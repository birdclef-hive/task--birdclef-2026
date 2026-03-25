"""PyTorch Lightning training loop for BirdCLEF 2026.

Reads configs/base.yaml, supports multi-fold training, MixUp augmentation
in the training step, metric logging, and checkpoint saving.

Usage:
    python -m src.train --config configs/base.yaml --fold 0
    python -m src.train --config configs/base.yaml --fold all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.dataset import BirdCLEFDataset
from src.models.losses import build_loss
from src.models.sed_model import SEDModel
from src.transforms import AudioAugmentations, AudioTransform

logger = logging.getLogger(__name__)


class BirdCLEFModule(pl.LightningModule):
    """Lightning module wrapping SEDModel with training/validation logic."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.model = SEDModel.from_config(config)
        self.criterion = build_loss(config)
        self.augmentations = AudioAugmentations(config.get("augmentations", {}))

        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mel = batch["mel"]
        target = batch["target"]

        # MixUp augmentation at batch level
        mixup_cfg = self.config.get("augmentations", {})
        if np.random.random() < mixup_cfg.get("mixup_prob", 0.5):
            # Shuffle indices for mixing partners
            indices = torch.randperm(mel.size(0), device=mel.device)
            alpha = mixup_cfg.get("mixup_alpha", 0.4)
            lam = np.random.beta(alpha, alpha)
            mel = lam * mel + (1 - lam) * mel[indices]
            target = torch.max(target, target[indices])

        clip_pred, _ = self.model(mel)
        loss = self.criterion(clip_pred, target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        mel = batch["mel"]
        target = batch["target"]

        clip_pred, _ = self.model(mel)
        loss = self.criterion(clip_pred, target)

        self.validation_step_outputs.append({
            "preds": clip_pred.detach().cpu(),
            "targets": target.detach().cpu(),
            "loss": loss.detach().cpu(),
        })

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        all_preds = torch.cat([o["preds"] for o in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([o["targets"] for o in self.validation_step_outputs], dim=0)

        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()

        # Compute macro-averaged ROC-AUC (skip classes with no positive samples)
        aucs = []
        for i in range(targets_np.shape[1]):
            if targets_np[:, i].sum() > 0:
                try:
                    auc = roc_auc_score(targets_np[:, i], preds_np[:, i])
                    aucs.append(auc)
                except ValueError:
                    pass

        macro_auc = np.mean(aucs) if aucs else 0.0
        self.log("val/macro_auc", macro_auc, prog_bar=True)
        self.log("val/classes_evaluated", float(len(aucs)))

        logger.info(
            "Epoch %d — val/macro_auc: %.4f (%d/%d classes)",
            self.current_epoch,
            macro_auc,
            len(aucs),
            targets_np.shape[1],
        )

        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        train_cfg = self.config.get("training", {})
        optimizer_name = train_cfg.get("optimizer", "adamw")
        lr = train_cfg.get("learning_rate", 1e-4)
        wd = train_cfg.get("weight_decay", 0.01)

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler_name = train_cfg.get("scheduler", "cosine")
        epochs = train_cfg.get("epochs", 50)
        min_lr = train_cfg.get("min_lr", 1e-6)

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=min_lr
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
        else:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config, optionally merge with model/augmentation overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        for override_path in overrides:
            with open(override_path) as f:
                override = yaml.safe_load(f)
            # Deep merge: override values take precedence
            for key, value in override.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value

    return config


def create_folds(
    metadata: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: str = "primary_label",
    group_by: str | None = "author",
) -> pd.DataFrame:
    """Create stratified K-fold splits with optional group constraint.

    Groups by recordist to prevent data leakage (same recorder in train+val).
    """
    metadata = metadata.copy()
    metadata["fold"] = -1

    stratify_col = metadata[stratify_by].fillna("unknown")
    if group_by and group_by in metadata.columns:
        groups = metadata[group_by].fillna("unknown")
    else:
        groups = np.arange(len(metadata))

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (_, val_idx) in enumerate(skf.split(metadata, stratify_col, groups)):
        metadata.loc[val_idx, "fold"] = fold_idx

    return metadata


def train_fold(
    config: dict[str, Any],
    fold: int,
    metadata: pd.DataFrame,
    species_list: list[str],
    audio_dir: Path,
    output_dir: Path,
) -> None:
    """Train a single fold."""
    logger.info("=== Training fold %d ===", fold)

    train_df = metadata[metadata["fold"] != fold].reset_index(drop=True)
    val_df = metadata[metadata["fold"] == fold].reset_index(drop=True)

    logger.info("Train: %d samples, Val: %d samples", len(train_df), len(val_df))

    data_cfg = config.get("data", {})
    train_transform = AudioTransform(data_cfg, train=True)
    val_transform = AudioTransform(data_cfg, train=False)
    augmentations = AudioAugmentations(config.get("augmentations", {}))

    train_ds = BirdCLEFDataset(
        train_df, audio_dir, species_list, data_cfg,
        train=True, transform=train_transform, augmentations=augmentations,
    )
    val_ds = BirdCLEFDataset(
        val_df, audio_dir, species_list, data_cfg,
        train=False, transform=val_transform,
    )

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    module = BirdCLEFModule(config)

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(fold_dir),
            filename="best-{epoch:02d}-{val/macro_auc:.4f}",
            monitor="val/macro_auc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/macro_auc",
            mode="max",
            patience=10,
            verbose=True,
        ),
    ]

    epochs = train_cfg.get("epochs", 50)
    precision = "16-mixed" if train_cfg.get("mixed_precision", False) else "32-true"

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir=str(fold_dir),
        precision=precision,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(module, train_loader, val_loader)

    logger.info("Fold %d complete. Best checkpoint: %s", fold, callbacks[0].best_model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    parser.add_argument("--override", type=str, nargs="*", help="Additional config files to merge")
    parser.add_argument("--fold", type=str, default="0", help="Fold number or 'all'")
    parser.add_argument("--data-dir", type=str, default="data/raw/train_audio", help="Audio directory")
    parser.add_argument("--metadata", type=str, default="data/raw/train.csv", help="Metadata CSV")
    parser.add_argument("--taxonomy", type=str, default="data/raw/taxonomy.csv", help="Taxonomy CSV (defines full species list)")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for checkpoints")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config, args.override)
    logger.info("Config loaded: %s", args.config)

    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        return

    metadata = pd.read_csv(metadata_path)
    # Ensure primary_label is string (some are numeric taxon IDs)
    metadata["primary_label"] = metadata["primary_label"].astype(str)
    logger.info("Loaded metadata: %d rows", len(metadata))

    # Build species list from taxonomy (includes species with 0 training samples)
    taxonomy_path = Path(args.taxonomy)
    if taxonomy_path.exists():
        taxonomy = pd.read_csv(taxonomy_path)
        species_list = sorted(taxonomy["primary_label"].astype(str).unique().tolist())
        logger.info("Species list from taxonomy: %d", len(species_list))
    else:
        species_list = sorted(metadata["primary_label"].astype(str).unique().tolist())
        logger.info("Species list from train metadata: %d", len(species_list))
    num_classes = len(species_list)
    logger.info("Species: %d", num_classes)

    # Update config with actual class count
    if "model" not in config:
        config["model"] = {}
    config["model"]["num_classes"] = num_classes

    # Create folds
    train_cfg = config.get("training", {})
    n_folds = train_cfg.get("n_folds", 5)
    metadata = create_folds(
        metadata,
        n_folds=n_folds,
        stratify_by=train_cfg.get("stratify_by", "primary_label"),
        group_by=train_cfg.get("group_by", "recordist"),
    )

    audio_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.fold == "all":
        folds = list(range(n_folds))
    else:
        folds = [int(args.fold)]

    for fold in folds:
        train_fold(config, fold, metadata, species_list, audio_dir, output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
