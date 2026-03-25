"""Pseudo-label generation with PowerTransform scaling for BirdCLEF 2026.

Multi-iterative pseudo-labeling pipeline (1st place: +5.8% AUC from 4 rounds).
PowerTransform prevents distribution collapse across rounds.
Soft targets preserve knowledge distillation benefits.

Usage:
    python -m src.pseudo_label \
        --config configs/pseudo_label/round1.yaml \
        --model-dir models/ \
        --data-dir data/raw/unlabeled_soundscapes/ \
        --output data/pseudo/round1.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import SoundscapeDataset
from src.models.sed_model import SEDModel
from src.transforms import AudioTransform

logger = logging.getLogger(__name__)


class PseudoLabelGenerator:
    """Multi-iterative pseudo-labeling with PowerTransform scaling.

    1st place key: PowerTransform prevents distribution collapse across rounds.
    2nd place key: Soft targets (don't binarize) = knowledge distillation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_prob_threshold: float = 0.1,
        power_scale: float = 0.7,
        replacement_prob: float = 0.4,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_prob_threshold = min_prob_threshold
        self.power_scale = power_scale
        self.replacement_prob = replacement_prob

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PseudoLabelGenerator":
        """Create from pseudo_labeling section of config."""
        cfg = config.get("pseudo_labeling", config)
        return cls(
            confidence_threshold=cfg.get("confidence_threshold", 0.5),
            min_prob_threshold=cfg.get("min_prob_threshold", 0.1),
            power_scale=cfg.get("power_scale", 0.7),
            replacement_prob=cfg.get("replacement_prob", 0.4),
        )

    def generate(
        self,
        models: list[torch.nn.Module],
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, list[str]]:
        """Generate ensemble predictions from multiple models.

        Returns:
            predictions: (num_samples, num_classes) averaged predictions
            row_ids: list of row identifiers
        """
        all_model_preds = []
        row_ids: list[str] = []

        for model_idx, model in enumerate(models):
            model.eval()
            model.to(device)
            preds = []
            if model_idx == 0:
                row_ids = []

            with torch.no_grad():
                for batch in dataloader:
                    mel = batch["mel"].to(device)
                    clip_pred, _ = model(mel)
                    preds.append(clip_pred.cpu())
                    if model_idx == 0:
                        row_ids.extend(batch["row_id"])

            all_model_preds.append(torch.cat(preds, dim=0))
            logger.info("Model %d: %d predictions generated", model_idx, len(preds))

        # Ensemble average
        predictions = torch.stack(all_model_preds).mean(dim=0)
        return predictions, row_ids

    def filter_and_scale(
        self, predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter low-confidence predictions and apply PowerTransform.

        PowerTransform (x^power_scale where power_scale < 1) pushes predictions
        toward more uniform values, preventing the distribution from collapsing
        to 0/1 across pseudo-labeling rounds.

        Returns:
            filtered: Filtered and scaled predictions
            mask: Boolean mask of which samples passed the confidence threshold
        """
        max_probs = predictions.max(dim=-1).values
        mask = max_probs >= self.confidence_threshold

        filtered = predictions[mask].clone()
        # Zero out very low probabilities
        filtered[filtered < self.min_prob_threshold] = 0
        # Apply PowerTransform to non-zero values
        nonzero = filtered > 0
        filtered[nonzero] = filtered[nonzero] ** self.power_scale

        logger.info(
            "Filtered: %d/%d samples passed threshold %.2f",
            mask.sum().item(),
            len(predictions),
            self.confidence_threshold,
        )

        return filtered, mask

    def create_training_data(
        self,
        original_data: list[dict[str, Any]],
        pseudo_data: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Merge original and pseudo-labeled data with replacement probability.

        With probability replacement_prob, replace an original sample with
        a pseudo-labeled sample of the same class.
        """
        merged = []
        replaced = 0

        for item in original_data:
            item_class = item.get("class", item.get("primary_label", ""))
            if (
                np.random.random() < self.replacement_prob
                and item_class in pseudo_data
                and len(pseudo_data[item_class]) > 0
            ):
                pseudo_item = pseudo_data[item_class][
                    np.random.randint(len(pseudo_data[item_class]))
                ]
                merged.append({**pseudo_item, "is_pseudo": True})
                replaced += 1
            else:
                merged.append({**item, "is_pseudo": False})

        logger.info(
            "Merged data: %d total, %d replaced with pseudo (%.1f%%)",
            len(merged),
            replaced,
            100 * replaced / max(len(merged), 1),
        )
        return merged


def load_models(
    model_dir: Path,
    config: dict[str, Any],
    device: str = "cpu",
) -> list[torch.nn.Module]:
    """Load all checkpoint models from a directory."""
    models = []
    checkpoint_paths = sorted(model_dir.glob("**/*.ckpt"))

    if not checkpoint_paths:
        logger.warning("No checkpoints found in %s", model_dir)
        return models

    for ckpt_path in checkpoint_paths:
        logger.info("Loading model from %s", ckpt_path)
        model = SEDModel.from_config(config)
        # Load from Lightning checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Remove 'model.' prefix if present (from Lightning module)
        cleaned = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=False)
        models.append(model)

    logger.info("Loaded %d models", len(models))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Pseudo-Label Generation")
    parser.add_argument("--config", type=str, required=True, help="Pseudo-label config YAML")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml", help="Base config for model/data")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with model checkpoints")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with unlabeled audio")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load configs
    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)
    with open(args.config) as f:
        pseudo_config = yaml.safe_load(f)

    # Merge pseudo config into base
    config = {**base_config, **pseudo_config}

    # Load models
    models = load_models(Path(args.model_dir), config, args.device)
    if not models:
        logger.error("No models loaded, exiting")
        return

    # Build dataset
    data_dir = Path(args.data_dir)
    audio_paths = sorted(
        list(data_dir.glob("*.ogg"))
        + list(data_dir.glob("*.wav"))
        + list(data_dir.glob("*.mp3"))
        + list(data_dir.glob("*.flac"))
    )
    logger.info("Found %d audio files", len(audio_paths))

    transform = AudioTransform(config.get("data", {}), train=False)
    dataset = SoundscapeDataset(audio_paths, config.get("data", {}), transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Generate predictions
    generator = PseudoLabelGenerator.from_config(config)
    predictions, row_ids = generator.generate(models, dataloader, args.device)

    # Filter and scale
    filtered, mask = generator.filter_and_scale(predictions)

    # Save to CSV
    filtered_ids = [row_ids[i] for i in range(len(row_ids)) if mask[i]]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use species columns from metadata or generate generic ones
    num_classes = filtered.shape[1]
    species_cols = [f"species_{i}" for i in range(num_classes)]

    df = pd.DataFrame(filtered.numpy(), columns=species_cols)
    df.insert(0, "row_id", filtered_ids)
    df.to_csv(output_path, index=False)
    logger.info("Pseudo-labels saved to %s (%d samples)", output_path, len(df))


if __name__ == "__main__":
    main()
