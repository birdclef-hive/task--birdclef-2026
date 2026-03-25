"""CPU inference pipeline for BirdCLEF 2026.

Processes soundscape recordings, runs predictions through OpenVINO or PyTorch
models, and outputs a submission CSV. Designed for the 90-minute CPU-only
inference constraint on Kaggle.

Usage:
    python -m src.inference \
        --model-dir models/ \
        --data-dir data/processed/val_soundscapes/ \
        --output predictions.csv \
        --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import SoundscapeDataset
from src.transforms import AudioTransform

logger = logging.getLogger(__name__)


class OpenVINOPredictor:
    """Inference using OpenVINO FP16 models for maximum CPU throughput."""

    def __init__(self, model_path: str | Path) -> None:
        import openvino as ov

        core = ov.Core()
        self.model = core.compile_model(str(model_path), "CPU")
        self.input_layer = self.model.input(0)
        self.output_layer = self.model.output(0)
        logger.info("OpenVINO model loaded: %s", model_path)

    def predict(self, mel: np.ndarray) -> np.ndarray:
        """Run inference on a batch of mel spectrograms.

        Args:
            mel: (batch, 1, n_mels, time_frames)

        Returns:
            predictions: (batch, num_classes)
        """
        result = self.model({self.input_layer: mel})
        return result[self.output_layer]


class PyTorchPredictor:
    """Inference using PyTorch models (fallback when OpenVINO not available)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()

    def predict(self, mel: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(mel).float()
            clip_pred, _ = self.model(tensor)
            return clip_pred.numpy()


def load_predictors(
    model_dir: Path,
    config: dict[str, Any],
) -> list[OpenVINOPredictor | PyTorchPredictor]:
    """Load all models from directory (OpenVINO .xml or PyTorch .ckpt/.pt)."""
    predictors = []

    # Try OpenVINO first
    xml_files = sorted(model_dir.glob("**/*.xml"))
    if xml_files:
        for xml_path in xml_files:
            try:
                predictors.append(OpenVINOPredictor(xml_path))
            except Exception as e:
                logger.warning("Failed to load OpenVINO model %s: %s", xml_path, e)
        if predictors:
            return predictors

    # Fall back to PyTorch
    from src.models.sed_model import SEDModel

    ckpt_files = sorted(
        list(model_dir.glob("**/*.ckpt")) + list(model_dir.glob("**/*.pt"))
    )
    for ckpt_path in ckpt_files:
        try:
            model = SEDModel.from_config(config)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            cleaned = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }
            model.load_state_dict(cleaned, strict=False)
            predictors.append(PyTorchPredictor(model))
            logger.info("Loaded PyTorch model: %s", ckpt_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", ckpt_path, e)

    return predictors


def run_inference(
    predictors: list[OpenVINOPredictor | PyTorchPredictor],
    dataloader: DataLoader,
    time_budget_seconds: float = 5100,  # 85 minutes
) -> tuple[np.ndarray, list[str]]:
    """Run ensemble inference with time budget monitoring.

    Returns:
        predictions: (num_segments, num_classes) ensemble-averaged predictions
        row_ids: list of row identifiers
    """
    start_time = time.time()
    all_preds: list[np.ndarray] = []
    row_ids: list[str] = []
    num_models = len(predictors)

    for batch_idx, batch in enumerate(dataloader):
        elapsed = time.time() - start_time
        if elapsed > time_budget_seconds:
            logger.warning(
                "Time budget exceeded (%.0fs / %.0fs) at batch %d",
                elapsed, time_budget_seconds, batch_idx,
            )
            break

        mel_np = batch["mel"].numpy()
        row_ids.extend(batch["row_id"])

        # Ensemble: average predictions across all models
        batch_preds = np.zeros(
            (mel_np.shape[0], predictors[0].predict(mel_np[:1]).shape[1]),
            dtype=np.float32,
        ) if batch_idx == 0 or True else None

        model_preds = []
        for predictor in predictors:
            pred = predictor.predict(mel_np)
            model_preds.append(pred)

        batch_preds = np.mean(model_preds, axis=0)
        all_preds.append(batch_preds)

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            logger.info(
                "Batch %d/%d (%.1f batches/s, %.0fs elapsed)",
                batch_idx + 1, len(dataloader), rate, elapsed,
            )

    total_time = time.time() - start_time
    predictions = np.concatenate(all_preds, axis=0)
    logger.info(
        "Inference complete: %d segments, %d models, %.1fs total",
        len(predictions), num_models, total_time,
    )

    return predictions, row_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Inference")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Soundscape audio directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config YAML")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--species-list", type=str, default=None, help="Path to species list file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    inf_cfg = config.get("inference", {})
    batch_size = args.batch_size or inf_cfg.get("batch_size", 32)
    time_budget = inf_cfg.get("time_budget_minutes", 85) * 60

    # Load models
    model_dir = Path(args.model_dir)
    predictors = load_predictors(model_dir, config)
    if not predictors:
        logger.error("No models found in %s", model_dir)
        return
    logger.info("Loaded %d model(s)", len(predictors))

    # Build dataset
    data_dir = Path(args.data_dir)
    audio_paths = sorted(
        list(data_dir.glob("*.ogg"))
        + list(data_dir.glob("*.wav"))
        + list(data_dir.glob("*.mp3"))
        + list(data_dir.glob("*.flac"))
    )
    logger.info("Found %d soundscape files", len(audio_paths))

    transform = AudioTransform(data_cfg, train=False)
    dataset = SoundscapeDataset(audio_paths, data_cfg, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=data_cfg.get("num_workers", 4), pin_memory=False,
    )

    # Run inference
    predictions, row_ids = run_inference(predictors, dataloader, time_budget)

    # Load species list
    if args.species_list and Path(args.species_list).exists():
        with open(args.species_list) as f:
            species_cols = [line.strip() for line in f if line.strip()]
    else:
        # Try to infer from metadata
        meta_path = Path("data/raw/train_metadata.csv")
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            species_cols = sorted(meta["primary_label"].unique().tolist())
        else:
            species_cols = [f"species_{i}" for i in range(predictions.shape[1])]

    # Build output DataFrame
    result = pd.DataFrame(predictions[:, : len(species_cols)], columns=species_cols)
    result.insert(0, "row_id", row_ids[: len(result)])
    result.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s (%d rows)", args.output, len(result))


if __name__ == "__main__":
    main()
