"""Export PyTorch SED models to OpenVINO FP16 for CPU inference.

Critical for the 90-minute CPU-only inference constraint on Kaggle.
OpenVINO FP16 provides ~2-3x speedup over PyTorch on CPU.

Usage:
    python -m src.export_openvino \
        --checkpoint models/fold_0/best.ckpt \
        --output models/fold_0/model.xml \
        --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models.sed_model import SEDModel

logger = logging.getLogger(__name__)


def export_to_openvino(
    model: torch.nn.Module,
    save_path: str | Path,
    input_shape: tuple[int, ...] = (1, 1, 128, 312),
) -> Path:
    """Export PyTorch model to OpenVINO FP16 via ONNX intermediate.

    Args:
        model: PyTorch model in eval mode.
        save_path: Output path for .xml file (companion .bin created automatically).
        input_shape: Model input shape (batch, channels, n_mels, time_frames).

    Returns:
        Path to the saved .xml model file.
    """
    import openvino as ov

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX first
    onnx_path = save_path.with_suffix(".onnx")
    logger.info("Exporting to ONNX: %s", onnx_path)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["clip_pred", "frame_pred"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17,
    )

    # Convert ONNX to OpenVINO IR with FP16 compression
    logger.info("Converting to OpenVINO IR (FP16): %s", save_path)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(save_path), compress_to_fp16=True)

    # Verify the model loads correctly
    core = ov.Core()
    compiled = core.compile_model(str(save_path), "CPU")
    test_result = compiled({compiled.input(0): dummy_input.numpy()})
    output_shape = test_result[compiled.output(0)].shape
    logger.info("Verification passed. Output shape: %s", output_shape)

    # Clean up ONNX intermediate
    onnx_path.unlink(missing_ok=True)
    logger.info("Saved OpenVINO model to %s", save_path)

    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 OpenVINO Export")
    parser.add_argument("--checkpoint", type=str, required=True, help="PyTorch checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output .xml path")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config YAML")
    parser.add_argument("--batch-size", type=int, default=1, help="Export batch dimension")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build model and load weights
    model = SEDModel.from_config(config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # Compute input shape from config
    data_cfg = config.get("data", {})
    n_mels = data_cfg.get("n_mels", 128)
    duration = data_cfg.get("duration", 5.0)
    hop_length = data_cfg.get("hop_length", 512)
    sample_rate = data_cfg.get("sample_rate", 32000)
    time_frames = int(sample_rate * duration) // hop_length + 1
    input_shape = (args.batch_size, 1, n_mels, time_frames)

    logger.info("Input shape: %s", input_shape)
    export_to_openvino(model, args.output, input_shape)


if __name__ == "__main__":
    main()
