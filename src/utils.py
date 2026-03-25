"""Utility functions for BirdCLEF 2026.

Audio loading helpers, Silero VAD integration for voice activity detection,
and data curation utilities for filtering low-quality recordings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(
    path: str | Path,
    target_sr: int = 32000,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and resample to target sample rate.

    Returns:
        waveform: (channels, num_samples) tensor
        sample_rate: target sample rate
    """
    waveform, sr = torchaudio.load(str(path))
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform, target_sr


def get_audio_duration(path: str | Path) -> float:
    """Get duration of an audio file in seconds without loading it."""
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


# ---------------------------------------------------------------------------
# Silero VAD
# ---------------------------------------------------------------------------

_vad_model = None
_vad_utils = None


def get_silero_vad() -> tuple[Any, Any]:
    """Load Silero VAD model (cached after first call).

    Returns:
        model: Silero VAD model
        get_speech_timestamps: function to get speech timestamps
    """
    global _vad_model, _vad_utils
    if _vad_model is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
    get_speech_timestamps = _vad_utils[0]
    return _vad_model, get_speech_timestamps


def detect_speech(
    waveform: torch.Tensor,
    sample_rate: int = 32000,
    threshold: float = 0.5,
) -> list[dict[str, int]]:
    """Detect speech/voice segments using Silero VAD.

    Useful for filtering out recordings dominated by human speech,
    or for removing voice segments from field recordings.

    Args:
        waveform: (1, num_samples) or (num_samples,) tensor
        sample_rate: audio sample rate
        threshold: VAD threshold (higher = stricter)

    Returns:
        List of dicts with 'start' and 'end' sample indices.
    """
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    # Silero VAD expects 16kHz
    if sample_rate != 16000:
        waveform_16k = torchaudio.functional.resample(waveform, sample_rate, 16000)
    else:
        waveform_16k = waveform

    model, get_speech_timestamps = get_silero_vad()
    timestamps = get_speech_timestamps(
        waveform_16k, model, threshold=threshold, sampling_rate=16000
    )

    # Scale timestamps back to original sample rate
    scale = sample_rate / 16000
    return [
        {"start": int(t["start"] * scale), "end": int(t["end"] * scale)}
        for t in timestamps
    ]


def remove_speech(
    waveform: torch.Tensor,
    sample_rate: int = 32000,
    threshold: float = 0.5,
    fade_samples: int = 160,
) -> torch.Tensor:
    """Remove speech segments from a waveform by zeroing them out.

    Applies a short fade to avoid clicks at segment boundaries.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    speech_segments = detect_speech(waveform, sample_rate, threshold)
    result = waveform.clone()

    for seg in speech_segments:
        start = max(0, seg["start"] - fade_samples)
        end = min(waveform.shape[1], seg["end"] + fade_samples)
        result[:, start:end] = 0.0

    return result


def compute_speech_ratio(
    waveform: torch.Tensor,
    sample_rate: int = 32000,
    threshold: float = 0.5,
) -> float:
    """Compute fraction of audio that contains speech."""
    total_samples = waveform.shape[-1]
    if total_samples == 0:
        return 0.0

    speech_segments = detect_speech(waveform, sample_rate, threshold)
    speech_samples = sum(seg["end"] - seg["start"] for seg in speech_segments)
    return speech_samples / total_samples


# ---------------------------------------------------------------------------
# Data curation
# ---------------------------------------------------------------------------

def compute_snr(waveform: torch.Tensor, frame_length: int = 2048) -> float:
    """Estimate signal-to-noise ratio of an audio clip.

    Uses the ratio of max frame energy to median frame energy as a proxy.
    """
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    num_frames = len(waveform) // frame_length
    if num_frames == 0:
        return 0.0

    frames = waveform[: num_frames * frame_length].reshape(num_frames, frame_length)
    frame_energy = (frames ** 2).mean(dim=1)

    if frame_energy.median() < 1e-10:
        return 0.0

    snr = 10 * torch.log10(frame_energy.max() / (frame_energy.median() + 1e-10))
    return snr.item()


def filter_low_quality(
    metadata: Any,
    audio_dir: str | Path,
    min_snr: float = 5.0,
    max_speech_ratio: float = 0.3,
    sample_rate: int = 32000,
) -> Any:
    """Filter out low-quality recordings based on SNR and speech content.

    Args:
        metadata: DataFrame with 'filename' column
        audio_dir: Directory containing audio files
        min_snr: Minimum SNR threshold (dB)
        max_speech_ratio: Maximum allowed speech ratio

    Returns:
        Filtered DataFrame
    """
    import pandas as pd

    audio_dir = Path(audio_dir)
    keep_mask = []

    for _, row in metadata.iterrows():
        filepath = audio_dir / row["filename"]
        try:
            waveform, sr = load_audio(filepath, sample_rate)
            snr = compute_snr(waveform)
            speech_ratio = compute_speech_ratio(waveform, sample_rate)
            keep = snr >= min_snr and speech_ratio <= max_speech_ratio
        except Exception as e:
            logger.warning("Error processing %s: %s", filepath, e)
            keep = False

        keep_mask.append(keep)

    filtered = metadata[keep_mask].reset_index(drop=True)
    logger.info(
        "Data curation: %d/%d recordings kept (min_snr=%.1f, max_speech=%.1f)",
        len(filtered), len(metadata), min_snr, max_speech_ratio,
    )
    return filtered


def upsample_rare_classes(
    metadata: Any,
    min_samples: int = 20,
) -> Any:
    """Upsample rare classes to a minimum number of samples.

    Duplicates rows for classes with fewer than min_samples recordings.
    """
    import pandas as pd

    class_counts = metadata["primary_label"].value_counts()
    rare_classes = class_counts[class_counts < min_samples].index.tolist()

    if not rare_classes:
        return metadata

    extra_rows = []
    for cls in rare_classes:
        cls_data = metadata[metadata["primary_label"] == cls]
        needed = min_samples - len(cls_data)
        if needed > 0:
            repeated = cls_data.sample(n=needed, replace=True, random_state=42)
            extra_rows.append(repeated)

    if extra_rows:
        upsampled = pd.concat([metadata] + extra_rows, ignore_index=True)
        logger.info(
            "Upsampled %d rare classes (<%d samples), total: %d -> %d",
            len(rare_classes), min_samples, len(metadata), len(upsampled),
        )
        return upsampled

    return metadata
