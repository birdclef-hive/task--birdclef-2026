"""PyTorch Dataset classes for BirdCLEF 2026.

Handles labeled training data (variable-length recordings segmented into
5-second clips) and unlabeled soundscape data for inference/pseudo-labeling.
Supports both raw audio files and HDF5 pre-cached mel spectrograms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from src.transforms import AudioAugmentations, AudioTransform

logger = logging.getLogger(__name__)


def load_audio(
    path: str | Path,
    target_sr: int = 32000,
    max_duration: float | None = None,
) -> torch.Tensor:
    """Load and resample an audio file.

    Returns:
        Waveform tensor of shape (1, num_samples).
    """
    waveform, sr = torchaudio.load(str(path))
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Truncate if max_duration specified
    if max_duration is not None:
        max_samples = int(target_sr * max_duration)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
    return waveform


def segment_waveform(
    waveform: torch.Tensor,
    sr: int = 32000,
    duration: float = 5.0,
) -> list[torch.Tensor]:
    """Split a waveform into fixed-length segments.

    Last segment is zero-padded if shorter than duration.
    """
    segment_samples = int(sr * duration)
    total_samples = waveform.shape[1]
    segments = []

    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        segment = waveform[:, start:end]
        # Pad last segment if needed
        if segment.shape[1] < segment_samples:
            pad_size = segment_samples - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, pad_size))
        segments.append(segment)

    return segments


class BirdCLEFDataset(Dataset):
    """Dataset for labeled BirdCLEF training data.

    Loads variable-length audio recordings, segments them into 5-second clips,
    and returns (mel_spectrogram, target) pairs. Supports random segment
    selection during training and full coverage during validation.

    Args:
        metadata: DataFrame with columns: filename, primary_label, secondary_labels, ...
        audio_dir: Path to directory containing audio files.
        species_list: List of all species labels (defines target vector order).
        config: Data config dict from base.yaml.
        train: If True, randomly sample one segment per recording per epoch.
        transform: AudioTransform instance for mel spectrogram conversion.
        augmentations: AudioAugmentations instance (only used if train=True).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_dir: str | Path,
        species_list: list[str],
        config: dict[str, Any] | None = None,
        train: bool = True,
        transform: AudioTransform | None = None,
        augmentations: AudioAugmentations | None = None,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.train = train

        config = config or {}
        self.sample_rate = config.get("sample_rate", 32000)
        self.duration = config.get("duration", 5.0)
        self.max_duration = config.get("max_duration_per_sample", 30.0)

        self.transform = transform or AudioTransform(config, train=train)
        self.augmentations = augmentations if train else None

    def __len__(self) -> int:
        return len(self.metadata)

    def _build_target(self, row: pd.Series) -> torch.Tensor:
        """Build multilabel target vector from primary + secondary labels."""
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        primary = str(row.get("primary_label", ""))
        if primary in self.species_to_idx:
            target[self.species_to_idx[primary]] = 1.0

        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary:
            for label in secondary.split():
                label = label.strip("[]',\" ")
                if label in self.species_to_idx:
                    target[self.species_to_idx[label]] = 1.0

        return target

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        filepath = self.audio_dir / row["filename"]

        # Load audio
        waveform = load_audio(filepath, self.sample_rate, self.max_duration)

        # Segment into clips
        segments = segment_waveform(waveform, self.sample_rate, self.duration)

        if self.train:
            # Random segment during training
            seg_idx = np.random.randint(len(segments))
            segment = segments[seg_idx]
        else:
            # During validation, use the first segment (or could iterate all)
            segment = segments[0]

        # Apply waveform-level augmentations
        if self.augmentations and self.train:
            if (
                self.augmentations.random_filtering_enabled
                and np.random.random() < 0.3
            ):
                segment = self.augmentations.random_filtering(
                    segment, self.sample_rate
                )

        # Convert to mel spectrogram
        mel = self.transform(segment)

        # Apply spectrogram-level augmentations
        if self.augmentations and self.train:
            mel = self.augmentations(mel)

        target = self._build_target(row)

        return {
            "mel": mel,
            "target": target,
            "filename": row["filename"],
        }


class SoundscapeDataset(Dataset):
    """Dataset for unlabeled soundscape audio (inference/pseudo-labeling).

    Processes full soundscape recordings by sliding a 5-second window
    with configurable stride, outputting all segments for prediction.

    Args:
        audio_paths: List of paths to soundscape audio files.
        config: Data config dict from base.yaml.
        transform: AudioTransform instance.
        stride: Stride in seconds between segments (default: 5.0 = no overlap).
    """

    def __init__(
        self,
        audio_paths: list[str | Path],
        config: dict[str, Any] | None = None,
        transform: AudioTransform | None = None,
        stride: float | None = None,
    ) -> None:
        config = config or {}
        self.sample_rate = config.get("sample_rate", 32000)
        self.duration = config.get("duration", 5.0)
        self.stride = stride or self.duration

        self.transform = transform or AudioTransform(config, train=False)

        # Pre-compute all segments across all files
        self.segments: list[dict[str, Any]] = []
        for audio_path in audio_paths:
            audio_path = Path(audio_path)
            try:
                info = torchaudio.info(str(audio_path))
                total_duration = info.num_frames / info.sample_rate
            except Exception:
                logger.warning("Could not read info for %s, skipping", audio_path)
                continue

            segment_dur = self.duration
            stride_dur = self.stride
            offset = 0.0
            seg_idx = 0
            while offset + segment_dur <= total_duration + 0.1:
                end_time = min(offset + segment_dur, total_duration)
                self.segments.append({
                    "path": audio_path,
                    "offset": offset,
                    "end_time": end_time,
                    "file_id": audio_path.stem,
                    "segment_idx": seg_idx,
                    "row_id": f"{audio_path.stem}_{int(offset)}",
                })
                offset += stride_dur
                seg_idx += 1

        logger.info(
            "SoundscapeDataset: %d files, %d segments",
            len(audio_paths),
            len(self.segments),
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seg_info = self.segments[idx]
        path = seg_info["path"]
        offset_samples = int(seg_info["offset"] * self.sample_rate)
        num_samples = int(self.duration * self.sample_rate)

        # Load segment
        try:
            waveform, sr = torchaudio.load(
                str(path),
                frame_offset=int(seg_info["offset"] * (torchaudio.info(str(path)).sample_rate)),
                num_frames=int(self.duration * (torchaudio.info(str(path)).sample_rate)),
            )
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        except Exception:
            logger.warning("Error loading segment from %s at %.1f", path, seg_info["offset"])
            waveform = torch.zeros(1, num_samples)

        # Pad if needed
        if waveform.shape[1] < num_samples:
            pad_size = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]

        mel = self.transform(waveform)

        return {
            "mel": mel,
            "row_id": seg_info["row_id"],
            "file_id": seg_info["file_id"],
        }
