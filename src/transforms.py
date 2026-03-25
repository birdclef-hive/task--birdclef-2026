"""Audio preprocessing and augmentation pipeline for BirdCLEF 2026.

Mel spectrogram extraction via nnAudio + augmentations from top-5 BirdCLEF 2025 solutions:
MixUp with element-wise max targets, SpecAugment, BackgroundMix, RandomFiltering.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioTransform:
    """Mel spectrogram extraction matching top BirdCLEF solutions.

    Uses nnAudio for GPU-accelerated mel spectrograms when available,
    falls back to torchaudio on CPU.
    """

    SAMPLE_RATE: int = 32000
    DURATION: float = 5.0
    N_MELS: int = 128
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    F_MIN: int = 20
    F_MAX: int = 16000
    TOP_DB: float = 80.0

    def __init__(self, config: dict[str, Any] | None = None, train: bool = True) -> None:
        if config is not None:
            self.SAMPLE_RATE = config.get("sample_rate", self.SAMPLE_RATE)
            self.DURATION = config.get("duration", self.DURATION)
            self.N_MELS = config.get("n_mels", self.N_MELS)
            self.N_FFT = config.get("n_fft", self.N_FFT)
            self.HOP_LENGTH = config.get("hop_length", self.HOP_LENGTH)
            self.F_MIN = config.get("f_min", self.F_MIN)
            self.F_MAX = config.get("f_max", self.F_MAX)
            self.TOP_DB = config.get("top_db", self.TOP_DB)

        self.train = train

        # Try nnAudio first (GPU-accelerated), fall back to torchaudio
        try:
            from nnAudio.features import MelSpectrogram as NNMelSpec

            self.mel_spec = NNMelSpec(
                sr=self.SAMPLE_RATE,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=self.N_MELS,
                fmin=self.F_MIN,
                fmax=self.F_MAX,
            )
            self._backend = "nnAudio"
        except ImportError:
            logger.warning("nnAudio not available, falling back to torchaudio")
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.SAMPLE_RATE,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=self.N_MELS,
                f_min=self.F_MIN,
                f_max=self.F_MAX,
            )
            self._backend = "torchaudio"

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=self.TOP_DB)

    @property
    def expected_frames(self) -> int:
        """Number of time frames for a clip of DURATION seconds."""
        num_samples = int(self.SAMPLE_RATE * self.DURATION)
        return num_samples // self.HOP_LENGTH + 1

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to normalized mel spectrogram.

        Args:
            waveform: Tensor of shape (1, num_samples) or (num_samples,).

        Returns:
            Normalized mel spectrogram of shape (1, n_mels, time_frames).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_spec(waveform)
        mel = self.amplitude_to_db(mel)

        # Min-max normalize per spectrogram
        mel_min = mel.amin(dim=(-2, -1), keepdim=True)
        mel_max = mel.amax(dim=(-2, -1), keepdim=True)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)

        return mel


class AudioAugmentations:
    """Augmentation pipeline from top-5 BirdCLEF 2025 solutions.

    Key augmentations:
    - MixUp with element-wise max targets (+3.6% AUC, 2nd place)
    - SpecAugment: time and frequency masking
    - BackgroundMix: additive noise at controlled SNR
    - RandomFiltering: random bandpass/lowpass for recorder diversity
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.mixup_prob: float = config.get("mixup_prob", 0.5)
        self.mixup_alpha: float = config.get("mixup_alpha", 0.4)
        self.background_mix_prob: float = config.get("background_mix_prob", 0.5)
        self.spec_augment_enabled: bool = config.get("spec_augment", True)
        self.random_filtering_enabled: bool = config.get("random_filtering", False)
        self.time_mask_param: int = config.get("time_mask_param", 30)
        self.freq_mask_param: int = config.get("freq_mask_param", 10)
        self.num_time_masks: int = config.get("num_time_masks", 2)
        self.num_freq_masks: int = config.get("num_freq_masks", 2)

    def mixup(
        self,
        waveform1: torch.Tensor,
        target1: torch.Tensor,
        waveform2: torch.Tensor,
        target2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MixUp augmentation with element-wise max for multilabel targets.

        Uses element-wise max (not weighted sum) to preserve multilabel
        semantics — both species are present in the mixed audio.
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed = lam * waveform1 + (1 - lam) * waveform2
        target = torch.max(target1, target2)
        return mixed, target

    def spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        """SpecAugment: time and frequency masking on mel spectrograms."""
        for _ in range(self.num_time_masks):
            mel = torchaudio.transforms.TimeMasking(
                self.time_mask_param
            )(mel)
        for _ in range(self.num_freq_masks):
            mel = torchaudio.transforms.FrequencyMasking(
                self.freq_mask_param
            )(mel)
        return mel

    def background_mix(
        self,
        waveform: torch.Tensor,
        noise_waveform: torch.Tensor,
        snr_range: tuple[float, float] = (3.0, 15.0),
    ) -> torch.Tensor:
        """Mix background noise at a random SNR.

        Noise sources: prior-year soundscapes, ESC-50, or Pantanal ambience.
        """
        snr = np.random.uniform(*snr_range)
        signal_power = waveform.norm()
        noise_norm = noise_waveform.norm() + 1e-8
        noise_power = signal_power / (10 ** (snr / 20))
        scaled_noise = noise_waveform * (noise_power / noise_norm)
        return waveform + scaled_noise

    def random_filtering(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 32000,
    ) -> torch.Tensor:
        """Random bandpass filtering to simulate recorder diversity.

        Critical for Pantanal data with 1,000 different recorder units.
        """
        low_cutoff = np.random.randint(0, 500)
        high_cutoff = np.random.randint(12000, 16000)

        # Apply highpass then lowpass using torchaudio
        if low_cutoff > 0:
            waveform = torchaudio.functional.highpass_biquad(
                waveform, sample_rate, float(low_cutoff)
            )
        waveform = torchaudio.functional.lowpass_biquad(
            waveform, sample_rate, float(high_cutoff)
        )
        return waveform

    def __call__(
        self,
        mel: torch.Tensor,
        waveform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply augmentations to a mel spectrogram.

        For waveform-level augmentations (mixup, background_mix, random_filtering),
        these should be called explicitly before mel conversion. This __call__
        applies spectrogram-level augmentations only.
        """
        if self.spec_augment_enabled:
            mel = self.spec_augment(mel)
        return mel
