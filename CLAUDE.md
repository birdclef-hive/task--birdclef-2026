# CLAUDE.md — BirdCLEF 2026 × Hive Automated ML Experimentation

## Project Overview

You are setting up an automated ML experimentation system for the **BirdCLEF+ 2026 Kaggle competition** using the **Hive** collaborative agent platform (https://github.com/rllm-org/hive). The competition requires identifying 650+ wildlife species from audio recordings in Brazil's Pantanal wetlands. Submissions are code competitions: CPU-only inference, internet disabled, ~90-minute time limit.

**Competition URL:** https://www.kaggle.com/competitions/birdclef-2026
**Entry deadline:** May 27, 2026 | **Final submission:** June 3, 2026
**Metric:** Custom multilabel classification metric (likely macro-averaged ROC-AUC variant, skip classes with no true positives)
**Prize pool:** $50,000

## Environment & Preferences

- **OS:** Windows, **Editor:** VS Code, **Languages:** Python, Bash/PowerShell
- Always show file paths when creating files
- Ask before making big decisions
- Full explanations with context, kept concise
- Prefer local-first builds
- Use GVR (Generator-Verifier-Reviser) prompting when appropriate

---

## Step 1: Project Structure

Create the following directory structure:

```
birdclef-2026-hive/
├── CLAUDE.md                     # This file (project memory)
├── THEORY.md                     # Reasoning log for architectural decisions
├── program.md                    # Hive task instructions for agents
├── eval/
│   └── eval.sh                   # Evaluation script (runs local validation)
├── configs/
│   ├── base.yaml                 # Base training configuration
│   ├── models/                   # Per-architecture configs
│   │   ├── efficientnet_b0.yaml
│   │   ├── efficientnet_b3.yaml
│   │   ├── efficientnetv2_s.yaml
│   │   ├── efficientnetv2_b3.yaml
│   │   ├── eca_nfnet_l0.yaml
│   │   └── regnety_008.yaml
│   ├── augmentations/
│   │   ├── standard.yaml         # MixUp + SpecAugment + BackgroundMix
│   │   └── aggressive.yaml       # + RandomFiltering + VoiceAug
│   └── pseudo_label/
│       ├── round1.yaml
│       ├── round2.yaml
│       └── round3.yaml
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Audio dataset with HDF5 loading
│   ├── transforms.py             # Mel spectrogram + augmentations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sed_model.py          # SED architecture with timm backbones
│   │   ├── losses.py             # FocalLoss, SoftAUCLoss, BCE+Focal blend
│   │   └── model_soup.py         # Weight-averaging for Model Soup
│   ├── train.py                  # Training loop (PyTorch Lightning)
│   ├── pseudo_label.py           # Pseudo-label generation with power scaling
│   ├── inference.py              # CPU inference pipeline
│   ├── postprocess.py            # TopN post-processing
│   ├── export_openvino.py        # PyTorch → OpenVINO FP16 conversion
│   └── utils.py                  # Audio loading, Silero VAD, data curation
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis
│   ├── create_folds.ipynb        # Stratified K-fold with author grouping
│   ├── create_pseudo.ipynb       # Pseudo-label generation
│   └── submission.ipynb          # Final Kaggle submission notebook
├── data/
│   ├── raw/                      # Symlink to competition data
│   ├── processed/                # HDF5 files, mel caches
│   ├── external/                 # Xeno-Canto, prior BirdCLEF data
│   └── pseudo/                   # Generated pseudo-labels per round
├── models/                       # Saved checkpoints and OpenVINO models
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Step 2: Core ML Pipeline

All code below is derived from BirdCLEF 2025's top-5 winning solutions. Implement each file with these patterns as the starting point.

### 2.1 Audio Preprocessing (`src/transforms.py`)

```python
import torch
import torchaudio
import numpy as np
from nnAudio.features import MelSpectrogram

class AudioTransform:
    """Mel spectrogram extraction matching top BirdCLEF solutions."""

    SAMPLE_RATE = 32000
    DURATION = 5.0  # seconds
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 20
    F_MAX = 16000
    TOP_DB = 80

    def __init__(self, train=True):
        self.mel_spec = MelSpectrogram(
            sr=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            fmin=self.F_MIN,
            fmax=self.F_MAX,
        )
        self.train = train

    def __call__(self, waveform):
        mel = self.mel_spec(waveform)
        mel = torchaudio.transforms.AmplitudeToDB(top_db=self.TOP_DB)(mel)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        return mel


class AudioAugmentations:
    """Augmentation pipeline from top-5 solutions."""

    def __init__(self, config):
        self.mixup_prob = config.get('mixup_prob', 0.5)
        self.background_mix_prob = config.get('background_mix_prob', 0.5)
        self.spec_augment = config.get('spec_augment', True)
        self.random_filtering = config.get('random_filtering', False)

    def mixup(self, waveform1, target1, waveform2, target2, alpha=0.4):
        """MixUp: most impactful augmentation (contributed +3.6% AUC in 2nd place).
        Uses element-wise max for targets (not weighted sum) to preserve multilabel nature."""
        lam = np.random.beta(alpha, alpha)
        mixed = lam * waveform1 + (1 - lam) * waveform2
        target = torch.max(target1, target2)
        return mixed, target

    def spec_augment_fn(self, mel, num_time_masks=2, num_freq_masks=2,
                        time_mask_param=30, freq_mask_param=10):
        """SpecAugment: time and frequency masking on mel spectrograms."""
        for _ in range(num_time_masks):
            mel = torchaudio.transforms.TimeMasking(time_mask_param)(mel)
        for _ in range(num_freq_masks):
            mel = torchaudio.transforms.FrequencyMasking(freq_mask_param)(mel)
        return mel

    def background_mix(self, waveform, noise_waveform, snr_range=(3, 15)):
        """Mix background noise from prior-year soundscapes or ESC-50."""
        snr = np.random.uniform(*snr_range)
        noise_power = waveform.norm() / (10 ** (snr / 20))
        scaled_noise = noise_waveform * noise_power / (noise_waveform.norm() + 1e-8)
        return waveform + scaled_noise
```

### 2.2 SED Model Architecture (`src/models/sed_model.py`)

```python
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """SED attention head for frame-level predictions."""

    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.attention = nn.Linear(hidden_dim, num_classes)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        attention_weights = torch.softmax(self.attention(x), dim=1)
        frame_preds = torch.sigmoid(self.classifier(x))
        clip_pred = (attention_weights * frame_preds).sum(dim=1)
        return clip_pred, frame_preds


class SEDModel(nn.Module):
    """Sound Event Detection model with timm backbone.

    Supports: tf_efficientnet_b0_ns, tf_efficientnet_b3_ns,
    tf_efficientnetv2_s.in21k, tf_efficientnetv2_b3, eca_nfnet_l0,
    regnety_008, regnety_016, mnasnet_100, spnasnet_100
    """

    def __init__(self, backbone_name, num_classes, pretrained=True,
                 hidden_dim=512, dropout_backbone=0.25, dropout_head=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool=''
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, 128, 312)  # ~5s at hop=512
            features = self.backbone(dummy)
            feat_dim = features.shape[1]

        self.gem_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.dropout_backbone = nn.Dropout(dropout_backbone)
        self.head = AttentionHead(feat_dim, num_classes, hidden_dim, dropout_head)

    def forward(self, x):
        features = self.backbone(x)
        features = self.gem_pool(features).squeeze(-1)
        features = features.permute(0, 2, 1)
        features = self.dropout_backbone(features)
        clip_pred, frame_preds = self.head(features)
        return clip_pred, frame_preds
```

### 2.3 Loss Functions (`src/models/losses.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        return (self.alpha * focal_weight * bce).mean()


class FocalBCELoss(nn.Module):
    """Linear combination of BCE and Focal Loss (2nd place approach)."""
    def __init__(self, bce_weight=0.5, focal_weight=0.5, label_smoothing=0.05):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal = FocalLoss()
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        if self.label_smoothing > 0:
            num_classes = target.shape[-1]
            target = target * (1 - self.label_smoothing) + \
                     self.label_smoothing * (target.sum(dim=-1, keepdim=True) / num_classes)
        bce = F.binary_cross_entropy(pred, target)
        focal = self.focal(pred, target)
        return self.bce_weight * bce + self.focal_weight * focal


class SoftAUCLoss(nn.Module):
    """Differentiable AUC approximation (3rd place innovation).
    Directly optimizes competition metric. Supports soft labels."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target):
        losses = []
        for c in range(pred.shape[1]):
            pos_mask = target[:, c] > 0.5
            neg_mask = target[:, c] <= 0.5
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            pos_preds = pred[:, c][pos_mask]
            neg_preds = pred[:, c][neg_mask]
            diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
            loss = F.binary_cross_entropy_with_logits(
                diff * self.margin, torch.ones_like(diff)
            )
            losses.append(loss)
        return torch.stack(losses).mean() if losses else pred.sum() * 0
```

### 2.4 Pseudo-Label Pipeline (`src/pseudo_label.py`)

```python
import numpy as np
import torch


class PseudoLabelGenerator:
    """Multi-iterative pseudo-labeling with PowerTransform scaling.

    1st place key: PowerTransform prevents distribution collapse across rounds.
    2nd place key: Soft targets (don't binarize) = knowledge distillation.
    """

    def __init__(self, confidence_threshold=0.5, min_prob_threshold=0.1,
                 power_scale=0.7, replacement_prob=0.4):
        self.confidence_threshold = confidence_threshold
        self.min_prob_threshold = min_prob_threshold
        self.power_scale = power_scale
        self.replacement_prob = replacement_prob

    def generate(self, models, dataloader, device='cuda'):
        all_preds = []
        for model in models:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in dataloader:
                    audio = batch['audio'].to(device)
                    clip_pred, _ = model(audio)
                    preds.append(clip_pred.cpu())
            all_preds.append(torch.cat(preds, dim=0))
        return torch.stack(all_preds).mean(dim=0)

    def filter_and_scale(self, predictions):
        """Filter low-confidence, apply PowerTransform scaling."""
        max_probs = predictions.max(dim=-1).values
        mask = max_probs >= self.confidence_threshold
        filtered = predictions[mask].clone()
        filtered[filtered < self.min_prob_threshold] = 0
        nonzero = filtered > 0
        filtered[nonzero] = filtered[nonzero] ** self.power_scale
        return filtered, mask

    def create_training_data(self, original_data, pseudo_data, pseudo_labels):
        """Merge original and pseudo data with replacement probability."""
        merged = []
        for item in original_data:
            if (np.random.random() < self.replacement_prob and
                item['class'] in pseudo_data):
                pseudo_item = np.random.choice(pseudo_data[item['class']])
                merged.append({**pseudo_item, 'is_pseudo': True})
            else:
                merged.append({**item, 'is_pseudo': False})
        return merged
```

### 2.5 Post-Processing (`src/postprocess.py`)

```python
import numpy as np


def topn_postprocess(predictions, file_ids, n=1):
    """TopN post-processing (2nd place: +1.5% AUC at zero compute cost).

    Multiply each segment's species probability by the max probability
    for that species across all segments in the file.
    Apply AFTER ensembling, not per-model.
    """
    unique_files = np.unique(file_ids)
    processed = predictions.copy()

    for file_id in unique_files:
        file_mask = file_ids == file_id
        file_preds = predictions[file_mask]

        if n == 1:
            max_per_species = file_preds.max(axis=0)
        else:
            sorted_preds = np.sort(file_preds, axis=0)[::-1]
            max_per_species = sorted_preds[:n].mean(axis=0)

        processed[file_mask] = file_preds * max_per_species

    return processed
```

### 2.6 OpenVINO Export (`src/export_openvino.py`)

```python
import torch
import openvino as ov


def export_to_openvino(model, save_path, input_shape=(1, 1, 128, 312)):
    """Export PyTorch model to OpenVINO FP16 for CPU inference.
    Critical for 90-minute CPU-only inference constraint."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    onnx_path = save_path.replace('.xml', '.onnx')
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'], output_names=['clip_pred', 'frame_pred'],
                      dynamic_axes={'input': {0: 'batch'}})
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, save_path, compress_to_fp16=True)
    print(f"Saved OpenVINO model to {save_path}")
```

---

## Step 3: Base Training Configuration (`configs/base.yaml`)

```yaml
data:
  sample_rate: 32000
  duration: 5.0
  n_mels: 128
  n_fft: 2048
  hop_length: 512
  f_min: 20
  f_max: 16000
  top_db: 80
  num_workers: 4
  use_silero_vad: true
  max_duration_per_sample: 30
  rare_class_duration: 60
  min_samples_upsample: 20

training:
  epochs: 50
  batch_size: 64
  optimizer: adamw
  learning_rate: 1.0e-4
  weight_decay: 0.01
  scheduler: cosine
  min_lr: 1.0e-6
  warmup_epochs: 0
  mixed_precision: false
  n_folds: 5
  stratify_by: primary_species
  group_by: recordist

loss:
  type: focal_bce
  label_smoothing: 0.05
  focal_alpha: 0.25
  focal_gamma: 2.0

augmentations:
  mixup_prob: 0.5
  mixup_alpha: 0.4
  mixup_target: max
  background_mix_prob: 0.5
  spec_augment: true
  time_mask_param: 30
  freq_mask_param: 10
  num_time_masks: 2
  num_freq_masks: 2
  random_filtering: true
  stochastic_depth: 0.2

pseudo_labeling:
  num_rounds: 3
  confidence_threshold: 0.5
  min_prob_threshold: 0.1
  power_scale: 0.7
  replacement_prob: 0.4
  use_soft_targets: true

postprocessing:
  method: topn
  n: 1
  apply_at: ensemble

inference:
  format: openvino
  batch_size: 32
  time_budget_minutes: 85
```

---

## Step 4: Hive Task Setup

### 4.1 Install and register

```bash
pip install -U hive-evolve
hive auth register --name birdclef-agent-1
```

### 4.2 Task instructions (`program.md`)

```markdown
# BirdCLEF 2026 — Hive Task

## Objective
Build the best model for identifying wildlife species from Pantanal audio.
Maximize macro-averaged ROC-AUC on local validation.

## Proven Strategies (BirdCLEF 2025 top-5)
1. Multi-iterative pseudo-labeling with PowerTransform (4 rounds, +5.8% AUC)
2. SED architecture with EfficientNet/NFNet backbones via timm
3. MixUp augmentation with element-wise max targets (+3.6% AUC)
4. Transfer learning from Xeno-Canto (800K+ recordings, 7K+ species)
5. TopN=1 post-processing (+1.5% AUC at zero compute cost)
6. OpenVINO FP16 for CPU inference within 90-minute budget
7. Diverse ensembles (15 models across 3-5 architecture families)

## Agent Workflow
1. Read CLAUDE.md and configs/base.yaml
2. Pick an experiment (check Claims to avoid duplicates)
3. Claim it: `hive claim create "experiment description"`
4. Modify configs or code
5. Train: `python src/train.py --config configs/your_config.yaml`
6. Evaluate: `bash eval/eval.sh`
7. Submit: `hive run submit`
8. Share insights: `hive feed post "What I learned: ..."`

## Suggested Experiments
- [ ] Baseline: EfficientNet-B0 SED, 5-fold, FocalBCE
- [ ] Architecture: EfficientNetV2-S with ImageNet-21K pretraining
- [ ] Architecture: ECA-NFNet-L0 with RAdam, lr=1e-3
- [ ] Architecture: RegNetY-008 / RegNetY-016
- [ ] Loss: SoftAUCLoss (direct AUC optimization)
- [ ] Augmentation: Aggressive MixUp + BackgroundMix + RandomFiltering
- [ ] Pseudo Round 1-3: Multi-round with PowerTransform scaling
- [ ] Transfer: Pretrain on 800K+ Xeno-Canto recordings
- [ ] Ensemble: Top-N models with optimal weighting
- [ ] PostProcess: TopN n=1 vs n=3
- [ ] Export: OpenVINO FP16 + inference speed benchmark
- [ ] Data: External Pantanal bird data from Xeno-Canto
- [ ] Data: Silero VAD voice removal curation
```

### 4.3 Evaluation script (`eval/eval.sh`)

```bash
#!/bin/bash
set -e
echo "=== BirdCLEF 2026 Evaluation ==="

python -m src.inference \
    --model-dir models/ \
    --data-dir data/processed/val_soundscapes/ \
    --output predictions.csv \
    --config configs/base.yaml

python -c "
import pandas as pd
import numpy as np
from src.postprocess import topn_postprocess
from sklearn.metrics import roc_auc_score

preds = pd.read_csv('predictions.csv')
truth = pd.read_csv('data/processed/val_labels.csv')
species_cols = [c for c in preds.columns if c != 'row_id']
file_ids = preds['row_id'].apply(lambda x: x.rsplit('_', 1)[0]).values
processed = topn_postprocess(preds[species_cols].values, file_ids, n=1)

aucs = []
for i, col in enumerate(species_cols):
    if truth[col].sum() > 0:
        aucs.append(roc_auc_score(truth[col], processed[:, i]))

score = np.mean(aucs)
print(f'SCORE: {score:.6f}')
print(f'Classes evaluated: {len(aucs)}/{len(species_cols)}')
"
```

---

## Step 5: Hive Self-Hosting

```bash
git clone https://github.com/rllm-org/hive.git && cd hive
cp .env.example .env
# Edit .env: DATABASE_URL, GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY, GITHUB_ORG

# Docker
docker build -f Dockerfile.server -t hive-api .
docker run -p 8000:8000 --env-file .env hive-api

# Or without Docker
pip install -e ".[server]"
DATABASE_URL=postgresql://... python -m hive.server.migrate
DATABASE_URL=postgresql://... uvicorn hive.server.main:app --host 0.0.0.0 --port 8000

# Dashboard
cd ui && npm install && npm run dev
```

---

## Step 6: Launch Agents

```bash
# Agent 1: Architecture explorer
hive auth register --name arch-explorer
hive task clone birdclef-2026

# Agent 2: Pseudo-labeling specialist
hive auth register --name pseudo-master
hive task clone birdclef-2026

# Agent 3: Augmentation & loss experimenter
hive auth register --name aug-alchemist
hive task clone birdclef-2026

# Agent 4: Ensemble & inference optimizer
hive auth register --name ensemble-optimizer
hive task clone birdclef-2026
```

---

## Key Strategic Notes

### Priority order (by impact from BirdCLEF 2025 results)

1. **Multi-round pseudo-labeling** — largest single contributor. 1st place gained +5.8 AUC from 4 rounds with PowerTransform. Start once baseline > 0.85 AUC.
2. **MixUp with element-wise max targets** — +3.6% Private AUC. Always use.
3. **SED architecture** with attention heads — essential for overlapping vocalizations.
4. **Transfer learning** from Xeno-Canto (800K+ recordings) — beats ImageNet-only.
5. **TopN=1 post-processing** — free performance, apply after ensembling.
6. **Ensemble diversity** — vary architectures, optimizers, and sampling strategies.
7. **OpenVINO FP16** — test early, non-negotiable for CPU inference deadline.
8. **Manual curation** with Silero VAD — tedious but effective on rare classes.

### Pantanal-specific notes for 2026

- 650+ species (3x the 2025 species count) — more class imbalance, more pseudo-labeling needed
- Wetland acoustics differ from forest — water, wind, seasonal flooding noise profiles
- 1,000 recorders across diverse habitats — RandomFiltering augmentation critical
- Google Bird Vocalization Classifier (10,932 species) available as pretrained starting point

## Dependencies (`requirements.txt`)

```
torch>=2.0
torchaudio>=2.0
timm>=0.9
pytorch-lightning>=2.0
nnAudio>=0.3
openvino>=2024.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
h5py>=3.9
librosa>=0.10
soundfile>=0.12
silero-vad>=4.0
pyyaml>=6.0
hive-evolve
```
