# THEORY.md — Architectural Decision Log

## Why SED (Sound Event Detection)?

BirdCLEF requires identifying species from 5-second audio segments extracted from
longer soundscape recordings. Multiple species often vocalize simultaneously.
SED with attention-based pooling learns which time frames contain vocalizations
and weights them accordingly — critical when most of a clip is silence or noise.

All top-5 BirdCLEF 2025 solutions used SED architectures.

## Why these backbone families?

**EfficientNet (B0, B3, V2-S, V2-B3):** Best accuracy-efficiency tradeoff for
audio classification. The `_ns` (noisy student) variants provide better
generalization from semi-supervised pretraining. V2-S with ImageNet-21K
pretraining gives richer features for fine-tuning.

**ECA-NFNet-L0:** Normalizer-free network avoids batch normalization artifacts
common in small-batch audio training. ECA (Efficient Channel Attention) adds
lightweight channel attention. Works well with RAdam optimizer at higher LR (1e-3).

**RegNetY-008:** Lightweight architecture for ensemble diversity. Different
computational profile than EfficientNet family, providing decorrelated predictions
for ensembling.

## Why FocalBCE as default loss?

650+ species with extreme class imbalance (some species have <5 recordings).
Standard BCE treats all classes equally — Focal Loss down-weights easy negatives.
The BCE+Focal blend (2nd place approach) prevents over-suppression of easy examples
that pure Focal Loss can cause. Label smoothing (0.05) regularizes against
overconfident predictions on noisy labels.

SoftAUCLoss is available as an alternative for direct metric optimization,
but is more computationally expensive and can be unstable early in training.

## Why MixUp with element-wise max targets?

Standard MixUp uses weighted target combination: `y = λy₁ + (1-λ)y₂`. This
fails for multilabel classification because it implies the mixed audio contains
"fractional" species presence. Element-wise max `y = max(y₁, y₂)` correctly
models that both species are present in the mixed audio.

2nd place BirdCLEF 2025 reported +3.6% AUC from this single change.

## Why multi-round pseudo-labeling?

The competition provides ~240 hours of unlabeled soundscape audio.
Self-training via pseudo-labeling is the largest single performance contributor:
1st place gained +5.8% AUC from 4 rounds.

**PowerTransform (x^0.7)** is critical: without it, predictions collapse toward
0 and 1 across rounds, losing calibration. Power scaling with exponent < 1
pushes predictions toward the center, maintaining useful soft-target information.

**Soft targets** (not binarized) act as knowledge distillation — the model
learns the ensemble's uncertainty, not just binary decisions.

## Why OpenVINO FP16?

The competition requires CPU-only inference within 90 minutes. OpenVINO FP16
provides ~2-3x speedup over PyTorch CPU inference through graph optimizations
and reduced-precision arithmetic. This enables larger ensembles (15+ models)
within the time budget.

## Why TopN=1 post-processing?

For each soundscape file, multiply each segment's predictions by the file-level
max per species. This suppresses species that appear weakly in single segments
but aren't consistently detected across the file. Zero compute cost, +1.5% AUC.

Applied after ensembling (not per-model) because the ensemble's aggregated
signal is more reliable for this file-level correction.
