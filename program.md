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
