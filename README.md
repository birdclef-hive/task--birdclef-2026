# BirdCLEF+ 2026 — Hive Automated ML Experimentation

Identify 650+ wildlife species from audio recordings in Brazil's Pantanal wetlands.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.train --config configs/base.yaml --fold 0
bash eval/eval.sh
```

## Architecture

- **SED models** with timm backbones (EfficientNet, NFNet, RegNet)
- **Multi-round pseudo-labeling** with PowerTransform scaling
- **MixUp** with element-wise max targets
- **OpenVINO FP16** for CPU-only inference (90-min budget)
- **Hive** for multi-agent experiment coordination

See `CLAUDE.md` for full details.
