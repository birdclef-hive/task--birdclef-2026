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
