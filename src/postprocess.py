"""Post-processing for BirdCLEF 2026 predictions.

TopN post-processing from 2nd place BirdCLEF 2025 (+1.5% AUC at zero cost).
Applied AFTER ensembling, not per-model.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def topn_postprocess(
    predictions: np.ndarray,
    file_ids: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    """TopN post-processing: multiply segment predictions by file-level max.

    For each soundscape file, multiply each segment's species probability by
    the max (or top-N mean) probability for that species across all segments.
    This suppresses species that appear weakly in isolated segments.

    Args:
        predictions: (num_segments, num_classes) prediction matrix.
        file_ids: (num_segments,) array of file identifiers.
        n: Number of top predictions to average. n=1 uses the max.

    Returns:
        Post-processed predictions of the same shape.
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
            top_n = min(n, sorted_preds.shape[0])
            max_per_species = sorted_preds[:top_n].mean(axis=0)

        processed[file_mask] = file_preds * max_per_species

    return processed


def main() -> None:
    """CLI for applying post-processing to a predictions CSV."""
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Post-Processing")
    parser.add_argument("--input", type=str, required=True, help="Input predictions CSV")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--n", type=int, default=1, help="TopN parameter (default: 1)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    preds_df = pd.read_csv(args.input)
    species_cols = [c for c in preds_df.columns if c != "row_id"]
    file_ids = preds_df["row_id"].apply(lambda x: x.rsplit("_", 1)[0]).values

    processed = topn_postprocess(preds_df[species_cols].values, file_ids, n=args.n)

    result = preds_df.copy()
    result[species_cols] = processed
    result.to_csv(args.output, index=False)
    logger.info("Post-processed predictions saved to %s", args.output)


if __name__ == "__main__":
    main()
