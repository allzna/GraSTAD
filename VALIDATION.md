# Release validation

The release was checked against the original research implementation using its
multivariate configuration: mean-centered similarity and temporal graph edges,
graph-neighbor positives and non-neighbor negatives, triplet loss, independent
channels and memory banks, 3 representation neighbors, and Mahalanobis amplitude
scoring with weight 0.3. The same configuration was used for single-channel data.

## Numerical equivalence

For each device and case below, all compared tensors were exactly equal, with
maximum absolute difference **0**:

| Device | Channels | RevIN | Compared tensors |
| --- | --- | --- | --- |
| CPU | 3 | Enabled | 38 |
| CPU | 1 | Enabled | 38 |
| CPU | 3 | Disabled | 32 |
| CUDA | 3 | Enabled | 38 |
| CUDA | 1 | Enabled | 38 |
| CUDA | 3 | Disabled | 32 |

Comparisons covered trained encoder weights, graph edges, memory banks,
Mahalanobis statistics, training-score calibration, patch scores, and point
scores. Each case used 560 synthetic training patches, four optimizer updates,
and 48 separate test patches, including injected amplitude anomalies. Coreset
selection was exercised with 500 representatives per channel.

## Integration checks

- All four bundled unit tests passed, including three training/scoring cases.
- CLI help exposes exactly the nine requested public parameters plus help.
- The complete CLI processed single-channel and multivariate synthetic CSVs,
  saved 360 point scores per file, and generated metrics and timing outputs.
- A second invocation resumed both completed files without rewriting their scores.
- All Python files passed syntax checks. Python comments and docstrings contain
  no Chinese text. The Chinese dataset directory name remains in example paths.
- Hash checks confirmed the original research source files were unchanged.

## Validation environment and scope

Checks used the existing local Python 3.10.18 environment with PyTorch
2.8.0+cu128, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.15.3,
statsmodels 0.14.5, and tqdm 4.64.1. Both versions ran in that same environment.

The original pinned `requirements.txt` is retained; it was not reinstalled for
these checks. Use Python 3.11 or 3.12 to install those pins. Validation covers
small synthetic cases and does not constitute a full TSB-AD benchmark run.
