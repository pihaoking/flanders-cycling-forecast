# Model Diagnostics: RF vs XGBoost

Two questions:

1. Is the predictive model overfitting?
2. RF or XGBoost as primary?

Setup: panel from `02_build_panel.py`, `temporal_split` (train 2023 / val 2024 / test 2025), 9 stations new in 2025 excluded from val and test. Numbers come from `src/model_diagnostics.py`.

## 1. Train-internal 5-fold CV (the overfitting test)

| Model | CV mean R² | In-sample R² | Gap | Test R² |
|---|---|---|---|---|
| RF  | 0.835 (sd 0.003) | 0.883 | 0.048 | 0.638 |
| XGB | 0.758 (sd 0.005) | 0.770 | 0.012 | 0.608 |

CV is close to in-sample for both, so neither model is overfitting the train set. The drop to test is year-over-year drift.

## 2. Per-segment RMSE on test

By day of week (RF wins all 7), by month (RF wins 11/12, XGB wins July), by weekend:

| | n | RF | XGB | Winner |
|---|---|---|---|---|
| Weekday | 1,712,718 | 17.72 | 18.60 | RF |
| Weekend |   682,416 | 13.72 | 13.81 | RF |

By hour: split (RF on rush 5-8 and 16-17 / 20-23, XGB on midday peak 12-15 and overnight 0-3).

By station volume quartile:

| Quartile | n | RF | XGB | Winner |
|---|---|---|---|---|
| Q1 (low)  | 604,966 | 11.48 | **10.41** | XGB |
| Q2        | 595,680 |  7.89 |  8.78 | RF |
| Q3        | 581,288 | 10.68 | 12.09 | RF |
| Q4 (high) | 613,200 | 28.07 | 29.30 | RF |

## 3. Residual stats on test

| Model | Bias | Std | q05 | q95 | q95 - q05 |
|---|---|---|---|---|---|
| RF  | +0.523 | 16.67 | -10.78 | 15.40 | 26.18 |
| XGB | +0.073 | 17.37 | -13.90 | 16.44 | 30.34 |

XGB nearly unbiased; RF tighter spread. Plot: `figures/model_diagnostics.png`.

## Verdict

| Criterion | Winner |
|---|---|
| Overfitting | tie (neither) |
| Test R² | RF (+0.030) |
| Bias (policy line) | XGB |
| Spread / consistency (predictive line) | RF |
| Low-volume stations | XGB |
| Mid / high-volume stations | RF |

Plan: RF primary, XGB as comparison baseline.

## Reproduce

```bash
python src/model_diagnostics.py
```
