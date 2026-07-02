# PD Scorecard (IRB-style)

Industry-grade retail Probability of Default scorecard on a synthetic
loan portfolio. WoE binning, Information Value selection, regularized
logistic regression on WoE-encoded features, calibration to long-run-
average PD, master rating scale, out-of-time backtesting.

## Run

```
python generate_data.py   # 100,000 loans across 4 origination vintages
python train.py           # WoE binning, IV selection, training, calibration, backtest
```

## Cohort design

- TRAIN:  2022H1 + 2022H2 (50,000 loans)
- VAL:    2023H1 (25,000 loans, in-time)
- OOT:    2023H2 (25,000 loans, out-of-time stress)

## Results

| Cohort | AUC    | Gini   | KS     |
|--------|--------|--------|--------|
| TRAIN  | 0.7779 | 0.5558 | 0.4159 |
| VAL    | 0.7765 | 0.5531 | 0.4227 |
| OOT    | 0.7718 | 0.5437 | 0.4086 |

PSI TRAIN vs OOT: 0.0002 (well below the 0.10 stability threshold).

## Files

- `generate_data.py` - synthetic generator with bureau attributes and vintages
- `train.py` - WoE binning, IV selection, logistic, calibration, backtest
- `data/loan_applicants.csv` - 100k loans, full schema
- `data/bank.duckdb` - DuckDB warehouse
- `figures/` - 6 diagnostic plots
- `outputs/coefficients.csv`, `metrics.json`, `grade_summary.csv`, `information_values.csv`
- `report.tex` - full report
