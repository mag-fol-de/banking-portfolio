# Loss Given Default (LGD) Model

Retail LGD scorecard on $\sim$15,000 defaulted facilities across
mortgage, auto, personal secured, and unsecured collateral. Ten
origination vintages (2019H1--2023H2). Built with Basel IRB downturn
requirements and IFRS 9 lifetime LGD needs in mind.

## Methodology

- Feature set: collateral type, EAD, LTV at default, DPD at workout
  start, workout duration, borrower age, employment status, vintage
  macro (unemployment, HPI YoY)
- Three models compared: fractional logit (statsmodels GLM),
  OLS on logit-transformed LGD, XGBoost
- Temporal split: train 2019H1--2022H2, out-of-time 2023H1--2023H2
- Downturn LGD via 90th-percentile of predicted LGD per collateral
  bucket
- PSI per feature between train and OOT
- Calibration curves at decile resolution

## Headline results (OOT)

- Fractional logit MAE 0.100, RMSE 0.126, $R^2$ 0.724
- Mean predicted 0.442 vs mean realised 0.439 (portfolio-level unbiased)
- Downturn uplifts range from +12 pp (mortgage) to +18 pp (personal
  secured) over long-run-average

## Files

- `generate_data.py` --- synthetic default panel generator
- `train.py` --- pipeline (three models, calibration, downturn, PSI)
- `report.tex` / `report.pdf` --- full report
- `data/` --- generated defaulted facilities and vintage macro
- `outputs/` --- metrics, PSI, downturn LGD CSVs
- `figures/` --- five diagnostic plots

## Reproducibility

```bash
python generate_data.py
python train.py
```
