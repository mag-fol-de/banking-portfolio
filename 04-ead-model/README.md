# Exposure at Default (EAD) Model

CCF regression for revolving retail credit facilities (credit cards,
overdrafts, lines of credit). Synthetic panel of $\sim$20,000
defaulted facilities across ten origination vintages (2019H1--2023H2).

## Methodology

- Target: Credit Conversion Factor,
  `CCF = (EAD - drawn_t0) / (limit - drawn_t0)`, then derive
  `EAD = drawn + CCF * (limit - drawn)`
- Features: log limit, log drawn, utilisation at t_0, time to default,
  risk score, minimum-payment-ratio, borrower age, product,
  vintage macro (unemployment, GDP YoY)
- Three models compared: fractional logit, OLS on logit(CCF), XGBoost
- Temporal split: train 2019H1--2022H2, out-of-time 2023H1--2023H2
- Downturn CCF via 90th percentile of predicted CCF per product
- PSI per feature train vs OOT

## Headline results (OOT)

- Fractional logit CCF: MAE 0.127, RMSE 0.157, $R^2$ 0.31
- Derived EAD $R^2$ 0.98 (mean pred 9,325 EUR vs mean true 9,277 EUR)
- Mean-level bias $\sim$0.5\% at portfolio level

## Files

- `generate_data.py` --- synthetic revolving-credit panel
- `train.py` --- CCF pipeline with derived EAD
- `report.tex` / `report.pdf` --- full report
- `data/`, `outputs/`, `figures/`

## Reproducibility

```bash
python generate_data.py
python train.py
```
