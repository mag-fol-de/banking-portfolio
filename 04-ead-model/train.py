"""EAD model training: CCF regression on revolving credit facilities.

Fits three CCF (Credit Conversion Factor) models on the synthetic
default panel:

    1. Fractional logit (statsmodels GLM, quasi-binomial)
    2. OLS on logit-transformed CCF
    3. XGBoost on logit-transformed CCF

CCF is the fraction of undrawn commitment that gets used before
default. Once CCF is estimated, EAD follows from:

    EAD = drawn + CCF * (limit - drawn)

Temporal split: train 2019H1-2022H2, out-of-time 2023H1-2023H2.

Outputs
-------
outputs/metrics.csv       : CCF model comparison
outputs/ead_metrics.csv   : EAD prediction quality
outputs/psi_features.csv  : PSI train vs OOT
figures/*.png             : diagnostic plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG  = ROOT / "figures"
OUT  = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120})

# -----------------------------------------------------------------------------
# Load and split
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA / "ead_facilities.csv")
print(f"Loaded {len(df):,} facilities.")

OOT_VINTAGES = {"2023H1", "2023H2"}
train = df[~df["vintage"].isin(OOT_VINTAGES)].copy()
oot   = df[df["vintage"].isin(OOT_VINTAGES)].copy()
print(f"Train: {len(train):,}   OOT: {len(oot):,}")


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["log_limit"] = np.log1p(d["credit_limit_eur"])
    d["log_drawn"] = np.log1p(d["drawn_t0_eur"] + 1)
    d = pd.get_dummies(d, columns=["product"], drop_first=True)
    return d

train_p = prepare(train)
oot_p   = prepare(oot)

feature_cols = ["log_limit", "log_drawn", "utilisation_t0",
                "time_to_default_months", "risk_score_t0",
                "min_payment_ratio", "borrower_age",
                "unemployment_pct", "gdp_yoy_pct"]
feature_cols += [c for c in train_p.columns if c.startswith("product_")]

X_train = train_p[feature_cols].astype(float).to_numpy()
y_train = np.clip(train_p["ccf_realised"].to_numpy(), 1e-4, 1 - 1e-4)
X_oot   = oot_p[feature_cols].astype(float).to_numpy()
y_oot   = np.clip(oot_p["ccf_realised"].to_numpy(), 1e-4, 1 - 1e-4)

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0) + 1e-9
X_train_z = (X_train - mu) / sd
X_oot_z   = (X_oot - mu) / sd
X_train_c = sm.add_constant(X_train_z)
X_oot_c   = sm.add_constant(X_oot_z)


# -----------------------------------------------------------------------------
# Model 1: Fractional logit
# -----------------------------------------------------------------------------
print("\n>>> Fitting fractional logit")
frac_logit = sm.GLM(
    y_train, X_train_c,
    family=sm.families.Binomial(link=sm.families.links.Logit()),
).fit(scale="X2")
pred_frac = frac_logit.predict(X_oot_c)

# -----------------------------------------------------------------------------
# Model 2: OLS on logit
# -----------------------------------------------------------------------------
print(">>> Fitting OLS on logit(CCF)")
y_train_logit = np.log(y_train / (1 - y_train))
ols = sm.OLS(y_train_logit, X_train_c).fit()
pred_ols = 1.0 / (1.0 + np.exp(-ols.predict(X_oot_c)))

# -----------------------------------------------------------------------------
# Model 3: XGBoost
# -----------------------------------------------------------------------------
print(">>> Fitting XGBoost")
xgb = XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)
xgb.fit(X_train, y_train_logit)
pred_xgb = 1.0 / (1.0 + np.exp(-xgb.predict(X_oot)))


# -----------------------------------------------------------------------------
# CCF metrics
# -----------------------------------------------------------------------------
def metric_row(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "mean_pred": float(np.mean(y_pred)),
        "mean_true": float(np.mean(y_true)),
    }

rows = [
    metric_row("FractionalLogit", y_oot, pred_frac),
    metric_row("OLS-logit",       y_oot, pred_ols),
    metric_row("XGBoost",         y_oot, pred_xgb),
]
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "metrics.csv", index=False)
print("\n== OOT CCF metrics ==")
print(metrics.to_string(index=False))


# -----------------------------------------------------------------------------
# Derived EAD prediction
# -----------------------------------------------------------------------------
def ead_from_ccf(ccf, drawn, limit):
    return drawn + ccf * (limit - drawn)

ead_true = oot["ead_eur"].to_numpy()
drawn    = oot["drawn_t0_eur"].to_numpy()
limit_v  = oot["credit_limit_eur"].to_numpy()

ead_frac = ead_from_ccf(pred_frac, drawn, limit_v)
ead_ols  = ead_from_ccf(pred_ols, drawn, limit_v)
ead_xgb  = ead_from_ccf(pred_xgb, drawn, limit_v)

ead_rows = []
for name, pred_ead in [("FractionalLogit", ead_frac),
                       ("OLS-logit", ead_ols),
                       ("XGBoost", ead_xgb)]:
    ead_rows.append({
        "model": name,
        "MAE_EUR": mean_absolute_error(ead_true, pred_ead),
        "RMSE_EUR": float(np.sqrt(mean_squared_error(ead_true, pred_ead))),
        "R2": r2_score(ead_true, pred_ead),
        "mean_pred_EAD": float(np.mean(pred_ead)),
        "mean_true_EAD": float(np.mean(ead_true)),
    })
ead_metrics = pd.DataFrame(ead_rows)
ead_metrics.to_csv(OUT / "ead_metrics.csv", index=False)
print("\n== OOT EAD metrics ==")
print(ead_metrics.to_string(index=False))


# -----------------------------------------------------------------------------
# CCF calibration (deciles)
# -----------------------------------------------------------------------------
def calib_curve(y_true, y_pred, bins=10):
    d = pd.DataFrame({"y": y_true, "p": y_pred})
    d["bucket"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    return d.groupby("bucket", observed=True).agg(
        mean_pred=("p", "mean"), mean_true=("y", "mean"),
    )

fig, ax = plt.subplots(figsize=(7.5, 6))
for name, pred in [("FractionalLogit", pred_frac),
                   ("OLS-logit", pred_ols),
                   ("XGBoost", pred_xgb)]:
    c = calib_curve(y_oot, pred, bins=10)
    ax.plot(c["mean_pred"], c["mean_true"], marker="o", label=name)
ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, label="Perfect")
ax.set_xlabel("Predicted CCF"); ax.set_ylabel("Realised CCF")
ax.set_title("Out-of-time CCF calibration (deciles)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "01_ccf_calibration.png"); plt.close()

# -----------------------------------------------------------------------------
# CCF by product bucket
# -----------------------------------------------------------------------------
oot_view = oot.copy()
oot_view["pred_ccf"] = pred_frac
prod_summary = oot_view.groupby("product").agg(
    mean_true=("ccf_realised", "mean"),
    mean_pred=("pred_ccf", "mean"),
    n=("ccf_realised", "size"),
).reset_index()
prod_summary.to_csv(OUT / "ccf_by_product.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(prod_summary)); w = 0.35
ax.bar(x - w/2, prod_summary["mean_true"], w, label="Realised", color="#2b6b8b")
ax.bar(x + w/2, prod_summary["mean_pred"], w, label="Predicted", color="#e0a04a")
ax.set_xticks(x); ax.set_xticklabels(prod_summary["product"], rotation=10)
ax.set_ylabel("CCF"); ax.set_title("Realised vs predicted CCF by product (OOT)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "02_ccf_by_product.png"); plt.close()

# -----------------------------------------------------------------------------
# EAD scatter true vs predicted (log-log)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(ead_true, ead_frac, alpha=0.15, s=6, color="#2b6b8b")
lim = max(ead_true.max(), ead_frac.max()) * 1.05
ax.plot([0, lim], [0, lim], color="black", ls="--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Realised EAD (EUR)"); ax.set_ylabel("Predicted EAD (EUR)")
ax.set_title("EAD prediction: realised vs predicted (log-log)")
ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(FIG / "03_ead_scatter.png"); plt.close()

# -----------------------------------------------------------------------------
# Downturn CCF: P90 per product
# -----------------------------------------------------------------------------
lra_ccf = train.groupby("product")["ccf_realised"].mean()
downturn_p90 = oot_view.groupby("product")["pred_ccf"].quantile(0.90)
downturn = pd.DataFrame({
    "product": lra_ccf.index,
    "LRA_CCF": lra_ccf.values,
    "downturn_p90": downturn_p90.reindex(lra_ccf.index).values,
})
downturn["uplift"] = downturn["downturn_p90"] - downturn["LRA_CCF"]
downturn.to_csv(OUT / "downturn_ccf.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(downturn))
ax.bar(x - w/2, downturn["LRA_CCF"], w, label="LRA", color="#2b6b8b")
ax.bar(x + w/2, downturn["downturn_p90"], w, label="Downturn P90", color="#c14e4e")
ax.set_xticks(x); ax.set_xticklabels(downturn["product"], rotation=10)
ax.set_ylabel("CCF"); ax.set_title("Downturn CCF vs long-run-average")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "04_downturn_ccf.png"); plt.close()

# -----------------------------------------------------------------------------
# Feature importance
# -----------------------------------------------------------------------------
importance = pd.DataFrame({
    "feature": feature_cols, "importance": xgb.feature_importances_,
}).sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance["feature"], importance["importance"], color="#5a8b6a")
ax.set_xlabel("XGBoost gain importance")
ax.set_title("Feature importance (CCF model)")
plt.tight_layout(); plt.savefig(FIG / "05_feature_importance.png"); plt.close()


# -----------------------------------------------------------------------------
# PSI
# -----------------------------------------------------------------------------
def psi(a, b, bins=10):
    lo, hi = a.min(), a.max()
    edges = np.linspace(lo, hi, bins + 1)
    a_pct = np.histogram(a, bins=edges)[0] / max(len(a), 1) + 1e-6
    b_pct = np.histogram(b, bins=edges)[0] / max(len(b), 1) + 1e-6
    return float(np.sum((a_pct - b_pct) * np.log(a_pct / b_pct)))

psi_rows = []
for col in ["credit_limit_eur", "drawn_t0_eur", "utilisation_t0",
            "time_to_default_months", "risk_score_t0",
            "min_payment_ratio", "borrower_age",
            "unemployment_pct", "gdp_yoy_pct"]:
    psi_rows.append({"feature": col,
                     "psi": psi(train[col].values, oot[col].values)})
psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
psi_df.to_csv(OUT / "psi_features.csv", index=False)
print("\n== PSI train vs OOT (top 5) ==")
print(psi_df.head())

# -----------------------------------------------------------------------------
# Headline
# -----------------------------------------------------------------------------
best_ead = ead_metrics.iloc[0]
headline = {
    "n_train": int(len(train)),
    "n_oot": int(len(oot)),
    "mean_ccf_train": float(train["ccf_realised"].mean()),
    "mean_ccf_oot": float(oot["ccf_realised"].mean()),
    "chosen_model": "FractionalLogit",
    "ccf_oot_mae": float(metric_row("_", y_oot, pred_frac)["MAE"]),
    "ccf_oot_rmse": float(metric_row("_", y_oot, pred_frac)["RMSE"]),
    "ccf_oot_r2": float(metric_row("_", y_oot, pred_frac)["R2"]),
    "ead_oot_r2_frac": float(best_ead["R2"]),
    "max_psi": float(psi_df["psi"].max()),
}
pd.Series(headline).to_csv(OUT / "headline.csv")
print("\nHeadline:")
for k, v in headline.items():
    print(f"  {k}: {v}")

print("\nDone.")
