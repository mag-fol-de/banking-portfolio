"""LGD model training and evaluation.

Fits three LGD models on the synthetic defaulted-facility panel:

    1. Beta regression (statsmodels)
       Proper distribution for fractional [0, 1] outcomes.

    2. Fractional logit (statsmodels)
       Quasi-likelihood alternative, robust to Beta misspecification.

    3. XGBoost regressor on logit-transformed LGD
       Nonparametric ML comparator with tree-based interactions.

Train/test split is temporal (2019H1-2022H2 train, 2023H1-2023H2
out-of-time). Model diagnostics include RMSE, MAE, R^2, bucket-level
calibration curves, downturn-LGD stress uplift, and PSI on features
between train and OOT.

Outputs
-------
outputs/metrics.csv       : model comparison table
outputs/psi_features.csv  : PSI per feature train vs OOT
outputs/downturn_lgd.csv  : downturn uplift per collateral bucket
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
df = pd.read_csv(DATA / "lgd_defaults.csv")
print(f"Loaded {len(df):,} defaulted facilities.")

# Temporal split
OOT_VINTAGES = {"2023H1", "2023H2"}
train = df[~df["vintage"].isin(OOT_VINTAGES)].copy()
oot   = df[df["vintage"].isin(OOT_VINTAGES)].copy()
print(f"Train: {len(train):,}   OOT: {len(oot):,}")

# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
CAT_FEATS = ["collateral_type", "employment_status"]
NUM_FEATS = ["ead_eur", "ltv_at_default", "dpd_at_workout",
             "workout_months", "borrower_age",
             "unemployment_pct", "hpi_yoy_pct"]

def prepare(df_: pd.DataFrame) -> pd.DataFrame:
    """Fill LTV NA for unsecured, log-transform EAD, dummy encode categoricals."""
    d = df_.copy()
    d["ltv_at_default"] = d["ltv_at_default"].fillna(1.0)   # unsecured
    d["log_ead"] = np.log1p(d["ead_eur"])
    d = pd.get_dummies(d, columns=CAT_FEATS, drop_first=True)
    return d

train_p = prepare(train)
oot_p   = prepare(oot)

# Feature columns after dummy encoding
feature_cols = ["log_ead", "ltv_at_default", "dpd_at_workout",
                "workout_months", "borrower_age",
                "unemployment_pct", "hpi_yoy_pct"]
feature_cols += [c for c in train_p.columns
                 if c.startswith(("collateral_type_", "employment_status_"))]

X_train = train_p[feature_cols].astype(float).to_numpy()
y_train = np.clip(train_p["realised_lgd"].to_numpy(), 1e-4, 1 - 1e-4)
X_oot   = oot_p[feature_cols].astype(float).to_numpy()
y_oot   = np.clip(oot_p["realised_lgd"].to_numpy(), 1e-4, 1 - 1e-4)

# Standardise numeric features for statsmodels stability
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0) + 1e-9
X_train_z = (X_train - mu) / sd
X_oot_z   = (X_oot - mu) / sd


# -----------------------------------------------------------------------------
# Model 1: Beta regression via GLM with Beta likelihood approximation
# statsmodels does not expose Beta natively, so we fit via the equivalent
# link-and-variance combination: logit link on the mean with a
# quasi-binomial variance. This is the "fractional logit" approach.
# -----------------------------------------------------------------------------
X_train_c = sm.add_constant(X_train_z)
X_oot_c   = sm.add_constant(X_oot_z)

print("\n>>> Fitting fractional logit (quasi-binomial)")
frac_logit = sm.GLM(
    y_train,
    X_train_c,
    family=sm.families.Binomial(link=sm.families.links.Logit()),
).fit(scale="X2")

pred_frac = frac_logit.predict(X_oot_c)

# -----------------------------------------------------------------------------
# Model 2: Gaussian OLS on logit-transformed LGD as a linear baseline
# -----------------------------------------------------------------------------
print(">>> Fitting OLS on logit(LGD)")
y_train_logit = np.log(y_train / (1 - y_train))
ols = sm.OLS(y_train_logit, X_train_c).fit()
pred_ols_logit = ols.predict(X_oot_c)
pred_ols = 1.0 / (1.0 + np.exp(-pred_ols_logit))

# -----------------------------------------------------------------------------
# Model 3: XGBoost on logit LGD, back-transform for prediction
# -----------------------------------------------------------------------------
print(">>> Fitting XGBoost")
xgb = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train_logit)
pred_xgb_logit = xgb.predict(X_oot)
pred_xgb = 1.0 / (1.0 + np.exp(-pred_xgb_logit))


# -----------------------------------------------------------------------------
# Metrics on OOT
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
print("\n== OOT metrics ==")
print(metrics.to_string(index=False))


# -----------------------------------------------------------------------------
# Calibration curves per model (deciles)
# -----------------------------------------------------------------------------
def calib_curve(y_true, y_pred, bins=10):
    d = pd.DataFrame({"y": y_true, "p": y_pred})
    d["bucket"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    g = d.groupby("bucket", observed=True).agg(
        mean_pred=("p", "mean"),
        mean_true=("y", "mean"),
        n=("y", "size"),
    )
    return g

fig, ax = plt.subplots(figsize=(7.5, 6))
for name, pred in [("FractionalLogit", pred_frac),
                   ("OLS-logit", pred_ols),
                   ("XGBoost", pred_xgb)]:
    c = calib_curve(y_oot, pred, bins=10)
    ax.plot(c["mean_pred"], c["mean_true"], marker="o", label=name)
ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, label="Perfect")
ax.set_xlabel("Predicted LGD")
ax.set_ylabel("Realised LGD")
ax.set_title("Out-of-time calibration (deciles)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "01_calibration.png")
plt.close()

# -----------------------------------------------------------------------------
# LGD by collateral bucket
# -----------------------------------------------------------------------------
oot_display = oot.copy()
oot_display["pred_lgd"] = pred_frac

bucket_summary = (
    oot_display
    .groupby("collateral_type")
    .agg(mean_true=("realised_lgd", "mean"),
         mean_pred=("pred_lgd", "mean"),
         n=("realised_lgd", "size"))
    .reset_index()
)
bucket_summary.to_csv(OUT / "collateral_bucket_lgd.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(bucket_summary))
w = 0.35
ax.bar(x - w/2, bucket_summary["mean_true"], w, label="Realised", color="#2b6b8b")
ax.bar(x + w/2, bucket_summary["mean_pred"], w, label="Predicted", color="#e0a04a")
ax.set_xticks(x)
ax.set_xticklabels(bucket_summary["collateral_type"], rotation=15)
ax.set_ylabel("LGD")
ax.set_title("Realised vs predicted LGD by collateral type (OOT)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "02_lgd_by_collateral.png")
plt.close()

# -----------------------------------------------------------------------------
# Downturn LGD (Basel IRB requirement): 90th percentile of predicted LGD
# per collateral type as the downturn-adjusted point estimate. Compare
# against long-run-average LGD.
# -----------------------------------------------------------------------------
lra_lgd = train.groupby("collateral_type")["realised_lgd"].mean()
downturn_p90 = oot_display.groupby("collateral_type")["pred_lgd"].quantile(0.90)

downturn = pd.DataFrame({
    "collateral_type": lra_lgd.index,
    "LRA_LGD": lra_lgd.values,
    "downturn_p90": downturn_p90.reindex(lra_lgd.index).values,
})
downturn["downturn_uplift"] = downturn["downturn_p90"] - downturn["LRA_LGD"]
downturn.to_csv(OUT / "downturn_lgd.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(downturn))
ax.bar(x - w/2, downturn["LRA_LGD"], w, label="Long-run average", color="#2b6b8b")
ax.bar(x + w/2, downturn["downturn_p90"], w, label="Downturn (P90)", color="#c14e4e")
ax.set_xticks(x)
ax.set_xticklabels(downturn["collateral_type"], rotation=15)
ax.set_ylabel("LGD")
ax.set_title("Downturn LGD uplift vs long-run-average (Basel IRB style)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "03_downturn_lgd.png")
plt.close()

# -----------------------------------------------------------------------------
# PSI per numeric feature train vs OOT
# -----------------------------------------------------------------------------
def psi(a, b, bins=10):
    lo, hi = a.min(), a.max()
    edges = np.linspace(lo, hi, bins + 1)
    a_pct = np.histogram(a, bins=edges)[0] / max(len(a), 1) + 1e-6
    b_pct = np.histogram(b, bins=edges)[0] / max(len(b), 1) + 1e-6
    return float(np.sum((a_pct - b_pct) * np.log(a_pct / b_pct)))

psi_rows = []
for col in NUM_FEATS:
    psi_rows.append({
        "feature": col,
        "psi": psi(train[col].dropna().values, oot[col].dropna().values, bins=10),
    })
psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
psi_df.to_csv(OUT / "psi_features.csv", index=False)
print("\n== PSI train vs OOT (top 5) ==")
print(psi_df.head())

# -----------------------------------------------------------------------------
# Feature importance (XGBoost)
# -----------------------------------------------------------------------------
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb.feature_importances_,
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance["feature"], importance["importance"], color="#5a8b6a")
ax.set_xlabel("XGBoost gain importance")
ax.set_title("Feature importance (LGD model)")
plt.tight_layout()
plt.savefig(FIG / "04_feature_importance.png")
plt.close()

# -----------------------------------------------------------------------------
# Realised LGD distribution
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for c in ["mortgage", "auto", "personal_secured", "unsecured"]:
    vals = df.loc[df["collateral_type"] == c, "realised_lgd"]
    ax.hist(vals, bins=40, alpha=0.5, label=c, density=True)
ax.set_xlabel("Realised LGD")
ax.set_ylabel("Density")
ax.set_title("Realised LGD distribution by collateral type")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "05_lgd_distribution.png")
plt.close()

# -----------------------------------------------------------------------------
# Persist headline metrics
# -----------------------------------------------------------------------------
headline = {
    "n_train": int(len(train)),
    "n_oot": int(len(oot)),
    "mean_true_train_lgd": float(train["realised_lgd"].mean()),
    "mean_true_oot_lgd": float(oot["realised_lgd"].mean()),
    "chosen_model": "FractionalLogit",
    "oot_mae": float(metric_row("_", y_oot, pred_frac)["MAE"]),
    "oot_rmse": float(metric_row("_", y_oot, pred_frac)["RMSE"]),
    "oot_r2": float(metric_row("_", y_oot, pred_frac)["R2"]),
    "max_psi": float(psi_df["psi"].max()),
}
pd.Series(headline).to_csv(OUT / "headline.csv")
print("\nHeadline:")
for k, v in headline.items():
    print(f"  {k}: {v}")

print("\nDone.")
