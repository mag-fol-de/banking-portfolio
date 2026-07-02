"""IRB-style PD scorecard with WoE binning, IV-based feature selection,
calibration to long-run-average PD, and out-of-time backtesting.

Cohort design:
- TRAIN:        2022H1 + 2022H2 (50k loans)
- VALIDATION:   2023H1 (25k loans), in-time
- OUT-OF-TIME:  2023H2 (25k loans), out-of-time stress test

Pipeline:
1. Load from DuckDB, mirror the dbt mart in-memory.
2. Numeric WoE binning via quantile cuts, categorical WoE encoding.
3. Information Value per feature, drop weak features (IV < 0.02).
4. L2-regularized logistic on WoE-encoded features.
5. Calibrate predicted PD intercept to long-run-average default rate.
6. Map calibrated PDs to a master rating scale (A1..C3).
7. Backtest on OOT vintage: discrimination, calibration, PSI.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SEED = 42
TRAIN_VINTAGES = ["2019H1", "2019H2", "2020H1", "2020H2", "2021H1", "2021H2"]
VAL_VINTAGES = ["2022H1", "2022H2"]
OOT_VINTAGES = ["2023H1", "2023H2"]

NUMERIC_FEATURES = [
    "age", "annual_income", "employment_length_years",
    "loan_amount", "loan_term_months", "interest_rate_pct",
    "debt_to_income_pct", "credit_history_length_years",
    "num_credit_lines", "delinquencies_2y", "worst_status_24m",
    "months_since_last_delinquency", "inquiries_6m", "utilization_pct",
    "public_records", "prior_bankruptcies",
    "bureau_score",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
con = duckdb.connect(str(DATA / "bank.duckdb"), read_only=True)
df = con.execute("SELECT * FROM raw_loan_applicants").df()
con.close()

print(f"Loaded {len(df):,} loans across {df['origination_vintage'].nunique()} vintages")
print(df["origination_vintage"].value_counts().sort_index())

# Missingness flags BEFORE imputation, only where missingness signals risk
df["annual_income_missing"] = df["annual_income"].isna().astype(int)
df["employment_length_years_missing"] = df["employment_length_years"].isna().astype(int)
df["months_since_last_delinquency_missing"] = df["months_since_last_delinquency"].isna().astype(int)

# Impute with vintage-specific medians to respect the cohort boundary
for col in ["annual_income", "employment_length_years"]:
    df[col] = df.groupby("origination_vintage")[col].transform(
        lambda s: s.fillna(s.median()))
# For "months_since_last_delinquency", missing means "no delinquency in window" -> impute to max
df["months_since_last_delinquency"] = df["months_since_last_delinquency"].fillna(24)

# Derived ratios
df["loan_to_income"] = df["loan_amount"] / df["annual_income"].clip(lower=1)
r = df["interest_rate_pct"] / 100.0 / 12.0
n_term = df["loan_term_months"]
df["estimated_monthly_payment"] = df["loan_amount"] * r / (1 - (1 + r) ** (-n_term))
df["payment_to_income"] = df["estimated_monthly_payment"] * 12 / df["annual_income"].clip(lower=1)
df["log_annual_income"] = np.log1p(df["annual_income"])
df["log_loan_amount"] = np.log1p(df["loan_amount"])

NUMERIC_FEATURES = NUMERIC_FEATURES + [
    "loan_to_income", "payment_to_income",
    "log_annual_income", "log_loan_amount",
    "annual_income_missing", "employment_length_years_missing",
    "months_since_last_delinquency_missing",
]

# -----------------------------------------------------------------------------
# Vintage split
# -----------------------------------------------------------------------------
train_mask = df["origination_vintage"].isin(TRAIN_VINTAGES)
val_mask = df["origination_vintage"].isin(VAL_VINTAGES)
oot_mask = df["origination_vintage"].isin(OOT_VINTAGES)

df_train = df[train_mask].reset_index(drop=True)
df_val = df[val_mask].reset_index(drop=True)
df_oot = df[oot_mask].reset_index(drop=True)

y_train = df_train["default"].to_numpy()
y_val = df_val["default"].to_numpy()
y_oot = df_oot["default"].to_numpy()

base_rate_train = y_train.mean()
print(f"\nTrain: {len(df_train):,}, DR={base_rate_train:.4f}")
print(f"Val:   {len(df_val):,}, DR={y_val.mean():.4f}")
print(f"OOT:   {len(df_oot):,}, DR={y_oot.mean():.4f}")

# -----------------------------------------------------------------------------
# WoE binning utilities
# -----------------------------------------------------------------------------
def make_numeric_bins(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    qs[0], qs[-1] = -np.inf, np.inf
    return qs


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)


def woe_iv(y: np.ndarray, bin_ix: np.ndarray, n_bins: int) -> tuple[np.ndarray, float]:
    """Return WoE per bin and total Information Value."""
    woes = np.zeros(n_bins)
    iv_total = 0.0
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    for b in range(n_bins):
        in_bin = bin_ix == b
        pos = (y[in_bin] == 1).sum()
        neg = (y[in_bin] == 0).sum()
        # Smoothing
        share_pos = (pos + 0.5) / (n_pos + 0.5)
        share_neg = (neg + 0.5) / (n_neg + 0.5)
        woe = float(np.log(share_neg / share_pos))
        woes[b] = woe
        iv_total += (share_neg - share_pos) * woe
    return woes, iv_total


# Build WoE encoder per feature (fitted on training data only)
encoders: dict[str, dict] = {}

LOW_CARDINALITY_THRESHOLD = 8

for col in NUMERIC_FEATURES:
    x_tr = df_train[col].to_numpy()
    unique_vals = np.unique(x_tr)
    if len(unique_vals) <= LOW_CARDINALITY_THRESHOLD:
        # Treat as discrete: each unique value is its own bin
        val_to_ix = {v: i for i, v in enumerate(unique_vals)}
        bin_ix = np.array([val_to_ix[v] for v in x_tr])
        woes, iv = woe_iv(y_train, bin_ix, len(unique_vals))
        encoders[col] = {"type": "discrete_numeric",
                         "values": unique_vals,
                         "val_to_ix": val_to_ix,
                         "woe": woes, "iv": iv}
    else:
        edges = make_numeric_bins(x_tr, n_bins=10)
        bin_ix = assign_bins(x_tr, edges)
        n_bins_actual = len(edges) - 1
        woes, iv = woe_iv(y_train, bin_ix, n_bins_actual)
        encoders[col] = {"type": "numeric", "edges": edges, "woe": woes, "iv": iv}

for col in CATEGORICAL_FEATURES:
    x_tr = df_train[col].to_numpy()
    cats = np.unique(x_tr)
    cat_to_ix = {c: i for i, c in enumerate(cats)}
    bin_ix = np.array([cat_to_ix[c] for c in x_tr])
    woes, iv = woe_iv(y_train, bin_ix, len(cats))
    encoders[col] = {"type": "categorical", "categories": cats,
                     "cat_to_ix": cat_to_ix, "woe": woes, "iv": iv}

# -----------------------------------------------------------------------------
# Information Value table and feature selection
# -----------------------------------------------------------------------------
iv_table = pd.DataFrame([
    {"feature": col, "iv": enc["iv"], "type": enc["type"]}
    for col, enc in encoders.items()
]).sort_values("iv", ascending=False)
iv_table.to_csv(OUT / "information_values.csv", index=False)

print("\n=== Information Values ===")
print(iv_table.to_string(index=False,
      formatters={"iv": "{:.4f}".format}))

# Drop features below the standard IV threshold (0.02 is "unpredictive")
IV_THRESHOLD = 0.02
kept_features = iv_table[iv_table["iv"] >= IV_THRESHOLD]["feature"].tolist()
dropped_features = iv_table[iv_table["iv"] < IV_THRESHOLD]["feature"].tolist()
print(f"\nKept ({len(kept_features)}) features with IV >= {IV_THRESHOLD}")
print(f"Dropped ({len(dropped_features)}): {dropped_features}")


# -----------------------------------------------------------------------------
# Apply WoE encoding
# -----------------------------------------------------------------------------
def transform(df_in: pd.DataFrame, features: list[str]) -> np.ndarray:
    out = np.zeros((len(df_in), len(features)), dtype=float)
    for j, col in enumerate(features):
        enc = encoders[col]
        if enc["type"] == "numeric":
            ix = assign_bins(df_in[col].to_numpy(), enc["edges"])
        elif enc["type"] == "discrete_numeric":
            ix = np.array([enc["val_to_ix"].get(v, 0) for v in df_in[col]])
        else:
            unknown = -1
            ix = np.array([enc["cat_to_ix"].get(c, unknown) for c in df_in[col]])
            ix = np.where(ix == unknown, 0, ix)
        out[:, j] = enc["woe"][ix]
    return out


X_train = transform(df_train, kept_features)
X_val = transform(df_val, kept_features)
X_oot = transform(df_oot, kept_features)

# -----------------------------------------------------------------------------
# Fit L2-regularized logistic regression on WoE features
# -----------------------------------------------------------------------------
model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                          class_weight=None, random_state=SEED)
model.fit(X_train, y_train)

p_train_raw = model.predict_proba(X_train)[:, 1]
p_val_raw = model.predict_proba(X_val)[:, 1]
p_oot_raw = model.predict_proba(X_oot)[:, 1]

# -----------------------------------------------------------------------------
# Calibration: shift intercept so train mean predicted PD = train DR
# This is the "Platt-style intercept correction" used in IRB scorecards
# to map output to long-run-average PD.
# -----------------------------------------------------------------------------
def calibrate_intercept(p_raw: np.ndarray, target_rate: float) -> tuple[np.ndarray, float]:
    """Solve for offset c such that mean(sigmoid(logit(p) + c)) = target_rate."""
    logits = np.log(p_raw / (1 - p_raw))
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logits + mid)))).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    offset = (lo + hi) / 2
    return 1 / (1 + np.exp(-(logits + offset))), offset


p_train, calib_offset = calibrate_intercept(p_train_raw, base_rate_train)
# Apply same offset to val and oot
p_val = 1 / (1 + np.exp(-(np.log(p_val_raw / (1 - p_val_raw)) + calib_offset)))
p_oot = 1 / (1 + np.exp(-(np.log(p_oot_raw / (1 - p_oot_raw)) + calib_offset)))

# -----------------------------------------------------------------------------
# Master rating scale (A1..C3, 9 grades)
# -----------------------------------------------------------------------------
GRADE_EDGES = [0.005, 0.01, 0.025, 0.05, 0.08, 0.12, 0.18, 0.28, 1.0]
GRADE_LABELS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]


def assign_grade(p: np.ndarray) -> np.ndarray:
    ix = np.searchsorted(GRADE_EDGES, p, side="right")
    ix = np.clip(ix, 0, len(GRADE_LABELS) - 1)
    return np.array([GRADE_LABELS[i] for i in ix])


grade_train = assign_grade(p_train)
grade_val = assign_grade(p_val)
grade_oot = assign_grade(p_oot)


# -----------------------------------------------------------------------------
# Discrimination metrics per split
# -----------------------------------------------------------------------------
def discrimination(y, p, label):
    auc = roc_auc_score(y, p)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y, p)
    ks = float(np.max(tpr - fpr))
    ll = log_loss(y, p)
    br = brier_score_loss(y, p)
    ap = average_precision_score(y, p)
    print(f"  {label:<5}  AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}"
          f"  log-loss={ll:.4f}  Brier={br:.4f}  AP={ap:.4f}")
    return dict(auc=auc, gini=gini, ks=ks, log_loss=ll, brier=br, ap=ap)


print("\n=== Discrimination ===")
m_train = discrimination(y_train, p_train, "TRAIN")
m_val = discrimination(y_val, p_val, "VAL")
m_oot = discrimination(y_oot, p_oot, "OOT")

# -----------------------------------------------------------------------------
# Calibration check: predicted PD vs actual default rate per grade
# -----------------------------------------------------------------------------
def grade_table(df_split, p, grade, label):
    out = pd.DataFrame({
        "grade": grade,
        "default": df_split["default"].to_numpy(),
        "pd_pred": p,
    })
    g = (out.groupby("grade")
            .agg(n=("default", "size"),
                 defaults=("default", "sum"),
                 obs_dr=("default", "mean"),
                 pred_pd=("pd_pred", "mean"))
            .reindex(GRADE_LABELS).fillna(0))
    g["population_pct"] = g["n"] / g["n"].sum()
    g["split"] = label
    return g


grade_summary = pd.concat([
    grade_table(df_train, p_train, grade_train, "TRAIN"),
    grade_table(df_val, p_val, grade_val, "VAL"),
    grade_table(df_oot, p_oot, grade_oot, "OOT"),
])
grade_summary.to_csv(OUT / "grade_summary.csv")
print("\n=== Grade summary (TRAIN) ===")
print(grade_summary.loc[grade_summary["split"] == "TRAIN"]
      .to_string(formatters={"obs_dr": "{:.3%}".format,
                              "pred_pd": "{:.3%}".format,
                              "population_pct": "{:.1%}".format}))

# -----------------------------------------------------------------------------
# Population Stability Index (PSI): TRAIN vs OOT
# -----------------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: list[str]) -> float:
    psi_value = 0.0
    for grade in bins:
        p_exp = (expected == grade).mean()
        p_act = (actual == grade).mean()
        if p_exp > 0 and p_act > 0:
            psi_value += (p_act - p_exp) * np.log(p_act / p_exp)
    return psi_value


psi_train_val = psi(grade_train, grade_val, GRADE_LABELS)
psi_train_oot = psi(grade_train, grade_oot, GRADE_LABELS)
print(f"\nPSI TRAIN vs VAL: {psi_train_val:.4f}")
print(f"PSI TRAIN vs OOT: {psi_train_oot:.4f}")
print("PSI < 0.10: stable, 0.10-0.25: minor shift, > 0.25: material shift")

# -----------------------------------------------------------------------------
# Save coefficients and summary metrics
# -----------------------------------------------------------------------------
coef_table = pd.DataFrame({
    "feature": kept_features,
    "coef_woe": model.coef_.ravel(),
    "iv": [encoders[c]["iv"] for c in kept_features],
}).sort_values("iv", ascending=False)
coef_table.to_csv(OUT / "coefficients.csv", index=False)

metrics_summary = {
    "train": m_train, "val": m_val, "oot": m_oot,
    "psi_train_val": psi_train_val,
    "psi_train_oot": psi_train_oot,
    "base_rate_train": float(base_rate_train),
    "base_rate_val": float(y_val.mean()),
    "base_rate_oot": float(y_oot.mean()),
    "calib_offset": float(calib_offset),
    "n_features_kept": len(kept_features),
    "n_features_dropped": len(dropped_features),
    "kept_features": kept_features,
    "dropped_features": dropped_features,
}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120})

# 1. ROC across all three splits
fig, ax = plt.subplots(figsize=(7, 6))
for y, p, label, color in [(y_train, p_train, "TRAIN", "steelblue"),
                           (y_val, p_val, "VAL", "darkorange"),
                           (y_oot, p_oot, "OOT", "crimson")]:
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC: train, validation, out-of-time")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "01_roc_three_splits.png"); plt.close()

# 2. Information Value bar
fig, ax = plt.subplots(figsize=(10, 7))
iv_plot = iv_table.iloc[::-1]
colors = ["steelblue" if v >= IV_THRESHOLD else "lightgrey" for v in iv_plot["iv"]]
ax.barh(iv_plot["feature"], iv_plot["iv"], color=colors)
ax.axvline(IV_THRESHOLD, color="red", linestyle="--", lw=1, label=f"IV threshold = {IV_THRESHOLD}")
ax.set_xlabel("Information Value"); ax.set_title("Feature IV ranking")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "02_information_values.png"); plt.close()

# 3. Calibration curve, OOT split
prob_true, prob_pred = calibration_curve(y_oot, p_oot, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Perfect calibration")
ax.plot(prob_pred, prob_true, marker="o", color="crimson", lw=2, label="OOT")
ax.set_xlabel("Mean predicted PD"); ax.set_ylabel("Observed default rate")
ax.set_title("Calibration on out-of-time cohort (2023H2)")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "03_calibration_oot.png"); plt.close()

# 4. Predicted PD vs observed DR per grade (OOT)
g_oot = grade_summary.loc[grade_summary["split"] == "OOT"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(GRADE_LABELS))
ax.bar(x - 0.2, g_oot["pred_pd"], width=0.4, color="steelblue", label="Predicted PD")
ax.bar(x + 0.2, g_oot["obs_dr"],  width=0.4, color="darkorange", label="Observed default rate")
ax.set_xticks(x); ax.set_xticklabels(GRADE_LABELS)
ax.set_ylabel("Probability"); ax.set_title("Calibration per grade, OOT cohort")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "04_grade_calibration_oot.png"); plt.close()

# 5. PSI grade distribution
fig, ax = plt.subplots(figsize=(10, 5))
for grade_arr, label, color in [(grade_train, "TRAIN", "steelblue"),
                                (grade_val, "VAL", "darkorange"),
                                (grade_oot, "OOT", "crimson")]:
    shares = [(grade_arr == g).mean() for g in GRADE_LABELS]
    ax.plot(GRADE_LABELS, shares, marker="o", lw=2, label=label, color=color)
ax.set_xlabel("Rating grade"); ax.set_ylabel("Share of population")
ax.set_title(f"Population stability across splits (PSI OOT = {psi_train_oot:.3f})")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "05_population_stability.png"); plt.close()

# 6. Score distribution by class on OOT
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(p_oot[y_oot == 0], bins=40, alpha=0.6, color="steelblue",
        label="Non-default", density=True)
ax.hist(p_oot[y_oot == 1], bins=40, alpha=0.6, color="crimson",
        label="Default", density=True)
ax.set_xlabel("Predicted PD"); ax.set_ylabel("Density")
ax.set_title("Score distribution on OOT cohort")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "06_score_distribution_oot.png"); plt.close()

print(f"\nFigures written to {FIG}")
print(f"Tables written to {OUT}")
print("Done.")
