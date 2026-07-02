# =============================================================================
# PD MODEL — SEB Quantitative Case
# =============================================================================
# Task   : Predict probability of default within 12 months (binary: DEFAULT_FLG)
# Data   : case_data.csv (70 975 observations, 45 features)
# Split  : Train 2008–2011  |  Test 2012–2013  (temporal, no leakage)
# Output : Probability of default per borrower (no threshold applied)
# Metrics: Gini, KS-statistic, Brier score, calibration curve
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             precision_recall_curve, average_precision_score,
                             log_loss)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

plt.style.use('seaborn-v0_8-whitegrid')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("1. LOAD DATA")
print("=" * 70)

df = pd.read_csv('case_data.csv', sep=';')
varinfo = pd.read_csv('varinfo.csv', sep=';', header=0)

TARGET = 'DEFAULT_FLG'
ID_COL = 'ID'
PERIOD_COL = 'PERIOD'

# Drop non-feature columns
DROP_COLS = [ID_COL, PERIOD_COL]
feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]

print(f"Shape: {df.shape}")
print(f"Default rate: {df[TARGET].mean():.2%}")
print(f"Features: {len(feature_cols)}")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("2. PREPROCESSING")
print("=" * 70)

# --- 2a. Remove the single PERIOD=1956 row (data anomaly) ---
# The observation year 1956 is implausible for modern banking data.
n_before = len(df)
df = df[df[PERIOD_COL] != 1956].copy()
print(f"Removed {n_before - len(df)} row(s) with PERIOD=1956 (data anomaly)")

# --- 2b. Missing values ---
# Very few missing values (<5 per column) — impute with median per column.
missing = df[feature_cols].isnull().sum()
missing_cols = missing[missing > 0]
print(f"\nMissing values (before imputation):\n{missing_cols}")
for col in missing_cols.index:
    df[col].fillna(df[col].median(), inplace=True)
print("Imputed with column median.")

# --- 2c. Winsorization (considered and rejected) ---
# Clipping at the 1st/99th percentile was considered but not applied.
# WoE binning assigns extreme values to the edge bin, so they have no further
# influence on the linear predictor — winsorization is redundant for the LR
# pipeline. Tree models are invariant to monotonic transformations and do not
# require it either. Both sets of models receive the same imputed data.

# --- 2d. Temporal train/test split ---
# Train: 2008–2011 (in-sample). Test: 2012–2013 (out-of-time validation).
# A two-year holdout (~25% of data) provides ~400 defaults in the test set —
# sufficient for statistically reliable Gini estimates.
# Testing on 2013 only yields ~75 defaults, too few for meaningful model comparison.
train_mask = df[PERIOD_COL].isin([2008, 2009, 2010, 2011])
test_mask  = df[PERIOD_COL].isin([2012, 2013])

X_train = df.loc[train_mask, feature_cols].copy()
y_train = df.loc[train_mask, TARGET].copy()
X_test  = df.loc[test_mask,  feature_cols].copy()
y_test  = df.loc[test_mask,  TARGET].copy()

X_train_raw = df.loc[train_mask, feature_cols].copy()
X_test_raw  = df.loc[test_mask,  feature_cols].copy()

print(f"\nTrain: {len(X_train):,} rows | Default rate: {y_train.mean():.2%}")
print(f"Test : {len(X_test):,}  rows | Default rate: {y_test.mean():.2%}")

# =============================================================================
# 3. FEATURE SELECTION
# =============================================================================
print("\n" + "=" * 70)
print("3. FEATURE SELECTION")
print("=" * 70)

# ---------------------------------------------------------------------------
# 3a. Information Value (IV) — industry standard in credit risk modelling.
#     Captures non-linear relationships, works natively with binary targets.
#     IV thresholds: <0.02 useless | 0.02–0.1 weak | 0.1–0.3 medium |
#                   0.3–0.5 strong | >0.5 suspicious (check for leakage)
# ---------------------------------------------------------------------------
def compute_iv(X: pd.DataFrame, y: pd.Series, n_bins: int = 10) -> pd.Series:
    """Compute Information Value for all columns in X vs binary target y."""
    iv_dict = {}
    total_events = y.sum()
    total_non_events = (1 - y).sum()
    for col in X.columns:
        try:
            # Bin into at most n_bins quantile-based buckets
            bins = pd.qcut(X[col], q=n_bins, duplicates='drop')
            grouped = pd.crosstab(bins, y)
            if grouped.shape[1] < 2:
                iv_dict[col] = 0.0
                continue
            grouped.columns = ['non_event', 'event']
            grouped['dist_event']     = grouped['event']     / total_events
            grouped['dist_non_event'] = grouped['non_event'] / total_non_events
            # Replace 0 to avoid log(0)
            grouped['dist_event']     = grouped['dist_event'].replace(0, 1e-6)
            grouped['dist_non_event'] = grouped['dist_non_event'].replace(0, 1e-6)
            grouped['woe'] = np.log(grouped['dist_event'] / grouped['dist_non_event'])
            grouped['iv']  = (grouped['dist_event'] - grouped['dist_non_event']) * grouped['woe']
            iv_dict[col] = grouped['iv'].sum()
        except Exception:
            iv_dict[col] = 0.0
    return pd.Series(iv_dict).sort_values(ascending=False)

iv_scores = compute_iv(X_train, y_train, n_bins=10)
print("\nInformation Value (all features):")
print(iv_scores.to_string())

IV_THRESHOLD = 0.02
selected_iv = iv_scores[iv_scores >= IV_THRESHOLD].index.tolist()
print(f"\nFeatures with IV >= {IV_THRESHOLD}: {len(selected_iv)}")
print(selected_iv)

# ---------------------------------------------------------------------------
# 3b. Iterative VIF removal — handles multicollinearity properly.
#     Remove the feature with highest VIF (> 10), recompute, repeat.
#     This is iterative because removing one feature changes all VIF values.
# ---------------------------------------------------------------------------
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    # Standardize before VIF to avoid numerical issues with large-scale features
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    Xs = Xs.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    vif_data = pd.DataFrame()
    vif_data['feature'] = Xs.columns
    vif_data['VIF'] = [variance_inflation_factor(Xs.values, i)
                       for i in range(Xs.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

VIF_THRESHOLD = 10.0
selected_vif = selected_iv.copy()

print(f"\nIterative VIF removal (threshold = {VIF_THRESHOLD}):")
iteration = 0
while True:
    vif_df = compute_vif(X_train[selected_vif])
    max_vif = vif_df['VIF'].max()
    if max_vif <= VIF_THRESHOLD:
        break
    worst_feature = vif_df.iloc[0]['feature']
    print(f"  Iter {iteration+1}: removing '{worst_feature}' (VIF={max_vif:.1f})")
    selected_vif.remove(worst_feature)
    iteration += 1

print(f"Remaining after VIF removal: {len(selected_vif)} features")
print(selected_vif)

# ---------------------------------------------------------------------------
# 3c. Forward stepwise selection — combined Gini + Brier criterion (CV).
#     Select up to MAX_FEATURES features from the VIF-filtered pool.
#     Scoring: alpha*Gini + (1-alpha)*BSS where BSS = 1 - Brier/Brier_naive.
#     Optimising Gini alone may ignore probability calibration. Including
#     Brier penalises features that inflate predicted probabilities.
# ---------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer

MAX_FEATURES     = 10
SELECTION_ALPHA  = 0.5   # weight on Gini; (1-alpha) goes to BSS
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scaler_fs = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler_fs.fit_transform(X_train[selected_vif]),
    columns=selected_vif
).fillna(0).replace([np.inf, -np.inf], 0)

lr_fs = LogisticRegression(
    max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs'
)

def combined_gini_bss(y_true, y_prob):
    """alpha*Gini + (1-alpha)*BSS — optimises discrimination AND calibration.
    Gini = 2*AUC - 1 is the standard credit risk discrimination metric.
    BSS  = 1 - Brier/Brier_naive normalises calibration to a similar scale."""
    gini  = 2 * roc_auc_score(y_true, y_prob) - 1
    brier = brier_score_loss(y_true, y_prob)
    p_bar       = float(y_true.mean())
    brier_naive = p_bar * (1.0 - p_bar)
    bss         = 1.0 - brier / max(brier_naive, 1e-9)
    return SELECTION_ALPHA * gini + (1.0 - SELECTION_ALPHA) * bss

combined_scorer = make_scorer(combined_gini_bss, needs_proba=True)

remaining   = selected_vif.copy()
chosen      = []
best_scores = []   # combined scores per step (used in plot 6)

print(f"\nForward stepwise selection (max {MAX_FEATURES} features):")
print(f"  Criterion: {SELECTION_ALPHA:.0%} x Gini  +  {1-SELECTION_ALPHA:.0%} x Brier (5-fold CV)")
for step in range(MAX_FEATURES):
    best_score = -np.inf
    best_feat  = None
    for feat in remaining:
        candidate = chosen + [feat]
        score = cross_val_score(
            lr_fs, X_train_scaled[candidate], y_train,
            cv=cv, scoring=combined_scorer, n_jobs=-1
        ).mean()
        if score > best_score:
            best_score = score
            best_feat  = feat
    chosen.append(best_feat)
    remaining.remove(best_feat)
    best_scores.append(best_score)
    print(f"  Step {step+1:2d}: +'{best_feat}'  ->  CV Score = {best_score:.4f}")

# Keep best_aucs alias so plot-6 code needs no rename
best_aucs = best_scores

FINAL_FEATURES = chosen
print(f"\nFinal feature set ({len(FINAL_FEATURES)} features): {FINAL_FEATURES}")

# ---------------------------------------------------------------------------
# 3e. WoE transformation for Logistic Regression
#     Gold standard in bank PD models. Replaces raw features with Weight of
#     Evidence values computed per bin. Benefits:
#       - Captures non-linear (monotonic) feature-target relationships
#       - Handles outliers automatically (extreme values fall in edge bins)
#       - Eliminates need for winsorization and standardization for LR
#       - Coefficients remain interpretable in log-odds units
#     WoE is fitted on training data only and applied to test data.
# ---------------------------------------------------------------------------
def fit_woe(X: pd.DataFrame, y: pd.Series, features: list, n_bins: int = 10):
    """Fit WoE bins using numpy searchsorted — avoids pandas typing issues.
    Returns woe_arrays (numpy arrays of WoE per bin) and bin_edges dicts."""
    total_events     = float(y.sum())
    total_non_events = float((1 - y).sum())
    y_vals    = y.values.astype(float)
    woe_maps  = {}
    bin_edges = {}
    for col in features:
        try:
            _, edges = pd.qcut(X[col], q=n_bins, duplicates='drop', retbins=True)
            bin_edges[col] = edges
            interior = edges[1:-1]               # interior cut-points
            vals     = X[col].values.astype(float)
            # searchsorted with side='left' matches pd.cut (right-closed bins,
            # include_lowest on the first bin)
            bin_idx  = np.searchsorted(interior, vals, side='left')  # 0..n_bins-1
            n_bins_actual = len(edges) - 1
            woe_arr  = np.zeros(n_bins_actual)
            for b in range(n_bins_actual):
                mask     = (bin_idx == b)
                n_ev     = float(y_vals[mask].sum())
                n_nev    = float((1 - y_vals[mask]).sum())
                dist_ev  = max(n_ev  / total_events,     1e-6)
                dist_nev = max(n_nev / total_non_events, 1e-6)
                woe_arr[b] = np.log(dist_ev / dist_nev)
            woe_maps[col] = woe_arr
        except Exception:
            woe_maps[col]  = None
            bin_edges[col] = None
    return woe_maps, bin_edges

def apply_woe(X: pd.DataFrame, features: list, woe_maps: dict,
              bin_edges: dict) -> np.ndarray:
    """Apply WoE using numpy searchsorted. Values outside training range are
    clipped to the nearest edge bin (conservative: no extrapolation)."""
    X_woe = np.zeros((len(X), len(features)))
    for j, col in enumerate(features):
        if bin_edges[col] is not None and woe_maps[col] is not None:
            interior    = bin_edges[col][1:-1]
            vals        = X[col].values.astype(float)
            bin_idx     = np.searchsorted(interior, vals, side='left')
            bin_idx     = np.clip(bin_idx, 0, len(woe_maps[col]) - 1)
            X_woe[:, j] = woe_maps[col][bin_idx]
    return X_woe

woe_maps, bin_edges = fit_woe(X_train, y_train, FINAL_FEATURES)
X_tr_woe = apply_woe(X_train, FINAL_FEATURES, woe_maps, bin_edges)
X_te_woe = apply_woe(X_test,  FINAL_FEATURES, woe_maps, bin_edges)
print("\nWoE transformation fitted and applied (LR only).")

# =============================================================================
# 4. MODEL TRAINING
# =============================================================================
print("\n" + "=" * 70)
print("4. MODEL TRAINING")
print("=" * 70)

# Final LR uses WoE features (X_tr_woe / X_te_woe) — no standardization needed.

# ---------------------------------------------------------------------------
# 3f. Forward stepwise selection for tree models (XGBoost internally)
#     Trees use raw features without WoE. Running a separate forward stepwise
#     with XGBoost ensures tree models are evaluated on their own optimal
#     feature set rather than features selected for LR.
# ---------------------------------------------------------------------------
print("\nForward stepwise selection for tree models (XGBoost internally):")
print(f"  Criterion: {SELECTION_ALPHA:.0%} x Gini  +  {1-SELECTION_ALPHA:.0%} x Brier (5-fold CV)")

xgb_fs = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=1,
    eval_metric='auc', use_label_encoder=False,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)

remaining_tree = selected_vif.copy()
chosen_tree    = []
for step in range(MAX_FEATURES):
    best_score = -np.inf
    best_feat  = None
    for feat in remaining_tree:
        candidate = chosen_tree + [feat]
        score = cross_val_score(
            xgb_fs, X_train_raw[candidate].values, y_train,
            cv=cv, scoring=combined_scorer, n_jobs=-1
        ).mean()
        if score > best_score:
            best_score = score
            best_feat  = feat
    chosen_tree.append(best_feat)
    remaining_tree.remove(best_feat)
    print(f"  Step {step+1:2d}: +'{best_feat}'  ->  CV Score = {best_score:.4f}")

TREE_FEATURES = chosen_tree
print(f"\nTree feature set ({len(TREE_FEATURES)} features): {TREE_FEATURES}")

# Trees: raw data using tree-specific feature set
X_tr_tree = X_train_raw[TREE_FEATURES].values
X_te_tree = X_test_raw[TREE_FEATURES].values

models = {}

# --- Model 1: Logistic Regression (WoE) ---
# LR with MLE is well-calibrated by construction: the first-order condition
# for the intercept forces sum(y_i) = sum(p_hat_i), meaning mean predicted
# probability matches observed default rate. WoE helps by making the model
# well-specified (log-odds linearity holds approximately), which improves
# calibration beyond just the mean. C=1 is a standard default.
lr = LogisticRegression(
    penalty='l2', C=1, max_iter=1000,
    random_state=RANDOM_STATE, solver='lbfgs'
)
lr.fit(X_tr_woe, y_train.values)
models['Logistic Regression (WoE)'] = lr

# --- Model 2: Logistic Regression (L1 / Lasso, WoE) ---
# L1 regularization serves as a validation tool: features driven to zero have
# marginal predictive value after the others are included.
# Also demonstrates awareness of overfitting even in a small-feature setting.
# C=0.1 applies stronger shrinkage than Model 1 — a deliberate stress test.
lr_l1 = LogisticRegression(
    penalty='l1', C=0.1, max_iter=1000,
    random_state=RANDOM_STATE, solver='liblinear'
)
lr_l1.fit(X_tr_woe, y_train.values)
models['Logistic Regression (L1/WoE)'] = lr_l1

# L1 coefficient check — which features survive strong shrinkage?
l1_coefs = pd.Series(lr_l1.coef_[0], index=FINAL_FEATURES)
zeroed   = l1_coefs[l1_coefs == 0].index.tolist()
print(f"\nL1 validation — features driven to zero (C=0.1): "
      f"{zeroed if zeroed else 'none — all features confirmed relevant'}")

# --- Tune scale_pos_weight for tree models via cross-validation ---
# Using scale_pos_weight = class ratio (~60) optimises purely for discrimination
# at the cost of probability calibration. We instead search for the value that
# maximises the same combined Gini + BSS criterion used in feature selection.
# This gives tree models a fair chance at both discrimination and calibration.
def gini(auc):
    return 2 * auc - 1

print("\nTuning scale_pos_weight for tree models (combined Gini + Brier, 3-fold CV):")
spw_candidates = [1, 5, 10, 20, 40, 60]
cv_tree = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

best_spw_xgb, best_score_xgb = 1, -np.inf
for spw in spw_candidates:
    xgb_cv = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='auc', use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
    xgb_cv.fit(X_tr_tree, y_train)
    tr_gini = gini(roc_auc_score(y_train, xgb_cv.predict_proba(X_tr_tree)[:,1]))
    te_gini = gini(roc_auc_score(y_test,  xgb_cv.predict_proba(X_te_tree)[:,1]))
    cv_score = cross_val_score(xgb_cv, X_tr_tree, y_train,
                               cv=cv_tree, scoring=combined_scorer, n_jobs=-1).mean()
    print(f"  XGBoost  spw={spw:<4} -> CV={cv_score:.4f}  Gini Train={tr_gini:.4f}  Test={te_gini:.4f}  Gap={tr_gini-te_gini:.4f}")
    if cv_score > best_score_xgb:
        best_score_xgb, best_spw_xgb = cv_score, spw

best_spw_lgb, best_score_lgb = 1, -np.inf
for spw in spw_candidates:
    lgb_cv = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_cv.fit(X_tr_tree, y_train)
    tr_gini = gini(roc_auc_score(y_train, lgb_cv.predict_proba(X_tr_tree)[:,1]))
    te_gini = gini(roc_auc_score(y_test,  lgb_cv.predict_proba(X_te_tree)[:,1]))
    cv_score = cross_val_score(lgb_cv, X_tr_tree, y_train,
                               cv=cv_tree, scoring=combined_scorer, n_jobs=-1).mean()
    print(f"  LightGBM spw={spw:<4} -> CV={cv_score:.4f}  Gini Train={tr_gini:.4f}  Test={te_gini:.4f}  Gap={tr_gini-te_gini:.4f}")
    if cv_score > best_score_lgb:
        best_score_lgb, best_spw_lgb = cv_score, spw

print(f"\nBest scale_pos_weight -> XGBoost: {best_spw_xgb}, LightGBM: {best_spw_lgb}")

# --- Model 3: XGBoost (tuned scale_pos_weight) ---
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=best_spw_xgb,
    eval_metric='auc', use_label_encoder=False,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_tr_tree, y_train)
models['XGBoost'] = xgb_model

# --- Model 4: LightGBM (tuned scale_pos_weight) ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=best_spw_lgb,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_tr_tree, y_train)
models['LightGBM'] = lgb_model

# =============================================================================
# 5. EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("5. EVALUATION")
print("=" * 70)

def ks_statistic(y_true, y_prob):
    """KS statistic: max separation between CDFs of events and non-events."""
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    return ks_stat

def get_data(name):
    """Return correct train/test arrays per model type.
    LR uses WoE-transformed features (no winsorization/scaling needed).
    Trees use raw (non-winsorized) data — invariant to monotonic transforms.
    """
    if 'XGBoost' in name or 'LightGBM' in name:
        return X_tr_tree, X_te_tree
    return X_tr_woe, X_te_woe

results = {}
for name, model in models.items():
    Xtr_, Xte_ = get_data(name)
    y_prob_tr = model.predict_proba(Xtr_)[:, 1]
    y_prob_te = model.predict_proba(Xte_)[:, 1]

    auc_tr = roc_auc_score(y_train, y_prob_tr)
    auc_te = roc_auc_score(y_test,  y_prob_te)
    ks_te  = ks_statistic(y_test.values, y_prob_te)
    brier_tr = brier_score_loss(y_train, y_prob_tr)
    brier_te = brier_score_loss(y_test,  y_prob_te)
    ll     = log_loss(y_test, y_prob_te)

    results[name] = {
        'Gini Train':  gini(auc_tr),
        'Gini Test':   gini(auc_te),
        'Gini Gap':    gini(auc_tr) - gini(auc_te),
        'Brier Train': brier_tr,
        'Brier Test':  brier_te,
        'Brier Gap':   brier_te - brier_tr,
        'KS Test':     ks_te,
        'Log-Loss':    ll,
    }
    print(f"\n{name}")
    print(f"  Gini  Train={gini(auc_tr):.4f}  Test={gini(auc_te):.4f}  Gap={gini(auc_tr)-gini(auc_te):.4f}")
    print(f"  Brier Train={brier_tr:.4f}  Test={brier_te:.4f}  Gap={brier_te-brier_tr:.4f}")
    print(f"  KS    Test={ks_te:.4f}")

results_df = pd.DataFrame(results).T
print("\n\nSummary table:")
print(results_df.round(4).to_string())

# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# Measures how much the feature distribution has shifted between train and
# test (out-of-time). Critical in banking — a stable model on a shifted
# population is unreliable.
# PSI < 0.10: stable | 0.10–0.25: moderate shift | > 0.25: significant shift
# ---------------------------------------------------------------------------
def compute_psi(train_col: pd.Series, test_col: pd.Series, n_bins: int = 10) -> float:
    _, edges = pd.qcut(train_col, q=n_bins, duplicates='drop', retbins=True)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    train_dist = pd.cut(train_col, bins=edges).value_counts(normalize=True).sort_index()
    test_dist  = pd.cut(test_col,  bins=edges).value_counts(normalize=True).sort_index()
    train_dist = train_dist.reindex(test_dist.index, fill_value=1e-6).clip(lower=1e-6)
    test_dist  = test_dist.reindex(train_dist.index, fill_value=1e-6).clip(lower=1e-6)
    return float(((test_dist - train_dist) * np.log(test_dist / train_dist)).sum())

print("\n\nPSI — Feature stability (train 2008-2011 vs test 2012-2013):")
psi_results = {}
for col in FINAL_FEATURES:
    psi_val = compute_psi(X_train[col], X_test[col])
    psi_results[col] = psi_val
psi_series = pd.Series(psi_results).sort_values(ascending=False)
for feat, psi_val in psi_series.items():
    flag = "stable" if psi_val < 0.10 else ("moderate" if psi_val < 0.25 else "SHIFT")
    print(f"  {feat:<30} PSI={psi_val:.4f}  [{flag}]")

# ---------------------------------------------------------------------------
# Decile analysis — Cumulative Capture Rate
# Sort borrowers by predicted PD (descending), split into 10 equal groups.
# Shows what % of all defaults are captured in the top N% highest-risk group.
# Key metric for communicating business value and comparing LR vs XGBoost.
# ---------------------------------------------------------------------------
def decile_table(y_true: np.ndarray, y_prob: np.ndarray,
                 model_name: str) -> pd.DataFrame:
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df = df.sort_values('p', ascending=False).reset_index(drop=True)
    df['decile'] = pd.qcut(df.index, q=10, labels=range(1, 11))
    tbl = df.groupby('decile', observed=True).agg(
        N=('y', 'count'), Defaults=('y', 'sum')
    ).reset_index()
    tbl['Default_Rate']     = tbl['Defaults'] / tbl['N']
    tbl['Cum_Defaults']     = tbl['Defaults'].cumsum()
    tbl['Cum_Capture_Rate'] = tbl['Cum_Defaults'] / tbl['Defaults'].sum()
    tbl.insert(0, 'Model', model_name)
    return tbl

print("\n\nDecile analysis — Cumulative Capture Rate (test set):")
for name in ['Logistic Regression (WoE)', 'XGBoost']:
    _, Xte_ = get_data(name)
    y_prob  = models[name].predict_proba(Xte_)[:, 1]
    tbl     = decile_table(y_test.values, y_prob, name)
    print(f"\n  {name}")
    print(f"  {'Decile':<8} {'N':>6} {'Defaults':>9} {'Default%':>10} {'Cum Capture%':>14}")
    for _, row in tbl.iterrows():
        print(f"  {int(row.decile):<8} {int(row.N):>6} {int(row.Defaults):>9} "
              f"{row.Default_Rate:>9.2%} {row.Cum_Capture_Rate:>13.1%}")

# =============================================================================
# 6. PLOTS
# =============================================================================
print("\n" + "=" * 70)
print("6. GENERATING PLOTS")
print("=" * 70)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- Plot 1: Cumulative Accuracy Profile (CAP) curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
n_total = len(y_test)
n_defaults = y_test.sum()
for (name, model), color in zip(models.items(), colors):
    _, Xte_ = get_data(name)
    y_prob_te = model.predict_proba(Xte_)[:, 1]
    order = np.argsort(y_prob_te)[::-1]
    y_sorted = np.array(y_test)[order]
    cum_defaults = np.cumsum(y_sorted) / n_defaults
    cum_pop = np.arange(1, n_total + 1) / n_total
    g = gini(roc_auc_score(y_test, y_prob_te))
    ax.plot(cum_pop, cum_defaults, color=color, lw=2,
            label=f'{name}  (Gini={g:.3f})')
# Random model
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random model')
# Perfect model
ax.plot([0, n_defaults/n_total, 1], [0, 1, 1], 'k:', lw=1,
        label='Perfect model')
ax.set_xlabel('Fraction of borrowers (sorted by predicted risk, high to low)')
ax.set_ylabel('Fraction of defaults captured')
ax.set_title('Cumulative Accuracy Profile (CAP) - Test Set (2012-2013)')
ax.legend(loc='lower right', fontsize=9)

# --- Plot 2: Calibration curves ---
ax = axes[1]
for (name, model), color in zip(models.items(), colors):
    _, Xte_ = get_data(name)
    y_prob_te = model.predict_proba(Xte_)[:, 1]
    fraction_of_positives, mean_predicted = calibration_curve(
        y_test, y_prob_te, n_bins=10, strategy='quantile'
    )
    ax.plot(mean_predicted, fraction_of_positives, 's-', color=color,
            lw=2, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax.set_xlabel('Mean predicted probability per bin')
ax.set_ylabel('Actual default rate per bin')
ax.set_title('Calibration Curves - Test Set')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('cap_calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cap_calibration.png")

# --- Plot 3: IV bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))
iv_final = iv_scores[FINAL_FEATURES].sort_values(ascending=True)
bars = ax.barh(iv_final.index, iv_final.values, color='steelblue')
ax.axvline(0.1, color='orange', linestyle='--', lw=1.5, label='IV=0.1 (medium)')
ax.axvline(0.3, color='green',  linestyle='--', lw=1.5, label='IV=0.3 (strong)')
ax.set_xlabel('Information Value')
ax.set_title('Information Value — Final 10 Features')
ax.legend()
plt.tight_layout()
plt.savefig('iv_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: iv_chart.png")

# --- Plot 4: Feature importance (best model) ---
# Determine best model by test AUC
best_model_name = results_df['Gini Test'].idxmax()
best_model      = models[best_model_name]
_, best_Xte     = get_data(best_model_name)
print(f"\nBest model by AUC: {best_model_name}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LR coefficients (WoE model — coefficients are in log-odds / WoE units)
ax = axes[0]
lr_coefs = pd.Series(
    models['Logistic Regression (WoE)'].coef_[0],
    index=FINAL_FEATURES
).sort_values()
colors_coef = ['#d62728' if v > 0 else '#1f77b4' for v in lr_coefs]
ax.barh(lr_coefs.index, lr_coefs.values, color=colors_coef)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Coefficient (WoE units = log-odds contribution)')
ax.set_title('Logistic Regression (WoE) — Coefficients')

# XGBoost feature importance
ax = axes[1]
if 'XGBoost' in models:
    fi = pd.Series(
        models['XGBoost'].feature_importances_,
        index=TREE_FEATURES
    ).sort_values(ascending=True)
    ax.barh(fi.index, fi.values, color='steelblue')
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title('XGBoost — Feature Importance')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# --- Plot 5: KS plot for best model ---
fig, ax = plt.subplots(figsize=(9, 5))
y_prob_best = best_model.predict_proba(best_Xte)[:, 1]
thresholds  = np.linspace(0, 1, 200)
tpr_list, fpr_list = [], []
for t in thresholds:
    pred = (y_prob_best >= t).astype(int)
    tp = ((pred == 1) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum()
    fn = ((pred == 0) & (y_test == 1)).sum()
    tn = ((pred == 0) & (y_test == 0)).sum()
    tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

tpr_arr = np.array(tpr_list)
fpr_arr = np.array(fpr_list)
ks_idx  = np.argmax(tpr_arr - fpr_arr)

ax.plot(thresholds, tpr_arr, label='TPR (Sensitivity)',  color='green', lw=2)
ax.plot(thresholds, fpr_arr, label='FPR (1-Specificity)', color='red',   lw=2)
ax.axvline(thresholds[ks_idx], color='navy', linestyle='--', lw=1.5,
           label=f'KS-threshold = {thresholds[ks_idx]:.3f}')
ax.annotate(f'KS = {tpr_arr[ks_idx]-fpr_arr[ks_idx]:.3f}',
            xy=(thresholds[ks_idx], (tpr_arr[ks_idx]+fpr_arr[ks_idx])/2),
            xytext=(thresholds[ks_idx]+0.05, 0.5), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='navy'))
ax.set_xlabel('Threshold')
ax.set_ylabel('Rate')
ax.set_title(f'KS Plot — {best_model_name} (Test Set)')
ax.legend()
plt.tight_layout()
plt.savefig('ks_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ks_plot.png")

# --- Plot 6: Forward stepwise AUC progression ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(best_aucs)+1), best_aucs, 'o-', color='steelblue', lw=2)
ax.set_xlabel('Number of features')
ax.set_ylabel('CV Combined Score (50% Gini + 50% Brier, 5-fold)')
ax.set_title('Forward Stepwise — Combined Score vs Number of Features')
ax.set_xticks(range(1, len(best_aucs)+1))
for i, (n, auc) in enumerate(zip(range(1, len(best_aucs)+1), best_aucs)):
    ax.annotate(f'{auc:.3f}', (n, auc), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('stepwise_auc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stepwise_auc.png")

# =============================================================================
# 7. FINAL MODEL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("7. FINAL MODEL SUMMARY")
print("=" * 70)
print(f"\nSelected model : {best_model_name}")
print(f"Final features : {FINAL_FEATURES}")
print(f"\nTest set performance:")
print(f"  Gini Train : {results[best_model_name]['Gini Train']:.4f}")
print(f"  Gini Test  : {results[best_model_name]['Gini Test']:.4f}")
print(f"  KS         : {results[best_model_name]['KS Test']:.4f}")
print(f"  Brier Test : {results[best_model_name]['Brier Test']:.4f}")

# --- Logistic Regression coefficient table (interpretability) ---
# Coefficients are in WoE units. Because WoE = log(dist_ev / dist_nev),
# the LR coefficients directly represent the log-odds weight assigned to
# each feature's binned risk signal. Positive => higher PD, negative => lower.
print("\nLogistic Regression (WoE) — coefficients:")
lr_coef_df = pd.DataFrame({
    'Feature':     FINAL_FEATURES,
    'Coefficient': models['Logistic Regression (WoE)'].coef_[0],
    'IV':          [iv_scores[f] for f in FINAL_FEATURES]
}).sort_values('Coefficient', key=abs, ascending=False)
print(lr_coef_df.round(4).to_string(index=False))

print("\nDone. All plots saved to working directory.")
