# -*- coding: utf-8 -*-
"""eda_processor.py

Reproducible EDA Processor for the toxicity dataset.

Process flow:
1. Load train/test split from split.pkl (contains X_train, X_test, y_train, y_test)
2. Apply DenseFeatureTransformer from build_features.py to generate engineered features
3. Run comprehensive EDA on the training set
4. Save all EDA results to eda_cache.json for consumption by the web app
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

#  parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.build_features import DenseFeatureTransformer

# Configuration
SPLIT_PATH = "analysis_and_inference/models/split_and_features/split.pkl"
OUTPUT_PATH = "analysis_and_inference/EDA/eda_cache.json"
TARGET_COLUMN = "toxic"

# ============================================================================
# Load split.pkl
# ============================================================================
print(f"Loading split from {SPLIT_PATH}...")
with open(SPLIT_PATH, 'rb') as f:
    split_data = pickle.load(f)

X_train = split_data['X_train']
X_test = split_data['X_test']
y_train = split_data['y_train']
y_test = split_data['y_test']

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# ============================================================================
# Generate features using DenseFeatureTransformer
# ============================================================================
print("Generating features for training set...")
transformer = DenseFeatureTransformer()
features_train = transformer.fit_transform(X_train)
print(f"Generated features shape: {features_train.shape}")

# Add target column
features_train[TARGET_COLUMN] = y_train.values
X_train_with_features = pd.concat([X_train.reset_index(drop=True), features_train.reset_index(drop=True)], axis=1)

print(f"Combined train data shape: {X_train_with_features.shape}")

# ============================================================================
# EDA-1: Initial Report
# ============================================================================
print("\n=== EDA-1: Initial Report ===")
eda_cache = {}

# Missing values report
missing_counts = X_train_with_features.isna().sum().sort_values(ascending=False)
missing_pct = (X_train_with_features.isna().mean() * 100).sort_values(ascending=False)
missing_report = pd.DataFrame({"missing_count": missing_counts, "missing_pct": missing_pct})
missing_with_values = missing_report[missing_report["missing_count"] > 0]

eda_cache['missing_values'] = missing_with_values.to_dict()
print(f"Missing values: {len(missing_with_values)} columns with missing data")

# Duplicate rows
duplicate_count = X_train_with_features.duplicated().sum()
eda_cache['duplicate_rows'] = int(duplicate_count)
print(f"Duplicate rows: {duplicate_count}")

# Data types
dtype_counts = X_train_with_features.dtypes.astype(str).value_counts()
eda_cache['dtype_distribution'] = dtype_counts.to_dict()

# ============================================================================
# EDA-2: Outcome Variable Deep Dive
# ============================================================================
print("\n=== EDA-2: Outcome Variable Deep Dive ===")

target_counts = X_train_with_features[TARGET_COLUMN].value_counts(dropna=False)
target_pct = X_train_with_features[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100

target_report = pd.DataFrame({
    "count": target_counts,
    "pct": target_pct.round(2)
})

eda_cache['target_distribution'] = {
    'counts': target_counts.to_dict(),
    'percentages': target_pct.round(2).to_dict()
}

# Imbalance ratio
if len(target_counts) > 1:
    imbalance_ratio = target_counts.max() / target_counts.min()
    eda_cache['imbalance_ratio'] = float(imbalance_ratio)
    print(f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")
else:
    print("Only one class detected in target.")

print(f"Target distribution:\n{target_report}")

# ============================================================================
# EDA-3: Train-Test Split Check
# ============================================================================
print("\n=== EDA-3: Train-Test Split Check ===")

train_target_rate = X_train_with_features[TARGET_COLUMN].mean()
test_target_rate = y_test.mean()
abs_gap = abs(train_target_rate - test_target_rate)

split_report = {
    'train': {
        'rows': len(X_train_with_features),
        f'{TARGET_COLUMN}_rate': float(train_target_rate)
    },
    'test': {
        'rows': len(X_test),
        f'{TARGET_COLUMN}_rate': float(test_target_rate)
    },
    'absolute_rate_gap': float(abs_gap)
}

eda_cache['split_report'] = split_report
print(f"Train {TARGET_COLUMN} rate: {train_target_rate:.4f}")
print(f"Test  {TARGET_COLUMN} rate: {test_target_rate:.4f}")
print(f"Absolute rate gap: {abs_gap:.4f}")

# ============================================================================
# EDA-4: Feature-Level Deep Dive (Train Set Only)
# ============================================================================
print("\n=== EDA-4: Feature-Level Deep Dive ===")

# Exclude non-numeric and metadata columns
exclude_cols = {
    "id",
    "comment_text",
    TARGET_COLUMN,
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
}

numeric_cols = [
    c for c in X_train_with_features.select_dtypes(include=[np.number]).columns
    if c not in exclude_cols
]

print(f"Numeric feature columns analyzed: {len(numeric_cols)}")

# Occurrence table: how often each feature is non-zero
occurrence_df = pd.DataFrame(index=numeric_cols)
occurrence_df["non_zero_count"] = (X_train_with_features[numeric_cols] != 0).sum(axis=0)
occurrence_df["non_zero_pct"] = (occurrence_df["non_zero_count"] / len(X_train_with_features) * 100).round(2)

non_toxic_mask = X_train_with_features[TARGET_COLUMN] == 0
toxic_mask = X_train_with_features[TARGET_COLUMN] == 1

occurrence_df["non_zero_count_non_toxic"] = (X_train_with_features.loc[non_toxic_mask, numeric_cols] != 0).sum(axis=0)
occurrence_df["non_zero_count_toxic"] = (X_train_with_features.loc[toxic_mask, numeric_cols] != 0).sum(axis=0)
occurrence_df["non_zero_pct_non_toxic"] = (
    occurrence_df["non_zero_count_non_toxic"] / max(non_toxic_mask.sum(), 1) * 100
).round(2)
occurrence_df["non_zero_pct_toxic"] = (
    occurrence_df["non_zero_count_toxic"] / max(toxic_mask.sum(), 1) * 100
).round(2)

occurrence_df = occurrence_df.sort_values("non_zero_count", ascending=False)
eda_cache['feature_occurrence'] = occurrence_df.to_dict()

# Correlation with target
feature_target_corr = (
    X_train_with_features[numeric_cols + [TARGET_COLUMN]]
    .corr(numeric_only=True)[TARGET_COLUMN]
    .drop(TARGET_COLUMN)
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

eda_cache['feature_target_correlation'] = {
    col: float(val) for col, val in feature_target_corr.items()
}

# Feature means by target class
feature_means = (
    X_train_with_features.groupby(TARGET_COLUMN)[numeric_cols]
    .mean()
    .T
    .rename(columns={0: "non_toxic", 1: "toxic"})
)
feature_means["toxic_minus_non_toxic"] = feature_means["toxic"] - feature_means["non_toxic"]
feature_means = feature_means.sort_values(
    "toxic_minus_non_toxic", key=lambda s: s.abs(), ascending=False
)

eda_cache['feature_means_by_class'] = {
    col: {
        "non_toxic": float(feature_means.loc[col, "non_toxic"]),
        "toxic": float(feature_means.loc[col, "toxic"]),
        "diff": float(feature_means.loc[col, "toxic_minus_non_toxic"]),
    }
    for col in feature_means.index  # all features, not just top 15
}

print(f"Top correlated features: {list(feature_target_corr.head(5).index)}")

# ============================================================================
# EDA-5: Identity Risk Analysis
# ============================================================================
print("\n=== EDA-5: Identity Risk Analysis ===")

TRAIN_FEATURES_PATH = "data/processed/train_set_with_features.csv"
identity_groups = [
    ("identity_race",        "Race / Ethnicity"),
    ("identity_gender",      "Gender"),
    ("identity_sexuality",   "Sexual Orientation"),
    ("identity_religion",    "Religion"),
    ("identity_disability",  "Disability"),
    ("identity_nationality", "Nationality"),
]

try:
    df_full = pd.read_csv(TRAIN_FEATURES_PATH)
    baseline_tox = float(df_full[TARGET_COLUMN].mean())
    total_n = len(df_full)

    groups = []
    for col, label in identity_groups:
        if col not in df_full.columns:
            continue
        mask = df_full[col] == 1
        count = int(mask.sum())
        if count == 0:
            continue
        tox_rate = float(df_full.loc[mask, TARGET_COLUMN].mean())
        ih_rate = float(df_full.loc[mask, "identity_hate"].mean()) if "identity_hate" in df_full.columns else 0.0
        rel_risk = round(tox_rate / baseline_tox, 3) if baseline_tox > 0 else 0.0
        groups.append({
            "key": col,
            "label": label,
            "count": count,
            "pct_of_dataset": round(count / total_n * 100, 2),
            "toxic_rate": round(tox_rate, 4),
            "relative_risk": rel_risk,
            "identity_hate_rate": round(ih_rate, 4),
        })

    any_mask = df_full.get("identity_mention_count", pd.Series(0, index=df_full.index)) > 0
    any_count = int(any_mask.sum())
    any_tox = float(df_full.loc[any_mask, TARGET_COLUMN].mean()) if any_count > 0 else 0.0
    ih_total = int(df_full["identity_hate"].sum()) if "identity_hate" in df_full.columns else 0

    eda_cache["identity_risk"] = {
        "baseline_toxic_rate": round(baseline_tox, 4),
        "total_comments": total_n,
        "any_identity_mention_count": any_count,
        "any_identity_mention_pct": round(any_count / total_n * 100, 2),
        "any_identity_mention_toxic_rate": round(any_tox, 4),
        "identity_hate_total": ih_total,
        "identity_hate_rate": round(ih_total / total_n, 5) if total_n > 0 else 0.0,
        "groups": sorted(groups, key=lambda g: g["relative_risk"], reverse=True),
    }
    print(f"Identity groups analysed: {len(groups)}")
    for g in eda_cache["identity_risk"]["groups"]:
        print(f"  {g['label']}: n={g['count']}, rel_risk={g['relative_risk']}x")
except Exception as exc:
    print(f"WARNING: Identity risk analysis skipped: {exc}")
    eda_cache["identity_risk"] = {}

# ============================================================================
# EDA-6: Feature-Feature Correlation Matrix
# ============================================================================
print("\n=== EDA-6: Feature Correlation Matrix ===")

corr_matrix = (
    X_train_with_features[numeric_cols]
    .corr(numeric_only=True)
    .values
    .round(4)
    .tolist()
)

eda_cache['feature_correlation_matrix'] = {
    'features': numeric_cols,
    'matrix': corr_matrix,
}
print(f"Correlation matrix: {len(numeric_cols)}×{len(numeric_cols)}")

# ============================================================================
# EDA-7: Feature Distributions (per-class histogram densities)
# ============================================================================
print("\n=== EDA-7: Feature Distributions ===")

BINARY_FEATURES = {
    'vader_is_negative', 'has_second_person', 'has_url_or_ip',
    'identity_race', 'identity_gender', 'identity_sexuality',
    'identity_religion', 'identity_disability', 'identity_nationality',
}

non_toxic_mask = X_train_with_features[TARGET_COLUMN] == 0
toxic_mask     = X_train_with_features[TARGET_COLUMN] == 1
nt_total = non_toxic_mask.sum()
t_total  = toxic_mask.sum()

feature_distributions = {}
for feat in numeric_cols:
    vals_all = X_train_with_features[feat].values.astype(float)
    vals_nt  = X_train_with_features.loc[non_toxic_mask, feat].values.astype(float)
    vals_t   = X_train_with_features.loc[toxic_mask,     feat].values.astype(float)

    if feat in BINARY_FEATURES:
        edges = np.array([-0.5, 0.5, 1.5])
    else:
        lo = float(np.percentile(vals_all, 1))
        hi = float(np.percentile(vals_all, 99))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, 31)  # 30 bins

    cnt_nt, _ = np.histogram(vals_nt, bins=edges)
    cnt_t,  _ = np.histogram(vals_t,  bins=edges)

    # Density: fraction of each class (%) so imbalanced classes compare fairly
    density_nt = (cnt_nt / nt_total * 100).round(4).tolist()
    density_t  = (cnt_t  / t_total  * 100).round(4).tolist()

    feature_distributions[feat] = {
        'bin_edges': [round(float(e), 6) for e in edges],
        'non_toxic': density_nt,
        'toxic':     density_t,
        'is_binary': feat in BINARY_FEATURES,
    }

eda_cache['feature_distributions'] = feature_distributions
print(f"Distributions computed for {len(feature_distributions)} features")

# ============================================================================
# EDA-8: Modeling Readiness Checklist
# ============================================================================
print("\n=== EDA-8: Modeling Readiness Checklist ===")

readiness = {
    "split_fixed_and_leakage_safe": True,
    "train_has_engineered_features": True,
    "test_is_raw": True,
    "target_imbalance_detected": eda_cache['imbalance_ratio'] > 1.5,
    "candidate_features_count": len(numeric_cols),
    "top_candidate_features": list(feature_target_corr.head(10).index),
}

eda_cache['modeling_readiness'] = readiness
print("Readiness checklist:")
for k, v in readiness.items():
    print(f"  {k}: {v}")

# ============================================================================
# Save to eda_cache.json
# ============================================================================
print(f"\nSaving EDA cache to {OUTPUT_PATH}...")

# Convert numpy types to native Python types for JSON serialization
def convert_types(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(item) for item in obj]
    return obj

eda_cache_serializable = convert_types(eda_cache)

with open(OUTPUT_PATH, 'w') as f:
    json.dump(eda_cache_serializable, f, indent=2)

print(f"✓ EDA cache saved to {OUTPUT_PATH}")
print(f"Cache keys: {list(eda_cache_serializable.keys())}")
