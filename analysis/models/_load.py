"""
Shared loading logic for feature_evaluation, error_analysis, test_evaluation.
Import with: from analysis.models._load import load_bundle, load_val, load_test
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from analysis.models.data_pipeline import DataPipeline
from analysis.features.build_features import DenseFeatureTransformer


PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "analysis/models/artifacts"


def load_bundle():
    """Load best_model.pkl — returns model, scaler_dense, scaler_bert, threshold."""
    with open(os.path.join(ARTIFACTS_DIR, "best_model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler_dense"], bundle["scaler_bert"], bundle["threshold"]


def _build_features(X_raw, scaler_dense, scaler_bert, split="train"):
    """Reconstruct the same feature matrix used during training."""
    dense_transformer = DenseFeatureTransformer()
    dense_scaled = scaler_dense.transform(dense_transformer.transform(X_raw))

    tfidf = sp.load_npz(os.path.join(PROCESSED_DIR, f"tfidf_{split}.npz"))

    with open(os.path.join(PROCESSED_DIR, f"bert_{split}.pkl"), "rb") as f:
        bert = pickle.load(f)
    bert_scaled = scaler_bert.transform(bert)

    return hstack([
        sp.csr_matrix(dense_scaled),
        tfidf,
        sp.csr_matrix(bert_scaled),
    ]).tocsr()


def load_val(model, scaler_dense, scaler_bert):
    """
    Reconstruct the last CV fold val split — same logic as hypertuning_parameters.py.
    Returns X_val, y_val, X_val_raw, last_val_idx.
    """
    dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
    X_train_raw, _, y_train, _ = dp.get_data()
    y_train = y_train.values.ravel()

    # Rebuild full train feature matrix to get val indices
    X_train_proc = _build_features(X_train_raw, scaler_dense, scaler_bert, split="train")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _, last_val_idx = list(cv.split(X_train_proc, y_train))[-1]

    X_val     = X_train_proc[last_val_idx]
    y_val     = y_train[last_val_idx]
    X_val_raw = X_train_raw.iloc[last_val_idx].reset_index(drop=True)

    return X_val, y_val, X_val_raw, last_val_idx


def load_test(scaler_dense, scaler_bert):
    """Load and return X_test, y_test."""
    dp = DataPipeline("data/processed/test_train_data.pkl", label_columns=["toxic"])
    _, X_test_raw, _, y_test = dp.get_data()
    y_test = y_test.values.ravel()

    X_test = _build_features(X_test_raw, scaler_dense, scaler_bert, split="test")
    return X_test, y_test
