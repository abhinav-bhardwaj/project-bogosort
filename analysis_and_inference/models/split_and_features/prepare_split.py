"""One-time data prep for models.

Reads the raw zipped Jigsaw CSV, keeps only `comment_text` and `toxic`,
makes a stratified 80/20 split, and saves it to disk so every model
loads the exact same X_train/X_test/y_train/y_test.

Run with:
    uv run python analysis_and_inference/models/split_and_features/prepare_split.py
"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH    = "data/raw/jigsaw-dataset/train.csv.zip"
OUTPUT_PATH = "analysis_and_inference/models/split_and_features/split.pkl"
TEST_SIZE   = 0.2
SEED        = 42


def main():
    print(f"Reading {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, compression="zip", usecols=["comment_text", "toxic"])
    df = df.dropna(subset=["comment_text", "toxic"])

    X = df[["comment_text"]]
    y = df["toxic"].astype(int)

    print(f"Total rows: {len(df)}  |  Toxic rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"X_train": X_train, "X_test": X_test,
                     "y_train": y_train, "y_test": y_test}, f)

    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Saved split to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
