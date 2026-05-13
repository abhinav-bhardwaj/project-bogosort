"""
This code executes a quick end-to-end sanity check on the dense feature engineering
pipeline implemented in build_features.py. The script validates that:

1. The transformer runs successfully on real Jigsaw toxicity data.
2. Feature generation produces the expected output shape and columns.
3. Runtime performance is acceptable on a small representative sample.
4. Individual engineered features behave plausibly on hand-written examples.

Spot-check examples verify behavior for:
- highly toxic text,
- neutral/non-toxic text,
- slang + profanity-heavy text.

Note: This is not a formal unit test suite. Instead, it acts as a fast diagnostic
tool for debugging feature engineering logic during development before
running full model training or pytest-based validation.

Run with: uv run python analysis_and_inference/features/sanity_check.py
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from analysis_and_inference.features.build_features import DenseFeatureTransformer

DATA_PATH = "data/raw/jigsaw-dataset/train.csv"

# ---------------------------------------------------------------------------
# Load a small sample
# ---------------------------------------------------------------------------
df     = pd.read_csv(DATA_PATH)
sample = df.sample(1000, random_state=42)
train, test = train_test_split(sample, test_size=0.2, random_state=42)

print(f"Train rows: {len(train)}  |  Test rows: {len(test)}")

# ---------------------------------------------------------------------------
# DenseFeatureTransformer
# ---------------------------------------------------------------------------
print("\n--- DenseFeatureTransformer ---")
t0      = time.time()
dense   = DenseFeatureTransformer()
result  = dense.fit_transform(train)
elapsed = time.time() - t0

print(f"Output shape : {result.shape}")
print(f"Time (800 rows): {elapsed:.1f}s  →  ~{elapsed * 75:.0f}s estimated for 60K rows")
print(f"New columns  : {[c for c in result.columns if c not in train.columns]}")

# Spot-check a known comment
test_cases = [
    ("YOU are TOTALLY WORTHLESS you idiot!!!", "should score high on second_person, uppercase, profanity"),
    ("Thanks for the edit, looks great.",       "should score near zero on everything"),
    ("kys you f**king retard stfu",             "should score high on slang and profanity"),
]
print("\n--- Spot checks ---")
for text, description in test_cases:
    row    = pd.DataFrame({"comment_text": [text]})
    output = dense.transform(row).iloc[0]
    print(f"\n'{text[:60]}'")
    print(f"  ({description})")
    print(f"  vader_compound       : {output['vader_compound']:.3f}")
    print(f"  second_person_count  : {output['second_person_count']}")
    print(f"  uppercase_ratio      : {output['uppercase_ratio']:.2f}")
    print(f"  profanity_count      : {output['profanity_count']}")
    print(f"  slang_count          : {output['slang_count']}")

