"""
conftest.py — shared pytest fixtures for model smoke tests

This module centralizes reusable pytest fixtures so all model tests share
the same lightweight synthetic dataset and environment setup.

Using a shared fixture avoids duplicated test data definitions across files
and guarantees consistent testing conditions between models.


The fixture uses a small balanced dataset of toxic and non-toxic comments
to keep smoke tests fast, deterministic, and independent of external files.
This allows tests to validate pipeline behavior, prediction flow, and sklearn
compatibility without requiring full dataset loading or expensive training.

The dataset is intentionally minimal because the goal is not benchmarking
model quality, but verifying that pipelines:
- fit successfully,
- produce valid predictions,
- expose expected sklearn interfaces.

The project root is added dynamically to sys.path so tests remain runnable
from any working directory without manual path configuration.

Run with: Automatically loaded by pytest for all tests in the project.

"""

import os
import sys

from pathlib import Path
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_data():
    """50 synthetic comments, balanced 0/1 labels — enough for smoke tests."""
    toxic_examples = [
        "you are an idiot and a moron",
        "fuck off you piece of trash",
        "kys you worthless noob",
        "shut up you stupid loser",
        "i hate you so much, retard",
    ]
    clean_examples = [
        "thanks for the help today",
        "this is a wonderful comment",
        "i appreciate your input",
        "great work on the project",
        "really enjoyed reading this article",
    ]

    n_each = 25
    texts  = (toxic_examples * (n_each // len(toxic_examples) + 1))[:n_each] + \
             (clean_examples * (n_each // len(clean_examples) + 1))[:n_each]
    labels = [1] * n_each + [0] * n_each

    X = pd.DataFrame({"comment_text": texts})
    y = pd.Series(labels, name="toxic")
    return X, y
