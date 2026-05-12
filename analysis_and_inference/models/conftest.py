"""Shared pytest fixtures for models smoke tests.

Pytest auto-loads this file, so every test in any subfolder gets these
fixtures and the project root is on sys.path.
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
