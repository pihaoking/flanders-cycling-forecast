"""
Train/val/test split: 2023 / 2024 / 2025.

Earlier years are dropped because most stations were not yet installed:
only ~25 stations before 2022, and ~100 more came online during 2022.
"""

import pandas as pd

TRAIN_YEAR = 2023
VAL_YEAR = 2024
TEST_YEAR = 2025


def temporal_split(panel):
    train = panel[panel["year"] == TRAIN_YEAR].copy()
    val = panel[panel["year"] == VAL_YEAR].copy()
    test = panel[panel["year"] == TEST_YEAR].copy()
    return train, val, test
