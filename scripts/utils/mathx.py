# scripts/utils/mathx.py
"""
Mathematical and statistical helpers for pricing, Monte Carlo, etc.
"""

import numpy as np
import pandas as pd

def zscore(series):
    """Return z-score normalized pandas Series."""
    return (series - series.mean()) / (series.std(ddof=0) + 1e-9)


def weighted_mean(values, weights):
    """Compute weighted mean with numeric stability."""
    values, weights = np.array(values), np.array(weights)
    if weights.sum() == 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)


def clip_probabilities(p):
    """Clip probabilities to avoid 0/1 extremes."""
    return np.clip(p, 1e-6, 1 - 1e-6)


def blend_probs(p1, p2, w1=0.5, w2=0.5):
    """Blend two probability estimates."""
    p1, p2 = clip_probabilities(p1), clip_probabilities(p2)
    return (p1 * w1 + p2 * w2) / (w1 + w2)


def log_loss(y_true, y_pred):
    """Log loss for calibration diagnostics."""
    y_pred = clip_probabilities(y_pred)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def kelly_fraction(edge, odds):
    """Compute Kelly Criterion fraction (edge% / odds)."""
    odds_decimal = odds if odds > 1 else 1 + abs(odds)
    return np.clip(edge / (odds_decimal - 1), 0, 1)


def rolling_corr(x, y, window=5):
    """Rolling correlation used in SGP correlation models."""
    return pd.Series(x).rolling(window).corr(pd.Series(y))
