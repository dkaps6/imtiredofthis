# scripts/volume.py
# Strict volume × efficiency projection helpers
# Uses team/opp context (pace, PROE, EPA, script) + player shares to build μ for each market.

from __future__ import annotations
import math
from typing import Optional, Dict
import numpy as np
import pandas as pd

# ---------- Tunables (conservative, easy to adjust) ----------
SECONDS_PER_PLAY_BASE = 26.8  # neutral seconds/snap baseline
PLAYS_NOISE_CAP = 0.12        # cap total plays inflation/deflation from pace z-blend
PROE_INFLUENCE = 0.45         # how much PROE tilts pass volume
SCRIPT_ATT_DELTA = 0.08       # favored teams pass a bit less, dogs a bit more (±8%)
RUN_FUNNEL_DELTA = 0.05       # shift pass→run in run-funnel and reverse in pass-funnel

# Efficiency priors used only if player_form lacks per-player priors.
DEFAULT_YPRR = 1.62           # yards per route run baseline
DEFAULT_YPT = 7.5             # yards per target baseline
DEFAULT_YPC = 4.25            # yards per carry baseline
DEFAULT_CATCH_RATE = 0.67     # receptions = targets * catch_rate

# Market name normalization (map your internal markets -> handlers)
RECEPTIONS_KEYS = {"receptions", "rec", "player_receptions"}
REC_YARDS_KEYS  = {"rec_yards", "receiving_yards", "player_reception_yds"}
RUSH_YARDS_KEYS = {"rush_yards", "rushing_yards", "player_rush_yds"}
RUSH_ATT_KEYS   = {"rush_att", "rushing_attempts", "player_rush_att"}
PASS_YARDS_KEYS = {"pass_yards", "passing_yards", "player_pass_yds"}

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _mean_if_any(values: list[float], default: float) -> float:
    arr = [v for v in values if v is not None and not np.isnan(v)]
    return np.mean(arr) if arr else default

# ---------------- Core: game-level plays & splits ----------------

def project_total_plays(row: pd.Series) -> float:
    """
    Estimate team offensive plays using pace_z (off+def blended), script, and mild noise caps.
    Expects: row['pace_z'] (blend), row['home_wp' or 'away_wp'] depending on team flag,
    row['is_home'] boolean (0/1). If pace_z missing, uses 0.
    """
    pace_z = _safe_float(row.get("pace_z"), 0.0)
    # Bound pace effect
    pace_mult = 1.0 + np.clip(0.5 * pace_z, -PLAYS_NOISE_CAP, PLAYS_NOISE_CAP)

    # Convert seconds
