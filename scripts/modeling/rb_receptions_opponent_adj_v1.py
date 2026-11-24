"""RB_RECEPTIONS_OPPONENT_ADJ_V1: opponent defense adjustment for RB receptions.

Promoted 2026-11-24 from OPPONENT_POSITION_DEFENSE_ADJUSTMENT_RESULTS_V1 (preregistered,
commit c8cdb98). It was the only market x position group to pass all three gates:
- MAE improved in both test folds (2024 -> 2025, 2024+2025 -> 2026);
- b > 0 in both fits;
- game-clustered P(improve) 0.988 on the 2026 fold.

    final RB receptions mean = model mean x (1 + B x D)
    D = n / (n + K) x (allowed_pg / league_pg - 1)

`allowed_pg` is the opponent's mean per-game receptions allowed to RBs over this season's
weeks BEFORE the slate week; `league_pg` is the mean of that over all defenses; `n` is the
opponent's prior games. Football-only: no sportsbook variable is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "RB_RECEPTIONS_OPPONENT_ADJ_V1"
B = 0.1932
SHRINK_K = 4.0
CONTEXT = Path("data/rb_receptions_opponent_context.csv")
POSITIONS = {"RB"}


def build_context(stats: pd.DataFrame, opponents: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Per-defense D for the slate week, strictly from weeks before `week`.

    stats: weekly player rows with season, week, team, position, receptions.
    opponents: season, week, team, opponent (one row per team-game).
    """
    s = stats.copy()
    s = s[(pd.to_numeric(s["season"], errors="coerce") == int(season)) & (pd.to_numeric(s["week"], errors="coerce") < int(week))]
    s = s[s["position"].astype(str).str.upper().isin(POSITIONS)]
    o = opponents[["season", "week", "team", "opponent"]].drop_duplicates()
    s = s.merge(o, on=["season", "week", "team"], how="inner")
    per_game = s.groupby(["week", "opponent"])["receptions"].sum().reset_index()
    if per_game.empty:
        return pd.DataFrame(columns=["season", "week", "defense", "games", "allowed_pg", "league_pg", "D", "version"])
    per = per_game.groupby("opponent")["receptions"].agg(["mean", "size"])
    league = float(per["mean"].mean())
    out = pd.DataFrame({
        "season": int(season), "week": int(week), "defense": per.index.astype(str),
        "games": per["size"].astype(int).values, "allowed_pg": per["mean"].values, "league_pg": league,
    })
    out["D"] = (out["games"] / (out["games"] + SHRINK_K)) * (out["allowed_pg"] / league - 1.0) if league > 0 else 0.0
    out["version"] = VERSION
    return out.reset_index(drop=True)


def load_context(season: int, week: int, path: Path = CONTEXT) -> dict[str, float]:
    """Return {defense: D}. Fails closed when the artifact is missing or for another slate."""
    if int(week) <= 1:
        return {}
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{VERSION}: {path} missing; build_rb_receptions_opponent_context_v1.py must run before pricing")
    ctx = pd.read_csv(path)
    if ctx.empty:
        raise RuntimeError(f"{VERSION}: {path} is empty for week {week}")
    bad = (pd.to_numeric(ctx["season"], errors="coerce") != int(season)) | (pd.to_numeric(ctx["week"], errors="coerce") != int(week))
    if bad.any():
        raise RuntimeError(f"{VERSION}: context is for another slate (expected season={season} week={week})")
    if (ctx["version"].astype(str) != VERSION).any():
        raise RuntimeError(f"{VERSION}: context version mismatch")
    d = pd.to_numeric(ctx["D"], errors="coerce")
    if not np.isfinite(d).all():
        raise RuntimeError(f"{VERSION}: non-finite D in context")
    return dict(zip(ctx["defense"].astype(str), d.astype(float)))


def factor(market: str, position: str, opponent: str, week: int, ctx: dict[str, float]) -> tuple[float, float, int]:
    """(multiplier, D, applied) for one priced row. Non-RB-receptions rows get (1, nan, 0)."""
    if market != "receptions" or str(position).upper() not in POSITIONS or int(week) <= 1:
        return 1.0, np.nan, 0
    opp = str(opponent).upper().strip()
    if opp not in ctx:
        # A defense with prior games must be in the context; a miss means a team-code
        # mismatch, and silently using D=0 would hide it.
        raise RuntimeError(f"{VERSION}: opponent {opp!r} not in context for week {week}")
    d = float(ctx[opp])
    return float(1.0 + B * d), d, 1
