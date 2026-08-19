"""Leakage-safe QB role/opportunity inference for historical backtests.

The historical roster universe can contain multiple active quarterbacks, while
sportsbook prop slates normally expose only the expected passer(s).  This module
uses information strictly before the prediction cutoff to select one primary QB
per team and estimate his share of team passing attempts.  It never uses the
target week's participation or result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _role_bonus(role: object) -> float:
    text = str(role or "").upper().replace(" ", "")
    if any(token in text for token in ("QB1", "QB01", "START", "FIRST")):
        return 1_000_000.0
    if any(token in text for token in ("QB2", "QB02", "SECOND")):
        return 100.0
    return 0.0


def _weighted_recent_attempts(rows: pd.DataFrame, *, team: str | None = None, max_games: int = 4) -> float:
    x = rows.copy()
    if team is not None and "team" in x.columns:
        x = x.loc[x["team"].astype(str).eq(str(team))]
    if x.empty:
        return 0.0
    x = x.sort_values(["season", "week"]).tail(max_games)
    attempts = _num(x, "pass_att").fillna(0.0).to_numpy(dtype=float)
    if not len(attempts):
        return 0.0
    weights = np.arange(1.0, len(attempts) + 1.0)
    return float(np.dot(attempts, weights) / weights.sum())


def _historical_attempt_share(rows: pd.DataFrame, *, team: str, season: int, prior_season: int) -> float:
    """Estimate primary-QB attempt share from pre-cutoff games only."""
    x = rows.copy()
    if x.empty:
        return np.nan
    current = x.loc[
        pd.to_numeric(x["season"], errors="coerce").eq(int(season))
        & x["team"].astype(str).eq(str(team))
    ].sort_values(["season", "week"]).tail(4)
    source = current
    if source.empty:
        source = x.loc[pd.to_numeric(x["season"], errors="coerce").eq(int(prior_season))].sort_values(["season", "week"]).tail(4)
    if source.empty:
        return np.nan
    att = _num(source, "pass_att")
    team_att = _num(source, "team_dropbacks")
    valid = att.notna() & team_att.gt(0)
    if not valid.any():
        return np.nan
    shares = (att.loc[valid] / team_att.loc[valid]).clip(0.0, 1.0)
    # The selected primary should receive starter-like volume.  Clamp only the
    # inferred share, not the team passing environment itself.
    return float(np.clip(shares.mean(), 0.70, 1.00))


def add_qb_opportunity(
    player_form: pd.DataFrame,
    pregame_universe: pd.DataFrame,
    history: pd.DataFrame,
    *,
    season: int,
    prior_season: int,
) -> pd.DataFrame:
    """Annotate player_form with one leakage-safe primary QB per active team.

    Selection priority is: explicit pregame depth role when available, recent
    pass attempts for the same team in the current season, recent current-season
    attempts elsewhere, then prior-season attempts.  Target-week data is absent
    because ``history`` is already cutoff-filtered by historical_context.py.
    """
    out = player_form.copy()
    out["qb_projection_eligible"] = 0
    out["qb_pass_att_share"] = np.nan
    out["qb_role_score"] = np.nan
    out["qb_role_source"] = "not_qb"

    u = pregame_universe.copy()
    u.columns = [str(c).strip().lower() for c in u.columns]
    if "role" not in u.columns:
        u["role"] = ""
    if "player_clean_key" not in u.columns:
        return out
    h = history.copy()
    h.columns = [str(c).strip().lower() for c in h.columns]

    qb_mask = out["position"].astype(str).str.upper().eq("QB")
    for team, qbs in out.loc[qb_mask].groupby("team", sort=False):
        scored: list[tuple[float, int, str]] = []
        for idx, row in qbs.iterrows():
            key = str(row["player_clean_key"])
            ph = h.loc[h["player_clean_key"].astype(str).eq(key)].copy()
            role_rows = u.loc[(u["team"].astype(str).eq(str(team))) & (u["player_clean_key"].astype(str).eq(key))]
            role = role_rows.iloc[0].get("role", "") if not role_rows.empty else row.get("role", "")
            same_team_recent = _weighted_recent_attempts(
                ph.loc[pd.to_numeric(ph.get("season"), errors="coerce").eq(int(season))],
                team=str(team),
            )
            current_any = _weighted_recent_attempts(
                ph.loc[pd.to_numeric(ph.get("season"), errors="coerce").eq(int(season))]
            )
            prior_any = _weighted_recent_attempts(
                ph.loc[pd.to_numeric(ph.get("season"), errors="coerce").eq(int(prior_season))]
            )
            score = _role_bonus(role) + same_team_recent * 1000.0 + current_any * 10.0 + prior_any
            source = "depth_role" if _role_bonus(role) >= 1_000_000 else "current_team_history" if same_team_recent > 0 else "current_history" if current_any > 0 else "prior_history" if prior_any > 0 else "no_history"
            out.at[idx, "qb_role_score"] = score
            out.at[idx, "qb_role_source"] = source
            scored.append((score, idx, key))

        if not scored:
            continue
        # Deterministic tie-break keeps the backtest reproducible when no role or
        # history exists.  We still project only one QB rather than assigning the
        # full team workload to every rostered quarterback.
        scored.sort(key=lambda item: (-item[0], item[2]))
        _, primary_idx, primary_key = scored[0]
        ph = h.loc[h["player_clean_key"].astype(str).eq(primary_key)].copy()
        share = _historical_attempt_share(ph, team=str(team), season=int(season), prior_season=int(prior_season))
        if not np.isfinite(share):
            share = 0.95
        out.at[primary_idx, "qb_projection_eligible"] = 1
        out.at[primary_idx, "qb_pass_att_share"] = float(share)
        for _, idx, _ in scored[1:]:
            out.at[idx, "qb_pass_att_share"] = 0.0
    return out
