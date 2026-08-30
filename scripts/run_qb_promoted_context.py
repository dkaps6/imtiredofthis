#!/usr/bin/env python3
"""Build production team context matching the M89/M90 QB feature semantics.

This is deliberately run after the legacy TeamForm builder and before model
bridges. It overlays only the fields whose semantics were corrected in M89,
using completed games strictly before the active slate week. The prior season is
required; current-season history is added when it exists. Sportsbook data is not
read here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.correct_m89_team_semantics import build_observations
from scripts.runtime_context import resolve_prior_season, resolve_season, resolve_week
from scripts.utils.pbp import get_pbp

TEAM_FORM = Path("data/team_form.csv")
OUT = Path("data/qb_promoted_team_context.csv")
AUDIT = Path("data/qb_promoted_context_audit.csv")
HISTORY_GAMES = 8


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _regular(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "season_type" in x.columns:
        q = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    if "game_type" in x.columns:
        q = x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    return x


def _play_counts(season: int) -> pd.DataFrame:
    """Completed offensive play counts used only for M89 off_plays semantics."""
    x = _regular(_lower(get_pbp(int(season), min_rows=1)))
    if not {"week", "posteam"}.issubset(x.columns):
        raise RuntimeError(f"PBP {season} missing week/posteam for play counts")
    x["posteam"] = x["posteam"].map(canon_team)
    qb = _num(x, "qb_dropback", 0).fillna(0).eq(1)
    rush = _num(x, "rush_attempt", 0).fillna(0).eq(1)
    x["_off_play"] = (qb | rush).astype(int)
    q = x.loc[x["posteam"].ne("") & x["_off_play"].eq(1)].copy()
    out = q.groupby(["week", "posteam"], as_index=False).size().rename(
        columns={"posteam": "team", "size": "plays_est"}
    )
    out["season"] = int(season)
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype(int)
    return out[["season", "week", "team", "plays_est"]]


def _season_observations(season: int, *, required: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    try:
        obs, audit = build_observations([int(season)])
        plays = _play_counts(int(season))
        obs = obs.merge(plays, on=["season", "week", "team"], how="left", validate="one_to_one")
        return obs, audit, "available"
    except Exception as exc:
        if required:
            raise
        print(f"[qb_promoted_context] current-season history unavailable for {season}: {exc}")
        return pd.DataFrame(), pd.DataFrame(), f"unavailable:{type(exc).__name__}"


def _prior(obs: pd.DataFrame, team: str, season: int, week: int) -> pd.DataFrame:
    q = obs.loc[
        obs["team"].eq(canon_team(team))
        & (
            obs["season"].lt(int(season))
            | (obs["season"].eq(int(season)) & obs["week"].lt(int(week)))
        )
    ].sort_values(["season", "week"])
    return q.tail(HISTORY_GAMES)


def _mean(q: pd.DataFrame, name: str) -> float:
    if name not in q.columns:
        return np.nan
    s = pd.to_numeric(q[name], errors="coerce")
    return float(s.mean()) if s.notna().any() else np.nan


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--prior-season", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    args = p.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    prior = int(args.prior_season if args.prior_season is not None else resolve_prior_season())
    week = int(args.week if args.week is not None else resolve_week())
    if prior >= season:
        raise RuntimeError(f"prior season must precede active season: prior={prior} season={season}")
    if not TEAM_FORM.exists() or TEAM_FORM.stat().st_size == 0:
        raise RuntimeError(f"missing TeamForm: {TEAM_FORM}")

    prior_obs, prior_audit, prior_status = _season_observations(prior, required=True)
    current_obs, current_audit, current_status = _season_observations(season, required=False)
    obs = (
        pd.concat([prior_obs, current_obs], ignore_index=True)
        if not current_obs.empty
        else prior_obs.copy()
    )
    obs["team"] = obs["team"].map(canon_team)

    tf = _lower(pd.read_csv(TEAM_FORM, low_memory=False))
    team_col = "team" if "team" in tf.columns else "team_abbr" if "team_abbr" in tf.columns else None
    if not team_col:
        raise RuntimeError("TeamForm missing team/team_abbr")
    tf["team"] = tf[team_col].map(canon_team)
    if tf["team"].eq("").any() or tf.duplicated("team").any():
        raise RuntimeError("TeamForm has invalid/duplicate canonical teams")

    mappings = {
        "true_proe": "true_proe",
        "proe": "true_proe",
        "neutral_pace_true": "neutral_pace_true",
        "neutral_pace": "neutral_pace_true",
        "pressure_rate_allowed": "hit_sack_pressure_rate_allowed",
        "pressure_rate_generated": "hit_sack_pressure_rate_generated",
        "hit_sack_pressure_rate_allowed": "hit_sack_pressure_rate_allowed",
        "hit_sack_pressure_rate_generated": "hit_sack_pressure_rate_generated",
        "pass_attempts_per_dropback": "pass_attempts_per_dropback",
        "pass_rate_off": "pass_rate_off",
        "pass_rate_faced": "pass_rate_faced",
        "def_pass_epa_allowed": "def_pass_epa_allowed",
        "def_pass_success_allowed": "def_pass_success_allowed",
        "def_ypa_allowed": "def_ypa_allowed",
        "off_ypa": "off_ypa",
        "off_pass_epa": "off_pass_epa",
        "plays_est": "plays_est",
    }
    context_rows = []
    for idx, row in tf.iterrows():
        team = canon_team(row["team"])
        hist = _prior(obs, team, season, week)
        if hist.empty:
            raise RuntimeError(
                f"no pregame promoted QB history for team={team} season={season} week={week}"
            )
        hist_season = pd.to_numeric(hist["season"], errors="coerce")
        current_games = int(hist_season.eq(season).sum())
        prior_games = int(hist_season.eq(prior).sum())
        latest = hist.sort_values(["season", "week"]).iloc[-1]
        source_seasons = sorted({int(v) for v in hist_season.dropna().tolist()})

        rec = {
            "team": team,
            "season": season,
            "week": week,
            "qb_context_history_games": int(len(hist)),
            "qb_context_current_games": current_games,
            "qb_context_prior_games": prior_games,
            "qb_context_latest_season": int(latest["season"]),
            "qb_context_latest_week": int(latest["week"]),
            "qb_context_source_seasons": ",".join(str(v) for v in source_seasons),
        }
        for target, source in mappings.items():
            value = _mean(hist, source)
            rec[target] = value
            if np.isfinite(value):
                tf.at[idx, target] = value
        rec["qb_context_version"] = "M89_M90_PROMOTED_V1"
        context_rows.append(rec)

    context = pd.DataFrame(context_rows)
    conv = pd.to_numeric(context["pass_attempts_per_dropback"], errors="coerce")
    if conv.isna().any() or not conv.between(0.50, 1.0, inclusive="both").all():
        bad = context.loc[
            conv.isna() | ~conv.between(0.50, 1.0, inclusive="both"),
            ["team", "pass_attempts_per_dropback"],
        ]
        raise RuntimeError(f"invalid promoted attempt conversion rows: {bad.to_dict('records')}")

    # Keep the legacy TeamForm file populated for compatibility while Team
    # Context v3 becomes the canonical downstream authority.
    tf["qb_promoted_context_applied"] = 1
    tf["qb_promoted_context_version"] = "M89_M90_PROMOTED_V1"
    tf["qb_promoted_context_target_week"] = week
    tf.to_csv(TEAM_FORM, index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(OUT, index=False)

    audit = (
        pd.concat([prior_audit, current_audit], ignore_index=True)
        if not current_audit.empty
        else prior_audit.copy()
    )
    audit["active_season"] = season
    audit["active_week"] = week
    audit["prior_history_status"] = prior_status
    audit["current_history_status"] = current_status
    audit["teams_contextualized"] = len(context)
    audit.to_csv(AUDIT, index=False)
    print(
        f"[qb_promoted_context] season={season} week={week} teams={len(context)} "
        f"prior={prior_status} current={current_status}"
    )
    print(f"[qb_promoted_context] attempt conversion range={conv.min():.4f}..{conv.max():.4f}")
    print(f"[qb_promoted_context] wrote {OUT}, overlaid {TEAM_FORM}, audit={AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())