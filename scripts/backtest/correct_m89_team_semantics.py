#!/usr/bin/env python3
"""M89 Phase 0: correct historical team-feature semantics from nflverse PBP.

This script is intentionally narrow. It takes the existing leakage-safe
team_weekly_history.csv and replaces/augments only fields whose prior labels were
stronger than their construction:

- `proe` becomes true situation-adjusted pass rate over expected using xpass /
  pass_probability when available.
- `neutral_pace` becomes within-drive seconds/play in neutral game states.
- pressure fields remain the public-PBP sack-or-QB-hit proxy, but are also
  emitted under explicit `hit_sack_pressure_*` names.

It also emits a few strictly historical team-game observables used by the M89
synthesis/casebook. Target-week cutoffs are still enforced later by the normal
historical context factory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.pbp import get_pbp


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _regular(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "season_type" in x.columns:
        reg = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            return reg
    if "game_type" in x.columns:
        reg = x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            return reg
    return x


def _neutral_pace(g: pd.DataFrame) -> tuple[float, int]:
    """Within-drive offensive seconds/play under a frozen neutral-state mask."""
    if g.empty or "game_seconds_remaining" not in g.columns or "play_id" not in g.columns:
        return np.nan, 0
    x = g.copy()
    mask = pd.Series(True, index=x.index)
    if "qtr" in x.columns:
        mask &= _num(x, "qtr").le(3)
    if "score_differential" in x.columns:
        mask &= _num(x, "score_differential").between(-7, 7, inclusive="both")
    if "wp" in x.columns:
        mask &= _num(x, "wp").between(0.20, 0.80, inclusive="both")
    x = x.loc[mask].copy()
    if len(x) < 2:
        return np.nan, 0
    keys = [c for c in ["game_id", "posteam", "drive"] if c in x.columns]
    if "game_id" not in keys or "posteam" not in keys:
        return np.nan, 0
    if "drive" not in keys:
        # Without a drive key, do not accidentally count the opponent possession
        # between two offensive snaps as offensive pace.
        return np.nan, 0
    x = x.sort_values(keys + ["play_id"])
    x["_prev_clock"] = x.groupby(keys)["game_seconds_remaining"].shift(1)
    delta = _num(x, "_prev_clock") - _num(x, "game_seconds_remaining")
    delta = delta.loc[delta.between(0, 90, inclusive="right")]
    if delta.empty:
        return np.nan, 0
    return float(delta.mean()), int(len(delta))


def build_observations(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    audits: list[dict] = []
    for season in sorted(set(int(s) for s in seasons)):
        x = _regular(_lower(get_pbp(season, min_rows=1)))
        if not {"week", "posteam", "defteam"}.issubset(x.columns):
            raise RuntimeError(f"PBP {season} missing week/posteam/defteam")
        x["posteam"] = x["posteam"].map(canon_team)
        x["defteam"] = x["defteam"].map(canon_team)
        for c in [
            "qb_dropback", "rush_attempt", "pass_attempt", "sack", "qb_hit",
            "success", "epa", "passing_yards", "yards_gained", "air_yards",
            "xpass", "pass_probability", "qtr", "score_differential", "wp",
            "game_seconds_remaining", "play_id", "drive",
        ]:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x["_off_play"] = (_num(x, "qb_dropback", 0).fillna(0).eq(1) | _num(x, "rush_attempt", 0).fillna(0).eq(1)).astype(int)
        x["_dropback"] = _num(x, "qb_dropback", 0).fillna(0).eq(1)
        x["_pass_attempt"] = _num(x, "pass_attempt", 0).fillna(0).eq(1)
        x["_hit_sack"] = (_num(x, "sack", 0).fillna(0).eq(1) | _num(x, "qb_hit", 0).fillna(0).eq(1)).astype(int)
        x["_deep20"] = (_num(x, "air_yards").ge(20) & x["_pass_attempt"]).astype(int)
        x["_complete20"] = (_num(x, "passing_yards").ge(20) & x["_pass_attempt"]).astype(int)

        expected_col = "xpass" if "xpass" in x.columns and _num(x, "xpass").notna().any() else "pass_probability" if "pass_probability" in x.columns and _num(x, "pass_probability").notna().any() else None
        audits.append({
            "season": season,
            "pbp_rows": int(len(x)),
            "expected_pass_source": expected_col or "unavailable",
            "xpass_coverage": float(_num(x, "xpass").notna().mean()) if "xpass" in x.columns else 0.0,
            "pass_probability_coverage": float(_num(x, "pass_probability").notna().mean()) if "pass_probability" in x.columns else 0.0,
            "drive_available": int("drive" in x.columns),
            "score_differential_available": int("score_differential" in x.columns),
            "wp_available": int("wp" in x.columns),
        })

        offense = x.loc[x["_off_play"].eq(1) & x["posteam"].ne("")].copy()
        for (week, team), g in offense.groupby(["week", "posteam"]):
            defense_g = offense.loc[offense["week"].eq(week) & offense["defteam"].eq(team)].copy()
            drop = g["_dropback"]
            opp_drop = defense_g["_dropback"]
            pass_att = g["_pass_attempt"]
            opp_pass_att = defense_g["_pass_attempt"]

            true_proe = np.nan
            proe_n = 0
            if expected_col:
                ep = _num(g, expected_col)
                valid = ep.notna() & g["_dropback"].notna()
                if valid.any():
                    true_proe = float((g.loc[valid, "_dropback"].astype(float) - ep.loc[valid]).mean())
                    proe_n = int(valid.sum())

            pace, pace_n = _neutral_pace(g)
            off_pass_yards = _num(g.loc[pass_att], "passing_yards", 0).fillna(0).sum() if pass_att.any() else 0.0
            def_pass_yards = _num(defense_g.loc[opp_pass_att], "passing_yards", 0).fillna(0).sum() if opp_pass_att.any() else 0.0

            rows.append({
                "season": int(season),
                "week": int(week),
                "team": canon_team(team),
                "proe": true_proe,
                "true_proe": true_proe,
                "true_proe_n": proe_n,
                "true_proe_source": expected_col or "unavailable",
                "neutral_pace": pace,
                "neutral_pace_true": pace,
                "neutral_pace_n": pace_n,
                "neutral_pace_source": "within_drive_neutral_score_wp" if pace_n else "unavailable",
                "pressure_rate_allowed": float(g.loc[drop, "_hit_sack"].mean()) if drop.any() else np.nan,
                "pressure_rate_generated": float(defense_g.loc[opp_drop, "_hit_sack"].mean()) if opp_drop.any() else np.nan,
                "hit_sack_pressure_rate_allowed": float(g.loc[drop, "_hit_sack"].mean()) if drop.any() else np.nan,
                "hit_sack_pressure_rate_generated": float(defense_g.loc[opp_drop, "_hit_sack"].mean()) if opp_drop.any() else np.nan,
                "pressure_semantic": "sack_or_qb_hit_proxy_not_full_pressure",
                "pass_rate_off": float(drop.mean()) if len(g) else np.nan,
                "pass_rate_faced": float(opp_drop.mean()) if len(defense_g) else np.nan,
                "deep20_attempt_rate_off": float(g.loc[pass_att, "_deep20"].mean()) if pass_att.any() else np.nan,
                "deep20_completion_rate_allowed": float(defense_g.loc[opp_pass_att, "_complete20"].mean()) if opp_pass_att.any() else np.nan,
                "off_ypa": float(off_pass_yards / pass_att.sum()) if pass_att.sum() else np.nan,
                "def_ypa_allowed": float(def_pass_yards / opp_pass_att.sum()) if opp_pass_att.sum() else np.nan,
                "def_pass_success_allowed": float(defense_g.loc[opp_drop, "success"].mean()) if opp_drop.any() and "success" in defense_g.columns else np.nan,
                "off_pass_epa": float(g.loc[drop, "epa"].mean()) if drop.any() and "epa" in g.columns else np.nan,
                "def_pass_epa_allowed": float(defense_g.loc[opp_drop, "epa"].mean()) if opp_drop.any() and "epa" in defense_g.columns else np.nan,
            })

    obs = pd.DataFrame(rows)
    if obs.empty:
        raise RuntimeError("M89 semantic builder produced zero rows")
    if obs.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("M89 semantic observations contain duplicate season/week/team")
    return obs.sort_values(["season", "week", "team"]).reset_index(drop=True), pd.DataFrame(audits)


def apply_corrections(base: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    b = _lower(base)
    b["team"] = b["team"].map(canon_team)
    c = _lower(corrected)
    overwrite = [
        "proe", "neutral_pace", "pressure_rate_allowed", "pressure_rate_generated",
    ]
    for col in overwrite:
        if col in b.columns:
            b = b.drop(columns=[col])
    extras = [c for c in corrected.columns if c not in {"season", "week", "team"}]
    b = b.drop(columns=[c for c in extras if c in b.columns], errors="ignore")
    out = b.merge(corrected, on=["season", "week", "team"], how="left", validate="one_to_one")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--seasons", required=True, help="Comma-separated seasons")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    args = p.parse_args()
    if not args.team_weekly.exists():
        raise RuntimeError(f"missing team weekly file {args.team_weekly}")
    seasons = [int(v.strip()) for v in args.seasons.split(",") if v.strip()]
    obs, audit = build_observations(seasons)
    base = pd.read_csv(args.team_weekly, low_memory=False)
    out = apply_corrections(base, obs)
    out.to_csv(args.team_weekly, index=False)
    args.observations.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(args.observations, index=False)
    audit.to_csv(args.audit, index=False)
    print("=== M89 SEMANTIC SOURCE AUDIT ===")
    print(audit.to_string(index=False))
    print(f"[m89_semantics] corrected team-week rows={len(out)} -> {args.team_weekly}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
