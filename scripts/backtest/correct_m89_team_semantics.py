#!/usr/bin/env python3
"""M89 Phase 0: correct historical team-feature semantics.

This correction layer keeps two source families deliberately separate:

- nflverse parsed PBP is the source for dropbacks, sacks/hits, game state,
  expected-pass probability, pace, YAC/explosive mechanics, and EPA/success.
- nflverse official weekly player stats are the source of truth for official
  pass attempts and official passing yards.

That distinction is required because parsed PBP is not an official-stat ledger:
its pass-attempt semantics include plays (notably sacks and some nullified play
representations) that need not reproduce the official box score exactly.

The script takes the existing leakage-safe team_weekly_history.csv and replaces
or augments fields whose prior labels/construction were too strong:

- `proe` becomes situation-adjusted pass rate over expected using xpass /
  pass_probability when available.
- `neutral_pace` becomes within-drive seconds/play in neutral game states.
- pressure remains the public-PBP sack-or-QB-hit proxy and is explicitly named.
- `pass_attempts_per_dropback` uses official team attempts divided by PBP
  dropbacks, so the MC converts projected dropbacks to official attempts using
  the correct statistical target.
- team offensive/defensive YPA uses official weekly attempts/yards.

Target-week cutoffs are still enforced later by the historical context factory;
these rows are completed-game observations, not target-game pregame features.
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


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


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


def _load_official_team_passing(season: int) -> pd.DataFrame:
    """Return official team-week pass attempts/yards from nflverse weekly stats."""
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    w = _lower(_to_pandas(raw))
    if "season" not in w.columns:
        w["season"] = int(season)
    w["season"] = pd.to_numeric(w["season"], errors="coerce")
    w["week"] = pd.to_numeric(w.get("week"), errors="coerce")
    w = w.loc[w["season"].eq(int(season)) & w["week"].between(1, 18)].copy()

    team_col = next((c for c in ["recent_team", "team", "posteam"] if c in w.columns), None)
    att_col = next((c for c in ["attempts", "passing_attempts", "pass_attempts"] if c in w.columns), None)
    yd_col = next((c for c in ["passing_yards", "pass_yards"] if c in w.columns), None)
    if not team_col or not att_col or not yd_col:
        raise RuntimeError(
            f"official weekly stats {season} missing team/attempts/yards: "
            f"team={team_col} attempts={att_col} yards={yd_col}"
        )

    w["team"] = w[team_col].map(canon_team)
    w["official_pass_attempts"] = pd.to_numeric(w[att_col], errors="coerce").fillna(0.0)
    w["official_pass_yards"] = pd.to_numeric(w[yd_col], errors="coerce").fillna(0.0)
    w = w.loc[w["team"].ne("")].copy()
    out = (
        w.groupby(["season", "week", "team"], as_index=False)
        .agg(
            official_pass_attempts=("official_pass_attempts", "sum"),
            official_pass_yards=("official_pass_yards", "sum"),
        )
    )
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


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
    if "game_id" not in keys or "posteam" not in keys or "drive" not in keys:
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
        x["_off_play"] = (
            _num(x, "qb_dropback", 0).fillna(0).eq(1)
            | _num(x, "rush_attempt", 0).fillna(0).eq(1)
        ).astype(int)
        x["_dropback"] = _num(x, "qb_dropback", 0).fillna(0).eq(1)
        # Parsed-PBP attempt is retained only for route/deep-play denominators.
        # It is NOT the official-attempt source of truth.
        x["_parsed_pass_attempt"] = _num(x, "pass_attempt", 0).fillna(0).eq(1)
        x["_hit_sack"] = (
            _num(x, "sack", 0).fillna(0).eq(1)
            | _num(x, "qb_hit", 0).fillna(0).eq(1)
        ).astype(int)
        x["_deep20"] = (
            _num(x, "air_yards").ge(20) & x["_parsed_pass_attempt"]
        ).astype(int)
        x["_complete20"] = (
            _num(x, "passing_yards").ge(20) & x["_parsed_pass_attempt"]
        ).astype(int)

        official = _load_official_team_passing(season)
        official_lookup = {
            (int(r.week), canon_team(r.team)): (
                float(r.official_pass_attempts), float(r.official_pass_yards)
            )
            for r in official.itertuples(index=False)
        }

        expected_col = (
            "xpass"
            if "xpass" in x.columns and _num(x, "xpass").notna().any()
            else "pass_probability"
            if "pass_probability" in x.columns and _num(x, "pass_probability").notna().any()
            else None
        )

        offense = x.loc[x["_off_play"].eq(1) & x["posteam"].ne("")].copy()
        pbp_keys = {
            (int(w), canon_team(t))
            for w, t in offense[["week", "posteam"]].drop_duplicates().itertuples(index=False, name=None)
        }
        official_keys = set(official_lookup)
        missing_official = sorted(pbp_keys - official_keys)
        if missing_official:
            raise RuntimeError(
                f"official weekly passing stats missing {len(missing_official)} team-weeks "
                f"for {season}: {missing_official[:8]}"
            )

        audits.append({
            "season": season,
            "pbp_rows": int(len(x)),
            "expected_pass_source": expected_col or "unavailable",
            "xpass_coverage": float(_num(x, "xpass").notna().mean()) if "xpass" in x.columns else 0.0,
            "pass_probability_coverage": float(_num(x, "pass_probability").notna().mean()) if "pass_probability" in x.columns else 0.0,
            "drive_available": int("drive" in x.columns),
            "score_differential_available": int("score_differential" in x.columns),
            "wp_available": int("wp" in x.columns),
            "official_attempt_source": "nflverse_weekly_player_stats",
            "pbp_team_weeks": int(len(pbp_keys)),
            "official_team_weeks_matched": int(len(pbp_keys & official_keys)),
            "official_team_week_coverage": float(len(pbp_keys & official_keys) / len(pbp_keys)) if pbp_keys else 0.0,
        })

        for (week, team), g in offense.groupby(["week", "posteam"]):
            week = int(week)
            team = canon_team(team)
            defense_g = offense.loc[offense["week"].eq(week) & offense["defteam"].eq(team)].copy()
            drop = g["_dropback"]
            opp_drop = defense_g["_dropback"]
            parsed_pass_att = g["_parsed_pass_attempt"]
            opp_parsed_pass_att = defense_g["_parsed_pass_attempt"]

            official_att, official_yards = official_lookup[(week, team)]
            opp_teams = [canon_team(v) for v in defense_g["posteam"].dropna().unique() if canon_team(v)]
            if len(opp_teams) != 1:
                raise RuntimeError(
                    f"cannot resolve single opponent offense for {season} W{week} defense {team}: {opp_teams}"
                )
            opp_team = opp_teams[0]
            if (week, opp_team) not in official_lookup:
                raise RuntimeError(f"missing official passing stats for {season} W{week} {opp_team}")
            opp_official_att, opp_official_yards = official_lookup[(week, opp_team)]

            true_proe = np.nan
            proe_n = 0
            if expected_col:
                ep = _num(g, expected_col)
                valid = ep.notna() & g["_dropback"].notna()
                if valid.any():
                    true_proe = float(
                        (g.loc[valid, "_dropback"].astype(float) - ep.loc[valid]).mean()
                    )
                    proe_n = int(valid.sum())

            pace, pace_n = _neutral_pace(g)
            dropbacks = int(drop.sum())
            opp_dropbacks = int(opp_drop.sum())
            att_per_dropback = float(official_att / dropbacks) if dropbacks else np.nan
            opp_att_per_dropback = float(opp_official_att / opp_dropbacks) if opp_dropbacks else np.nan
            if np.isfinite(att_per_dropback) and not (0.0 <= att_per_dropback <= 1.001):
                raise RuntimeError(
                    f"invalid official attempts/dropback {season} W{week} {team}: "
                    f"attempts={official_att} dropbacks={dropbacks} ratio={att_per_dropback}"
                )

            rows.append({
                "season": int(season),
                "week": week,
                "team": team,
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
                "pass_attempts_per_dropback": att_per_dropback,
                "official_pass_attempts": official_att,
                "official_pass_yards": official_yards,
                "official_pass_stat_source": "nflverse_weekly_player_stats",
                "opponent_official_pass_attempts": opp_official_att,
                "opponent_official_pass_yards": opp_official_yards,
                "opponent_pass_attempts_per_dropback": opp_att_per_dropback,
                "deep20_attempt_rate_off": float(g.loc[parsed_pass_att, "_deep20"].mean()) if parsed_pass_att.any() else np.nan,
                "deep20_completion_rate_allowed": float(defense_g.loc[opp_parsed_pass_att, "_complete20"].mean()) if opp_parsed_pass_att.any() else np.nan,
                "off_ypa": float(official_yards / official_att) if official_att else np.nan,
                "def_ypa_allowed": float(opp_official_yards / opp_official_att) if opp_official_att else np.nan,
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
    overwrite = [
        "proe", "neutral_pace", "pressure_rate_allowed", "pressure_rate_generated",
        "pass_attempts_per_dropback",
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
