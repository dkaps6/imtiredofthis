#!/usr/bin/env python3
"""Football Matchup Transmission V1 Phase B/C historical residual audit.

Frozen by docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PHASE_BC_METHODS.md.
No sportsbook inputs. No candidate fitting. No production changes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.pbp import get_pbp

VERSION = "FOOTBALL_MATCHUP_TRANSMISSION_V1_PHASE_BC"
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(2, 19))
MIN_ROWS = 200
MIN_GAMES = 50
BOOT_REPS = 5000
BOOT_SEED = 20261006

FORBIDDEN_TOKENS = (
    "sportsbook", "bookmaker", "prop_line", "market_line", "over_odds", "under_odds",
    "spread_line", "total_line", "moneyline", "closing_line", "no_vig", "implied_prob",
)

RB_POS = {"RB", "FB", "HB"}


@dataclass(frozen=True)
class FeatureSpec:
    cohort: str
    feature: str
    family: str
    sign: int
    phase: str
    parity: str
    closed_family: str = ""
    opportunity: str = ""
    notes: str = ""


SOURCE_ROWS = [
    ("def_rush_epa", "nflverse PBP; same make_team_form defensive rush EPA definition", "EXACT_LIVE_SEMANTICS", 1, "M95A_M95B_CLOSED_OVERLAP"),
    ("dl_stuff_rate", "Sharp team form", "SOURCE_BLOCKED", 0, "no same-semantic historical archive"),
    ("dl_ybc_per_rush", "Sharp team form", "SOURCE_BLOCKED", 0, "no same-semantic historical archive"),
    ("light_box_rate", "historical nflverse participation vs live Sharp-preferred box field", "PARITY_UNPROVEN_DIAGNOSTIC_ONLY", 0, "provider mismatch"),
    ("heavy_box_rate", "historical nflverse participation vs live Sharp-preferred box field", "PARITY_UNPROVEN_DIAGNOSTIC_ONLY", 0, "provider mismatch"),
    ("wr_ypt_allowed", "historical official weekly player stats vs live Sharp field", "PARITY_UNPROVEN_DIAGNOSTIC_ONLY", 0, "provider mismatch"),
    ("te_ypt_allowed", "historical official weekly player stats vs live Sharp field", "PARITY_UNPROVEN_DIAGNOSTIC_ONLY", 0, "provider mismatch"),
    ("rb_ypt_allowed", "historical official weekly player stats vs live Sharp field", "PARITY_UNPROVEN_DIAGNOSTIC_ONLY", 0, "provider mismatch"),
    ("outside_ypt_allowed", "live Sharp field", "SOURCE_BLOCKED", 0, "no leakage-safe same-semantic history"),
    ("slot_ypt_allowed", "live Sharp field", "SOURCE_BLOCKED", 0, "no leakage-safe same-semantic history"),
    ("coverage_man_rate", "historical nflverse participation vs live Coverage v2/Sharp", "PARITY_UNPROVEN_DIAGNOSTIC_ONLY", 0, "provider mismatch"),
    ("coverage_zone_rate", "historical nflverse participation vs live Coverage v2/Sharp", "PARITY_UNPROVEN_DIAGNOSTIC_ONLY", 0, "provider mismatch"),
    ("middle_open_rate", "legacy live provider field", "SOURCE_BLOCKED", 0, "no exact historical source"),
    ("def_pass_epa_allowed", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("def_pass_success_allowed", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("def_ypa_allowed", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("true_proe", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("neutral_pace_true", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("pass_rate_off", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("plays_est", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("pass_rate_faced", "M89/M90 corrected public-football context", "EXACT_LIVE_SEMANTICS", 1, ""),
    ("pressure_mismatch", "M89/M90 sack-or-QB-hit pressure proxy", "EXACT_LIVE_SEMANTICS", 1, ""),
]


def _source_matrix() -> pd.DataFrame:
    return pd.DataFrame(SOURCE_ROWS, columns=["feature", "source", "parity_status", "same_live_semantics", "block_reason"])


def _feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []

    def add(
        cohort,
        feature,
        family,
        sign,
        *,
        parity="EXACT_LIVE_SEMANTICS",
        closed="",
        phase="B",
        opportunity="",
        notes="",
    ):
        specs.append(FeatureSpec(cohort, feature, family, int(sign), phase, parity, closed, opportunity, notes))

    add("RB_RUSH", "def_rush_epa", "run_defense", +1, closed="M95A_M95B_CLOSED_OVERLAP")
    add("RB_RUSH", "off_true_proe", "game_environment", -1)
    add("RB_RUSH", "off_pass_rate", "game_environment", -1)
    add("RB_RUSH", "off_plays", "game_environment", +1)
    add("RB_RUSH", "off_neutral_pace", "game_environment", -1)
    add("RB_RUSH", "def_pass_rate_faced", "game_environment", -1)
    add("RB_RUSH", "def_light_box_rate", "run_defense", +1, parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY")
    add("RB_RUSH", "def_heavy_box_rate", "run_defense", -1, parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY")

    for cohort, ypt_feature in [
        ("WR_REC", "def_wr_ypt_allowed"),
        ("TE_REC", "def_te_ypt_allowed"),
        ("RB_REC", "def_rb_ypt_allowed"),
    ]:
        add(cohort, "def_pass_epa_allowed", "receiving_defense", +1)
        add(cohort, "def_ypa_allowed", "receiving_defense", +1)
        add(cohort, "def_pass_success_allowed", "receiving_defense", +1)
        add(cohort, "off_true_proe", "game_environment", +1)
        add(cohort, "off_pass_rate", "game_environment", +1)
        add(cohort, "off_plays", "game_environment", +1)
        add(cohort, "off_neutral_pace", "game_environment", -1)
        add(cohort, "def_pass_rate_faced", "game_environment", +1)
        add(cohort, "pressure_mismatch", "game_environment", +1 if cohort == "RB_REC" else -1)
        add(cohort, ypt_feature, "position_receiving_defense", +1, parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY")
    add("TE_REC", "def_zone_rate", "coverage", +1, parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY")

    add("RB_RUSH_REC", "def_rush_epa", "run_defense", +1, closed="M95A_M95B_CLOSED_OVERLAP")
    add("RB_RUSH_REC", "def_pass_epa_allowed", "receiving_defense", +1)
    add("RB_RUSH_REC", "def_ypa_allowed", "receiving_defense", +1)
    add("RB_RUSH_REC", "def_pass_success_allowed", "receiving_defense", +1)
    add("RB_RUSH_REC", "off_plays", "game_environment", +1)
    add("RB_RUSH_REC", "off_neutral_pace", "game_environment", -1)
    add("RB_RUSH_REC", "def_rb_ypt_allowed", "position_receiving_defense", +1, parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY")

    for feature, family, sign in [
        ("def_pass_epa_allowed", "receiving_defense", +1),
        ("def_ypa_allowed", "receiving_defense", +1),
        ("def_pass_success_allowed", "receiving_defense", +1),
        ("off_true_proe", "game_environment", +1),
        ("off_pass_rate", "game_environment", +1),
        ("off_plays", "game_environment", +1),
        ("off_neutral_pace", "game_environment", -1),
        ("def_pass_rate_faced", "game_environment", +1),
        ("pressure_mismatch", "game_environment", -1),
    ]:
        add("QB_PASS_CONTROL", feature, family, sign, closed="M56_M83_CONTROL_ONLY")

    add(
        "RB_RUSH",
        "def_rush_epa",
        "run_defense",
        +1,
        phase="C",
        opportunity="rush_share",
        closed="M95A_M95B_CLOSED_OVERLAP",
    )
    add(
        "WR_REC",
        "def_wr_ypt_allowed",
        "position_receiving_defense",
        +1,
        phase="C",
        opportunity="target_share",
        parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY",
    )
    add(
        "TE_REC",
        "def_te_ypt_allowed",
        "position_receiving_defense",
        +1,
        phase="C",
        opportunity="target_share",
        parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY",
    )
    add(
        "TE_REC",
        "def_zone_rate",
        "coverage",
        +1,
        phase="C",
        opportunity="target_share",
        parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY",
    )
    add(
        "RB_REC",
        "def_rb_ypt_allowed",
        "position_receiving_defense",
        +1,
        phase="C",
        opportunity="target_share",
        parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY",
    )
    add(
        "RB_RUSH_REC",
        "def_rush_epa",
        "run_defense",
        +1,
        phase="C",
        opportunity="rush_share",
        closed="M95A_M95B_CLOSED_OVERLAP",
    )
    add(
        "RB_RUSH_REC",
        "def_rb_ypt_allowed",
        "position_receiving_defense",
        +1,
        phase="C",
        opportunity="target_share",
        parity="PARITY_UNPROVEN_DIAGNOSTIC_ONLY",
    )
    return specs


def _read(path: Path, label: str, usecols=None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False, usecols=usecols)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _key(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _regular(x: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in x.columns:
        q = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    if "game_type" in x.columns:
        q = x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    return x


def _mean_col(x: pd.DataFrame, names: Iterable[str]) -> float:
    for name in names:
        if name in x.columns:
            s = _num(x[name])
            if s.notna().any():
                return float(s.mean())
    return np.nan


def _prior_team_rows(
    team_weekly: pd.DataFrame,
    season: int,
    week: int,
    team: str,
    n_games: int = 8,
) -> pd.DataFrame:
    q = team_weekly.loc[
        team_weekly["team"].eq(team)
        & (
            team_weekly["season"].lt(season)
            | (team_weekly["season"].eq(season) & team_weekly["week"].lt(week))
        )
    ].sort_values(["season", "week"])
    return q.tail(n_games)


def build_exact_rush_epa_pregame(seasons: Iterable[int]) -> pd.DataFrame:
    """Mirror make_team_form's defensive rushing EPA definition, strict-prior."""
    rows: list[dict] = []
    for season in sorted(set(int(s) for s in seasons)):
        x = get_pbp(season, min_rows=1).copy()
        x.columns = [str(c).strip().lower() for c in x.columns]
        x = _regular(x)
        required = {"week", "defteam", "epa", "rush"}
        if not required.issubset(x.columns):
            raise RuntimeError(
                f"PBP exact rush EPA parity missing columns for {season}: "
                f"{sorted(required - set(x.columns))}"
            )
        x["week"] = _num(x["week"])
        x["defteam"] = x["defteam"].map(canon_team)
        x["epa"] = _num(x["epa"])
        rush = x["rush"].fillna(False).astype(bool)
        q = x.loc[
            rush
            & x["week"].between(1, 18)
            & x["defteam"].ne("")
            & x["epa"].notna()
        ].copy()
        weekly = (
            q.groupby(["week", "defteam"], as_index=False)
            .agg(epa_sum=("epa", "sum"), epa_n=("epa", "count"))
        )
        for team, g in weekly.groupby("defteam"):
            g = g.sort_values("week")
            for target_week in TARGET_WEEKS:
                h = g.loc[g["week"].lt(target_week)]
                n = int(h["epa_n"].sum())
                rows.append(
                    {
                        "season": season,
                        "week": target_week,
                        "team": team,
                        "def_rush_epa": float(h["epa_sum"].sum() / n) if n else np.nan,
                        "def_rush_epa_plays": n,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("exact rush EPA builder produced zero rows")
    return out


def build_position_ypt_pregame(logs: pd.DataFrame) -> pd.DataFrame:
    x = logs.copy()
    x["position"] = x["position"].astype(str).str.upper().str.strip()
    x["pos_group"] = np.select(
        [
            x["position"].eq("WR"),
            x["position"].eq("TE"),
            x["position"].isin(RB_POS),
        ],
        ["WR", "TE", "RB"],
        default="OTHER",
    )
    x = x.loc[x["pos_group"].isin(["WR", "TE", "RB"])].copy()
    x["targets"] = _num(x["targets"]).fillna(0.0)
    x["rec_yards"] = _num(x["rec_yards"]).fillna(0.0)
    x["defense"] = x["opponent"].map(canon_team)
    weekly = (
        x.groupby(["season", "week", "defense", "pos_group"], as_index=False)
        .agg(targets=("targets", "sum"), rec_yards=("rec_yards", "sum"))
    )
    rows: list[dict] = []
    for season in TARGET_SEASONS:
        sx = weekly.loc[weekly["season"].eq(season)].copy()
        defenses = sorted(set(sx["defense"].dropna().astype(str)))
        for target_week in TARGET_WEEKS:
            h = sx.loc[sx["week"].lt(target_week)].copy()
            for defense in defenses:
                d = h.loc[h["defense"].eq(defense)].copy()
                if d.empty:
                    continue
                keep_weeks = sorted(d["week"].dropna().astype(int).unique())[-8:]
                d = d.loc[d["week"].isin(keep_weeks)]
                rec = {"season": season, "week": target_week, "team": defense}
                for pos in ("WR", "TE", "RB"):
                    p = d.loc[d["pos_group"].eq(pos)]
                    den = float(p["targets"].sum())
                    rec[f"{pos.lower()}_ypt_allowed_hist"] = (
                        float(p["rec_yards"].sum() / den) if den > 0 else np.nan
                    )
                    rec[f"{pos.lower()}_targets_faced_hist"] = den
                rows.append(rec)
    return pd.DataFrame(rows)


def _aggregate_player_totals(x: pd.DataFrame) -> pd.DataFrame:
    if x.empty:
        return pd.DataFrame(
            columns=["player_identity_key", "games", "target_share", "rush_share"]
        )
    g = (
        x.groupby("player_identity_key", as_index=False)
        .agg(
            games=("week", "nunique"),
            targets=("targets", "sum"),
            team_targets=("team_targets", "sum"),
            rushes=("rushes", "sum"),
            team_rushes=("team_rushes", "sum"),
        )
    )
    g["target_share"] = np.where(
        g["team_targets"] > 0,
        g["targets"] / g["team_targets"],
        np.nan,
    )
    g["rush_share"] = np.where(
        g["team_rushes"] > 0,
        g["rushes"] / g["team_rushes"],
        np.nan,
    )
    return g[
        ["player_identity_key", "games", "target_share", "rush_share"]
    ]


def build_player_opportunity_pregame(logs: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in TARGET_SEASONS:
        prior = _aggregate_player_totals(
            logs.loc[logs["season"].eq(season - 1)]
        )
        prior = prior.rename(
            columns={
                "games": "prior_games",
                "target_share": "target_share_prior",
                "rush_share": "rush_share_prior",
            }
        )
        for week in TARGET_WEEKS:
            current = _aggregate_player_totals(
                logs.loc[
                    logs["season"].eq(season)
                    & logs["week"].lt(week)
                ]
            )
            current = current.rename(
                columns={
                    "games": "current_games",
                    "target_share": "target_share_current",
                    "rush_share": "rush_share_current",
                }
            )
            ids = pd.DataFrame(
                {
                    "player_identity_key": sorted(
                        set(prior.get("player_identity_key", []))
                        | set(current.get("player_identity_key", []))
                    )
                }
            )
            z = (
                ids.merge(prior, on="player_identity_key", how="left")
                .merge(current, on="player_identity_key", how="left")
            )
            z["prior_games"] = _num(z["prior_games"]).fillna(0.0)
            z["current_games"] = _num(z["current_games"]).fillna(0.0)
            w = z["current_games"] / (z["current_games"] + 4.0)
            for metric in ("target_share", "rush_share"):
                pv = _num(z[f"{metric}_prior"])
                cv = _num(z[f"{metric}_current"])
                val = pv.copy()
                only_cur = cv.notna() & pv.isna()
                both = cv.notna() & pv.notna()
                val.loc[only_cur] = cv.loc[only_cur]
                val.loc[both] = (
                    (1.0 - w.loc[both]) * pv.loc[both]
                    + w.loc[both] * cv.loc[both]
                )
                z[metric] = val
            z["season"], z["week"] = season, week
            rows.append(
                z[
                    [
                        "season",
                        "week",
                        "player_identity_key",
                        "target_share",
                        "rush_share",
                        "prior_games",
                        "current_games",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_team_target_features(
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    exact_rush: pd.DataFrame,
    pos_ypt: pd.DataFrame,
) -> pd.DataFrame:
    tw = team_weekly.copy()
    tw["season"] = _num(tw["season"]).astype("Int64")
    tw["week"] = _num(tw["week"]).astype("Int64")
    tw["team"] = tw["team"].map(canon_team)

    s = schedule.copy()
    s["season"] = _num(s["season"]).astype("Int64")
    s["week"] = _num(s["week"]).astype("Int64")
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    s = (
        s.loc[
            s["season"].isin(TARGET_SEASONS)
            & s["week"].isin(TARGET_WEEKS),
            ["season", "week", "team", "opponent"],
        ]
        .drop_duplicates()
    )
    if s.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("schedule duplicate target team-week")

    rush_lookup = (
        exact_rush.set_index(["season", "week", "team"])
        .to_dict("index")
    )
    ypt_lookup = (
        pos_ypt.set_index(["season", "week", "team"]).to_dict("index")
        if not pos_ypt.empty
        else {}
    )

    rows: list[dict] = []
    for r in s.itertuples(index=False):
        season, week = int(r.season), int(r.week)
        team, opponent = str(r.team), str(r.opponent)
        oh = _prior_team_rows(tw, season, week, team)
        dh = _prior_team_rows(tw, season, week, opponent)
        rec = {
            "season": season,
            "week": week,
            "team": team,
            "opponent": opponent,
            "off_true_proe": _mean_col(oh, ["true_proe", "proe"]),
            "off_neutral_pace": _mean_col(
                oh, ["neutral_pace_true", "neutral_pace"]
            ),
            "off_pass_rate": _mean_col(
                oh, ["pass_rate_off", "dropback_rate"]
            ),
            "off_plays": _mean_col(oh, ["plays_est"]),
            "off_pressure_allowed": _mean_col(
                oh,
                [
                    "hit_sack_pressure_rate_allowed",
                    "pressure_rate_allowed",
                ],
            ),
            "def_pass_epa_allowed": _mean_col(
                dh,
                ["def_pass_epa_allowed", "def_pass_epa"],
            ),
            "def_pass_success_allowed": _mean_col(
                dh,
                ["def_pass_success_allowed", "success_rate_def"],
            ),
            "def_ypa_allowed": _mean_col(dh, ["def_ypa_allowed"]),
            "def_pass_rate_faced": _mean_col(dh, ["pass_rate_faced"]),
            "def_pressure_generated": _mean_col(
                dh,
                [
                    "hit_sack_pressure_rate_generated",
                    "pressure_rate_generated",
                ],
            ),
            "def_light_box_rate": _mean_col(dh, ["light_box_rate"]),
            "def_heavy_box_rate": _mean_col(dh, ["heavy_box_rate"]),
            "def_man_rate": _mean_col(dh, ["coverage_man_rate"]),
            "def_zone_rate": _mean_col(dh, ["coverage_zone_rate"]),
        }
        rec["pressure_mismatch"] = (
            rec["def_pressure_generated"]
            - rec["off_pressure_allowed"]
            if np.isfinite(rec["def_pressure_generated"])
            and np.isfinite(rec["off_pressure_allowed"])
            else np.nan
        )
        rr = rush_lookup.get((season, week, opponent), {})
        rec["def_rush_epa"] = rr.get("def_rush_epa", np.nan)
        rec["def_rush_epa_plays"] = rr.get("def_rush_epa_plays", 0)
        yy = ypt_lookup.get((season, week, opponent), {})
        rec["def_wr_ypt_allowed"] = yy.get(
            "wr_ypt_allowed_hist", np.nan
        )
        rec["def_te_ypt_allowed"] = yy.get(
            "te_ypt_allowed_hist", np.nan
        )
        rec["def_rb_ypt_allowed"] = yy.get(
            "rb_ypt_allowed_hist", np.nan
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def add_week_zscores(
    team_features: pd.DataFrame,
    specs: list[FeatureSpec],
) -> pd.DataFrame:
    out = team_features.copy()
    unique = sorted({s.feature for s in specs})
    for feature in unique:
        if feature not in out.columns:
            continue
        raw = _num(out[feature])
        z = pd.Series(np.nan, index=out.index, dtype=float)
        for _, idx in out.groupby(["season", "week"]).groups.items():
            vals = raw.loc[idx]
            m = float(vals.mean()) if vals.notna().any() else np.nan
            sd = (
                float(vals.std(ddof=0))
                if vals.notna().sum() >= 2
                else np.nan
            )
            if np.isfinite(sd) and sd > 0:
                z.loc[idx] = (vals - m) / sd
        out[f"{feature}__z"] = z
    return out


def load_skill_detail(
    path: Path,
    logs: pd.DataFrame,
) -> pd.DataFrame:
    allowed = {
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "player_clean_key",
        "market",
        "actual",
        "final_mean",
    }
    x = _read(
        path,
        "right-tail football-only detail",
        usecols=lambda c: str(c).strip().lower() in allowed,
    )
    missing = sorted(allowed - set(x.columns))
    if missing:
        raise RuntimeError(
            f"right-tail detail missing allowed fields: {missing}"
        )
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x = x.loc[
        x["season"].isin(TARGET_SEASONS)
        & x["week"].isin(TARGET_WEEKS)
    ].copy()
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].map(_key)
    x["actual"] = _num(x["actual"])
    x["projection"] = _num(x["final_mean"])
    x["residual"] = x["actual"] - x["projection"]

    meta = (
        logs[
            [
                "season",
                "week",
                "team",
                "player_clean_key",
                "player_identity_key",
                "position",
            ]
        ]
        .drop_duplicates()
    )
    x = x.merge(
        meta,
        on=["season", "week", "team", "player_clean_key"],
        how="left",
        validate="many_to_one",
    )
    if x["position"].isna().mean() > 0.02:
        raise RuntimeError(
            "position identity coverage too low: "
            f"missing={int(x['position'].isna().sum())}/{len(x)}"
        )
    x["position"] = (
        x["position"].astype(str).str.upper().str.strip()
    )
    return x


def load_qb_control(
    path: Path,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    allowed = {
        "season",
        "week",
        "team",
        "opponent",
        "player_clean_key",
        "actual_pass_yards",
        "football_synthesis",
    }
    x = _read(
        path,
        "M89/M90 QB control",
        usecols=lambda c: str(c).strip().lower() in allowed,
    )
    required = {
        "season",
        "week",
        "team",
        "player_clean_key",
        "actual_pass_yards",
        "football_synthesis",
    }
    if not required.issubset(x.columns):
        raise RuntimeError(
            "QB control missing allowed fields: "
            f"{sorted(required - set(x.columns))}"
        )
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x = x.loc[
        x["season"].isin(TARGET_SEASONS)
        & x["week"].isin(TARGET_WEEKS)
    ].copy()
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].map(_key)
    if "opponent" not in x.columns or x["opponent"].isna().any():
        s = (
            schedule[
                ["season", "week", "team", "opponent"]
            ]
            .drop_duplicates()
        )
        x = (
            x.drop(columns=["opponent"], errors="ignore")
            .merge(
                s,
                on=["season", "week", "team"],
                how="left",
                validate="many_to_one",
            )
        )
    x["opponent"] = x["opponent"].map(canon_team)
    x["actual"] = _num(x["actual_pass_yards"])
    x["projection"] = _num(x["football_synthesis"])
    x["residual"] = x["actual"] - x["projection"]
    lo = x[["team", "opponent"]].min(axis=1)
    hi = x[["team", "opponent"]].max(axis=1)
    x["game_id"] = (
        x["season"].astype(str)
        + "_"
        + x["week"].astype(str).str.zfill(2)
        + "_"
        + lo
        + "_"
        + hi
    )
    x["player_identity_key"] = x["player_clean_key"]
    x["position"] = "QB"
    x["market"] = "pass_yards"
    return x


def build_cohorts(
    skill: pd.DataFrame,
    qb: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "RB_RUSH": skill.loc[
            skill["market"].eq("rush_yards")
            & skill["position"].isin(RB_POS)
        ].copy(),
        "RB_RUSH_REC": skill.loc[
            skill["market"].eq("rush_rec_yards")
            & skill["position"].isin(RB_POS)
        ].copy(),
        "WR_REC": skill.loc[
            skill["market"].eq("rec_yards")
            & skill["position"].eq("WR")
        ].copy(),
        "TE_REC": skill.loc[
            skill["market"].eq("rec_yards")
            & skill["position"].eq("TE")
        ].copy(),
        "RB_REC": skill.loc[
            skill["market"].eq("rec_yards")
            & skill["position"].isin(RB_POS)
        ].copy(),
        "QB_PASS_CONTROL": qb.copy(),
    }


def _corr_from_sums(
    n,
    sx,
    sy,
    sxx,
    syy,
    sxy,
):
    denx = n * sxx - sx * sx
    deny = n * syy - sy * sy
    den = np.sqrt(np.maximum(denx * deny, 0.0))
    return np.where(
        (n > 1) & (den > 0),
        (n * sxy - sx * sy) / den,
        np.nan,
    )


def _cluster_bootstrap_rank_corr(
    x: pd.Series,
    y: pd.Series,
    clusters: pd.Series,
    reps: int,
    seed: int,
) -> dict:
    z = pd.DataFrame(
        {
            "x": _num(x),
            "y": _num(y),
            "cluster": clusters.astype(str),
        }
    ).dropna()
    if z.empty:
        return {
            "ci_low": np.nan,
            "ci_high": np.nan,
            "valid_reps": 0,
        }
    z["xr"] = z["x"].rank(method="average")
    z["yr"] = z["y"].rank(method="average")
    z["xx"] = z["xr"] * z["xr"]
    z["yy"] = z["yr"] * z["yr"]
    z["xy"] = z["xr"] * z["yr"]
    g = (
        z.groupby("cluster", as_index=False)
        .agg(
            n=("xr", "size"),
            sx=("xr", "sum"),
            sy=("yr", "sum"),
            sxx=("xx", "sum"),
            syy=("yy", "sum"),
            sxy=("xy", "sum"),
        )
    )
    if len(g) < 2:
        return {
            "ci_low": np.nan,
            "ci_high": np.nan,
            "valid_reps": 0,
        }

    a = g[["n", "sx", "sy", "sxx", "syy", "sxy"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals: list[np.ndarray] = []
    done = 0
    batch = 250
    p = np.full(len(g), 1 / len(g), dtype=float)
    while done < reps:
        k = min(batch, reps - done)
        counts = rng.multinomial(
            len(g),
            p,
            size=k,
        ).astype(float)
        s = counts @ a
        vals.append(
            _corr_from_sums(
                s[:, 0],
                s[:, 1],
                s[:, 2],
                s[:, 3],
                s[:, 4],
                s[:, 5],
            )
        )
        done += k
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    return {
        "ci_low": (
            float(np.quantile(v, 0.025))
            if len(v)
            else np.nan
        ),
        "ci_high": (
            float(np.quantile(v, 0.975))
            if len(v)
            else np.nan
        ),
        "valid_reps": int(len(v)),
    }


def evaluate_cell(
    q: pd.DataFrame,
    signal_col: str,
    season: int,
    seed_offset: int,
) -> dict:
    z = q.loc[
        q["season"].eq(season),
        [
            "residual",
            signal_col,
            "game_id",
            "player_identity_key",
        ],
    ].copy()
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    rows = len(z)
    games = z["game_id"].nunique()
    players = z["player_identity_key"].nunique()
    support = (
        rows >= MIN_ROWS
        and games >= MIN_GAMES
        and players >= 25
    )
    rec = {
        "season": season,
        "rows": int(rows),
        "games": int(games),
        "players": int(players),
        "support": bool(support),
    }
    if not support:
        rec.update(
            {
                "rho": np.nan,
                "game_ci_low": np.nan,
                "game_ci_high": np.nan,
                "player_ci_low": np.nan,
                "player_ci_high": np.nan,
                "directional_cluster_support": False,
            }
        )
        return rec

    rho = float(
        z["residual"].corr(
            z[signal_col],
            method="spearman",
        )
    )
    game = _cluster_bootstrap_rank_corr(
        z[signal_col],
        z["residual"],
        z["game_id"],
        BOOT_REPS,
        BOOT_SEED + seed_offset,
    )
    player = _cluster_bootstrap_rank_corr(
        z[signal_col],
        z["residual"],
        z["player_identity_key"],
        BOOT_REPS,
        BOOT_SEED + 10000 + seed_offset,
    )
    passed = bool(
        np.isfinite(rho)
        and rho > 0
        and np.isfinite(game["ci_low"])
        and game["ci_low"] > 0
        and np.isfinite(player["ci_low"])
        and player["ci_low"] > 0
    )
    rec.update(
        {
            "rho": rho,
            "game_ci_low": game["ci_low"],
            "game_ci_high": game["ci_high"],
            "game_boot_valid": game["valid_reps"],
            "player_ci_low": player["ci_low"],
            "player_ci_high": player["ci_high"],
            "player_boot_valid": player["valid_reps"],
            "directional_cluster_support": passed,
        }
    )
    return rec


def evaluate_specs(
    cohorts: dict[str, pd.DataFrame],
    team_features: pd.DataFrame,
    opportunity: pd.DataFrame,
    specs: list[FeatureSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf = team_features.copy()
    opp = opportunity.copy()
    cells: list[dict] = []
    summaries: list[dict] = []

    for idx, spec in enumerate(specs):
        base = cohorts.get(
            spec.cohort,
            pd.DataFrame(),
        ).copy()
        if base.empty or spec.feature not in tf.columns:
            summaries.append(
                {
                    **spec.__dict__,
                    "replicated": False,
                    "integration_candidate_eligible": False,
                    "disposition": "UNAVAILABLE",
                }
            )
            continue

        q = base.merge(
            tf,
            on=["season", "week", "team", "opponent"],
            how="left",
            validate="many_to_one",
        )
        rawz = f"{spec.feature}__z"
        if rawz not in q.columns:
            summaries.append(
                {
                    **spec.__dict__,
                    "replicated": False,
                    "integration_candidate_eligible": False,
                    "disposition": "UNAVAILABLE",
                }
            )
            continue

        q["weakness_z"] = _num(q[rawz]) * float(spec.sign)
        signal_col = "weakness_z"
        if spec.phase == "C":
            if not spec.opportunity:
                raise RuntimeError(
                    f"Phase C spec missing opportunity: {spec}"
                )
            q = q.merge(
                opp[
                    [
                        "season",
                        "week",
                        "player_identity_key",
                        spec.opportunity,
                    ]
                ],
                on=["season", "week", "player_identity_key"],
                how="left",
                validate="many_to_one",
            )
            q["interaction"] = (
                _num(q[spec.opportunity])
                * _num(q["weakness_z"])
            )
            signal_col = "interaction"

        season_rows = []
        for season in TARGET_SEASONS:
            rec = evaluate_cell(
                q,
                signal_col,
                season,
                seed_offset=idx * 100 + season,
            )
            cells.append(
                {
                    **spec.__dict__,
                    **rec,
                }
            )
            season_rows.append(rec)

        replicated = all(
            r.get("directional_cluster_support", False)
            for r in season_rows
        )
        same_semantics = (
            spec.parity == "EXACT_LIVE_SEMANTICS"
        )
        closed = bool(spec.closed_family)
        control = spec.cohort == "QB_PASS_CONTROL"
        eligible = bool(
            replicated
            and same_semantics
            and not closed
            and not control
        )

        if eligible:
            disposition = "INTEGRATION_CANDIDATE_ELIGIBLE"
        elif replicated and closed:
            disposition = "REPLICATED_BUT_CLOSED_PRIOR_FAMILY"
        elif replicated and not same_semantics:
            disposition = (
                "REPLICATED_DIAGNOSTIC_SOURCE_PARITY_BLOCKED"
            )
        elif replicated and control:
            disposition = (
                "REPLICATED_CONTROL_ONLY_NO_QB_REOPEN"
            )
        else:
            disposition = "NOT_REPLICATED"

        summaries.append(
            {
                **spec.__dict__,
                "replicated": bool(replicated),
                "integration_candidate_eligible": eligible,
                "disposition": disposition,
            }
        )

    return pd.DataFrame(cells), pd.DataFrame(summaries)


def _validate_no_forbidden_loaded(
    *frames: tuple[str, pd.DataFrame],
) -> None:
    for label, frame in frames:
        bad = [
            c
            for c in frame.columns
            if any(
                t in c.lower()
                for t in FORBIDDEN_TOKENS
            )
        ]
        if bad:
            raise RuntimeError(
                f"sportsbook/odds columns prohibited in {label}: {bad}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--right-tail-detail",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--qb-control-trace",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--team-weekly",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--player-logs",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--schedule",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    args = ap.parse_args()

    team = _read(
        args.team_weekly,
        "corrected team-week history",
    )
    logs = _read(
        args.player_logs,
        "historical player logs",
    )
    schedule_allowed = {
        "season",
        "week",
        "team",
        "opponent",
        "game_id",
    }
    schedule = _read(
        args.schedule,
        "authoritative schedule",
        usecols=lambda c: (
            str(c).strip().lower()
            in schedule_allowed
        ),
    )
    _validate_no_forbidden_loaded(
        ("team_weekly", team),
        ("player_logs", logs),
        ("schedule", schedule),
    )

    for frame in (team, logs, schedule):
        if "season" in frame.columns:
            frame["season"] = _num(
                frame["season"]
            ).astype("Int64")
        if "week" in frame.columns:
            frame["week"] = _num(
                frame["week"]
            ).astype("Int64")

    team["team"] = team["team"].map(canon_team)
    logs["team"] = logs["team"].map(canon_team)
    logs["opponent"] = logs["opponent"].map(canon_team)
    logs["player_clean_key"] = logs[
        "player_clean_key"
    ].map(_key)
    if "player_identity_key" not in logs.columns:
        logs["player_identity_key"] = logs[
            "player_clean_key"
        ]
    logs["player_identity_key"] = logs[
        "player_identity_key"
    ].astype(str)
    schedule["team"] = schedule["team"].map(canon_team)
    schedule["opponent"] = schedule[
        "opponent"
    ].map(canon_team)

    exact_rush = build_exact_rush_epa_pregame(
        TARGET_SEASONS
    )
    pos_ypt = build_position_ypt_pregame(logs)
    opportunity = build_player_opportunity_pregame(
        logs
    )
    specs = _feature_specs()
    team_features = build_team_target_features(
        team,
        schedule,
        exact_rush,
        pos_ypt,
    )
    team_features = add_week_zscores(
        team_features,
        specs,
    )

    skill = load_skill_detail(
        args.right_tail_detail,
        logs,
    )
    qb = load_qb_control(
        args.qb_control_trace,
        schedule,
    )
    _validate_no_forbidden_loaded(
        ("skill_detail", skill),
        ("qb_control", qb),
        ("team_features", team_features),
        ("opportunity", opportunity),
    )

    cohorts = build_cohorts(skill, qb)
    cells, summary = evaluate_specs(
        cohorts,
        team_features,
        opportunity,
        specs,
    )

    sources = _source_matrix()
    candidates = (
        summary.loc[
            summary[
                "integration_candidate_eligible"
            ].eq(True)
        ].copy()
        if not summary.empty
        else pd.DataFrame()
    )

    result = {
        "version": VERSION,
        "methods": (
            "docs/research/"
            "FOOTBALL_MATCHUP_TRANSMISSION_V1_PHASE_BC_METHODS.md"
        ),
        "evaluation_seasons": list(TARGET_SEASONS),
        "evaluation_weeks": [2, 18],
        "sportsbook_inputs_used": 0,
        "candidate_models_fit": 0,
        "production_changed": False,
        "phase_a_rerun": False,
        "bootstrap_reps": BOOT_REPS,
        "bootstrap_seed": BOOT_SEED,
        "skill_rows": int(len(skill)),
        "qb_control_rows": int(len(qb)),
        "team_target_rows": int(len(team_features)),
        "tested_specs": int(len(summary)),
        "replicated_specs": (
            int(summary["replicated"].sum())
            if len(summary)
            else 0
        ),
        "integration_candidate_eligible_specs": int(
            len(candidates)
        ),
        "integration_candidates": (
            candidates[
                [
                    "phase",
                    "cohort",
                    "feature",
                    "family",
                    "disposition",
                ]
            ].to_dict("records")
            if len(candidates)
            else []
        ),
        "blocked_source_features": (
            sources.loc[
                sources["same_live_semantics"].eq(0),
                "feature",
            ].tolist()
        ),
        "qb_control_reopens_closed_family": False,
        "m95_generic_rb_matchup_reopened": False,
    }

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    team_features.to_csv(
        args.out_dir
        / "football_matchup_phase_bc_team_features.csv",
        index=False,
    )
    opportunity.to_csv(
        args.out_dir
        / "football_matchup_phase_bc_player_opportunity.csv",
        index=False,
    )
    sources.to_csv(
        args.out_dir
        / "football_matchup_phase_bc_source_readiness.csv",
        index=False,
    )
    cells.to_csv(
        args.out_dir
        / "football_matchup_phase_bc_cells.csv",
        index=False,
    )
    summary.to_csv(
        args.out_dir
        / "football_matchup_phase_bc_summary.csv",
        index=False,
    )
    (
        args.out_dir
        / "football_matchup_phase_bc_result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    print("\n=== REPLICATED / CANDIDATE SUMMARY ===")
    if len(summary):
        print(
            summary.loc[
                summary["replicated"].eq(True)
                | summary[
                    "integration_candidate_eligible"
                ].eq(True)
            ].to_string(index=False)
        )
    print("\n=== SOURCE READINESS ===")
    print(sources.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
