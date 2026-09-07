#!/usr/bin/env python3
"""Cross-position Catastrophic Casebook V1 — Phase C pregame game-spot overlay.

Builds leakage-safe rolling team/opponent environment profiles from nflverse PBP
and overlays them onto the frozen Phase-A all-row casebook and Phase-B
catastrophic casebook. Sportsbook fields are not used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.pbp import get_pbp


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def canon(v) -> str:
    return canon_team(v)


def regular(x: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in x.columns:
        q = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    if "game_type" in x.columns:
        q = x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    return x


def aggregate_season(season: int) -> pd.DataFrame:
    x = regular(lower(get_pbp(int(season), min_rows=1)))
    x["season"] = pd.to_numeric(x.get("season"), errors="coerce")
    x["week"] = pd.to_numeric(x.get("week"), errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].between(1, 18)].copy()
    x["season"] = int(season)
    x["week"] = x["week"].astype(int)
    x["team"] = x.get("posteam", pd.Series("", index=x.index)).map(canon)
    x["opponent"] = x.get("defteam", pd.Series("", index=x.index)).map(canon)
    x = x.loc[x["team"].ne("") & x["opponent"].ne("")].copy()

    sack = num(x, "sack", 0).fillna(0).eq(1)
    pass_attempt = num(x, "pass_attempt", 0).fillna(0).eq(1) & ~sack
    qb_scramble = num(x, "qb_scramble", 0).fillna(0).eq(1)
    qb_kneel = num(x, "qb_kneel", 0).fillna(0).eq(1)
    rush_attempt = num(x, "rush_attempt", 0).fillna(0).eq(1) & ~qb_scramble & ~qb_kneel
    off_play = num(x, "qb_dropback", 0).fillna(0).eq(1) | num(x, "rush_attempt", 0).fillna(0).eq(1)

    py = num(x, "passing_yards", np.nan)
    if py.notna().sum() == 0:
        py = num(x, "yards_gained", 0)
    py = py.fillna(0)
    ry = num(x, "rushing_yards", np.nan)
    if ry.notna().sum() == 0:
        ry = num(x, "yards_gained", 0)
    ry = ry.fillna(0)
    epa = num(x, "epa")
    success = num(x, "success")
    complete = pass_attempt & num(x, "complete_pass", 0).fillna(0).eq(1)
    yac = num(x, "yards_after_catch")

    x["_off_play"] = off_play.astype(float)
    x["_pass"] = pass_attempt.astype(float)
    x["_rush"] = rush_attempt.astype(float)
    x["_pass_yards"] = np.where(pass_attempt, py, 0.0)
    x["_rush_yards"] = np.where(rush_attempt, ry, 0.0)
    x["_pass_epa"] = np.where(pass_attempt, epa, np.nan)
    x["_rush_epa"] = np.where(rush_attempt, epa, np.nan)
    x["_pass_success"] = np.where(pass_attempt, success, np.nan)
    x["_rush_success"] = np.where(rush_attempt, success, np.nan)
    x["_pass_expl20"] = np.where(pass_attempt, py.ge(20).astype(float), np.nan)
    x["_pass_expl40"] = np.where(pass_attempt, py.ge(40).astype(float), np.nan)
    x["_rush_expl10"] = np.where(rush_attempt, ry.ge(10).astype(float), np.nan)
    x["_rush_expl20"] = np.where(rush_attempt, ry.ge(20).astype(float), np.nan)
    x["_yac_comp"] = np.where(complete, yac, np.nan)

    g = (
        x.groupby(["season", "week", "team", "opponent"], as_index=False)
        .agg(
            off_plays=("_off_play", "sum"),
            pass_att=("_pass", "sum"),
            rush_att=("_rush", "sum"),
            pass_yards=("_pass_yards", "sum"),
            rush_yards=("_rush_yards", "sum"),
            pass_epa=("_pass_epa", "mean"),
            rush_epa=("_rush_epa", "mean"),
            pass_success=("_pass_success", "mean"),
            rush_success=("_rush_success", "mean"),
            pass_expl20=("_pass_expl20", "mean"),
            pass_expl40=("_pass_expl40", "mean"),
            rush_expl10=("_rush_expl10", "mean"),
            rush_expl20=("_rush_expl20", "mean"),
            yac_per_completion=("_yac_comp", "mean"),
        )
    )
    den = g["pass_att"] + g["rush_att"]
    g["pass_rate"] = np.where(den > 0, g["pass_att"] / den, np.nan)
    g["rush_rate"] = np.where(den > 0, g["rush_att"] / den, np.nan)
    g["pass_ypa"] = np.where(g["pass_att"] > 0, g["pass_yards"] / g["pass_att"], np.nan)
    g["rush_ypc"] = np.where(g["rush_att"] > 0, g["rush_yards"] / g["rush_att"], np.nan)
    print(f"[phase_c] aggregated PBP {season}: {len(g)} team-games")
    return g


def rolling_prior(df: pd.DataFrame, team_col: str, features: list[str], prefix: str) -> pd.DataFrame:
    x = df.copy()
    x["_order"] = x["season"] * 100 + x["week"]
    x = x.sort_values([team_col, "_order", "opponent"]).copy()
    for c in features:
        x[f"{prefix}{c}"] = (
            x.groupby(team_col, sort=False)[c]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
    return x


def build_team_spots() -> pd.DataFrame:
    games = pd.concat([aggregate_season(s) for s in range(2019, 2026)], ignore_index=True, sort=False)

    off_features = ["off_plays", "pass_att", "rush_att", "pass_rate", "rush_rate"]
    off = rolling_prior(games, "team", off_features, "off_prior_")
    off = off[["season", "week", "team", "opponent"] + [f"off_prior_{c}" for c in off_features]].copy()

    d = games.rename(
        columns={
            "team": "offense",
            "opponent": "team",
            "pass_att": "pass_att_faced",
            "rush_att": "rush_att_faced",
            "pass_ypa": "pass_ypa_allowed",
            "rush_ypc": "rush_ypc_allowed",
            "pass_epa": "pass_epa_allowed",
            "rush_epa": "rush_epa_allowed",
            "pass_success": "pass_success_allowed",
            "rush_success": "rush_success_allowed",
            "pass_expl20": "pass_expl20_allowed",
            "pass_expl40": "pass_expl40_allowed",
            "rush_expl10": "rush_expl10_allowed",
            "rush_expl20": "rush_expl20_allowed",
            "yac_per_completion": "yac_per_completion_allowed",
        }
    )
    d["opponent"] = d["offense"]
    def_features = [
        "pass_att_faced", "rush_att_faced",
        "pass_ypa_allowed", "rush_ypc_allowed",
        "pass_epa_allowed", "rush_epa_allowed",
        "pass_success_allowed", "rush_success_allowed",
        "pass_expl20_allowed", "pass_expl40_allowed",
        "rush_expl10_allowed", "rush_expl20_allowed",
        "yac_per_completion_allowed",
    ]
    d = rolling_prior(d, "team", def_features, "def_prior_")
    d = d[["season", "week", "team"] + [f"def_prior_{c}" for c in def_features]].drop_duplicates(
        ["season", "week", "team"]
    )

    t = off.merge(
        d.rename(columns={"team": "opponent"}),
        on=["season", "week", "opponent"],
        how="left",
        validate="many_to_one",
    )

    zcols = [
        "off_prior_off_plays", "off_prior_pass_att", "off_prior_rush_att", "off_prior_pass_rate", "off_prior_rush_rate",
        "def_prior_pass_att_faced", "def_prior_rush_att_faced",
        "def_prior_pass_ypa_allowed", "def_prior_rush_ypc_allowed",
        "def_prior_pass_epa_allowed", "def_prior_rush_epa_allowed",
        "def_prior_pass_success_allowed", "def_prior_rush_success_allowed",
        "def_prior_pass_expl20_allowed", "def_prior_pass_expl40_allowed",
        "def_prior_rush_expl10_allowed", "def_prior_rush_expl20_allowed",
        "def_prior_yac_per_completion_allowed",
    ]
    for c in zcols:
        mean = t.groupby(["season", "week"])[c].transform("mean")
        std = t.groupby(["season", "week"])[c].transform("std").replace(0, np.nan)
        t[f"z_{c}"] = (t[c] - mean) / std

    t["pass_opportunity_spot"] = t[
        ["z_off_prior_pass_att", "z_off_prior_pass_rate", "z_off_prior_off_plays", "z_def_prior_pass_att_faced"]
    ].mean(axis=1, skipna=True)
    t["pass_efficiency_spot"] = t[
        [
            "z_def_prior_pass_ypa_allowed", "z_def_prior_pass_epa_allowed", "z_def_prior_pass_success_allowed",
            "z_def_prior_pass_expl20_allowed", "z_def_prior_yac_per_completion_allowed",
        ]
    ].mean(axis=1, skipna=True)
    t["rush_opportunity_spot"] = t[
        ["z_off_prior_rush_att", "z_off_prior_rush_rate", "z_off_prior_off_plays", "z_def_prior_rush_att_faced"]
    ].mean(axis=1, skipna=True)
    t["rush_efficiency_spot"] = t[
        [
            "z_def_prior_rush_ypc_allowed", "z_def_prior_rush_epa_allowed", "z_def_prior_rush_success_allowed",
            "z_def_prior_rush_expl10_allowed", "z_def_prior_rush_expl20_allowed",
        ]
    ].mean(axis=1, skipna=True)
    t["pass_composite_spot"] = 0.5 * t["pass_opportunity_spot"] + 0.5 * t["pass_efficiency_spot"]
    t["rush_composite_spot"] = 0.5 * t["rush_opportunity_spot"] + 0.5 * t["rush_efficiency_spot"]

    for family in ["pass", "rush"]:
        c = f"{family}_composite_spot"
        pct = t.groupby(["season", "week"])[c].rank(method="average", pct=True)
        t[f"{family}_spot_percentile"] = pct
        t[f"{family}_spot_bucket"] = np.select(
            [pct >= 0.75, pct <= 0.25],
            ["FAVORABLE", "ADVERSE"],
            default="NEUTRAL",
        )
    return t.loc[t["season"].between(2020, 2025)].copy()


def attach_spot(rows: pd.DataFrame, spots: pd.DataFrame) -> pd.DataFrame:
    x = rows.copy()
    x["team"] = x["team"].map(canon)
    tcols = [
        "season", "week", "team", "opponent",
        "pass_opportunity_spot", "pass_efficiency_spot", "pass_composite_spot", "pass_spot_percentile", "pass_spot_bucket",
        "rush_opportunity_spot", "rush_efficiency_spot", "rush_composite_spot", "rush_spot_percentile", "rush_spot_bucket",
    ]
    x = x.merge(spots[tcols], on=["season", "week", "team"], how="left", validate="many_to_one")
    is_rb = x["position"].astype(str).str.upper().eq("RB")
    x["opportunity_spot_score"] = np.where(is_rb, x["rush_opportunity_spot"], x["pass_opportunity_spot"])
    x["efficiency_spot_score"] = np.where(is_rb, x["rush_efficiency_spot"], x["pass_efficiency_spot"])
    x["game_spot_score"] = np.where(is_rb, x["rush_composite_spot"], x["pass_composite_spot"])
    x["game_spot_percentile"] = np.where(is_rb, x["rush_spot_percentile"], x["pass_spot_percentile"])
    x["game_spot_bucket"] = np.where(is_rb, x["rush_spot_bucket"], x["pass_spot_bucket"])
    return x


def q90(s: pd.Series) -> float:
    z = pd.to_numeric(s, errors="coerce").dropna()
    return float(z.quantile(0.90)) if len(z) else np.nan


def bucket_scorecard(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pos, direction, bucket), g in all_rows.groupby(["position", "direction", "game_spot_bucket"], dropna=False):
        cat = g.loc[g["catastrophic"].astype(bool)]
        rows.append({
            "position": pos, "direction": direction, "game_spot_bucket": bucket,
            "all_n": len(g), "catastrophic_n": len(cat),
            "catastrophic_rate": float(len(cat) / len(g)) if len(g) else np.nan,
            "mean_abs_error_all": float(g["abs_error"].mean()) if len(g) else np.nan,
            "p90_abs_error_all": q90(g["abs_error"]),
            "catastrophic_abs_error_mass": float(cat["abs_error"].sum()),
            "mean_opportunity_spot": float(g["opportunity_spot_score"].mean()),
            "mean_efficiency_spot": float(g["efficiency_spot_score"].mean()),
            "mean_game_spot": float(g["game_spot_score"].mean()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["cat_error_mass_share_within_direction"] = (
            out["catastrophic_abs_error_mass"]
            / out.groupby(["position", "direction"])["catastrophic_abs_error_mass"].transform("sum")
        )
    return out


def q4_scorecard(all_rows: pd.DataFrame) -> pd.DataFrame:
    return bucket_scorecard(all_rows.loc[all_rows["opportunity_quartile"].astype(str).eq("Q4")].copy())


def mechanism_spots(cat: pd.DataFrame) -> pd.DataFrame:
    return (
        cat.groupby(["position", "direction", "pbp_primary_label", "game_spot_bucket"], dropna=False)
        .agg(
            n=("abs_error", "size"),
            abs_error_mass=("abs_error", "sum"),
            mean_abs_error=("abs_error", "mean"),
            mean_opportunity_spot=("opportunity_spot_score", "mean"),
            mean_efficiency_spot=("efficiency_spot_score", "mean"),
            mean_game_spot=("game_spot_score", "mean"),
        )
        .reset_index()
    )


def alignment_table(cat: pd.DataFrame) -> pd.DataFrame:
    x = cat.copy()
    under = x["direction"].astype(str).eq("UNDERPROJECTED")
    over = x["direction"].astype(str).eq("OVERPROJECTED")
    fav = x["game_spot_bucket"].astype(str).eq("FAVORABLE")
    adv = x["game_spot_bucket"].astype(str).eq("ADVERSE")
    x["spot_alignment"] = np.select(
        [(under & fav) | (over & adv), (under & adv) | (over & fav)],
        ["ALIGNED", "OPPOSITE"],
        default="NEUTRAL",
    )
    out = (
        x.groupby(["position", "direction", "spot_alignment"], dropna=False)
        .agg(n=("abs_error", "size"), abs_error_mass=("abs_error", "sum"), mean_abs_error=("abs_error", "mean"))
        .reset_index()
    )
    out["mass_share_within_direction"] = (
        out["abs_error_mass"] / out.groupby(["position", "direction"])["abs_error_mass"].transform("sum")
    )
    return out


def result_summary(all_rows: pd.DataFrame, cat: pd.DataFrame) -> dict:
    coverage = {}
    for pos, g in all_rows.groupby("position"):
        coverage[str(pos)] = float(g["game_spot_score"].notna().mean())
    overall = float(all_rows["game_spot_score"].notna().mean())
    integrity = bool(overall >= 0.98 and all(v >= 0.98 for v in coverage.values()) and len(cat) == 1911)
    return {
        "disposition": "PHASE_C_GAME_SPOT_OVERLAY_COMPLETE" if integrity else "MECHANICAL_OR_INTEGRITY_FAILURE",
        "all_rows": int(len(all_rows)),
        "catastrophic_rows": int(len(cat)),
        "spot_coverage_overall": overall,
        "spot_coverage_by_position": coverage,
        "same_or_future_pbp_used": 0,
        "sportsbook_features_used": 0,
        "postgame_phase_b_fields_used_to_construct_spot": False,
        "spot_definition": {
            "composite": "0.50 opportunity + 0.50 efficiency",
            "bucket": "within-season-week team-game quartile; top=favorable bottom=adverse",
            "history": "strict-prior trailing five team games via shift(1), prior-season carryover allowed",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-a-all-rows", type=Path, required=True)
    p.add_argument("--phase-b-casebook", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = pd.read_csv(args.phase_a_all_rows, low_memory=False)
    phase_b = pd.read_csv(args.phase_b_casebook, low_memory=False)
    if len(phase_b) != 1911:
        raise RuntimeError(f"Expected 1911 Phase-B catastrophic rows; got {len(phase_b)}")

    spots = build_team_spots()
    all_spot = attach_spot(all_rows, spots)
    cat_spot = attach_spot(phase_b, spots)

    score = bucket_scorecard(all_spot)
    q4 = q4_scorecard(all_spot)
    mech = mechanism_spots(cat_spot)
    align = alignment_table(cat_spot)
    result = result_summary(all_spot, cat_spot)

    spots.to_csv(args.out_dir / "cross_position_team_game_spots.csv", index=False)
    all_spot.to_csv(args.out_dir / "cross_position_all_rows_game_spot.csv", index=False)
    cat_spot.to_csv(args.out_dir / "cross_position_catastrophic_game_spot.csv", index=False)
    score.to_csv(args.out_dir / "cross_position_game_spot_scorecard.csv", index=False)
    q4.to_csv(args.out_dir / "cross_position_q4_game_spot_scorecard.csv", index=False)
    mech.to_csv(args.out_dir / "cross_position_phase_b_mechanism_game_spot.csv", index=False)
    align.to_csv(args.out_dir / "cross_position_game_spot_alignment.csv", index=False)
    with open(args.out_dir / "cross_position_phase_c_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\n[phase_c] scorecard")
    print(score.to_string(index=False))
    print("\n[phase_c] Q4 scorecard")
    print(q4.to_string(index=False))
    print("\n[phase_c] alignment")
    print(align.to_string(index=False))

    if result["disposition"] != "PHASE_C_GAME_SPOT_OVERLAY_COMPLETE":
        raise RuntimeError(result["disposition"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
