#!/usr/bin/env python3
"""Source audit for player targeted-route x opponent route-family matchup V1.

Research only. This script proves whether nflverse participation + PBP + weekly
rosters can support the frozen player-attributed *targeted-route* semantics.
It never treats the participation route label as total routes run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_V1"
SEASONS = (2022, 2023, 2024, 2025)
SCORED_SEASONS = (2023, 2024, 2025)
VALID_POS = {"WR", "TE"}
ROUTE_FAMILIES = {
    "CORNER",
    "DEEP OUT",
    "GO",
    "HITCH/CURL",
    "IN/DIG",
    "POST",
    "QUICK OUT",
    "SCREEN",
    "SHALLOW CROSS/DRAG",
    "SLANT",
    "SWING",
    "TEXAS/ANGLE",
    "WHEEL",
}


def to_pandas(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def text_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def route_norm(s: pd.Series) -> pd.Series:
    q = text_series(s).str.upper().str.replace(r"\s+", " ", regex=True)
    q = q.str.replace("_", " ", regex=False)
    q = q.str.replace("-", " ", regex=False)
    # Only lexical punctuation normalization; no outcome-driven family merging.
    lexical = {
        "HITCH CURL": "HITCH/CURL",
        "IN DIG": "IN/DIG",
        "SHALLOW CROSS DRAG": "SHALLOW CROSS/DRAG",
        "TEXAS ANGLE": "TEXAS/ANGLE",
        "DEEP  OUT": "DEEP OUT",
        "QUICK  OUT": "QUICK OUT",
    }
    return q.replace(lexical)


def join_frames(part: pd.DataFrame, pbp: pd.DataFrame, rosters: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = lower(part)
    b = lower(pbp)
    r = lower(rosters)

    required_p = {"nflverse_game_id", "play_id", "route"}
    required_b = {
        "game_id", "play_id", "season", "week", "posteam", "defteam",
        "receiver_player_id", "receiver_player_name",
    }
    required_r = {"season", "week", "gsis_id", "position"}
    missing = {
        "participation": sorted(required_p - set(p.columns)),
        "pbp": sorted(required_b - set(b.columns)),
        "rosters": sorted(required_r - set(r.columns)),
    }
    if any(missing.values()):
        raise RuntimeError(f"missing source columns: {missing}")

    p = p[["nflverse_game_id", "play_id", "route"]].copy()
    bcols = [
        "game_id", "play_id", "season", "week", "posteam", "defteam",
        "receiver_player_id", "receiver_player_name",
    ]
    for c in ["season_type", "play_type", "pass_attempt", "complete_pass", "yards_gained", "air_yards"]:
        if c in b.columns:
            bcols.append(c)
    b = b[bcols].copy()

    p["nflverse_game_id"] = text_series(p["nflverse_game_id"])
    b["game_id"] = text_series(b["game_id"])
    p["play_id"] = pd.to_numeric(p["play_id"], errors="coerce")
    b["play_id"] = pd.to_numeric(b["play_id"], errors="coerce")

    route_rows = p.loc[text_series(p["route"]).ne("")].copy()
    merged = route_rows.merge(
        b,
        left_on=["nflverse_game_id", "play_id"],
        right_on=["game_id", "play_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if "season_type" in merged.columns:
        merged = merged.loc[text_series(merged["season_type"]).str.upper().eq("REG")].copy()

    merged["route_family"] = route_norm(merged["route"])
    merged["receiver_id"] = text_series(merged["receiver_player_id"])
    merged["receiver_name"] = text_series(merged["receiver_player_name"])
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce").astype("Int64")
    merged["week"] = pd.to_numeric(merged["week"], errors="coerce").astype("Int64")
    merged["posteam"] = text_series(merged["posteam"]).str.upper()
    merged["defteam"] = text_series(merged["defteam"]).str.upper()

    rr = r[["season", "week", "gsis_id", "position"]].copy()
    rr["season"] = pd.to_numeric(rr["season"], errors="coerce").astype("Int64")
    rr["week"] = pd.to_numeric(rr["week"], errors="coerce").astype("Int64")
    rr["gsis_id"] = text_series(rr["gsis_id"])
    rr["position"] = text_series(rr["position"]).str.upper()
    rr = rr.drop_duplicates(["season", "week", "gsis_id"], keep="last")

    merged = merged.merge(
        rr,
        left_on=["season", "week", "receiver_id"],
        right_on=["season", "week", "gsis_id"],
        how="left",
        validate="many_to_one",
    )
    merged["position"] = text_series(merged["position"]).str.upper()
    merged["valid_receiver_identity"] = merged["receiver_id"].ne("")
    merged["valid_route_taxonomy"] = merged["route_family"].isin(ROUTE_FAMILIES)
    merged["valid_wr_te"] = merged["position"].isin(VALID_POS)

    audit = {
        "route_labeled_participation_rows": int(len(route_rows)),
        "joined_rows": int((merged["_merge"] == "both").sum()) if "_merge" in merged else int(len(merged)),
        "join_rate": float((merged["_merge"] == "both").mean()) if len(merged) and "_merge" in merged else 0.0,
    }
    return merged, audit


def support_metrics(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = rows.copy()
    valid_route = x["valid_route_taxonomy"]
    identity_den = x.loc[valid_route].copy()
    identity_cov = float(identity_den["valid_receiver_identity"].mean()) if len(identity_den) else 0.0

    valid = x.loc[
        valid_route
        & x["valid_receiver_identity"]
        & x["valid_wr_te"]
        & x["season"].isin(SEASONS)
        & x["week"].notna()
        & x["posteam"].ne("")
        & x["defteam"].ne("")
    ].copy()
    valid["game_key"] = valid["nflverse_game_id"].astype(str)

    by_season = []
    eligible_players_by_season = {}
    for season in SCORED_SEASONS:
        s = valid.loc[valid["season"].eq(season)].copy()
        # At least one target-route observation in a game defines an eligible player-game.
        pg = s[["receiver_id", "game_key", "week"]].drop_duplicates().sort_values(["receiver_id", "week", "game_key"])
        counts = pg.groupby("receiver_id")["game_key"].nunique()
        eligible_players = int((counts >= 5).sum())  # >=4 strictly-prior games exists by the 5th eligible game
        eligible_players_by_season[str(season)] = eligible_players
        by_season.append({
            "season": season,
            "valid_targeted_route_rows": int(len(s)),
            "distinct_wr_te": int(s["receiver_id"].nunique()),
            "players_with_at_least_4_prior_eligible_games": eligible_players,
        })

    route_counts = (
        valid.groupby("route_family", as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )

    pg_counts = (
        valid.groupby(["season", "receiver_id", "game_key"], as_index=False)
        .size()
        .rename(columns={"size": "targeted_route_rows"})
    )
    player_summary = (
        pg_counts.groupby(["season", "receiver_id"], as_index=False)
        .agg(
            eligible_games=("game_key", "nunique"),
            median_targeted_route_rows_per_game=("targeted_route_rows", "median"),
        )
    )

    defense_rows = []
    defense_support_by_season = {}
    for season in SCORED_SEASONS:
        pre8 = valid.loc[valid["season"].eq(season) & valid["week"].lt(8)].copy()
        d = (
            pre8.groupby("defteam", as_index=False)
            .size()
            .rename(columns={"size": "prior_route_observations_by_week8"})
        )
        support_n = int((d["prior_route_observations_by_week8"] >= 100).sum())
        defense_support_by_season[str(season)] = support_n
        d["season"] = season
        defense_rows.append(d)
    defense_support = pd.concat(defense_rows, ignore_index=True) if defense_rows else pd.DataFrame()

    season_rows = {str(r["season"]): int(r["valid_targeted_route_rows"]) for r in by_season}
    families_1000 = int((route_counts["rows"] >= 1000).sum())
    metrics = {
        "identity_coverage_on_taxonomy_route_rows": identity_cov,
        "valid_targeted_route_rows": int(len(valid)),
        "valid_wr_te_players": int(valid["receiver_id"].nunique()),
        "season_rows": season_rows,
        "eligible_players_by_season": eligible_players_by_season,
        "route_families_ge_1000_rows": families_1000,
        "defenses_ge_100_prior_route_obs_by_week8": defense_support_by_season,
    }
    return metrics, pd.DataFrame(by_season), route_counts, defense_support


def disposition(metrics: dict, audit: dict) -> tuple[str, dict]:
    gates = {
        "join_rate_ge_95pct": bool(audit.get("join_rate", 0.0) >= 0.95),
        "identity_coverage_ge_90pct": bool(metrics["identity_coverage_on_taxonomy_route_rows"] >= 0.90),
        "five_route_families_ge_1000": bool(metrics["route_families_ge_1000_rows"] >= 5),
    }
    for s in SCORED_SEASONS:
        gates[f"{s}_rows_ge_10000"] = bool(metrics["season_rows"].get(str(s), 0) >= 10000)
        gates[f"{s}_eligible_players_ge_150"] = bool(metrics["eligible_players_by_season"].get(str(s), 0) >= 150)
        gates[f"{s}_defenses_supported_ge_24"] = bool(metrics["defenses_ge_100_prior_route_obs_by_week8"].get(str(s), 0) >= 24)
    if all(gates.values()):
        disp = "PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY"
    else:
        disp = "PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_LIMITED"
    return disp, gates


def run(participation: pd.DataFrame, pbp: pd.DataFrame, rosters: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    rows, audit = join_frames(participation, pbp, rosters)
    metrics, by_season, route_counts, defense_support = support_metrics(rows)
    disp, gates = disposition(metrics, audit)
    result = {
        "version": VERSION,
        "disposition": disp,
        "source_seasons": list(SEASONS),
        "scored_seasons": list(SCORED_SEASONS),
        "source_audit": audit,
        "support": metrics,
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
        "route_semantics": "primary_or_targeted_receiver_route_only_not_all_routes_run",
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_read": 0,
        "production_changed": False,
    }
    return result, {
        "joined_rows": rows,
        "by_season": by_season,
        "route_counts": route_counts,
        "defense_support": defense_support,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import nflreadpy as nfl

    participation = to_pandas(nfl.load_participation(seasons=list(SEASONS)))
    pbp = to_pandas(nfl.load_pbp(seasons=list(SEASONS)))
    rosters = to_pandas(nfl.load_rosters_weekly(seasons=list(SEASONS)))

    result, frames = run(participation, pbp, rosters)
    (args.out_dir / "player_targeted_route_matchup_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    frames["by_season"].to_csv(args.out_dir / "player_targeted_route_matchup_by_season.csv", index=False)
    frames["route_counts"].to_csv(args.out_dir / "player_targeted_route_matchup_route_counts.csv", index=False)
    frames["defense_support"].to_csv(args.out_dir / "player_targeted_route_matchup_defense_support.csv", index=False)
    # Keep a compact source sample for semantic review; do not persist the full multi-season joined table.
    frames["joined_rows"].head(1000).to_csv(args.out_dir / "player_targeted_route_matchup_source_sample.csv", index=False)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
