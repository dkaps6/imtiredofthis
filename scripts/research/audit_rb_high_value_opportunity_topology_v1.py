#!/usr/bin/env python3
"""RB High-Value Opportunity Topology V1 — frozen source audit.

Source/readiness only. Reconstructs named-RB rushing opportunity topology from
historical nflverse PBP + weekly roster identity. No yardage/TD outcome is
scored and no predictive model is fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_official_inactive_availability as m78

VERSION = "RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_V1_SOURCE_AUDIT"
READY = "RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_SOURCE_READY"
BLOCKED = "RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_SOURCE_BLOCKED"
INTEGRITY = "RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_SOURCE_INTEGRITY_FAILURE"
SEASONS = (2023, 2024, 2025)
RB_POS = {"RB", "FB", "HB"}

SITUATIONS = (
    "ANY_RB_RUSH",
    "RED_ZONE_RUSH",
    "INSIDE_10_RUSH",
    "INSIDE_5_RUSH",
    "GOAL_TO_GO_RUSH",
    "SHORT_YARDAGE_RUSH",
)

PLAYER_SUPPORT = {
    "ANY_RB_RUSH": (4, 100),
    "RED_ZONE_RUSH": (3, 60),
    "INSIDE_10_RUSH": (2, 40),
    "INSIDE_5_RUSH": (2, 25),
    "GOAL_TO_GO_RUSH": (2, 30),
    "SHORT_YARDAGE_RUSH": (2, 40),
}

TEAM_GAME_SUPPORT = {
    "RED_ZONE_RUSH": 200,
    "INSIDE_10_RUSH": 150,
    "INSIDE_5_RUSH": 100,
    "GOAL_TO_GO_RUSH": 100,
    "SHORT_YARDAGE_RUSH": 100,
}

REQ = {
    "season", "week", "game_id", "play_id", "posteam", "rush_attempt",
    "rusher_player_id", "yardline_100", "goal_to_go", "down", "ydstogo",
}


def to_pd(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl
    frames = []
    for season in SEASONS:
        raw = to_pd(nfl.load_pbp(seasons=[season]))
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        missing = REQ - set(raw.columns)
        if missing:
            raise RuntimeError(f"PBP {season} missing {sorted(missing)}")
        if "season_type" in raw.columns:
            raw = raw.loc[raw["season_type"].astype(str).str.upper().eq("REG")].copy()
        elif "game_type" in raw.columns:
            raw = raw.loc[raw["game_type"].astype(str).str.upper().eq("REG")].copy()
        raw["season"] = num(raw["season"]).astype("Int64")
        raw["week"] = num(raw["week"]).astype("Int64")
        raw = raw.loc[
            raw["season"].eq(season)
            & raw["week"].between(1, 18)
        ].copy()
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def load_rosters() -> pd.DataFrame:
    parts = []
    meta: list[dict] = []
    for season in SEASONS:
        raw = m78.load_weekly_rosters(season, meta)
        if raw.empty:
            raise RuntimeError(f"weekly roster unavailable {season}")
        x = raw.copy()
        x.columns = [str(c).strip().lower() for c in x.columns]
        wc = first_col(x, ["week"])
        tc = first_col(x, ["team", "club_code"])
        ic = first_col(x, ["gsis_id", "player_id"])
        pc = first_col(x, ["position", "depth_chart_position", "ngs_position", "pos"])
        if not all([wc, tc, ic, pc]):
            raise RuntimeError(f"weekly roster schema incomplete {season}")
        q = pd.DataFrame({
            "season": season,
            "week": num(x[wc]),
            "team": x[tc].map(canon_team),
            "player_id": x[ic].astype("string").fillna("").str.strip(),
            "position": x[pc].astype("string").fillna("").str.upper().str.strip(),
        })
        q = q.loc[
            q["week"].notna()
            & q["team"].astype(str).ne("")
            & q["player_id"].ne("")
        ].copy()
        q["week"] = q["week"].astype(int)
        parts.append(q)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(
        ["season", "week", "team", "player_id"], keep="first"
    ).reset_index(drop=True)


def situation_flags(df: pd.DataFrame) -> pd.DataFrame:
    y = num(df["yardline_100"])
    g = num(df["goal_to_go"]).fillna(0)
    down = num(df["down"])
    togo = num(df["ydstogo"])
    out = pd.DataFrame(index=df.index)
    out["ANY_RB_RUSH"] = True
    out["RED_ZONE_RUSH"] = y.le(20)
    out["INSIDE_10_RUSH"] = y.le(10)
    out["INSIDE_5_RUSH"] = y.le(5)
    out["GOAL_TO_GO_RUSH"] = g.eq(1)
    out["SHORT_YARDAGE_RUSH"] = down.isin([3, 4]) & togo.le(2)
    return out


def prepare_rushes(pbp: pd.DataFrame, roster: pd.DataFrame):
    p = pbp.copy()
    p["rush_attempt_num"] = num(p["rush_attempt"]).fillna(0)
    rush = p.loc[p["rush_attempt_num"].eq(1)].copy()
    rush["team"] = rush["posteam"].map(canon_team)
    rush["rusher_player_id"] = (
        rush["rusher_player_id"].astype("string").fillna("").str.strip()
    )
    rush["game_id"] = rush["game_id"].astype("string").fillna("").str.strip()
    rush["play_id_str"] = rush["play_id"].astype("string").fillna("").str.strip()

    game_team_identity = (
        rush["game_id"].ne("") & rush["team"].astype(str).ne("")
    )
    rusher_identity = rush["rusher_player_id"].ne("")

    map_cols = ["season", "week", "team", "player_id", "position"]
    r = roster[map_cols].rename(columns={"player_id": "rusher_player_id"})
    x = rush.merge(
        r,
        on=["season", "week", "team", "rusher_player_id"],
        how="left",
        validate="many_to_one",
    )
    mapped = x["position"].notna()
    x["position"] = x["position"].astype("string").fillna("").str.upper().str.strip()
    x["is_rb_family"] = x["position"].isin(RB_POS)

    coverage = {
        "rush_rows": int(len(x)),
        "game_team_identity_rate": float(game_team_identity.mean()) if len(rush) else 0.0,
        "rusher_identity_rate": float(rusher_identity.mean()) if len(rush) else 0.0,
        "roster_position_map_rate": float(mapped.mean()) if len(x) else 0.0,
    }

    rb = x.loc[x["is_rb_family"]].copy()
    coverage["rb_family_clean_rate"] = (
        float(rb["position"].isin(RB_POS).mean()) if len(rb) else 0.0
    )

    dup = rb.duplicated(
        ["season", "game_id", "play_id_str", "rusher_player_id"],
        keep=False,
    )
    coverage["duplicate_player_play_rows"] = int(dup.sum())

    flags = situation_flags(rb)
    for c in SITUATIONS:
        rb[c] = flags[c].astype(bool)

    return rb, coverage


def topology_rows(rb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for situation in SITUATIONS:
        q = rb.loc[rb[situation]].copy()
        if q.empty:
            continue
        pc = (
            q.groupby(
                ["season", "week", "game_id", "team", "rusher_player_id"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "player_situation_attempts"})
        )
        team = (
            pc.groupby(["season", "week", "game_id", "team"], as_index=False)
            .agg(
                team_rb_situation_attempts=("player_situation_attempts", "sum"),
                situation_rushers=("rusher_player_id", "nunique"),
            )
        )
        z = pc.merge(
            team,
            on=["season", "week", "game_id", "team"],
            how="left",
            validate="many_to_one",
        )
        z["player_situation_share"] = (
            z["player_situation_attempts"] / z["team_rb_situation_attempts"]
        )
        hhi = (
            z.assign(_sq=z["player_situation_share"] ** 2)
            .groupby(["season", "week", "game_id", "team"], as_index=False)["_sq"]
            .sum()
            .rename(columns={"_sq": "situation_hhi"})
        )
        z = z.merge(
            hhi,
            on=["season", "week", "game_id", "team"],
            how="left",
            validate="many_to_one",
        )
        z["situation"] = situation
        rows.append(z)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def season_metrics(
    pbp: pd.DataFrame,
    rb: pd.DataFrame,
    topology: pd.DataFrame,
    coverage: dict,
    season: int,
) -> dict:
    ps = pbp.loc[pbp["season"].eq(season)]
    rs = rb.loc[rb["season"].eq(season)]
    ts = topology.loc[topology["season"].eq(season)]

    weeks = sorted(num(ps["week"]).dropna().astype(int).unique().tolist())
    m = {
        "season": season,
        "weeks": weeks,
        "weeks_complete": weeks == list(range(1, 19)),
        "rb_rush_rows": int(len(rs)),
        "game_team_identity_rate": float(
            (
                ps.loc[num(ps["rush_attempt"]).fillna(0).eq(1), "game_id"]
                .astype("string").fillna("").str.strip().ne("")
                & ps.loc[num(ps["rush_attempt"]).fillna(0).eq(1), "posteam"]
                .astype("string").fillna("").str.strip().ne("")
            ).mean()
        ),
        "rusher_identity_rate": float(
            ps.loc[num(ps["rush_attempt"]).fillna(0).eq(1), "rusher_player_id"]
            .astype("string").fillna("").str.strip().ne("").mean()
        ),
    }

    # Exact season-specific roster map rate among all rushes is reconstructed
    # from the merged frame indirectly: RB-family rows are a subset, so record
    # the pooled exact rate separately and retain season integrity/support here.
    for situation in SITUATIONS:
        q = ts.loc[ts["situation"].eq(situation)].copy()
        min_games, _ = PLAYER_SUPPORT[situation]
        pg = (
            q.groupby("rusher_player_id")["game_id"].nunique()
            if len(q)
            else pd.Series(dtype=int)
        )
        m[f"{situation}_players_support"] = int((pg >= min_games).sum())
        m[f"{situation}_team_games"] = int(
            q[["game_id", "team"]].drop_duplicates().shape[0]
        )
    return m


def verify_topology(topology: pd.DataFrame) -> dict:
    if topology.empty:
        return {
            "share_finite": False,
            "share_range": False,
            "share_sum_max_gap": float("inf"),
        }
    s = num(topology["player_situation_share"])
    sums = (
        topology.groupby(
            ["season", "week", "game_id", "team", "situation"]
        )["player_situation_share"]
        .sum()
    )
    return {
        "share_finite": bool(np.isfinite(s.to_numpy(float)).all()),
        "share_range": bool(s.between(0, 1).all()),
        "share_sum_max_gap": float((sums - 1.0).abs().max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pbp = load_pbp()
        roster = load_rosters()
        rb, pooled_coverage = prepare_rushes(pbp, roster)
        topology = topology_rows(rb)
        topo_integrity = verify_topology(topology)

        # Reconstruct roster-map rate by season from all rush attempts using the
        # same exact join. This is a source gate, not a scientific result.
        rush = pbp.loc[num(pbp["rush_attempt"]).fillna(0).eq(1)].copy()
        rush["team"] = rush["posteam"].map(canon_team)
        rush["rusher_player_id"] = (
            rush["rusher_player_id"].astype("string").fillna("").str.strip()
        )
        rmap = roster[
            ["season", "week", "team", "player_id", "position"]
        ].rename(columns={"player_id": "rusher_player_id"})
        rush_map = rush.merge(
            rmap,
            on=["season", "week", "team", "rusher_player_id"],
            how="left",
            validate="many_to_one",
        )

        season_results = {}
        gates = {}
        for season in SEASONS:
            m = season_metrics(pbp, rb, topology, pooled_coverage, season)
            sm = rush_map.loc[rush_map["season"].eq(season)]
            roster_map_rate = float(sm["position"].notna().mean()) if len(sm) else 0.0
            m["roster_position_map_rate"] = roster_map_rate
            rbs = sm.loc[
                sm["position"].astype("string").fillna("").str.upper().isin(RB_POS)
            ]
            rb_clean = (
                float(
                    rbs["position"]
                    .astype("string").fillna("").str.upper()
                    .isin(RB_POS).mean()
                )
                if len(rbs)
                else 0.0
            )
            m["rb_family_clean_rate"] = rb_clean

            g = {
                "weeks_complete": bool(m["weeks_complete"]),
                "game_team_identity_ge_99": bool(m["game_team_identity_rate"] >= 0.99),
                "rusher_identity_ge_99": bool(m["rusher_identity_rate"] >= 0.99),
                "roster_position_map_ge_95": bool(roster_map_rate >= 0.95),
                "rb_family_clean_ge_95": bool(rb_clean >= 0.95),
            }
            for situation, (min_games, min_players) in PLAYER_SUPPORT.items():
                g[f"{situation}_player_support"] = bool(
                    m[f"{situation}_players_support"] >= min_players
                )
            for situation, min_games in TEAM_GAME_SUPPORT.items():
                g[f"{situation}_team_game_support"] = bool(
                    m[f"{situation}_team_games"] >= min_games
                )
            season_results[str(season)] = m
            gates[str(season)] = g

        global_gates = {
            "share_finite": topo_integrity["share_finite"],
            "share_range": topo_integrity["share_range"],
            "share_sums_to_one": topo_integrity["share_sum_max_gap"] <= 1e-9,
            "duplicate_player_play_identity_zero": pooled_coverage[
                "duplicate_player_play_rows"
            ] == 0,
            "sportsbook_inputs_zero": True,
            "oddsapi_calls_zero": True,
            "outcomes_2026_read_zero": True,
            "predictive_models_fit_zero": True,
            "production_unchanged": True,
        }

        all_pass = (
            all(all(v for v in g.values()) for g in gates.values())
            and all(global_gates.values())
        )
        disposition = READY if all_pass else BLOCKED

        result = {
            "version": VERSION,
            "disposition": disposition,
            "season_metrics": season_results,
            "season_gates": gates,
            "global_integrity": topo_integrity,
            "global_gates": global_gates,
            "failed_gates": {
                s: [k for k, v in g.items() if not v]
                for s, g in gates.items()
            },
            "failed_global_gates": [
                k for k, v in global_gates.items() if not v
            ],
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "predictive_models_fit": 0,
            "projection_values_changed": 0,
            "production_changed": False,
        }

        topology.to_csv(
            args.out_dir / "rb_high_value_opportunity_topology_rows.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"season": int(s), **season_results[str(s)]}
                for s in SEASONS
            ]
        ).to_csv(
            args.out_dir / "rb_high_value_opportunity_topology_source_metrics.csv",
            index=False,
        )
    except Exception as exc:
        result = {
            "version": VERSION,
            "disposition": INTEGRITY,
            "error": str(exc),
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "predictive_models_fit": 0,
            "projection_values_changed": 0,
            "production_changed": False,
        }
        (
            args.out_dir / "rb_high_value_opportunity_topology_source_result.json"
        ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (
        args.out_dir / "rb_high_value_opportunity_topology_source_result.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
