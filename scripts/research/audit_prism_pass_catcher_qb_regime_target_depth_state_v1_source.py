#!/usr/bin/env python3
"""PRISM pass-catcher x QB-regime target-depth state V1 source audit.

Source/chronology only. This script does not score a predictive candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research.audit_prism_pass_catcher_qb_regime_allocation_state_v1_source import (
    qb_regimes,
    universe,
    weekly,
)

VERSION = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_AUDIT"
SEASONS = (2024, 2025)
FAMILIES = ("WR", "TE")
HISTORY_SEASONS = (2023, 2024, 2025)


def num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def to_pd(x: Any) -> pd.DataFrame:
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    if hasattr(x, "to_dicts"):
        return pd.DataFrame(x.to_dicts())
    return pd.DataFrame(x)


def clean(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def regular_only(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    c = "season_type" if "season_type" in y.columns else "game_type" if "game_type" in y.columns else None
    if c:
        s = y[c].astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            y = y.loc[keep].copy()
    return y


def before_mask(frame: pd.DataFrame, season: int, week: int) -> pd.Series:
    s = num(frame["season"])
    w = num(frame["week"])
    return s.lt(int(season)) | (s.eq(int(season)) & w.lt(int(week)))


def load_target_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    events = []
    provider = []
    for season in HISTORY_SEASONS:
        x = regular_only(to_pd(nfl.load_pbp(seasons=[season])))
        for c in [
            "season", "week", "game_id", "posteam", "pass_attempt", "sack",
            "two_point_attempt", "receiver_player_id", "receiver_player_name",
            "passer_player_id", "air_yards",
        ]:
            if c not in x.columns:
                x[c] = np.nan
        x["season"] = num(x["season"]).fillna(season).astype(int)
        x["week"] = num(x["week"]).astype("Int64")
        x["team"] = x["posteam"].map(canon_team)
        x["receiver_id"] = x["receiver_player_id"].map(clean)
        x["receiver_name"] = x["receiver_player_name"].map(clean)
        x["passer_id"] = x["passer_player_id"].map(clean)
        x["air_yards"] = num(x["air_yards"])

        pass_attempt = num(x["pass_attempt"]).fillna(0).eq(1)
        sack = num(x["sack"]).fillna(0).eq(1)
        two = num(x["two_point_attempt"]).fillna(0).eq(1)
        # A target-like play needs a receiver identity or receiver name. If the
        # source omits receiver names entirely, receiver ID is the denominator.
        has_receiver_semantic = x["receiver_id"].ne("")
        if x["receiver_name"].ne("").any():
            has_receiver_semantic = has_receiver_semantic | x["receiver_name"].ne("")
        potential = pass_attempt & ~sack & ~two & has_receiver_semantic
        p = x.loc[potential].copy()
        provider.append(
            {
                "season": season,
                "target_semantic_rows": int(len(p)),
                "receiver_id_rate": float(p["receiver_id"].ne("").mean()) if len(p) else 0.0,
                "passer_id_rate": float(p["passer_id"].ne("").mean()) if len(p) else 0.0,
                "air_yards_rate": float(p["air_yards"].notna().mean()) if len(p) else 0.0,
            }
        )
        target = potential & x["receiver_id"].ne("")
        events.append(
            x.loc[target, ["season", "week", "game_id", "team", "receiver_id", "passer_id", "air_yards"]].copy()
        )
    return pd.concat(events, ignore_index=True), pd.DataFrame(provider)


def latest_games(events: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if events.empty:
        return events
    g = (
        events.loc[events["air_yards"].notna(), ["season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["season", "week", "game_id"])
        .tail(limit)
    )
    if g.empty:
        return events.iloc[0:0].copy()
    keys = set(zip(g["season"].astype(int), g["week"].astype(int), g["game_id"].astype(str)))
    keep = [
        (int(s), int(w), str(gid)) in keys
        for s, w, gid in zip(events["season"], events["week"], events["game_id"])
    ]
    return events.loc[keep].copy()


def event_summary(x: pd.DataFrame) -> dict[str, Any]:
    z = x.loc[x["air_yards"].notna()].copy()
    air = num(z["air_yards"]).dropna().to_numpy(float)
    games = int(z[["season", "week", "game_id"]].drop_duplicates().shape[0]) if len(z) else 0
    if len(air):
        max_season = int(num(z["season"]).max())
        max_week = int(num(z.loc[num(z["season"]).eq(max_season), "week"]).max())
    else:
        max_season = None
        max_week = None
    return {
        "games": games,
        "finite_air_targets": int(len(air)),
        "mean_air_yards": float(np.mean(air)) if len(air) else np.nan,
        "target_depth_sd": float(np.std(air, ddof=0)) if len(air) else np.nan,
        "deep20_rate": float(np.mean(air >= 20.0)) if len(air) else np.nan,
        "max_source_season": max_season,
        "max_source_week": max_week,
    }


def build_state(u: pd.DataFrame, reg: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    target = u.loc[u["family"].isin(FAMILIES)].copy()
    target = target.merge(reg, on=["season", "week", "team"], how="left", validate="many_to_one")
    groups = {
        (str(team), str(pid)): g.sort_values(["season", "week", "game_id"]).copy()
        for (team, pid), g in events.groupby(["team", "receiver_id"], sort=False)
    }

    rows = []
    for r in target.itertuples(index=False):
        season = int(r.season)
        week = int(r.week)
        team = str(r.team)
        player = str(r.player_id)
        qb = clean(getattr(r, "selected_qb_id", ""))

        h = groups.get((team, player), pd.DataFrame(columns=events.columns))
        if len(h):
            h = h.loc[before_mask(h, season, week)].copy()
        recent = latest_games(h, 8)
        overall = event_summary(recent)
        same = recent.loc[recent["passer_id"].astype(str).eq(qb)].copy() if qb else recent.iloc[0:0].copy()
        pair = event_summary(same)

        baseline_ready = (
            overall["games"] >= 4
            and overall["finite_air_targets"] >= 10
            and np.isfinite(overall["mean_air_yards"])
        )
        same_qb_ready = (
            pair["games"] >= 2
            and pair["finite_air_targets"] >= 4
            and np.isfinite(pair["mean_air_yards"])
        )
        violation = bool(
            overall["max_source_season"] is not None
            and (
                overall["max_source_season"] > season
                or (
                    overall["max_source_season"] == season
                    and int(overall["max_source_week"]) >= week
                )
            )
        )

        rows.append(
            {
                "season": season,
                "week": week,
                "team": team,
                "player_id": player,
                "family": str(r.family),
                "position": str(r.position),
                "stable_roster_identity": int(r.stable_roster_identity),
                "selected_qb_id": qb,
                "qb_regime_source": str(getattr(r, "qb_regime_source", "")),
                "qb_regime_source_valid": bool(getattr(r, "qb_regime_source_valid", False)),
                "prior8_receiver_games": overall["games"],
                "prior8_finite_air_targets": overall["finite_air_targets"],
                "prior8_mean_air_yards": overall["mean_air_yards"],
                "prior8_target_depth_sd": overall["target_depth_sd"],
                "prior8_deep20_rate": overall["deep20_rate"],
                "same_qb_prior_games": pair["games"],
                "same_qb_finite_air_targets": pair["finite_air_targets"],
                "same_qb_mean_air_yards": pair["mean_air_yards"],
                "same_qb_target_depth_sd": pair["target_depth_sd"],
                "same_qb_deep20_rate": pair["deep20_rate"],
                "max_source_season": overall["max_source_season"],
                "max_source_week": overall["max_source_week"],
                "baseline_history_ready": bool(baseline_ready),
                "same_qb_history_ready": bool(same_qb_ready),
                "mechanism_source_ready": bool(
                    int(r.stable_roster_identity) == 1
                    and bool(getattr(r, "qb_regime_source_valid", False))
                    and baseline_ready
                    and same_qb_ready
                    and not violation
                ),
                "chronology_violation": violation,
            }
        )
    return pd.DataFrame(rows)


def source_metrics(state: pd.DataFrame, reg: pd.DataFrame, provider: pd.DataFrame) -> pd.DataFrame:
    rows = []
    provider_idx = provider.set_index("season")
    for season in SEASONS:
        q = reg.loc[reg["season"].eq(season)]
        q_rate = float(q["qb_regime_source_valid"].mean()) if len(q) else 0.0
        p = provider_idx.loc[season]
        for family in FAMILIES:
            x = state.loc[state["season"].eq(season) & state["family"].eq(family)].copy()
            stable = float(x["stable_roster_identity"].eq(1).mean()) if len(x) else 0.0
            ready = x.loc[x["mechanism_source_ready"]].copy()
            min_rows = 1000 if family == "WR" else 600
            min_players = 150 if family == "WR" else 80
            gates = {
                "stable_roster_identity_99": stable >= 0.99,
                "qb_regime_source_90": q_rate >= 0.90,
                "pbp_receiver_id_99": float(p["receiver_id_rate"]) >= 0.99,
                "pbp_passer_id_99": float(p["passer_id_rate"]) >= 0.99,
                "pbp_air_yards_98": float(p["air_yards_rate"]) >= 0.98,
                "mechanism_support": len(ready) >= min_rows and ready["player_id"].nunique() >= min_players,
                "chronology_zero": int(x["chronology_violation"].sum()) == 0,
            }
            rows.append(
                {
                    "season": season,
                    "family": family,
                    "target_state_rows": int(len(x)),
                    "stable_roster_identity_rate": stable,
                    "qb_regime_source_valid_teamweek_rate": q_rate,
                    "pbp_target_receiver_id_rate": float(p["receiver_id_rate"]),
                    "pbp_target_passer_id_rate": float(p["passer_id_rate"]),
                    "pbp_target_air_yards_rate": float(p["air_yards_rate"]),
                    "baseline_history_ready_rows": int(x["baseline_history_ready"].sum()),
                    "same_qb_history_ready_rows": int(x["same_qb_history_ready"].sum()),
                    "mechanism_source_ready_rows": int(len(ready)),
                    "mechanism_source_ready_players": int(ready["player_id"].nunique()),
                    "chronology_violations": int(x["chronology_violation"].sum()),
                    **{f"gate_{k}": bool(v) for k, v in gates.items()},
                    "source_ready": bool(all(gates.values())),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregame-universe", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    u = universe(a.pregame_universe)
    logs = weekly()
    reg = qb_regimes(u, logs)
    events, provider = load_target_events()
    state = build_state(u, reg, events)
    metrics = source_metrics(state, reg, provider)

    ready = int(metrics["source_ready"].sum())
    if ready == len(metrics):
        disposition = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_READY_ALL"
    elif ready:
        disposition = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_READY_PARTIAL"
    else:
        disposition = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_BLOCKED"

    result = {
        "version": VERSION,
        "disposition": disposition,
        "target_seasons": list(SEASONS),
        "target_families": list(FAMILIES),
        "pregame_universe_rows": int(len(u)),
        "qb_regime_teamweeks": int(len(reg)),
        "target_event_rows": int(len(events)),
        "player_target_depth_state_rows": int(len(state)),
        "ready_family_seasons": ready,
        "total_family_seasons": int(len(metrics)),
        "metrics": metrics.to_dict("records"),
        "integrity": {
            "chronology_violations": int(state["chronology_violation"].sum()),
            "target_game_outcomes_used_in_predictor": 0,
            "predictive_candidates_scored": 0,
            "receiving_yards_scored": False,
            "receptions_scored": False,
            "touchdowns_scored": False,
            "market_translation_scored": False,
            "coefficients_fit": 0,
            "thresholds_searched": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "production_changed": False,
        },
        "boundary": (
            "A source pass authorizes only a separately frozen target-depth scoring test. "
            "Target entitlement and market translation remain untouched."
        ),
    }

    provider.to_csv(a.out_dir / "qb_regime_target_depth_provider_quality.csv", index=False)
    state.to_csv(a.out_dir / "qb_regime_target_depth_state.csv", index=False)
    metrics.to_csv(a.out_dir / "qb_regime_target_depth_source_metrics.csv", index=False)
    (a.out_dir / "qb_regime_target_depth_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
