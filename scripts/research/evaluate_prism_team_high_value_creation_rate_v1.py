#!/usr/bin/env python3
"""Evaluate PRISM Team High-Value Opportunity Creation Rate V1.

Frozen contract:
docs/research/PRISM_TEAM_HIGH_VALUE_CREATION_RATE_V1_FREEZE.md

The only candidate change is team strictly-prior high-value creation rate.
Canonical parent opportunity quantity and named-player entitlement remain fixed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from scripts._opponent_map import canon_team

VERSION = "PRISM_TEAM_HIGH_VALUE_CREATION_RATE_V1"
FREEZE_COMMIT = "02e25a29e26c0562f298fdd71a517f7602f424cf"
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
HISTORY_WINDOW = 4
HISTORY_MIN = 3
BOOT = 5000
TOL = 1e-10

CELL_MAP = {
    "RB_RUSH_INSIDE10": {
        "stage2_cell": "RB_RUSHING",
        "family": "RB",
        "source_flag": "rush_source_present",
        "team_hv": "inside10_rush",
        "team_parent": "team_rush_attempts",
        "topology_channel": "RB_INSIDE10_RUSH",
        "min_player_games": 800,
        "min_players": 80,
        "qualified": "RB_RUSH_INSIDE10_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED",
        "seed": 2026101701,
    },
    "RB_RECEIVING_REDZONE": {
        "stage2_cell": "RB_RECEIVING",
        "family": "RB",
        "source_flag": "target_source_present",
        "team_hv": "red_zone_targets",
        "team_parent": "team_targets",
        "topology_channel": "RED_ZONE_TARGET",
        "min_player_games": 800,
        "min_players": 80,
        "qualified": "RB_RECEIVING_REDZONE_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED",
        "seed": 2026101711,
    },
    "TE_RECEIVING_REDZONE": {
        "stage2_cell": "TE_RECEIVING",
        "family": "TE",
        "source_flag": "target_source_present",
        "team_hv": "red_zone_targets",
        "team_parent": "team_targets",
        "topology_channel": "RED_ZONE_TARGET",
        "min_player_games": 700,
        "min_players": 75,
        "qualified": "TE_RECEIVING_REDZONE_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED",
        "seed": 2026101721,
    },
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype("string").fillna("").str.lower().isin(
        ["true", "1", "yes", "y"]
    )


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def adjacent_binomial_pmf(mu: float, p: float) -> np.ndarray:
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError(f"invalid parent mean {mu}")
    if not np.isfinite(p) or p < -TOL or p > 1 + TOL:
        raise ValueError(f"invalid probability {p}")
    p = float(np.clip(p, 0.0, 1.0))
    n0 = int(math.floor(mu))
    n1 = int(math.ceil(mu))
    w = float(mu - n0)
    ks = np.arange(n1 + 1)
    pmf0 = binom.pmf(ks, n0, p)
    if n1 == n0:
        out = pmf0
    else:
        pmf1 = binom.pmf(ks, n1, p)
        out = (1.0 - w) * pmf0 + w * pmf1
    out = np.asarray(out, dtype=float)
    out /= float(out.sum())
    return out


def adjacent_binomial_crps(mu: float, p: float, actual: float) -> float:
    if not np.isfinite(actual) or actual < -TOL:
        raise ValueError(f"invalid actual count {actual}")
    y = int(round(float(actual)))
    if abs(float(actual) - y) > TOL:
        raise ValueError(f"non-integer actual count {actual}")
    pmf = adjacent_binomial_pmf(mu, p)
    nmax = len(pmf) - 1
    top = max(nmax, y)
    cdf = np.ones(top + 1, dtype=float)
    cdf[: nmax + 1] = np.cumsum(pmf)
    obs = (np.arange(top + 1) >= y).astype(float)
    return float(np.square(cdf - obs).sum())


def distribution_mean(mu: float, p: float) -> float:
    return float(mu) * float(p)


def prepare_teamgames(x: pd.DataFrame) -> pd.DataFrame:
    need = {
        "season", "week", "game_id", "team", "opponent",
        "rush_source_present", "target_source_present",
        "team_rush_attempts", "inside10_rush",
        "team_targets", "red_zone_targets",
    }
    if need - set(x.columns):
        raise RuntimeError(f"team source missing {sorted(need-set(x.columns))}")
    q = x.copy()
    q["season"] = num(q["season"]).astype("Int64")
    q["week"] = num(q["week"]).astype("Int64")
    if int(q["season"].max()) > 2025:
        raise RuntimeError("2026 outcome leak in team source")
    q["team"] = q["team"].map(canon_team)
    q["opponent"] = q["opponent"].map(canon_team)
    q["game_id"] = q["game_id"].astype("string").fillna("").str.strip()
    for c in ["team_rush_attempts", "inside10_rush", "team_targets", "red_zone_targets"]:
        q[c] = num(q[c])
    for c in ["rush_source_present", "target_source_present"]:
        q[c] = as_bool(q[c])
    if q.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate raw team-game")
    return q


def prior_rate_state(
    teamgames: pd.DataFrame, spec: dict, season: int
) -> pd.DataFrame:
    x = teamgames.loc[
        teamgames["season"].eq(season)
        & teamgames["week"].between(1, 18)
    ].copy()
    flag = spec["source_flag"]
    hv = spec["team_hv"]
    parent = spec["team_parent"]

    valid = (
        x[flag]
        & num(x[hv]).notna()
        & num(x[parent]).notna()
        & num(x[parent]).gt(0)
    )
    x["_valid"] = valid

    out: list[dict] = []
    for r in x.loc[x["week"].isin(TARGET_WEEKS)].itertuples(index=False):
        week = int(r.week)
        team = str(r.team)
        league = x.loc[x["_valid"] & x["week"].lt(week)].copy()
        own = x.loc[
            x["_valid"] & x["team"].eq(team) & x["week"].lt(week)
        ].sort_values("week").tail(HISTORY_WINDOW)

        league_parent = float(num(league[parent]).sum()) if len(league) else 0.0
        team_parent = float(num(own[parent]).sum()) if len(own) else 0.0
        league_hv = float(num(league[hv]).sum()) if len(league) else 0.0
        team_hv = float(num(own[hv]).sum()) if len(own) else 0.0

        p_league = league_hv / league_parent if league_parent > 0 else np.nan
        p_team = team_hv / team_parent if team_parent > 0 else np.nan
        if np.isfinite(p_league) and (p_league < -TOL or p_league > 1 + TOL):
            raise RuntimeError("league high-value rate outside [0,1]")
        if np.isfinite(p_team) and (p_team < -TOL or p_team > 1 + TOL):
            raise RuntimeError("team high-value rate outside [0,1]")

        max_prior = int(own["week"].max()) if len(own) else 0
        if max_prior >= week:
            raise RuntimeError("same/future team history leakage")

        out.append({
            "season": season,
            "week": week,
            "game_id": str(r.game_id),
            "team": team,
            "opponent": str(r.opponent),
            "actual_team_high_value": float(getattr(r, hv)),
            "actual_team_parent": float(getattr(r, parent)),
            "target_source_present": bool(getattr(r, flag)),
            "history_games": int(len(own)),
            "history_max_week": max_prior,
            "history_parent_sum": team_parent,
            "history_high_value_sum": team_hv,
            "league_prior_teamgames": int(len(league)),
            "league_parent_sum": league_parent,
            "league_high_value_sum": league_hv,
            "p_league": float(np.clip(p_league, 0, 1)) if np.isfinite(p_league) else np.nan,
            "p_team": float(np.clip(p_team, 0, 1)) if np.isfinite(p_team) else np.nan,
        })
    return pd.DataFrame(out)


def prepare_stage2(
    stage2: pd.DataFrame, cell: str, spec: dict, season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    need = {
        "season", "week", "team", "player_id", "cell",
        "authority_q", "authority_denominator_mean",
        "authority_covered", "denominator_finite",
    }
    if need - set(stage2.columns):
        raise RuntimeError(f"Stage-2 detail missing {sorted(need-set(stage2.columns))}")
    q = stage2.loc[
        num(stage2["season"]).eq(season)
        & num(stage2["week"]).isin(TARGET_WEEKS)
        & stage2["cell"].astype(str).eq(spec["stage2_cell"])
    ].copy()
    q["season"] = num(q["season"]).astype(int)
    q["week"] = num(q["week"]).astype(int)
    q["team"] = q["team"].map(canon_team)
    q["player_id"] = q["player_id"].astype("string").fillna("").str.strip()
    q["authority_q"] = num(q["authority_q"])
    q["authority_denominator_mean"] = num(q["authority_denominator_mean"])
    q["authority_covered"] = as_bool(q["authority_covered"])
    q["denominator_finite"] = as_bool(q["denominator_finite"])
    q = q.loc[q["authority_covered"] & q["denominator_finite"]].copy()

    if q.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError(f"duplicate Stage-2 player identity {cell} {season}")
    if (
        q["authority_q"].isna().any()
        or (~np.isfinite(q["authority_q"])).any()
        or (q["authority_q"] < -TOL).any()
        or (q["authority_q"] > 1 + TOL).any()
    ):
        raise RuntimeError(f"invalid canonical q {cell} {season}")
    if (
        q["authority_denominator_mean"].isna().any()
        or (~np.isfinite(q["authority_denominator_mean"])).any()
        or (q["authority_denominator_mean"] <= 0).any()
    ):
        raise RuntimeError(f"invalid canonical denominator {cell} {season}")

    rooms = []
    for keys, g in q.groupby(["season", "week", "team"], sort=False):
        d = g["authority_denominator_mean"].to_numpy(float)
        spread = float(np.max(d) - np.min(d)) if len(d) else np.inf
        if spread > 1e-8:
            raise RuntimeError(
                f"canonical denominator not team-consistent {cell} {keys} spread={spread}"
            )
        qsum = float(g["authority_q"].sum())
        if qsum > 1.0 + TOL:
            raise RuntimeError(f"canonical family room q exceeds 1 {cell} {keys}: {qsum}")
        rooms.append({
            "season": int(keys[0]),
            "week": int(keys[1]),
            "team": str(keys[2]),
            "authority_denominator_mean": float(np.mean(d)),
            "family_room_q_sum": qsum,
            "player_rows": int(len(g)),
        })
    return q, pd.DataFrame(rooms)


def prepare_player_labels(
    topology: pd.DataFrame, spec: dict, season: int
) -> pd.DataFrame:
    need = {
        "season", "week", "team", "player_id", "family",
        "channel", "high_value_opportunities",
    }
    if need - set(topology.columns):
        raise RuntimeError(f"player topology missing {sorted(need-set(topology.columns))}")
    q = topology.loc[
        num(topology["season"]).eq(season)
        & num(topology["week"]).isin(TARGET_WEEKS)
        & topology["channel"].astype(str).eq(spec["topology_channel"])
        & topology["family"].astype(str).eq(spec["family"])
    ].copy()
    q["season"] = num(q["season"]).astype(int)
    q["week"] = num(q["week"]).astype(int)
    q["team"] = q["team"].map(canon_team)
    q["player_id"] = q["player_id"].astype("string").fillna("").str.strip()
    q["actual_player_high_value"] = num(q["high_value_opportunities"]).fillna(0)
    q = (
        q.groupby(["season", "week", "team", "player_id"], as_index=False)
        ["actual_player_high_value"].sum()
    )
    return q


def score_team_rows(x: pd.DataFrame) -> pd.DataFrame:
    q = x.copy()
    q["baseline_mean"] = q["authority_denominator_mean"] * q["p_league"]
    q["candidate_mean"] = q["authority_denominator_mean"] * q["p_team"]
    q["baseline_ae"] = (q["actual_team_high_value"] - q["baseline_mean"]).abs()
    q["candidate_ae"] = (q["actual_team_high_value"] - q["candidate_mean"]).abs()
    q["ae_gain"] = q["baseline_ae"] - q["candidate_ae"]
    q["baseline_crps"] = [
        adjacent_binomial_crps(mu, p, y)
        for mu, p, y in zip(
            q["authority_denominator_mean"], q["p_league"], q["actual_team_high_value"]
        )
    ]
    q["candidate_crps"] = [
        adjacent_binomial_crps(mu, p, y)
        for mu, p, y in zip(
            q["authority_denominator_mean"], q["p_team"], q["actual_team_high_value"]
        )
    ]
    q["crps_gain"] = q["baseline_crps"] - q["candidate_crps"]
    q["team_game_cluster"] = (
        q["season"].astype(str) + "-" + q["week"].astype(str) + "-" + q["team"]
    )
    return q


def score_player_rows(
    players: pd.DataFrame,
    labels: pd.DataFrame,
    team_scored: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["season", "week", "team"]
    p = players.merge(
        labels,
        on=keys + ["player_id"],
        how="left",
        validate="one_to_one",
    )
    p["actual_player_high_value"] = num(p["actual_player_high_value"]).fillna(0.0)
    # Keep the player's own persisted canonical denominator from Stage-2.
    # The team-scored frame carries the same room-level value; merging it again
    # would create pandas suffix columns and hide the authoritative player field.
    keep = keys + [
        "game_id", "actual_team_high_value", "p_league", "p_team",
    ]
    p = p.merge(team_scored[keep], on=keys, how="inner", validate="many_to_one")
    if (p["actual_player_high_value"] > p["actual_team_high_value"] + TOL).any():
        raise RuntimeError("player high-value count exceeds team pool")
    p["baseline_event_p"] = p["p_league"] * p["authority_q"]
    p["candidate_event_p"] = p["p_team"] * p["authority_q"]
    if (
        (p[["baseline_event_p", "candidate_event_p"]] < -TOL).any().any()
        or (p[["baseline_event_p", "candidate_event_p"]] > 1 + TOL).any().any()
    ):
        raise RuntimeError("player marginal event probability outside [0,1]")
    p["baseline_mean"] = (
        p["authority_denominator_mean"] * p["baseline_event_p"]
    )
    p["candidate_mean"] = (
        p["authority_denominator_mean"] * p["candidate_event_p"]
    )
    p["baseline_ae"] = (p["actual_player_high_value"] - p["baseline_mean"]).abs()
    p["candidate_ae"] = (p["actual_player_high_value"] - p["candidate_mean"]).abs()
    p["ae_gain"] = p["baseline_ae"] - p["candidate_ae"]
    p["baseline_crps"] = [
        adjacent_binomial_crps(mu, pp, y)
        for mu, pp, y in zip(
            p["authority_denominator_mean"],
            p["baseline_event_p"],
            p["actual_player_high_value"],
        )
    ]
    p["candidate_crps"] = [
        adjacent_binomial_crps(mu, pp, y)
        for mu, pp, y in zip(
            p["authority_denominator_mean"],
            p["candidate_event_p"],
            p["actual_player_high_value"],
        )
    ]
    p["crps_gain"] = p["baseline_crps"] - p["candidate_crps"]
    p["team_game_cluster"] = (
        p["season"].astype(str) + "-" + p["week"].astype(str) + "-" + p["team"]
    )
    return p


def metrics(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "baseline_crps": float(df["baseline_crps"].mean()),
        "candidate_crps": float(df["candidate_crps"].mean()),
        "crps_gain": float(df["crps_gain"].mean()),
        "baseline_expected_count_ae": float(df["baseline_ae"].mean()),
        "candidate_expected_count_ae": float(df["candidate_ae"].mean()),
        "expected_count_ae_gain": float(df["ae_gain"].mean()),
    }


def cluster_bootstrap(
    df: pd.DataFrame, gain: str, cluster: str, seed: int
) -> dict:
    g = (
        df[[cluster, gain]]
        .dropna()
        .groupby(cluster)[gain]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )
    if len(g) < 2:
        return {
            "clusters": int(len(g)), "reps": BOOT,
            "observed": float(df[gain].mean()) if len(df) else np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
        }
    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT, dtype=float)
    n = len(g)
    for i in range(BOOT):
        ix = rng.integers(0, n, n)
        vals[i] = sums[ix].sum() / counts[ix].sum()
    return {
        "clusters": int(n),
        "reps": BOOT,
        "observed": float(df[gain].mean()),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def equal_player_bootstrap(df: pd.DataFrame, seed: int) -> tuple[dict, dict]:
    g = df.groupby("player_id")["crps_gain"].agg(["mean", "count"])
    g = g.loc[g["count"].ge(4)].copy()
    vals0 = g["mean"].to_numpy(float)
    if len(vals0) < 2:
        boot = {
            "players": int(len(g)), "reps": BOOT,
            "observed": float(np.mean(vals0)) if len(vals0) else np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
        }
    else:
        rng = np.random.default_rng(seed)
        vals = np.empty(BOOT, dtype=float)
        n = len(vals0)
        for i in range(BOOT):
            vals[i] = float(vals0[rng.integers(0, n, n)].mean())
        boot = {
            "players": int(n), "reps": BOOT,
            "observed": float(vals0.mean()),
            "ci_low": float(np.quantile(vals, 0.025)),
            "ci_high": float(np.quantile(vals, 0.975)),
        }
    hetero = {
        "players_ge4": int(len(g)),
        "median_player_crps_gain": float(g["mean"].median()) if len(g) else np.nan,
        "positive_player_share": float(g["mean"].gt(0).mean()) if len(g) else np.nan,
        "q25": float(g["mean"].quantile(0.25)) if len(g) else np.nan,
        "q75": float(g["mean"].quantile(0.75)) if len(g) else np.nan,
    }
    return boot, hetero


def evaluate_cell(
    *,
    cell: str,
    spec: dict,
    teamgames: pd.DataFrame,
    topology: pd.DataFrame,
    stage2: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    by_season = {}
    team_parts = []
    player_parts = []
    support_ok = True
    source_integrity = True

    for season in TARGET_SEASONS:
        rates = prior_rate_state(teamgames, spec, season)
        players, rooms = prepare_stage2(stage2, cell, spec, season)

        target = rates.loc[
            rates["target_source_present"]
            & rates["history_games"].ge(HISTORY_MIN)
            & rates["p_league"].notna()
            & rates["p_team"].notna()
        ].copy()

        topology_target = teamgames.loc[
            teamgames["season"].eq(season)
            & teamgames["week"].isin(TARGET_WEEKS)
            & teamgames[spec["source_flag"]]
        ].copy()
        topology_cov = len(topology_target) / max(
            1,
            len(
                teamgames.loc[
                    teamgames["season"].eq(season)
                    & teamgames["week"].isin(TARGET_WEEKS)
                ]
            ),
        )

        room_cov = len(
            rooms.merge(
                target[["season", "week", "team"]].drop_duplicates(),
                on=["season", "week", "team"],
                how="inner",
            )
        ) / max(1, len(target))

        target = target.merge(
            rooms,
            on=["season", "week", "team"],
            how="inner",
            validate="one_to_one",
        )

        if (
            (target["actual_team_high_value"] < -TOL).any()
            or (target["actual_team_parent"] <= 0).any()
            or (
                target["actual_team_high_value"]
                > target["actual_team_parent"] + TOL
            ).any()
        ):
            raise RuntimeError(f"finite team-pool integrity failure {cell} {season}")

        team_scored = score_team_rows(target)
        labels = prepare_player_labels(topology, spec, season)
        player_scored = score_player_rows(players, labels, team_scored)

        team_scored["quality_cell"] = cell
        player_scored["quality_cell"] = cell
        team_parts.append(team_scored)
        player_parts.append(player_scored)

        team_m = metrics(team_scored)
        player_m = metrics(player_scored)
        team_m["teams"] = int(team_scored["team"].nunique())
        team_m["positive_high_value_teamgames"] = int(
            team_scored["actual_team_high_value"].gt(0).sum()
        )
        team_m["history_attach_rate"] = float(
            rates["history_games"].ge(HISTORY_MIN).mean()
        )
        team_m["league_prior_available_rate"] = float(
            rates["p_league"].notna().mean()
        )
        team_m["parent_denominator_teamgame_coverage"] = float(room_cov)
        team_m["topology_source_coverage"] = float(topology_cov)
        player_m["players"] = int(player_scored["player_id"].nunique())

        support_gates = {
            "team_games_ge_400": team_m["rows"] >= 400,
            "teams_ge_30": team_m["teams"] >= 30,
            "positive_high_value_teamgames_ge_300":
                team_m["positive_high_value_teamgames"] >= 300,
            "player_games_floor":
                player_m["rows"] >= int(spec["min_player_games"]),
            "players_floor":
                player_m["players"] >= int(spec["min_players"]),
            "history_attach_ge_95":
                team_m["history_attach_rate"] >= 0.95,
            "league_prior_100":
                team_m["league_prior_available_rate"] >= 1.0 - TOL,
            "parent_denominator_teamgame_coverage_ge_99":
                team_m["parent_denominator_teamgame_coverage"] >= 0.99,
            "topology_source_coverage_ge_99":
                team_m["topology_source_coverage"] >= 0.99,
        }
        support_ok = support_ok and all(support_gates.values())
        by_season[str(season)] = {
            "team": team_m,
            "player": player_m,
            "support_gates": support_gates,
        }

    team_all = pd.concat(team_parts, ignore_index=True)
    player_all = pd.concat(player_parts, ignore_index=True)
    conf_team = team_all.loc[team_all["season"].eq(2025)].copy()
    conf_player = player_all.loc[player_all["season"].eq(2025)].copy()

    seed = int(spec["seed"])
    team_team_boot = cluster_bootstrap(conf_team, "crps_gain", "team", seed)
    team_game_boot = cluster_bootstrap(conf_team, "crps_gain", "game_id", seed + 1)
    player_player_boot = cluster_bootstrap(
        conf_player, "crps_gain", "player_id", seed + 2
    )
    player_teamgame_boot = cluster_bootstrap(
        conf_player, "crps_gain", "team_game_cluster", seed + 3
    )
    equal_boot, hetero = equal_player_bootstrap(conf_player, seed + 4)

    d24 = by_season["2024"]
    d25 = by_season["2025"]
    gates = {
        "source_integrity": bool(source_integrity),
        "support_2024": bool(all(d24["support_gates"].values())),
        "support_2025": bool(all(d25["support_gates"].values())),
        "2024_team_crps_gain_positive": d24["team"]["crps_gain"] > 0,
        "2024_team_ae_gain_nonnegative":
            d24["team"]["expected_count_ae_gain"] >= 0,
        "2024_player_crps_gain_positive": d24["player"]["crps_gain"] > 0,
        "2024_player_ae_gain_nonnegative":
            d24["player"]["expected_count_ae_gain"] >= 0,
        "2025_team_crps_gain_positive": d25["team"]["crps_gain"] > 0,
        "2025_team_ae_gain_nonnegative":
            d25["team"]["expected_count_ae_gain"] >= 0,
        "2025_player_crps_gain_positive": d25["player"]["crps_gain"] > 0,
        "2025_player_ae_gain_nonnegative":
            d25["player"]["expected_count_ae_gain"] >= 0,
        "2025_team_cluster_ci_low_positive":
            team_team_boot["ci_low"] > 0,
        "2025_game_cluster_ci_low_positive":
            team_game_boot["ci_low"] > 0,
        "2025_player_cluster_ci_low_positive":
            player_player_boot["ci_low"] > 0,
        "2025_teamgame_cluster_player_ci_low_positive":
            player_teamgame_boot["ci_low"] > 0,
        "2025_equal_player_ci_low_positive":
            equal_boot["ci_low"] > 0,
        "2025_median_player_crps_gain_positive":
            bool(
                np.isfinite(hetero["median_player_crps_gain"])
                and hetero["median_player_crps_gain"] > 0
            ),
        "sportsbook_zero": True,
        "oddsapi_zero": True,
        "outcomes_2026_zero": True,
        "fitted_coefficients_zero": True,
        "production_unchanged": True,
    }

    disposition = (
        spec["qualified"]
        if all(gates.values())
        else f"{cell}_PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED"
    )

    result = {
        "cell": cell,
        "disposition": disposition,
        "by_season": by_season,
        "confirmation_2025_inference": {
            "team_cluster": team_team_boot,
            "game_cluster": team_game_boot,
            "player_cluster": player_player_boot,
            "player_teamgame_cluster": player_teamgame_boot,
            "equal_player": equal_boot,
        },
        "confirmation_2025_player_heterogeneity": hetero,
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not bool(v)],
    }
    return result, team_all, player_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teamgames", type=Path, required=True)
    ap.add_argument("--player-topology", type=Path, required=True)
    ap.add_argument("--team-source-result", type=Path, required=True)
    ap.add_argument("--stage2-detail", type=Path, required=True)
    ap.add_argument("--stage2-result", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        team_source_result = read_json(args.team_source_result, "team source result")
        stage2_result = read_json(args.stage2_result, "Stage-2 source result")
        if (
            team_source_result.get("disposition")
            != "PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY"
        ):
            raise RuntimeError("raw team high-value parent disposition drift")
        if (
            stage2_result.get("disposition")
            != "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL"
        ):
            raise RuntimeError("Stage-2 parent disposition drift")
        if sorted(stage2_result.get("ready_cells", [])) != sorted(
            ["QB_PASSING", "RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING"]
        ):
            raise RuntimeError("Stage-2 ready-cell drift")

        teamgames = prepare_teamgames(read_csv(args.teamgames, "teamgames"))
        topology = read_csv(args.player_topology, "player topology")
        stage2 = read_csv(args.stage2_detail, "Stage-2 detail")
        if int(num(stage2["season"]).max()) > 2025:
            raise RuntimeError("2026 outcome leak in Stage-2 detail")

        cell_results = {}
        team_parts = []
        player_parts = []
        qualified = []
        for cell, spec in CELL_MAP.items():
            result, team_rows, player_rows = evaluate_cell(
                cell=cell,
                spec=spec,
                teamgames=teamgames,
                topology=topology,
                stage2=stage2,
            )
            cell_results[cell] = result
            team_parts.append(team_rows)
            player_parts.append(player_rows)
            qualified.append(result["disposition"] == spec["qualified"])

        if all(qualified):
            overall = "PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_ALL"
        elif any(qualified):
            overall = "PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_PARTIAL"
        else:
            overall = "PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED"

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": overall,
            "target_seasons": list(TARGET_SEASONS),
            "target_weeks": [min(TARGET_WEEKS), max(TARGET_WEEKS)],
            "history_window": HISTORY_WINDOW,
            "history_min": HISTORY_MIN,
            "bootstrap_reps": BOOT,
            "cells": cell_results,
            "integrity": {
                "player_recent_high_value_share_used": False,
                "opponent_environment_predictor_used": False,
                "td_outcomes_used": 0,
                "sportsbook_inputs_used": 0,
                "oddsapi_calls": 0,
                "outcomes_2026_used": 0,
                "fitted_predictive_coefficients": 0,
                "threshold_searches": 0,
                "production_changed": False,
                "parent_denominator_formula_changed": False,
                "player_entitlement_formula_changed": False,
            },
        }

        pd.concat(team_parts, ignore_index=True).to_csv(
            args.out_dir / "prism_team_high_value_creation_team_rows.csv",
            index=False,
        )
        pd.concat(player_parts, ignore_index=True).to_csv(
            args.out_dir / "prism_team_high_value_creation_player_rows.csv",
            index=False,
        )
        (args.out_dir / "prism_team_high_value_creation_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_TEAM_HIGH_VALUE_CREATION_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "integrity": {
                "player_recent_high_value_share_used": False,
                "opponent_environment_predictor_used": False,
                "td_outcomes_used": 0,
                "sportsbook_inputs_used": 0,
                "oddsapi_calls": 0,
                "outcomes_2026_used": 0,
                "fitted_predictive_coefficients": 0,
                "threshold_searches": 0,
                "production_changed": False,
                "parent_denominator_formula_changed": False,
                "player_entitlement_formula_changed": False,
            },
        }
        (args.out_dir / "prism_team_high_value_creation_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
