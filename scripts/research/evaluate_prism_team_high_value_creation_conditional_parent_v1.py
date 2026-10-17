#!/usr/bin/env python3
"""Score PRISM Team High-Value Opportunity Creation Conditional Parent V1.

Frozen contract:
docs/research/PRISM_TEAM_HIGH_VALUE_CREATION_CONDITIONAL_PARENT_V1_FREEZE.md

The only candidate change is TEAM high-value-opportunity creation state.
The canonical PRISM parent opportunity mean and qualified Stage-1/Stage-2
player allocation law are identical between baseline and candidate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from scripts._opponent_map import canon_team
from scripts.research.audit_player_active_role_conditional_allocation_source_v1 import (
    prepare_stage1,
)
from scripts.research.evaluate_player_active_role_conditional_allocation_bridge_v1 import (
    enumerate_active_states,
)

VERSION = "PRISM_TEAM_HIGH_VALUE_CREATION_CONDITIONAL_PARENT_V1"
FREEZE_COMMIT = "b15573fb7e5e0662a2d6b4a894cc2a13814c3df5"

RAW_RUN = 38991605252
RAW_ARTIFACT = 12085367606
RAW_DIGEST = "sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90"
STAGE2_SOURCE_RUN = 38991915503
STAGE2_SOURCE_ARTIFACT = 12085194208
STAGE2_SOURCE_DIGEST = "sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569"
STAGE2_BRIDGE_RUN = 38957575645
STAGE2_BRIDGE_ARTIFACT = 12070774378
STAGE2_BRIDGE_DIGEST = "sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0"
RECON_RUN = 38991737241
RECON_ARTIFACT = 12085278127
RECON_DIGEST = "sha256:0be3d571129876ce0a8df755762215e147faec2ff1ac1822e05cbd84508a12bb"

SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
HISTORY_WINDOW = 4
HISTORY_MIN = 3
BOOT_REPS = 5000
BOOT_SEED = 2026101701
TOL = 1e-12
TEAM_MIN_ROWS = 400
TEAM_MIN_POSITIVE = 300

CELL_MAP = {
    "RB_RUSH_INSIDE10": {
        "stage2_cell": "RB_RUSHING",
        "family": "RB",
        "source_flag": "rush_source_present",
        "team_hv_col": "inside10_rush",
        "team_parent_col": "team_rush_attempts",
        "player_channel": "RB_INSIDE10_RUSH",
        "min_player_rows": 700,
        "min_players": 80,
    },
    "RB_RECEIVING_REDZONE": {
        "stage2_cell": "RB_RECEIVING",
        "family": "RB",
        "source_flag": "target_source_present",
        "team_hv_col": "red_zone_targets",
        "team_parent_col": "team_targets",
        "player_channel": "RED_ZONE_TARGET",
        "min_player_rows": 700,
        "min_players": 80,
    },
    "TE_RECEIVING_REDZONE": {
        "stage2_cell": "TE_RECEIVING",
        "family": "TE",
        "source_flag": "target_source_present",
        "team_hv_col": "red_zone_targets",
        "team_parent_col": "team_targets",
        "player_channel": "RED_ZONE_TARGET",
        "min_player_rows": 700,
        "min_players": 75,
    },
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bools(s: Any) -> pd.Series:
    if isinstance(s, pd.Series):
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        return s.astype("string").fillna("").str.strip().str.lower().isin(
            {"true", "1", "yes", "y"}
        )
    return pd.Series(dtype=bool)


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    forbidden_tokens = (
        "sportsbook", "bookmaker", "prop_line", "market_line",
        "over_odds", "under_odds", "moneyline", "closing_line",
        "no_vig", "implied_prob",
    )
    forbidden = [c for c in x.columns if any(t in c for t in forbidden_tokens)]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns in {label}: {forbidden}")
    return x


def read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_parents(raw_result: dict, stage2_result: dict, parity: dict, recon: dict) -> None:
    if raw_result.get("disposition") != "PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY":
        raise RuntimeError(f"raw source disposition drift: {raw_result.get('disposition')}")
    if stage2_result.get("disposition") != "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL":
        raise RuntimeError(f"Stage-2 source disposition drift: {stage2_result.get('disposition')}")
    ready = set(stage2_result.get("ready_cells") or [])
    required = {v["stage2_cell"] for v in CELL_MAP.values()}
    if not required.issubset(ready):
        raise RuntimeError(f"required Stage-2 source cells missing: {sorted(required-ready)}")
    if parity.get("disposition") != "PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS":
        raise RuntimeError(f"parent denominator parity drift: {parity.get('disposition')}")
    if not bool(parity.get("key_sets_exact")):
        raise RuntimeError("parent denominator key parity not exact")
    if not bool(parity.get("denominator_finite_flags_exact")):
        raise RuntimeError("parent denominator finite-flag parity not exact")
    if not bool(parity.get("scientific_disposition_exact")):
        raise RuntimeError("parent denominator scientific disposition drift")
    if not bool(parity.get("ready_cells_exact")):
        raise RuntimeError("parent denominator ready-cell drift")
    if recon.get("disposition") != "PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL":
        raise RuntimeError(f"source reconciliation drift: {recon.get('disposition')}")
    for parent in (raw_result, stage2_result, parity, recon):
        for key in (
            "sportsbook_inputs_used", "oddsapi_calls", "outcomes_2026_read",
            "outcomes_2026_used",
        ):
            if key in parent and int(parent.get(key, 0) or 0) != 0:
                raise RuntimeError(f"parent integrity drift {key}={parent.get(key)}")
        if bool(parent.get("production_changed", False)):
            raise RuntimeError("parent production mutation drift")


def prepare_raw_teamgames(teamgames: pd.DataFrame) -> pd.DataFrame:
    need = {
        "season", "week", "game_id", "team", "opponent",
        "rush_source_present", "target_source_present",
        "team_rush_attempts", "inside10_rush",
        "team_targets", "red_zone_targets",
    }
    if need - set(teamgames.columns):
        raise RuntimeError(f"raw teamgames missing {sorted(need-set(teamgames.columns))}")
    x = teamgames.copy()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x = x.loc[x["season"].isin(SEASONS) & x["week"].between(1, 18)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["rush_source_present"] = bools(x["rush_source_present"])
    x["target_source_present"] = bools(x["target_source_present"])
    for c in ("team_rush_attempts", "inside10_rush", "team_targets", "red_zone_targets"):
        x[c] = num(x[c])
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate raw team-game identity")
    if (x.loc[x["rush_source_present"], "inside10_rush"] >
            x.loc[x["rush_source_present"], "team_rush_attempts"] + TOL).any():
        raise RuntimeError("inside10 rush exceeds parent rushing opportunity")
    if (x.loc[x["target_source_present"], "red_zone_targets"] >
            x.loc[x["target_source_present"], "team_targets"] + TOL).any():
        raise RuntimeError("red-zone targets exceed parent target opportunity")
    game = x["game_id"].astype("string").fillna("").str.strip()
    fallback = [
        f"{s}_{w:02d}_{'_'.join(sorted([str(t),str(o)]))}"
        for s, w, t, o in zip(x["season"], x["week"], x["team"], x["opponent"])
    ]
    x["game_key"] = np.where(game.ne(""), game.astype(str), fallback)
    return x.sort_values(["season", "week", "team"]).reset_index(drop=True)


def build_team_rate_rows(teamgames: pd.DataFrame, cell: str) -> pd.DataFrame:
    spec = CELL_MAP[cell]
    flag = spec["source_flag"]
    hv = spec["team_hv_col"]
    parent = spec["team_parent_col"]
    rows: list[dict] = []
    for r in teamgames.loc[
        teamgames["season"].isin(SEASONS)
        & teamgames["week"].isin(TARGET_WEEKS)
        & teamgames[flag]
    ].itertuples(index=False):
        season = int(r.season)
        week = int(r.week)
        team = str(r.team)
        source = teamgames.loc[
            teamgames["season"].eq(season)
            & teamgames[flag]
            & teamgames["week"].lt(week)
        ].copy()
        team_hist = source.loc[source["team"].eq(team)].sort_values("week").tail(HISTORY_WINDOW)
        if len(team_hist) < HISTORY_MIN:
            continue

        team_parent = float(num(team_hist[parent]).sum())
        team_hv = float(num(team_hist[hv]).sum())
        league_parent = float(num(source[parent]).sum())
        league_hv = float(num(source[hv]).sum())
        if not np.isfinite(team_parent) or not np.isfinite(league_parent):
            continue
        if team_parent <= 0 or league_parent <= 0:
            continue

        r_team = team_hv / team_parent
        r_league = league_hv / league_parent
        if not (np.isfinite(r_team) and np.isfinite(r_league)):
            continue
        if r_team < -TOL or r_team > 1.0 + TOL or r_league < -TOL or r_league > 1.0 + TOL:
            raise RuntimeError(f"high-value rate outside [0,1] {season} W{week} {team} {cell}")

        target_hv = float(getattr(r, hv))
        target_parent = float(getattr(r, parent))
        if not np.isfinite(target_hv) or not np.isfinite(target_parent):
            raise RuntimeError(f"non-finite target finite-pool label {season} W{week} {team} {cell}")
        if target_hv < -TOL or target_parent < -TOL or target_hv > target_parent + TOL:
            raise RuntimeError(f"target high-value finite-pool contradiction {season} W{week} {team} {cell}")
        if abs(target_hv - round(target_hv)) > TOL:
            raise RuntimeError("non-integer target high-value team count")

        rows.append({
            "season": season,
            "week": week,
            "game_key": str(r.game_key),
            "team": team,
            "opponent": str(r.opponent),
            "quality_cell": cell,
            "target_team_high_value": int(round(target_hv)),
            "target_team_parent_observed": float(target_parent),
            "prior_team_games": int(len(team_hist)),
            "team_history_min_week": int(team_hist["week"].min()),
            "team_history_max_week": int(team_hist["week"].max()),
            "league_history_max_week": int(source["week"].max()) if len(source) else 0,
            "team_prior_high_value_sum": team_hv,
            "team_prior_parent_sum": team_parent,
            "league_prior_high_value_sum": league_hv,
            "league_prior_parent_sum": league_parent,
            "team_high_value_rate": float(r_team),
            "league_high_value_rate": float(r_league),
            "chronology_violation": int(
                int(team_hist["week"].max()) >= week
                or int(source["week"].max()) >= week
            ),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"zero scoreable team-rate rows cell={cell}")
    if out.duplicated(["season", "week", "team", "quality_cell"]).any():
        raise RuntimeError(f"duplicate team-rate rows cell={cell}")
    return out


def prepare_stage2_expected_shares(
    detail: pd.DataFrame,
    stage1_universe: pd.DataFrame,
    stage1_mechanism: pd.DataFrame,
) -> pd.DataFrame:
    need = {
        "season", "week", "team", "cell", "family", "player_id",
        "player_name_key", "authority_q", "authority_denominator_mean",
        "authority_covered", "denominator_finite", "ambiguous_identity",
    }
    if need - set(detail.columns):
        raise RuntimeError(f"Stage-2 detail missing {sorted(need-set(detail.columns))}")

    d = detail.copy()
    d["season"] = num(d["season"]).astype("Int64")
    d["week"] = num(d["week"]).astype("Int64")
    d = d.loc[d["season"].isin(SEASONS) & d["week"].isin(TARGET_WEEKS)].copy()
    d["season"] = d["season"].astype(int)
    d["week"] = d["week"].astype(int)
    d["team"] = d["team"].map(canon_team)
    d["player_id"] = d["player_id"].astype("string").fillna("").str.strip()
    d["authority_covered"] = bools(d["authority_covered"])
    d["denominator_finite"] = bools(d["denominator_finite"])
    d["ambiguous_identity"] = bools(d["ambiguous_identity"])
    d["authority_q"] = num(d["authority_q"])
    d["authority_denominator_mean"] = num(d["authority_denominator_mean"])

    required_cells = {v["stage2_cell"] for v in CELL_MAP.values()}
    d = d.loc[
        d["cell"].isin(required_cells)
        & d["authority_covered"]
        & d["denominator_finite"]
        & ~d["ambiguous_identity"]
    ].copy()
    if d.duplicated(["season", "week", "team", "cell", "player_id"]).any():
        raise RuntimeError("duplicate Stage-2 source player identity")

    _, pstate = prepare_stage1(stage1_universe, stage1_mechanism)
    pstate["season"] = num(pstate["season"]).astype(int)
    pstate["week"] = num(pstate["week"]).astype(int)
    pstate["team"] = pstate["team"].map(canon_team)
    pstate["player_id"] = pstate["player_id"].astype("string").fillna("").str.strip()

    d = d.merge(
        pstate[
            [
                "season", "week", "team", "family", "player_id",
                "stage1_p_active", "stage1_named_player_supported",
            ]
        ],
        on=["season", "week", "team", "family", "player_id"],
        how="left",
        validate="many_to_one",
    )
    if d["stage1_p_active"].isna().any():
        raise RuntimeError(f"Stage-1 p_active attach misses={int(d['stage1_p_active'].isna().sum())}")

    out_parts = []
    for keys, g in d.groupby(["season", "week", "team", "cell"], sort=True):
        g = g.copy().sort_values("player_id").reset_index(drop=True)
        den = num(g["authority_denominator_mean"]).to_numpy(float)
        if not np.isfinite(den).all() or (den <= 0).any():
            raise RuntimeError(f"non-finite canonical parent denominator room={keys}")
        if float(np.nanmax(den) - np.nanmin(den)) > 1e-9:
            raise RuntimeError(f"canonical parent denominator not unique room={keys}")
        q_all = num(g["authority_q"]).to_numpy(float)
        if not np.isfinite(q_all).all() or (q_all < -TOL).any():
            raise RuntimeError(f"invalid authority_q room={keys}")
        if float(q_all.sum()) > 1.0 + TOL:
            raise RuntimeError(f"authority_q mass exceeds one room={keys} sum={q_all.sum()}")

        expected = np.zeros(len(g), dtype=float)
        pos = q_all > TOL
        if pos.any():
            p = num(g.loc[pos, "stage1_p_active"]).to_numpy(float)
            if not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
                raise RuntimeError(f"invalid Stage-1 p_active room={keys}")
            weights, state_named, _ = enumerate_active_states(q_all[pos], p)
            expected[pos] = weights @ state_named
        if not np.isfinite(expected).all() or (expected < -TOL).any():
            raise RuntimeError(f"invalid Stage-2 expected allocation room={keys}")
        if float(expected.sum()) > 1.0 + TOL:
            raise RuntimeError(f"Stage-2 expected named mass exceeds one room={keys}")

        g["stage2_expected_share"] = expected
        g["canonical_parent_mean"] = float(den[0])
        g["baseline_stage2_expected_share"] = expected
        g["candidate_stage2_expected_share"] = expected
        g["allocation_share_drift"] = (
            g["candidate_stage2_expected_share"] - g["baseline_stage2_expected_share"]
        ).abs()
        out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)
    if float(out["allocation_share_drift"].max()) > TOL:
        raise RuntimeError("Stage-2 allocation changed between baseline and candidate")
    return out


def prepare_player_labels(topology: pd.DataFrame) -> pd.DataFrame:
    need = {
        "season", "week", "team", "player_id", "family",
        "channel", "high_value_opportunities",
    }
    if need - set(topology.columns):
        raise RuntimeError(f"raw player topology missing {sorted(need-set(topology.columns))}")
    x = topology.copy()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x = x.loc[x["season"].isin(SEASONS) & x["week"].isin(TARGET_WEEKS)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].astype("string").fillna("").str.strip()
    x["high_value_opportunities"] = num(x["high_value_opportunities"])
    if (x["high_value_opportunities"] < -TOL).any():
        raise RuntimeError("negative player high-value topology count")
    return x


def poisson_crps(mu: float, y: int) -> float:
    if not np.isfinite(mu) or mu < 0:
        raise RuntimeError(f"invalid Poisson mean {mu}")
    if int(y) < 0:
        raise RuntimeError(f"invalid Poisson outcome {y}")
    if mu <= TOL:
        return float(y)
    q = poisson.ppf(1.0 - 1e-12, mu)
    upper = int(max(int(y) + 12, (int(q) if np.isfinite(q) else int(mu + 12 * math.sqrt(mu + 1))) + 4))
    k = np.arange(upper + 1, dtype=int)
    cdf = poisson.cdf(k, mu)
    obs = (k >= int(y)).astype(float)
    return float(np.square(cdf - obs).sum())


def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str, seed: int) -> dict:
    z = df[[value_col, cluster_col]].dropna().copy()
    g = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(g) < 2:
        return {
            "clusters": int(len(g)), "reps": BOOT_REPS, "valid_reps": 0,
            "observed": float(z[value_col].mean()) if len(z) else math.nan,
            "ci95_low": math.nan, "ci95_high": math.nan,
        }
    a = g[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    n = len(a)
    for i in range(BOOT_REPS):
        ix = rng.integers(0, n, size=n)
        total = a[ix].sum(axis=0)
        vals[i] = total[0] / total[1] if total[1] > 0 else np.nan
    vals = vals[np.isfinite(vals)]
    return {
        "clusters": int(n), "reps": BOOT_REPS, "valid_reps": int(len(vals)),
        "observed": float(z[value_col].mean()) if len(z) else math.nan,
        "ci95_low": float(np.quantile(vals, 0.025)) if len(vals) else math.nan,
        "ci95_high": float(np.quantile(vals, 0.975)) if len(vals) else math.nan,
    }


def equal_player_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("player_id", sort=True):
        if len(g) < 4:
            continue
        rows.append({
            "player_id": str(pid),
            "rows": int(len(g)),
            "mean_crps_gain": float(g["crps_gain"].mean()),
            "mean_ae_gain": float(g["ae_gain"].mean()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(player_df: pd.DataFrame, seed: int) -> dict:
    vals = num(player_df["mean_crps_gain"]).dropna().to_numpy(float)
    if len(vals) < 2:
        return {
            "players": int(len(vals)), "reps": BOOT_REPS, "valid_reps": 0,
            "observed": float(vals.mean()) if len(vals) else math.nan,
            "ci95_low": math.nan, "ci95_high": math.nan,
        }
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        out[i] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    return {
        "players": int(len(vals)), "reps": BOOT_REPS, "valid_reps": BOOT_REPS,
        "observed": float(vals.mean()),
        "ci95_low": float(np.quantile(out, 0.025)),
        "ci95_high": float(np.quantile(out, 0.975)),
    }


def score_cell(
    *,
    cell: str,
    team_rate_rows: pd.DataFrame,
    stage2: pd.DataFrame,
    player_topology: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = CELL_MAP[cell]
    s2 = stage2.loc[stage2["cell"].eq(spec["stage2_cell"])].copy()
    team = team_rate_rows.merge(
        s2[
            ["season", "week", "team", "canonical_parent_mean"]
        ].drop_duplicates(["season", "week", "team"]),
        on=["season", "week", "team"],
        how="inner",
        validate="one_to_one",
    )
    if team.empty:
        raise RuntimeError(f"zero joined team score rows cell={cell}")

    team["baseline_parent_mean"] = num(team["canonical_parent_mean"])
    team["candidate_parent_mean"] = num(team["canonical_parent_mean"])
    team["parent_mean_drift"] = (
        team["candidate_parent_mean"] - team["baseline_parent_mean"]
    ).abs()
    if float(team["parent_mean_drift"].max()) > TOL:
        raise RuntimeError(f"parent denominator changed cell={cell}")

    team["baseline_mean"] = team["baseline_parent_mean"] * num(team["league_high_value_rate"])
    team["candidate_mean"] = team["candidate_parent_mean"] * num(team["team_high_value_rate"])
    team["baseline_crps"] = [
        poisson_crps(float(mu), int(y))
        for mu, y in zip(team["baseline_mean"], team["target_team_high_value"])
    ]
    team["candidate_crps"] = [
        poisson_crps(float(mu), int(y))
        for mu, y in zip(team["candidate_mean"], team["target_team_high_value"])
    ]
    team["crps_gain"] = team["baseline_crps"] - team["candidate_crps"]
    team["baseline_ae"] = (
        num(team["target_team_high_value"]) - num(team["baseline_mean"])
    ).abs()
    team["candidate_ae"] = (
        num(team["target_team_high_value"]) - num(team["candidate_mean"])
    ).abs()
    team["ae_gain"] = team["baseline_ae"] - team["candidate_ae"]

    labels = player_topology.loc[
        player_topology["channel"].astype(str).eq(spec["player_channel"])
        & player_topology["family"].astype(str).eq(spec["family"])
    ].copy()
    labels = (
        labels.groupby(["season", "week", "team", "player_id"], as_index=False)
        ["high_value_opportunities"].sum()
    )
    labels = labels.rename(columns={"high_value_opportunities": "actual_player_high_value"})

    p = s2.merge(
        team[
            [
                "season", "week", "team", "game_key", "target_team_high_value",
                "baseline_mean", "candidate_mean", "baseline_parent_mean",
                "candidate_parent_mean",
            ]
        ],
        on=["season", "week", "team"],
        how="inner",
        validate="many_to_one",
    )
    p = p.merge(
        labels,
        on=["season", "week", "team", "player_id"],
        how="left",
        validate="one_to_one",
    )
    p["actual_player_high_value"] = num(p["actual_player_high_value"]).fillna(0.0)
    if not np.allclose(
        p["actual_player_high_value"],
        np.rint(p["actual_player_high_value"]),
        atol=TOL,
    ):
        raise RuntimeError(f"non-integer player high-value count cell={cell}")
    if (p["actual_player_high_value"] < -TOL).any():
        raise RuntimeError(f"negative player high-value count cell={cell}")

    room_y = (
        p.groupby(["season", "week", "team"])["actual_player_high_value"]
        .sum()
        .reset_index(name="explicit_family_high_value")
    )
    chk = team.merge(room_y, on=["season", "week", "team"], how="left", validate="one_to_one")
    chk["explicit_family_high_value"] = num(chk["explicit_family_high_value"]).fillna(0.0)
    if (chk["explicit_family_high_value"] > chk["target_team_high_value"] + TOL).any():
        raise RuntimeError(f"player family count exceeds finite team pool cell={cell}")

    p["baseline_player_mean"] = (
        num(p["baseline_mean"]) * num(p["baseline_stage2_expected_share"])
    )
    p["candidate_player_mean"] = (
        num(p["candidate_mean"]) * num(p["candidate_stage2_expected_share"])
    )
    p["allocation_share_drift"] = (
        num(p["candidate_stage2_expected_share"])
        - num(p["baseline_stage2_expected_share"])
    ).abs()
    if float(p["allocation_share_drift"].max()) > TOL:
        raise RuntimeError(f"player allocation changed between baseline/candidate cell={cell}")

    y = np.rint(num(p["actual_player_high_value"])).astype(int)
    p["baseline_crps"] = [
        poisson_crps(float(mu), int(yy))
        for mu, yy in zip(p["baseline_player_mean"], y)
    ]
    p["candidate_crps"] = [
        poisson_crps(float(mu), int(yy))
        for mu, yy in zip(p["candidate_player_mean"], y)
    ]
    p["crps_gain"] = p["baseline_crps"] - p["candidate_crps"]
    p["baseline_ae"] = (y - num(p["baseline_player_mean"])).abs()
    p["candidate_ae"] = (y - num(p["candidate_player_mean"])).abs()
    p["ae_gain"] = p["baseline_ae"] - p["candidate_ae"]
    p["quality_cell"] = cell
    return team, p


def season_metrics(
    cell: str,
    season: int,
    team: pd.DataFrame,
    player: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    spec = CELL_MAP[cell]
    t = team.loc[team["season"].eq(season)].copy()
    p = player.loc[player["season"].eq(season)].copy()
    ep = equal_player_frame(p)
    idx = list(CELL_MAP).index(cell)
    seed = BOOT_SEED + idx * 10000 + season

    rec = {
        "cell": cell,
        "season": season,
        "team_games": int(len(t)),
        "positive_high_value_team_games": int((num(t["target_team_high_value"]) > 0).sum()),
        "player_games": int(len(p)),
        "players": int(p["player_id"].nunique()),
        "heterogeneity_players_ge4": int(len(ep)),
        "mean_league_high_value_rate": float(num(t["league_high_value_rate"]).mean()),
        "mean_team_high_value_rate": float(num(t["team_high_value_rate"]).mean()),
        "mean_parent_opportunity_mean": float(num(t["baseline_parent_mean"]).mean()),
        "team_baseline_mean": float(num(t["baseline_mean"]).mean()),
        "team_candidate_mean": float(num(t["candidate_mean"]).mean()),
        "team_baseline_crps": float(num(t["baseline_crps"]).mean()),
        "team_candidate_crps": float(num(t["candidate_crps"]).mean()),
        "team_mean_crps_gain": float(num(t["crps_gain"]).mean()),
        "team_baseline_ae": float(num(t["baseline_ae"]).mean()),
        "team_candidate_ae": float(num(t["candidate_ae"]).mean()),
        "team_mean_ae_gain": float(num(t["ae_gain"]).mean()),
        "player_baseline_crps": float(num(p["baseline_crps"]).mean()),
        "player_candidate_crps": float(num(p["candidate_crps"]).mean()),
        "player_mean_crps_gain": float(num(p["crps_gain"]).mean()),
        "player_baseline_ae": float(num(p["baseline_ae"]).mean()),
        "player_candidate_ae": float(num(p["candidate_ae"]).mean()),
        "player_mean_ae_gain": float(num(p["ae_gain"]).mean()),
        "share_player_rows_crps_improved": float((num(p["crps_gain"]) > 0).mean()),
        "max_parent_mean_drift": float(num(t["parent_mean_drift"]).max()),
        "max_allocation_share_drift": float(num(p["allocation_share_drift"]).max()),
        "chronology_violations": int(num(t["chronology_violation"]).sum()),
        "player_mean_per_player_crps_gain": float(num(ep["mean_crps_gain"]).mean()) if len(ep) else math.nan,
        "player_median_per_player_crps_gain": float(num(ep["mean_crps_gain"]).median()) if len(ep) else math.nan,
        "player_positive_supported_share": float((num(ep["mean_crps_gain"]) > 0).mean()) if len(ep) else math.nan,
    }
    support = {
        "team_games_ge_400": rec["team_games"] >= TEAM_MIN_ROWS,
        "positive_high_value_team_games_ge_300": rec["positive_high_value_team_games"] >= TEAM_MIN_POSITIVE,
        "player_games": rec["player_games"] >= int(spec["min_player_rows"]),
        "players": rec["players"] >= int(spec["min_players"]),
    }
    rec["support_gates"] = support

    if season == 2025:
        rec["team_cluster_bootstrap"] = cluster_bootstrap(
            t, "crps_gain", "team", seed + 1
        )
        rec["team_game_cluster_bootstrap"] = cluster_bootstrap(
            t, "crps_gain", "game_key", seed + 2
        )
        rec["player_cluster_bootstrap"] = cluster_bootstrap(
            p, "crps_gain", "player_id", seed + 3
        )
        rec["player_game_cluster_bootstrap"] = cluster_bootstrap(
            p, "crps_gain", "game_key", seed + 4
        )
        rec["equal_player_bootstrap"] = equal_player_bootstrap(ep, seed + 5)
    return rec, ep


def qualify(cell: str, m24: dict, m25: dict) -> tuple[dict, str]:
    gates = {
        "support_2024": all(m24["support_gates"].values()),
        "support_2025": all(m25["support_gates"].values()),
        "development_2024_team_crps_gt_0": m24["team_mean_crps_gain"] > 0,
        "development_2024_team_ae_nonworse": m24["team_mean_ae_gain"] >= -TOL,
        "development_2024_player_crps_gt_0": m24["player_mean_crps_gain"] > 0,
        "development_2024_player_ae_nonworse": m24["player_mean_ae_gain"] >= -TOL,
        "confirm_2025_team_crps_gt_0": m25["team_mean_crps_gain"] > 0,
        "confirm_2025_team_ae_nonworse": m25["team_mean_ae_gain"] >= -TOL,
        "confirm_2025_team_cluster_ci_low_gt_0": (
            m25["team_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirm_2025_game_cluster_team_ci_low_gt_0": (
            m25["team_game_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirm_2025_player_crps_gt_0": m25["player_mean_crps_gain"] > 0,
        "confirm_2025_player_ae_nonworse": m25["player_mean_ae_gain"] >= -TOL,
        "confirm_2025_player_cluster_ci_low_gt_0": (
            m25["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirm_2025_game_cluster_player_ci_low_gt_0": (
            m25["player_game_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirm_2025_equal_player_mean_gt_0": m25["player_mean_per_player_crps_gain"] > 0,
        "confirm_2025_equal_player_ci_low_gt_0": (
            m25["equal_player_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "parent_denominator_unchanged_2024": m24["max_parent_mean_drift"] <= TOL,
        "parent_denominator_unchanged_2025": m25["max_parent_mean_drift"] <= TOL,
        "stage2_allocation_unchanged_2024": m24["max_allocation_share_drift"] <= TOL,
        "stage2_allocation_unchanged_2025": m25["max_allocation_share_drift"] <= TOL,
        "chronology_clean_2024": m24["chronology_violations"] == 0,
        "chronology_clean_2025": m25["chronology_violations"] == 0,
    }
    disposition = (
        f"{cell}_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED"
        if all(gates.values())
        else f"{cell}_PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED"
    )
    return gates, disposition


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-teamgames", type=Path, required=True)
    ap.add_argument("--raw-player-topology", type=Path, required=True)
    ap.add_argument("--raw-result", type=Path, required=True)
    ap.add_argument("--stage2-detail", type=Path, required=True)
    ap.add_argument("--stage2-result", type=Path, required=True)
    ap.add_argument("--denominator-parity", type=Path, required=True)
    ap.add_argument("--reconciliation-result", type=Path, required=True)
    ap.add_argument("--stage1-universe", type=Path, required=True)
    ap.add_argument("--stage1-mechanism", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_result = read_json(args.raw_result, "raw source result")
        stage2_result = read_json(args.stage2_result, "Stage-2 source result")
        parity = read_json(args.denominator_parity, "denominator parity")
        recon = read_json(args.reconciliation_result, "source reconciliation")
        validate_parents(raw_result, stage2_result, parity, recon)

        raw_teamgames = prepare_raw_teamgames(
            read_csv(args.raw_teamgames, "raw opportunity-quality teamgames")
        )
        raw_topology = prepare_player_labels(
            read_csv(args.raw_player_topology, "raw opportunity-quality player topology")
        )
        stage2 = prepare_stage2_expected_shares(
            read_csv(args.stage2_detail, "Stage-2 source detail"),
            read_csv(args.stage1_universe, "Stage-1 universe"),
            read_csv(args.stage1_mechanism, "Stage-1 mechanism"),
        )

        team_parts = []
        player_parts = []
        player_diag_parts = []
        season_metrics_map: dict[str, dict] = {}
        qualification_gates: dict[str, dict] = {}
        dispositions: dict[str, str] = {}

        for cell in CELL_MAP:
            rates = build_team_rate_rows(raw_teamgames, cell)
            team_scored, player_scored = score_cell(
                cell=cell,
                team_rate_rows=rates,
                stage2=stage2,
                player_topology=raw_topology,
            )
            team_parts.append(team_scored)
            player_parts.append(player_scored)
            season_metrics_map[cell] = {}
            by_season = {}
            for season in SEASONS:
                metrics, pdiag = season_metrics(
                    cell, season, team_scored, player_scored
                )
                season_metrics_map[cell][str(season)] = metrics
                pdiag["quality_cell"] = cell
                pdiag["season"] = season
                player_diag_parts.append(pdiag)
                by_season[season] = metrics
            gates, disp = qualify(cell, by_season[2024], by_season[2025])
            qualification_gates[cell] = gates
            dispositions[cell] = disp

        n_qualified = sum(v.endswith("_QUALIFIED") for v in dispositions.values())
        if n_qualified == len(CELL_MAP):
            overall = "PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_ALL"
        elif n_qualified:
            overall = "PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_PARTIAL"
        else:
            overall = "PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED"

        team_all = pd.concat(team_parts, ignore_index=True)
        player_all = pd.concat(player_parts, ignore_index=True)
        player_diag = (
            pd.concat(player_diag_parts, ignore_index=True)
            if player_diag_parts else pd.DataFrame()
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": overall,
            "position_dispositions": dispositions,
            "qualification_gates": qualification_gates,
            "season_metrics": season_metrics_map,
            "frozen_parameters": {
                "development_season": 2024,
                "confirmation_season": 2025,
                "target_weeks": [5, 18],
                "history_window": HISTORY_WINDOW,
                "history_min": HISTORY_MIN,
                "baseline_rate": "STRICTLY_PRIOR_SAME_SEASON_LEAGUE_HV_OVER_PARENT",
                "candidate_rate": "LAST4_MIN3_SAME_TEAM_HV_OVER_PARENT",
                "team_count_law": "POISSON",
                "player_count_law": "POISSON",
                "player_allocation": "QUALIFIED_STAGE1_PLUS_STAGE2_EXACT_ACTIVE_SET_EXPECTATION",
                "bootstrap_replicates": BOOT_REPS,
                "bootstrap_seed": BOOT_SEED,
            },
            "parent_lineage": {
                "raw_source": {
                    "run": RAW_RUN, "artifact": RAW_ARTIFACT, "digest": RAW_DIGEST,
                },
                "stage2_source": {
                    "run": STAGE2_SOURCE_RUN,
                    "artifact": STAGE2_SOURCE_ARTIFACT,
                    "digest": STAGE2_SOURCE_DIGEST,
                },
                "stage2_bridge": {
                    "run": STAGE2_BRIDGE_RUN,
                    "artifact": STAGE2_BRIDGE_ARTIFACT,
                    "digest": STAGE2_BRIDGE_DIGEST,
                },
                "source_reconciliation": {
                    "run": RECON_RUN, "artifact": RECON_ARTIFACT, "digest": RECON_DIGEST,
                },
            },
            "target_game_outcomes_used_as_predictors": 0,
            "predictive_models_fit": 0,
            "fitted_coefficients": 0,
            "threshold_searches": 0,
            "player_high_value_share_candidate_used": False,
            "opponent_environment_used": False,
            "td_outcomes_used": False,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "production_changed": False,
        }

        team_all.to_csv(
            args.out_dir / "prism_team_high_value_creation_team_rows.csv", index=False
        )
        player_all.to_csv(
            args.out_dir / "prism_team_high_value_creation_player_rows.csv", index=False
        )
        player_diag.to_csv(
            args.out_dir / "prism_team_high_value_creation_player_diagnostics.csv",
            index=False,
        )
        (args.out_dir / "prism_team_high_value_creation_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_TEAM_HIGH_VALUE_CREATION_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "target_game_outcomes_used_as_predictors": 0,
            "predictive_models_fit": 0,
            "fitted_coefficients": 0,
            "threshold_searches": 0,
            "player_high_value_share_candidate_used": False,
            "opponent_environment_used": False,
            "td_outcomes_used": False,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "production_changed": False,
        }
        (args.out_dir / "prism_team_high_value_creation_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
