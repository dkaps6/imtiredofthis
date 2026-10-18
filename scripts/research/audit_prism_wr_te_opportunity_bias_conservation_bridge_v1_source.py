#!/usr/bin/env python3
"""Source/topology audit for PRISM WR/TE opportunity-bias conservation bridge V1.

No target or receiving-yard outcome is scored. This verifies that the qualified
signed opportunity-bias signal maps exactly onto complete promoted WR/TE room
authorities with enough multi-player support for a later zero-sum intervention.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "eb9698dfb44883e3bdd2ebf8706f26f63718e0ce"

SIGNAL_RUN = 39109423266
SIGNAL_ARTIFACT = 12128983909
SIGNAL_DIGEST = "sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490"

WR_RUN = 34238301577
WR_ARTIFACT = 10061328722
WR_DIGEST = "sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce"

TE_RUN = 34152797603
TE_ARTIFACT = 10029942404
TE_DIGEST = "sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb"

SUPPORT = {
    ("WR", 2023): {"rows": 250, "players": 50},
    ("WR", 2024): {"rows": 250, "players": 50},
    ("TE", 2024): {"rows": 250, "players": 50},
    ("TE", 2025): {"rows": 250, "players": 50},
}

SEASONS = {"WR": (2023, 2024), "TE": (2024, 2025)}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_signal_rows(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing signal rows: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "position", "component", "season", "week", "team", "player_clean_key",
        "source_eligible", "hierarchical_signed_bias",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"signal rows missing columns: {sorted(missing)}")
    x["position"] = x["position"].astype("string").fillna("").str.strip().str.upper()
    x["component"] = x["component"].astype("string").fillna("").str.strip().str.upper()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["player_clean_key"] = x["player_clean_key"].astype("string").fillna("").str.strip()
    x["hierarchical_signed_bias"] = num(x["hierarchical_signed_bias"])
    if x["source_eligible"].dtype != bool:
        x["source_eligible"] = (
            x["source_eligible"].astype("string").str.strip().str.lower()
            .map({"true": True, "false": False}).fillna(False).astype(bool)
        )
    x = x.loc[
        x["component"].eq("OPPORTUNITY")
        & x["position"].isin(["WR", "TE"])
        & x["source_eligible"].eq(True)
    ].copy()
    if x.empty:
        raise RuntimeError("zero qualified opportunity signal rows")
    if x.duplicated(["position", "season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate signal identity")
    if not np.isfinite(x["hierarchical_signed_bias"]).all():
        raise RuntimeError("nonfinite signal bias")
    return x


def load_signal_result(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing signal result: {path}")
    p = json.loads(path.read_text())
    if p.get("version") != "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SIGNAL":
        raise RuntimeError("wrong signal result version")
    expected = {
        "WR_OPPORTUNITY": "WR_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED",
        "TE_OPPORTUNITY": "TE_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED",
    }
    for cell, disp in expected.items():
        if p.get("cell_dispositions", {}).get(cell) != disp:
            raise RuntimeError(f"{cell} not qualified in signal authority")
    for cell in ("WR_EFFICIENCY", "TE_EFFICIENCY"):
        if not str(p.get("cell_dispositions", {}).get(cell, "")).endswith("_CLOSED"):
            raise RuntimeError(f"{cell} unexpectedly not closed")
    if p.get("component_means_changed") != 0 or p.get("projection_means_changed") != 0:
        raise RuntimeError("signal authority changed projection/component means")
    return p


def load_authority(path: Path, position: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {position} authority: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if position == "WR":
        target_col = "pred_targets"
        yards_col = "mc_rec_yards"
    else:
        target_col = "candidate_targets_r5p"
        yards_col = "candidate_rec_yards_r5p"
    required = {"season", "week", "team", "player_clean_key", target_col, yards_col}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"{position} authority missing columns: {sorted(missing)}")
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["player_clean_key"] = x["player_clean_key"].astype("string").fillna("").str.strip()
    x["pred_targets_bridge"] = num(x[target_col])
    x["pred_rec_yards_bridge"] = num(x[yards_col])
    x["position"] = position
    x = x.loc[x["season"].isin(SEASONS[position])].copy()
    if x.empty:
        raise RuntimeError(f"zero {position} authority rows after season filter")
    if x["team"].eq("").any() or x["player_clean_key"].eq("").any():
        raise RuntimeError(f"blank {position} authority identity")
    if x.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError(f"duplicate {position} authority identity")
    if not np.isfinite(x["pred_targets_bridge"]).all() or not np.isfinite(x["pred_rec_yards_bridge"]).all():
        raise RuntimeError(f"nonfinite {position} projected authority values")
    room_keys = ["position", "season", "week", "team"]
    room_sum = x.groupby(room_keys, sort=False)["pred_targets_bridge"].transform("sum")
    if (room_sum <= 0).any():
        raise RuntimeError(f"{position} nonpositive room target sum")
    x["room_target_sum"] = room_sum
    x["room_size"] = x.groupby(room_keys, sort=False)["player_clean_key"].transform("size")
    return x


def q(v: pd.Series, p: float) -> float:
    if len(v) == 0:
        return math.nan
    return float(num(v).quantile(p))


def summarize(position: str, season: int, authority: pd.DataFrame, signal: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    a = authority.loc[authority["season"].eq(season)].copy()
    s = signal.loc[
        signal["position"].eq(position) & signal["season"].eq(season)
    ].copy()
    keys = ["position", "season", "week", "team", "player_clean_key"]
    merged = s.merge(
        a[keys + [
            "pred_targets_bridge", "pred_rec_yards_bridge",
            "room_target_sum", "room_size",
        ]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    matched = merged["_merge"].eq("both")
    merge_coverage = float(matched.mean()) if len(merged) else 0.0
    m = merged.loc[matched].copy()
    m["pred_ypt"] = m["pred_rec_yards_bridge"] / m["pred_targets_bridge"]
    positive_targets = m["pred_targets_bridge"].gt(0) & np.isfinite(m["pred_targets_bridge"])
    positive_ypt = m["pred_ypt"].gt(0) & np.isfinite(m["pred_ypt"])
    m["bridge_topology_eligible"] = (
        positive_targets & positive_ypt & m["room_size"].ge(2)
    )
    eligible = m.loc[m["bridge_topology_eligible"]].copy()
    eligible["bias_factor"] = np.exp(
        -eligible["hierarchical_signed_bias"] / eligible["pred_rec_yards_bridge"]
    )
    if not np.isfinite(eligible["bias_factor"]).all():
        raise RuntimeError(f"{position} {season} nonfinite bias factor")

    room_sizes = (
        a[["position", "season", "week", "team", "room_size"]]
        .drop_duplicates(["position", "season", "week", "team"])
        ["room_size"]
    )
    req = SUPPORT[(position, season)]
    gates = {
        "rows": int(len(eligible)) >= req["rows"],
        "players": int(eligible["player_clean_key"].nunique()) >= req["players"],
        "join_coverage_100pct": bool(len(s) > 0 and abs(merge_coverage - 1.0) <= 1e-12),
        "matched_positive_targets": bool(len(m) > 0 and positive_targets.all()),
        "matched_positive_pred_ypt": bool(len(m) > 0 and positive_ypt.all()),
    }
    factors = eligible["bias_factor"]
    rec = {
        "position": position,
        "season": season,
        "authority_rows": int(len(a)),
        "authority_rooms": int(a[["week", "team"]].drop_duplicates().shape[0]),
        "room_size_p10": q(room_sizes, 0.10),
        "room_size_median": q(room_sizes, 0.50),
        "room_size_p90": q(room_sizes, 0.90),
        "room_size_max": float(room_sizes.max()) if len(room_sizes) else math.nan,
        "signal_supported_rows": int(len(s)),
        "exact_matched_rows": int(matched.sum()),
        "join_coverage": merge_coverage,
        "bridge_eligible_rows": int(len(eligible)),
        "bridge_eligible_players": int(eligible["player_clean_key"].nunique()),
        "single_player_room_supported_rows": int((m["room_size"] < 2).sum()),
        "nonpositive_pred_target_rows": int((~positive_targets).sum()),
        "nonpositive_pred_ypt_rows": int((~positive_ypt).sum()),
        "bias_factor_min": float(factors.min()) if len(factors) else math.nan,
        "bias_factor_p01": q(factors, 0.01),
        "bias_factor_p10": q(factors, 0.10),
        "bias_factor_median": q(factors, 0.50),
        "bias_factor_p90": q(factors, 0.90),
        "bias_factor_p99": q(factors, 0.99),
        "bias_factor_max": float(factors.max()) if len(factors) else math.nan,
        "bias_factor_finite_rows": int(np.isfinite(factors).sum()),
        "source_gates": gates,
        "source_ready": bool(all(gates.values())),
    }
    return rec, eligible


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-authority", required=True, type=Path)
    ap.add_argument("--te-authority", required=True, type=Path)
    ap.add_argument("--signal-rows", required=True, type=Path)
    ap.add_argument("--signal-result", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        load_signal_result(args.signal_result)
        signal = load_signal_rows(args.signal_rows)
        wr = load_authority(args.wr_authority, "WR")
        te = load_authority(args.te_authority, "TE")
        auth = {"WR": wr, "TE": te}

        summaries: dict[str, dict] = {}
        eligible_parts = []
        overall_ready = True
        for position in ("WR", "TE"):
            summaries[position] = {}
            for season in SEASONS[position]:
                rec, eligible = summarize(position, season, auth[position], signal)
                summaries[position][str(season)] = rec
                eligible_parts.append(eligible)
                overall_ready = overall_ready and rec["source_ready"]

        eligible_all = pd.concat(eligible_parts, ignore_index=True, sort=False)
        eligible_all.to_csv(
            args.out_dir / "wr_te_opportunity_bias_conservation_bridge_source_rows.csv",
            index=False,
        )
        flat = []
        for position in ("WR", "TE"):
            for season in SEASONS[position]:
                rec = summaries[position][str(season)]
                row = {k: v for k, v in rec.items() if k != "source_gates"}
                for k, v in rec["source_gates"].items():
                    row[f"gate_{k}"] = v
                flat.append(row)
        pd.DataFrame(flat).to_csv(
            args.out_dir / "wr_te_opportunity_bias_conservation_bridge_source_support.csv",
            index=False,
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "signal_run": SIGNAL_RUN,
            "signal_artifact": SIGNAL_ARTIFACT,
            "signal_digest": SIGNAL_DIGEST,
            "wr_parent_run": WR_RUN,
            "wr_parent_artifact": WR_ARTIFACT,
            "wr_parent_digest": WR_DIGEST,
            "te_parent_run": TE_RUN,
            "te_parent_artifact": TE_ARTIFACT,
            "te_parent_digest": TE_DIGEST,
            "qualified_signal_cells": ["WR_OPPORTUNITY", "TE_OPPORTUNITY"],
            "excluded_closed_cells": ["WR_EFFICIENCY", "TE_EFFICIENCY"],
            "support_floor_rows": 250,
            "support_floor_players": 50,
            "summaries": summaries,
            "disposition": (
                "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_READY_ALL"
                if overall_ready
                else "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_BLOCKED"
            ),
            "outcomes_scored": 0,
            "projection_actions_scored": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_INTEGRITY_FAILURE",
            "error": str(exc),
            "outcomes_scored": 0,
            "projection_actions_scored": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        (args.out_dir / "wr_te_opportunity_bias_conservation_bridge_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "wr_te_opportunity_bias_conservation_bridge_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
