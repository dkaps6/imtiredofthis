#!/usr/bin/env python3
"""Source-only audit for PRISM WR/TE player signed component bias V1.

No predictive relationship is scored. The script verifies that the immutable
historical component-decomposition source can support a later, separately
frozen named-player signed-bias experiment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "bec1e2becbe37fc68aab9da25bf28898ee3f41bf"
PARENT_RUN = 39061606501
PARENT_ARTIFACT = 12111278777
PARENT_DIGEST = "sha256:f64c6e7efde9deaf91b7bc19076693a5106cb632338c461acc4d2cfe2ae37ea9"
TOL = 1e-10

EXPECTED_SEASONS = {
    "WR": (2023, 2024),
    "TE": (2024, 2025),
}
GATES = {
    ("WR", 2023): {"rows": 700, "players": 120, "teams": 28},
    ("WR", 2024): {"rows": 1100, "players": 135, "teams": 28},
    ("TE", 2024): {"rows": 575, "players": 70, "teams": 28},
    ("TE", 2025): {"rows": 600, "players": 75, "teams": 28},
}
REQUIRED_SIGNED = (
    "prior8_opportunity_error_mean",
    "prior8_efficiency_error_mean",
    "current_opportunity_error",
    "current_efficiency_error",
)


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
        .astype(bool)
    )


def load_source(path: Path, position: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {position} source rows: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "player_clean_key",
        "prior_component_rows_available",
        "prior_component_min_week", "prior_component_max_week",
        "instability_state_source_eligible",
        "identity_gap",
        *REQUIRED_SIGNED,
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"{position} source missing columns: {sorted(missing)}")

    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns in {position}: {forbidden}")

    x["position"] = position
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["player_clean_key"] = (
        x["player_clean_key"].astype("string").fillna("").str.strip()
    )
    x["prior_component_rows_available"] = num(
        x["prior_component_rows_available"]
    ).astype("Int64")
    x["prior_component_min_week"] = num(x["prior_component_min_week"])
    x["prior_component_max_week"] = num(x["prior_component_max_week"])
    x["identity_gap"] = num(x["identity_gap"])
    x["source_eligible"] = bool_series(x["instability_state_source_eligible"])
    for c in REQUIRED_SIGNED:
        x[c] = num(x[c])

    allowed = set(EXPECTED_SEASONS[position])
    x = x.loc[x["season"].isin(allowed)].copy()
    if x.empty:
        raise RuntimeError(f"zero {position} source rows after season filter")
    if x["player_clean_key"].eq("").any() or x["team"].eq("").any():
        raise RuntimeError(f"blank {position} identity")
    if x.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError(f"duplicate {position} player/week/team identity")
    return x


def audit_slice(x: pd.DataFrame, position: str, season: int) -> dict:
    g = x.loc[
        x["season"].eq(season) & x["source_eligible"].eq(True)
    ].copy()
    req = GATES[(position, season)]
    if g.empty:
        raise RuntimeError(f"zero eligible rows for {position} {season}")

    rows_by_prior = (
        g["prior_component_rows_available"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    chronology_bad = int(
        (
            g["prior_component_rows_available"].lt(4)
            | g["prior_component_rows_available"].gt(8)
            | g["prior_component_min_week"].ge(g["week"])
            | g["prior_component_max_week"].ge(g["week"])
        ).sum()
    )
    missing = {c: int(g[c].isna().sum()) for c in REQUIRED_SIGNED}
    finite = {
        c: bool(np.isfinite(g[c].to_numpy(float)).all())
        for c in REQUIRED_SIGNED
    }
    max_gap = float(g["identity_gap"].abs().max())
    gates = {
        "rows": int(len(g)) >= req["rows"],
        "players": int(g["player_clean_key"].nunique()) >= req["players"],
        "teams": int(g["team"].nunique()) >= req["teams"],
        "prior_signed_opportunity_complete": missing["prior8_opportunity_error_mean"] == 0
            and finite["prior8_opportunity_error_mean"],
        "prior_signed_efficiency_complete": missing["prior8_efficiency_error_mean"] == 0
            and finite["prior8_efficiency_error_mean"],
        "current_signed_opportunity_complete": missing["current_opportunity_error"] == 0
            and finite["current_opportunity_error"],
        "current_signed_efficiency_complete": missing["current_efficiency_error"] == 0
            and finite["current_efficiency_error"],
        "decomposition_identity": max_gap <= TOL,
        "chronology": chronology_bad == 0,
    }

    return {
        "position": position,
        "season": season,
        "eligible_rows": int(len(g)),
        "distinct_players": int(g["player_clean_key"].nunique()),
        "teams": int(g["team"].nunique()),
        "prior_component_rows_median": float(g["prior_component_rows_available"].median()),
        "prior_component_rows_p25": float(g["prior_component_rows_available"].quantile(0.25)),
        "prior_component_rows_p75": float(g["prior_component_rows_available"].quantile(0.75)),
        "rows_by_prior_component_count": {str(int(k)): int(v) for k, v in rows_by_prior.items()},
        "signed_field_missingness": missing,
        "chronology_violations": chronology_bad,
        "max_decomposition_identity_gap": max_gap,
        "source_gates": gates,
        "source_ready": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-source-rows", required=True, type=Path)
    ap.add_argument("--te-source-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        wr = load_source(args.wr_source_rows, "WR")
        te = load_source(args.te_source_rows, "TE")
        sources = {"WR": wr, "TE": te}

        summaries: dict[str, dict] = {}
        dispositions: dict[str, str] = {}
        combined = []
        overall_ready = True

        for position, x in sources.items():
            summaries[position] = {}
            position_ready = True
            for season in EXPECTED_SEASONS[position]:
                rec = audit_slice(x, position, season)
                summaries[position][str(season)] = rec
                position_ready = position_ready and rec["source_ready"]
            dispositions[position] = (
                f"{position}_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_READY"
                if position_ready
                else f"{position}_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_BLOCKED"
            )
            overall_ready = overall_ready and position_ready

            keep = x.loc[x["source_eligible"].eq(True)].copy()
            cols = [
                "position", "season", "week", "team", "player_clean_key",
                "prior_component_rows_available",
                "prior_component_min_week", "prior_component_max_week",
                "prior8_opportunity_error_mean",
                "prior8_efficiency_error_mean",
                "current_opportunity_error",
                "current_efficiency_error",
                "identity_gap",
            ]
            if position == "WR" and "wr_rank" in keep.columns:
                cols.insert(5, "wr_rank")
            combined.append(keep[cols])

        combined_rows = pd.concat(combined, ignore_index=True, sort=False)
        combined_rows.to_csv(
            args.out_dir / "wr_te_player_signed_component_bias_source_rows.csv",
            index=False,
        )

        flat = []
        for position in ("WR", "TE"):
            for season in EXPECTED_SEASONS[position]:
                rec = summaries[position][str(season)]
                row = {
                    k: v
                    for k, v in rec.items()
                    if k not in {
                        "rows_by_prior_component_count",
                        "signed_field_missingness",
                        "source_gates",
                    }
                }
                for k, v in rec["rows_by_prior_component_count"].items():
                    row[f"prior_count_{k}"] = v
                for k, v in rec["signed_field_missingness"].items():
                    row[f"missing_{k}"] = v
                for k, v in rec["source_gates"].items():
                    row[f"gate_{k}"] = v
                flat.append(row)
        pd.DataFrame(flat).to_csv(
            args.out_dir / "wr_te_player_signed_component_bias_source_support.csv",
            index=False,
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "positions": ["WR", "TE"],
            "components": ["OPPORTUNITY", "EFFICIENCY"],
            "summaries": summaries,
            "position_dispositions": dispositions,
            "disposition": (
                "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_READY_ALL"
                if overall_ready
                else "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_BLOCKED"
            ),
            "predictive_relationships_scored": 0,
            "models_fit": 0,
            "threshold_searches": 0,
            "component_means_changed": 0,
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
            "parent_run": PARENT_RUN,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "disposition": "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_INTEGRITY_FAILURE",
            "error": str(exc),
            "predictive_relationships_scored": 0,
            "models_fit": 0,
            "threshold_searches": 0,
            "component_means_changed": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        (args.out_dir / "wr_te_player_signed_component_bias_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "wr_te_player_signed_component_bias_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
