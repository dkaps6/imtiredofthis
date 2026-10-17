#!/usr/bin/env python3
"""Score PRISM Team Scoring-Territory Drive Creation V1.

Frozen contract:
docs/research/PRISM_TEAM_SCORING_TERRITORY_DRIVE_CREATION_V1_FREEZE.md

This scorer reuses the certified PRISM parent denominator, Stage-1/Stage-2
player-allocation law, Poisson count law, proper scoring, and inference from
the first binding opportunity-quality V1. The only changed scientific object
is the strictly-prior TEAM scoring-territory drive-creation state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_team_high_value_creation_conditional_parent_v1 import (
    BOOT_REPS,
    BOOT_SEED,
    CELL_MAP as BASE_CELL_MAP,
    SEASONS,
    TARGET_WEEKS,
    TOL,
    bools,
    num,
    prepare_player_labels,
    prepare_raw_teamgames,
    prepare_stage2_expected_shares,
    read_csv,
    read_json,
    score_cell as score_fixed_allocation_cell,
    season_metrics as base_season_metrics,
    validate_parents,
)

VERSION = "PRISM_TEAM_SCORING_TERRITORY_DRIVE_CREATION_V1"
FREEZE_COMMIT = "82f74487e68729efdbcc0868968fd6ddccd4eef1"

HISTORY_WINDOW = 4
HISTORY_MIN = 3
TEAM_MIN_ROWS = 400
TEAM_MIN_POSITIVE = 300

CELL_MAP = {
    "RB_RUSH_INSIDE10": {
        **BASE_CELL_MAP["RB_RUSH_INSIDE10"],
        "drive_reach_col": "drives_reached10",
        "territory": "OPPONENT_10",
    },
    "RB_RECEIVING_REDZONE": {
        **BASE_CELL_MAP["RB_RECEIVING_REDZONE"],
        "drive_reach_col": "drives_reached20",
        "territory": "OPPONENT_20",
    },
    "TE_RECEIVING_REDZONE": {
        **BASE_CELL_MAP["TE_RECEIVING_REDZONE"],
        "drive_reach_col": "drives_reached20",
        "territory": "OPPONENT_20",
    },
}


def prepare_teamgames_with_drive(teamgames: pd.DataFrame) -> pd.DataFrame:
    need = {
        "drive_source_present",
        "offensive_drives",
        "drives_reached20",
        "drives_reached10",
    }
    if need - set(teamgames.columns):
        raise RuntimeError(
            f"raw teamgames missing drive topology columns {sorted(need-set(teamgames.columns))}"
        )
    x = prepare_raw_teamgames(teamgames)
    x["drive_source_present"] = bools(x["drive_source_present"])
    for c in ("offensive_drives", "drives_reached20", "drives_reached10"):
        x[c] = num(x[c])

    present = x["drive_source_present"]
    if x.loc[present, ["offensive_drives", "drives_reached20", "drives_reached10"]].isna().any(axis=None):
        raise RuntimeError("drive-source-present row has non-finite drive topology")
    if (x.loc[present, "offensive_drives"] <= 0).any():
        raise RuntimeError("drive-source-present row has non-positive offensive_drives")
    if (x.loc[present, "drives_reached20"] < -TOL).any() or (
        x.loc[present, "drives_reached10"] < -TOL
    ).any():
        raise RuntimeError("negative scoring-territory drive count")
    if (
        x.loc[present, "drives_reached20"]
        > x.loc[present, "offensive_drives"] + TOL
    ).any():
        raise RuntimeError("drives_reached20 exceeds offensive_drives")
    if (
        x.loc[present, "drives_reached10"]
        > x.loc[present, "drives_reached20"] + TOL
    ).any():
        raise RuntimeError("drives_reached10 exceeds drives_reached20")
    return x


def build_team_drive_rows(teamgames: pd.DataFrame, cell: str) -> pd.DataFrame:
    spec = CELL_MAP[cell]
    flag = spec["source_flag"]
    hv = spec["team_hv_col"]
    parent = spec["team_parent_col"]
    reach = spec["drive_reach_col"]

    rows: list[dict] = []
    target_mask = (
        teamgames["season"].isin(SEASONS)
        & teamgames["week"].isin(TARGET_WEEKS)
        & teamgames[flag]
    )
    for r in teamgames.loc[target_mask].itertuples(index=False):
        season = int(r.season)
        week = int(r.week)
        team = str(r.team)

        league_hv_hist = teamgames.loc[
            teamgames["season"].eq(season)
            & teamgames[flag]
            & teamgames["week"].lt(week)
        ].copy()
        league_drive_hist = teamgames.loc[
            teamgames["season"].eq(season)
            & teamgames["drive_source_present"]
            & teamgames["week"].lt(week)
        ].copy()
        team_drive_hist = (
            league_drive_hist.loc[league_drive_hist["team"].eq(team)]
            .sort_values("week")
            .tail(HISTORY_WINDOW)
        )
        if len(team_drive_hist) < HISTORY_MIN:
            continue

        league_hv_parent = float(num(league_hv_hist[parent]).sum())
        league_hv_count = float(num(league_hv_hist[hv]).sum())
        league_drives = float(num(league_drive_hist["offensive_drives"]).sum())
        league_reached = float(num(league_drive_hist[reach]).sum())
        team_drives = float(num(team_drive_hist["offensive_drives"]).sum())
        team_reached = float(num(team_drive_hist[reach]).sum())

        if (
            not np.isfinite(league_hv_parent)
            or not np.isfinite(league_hv_count)
            or not np.isfinite(league_drives)
            or not np.isfinite(league_reached)
            or not np.isfinite(team_drives)
            or not np.isfinite(team_reached)
            or league_hv_parent <= 0
            or league_drives <= 0
            or team_drives <= 0
        ):
            continue

        league_hv_rate = league_hv_count / league_hv_parent
        league_drive_rate = league_reached / league_drives
        team_drive_rate = team_reached / team_drives
        if league_drive_rate <= 0:
            continue

        multiplier = team_drive_rate / league_drive_rate
        candidate_raw = league_hv_rate * multiplier
        candidate_rate = float(np.clip(candidate_raw, 0.0, 1.0))
        clipped = not np.isclose(candidate_raw, candidate_rate, atol=TOL, rtol=0)

        for label, value in (
            ("league_hv_rate", league_hv_rate),
            ("league_drive_rate", league_drive_rate),
            ("team_drive_rate", team_drive_rate),
            ("territory_multiplier", multiplier),
            ("candidate_rate", candidate_rate),
        ):
            if not np.isfinite(value):
                raise RuntimeError(
                    f"non-finite {label} {season} W{week} {team} {cell}"
                )
        if not (0.0 - TOL <= league_hv_rate <= 1.0 + TOL):
            raise RuntimeError("league high-value rate outside [0,1]")
        if not (0.0 - TOL <= league_drive_rate <= 1.0 + TOL):
            raise RuntimeError("league drive-entry rate outside [0,1]")
        if not (0.0 - TOL <= team_drive_rate <= 1.0 + TOL):
            raise RuntimeError("team drive-entry rate outside [0,1]")
        if not (0.0 - TOL <= candidate_rate <= 1.0 + TOL):
            raise RuntimeError("candidate high-value rate outside [0,1]")

        target_hv = float(getattr(r, hv))
        target_parent = float(getattr(r, parent))
        if not np.isfinite(target_hv) or not np.isfinite(target_parent):
            raise RuntimeError(
                f"non-finite target label {season} W{week} {team} {cell}"
            )
        if target_hv < -TOL or target_hv > target_parent + TOL:
            raise RuntimeError(
                f"finite-pool target contradiction {season} W{week} {team} {cell}"
            )
        if abs(target_hv - round(target_hv)) > TOL:
            raise RuntimeError("non-integer target high-value count")

        rows.append(
            {
                "season": season,
                "week": week,
                "game_key": str(r.game_key),
                "team": team,
                "opponent": str(r.opponent),
                "quality_cell": cell,
                "territory": spec["territory"],
                "target_team_high_value": int(round(target_hv)),
                "target_team_parent_observed": target_parent,
                "prior_team_drive_games": int(len(team_drive_hist)),
                "team_drive_history_min_week": int(team_drive_hist["week"].min()),
                "team_drive_history_max_week": int(team_drive_hist["week"].max()),
                "league_hv_history_max_week": int(league_hv_hist["week"].max()),
                "league_drive_history_max_week": int(league_drive_hist["week"].max()),
                "team_prior_drives": team_drives,
                "team_prior_drives_reached": team_reached,
                "league_prior_drives": league_drives,
                "league_prior_drives_reached": league_reached,
                "league_prior_high_value_sum": league_hv_count,
                "league_prior_parent_sum": league_hv_parent,
                "league_high_value_rate": float(league_hv_rate),
                "team_drive_entry_rate": float(team_drive_rate),
                "league_drive_entry_rate": float(league_drive_rate),
                "territory_multiplier": float(multiplier),
                "candidate_high_value_rate_raw": float(candidate_raw),
                "candidate_high_value_rate": float(candidate_rate),
                # Compatibility field consumed by the already-certified
                # fixed-allocation score_cell. It is the DRIVE-TRANSMITTED
                # candidate rate, never the team's recent HV event rate.
                "team_high_value_rate": float(candidate_rate),
                "candidate_rate_clipped": bool(clipped),
                "chronology_violation": int(
                    int(team_drive_hist["week"].max()) >= week
                    or int(league_hv_hist["week"].max()) >= week
                    or int(league_drive_hist["week"].max()) >= week
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"zero scoreable drive-state rows cell={cell}")
    if out.duplicated(["season", "week", "team", "quality_cell"]).any():
        raise RuntimeError(f"duplicate drive-state rows cell={cell}")
    return out


def drive_season_metrics(
    cell: str,
    season: int,
    team: pd.DataFrame,
    player: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    rec, ep = base_season_metrics(cell, season, team, player)
    t = team.loc[team["season"].eq(season)].copy()

    # Rename the inherited generic candidate-rate label so the result cannot
    # be misread as direct team high-value-event-rate persistence.
    rec["mean_candidate_high_value_rate"] = rec.pop("mean_team_high_value_rate")
    rec.update(
        {
            "drive_history_candidate_team_games": int(len(t)),
            "mean_team_drive_entry_rate": float(num(t["team_drive_entry_rate"]).mean()),
            "mean_league_drive_entry_rate": float(num(t["league_drive_entry_rate"]).mean()),
            "mean_territory_multiplier": float(num(t["territory_multiplier"]).mean()),
            "median_territory_multiplier": float(num(t["territory_multiplier"]).median()),
            "candidate_rate_clipping_share": float(
                bools(t["candidate_rate_clipped"]).mean()
            ),
        }
    )
    rec["support_gates"]["drive_history_candidate_team_games_ge_400"] = (
        rec["drive_history_candidate_team_games"] >= TEAM_MIN_ROWS
    )
    return rec, ep


def qualify_drive(cell: str, m24: dict, m25: dict) -> tuple[dict, str]:
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
        "confirm_2025_equal_player_mean_gt_0": (
            m25["player_mean_per_player_crps_gain"] > 0
        ),
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
        f"{cell}_PRISM_TEAM_DRIVE_CREATION_QUALIFIED"
        if all(gates.values())
        else f"{cell}_PRISM_TEAM_DRIVE_CREATION_CLOSED"
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

        raw_teamgames = prepare_teamgames_with_drive(
            read_csv(args.raw_teamgames, "raw opportunity-quality teamgames")
        )
        raw_topology = prepare_player_labels(
            read_csv(args.raw_player_topology, "raw player topology")
        )
        stage2 = prepare_stage2_expected_shares(
            read_csv(args.stage2_detail, "Stage-2 detail"),
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
            drive_rows = build_team_drive_rows(raw_teamgames, cell)
            team_scored, player_scored = score_fixed_allocation_cell(
                cell=cell,
                team_rate_rows=drive_rows,
                stage2=stage2,
                player_topology=raw_topology,
            )
            team_parts.append(team_scored)
            player_parts.append(player_scored)
            season_metrics_map[cell] = {}
            by_season = {}
            for season in SEASONS:
                metrics, pdiag = drive_season_metrics(
                    cell, season, team_scored, player_scored
                )
                season_metrics_map[cell][str(season)] = metrics
                pdiag["quality_cell"] = cell
                pdiag["season"] = season
                player_diag_parts.append(pdiag)
                by_season[season] = metrics
            gates, disposition = qualify_drive(
                cell, by_season[2024], by_season[2025]
            )
            qualification_gates[cell] = gates
            dispositions[cell] = disposition

        n_qualified = sum(v.endswith("_QUALIFIED") for v in dispositions.values())
        if n_qualified == len(CELL_MAP):
            overall = "PRISM_TEAM_DRIVE_CREATION_QUALIFIED_ALL"
        elif n_qualified:
            overall = "PRISM_TEAM_DRIVE_CREATION_QUALIFIED_PARTIAL"
        else:
            overall = "PRISM_TEAM_DRIVE_CREATION_CLOSED"

        team_all = pd.concat(team_parts, ignore_index=True)
        player_all = pd.concat(player_parts, ignore_index=True)
        player_diag = (
            pd.concat(player_diag_parts, ignore_index=True)
            if player_diag_parts
            else pd.DataFrame()
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
                "rush_territory": "DRIVES_REACHED_OPPONENT_10",
                "receiving_territory": "DRIVES_REACHED_OPPONENT_20",
                "baseline_rate": "STRICTLY_PRIOR_SAME_SEASON_LEAGUE_HV_OVER_PARENT",
                "candidate_law": "BASELINE_HV_RATE_X_TEAM_TO_LEAGUE_DRIVE_ENTRY_RATE_RATIO_CLIPPED_0_1",
                "team_count_law": "POISSON",
                "player_count_law": "POISSON",
                "player_allocation": "QUALIFIED_STAGE1_PLUS_STAGE2_EXACT_ACTIVE_SET_EXPECTATION",
                "bootstrap_replicates": BOOT_REPS,
                "bootstrap_seed": BOOT_SEED,
            },
            "integrity": {
                "target_game_outcomes_used_as_predictors": 0,
                "predictive_models_fit": 0,
                "fitted_coefficients": 0,
                "threshold_searches": 0,
                "direct_recent_team_high_value_rate_candidate_used": False,
                "player_high_value_share_candidate_used": False,
                "opponent_environment_used": False,
                "td_outcomes_used": False,
                "sportsbook_inputs_used": 0,
                "oddsapi_calls": 0,
                "outcomes_2026_used": 0,
                "production_changed": False,
            },
        }

        team_all.to_csv(
            args.out_dir / "prism_team_drive_creation_team_rows.csv", index=False
        )
        player_all.to_csv(
            args.out_dir / "prism_team_drive_creation_player_rows.csv", index=False
        )
        player_diag.to_csv(
            args.out_dir / "prism_team_drive_creation_player_diagnostics.csv",
            index=False,
        )
        (args.out_dir / "prism_team_drive_creation_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_TEAM_DRIVE_CREATION_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "integrity": {
                "target_game_outcomes_used_as_predictors": 0,
                "predictive_models_fit": 0,
                "fitted_coefficients": 0,
                "threshold_searches": 0,
                "direct_recent_team_high_value_rate_candidate_used": False,
                "player_high_value_share_candidate_used": False,
                "opponent_environment_used": False,
                "td_outcomes_used": False,
                "sportsbook_inputs_used": 0,
                "oddsapi_calls": 0,
                "outcomes_2026_used": 0,
                "production_changed": False,
            },
        }
        (args.out_dir / "prism_team_drive_creation_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
