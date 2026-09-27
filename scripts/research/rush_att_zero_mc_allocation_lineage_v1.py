#!/usr/bin/env python3
"""Rush-Attempt Zero-MC Allocation Lineage Audit V1.

Diagnostic-only. This script traces the canonical historical rushing-attempt
Monte Carlo path from the output-row rushing share through the simulator's
exact one-row-per-player authority, literal top-five selector, multinomial
probability, realized carry array, keyed lookup, and canonical mc_proj.

No candidate repair is implemented here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
import scripts.simulation_v2 as simv2

TOL = 1e-9


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _selected_authority(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mirror simulation_v2's exact row collapse and top-five selector."""
    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(simv2._player_key, axis=1)
    game_key = "event_id" if "event_id" in frame.columns and frame["event_id"].notna().any() else None
    if game_key is None:
        frame["_game_key"] = frame.apply(
            lambda r: "|".join(sorted([str(r.get("team", "")), str(r.get("opponent", ""))])),
            axis=1,
        )
        game_key = "_game_key"

    player_cols = [game_key, "team", "player_clean_key"]
    selected = frame.sort_values(player_cols).drop_duplicates(player_cols, keep="last").copy()

    rows: list[dict] = []
    for game, game_df in selected.groupby(game_key, dropna=False):
        for team, team_df in game_df.groupby("team", dropna=False):
            if pd.isna(team) or not str(team).strip():
                continue
            raw = np.array(
                [
                    simv2._num(
                        r,
                        "rules_rush_share",
                        "bayes_rush_share",
                        "rush_share",
                        default=0.0,
                    )
                    for _, r in team_df.iterrows()
                ],
                dtype=float,
            )
            clean = np.clip(
                np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
                0.95,
            )
            post_top5 = simv2._top_n_shares(raw, 5)

            order = np.argsort(-clean, kind="stable")
            ranks = np.empty(len(clean), dtype=int)
            ranks[order] = np.arange(1, len(clean) + 1)
            top5_idx = set(order[: min(5, len(order))].tolist())

            for j, (_, row) in enumerate(team_df.iterrows()):
                rows.append(
                    {
                        "event_id": str(game),
                        "team": str(team),
                        "player_clean_key": simv2._player_key(row),
                        "sim_selected_market": str(row.get("market", "")),
                        "sim_selected_rules_rush_share": float(clean[j]),
                        "sim_selected_share_rank": int(ranks[j]),
                        "sim_selected_top5_member": int(j in top5_idx),
                        "sim_post_top5_share": float(post_top5[j]),
                    }
                )
    return pd.DataFrame(rows)


def _classify(row: pd.Series) -> str:
    output_share = pd.to_numeric(pd.Series([row.get("rush_att_row_rules_rush_share")]), errors="coerce").iloc[0]
    selected_share = pd.to_numeric(pd.Series([row.get("sim_selected_rules_rush_share")]), errors="coerce").iloc[0]
    selected_top5 = pd.to_numeric(pd.Series([row.get("sim_selected_top5_member")]), errors="coerce").iloc[0]
    final_prob = pd.to_numeric(pd.Series([row.get("final_player_probability")]), errors="coerce").iloc[0]
    realized = pd.to_numeric(pd.Series([row.get("realized_multinomial_mean_carries")]), errors="coerce").iloc[0]
    lookup = pd.to_numeric(pd.Series([row.get("keyed_lookup_mean")]), errors="coerce").iloc[0]
    canonical = pd.to_numeric(pd.Series([row.get("canonical_mc_proj")]), errors="coerce").iloc[0]

    if not np.isfinite(output_share) or output_share <= TOL:
        return "OUTPUT_SHARE_NONPOSITIVE"
    if not np.isfinite(selected_share):
        return "SELECTED_ROW_MISSING"
    if selected_share <= TOL:
        return "SELECTED_ROW_SHARE_NONPOSITIVE"
    if not np.isfinite(selected_top5) or int(selected_top5) != 1:
        return "TOP5_EXCLUDED"
    if not np.isfinite(final_prob) or final_prob <= TOL:
        return "FINAL_PROBABILITY_ZERO"
    if not np.isfinite(realized) or realized <= TOL:
        return "REALIZED_MC_ZERO_WITH_POSITIVE_PROBABILITY"
    if not np.isfinite(lookup) or lookup <= TOL:
        return "LOOKUP_ZERO_AFTER_POSITIVE_REALIZED"
    if not np.isfinite(canonical) or canonical <= TOL:
        return "CANONICAL_ZERO_AFTER_POSITIVE_LOOKUP"
    return "UNEXPLAINED"


def _position_family(value: object) -> str:
    p = str(value or "").upper().strip()
    if p in {"HB", "TB"}:
        return "RB"
    if p in {"LWR", "RWR", "SWR"}:
        return "WR"
    return p or "UNKNOWN"


def _add_components(
    prepared: pd.DataFrame,
    *,
    logs: pd.DataFrame,
    bundle,
    season: int,
    week: int,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ML/State and the current calibrated ensemble to the full pregame frame."""
    out = prepared.copy()
    _, ml_pred = cp.build_ml(logs, bundle.player_consensus, int(season), int(week))
    _, state_pred = cp.build_state_predictions(logs, bundle.player_consensus, int(season), int(week))
    out = cp._attach_component_projection(out, ml_pred, "ml")
    out = cp._attach_component_projection(out, state_pred, "state")
    out = apply_ensemble(out, weights=weights)
    return out


def _run_week(
    *,
    logs: pd.DataFrame,
    team: pd.DataFrame,
    sched: pd.DataFrame,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    universe_dir: Path,
    season: int,
    prior_season: int,
    week: int,
    iterations: int,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    universe = _read(universe_dir / f"{season}_week_{week:02d}.csv", f"universe {season} W{week}")
    bundle = build_historical_context_bundle(
        player_logs=logs,
        team_weekly=team,
        pregame_universe=universe,
        schedule=sched,
        season=int(season),
        week=int(week),
        prior_season=int(prior_season),
        injuries=_exact_week(injuries, int(season), int(week)),
        weather=_exact_week(weather, int(season), int(week)),
    )
    seed = 42 + int(week)

    # This is the canonical pregame football frame, including the canonical MC.
    prepared = cp.build_mc_predictions(bundle, iterations=int(iterations), seed=seed)
    enriched = _add_components(
        prepared,
        logs=logs,
        bundle=bundle,
        season=season,
        week=week,
        weights=weights,
    )

    rush = enriched.loc[enriched["market"].astype(str).eq("rush_att")].copy()
    rush = rush.rename(
        columns={
            "mc_proj": "canonical_mc_proj",
            "rules_rush_share": "rush_att_row_rules_rush_share",
        }
    )
    keep = [
        "event_id",
        "team",
        "player_clean_key",
        "player",
        "position",
        "role",
        "rush_att_row_rules_rush_share",
        "canonical_mc_proj",
        "ml_proj",
        "state_proj",
        "ensemble_proj",
        "ensemble_status",
        "ensemble_weight_mc",
        "ensemble_weight_ml",
        "ensemble_weight_state",
    ]
    keep = [c for c in keep if c in rush.columns]
    rush = rush[keep].drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")

    selected = _selected_authority(prepared)

    allocation_trace: list[dict] = []
    keyed_result = simv2.simulate(
        prepared,
        iterations=int(iterations),
        seed=seed,
        allocation_trace=allocation_trace,
    )
    allocated = pd.DataFrame(allocation_trace)
    if allocated.empty:
        raise RuntimeError(f"{season} W{week}: canonical allocation trace is empty")
    allocated = allocated.rename(
        columns={
            "raw_player_rush_share": "allocation_post_top5_share",
            "expected_carries_from_final_probability": "expected_carries_from_final_probability",
            "realized_multinomial_mean_carries": "realized_multinomial_mean_carries",
        }
    )
    alloc_keep = [
        "event_id",
        "team",
        "player_clean_key",
        "allocation_post_top5_share",
        "raw_team_rush_share_sum",
        "final_player_probability",
        "residual_probability",
        "team_rush_total_mean",
        "expected_carries_from_final_probability",
        "realized_multinomial_mean_carries",
    ]
    allocated = allocated[alloc_keep]

    keyed_lookup = []
    for _, row in rush.iterrows():
        arr = simv2.lookup(keyed_result, row, "rush_att")
        keyed_lookup.append(float(np.mean(arr)) if arr is not None and len(arr) else np.nan)
    rush["keyed_lookup_mean"] = keyed_lookup

    trace = rush.merge(
        selected,
        on=["event_id", "team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    trace = trace.merge(
        allocated,
        on=["event_id", "team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )

    actual = cp.build_actual_rows(logs, int(season), int(week))
    actual = actual.loc[
        actual["market"].astype(str).eq("rush_att"),
        ["team", "player_clean_key", "actual"],
    ].drop_duplicates(["team", "player_clean_key"])
    trace = trace.merge(actual, on=["team", "player_clean_key"], how="left", validate="one_to_one")

    trace.insert(0, "season", int(season))
    trace.insert(1, "week", int(week))
    trace["position_family"] = trace.get("position", "").map(_position_family)
    trace["share_row_vs_selected_abs"] = (
        pd.to_numeric(trace["rush_att_row_rules_rush_share"], errors="coerce")
        - pd.to_numeric(trace["sim_selected_rules_rush_share"], errors="coerce")
    ).abs()
    trace["selected_post_top5_abs"] = (
        pd.to_numeric(trace["sim_post_top5_share"], errors="coerce")
        - pd.to_numeric(trace["allocation_post_top5_share"], errors="coerce")
    ).abs()
    trace["canonical_vs_lookup_abs"] = (
        pd.to_numeric(trace["canonical_mc_proj"], errors="coerce")
        - pd.to_numeric(trace["keyed_lookup_mean"], errors="coerce")
    ).abs()
    trace["lookup_vs_realized_abs"] = (
        pd.to_numeric(trace["keyed_lookup_mean"], errors="coerce")
        - pd.to_numeric(trace["realized_multinomial_mean_carries"], errors="coerce")
    ).abs()

    trace["zero_mc_nonzero_ensemble"] = (
        pd.to_numeric(trace["canonical_mc_proj"], errors="coerce").fillna(0.0).abs().le(TOL)
        & pd.to_numeric(trace["ensemble_proj"], errors="coerce").fillna(0.0).gt(TOL)
    ).astype(int)
    trace["positive_output_share_zero_mc"] = (
        pd.to_numeric(trace["rush_att_row_rules_rush_share"], errors="coerce").fillna(0.0).gt(TOL)
        & pd.to_numeric(trace["canonical_mc_proj"], errors="coerce").fillna(0.0).abs().le(TOL)
    ).astype(int)
    trace["first_zero_stage"] = ""
    mask = trace["zero_mc_nonzero_ensemble"].eq(1)
    trace.loc[mask, "first_zero_stage"] = trace.loc[mask].apply(_classify, axis=1)

    max_can = float(pd.to_numeric(trace["canonical_vs_lookup_abs"], errors="coerce").max())
    max_lookup = float(pd.to_numeric(trace["lookup_vs_realized_abs"], errors="coerce").max())
    max_top5 = float(pd.to_numeric(trace["selected_post_top5_abs"], errors="coerce").max())
    if not np.isfinite(max_can) or max_can > TOL:
        raise RuntimeError(f"{season} W{week}: canonical/keyed lookup mismatch {max_can}")
    if not np.isfinite(max_lookup) or max_lookup > TOL:
        raise RuntimeError(f"{season} W{week}: lookup/realized mismatch {max_lookup}")
    if not np.isfinite(max_top5) or max_top5 > TOL:
        raise RuntimeError(f"{season} W{week}: top-five trace mismatch {max_top5}")

    print(
        f"[lineage] {season} W{week:02d} rows={len(trace)} "
        f"blocked={int(trace.zero_mc_nonzero_ensemble.sum())} "
        f"positive-share-zero-mc={int(trace.positive_output_share_zero_mc.sum())} "
        f"share-authority-mismatches={int((trace.share_row_vs_selected_abs > TOL).sum())}"
    )
    return trace


def _summaries(trace: pd.DataFrame, out_dir: Path) -> None:
    blocked = trace.loc[trace["zero_mc_nonzero_ensemble"].eq(1)].copy()

    stage = (
        blocked.groupby(["season", "position_family", "first_zero_stage"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["season", "position_family", "rows"], ascending=[True, True, False])
    )
    stage.to_csv(out_dir / "first_zero_stage_summary.csv", index=False)

    selected_market = (
        blocked.groupby(["season", "position_family", "sim_selected_market"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["season", "position_family", "rows"], ascending=[True, True, False])
    )
    selected_market.to_csv(out_dir / "selected_market_summary.csv", index=False)

    rank_frame = trace.loc[
        pd.to_numeric(trace["rush_att_row_rules_rush_share"], errors="coerce").fillna(0.0).gt(TOL)
    ].copy()
    rank_frame["mc_positive"] = (
        pd.to_numeric(rank_frame["canonical_mc_proj"], errors="coerce").fillna(0.0).gt(TOL)
    ).astype(int)
    rank = (
        rank_frame.groupby(["season", "sim_selected_share_rank"], dropna=False)
        .agg(rows=("mc_positive", "size"), positive_mc_rows=("mc_positive", "sum"), positive_mc_rate=("mc_positive", "mean"))
        .reset_index()
        .sort_values(["season", "sim_selected_share_rank"])
    )
    rank.to_csv(out_dir / "selected_rank_positive_mc_summary.csv", index=False)

    checks = pd.DataFrame(
        [
            {
                "rows": len(trace),
                "zero_mc_nonzero_ensemble_rows": int(trace["zero_mc_nonzero_ensemble"].sum()),
                "positive_output_share_zero_mc_rows": int(trace["positive_output_share_zero_mc"].sum()),
                "rushrow_vs_selected_share_mismatch_rows": int((trace["share_row_vs_selected_abs"] > TOL).sum()),
                "rushrow_vs_selected_share_max_abs": float(pd.to_numeric(trace["share_row_vs_selected_abs"], errors="coerce").max()),
                "canonical_vs_keyed_lookup_mismatch_rows": int((trace["canonical_vs_lookup_abs"] > TOL).sum()),
                "canonical_vs_keyed_lookup_max_abs": float(pd.to_numeric(trace["canonical_vs_lookup_abs"], errors="coerce").max()),
                "lookup_vs_realized_mismatch_rows": int((trace["lookup_vs_realized_abs"] > TOL).sum()),
                "lookup_vs_realized_max_abs": float(pd.to_numeric(trace["lookup_vs_realized_abs"], errors="coerce").max()),
                "top5_trace_mismatch_rows": int((trace["selected_post_top5_abs"] > TOL).sum()),
                "top5_trace_max_abs": float(pd.to_numeric(trace["selected_post_top5_abs"], errors="coerce").max()),
                "candidate_variants": 0,
                "fitted_parameters": 0,
                "sportsbook_inputs_upstream": 0,
                "production_mutations": 0,
            }
        ]
    )
    checks.to_csv(out_dir / "lineage_checks.csv", index=False)

    kirk = trace.loc[
        trace["season"].eq(2024)
        & trace["week"].eq(1)
        & trace["team"].astype(str).eq("ATL")
        & trace["player"].astype(str).str.contains("Kirk Cousins", case=False, na=False)
    ].copy()
    kirk.to_csv(out_dir / "kirk_cousins_2024_week1_atl.csv", index=False)

    lines = [
        "# Rush-Attempt Zero-MC Allocation Lineage V1 — Raw Diagnostic Result",
        "",
        f"Rows traced: {len(trace):,}",
        f"Zero-MC / nonzero-ensemble rows: {int(trace['zero_mc_nonzero_ensemble'].sum()):,}",
        f"Positive output-share / zero-MC rows: {int(trace['positive_output_share_zero_mc'].sum()):,}",
        "",
        "## First-zero stages",
        "",
        stage.to_markdown(index=False) if not stage.empty else "No blocked rows.",
        "",
        "## Exact parity checks",
        "",
        checks.to_markdown(index=False),
        "",
        "No repair is authorized by this raw diagnostic artifact.",
    ]
    (out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--injuries", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    sched = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    weights = load_weights()
    if weights.empty or not weights["market"].astype(str).str.lower().eq("rush_att").any():
        raise RuntimeError("current calibrated rush_att ensemble weights are unavailable")

    rows = []
    for week in _parse_weeks(a.weeks):
        rows.append(
            _run_week(
                logs=logs,
                team=team,
                sched=sched,
                injuries=injuries,
                weather=weather,
                universe_dir=a.universe_dir,
                season=int(a.season),
                prior_season=int(a.prior_season),
                week=int(week),
                iterations=int(a.iterations),
                weights=weights,
            )
        )

    out = pd.concat(rows, ignore_index=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "rush_att_allocation_lineage.csv", index=False)
    _summaries(out, a.out_dir)
    print((a.out_dir / "RESULT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
