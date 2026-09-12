#!/usr/bin/env python3
"""Compare historical component-SD probabilities with production-style empirical probabilities.

The OLD arm is exactly ``grade_full_stack_vegas_benchmark_v1.grade``: same clean
identity checks, same selected sportsbook row, same final football mean, same
Normal(mean=proj, sd=component_sd) translator, and same frozen betting gates.

The NEW arm changes only the probability translator. It loads the exact Monte
Carlo arrays persisted by ``walk_forward_distribution_v1.py``, verifies that
their mean reproduces the same historical ``mc_proj``, rescales each outcome
array to the same final football mean exactly as production does, and computes
empirical P(outcome > line). Sportsbook data remains downstream throughout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import (
    ev_roi,
    grade as grade_old,
    signal,
)
from scripts.backtest.walk_forward_distribution_v1 import (
    IDENTITY_COLUMNS,
    SERIALIZATION_ATOL,
    SERIALIZATION_RTOL,
)
from scripts.operations.grade_market_track_record_v1 import american_profit, num


EXPECTED_ITERATIONS = 2000
ALIGNMENT_RTOL = 5e-6
ALIGNMENT_ATOL = 5e-4


def _normalize_identity(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    out = frame.copy()
    required = set(IDENTITY_COLUMNS)
    missing = sorted(required - set(out.columns))
    if missing:
        raise RuntimeError(f"{label} missing identity columns: {missing}")
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    out["team"] = out["team"].map(canon_team)
    out["opponent"] = out["opponent"].map(canon_team)
    out["player_clean_key"] = out["player_clean_key"].fillna("").astype(str).str.strip()
    out["market"] = out["market"].fillna("").astype(str).str.lower().str.strip()
    if out["team"].eq("").any() or out["opponent"].eq("").any():
        raise RuntimeError(f"{label} contains uncanonicalizable team/opponent")
    if out["player_clean_key"].eq("").any() or out["market"].eq("").any():
        raise RuntimeError(f"{label} contains blank player/market identity")
    if out.duplicated(IDENTITY_COLUMNS).any():
        sample = out.loc[out.duplicated(IDENTITY_COLUMNS, keep=False), IDENTITY_COLUMNS].head(20)
        raise RuntimeError(f"{label} duplicate identity rows: {sample.to_dict('records')}")
    return out


def load_distribution_manifest(distribution_dir: Path) -> pd.DataFrame:
    manifests = sorted(Path(distribution_dir).glob("sim_distribution_*_manifest.csv"))
    if not manifests:
        raise RuntimeError(f"no simulation distribution manifests found in {distribution_dir}")
    parts = []
    required = {
        *IDENTITY_COLUMNS,
        "event_id",
        "array_row",
        "distribution_file",
        "simulation_iterations",
        "simulation_seed",
        "canonical_mc_proj",
        "stored_mc_proj",
        "stored_model_sd",
    }
    for path in manifests:
        x = pd.read_csv(path)
        x.columns = [str(c).strip().lower() for c in x.columns]
        missing = sorted(required - set(x.columns))
        if missing:
            raise RuntimeError(f"distribution manifest {path} missing columns: {missing}")
        x["manifest_file"] = path.name
        parts.append(x)
    out = _normalize_identity(pd.concat(parts, ignore_index=True), label="distribution manifest")
    out["array_row"] = pd.to_numeric(out["array_row"], errors="raise").astype(int)
    out["simulation_iterations"] = pd.to_numeric(
        out["simulation_iterations"], errors="raise"
    ).astype(int)
    out["simulation_seed"] = pd.to_numeric(out["simulation_seed"], errors="raise").astype(int)
    out["distribution_file"] = out["distribution_file"].fillna("").astype(str).str.strip()
    if out["distribution_file"].eq("").any():
        raise RuntimeError("distribution manifest contains blank distribution_file")
    for name in out["distribution_file"].unique():
        if Path(name).name != name:
            raise RuntimeError(f"distribution manifest contains non-basename path: {name}")
        if not (Path(distribution_dir) / name).exists():
            raise RuntimeError(f"distribution matrix missing: {Path(distribution_dir) / name}")
    return out


def _attach_empirical_probabilities(
    old_detail: pd.DataFrame,
    manifest: pd.DataFrame,
    distribution_dir: Path,
    *,
    expected_iterations: int,
) -> pd.DataFrame:
    z = _normalize_identity(old_detail, label="old translator detail")
    manifest_cols = IDENTITY_COLUMNS + [
        "event_id",
        "array_row",
        "distribution_file",
        "simulation_iterations",
        "simulation_seed",
        "canonical_mc_proj",
        "stored_mc_proj",
        "stored_model_sd",
        "serialization_abs_error",
        "manifest_file",
    ]
    manifest_cols = [c for c in manifest_cols if c in manifest.columns]
    z = z.merge(
        manifest[manifest_cols],
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_distribution"),
    )
    missing_dist = z["distribution_file"].isna()
    if missing_dist.any():
        sample = z.loc[missing_dist, IDENTITY_COLUMNS].head(20).to_dict("records")
        raise RuntimeError(f"missing empirical distribution for graded rows: {sample}")

    iteration_values = pd.to_numeric(z["simulation_iterations"], errors="raise").astype(int)
    if not iteration_values.eq(int(expected_iterations)).all():
        bad = sorted(iteration_values.loc[~iteration_values.eq(int(expected_iterations))].unique().tolist())
        raise RuntimeError(
            f"empirical reconstruction iteration policy violated: expected={expected_iterations} found={bad}"
        )

    z["old_p_over"] = num(z["p_over"])
    z["old_p_under"] = num(z["p_under"])
    z["old_side"] = z["side"].astype(str)
    z["old_best_ev"] = num(z["best_ev"])
    z["old_best_model_p"] = num(z["best_model_p"])
    z["old_best_market_p"] = num(z["best_market_p"])
    z["old_prob_edge"] = num(z["prob_edge"])
    z["old_chosen_odds"] = num(z["chosen_odds"])
    z["old_signal"] = z["signal"].astype(str)
    z["old_bet_result"] = z["bet_result"].astype(str)
    z["old_unit_result"] = num(z["unit_result"])
    z["old_component_sd"] = num(z["component_sd"])
    z["target_mean"] = num(z["proj"])

    n = len(z)
    new_p_over = np.full(n, np.nan, dtype=float)
    new_model_mean = np.full(n, np.nan, dtype=float)
    new_model_sd = np.full(n, np.nan, dtype=float)
    mean_alignment_error = np.full(n, np.nan, dtype=float)
    stored_mean_from_array = np.full(n, np.nan, dtype=float)

    for distribution_file, group in z.groupby("distribution_file", sort=False):
        path = Path(distribution_dir) / str(distribution_file)
        with np.load(path, allow_pickle=False) as payload:
            if "outcomes" not in payload.files:
                raise RuntimeError(f"distribution matrix missing outcomes key: {path}")
            matrix = payload["outcomes"]
        if matrix.ndim != 2:
            raise RuntimeError(f"distribution matrix must be 2D: {path} shape={matrix.shape}")
        if matrix.shape[1] != int(expected_iterations):
            raise RuntimeError(
                f"distribution matrix iteration mismatch: {path} shape={matrix.shape} "
                f"expected second dimension={expected_iterations}"
            )

        for idx, row in group.iterrows():
            array_row = int(row["array_row"])
            if array_row < 0 or array_row >= matrix.shape[0]:
                raise RuntimeError(
                    f"distribution array_row out of bounds: file={distribution_file} row={array_row} "
                    f"matrix_rows={matrix.shape[0]}"
                )
            outcomes = np.asarray(matrix[array_row], dtype=np.float64)
            arr_mean = float(np.mean(outcomes))
            stored_mean_from_array[z.index.get_loc(idx)] = arr_mean

            mc_proj = float(row["mc_proj"])
            manifest_mc = float(row["canonical_mc_proj"])
            if not np.isclose(
                manifest_mc, mc_proj, rtol=SERIALIZATION_RTOL, atol=SERIALIZATION_ATOL
            ):
                raise RuntimeError(
                    "manifest canonical_mc_proj does not match graded mc_proj "
                    f"for {tuple(row[c] for c in IDENTITY_COLUMNS)}: "
                    f"manifest={manifest_mc} graded={mc_proj}"
                )
            if not np.isclose(
                arr_mean, mc_proj, rtol=SERIALIZATION_RTOL, atol=SERIALIZATION_ATOL
            ):
                raise RuntimeError(
                    "stored empirical distribution does not reproduce graded mc_proj "
                    f"for {tuple(row[c] for c in IDENTITY_COLUMNS)}: "
                    f"stored={arr_mean} graded={mc_proj}"
                )

            target_mean = float(row["target_mean"])
            if np.isfinite(mc_proj) and mc_proj > 0 and np.isfinite(target_mean):
                adjusted = outcomes * max(0.0, target_mean / mc_proj)
            else:
                adjusted = outcomes

            line = float(row["line"])
            if not np.isfinite(line):
                raise RuntimeError(f"non-finite sportsbook line for {tuple(row[c] for c in IDENTITY_COLUMNS)}")
            p_over = float(np.mean(adjusted > line))
            aligned_mean = float(np.mean(adjusted))
            aligned_sd = float(np.std(adjusted, ddof=1)) if len(adjusted) > 1 else 0.0

            loc = z.index.get_loc(idx)
            new_p_over[loc] = p_over
            new_model_mean[loc] = aligned_mean
            new_model_sd[loc] = aligned_sd
            if np.isfinite(mc_proj) and mc_proj > 0 and np.isfinite(target_mean):
                expected_mean = max(0.0, target_mean)
                error = abs(aligned_mean - expected_mean)
                mean_alignment_error[loc] = error
                if not np.isclose(
                    aligned_mean,
                    expected_mean,
                    rtol=ALIGNMENT_RTOL,
                    atol=ALIGNMENT_ATOL,
                ):
                    raise RuntimeError(
                        "production-style mean alignment failed "
                        f"for {tuple(row[c] for c in IDENTITY_COLUMNS)}: "
                        f"aligned={aligned_mean} target={expected_mean}"
                    )

    z["stored_mc_proj_from_array"] = stored_mean_from_array
    z["new_p_over"] = new_p_over
    z["new_p_under"] = 1.0 - z["new_p_over"]
    z["new_model_proj"] = new_model_mean
    z["new_model_sd"] = new_model_sd
    z["mean_alignment_abs_error"] = mean_alignment_error

    z["new_ev_over"] = [ev_roi(p, o) for p, o in zip(z.new_p_over, z.over_odds)]
    z["new_ev_under"] = [ev_roi(p, o) for p, o in zip(z.new_p_under, z.under_odds)]
    best_over = z.new_ev_under.isna() | (
        z.new_ev_over.fillna(-np.inf) >= z.new_ev_under.fillna(-np.inf)
    )
    z["new_side"] = np.where(best_over, "OVER", "UNDER")
    z["new_best_ev"] = np.where(best_over, z.new_ev_over, z.new_ev_under)
    z["new_best_model_p"] = np.where(best_over, z.new_p_over, z.new_p_under)
    z["new_best_market_p"] = np.where(best_over, z.over_novig, z.under_novig)
    z["new_prob_edge"] = z.new_best_model_p - z.new_best_market_p
    z["new_chosen_odds"] = np.where(best_over, z.over_odds, z.under_odds)
    z["new_signal"] = [signal(e, q) for e, q in zip(z.new_best_ev, z.new_prob_edge)]
    z["new_bet_result"] = np.select(
        [z.actual_side.eq("PUSH"), z.new_side.eq(z.actual_side)],
        ["PUSH", "WIN"],
        default="LOSS",
    )
    z["new_unit_result"] = np.where(
        z.new_bet_result.eq("WIN"),
        [american_profit(o) for o in z.new_chosen_odds],
        np.where(z.new_bet_result.eq("LOSS"), -1.0, 0.0),
    )

    z["old_strong"] = z.old_signal.eq("STRONG_EDGE").astype(int)
    z["new_strong"] = z.new_signal.eq("STRONG_EDGE").astype(int)
    z["delta_p_over"] = z.new_p_over - z.old_p_over
    z["delta_best_ev"] = z.new_best_ev - z.old_best_ev
    z["delta_prob_edge"] = z.new_prob_edge - z.old_prob_edge
    z["delta_unit_result"] = z.new_unit_result - z.old_unit_result
    z["delta_sd_new_minus_component"] = z.new_model_sd - z.old_component_sd
    z["side_changed"] = z.new_side.ne(z.old_side).astype(int)
    z["signal_changed"] = z.new_signal.ne(z.old_signal).astype(int)
    return z


def _slice_frames(frame: pd.DataFrame, *, side_col: str):
    yield "ALL", "ALL", frame
    for dimension, column in (
        ("MARKET", "market"),
        ("SIDE", side_col),
        ("SEASON", "season"),
        ("POSITION", "position"),
    ):
        if column not in frame.columns:
            continue
        values = frame[column].fillna("UNKNOWN").astype(str)
        for value in sorted(values.unique().tolist()):
            yield dimension, value, frame.loc[values.eq(value)]


def build_decision_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ("old", "new"):
        signal_col = f"{method}_signal"
        side_col = f"{method}_side"
        result_col = f"{method}_bet_result"
        unit_col = f"{method}_unit_result"
        odds_col = f"{method}_chosen_odds"
        model_p_col = f"{method}_best_model_p"
        edge_col = f"{method}_prob_edge"
        for dimension, value, base in _slice_frames(detail, side_col=side_col):
            strong_coverage = float(base[signal_col].eq("STRONG_EDGE").mean()) if len(base) else np.nan
            tiers = {
                "ALL_NO_FILTER": base,
                "LEAN_OR_STRONG": base.loc[base[signal_col].isin(["LEAN_EDGE", "STRONG_EDGE"])],
                "STRONG_ONLY_PLAY_TIER": base.loc[base[signal_col].eq("STRONG_EDGE")],
            }
            for tier, g in tiers.items():
                decided = g.loc[
                    g[result_col].isin(["WIN", "LOSS"]) & num(g[odds_col]).notna()
                ]
                rows.append(
                    {
                        "method": method,
                        "slice_dimension": dimension,
                        "slice_value": value,
                        "tier": tier,
                        "rows": int(len(g)),
                        "strong_coverage": strong_coverage,
                        "decided_bets": int(len(decided)),
                        "wins": int(decided[result_col].eq("WIN").sum()),
                        "losses": int(decided[result_col].eq("LOSS").sum()),
                        "win_rate": float(decided[result_col].eq("WIN").mean()) if len(decided) else np.nan,
                        "units": float(num(decided[unit_col]).sum()) if len(decided) else np.nan,
                        "roi_per_unit": float(num(decided[unit_col]).mean()) if len(decided) else np.nan,
                        "mean_model_probability": float(num(g[model_p_col]).mean()) if len(g) else np.nan,
                        "mean_prob_edge": float(num(g[edge_col]).mean()) if len(g) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _binary_actual(detail: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(detail.actual_side.eq("OVER"), 1.0, np.where(detail.actual_side.eq("UNDER"), 0.0, np.nan)),
        index=detail.index,
        dtype=float,
    )


def build_probability_scores(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = _binary_actual(detail)
    for method in ("old", "new"):
        p_col = f"{method}_p_over"
        for dimension, value, g in _slice_frames(detail, side_col=f"{method}_side"):
            if dimension == "SIDE":
                continue
            y = actual.loc[g.index]
            p = num(g[p_col])
            valid = y.notna() & p.notna()
            yv = y.loc[valid].to_numpy(dtype=float)
            pv = p.loc[valid].to_numpy(dtype=float)
            if len(pv):
                clipped = np.clip(pv, 1e-6, 1.0 - 1e-6)
                brier = float(np.mean((pv - yv) ** 2))
                log_loss = float(np.mean(-(yv * np.log(clipped) + (1.0 - yv) * np.log(1.0 - clipped))))
            else:
                brier = log_loss = np.nan
            rows.append(
                {
                    "method": method,
                    "slice_dimension": dimension,
                    "slice_value": value,
                    "graded_binary_rows": int(len(pv)),
                    "brier_score": brier,
                    "log_loss": log_loss,
                    "mean_p_over": float(np.mean(pv)) if len(pv) else np.nan,
                    "realized_over_rate": float(np.mean(yv)) if len(yv) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_calibration_bins(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = _binary_actual(detail)
    edges = np.linspace(0.0, 1.0, 11)
    for method in ("old", "new"):
        p_col = f"{method}_p_over"
        scopes = [("ALL", "ALL", detail)]
        for market in sorted(detail.market.astype(str).unique().tolist()):
            scopes.append(("MARKET", market, detail.loc[detail.market.astype(str).eq(market)]))
        for dimension, value, g in scopes:
            y = actual.loc[g.index]
            p = num(g[p_col])
            valid = y.notna() & p.notna()
            if not valid.any():
                continue
            work = pd.DataFrame({"p": p.loc[valid], "y": y.loc[valid]})
            work["bin"] = pd.cut(work["p"], bins=edges, include_lowest=True, right=True)
            for bucket, b in work.groupby("bin", observed=True):
                rows.append(
                    {
                        "method": method,
                        "slice_dimension": dimension,
                        "slice_value": value,
                        "probability_bin": str(bucket),
                        "rows": int(len(b)),
                        "mean_predicted_p_over": float(b.p.mean()),
                        "realized_over_rate": float(b.y.mean()),
                        "brier_score": float(np.mean((b.p.to_numpy() - b.y.to_numpy()) ** 2)),
                    }
                )
    return pd.DataFrame(rows)


def _quantiles(series: pd.Series, prefix: str) -> dict:
    x = num(series).dropna()
    if x.empty:
        return {f"{prefix}_{name}": np.nan for name in ("mean", "p10", "p25", "p50", "p75", "p90")}
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_p10": float(x.quantile(0.10)),
        f"{prefix}_p25": float(x.quantile(0.25)),
        f"{prefix}_p50": float(x.quantile(0.50)),
        f"{prefix}_p75": float(x.quantile(0.75)),
        f"{prefix}_p90": float(x.quantile(0.90)),
    }


def build_distribution_diagnostics(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    residual = num(detail.actual) - num(detail.target_mean)
    temp = detail.copy()
    temp["actual_residual"] = residual
    temp["abs_actual_residual"] = residual.abs()
    for dimension, value, g in _slice_frames(temp, side_col="new_side"):
        if dimension == "SIDE":
            continue
        record = {
            "slice_dimension": dimension,
            "slice_value": value,
            "rows": int(len(g)),
            "actual_residual_sd": float(num(g.actual_residual).std(ddof=1)) if len(g) > 1 else np.nan,
            "max_mean_alignment_abs_error": float(num(g.mean_alignment_abs_error).max()) if len(g) else np.nan,
        }
        record.update(_quantiles(g.old_component_sd, "old_component_sd"))
        record.update(_quantiles(g.new_model_sd, "new_simulated_sd"))
        record.update(_quantiles(g.abs_actual_residual, "abs_actual_residual"))
        rows.append(record)
    return pd.DataFrame(rows)


def grade_empirical_ab(
    proj: pd.DataFrame,
    props: pd.DataFrame,
    distribution_dir: Path,
    *,
    proj_col: str = "ensemble_proj",
    expected_iterations: int = EXPECTED_ITERATIONS,
):
    old_detail, old_summary = grade_old(proj, props, proj_col=proj_col)
    if old_detail.empty:
        raise RuntimeError("old translator produced zero matched rows; cannot run same-row A/B")
    manifest = load_distribution_manifest(distribution_dir)
    detail = _attach_empirical_probabilities(
        old_detail,
        manifest,
        distribution_dir,
        expected_iterations=int(expected_iterations),
    )
    decisions = build_decision_summary(detail)
    scores = build_probability_scores(detail)
    calibration = build_calibration_bins(detail)
    diagnostics = build_distribution_diagnostics(detail)
    return detail, old_summary, decisions, scores, calibration, diagnostics


def _overall_json(
    detail: pd.DataFrame,
    decisions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    proj_col: str,
    expected_iterations: int,
) -> dict:
    def _row(method: str, table: pd.DataFrame, tier: str | None = None):
        q = table.loc[
            table["method"].eq(method)
            & table["slice_dimension"].eq("ALL")
            & table["slice_value"].eq("ALL")
        ]
        if tier is not None and "tier" in q.columns:
            q = q.loc[q.tier.eq(tier)]
        return q.iloc[0].to_dict() if len(q) else {}

    return {
        "version": "HISTORICAL_PRODUCTION_FAIR_PROBABILITY_V1",
        "projection_mean_authority": proj_col,
        "expected_iterations": int(expected_iterations),
        "matched_rows": int(len(detail)),
        "mean_alignment_max_abs_error": float(num(detail.mean_alignment_abs_error).max()),
        "same_row_side_changes": int(detail.side_changed.sum()),
        "same_row_signal_changes": int(detail.signal_changed.sum()),
        "old_strong": _row("old", decisions, "STRONG_ONLY_PLAY_TIER"),
        "new_strong": _row("new", decisions, "STRONG_ONLY_PLAY_TIER"),
        "old_probability_scores": _row("old", scores),
        "new_probability_scores": _row("new", scores),
        "contract": {
            "sportsbook_downstream_only": True,
            "component_sd_not_used_as_new_outcome_variance": True,
            "same_row_ab": True,
            "mean_alignment_matches_production": True,
            "ensemble_weights_frozen": True,
            "betting_gates_frozen": True,
            "football_model_science_changed": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--proj-col", default="ensemble_proj")
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-iterations", type=int, default=EXPECTED_ITERATIONS)
    a = ap.parse_args()

    proj = pd.concat([pd.read_csv(Path(p)) for p in a.projection_file], ignore_index=True)
    props = pd.read_csv(a.props)
    detail, old_summary, decisions, scores, calibration, diagnostics = grade_empirical_ab(
        proj,
        props,
        a.distribution_dir,
        proj_col=a.proj_col,
        expected_iterations=a.expected_iterations,
    )

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "fair_probability_ab_detail.csv", index=False)
    old_summary.to_csv(a.out_dir / "old_component_translator_summary.csv", index=False)
    decisions.to_csv(a.out_dir / "fair_probability_decision_summary.csv", index=False)
    scores.to_csv(a.out_dir / "fair_probability_scores.csv", index=False)
    calibration.to_csv(a.out_dir / "fair_probability_calibration_bins.csv", index=False)
    diagnostics.to_csv(a.out_dir / "fair_probability_distribution_diagnostics.csv", index=False)

    result = _overall_json(
        detail,
        decisions,
        scores,
        proj_col=a.proj_col,
        expected_iterations=a.expected_iterations,
    )
    with open(a.out_dir / "HISTORICAL_PRODUCTION_FAIR_PROBABILITY_V1.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print("=== HISTORICAL PRODUCTION FAIR PROBABILITY V1 ===")
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
