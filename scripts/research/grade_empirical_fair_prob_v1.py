#!/usr/bin/env python3
"""Same-row A/B test of historical fair-probability translators.

The legacy arm is the frozen Normal(mean=projection, sd=component_sd) benchmark.
The empirical arm uses the exact historical Monte Carlo outcome arrays rebuilt
without sportsbook inputs, rescales each array to the same frozen football mean
with production semantics, then computes P(outcome > line) empirically.

No model mean, ensemble weight, sportsbook threshold, or decision gate is fit or
changed in this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import (
    ev_roi,
    grade as legacy_grade,
    implied_prob,
    no_vig,
    signal,
)
from scripts.operations.grade_market_track_record_v1 import american_profit, num, outcome_side
from scripts.utils.canonical_names import canon_team

# Opponent is intentionally part of the distribution-lineage key.  Team/week
# functionally identifies it in the authoritative schedule, but carrying it here
# makes a stale or wrong-opponent sidecar fail closed instead of joining anyway.
KEYS = ["season", "week", "team", "opponent", "player_clean_key", "market"]
EXPECTED_DRAWS = 2000


def rescale_outcomes(outcomes: np.ndarray, target_mean: float) -> np.ndarray:
    base = np.asarray(outcomes, dtype=float)
    mc_mean = float(np.mean(base)) if len(base) else np.nan
    if np.isfinite(mc_mean) and mc_mean > 0 and np.isfinite(target_mean):
        return base * max(0.0, float(target_mean) / mc_mean)
    return base


def empirical_over_probability(outcomes: np.ndarray, line: float) -> float:
    arr = np.asarray(outcomes, dtype=float)
    if not len(arr) or not np.isfinite(float(line)):
        return np.nan
    return float(np.mean(arr > float(line)))


def _canon_keys(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    required = set(KEYS)
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"distribution identity missing columns: {missing}")
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    x["market"] = x["market"].astype(str).str.lower()
    if x["team"].eq("").any() or x["opponent"].eq("").any():
        raise RuntimeError("distribution identity contains uncanonicalizable team/opponent")
    return x


def _load_metadata(distribution_dir: Path) -> pd.DataFrame:
    files = sorted(distribution_dir.glob("*_metadata.csv"))
    if not files:
        raise RuntimeError(f"no distribution metadata shards in {distribution_dir}")
    x = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    x = _canon_keys(x)
    if x.duplicated(KEYS).any():
        bad = x.loc[x.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")
        raise RuntimeError(f"duplicate distribution lineage rows: {bad}")
    return x


def _summarize(z: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict] = []
    tiers = {
        "ALL_NO_FILTER": z,
        "LEAN_OR_STRONG": z.loc[z.signal.isin(["LEAN_EDGE", "STRONG_EDGE"])],
        "STRONG_ONLY_PLAY_TIER": z.loc[z.signal.eq("STRONG_EDGE")],
    }
    scope = list(z.market.unique()) + ["ALL_MARKETS"]
    for market in scope:
        for tier_name, tier_df in tiers.items():
            g = tier_df if market == "ALL_MARKETS" else tier_df.loc[tier_df.market.eq(market)]
            decided = g.loc[g.bet_result.isin(["WIN", "LOSS"]) & num(g.chosen_odds).notna()]
            summaries.append(
                {
                    "market": market,
                    "tier": tier_name,
                    "matched_rows": int(len(g)),
                    "decided_bets": int(len(decided)),
                    "wins": int(decided.bet_result.eq("WIN").sum()),
                    "losses": int(decided.bet_result.eq("LOSS").sum()),
                    "win_rate": float(decided.bet_result.eq("WIN").mean()) if len(decided) else np.nan,
                    "units": float(decided.unit_result.sum()) if len(decided) else np.nan,
                    "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
                    "model_mae": float(g.model_error.abs().mean()) if len(g) else np.nan,
                    "vegas_mae": float(g.vegas_error.abs().mean()) if len(g) else np.nan,
                }
            )
    return pd.DataFrame(summaries)


def _assert_same_base_cohort(comparison: pd.DataFrame) -> None:
    """Require identical unfiltered rows while allowing the translator to move tiers.

    LEAN/STRONG membership is an experimental outcome, not an identity invariant.
    Requiring equal tier counts would make the experiment fail precisely when the
    new probability translator changes calibration/selectivity as intended.
    """
    base = comparison.loc[comparison["tier"].eq("ALL_NO_FILTER")]
    if base.empty:
        raise RuntimeError("same-row A/B contract violated: missing ALL_NO_FILTER cohort")
    equal = (
        base["matched_rows_legacy"].astype(int)
        == base["matched_rows_empirical"].astype(int)
    )
    if not equal.all():
        bad = base.loc[
            ~equal,
            ["market", "matched_rows_legacy", "matched_rows_empirical"],
        ].to_dict("records")
        raise RuntimeError(f"same-row A/B contract violated in unfiltered cohort: {bad}")


def _probability_diagnostics(detail: pd.DataFrame, translator: str) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = list(detail.market.unique()) + ["ALL_MARKETS"]
    for market in scopes:
        g = detail if market == "ALL_MARKETS" else detail.loc[detail.market.eq(market)]
        decided = g.loc[g.actual_side.ne("PUSH")].copy()
        if len(decided):
            y = (num(decided.actual) > num(decided.line)).astype(float).to_numpy()
            p = np.clip(num(decided.p_over).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            brier = float(np.mean((p - y) ** 2))
            log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            brier = log_loss = np.nan
        sd_col = "model_sd" if "model_sd" in g.columns else "component_sd"
        rows.append(
            {
                "translator": translator,
                "market": market,
                "matched_rows": int(len(g)),
                "strong_rows": int(g.signal.eq("STRONG_EDGE").sum()),
                "strong_coverage": float(g.signal.eq("STRONG_EDGE").mean()) if len(g) else np.nan,
                "brier_over": brier,
                "log_loss_over": log_loss,
                "mean_translator_sd": float(num(g[sd_col]).mean()) if len(g) and sd_col in g else np.nan,
                "residual_std": float((num(g.actual) - num(g.proj)).std(ddof=1)) if len(g) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _calibration_bins(detail: pd.DataFrame, translator: str) -> pd.DataFrame:
    rows: list[dict] = []
    x = detail.loc[detail.actual_side.ne("PUSH")].copy()
    x["p_over_num"] = num(x.p_over).clip(0.0, 1.0)
    x["actual_over"] = (num(x.actual) > num(x.line)).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    x["prob_bin"] = pd.cut(x["p_over_num"], bins=edges, include_lowest=True, duplicates="drop")
    for market in list(x.market.unique()) + ["ALL_MARKETS"]:
        g = x if market == "ALL_MARKETS" else x.loc[x.market.eq(market)]
        for bucket, b in g.groupby("prob_bin", observed=True):
            rows.append(
                {
                    "translator": translator,
                    "market": market,
                    "prob_bin": str(bucket),
                    "rows": int(len(b)),
                    "mean_predicted_over": float(b.p_over_num.mean()),
                    "realized_over_rate": float(b.actual_over.mean()),
                    "calibration_error": float(b.p_over_num.mean() - b.actual_over.mean()),
                }
            )
    return pd.DataFrame(rows)


def grade_empirical(
    proj: pd.DataFrame,
    props: pd.DataFrame,
    *,
    distribution_dir: Path,
    proj_col: str = "ensemble_proj",
    expected_draws: int = EXPECTED_DRAWS,
):
    legacy_detail, legacy_summary = legacy_grade(proj, props, proj_col=proj_col)
    if legacy_detail.empty:
        raise RuntimeError("legacy benchmark produced no matched rows")

    # Preserve the OLD arm row-level values before NEW overwrites probability,
    # side, signal, and unit-result columns. These deltas are required to isolate
    # the translator on identical rows rather than only compare aggregate tables.
    detail = _canon_keys(legacy_detail)
    detail["legacy_p_over"] = num(detail["p_over"])
    detail["legacy_p_under"] = num(detail["p_under"])
    detail["legacy_side"] = detail["side"].astype(str)
    detail["legacy_signal"] = detail["signal"].astype(str)
    detail["legacy_best_ev"] = num(detail["best_ev"])
    detail["legacy_prob_edge"] = num(detail["prob_edge"])
    detail["legacy_unit_result"] = num(detail["unit_result"])
    detail["legacy_component_sd"] = num(detail["component_sd"])

    meta = _load_metadata(distribution_dir)
    detail = detail.merge(
        meta[KEYS + ["array_key", "npz_file", "draws", "mc_mean", "mc_sd"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if len(detail) != len(legacy_detail):
        raise RuntimeError(
            f"same-row A/B contract violated by distribution join: "
            f"legacy={len(legacy_detail)} joined={len(detail)}"
        )
    missing = int(detail["array_key"].isna().sum())
    if missing:
        sample = detail.loc[detail["array_key"].isna(), KEYS].head(10).to_dict("records")
        raise RuntimeError(f"{missing} graded rows lack defensible simulation lineage: {sample}")

    draws = pd.to_numeric(detail["draws"], errors="raise").astype(int)
    if not draws.eq(int(expected_draws)).all():
        bad = sorted(draws.loc[~draws.eq(int(expected_draws))].unique().tolist())
        raise RuntimeError(
            f"empirical reconstruction iteration policy violated: "
            f"expected={int(expected_draws)} found={bad}"
        )

    cache: dict[str, object] = {}
    p_over: list[float] = []
    model_sd: list[float] = []
    aligned_mean: list[float] = []
    mean_delta: list[float] = []

    for _, row in detail.iterrows():
        file_name = str(row["npz_file"])
        if Path(file_name).name != file_name:
            raise RuntimeError(f"invalid distribution shard path in metadata: {file_name}")
        if file_name not in cache:
            path = distribution_dir / file_name
            if not path.exists():
                raise RuntimeError(f"missing distribution shard: {path}")
            cache[file_name] = np.load(path, allow_pickle=False)
        arr = np.asarray(cache[file_name][str(row["array_key"])], dtype=float)

        if len(arr) != int(expected_draws):
            raise RuntimeError(
                f"distribution draw-count mismatch {row['season']} W{row['week']} "
                f"{row['team']} {row['player_clean_key']} {row['market']}: "
                f"{len(arr)} != {int(expected_draws)}"
            )
        stored_mean = float(np.mean(arr))
        expected_mc = float(row["mc_proj"])
        delta = abs(stored_mean - expected_mc)
        if not np.isfinite(delta) or delta > 1e-8:
            raise RuntimeError(
                f"distribution/MC mean mismatch {row['season']} W{row['week']} "
                f"{row['team']} {row['player_clean_key']} {row['market']}: {delta}"
            )

        adjusted = rescale_outcomes(arr, float(row["proj"]))
        p_over.append(empirical_over_probability(adjusted, float(row["line"])))
        # Production run_pricing_v2 records sample SD (ddof=1).
        model_sd.append(float(np.std(adjusted, ddof=1)) if len(adjusted) > 1 else 0.0)
        aligned_mean.append(float(np.mean(adjusted)))
        mean_delta.append(abs(float(np.mean(adjusted)) - float(row["proj"])))

    detail["p_over"] = p_over
    detail["p_under"] = 1.0 - detail["p_over"]
    detail["model_sd"] = model_sd
    detail["aligned_distribution_mean"] = aligned_mean
    detail["aligned_mean_abs_delta"] = mean_delta
    if float(detail["aligned_mean_abs_delta"].max()) > 1e-8:
        raise RuntimeError(
            f"production-style mean alignment failed max_abs={detail['aligned_mean_abs_delta'].max()}"
        )

    detail["over_implied"] = detail.over_odds.map(implied_prob)
    detail["under_implied"] = detail.under_odds.map(implied_prob)
    detail["over_novig"] = [no_vig(a, b) for a, b in zip(detail.over_implied, detail.under_implied)]
    detail["under_novig"] = [no_vig(a, b) for a, b in zip(detail.under_implied, detail.over_implied)]
    detail["ev_over"] = [ev_roi(p, o) for p, o in zip(detail.p_over, detail.over_odds)]
    detail["ev_under"] = [ev_roi(p, o) for p, o in zip(detail.p_under, detail.under_odds)]

    best_over = detail.ev_under.isna() | (
        detail.ev_over.fillna(-np.inf) >= detail.ev_under.fillna(-np.inf)
    )
    detail["side"] = np.where(best_over, "OVER", "UNDER")
    detail["best_ev"] = np.where(best_over, detail.ev_over, detail.ev_under)
    detail["best_model_p"] = np.where(best_over, detail.p_over, detail.p_under)
    detail["best_market_p"] = np.where(best_over, detail.over_novig, detail.under_novig)
    detail["prob_edge"] = detail.best_model_p - detail.best_market_p
    detail["chosen_odds"] = np.where(best_over, detail.over_odds, detail.under_odds)
    detail["signal"] = [signal(e, q) for e, q in zip(detail.best_ev, detail.prob_edge)]

    detail["actual_side"] = [outcome_side(a, l) for a, l in zip(detail.actual, detail.line)]
    detail["bet_result"] = np.select(
        [detail.actual_side.eq("PUSH"), detail.side.eq(detail.actual_side)],
        ["PUSH", "WIN"],
        default="LOSS",
    )
    detail["unit_result"] = np.where(
        detail.bet_result.eq("WIN"),
        [american_profit(o) for o in detail.chosen_odds],
        np.where(detail.bet_result.eq("LOSS"), -1.0, 0.0),
    )
    detail["model_error"] = num(detail.proj) - num(detail.actual)
    detail["vegas_error"] = num(detail.line) - num(detail.actual)
    detail["translator"] = "EMPIRICAL_MC_RESCALED_V1"

    detail["delta_p_over_empirical_minus_legacy"] = detail["p_over"] - detail["legacy_p_over"]
    detail["delta_best_ev_empirical_minus_legacy"] = detail["best_ev"] - detail["legacy_best_ev"]
    detail["delta_prob_edge_empirical_minus_legacy"] = detail["prob_edge"] - detail["legacy_prob_edge"]
    detail["delta_unit_result_empirical_minus_legacy"] = detail["unit_result"] - detail["legacy_unit_result"]
    detail["side_changed"] = detail["side"].ne(detail["legacy_side"]).astype(int)
    detail["signal_changed"] = detail["signal"].ne(detail["legacy_signal"]).astype(int)

    empirical_summary = _summarize(detail)
    comparison = legacy_summary.merge(
        empirical_summary,
        on=["market", "tier"],
        how="inner",
        suffixes=("_legacy", "_empirical"),
        validate="one_to_one",
    )
    _assert_same_base_cohort(comparison)
    comparison["coverage_row_delta_empirical_minus_legacy"] = (
        comparison["matched_rows_empirical"] - comparison["matched_rows_legacy"]
    )
    comparison["roi_delta_empirical_minus_legacy"] = (
        comparison["roi_per_unit_empirical"] - comparison["roi_per_unit_legacy"]
    )
    comparison["win_rate_delta_empirical_minus_legacy"] = (
        comparison["win_rate_empirical"] - comparison["win_rate_legacy"]
    )

    legacy_diag = _probability_diagnostics(legacy_detail, "LEGACY_COMPONENT_SD_NORMAL")
    empirical_diag = _probability_diagnostics(detail, "EMPIRICAL_MC_RESCALED_V1")
    diagnostics = pd.concat([legacy_diag, empirical_diag], ignore_index=True)
    bins = pd.concat(
        [
            _calibration_bins(legacy_detail, "LEGACY_COMPONENT_SD_NORMAL"),
            _calibration_bins(detail, "EMPIRICAL_MC_RESCALED_V1"),
        ],
        ignore_index=True,
    )

    alignment = detail[
        KEYS
        + [
            "game_id",
            "proj",
            "mc_proj",
            "mc_mean",
            "mc_sd",
            "model_sd",
            "aligned_distribution_mean",
            "aligned_mean_abs_delta",
            "draws",
            "npz_file",
            "array_key",
        ]
    ].copy()
    return detail, empirical_summary, legacy_summary, comparison, diagnostics, bins, alignment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--proj-col", default="ensemble_proj")
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-draws", type=int, default=EXPECTED_DRAWS)
    a = ap.parse_args()

    proj = pd.concat([pd.read_csv(Path(p)) for p in a.projection_file], ignore_index=True)
    props = pd.read_csv(a.props)
    (
        detail,
        empirical_summary,
        legacy_summary,
        comparison,
        diagnostics,
        bins,
        alignment,
    ) = grade_empirical(
        proj,
        props,
        distribution_dir=a.distribution_dir,
        proj_col=a.proj_col,
        expected_draws=a.expected_draws,
    )

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "empirical_fair_prob_detail.csv", index=False)
    empirical_summary.to_csv(a.out_dir / "empirical_fair_prob_summary.csv", index=False)
    legacy_summary.to_csv(a.out_dir / "legacy_component_sd_summary.csv", index=False)
    comparison.to_csv(a.out_dir / "translator_comparison_summary.csv", index=False)
    diagnostics.to_csv(a.out_dir / "probability_diagnostics.csv", index=False)
    bins.to_csv(a.out_dir / "calibration_bins.csv", index=False)
    alignment.to_csv(a.out_dir / "distribution_alignment_audit.csv", index=False)

    print("=== EMPIRICAL FAIR-PROBABILITY SAME-ROW A/B ===")
    print(comparison.to_string(index=False))
    print("\n=== PROBABILITY DIAGNOSTICS ===")
    print(diagnostics.to_string(index=False))
    print(
        "\n=== SAME-ROW CHANGE COUNTS ===\n"
        f"rows={len(detail)} side_changed={int(detail['side_changed'].sum())} "
        f"signal_changed={int(detail['signal_changed'].sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
