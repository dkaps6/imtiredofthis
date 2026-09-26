#!/usr/bin/env python3
"""Discrete Count Mean Alignment V1.

Research-only A/B on historical Monte Carlo count distributions.

A0 reproduces current production-style multiplicative mean alignment.
A1 applies one frozen integer-preserving largest-remainder projection of the
same continuously aligned draws.

No model mean, ensemble weight, sportsbook threshold, or decision gate is fit
or changed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import grade as legacy_grade
from scripts.utils.canonical_names import canon_team

KEYS = ["season", "week", "team", "opponent", "player_clean_key", "market"]
COUNT_MARKETS = {"receptions", "rush_att"}
EXPECTED_DRAWS = 2000
ATOL = 1e-12


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _canon(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = sorted(set(KEYS) - set(x.columns))
    if missing:
        raise RuntimeError(f"missing count-distribution identity columns: {missing}")
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    x["market"] = x["market"].astype(str).str.lower()
    if x["team"].eq("").any() or x["opponent"].eq("").any():
        raise RuntimeError("uncanonicalizable team/opponent in count-distribution identity")
    return x


def load_metadata(distribution_dir: Path) -> pd.DataFrame:
    files = sorted(distribution_dir.glob("*_metadata.csv"))
    if not files:
        raise RuntimeError(f"no distribution metadata in {distribution_dir}")
    x = _canon(pd.concat([pd.read_csv(p) for p in files], ignore_index=True))
    x = x.loc[x["market"].isin(COUNT_MARKETS)].copy()
    if x.empty:
        raise RuntimeError("distribution metadata contains zero target count markets")
    if x.duplicated(KEYS).any():
        bad = x.loc[x.duplicated(KEYS, keep=False), KEYS].head(20).to_dict("records")
        raise RuntimeError(f"duplicate count-distribution metadata: {bad}")
    return x


def continuous_align(raw: np.ndarray, target_mean: float) -> np.ndarray:
    x = np.asarray(raw, dtype=float)
    mc_mean = float(np.mean(x)) if len(x) else np.nan
    if np.isfinite(mc_mean) and mc_mean > 0 and np.isfinite(target_mean):
        return x * max(0.0, float(target_mean) / mc_mean)
    return x.copy()


def discrete_largest_remainder(continuous: np.ndarray) -> np.ndarray:
    """Nearest integer-support projection under the frozen largest-remainder rule."""
    z = np.asarray(continuous, dtype=float)
    if not np.isfinite(z).all() or (z < -ATOL).any():
        raise RuntimeError("nonfinite/negative continuously aligned count draw")
    z = np.clip(z, 0.0, None)
    floors = np.floor(z).astype(np.int64)
    desired_total = int(np.rint(float(z.sum())))
    need = desired_total - int(floors.sum())
    if need < 0 or need > len(z):
        raise RuntimeError(
            f"largest-remainder residual out of bounds need={need} draws={len(z)}"
        )
    y = floors.copy()
    if need:
        frac = z - floors
        # Stable descending fractional remainder; original draw index breaks ties.
        order = np.argsort(-frac, kind="mergesort")
        y[order[:need]] += 1
    return y.astype(float)


def empirical_crps(samples: np.ndarray, actual: float) -> float:
    """CRPS for an empirical equal-weight sample in O(n log n)."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0 or not np.isfinite(actual):
        return np.nan
    first = float(np.mean(np.abs(x - float(actual))))
    # 0.5 * E|X-X'| = sum_i (2i-n-1)x_i / n^2 for sorted 1-indexed i.
    coeff = 2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    second = float(np.sum(coeff * x) / (n * n))
    return first - second


def coverage(samples: np.ndarray, actual: float, level: float) -> int:
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [alpha, 1.0 - alpha])
    return int(float(actual) >= float(lo) and float(actual) <= float(hi))


def prob_over(samples: np.ndarray, line: float) -> float:
    x = np.asarray(samples, dtype=float)
    return float(np.mean(x > float(line)))


def binary_scores(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return (
        float(np.mean((p - y) ** 2)),
        float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
    )


def build_count_detail(
    projections: pd.DataFrame,
    *,
    distribution_dir: Path,
    expected_draws: int,
) -> pd.DataFrame:
    p = _canon(projections)
    required = {"ensemble_proj", "mc_proj", "actual"}
    missing = sorted(required - set(p.columns))
    if missing:
        raise RuntimeError(f"projection trace missing required fields: {missing}")
    p = p.loc[p["market"].isin(COUNT_MARKETS)].copy()
    if p.empty:
        raise RuntimeError("projection trace contains zero count-market rows")
    if p.duplicated(KEYS).any():
        bad = p.loc[p.duplicated(KEYS, keep=False), KEYS].head(20).to_dict("records")
        raise RuntimeError(f"duplicate projection count identities: {bad}")

    meta = load_metadata(distribution_dir)
    x = p.merge(
        meta[KEYS + ["array_key", "npz_file", "draws", "mc_mean"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if x["array_key"].isna().any():
        bad = x.loc[x["array_key"].isna(), KEYS].head(20).to_dict("records")
        raise RuntimeError(f"count rows missing exact distribution lineage: {bad}")

    cache: dict[str, object] = {}
    rows: list[dict] = []
    for r in x.itertuples(index=False):
        file_name = str(r.npz_file)
        if Path(file_name).name != file_name:
            raise RuntimeError(f"invalid distribution shard path: {file_name}")
        if file_name not in cache:
            path = distribution_dir / file_name
            if not path.is_file():
                raise RuntimeError(f"missing distribution shard: {path}")
            cache[file_name] = np.load(path, allow_pickle=False)
        raw = np.asarray(cache[file_name][str(r.array_key)], dtype=float)
        if len(raw) != int(expected_draws):
            raise RuntimeError(f"draw-count mismatch {len(raw)} != {expected_draws}")

        raw_integer_gap = float(np.max(np.abs(raw - np.rint(raw)))) if len(raw) else np.inf
        if raw_integer_gap > ATOL:
            raise RuntimeError(
                f"raw count MC not integer-valued for {r.season} W{r.week} "
                f"{r.player_clean_key} {r.market}: {raw_integer_gap}"
            )

        raw_mean = float(np.mean(raw))
        if abs(raw_mean - float(r.mc_proj)) > 1e-8:
            raise RuntimeError(
                f"raw count mean mismatch for {r.player_clean_key} {r.market}: "
                f"{raw_mean} vs {r.mc_proj}"
            )

        target = float(r.ensemble_proj)
        a0 = continuous_align(raw, target)
        a1 = discrete_largest_remainder(a0)

        a0_mean_gap = abs(float(np.mean(a0)) - target)
        a1_mean_gap = abs(float(np.mean(a1)) - target)
        max_allowed = 0.5 / float(expected_draws) + ATOL
        if a0_mean_gap > 1e-8:
            raise RuntimeError(f"A0 failed exact production mean alignment: {a0_mean_gap}")
        if a1_mean_gap > max_allowed:
            raise RuntimeError(
                f"A1 mean alignment exceeds frozen bound: {a1_mean_gap} > {max_allowed}"
            )
        a1_integer_gap = float(np.max(np.abs(a1 - np.rint(a1)))) if len(a1) else np.inf
        if a1_integer_gap > ATOL or (a1 < -ATOL).any():
            raise RuntimeError("A1 violated nonnegative integer support")

        frac_current = float(np.mean(np.abs(a0 - np.rint(a0)) > ATOL))
        actual = float(r.actual)

        row = {k: getattr(r, k) for k in KEYS}
        row.update(
            {
                "actual": actual,
                "mc_proj": raw_mean,
                "target_mean": target,
                "a0_mean": float(np.mean(a0)),
                "a1_mean": float(np.mean(a1)),
                "a0_mean_gap": a0_mean_gap,
                "a1_mean_gap": a1_mean_gap,
                "a0_fractional_draw_rate": frac_current,
                "raw_integer_max_gap": raw_integer_gap,
                "a1_integer_max_gap": a1_integer_gap,
                "a0_crps": empirical_crps(a0, actual),
                "a1_crps": empirical_crps(a1, actual),
                "a0_cov80": coverage(a0, actual, 0.80),
                "a1_cov80": coverage(a1, actual, 0.80),
                "a0_cov90": coverage(a0, actual, 0.90),
                "a1_cov90": coverage(a1, actual, 0.90),
                "npz_file": file_name,
                "array_key": str(r.array_key),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_football(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for market in sorted(detail["market"].unique()):
        for season_label, g in list(detail.loc[detail.market.eq(market)].groupby("season")) + [
            ("POOLED", detail.loc[detail.market.eq(market)])
        ]:
            rows.append(
                {
                    "market": market,
                    "season": season_label,
                    "rows": int(len(g)),
                    "a0_crps": float(g.a0_crps.mean()),
                    "a1_crps": float(g.a1_crps.mean()),
                    "crps_delta_a1_minus_a0": float((g.a1_crps - g.a0_crps).mean()),
                    "a0_cov80": float(g.a0_cov80.mean()),
                    "a1_cov80": float(g.a1_cov80.mean()),
                    "a0_cov80_abs_error": abs(float(g.a0_cov80.mean()) - 0.80),
                    "a1_cov80_abs_error": abs(float(g.a1_cov80.mean()) - 0.80),
                    "a0_cov90": float(g.a0_cov90.mean()),
                    "a1_cov90": float(g.a1_cov90.mean()),
                    "a0_cov90_abs_error": abs(float(g.a0_cov90.mean()) - 0.90),
                    "a1_cov90_abs_error": abs(float(g.a1_cov90.mean()) - 0.90),
                    "mean_fractional_draw_rate_a0": float(g.a0_fractional_draw_rate.mean()),
                    "max_a1_mean_gap": float(g.a1_mean_gap.max()),
                }
            )
    return pd.DataFrame(rows)


def receptions_probability_detail(
    projections: pd.DataFrame,
    props: pd.DataFrame,
    count_detail: pd.DataFrame,
    *,
    distribution_dir: Path,
    expected_draws: int,
) -> pd.DataFrame:
    legacy_detail, _ = legacy_grade(projections, props, proj_col="ensemble_proj")
    if legacy_detail.empty:
        raise RuntimeError("historical market archive produced zero matched rows")
    legacy_detail = _canon(legacy_detail)
    legacy_detail = legacy_detail.loc[legacy_detail["market"].eq("receptions")].copy()
    if legacy_detail.empty:
        raise RuntimeError("historical market archive produced zero matched receptions rows")

    lineage = count_detail.loc[count_detail.market.eq("receptions"), KEYS + [
        "npz_file", "array_key", "target_mean"
    ]]
    x = legacy_detail.merge(lineage, on=KEYS, how="inner", validate="many_to_one")
    if x.empty:
        raise RuntimeError("receptions archived rows did not join count-distribution lineage")

    cache: dict[str, object] = {}
    p0, p1 = [], []
    for r in x.itertuples(index=False):
        file_name = str(r.npz_file)
        if file_name not in cache:
            cache[file_name] = np.load(distribution_dir / file_name, allow_pickle=False)
        raw = np.asarray(cache[file_name][str(r.array_key)], dtype=float)
        if len(raw) != int(expected_draws):
            raise RuntimeError("receptions probability draw-count mismatch")
        a0 = continuous_align(raw, float(r.target_mean))
        a1 = discrete_largest_remainder(a0)
        p0.append(prob_over(a0, float(r.line)))
        p1.append(prob_over(a1, float(r.line)))

    x["a0_p_over"] = p0
    x["a1_p_over"] = p1
    x["actual_over"] = (_num(x["actual"]) > _num(x["line"])).astype(float)
    x["push"] = _num(x["actual"]).eq(_num(x["line"]))
    x["p_over_delta_a1_minus_a0"] = x["a1_p_over"] - x["a0_p_over"]
    x["prob_side_changed"] = (
        (x["a0_p_over"] >= 0.5) != (x["a1_p_over"] >= 0.5)
    ).astype(int)
    return x


def summarize_probability(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season_label, g0 in list(detail.groupby("season")) + [("POOLED", detail)]:
        g = g0.loc[~g0["push"]].copy()
        b0, l0 = binary_scores(g.a0_p_over.to_numpy(float), g.actual_over.to_numpy(float))
        b1, l1 = binary_scores(g.a1_p_over.to_numpy(float), g.actual_over.to_numpy(float))
        rows.append(
            {
                "market": "receptions",
                "season": season_label,
                "rows_nonpush": int(len(g)),
                "a0_brier": b0,
                "a1_brier": b1,
                "brier_delta_a1_minus_a0": b1 - b0,
                "a0_log_loss": l0,
                "a1_log_loss": l1,
                "log_loss_delta_a1_minus_a0": l1 - l0,
                "mean_abs_p_over_change": float(np.mean(np.abs(g.a1_p_over - g.a0_p_over))),
                "prob_side_changed_rows": int(g.prob_side_changed.sum()),
            }
        )
    return pd.DataFrame(rows)


def disposition(football: pd.DataFrame, probability: pd.DataFrame, detail: pd.DataFrame) -> dict:
    def frow(market, season):
        r = football.loc[
            football.market.eq(market) & football.season.astype(str).eq(str(season))
        ]
        if len(r) != 1:
            raise RuntimeError(f"missing summary row {market} {season}")
        return r.iloc[0]

    def prow(season):
        r = probability.loc[probability.season.astype(str).eq(str(season))]
        if len(r) != 1:
            raise RuntimeError(f"missing probability row {season}")
        return r.iloc[0]

    mechanics = {
        "raw_counts_integer": bool(detail.raw_integer_max_gap.max() <= ATOL),
        "candidate_counts_integer": bool(detail.a1_integer_max_gap.max() <= ATOL),
        "candidate_mean_bound": bool(detail.a1_mean_gap.max() <= 0.5 / EXPECTED_DRAWS + ATOL),
        "current_fractional_support_observed": bool(detail.a0_fractional_draw_rate.mean() > 0),
        "sportsbook_inputs_to_football_zero": True,
    }

    science = {
        "pooled_receptions_crps_improves": bool(frow("receptions", "POOLED").a1_crps < frow("receptions", "POOLED").a0_crps),
        "pooled_rush_att_crps_improves": bool(frow("rush_att", "POOLED").a1_crps < frow("rush_att", "POOLED").a0_crps),
        "receptions_2024_crps_nonworse": bool(frow("receptions", 2024).a1_crps <= frow("receptions", 2024).a0_crps),
        "receptions_2025_crps_nonworse": bool(frow("receptions", 2025).a1_crps <= frow("receptions", 2025).a0_crps),
        "rush_att_2024_crps_nonworse": bool(frow("rush_att", 2024).a1_crps <= frow("rush_att", 2024).a0_crps),
        "rush_att_2025_crps_nonworse": bool(frow("rush_att", 2025).a1_crps <= frow("rush_att", 2025).a0_crps),
        "pooled_receptions_brier_improves": bool(prow("POOLED").a1_brier < prow("POOLED").a0_brier),
        "pooled_receptions_logloss_nonworse": bool(prow("POOLED").a1_log_loss <= prow("POOLED").a0_log_loss),
        "receptions_2024_brier_nonworse": bool(prow(2024).a1_brier <= prow(2024).a0_brier),
        "receptions_2025_brier_nonworse": bool(prow(2025).a1_brier <= prow(2025).a0_brier),
    }

    # Frozen plan states six conceptual gates; the season non-worse CRPS gate
    # expands to four market-season checks here for explicit auditability.
    passed = all(mechanics.values()) and all(science.values())
    if passed:
        disp = "DISCRETE_COUNT_MEAN_ALIGNMENT_V1_QUALIFIED_FOR_INTEGRATION_TEST"
    elif mechanics["current_fractional_support_observed"] and all(
        v for k, v in mechanics.items() if k != "current_fractional_support_observed"
    ):
        disp = "COUNT_SUPPORT_CONTRADICTION_CONFIRMED_DISCRETE_REPAIR_FAILED_CLOSED"
    else:
        disp = "DISCRETE_COUNT_MEAN_ALIGNMENT_V1_NOT_INTERPRETABLE"

    return {
        "disposition": disp,
        "mechanical_gates": mechanics,
        "science_gates": science,
        "production_changed": False,
        "week3_outcomes_used": False,
        "sportsbook_inputs_to_football": 0,
        "candidate_variants_scored": 1,
        "parameters_fit": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", type=Path, required=True)
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-draws", type=int, default=EXPECTED_DRAWS)
    a = ap.parse_args()

    projections = pd.read_csv(a.projection_file)
    props = pd.read_csv(a.props)
    detail = build_count_detail(
        projections,
        distribution_dir=a.distribution_dir,
        expected_draws=int(a.expected_draws),
    )
    football = summarize_football(detail)
    prob_detail = receptions_probability_detail(
        projections,
        props,
        detail,
        distribution_dir=a.distribution_dir,
        expected_draws=int(a.expected_draws),
    )
    probability = summarize_probability(prob_detail)
    result = disposition(football, probability, detail)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "discrete_count_alignment_detail.csv", index=False)
    football.to_csv(a.out_dir / "discrete_count_alignment_football_summary.csv", index=False)
    prob_detail.to_csv(a.out_dir / "discrete_count_alignment_receptions_probability_detail.csv", index=False)
    probability.to_csv(a.out_dir / "discrete_count_alignment_receptions_probability_summary.csv", index=False)
    (a.out_dir / "discrete_count_alignment_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== DISCRETE COUNT MEAN ALIGNMENT V1 ===")
    print(football.to_string(index=False))
    print("\n=== RECEPTIONS PROBABILITY ===")
    print(probability.to_string(index=False))
    print("\n=== DISPOSITION ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
