#!/usr/bin/env python3
"""Blind-season TE-R5P receiving-yard distribution-width validation.

Research only. Reconstructs the exact fold-safe production-order TE-R5P
specialist distributions, estimates one football-only width factor on one
historical season, and applies it unchanged to the other season. The football
mean is invariant by construction. Sportsbook lines are secondary evaluation
only and never enter the fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import (
    ev_roi,
    implied_prob,
    no_vig,
    signal,
)
from scripts.backtest.grade_historical_market_vegas_benchmark_v1 import (
    select_one_book_row,
)
from scripts.operations.grade_market_track_record_v1 import (
    american_profit,
    num,
    outcome_side,
)
from scripts.research.grade_empirical_fair_prob_v1 import (
    KEYS,
    _canon_keys,
    empirical_over_probability,
    rescale_outcomes,
)

MARKET = "rec_yards"
POSITION = "TE"
FIT_TEST_DIRECTIONS = ((2024, 2025), (2025, 2024))


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _load_meta(distribution_dir: Path) -> pd.DataFrame:
    paths = sorted(distribution_dir.glob("*_metadata.csv"))
    if not paths:
        raise RuntimeError(f"no metadata files in {distribution_dir}")
    meta = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    meta = _canon_keys(meta)
    meta = meta.loc[meta["market"].eq(MARKET)].copy()
    if meta.duplicated(KEYS).any():
        raise RuntimeError("duplicate TE rec_yards distribution metadata identity")
    return meta


def _row_arrays(meta: pd.DataFrame, distribution_dir: Path) -> dict:
    cache: dict[str, object] = {}
    out: dict[tuple, np.ndarray] = {}
    for _, row in meta.iterrows():
        fn = str(row["npz_file"])
        if Path(fn).name != fn:
            raise RuntimeError(f"invalid shard path: {fn}")
        if fn not in cache:
            path = distribution_dir / fn
            if not path.exists():
                raise RuntimeError(f"missing distribution shard: {path}")
            cache[fn] = np.load(path, allow_pickle=False)
        arr = np.asarray(cache[fn][str(row["array_key"])], dtype=float)
        if len(arr) != int(row["draws"]):
            raise RuntimeError("distribution draw-count drift")
        if len(arr) != 2000:
            raise RuntimeError(f"unexpected TE replay draw count: {len(arr)}")
        if not np.isfinite(arr).all():
            raise RuntimeError("non-finite distribution draw")
        key = tuple(row[c] for c in KEYS)
        out[key] = arr
    return out


def empirical_crps(draws: np.ndarray, actual: float) -> float:
    """Exact CRPS of an equally weighted empirical sample in O(n log n)."""
    x = np.sort(np.asarray(draws, dtype=float))
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(actual):
        return np.nan
    n = len(x)
    first = float(np.mean(np.abs(x - float(actual))))
    # 0.5 E|X-X'| = sum((2i-n-1)*x_i) / n^2 for sorted 1-indexed i.
    i = np.arange(1, n + 1, dtype=float)
    second = float(np.sum((2.0 * i - n - 1.0) * x) / (n * n))
    return first - second


def _aligned(arr: np.ndarray, mean: float) -> np.ndarray:
    out = rescale_outcomes(arr, float(mean))
    delta = abs(float(np.mean(out)) - float(mean))
    if not np.isfinite(delta) or delta > 1e-8:
        raise RuntimeError(f"mean alignment failed: delta={delta}")
    return out


def _widen(arr: np.ndarray, mean: float, k: float) -> np.ndarray:
    out = float(mean) + (arr - float(mean)) * float(k)
    # remove floating drift without changing the declared one-variable transform
    out = out + (float(mean) - float(np.mean(out)))
    delta = abs(float(np.mean(out)) - float(mean))
    if delta > 1e-8:
        raise RuntimeError(f"widened mean drift: {delta}")
    return out


def _prepare_projection(projection_file: Path) -> pd.DataFrame:
    p = _read(projection_file, "specialist projection")
    if "ensemble_proj" not in p.columns:
        raise RuntimeError("specialist projection missing ensemble_proj")
    p["proj"] = num(p["ensemble_proj"])
    p["actual"] = num(p["actual"])
    p["season"] = pd.to_numeric(p["season"], errors="coerce")
    p["week"] = pd.to_numeric(p["week"], errors="coerce")
    p["position"] = p["position"].astype(str).str.upper().str.strip()
    p = p.loc[
        p["position"].eq(POSITION)
        & p["market"].astype(str).eq(MARKET)
        & p["season"].isin([2024, 2025])
        & p["proj"].notna()
        & p["actual"].notna()
    ].copy()
    if p.empty:
        raise RuntimeError("no TE rec_yards specialist rows")
    if p.duplicated(KEYS).any():
        raise RuntimeError("duplicate TE rec_yards projection identity")
    return _canon_keys(p)


def fit_k(
    proj: pd.DataFrame,
    meta: pd.DataFrame,
    arrays: dict,
    fit_season: int,
) -> dict:
    q = proj.loc[proj["season"].eq(fit_season)].merge(
        meta[KEYS + ["array_key", "npz_file", "draws"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    expected = int(proj["season"].eq(fit_season).sum())
    if len(q) != expected or len(q) == 0:
        raise RuntimeError(f"fit distribution coverage mismatch: {len(q)} != {expected}")
    row_sds = []
    residuals = []
    for _, row in q.iterrows():
        key = tuple(row[c] for c in KEYS)
        aligned = _aligned(arrays[key], float(row["proj"]))
        row_sds.append(float(np.std(aligned, ddof=1)))
        residuals.append(float(row["actual"]) - float(row["proj"]))
    mean_sd = float(np.mean(row_sds))
    residual_sd = float(np.std(residuals, ddof=1))
    if not (np.isfinite(mean_sd) and mean_sd > 0 and np.isfinite(residual_sd) and residual_sd > 0):
        raise RuntimeError("invalid fit dispersion")
    k = residual_sd / mean_sd
    return {
        "fit_season": int(fit_season),
        "n": int(len(q)),
        "mean_row_mc_sd": mean_sd,
        "residual_sd": residual_sd,
        "k": float(k),
    }


def evaluate_season(
    proj: pd.DataFrame,
    meta: pd.DataFrame,
    arrays: dict,
    *,
    test_season: int,
    k: float,
) -> tuple[pd.DataFrame, dict]:
    q = proj.loc[proj["season"].eq(test_season)].merge(
        meta[KEYS + ["array_key", "npz_file", "draws"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    expected = int(proj["season"].eq(test_season).sum())
    if len(q) != expected or len(q) == 0:
        raise RuntimeError(f"test distribution coverage mismatch: {len(q)} != {expected}")

    rows = []
    for _, row in q.iterrows():
        key = tuple(row[c] for c in KEYS)
        base = _aligned(arrays[key], float(row["proj"]))
        wide = _widen(base, float(row["proj"]), float(k))
        actual = float(row["actual"])
        base_q05, base_q10, base_q90, base_q95 = np.quantile(base, [0.05, 0.10, 0.90, 0.95])
        wide_q05, wide_q10, wide_q90, wide_q95 = np.quantile(wide, [0.05, 0.10, 0.90, 0.95])
        rows.append({
            **{c: row[c] for c in KEYS},
            "position": POSITION,
            "proj": float(row["proj"]),
            "actual": actual,
            "k": float(k),
            "base_sd": float(np.std(base, ddof=1)),
            "wide_sd": float(np.std(wide, ddof=1)),
            "base_crps": empirical_crps(base, actual),
            "wide_crps": empirical_crps(wide, actual),
            "base_cover80": bool(base_q10 <= actual <= base_q90),
            "wide_cover80": bool(wide_q10 <= actual <= wide_q90),
            "base_cover90": bool(base_q05 <= actual <= base_q95),
            "wide_cover90": bool(wide_q05 <= actual <= wide_q95),
            "base_width80": float(base_q90 - base_q10),
            "wide_width80": float(wide_q90 - wide_q10),
            "base_width90": float(base_q95 - base_q05),
            "wide_width90": float(wide_q95 - wide_q05),
            "base_mean": float(np.mean(base)),
            "wide_mean": float(np.mean(wide)),
        })
    d = pd.DataFrame(rows)
    base_cov80 = float(d["base_cover80"].mean())
    wide_cov80 = float(d["wide_cover80"].mean())
    base_cov90 = float(d["base_cover90"].mean())
    wide_cov90 = float(d["wide_cover90"].mean())
    summary = {
        "test_season": int(test_season),
        "n": int(len(d)),
        "k": float(k),
        "point_mae_base": float((d["proj"] - d["actual"]).abs().mean()),
        "point_mae_wide": float((d["proj"] - d["actual"]).abs().mean()),
        "max_abs_mean_shift": float((d["wide_mean"] - d["base_mean"]).abs().max()),
        "crps_base": float(d["base_crps"].mean()),
        "crps_wide": float(d["wide_crps"].mean()),
        "crps_improvement_pct": float(
            (d["base_crps"].mean() - d["wide_crps"].mean()) / d["base_crps"].mean()
        ),
        "coverage80_base": base_cov80,
        "coverage80_wide": wide_cov80,
        "coverage80_gap_base": abs(base_cov80 - 0.80),
        "coverage80_gap_wide": abs(wide_cov80 - 0.80),
        "coverage90_base": base_cov90,
        "coverage90_wide": wide_cov90,
        "coverage90_gap_base": abs(base_cov90 - 0.90),
        "coverage90_gap_wide": abs(wide_cov90 - 0.90),
        "width80_base": float(d["base_width80"].mean()),
        "width80_wide": float(d["wide_width80"].mean()),
        "width90_base": float(d["base_width90"].mean()),
        "width90_wide": float(d["wide_width90"].mean()),
    }
    return d, summary


def _secondary_market_eval(
    detail: pd.DataFrame,
    meta: pd.DataFrame,
    arrays: dict,
    props: pd.DataFrame,
    *,
    test_season: int,
    k: float,
) -> tuple[pd.DataFrame, dict]:
    p = props.copy()
    p["season"] = pd.to_numeric(p["season"], errors="coerce")
    p = p.loc[p["season"].eq(test_season) & p["market"].astype(str).eq(MARKET)].copy()
    selected = select_one_book_row(p)
    keep = ["game_id", "player_clean_key", "market", "book", "line", "over_odds", "under_odds", "player"]
    q = detail.merge(selected[keep], on=["game_id", "player_clean_key", "market"], how="inner")
    if q.empty:
        raise RuntimeError(f"no historical market matches for TE rec_yards season {test_season}")
    q["line"] = num(q["line"])
    p_base, p_wide = [], []
    for _, row in q.iterrows():
        key = tuple(row[c] for c in KEYS)
        base = _aligned(arrays[key], float(row["proj"]))
        wide = _widen(base, float(row["proj"]), float(k))
        p_base.append(empirical_over_probability(base, float(row["line"])))
        p_wide.append(empirical_over_probability(wide, float(row["line"])))

    outputs = []
    summaries = {}
    for label, p_over in (("base", p_base), ("wide", p_wide)):
        z = q.copy()
        z["p_over"] = p_over
        z["p_under"] = 1.0 - z["p_over"]
        y = (num(z["actual"]) > num(z["line"])).astype(float)
        pp = np.clip(num(z["p_over"]), 1e-6, 1 - 1e-6)
        z["over_implied"] = z["over_odds"].map(implied_prob)
        z["under_implied"] = z["under_odds"].map(implied_prob)
        z["over_novig"] = [no_vig(a, b) for a, b in zip(z["over_implied"], z["under_implied"])]
        z["under_novig"] = [no_vig(a, b) for a, b in zip(z["under_implied"], z["over_implied"])]
        z["ev_over"] = [ev_roi(p, o) for p, o in zip(z["p_over"], z["over_odds"])]
        z["ev_under"] = [ev_roi(p, o) for p, o in zip(z["p_under"], z["under_odds"])]
        best_over = z["ev_under"].isna() | (z["ev_over"].fillna(-np.inf) >= z["ev_under"].fillna(-np.inf))
        z["side"] = np.where(best_over, "OVER", "UNDER")
        z["best_ev"] = np.where(best_over, z["ev_over"], z["ev_under"])
        z["best_model_p"] = np.where(best_over, z["p_over"], z["p_under"])
        z["best_market_p"] = np.where(best_over, z["over_novig"], z["under_novig"])
        z["prob_edge"] = z["best_model_p"] - z["best_market_p"]
        z["chosen_odds"] = np.where(best_over, z["over_odds"], z["under_odds"])
        z["signal"] = [signal(e, pe) for e, pe in zip(z["best_ev"], z["prob_edge"])]
        z["actual_side"] = [outcome_side(a, l) for a, l in zip(z["actual"], z["line"])]
        z["bet_result"] = np.select(
            [z["actual_side"].eq("PUSH"), z["side"].eq(z["actual_side"])],
            ["PUSH", "WIN"],
            default="LOSS",
        )
        z["unit_result"] = np.where(
            z["bet_result"].eq("WIN"),
            [american_profit(o) for o in z["chosen_odds"]],
            np.where(z["bet_result"].eq("LOSS"), -1.0, 0.0),
        )
        strong = z.loc[z["signal"].eq("STRONG_EDGE") & z["bet_result"].isin(["WIN", "LOSS"])]
        summaries[label] = {
            "rows": int(len(z)),
            "brier": float(np.mean((pp - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(pp) + (1-y) * np.log(1-pp))),
            "strong_rows": int(len(strong)),
            "strong_win_rate": float(strong["bet_result"].eq("WIN").mean()) if len(strong) else np.nan,
            "strong_roi": float(strong["unit_result"].mean()) if len(strong) else np.nan,
        }
        z["variant"] = label
        outputs.append(z)
    return pd.concat(outputs, ignore_index=True), summaries


def run(projection_file: Path, distribution_dir: Path, props_file: Path, out_dir: Path) -> dict:
    proj = _prepare_projection(projection_file)
    meta = _load_meta(distribution_dir)
    meta = meta.loc[meta["season"].isin([2024, 2025])].copy()
    arrays = _row_arrays(meta, distribution_dir)
    props = _read(props_file, "historical props")

    fits = {season: fit_k(proj, meta, arrays, season) for season in (2024, 2025)}
    evaluations = {}
    market_eval = {}
    all_detail = []
    all_market = []

    for fit_season, test_season in FIT_TEST_DIRECTIONS:
        d, summary = evaluate_season(
            proj, meta, arrays, test_season=test_season, k=fits[fit_season]["k"]
        )
        d["fit_season"] = fit_season
        d["test_season"] = test_season
        all_detail.append(d)
        m, ms = _secondary_market_eval(
            d, meta, arrays, props, test_season=test_season, k=fits[fit_season]["k"]
        )
        m["fit_season"] = fit_season
        m["test_season"] = test_season
        all_market.append(m)
        evaluations[f"fit{fit_season}_test{test_season}"] = summary
        market_eval[f"fit{fit_season}_test{test_season}"] = ms

    pooled_market_base = []
    pooled_market_wide = []
    for key, v in market_eval.items():
        # Weight Brier/log loss by matched rows in each blind direction.
        for variant, bucket in (("base", pooled_market_base), ("wide", pooled_market_wide)):
            bucket.append((v[variant]["rows"], v[variant]["brier"], v[variant]["log_loss"]))
    def weighted(bucket, idx):
        n = sum(x[0] for x in bucket)
        return sum(x[0] * x[idx] for x in bucket) / n
    pooled = {
        "matched_rows": int(sum(x[0] for x in pooled_market_base)),
        "brier_base": float(weighted(pooled_market_base, 1)),
        "brier_wide": float(weighted(pooled_market_wide, 1)),
        "log_loss_base": float(weighted(pooled_market_base, 2)),
        "log_loss_wide": float(weighted(pooled_market_wide, 2)),
    }

    gates = {
        "point_mae_invariant_both_directions": all(
            abs(v["point_mae_wide"] - v["point_mae_base"]) <= 1e-10
            for v in evaluations.values()
        ),
        "mean_shift_le_1e_8_both_directions": all(
            v["max_abs_mean_shift"] <= 1e-8 for v in evaluations.values()
        ),
        "crps_strict_improve_both_directions": all(
            v["crps_wide"] < v["crps_base"] for v in evaluations.values()
        ),
        "coverage80_gap_improve_both_directions": all(
            v["coverage80_gap_wide"] < v["coverage80_gap_base"]
            for v in evaluations.values()
        ),
        "coverage90_gap_improve_both_directions": all(
            v["coverage90_gap_wide"] < v["coverage90_gap_base"]
            for v in evaluations.values()
        ),
        "pooled_brier_nonworse": bool(pooled["brier_wide"] <= pooled["brier_base"] + 1e-12),
        "pooled_log_loss_nonworse": bool(
            pooled["log_loss_wide"] <= pooled["log_loss_base"] + 1e-12
        ),
        "sportsbook_inputs_used_to_fit_k": 0,
    }
    gates["qualified"] = bool(
        all(v for k, v in gates.items() if k != "sportsbook_inputs_used_to_fit_k")
    )

    # This is the predeclared future-only factor if blind qualification passes.
    row_sds = []
    residuals = []
    for _, row in proj.iterrows():
        key = tuple(row[c] for c in KEYS)
        a = _aligned(arrays[key], float(row["proj"]))
        row_sds.append(float(np.std(a, ddof=1)))
        residuals.append(float(row["actual"]) - float(row["proj"]))
    future_k = float(np.std(residuals, ddof=1) / np.mean(row_sds))

    result = {
        "study": "TE_R5P_REC_YARDS_WIDTH_V2",
        "status": "research_only",
        "production_changed": False,
        "sportsbook_inputs_used_to_fit_k": 0,
        "fit": fits,
        "blind_evaluations": evaluations,
        "secondary_market_calibration": market_eval,
        "pooled_secondary_market_calibration": pooled,
        "gates": gates,
        "future_k_if_qualified": future_k,
        "disposition": (
            "TE_R5P_REC_YARDS_WIDTH_V2_QUALIFIED"
            if gates["qualified"]
            else "TE_R5P_REC_YARDS_WIDTH_V2_FAILED_CLOSED"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_detail, ignore_index=True).to_csv(out_dir / "blind_row_detail.csv", index=False)
    pd.concat(all_market, ignore_index=True).to_csv(out_dir / "blind_market_detail.csv", index=False)
    pd.DataFrame(fits.values()).to_csv(out_dir / "fit_factors.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# TE-R5P Receiving-Yards Width V2 — Result",
        "",
        f"**Disposition:** `{result['disposition']}`",
        "",
        "Research only. No production change.",
        "",
        "## Fit factors",
        "",
    ]
    for season in (2024, 2025):
        x = fits[season]
        lines.append(
            f"- {season}: n={x['n']}, mean MC SD={x['mean_row_mc_sd']:.3f}, "
            f"residual SD={x['residual_sd']:.3f}, k=**{x['k']:.4f}**"
        )
    lines += ["", "## Blind directions", ""]
    for key, v in evaluations.items():
        lines += [
            f"### {key}",
            f"- n: **{v['n']}**",
            f"- CRPS: **{v['crps_base']:.4f} -> {v['crps_wide']:.4f}** "
            f"({v['crps_improvement_pct']*100:+.2f}%)",
            f"- 80% coverage: **{v['coverage80_base']:.3f} -> {v['coverage80_wide']:.3f}** "
            f"(gap {v['coverage80_gap_base']:.3f} -> {v['coverage80_gap_wide']:.3f})",
            f"- 90% coverage: **{v['coverage90_base']:.3f} -> {v['coverage90_wide']:.3f}** "
            f"(gap {v['coverage90_gap_base']:.3f} -> {v['coverage90_gap_wide']:.3f})",
            f"- point MAE invariant: **{v['point_mae_base']:.4f}**",
            f"- max mean shift: **{v['max_abs_mean_shift']:.3g}**",
            "",
        ]
    lines += [
        "## Secondary historical-line calibration",
        "",
        f"- pooled Brier: **{pooled['brier_base']:.5f} -> {pooled['brier_wide']:.5f}**",
        f"- pooled log loss: **{pooled['log_loss_base']:.5f} -> {pooled['log_loss_wide']:.5f}**",
        "",
        "## Frozen gates",
        "",
    ]
    for k, v in gates.items():
        lines.append(f"- {k}: **{v}**")
    lines += [
        "",
        f"Predeclared future-only pooled factor if qualified: **{future_k:.4f}**.",
        "",
        "No live 2026 outcome was used to fit k. Any production integration requires a separate validation.",
    ]
    (out_dir / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection-file", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    run(a.projection_file, a.distribution_dir, a.props, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
