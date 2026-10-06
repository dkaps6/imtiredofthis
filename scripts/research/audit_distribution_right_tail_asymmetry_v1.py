#!/usr/bin/env python3
"""Distribution Right-Tail Asymmetry V1.

Frozen in docs/research/DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_V1_PLAN.md.

Uses football-only historical projection/distribution authority. Sportsbook data
is prohibited. No mean, probability, threshold or production parameter changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.grade_empirical_fair_prob_v1 import (
    KEYS,
    _load_metadata,
    rescale_outcomes,
)

VERSION = "DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_V1"
MARKETS = ("pass_yards", "rush_yards", "rec_yards", "receptions", "rush_rec_yards")
EXPECTED_DRAWS = 2000
MIN_ROWS = 200
MIN_GAMES = 50
REPS = 10_000
SEED = 20261006


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _cluster_bootstrap(q: pd.DataFrame) -> dict:
    grouped = q.groupby("game_id", sort=True)
    games = list(grouped.groups)
    if len(games) < 2:
        return {"valid_reps": 0, "ci_low": None, "ci_high": None}

    stats = []
    for gid, z in grouped:
        stats.append([
            float(z["upper_break_10"].sum()),
            float(z["lower_break_10"].sum()),
            float(len(z)),
        ])
    a = np.asarray(stats, dtype=float)

    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(
        len(games),
        [1.0 / len(games)] * len(games),
        size=REPS,
    ).astype(float)

    upper = counts @ a[:, 0]
    lower = counts @ a[:, 1]
    n = counts @ a[:, 2]
    valid = n > 0
    diff = (upper[valid] - lower[valid]) / n[valid]
    return {
        "valid_reps": int(valid.sum()),
        "ci_low": float(np.quantile(diff, 0.025)),
        "ci_high": float(np.quantile(diff, 0.975)),
        "p_diff_gt_0": float((diff > 0).mean()),
    }


def _cell(q: pd.DataFrame, season: int, market: str) -> dict:
    support = (
        len(q) >= MIN_ROWS
        and q["game_id"].nunique() >= MIN_GAMES
        and q["actual"].notna().all()
    )
    rec = {
        "season": int(season),
        "market": market,
        "rows": int(len(q)),
        "games": int(q["game_id"].nunique()),
        "support": "PASS" if support else "INSUFFICIENT_SUPPORT",
    }
    if not support:
        rec["cell_pass"] = False
        return rec

    upper10 = float(q["upper_break_10"].mean())
    lower10 = float(q["lower_break_10"].mean())
    upper05 = float(q["upper_break_05"].mean())
    lower05 = float(q["lower_break_05"].mean())
    boot = _cluster_bootstrap(q)
    ci_low = boot["ci_low"]
    passed = bool(
        upper10 > lower10
        and ci_low is not None
        and np.isfinite(ci_low)
        and ci_low > 0
    )
    rec.update({
        "upper_break_rate_q90": upper10,
        "lower_break_rate_q10": lower10,
        "upper_minus_lower_q10q90": upper10 - lower10,
        "upper_break_rate_q95": upper05,
        "lower_break_rate_q05": lower05,
        "upper_minus_lower_q05q95": upper05 - lower05,
        "bootstrap_reps": REPS,
        "bootstrap_valid_reps": boot["valid_reps"],
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": boot["ci_high"],
        "bootstrap_p_diff_gt_0": boot["p_diff_gt_0"],
        "cell_pass": passed,
    })
    return rec


def classify(cells: pd.DataFrame) -> dict:
    per_market = []
    for market in MARKETS:
        q = cells.loc[cells["market"].eq(market)]
        by = {int(r.season): bool(r.cell_pass) for r in q.itertuples()}
        replicated = by.get(2024, False) and by.get(2025, False)
        per_market.append({
            "market": market,
            "season_2024_pass": bool(by.get(2024, False)),
            "season_2025_pass": bool(by.get(2025, False)),
            "disposition": (
                "RIGHT_TAIL_ASYMMETRY_REPLICATED"
                if replicated
                else "RIGHT_TAIL_ASYMMETRY_NOT_REPLICATED"
            ),
        })
    any_rep = any(x["disposition"] == "RIGHT_TAIL_ASYMMETRY_REPLICATED" for x in per_market)
    return {
        "disposition": (
            "DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_SIGNAL_CONFIRMED"
            if any_rep
            else "DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_NULL"
        ),
        "per_market": per_market,
    }


def build_detail(
    projection: pd.DataFrame,
    distribution_dir: Path,
    *,
    expected_draws: int = EXPECTED_DRAWS,
) -> pd.DataFrame:
    required = set(KEYS + ["actual", "ensemble_proj", "mc_proj", "game_id"])
    missing = sorted(required - set(projection.columns))
    if missing:
        raise RuntimeError(f"projection trace missing required columns: {missing}")

    x = projection.loc[projection["market"].astype(str).str.lower().isin(MARKETS)].copy()
    x["market"] = x["market"].astype(str).str.lower()
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["actual"] = pd.to_numeric(x["actual"], errors="coerce")
    x["ensemble_proj"] = pd.to_numeric(x["ensemble_proj"], errors="coerce")
    x["mc_proj"] = pd.to_numeric(x["mc_proj"], errors="coerce")
    x = x.loc[
        x["season"].isin([2024, 2025])
        & x["actual"].notna()
        & x["ensemble_proj"].notna()
        & x["mc_proj"].notna()
    ].copy()
    if x.empty:
        raise RuntimeError("no scoreable 2024/2025 projection rows")
    if x.duplicated(KEYS).any():
        raise RuntimeError("projection trace contains duplicate distribution identities")

    meta = _load_metadata(distribution_dir)
    z = x.merge(
        meta[KEYS + ["array_key", "npz_file", "draws", "mc_mean"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if z["array_key"].isna().any():
        sample = z.loc[z["array_key"].isna(), KEYS].head(20).to_dict("records")
        raise RuntimeError(f"scoreable rows missing distribution lineage: {sample}")

    cache = {}
    rows = []
    for r in z.itertuples(index=False):
        draws = int(r.draws)
        if draws != expected_draws:
            raise RuntimeError(f"draw-count mismatch {r.season} W{r.week} {r.player_clean_key} {r.market}: {draws}")
        fname = str(r.npz_file)
        if Path(fname).name != fname:
            raise RuntimeError(f"unsafe distribution shard path: {fname}")
        if fname not in cache:
            p = distribution_dir / fname
            if not p.exists():
                raise RuntimeError(f"missing distribution shard: {p}")
            cache[fname] = np.load(p, allow_pickle=False)
        arr = np.asarray(cache[fname][str(r.array_key)], dtype=float)
        if len(arr) != expected_draws or not np.isfinite(arr).all():
            raise RuntimeError(f"invalid simulated array {r.season} W{r.week} {r.player_clean_key} {r.market}")

        stored_mean = float(np.mean(arr))
        if abs(stored_mean - float(r.mc_proj)) > 1e-8:
            raise RuntimeError(
                f"MC lineage mismatch {r.season} W{r.week} {r.player_clean_key} {r.market}: "
                f"{stored_mean} vs {r.mc_proj}"
            )

        adj = rescale_outcomes(arr, float(r.ensemble_proj))
        aligned_mean = float(np.mean(adj))
        if abs(aligned_mean - float(r.ensemble_proj)) > 1e-8:
            raise RuntimeError(
                f"mean alignment mismatch {r.season} W{r.week} {r.player_clean_key} {r.market}"
            )

        q05, q10, q50, q90, q95 = np.quantile(adj, [0.05, 0.10, 0.50, 0.90, 0.95])
        actual = float(r.actual)
        rows.append({
            "season": int(r.season),
            "week": int(r.week),
            "game_id": str(r.game_id),
            "team": str(r.team),
            "opponent": str(r.opponent),
            "player_clean_key": str(r.player_clean_key),
            "market": str(r.market),
            "actual": actual,
            "final_mean": float(r.ensemble_proj),
            "sim_sd": float(np.std(adj, ddof=1)),
            "q05": float(q05),
            "q10": float(q10),
            "q50": float(q50),
            "q90": float(q90),
            "q95": float(q95),
            "upper_break_10": int(actual > q90),
            "lower_break_10": int(actual < q10),
            "upper_break_05": int(actual > q95),
            "lower_break_05": int(actual < q05),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection-file", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-draws", type=int, default=EXPECTED_DRAWS)
    args = ap.parse_args()

    projection = _read(args.projection_file, "historical projection trace")
    detail = build_detail(
        projection,
        args.distribution_dir,
        expected_draws=int(args.expected_draws),
    )

    rows = []
    for season in (2024, 2025):
        for market in MARKETS:
            q = detail.loc[detail["season"].eq(season) & detail["market"].eq(market)].copy()
            rows.append(_cell(q, season, market))
    cells = pd.DataFrame(rows)
    result = classify(cells)
    result.update({
        "version": VERSION,
        "plan": "docs/research/DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_V1_PLAN.md",
        "football_only": True,
        "sportsbook_inputs_used": 0,
        "projection_mean_changed": False,
        "parameters_fit": 0,
        "expected_draws": int(args.expected_draws),
        "scoreable_rows": int(len(detail)),
        "bootstrap_reps": REPS,
        "bootstrap_seed": SEED,
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "distribution_right_tail_asymmetry_detail.csv", index=False)
    cells.to_csv(args.out_dir / "distribution_right_tail_asymmetry_cells.csv", index=False)
    (args.out_dir / "distribution_right_tail_asymmetry_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(cells.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
