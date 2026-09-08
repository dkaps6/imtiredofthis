#!/usr/bin/env python3
"""RB-R17: mean-preserving empirical tail-mixture distribution candidate.

Consumes mechanically corrected R12 player-game predictions/outcomes and OOS R16
cat30/cat50 tail probabilities. Compares an unconditional empirical residual
bootstrap against a nested R16-conditioned residual mixture. Both distributions
are floored at zero and then rescaled per player-game to preserve the frozen
baseline receiving-yard mean exactly.

Research-only. No production mean, target entitlement, RB-room mass, sportsbook
input, or production parameter is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = [((2023,), 2024), ((2023, 2024), 2025)]
DRAWS = 2000
SEED = 917
MEAN_TOL = 1e-6


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _row_seed(season: int, week: int, team: str, player: str, salt: str) -> int:
    text = f"{SEED}|{season}|{week}|{team}|{player}|{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def _mean_preserve(mu: float, residuals: np.ndarray) -> np.ndarray:
    mu = max(0.0, float(mu))
    if mu <= 0.0:
        return np.zeros_like(residuals, dtype=float)
    draws = np.clip(mu + np.asarray(residuals, dtype=float), 0.0, None)
    m = float(draws.mean())
    if not np.isfinite(m) or m <= 0.0:
        return np.full_like(draws, mu, dtype=float)
    draws = draws * (mu / m)
    m2 = float(draws.mean())
    if m2 > 0:
        draws *= mu / m2
    return draws


def _crps(samples: np.ndarray, y: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return np.nan
    first = float(np.mean(np.abs(x - float(y))))
    i = np.arange(1, n + 1, dtype=float)
    e_pair = float((2.0 / (n * n)) * np.sum((2.0 * i - n - 1.0) * x))
    return first - 0.5 * e_pair


def _pinball(y: float, qhat: float, q: float) -> float:
    err = float(y) - float(qhat)
    return float(q * err if err >= 0 else (1.0 - q) * (-err))


def _distribution_row(mu: float, y: float, samples: np.ndarray) -> dict:
    q05, q10, q90, q95 = np.quantile(samples, [0.05, 0.10, 0.90, 0.95])
    e30 = float(y >= mu + 30.0)
    e50 = float(y >= mu + 50.0)
    p30 = float(np.mean(samples >= mu + 30.0))
    p50 = float(np.mean(samples >= mu + 50.0))
    return {
        "sim_mean": float(samples.mean()),
        "mean_abs_delta": abs(float(samples.mean()) - float(mu)),
        "crps": _crps(samples, y),
        "p30": p30,
        "p50": p50,
        "brier30": (p30 - e30) ** 2,
        "brier50": (p50 - e50) ** 2,
        "q90": float(q90),
        "q95": float(q95),
        "pinball90": _pinball(y, q90, 0.90),
        "pinball95": _pinball(y, q95, 0.95),
        "covered80": float(q10 <= y <= q90),
        "covered90": float(q05 <= y <= q95),
    }


def _aggregate(g: pd.DataFrame, variant: str, test_season: str) -> dict:
    return {
        "variant": variant,
        "test_season": test_season,
        "n": int(len(g)),
        "crps": float(g.crps.mean()),
        "brier30": float(g.brier30.mean()),
        "brier50": float(g.brier50.mean()),
        "pinball90": float(g.pinball90.mean()),
        "pinball95": float(g.pinball95.mean()),
        "coverage80": float(g.covered80.mean()),
        "coverage90": float(g.covered90.mean()),
        "max_mean_abs_delta": float(g.mean_abs_delta.max()),
        "avg_pred_p30": float(g.p30.mean()),
        "avg_pred_p50": float(g.p50.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--r16-tail-predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    pred = pd.read_csv(a.predictions, low_memory=False)
    req = {"season", "week", "team", "player_clean_key", "actual_rec_yards", "baseline_pred_rec_yards"}
    missing = sorted(req - set(pred.columns))
    if missing:
        raise RuntimeError(f"R17 predictions missing columns: {missing}")
    for c in ["season", "week", "actual_rec_yards", "baseline_pred_rec_yards"]:
        pred[c] = _num(pred[c])
    pred = pred.dropna(subset=list(req)).copy()
    pred["season"] = pred.season.astype(int)
    pred["week"] = pred.week.astype(int)
    pred["team"] = pred.team.astype(str)
    pred["player_clean_key"] = pred.player_clean_key.astype(str)
    pred["residual"] = pred.actual_rec_yards - pred.baseline_pred_rec_yards
    if pred.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("R17 predictions must be unique by player-game")

    r16 = pd.read_csv(a.r16_tail_predictions, low_memory=False)
    r16 = r16.loc[r16.label.isin(["cat30_under", "cat50_under"]), [
        "season", "week", "team", "player_clean_key", "label", "p_full"
    ]].copy()
    r16["season"] = _num(r16.season).astype(int)
    r16["week"] = _num(r16.week).astype(int)
    wide = r16.pivot_table(
        index=["season", "week", "team", "player_clean_key"], columns="label", values="p_full", aggfunc="first"
    ).reset_index().rename(columns={"cat30_under": "r16_p30", "cat50_under": "r16_p50"})
    if wide[["r16_p30", "r16_p50"]].isna().any().any():
        raise RuntimeError("R17 R16 probabilities incomplete")

    all_rows = []
    fold_audits = []
    for train_seasons, test_season in FOLDS:
        train = pred.loc[pred.season.isin(train_seasons)].copy()
        test = pred.loc[pred.season.eq(test_season)].merge(
            wide.loc[wide.season.eq(test_season)],
            on=["season", "week", "team", "player_clean_key"], how="inner", validate="one_to_one"
        )
        if len(test) < 100:
            raise RuntimeError(f"R17 too few test rows for {test_season}: {len(test)}")

        pool_all = train.residual.to_numpy(float)
        pool_non = train.loc[train.residual.lt(30.0), "residual"].to_numpy(float)
        pool_30 = train.loc[train.residual.ge(30.0) & train.residual.lt(50.0), "residual"].to_numpy(float)
        pool_50 = train.loc[train.residual.ge(50.0), "residual"].to_numpy(float)
        if min(len(pool_all), len(pool_non), len(pool_30), len(pool_50)) == 0:
            raise RuntimeError(f"R17 empty residual pool for fold {test_season}")
        fold_audits.append({
            "train_seasons": ",".join(map(str, train_seasons)), "test_season": test_season,
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "pool_all": int(len(pool_all)), "pool_non": int(len(pool_non)),
            "pool_30_49": int(len(pool_30)), "pool_50_plus": int(len(pool_50)),
            "train_cat30_rate": float(np.mean(pool_all >= 30.0)),
            "train_cat50_rate": float(np.mean(pool_all >= 50.0)),
        })

        for row in test.itertuples(index=False):
            mu = max(0.0, float(row.baseline_pred_rec_yards))
            y = max(0.0, float(row.actual_rec_yards))
            p30 = float(np.clip(row.r16_p30, 0.0, 1.0))
            p50 = float(np.clip(row.r16_p50, 0.0, 1.0))
            w50 = min(p50, p30)
            w30 = max(p30 - w50, 0.0)
            wnon = max(0.0, 1.0 - p30)
            total = wnon + w30 + w50
            if total <= 0:
                wnon, w30, w50 = 1.0, 0.0, 0.0
            else:
                wnon, w30, w50 = wnon / total, w30 / total, w50 / total

            rng_b = np.random.default_rng(_row_seed(row.season, row.week, row.team, row.player_clean_key, "baseline"))
            resid_b = rng_b.choice(pool_all, size=DRAWS, replace=True)
            draws_b = _mean_preserve(mu, resid_b)

            rng_c = np.random.default_rng(_row_seed(row.season, row.week, row.team, row.player_clean_key, "candidate"))
            comp = rng_c.choice(3, size=DRAWS, p=[wnon, w30, w50])
            resid_c = np.empty(DRAWS, dtype=float)
            m = comp == 0
            resid_c[m] = rng_c.choice(pool_non, size=int(m.sum()), replace=True)
            m = comp == 1
            resid_c[m] = rng_c.choice(pool_30, size=int(m.sum()), replace=True)
            m = comp == 2
            resid_c[m] = rng_c.choice(pool_50, size=int(m.sum()), replace=True)
            draws_c = _mean_preserve(mu, resid_c)

            base = _distribution_row(mu, y, draws_b)
            cand = _distribution_row(mu, y, draws_c)
            common = {
                "season": int(row.season), "week": int(row.week), "team": str(row.team),
                "player_clean_key": str(row.player_clean_key), "actual_rec_yards": y,
                "frozen_mean": mu, "r16_p30": p30, "r16_p50": p50,
                "mix_wnon": wnon, "mix_w30": w30, "mix_w50": w50,
            }
            all_rows.append({**common, "variant": "GLOBAL_EMPIRICAL_MEAN_PRESERVED", **base})
            all_rows.append({**common, "variant": "R16_NESTED_TAIL_EMPIRICAL_MEAN_PRESERVED", **cand})

    casebook = pd.DataFrame(all_rows)
    fold_rows = []
    for (variant, season), g in casebook.groupby(["variant", "season"], sort=True):
        fold_rows.append(_aggregate(g, variant, str(int(season))))
    folds = pd.DataFrame(fold_rows)
    combined_rows = []
    for variant, g in casebook.groupby("variant", sort=True):
        combined_rows.append(_aggregate(g, variant, "COMBINED"))
    combined = pd.DataFrame(combined_rows)

    def one(frame: pd.DataFrame, variant: str, season: str):
        q = frame.loc[frame.variant.eq(variant) & frame.test_season.eq(season)]
        if len(q) != 1:
            raise RuntimeError(f"R17 lookup failed {variant=} {season=}")
        return q.iloc[0]

    b = one(combined, "GLOBAL_EMPIRICAL_MEAN_PRESERVED", "COMBINED")
    c = one(combined, "R16_NESTED_TAIL_EMPIRICAL_MEAN_PRESERVED", "COMBINED")
    fold_guard = True
    for s in ["2024", "2025"]:
        bf = one(folds, "GLOBAL_EMPIRICAL_MEAN_PRESERVED", s)
        cf = one(folds, "R16_NESTED_TAIL_EMPIRICAL_MEAN_PRESERVED", s)
        fold_guard = fold_guard and bool(float(cf.crps) <= float(bf.crps) * 1.01)

    gates = {
        "mean_preservation": bool(float(c.max_mean_abs_delta) <= MEAN_TOL),
        "combined_crps": bool(float(c.crps) <= float(b.crps)),
        "fold_crps_guard": bool(fold_guard),
        "cat30_brier": bool(float(c.brier30) < float(b.brier30)),
        "cat50_brier": bool(float(c.brier50) < float(b.brier50)),
        "q90_pinball": bool(float(c.pinball90) < float(b.pinball90)),
        "q95_pinball": bool(float(c.pinball95) < float(b.pinball95)),
        "coverage80_guard": bool(abs(float(c.coverage80) - 0.80) <= abs(float(b.coverage80) - 0.80) + 0.02),
        "coverage90_guard": bool(abs(float(c.coverage90) - 0.90) <= abs(float(b.coverage90) - 0.90) + 0.02),
        "sportsbook_zero": True,
    }
    supported = all(gates.values())

    result = {
        "candidate": "RB_R17_MEAN_PRESERVING_TAIL_MIXTURE_V1",
        "disposition": "RB_R17_DISTRIBUTION_SIGNAL_SUPPORTED_RESEARCH_ONLY" if supported else "RB_R17_DISTRIBUTION_SIGNAL_NOT_SUPPORTED_RESEARCH_ONLY",
        "science_pass": bool(supported),
        "parent": "RB_R16_UPSIDE_TAIL_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY",
        "mean_policy": "FROZEN_BASELINE_RECEIVING_MEAN_PRESERVED_EXACTLY",
        "draws_per_player_game": DRAWS,
        "seed": SEED,
        "folds": [{"train": list(t), "test": s} for t, s in FOLDS],
        "baseline": {k: (float(b[k]) if k not in {"variant", "test_season", "n"} else int(b[k]) if k == "n" else str(b[k])) for k in b.index},
        "candidate_metrics": {k: (float(c[k]) if k not in {"variant", "test_season", "n"} else int(c[k]) if k == "n" else str(c[k])) for k in c.index},
        "gates": gates,
        "thresholds": {
            "max_mean_abs_delta": MEAN_TOL,
            "combined_crps_candidate_lte_baseline": True,
            "fold_crps_max_relative_worsening": 0.01,
            "cat30_brier_strict_gain": True,
            "cat50_brier_strict_gain": True,
            "q90_pinball_strict_gain": True,
            "q95_pinball_strict_gain": True,
            "coverage_guard_extra_absolute_tolerance": 0.02,
        },
        "fold_pool_audit": fold_audits,
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": 0,
        "governance_note": "PASS authorizes only a separately frozen production-parity/distribution integration candidate; no RB receiving mean or R12 promotion occurs here.",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    casebook.to_csv(a.out_dir / "rb_r17_distribution_casebook.csv", index=False)
    folds.to_csv(a.out_dir / "rb_r17_fold_summary.csv", index=False)
    combined.to_csv(a.out_dir / "rb_r17_combined_summary.csv", index=False)
    (a.out_dir / "rb_r17_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\n=== combined ===\n", combined.to_string(index=False))
    print("\n=== folds ===\n", folds.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
