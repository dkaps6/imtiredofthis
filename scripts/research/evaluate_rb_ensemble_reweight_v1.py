"""Fix B: evaluate frozen ensemble-weight candidates for RB rush_yards.

Preregistration: docs/research/PRISM_RB_ENSEMBLE_REWEIGHT_PREREGISTRATION_V1.md
Candidate set and metrics were frozen before this ran. No candidate may be added.

Fit on 2026 weeks 1-2 only; evaluate on weeks 6-7 only. Writes a RESEARCH weights
file; never touches data/model_ensemble_weights.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.modeling.ensemble_v2 import fit_market_weights

ROWS = Path("data/research/live_2026/backtest_2026_live_calibration_v1_rows.csv")
COMPONENTS = ("mc_proj", "ml_proj", "state_proj")
FIT_WEEKS, TEST_WEEKS = (1, 2), (6, 7)
INCUMBENT = {"rush_yards": (0.5569542426070742, 0.4430457573929258, 0.0),
             "rush_rec_yards": None}
RUSH_MARKETS = {"rush_yards", "rush_rec_yards"}
BOOT, SEED = 4000, 42
MAE_TOLERANCE = 1.0


def load(market: str) -> pd.DataFrame:
    d = pd.read_csv(ROWS, low_memory=False)
    qb = set(d.loc[d["market"] == "pass_yards", "player_clean_key"].dropna())
    x = d[d["market"] == market].copy()
    if market in RUSH_MARKETS:
        x = x[~x["player_clean_key"].isin(qb)]
    for c in (*COMPONENTS, "actual", "vegas_line"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=[*COMPONENTS, "actual"])


def blend(df: pd.DataFrame, w) -> np.ndarray:
    return df[list(COMPONENTS)].to_numpy(float) @ np.asarray(w, float)


def metrics(pred, actual, line) -> dict:
    ok = np.isfinite(line)
    side = (np.sign(pred - line) == np.sign(actual - line))[ok]
    return {"bias": float(np.mean(pred - actual)),
            "mae": float(np.mean(np.abs(pred - actual))),
            "side_acc": float(np.mean(side)) if ok.any() else float("nan"),
            "side_n": int(ok.sum())}


def main() -> int:
    out: dict = {"version": "RB_ENSEMBLE_REWEIGHT_EVAL_V1",
                 "preregistration": "docs/research/PRISM_RB_ENSEMBLE_REWEIGHT_PREREGISTRATION_V1.md",
                 "sportsbook_inputs_used_in_any_weight_or_fit": False, "markets": {}}

    for market in ("rush_yards", "rush_rec_yards"):
        d = load(market)
        fit = d[d["week"].isin(FIT_WEEKS)]
        test = d[d["week"].isin(TEST_WEEKS)]
        print(f"\n{'='*74}\n{market}: fit n={len(fit)} (wk{FIT_WEEKS})  test n={len(test)} (wk{TEST_WEEKS})\n{'='*74}")
        if len(fit) < 40 or len(test) < 40:
            print("  K4: insufficient rows; failing closed for this market.")
            out["markets"][market] = {"status": "FAIL_CLOSED_INSUFFICIENT_ROWS",
                                      "fit_n": int(len(fit)), "test_n": int(len(test))}
            if market == "rush_yards":
                return 2
            continue

        # P3 synthesis bypasses the ensemble for Week-1 rush_yards; report its reach
        syn = pd.to_numeric(d.get("rb_synthesis_applied", 0), errors="coerce").fillna(0)
        print(f"  rb_synthesis_applied rows: fit {int(syn[fit.index].sum())}, test {int(syn[test.index].sum())}"
              f"  (P3 is Week-1 only and bypasses the ensemble on those rows)")

        # C1: the promoted fitter, on fit rows only
        fw = fit_market_weights(fit.assign(market=market))
        if fw.empty:
            print("  K4: fitter returned no weights.")
            return 2
        r = fw.iloc[0]
        c1 = (float(r["mc_weight"]), float(r["ml_weight"]), float(r["state_weight"]))

        # diagnostic: what the sum-to-1 normalization discards (alters no candidate)
        from sklearn.linear_model import LinearRegression
        raw = LinearRegression(fit_intercept=False, positive=True).fit(
            fit[list(COMPONENTS)].to_numpy(float), fit["actual"].to_numpy(float)).coef_
        print(f"  [diagnostic] raw NNLS coefs {np.round(raw,4)} sum={raw.sum():.4f}"
              f"  -> normalization {'DISCARDS a scale correction' if abs(raw.sum()-1)>0.02 else 'discards ~nothing'}")

        cands = {"C0_incumbent": INCUMBENT[market], "C1_refit_wk12": c1,
                 "C2_pure_ml": (0.0, 1.0, 0.0), "C3_equal": (1/3, 1/3, 1/3)}
        if cands["C0_incumbent"] is None:
            cands["C0_incumbent"] = (1.0, 0.0, 0.0)  # uncalibrated fallback is 100% MC
            print("  note: no incumbent weight row for this market; C0 = uncalibrated MC fallback")

        act = test["actual"].to_numpy(float); line = test["vegas_line"].to_numpy(float)
        res = {k: metrics(blend(test, w), act, line) for k, w in cands.items()}
        base = res["C0_incumbent"]

        print(f"\n  {'candidate':16s} {'weights (mc/ml/state)':28s} {'bias':>8} {'MAE':>7} {'side%':>7}")
        for k, w in cands.items():
            m = res[k]
            print(f"  {k:16s} {np.round(w,4)!s:28s} {m['bias']:>+8.2f} {m['mae']:>7.2f} {100*m['side_acc']:>6.1f}%")

        # paired bootstrap on improvement in |bias| vs C0, identical rows
        rng = np.random.default_rng(SEED)
        preds = {k: blend(test, w) for k, w in cands.items()}
        idx = np.arange(len(test))
        for k in cands:
            if k == "C0_incumbent":
                continue
            draws = []
            for _ in range(BOOT):
                s = rng.choice(idx, len(idx), replace=True)
                b0 = abs(np.mean(preds["C0_incumbent"][s] - act[s]))
                bk = abs(np.mean(preds[k][s] - act[s]))
                draws.append(b0 - bk)      # >0 means candidate closer to zero
            lo, hi = np.percentile(draws, [2.5, 97.5])
            imp = abs(base["bias"]) - abs(res[k]["bias"])
            res[k]["abs_bias_improvement_vs_C0"] = round(float(imp), 3)
            res[k]["improvement_ci95"] = [round(float(lo), 3), round(float(hi), 3)]
            res[k]["improvement_ci_excludes_zero"] = bool(lo > 0)
            res[k]["mae_delta_vs_C0"] = round(float(res[k]["mae"] - base["mae"]), 3)
            res[k]["side_acc_delta_vs_C0"] = round(float(res[k]["side_acc"] - base["side_acc"]), 4)

        print(f"\n  {'candidate':16s} {'|bias| gain vs C0':>18} {'95% CI':>22} {'MAE d':>7} {'side d':>8}")
        for k in cands:
            if k == "C0_incumbent":
                continue
            m = res[k]
            print(f"  {k:16s} {m['abs_bias_improvement_vs_C0']:>+18.2f}"
                  f" {str(m['improvement_ci95']):>22} {m['mae_delta_vs_C0']:>+7.2f}"
                  f" {100*m['side_acc_delta_vs_C0']:>+7.1f}%")

        # K3: per-week stability of the leader
        print(f"\n  per-week stability (K3):")
        wk = {}
        for w_ in TEST_WEEKS:
            t = test[test["week"] == w_]
            a_ = t["actual"].to_numpy(float); l_ = t["vegas_line"].to_numpy(float)
            wk[int(w_)] = {k: metrics(blend(t, v), a_, l_) for k, v in cands.items()}
            cells = "  ".join(f"{k.split('_')[0]} {wk[int(w_)][k]['bias']:+.1f}" for k in cands)
            print(f"    week {w_} (n={len(t)}): {cells}")

        # gates
        g = {}
        for k in cands:
            if k == "C0_incumbent":
                continue
            m = res[k]
            g[k] = {"G1_bias_closer_ci_excludes_0": bool(m["abs_bias_improvement_vs_C0"] > 0
                                                         and m["improvement_ci_excludes_zero"]),
                    "G2_side_not_worse": bool(m["side_acc"] >= base["side_acc"]),
                    "G3_mae_within_tolerance": bool(m["mae"] - base["mae"] <= MAE_TOLERANCE)}
            g[k]["passes_G1_G3"] = all(g[k].values())
        c3_passes = g.get("C3_equal", {}).get("passes_G1_G3", False)
        print(f"\n  GATES (G4: C3 naive control must NOT pass -> C3 passes = {c3_passes})")
        for k, v in g.items():
            print(f"    {k:16s} G1 {str(v['G1_bias_closer_ci_excludes_0']):5s}"
                  f" G2 {str(v['G2_side_not_worse']):5s} G3 {str(v['G3_mae_within_tolerance']):5s}"
                  f" -> {'PASS' if v['passes_G1_G3'] and not c3_passes else 'no'}")
        if c3_passes:
            print("    K2 TRIGGERED: the naive equal-weight control also passes. Promote nothing.")

        out["markets"][market] = {
            "fit_n": int(len(fit)), "test_n": int(len(test)),
            "weights": {k: [round(float(x), 6) for x in v] for k, v in cands.items()},
            "raw_nnls_coefs": [round(float(x), 6) for x in raw],
            "raw_nnls_sum": round(float(raw.sum()), 6),
            "test_metrics": res, "per_week": wk, "gates": g,
            "K2_naive_control_also_passes": c3_passes,
            "promotion_candidate": market == "rush_yards",
        }

    out_dir = Path("data/research/rb_ensemble_reweight")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rb_ensemble_reweight_eval_v1.json").write_text(json.dumps(out, indent=2) + "\n")
    # RESEARCH weights file only -- never data/model_ensemble_weights.csv
    rowsout = []
    for market, m in out["markets"].items():
        if "weights" not in m:
            continue
        for cand, w in m["weights"].items():
            rowsout.append({"market": market, "candidate": cand, "mc_weight": w[0],
                            "ml_weight": w[1], "state_weight": w[2],
                            "fit_scope": "2026_wk1_2_only" if cand == "C1_refit_wk12" else "no_fit",
                            "promoted": False})
    pd.DataFrame(rowsout).to_csv(out_dir / "candidate_weights_research_only_v1.csv", index=False)
    print(f"\nwrote {out_dir}/ (research only; data/model_ensemble_weights.csv untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
