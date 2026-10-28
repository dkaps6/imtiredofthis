"""Validate the mean-preserving MC dispersion recalibration on real Week 7 draws.

Executes MC_DISPERSION_RECALIBRATION_PREREGISTRATION_V1, frozen at 73c31697 before any
scoring.

k is DERIVED from 2026 weeks 1, 2 and 6 board residuals and NEVER from week 7. Week 7 is
then re-simulated from the archived fixture -- `simulation_v2.simulate()` runs on it and
produces genuine draws -- and baseline vs widened probabilities are graded against actual
Week 7 outcomes using the canonical GSIS identity resolver.

No sportsbook variable enters a projection; lines are the graded boundary only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.operations.grade_market_track_record_gsis_v1 import (
    build_alias_index, load_actual_stats_unfiltered, load_roster_identity, resolve_gsis,
)
from scripts.simulation_v2 import lookup, simulate

FIXTURE = Path("data/research/week7_opportunity_trace/week7_trace__data__model_rule_simulation_inputs.csv")
BOARDS = Path("data/market_track_record/boards")
OUT_JSON = Path("data/research/mc_dispersion/mc_dispersion_recalibration_v1.json")
OUT_ROWS = Path("data/research/mc_dispersion/mc_dispersion_recalibration_v1_rows.csv")

SEASON, TARGET_WEEK = 2026, 7
DERIVE_WEEKS = (1, 2, 6)            # frozen: k never sees week 7
ITERATIONS, SEED = 25000, 42        # production defaults
MEAN_TOLERANCE = 0.02               # frozen rule 4
ACTUAL_COLUMNS = {
    "pass_yards": ["pass_yards"], "rush_yards": ["rush_yards"], "rec_yards": ["rec_yards"],
    "receptions": ["receptions"], "rush_rec_yards": ["rush_yards", "rec_yards"],
}
BUCKETS = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]


def derive_k() -> dict:
    """k_market = sd(resid / model_sd) on the derive weeks only."""
    rows = []
    for w in DERIVE_WEEKS:
        p = BOARDS / f"{SEASON}_wk{w:02d}.csv"
        if p.exists():
            d = pd.read_csv(p, low_memory=False)
            d["_w"] = w
            rows.append(d)
    board = pd.concat(rows, ignore_index=True)
    weeks = sorted(board["_w"].unique().tolist())
    actual = load_actual_stats_unfiltered(SEASON, weeks)
    roster = load_roster_identity(SEASON, weeks)
    idx = build_alias_index(actual, roster)
    board["gsis_id"] = [resolve_gsis(k, t, idx)[0] for k, t in
                        zip(board["player_clean_key"].astype(str), board["team"].astype(str))]
    stat_cols = sorted({c for v in ACTUAL_COLUMNS.values() for c in v})
    a = actual.copy()
    for c in stat_cols:
        a[c] = pd.to_numeric(a.get(c), errors="coerce")
    a = a[["gsis_id", "season", "week"] + stat_cols].drop_duplicates(["gsis_id", "season", "week"])
    board["week"] = pd.to_numeric(board["week"], errors="coerce")
    b = board.merge(a, on=["gsis_id", "season", "week"], how="left")
    b["actual"] = b.apply(lambda r: (np.nan if not ACTUAL_COLUMNS.get(str(r["market"]))
                                     or any(pd.isna(r.get(c)) for c in ACTUAL_COLUMNS[str(r["market"])])
                                     else float(sum(float(r[c]) for c in ACTUAL_COLUMNS[str(r["market"])]))),
                          axis=1)
    b["model_sd"] = pd.to_numeric(b["model_sd"], errors="coerce")
    b["model_proj"] = pd.to_numeric(b["model_proj"], errors="coerce")
    b = b.dropna(subset=["actual", "model_sd", "model_proj"])
    b = b[b["model_sd"] > 0]
    b = b.drop_duplicates(["season", "week", "player_clean_key", "market"])
    b["z"] = (b["actual"] - b["model_proj"]) / b["model_sd"]
    out = {}
    for m, g in b.groupby("market"):
        if len(g) < 40:
            continue
        out[str(m)] = {"k": float(g["z"].std()), "rows": int(len(g)), "weeks": weeks}
    return out


def recalibrate_naive(draws: np.ndarray, k: float) -> np.ndarray:
    """First implementation: widen then clip. NOT actually mean-preserving.

    Recorded because it FAILED frozen rule 4 -- clipping at zero lifts a low-mean
    player's mean by up to 27.8%, which would break the promoted QB/RB synthesis mean
    invariants (AGENTS.md section 7).
    """
    mu = float(draws.mean())
    return np.clip(mu + k * (draws - mu), 0.0, None)


def recalibrate(draws: np.ndarray, k: float) -> np.ndarray:
    """The preregistered design, implemented correctly: genuinely mean-preserving.

    Widen, clip to the non-negative domain, then multiplicatively restore the original
    mean. A multiplicative rescale preserves the zero floor and restores the mean
    exactly, so the promoted-synthesis invariant holds. The realised dispersion is
    therefore slightly below k x sd_base, which is the honest cost of respecting the
    domain -- reported rather than hidden.
    """
    mu = float(draws.mean())
    wide = np.clip(mu + k * (draws - mu), 0.0, None)
    m = float(wide.mean())
    if m <= 0.0 or not np.isfinite(m):
        return np.clip(draws, 0.0, None)
    return wide * (mu / m)


def _pack_actuals(board: pd.DataFrame) -> pd.DataFrame:
    actual = load_actual_stats_unfiltered(SEASON, [TARGET_WEEK])
    roster = load_roster_identity(SEASON, [TARGET_WEEK])
    idx = build_alias_index(actual, roster)
    board = board.copy()
    board["gsis_id"] = [resolve_gsis(k, t, idx)[0] for k, t in
                        zip(board["player_clean_key"].astype(str), board["team"].astype(str))]
    stat_cols = sorted({c for v in ACTUAL_COLUMNS.values() for c in v})
    a = actual.copy()
    for c in stat_cols:
        a[c] = pd.to_numeric(a.get(c), errors="coerce")
    a = a[["gsis_id", "season", "week"] + stat_cols].drop_duplicates(["gsis_id", "season", "week"])
    board["week"] = pd.to_numeric(board["week"], errors="coerce")
    b = board.merge(a, on=["gsis_id", "season", "week"], how="left")
    b["actual"] = b.apply(lambda r: (np.nan if not ACTUAL_COLUMNS.get(str(r["market"]))
                                     or any(pd.isna(r.get(c)) for c in ACTUAL_COLUMNS[str(r["market"])])
                                     else float(sum(float(r[c]) for c in ACTUAL_COLUMNS[str(r["market"])]))),
                          axis=1)
    return b


def _metrics(g: pd.DataFrame, pcol: str) -> dict:
    g = g.dropna(subset=[pcol, "actual", "vegas_line"])
    g = g[g["actual"] != g["vegas_line"]]
    if g.empty:
        return {"rows": 0}
    over = (g["actual"] > g["vegas_line"]).astype(float)
    brier = float(((g[pcol] - over) ** 2).mean())
    bucket = pd.cut(g[pcol], BUCKETS, include_lowest=True)
    gaps, curve = [], []
    for b, sub in g.groupby(bucket, observed=True):
        if len(sub) < 15:
            continue
        o = (sub["actual"] > sub["vegas_line"]).astype(float)
        gap = float(sub[pcol].mean() - o.mean())
        gaps.append(abs(gap))
        curve.append({"bucket": str(b), "rows": int(len(sub)),
                      "predicted": float(sub[pcol].mean()), "actual": float(o.mean()), "gap": gap})
    return {"rows": int(len(g)), "brier": brier,
            "mean_abs_calibration_gap": float(np.mean(gaps)) if gaps else None,
            "curve": curve,
            "predicted_over_rate": float(g[pcol].mean()),
            "actual_over_rate": float(over.mean())}


def main() -> int:
    prereg = Path("docs/production/MC_DISPERSION_RECALIBRATION_PREREGISTRATION_V1.md")
    if not prereg.exists():
        raise RuntimeError("preregistration missing; this may not be scored unfrozen")
    txt = prereg.read_text(encoding="utf-8")
    for token in ("weeks 1, 2 and 6 ONLY", "0.02", "mean-preserving"):
        if token not in txt:
            raise RuntimeError(f"preregistration no longer carries frozen value {token!r}")

    ks = derive_k()
    print("k derived from weeks", DERIVE_WEEKS, "(week 7 never used):")
    for m, v in sorted(ks.items()):
        print(f"  {m:16s} k={v['k']:.3f}  (n={v['rows']})")

    metrics = pd.read_csv(FIXTURE, low_memory=False)
    sims = simulate(metrics, iterations=ITERATIONS, seed=SEED)
    board = pd.read_csv(BOARDS / f"{SEASON}_wk{TARGET_WEEK:02d}.csv", low_memory=False)
    board = _pack_actuals(board)

    rows, mean_checks = [], []
    for _, r in board.iterrows():
        market, raw = str(r.get("market")), str(r.get("source_market"))
        if market not in ks or market not in ACTUAL_COLUMNS:
            continue
        line = pd.to_numeric(r.get("vegas_line"), errors="coerce")
        if not np.isfinite(line):
            continue
        # The fixture carries no event_id, so simulate() keyed draws by the sorted
        # team|opponent pair. The board DOES carry event_id, which would make lookup()
        # use a key the simulation never emitted -- so blank it and take the same
        # fallback convention the draws were built under.
        lookup_row = r.copy()
        lookup_row["event_id"] = ""
        draws = lookup(sims, lookup_row, raw)
        if draws is None:
            continue
        base = np.asarray(draws, dtype=float)
        if base.size < 2:
            continue
        k = ks[market]["k"]
        wide = recalibrate(base, k)
        naive = recalibrate_naive(base, k)
        mb, mw = float(base.mean()), float(wide.mean())
        mean_checks.append({"market": market, "mean_before": mb, "mean_after": mw,
                            "rel_shift": abs(mw - mb) / max(mb, 1.0)})
        rows.append({
            "player_clean_key": r.get("player_clean_key"), "market": market,
            "vegas_line": float(line), "actual": r.get("actual"),
            "p_over_base": float(np.mean(base > line)), "p_over_wide": float(np.mean(wide > line)),
            "sd_base": float(base.std(ddof=1)), "sd_wide": float(wide.std(ddof=1)),
            "mean_base": mb, "mean_wide": mw, "k": k,
            "p_over_naive": float(np.mean(naive > line)),
            "mean_naive": float(naive.mean()),
            "rel_shift_naive": abs(float(naive.mean()) - mb) / max(mb, 1.0),
        })
    d = pd.DataFrame(rows).drop_duplicates(["player_clean_key", "market", "vegas_line"])
    mc = pd.DataFrame(mean_checks)

    base_m, wide_m = _metrics(d, "p_over_base"), _metrics(d, "p_over_wide")
    naive_m = _metrics(d, "p_over_naive")
    dd = d.dropna(subset=["actual"])
    z_base = ((dd["actual"] - dd["mean_base"]) / dd["sd_base"].replace(0, np.nan)).std()
    z_wide = ((dd["actual"] - dd["mean_wide"]) / dd["sd_wide"].replace(0, np.nan)).std()

    r1 = wide_m["brier"] < base_m["brier"]
    r2 = wide_m["mean_abs_calibration_gap"] < base_m["mean_abs_calibration_gap"]
    # Frozen rule 3: move toward 1.0 AND do not overshoot past it.
    # The first implementation was `A and B or A`, which Python evaluates as
    # `(A and B) or A` == A, silently dropping the no-overshoot condition.
    # Frozen rule 3: move toward 1.0 AND do not cross to the other side of it.
    # Two bugs were found here. First, `A and B or A` evaluates as `(A and B) or A` == A,
    # which dropped the no-overshoot condition entirely. Second, the replacement hard-coded
    # `z_wide <= 1.0`, which is backwards when the baseline sits ABOVE 1.0 (it is 1.767):
    # that permits exactly the overshoot the rule exists to block. The side to guard
    # depends on which side the baseline starts from.
    closer = abs(z_wide - 1.0) < abs(z_base - 1.0)
    no_overshoot = (z_wide >= 1.0 - 1e-9) if z_base > 1.0 else (z_wide <= 1.0 + 1e-9)
    r3 = bool(closer and no_overshoot)
    worst = float(mc["rel_shift"].max()) if len(mc) else 0.0
    r4 = worst <= MEAN_TOLERANCE

    decision = {"rule_1_brier_improves": bool(r1), "rule_2_calibration_gaps_shrink": bool(r2),
                "rule_3_sd_z_toward_one": bool(r3), "rule_4_mean_preserved": bool(r4),
                "worst_relative_mean_shift": worst, "mean_tolerance": MEAN_TOLERANCE,
                "SUPPORTED": bool(r1 and r2 and r3 and r4)}
    report = {"preregistration": prereg.as_posix(), "k_by_market": ks,
              "iterations": ITERATIONS, "seed": SEED,
              "target_week": TARGET_WEEK, "derive_weeks": list(DERIVE_WEEKS),
              "baseline": base_m, "widened": wide_m,
              "naive_clip_only_REJECTED_rule4": {
                  **naive_m,
                  "worst_relative_mean_shift": float(d["rel_shift_naive"].max()),
                  "note": "widen-then-clip is NOT mean-preserving; failed frozen rule 4",
              },
              "sd_z_baseline": float(z_base), "sd_z_widened": float(z_wide),
              "mean_preservation_by_market": {
                  m: {"worst_rel_shift": float(g["rel_shift"].max()),
                      "median_rel_shift": float(g["rel_shift"].median())}
                  for m, g in mc.groupby("market")} if len(mc) else {},
              "decision": decision, "sportsbook_inputs_used_for_projection": False}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    d.to_csv(OUT_ROWS, index=False)

    print(f"\nwrote {OUT_JSON}")
    print(f"\nWEEK 7 validation ({base_m['rows']} decided rows, real draws, {ITERATIONS} iters)")
    print(f"  Brier        baseline {base_m['brier']:.4f}  ->  widened {wide_m['brier']:.4f}  "
          f"({wide_m['brier'] - base_m['brier']:+.4f})")
    print(f"  mean |gap|   baseline {base_m['mean_abs_calibration_gap']:.4f}  ->  "
          f"widened {wide_m['mean_abs_calibration_gap']:.4f}")
    print(f"  sd(z)        baseline {z_base:.3f}  ->  widened {z_wide:.3f}   (target 1.000)")
    print(f"  worst relative mean shift: {worst:.5f}  (tolerance {MEAN_TOLERANCE})")
    print(f"  [rejected naive widen-then-clip: Brier {naive_m['brier']:.4f}, "
          f"worst mean shift {float(d['rel_shift_naive'].max()):.5f} -- failed rule 4]")
    print("\n  calibration curve, baseline -> widened:")
    wb = {c["bucket"]: c for c in wide_m["curve"]}
    for c in base_m["curve"]:
        w = wb.get(c["bucket"])
        print(f"    {c['bucket']:>14s} n={c['rows']:4d}  pred {c['predicted']:.3f}->"
              f"{w['predicted'] if w else float('nan'):.3f}  actual {c['actual']:.3f}  "
              f"gap {c['gap']:+.3f}->{w['gap'] if w else float('nan'):+.3f}")
    print("\n  FROZEN DECISION RULE:")
    for key in ("rule_1_brier_improves", "rule_2_calibration_gaps_shrink",
                "rule_3_sd_z_toward_one", "rule_4_mean_preserved"):
        print(f"    {key}: {decision[key]}")
    print(f"\n  SUPPORTED: {decision['SUPPORTED']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
