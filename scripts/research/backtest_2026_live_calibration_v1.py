"""2026 live-season backtest: how did production actually do, weeks 1-7?

TWO THINGS THIS IS NOT:

1. **Not a betting track record.** Hit rate, units and ROI require knowing which board
   rows were real production decisions, and `PRODUCTION_DECISION_GATES_V1.json`
   preserves that gate evidence for **weeks 1-2 only**. Weeks 6-7 have clean board
   provenance but no preserved record of which rows were blocked, so the canonical
   grader fails closed on them -- correctly, since grading the whole board would
   overstate the decision set. That gap is reported, not worked around.
2. **Not a comparison against a constant-sigma normal.** CORRECTION: production does
   NOT price from a constant sigma or a normal CDF. The live path is
   full-slate.yml -> run_pricing_with_full_roster_universe_v3 -> v4/v5/v6 -> ...v1 ->
   `run_pricing_v2.py`, which at line 400 computes
   `p_over = np.mean(adjusted_outcomes > line)` -- the EMPIRICAL Monte Carlo
   distribution -- and reports `model_sd = np.std(adjusted_outcomes)` as a per-player
   diagnostic. `scripts/pricing_v2.py`'s `SIGMA` constant is imported by nothing on the
   production path; `run_pricing_v2` takes only `_fair_market_prob` and `_fair_odds`
   from it and never references `SIGMA`. So production's dispersion IS player-specific
   and its distribution is NOT forced symmetric. This script therefore tests
   production's real object: the calibration of `fair_prob`.

WHAT IT IS: a model-accuracy and interval-calibration read on live 2026 data, which
needs only projections and actuals. Weeks 3-5 are permanently lost (AGENTS.md 5a), so
the gradable set is weeks 1, 2, 6, 7.

Identity resolution reuses the canonical grader's deterministic, fail-closed
GSIS alias machinery rather than a second name matcher.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.operations.grade_market_track_record_gsis_v1 import (
    build_alias_index, load_actual_stats_unfiltered, load_roster_identity, resolve_gsis,
)
# load_actual_stats_unfiltered already renames to MARKET names (rec_yards,
# rush_yards, pass_yards, receptions), so the nflverse-named map does not apply here.
ACTUAL_COLUMNS: dict[str, list[str]] = {
    "pass_yards": ["pass_yards"],
    "rush_yards": ["rush_yards"],
    "rec_yards": ["rec_yards"],
    "receptions": ["receptions"],
    "rush_rec_yards": ["rush_yards", "rec_yards"],
}

SEASON = 2026
BOARD_DIR = Path("data/market_track_record/boards")
GATES = Path("data/market_track_record/PRODUCTION_DECISION_GATES_V1.json")
OUT_JSON = Path("data/research/live_2026/backtest_2026_live_calibration_v1.json")
OUT_ROWS = Path("data/research/live_2026/backtest_2026_live_calibration_v1_rows.csv")

Z20, Z80 = -0.8416212335729143, 0.8416212335729143
TARGET_COVERAGE = 0.60


def available_weeks() -> list[int]:
    weeks = sorted(int(p.stem.split("_wk")[1]) for p in BOARD_DIR.glob(f"{SEASON}_wk*.csv"))
    return weeks


def bet_gradable_weeks() -> list[int]:
    if not GATES.exists():
        return []
    return sorted(int(k) for k in json.loads(GATES.read_text(encoding="utf-8")).get("weeks", {}))


def load_boards(weeks: list[int]) -> pd.DataFrame:
    frames = []
    for w in weeks:
        p = BOARD_DIR / f"{SEASON}_wk{w:02d}.csv"
        d = pd.read_csv(p, low_memory=False)
        d["_board_week"] = w
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def attach_actuals(board: pd.DataFrame, weeks: list[int]) -> tuple[pd.DataFrame, dict]:
    actual = load_actual_stats_unfiltered(SEASON, weeks)
    roster = load_roster_identity(SEASON, weeks)
    idx = build_alias_index(actual, roster)

    resolved, status = [], []
    for key, team in zip(board["player_clean_key"].astype(str), board["team"].astype(str)):
        gsis, st = resolve_gsis(key, team, idx)
        resolved.append(gsis)
        status.append(st)
    out = board.copy()
    out["gsis_id"], out["identity_status"] = resolved, status

    stat_cols = sorted({c for cols in ACTUAL_COLUMNS.values() for c in cols})
    a = actual.copy()
    for c in stat_cols:
        a[c] = pd.to_numeric(a.get(c), errors="coerce")
    a = a[["gsis_id", "season", "week"] + stat_cols].drop_duplicates(["gsis_id", "season", "week"])

    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out.merge(a, on=["gsis_id", "season", "week"], how="left", suffixes=("", "_act"))

    def actual_for(row):
        cols = ACTUAL_COLUMNS.get(str(row["market"]))
        if not cols:
            return np.nan
        vals = [row.get(c) for c in cols]
        if any(pd.isna(v) for v in vals):
            return np.nan
        return float(sum(float(v) for v in vals))

    out["actual"] = out.apply(actual_for, axis=1)
    diag = {
        "board_rows": int(len(board)),
        "identity_status_counts": dict(pd.Series(status).value_counts()),
        "rows_with_actual": int(out["actual"].notna().sum()),
        "rows_without_actual": int(out["actual"].isna().sum()),
    }
    return out, diag


def accuracy(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["actual"]).copy()
    d["model_proj"] = pd.to_numeric(d["model_proj"], errors="coerce")
    d["vegas_line"] = pd.to_numeric(d["vegas_line"], errors="coerce")
    d = d.dropna(subset=["model_proj", "vegas_line"])
    # one row per player-market-week; a board carries several book rows
    d = d.drop_duplicates(["season", "week", "player_clean_key", "market"])
    d["model_err"] = (d["model_proj"] - d["actual"]).abs()
    d["vegas_err"] = (d["vegas_line"] - d["actual"]).abs()
    d["model_closer"] = d["model_err"] < d["vegas_err"]

    def block(g):
        return {
            "rows": int(len(g)),
            "model_mae": float(g["model_err"].mean()),
            "vegas_mae": float(g["vegas_err"].mean()),
            "model_minus_vegas_mae": float(g["model_err"].mean() - g["vegas_err"].mean()),
            "frac_model_closer": float(g["model_closer"].mean()),
        }

    return {
        "overall": block(d),
        "by_market": {str(k): block(g) for k, g in d.groupby("market")},
        "by_week": {str(int(k)): block(g) for k, g in d.groupby("week")},
    }, d


def fair_prob_calibration(d: pd.DataFrame) -> dict:
    """Calibration of production's REAL object: fair_prob from the MC distribution.

    A well-calibrated p_over means that among rows where production said ~70%, the
    actual went over ~70% of the time. This needs no distributional assumption and no
    reconstruction, which is why it is the right test.
    """
    g = d.copy()
    g["fair_prob"] = pd.to_numeric(g.get("fair_prob"), errors="coerce")
    g["vegas_line"] = pd.to_numeric(g["vegas_line"], errors="coerce")
    g = g.dropna(subset=["fair_prob", "vegas_line", "actual"])
    g = g[g["actual"] != g["vegas_line"]]          # exact pushes are not over/under
    if g.empty:
        return {"rows": 0, "note": "no decided rows"}
    g["went_over"] = (g["actual"] > g["vegas_line"]).astype(float)

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    g["bucket"] = pd.cut(g["fair_prob"], bins, include_lowest=True)
    curve = []
    for b, sub in g.groupby("bucket", observed=True):
        if len(sub) < 15:
            continue
        curve.append({
            "bucket": str(b), "rows": int(len(sub)),
            "mean_predicted": float(sub["fair_prob"].mean()),
            "actual_over_rate": float(sub["went_over"].mean()),
            "gap": float(sub["fair_prob"].mean() - sub["went_over"].mean()),
        })
    # Brier score and a market benchmark on the same rows
    brier = float(((g["fair_prob"] - g["went_over"]) ** 2).mean())
    mkt = pd.to_numeric(g.get("market_prob"), errors="coerce")
    have_mkt = mkt.notna()
    brier_mkt = (float(((mkt[have_mkt] - g.loc[have_mkt, "went_over"]) ** 2).mean())
                 if have_mkt.any() else None)
    return {
        "rows": int(len(g)),
        "overall_predicted": float(g["fair_prob"].mean()),
        "overall_actual_over_rate": float(g["went_over"].mean()),
        "brier_model": brier,
        "brier_market": brier_mkt,
        "brier_model_minus_market": (brier - brier_mkt) if brier_mkt is not None else None,
        "calibration_curve": curve,
        "by_market": {
            str(m): {
                "rows": int(len(sub)),
                "predicted": float(sub["fair_prob"].mean()),
                "actual": float(sub["went_over"].mean()),
                "brier": float(((sub["fair_prob"] - sub["went_over"]) ** 2).mean()),
            } for m, sub in g.groupby("market") if len(sub) >= 30
        },
    }


def main() -> int:
    weeks = available_weeks()
    gradable = bet_gradable_weeks()
    board = load_boards(weeks)
    enriched, diag = attach_actuals(board, weeks)
    acc, d = accuracy(enriched)
    cal = fair_prob_calibration(d)

    report = {
        "season": SEASON,
        "weeks_with_archived_boards": weeks,
        "weeks_permanently_lost": [w for w in range(1, max(weeks) + 1) if w not in weeks],
        "weeks_bet_gradable": gradable,
        "weeks_blocked_from_bet_grading": [w for w in weeks if w not in gradable],
        "bet_grading_blocker": (
            "PRODUCTION_DECISION_GATES_V1.json preserves decision-gate evidence for "
            f"weeks {gradable} only; the canonical grader fails closed on the rest because "
            "grading a full board without knowing which rows were blocked would overstate "
            "the production decision set"
        ),
        "diagnostics": diag,
        "model_vs_vegas_accuracy": acc,
        "fair_prob_calibration": cal,
        "sportsbook_inputs_used_for_projection": False,
        "note": "sportsbook lines appear here only as a comparison benchmark and as the "
                "graded boundary, never as a projection input",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    d.to_csv(OUT_ROWS, index=False)

    print(f"wrote {OUT_JSON}\n")
    print(f"archived boards: weeks {weeks}   PERMANENTLY LOST: {report['weeks_permanently_lost']}")
    print(f"bet-gradable: weeks {gradable}   blocked: {report['weeks_blocked_from_bet_grading']}")
    print(f"identity: {diag['identity_status_counts']}")
    print(f"rows with actuals: {diag['rows_with_actual']} / {diag['board_rows']}\n")
    o = acc["overall"]
    print(f"=== MODEL vs VEGAS, 2026 weeks {weeks} ({o['rows']} deduped player-market-weeks) ===")
    print(f"  model MAE {o['model_mae']:.2f}   vegas MAE {o['vegas_mae']:.2f}   "
          f"diff {o['model_minus_vegas_mae']:+.2f}   model closer on {o['frac_model_closer']:.1%}")
    print("\n  by market:")
    for m, b in sorted(acc["by_market"].items(), key=lambda kv: -kv[1]["rows"]):
        print(f"    {m:16s} n={b['rows']:4d}  model {b['model_mae']:6.2f}  vegas {b['vegas_mae']:6.2f}  "
              f"diff {b['model_minus_vegas_mae']:+6.2f}  closer {b['frac_model_closer']:.1%}")
    print("\n  by week:")
    for w, b in sorted(acc["by_week"].items(), key=lambda kv: int(kv[0])):
        print(f"    wk{w:>2s}  n={b['rows']:4d}  model {b['model_mae']:6.2f}  vegas {b['vegas_mae']:6.2f}  "
              f"diff {b['model_minus_vegas_mae']:+6.2f}  closer {b['frac_model_closer']:.1%}")
    print(f"\n=== PRODUCTION'S REAL OBJECT: fair_prob calibration (MC-derived) ===")
    if cal.get("rows"):
        print(f"  decided rows {cal['rows']}  predicted over-rate {cal['overall_predicted']:.3f}  "
              f"actual {cal['overall_actual_over_rate']:.3f}")
        bm = cal["brier_market"]
        print(f"  Brier: model {cal['brier_model']:.4f}" +
              (f"   market {bm:.4f}   diff {cal['brier_model_minus_market']:+.4f}" if bm else ""))
        print("  calibration curve:")
        for c in cal["calibration_curve"]:
            print(f"    {c['bucket']:>14s} n={c['rows']:4d}  predicted {c['mean_predicted']:.3f}  "
                  f"actual {c['actual_over_rate']:.3f}  gap {c['gap']:+.3f}")
        print("  by market:")
        for m, b in sorted(cal["by_market"].items(), key=lambda kv: -kv[1]["rows"]):
            print(f"    {m:16s} n={b['rows']:4d}  predicted {b['predicted']:.3f}  "
                  f"actual {b['actual']:.3f}  brier {b['brier']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
