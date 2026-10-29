"""Grade the frozen CAR @ GB predictions once nflverse has week-8 actuals.

Card: docs/research/PRISM_TNF_WEEK8_PROSPECTIVE_PREDICTIONS_V1.md
Predictions were committed BEFORE kickoff. This script only scores them.

Exits 3 if actuals are not yet published, so it can be re-run without inventing a result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

FROZEN = Path("data/research/tnf_week8_prospective/frozen_board_rows_v1.csv")
OUT = Path("data/research/tnf_week8_prospective")
TEAMS, SEASON, WEEK = ("CAR", "GB"), 2026, 8
QBS = {"Bryce Young", "Jordan Love"}
RB_POS = {"RB", "HB", "TB", "FB"}


def _is_rb(p: object) -> bool:
    s = str(p).strip().upper()
    return s in RB_POS or s.startswith("RB")


def main() -> int:
    import nflreadpy as nfl
    fr = pd.read_csv(FROZEN)
    w = nfl.load_player_stats(seasons=[SEASON]).to_pandas()
    w = w[(w["week"] == WEEK) & (w["team"].astype(str).isin(TEAMS))].copy()
    if w.empty or w["carries"].fillna(0).sum() == 0:
        print("ACTUALS NOT PUBLISHED YET for 2026 wk8 CAR/GB — nothing graded, nothing invented.")
        return 3
    for c in ("carries", "rushing_yards", "receiving_yards", "targets"):
        w[c] = pd.to_numeric(w.get(c), errors="coerce").fillna(0.0)
    w["rush_rec"] = w["rushing_yards"] + w["receiving_yards"]

    act = {(r.player_display_name, r.team): r for r in w.itertuples()}
    print(f"=== ACTUALS, CAR @ GB wk{WEEK} ===")
    print(w[w["carries"] > 0][["player_display_name", "team", "position", "carries",
                               "rushing_yards", "receiving_yards"]]
          .sort_values("rushing_yards", ascending=False).to_string(index=False))

    print("\n=== FROZEN ROWS vs ACTUAL ===")
    rows = []
    for r in fr.itertuples():
        key = next((k for k in act if k[0] == r.player and k[1] == r.team), None)
        a = act.get(key)
        if a is None:
            print(f"  {r.player:18s} {r.team} {r.market:14s} DID NOT APPEAR")
            continue
        actual = a.rushing_yards if r.market == "rush_yards" else a.rush_rec
        rows.append({"player": r.player, "team": r.team, "market": r.market,
                     "line": r.vegas_line, "model": r.model_proj, "actual": float(actual),
                     "model_err": float(r.model_proj - actual),
                     "line_err": float(r.vegas_line - actual)})
        print(f"  {r.player:18s} {r.team} {r.market:14s} line {r.vegas_line:6.1f}"
              f"  model {r.model_proj:6.2f}  ACTUAL {actual:6.1f}"
              f"  model_err {r.model_proj-actual:+7.2f}  line_err {r.vegas_line-actual:+7.2f}")
    d = pd.DataFrame(rows)

    # the four weakly-dependent signs
    print("\n=== FROZEN PREDICTIONS ===")
    verdict = {}
    rb_act = w[w["position"].map(_is_rb) & (w["carries"] > 0)]
    for tm, model_total, model_lead in (("GB", 70.8, 0.558), ("CAR", 79.2, 0.782)):
        t = rb_act[rb_act["team"] == tm].sort_values("rushing_yards", ascending=False)
        tot = float(t["rushing_yards"].sum())
        lead = float(t.iloc[0]["rushing_yards"] / tot) if tot > 0 else float("nan")
        p4 = tot > model_total
        p3 = lead > model_lead
        verdict[f"P4_{tm}"] = bool(p4); verdict[f"P3_{tm}"] = bool(p3)
        print(f"  P4 {tm}: actual team RB rush {tot:.1f} vs model {model_total:.1f}"
              f"  -> {'HIT' if p4 else 'MISS'}")
        print(f"  P3 {tm}: actual lead share {lead:.3f} vs model {model_lead:.3f}"
              f"  (lead = {t.iloc[0]['player_display_name']})  -> {'HIT' if p3 else 'MISS'}")
    hits = sum(verdict.values())
    print(f"\n  FOUR WEAKLY-DEPENDENT SIGNS: {hits}/4")
    rule = ("consistent with the diagnosis; NOT promotion evidence" if hits == 4
            else "uninformative; change nothing" if hits >= 2
            else "a point AGAINST the flattening story on this game; record it, do not explain it away")
    print(f"  pre-committed interpretation: {rule}")

    print("\n=== P5: is rush_rec_yards worse than rush_yards for the lead backs? ===")
    for p in ("Josh Jacobs", "Chuba Hubbard"):
        a_ = d[(d["player"] == p) & (d["market"] == "rush_yards")]
        b_ = d[(d["player"] == p) & (d["market"] == "rush_rec_yards")]
        if len(a_) and len(b_):
            ra, rb = abs(a_.iloc[0]["model_err"]), abs(b_.iloc[0]["model_err"])
            print(f"  {p:16s} |rush err| {ra:6.2f}  |rush_rec err| {rb:6.2f}"
                  f"  -> {'HIT' if rb > ra else 'MISS'}")
            verdict[f"P5_{p.split()[-1]}"] = bool(rb > ra)

    if not d.empty:
        print(f"\n=== model vs market on these rows (n={len(d)}) ===")
        print(f"  model mean |err| {d['model_err'].abs().mean():.2f}"
              f"   market mean |err| {d['line_err'].abs().mean():.2f}")
        print(f"  model bias {d['model_err'].mean():+.2f}"
              f"   market bias {d['line_err'].mean():+.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "graded_rows_v1.csv", index=False)
    (OUT / "graded_v1.json").write_text(json.dumps({
        "version": "TNF_WEEK8_PROSPECTIVE_GRADE_V1",
        "card": "docs/research/PRISM_TNF_WEEK8_PROSPECTIVE_PREDICTIONS_V1.md",
        "game": "CAR @ GB", "season": SEASON, "week": WEEK,
        "source_run_id": "40953584718",
        "predictions": verdict, "signs_hit": int(hits), "signs_total": 4,
        "interpretation": rule,
        "model_mean_abs_err": float(d["model_err"].abs().mean()) if not d.empty else None,
        "market_mean_abs_err": float(d["line_err"].abs().mean()) if not d.empty else None,
        "n_is_one_game_not_evidence": True,
        "changes_any_promotion_state": False,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT}/graded_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
