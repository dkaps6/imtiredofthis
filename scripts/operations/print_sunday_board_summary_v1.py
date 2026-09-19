#!/usr/bin/env python3
"""Print a plain-text summary of Sunday HAS-EDGE rows from the master workbook.

Read-only reporting only. Does not touch projections, probabilities, EV,
pricing, or the Decision policy in scripts/master_betting_workbook_core_v2.py.
It only filters and prints rows that script already produced, so this is
safe to run against any already-built outputs/NFL_BETTING_MODEL_MASTER.xlsx.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workbook", default="outputs/NFL_BETTING_MODEL_MASTER.xlsx")
    p.add_argument("--weekday", default="Sunday", help="Python weekday name to filter Kickoff UTC to")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    wb = load_workbook(Path(a.workbook))
    ws = wb["Master Betting Board"]
    headers = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(headers)}

    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        kickoff = r[idx["Kickoff UTC"]]
        signal = r[idx["Snapshot Signal"]]
        if not kickoff or signal != "HAS EDGE":
            continue
        try:
            dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.astimezone(timezone.utc).strftime("%A") != a.weekday:
            continue
        rows.append({
            "player": r[idx["Player"]],
            "team": r[idx["Team"]],
            "opp": r[idx["Opponent"]],
            "market": r[idx["Market"]],
            "book": r[idx["Book"]],
            "line": r[idx["Vegas Line"]],
            "proj": r[idx["Model Projection"]],
            "side": r[idx["Best Side"]],
            "odds": r[idx["Best Odds"]],
            "prob_edge": r[idx["Probability Edge"]],
            "best_ev": r[idx["Best EV ROI"]],
            "decision": r[idx["Decision"]],
            "kickoff": kickoff,
            "bettable_now": r[idx["Bettable Now"]],
        })

    rows.sort(key=lambda x: (x["best_ev"] if x["best_ev"] is not None else -999), reverse=True)

    print(f"=== {a.weekday} HAS-EDGE rows: {len(rows)} ===")
    print(
        f"{'Player':<24}{'Team':<5}{'Opp':<5}{'Market':<18}{'Side':<6}{'Line':>8}"
        f"{'Proj':>8}{'Odds':>8}{'ProbEdge':>10}{'BestEV':>10}  Kickoff"
    )
    for r in rows:
        pe = f"{r['prob_edge']*100:.1f}%" if isinstance(r["prob_edge"], (int, float)) else ""
        ev = f"{r['best_ev']*100:.1f}%" if isinstance(r["best_ev"], (int, float)) else ""
        print(
            f"{str(r['player'])[:23]:<24}{str(r['team'])[:4]:<5}{str(r['opp'])[:4]:<5}"
            f"{str(r['market'])[:17]:<18}{str(r['side'])[:5]:<6}{str(r['line']):>8}"
            f"{str(r['proj']):>8}{str(r['odds']):>8}{pe:>10}{ev:>10}  {r['kickoff']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
