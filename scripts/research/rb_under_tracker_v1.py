#!/usr/bin/env python3
"""Weekly forward record of the RB rushing-UNDER rule. Research only.

The rule is frozen as of 2026-11-19 and is not tuned after any graded week:
  * FanDuel main-line RB rush_yards UNDER;
  * at least 2 of 3 flags from rb_rush_situation_flags_v1.py: team underdog by 3+,
    YPC below 95% of the league RB average, volume projection > 1.2x the line;
  * line >= 30 (lines under 30 hit 51% in the backtest: backups).

  log    --week N --board <archived board csv> --capture <label>   appends picks
  grade  --week N                                                  fills results

The ledger is data/research/rb_under_tracker/ledger.csv (committed). A pick is VOID when
the player takes no offensive snap or carries an Out/Doubtful designation before kickoff.
Graded at the captured price; the price actually bet may differ.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.research.rb_rush_situation_flags_v1 import flag

LEDGER = Path("data/research/rb_under_tracker/ledger.csv")
MIN_FLAGS, MIN_LINE = 2, 30.0
COLS = ["season", "week", "capture", "player", "team", "opponent", "fav_by", "line", "under_price",
        "dog", "cold", "cut", "n_flags", "injury_at_log", "status", "actual", "result", "units"]


def log(a) -> int:
    b = pd.read_csv(a.board); b = b[(b.side == "UNDER") & (b.week == a.week)]
    f = flag(b, a.season)
    f = f[(f.n_flags >= MIN_FLAGS) & (f.vegas_line >= MIN_LINE)].copy()
    ij = nfl.load_injuries(seasons=[a.season]).to_pandas(); ij = ij[ij.week == a.week]
    inj = ij.set_index("full_name").report_status.astype(str).to_dict()
    rows = pd.DataFrame({"season": a.season, "week": a.week, "capture": a.capture, "player": f.player, "team": f.team,
                         "opponent": f.opponent, "fav_by": f.fav_by, "line": f.vegas_line, "under_price": f.vegas_under_odds,
                         "dog": f.dog, "cold": f.cold, "cut": f.cut, "n_flags": f.n_flags,
                         "injury_at_log": f.player.map(inj).fillna("none"), "status": "OPEN",
                         "actual": np.nan, "result": "", "units": np.nan})
    rows.loc[rows.injury_at_log.str.lower().isin(["out", "doubtful"]), "status"] = "VOID"
    old = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=COLS)
    key = ["season", "week", "player"]
    new = rows[~rows.set_index(key).index.isin(old.set_index(key).index)]
    out = pd.concat([old, new], ignore_index=True)[COLS]
    LEDGER.parent.mkdir(parents=True, exist_ok=True); out.to_csv(LEDGER, index=False)
    print(new[["player", "team", "opponent", "line", "under_price", "n_flags", "injury_at_log", "status"]].to_string(index=False))
    return 0


def grade(a) -> int:
    L = pd.read_csv(LEDGER)
    st = nfl.load_player_stats(seasons=[a.season]).to_pandas(); st = st[st.season_type == "REG"]
    sn = nfl.load_snap_counts(seasons=[a.season]).to_pandas()
    played = set(zip(sn[sn.offense_snaps.fillna(0) > 0].player, sn[sn.offense_snaps.fillna(0) > 0].week))
    ry = st.groupby(["player_display_name", "week"]).rushing_yards.sum().to_dict()
    m = (L.season == a.season) & (L.week == a.week) & (L.status == "OPEN")
    for i in L.index[m]:
        p, w = L.at[i, "player"], int(L.at[i, "week"])
        if (p, w) not in played:
            L.loc[i, ["status", "result", "units"]] = ["VOID", "no snap", 0.0]; continue
        act = float(ry.get((p, w), 0.0)); price = float(L.at[i, "under_price"])
        win = act < float(L.at[i, "line"])
        L.loc[i, ["status", "actual", "result", "units"]] = ["GRADED", act, "WIN" if win else "LOSS",
                                                               (price / 100 if price > 0 else 100 / abs(price)) if win else -1.0]
    L.to_csv(LEDGER, index=False)
    g = L[L.status == "GRADED"]
    print(L[(L.week == a.week) & (L.season == a.season)][["player", "line", "actual", "result", "units"]].to_string(index=False))
    if len(g):
        print(f"\nRUNNING RECORD: {int((g.result == 'WIN').sum())}-{int((g.result == 'LOSS').sum())}, "
              f"units {g.units.sum():+.2f} on {len(g)} bets ({g.units.mean():+.1%} per bet)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["log", "grade"])
    ap.add_argument("--season", type=int, default=2026); ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--board"); ap.add_argument("--capture", default="")
    a = ap.parse_args()
    return log(a) if a.cmd == "log" else grade(a)


if __name__ == "__main__":
    raise SystemExit(main())
