#!/usr/bin/env python3
"""Grade an archived main-line production board against nflverse actuals.

Weeks 1-8 of 2026 have NO alternate-line data -- the alt ladder capture
(`fetch_alt_lines` in full-slate.yml) was first run in week 9. So the week-9
ladder analysis cannot be reproduced for earlier weeks. What CAN be tested on
earlier weeks is the mechanism that analysis uncovered: whether a roughly
constant calibration error costs progressively more ROI as the price lengthens.
This script grades the archived boards so that test can run.

TWO TRAPS THIS SCRIPT EXISTS TO AVOID
-------------------------------------
1. UNPRICED ROWS. The board carries an OVER and an UNDER row for every offer,
   but `anytime_td` has no UNDER price (books do not offer "will not score"):
   `vegas_under_odds` is null on 100% of those rows. Grading them anyway yields
   a hit rate of 81% and an ROI of -100% simultaneously, because pandas averages
   only the priced losers. Rows without a finite price are not bets and are
   dropped.

2. THE ZERO-INVOLVEMENT DENOMINATOR. `load_player_stats` omits a player-game
   with no recorded stat, so a player who played and caught nothing is missing
   rather than zero. Treating "missing" as "no bet" silently deletes losses.
   This script resolves it with `load_snap_counts`: offense_snaps > 0 and no
   stat row means a true zero; no snaps at all means the player did not play and
   the row is dropped as unsettleable.

Weeks 1-2 are already archived graded, but they hold only the model's SELECTED
side (626 UNDER vs 179 OVER), not a full two-sided board, so they are a
different population and must not be pooled with a full board.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MARKETS = ("pass_yards", "rec_yards", "receptions", "rush_yards",
           "rush_rec_yards", "anytime_td")


def dec(american):
    a = np.asarray(american, dtype=float)
    return np.where(a > 0, a / 100.0, 100.0 / (-a))


def build_actuals(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    w = nfl.load_player_stats(seasons=[season]).to_pandas()
    w = w[w.season_type == "REG"]
    z = lambda c: w[c].fillna(0)
    A = w.assign(
        pass_yards=z("passing_yards"), rec_yards=z("receiving_yards"),
        receptions=z("receptions"), rush_yards=z("rushing_yards"),
        rush_rec_yards=z("rushing_yards") + z("receiving_yards"),
        anytime_td=((z("rushing_tds") + z("receiving_tds")) > 0).astype(float),
    )[["player_display_name", "week", *MARKETS]]
    A = A.melt(id_vars=["player_display_name", "week"],
               var_name="market", value_name="actual")
    return A.rename(columns={"player_display_name": "player"})


def build_played(season: int) -> pd.DataFrame:
    """Player-weeks with at least one offensive snap -- the real denominator."""
    import nflreadpy as nfl
    s = nfl.load_snap_counts(seasons=[season]).to_pandas()
    if "game_type" in s.columns:
        s = s[s.game_type == "REG"]
    return (s[s.offense_snaps.fillna(0) > 0][["player", "week"]]
            .drop_duplicates().assign(played=1))


def grade(board: pd.DataFrame, actuals: pd.DataFrame,
          played: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    b = board.copy()
    b["week"] = b["week"].astype(int)
    m = (b.merge(actuals, on=["player", "week", "market"], how="left")
          .merge(played, on=["player", "week"], how="left"))
    m["played"] = m["played"].fillna(0)

    n_board = len(m)
    fill = m.actual.isna() & (m.played == 1)
    m.loc[fill, "actual"] = 0.0          # played, no stat row -> a true zero
    n_zero = int(fill.sum())
    n_nosnap = int(m.actual.isna().sum())
    m = m[m.actual.notna()].copy()

    priced = np.isfinite(pd.to_numeric(m.vegas_odds, errors="coerce"))
    n_unpriced = int((~priced).sum())
    m = m[priced].copy()
    m["vegas_odds"] = pd.to_numeric(m.vegas_odds, errors="coerce")

    over = m.side.astype(str).str.upper() == "OVER"
    binary = m.market == "anytime_td"
    m["hit"] = np.where(
        binary,
        np.where(over, (m.actual > 0).astype(int), (m.actual == 0).astype(int)),
        np.where(over, (m.actual > m.vegas_line).astype(int),
                 (m.actual < m.vegas_line).astype(int)))
    m["pnl"] = np.where(m.hit == 1, dec(m.vegas_odds), -1.0)
    m["breakeven"] = 1.0 / (1.0 + dec(m.vegas_odds))

    assert m.pnl.notna().all(), "unpriced row survived grading"
    return m, {"board_rows": n_board, "graded": len(m),
               "zeros_via_snaps": n_zero, "dropped_no_snaps": n_nosnap,
               "dropped_unpriced": n_unpriced}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", nargs="+", required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    boards = pd.concat([pd.read_csv(p) for p in a.boards], ignore_index=True)
    m, st = grade(boards, build_actuals(a.season), build_played(a.season))
    for k, v in st.items():
        print(f"  {k}: {v}")
    print(f"\noverall ROI {100 * m.pnl.mean():+.1f}%  hit {100 * m.hit.mean():.1f}%")
    print("\nby market (priced rows only):")
    print(m.groupby("market").agg(n=("pnl", "size"), hit=("hit", "mean"),
                                  roi=("pnl", "mean"))
          .assign(hit=lambda d: (100 * d.hit).round(1),
                  roi=lambda d: (100 * d.roi).round(1)).to_string())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
