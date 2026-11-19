#!/usr/bin/env python3
"""Re-test of REVERSE_ENGINEER_2026_W1_10_V1.md §2 with a valid design.

Why V2. The first walk-forward (2026-11-19, part 1) compared an UNFIT team
average against a FITTED "team average + spread + total" regression, so the
fitted arm could win merely by shrinking the team average -- the exact flaw
PR #558's review caught (MARKET_IMPLIED_GAME_SCRIPT_V1_PLAN.md, amendment 3).
Its "production constant 0.57" baseline also used LEAGUE plays, while
production's rules_v2.project_game_script() uses team-specific plays x 0.57.
And it had seven weeks of one season.

This version:
  * targets: team rush attempts, dropbacks, plays, pass rate (the last two are
    #558's targets, so the two studies can be compared);
  * arms, every one strictly prior:
      A0 production proxy -- team rolling plays x (1-0.57) for rushes,
         x 0.57 for dropbacks, 0.57 for pass rate (rules_v2's constant)
      A1 team rolling mean of the target itself, unfit
      A2 A1 refit by linear regression           (the correct comparator)
      A3 A1 + signed team spread + total, fit     (market value = A2 vs A3)
  * rolling window 8 team-games, min 3, crossing season boundaries (as #558);
  * holdouts: fit 2023-2024 -> test 2025; fit 2023-2025 -> test 2026 wk1-10.
  * game-clustered bootstrap CI on the A2-A3 MAE difference.
Spread/total: nflverse schedule spread_line/total_line (pregame closing-ish
lines; sign: + = home favored). No player props are used.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

PROD_PASS_SHARE = 0.57
WINDOW, MIN_PRIOR = 8, 3


def team_games(seasons) -> pd.DataFrame:
    p = nfl.load_pbp(seasons=list(seasons)).to_pandas()
    p = p[p.season_type == "REG"]
    x = p[((p["pass"] == 1) | (p["rush"] == 1)) & (p.qb_kneel != 1) & (p.qb_spike != 1) & p.posteam.notna()]
    T = (x.groupby(["season", "week", "game_id", "posteam", "defteam"])
          .agg(plays=("pass", "size"), dropbacks=("pass", "sum"), rushes=("rush", "sum")).reset_index())
    T["pass_rate"] = T.dropbacks / T.plays
    s = nfl.load_schedules(seasons=list(seasons)).to_pandas()[["game_id", "home_team", "spread_line", "total_line"]]
    T = T.merge(s, on="game_id", how="left")
    T["fav_by"] = np.where(T.posteam == T.home_team, T.spread_line, -T.spread_line)
    T = T.sort_values(["posteam", "season", "week"])
    for c in ["plays", "dropbacks", "rushes", "pass_rate"]:
        T[c + "_prior"] = T.groupby("posteam")[c].transform(lambda v: v.shift().rolling(WINDOW, min_periods=MIN_PRIOR).mean())
    return T.dropna(subset=["plays_prior", "fav_by", "total_line"])


def fit(X, y):
    X = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def pred(b, X):
    return np.c_[np.ones(len(X)), X] @ b


def evaluate(T, train_mask, test_mask, label, rng):
    tr, te = T[train_mask], T[test_mask]
    rows = []
    for y in ["rushes", "dropbacks", "plays", "pass_rate"]:
        base = y + "_prior"
        a0 = {"rushes": te.plays_prior * (1 - PROD_PASS_SHARE), "dropbacks": te.plays_prior * PROD_PASS_SHARE,
              "plays": te.plays_prior, "pass_rate": pd.Series(PROD_PASS_SHARE, index=te.index)}[y].to_numpy()
        a1 = te[base].to_numpy()
        b2 = fit(tr[[base]].to_numpy(), tr[y].to_numpy()); a2 = pred(b2, te[[base]].to_numpy())
        cols = [base, "fav_by", "total_line"]
        b3 = fit(tr[cols].to_numpy(), tr[y].to_numpy()); a3 = pred(b3, te[cols].to_numpy())
        yt = te[y].to_numpy()
        e2, e3 = np.abs(yt - a2), np.abs(yt - a3)
        d = pd.DataFrame({"g": te.game_id.to_numpy(), "diff": e2 - e3})
        gs = d.groupby("g")["diff"].agg(["sum", "size"])
        bs = [gs["sum"].to_numpy()[i].sum() / gs["size"].to_numpy()[i].sum()
              for i in (rng.integers(0, len(gs), len(gs)) for _ in range(2000))]
        rows.append(dict(holdout=label, target=y, test_rows=len(te),
                         A0_production_proxy=np.abs(yt - a0).mean(), A1_team_avg_unfit=np.abs(yt - a1).mean(),
                         A2_team_avg_fitted=e2.mean(), A3_plus_spread_total=e3.mean(),
                         market_gain=e2.mean() - e3.mean(), gain_ci_lo=np.percentile(bs, 2.5), gain_ci_hi=np.percentile(bs, 97.5),
                         spread_coef=b3[2], total_coef=b3[3]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/reverse_engineer/team_volume_market_increment_v2.csv")
    a = ap.parse_args()
    T = team_games(range(2022, 2027))
    T = T[T.season >= 2023]
    rng = np.random.default_rng(558)
    rows = evaluate(T, T.season.isin([2023, 2024]), T.season == 2025, "fit 2023-24, test 2025", rng)
    rows += evaluate(T, T.season.isin([2023, 2024, 2025]), T.season == 2026, "fit 2023-25, test 2026 wk1-10", rng)
    R = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); R.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
