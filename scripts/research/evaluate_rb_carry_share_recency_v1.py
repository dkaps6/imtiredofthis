#!/usr/bin/env python3
"""Evaluate RB_CARRY_SHARE_RECENCY_PREREGISTRATION_V1.md exactly as frozen.

C0 = PlayerForm's blend: (1-w)*prior_season_share + w*season_to_date, w = g/(g+4).
C1 = same, season_to_date replaced by (1-a)*season_to_date + a*last3.
a fitted on 2023-2024 (grid 0..1 step 0.05), frozen; scored on 2025 and 2026 wk1-10.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

PRIOR_K = 4.0


def rb_games(seasons) -> pd.DataFrame:
    # Join snaps to stats on IDs (pfr -> gsis), not names: a name join silently zeroed
    # "Kenneth"/"Kenny" Gainwell, Brian Robinson Jr., Ollie Gordon II, Chase Brown, etc.
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    out = []
    for yr in seasons:
        st = nfl.load_player_stats(seasons=[yr]).to_pandas(); st = st[st.season_type == "REG"]
        tm = st.groupby(["team", "week"]).carries.sum().rename("team_carries").reset_index()
        sn = nfl.load_snap_counts(seasons=[yr]).to_pandas()
        sn = sn[(sn.offense_snaps.fillna(0) > 0) & (sn.position == "RB")]
        sn = sn.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
        sn["name"] = sn.gsis_id.fillna("pfr:" + sn.pfr_player_id.astype(str))
        sn = sn[["name", "week", "team"]]
        d = sn.merge(
            st[["player_id", "week", "team", "carries"]].rename(columns={"player_id": "name"}),
            on=["name", "week", "team"], how="left")
        d["carries"] = d.carries.fillna(0)
        d = d.merge(tm, on=["team", "week"], how="inner")
        d = d[d.team_carries > 0]; d["season"] = yr
        out.append(d)
    return pd.concat(out, ignore_index=True)


def build(D: pd.DataFrame) -> pd.DataFrame:
    D = D.sort_values(["season", "week"])
    # prior-season share: total carries / total team carries in games played, previous season
    tot = D.groupby(["season", "name"]).agg(c=("carries", "sum"), t=("team_carries", "sum")).reset_index()
    tot["prior_share"] = tot.c / tot.t; tot["season"] += 1
    D = D.merge(tot[["season", "name", "prior_share"]], on=["season", "name"], how="left")
    g = D.groupby(["season", "name", "team"])
    D["share"] = D.carries / D.team_carries
    D["cum_c"] = g.carries.transform(lambda s: s.shift().cumsum()); D["cum_t"] = g.team_carries.transform(lambda s: s.shift().cumsum())
    D["std_share"] = D.cum_c / D.cum_t
    D["l3_c"] = g.carries.transform(lambda s: s.shift().rolling(3, min_periods=3).sum())
    D["l3_t"] = g.team_carries.transform(lambda s: s.shift().rolling(3, min_periods=3).sum())
    D["l3_share"] = D.l3_c / D.l3_t
    D["g"] = g.cumcount()
    D["oracle"] = g.share.transform("mean")   # full-season mean, uses future games: headroom only
    D = D[(D.g >= 3) & D.std_share.notna() & D.l3_share.notna()].copy()
    D["w"] = D.g / (D.g + PRIOR_K)
    return D


def arm(D, a):
    cur = (1 - a) * D.std_share + a * D.l3_share
    return np.where(D.prior_share.notna(), (1 - D.w) * D.prior_share + D.w * cur, cur)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/rb_carry_share_recency_v1.csv")
    a_ = ap.parse_args()
    D = build(rb_games(range(2022, 2027)))
    D = D[D.season >= 2023]
    fit = D[D.season.isin([2023, 2024])]
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    fit_mae = {a: np.abs(fit.share - arm(fit, a)).mean() for a in grid}
    a_star = min(fit_mae, key=fit_mae.get)
    print(f"fitted a on 2023-2024 = {a_star}  (fit MAE C0 {fit_mae[0.0]:.5f} -> {fit_mae[a_star]:.5f})")
    rng = np.random.default_rng(20261119)
    rows = []
    for label, te in [("2025", D[D.season == 2025]), ("2026 wk1-10", D[D.season == 2026])]:
        te = te.copy(); te["c0"] = arm(te, 0.0); te["c1"] = arm(te, a_star)
        te["gid"] = te.season.astype(str) + "_" + te.week.astype(str) + "_" + te.team
        def stat(x):
            e0, e1 = np.abs(x.share - x.c0), np.abs(x.share - x.c1)
            gg = pd.DataFrame({"gid": x.gid, "d": e0 - e1}).groupby("gid").d.agg(["sum", "size"])
            bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum()
                  for i in (rng.integers(0, len(gg), len(gg)) for _ in range(2000))]
            return dict(n=len(x), mae_c0=e0.mean(), mae_c1=e1.mean(), gain=e0.mean() - e1.mean(),
                        ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                        oracle_mae=np.abs(x.share - x.oracle).mean())
        for sub, x in [("all", te), ("lead backs (C0>=0.40)", te[te.c0 >= 0.40]),
                       ("role change (|l3-std|>=0.15)", te[(te.l3_share - te.std_share).abs() >= 0.15]),
                       ("steady (|l3-std|<0.15)", te[(te.l3_share - te.std_share).abs() < 0.15])]:
            rows.append(dict(holdout=label, subset=sub, a=a_star, **stat(x)))
        ts = te.groupby("gid")[["c0", "c1", "share"]].sum().mean()
        rows.append(dict(holdout=label, subset="team-sum mean (C0 / C1 / actual)", a=a_star,
                         mae_c0=ts.c0, mae_c1=ts.c1, oracle_mae=ts.share))
    R = pd.DataFrame(rows)
    Path(a_.out).parent.mkdir(parents=True, exist_ok=True); R.to_csv(a_.out, index=False)
    pd.set_option("display.width", 230)
    print(R.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
