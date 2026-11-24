#!/usr/bin/env python3
"""ROLE_CHANGE_AND_RB_UNDER_HISTORICAL_PREREGISTRATION_V1.md, exactly as frozen.

Tests A (RB-under rule), B (role-change rule) and C (blind sides) on the 2024-2025
historical prop archive. Players matched to gsis by team + week + first-initial.lastname.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

ARCH = "data/backtests/historical_market_vegas_benchmark_v1/historical_market_vegas_benchmark_detail.csv"
SUF = {"jr", "sr", "ii", "iii", "iv", "v"}
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "JAC": "JAX"}


def key(name: str) -> str:
    t = [re.sub(r"[^a-z]", "", w) for w in str(name).lower().replace("-", " ").split()]
    t = [w for w in t if w and w not in SUF]
    return f"{t[0][0]}.{t[-1]}" if len(t) >= 2 else ""


def dec(a):
    a = np.asarray(a, float); return np.where(a > 0, a / 100, 100 / np.abs(a))


def logs(seasons):
    st = nfl.load_player_stats(seasons=list(seasons)).to_pandas(); st = st[st.season_type == "REG"].copy()
    for c in ["carries", "rushing_yards", "targets", "receiving_yards"]:
        st[c] = st[c].fillna(0)
    tm = st.groupby(["season", "week", "team"])[["carries", "targets"]].sum().add_prefix("tm_").reset_index()
    st = st.merge(tm, on=["season", "week", "team"])
    st["k"] = st.player_display_name.map(key)
    return st.sort_values(["season", "week"])


def features(st, season, week):
    """Strictly-prior features for every player as of (season, week)."""
    cur = st[(st.season == season) & (st.week < week)]
    prev = st[st.season == season - 1]
    rb = pd.concat([prev, cur]) if week <= 2 else cur
    rbp = rb[rb.position == "RB"]; lg = rbp.rushing_yards.sum() / max(rbp.carries.sum(), 1)
    g = rb.groupby("player_id").agg(gp=("week", "size"), car=("carries", "sum"), ry=("rushing_yards", "sum"))
    g["cpg"] = g.car / g.gp; g["ypc"] = g.ry / g.car.replace(0, np.nan); g["lg_ypc"] = lg
    # role change: current season, current team only
    rows = []
    for pid, h in cur.groupby("player_id"):
        h = h.sort_values("week"); t = h.team.iloc[-1]; h = h[h.team == t]
        if len(h) < 4:
            continue
        std_c = h.carries.sum() / max(h.tm_carries.sum(), 1); l3 = h.tail(3)
        l3_c = l3.carries.sum() / max(l3.tm_carries.sum(), 1)
        std_t = h.targets.sum() / max(h.tm_targets.sum(), 1); l3_t = l3.targets.sum() / max(l3.tm_targets.sum(), 1)
        rows.append(dict(player_id=pid, d_car=l3_c - std_c, d_tgt=l3_t - std_t, cur_games=len(h)))
    rc = pd.DataFrame(rows).set_index("player_id") if rows else pd.DataFrame(columns=["d_car", "d_tgt", "cur_games"])
    return g.join(rc, how="left")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="data/research/historical_rules")
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    H = pd.read_csv(ARCH, low_memory=False)
    H["team"] = H.team.replace(TEAM_FIX); H["k"] = H.player.astype(str).str.lower().str.replace(r"[^a-z.]", "", regex=True)
    st = logs([2023, 2024, 2025])
    ids = st[["season", "week", "team", "k", "player_id", "position"]].drop_duplicates(["season", "week", "team", "k"])
    H = H.merge(ids, on=["season", "week", "team", "k"], how="left")
    print(f"archive rows {len(H)}, matched to gsis {H.player_id.notna().mean():.1%}")
    sch = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()[["season", "week", "home_team", "away_team", "spread_line", "game_id"]]
    sp = pd.concat([sch.rename(columns={"home_team": "team"}).assign(fav_by=sch.spread_line)[["season", "week", "team", "fav_by", "game_id"]],
                    sch.rename(columns={"away_team": "team"}).assign(fav_by=-sch.spread_line)[["season", "week", "team", "fav_by", "game_id"]]])
    H = H.drop(columns=["game_id"], errors="ignore").merge(sp, on=["season", "week", "team"], how="left")
    F = []
    for (s, w), _ in H.groupby(["season", "week"]):
        f = features(st, s, w).reset_index().assign(season=s, week=w); F.append(f)
    F = pd.concat(F)
    H = H.merge(F, on=["season", "week", "player_id"], how="left")
    H["u_win"] = (H.actual < H.line).astype(float); H["o_win"] = (H.actual > H.line).astype(float)
    H["push"] = H.actual == H.line; H = H[~H.push]
    H["u_pnl"] = np.where(H.u_win == 1, dec(H.under_odds), -1.0); H["o_pnl"] = np.where(H.o_win == 1, dec(H.over_odds), -1.0)
    rng = np.random.default_rng(20261120)

    def summ(d, pnl, win, label):
        if len(d) == 0:
            return dict(test=label, n=0)
        gg = d.groupby("game_id")[pnl].agg(["sum", "size"])
        bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum() for i in (rng.integers(0, len(gg), len(gg)) for _ in range(3000))]
        by = d.groupby("season").agg(n=(pnl, "size"), hit=(win, "mean"), roi=(pnl, "mean"))
        wk = d.groupby(["season", "week"])[win].mean()
        return dict(test=label, n=len(d), hit=d[win].mean(), roi=d[pnl].mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                    n_2024=by.n.get(2024, 0), hit_2024=by.hit.get(2024, np.nan), roi_2024=by.roi.get(2024, np.nan),
                    n_2025=by.n.get(2025, 0), hit_2025=by.hit.get(2025, np.nan), roi_2025=by.roi.get(2025, np.nan),
                    weeks_above_breakeven=f"{int((wk > 0.524).sum())}/{len(wk)}")
    R = []
    rb = H[(H.market == "rush_yards") & (H.position == "RB") & (H.gp >= 3)].copy()
    rb["dog"] = (rb.fav_by < -2.5).astype(int); rb["cold"] = (rb.ypc / rb.lg_ypc < 0.95).astype(int)
    rb["cut"] = (rb.cpg * (0.79 * rb.lg_ypc + 0.21 * rb.ypc.fillna(rb.lg_ypc)) / rb.line > 1.2).astype(int)
    rb["n_flags"] = rb.dog + rb.cold + rb.cut
    R.append(summ(rb[(rb.n_flags >= 2) & (rb.line >= 30)], "u_pnl", "u_win", "A: RB UNDER, 2+ flags, line>=30 (the frozen rule)"))
    R.append(summ(rb[(rb.n_flags >= 2) & (rb.line < 30)], "u_pnl", "u_win", "A-report: RB UNDER, 2+ flags, line<30"))
    R.append(summ(rb[(rb.n_flags <= 1) & (rb.line >= 30)], "u_pnl", "u_win", "A-report: RB UNDER, 0-1 flags, line>=30"))
    for nf in (0, 1, 2, 3):
        R.append(summ(rb[(rb.n_flags == nf) & (rb.line >= 30)], "u_pnl", "u_win", f"A-report: RB UNDER, exactly {nf} flags, line>=30"))
    rc = H[(H.market == "rush_yards") & (H.position == "RB") & H.d_car.notna()]
    R.append(summ(rc[rc.d_car >= 0.15], "o_pnl", "o_win", "B: RB rising role (d_car>=+0.15) -> OVER"))
    R.append(summ(rc[rc.d_car <= -0.15], "u_pnl", "u_win", "B: RB falling role (d_car<=-0.15) -> UNDER"))
    rr = H[(H.market == "rec_yards") & H.position.isin(["WR", "TE"]) & H.d_tgt.notna()]
    R.append(summ(rr[rr.d_tgt >= 0.08], "o_pnl", "o_win", "B-secondary: WR/TE rising targets (>=+0.08) -> OVER"))
    R.append(summ(rr[rr.d_tgt <= -0.08], "u_pnl", "u_win", "B-secondary: WR/TE falling targets (<=-0.08) -> UNDER"))
    for mk in ["rush_yards", "rec_yards", "receptions", "pass_yards", "rush_rec_yards"]:
        d = H[H.market == mk]
        R.append(summ(d, "o_pnl", "o_win", f"C: every {mk} OVER")); R.append(summ(d, "u_pnl", "u_win", f"C: every {mk} UNDER"))
    T = pd.DataFrame(R); T.to_csv(out / "role_change_rb_under_historical_v1.csv", index=False)
    rb.to_csv(out / "rb_under_flags_2024_2025_rows.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
