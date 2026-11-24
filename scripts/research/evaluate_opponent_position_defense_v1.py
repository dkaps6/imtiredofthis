#!/usr/bin/env python3
"""OPPONENT_POSITION_DEFENSE_ADJUSTMENT_PREREGISTRATION_V1.md, exactly as frozen.

adjusted = proj * (1 + b * D), D = n/(n+4) * (allowed_pg / league_pg - 1), strictly prior,
per market x position group (QB / RB / WR / TE). Fold A: fit 2024 -> test 2025.
Fold B: fit 2024+2025 -> test 2026. QB rushing: fit 2026 wk 1-7, test wk 8-10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

FULL = "data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv"
BASE = "data/backtests/historical_market_vegas_benchmark_v1/historical_market_vegas_benchmark_detail.csv"
SLIM26 = "data/research/selective_pick/prism_2026_w1_2_6_10_mainline_graded_slim_v1.csv"
STAT = {"rush_yards": "rushing_yards", "rec_yards": "receiving_yards", "receptions": "receptions", "pass_yards": "passing_yards"}
GROUPS = ("QB", "RB", "WR", "TE")
SHRINK_K = 4.0
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "JAC": "JAX"}
nm = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)


def load_rows() -> pd.DataFrame:
    f = pd.read_csv(FULL, low_memory=False)
    f = f[f.market.isin(["rush_yards", "rec_yards", "receptions"])][["season", "week", "team", "player_clean_key", "market", "proj", "actual"]]
    b = pd.read_csv(BASE, low_memory=False)
    b = b[b.market == "pass_yards"].rename(columns={"mc_proj": "proj"})[["season", "week", "team", "player_clean_key", "market", "proj", "actual"]]
    m = pd.read_csv(SLIM26)
    m = m[(m.played == 1) & m.market.isin(list(STAT))].drop_duplicates(["season", "week", "player_clean_key", "market"])
    m = m.rename(columns={"model_proj": "proj"})[["season", "week", "team", "player_clean_key", "market", "proj", "actual"]]
    d = pd.concat([f, b, m], ignore_index=True).dropna(subset=["proj", "actual"])
    d["team"] = d.team.replace(TEAM_FIX)
    d["pk"] = nm(d.player_clean_key)
    return d[d.proj > 0]


def defense_table(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = nfl.load_player_stats(seasons=seasons).to_pandas()
    st = st[st.season_type == "REG"].copy()
    st["team"] = st.team.replace(TEAM_FIX)
    st["pk"] = nm(st.player_display_name)
    sch = nfl.load_schedules(seasons=seasons).to_pandas()
    sch = sch[sch.game_type == "REG"]
    opp = pd.concat([sch.assign(team=sch.home_team, opp=sch.away_team), sch.assign(team=sch.away_team, opp=sch.home_team)])[["season", "week", "team", "opp"]]
    opp["team"] = opp.team.replace(TEAM_FIX); opp["opp"] = opp.opp.replace(TEAM_FIX)
    s = st.merge(opp, on=["season", "week", "team"])
    rows = []
    for mk, col in STAT.items():
        for g in GROUPS:
            gm = s[s.position == g].groupby(["season", "week", "opp"])[col].sum().reset_index()
            for season, gs in gm.groupby("season"):
                for wk in range(2, 19):
                    prior = gs[gs.week < wk]
                    if prior.empty:
                        continue
                    per = prior.groupby("opp")[col].agg(["mean", "size"])
                    lg = per["mean"].mean()
                    if not lg > 0:
                        continue
                    shrink = per["size"] / (per["size"] + SHRINK_K)
                    rows.append(pd.DataFrame({"season": season, "week": wk, "opp": per.index, "market": mk, "group": g,
                                              "D": (shrink * (per["mean"] / lg - 1)).values}))
    pos = st.sort_values("week").groupby(["season", "pk"]).position.last().rename("group").reset_index()
    return pd.concat(rows, ignore_index=True), pos


def fit_b(d: pd.DataFrame) -> float:
    x = (d.proj * d.D).to_numpy(float); y = (d.actual - d.proj).to_numpy(float)
    den = float((x * x).sum())
    return float((x * y).sum() / den) if den > 0 else 0.0


def score(test: pd.DataFrame, b: float, rng) -> dict:
    adj = test.proj * (1 + b * test.D)
    e0, e1 = (test.actual - test.proj).abs(), (test.actual - adj).abs()
    gkey = test.season.astype(str) + "_" + test.week.astype(str) + "_" + test.opp.astype(str)
    diff = (e0 - e1).groupby(gkey).agg(["sum", "size"])
    s, n = diff["sum"].to_numpy(), diff["size"].to_numpy(); k = len(diff)
    boots = [s[i].sum() / n[i].sum() for i in (rng.integers(0, k, k) for _ in range(3000))] if k else [0]
    return dict(n=len(test), mae_model=float(e0.mean()), mae_adjusted=float(e1.mean()), mae_gain=float(e0.mean() - e1.mean()),
                p_improve=float(np.mean(np.array(boots) > 0)))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="data/research/opponent_position_defense")
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261122)
    d = load_rows()
    D, pos = defense_table([2024, 2025, 2026])
    d = d.merge(pos, on=["season", "pk"], how="left")
    sch = nfl.load_schedules(seasons=[2024, 2025, 2026]).to_pandas(); sch = sch[sch.game_type == "REG"]
    opp = pd.concat([sch.assign(team=sch.home_team, opp=sch.away_team), sch.assign(team=sch.away_team, opp=sch.home_team)])[["season", "week", "team", "opp"]]
    opp["team"] = opp.team.replace(TEAM_FIX); opp["opp"] = opp.opp.replace(TEAM_FIX)
    d = d.merge(opp, on=["season", "week", "team"], how="left")
    d = d.merge(D, on=["season", "week", "opp", "market", "group"], how="left")
    d["D"] = d.D.fillna(0.0)
    print(f"rows {len(d)}; position found {d.group.notna().mean():.1%}; opponent found {d.opp.notna().mean():.1%}")
    d = d[d.group.isin(GROUPS)]
    res = []
    for (mk, g), c in d.groupby(["market", "group"]):
        r = dict(market=mk, group=g)
        a_fit, a_test = c[c.season == 2024], c[c.season == 2025]
        b_fit, b_test = c[c.season.isin([2024, 2025])], c[c.season == 2026]
        if mk == "rush_yards" and g == "QB":
            a_fit, a_test = c[(c.season == 2026) & (c.week <= 7)], c[(c.season == 2026) & (c.week >= 8)]
            b_fit, b_test = a_fit, a_test
        for lab, fit, test in [("A", a_fit, a_test), ("B", b_fit, b_test)]:
            b = fit_b(fit) if len(fit) else np.nan
            s = score(test, b, rng) if len(test) and np.isfinite(b) else dict(n=len(test))
            r.update({f"{lab}_fit_n": len(fit), f"{lab}_b": b, **{f"{lab}_{k}": v for k, v in s.items()}})
        ok_n = r.get("A_n", 0) >= 150 and r.get("B_n", 0) >= 60
        if mk == "rush_yards" and g == "QB":
            ok_n = r.get("B_n", 0) >= 40
        passes = (ok_n and r.get("A_mae_gain", -1) > 0 and r.get("B_mae_gain", -1) > 0 and r.get("A_b", -1) > 0
                  and r.get("B_b", -1) > 0 and r.get("B_p_improve", 0) >= 0.90)
        r["verdict"] = "PROMOTE" if passes else ("TOO FEW ROWS" if not ok_n else "DO NOT PROMOTE")
        res.append(r)
    T = pd.DataFrame(res)
    T.to_csv(out / "opponent_position_defense_v1.csv", index=False)
    promoted = {f"{r.market}|{r.group}": round(float(r.B_b), 4) for r in T.itertuples() if r.verdict == "PROMOTE"}
    (out / "promoted_coefficients_v1.json").write_text(json.dumps(dict(shrink_k=SHRINK_K, coefficients=promoted), indent=2))
    pd.set_option("display.width", 250)
    cols = ["market", "group", "A_n", "A_b", "A_mae_model", "A_mae_gain", "B_n", "B_b", "B_mae_model", "B_mae_gain", "B_p_improve", "verdict"]
    print(T[cols].round(3).to_string(index=False))
    print("promoted:", promoted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
