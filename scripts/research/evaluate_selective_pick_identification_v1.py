#!/usr/bin/env python3
"""SELECTIVE_PICK_IDENTIFICATION_PREREGISTRATION_V1.md, exactly as frozen.

Can a pregame-recognizable subset of the model's picks beat the price?
Selectors S1-S6 are chosen/fit on one season and scored on the other (both directions).

Data defect handled here (disclosed in the results doc): the full-stack detail's `game_id`
is the PRIOR season's game for that team/week (e.g. 2024 wk1 NYJ carries 2023_01_BUF_NYJ).
Team, opponent, line and actual are correct (actual verified 100% against nflverse), so the
cluster key is rebuilt from season + week + the two teams, and that key is checked against
the nflverse schedule.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FULL = "data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv"
BASE = "data/backtests/historical_market_vegas_benchmark_v1/historical_market_vegas_benchmark_detail.csv"
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "JAC": "JAX"}
MARKETS = ["rush_yards", "rec_yards", "receptions", "rush_rec_yards"]
SEED = 20261120


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, a / 100, 100 / np.abs(a))


def prep(d: pd.DataFrame, proj: str) -> pd.DataFrame:
    d = d.copy()
    d["team"] = d.team.replace(TEAM_FIX)
    if "opponent" in d:
        d["opponent"] = d.opponent.replace(TEAM_FIX)
    d = d[d[proj] != d.line].dropna(subset=["over_odds", "under_odds", "actual", "line", proj])
    d = d[d.actual != d.line]
    d["pick"] = np.where(d[proj] > d.line, "OVER", "UNDER")
    d["win"] = np.where(d.pick == "OVER", d.actual > d.line, d.actual < d.line).astype(int)
    d["price"] = np.where(d.pick == "OVER", d.over_odds, d.under_odds)
    d["pnl"] = np.where(d.win == 1, dec(d.price), -1.0)
    d["u_win"] = (d.actual < d.line).astype(int)
    d["u_pnl"] = np.where(d.u_win == 1, dec(d.under_odds), -1.0)
    d["rel_edge"] = (d[proj] - d.line) / np.maximum(d.line, 0.5)
    d["abs_rel"] = d.rel_edge.abs()
    d["imp"] = np.where(d.price < 0, -d.price / (-d.price + 100), 100 / (d.price + 100))
    return d


def add_schedule(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    sch = nfl.load_schedules(seasons=sorted(d.season.unique().tolist())).to_pandas()
    sch = sch[sch.game_type == "REG"]
    h = sch.assign(team=sch.home_team, opp=sch.away_team, fav_by=sch.spread_line)
    a = sch.assign(team=sch.away_team, opp=sch.home_team, fav_by=-sch.spread_line)
    s = pd.concat([h, a])[["season", "week", "team", "opp", "fav_by", "total_line", "game_id"]]
    s = s.rename(columns={"game_id": "gkey"})
    d = d.drop(columns=["game_id"], errors="ignore").merge(s, on=["season", "week", "team"], how="left")
    opp_ok = (d.opp == d.opponent).mean() if "opponent" in d else np.nan
    return d, float(opp_ok)


def player_record(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["season", "player_clean_key", "market", "week"]).copy()
    g = d.groupby(["season", "player_clean_key", "market"])["win"]
    d["prior_n"] = g.cumcount()
    d["prior_w"] = g.cumsum() - d.win
    d["prior_rate"] = np.where(d.prior_n > 0, d.prior_w / d.prior_n.replace(0, 1), 0.5)
    return d


def boot(d: pd.DataFrame, pnl: str, key: str, rng, reps=3000):
    g = d.groupby(key)[pnl].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    k = len(g)
    r = [s[i].sum() / n[i].sum() for i in (rng.integers(0, k, k) for _ in range(reps))]
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


S6_NUM = ["rel_edge", "abs_rel", "line", "imp", "sd_rel", "fav_by", "total_line", "week", "prior_n", "prior_rate"]


def s6_matrix(d: pd.DataFrame, cells: list[str], fill: dict) -> np.ndarray:
    x = d[S6_NUM].copy()
    for c in S6_NUM:
        x[c] = x[c].fillna(fill[c])
    oh = np.column_stack([(d.market + "_" + d.pick == c).astype(float) for c in cells])
    return np.column_stack([x.to_numpy(float), oh])


def select(fit: pd.DataFrame, sc: pd.DataFrame) -> dict[str, pd.Series]:
    """Every selector is chosen on `fit` and returned as a mask over `sc`."""
    out = {}
    out["S1 model UNDER picks"] = sc.pick == "UNDER"
    cut = fit.loc[fit.pick == "UNDER", "abs_rel"].quantile(2 / 3)
    out[f"S2 model UNDER, top-third edge (cut {cut:.3f})"] = (sc.pick == "UNDER") & (sc.abs_rel >= cut)
    cells = fit.groupby(["market", "pick"]).agg(n=("pnl", "size"), roi=("pnl", "mean"))
    keep = cells[(cells.roi > 0) & (cells.n >= 150)].index.tolist()
    lab = ",".join(f"{m}:{p}" for m, p in keep) or "none"
    out[f"S3 cells positive in fit season [{lab}]"] = pd.Series(
        [(m, p) in keep for m, p in zip(sc.market, sc.pick)], index=sc.index)
    out["S4 price better than -110"] = pd.Series(dec(sc.price) > 100 / 110 + 1e-9, index=sc.index)
    out["S5 model's own record on player >=60% (>=4 prior)"] = (sc.prior_n >= 4) & (sc.prior_rate >= 0.6)
    cells_all = sorted((fit.market + "_" + fit.pick).unique())
    fill = {c: fit[c].mean() for c in S6_NUM}
    sca = StandardScaler().fit(s6_matrix(fit, cells_all, fill))
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sca.transform(s6_matrix(fit, cells_all, fill)), fit.win)
    p = lr.predict_proba(sca.transform(s6_matrix(sc, cells_all, fill)))[:, 1]
    out["S6 logistic top 20%"] = pd.Series(p >= np.quantile(p, 0.8), index=sc.index)
    return out


def summarize(d: pd.DataFrame, pnl: str, win: str, rng) -> dict:
    if len(d) == 0:
        return dict(n=0)
    wk = d.groupby(["season", "week"])[pnl].mean()
    gl, gh = boot(d, pnl, "gkey", rng)
    pl, ph = boot(d, pnl, "player_clean_key", rng)
    by = d.groupby("season").agg(n=(pnl, "size"), hit=(win, "mean"), roi=(pnl, "mean"))
    r = dict(n=len(d), hit=d[win].mean(), roi=d[pnl].mean(), ci_game_lo=gl, ci_game_hi=gh,
             ci_player_lo=pl, ci_player_hi=ph, weeks_pos=f"{int((wk > 0).sum())}/{len(wk)}")
    for s in (2024, 2025, 2026):
        if s in by.index:
            r.update({f"n_{s}": by.n[s], f"hit_{s}": by.hit[s], f"roi_{s}": by.roi[s]})
    return r


def verdict(r: dict) -> str:
    if r.get("n", 0) == 0:
        return "FAIL (selects nothing)"
    both = r.get("roi_2024", -1) > 0 and r.get("roi_2025", -1) > 0
    wk_ok = int(r["weeks_pos"].split("/")[0]) >= int(r["weeks_pos"].split("/")[1]) / 2
    if both and r["ci_game_lo"] > 0 and r["ci_player_lo"] > 0 and wk_ok:
        return "PASS"
    return "TRACK" if both else "FAIL"


def board_2026(path_glob: list[str]) -> pd.DataFrame:
    m = pd.concat([pd.read_csv(p) for p in path_glob], ignore_index=True)
    m = m[m.played == 1]
    m["pick"] = np.where(m.model_proj > m.vegas_line, "OVER", np.where(m.model_proj < m.vegas_line, "UNDER", "TIE"))
    both = m.pivot_table(index=["season", "week", "book", "player_clean_key", "team", "market", "vegas_line", "model_proj", "actual"],
                         columns="side", values="vegas_odds", aggfunc="first").reset_index()
    both = both.rename(columns={"OVER": "over_odds", "UNDER": "under_odds", "vegas_line": "line", "model_proj": "proj"})
    return both


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/research/selective_pick")
    ap.add_argument("--board-2026", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    F = pd.read_csv(FULL, low_memory=False)
    F = F[F.market.isin(MARKETS)]
    F = prep(F, "proj")
    F["sd_rel"] = F.component_sd / np.maximum(F.line, 0.5)
    F, opp_ok = add_schedule(F)
    print(f"rows {len(F)}; rebuilt game key matches the file's opponent on {opp_ok:.1%}; gkey missing {F.gkey.isna().mean():.2%}")
    F = player_record(F)

    rows, picked = [], []
    for fit_s, sc_s in [(2024, 2025), (2025, 2024)]:
        fit, sc = F[F.season == fit_s], F[F.season == sc_s]
        for name, mask in select(fit, sc).items():
            picked.append(sc[mask.reindex(sc.index).fillna(False).astype(bool)].assign(selector=name.split(" ")[0], label=name, fit_season=fit_s))
    P = pd.concat(picked, ignore_index=True)
    for sid, g in P.groupby("selector"):
        r = summarize(g, "pnl", "win", rng)
        labs = " || ".join(sorted(g.label.unique()))
        rows.append(dict(selector=sid, label=labs, **r))
    # comparators on all rows
    rows.append(dict(selector="C0", label="every model pick", **summarize(F, "pnl", "win", rng)))
    rows.append(dict(selector="C1", label="blind UNDER, model ignored", **summarize(F, "u_pnl", "u_win", rng)))
    ov = F[F.pick == "OVER"]
    rows.append(dict(selector="C2", label="model OVER picks (contrast to S1)", **summarize(ov, "pnl", "win", rng)))
    # does the model add anything once the side is fixed? model-UNDER vs blind-UNDER on rows where model says OVER
    rows.append(dict(selector="C3", label="blind UNDER on rows where the model says OVER", **summarize(ov, "u_pnl", "u_win", rng)))
    T = pd.DataFrame(rows)
    T["verdict"] = [verdict(r) if str(r["selector"]).startswith("S") else "comparator" for r in T.to_dict("records")]

    # by market x side, out-of-sample is not needed (descriptive)
    M = F.groupby(["season", "market", "pick"]).agg(n=("pnl", "size"), hit=("win", "mean"), roi=("pnl", "mean")).reset_index()

    # secondary: pass_yards, base engine only
    B = pd.read_csv(BASE, low_memory=False)
    B = B[B.market == "pass_yards"]
    B = prep(B, "mc_proj")
    B, _ = add_schedule(B)
    B = player_record(B)
    sec = []
    for lab, d, pnl, win in [("pass_yards every model pick", B, "pnl", "win"),
                             ("pass_yards model UNDER picks", B[B.pick == "UNDER"], "pnl", "win"),
                             ("pass_yards model OVER picks", B[B.pick == "OVER"], "pnl", "win"),
                             ("pass_yards blind UNDER", B, "u_pnl", "u_win")]:
        sec.append(dict(label=lab, **summarize(d, pnl, win, rng)))
    SEC = pd.DataFrame(sec)

    # 2026 check (unfitted selectors only unless something PASSes)
    if a.board_2026:
        W = board_2026(a.board_2026)
        W["season"] = 2026
        W = prep(W, "proj")
        W, _ = add_schedule(W)
        W = player_record(W)
        w26 = []
        for mk in MARKETS + ["pass_yards"]:
            d = W[W.market == mk]
            for lab, dd, pnl, win in [("every model pick", d, "pnl", "win"), ("model UNDER picks", d[d.pick == "UNDER"], "pnl", "win"),
                                      ("model OVER picks", d[d.pick == "OVER"], "pnl", "win"), ("blind UNDER", d, "u_pnl", "u_win"),
                                      ("price better than -110", d[dec(d.price) > 100 / 110 + 1e-9], "pnl", "win")]:
                w26.append(dict(market=mk, label=lab, **summarize(dd, pnl, win, rng)))
        d = W[W.market.isin(MARKETS)]
        for lab, dd, pnl, win in [("ALL4 every model pick", d, "pnl", "win"), ("ALL4 model UNDER picks", d[d.pick == "UNDER"], "pnl", "win"),
                                  ("ALL4 blind UNDER", d, "u_pnl", "u_win")]:
            w26.append(dict(market="all4", label=lab, **summarize(dd, pnl, win, rng)))
        W26 = pd.DataFrame(w26)
        W26.to_csv(out / "selective_pick_2026_check_v1.csv", index=False)

    T.to_csv(out / "selective_pick_selectors_v1.csv", index=False)
    M.to_csv(out / "selective_pick_market_side_v1.csv", index=False)
    SEC.to_csv(out / "selective_pick_pass_yards_secondary_v1.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    cols = ["selector", "n", "hit", "roi", "ci_game_lo", "ci_game_hi", "ci_player_lo", "ci_player_hi", "weeks_pos",
            "hit_2024", "roi_2024", "hit_2025", "roi_2025", "verdict"]
    print(T[cols].round(3).to_string(index=False))
    print(); print(T[["selector", "label"]].to_string(index=False))
    print("\nmarket x side, both seasons:"); print(M.round(3).to_string(index=False))
    print("\npass_yards secondary:"); print(SEC[["label", "n", "hit", "roi", "ci_game_lo", "ci_player_lo", "hit_2024", "roi_2024", "hit_2025", "roi_2025"]].round(3).to_string(index=False))
    if a.board_2026:
        print("\n2026 check:"); print(W26[["market", "label", "n", "hit", "roi", "ci_game_lo", "ci_game_hi", "weeks_pos"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
