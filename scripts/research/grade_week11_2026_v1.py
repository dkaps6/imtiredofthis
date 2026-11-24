#!/usr/bin/env python3
"""Week 11 2026 complete backtest: every card, the whole board, alt rungs, screens, line movement.

Actuals: nflverse week-11 player stats; a leg is VOID if the player took no offensive snap.
Writes data/market_track_record/graded/w11_2026_*.csv and prints the summaries used in
docs/research/WK11_2026_BACKTEST_V1.md.
"""
from __future__ import annotations

import io
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

SEASON, WEEK = 2026, 11
OUT = Path("data/market_track_record/graded")
STAT = {"rec_yards": "receiving_yards", "receptions": "receptions", "rush_yards": "rushing_yards",
        "pass_yards": "passing_yards", "rush_rec_yards": "rush_rec_yards"}
ALT_MK = {"player_reception_yds_alternate": "rec_yards", "player_rush_yds_alternate": "rush_yards",
          "player_pass_yds_alternate": "pass_yards", "player_receptions_alternate": "receptions",
          "player_rush_reception_yds_alternate": "rush_rec_yards"}
SUF = {"jr", "sr", "ii", "iii", "iv", "v"}


def key(name) -> str:
    toks = [re.sub(r"[^a-z]", "", t) for t in str(name).lower().replace("-", " ").split()]
    return "".join(t for t in toks if t and t not in SUF)


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, a / 100, 100 / np.abs(a))


def actuals() -> tuple[dict, dict, dict]:
    st = nfl.load_player_stats(seasons=[SEASON]).to_pandas()
    st = st[(st.season_type == "REG") & (st.week == WEEK)].copy()
    for c in ["receiving_yards", "receptions", "rushing_yards", "passing_yards", "carries", "targets", "attempts"]:
        st[c] = st[c].fillna(0)
    st["rush_rec_yards"] = st.rushing_yards + st.receiving_yards
    st["k"] = st.player_display_name.map(key)
    st = st[st.position.isin(["QB", "RB", "FB", "WR", "TE"])].drop_duplicates("k")
    sn = nfl.load_snap_counts(seasons=[SEASON]).to_pandas()
    sn = sn[sn.week == WEEK].copy(); sn["k"] = sn.player.map(key)
    snaps = sn.groupby("k").offense_pct.max().to_dict()
    pos = nfl.load_rosters_weekly(seasons=[SEASON]).to_pandas().drop_duplicates("full_name")
    posd = {key(n): p for n, p in zip(pos.full_name, pos.position)}
    return st.set_index("k").to_dict("index"), snaps, posd


ACT, SNAP, POS = {}, {}, {}


def grade(player, market, side, thr):
    k = key(player); a = ACT.get(k); s = SNAP.get(k, 0.0)
    if a is None or not s > 0:
        return np.nan, "VOID", s
    v = float(a[STAT[market]])
    if v == thr:
        return v, "PUSH", s
    win = v > thr if str(side).upper() == "OVER" else v < thr
    return v, "WIN" if win else "LOSS", s


def grade_card(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in df.to_dict("records"):
        v, res, s = grade(r["player"], r["market"], r.get("side", "OVER"), float(r["threshold"]))
        rows.append({**r, "actual": v, "snap_pct": s, "result": res})
    legs = pd.DataFrame(rows).assign(card=name)
    t = []
    for tk, g in legs.groupby("ticket", sort=False):
        live = g[g.result != "VOID"]
        won = bool(len(live)) and (live.result == "WIN").all()
        x = float(np.prod(1 + dec(live.price_american))) if len(live) else 1.0
        t.append(dict(card=name, ticket=tk, legs=len(g), hit=int((g.result == "WIN").sum()), void=int((g.result == "VOID").sum()),
                      result="WON" if won else "LOST", naive_payout_x=round(x, 2),
                      missed="; ".join(f"{p} {m} {th}+ ({a:g})" if sd == "OVER" else f"{p} {m} U{th} ({a:g})"
                                       for p, m, th, a, sd in zip(g.player, g.market, g.threshold, g.actual, g.get("side", pd.Series(["OVER"] * len(g), index=g.index)))
                                       if g.loc[g.player == p].result.iloc[0] == "LOSS")))
    return legs, pd.DataFrame(t)


def board_rows() -> pd.DataFrame:
    b = pd.read_csv("data/market_track_record/boards/2026_wk11.csv", low_memory=False)
    b = b[b.market.isin(list(STAT))]
    piv = b.pivot_table(index=["source_run_id", "book", "player", "team", "market", "vegas_line", "model_proj"], columns="side",
                        values="vegas_odds", aggfunc="first").reset_index()
    piv = piv.rename(columns={"OVER": "over_odds", "UNDER": "under_odds", "vegas_line": "line"})
    g = piv.apply(lambda r: grade(r.player, r.market, "OVER", float(r.line)), axis=1, result_type="expand")
    piv[["actual", "over_result", "snap"]] = g
    piv = piv[piv.over_result.isin(["WIN", "LOSS"])].copy()
    piv["o_win"] = (piv.over_result == "WIN").astype(int); piv["u_win"] = 1 - piv.o_win
    piv["o_pnl"] = np.where(piv.o_win == 1, dec(piv.over_odds), -1.0); piv["u_pnl"] = np.where(piv.u_win == 1, dec(piv.under_odds), -1.0)
    piv["pick"] = np.where(piv.model_proj > piv.line, "OVER", "UNDER")
    piv["pick_win"] = np.where(piv.pick == "OVER", piv.o_win, piv.u_win); piv["pick_pnl"] = np.where(piv.pick == "OVER", piv.o_pnl, piv.u_pnl)
    piv["pos"] = piv.player.map(lambda n: POS.get(key(n), ""))
    return piv


def alt_rows(board: pd.DataFrame) -> pd.DataFrame:
    a = pd.concat([pd.read_csv("data/market_track_record/alt_props/2026_wk11_sun.csv"),
                   pd.read_csv("data/market_track_record/alt_props/2026_wk11_thu.csv")], ignore_index=True)
    a = a[(a.book == "fanduel") & (a.side.str.upper() == "OVER")].copy()
    a["market"] = a.market.map(ALT_MK); a = a[a.market.notna()]
    # latest capture per event (the Thursday capture priced the whole week)
    last = a.groupby("event_id").fetched_at.max()
    a = a[a.fetched_at == a.event_id.map(last)]
    a = a.sort_values("price_american", ascending=False).drop_duplicates(["event_id", "player", "market", "line"])
    proj = board[board.book == "fanduel"].drop_duplicates(["player", "market"]).set_index(["player", "market"]).model_proj
    a["model_proj"] = [proj.get((p, m), np.nan) for p, m in zip(a.player, a.market)]
    g = a.apply(lambda r: grade(r.player, r.market, "OVER", float(r.line)), axis=1, result_type="expand")
    a[["actual", "result", "snap"]] = g
    a = a[a.result.isin(["WIN", "LOSS"])].copy()
    a["hit"] = (a.result == "WIN").astype(int); a["pnl"] = np.where(a.hit == 1, dec(a.price_american), -1.0)
    a["be"] = np.where(a.price_american < 0, -a.price_american / (-a.price_american + 100), 100 / (a.price_american + 100))
    a["cushion"] = a.model_proj / a.line
    a["pos"] = a.player.map(lambda n: POS.get(key(n), ""))
    return a


def main() -> int:
    global ACT, SNAP, POS
    ACT, SNAP, POS = actuals()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)

    # ---- cards
    sun = pd.read_csv("data/research/cards/wk11_2026_sunday_card_v2.csv")
    removed = pd.DataFrame([dict(ticket="REMOVED_IN_FINAL_CHECK", tier="-", player="Rome Odunze", market="rec_yards", side="OVER", threshold=24.5, price_american=-360),
                            dict(ticket="REMOVED_IN_FINAL_CHECK", tier="-", player="Chuba Hubbard", market="rush_yards", side="UNDER", threshold=75.5, price_american=-113)])
    snf = pd.DataFrame([dict(ticket="SNF_SCRIPT", player="Christian McCaffrey", market="rush_yards", side="OVER", threshold=39.5, price_american=-245),
                        dict(ticket="SNF_SCRIPT", player="Kaelon Black", market="rush_yards", side="OVER", threshold=14.5, price_american=-440),
                        dict(ticket="SNF_SCRIPT", player="Brock Purdy", market="rush_yards", side="OVER", threshold=9.5, price_american=-300)])
    ch = pd.read_csv("data/research/cards/wk11_2026_mnf_challenge_entries.csv").rename(columns={"rung": "threshold", "price": "price_american", "entry": "ticket"})
    ch["side"] = "OVER"
    L, T = [], []
    for df, name in [(sun, "SUNDAY_CARD_V2"), (removed, "SUNDAY_REMOVED"), (snf, "SNF_SUGGESTED"), (ch, "MNF_CHALLENGE")]:
        l, t = grade_card(df, name); L.append(l); T.append(t)
    legs, tickets = pd.concat(L, ignore_index=True), pd.concat(T, ignore_index=True)
    legs.to_csv(OUT / "w11_2026_card_legs_graded_v1.csv", index=False); tickets.to_csv(OUT / "w11_2026_card_tickets_graded_v1.csv", index=False)
    print("== TICKETS"); print(tickets.to_string(index=False))
    u = legs[legs.card == "SUNDAY_CARD_V2"].drop_duplicates(["player", "market", "threshold"])
    print("\n== SUNDAY CARD, unique legs"); print(u[["player", "market", "side", "threshold", "price_american", "actual", "snap_pct", "result"]].to_string(index=False))
    print("\n== OTHER LEGS"); print(legs[legs.card != "SUNDAY_CARD_V2"][["card", "ticket", "player", "market", "side", "threshold", "actual", "result"]].to_string(index=False))

    # ---- whole main-line board
    b = board_rows(); b.to_csv(OUT / "w11_2026_mainline_board_graded_v1.csv", index=False)
    def s(d, w, p): return pd.Series(dict(n=len(d), hit=d[w].mean(), roi=d[p].mean()))
    print("\n== MAIN LINES (FanDuel + DraftKings), blind sides and model picks by market")
    rows = []
    for mk, d in b.groupby("market"):
        rows.append(dict(market=mk, n=len(d), over_hit=d.o_win.mean(), over_roi=d.o_pnl.mean(), under_hit=d.u_win.mean(), under_roi=d.u_pnl.mean(),
                         model_over_n=(d.pick == "OVER").sum(), model_over_roi=d[d.pick == "OVER"].pick_pnl.mean(),
                         model_under_n=(d.pick == "UNDER").sum(), model_under_roi=d[d.pick == "UNDER"].pick_pnl.mean()))
    d = b
    rows.append(dict(market="ALL", n=len(d), over_hit=d.o_win.mean(), over_roi=d.o_pnl.mean(), under_hit=d.u_win.mean(), under_roi=d.u_pnl.mean(),
                     model_over_n=(d.pick == "OVER").sum(), model_over_roi=d[d.pick == "OVER"].pick_pnl.mean(),
                     model_under_n=(d.pick == "UNDER").sum(), model_under_roi=d[d.pick == "UNDER"].pick_pnl.mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- screened main lines (Sunday FanDuel, frozen screens)
    sc = pd.read_csv("data/research/main_line_screens/week11_2026_sunday_fanduel_screened_v1.csv")
    g = sc.apply(lambda r: grade(r.player, r.market, r.side, float(r.line)), axis=1, result_type="expand")
    sc[["actual", "result", "snap_now"]] = g
    sc = sc[sc.result.isin(["WIN", "LOSS"])].copy(); sc["win"] = (sc.result == "WIN").astype(int); sc["pnl2"] = np.where(sc.win == 1, dec(sc.price), -1.0)
    sc.to_csv(OUT / "w11_2026_screened_mainlines_graded_v1.csv", index=False)
    print("\n== SIX-SCREEN MAIN LINES (Sunday FanDuel)")
    print(sc.groupby(["market", "side", "SCREENED"]).agg(n=("win", "size"), hit=("win", "mean"), roi=("pnl2", "mean")).round(3).to_string())
    r = sc[(sc.market == "rush_yards") & (sc.side == "UNDER") & sc.SCREENED]
    print(r[["player", "team", "line", "proj", "actual", "result"]].to_string(index=False))
    for p in ["Derrick Henry", "Saquon Barkley", "Jordan Mason", "Cam Skattebo", "Omarion Hampton"]:
        q = sc[(sc.player == p) & (sc.market == "rush_yards")]
        if len(q):
            print(f"   screen-rejected {p}: U{q.line.iloc[0]} actual {q.actual.iloc[0]} -> {q.result.iloc[0]}")

    # ---- alt rungs
    a = alt_rows(b); a.to_csv(OUT / "w11_2026_alt_board_graded_v1.csv", index=False)
    a["cb"] = pd.cut(a.cushion, [0, 1.0, 1.25, 1.5, 2.0, 3.0, 99], labels=["<1.0", "1.0-1.25", "1.25-1.5", "1.5-2.0", "2.0-3.0", ">3.0"])
    print("\n== ALT RUNGS (FanDuel overs) by cushion"); print(a.groupby("cb", observed=True).agg(n=("hit", "size"), hit=("hit", "mean"), be=("be", "mean"), roi=("pnl", "mean")).round(3).to_string())
    band = a[a.cushion.between(1.5, 3.0) & a.price_american.between(-400, -150)]
    print("\n== SHORTLIST BAND (1.5-3.0x, -150..-400) by leg type"); print(band.groupby(["market", "pos"]).agg(n=("hit", "size"), hit=("hit", "mean"), be=("be", "mean"), roi=("pnl", "mean")).round(3).to_string())

    # ---- line movement Thursday -> Sunday
    thu = pd.read_csv(io.StringIO(subprocess.check_output(["git", "show", "86a3e36:data/market_track_record/boards/2026_wk11.csv"]).decode()), low_memory=False)
    thu = thu[thu.market.isin(list(STAT))]
    tp = thu.pivot_table(index=["book", "player", "team", "market", "vegas_line", "model_proj"], columns="side", values="vegas_odds", aggfunc="first").reset_index()
    tp = tp.rename(columns={"vegas_line": "line_thu", "model_proj": "proj_thu", "OVER": "over_thu", "UNDER": "under_thu"})
    sb = b[b.source_run_id == 45024058278][["book", "player", "team", "market", "line", "actual", "o_win"]]
    m = tp.merge(sb, on=["book", "player", "team", "market"])
    m["pick_thu"] = np.where(m.proj_thu > m.line_thu, "OVER", "UNDER")
    m["move"] = m.line - m.line_thu
    m["toward"] = np.sign(m.move) == np.where(m.pick_thu == "OVER", 1, -1)
    m["thu_win"] = np.where(m.pick_thu == "OVER", m.actual > m.line_thu, m.actual < m.line_thu).astype(int)
    m = m[m.actual != m.line_thu]
    m["thu_pnl"] = np.where(m.thu_win == 1, dec(np.where(m.pick_thu == "OVER", m.over_thu, m.under_thu)), -1.0)
    m.to_csv(OUT / "w11_2026_line_movement_graded_v1.csv", index=False)
    print("\n== LINE MOVEMENT: the model's Thursday pick, bet at Thursday's line and price")
    for lab, d in [("moved toward the model", m[(m.move != 0) & m.toward]), ("moved away", m[(m.move != 0) & ~m.toward]),
                   ("did not move", m[m.move == 0]), ("all Thursday picks", m)]:
        print(f"   {lab:24s} n={len(d):4d} hit={d.thu_win.mean():.3f} roi={d.thu_pnl.mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
