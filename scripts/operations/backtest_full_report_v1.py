#!/usr/bin/env python3
"""Complete graded backtest over the archived market track record.

Grades every archived board for the requested weeks and emits the whole
picture in one pass: the pure bet record, the projection-vs-actual error
decomposition, calibration against the model's own stated probability, and
whether the model's declared edge predicts anything.

Writes the full graded row set to disk so the analysis is auditable from the
rows rather than from this script's summary.

Read-only. Replays already-paid-for boards; fetches no sportsbook odds and
touches no pricing, projection or model-selection code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.operations import grade_market_track_record_v1 as G
from scripts.operations import grade_market_track_record_gsis_v1 as GG

POSITIONS = ["QB", "RB", "WR", "TE"]
MARKETS = ["pass_yards", "rush_yards", "rec_yards", "receptions", "rush_rec_yards"]


def build_graded(season: int, weeks: list[int]) -> pd.DataFrame:
    board = G.load_boards(season, weeks)
    if board.empty:
        raise SystemExit(f"no archived boards for {season} weeks {weeks}")
    board = G.apply_final_board_quarantine(board)
    board = G.apply_production_decision_gates(board)
    bets = G.select_model_bet(board)
    if bets.empty:
        return GG.empty_graded_frame(bets)
    bets["team"] = bets["team"].map(canon_team)

    actual = GG.load_actual_stats_unfiltered(season, weeks)
    roster = GG.load_roster_identity(season, weeks)
    snaps = GG.load_snap_participation(season, weeks)
    idx = GG.build_alias_index(actual, roster)
    confirmed = set(zip(roster.season, roster.week, roster.team, roster.gsis_id))
    roster_status = GG._status_map(roster)
    snap_exact, snap_base = GG._snap_participation_sets(snaps)

    resolved = [GG.resolve_gsis(r.player_clean_key, r.team, idx)
                for r in bets.itertuples(index=False)]
    b = bets.reset_index(drop=True)
    b["gsis_id"] = [g for g, _ in resolved]
    b["identity_status"] = [s for _, s in resolved]
    b["roster_confirmed_this_team_week"] = [
        (s, w, t, g) in confirmed
        for s, w, t, g in zip(b.season, b.week, b.team, b.gsis_id)
    ]
    b["roster_status"] = [
        roster_status.get((s, w, t, g), "")
        for s, w, t, g in zip(b.season, b.week, b.team, b.gsis_id)
    ]
    b["snap_participated"] = [
        (
            (int(s), int(w), t, k) in snap_exact
            or (int(s), int(w), t, GG._suffix_strip(k)) in snap_base
        )
        for s, w, t, k in zip(
            b.season, b.week, b.team, b.player_clean_key
        )
    ]

    parts = []
    for market, col in [("pass_yards", "pass_yards"), ("rush_yards", "rush_yards"),
                        ("rec_yards", "rec_yards"), ("receptions", "receptions")]:
        m = b.loc[b.market.eq(market)]
        if len(m):
            a = actual[["season", "week", "gsis_id", col]].rename(columns={col: "actual"})
            parts.append(m.merge(a, on=["season", "week", "gsis_id"], how="left"))
    rr = b.loc[b.market.eq("rush_rec_yards")]
    if len(rr):
        a = actual[["season", "week", "gsis_id", "rec_yards", "rush_yards"]].copy()
        a["actual"] = a.rec_yards + a.rush_yards
        parts.append(rr.merge(a[["season", "week", "gsis_id", "actual"]],
                              on=["season", "week", "gsis_id"], how="left"))
    d = pd.concat(parts, ignore_index=True, sort=False)

    d = GG.apply_postgame_settlement(d)

    g = d.loc[~d.settlement_status.eq("UNRESOLVED")].copy()
    g["vegas_line"] = G.num(g.vegas_line)
    g["model_proj"] = G.num(g.model_proj)
    g["model_error"] = g.model_proj - g.actual
    g["vegas_error"] = g.vegas_line - g.actual
    closer = g.model_error.abs() < g.vegas_error.abs()
    g["model_closer"] = closer.where(g["settlement_status"].eq("SETTLED"))
    g["model_closer_than_vegas"] = g["model_closer"]
    g["actual_side"] = [
        G.outcome_side(a, l) if pd.notna(a) else "VOID"
        for a, l in zip(g.actual, g.vegas_line)
    ]
    g["bet_result"] = np.select(
        [
            g.settlement_status.eq("VOID"),
            g.actual_side.eq("PUSH"),
            g.side.astype(str).str.upper().eq(g.actual_side),
        ],
        ["VOID", "PUSH", "WIN"],
        default="LOSS",
    )
    g["unit_result"] = np.where(
        g.bet_result.eq("WIN"), [G.american_profit(o) for o in g.vegas_odds],
        np.where(g.bet_result.eq("LOSS"), -1.0, 0.0))

    pos = {}
    for source in (roster, actual):
        for gid, p in zip(source.gsis_id, source.position):
            if str(p).strip():
                pos[gid] = str(p).strip().upper()
    g["position"] = g.gsis_id.map(pos).fillna("UNKNOWN")
    return g


def _cell(f: pd.DataFrame) -> dict:
    dec = f[f.bet_result.isin(["WIN", "LOSS"])]
    w, n = int(dec.bet_result.eq("WIN").sum()), len(dec)
    return {
        "W": w, "L": n - w, "bets": n,
        "win": w / n if n else np.nan,
        "units": float(dec.unit_result.sum()),
        "roi": float(dec.unit_result.mean()) if n else np.nan,
        "m_mae": float(dec.model_error.abs().mean()),
        "v_mae": float(dec.vegas_error.abs().mean()),
        "m_bias": float(dec.model_error.mean()),
        "v_bias": float(dec.vegas_error.mean()),
        "closer": float(dec.model_closer.mean()),
    }


def _print(rows: dict, title: str) -> None:
    print(f"\n{title}")
    hdr = (f"  {'':<16}{'W':>4}{'L':>4}{'bets':>6}{'win%':>8}{'units':>9}"
           f"{'mMAE':>8}{'vMAE':>8}{'mBias':>8}{'vBias':>8}{'closer':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, f in rows.items():
        if f.empty:
            continue
        c = _cell(f)
        if not c["bets"]:
            continue
        print(f"  {label:<16}{c['W']:>4}{c['L']:>4}{c['bets']:>6}{c['win']*100:>7.1f}%"
              f"{c['units']:>+9.2f}{c['m_mae']:>8.2f}{c['v_mae']:>8.2f}"
              f"{c['m_bias']:>+8.2f}{c['v_bias']:>+8.2f}{c['closer']:>8.3f}")


def report(g: pd.DataFrame, season: int, weeks: list[int]) -> None:
    bar = "=" * 104
    print(bar)
    print(f"FULL GRADED BACKTEST — {season}, weeks {weeks}")
    print("production Best Snapshot only: highest-EV real offer per player-market; nonpositive EV passes; DNP voids")
    print(bar)
    if g.empty:
        print("selected settlement rows: 0   decided: 0   voids: 0   pushes: 0   snap-confirmed zero outcomes: 0")
        print("\nNo production BET rows: every publishable player-market PASSed or failed closed.")
        return
    print(
        f"selected settlement rows: {len(g)}   decided: {int(g.bet_result.isin(['WIN','LOSS']).sum())}   "
        f"voids: {int(g.bet_result.eq('VOID').sum())}   pushes: {int(g.bet_result.eq('PUSH').sum())}   "
        f"snap-confirmed zero outcomes: {int(g.actual_source.eq('snap_confirmed_verified_zero').sum())}"
    )
    print("\nmBias / vBias are signed (projection - actual). Negative means the")
    print("number was too low. closer = share of bets where the model's absolute")
    print("error beat the line's.")

    _print({"ALL": g}, "OVERALL")
    _print({f"week {w}": g[g.week.eq(w)] for w in weeks}, "BY WEEK")
    _print({p: g[g.position.eq(p)] for p in POSITIONS}, "BY POSITION")
    _print({m: g[g.market.eq(m)] for m in MARKETS}, "BY MARKET")

    for p in POSITIONS:
        gp = g[g.position.eq(p)]
        if gp.empty:
            continue
        _print({m: gp[gp.market.eq(m)] for m in MARKETS}, f"{p} — BY MARKET")

    for w in weeks:
        gw = g[g.week.eq(w)]
        _print({p: gw[gw.position.eq(p)] for p in POSITIONS}, f"WEEK {w} — BY POSITION")

    _print({s: g[g.side.astype(str).str.upper().eq(s)] for s in ("OVER", "UNDER")}, "BY SIDE")

    # Does the model's own declared edge predict anything?
    if "edge_pct" in g.columns:
        e = g.copy()
        e["edge_abs_pct"] = pd.to_numeric(e.edge_pct, errors="coerce").abs()
        q = e.edge_abs_pct.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        e["edge_q"] = pd.cut(e.edge_abs_pct, [-np.inf] + q + [np.inf],
                             labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"])
        _print({str(k): v for k, v in e.groupby("edge_q", observed=True)},
               "BY THE MODEL'S OWN DECLARED EDGE (quintiles of |edge_pct|)")

    # Is the stated probability honest?
    if "fair_prob" in g.columns:
        c = g.copy()
        c["p"] = pd.to_numeric(c.fair_prob, errors="coerce")
        c = c[c.p.notna() & c.bet_result.isin(["WIN", "LOSS"])]
        if len(c):
            c["band"] = pd.cut(c.p, [0, .5, .55, .6, .65, .7, 1.0])
            print("\nCALIBRATION — model's stated win probability vs what happened")
            print(f"  {'band':<16}{'bets':>6}{'stated':>9}{'actual':>9}{'gap':>9}")
            print("  " + "-" * 47)
            for band, f in c.groupby("band", observed=True):
                hit = f.bet_result.eq("WIN").mean()
                print(f"  {str(band):<16}{len(f):>6}{f.p.mean()*100:>8.1f}%"
                      f"{hit*100:>8.1f}%{(hit - f.p.mean())*100:>+8.1f}pp")

    print("\n\nTE ROWS — the largest defect on the board")
    te = g[g.position.eq("TE")].sort_values(["week", "market", "player"])
    cols = [c for c in ["week", "player", "team", "opponent", "market", "side",
                        "vegas_line", "model_proj", "actual", "bet_result"] if c in te.columns]
    print(te[cols].to_string(index=False))

    print("\n\nQB PASS YARDS — every bet")
    q = g[g.market.eq("pass_yards")].sort_values(["week", "player"])
    print(q[cols].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", default="1,2")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the full graded row set here")
    a = ap.parse_args()
    weeks = [int(w) for w in a.weeks.split(",") if w.strip()]
    g = build_graded(a.season, weeks)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(a.out, index=False)
        print(f"full graded rows written: {len(g)} -> {a.out}\n")
    report(g, a.season, weeks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
