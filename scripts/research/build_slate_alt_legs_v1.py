#!/usr/bin/env python3
"""Turn a priced board plus its alternate ladders into gate-ready candidate legs.

Input for scripts/research/build_alt_cards_v1.py (the nine-gate builder). One row
per (player, market, rung, book) with every column the gates read:
offered_american, alt_threshold, model_proj, main_line, frac_of_line, snap_pct,
own_games, own_hits, def_allowed, injury_status, quarantined.

Also attaches model_prob: P(stat > rung) from a gamma matched to the board's own
(model_proj, model_sd). On the week-10 board that family reproduced the model's
fair_prob at the main line to a mean gap of 0.027, so it is the model's opinion
at the rung, not a new estimator. It is reported for the card validator's edge
check; it is NOT a calibrated probability (V27 records ~12 points of
overstatement on the picked side).

Strictly prior evidence only: own history and matchup use weeks < target week.
No sportsbook variable reaches model_proj (AGENTS.md section 3); price is used
for breakeven and the market's main line for the real-alternate gate.

Team-change check (added after Keenan Allen, wk10 2026: traded IND -> SF, all 8
prior games with IND, first game with SF on the day of the card). A player whose
prior games were for another team and who has fewer than NEW_TEAM_MIN_GAMES games
with today's team is labeled UNRESOLVED_CONTEXT: his own history describes a role
in a different offense, so it cannot be affirmative evidence (AGENTS.md anomaly
gate). The card validator refuses such legs.
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
import nflreadpy as nfl

MK = {"player_reception_yds_alternate": "rec_yards", "player_rush_yds_alternate": "rush_yards",
      "player_pass_yds_alternate": "pass_yards", "player_receptions_alternate": "receptions",
      "player_rush_reception_yds_alternate": "rush_rec_yards"}
STAT = {"rec_yards": "receiving_yards", "rush_yards": "rushing_yards", "pass_yards": "passing_yards",
        "receptions": "receptions", "rush_rec_yards": "rush_rec_yards"}
FAM_COL = {"rec_yards": (("WR", "TE"), "receiving_yards"), "receptions": (("WR", "TE"), "receiving_yards"),
           "rush_yards": (("RB",), "rushing_yards"), "rush_rec_yards": (("RB",), "rushing_yards"),
           "pass_yards": (("QB",), "passing_yards")}
NFL_TEAM = {"LAR": "LA"}
NEW_TEAM_MIN_GAMES = 3
nm = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)


def ip(x):
    x = float(x); return (-x) / ((-x) + 100) if x < 0 else 100 / (x + 100)


def team_change_state(prior_teams, team_today) -> tuple[int, str, str]:
    """(games with today's team, state_label, history) from the team of each prior game."""
    today = NFL_TEAM.get(str(team_today), str(team_today))
    teams = [NFL_TEAM.get(str(t), str(t)) for t in prior_teams]
    with_today = sum(t == today for t in teams)
    hist = ",".join(f"{t}:{teams.count(t)}" for t in dict.fromkeys(teams))
    if len(teams) > with_today and with_today < NEW_TEAM_MIN_GAMES:
        return with_today, "UNRESOLVED_CONTEXT", hist
    return with_today, "HEALTHY_FULL", hist


def main_line(g):
    g = g.sort_values("line"); p = g.price_american.map(ip).to_numpy(); L = g.line.to_numpy(); t = ip(-110)
    for i in range(len(L) - 1):
        if (p[i] - t) * (p[i + 1] - t) <= 0 and p[i] != p[i + 1]:
            w = (p[i] - t) / (p[i] - p[i + 1]); return float(L[i] + w * (L[i + 1] - L[i]))
    return np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True); ap.add_argument("--alt", required=True)
    ap.add_argument("--season", type=int, required=True); ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261115)

    board = pd.read_csv(a.board); board["k"] = nm(board.player)
    alt = pd.read_csv(a.alt); alt = alt[alt.side.astype(str).str.upper() == "OVER"].copy()
    alt["market"] = alt.market.map(MK); alt = alt[alt.market.notna()]; alt["k"] = nm(alt.player)
    alt = alt[alt.event_id.isin(set(board.event_id))]

    st = nfl.load_player_stats(seasons=[a.season]).to_pandas()
    st = st[(st.season_type == "REG") & (st.week < a.week)].copy()
    st["rush_rec_yards"] = st.rushing_yards.fillna(0) + st.receiving_yards.fillna(0)
    st["k"] = nm(st.player_display_name)
    sn = nfl.load_snap_counts(seasons=[a.season]).to_pandas(); sn = sn[sn.week < a.week].copy(); sn["k"] = nm(sn.player)
    snap = sn.groupby("k").offense_pct.mean()
    played = sn[sn.offense_snaps.fillna(0) > 0].groupby("k").week.apply(set)
    pos = st.groupby("k").position.first(); st["fam"] = st.k.map(pos)

    allowed = {}
    for mk, (fams, col) in FAM_COL.items():
        sub = st[st.fam.isin(fams)]
        allowed[mk] = sub.groupby(["opponent_team", "week"])[col].sum().groupby("opponent_team").mean()

    first = board.drop_duplicates(["k", "market"]).set_index(["k", "market"])
    rows = []
    for (k, mk), g in alt.groupby(["k", "market"]):
        if (k, mk) not in first.index:
            continue
        b = first.loc[(k, mk)]
        proj, sd = float(b.model_proj), float(b.model_sd) if pd.notna(b.model_sd) else np.nan
        if not (proj > 0 and sd > 0):
            continue
        if mk == "receptions":
            var = max(sd * sd, proj + 1e-6); r = proj * proj / (var - proj) if var > proj * 1.02 else None
            draws = rng.negative_binomial(r, r / (r + proj), 100_000) if r else rng.poisson(proj, 100_000)
        else:
            draws = rng.gamma((proj / sd) ** 2, sd * sd / proj, 100_000)
        s = st[st.k == k]
        wk = played.get(k, set())
        s = s[s.week.isin(wk)] if wk else s
        vals = s[STAT[mk]].fillna(0).to_numpy(float)
        gwt, state, hist = team_change_state(s.sort_values("week").team.tolist(), b.team)
        opp = NFL_TEAM.get(str(b.opponent), str(b.opponent))
        da = float(allowed[mk].get(opp, np.nan))
        ml = main_line(g)
        best = g.loc[g.groupby(["line", "book"]).price_american.idxmax()]
        for r in best.itertuples(index=False):
            thr = float(r.line)
            rows.append(dict(event_id=r.event_id, player=r.player, team=b.team, opponent=b.opponent, market=mk,
                book=r.book, offered_american=float(r.price_american), alt_threshold=thr, model_proj=proj, model_sd=sd,
                main_line=ml, frac_of_line=(thr / ml if ml == ml and ml > 0 else np.nan),
                snap_pct=float(snap.get(k, np.nan)), own_games=int(len(vals)), own_hits=int((vals > thr).sum()),
                own_floor=float(vals.min()) if len(vals) else np.nan, own_median=float(np.median(vals)) if len(vals) else np.nan,
                def_allowed=da, injury_status="", quarantined=False,
                games_with_team=gwt, team_history=hist, state_label=state,
                model_prob=float((draws > thr).mean()), breakeven=ip(r.price_american)))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"candidate rungs {len(out)} | players {out.player.nunique()} | events {out.event_id.nunique()} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
