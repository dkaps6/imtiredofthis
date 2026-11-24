#!/usr/bin/env python3
"""Audit every leg on a ticket card against ONE fixed checklist.

Why this exists (2026-11-15, Sunday wk10): legs were reaching the card through
different paths -- SAFE legs through the nine-gate builder, MEDIUM/HIGH/LOTTERY
through a relaxed subset, and same-day replacements through an ad hoc filter
(model edge + hit rate + snaps + manual news checks). The operator asked, fairly,
whether tickets were being built from hit rates alone and with different inputs
every time. This script answers that per leg with the same columns every time.

For every leg it reports, never short-circuiting, so a leg shows ALL its fails:

  MODEL        model_proj, model_prob at the rung, edge vs the price's breakeven
  VEGAS        price, breakeven, the market's own main line, model/main ratio
  ROLE         snap share (G1), main-line role floor (G2)
  CUSHION      model_proj / rung (G3; required >= 1.50 on SAFE only)
  ALTERNATE    rung / main line inside the real-alternate band (G4)
  HISTORY      own hit rate at the rung over games played (G6)
  MATCHUP      opponent's prior yards allowed to the position group, percentile
               WITHIN THE SAME MARKET (G7). build_alt_cards_v1.apply_gates
               computed one quantile over all markets mixed, which rejected
               every rush_yards rung and could never reject a pass_yards rung.
  ENVIRONMENT  game total
  AVAILABILITY week-10 club designation (G8)
  PRICE        inside the builder's price window (G9; reported, tier-dependent)
  CONTEXT      team change (player traded) and TEAMMATE context: key teammates
               at the same position group whose presence today differs from the
               player's history (returning from injury, traded away, newly out).
               The hit rate is recomputed on games that match today's teammate
               availability; that is the history that describes today's role.

Pure research tooling; reads strictly-prior stats (weeks < target week).
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
import nflreadpy as nfl

from scripts.research.build_alt_cards_v1 import (SNAP_MIN, CUSHION_MIN, FRAC_MAX, NEAR_MAIN_LO,
    NEAR_MAIN_HI, OWN_GAMES_MIN, OWN_RATE_MIN, DEF_EXCLUDE_Q, PRICE_MIN, PRICE_MAX, MAIN_LINE_FLOOR)
from scripts.research.build_slate_alt_legs_v1 import team_change_state, NFL_TEAM

STAT = {"rec_yards": "receiving_yards", "rush_yards": "rushing_yards", "pass_yards": "passing_yards",
        "receptions": "receptions"}
# Position group whose availability shapes the leg, and the usage that defines a key teammate.
GROUP = {"rush_yards": (("RB",), "carries", 0.25), "rec_yards": (("WR", "TE"), "targets", 0.15),
         "receptions": (("WR", "TE"), "targets", 0.15), "pass_yards": (("WR", "TE"), "targets", 0.15)}
OUT_STATES = {"out", "doubtful"}
KEY_MIN_VOLUME = {"carries": 10.0, "targets": 5.0}
QB_MISSING_SHARE = 0.30


def matchup_pct(legs: pd.DataFrame) -> pd.Series:
    """Percentile of def_allowed within the leg's own market AND defense group
    (QB / RB / WRTE; 0 = toughest defense). Keyed (market, def_group, opponent)."""
    legs = legs.copy()
    if "def_group" not in legs:
        legs["def_group"] = ""
    legs["def_group"] = legs["def_group"].fillna("")
    one = legs.drop_duplicates(["market", "def_group", "opponent"])[["market", "def_group", "opponent", "def_allowed"]]
    one["pct"] = one.groupby(["market", "def_group"]).def_allowed.rank(pct=True)
    return one.set_index(["market", "def_group", "opponent"]).pct


def teammate_context(st, played, team_now, inj_out, player, team, market, thr):
    """(ctx_games, ctx_hits, notes) for history matching today's teammate availability."""
    vals, notes = teammate_context_values(st, played, team_now, inj_out, player, team, market)
    return len(vals), int((vals > thr).sum()), notes


def teammate_context_values(st, played, team_now, inj_out, player, team, market):
    """(stat values from games that do NOT flatter today's role, notes).

    Direction matters (corrected 2026-11-15 after the first version required an
    exact match on every key teammate and cut 32 of 72 players):
      * receiving / rushing legs: same-group teammates are COMPETITORS. A game is
        dropped when a competitor who plays today was missing from it (Braelon
        Allen's Breece-Hall-out games). A competitor out today leaves MORE work,
        so games with him present are conservative and are kept (Ayomanor with
        Carnell Tate out).
      * passing legs: receivers are COMPLEMENTS. A game is dropped when a
        receiver who is out today (IR, inactive, traded) played in it (Stafford
        without Nacua and Adams, Jones without Downs and Keenan Allen). A
        receiver back today only helps, so those games are kept.
    Presence is snaps > 0 (a stat line can be missing for a player who played).
    A competitor newly arrived by trade is reported as "new arrival".
    """
    fams, use, share_min = GROUP[market]
    complement = market == "pass_yards"
    T = NFL_TEAM.get(team, team)
    tg = st[(st.team == T) & (st.position.isin(fams))]
    tot = tg.groupby("week")[use].sum()
    notes, key = [], {}
    for q, g in tg.groupby("player_display_name"):
        if q == player:
            continue
        share = float((g.set_index("week")[use] / tot.reindex(g.week).values).mean())
        # Key = real share AND real volume in the games he played. Share alone
        # made a one-game backup (DJ Giddens, Will Mallory) look like a starter.
        if share >= share_min and g[use].mean() >= KEY_MIN_VOLUME[use]:
            active = team_now.get(q) == T and q not in inj_out
            on_team = set(g.week)
            weeks = (played.get(q, set()) & set(st[st.team == T].week)) | on_team
            key[q] = (weeks, active)
    if not complement:
        for q, t in team_now.items():
            if t == T and q != player and q not in set(tg.player_display_name):
                prior = st[(st.player_display_name == q) & st.position.isin(fams)]
                if len(prior) >= 2 and prior[use].mean() >= (8 if use == "carries" else 4) and q not in inj_out:
                    notes.append(f"{q} new arrival")
    me = st[(st.player_display_name == player) & (st.team == T)]
    wk = played.get(player, set())
    me = me[me.week.isin(wk)] if wk else me
    # For a QB, a missing receiver matters in proportion to his target share:
    # Stafford (Nacua + Adams) and Jones (Downs + Keenan Allen) lost ~40-50% of
    # their targets; one WR3 on IR does not erase a season. Games are dropped
    # only when the receivers out today carried >= QB_MISSING_SHARE together.
    missing_share = 0.0
    if complement:
        for q, (weeks, active) in key.items():
            if not active:
                g = tg[tg.player_display_name == q]
                missing_share += float((g.set_index("week")[use] / tot.reindex(g.week).values).mean())
        if missing_share:
            notes.append(f"receivers out today held {missing_share:.0%} of targets")
    keep = []
    for w in me.week:
        if complement:   # drop games a now-missing receiver group played in
            ok = missing_share < QB_MISSING_SHARE or not any(
                w in weeks for weeks, active in key.values() if not active)
        else:            # drop games a now-playing competitor missed
            ok = all(w in weeks for weeks, active in key.values() if active)
        keep.append(ok)
    for q, (weeks, active) in key.items():
        absent = [w for w in me.week if w not in weeks]
        if active and absent and not complement:
            notes.append(f"{q} back (absent {len(absent)}/{len(me)})")
        if not active:
            why = "traded" if team_now.get(q) not in (None, T) else ("out today" if q in inj_out else "inactive")
            notes.append(f"{q} {why}")
    m = me[np.array(keep, bool)] if len(me) else me
    return m[STAT[market]].fillna(0).to_numpy(float), "; ".join(notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True); ap.add_argument("--legs", required=True)
    ap.add_argument("--season", type=int, required=True); ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--game-odds", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    card = pd.read_csv(a.card)
    legs = pd.read_csv(a.legs); legs = legs[legs.book == "fanduel"]
    mpct = matchup_pct(legs)
    st = nfl.load_player_stats(seasons=[a.season]).to_pandas()
    st = st[(st.season_type == "REG") & (st.week < a.week)]
    sn = nfl.load_snap_counts(seasons=[a.season]).to_pandas(); sn = sn[sn.week < a.week]
    played = sn[sn.offense_snaps.fillna(0) > 0].groupby("player").week.apply(set).to_dict()
    ro = nfl.load_rosters_weekly(seasons=[a.season]).to_pandas(); ro = ro[ro.week == ro.week.max()]
    team_now = ro[ro.status == "ACT"].drop_duplicates("full_name").set_index("full_name").team.to_dict()
    ij = nfl.load_injuries(seasons=[a.season]).to_pandas(); ij = ij[ij.week == a.week]
    status = ij.set_index("full_name").report_status.astype(str).str.lower().to_dict()
    inj_out = {p for p, s in status.items() if s in OUT_STATES}
    env = {}
    if a.game_odds:
        g = pd.read_csv(a.game_odds)
        tot = g[g.market == "totals"].groupby("event_id").point.median()
        env = dict(tot=tot)
    rows = []
    for c in card.itertuples():
        r = legs[(legs.player == c.player) & (legs.market == c.market) & (legs.alt_threshold == c.threshold)]
        r = r.sort_values("offered_american", ascending=False).iloc[0]
        proj, thr, ml = float(r.model_proj), float(c.threshold), float(r.main_line)
        cush, frac = proj / thr, thr / ml if ml > 0 else np.nan
        own = r.own_hits / r.own_games if r.own_games else np.nan
        grp = r.def_group if "def_group" in r.index and isinstance(r.def_group, str) else ""
        pct = float(mpct.get((c.market, grp, r.opponent), np.nan))
        gwt, tstate, hist = team_change_state(
            st[st.player_display_name == c.player].sort_values("week").team.tolist(), c.team)
        n, h, notes = teammate_context(st, played, team_now, inj_out, c.player, c.team, c.market, thr)
        be = float(r.breakeven)
        fails = []
        if not r.snap_pct >= SNAP_MIN: fails.append("G1 snaps")
        if not ml >= MAIN_LINE_FLOOR.get(c.market, 0): fails.append("G2 role")
        if c.tier == "SAFE" and cush < CUSHION_MIN: fails.append("G3 cushion")
        if not (frac <= FRAC_MAX and not (NEAR_MAIN_LO <= frac <= NEAR_MAIN_HI)): fails.append("G4 alt")
        if not (r.own_games >= OWN_GAMES_MIN and own >= OWN_RATE_MIN): fails.append("G6 own")
        if pct < DEF_EXCLUDE_Q: fails.append("G7 matchup")
        if status.get(c.player, "nan") in {"out", "doubtful", "questionable"}: fails.append("G8 injury")
        if not (PRICE_MIN <= c.price_american <= PRICE_MAX): fails.append("G9 price")
        if not r.model_prob > be: fails.append("no model edge")
        if tstate != "HEALTHY_FULL": fails.append("team change")
        if n < OWN_GAMES_MIN or (n and h / n <= be): fails.append("context history")
        rows.append(dict(ticket=c.ticket, player=c.player, team=c.team, opp=r.opponent, market=c.market, rung=thr,
            price=int(c.price_american), breakeven=round(be, 3), model_proj=round(proj, 1),
            model_prob=round(float(r.model_prob), 3), main_line=round(ml, 1), model_vs_main=round(proj / ml, 2),
            cushion=round(cush, 2), snap=round(float(r.snap_pct), 2), own=f"{int(r.own_hits)}/{int(r.own_games)}",
            ctx_own=f"{h}/{n}", matchup_pct=round(pct, 2), total=float(env["tot"].get(c.event_id, np.nan)) if env else np.nan,
            injury=status.get(c.player, ""), context=notes, fails=", ".join(fails) or "PASS"))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 260, "display.max_colwidth", 60):
        print(out.drop(columns=["total"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
