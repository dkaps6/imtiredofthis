#!/usr/bin/env python3
"""THE weekly alternate-line ticket builder. One procedure, every week, in code.

Operator, 2026-11-15: "You have to stop doing this. I shouldn't have to remind
you how to do this every goddamn week." Until this script, tickets were
assembled by hand from pools built different ways (nine gates for SAFE, a
relaxed subset for the rest, an ad hoc filter for same-day replacements). That
produced Keenan Allen in his first SF game, Braelon Allen with Breece Hall back,
Daniel Jones with his receivers traded, and Stafford with Nacua and Adams on IR.
Nothing here is chosen by hand: a leg is on a ticket because it passed every
check for its tier, and tickets are assembled by rule.

INPUT: candidate rungs from build_slate_alt_legs_v1 (board + alt ladders).

CHECKS (every tier; see TIER RULES for the extras):
  G1 snaps >= SNAP_MIN; G2 main-line role floor; G5 market allowed;
  G7 matchup, toughest quartile of all 32 defenses WITHIN THE MARKET
    (per-game yards allowed to the position group; fixed 2026-11-15);
  G8 availability: no Out/Doubtful/Questionable designation, not quarantined;
  QB: team not in --exclude-teams (QB change / first start; fail closed);
  TEAM CHANGE: player has >= 3 games with today's team if he played elsewhere;
  TEAMMATES: own history recomputed on the games that do not flatter today's
    role (audit_card_legs_v1.teammate_context_values: drop games a competitor
    who plays today missed; for QBs, drop games a receiver who is out today
    played): >= CTX_MIN_GAMES such games, and the tier's history rule holds on
    them; a key competitor newly arrived by trade fails closed;
  EDGE: model probability at the rung above the price's breakeven.

TIER RULES (as documented on the 2026-11-15 card, now enforced):
  SAFE              all nine gates: + G3 cushion >= 1.50, G4 real alternate,
                    G6 own history (>= 5 games, >= 60%), G9 price window.
  MODERATE, MEDIUM  + G4 real alternate (not the main line in disguise),
                    G6 own history (>= 5 games, >= 60%).
  HIGH, LOTTERY     + G4 real alternate (ceiling rungs up to 1.5x allowed),
                    own history >= 5 games and above the price's breakeven.

TICKETS: one leg per game per ticket (a standard cross-game parlay at the
product of the leg prices); one book; no player on more than two tickets.
For each tier every one-leg-per-game combination of eligible rungs that reaches
the tier's minimum odds is enumerated (stopping once the price is reached), and
the tickets are chosen jointly: fill as many tiers as possible, then maximise
the summed log hit estimate (p_est = half model, half own rate with today's
teammates), never exceeding two tickets per player. (Superseded greedy fill:
filled LOTTERY (only if 250x is reachable) -> HIGH -> MEDIUM -> MODERATE -> SAFE,
since the upper tiers need the scarce plus-money rungs, each until its minimum
odds are reached:
  SAFE      legs ordered by risk per unit of payout (-ln p / ln decimal, with
            p = half model, half own rate with today's teammates): the most
            likely ticket that still clears +150;
  MOD/MED   best leg_value first (own rate minus implied; the validated ranking);
  HIGH/LOT  each player's highest-priced rung that passes, best leg_value first.
  Greedy filling left SAFE or MEDIUM short on 2026-11-15 when a valid
  four-ticket card existed, so it was replaced by the joint search.) SAFE must be over +150; LOTTERY must reach
250x or it is labeled UNSOLVED. validate_ticket_card_v1 then runs on the card.
"""
from __future__ import annotations
import argparse, re
import numpy as np, pandas as pd
import nflreadpy as nfl

from scripts.research.build_alt_cards_v1 import (SNAP_MIN, CUSHION_MIN, FRAC_MAX, NEAR_MAIN_LO,
    NEAR_MAIN_HI, OWN_GAMES_MIN, OWN_RATE_MIN, DEF_EXCLUDE_Q, PRICE_MIN, PRICE_MAX, MAIN_LINE_FLOOR,
    EXCLUDED_MARKETS, leg_value)
from scripts.research.build_slate_alt_legs_v1 import team_change_state
from scripts.research.audit_card_legs_v1 import teammate_context_values, OUT_STATES
from scripts.research.validate_ticket_card_v1 import validate, dec, amer

TIERS = ["SAFE", "MODERATE", "MEDIUM", "HIGH", "LOTTERY"]
# Minimum combined odds per ticket. SAFE and LOTTERY are operator rules; the
# middle three are the tier spacing of the 2026-11-15 card.
MIN_AMERICAN = {"SAFE": 151, "MODERATE": 500, "MEDIUM": 1200, "HIGH": 3000, "LOTTERY": 24900}
CTX_MIN_GAMES = 3
TOP_K = 60  # best candidate tickets kept per tier for the joint search
MAX_TICKETS_PER_PLAYER = 2
nm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())


def check(r, tier: str) -> list[str]:
    """Every failed check for this rung at this tier (empty list = eligible)."""
    f = list(r.common_fails)
    frac = r.frac_of_line
    real_alt = np.isfinite(frac) and frac <= FRAC_MAX and not (NEAR_MAIN_LO <= frac <= NEAR_MAIN_HI)
    if not real_alt:
        f.append("G4 alt")
    own_rate = r.own_hits / r.own_games if r.own_games else 0.0
    ctx_rate = r.ctx_hits / r.ctx_games if r.ctx_games else 0.0
    if r.own_games < OWN_GAMES_MIN:
        f.append("G6 games")
    # The tier's history rule applies to the whole season AND to the games that
    # reflect today's teammates (same threshold; the context sample is not held
    # to a stricter bar than the tier's own rule).
    if tier in ("SAFE", "MODERATE", "MEDIUM"):
        if own_rate < OWN_RATE_MIN:
            f.append("G6 own")
        if r.ctx_games and ctx_rate < OWN_RATE_MIN:
            f.append("context own")
    else:
        if own_rate <= r.breakeven:
            f.append("own below breakeven")
        if r.ctx_games and ctx_rate <= r.breakeven:
            f.append("context below breakeven")
    if tier == "SAFE":
        if r.model_proj / r.alt_threshold < CUSHION_MIN:
            f.append("G3 cushion")
        if not (PRICE_MIN <= r.offered_american <= PRICE_MAX):
            f.append("G9 price")
        if frac > NEAR_MAIN_HI:
            f.append("SAFE above main line")
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", required=True); ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True); ap.add_argument("--book", default="fanduel")
    ap.add_argument("--exclude-teams", default="", help="comma list: QB change / first start teams")
    ap.add_argument("--exclude-events", default="", help="comma list of event_ids already started")
    ap.add_argument("--quarantine", default="", help="comma list of players ruled out by hand, with reason in the doc")
    ap.add_argument("--out-card", required=True); ap.add_argument("--out-checks", required=True)
    a = ap.parse_args()

    L = pd.read_csv(a.legs)
    L = L[(L.book == a.book) & (L.side.astype(str).str.upper().eq("OVER") if "side" in L else True)].copy()
    L = L[~L.event_id.isin([e for e in a.exclude_events.split(",") if e])]
    excl_teams = {t for t in a.exclude_teams.split(",") if t}
    quarantine = {nm(p) for p in a.quarantine.split(",") if p}

    st = nfl.load_player_stats(seasons=[a.season]).to_pandas()
    st = st[(st.season_type == "REG") & (st.week < a.week)]
    sn = nfl.load_snap_counts(seasons=[a.season]).to_pandas(); sn = sn[sn.week < a.week]
    played = sn[sn.offense_snaps.fillna(0) > 0].groupby("player").week.apply(set).to_dict()
    ro = nfl.load_rosters_weekly(seasons=[a.season]).to_pandas(); ro = ro[ro.week == ro.week.max()]
    team_now = ro[ro.status == "ACT"].drop_duplicates("full_name").set_index("full_name").team.to_dict()
    ij = nfl.load_injuries(seasons=[a.season]).to_pandas(); ij = ij[ij.week == a.week]
    desig = {nm(p): str(s).lower() for p, s in zip(ij.full_name, ij.report_status)}
    inj_out = {p for p, s in zip(ij.full_name, ij.report_status) if str(s).lower() in OUT_STATES}

    # Matchup percentile within each market AMONG ALL 32 DEFENSES (0 = toughest).
    # Ranking only today's opponents cut the toughest quarter of whoever is on
    # the slate (on a two-team slate, one defense always), which is not the
    # gate's meaning: "toughest quartile of prior yards allowed to the position
    # group". Per-game yards allowed, same family/column as the leg builder.
    from scripts.research.build_slate_alt_legs_v1 import FAM_COL
    pos = st.groupby("player_display_name").position.first()
    stf = st.assign(fam=st.player_display_name.map(pos))
    league = {}
    for mk, (fams, col) in FAM_COL.items():
        pg = stf[stf.fam.isin(fams)].groupby(["opponent_team", "week"])[col].sum().groupby("opponent_team").mean()
        league[mk] = pg.rank(pct=True)
    from scripts.research.build_slate_alt_legs_v1 import NFL_TEAM as _NT
    mpct = {(mk, opp): float(league.get(mk, pd.Series(dtype=float)).get(_NT.get(opp, opp), np.nan))
            for mk, opp in L[["market", "opponent"]].drop_duplicates().itertuples(index=False)}

    # Per (player, market): checks that do not depend on the rung.
    ctx, common = {}, {}
    for (p, team, mk), g in L.groupby(["player", "team", "market"]):
        b = g.iloc[0]
        f = []
        if mk in EXCLUDED_MARKETS: f.append("G5 market")
        if not b.snap_pct >= SNAP_MIN: f.append("G1 snaps")
        if not b.main_line >= MAIN_LINE_FLOOR.get(mk, np.inf): f.append("G2 role")
        if mpct.get((mk, b.opponent), 1.0) <= DEF_EXCLUDE_Q: f.append("G7 matchup")
        s = desig.get(nm(p), "nan")
        if s in {"out", "doubtful", "questionable"}: f.append(f"G8 {s}")
        if nm(p) in quarantine: f.append("quarantined")
        if team in excl_teams: f.append("QB context")
        _, tstate, _ = team_change_state(st[st.player_display_name == p].sort_values("week").team.tolist(), team)
        if tstate != "HEALTHY_FULL": f.append("team change")
        if mk in ("rec_yards", "rush_yards", "receptions", "pass_yards") and not f:
            ctx[(p, mk)] = teammate_context_values(st, played, team_now, inj_out, p, team, mk)
        common[(p, mk)] = f

    rows = []
    for r in L.itertuples(index=False):
        f = list(common[(r.player, r.market)])
        vals, notes = ctx.get((r.player, r.market), (np.array([]), ""))
        n, h = len(vals), int((vals > r.alt_threshold).sum())
        if not f:
            if "new arrival" in notes:
                f.append("teammate arrival")  # a key competitor joined; no shared history exists
            if n < CTX_MIN_GAMES:
                f.append("teammate context")  # too few games that reflect today's role
            if not r.model_prob > r.breakeven:
                f.append("no model edge")
        rows.append(dict(r._asdict(), common_fails=f, ctx_games=n, ctx_hits=h, context=notes))
    C = pd.DataFrame(rows)
    for t in TIERS:
        C[f"fails_{t}"] = [", ".join(check(r, t)) for r in C.itertuples(index=False)]
    C["own_rate"] = C.own_hits / C.own_games.replace(0, np.nan)
    C["leg_value"] = [leg_value(r) for r in C.to_dict("records")]
    C.drop(columns=["common_fails"]).to_csv(a.out_checks, index=False)

    # p_est: half the model, half the player's own rate in games matching today's
    # teammates (Laplace-smoothed so 8/8 is not a certainty). Used only to choose
    # between legs that already passed every check.
    C["p_est"] = 0.5 * C.model_prob + 0.5 * (C.ctx_hits + 1) / (C.ctx_games + 2)
    C["dec"] = C.offered_american.map(dec)
    C["lp"] = np.log(C.p_est.clip(1e-6, 1))
    C["ld"] = np.log(C.dec)

    def candidates(t: str, keep: int = TOP_K) -> list[tuple[float, tuple]]:
        """Every one-leg-per-game ticket from tier-eligible rungs reaching the tier's
        minimum odds; the `keep` with the highest estimated hit probability."""
        E = C[C[f"fails_{t}"] == ""].reset_index(drop=True)
        target = np.log(dec(MIN_AMERICAN[t]))
        games = [g.index.tolist() for _, g in E.groupby("event_id")]
        out: list = []
        def walk(i, chosen, ld, lp):
            if ld >= target:
                out.append((lp, tuple(chosen))); return   # minimal: stop adding once the price is reached
            if i == len(games):
                return
            walk(i + 1, chosen, ld, lp)
            for j in games[i]:
                walk(i + 1, chosen + [j], ld + E.ld[j], lp + E.lp[j])
        walk(0, [], 0.0, 0.0)
        out.sort(key=lambda x: -x[0])
        return [(lp, tuple(E.loc[list(ix)].index_orig)) for lp, ix in out[:keep]]

    C["index_orig"] = C.index
    up = C[C.fails_LOTTERY == ""]
    ceiling = float(np.prod(up.groupby("event_id").dec.max())) if len(up) else 1.0
    order = ["HIGH", "MEDIUM", "MODERATE", "SAFE"]
    if ceiling >= dec(MIN_AMERICAN["LOTTERY"]):
        order = ["LOTTERY"] + order
    else:
        print(f"LOTTERY UNSOLVED: best passing price in every game multiplies to {ceiling:.0f}x < 250x")
    cands = {t: candidates(t) for t in order}
    pl = C.player.to_dict()

    # Joint choice: as many tiers filled as possible, then the highest summed
    # log hit probability, subject to no player on more than two tickets.
    best = [-1, -np.inf, {}]
    def search(k, uses, chosen, score):
        if len(chosen) + (len(order) - k) < best[0]:
            return                                   # cannot fill more tiers than the best found
        if k == len(order):
            if (len(chosen), score) > (best[0], best[1]):
                best[:] = [len(chosen), score, dict(chosen)]
            return
        t = order[k]
        for lp, ix in cands[t]:
            u = dict(uses)
            for i in ix:
                u[pl[i]] = u.get(pl[i], 0) + 1
            if max(u.values()) > MAX_TICKETS_PER_PLAYER:
                continue
            search(k + 1, u, {**chosen, t: ix}, score + lp)
        search(k + 1, uses, chosen, score)   # leave this tier unfilled
    search(0, {}, {}, 0.0)
    chosen = best[2]
    card = []
    for t in TIERS:
        ix = chosen.get(t)
        status = "" if ix else ("UNSOLVED" if t == "LOTTERY" else "SHORT")
        if not ix:
            print(f"{t}: no ticket reaches {MIN_AMERICAN[t]:+d} within the player cap [{status}]")
            continue
        for r in C.loc[list(ix)].itertuples(index=False):
            card.append(dict(ticket=t, tier=t, player=r.player, team=r.team, opponent=r.opponent,
                event_id=r.event_id, market=r.market, side="OVER", threshold=r.alt_threshold,
                price_american=int(r.offered_american), book=a.book, model_prob=round(float(r.model_prob), 4),
                breakeven=round(float(r.breakeven), 4), own=f"{int(r.own_hits)}/{int(r.own_games)}",
                ctx_own=f"{r.ctx_hits}/{r.ctx_games}", context=r.context, state_label="HEALTHY_FULL",
                price_basis="naive_product", leg_value=round(float(r.leg_value), 4), status=status,
                football_reason=(f"{r.team} vs {r.opponent}: {int(r.own_hits)}/{int(r.own_games)} over "
                                 f"{r.alt_threshold:g} ({r.ctx_hits}/{r.ctx_games} with today's teammates), "
                                 f"snaps {r.snap_pct:.0%}, model {r.model_proj:.0f} vs book {r.main_line:.0f}")))
    if not any(c["ticket"] == "LOTTERY" for c in card):
        pass
    K = pd.DataFrame(card)
    K["_o"] = K.ticket.map(TIERS.index); K = K.sort_values("_o", kind="stable").drop(columns="_o")
    K.to_csv(a.out_card, index=False)
    for t, g in K.groupby("ticket", sort=False):
        d = float(np.prod([dec(x) for x in g.price_american])); p = float(np.prod(g.model_prob))
        tag = f" [{g.status.iat[0]}]" if g.status.iat[0] else ""
        print(f"\n{t:9s} {amer(d):+7d} ({d:.1f}x) {len(g)} legs{tag} | model {p * 100:.2f}% vs breakeven {100 / d:.2f}%")
        for x in g.itertuples():
            print(f"   {x.player:20s} {x.team:3s} {x.market:10s} o{x.threshold:<6g} {x.price_american:+6d}  "
                  f"model {x.model_prob * 100:3.0f}% b/e {x.breakeven * 100:3.0f}%  own {x.own:5s} ctx {x.ctx_own:5s} {x.context}")
    res = validate(K)
    errs = [m for s, tk, m in res if s == "ERROR"]
    print(f"\nVALIDATOR: {len(errs)} error(s)"); [print("  ", m) for m in errs]
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
