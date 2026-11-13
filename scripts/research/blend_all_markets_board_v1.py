#!/usr/bin/env python3
"""One board, every market, one estimator -- single-event card builder.

P(leg) = 0.45 de-vigged market + 0.25 model + 0.30 current-role opportunity.

The opportunity component is conditioned on TONIGHT'S quarterback from
play-by-play: targets ~ Binomial(attempts, share of that QB's targets),
receptions ~ Binomial(targets, catch rate), yards = sum of the player's own
per-catch yards. Receptions are NOT excluded: the operator's rule is "no ticket
made only of reception alternates", which is a ticket-construction cap, not a
market ban. Rushing uses per-game history times the rushing matchup factor.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

EVENT, BOARD, ALT, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
QB = {"NYG": "J.J. McCarthy", "WAS": "Jayden Daniels"}
W = (0.45, 0.25, 0.30)
SCHEMES = [(0.45, 0.25, 0.30), (0.60, 0.20, 0.20), (0.45, 0.40, 0.15), (0.45, 0.10, 0.45)]
RNG = np.random.default_rng(20261113); N = 100_000
MK = {"player_reception_yds_alternate": "rec_yards", "player_rush_yds_alternate": "rush_yards",
      "player_pass_yds_alternate": "pass_yards", "player_rush_reception_yds_alternate": "rush_rec_yards",
      "player_receptions_alternate": "receptions"}
nm = lambda s: "".join(c for c in str(s).lower() if c.isalpha())
def ip(x): x = float(x); return (-x) / ((-x) + 100) if x < 0 else 100 / (x + 100)
def dec(x): x = float(x); return 1 + x / 100 if x > 0 else 1 + 100 / -x

import nflreadpy as nfl
st = nfl.load_player_stats(seasons=[2026]).to_pandas()
st["k"] = st.player_display_name.map(nm)
pbp = nfl.load_pbp(seasons=[2026]).to_pandas()
pp = pbp[(pbp.posteam.isin(QB)) & (pbp.pass_attempt == 1) & (pbp.sack.fillna(0) == 0) & (pbp.play_type == "pass")]
board = pd.read_csv(BOARD); board["k"] = board.player.map(nm)
alt = pd.read_csv(ALT); alt = alt[alt.event_id == EVENT].copy()
alt["market"] = alt.market.map(MK); alt = alt[alt.market.notna()]; alt["k"] = alt.player.map(nm)
pid = st.groupby("k").player_id.first(); team = board.groupby("k").team.first()
pos = st.groupby("k").position.first(); st["fam"] = st.k.map(pos)

def matchup(opp, fams, col):
    sub = st[st.fam.isin(fams)]
    pg = sub.groupby(["opponent_team", "week"])[col].sum().groupby("opponent_team").mean()
    return float(pg.loc[opp] / pg.mean()) if opp in pg.index else 1.0

# Per-ATTEMPT pass-defense factors. A per-game yards-allowed ratio double counts
# volume, which the projected attempts already carry; these are efficiency only.
_qb = st[st.attempts.fillna(0) > 0].groupby("opponent_team")[["passing_yards", "attempts", "completions"]].sum()
YPA_F = (_qb.passing_yards / _qb.attempts) / (_qb.passing_yards.sum() / _qb.attempts.sum())
CMP_F = (_qb.completions / _qb.attempts) / (_qb.completions.sum() / _qb.attempts.sum())
snaps = nfl.load_snap_counts(seasons=[2026]).to_pandas(); snaps["k"] = snaps.player.map(nm)
PLAYED = snaps[snaps.offense_snaps.fillna(0) > 0].groupby("k").week.apply(set)

# ---- opportunity simulation for pass catchers, conditioned on tonight's QB ----
SIM, INFO = {}, {}
for k in sorted(set(board[board.market.isin(["receptions", "rec_yards"])].k)):
    if k not in pid.index or k not in team.index: continue
    t = team[k]; opp = "WAS" if t == "NYG" else "NYG"
    qid = pid[nm(QB[t])]
    # Share denominator: the QB's attempts ONLY in games this receiver played.
    wk_played = PLAYED.get(k, set())
    q = pp[(pp.posteam == t) & (pp.passer_player_id == qid) & (pp.week.isin(wk_played))]
    mine = pp[pp.receiver_player_id == pid[k]]
    mq = mine[mine.passer_player_id == qid]; mo = mine[mine.passer_player_id != qid]
    tq = len(mq); n = len(q)
    T = len(mine); R = int(mine.complete_pass.sum())
    Yq = mq.yards_gained[mq.complete_pass == 1].to_numpy(float)
    Yall = mine.yards_gained[mine.complete_pass == 1].to_numpy(float)
    if n < 15 or T < 8 or len(Yall) < 4: continue
    att = float(board[(board.market == "pass_yards") & (board.k == nm(QB[t]))].qb_pred_attempts.iloc[0])
    A = np.clip(np.rint(RNG.normal(att, 6.0, N)), 12, 60).astype(int)
    share = RNG.beta(tq + 1, n - tq + 1, N)
    tg = RNG.binomial(A, share)
    # Efficiency leans on plays with tonight's QB; other-QB plays count half.
    Rq, Ro = int(mq.complete_pass.sum()), int(mo.complete_pass.sum())
    cr = RNG.beta(Rq + 0.5 * Ro + 1, (tq - Rq) + 0.5 * (len(mo) - Ro) + 1, N)
    cr = np.clip(cr * float(CMP_F.get(opp, 1.0)), 0.02, 0.98)
    rec = RNG.binomial(tg, cr)
    kmax = int(rec.max()) + 1
    wq = len(Yq) / (len(Yq) + 15.0)
    pick_q = RNG.random((N, kmax)) < wq
    draw_q = RNG.choice(Yq, size=(N, kmax)) if len(Yq) else np.zeros((N, kmax))
    cs = np.cumsum(np.where(pick_q, draw_q, RNG.choice(Yall, size=(N, kmax))), axis=1)
    yds = np.where(rec > 0, cs[np.arange(N), np.maximum(rec - 1, 0)], 0.0)
    yds = yds * float(YPA_F.get(opp, 1.0)) / float(CMP_F.get(opp, 1.0))
    SIM[k] = {"receptions": rec.astype(float), "rec_yards": yds}
    INFO[k] = dict(qb=QB[t], share=tq / n, tq=tq, n=n, exp_tgt=float(tg.mean()), catch=R / max(T, 1),
                   exp_rec=float(rec.mean()), exp_yds=float(yds.mean()))

def own_rush(k, mk):
    col = {"rush_yards": "rushing_yards", "pass_yards": "passing_yards"}.get(mk)
    s = st[st.k == k]
    if mk == "rush_rec_yards": v = (s.rushing_yards.fillna(0) + s.receiving_yards.fillna(0)).to_numpy(float)
    elif col: v = s[col].dropna().to_numpy(float)
    else: return None
    if len(v) < 5: return None
    t = team.get(k); opp = "WAS" if t == "NYG" else "NYG"
    f = matchup(opp, ("QB",), "passing_yards") if mk == "pass_yards" else matchup(opp, ("RB",), "rushing_yards")
    return v * f

def model_draws(mean, sd, discrete):
    if discrete:
        var = max(sd * sd, mean + 1e-6)
        if var > mean * 1.02:
            r = mean * mean / (var - mean); return RNG.negative_binomial(r, r / (r + mean), N).astype(float)
        return RNG.poisson(mean, N).astype(float)
    kk = (mean / sd) ** 2; return RNG.gamma(kk, sd * sd / mean, N)

def hold(k, mk):
    r = board[(board.k == k) & (board.market == mk)]
    if r.empty: return 1.0
    r = r.iloc[0]; s = ip(r.vegas_over_odds) + ip(r.vegas_under_odds)
    return s if s > 0.9 else 1.0

rows = []
def add(player, k, mk, side, line, price, book, p_mkt, p_model, p_own, kind):
    comps = [(W[0], p_mkt), (W[1], p_model)] + ([(W[2], p_own)] if p_own is not None else [])
    tot = sum(w for w, _ in comps); p = sum(w * x for w, x in comps) / tot
    evs = []
    for wm, wd, wo in SCHEMES:
        c = [(wm, p_mkt), (wd, p_model)] + ([(wo, p_own)] if p_own is not None else [])
        evs.append(sum(w * x for w, x in c) / sum(w for w, _ in c) * dec(price) - 1)
    rows.append(dict(player=player, market=mk, side=side, line=line, price=price, book=book, kind=kind,
                     p_mkt=p_mkt, p_model=p_model, p_own=p_own, p=p, be=1 / dec(price),
                     ev=p * dec(price) - 1, ev_min=min(evs), ev_max=max(evs), dec=dec(price)))

MD = {}
for r in board.drop_duplicates(["k", "market"]).itertuples():
    if r.market in MK.values() and r.model_proj > 0 and r.model_sd > 0:
        MD[(r.k, r.market)] = model_draws(float(r.model_proj), float(r.model_sd), r.market == "receptions")

# alternate ladders (overs)
for (k, mk), g in alt[alt.side.astype(str).str.upper() == "OVER"].groupby(["k", "market"]):
    if (k, mk) not in MD: continue
    h = hold(k, mk); own = SIM.get(k, {}).get(mk) if mk in ("receptions", "rec_yards") else own_rush(k, mk)
    x = g.loc[g.groupby("line").price_american.idxmax()].sort_values("line")
    pm = np.minimum.accumulate((x.price_american.map(ip) / h).clip(0.001, 0.999).to_numpy())
    for (r, m) in zip(x.itertuples(), pm):
        add(r.player, k, mk, "OVER", float(r.line), float(r.price_american), r.book, float(m),
            float((MD[(k, mk)] > r.line).mean()), None if own is None else float((own > r.line).mean()), "alt")
# main lines, both sides
for (k, mk, line), g in board[board.market.isin(MK.values())].groupby(["k", "market", "vegas_line"]):
    if (k, mk) not in MD: continue
    own = SIM.get(k, {}).get(mk) if mk in ("receptions", "rec_yards") else own_rush(k, mk)
    for side in ("OVER", "UNDER"):
        col = "vegas_over_odds" if side == "OVER" else "vegas_under_odds"
        b = g.loc[g[col].idxmax()]; po, pu = ip(b.vegas_over_odds), ip(b.vegas_under_odds)
        pmk = po / (po + pu); pmd = float((MD[(k, mk)] > line).mean()); pow_ = None if own is None else float((own > line).mean())
        if side == "UNDER": pmk, pmd, pow_ = 1 - pmk, 1 - pmd, (None if pow_ is None else 1 - pow_)
        add(b.player, k, mk, side, float(line), float(b[col]), b.book, pmk, pmd, pow_, "main")
B = pd.DataFrame(rows); B.to_csv(OUT, index=False)
I = pd.DataFrame(INFO).T
print(f"pass-defense factors per attempt -- vs NYG: YPA {YPA_F.get('NYG',1):.3f} cmp {CMP_F.get('NYG',1):.3f} | vs WAS: YPA {YPA_F.get('WAS',1):.3f} cmp {CMP_F.get('WAS',1):.3f}")
print("pass catchers, opportunity conditioned on tonight's QB:")
print(I[["qb", "tq", "n", "share", "exp_tgt", "catch", "exp_rec", "exp_yds"]].to_string(float_format=lambda x: f"{x:.2f}"))
print(f"\nboard rows: {len(B)}  -> {OUT}")
