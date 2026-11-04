"""Grade a ticket ledger (alt-line legs) against realized outcomes.
Handles overs AND unders, and treats a player who did not play as VOID, not a win.
Usage: python scripts/research/grade_ticket_ledger_v1.py <ledger.csv> <week>
"""
from __future__ import annotations
import re, sys
import numpy as np, pandas as pd, nflreadpy as nfl
MK={"pass_yards":"py","rec_yards":"ry","rush_yards":"rs","receptions":"rc",
    "rush_rec_yards":"rrl","anytime_td":"anytd"}
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def payout(o): return np.where(o>0,o/100.0,100.0/np.abs(o))


def grade(path, week):
    d=pd.read_csv(path)
    if "direction" not in d.columns: d["direction"]="over"
    d["direction"]=d.direction.fillna("over")
    d["k"]=d.player.map(norm)
    d["thr"]=pd.to_numeric(d.threshold.astype(str).str.replace(r"[^0-9.]","",regex=True),
                           errors="coerce")
    d.loc[d.market.eq("anytime_td"),"thr"]=1.0
    ps=nfl.load_player_stats(seasons=[2026],summary_level="week")
    ps=ps.to_pandas() if hasattr(ps,"to_pandas") else ps
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    for c in ("passing_yards","receiving_yards","rushing_yards","receptions",
              "rushing_tds","receiving_tds","attempts","targets","carries"):
        ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)
    w=ps[ps.wk==week].groupby("k").agg(py=("passing_yards","sum"),ry=("receiving_yards","sum"),
        rs=("rushing_yards","sum"),rc=("receptions","sum"),t1=("rushing_tds","sum"),
        t2=("receiving_tds","sum"),vol=("attempts","sum"),vt=("targets","sum"),
        vc=("carries","sum"))
    w["anytd"]=w.t1+w.t2; w["rrl"]=w.ry+w.rs
    # VOID unless the player recorded real volume. Presence in the weekly table is NOT
    # proof of playing: a scratched starter can appear with all-zero stats, and counting
    # that as an UNDER win flatters the result.
    w["touched"]=(w.vol+w.vt+w.vc)>0
    played=set(w.index[w.touched])
    d["played"]=d.k.isin(played)
    d["col"]=d.market.map(MK)
    d["actual"]=[w[r.col].get(r.k,np.nan) if isinstance(r.col,str) else np.nan
                 for r in d.itertuples()]
    d["hit"]=np.where(d.direction.eq("under"), d.actual<d.thr,
                      d.actual>=d.thr).astype(float)
    d.loc[~d.played,"hit"]=np.nan          # VOID: did not play
    return d


def report(d, label):
    g=d[d.played & d.hit.notna()].copy()
    void=int((~d.played).sum())
    print(f"=== {label} ===")
    print(f"  legs {len(d)}   graded {len(g)}   VOID (did not play) {void}")
    print(f"  LEG HIT RATE {g.hit.mean()*100:.1f}%  ({int(g.hit.sum())}/{len(g)})")
    gp=g.dropna(subset=["american_price"]).copy()
    if len(gp):
        gp["profit"]=np.where(gp.hit.astype(bool),payout(gp.american_price),-1.0)
        print(f"  EV/$1 on {len(gp)} priced legs bet straight: {gp.profit.mean():+.4f}"
              f"   median price {gp.american_price.median():.0f}")
    print(f"  {'market':16s} {'n':>4s} {'hit%':>7s} {'med thr':>8s}")
    for m,s in g.groupby("market"):
        print(f"  {m:16s} {len(s):4d} {s.hit.mean()*100:6.1f}% {s.thr.median():8.1f}")
    print(f"  {'direction':16s} {'n':>4s} {'hit%':>7s}")
    for dr,s in g.groupby("direction"):
        print(f"  {dr:16s} {len(s):4d} {s.hit.mean()*100:6.1f}%")
    print(f"\n  TICKETS (void legs removed; a ticket needs every remaining leg)")
    tk=g.groupby("ticket_id").agg(legs=("hit","size"),hits=("hit","sum"),tier=("tier","first"))
    tk["won"]=tk.legs==tk.hits
    print(f"  tickets {len(tk)}   WON {int(tk.won.sum())}  ({tk.won.mean()*100:.1f}%)")
    print(f"  {'tier':12s} {'legs':>5s} {'hits':>5s} {'won':>4s} {'leg hit%':>9s}")
    for t,s in tk.groupby("tier"):
        lh=g[g.tier==t].hit.mean()*100
        print(f"  {str(t):12s} {int(s.legs.sum()):5d} {int(s.hits.sum()):5d} "
              f"{int(s.won.sum()):4d} {lh:8.1f}%")
    nm=int(((tk.legs-tk.hits)==1).sum())
    print(f"  lost by exactly one leg: {nm} of {len(tk)}")
    return g


if __name__=="__main__":
    p,wk=sys.argv[1],int(sys.argv[2])
    d=grade(p,wk)
    g=report(d,f"{p.split('/')[-2]}  week {wk}")
    out=p.replace(".csv","_graded.csv"); d.to_csv(out,index=False)
    print(f"\n  wrote {out}")
