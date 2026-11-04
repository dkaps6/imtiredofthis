"""Widened vacancy underreaction: 3 vacancy x 2 beneficiary cells, plus 3 controls.
Preregistration: docs/research/PRISM_VACANCY_WIDENED_PREREGISTRATION_V1.md (frozen first).
EV at posted odds. Cluster bootstrap over PLAYERS. Shares from PRIOR WEEKS ONLY. Holm-corrected.
No odds fetched. No bet authorized.
"""
from __future__ import annotations
import re, glob
import numpy as np, pandas as pd, nflreadpy as nfl

TGT_MK=["rec_yards","receptions","rush_rec_yards"]
RUSH_MK=["rush_yards","rush_rec_yards"]
VDEF={"V-STRICT":(0.25,0.35),"V-MID":(0.15,0.25),"V-LOOSE":(0.10,0.20)}
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def payout(o): return np.where(o>0,o/100.0,100.0/np.abs(o))
def grp(p):
    p=str(p).upper().strip()
    if p in ("RB","FB","HB"): return "RB"
    return p if p in ("QB","WR","TE") else "OTHER"


def build():
    rows=[]
    for f in sorted(glob.glob("data/market_track_record/boards/*.csv")):
        wk=int(re.search(r"wk(\d+)",f).group(1))
        d=pd.read_csv(f,low_memory=False); d["wk"]=wk; rows.append(d)
    b=pd.concat(rows,ignore_index=True); b["k"]=b.player.map(norm)
    for c in ("vegas_line","model_proj","vegas_over_odds","vegas_under_odds"):
        b[c]=pd.to_numeric(b[c],errors="coerce")
    b=b.dropna(subset=["vegas_line","model_proj"])

    d=nfl.load_player_stats(seasons=[2026],summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    for c in ("receiving_yards","receptions","rushing_yards","passing_yards",
              "rushing_tds","receiving_tds","targets","carries"):
        ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)
    ps["pos"]=ps.position.map(grp)
    A=ps.groupby(["k","wk"]).agg(ry=("receiving_yards","sum"),rc=("receptions","sum"),
        rs=("rushing_yards","sum"),py=("passing_yards","sum"),t1=("rushing_tds","sum"),
        t2=("receiving_tds","sum")).reset_index()
    A["anytd"]=A.t1+A.t2; A["rrl"]=A.ry+A.rs
    b=b.merge(A,on=["k","wk"],how="inner")
    MK={"rec_yards":"ry","receptions":"rc","rush_yards":"rs","pass_yards":"py",
        "rush_rec_yards":"rrl","anytime_td":"anytd"}
    b["actual"]=[getattr(r,MK[r.market]) if r.market in MK else np.nan for r in b.itertuples()]
    b=b.dropna(subset=["actual"]); b=b[b.actual!=b.vegas_line].copy()
    b["over"]=b.model_proj>b.vegas_line
    b["win"]=(b.over==(b.actual>b.vegas_line))
    b["odds"]=np.where(b.over,b.vegas_over_odds,b.vegas_under_odds)
    b=b.dropna(subset=["odds"]).copy()
    b["profit"]=np.where(b.win,payout(b.odds),-1.0)
    b["over_profit"]=np.where(b.actual>b.vegas_line,payout(b.vegas_over_odds),-1.0)
    pos=ps.groupby("k").pos.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "OTHER")
    b["pos"]=b.k.map(pos).fillna("OTHER")

    i=nfl.load_injuries(seasons=[2026]); inj=i.to_pandas() if hasattr(i,"to_pandas") else i
    inj=inj.copy(); inj["k"]=inj.full_name.map(norm)
    inj["wk"]=pd.to_numeric(inj.week,errors="coerce")
    inj["st"]=inj.report_status.astype(str).str.upper().str.strip()
    inj["pos"]=inj.position.map(grp)

    # prior-weeks-only team shares
    share={}
    for wk in sorted(b.wk.unique()):
        h=ps[ps.wk<wk]
        if h.empty: continue
        tt=h.groupby("team").targets.sum().rename("tt")
        tc=h.groupby("team").carries.sum().rename("tc")
        pl=h.groupby(["k","team"]).agg(t=("targets","sum"),c=("carries","sum")).reset_index()
        pl=pl.merge(tt,on="team").merge(tc,on="team")
        pl["ts"]=np.where(pl.tt>0,pl.t/pl.tt,0.0)
        pl["cs"]=np.where(pl.tc>0,pl.c/pl.tc,0.0)
        share[wk]={(r.k,r.team):(r.ts,r.cs) for r in pl.itertuples()}
    return b, inj, share


def flag(b, inj, share, vname, bmode, status=("OUT","DOUBTFUL")):
    tmin,cmin=VDEF[vname]
    tgt_vac=set(); rush_vac=set(); outk={}
    sub=inj[inj.st.isin(status)]
    for _,r in sub.iterrows():
        wk=r["wk"]
        if wk not in share: continue
        ts,cs=share[wk].get((r["k"],r["team"]),(0.0,0.0))
        outk.setdefault((r["team"],int(wk)),set()).add(r["k"])
        if ts>=tmin:  tgt_vac.add((r["team"],int(wk),r["pos"]))
        if cs>=cmin: rush_vac.add((r["team"],int(wk),r["pos"]))
    def hit(row):
        key=(row.team,int(row.wk))
        if row.k in outk.get(key,set()): return False     # never bet the inactive player
        groups=[row.pos] if bmode=="B-SAME" else ["WR","TE","RB"]
        if row.market in TGT_MK and any((row.team,int(row.wk),g) in tgt_vac for g in groups):
            return True
        if row.market in RUSH_MK and any((row.team,int(row.wk),g) in rush_vac for g in groups):
            return True
        return False
    return b[[hit(r) for r in b.itertuples()]].copy()


def cluster(df, col="profit", iters=12000, seed=4):
    rng=np.random.default_rng(seed)
    gs=[g[col].to_numpy() for _,g in df.groupby("k")]
    n=len(gs)
    if n<2: return np.nan,(np.nan,np.nan),n,np.nan
    o=np.empty(iters)
    for i in range(iters):
        o[i]=np.concatenate([gs[j] for j in rng.integers(0,n,n)]).mean()
    p=float((o<=0).mean())*2        # two-sided bootstrap p for EV>0
    return df[col].mean(), tuple(np.percentile(o,[2.5,97.5])), n, min(p,1.0)


def main():
    b,inj,share=build()
    print(f"graded rows with odds: {len(b)}\n")
    print("SIX PREREGISTERED CELLS (model side)")
    print(f"  {'cell':24s} {'n':>5s} {'plyr':>5s} {'win%':>7s} {'EV/$1':>8s} {'95% CI':>19s} {'p':>7s}")
    cells={}
    for v in VDEF:
        for bm in ("B-SAME","B-SKILL"):
            s=flag(b,inj,share,v,bm)
            if len(s)<2: print(f"  {v+' x '+bm:24s} n={len(s):4d}  (empty)"); continue
            ev,(lo,hi),nq,p=cluster(s)
            cells[(v,bm)]=(s,ev,lo,hi,nq,p)
            print(f"  {v+' x '+bm:24s} {len(s):5d} {nq:5d} {s.win.mean()*100:6.2f}% "
                  f"{ev:+8.4f} [{lo:+8.4f},{hi:+7.4f}] {p:7.3f}")
    # Holm correction across tested cells
    print("\nHOLM-CORRECTED G1 (family-wise 95% across the tested cells)")
    ps_=sorted([(c[5],k) for k,c in cells.items()])
    m=len(ps_); passed=set()
    for i,(p,k) in enumerate(ps_):
        thr=0.05/(m-i)
        ok=p<thr
        print(f"  {k[0]+' x '+k[1]:24s} p={p:.3f}  Holm threshold {thr:.4f}  {'PASS' if ok else 'fail'}")
        if ok: passed.add(k)
        else: break
    print("\nFULL GATE TALLY")
    for k,(s,ev,lo,hi,nq,p) in cells.items():
        wk=s.groupby("wk").profit.mean(); g2=(wk>0).sum()
        best=s.groupby("k").profit.sum().idxmax(); s2=s[s.k!=best]
        lo2=cluster(s2,seed=7)[1][0] if s2.k.nunique()>1 else np.nan
        g1=k in passed; g3=bool(np.isfinite(lo2) and lo2>0); g4=(len(s)>=200 and nq>=25)
        print(f"  {k[0]+' x '+k[1]:24s} G1 {'Y' if g1 else 'n'} | G2 {g2}/{len(wk)} "
              f"{'Y' if g2>=4 else 'n'} | G3 {'Y' if g3 else 'n'} | G4 {'Y' if g4 else 'n'}"
              f"  => {'EDGE' if (g1 and g2>=4 and g3 and g4) else 'not an edge'}")
    print("\nCONTROLS on the largest cell (V-LOOSE x B-SKILL)")
    s=cells[("V-LOOSE","B-SKILL")][0]
    ev,(lo,hi),nq,_=cluster(s); print(f"  model side        n={len(s):5d} EV {ev:+.4f} [{lo:+.4f},{hi:+.4f}]")
    ec,(lc,hc),_,_=cluster(s,col="over_profit",seed=6)
    print(f"  C1 always-OVER    n={len(s):5d} EV {ec:+.4f} [{lc:+.4f},{hc:+.4f}]")
    nv=b[~b.index.isin(s.index)&b.market.isin(sorted(set(TGT_MK+RUSH_MK)))]
    en,(ln,hn),_,_=cluster(nv,seed=10)
    print(f"  C2 non-vacancy    n={len(nv):5d} EV {en:+.4f} [{ln:+.4f},{hn:+.4f}]")
    pl=flag(b,inj,share,"V-LOOSE","B-SKILL",status=("QUESTIONABLE",))
    if len(pl)>2:
        ep,(lp,hp),npq,_=cluster(pl,seed=12)
        print(f"  C3 PLACEBO (QUEST)n={len(pl):5d} EV {ep:+.4f} [{lp:+.4f},{hp:+.4f}] players={npq}")
    b.to_csv("data/research/vacancy_widened_v1.csv",index=False)


if __name__=="__main__":
    main()
