"""Situational edge: vacancy underreaction (H1) and the alpha-tier side (H2).
Preregistration: docs/research/PRISM_SITUATIONAL_EDGE_PREREGISTRATION_V1.md (frozen first).
EV at posted odds. Cluster bootstrap over PLAYERS. All tiers from PRIOR WEEKS ONLY.
No odds fetched. No bet authorized.
"""
from __future__ import annotations
import re, glob
import numpy as np, pandas as pd, nflreadpy as nfl
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

    # ---- player position (modal, from stats) ----
    pos=ps.groupby("k").pos.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "OTHER")
    b["pos"]=b.k.map(pos).fillna("OTHER")

    # ---- H1: teammate vacancy, from the PREGAME injury report ----
    i=nfl.load_injuries(seasons=[2026]); inj=i.to_pandas() if hasattr(i,"to_pandas") else i
    inj=inj.copy(); inj["k"]=inj.full_name.map(norm)
    inj["wk"]=pd.to_numeric(inj.week,errors="coerce")
    inj["st"]=inj.report_status.astype(str).str.upper().str.strip()
    inj["pos"]=inj.position.map(grp)
    out=inj[inj.st.isin(["OUT","DOUBTFUL"])]
    # prior-weeks-only usage, to identify whether the OUT player was a PRIMARY
    prior={}
    for wk in sorted(b.wk.unique()):
        h=ps[ps.wk<wk]
        if h.empty: continue
        u=h.groupby(["k","team"]).targets.sum().rename("t").reset_index()
        u["rk"]=u.groupby("team").t.rank(ascending=False,method="first")
        prior[wk]=set(u[(u.rk<=2)&(u.t>0)].k)
    vac=set()
    for _,r in out.iterrows():
        wk=r["wk"]
        if wk in prior and r["k"] in prior[wk]:
            vac.add((r["team"],int(wk),r["pos"]))
    b["vacancy"]=[1 if (r.team,int(r.wk),r.pos) in vac and r.k not in
                  {x for x in out[out.wk==r.wk].k} else 0 for r in b.itertuples()]

    # ---- H2: pregame volume quartile ----
    b["lvl"]=np.nan
    for wk in sorted(b.wk.unique()):
        h=ps[ps.wk<wk]
        if h.empty: continue
        v=h.groupby("k").agg(t=("targets","sum"),c=("carries","sum"))
        lv=(v.t+v.c)
        m=b.wk==wk
        b.loc[m,"lvl"]=b.loc[m,"k"].map(lv)
    bb=b.dropna(subset=["lvl"]).copy()
    bb["q"]=bb.groupby("market",observed=True).lvl.transform(
        lambda s: pd.qcut(s,4,labels=[0,1,2,3],duplicates="drop") if s.nunique()>=4 else np.nan)
    return b, bb.dropna(subset=["q"])


def cluster(df, iters=20000, seed=3, col="profit"):
    rng=np.random.default_rng(seed)
    gs=[g[col].to_numpy() for _,g in df.groupby("k")]
    n=len(gs)
    if n<2: return np.nan,(np.nan,np.nan),n
    o=np.empty(iters)
    for i in range(iters):
        o[i]=np.concatenate([gs[j] for j in rng.integers(0,n,n)]).mean()
    return df[col].mean(), tuple(np.percentile(o,[2.5,97.5])), n


def gates(name, s):
    if len(s)==0:
        print(f"  {name:34s} EMPTY"); return
    ev,(lo,hi),nq=cluster(s)
    g1=lo>0
    wk=s.groupby("wk").profit.mean()
    g2=(wk>0).sum()
    best=s.groupby("k").profit.sum().idxmax()
    s2=s[s.k!=best]
    _,(lo2,_),_=cluster(s2,seed=8) if s2.k.nunique()>1 else (np.nan,(np.nan,np.nan),0)
    g3=lo2>0 if np.isfinite(lo2) else False
    g4=(len(s)>=200) and (nq>=25)
    print(f"  {name:34s} n={len(s):5d} players={nq:3d} win {s.win.mean()*100:5.2f}% "
          f"EV {ev:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"     G1 CI>0 {'Y' if g1 else 'n'} | G2 weeks+ {g2}/{len(wk)} "
          f"{'Y' if g2>=4 else 'n'} | G3 drop-best {'Y' if g3 else 'n'} | "
          f"G4 size {'Y' if g4 else 'n'}  => "
          f"{'EDGE' if (g1 and g2>=4 and g3 and g4) else 'not an edge'}")


def main():
    b,bb=build()
    print(f"graded rows with odds: {len(b)}   vacancy rows: {int(b.vacancy.sum())}")
    print(f"vacancy weeks present: {sorted(b[b.vacancy==1].wk.unique())}")
    print("\n[H1] VACANCY UNDERREACTION")
    gates("V1 vacancy rows (model side)", b[b.vacancy==1])
    gates("V1b NO-vacancy rows (baseline)", b[b.vacancy==0])
    gates("V2 vacancy + model says OVER", b[(b.vacancy==1)&b.over])
    v3=b[b.vacancy==1].copy()
    v3["profit"]=np.where(v3.actual>v3.vegas_line,payout(v3.vegas_over_odds),-1.0)
    v3["win"]=v3.actual>v3.vegas_line
    gates("V3 control: ALWAYS over, vacancy", v3)
    print("\n[H2] ALPHA-TIER SIDE  (pregame volume quartile)")
    gates("A1 top quartile + model UNDER", bb[(bb.q==3)&(~bb.over)])
    gates("A2 top quartile + model OVER",  bb[(bb.q==3)&(bb.over)])
    gates("A3 bottom quartile + UNDER",    bb[(bb.q==0)&(~bb.over)])
    gates("A3 bottom quartile + OVER",     bb[(bb.q==0)&(bb.over)])
    b.to_csv("data/research/situational_edge_v1.csv",index=False)


if __name__=="__main__":
    main()
