"""Role-constant shrinkage fix: arms C0, A1-A5.
Preregistration: docs/research/PRISM_ROLE_SHRINKAGE_FIX_PREREGISTRATION_V1.md (frozen first,
amended V1a before any arm ran). Walk-forward. No sportsbook variable enters any arm.
"""
from __future__ import annotations
import re, sys
import numpy as np, pandas as pd, nflreadpy as nfl

GROUP_STRENGTH = {"tgt_share":3.0,"rush_share":3.0,"receptions_per_target":4.0,
                  "ypt":5.0,"ypc":5.0}
PRIOR_CAP      = {"tgt_share":6.0,"rush_share":6.0,"receptions_per_target":8.0,
                  "ypt":8.0,"ypc":8.0}
DEFAULTS = {
 "tgt_share":{"QB":0.0,"RB":0.08,"WR":0.16,"TE":0.12,"OTHER":0.06},
 "rush_share":{"QB":0.14,"RB":0.32,"WR":0.02,"TE":0.00,"OTHER":0.02},
 "receptions_per_target":{"QB":0.0,"RB":0.76,"WR":0.64,"TE":0.68,"OTHER":0.64},
 "ypt":{"QB":np.nan,"RB":6.0,"WR":8.0,"TE":7.4,"OTHER":7.0},
 "ypc":{"QB":5.0,"RB":4.2,"WR":6.0,"TE":4.0,"OTHER":4.2},
}
METRICS=list(GROUP_STRENGTH)
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def grp(p):
    p=str(p).upper()
    return p if p in ("QB","RB","WR","TE") else ("RB" if p=="FB" else "OTHER")

def load(seasons):
    d=nfl.load_player_stats(seasons=seasons,summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    ps["season"]=pd.to_numeric(ps.season,errors="coerce")
    for c in ["targets","receptions","receiving_yards","carries","rushing_yards",
              "attempts","passing_yards"]:
        ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)
    ps["g"]=grp_series=ps.position.map(grp)
    return ps

def rates(df):
    """Per-player metric values + games, from a block of weekly rows."""
    tt=df.groupby(["team","season","wk"]).targets.sum().rename("tt").reset_index()
    tc=df.groupby(["team","season","wk"]).carries.sum().rename("tc").reset_index()
    x=df.merge(tt,on=["team","season","wk"],how="left").merge(tc,on=["team","season","wk"],how="left")
    x["ts"]=np.where(x.tt>0,x.targets/x.tt,np.nan)
    x["rs"]=np.where(x.tc>0,x.carries/x.tc,np.nan)
    a=x.groupby("k").agg(games=("wk","nunique"),tgt=("targets","sum"),rec=("receptions","sum"),
        ry=("receiving_yards","sum"),car=("carries","sum"),rsh=("rushing_yards","sum"),
        ts=("ts","mean"),rs=("rs","mean"))
    out=pd.DataFrame(index=a.index)
    out["games"]=a.games
    out["tgt_share"]=a.ts
    out["rush_share"]=a.rs
    out["receptions_per_target"]=np.where(a.tgt>0,a.rec/a.tgt,np.nan)
    out["ypt"]=np.where(a.tgt>0,a.ry/a.tgt,np.nan)
    out["ypc"]=np.where(a.car>0,a.rsh/a.car,np.nan)
    return out

def weighted_group_mean(ev,metric,pos):
    """Replicates _weighted_group_mean: prior_games-weighted mean within position group."""
    m=(ev.g==pos)&ev[f"{metric}_prior"].notna()&(ev.prior_games>0)
    if m.any() and ev.loc[m,"prior_games"].sum()>0:
        return float(np.average(ev.loc[m,f"{metric}_prior"],weights=ev.loc[m,"prior_games"]))
    pv=ev[f"{metric}_prior"]
    if pv.notna().any(): return float(pv.median())
    return float(DEFAULTS[metric][pos])

def posterior(anchor,aw,pv,pg,cv,cg,metric,cap):
    w=[aw]; v=[anchor]
    if np.isfinite(pv) and pg>0: w.append(min(float(pg),cap)); v.append(float(pv))
    if np.isfinite(cv) and cg>0: w.append(float(cg)); v.append(float(cv))
    w=np.asarray(w,float); v=np.asarray(v,float)
    if not np.isfinite(v).all() or w.sum()<=0: return np.nan
    return float(np.average(v,weights=w))

ARMS=["C0","A1","A2","A3","A4","A5"]
def run_arm(arm,ev,metric,gm_pos,gm_rank,kappa):
    GS,CAP=GROUP_STRENGTH[metric],PRIOR_CAP[metric]
    out=np.full(len(ev),np.nan)
    pv=ev[f"{metric}_prior"].to_numpy(); cv=ev[f"{metric}_current"].to_numpy()
    pg=ev.prior_games.to_numpy(); cg=ev.current_games.to_numpy()
    for i in range(len(ev)):
        pos=ev.g.iloc[i]
        cap = np.inf if arm in ("A1","A3") else CAP
        if arm=="A4":   anchor=gm_rank.get((pos,ev.rank_band.iloc[i]),gm_pos[pos])
        elif arm=="A5": anchor=pv[i] if np.isfinite(pv[i]) and pg[i]>0 else gm_pos[pos]
        else:           anchor=gm_pos[pos]
        if arm in ("A2","A3"):
            n=float(pg[i])+float(cg[i]); aw=GS*kappa/(kappa+n)
        else: aw=GS
        out[i]=posterior(anchor,aw,pv[i],pg[i],cv[i],cg[i],metric,cap)
    return out


def build_evidence(ps, target_wk):
    """Pregame evidence for 2026 week `target_wk`: prior = 2025, current = 2026 W1..k-1."""
    prior=rates(ps[(ps.season==2025)])
    cur  =rates(ps[(ps.season==2026)&(ps.wk<target_wk)])
    univ =ps[(ps.season==2026)&(ps.wk==target_wk)][["k","team","position"]].drop_duplicates("k")
    ev=univ.set_index("k")
    ev["g"]=ev.position.map(grp)
    ev["prior_games"]=prior.games.reindex(ev.index).fillna(0.0)
    ev["current_games"]=cur.games.reindex(ev.index).fillna(0.0)
    for m in METRICS:
        ev[f"{m}_prior"]=prior[m].reindex(ev.index)
        ev[f"{m}_current"]=cur[m].reindex(ev.index)
    # depth band for A4: within team+group, by current-season volume (pregame only)
    vol=ps[(ps.season==2026)&(ps.wk<target_wk)].groupby("k").agg(
        v=("targets","sum"),c=("carries","sum"))
    ev["vol"]=(vol.v.reindex(ev.index).fillna(0.0)+vol.c.reindex(ev.index).fillna(0.0))
    r=ev.groupby(["team","g"]).vol.rank(ascending=False,method="first")
    ev["rank_band"]=np.where(r<=1,"R1",np.where(r<=2,"R2","R3"))
    return ev.reset_index()


def realized(ps, wk):
    return rates(ps[(ps.season==2026)&(ps.wk==wk)])


def main():
    ps=load([2025,2026])
    WEEKS=[w for w in range(2,9)]
    EV={}; ACT={}
    for w in WEEKS:
        EV[w]=build_evidence(ps,w); ACT[w]=realized(ps,w)
    GMC={}
    def gmean(w,m,pos):
        if (w,m,pos) not in GMC: GMC[(w,m,pos)]=weighted_group_mean(EV[w],m,pos)
        return GMC[(w,m,pos)]
    KGRID=[2.0,4.0,8.0,16.0,32.0]
    recs=[]
    gm_report=[]
    for wk in WEEKS:
        ev=EV[wk]; act=ACT[wk]
        gm_pos={}; gm_rank={}
        for m in METRICS:
            for pos in ("QB","RB","WR","TE","OTHER"):
                gm_pos[(m,pos)]=gmean(wk,m,pos)
            for (pos,band),s in ev.groupby(["g","rank_band"]):
                mm=s[f"{m}_prior"]; pgs=s.prior_games
                ok=mm.notna()&(pgs>0)
                gm_rank[(m,pos,band)]=(float(np.average(mm[ok],weights=pgs[ok]))
                                       if ok.any() and pgs[ok].sum()>0 else gm_pos[(m,pos)])
        for m in METRICS:
            gmp={p:gm_pos[(m,p)] for p in ("QB","RB","WR","TE","OTHER")}
            gmr={(p,bd):gm_rank[(m,p,bd)] for p in ("QB","RB","WR","TE","OTHER")
                 for bd in ("R1","R2","R3") if (m,p,bd) in gm_rank}
            gm_report.append({"wk":wk,"metric":m,**{f"gm_{p}":gmp[p] for p in ("RB","WR","TE")}})
            # kappa fitted on EARLIER weeks only
            kap=8.0
            if wk>WEEKS[0]:
                best=(np.inf,8.0)
                for kk in KGRID:
                    err=[]
                    for pw in [w for w in WEEKS if w<wk]:
                        pev=EV[pw]; pact=ACT[pw]
                        p=run_arm("A2",pev,m,{x:gmean(pw,m,x) for x in
                                 ("QB","RB","WR","TE","OTHER")},{},kk)
                        tv=pact[m].reindex(pev.k).to_numpy()
                        ok=np.isfinite(p)&np.isfinite(tv)
                        if ok.sum()>20: err.append(np.abs(p[ok]-tv[ok]).mean())
                    if err and np.mean(err)<best[0]: best=(np.mean(err),kk)
                kap=best[1]
            tv=act[m].reindex(ev.k).to_numpy()
            for arm in ARMS:
                est=run_arm(arm,ev,m,gmp,gmr,kap)
                for i in range(len(ev)):
                    if np.isfinite(est[i]) and np.isfinite(tv[i]):
                        recs.append({"wk":wk,"metric":m,"arm":arm,"k":ev.k.iloc[i],
                                     "g":ev.g.iloc[i],"team":ev.team.iloc[i],
                                     "est":est[i],"true":tv[i],"kappa":kap,
                                     "ev":ev.prior_games.iloc[i]+ev.current_games.iloc[i],
                                     "band":ev.rank_band.iloc[i]})
    r=pd.DataFrame(recs)
    r.to_csv("data/research/role_shrinkage_arms_v1.csv",index=False)
    pd.DataFrame(gm_report).drop_duplicates().to_csv(
        "data/research/role_shrinkage_group_means_v1.csv",index=False)
    print(f"rows={len(r)}  weeks={sorted(r.wk.unique())}  metrics={sorted(r.metric.unique())}")
    return r


if __name__=="__main__":
    main()
