"""W9 2026 PROSPECTIVE CAPTURE -- frozen before kickoff.
Preregistration: docs/research/PRISM_W9_PROSPECTIVE_CAPTURE_V1.md (written in the same commit).

Purpose: the holdout every V22 study lacked. Builds a W9 player universe from data available
through W8 only, then records BOTH the production-equivalent baseline and the D3 arm for every
player. The D1 overlay is added in a later commit once W9 designations publish (they are not
available as of capture time).

This is a RECONSTRUCTION of the production path, not a production run. It validated at r=+0.967
against shipped model_proj on W7+W8 (see PRISM_D3_PHANTOM_YARDS_RESULTS_V1 M4). It therefore
tests the FIX, not the production integration.
"""
from __future__ import annotations
import re, sys, json, datetime as dt
import numpy as np, pandas as pd, nflreadpy as nfl
sys.path.insert(0,".")
from scripts.simulation_v2 import _allocate_counts, _sharpen_wr_target_shares, \
    _sharpen_rb_rush_shares, _top_n_shares

TARGET_WEEK=9; PRIOR=8; ITER=4000
GS={"tgt_share":3.0,"rush_share":3.0,"ypt":5.0,"ypc":5.0}
CAP={"tgt_share":6.0,"rush_share":6.0,"ypt":8.0,"ypc":8.0}
DEF={"tgt_share":{"QB":0.0,"RB":0.08,"WR":0.16,"TE":0.12,"OTHER":0.06},
     "rush_share":{"QB":0.14,"RB":0.32,"WR":0.02,"TE":0.00,"OTHER":0.02},
     "ypt":{"QB":np.nan,"RB":6.0,"WR":8.0,"TE":7.4,"OTHER":7.0},
     "ypc":{"QB":5.0,"RB":4.2,"WR":6.0,"TE":4.0,"OTHER":4.2}}
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def grp(p):
    p=str(p).upper().strip()
    if p in ("RB","FB","HB"): return "RB"
    return p if p in ("QB","WR","TE") else "OTHER"


def build():
    d=nfl.load_player_stats(seasons=[2025,2026],summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    ps["season"]=pd.to_numeric(ps.season,errors="coerce")
    for c in ("targets","receptions","receiving_yards","carries","rushing_yards","attempts"):
        ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)
    # HARD GUARD: nothing from week >= TARGET_WEEK may be read
    cur=ps[(ps.season==2026)&(ps.wk<TARGET_WEEK)]
    assert cur.wk.max()<=PRIOR, f"LOOKAHEAD: read week {cur.wk.max()}"
    pri=ps[ps.season==2025]

    def rates(df):
        tt=df.groupby(["team","season","wk"]).targets.sum().rename("tt").reset_index()
        tc=df.groupby(["team","season","wk"]).carries.sum().rename("tc").reset_index()
        x=df.merge(tt,on=["team","season","wk"]).merge(tc,on=["team","season","wk"])
        x["ts"]=np.where(x.tt>0,x.targets/x.tt,np.nan)
        x["rs"]=np.where(x.tc>0,x.carries/x.tc,np.nan)
        a=x.groupby("k").agg(games=("wk","nunique"),tgt=("targets","sum"),ry=("receiving_yards","sum"),
                             car=("carries","sum"),rsh=("rushing_yards","sum"),
                             ts=("ts","mean"),rs=("rs","mean"))
        o=pd.DataFrame(index=a.index)
        o["games"]=a.games; o["tgt_share"]=a.ts; o["rush_share"]=a.rs
        o["ypt"]=np.where(a.tgt>0,a.ry/a.tgt,np.nan)
        o["ypc"]=np.where(a.car>0,a.rsh/a.car,np.nan)
        return o
    P,C=rates(pri),rates(cur)

    sch=nfl.load_schedules(seasons=[2026]); sc=sch.to_pandas() if hasattr(sch,"to_pandas") else sch
    sc["week"]=pd.to_numeric(sc.week,errors="coerce")
    w=sc[sc.week==TARGET_WEEK]
    opp={}
    for _,r in w.iterrows():
        opp[r.home_team]=r.away_team; opp[r.away_team]=r.home_team
    teams=sorted(opp)

    # universe: players with any 2026 usage through W8, on a W9 team
    last=cur.sort_values("wk").groupby("k").tail(1)[["k","team","position"]]
    use=cur.groupby("k").agg(t=("targets","sum"),c=("carries","sum"),a=("attempts","sum")).reset_index()
    u=last.merge(use,on="k")
    u=u[(u.t+u.c+u.a)>0]
    u=u[u.team.isin(teams)].copy()
    u["g"]=u.position.map(grp); u["opponent"]=u.team.map(opp)
    u["prior_games"]=P.games.reindex(u.k).fillna(0.0).to_numpy()
    u["current_games"]=C.games.reindex(u.k).fillna(0.0).to_numpy()

    def gm(metric,pos):
        m=(u.g==pos)
        pv=P[metric].reindex(u.k).to_numpy(); pg=u.prior_games.to_numpy()
        ok=m.to_numpy()&np.isfinite(pv)&(pg>0)
        if ok.any() and pg[ok].sum()>0: return float(np.average(pv[ok],weights=pg[ok]))
        v=pv[np.isfinite(pv)]
        return float(np.median(v)) if len(v) else float(DEF[metric][pos])
    for metric in GS:
        anchors={p:gm(metric,p) for p in ("QB","RB","WR","TE","OTHER")}
        pv=P[metric].reindex(u.k).to_numpy(); cv=C[metric].reindex(u.k).to_numpy()
        pg=u.prior_games.to_numpy(); cg=u.current_games.to_numpy()
        est=[]
        for i in range(len(u)):
            wts=[GS[metric]]; vals=[anchors[u.g.iloc[i]]]
            if np.isfinite(pv[i]) and pg[i]>0: wts.append(min(pg[i],CAP[metric])); vals.append(pv[i])
            if np.isfinite(cv[i]) and cg[i]>0: wts.append(cg[i]); vals.append(cv[i])
            est.append(float(np.average(vals,weights=wts)))
        u[metric]=est
    # team pace/pass-rate from prior weeks only
    tp=cur.groupby(["team","wk"]).agg(a=("attempts","sum"),c=("carries","sum")).reset_index()
    tp["plays"]=tp.a+tp.c
    agg=tp.groupby("team").agg(plays=("plays","mean"),att=("a","mean")).reset_index()
    agg["pass_rate"]=(agg.att/agg.plays).clip(0.35,0.75)
    agg["plays"]=agg.plays.clip(50,80)
    u=u.merge(agg[["team","plays","pass_rate"]],on="team",how="left")
    u["plays"]=u.plays.fillna(64.0); u["pass_rate"]=u.pass_rate.fillna(0.58)
    return u


def project(u, d3):
    out=[]
    for tm,g in u.groupby("team"):
        g=g.reset_index(drop=True)
        rng=np.random.default_rng(abs(hash((tm,TARGET_WEEK)))%(2**32))
        plays=int(round(float(g.plays.iloc[0]))); pr=float(g.pass_rate.iloc[0])
        pa=rng.binomial(plays,pr,ITER); ra=plays-pa
        tsh=_sharpen_wr_target_shares(g,g.tgt_share.to_numpy())
        rsh=_sharpen_rb_rush_shares(g,_top_n_shares(g.rush_share.to_numpy(),5))
        T=_allocate_counts(rng,pa,tsh); C=_allocate_counts(rng,ra,rsh)
        pes=np.clip(rng.normal(1.0,0.09,ITER),0.65,1.35)
        res=np.clip(rng.normal(1.0,0.10,ITER),0.60,1.40)
        for j in range(len(g)):
            ypt=g.ypt.iloc[j]; ypt=7.5 if not np.isfinite(ypt) or ypt<=0 else ypt
            ypc=g.ypc.iloc[j]; ypc=4.2 if not np.isfinite(ypc) or ypc<=0 else ypc
            t=T[:,j]; c=C[:,j]
            rsd=np.maximum(6.0,np.sqrt(np.maximum(t,1))*ypt*0.55)
            csd=np.maximum(3.0,np.sqrt(np.maximum(c,1))*ypc*0.65)
            ry=np.clip(rng.normal(t*ypt*pes,rsd),0.0,None)
            cy=np.clip(rng.normal(c*ypc*res,csd),0.0,None)
            if d3: ry=np.where(t>0,ry,0.0); cy=np.where(c>0,cy,0.0)
            out.append({"player":g.player_display_name.iloc[j] if "player_display_name" in g
                        else g.k.iloc[j],"k":g.k.iloc[j],"team":tm,"opponent":g.opponent.iloc[j],
                        "position":g.position.iloc[j],"targets":float(t.mean()),
                        "carries":float(c.mean()),"rec_yards":float(ry.mean()),
                        "rush_yards":float(cy.mean())})
    return pd.DataFrame(out)


def main():
    u=build()
    print(f"W9 universe: {len(u)} players across {u.team.nunique()} teams")
    base=project(u,False); d3=project(u,True)
    m=base.merge(d3[["k","rec_yards","rush_yards"]],on="k",suffixes=("_base","_d3"))
    m["week"]=TARGET_WEEK; m["season"]=2026
    m["captured_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
    m.to_csv("data/market_track_record/w9_prospective_frozen_v1.csv",index=False)
    meta={"season":2026,"week":9,"captured_utc":m.captured_utc.iloc[0],
          "players":int(len(m)),"teams":int(m.team.nunique()),
          "max_player_stats_week_read":int(PRIOR),
          "w9_injury_designations_available_at_capture":False,
          "arms_frozen":["baseline","D3"],"arms_pending":["D1 overlay"],
          "totals":{"rec_base":float(m.rec_yards_base.sum()),
                    "rec_d3":float(m.rec_yards_d3.sum()),
                    "rush_base":float(m.rush_yards_base.sum()),
                    "rush_d3":float(m.rush_yards_d3.sum())}}
    open("data/market_track_record/w9_prospective_frozen_v1.json","w").write(json.dumps(meta,indent=2))
    print(json.dumps(meta["totals"],indent=2))
    print(f"captured_utc = {meta['captured_utc']}")
    print(f"\nfrozen -> data/market_track_record/w9_prospective_frozen_v1.csv ({len(m)} rows)")


if __name__=="__main__":
    main()
