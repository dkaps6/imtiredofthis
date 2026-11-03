"""Joint D1 x D2 x D3 factorial, scored on yards.
Preregistration: docs/research/PRISM_JOINT_D1_D2_D3_PREREGISTRATION_V1.md (frozen first).
Prerequisite: PRISM_ALLOCATOR_CAP_SCALE_INVARIANCE_V1.md. Uses the production allocator
unmodified. No sportsbook variable enters any arm. No production code modified.
"""
from __future__ import annotations
import re, sys, itertools
import numpy as np, pandas as pd, nflreadpy as nfl
sys.path.insert(0,".")
from scripts.simulation_v2 import _allocate_counts, _team_inputs, \
    _sharpen_wr_target_shares, _sharpen_rb_rush_shares, _top_n_shares

U={7:"data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv",
   8:"data/market_track_record/explanatory/2026_wk08_trace__data__football_simulation_universe.csv"}
ITER=4000
Q_T, Q_R = 0.8535, 0.5718          # D1 QUESTIONABLE retention, fitted on W1-8 injuries
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def _n(row,*names,default=np.nan):
    for c in names:
        if c in row.index:
            v=pd.to_numeric(row[c],errors="coerce")
            if np.isfinite(v): return float(v)
    return float(default)


def injuries():
    d=nfl.load_injuries(seasons=[2026]); i=d.to_pandas() if hasattr(d,"to_pandas") else d
    i=i.copy(); i["k"]=i.full_name.map(norm); i["wk"]=pd.to_numeric(i.week,errors="coerce")
    i["st"]=i.report_status.astype(str).str.upper().str.strip()
    i=i[i.st.isin(["OUT","DOUBTFUL","QUESTIONABLE"])]
    return i.drop_duplicates(["k","wk"]).set_index(["k","wk"]).st.to_dict()


def team_arm(rng, g, d1, d2, d3, inj):
    pm,pr=_team_inputs(g); plays=int(round(pm))
    pass_att=rng.binomial(plays, pr, ITER); rush_tot=plays-pass_att
    t=np.array([_n(r,"rules_tgt_share","bayes_tgt_share","target_share","tgt_share",default=0.0)
                for _,r in g.iterrows()])
    rs=np.array([_n(r,"rules_rush_share","bayes_rush_share","rush_share",default=0.0)
                 for _,r in g.iterrows()])
    if d2:   # delete the matchup target multipliers -> revert to the Bayesian share
        t=np.array([_n(r,"bayes_tgt_share","rules_tgt_share",default=0.0) for _,r in g.iterrows()])
    if d1:   # zero inactives, calibrated retention for QUESTIONABLE
        for j,(_,r) in enumerate(g.iterrows()):
            st=inj.get((norm(r.get("player")), int(r.get("wk"))),"")
            if st in ("OUT","DOUBTFUL"): t[j]=0.0; rs[j]=0.0
            elif st=="QUESTIONABLE":     t[j]*=Q_T; rs[j]*=Q_R
    tsh=_sharpen_wr_target_shares(g, t)
    rsh=_sharpen_rb_rush_shares(g, _top_n_shares(rs,5))
    targets=_allocate_counts(rng, pass_att, tsh)
    carries=_allocate_counts(rng, rush_tot, rsh)
    pes=np.clip(rng.normal(1.0,0.09,ITER),0.65,1.35)
    res=np.clip(rng.normal(1.0,0.10,ITER),0.60,1.40)
    out=[]
    for j,(_,row) in enumerate(g.iterrows()):
        vol=float(np.clip(_n(row,"rules_volatility_mult",default=1.0),0.75,1.50))
        ypt=_n(row,"rules_ypt","bayes_ypt","ypt"); ypt=7.5 if not np.isfinite(ypt) or ypt<=0 else ypt
        ypc=_n(row,"rules_ypc","bayes_ypc","ypc"); ypc=4.2 if not np.isfinite(ypc) or ypc<=0 else ypc
        tt=targets[:,j]; cc=carries[:,j]
        rsd=np.maximum(6.0,np.sqrt(np.maximum(tt,1))*ypt*0.55)*vol
        csd=np.maximum(3.0,np.sqrt(np.maximum(cc,1))*ypc*0.65)*vol
        ry=np.clip(rng.normal(tt*ypt*pes,rsd),0.0,None)
        cy=np.clip(rng.normal(cc*ypc*res,csd),0.0,None)
        if d3: ry=np.where(tt>0,ry,0.0); cy=np.where(cc>0,cy,0.0)
        out.append({"k":norm(row.get("player")),"team":row.get("team"),"wk":int(row.get("wk")),
                    "pos":str(row.get("position","")).upper(),
                    "rec_yards":float(ry.mean()),"rush_yards":float(cy.mean()),
                    "targets":float(tt.mean()),"carries":float(cc.mean())})
    return out


def main():
    u=pd.concat([pd.read_csv(p,low_memory=False).assign(wk=w) for w,p in U.items()],
                ignore_index=True)
    inj=injuries()
    frames=[]
    for d1,d2,d3 in itertools.product((0,1),repeat=3):
        name=("D1" if d1 else "--")+("D2" if d2 else "--")+("D3" if d3 else "--")
        rows=[]
        for (tm,wk),g in u.groupby(["team","wk"]):
            rng=np.random.default_rng(abs(hash((tm,int(wk))))%(2**32))
            rows+=team_arm(rng,g.reset_index(drop=True),d1,d2,d3,inj)
        d=pd.DataFrame(rows); d["arm"]=name; d["d1"]=d1; d["d2"]=d2; d["d3"]=d3
        frames.append(d)
        print(f"  {name}: {len(d)} rows  team targets {d.targets.sum():8.1f}  "
              f"rec {d.rec_yards.sum():8.0f}  rush {d.rush_yards.sum():8.0f}")
    sim=pd.concat(frames,ignore_index=True)

    d=nfl.load_player_stats(seasons=[2026],summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps["k"]=ps.player_display_name.map(norm); ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    for c in ("receiving_yards","rushing_yards"):
        ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)
    act=ps.groupby(["k","wk"]).agg(a_rec=("receiving_yards","sum"),
                                   a_rush=("rushing_yards","sum")).reset_index()
    sim=sim.merge(act,on=["k","wk"],how="left")
    sim["a_rec"]=sim.a_rec.fillna(0.0); sim["a_rush"]=sim.a_rush.fillna(0.0)
    sim.to_csv("data/research/joint_d1d2d3_v1.csv",index=False)
    print(f"\nwrote data/research/joint_d1d2d3_v1.csv  ({len(sim)} rows, 8 arms)")


if __name__=="__main__":
    main()
