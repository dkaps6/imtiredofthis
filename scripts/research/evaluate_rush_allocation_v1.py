"""Rush carry allocation: arms C0/A1/A2/A3/A4.
Preregistration: docs/research/PRISM_RUSH_ALLOCATION_PREREGISTRATION_V1.md (frozen first).
D3 ON, D1 ON, D2 OFF (joint-study settings). Calibration fitted on W1-W6 only (M4).
No sportsbook variable. No production code modified.
"""
from __future__ import annotations
import re, sys
import numpy as np, pandas as pd, nflreadpy as nfl
sys.path.insert(0,".")
from scripts.simulation_v2 import _allocate_counts, _team_inputs, \
    _sharpen_wr_target_shares, _sharpen_rb_rush_shares

U={7:"data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv",
   8:"data/market_track_record/explanatory/2026_wk08_trace__data__football_simulation_universe.csv"}
ITER=4000; ARMS=["C0","A1","A2","A3","A4"]
Q_T,Q_R=0.8535,0.5718
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def _n(row,*names,default=np.nan):
    for c in names:
        if c in row.index:
            v=pd.to_numeric(row[c],errors="coerce")
            if np.isfinite(v): return float(v)
    return float(default)
def grp(p):
    p=str(p).upper().strip()
    if p=="FB": return "FB"
    if p in ("RB","HB"): return "RB"
    return p if p in ("QB","WR","TE") else "OTHER"


def top_n(shares, n):
    c=np.nan_to_num(np.asarray(shares,float),nan=0.0,posinf=0.0,neginf=0.0)
    c=np.clip(c,0.0,0.95)
    if len(c)<=n: return c
    o=np.argsort(-c,kind="stable"); keep=o[:n]; out=np.zeros_like(c); out[keep]=c[keep]
    return out


def calibrate(ps):
    """Positional carry-share factors from W1-W6 ONLY (M4). Model side from the W7/W8
    universes' own positional share mix, which is a stable property of the priors."""
    w=ps[(ps.wk>=1)&(ps.wk<=6)]
    assert not w.wk.isin([7,8]).any(), "M4 VIOLATION: calibration touched W7/W8"
    real=w.groupby(w.position.map(grp)).carries.sum()
    real=real/real.sum()
    u=pd.concat([pd.read_csv(p,low_memory=False) for p in U.values()],ignore_index=True)
    u["g"]=u.position.map(grp)
    u["rs"]=pd.to_numeric(u.rules_rush_share,errors="coerce").fillna(0.0)
    mod=u.groupby("g").rs.sum(); mod=mod/mod.sum()
    f={}
    for g_ in set(real.index)|set(mod.index):
        r=float(real.get(g_,0.0)); m=float(mod.get(g_,0.0))
        f[g_]= (r/m) if m>1e-9 else (3.0 if r>0 else 0.0)
    return f, real, mod


def team_arm(rng,g,arm,inj,fac):
    pm,pr=_team_inputs(g); plays=int(round(pm))
    pass_att=rng.binomial(plays,pr,ITER); rush_tot=plays-pass_att
    t=np.array([_n(r,"rules_tgt_share","bayes_tgt_share","target_share","tgt_share",default=0.0)
                for _,r in g.iterrows()])
    rs=np.array([_n(r,"rules_rush_share","bayes_rush_share","rush_share",default=0.0)
                 for _,r in g.iterrows()])
    pos=[grp(r.get("position")) for _,r in g.iterrows()]
    for j,(_,r) in enumerate(g.iterrows()):   # D1 ON
        st=inj.get((norm(r.get("player")),int(r.get("wk"))),"")
        if st in ("OUT","DOUBTFUL"): t[j]=0.0; rs[j]=0.0
        elif st=="QUESTIONABLE":     t[j]*=Q_T; rs[j]*=Q_R
    if arm in ("A1",):                        # FB out of the RB group
        rs=np.array([v*(0.05 if pos[j]=="FB" else 1.0) for j,v in enumerate(rs)])
    if arm in ("A3","A4"):                    # positional recalibration
        rs=np.array([v*fac.get(pos[j],1.0) for j,v in enumerate(rs)])
    N = 8 if arm in ("A2","A4") else 5
    tsh=_sharpen_wr_target_shares(g,t)
    rsh=_sharpen_rb_rush_shares(g,top_n(rs,N))
    targets=_allocate_counts(rng,pass_att,tsh)
    carries=_allocate_counts(rng,rush_tot,rsh)
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
        ry=np.where(tt>0,np.clip(rng.normal(tt*ypt*pes,rsd),0.0,None),0.0)   # D3 ON
        cy=np.where(cc>0,np.clip(rng.normal(cc*ypc*res,csd),0.0,None),0.0)
        out.append({"k":norm(row.get("player")),"team":row.get("team"),"wk":int(row.get("wk")),
                    "g":pos[j],"rec_yards":float(ry.mean()),"rush_yards":float(cy.mean()),
                    "targets":float(tt.mean()),"carries":float(cc.mean()),
                    "rush_att_tot":float(rush_tot.mean())})
    return out


def main():
    d=nfl.load_injuries(seasons=[2026]); i=d.to_pandas() if hasattr(d,"to_pandas") else d
    i=i.copy(); i["k"]=i.full_name.map(norm); i["wk"]=pd.to_numeric(i.week,errors="coerce")
    i["st"]=i.report_status.astype(str).str.upper().str.strip()
    inj=i[i.st.isin(["OUT","DOUBTFUL","QUESTIONABLE"])].drop_duplicates(
        ["k","wk"]).set_index(["k","wk"]).st.to_dict()
    p=nfl.load_player_stats(seasons=[2026],summary_level="week")
    ps=p.to_pandas() if hasattr(p,"to_pandas") else p
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    for c in ("carries","rushing_yards","receiving_yards"):
        ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)
    fac,real,mod=calibrate(ps)
    print("[M4] calibration fitted on W1-W6 only -> PASS (asserted)")
    print(f"  {'group':6s} {'realized W1-6':>14s} {'model mix':>10s} {'factor':>8s}")
    for g_ in sorted(set(real.index)|set(mod.index)):
        print(f"  {g_:6s} {float(real.get(g_,0)):14.4f} {float(mod.get(g_,0)):10.4f} {fac[g_]:8.3f}")

    u=pd.concat([pd.read_csv(p_,low_memory=False).assign(wk=w) for w,p_ in U.items()],
                ignore_index=True)
    frames=[]
    for arm in ARMS:
        rows=[]
        for (tm,wk),g in u.groupby(["team","wk"]):
            rng=np.random.default_rng(abs(hash((tm,int(wk))))%(2**32))
            rows+=team_arm(rng,g.reset_index(drop=True),arm,inj,fac)
        f=pd.DataFrame(rows); f["arm"]=arm; frames.append(f)
        print(f"  {arm}: carries {f.carries.sum():8.1f}  rush_yards {f.rush_yards.sum():8.0f}"
              f"  rec_yards {f.rec_yards.sum():8.0f}")
    sim=pd.concat(frames,ignore_index=True)
    act=ps.groupby(["k","wk"]).agg(ac=("carries","sum"),ay=("rushing_yards","sum"),
                                   arec=("receiving_yards","sum")).reset_index()
    sim=sim.merge(act,on=["k","wk"],how="left").fillna({"ac":0,"ay":0,"arec":0})
    sim.to_csv("data/research/rush_allocation_v1.csv",index=False)
    print(f"\nwrote data/research/rush_allocation_v1.csv ({len(sim)} rows)")


if __name__=="__main__":
    main()
