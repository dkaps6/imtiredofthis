"""D3: phantom yards at zero opportunity. Arms C0/A1/A2/A3.
Preregistration: docs/research/PRISM_D3_PHANTOM_YARDS_PREREGISTRATION_V1.md (frozen first).
Reproduces the simulation_v2 yards block faithfully, driven by the committed production
universes. D1 and D2 are OFF (C0 uses the shipped rules_* values). No production code modified.
"""
from __future__ import annotations
import re, sys
import numpy as np, pandas as pd, nflreadpy as nfl
sys.path.insert(0,".")
from scripts.simulation_v2 import _allocate_counts, _team_inputs, _sharpen_wr_target_shares, \
    _sharpen_rb_rush_shares, _top_n_shares

U={7:"data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv",
   8:"data/market_track_record/explanatory/2026_wk08_trace__data__football_simulation_universe.csv"}
ARMS=["C0","A1","A2","A3"]
ITER=4000
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())
def _n(row,*names,default=np.nan):
    for c in names:
        if c in row.index:
            v=pd.to_numeric(row[c],errors="coerce")
            if np.isfinite(v): return float(v)
    return float(default)


def gamma_draw(rng, mu, sd, n):
    """Mean exactly mu, sd exactly sd, support >= 0 -- no clip bias."""
    mu=np.maximum(mu,1e-9); sd=np.maximum(sd,1e-9)
    shape=(mu/sd)**2; scale=(sd**2)/mu
    return rng.gamma(shape, scale, n)


def run_team(rng, team_df, arm):
    plays_mean, pass_rate = _team_inputs(team_df)
    plays=rng.binomial(int(round(plays_mean)), 1.0, ITER)*0+int(round(plays_mean))
    pass_att=rng.binomial(plays, pass_rate)
    rush_att_total=plays-pass_att
    raw_t=np.array([_n(r,"rules_tgt_share","bayes_tgt_share","target_share","tgt_share",default=0.0)
                    for _,r in team_df.iterrows()])
    raw_r=np.array([_n(r,"rules_rush_share","bayes_rush_share","rush_share",default=0.0)
                    for _,r in team_df.iterrows()])
    tshare=_sharpen_wr_target_shares(team_df, raw_t)
    rshare=_sharpen_rb_rush_shares(team_df, _top_n_shares(raw_r,5))
    targets=_allocate_counts(rng, pass_att, tshare)
    carries=_allocate_counts(rng, rush_att_total, rshare)
    pes=np.clip(rng.normal(1.0,0.09,ITER),0.65,1.35)
    res=np.clip(rng.normal(1.0,0.10,ITER),0.60,1.40)
    out=[]
    for j,(_,row) in enumerate(team_df.iterrows()):
        vol=float(np.clip(_n(row,"rules_volatility_mult",default=1.0),0.75,1.50))
        ypt=_n(row,"rules_ypt","bayes_ypt","ypt"); ypt=7.5 if not np.isfinite(ypt) or ypt<=0 else ypt
        ypc=_n(row,"rules_ypc","bayes_ypc","ypc"); ypc=4.2 if not np.isfinite(ypc) or ypc<=0 else ypc
        t=targets[:,j]; c=carries[:,j]
        rmu=t*ypt*pes; cmu=c*ypc*res
        if arm=="A3":
            rsd=np.sqrt(t)*ypt*0.55*vol; csd=np.sqrt(c)*ypc*0.65*vol
        else:
            rsd=np.maximum(6.0,np.sqrt(np.maximum(t,1))*ypt*0.55)*vol
            csd=np.maximum(3.0,np.sqrt(np.maximum(c,1))*ypc*0.65)*vol
        if arm=="A2":
            ry=np.where(t>0, gamma_draw(rng,rmu,np.maximum(rsd,1e-9),ITER), 0.0)
            cy=np.where(c>0, gamma_draw(rng,cmu,np.maximum(csd,1e-9),ITER), 0.0)
        else:
            ry=np.clip(rng.normal(rmu,np.maximum(rsd,1e-9)),0.0,None)
            cy=np.clip(rng.normal(cmu,np.maximum(csd,1e-9)),0.0,None)
            if arm in ("A1","A3"):
                ry=np.where(t>0,ry,0.0); cy=np.where(c>0,cy,0.0)
        out.append({"k":norm(row.get("player")),"team":row.get("team"),
                    "pos":str(row.get("position","")).upper(),
                    "rec_yards":float(ry.mean()),"rush_yards":float(cy.mean()),
                    "p_zero_t":float((t==0).mean()),"p_zero_c":float((c==0).mean()),
                    "mean_t":float(t.mean()),"mean_c":float(c.mean())})
    return out


def main():
    u=pd.concat([pd.read_csv(p,low_memory=False).assign(wk=w) for w,p in U.items()],
                ignore_index=True)
    frames=[]
    for arm in ARMS:
        rows=[]
        for (tm,wk),g in u.groupby(["team","wk"]):
            rng=np.random.default_rng(abs(hash((tm,int(wk))))%(2**32))
            for r in run_team(rng,g.reset_index(drop=True),arm):
                r["wk"]=wk; rows.append(r)
        d=pd.DataFrame(rows); d["arm"]=arm; frames.append(d)
        print(f"  simulated arm {arm}: {len(d)} player-weeks")
    sim=pd.concat(frames,ignore_index=True)

    # ---- M1 mechanical: zero opportunity -> zero yards ----
    print("\n[M1] applies to A1,A2,A3 -- zero-opportunity rows must give exactly 0 yards")
    for arm in ARMS:
        s=sim[(sim.arm==arm)&(sim.p_zero_t>=0.999)]
        v=float(s.rec_yards.max()) if len(s) else np.nan
        print(f"   {arm}: rows always-zero-targets n={len(s):3d}  max rec_yards {v if np.isfinite(v) else float('nan'):.3e}"
              + ("  <- C0 leaks phantom yards" if arm=="C0" and np.isfinite(v) and v>1e-12 else
                 ("  PASS" if arm!="C0" and (not np.isfinite(v) or v<=1e-12) else "")))

    # ---- M4 faithfulness ----
    print("\n[M4] applies to C0 -- faithfulness of the reproduction vs shipped model_proj")
    import glob
    bo=[]
    for f in sorted(glob.glob("data/market_track_record/boards/*.csv")):
        wk=int(re.search(r"wk(\d+)",f).group(1))
        if wk not in U: continue
        d=pd.read_csv(f,low_memory=False); d["wk"]=wk; bo.append(d)
    b=pd.concat(bo,ignore_index=True)
    b=b[b.market=="rec_yards"].copy(); b["k"]=b.player.map(norm)
    b["model_proj"]=pd.to_numeric(b.model_proj,errors="coerce")
    j=b.merge(sim[sim.arm=="C0"][["k","wk","rec_yards"]],on=["k","wk"],how="inner").dropna(
        subset=["model_proj","rec_yards"])
    r=float(j.rec_yards.corr(j.model_proj))
    print(f"   n={len(j)}  r={r:+.3f}  mine {j.rec_yards.mean():.2f} vs shipped {j.model_proj.mean():.2f}"
          f"  ratio {j.rec_yards.mean()/j.model_proj.mean():.3f}")
    print(f"   {'PASS' if r>=0.85 else 'FAIL -> ANALYTIC-ONLY (K1)'}")

    # ---- realized yards ----
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
    sim.to_csv("data/research/d3_phantom_yards_v1.csv",index=False)

    # ---- S1 ----
    rng=np.random.default_rng(77)
    for mk,proj,actual in (("rec_yards","rec_yards","a_rec"),("rush_yards","rush_yards","a_rush")):
        pv=sim.pivot_table(index="k",columns="arm",values=[proj,actual],aggfunc="mean")
        bias={a:(pv[(proj,a)]-pv[(actual,a)]) for a in ARMS}
        idx=np.array(sorted(pv.dropna().index)); n=len(idx)
        dr=rng.integers(0,n,(4000,n)); base=np.abs(bias["C0"].reindex(idx).to_numpy())
        print(f"\n[S1] {mk}  per-player bias (n={n} players)")
        print(f"   {'arm':4s} {'mean|bias|':>11s} {'mean bias':>10s} {'gain':>10s} {'95% CI':>20s} {'S1':>5s}")
        for a in ARMS:
            v=np.abs(bias[a].reindex(idx).to_numpy()); gn=base-v
            gb=gn[dr].mean(1); lo,hi=np.percentile(gb,[2.5,97.5])
            print(f"   {a:4s} {v.mean():11.3f} {bias[a].reindex(idx).mean():+10.3f} "
                  f"{gn.mean():+10.3f} [{lo:+8.3f},{hi:+8.3f}] "
                  f"{'' if a=='C0' else ('PASS' if lo>0 else 'fail'):>5s}")

    # ---- S2 tiers, BOTTOM FIRST ----
    print("\n[S2] rec_yards tiers from C0 projection (pregame). BOTTOM TIER FIRST.")
    c0=sim[sim.arm=="C0"].set_index(["k","wk"]).rec_yards
    sim["tier"]=pd.qcut(sim.set_index(["k","wk"]).index.map(c0),4,labels=[0,1,2,3],duplicates="drop")
    for t,lab in [(0,"Q1 low"),(1,"Q2"),(2,"Q3"),(3,"Q4 TOP")]:
        s=sim[sim.tier==t]
        if s.empty: continue
        line=f"   {lab:8s} n={len(s)//len(ARMS):4d} actual {s[s.arm=='C0'].a_rec.mean():6.2f}  "
        for a in ARMS:
            x=s[s.arm==a]
            line+=f"{a} {x.rec_yards.mean()-x.a_rec.mean():+7.2f}  "
        print(line)
    print("\n[S3] total projected receiving yards vs realized")
    for a in ARMS:
        s=sim[sim.arm==a]
        print(f"   {a}: projected {s.rec_yards.sum():9.0f}   realized {s.a_rec.sum():9.0f}"
              f"   ratio {s.rec_yards.sum()/max(s.a_rec.sum(),1):.3f}")


if __name__=="__main__":
    main()
