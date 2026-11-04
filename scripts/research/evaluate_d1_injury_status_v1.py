"""D1: injury status never reaches the model. Arms C0/A1/A2/A3.
Preregistration: docs/research/PRISM_D1_INJURY_STATUS_PREREGISTRATION_V1.md (frozen first).
Part 1 calibrates retention by designation; Part 2 applies arms to the production universes.
No sportsbook variable is read anywhere in this study. D2 is OFF (C0 uses rules_tgt_share as
shipped, which already contains D2; arms modify only designated players, so D2 is held constant).
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd, nflreadpy as nfl

U={7:"data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv",
   8:"data/market_track_record/explanatory/2026_wk08_trace__data__football_simulation_universe.csv"}
SKILL={"WR","RB","TE","QB","FB"}
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())


def load_all():
    d=nfl.load_player_stats(seasons=[2026],summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    for c in ("targets","carries"): ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)
    i=nfl.load_injuries(seasons=[2026])
    inj=i.to_pandas() if hasattr(i,"to_pandas") else i
    inj=inj.copy(); inj["k"]=inj.full_name.map(norm)
    inj["wk"]=pd.to_numeric(inj.week,errors="coerce")
    inj["st"]=inj.report_status.astype(str).str.upper().str.strip()
    inj["pos"]=inj.position.astype(str).str.upper()
    inj=inj[inj.st.isin(["OUT","DOUBTFUL","QUESTIONABLE"])]
    return ps, inj[["k","wk","team","pos","st"]].drop_duplicates(["k","wk"])


def part1(ps, inj):
    """Empirical opportunity retention by designation. Baseline = PRIOR weeks only (G5)."""
    tt=ps.groupby(["team","wk"]).targets.sum().rename("tt").reset_index()
    tc=ps.groupby(["team","wk"]).carries.sum().rename("tc").reset_index()
    x=ps.merge(tt,on=["team","wk"]).merge(tc,on=["team","wk"])
    x["ts"]=np.where(x.tt>0,x.targets/x.tt,np.nan)
    x["rs"]=np.where(x.tc>0,x.carries/x.tc,np.nan)
    rows=[]
    for wk in sorted(x.wk.dropna().unique()):
        if wk<2: continue
        prior=x[x.wk<wk].groupby("k")[["ts","rs"]].mean().rename(columns={"ts":"b_ts","rs":"b_rs"})
        cur=x[x.wk==wk].groupby("k")[["ts","rs"]].mean()
        d=inj[inj.wk==wk].drop_duplicates("k").set_index("k")
        for k in d.index:
            if k not in prior.index: continue
            rows.append({"k":k,"wk":wk,"st":d.at[k,"st"],"pos":d.at[k,"pos"],
                         "b_ts":prior.at[k,"b_ts"],"b_rs":prior.at[k,"b_rs"],
                         "c_ts":cur.at[k,"ts"] if k in cur.index else 0.0,
                         "c_rs":cur.at[k,"rs"] if k in cur.index else 0.0,
                         "played":int(k in cur.index)})
    r=pd.DataFrame(rows)
    r=r[r.pos.isin(SKILL)]
    print("="*86); print("PART 1 -- EMPIRICAL OPPORTUNITY RETENTION BY DESIGNATION (skill positions)")
    print("  baseline = prior weeks only. retention = realized share / prior-weeks share.")
    print("="*86)
    print(f"  {'designation':14s} {'n':>5s} {'% played':>9s} {'tgt retention':>14s} {'rush retention':>15s}")
    ret={}
    for st in ("OUT","DOUBTFUL","QUESTIONABLE"):
        s=r[r.st==st]
        if s.empty: continue
        a=s[(s.b_ts>0)]; b=s[(s.b_rs>0)]
        rt=float((a.c_ts/a.b_ts).mean()) if len(a) else np.nan
        rr=float((b.c_rs/b.b_rs).mean()) if len(b) else np.nan
        ret[st]=(rt,rr)
        print(f"  {st:14s} {len(s):5d} {s.played.mean()*100:8.1f}% {rt:14.4f} {rr:15.4f}"
              f"   (n_tgt={len(a)}, n_rush={len(b)})")
    print(f"\n  production's IMPLIED retention: OUT 1.00, DOUBTFUL 1.00, QUESTIONABLE 1.00")
    print(f"  the INTENDED rule's retention:   OUT 0.50, DOUBTFUL 0.50, QUESTIONABLE 1.00")
    return ret, r


def part2(ps, inj, ret):
    u=pd.concat([pd.read_csv(p,low_memory=False).assign(wk=w) for w,p in U.items()],
                ignore_index=True)
    u["k"]=u.player.map(norm); u["pos"]=u.position.astype(str).str.upper()
    u["C0"]=pd.to_numeric(u.rules_tgt_share,errors="coerce")
    u=u.dropna(subset=["C0"]).copy()
    u["key"]=u.team.astype(str)+"_"+u.wk.astype(str)
    u=u.merge(inj[["k","wk","st"]],on=["k","wk"],how="left")
    u["st"]=u.st.fillna("")
    A1={"OUT":0.50,"DOUBTFUL":0.50,"QUESTIONABLE":1.00,"":1.00}
    A2={"OUT":0.00,"DOUBTFUL":float(ret.get("DOUBTFUL",(0.0,0))[0] or 0.0),
        "QUESTIONABLE":float(ret.get("QUESTIONABLE",(1.0,0))[0] or 1.0),"":1.00}
    print(f"\n  A2 retention actually used: {{k:round(v,4) for k,v in A2.items()}}".replace("{k:round(v,4) for k,v in A2.items()}",str({k:round(v,4) for k,v in A2.items()})))
    u["A1"]=u.C0*u.st.map(A1)
    u["A2"]=u.C0*u.st.map(A2)
    g=u.groupby("key")
    s0=g.C0.transform("sum"); s2=g.A2.transform("sum")
    u["A3"]=np.where(s2>0,u.A2*(s0/s2),u.A2)      # mass-preserving redistribution
    # realized share
    wk=ps[ps.wk.isin(U)]
    tt=wk.groupby(["team","wk"]).targets.sum().rename("tt").reset_index()
    pt=wk.groupby(["k","team","wk"]).targets.sum().rename("pt").reset_index()
    u=u.merge(pt,on=["k","team","wk"],how="left").merge(tt,on=["team","wk"],how="left")
    u["pt"]=u.pt.fillna(0.0)
    u["actual"]=np.where(u.tt>0,u.pt/u.tt,np.nan)
    u=u.dropna(subset=["actual"])
    ARMS=["C0","A1","A2","A3"]
    des=u.st.ne("")
    print("\n"+"="*86); print(f"PART 2 -- ARMS ON THE PRODUCTION UNIVERSES  (rows={len(u)}, designated={des.sum()})")
    print("="*86)
    g=u.groupby("key")
    e1=float((g.A3.sum()-g.C0.sum()).abs().max())
    e2=float(max((u.loc[~des,a]-u.loc[~des,"C0"]).abs().max() for a in ARMS))
    print(f"  [G1] A3 preserves team mass : max|diff| {e1:.3e}  {'PASS' if e1<=1e-12 else 'FAIL K1'}")
    print(f"  [G2] undesignated untouched : max|diff| {e2:.3e}  {'PASS' if e2<=1e-12 else 'FAIL K1'}")
    print(f"\n  team sums: " + "  ".join(f"{a} {g[a].sum().mean():.4f}" for a in ARMS)
          + f"   realized {g.actual.sum().mean():.4f}")
    print("\n  [G3] per-player bias vs realized target share")
    pv=u.pivot_table(index="k",values=ARMS+["actual"],aggfunc="mean")
    bias={a:(pv[a]-pv["actual"]) for a in ARMS}
    idx=np.array(sorted(pv.dropna().index)); n=len(idx)
    rng=np.random.default_rng(21); dr=rng.integers(0,n,(4000,n))
    base=np.abs(bias["C0"].reindex(idx).to_numpy())
    print(f"    {'arm':4s} {'mean|bias|':>11s} {'gain vs C0':>11s} {'95% CI':>22s} {'G3':>5s}")
    for a in ARMS:
        v=np.abs(bias[a].reindex(idx).to_numpy()); gn=base-v
        gb=gn[dr].mean(1); lo,hi=np.percentile(gb,[2.5,97.5])
        print(f"    {a:4s} {v.mean():11.5f} {gn.mean():+11.5f} [{lo:+9.5f},{hi:+9.5f}] "
              f"{'' if a=='C0' else ('PASS' if lo>0 else 'fail'):>5s}")
    print("\n  on the DESIGNATED players only (where the arms act):")
    for a in ARMS:
        s=u[des]
        print(f"    {a:4s} n={len(s):3d}  mean share given {s[a].mean():.4f}  "
              f"mean |bias| {(s[a]-s.actual).abs().mean():.4f}")
    print("\n  [G4] tiers from C0 (pregame), TOP FIRST")
    q=pd.qcut(u.C0,4,labels=[0,1,2,3],duplicates="drop"); u2=u.assign(q=q)
    for t,lab in [(3,"Q4 TOP"),(2,"Q3"),(1,"Q2"),(0,"Q1 low")]:
        s=u2[u2.q==t]
        if s.empty: continue
        b0=(s.C0-s.actual).abs().mean()
        print(f"    {lab:8s} n={len(s):4d} actual {s.actual.mean():.4f}  " +
              "  ".join(f"{a} {b0-(s[a]-s.actual).abs().mean():+.4f}" for a in ARMS[1:]))
    u.to_csv("data/research/d1_injury_status_v1.csv",index=False)


def main():
    ps,inj=load_all()
    ret,_=part1(ps,inj)
    part2(ps,inj,ret)


if __name__=="__main__":
    main()
