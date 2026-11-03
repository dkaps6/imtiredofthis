"""D2: matchup-multiplier mass violation. Arms C0/M1/M2/M3/M4.
Preregistration: docs/research/PRISM_D2_MULTIPLIER_MASS_PREREGISTRATION_V1.md (frozen first).
Scores the SHARE RULE in expectation on the committed production universes. No code modified.
"""
from __future__ import annotations
import re, glob
import numpy as np, pandas as pd, nflreadpy as nfl

U={7:"data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv",
   8:"data/market_track_record/explanatory/2026_wk08_trace__data__football_simulation_universe.csv"}
ARMS=["C0","M1","M2","M3","M4"]
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())


def build():
    u=pd.concat([pd.read_csv(p,low_memory=False).assign(wk=w) for w,p in U.items()],
                ignore_index=True)
    u["k"]=u.player.map(norm); u["pos"]=u.position.astype(str).str.upper()
    u["C0"]=pd.to_numeric(u.rules_tgt_share,errors="coerce")
    u["M2"]=pd.to_numeric(u.bayes_tgt_share,errors="coerce")
    u=u.dropna(subset=["C0","M2"]).copy()
    u["key"]=u.team.astype(str)+"_"+u.wk.astype(str)
    g=u.groupby("key")
    sc=g.C0.transform("sum"); sb=g.M2.transform("sum")
    u["M1"]=u.C0*(sb/sc)                 # multipliers ON, mass back to pre-multiplier sum
    u["M3"]=u.C0/sc                      # multipliers ON, sum -> 1.0
    u["M4"]=u.M2/sb                      # multipliers OFF, sum -> 1.0
    u["mult"]=np.where(u.M2>0,u.C0/u.M2,np.nan)

    # realized target share for the same week
    d=nfl.load_player_stats(seasons=[2026],summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce")
    ps["targets"]=pd.to_numeric(ps.targets,errors="coerce").fillna(0.0)
    wk=ps[ps.wk.isin(U)]
    tt=wk.groupby(["team","wk"]).targets.sum().rename("tt").reset_index()
    pt=wk.groupby(["k","team","wk"]).targets.sum().rename("pt").reset_index()
    u=u.merge(pt,on=["k","team","wk"],how="left").merge(tt,on=["team","wk"],how="left")
    u["pt"]=u.pt.fillna(0.0)
    u["actual"]=np.where(u.tt>0,u.pt/u.tt,np.nan)
    return u.dropna(subset=["actual"])


def main():
    u=build()
    print(f"rows={len(u)}  team-weeks={u.key.nunique()}  weeks={sorted(u.wk.unique())}")

    # ---- G1 / G2 mechanical ----
    g=u.groupby("key")
    sb=g.M2.sum(); s1=g.M1.sum(); s3=g.M3.sum(); s4=g.M4.sum(); s0=g.C0.sum()
    e1=float((s1-sb).abs().max()); e3=float((s3-1.0).abs().max()); e4=float((s4-1.0).abs().max())
    print(f"\n[G1] M1 sum == bayes sum : max|diff| {e1:.3e}  {'PASS' if e1<=1e-12 else 'FAIL K1'}")
    print(f"[G1] M3 sum == 1.0       : max|diff| {e3:.3e}  {'PASS' if e3<=1e-12 else 'FAIL K1'}")
    print(f"[G1] M4 sum == 1.0       : max|diff| {e4:.3e}  {'PASS' if e4<=1e-12 else 'FAIL K1'}")
    flat=[k for k,s in u.groupby("key") if np.allclose(s.mult.dropna(),1.0)]
    if flat:
        sub=u[u.key.isin(flat)]
        e2=float((sub.M1-sub.C0).abs().max())
        print(f"[G2] identity on {len(flat)} all-1.0 team-weeks: max|M1-C0| {e2:.3e}  "
              f"{'PASS' if e2<=1e-12 else 'FAIL K1'}")
    else:
        print("[G2] no team-week has all multipliers == 1.0; identity checked analytically "
              "(M1 = C0*(sb/sc) and sb==sc when every mult is 1)")
    print(f"\nteam sums: C0 {s0.mean():.4f}  M1 {s1.mean():.4f}  M2 {sb.mean():.4f}  "
          f"M3 {s3.mean():.4f}  M4 {s4.mean():.4f}   realized {u.groupby('key').actual.sum().mean():.4f}")

    # ---- G3 projection, bootstrapped over players ----
    print("\n[G3] per-player bias vs realized target share (players counted once)")
    pv=u.pivot_table(index="k",columns=[],values=ARMS+["actual"],aggfunc="mean")
    bias={a:(pv[a]-pv["actual"]) for a in ARMS}
    idx=np.array(sorted(pv.dropna().index)); n=len(idx)
    rng=np.random.default_rng(13); dr=rng.integers(0,n,(4000,n))
    base=np.abs(bias["C0"].reindex(idx).to_numpy())
    print(f"  {'arm':4s} {'mean|bias|':>11s} {'gain vs C0':>11s} {'95% CI':>22s} {'breadth':>8s} {'G3':>5s}")
    for a in ARMS:
        v=np.abs(bias[a].reindex(idx).to_numpy()); gn=base-v
        gb=gn[dr].mean(1); lo,hi=np.percentile(gb,[2.5,97.5])
        tag="" if a=="C0" else ("PASS" if lo>0 else "fail")
        print(f"  {a:4s} {v.mean():11.5f} {gn.mean():+11.5f} [{lo:+9.5f},{hi:+9.5f}] "
              f"{(v<base).mean()*100:7.1f}% {tag:>5s}")
    # K4: M1 vs M2
    d=(np.abs(bias["M2"].reindex(idx).to_numpy())-np.abs(bias["M1"].reindex(idx).to_numpy()))
    db=d[dr].mean(1); lo,hi=np.percentile(db,[2.5,97.5])
    print(f"\n[K4] M1 minus M2 gain = {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  "
          f"{'-> multipliers carry NO information (K4 fires)' if lo<0<hi else '-> multipliers informative'}")

    # ---- G4 tiers, TOP FIRST ----
    print("\n[G4] tiers from the C0 estimate (pregame). TOP TIER FIRST, per preregistration.")
    q=pd.qcut(u.C0,4,labels=[0,1,2,3],duplicates="drop")
    u2=u.assign(q=q)
    print(f"  {'tier':9s} {'n':>5s} {'actual':>8s} " + " ".join(f"{a:>9s}" for a in ARMS))
    for t,lab in [(3,"Q4 TOP"),(2,"Q3"),(1,"Q2"),(0,"Q1 low")]:
        s=u2[u2.q==t]
        if s.empty: continue
        print(f"  {lab:9s} {len(s):5d} {s.actual.mean():8.4f} "
              + " ".join(f"{(s[a]-s.actual).mean():+9.4f}" for a in ARMS))
    print(f"\n  per-tier mean|bias| change vs C0:")
    for t,lab in [(3,"Q4 TOP"),(2,"Q3"),(1,"Q2"),(0,"Q1 low")]:
        s=u2[u2.q==t]
        if s.empty: continue
        b0=(s.C0-s.actual).abs().mean()
        row=" ".join(f"{b0-(s[a]-s.actual).abs().mean():+9.4f}" for a in ARMS)
        bad=any((b0-(s[a]-s.actual).abs().mean())<-0.002 for a in ARMS[1:])
        print(f"  {lab:9s} {row}" + ("   <-- a tier is harmed" if bad else ""))

    # ---- per-position, per the named risk ----
    print("\n[named risk] per-position mean bias, which must be reported not pooled")
    print(f"  {'pos':5s} {'n':>5s} {'actual':>8s} " + " ".join(f"{a:>9s}" for a in ARMS))
    for p,s in u.groupby("pos"):
        if len(s)<20: continue
        print(f"  {p:5s} {len(s):5d} {s.actual.mean():8.4f} "
              + " ".join(f"{(s[a]-s.actual).mean():+9.4f}" for a in ARMS))
    u.to_csv("data/research/d2_multiplier_mass_v1.csv",index=False)
    return u


if __name__=="__main__":
    main()
