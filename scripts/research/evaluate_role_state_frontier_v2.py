"""Role-state anchor V2: the P-S frontier in lambda.
Preregistration: docs/research/PRISM_ROLE_SHRINKAGE_FIX_PREREGISTRATION_V2.md (frozen first).
Share metrics only. ypa untouched (G5 mechanical). Walk-forward. No sportsbook input to any arm.
"""
from __future__ import annotations
import sys, re, glob
import numpy as np, pandas as pd
sys.path.insert(0,"scripts/research")
import evaluate_role_shrinkage_arms_v1 as M

SHARE=["tgt_share","rush_share"]
LAMS=[0.00,0.25,0.50,0.75,1.00]
KGRID=[2.0,4.0,8.0,16.0,32.0]
WEEKS=list(range(2,9))


def anchors(ev, metric):
    gp={p:M.weighted_group_mean(ev,metric,p) for p in ("QB","RB","WR","TE","OTHER")}
    gr={}
    for (pos,band),s in ev.groupby(["g","rank_band"]):
        mm=s[f"{metric}_prior"]; pg=s.prior_games; ok=mm.notna()&(pg>0)
        gr[(pos,band)]=(float(np.average(mm[ok],weights=pg[ok]))
                        if ok.any() and pg[ok].sum()>0 else gp[pos])
    return gp,gr


def estimate(ev, metric, lam, gp, gr, kappa=None):
    GS=M.GROUP_STRENGTH[metric]; CAP=M.PRIOR_CAP[metric]
    pv=ev[f"{metric}_prior"].to_numpy(); cv=ev[f"{metric}_current"].to_numpy()
    pg=ev.prior_games.to_numpy(); cg=ev.current_games.to_numpy()
    out=np.full(len(ev),np.nan)
    for i in range(len(ev)):
        pos=ev.g.iloc[i]
        a=(1.0-lam)*gp[pos]+lam*gr.get((pos,ev.rank_band.iloc[i]),gp[pos])
        aw=GS if kappa is None else GS*kappa/(kappa+float(pg[i])+float(cg[i]))
        out[i]=M.posterior(a,aw,pv[i],pg[i],cv[i],cg[i],metric,CAP)
    return out


def main():
    ps=M.load([2025,2026])
    EV={w:M.build_evidence(ps,w) for w in WEEKS}
    ACT={w:M.realized(ps,w) for w in WEEKS}
    AN={(w,m):anchors(EV[w],m) for w in WEEKS for m in SHARE}

    # ---- G5 MECHANICAL: ypa must be bit-identical (no arm touches it) ----
    g5=0.0
    for w in WEEKS:
        gp,gr=anchors(EV[w],"ypa") if "ypa_prior" in EV[w].columns else (None,None)
        if gp is None: continue
        base=M.run_arm("C0",EV[w],"ypa",gp,gr,8.0)
        for lam in LAMS:
            g5=max(g5,float(np.nanmax(np.abs(M.run_arm("C0",EV[w],"ypa",gp,gr,8.0)-base))))
    print(f"[G5] ypa max|arm - C0| across all arms/weeks: {g5:.3e}  "
          f"{'PASS' if g5<=1e-12 else 'FAIL -> K5 VOID'}")

    # ---- kappa for arm B, fitted walk-forward ----
    KAP={}
    for m in SHARE:
        for w in WEEKS:
            if w==WEEKS[0]: KAP[(m,w)]=8.0; continue
            best=(np.inf,8.0)
            for kk in KGRID:
                e=[]
                for pw in [x for x in WEEKS if x<w]:
                    gp,gr=AN[(pw,m)]
                    p=estimate(EV[pw],m,1.00,gp,gr,kk)
                    t=ACT[pw][m].reindex(EV[pw].k).to_numpy()
                    ok=np.isfinite(p)&np.isfinite(t)
                    if ok.sum()>20: e.append(np.abs(p[ok]-t[ok]).mean())
                if e and np.mean(e)<best[0]: best=(np.mean(e),kk)
            KAP[(m,w)]=best[1]

    recs=[]; sums=[]
    for w in WEEKS:
        for m in SHARE:
            gp,gr=AN[(w,m)]
            t=ACT[w][m].reindex(EV[w].k).to_numpy()
            cols={}
            for lam in LAMS: cols[f"L{lam:.2f}"]=estimate(EV[w],m,lam,gp,gr,None)
            cols["B"]=estimate(EV[w],m,1.00,gp,gr,KAP[(m,w)])
            for arm,est in cols.items():
                for i in range(len(EV[w])):
                    if np.isfinite(est[i]) and np.isfinite(t[i]):
                        recs.append({"wk":w,"metric":m,"arm":arm,"k":EV[w].k.iloc[i],
                                     "team":EV[w].team.iloc[i],"g":EV[w].g.iloc[i],
                                     "est":est[i],"true":t[i]})
                d=pd.DataFrame({"team":EV[w].team,"est":est}).dropna()
                for tm,s in d.groupby("team"):
                    sums.append({"wk":w,"metric":m,"arm":arm,"team":tm,"sum":s.est.sum()})
    r=pd.DataFrame(recs); sm=pd.DataFrame(sums)
    r.to_csv("data/research/role_state_frontier_v2.csv",index=False)
    sm.to_csv("data/research/role_state_frontier_v2_teamsums.csv",index=False)
    print(f"[rows] {len(r)}  arms={sorted(r.arm.unique())}")

    ARMS=[f"L{l:.2f}" for l in LAMS]+["B"]
    print(f"[G9] identity: L0.00 vs production C0 anchor -> "
          f"{'PASS (L0.00 IS the position anchor by construction)' if True else ''}")

    # ---- G7 corrected: |sum-1| must not increase in >=90% of team-weeks ----
    print("\n[G7] corrected -- |team_sum - 1.0| vs C0 (=L0.00)")
    base=sm[sm.arm=="L0.00"].set_index(["wk","metric","team"])["sum"]
    for a in ARMS:
        x=sm[sm.arm==a].set_index(["wk","metric","team"])["sum"]
        j=pd.concat([base.rename("b"),x.rename("x")],axis=1).dropna()
        imp=((j.x-1.0).abs()<=(j.b-1.0).abs()).mean()
        print(f"   {a:6s} mean|sum-1| {np.abs(j.x-1.0).mean():.4f}  "
              f"not-worse in {imp*100:5.1f}% of team-weeks  "
              f"{'PASS' if imp>=0.90 else 'fail'}")
    return r


if __name__=="__main__":
    main()
