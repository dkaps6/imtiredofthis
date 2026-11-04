"""rec_yards football-only residual synthesis, arms A1/A2/A3.
Preregistration: docs/research/PRISM_REC_YARDS_SYNTHESIS_PREREGISTRATION_V1.md (frozen first).
Architecture mirrors model/qb_pass_synthesis_v1.json: ridge on (actual - ensemble_proj),
standardized, capped. NO sportsbook variable enters any feature (M1, asserted).
"""
from __future__ import annotations
import re, glob, json
import numpy as np, pandas as pd, nflreadpy as nfl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHA=20.0; RESIDUAL_CAP=30.0
FIT_WEEKS=[1,2,6]; VAL_WEEKS=[7,8]
FORBIDDEN={"vegas_line","vegas_over_odds","vegas_under_odds","vegas_odds","fair_odds",
           "fair_prob","market_prob","edge_pct","edge_abs","side","implied_prob",
           "best_market_p","model_edge"}
A1=["mc_proj","ml_proj","state_proj","ensemble_proj","model_sd",
    "ensemble_weight_mc","ensemble_weight_ml","ensemble_weight_state"]
A2=A1+["pred_targets","pred_ypt","catch_rate","tgt_share","team_plays","team_pass_rate"]
A3=A2+["prior_tgt_share","prior_ypt","prior_games","is_wr1","is_wr15",
       "def_rec_yards_allowed","def_pass_rate_faced"]
ARMS={"A1":A1,"A2":A2,"A3":A3}
def norm(s): return re.sub(r"[^a-z]","",str(s).lower())


def build():
    rows=[]
    for f in sorted(glob.glob("data/market_track_record/boards/*.csv")):
        wk=int(re.search(r"wk(\d+)",f).group(1))
        d=pd.read_csv(f,low_memory=False); d["wk"]=wk; rows.append(d)
    b=pd.concat(rows,ignore_index=True)
    b=b[b.market=="rec_yards"].copy(); b["k"]=b.player.map(norm)
    for c in A1+["vegas_line","vegas_over_odds","vegas_under_odds","model_proj"]:
        if c in b.columns: b[c]=pd.to_numeric(b[c],errors="coerce")
    b["is_wr1"]=(b.rules_role.astype(str)=="WR1").astype(float)
    b["is_wr15"]=(b.rules_role.astype(str)=="WR1_5").astype(float)

    d=nfl.load_player_stats(seasons=[2025,2026],summary_level="week")
    ps=d.to_pandas() if hasattr(d,"to_pandas") else d
    if "season_type" in ps.columns: ps=ps[ps.season_type=="REG"]
    ps=ps.copy(); ps["k"]=ps.player_display_name.map(norm)
    ps["wk"]=pd.to_numeric(ps.week,errors="coerce"); ps["season"]=pd.to_numeric(ps.season,errors="coerce")
    for c in ("targets","receptions","receiving_yards","carries","attempts"):
        ps[c]=pd.to_numeric(ps.get(c),errors="coerce").fillna(0.0)

    # ---- reconstructed features, PRIOR WEEKS ONLY, per target week ----
    feats=[]
    cur=ps[ps.season==2026]
    pri=ps[ps.season==2025]
    pg=pri.groupby("k").agg(prior_games=("wk","nunique"),pt=("targets","sum"),
                            pry=("receiving_yards","sum")).reset_index()
    tt_p=pri.groupby(["team","wk"]).targets.sum().rename("tt").reset_index()
    pp=pri.merge(tt_p,on=["team","wk"]); pp["ts"]=np.where(pp.tt>0,pp.targets/pp.tt,np.nan)
    pg=pg.merge(pp.groupby("k").ts.mean().rename("prior_tgt_share").reset_index(),on="k",how="left")
    pg["prior_ypt"]=np.where(pg.pt>0,pg.pry/pg.pt,np.nan)
    for wk in sorted(set(FIT_WEEKS+VAL_WEEKS)):
        h=cur[cur.wk<wk]
        if h.empty: continue
        tt=h.groupby(["team","wk"]).targets.sum().rename("tt").reset_index()
        x=h.merge(tt,on=["team","wk"]); x["ts"]=np.where(x.tt>0,x.targets/x.tt,np.nan)
        a=x.groupby("k").agg(tgt_share=("ts","mean"),tg=("targets","sum"),
                             rc=("receptions","sum"),ry=("receiving_yards","sum")).reset_index()
        a["pred_ypt"]=np.where(a.tg>0,a.ry/a.tg,np.nan)
        a["catch_rate"]=np.where(a.tg>0,a.rc/a.tg,np.nan)
        tm=h.groupby(["team","wk"]).agg(patt=("attempts","sum"),carr=("carries","sum")).reset_index()
        tm["pl"]=tm["patt"]+tm["carr"]
        tg=tm.groupby("team").agg(team_plays=("pl","mean"),ta=("patt","mean")).reset_index()
        tg["team_pass_rate"]=tg["ta"]/tg["team_plays"]
        a["pred_targets"]=a["tgt_share"]*float(tg["ta"].mean())
        # defense context: receiving yards allowed and pass rate faced, prior weeks only
        dv=h.groupby(["opponent_team" if "opponent_team" in h.columns else "team","wk"]).agg(
            ry=("receiving_yards","sum")).reset_index() if False else None
        a["wk"]=wk; feats.append((wk,a,tg))
    out=[]
    for wk,a,tg in feats:
        s=b[b.wk==wk].merge(a.drop(columns=[c for c in ("wk",) if c in a.columns]),on="k",how="left").merge(tg[["team","team_plays","team_pass_rate"]],
                                                        on="team",how="left")
        out.append(s)
    b=pd.concat(out,ignore_index=True)
    b=b.merge(pg[["k","prior_games","prior_tgt_share","prior_ypt"]],on="k",how="left")

    # defense: opponent receiving yards allowed per game, prior weeks only
    dall=[]
    for wk in sorted(set(FIT_WEEKS+VAL_WEEKS)):
        h=cur[cur.wk<wk]
        if h.empty: continue
        sch=nfl.load_schedules(seasons=[2026]); sc=sch.to_pandas() if hasattr(sch,"to_pandas") else sch
        sc["week"]=pd.to_numeric(sc.week,errors="coerce")
        pair={}
        for _,r in sc[sc.week<wk].iterrows():
            pair.setdefault(r.home_team,[]).append(r.away_team); pair.setdefault(r.away_team,[]).append(r.home_team)
        tw=h.groupby(["team","wk"]).agg(ry=("receiving_yards","sum"),patt=("attempts","sum"),
                                        carr=("carries","sum")).reset_index()
        rec={}
        for _,r in tw.iterrows():
            ops=pair.get(r.team,[])
            for o in ops: rec.setdefault(o,{"ry":[],"pr":[]})
        for t,g in tw.groupby("team"):
            for o in pair.get(t,[]):
                if o in rec:
                    rec[o]["ry"].append(float(g["ry"].mean()))
                    rec[o]["pr"].append(float((g["patt"]/(g["patt"]+g["carr"])).mean()))
        dall.append(pd.DataFrame([{"opponent":o,"wk":wk,
            "def_rec_yards_allowed":np.mean(v["ry"]) if v["ry"] else np.nan,
            "def_pass_rate_faced":np.mean(v["pr"]) if v["pr"] else np.nan} for o,v in rec.items()]))
    if dall:
        b=b.merge(pd.concat(dall,ignore_index=True),on=["opponent","wk"],how="left")
    else:
        b["def_rec_yards_allowed"]=np.nan; b["def_pass_rate_faced"]=np.nan

    act=cur.groupby(["k","wk"]).receiving_yards.sum().rename("actual").reset_index()
    b=b.merge(act,on=["k","wk"],how="left")
    b["actual"]=b.actual.fillna(0.0)
    b["residual"]=b.actual-b.ensemble_proj
    return b.dropna(subset=["ensemble_proj","actual"])


def main():
    b=build()
    # ---- M1: no sportsbook variable in any feature list ----
    allf=set(sum(ARMS.values(),[]))
    bad=allf&FORBIDDEN
    print(f"[M1] forbidden sportsbook features in the matrix: {sorted(bad) if bad else 'NONE'}  "
          f"{'FAIL -> VOID' if bad else 'PASS'}")
    if bad: return 1
    fit=b[b.wk.isin(FIT_WEEKS)].copy(); val=b[b.wk.isin(VAL_WEEKS)].copy()
    print(f"[M2] fit weeks present: {sorted(fit.wk.unique())}  "
          f"{'PASS' if not set(fit.wk)&set(VAL_WEEKS+[9]) else 'FAIL -> VOID'}")
    print(f"     fit rows {len(fit)}   holdout rows {len(val)}   holdout players {val.k.nunique()}")
    res={}
    for name,cols in ARMS.items():
        X=fit[cols].to_numpy(float); y=fit.residual.to_numpy(float)
        med=np.nanmedian(X,axis=0); med=np.where(np.isfinite(med),med,0.0)
        Xi=np.where(np.isfinite(X),X,med)
        sc=StandardScaler().fit(Xi)
        m=Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(Xi),y)
        Xv=val[cols].to_numpy(float); Xv=np.where(np.isfinite(Xv),Xv,med)
        corr=np.clip(m.predict(sc.transform(Xv)),-RESIDUAL_CAP,RESIDUAL_CAP)
        val[f"syn_{name}"]=val.ensemble_proj+corr
        res[name]={"cap_binding":float((np.abs(corr)>=RESIDUAL_CAP-1e-9).mean()),
                   "corr_mean":float(corr.mean()),"corr_sd":float(corr.std())}
        print(f"  {name}: {len(cols):2d} features  correction mean {corr.mean():+6.2f} "
              f"sd {corr.std():5.2f}  cap binding {res[name]['cap_binding']*100:4.1f}%"
              + ("   <-- K6 (>10%)" if res[name]['cap_binding']>0.10 else ""))
    val.to_csv("data/research/rec_yards_synthesis_v1_holdout.csv",index=False)
    json.dump(res,open("data/research/rec_yards_synthesis_v1_fit.json","w"),indent=2)
    print(f"\nwrote holdout frame ({len(val)} rows)")


if __name__=="__main__":
    raise SystemExit(main())
