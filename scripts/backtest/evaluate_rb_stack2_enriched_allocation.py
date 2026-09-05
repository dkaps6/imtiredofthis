#!/usr/bin/env python3
"""RB-STACK2: timestamp-safe backfield allocation + full-stack/M94C integration.

Protocol is frozen in docs/migrations/RB_STACK2_ENRICHED_ALLOCATION_INTEGRATION_PLAN.md.
2024 is the only fit season. 2025 is evaluation. Sportsbook data is downstream only.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.backtest.build_historical_injuries import normalize_historical_injuries

TEAM = {"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
RB_POS = {"RB","HB","FB"}
QB_POS = {"QB"}
ALLOWED_STATUS = {"ACT","INA"}


def lower(x: pd.DataFrame) -> pd.DataFrame:
    x=x.copy(); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def pdx(v) -> pd.DataFrame:
    if isinstance(v,pd.DataFrame): return v.copy()
    if hasattr(v,"to_pandas"): return v.to_pandas()
    if hasattr(v,"to_dicts"): return pd.DataFrame(v.to_dicts())
    return pd.DataFrame(v)

def tm(v) -> str:
    s=str(v).strip().upper() if not pd.isna(v) else ""
    return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""

def nk(v) -> str:
    return re.sub(r"[^a-z0-9]","",str(v or "").lower())

def first(df: pd.DataFrame, names: list[str], default=pd.NA) -> pd.Series:
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series(default,index=df.index)

def one(root: Path, name: str) -> pd.DataFrame:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected exactly one {name}, found {len(h)}")
    return lower(pd.read_csv(h[0],low_memory=False))

def ordv(season,week):
    return pd.to_numeric(season,errors="coerce")*100+pd.to_numeric(week,errors="coerce")


def load_weekly_logs(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    out=[]
    for s in seasons:
        raw=pdx(nfl.load_player_stats(seasons=[int(s)],summary_level="week"))
        q=_normalize_weekly(raw,int(s)); q=lower(q)
        q=q.loc[pd.to_numeric(q.get("week"),errors="coerce").between(1,18)].copy()
        q["season"]=pd.to_numeric(q["season"],errors="coerce").astype(int)
        q["week"]=pd.to_numeric(q["week"],errors="coerce").astype(int)
        q["team"]=q["team"].map(tm)
        q["position"]=q["position"].fillna("").astype(str).str.upper().str.strip()
        q["name_key"]=q.get("player_clean_key",q.get("player",pd.Series("",index=q.index))).astype(str).map(nk)
        q["rushes"]=pd.to_numeric(q.get("rushes"),errors="coerce").fillna(0.0)
        q["rush_yards"]=pd.to_numeric(q.get("rush_yards"),errors="coerce").fillna(0.0)
        q["order"]=ordv(q["season"],q["week"])
        out.append(q)
    x=pd.concat(out,ignore_index=True,sort=False)
    rb=x.loc[x.position.isin(RB_POS)].copy()
    den=rb.groupby(["season","week","team"],as_index=False).agg(team_rb_carries=("rushes","sum"))
    x=x.merge(den,on=["season","week","team"],how="left")
    x["rb_share"]=np.where(x.position.isin(RB_POS)&pd.to_numeric(x.team_rb_carries,errors="coerce").gt(0),x.rushes/x.team_rb_carries,0.0)
    return x


def load_rosters(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    out=[]
    for s in seasons:
        q=lower(pdx(nfl.load_rosters_weekly(int(s))))
        q["season"]=pd.to_numeric(q.get("season",s),errors="coerce").fillna(s).astype(int)
        q["week"]=pd.to_numeric(q.get("week"),errors="coerce")
        q=q.loc[q.season.eq(s)&q.week.between(1,18)].copy(); q["week"]=q.week.astype(int)
        q["team"]=first(q,["team","team_abbr","club_code"]).map(tm)
        q["position"]=first(q,["position","pos"]).fillna("").astype(str).str.upper().str.strip()
        q=q.loc[q.position.isin(RB_POS)].copy()
        if "status" in q.columns:
            z=q.status.fillna("").astype(str).str.upper().str.strip(); keep=z.isin(ALLOWED_STATUS)
            if keep.any(): q=q.loc[keep].copy()
        q["player"]=first(q,["full_name","football_name","player_name","player","name"]).astype(str).str.strip()
        q["name_key"]=q.player.map(nk)
        q["gsis_id"]=first(q,["gsis_id","player_id"]).fillna("").astype(str)
        q["status_raw"]=first(q,["status"]).fillna("").astype(str).str.upper()
        q["years_exp"]=pd.to_numeric(first(q,["years_exp","years_experience","experience"]),errors="coerce")
        q["rookie_year"]=pd.to_numeric(first(q,["rookie_year"]),errors="coerce")
        q["entry_year"]=pd.to_numeric(first(q,["entry_year"]),errors="coerce")
        q["draft_number"]=pd.to_numeric(first(q,["draft_number","draft_pick"]),errors="coerce")
        out.append(q[["season","week","team","position","player","name_key","gsis_id","status_raw","years_exp","rookie_year","entry_year","draft_number"]])
    return pd.concat(out,ignore_index=True).drop_duplicates(["season","week","team","name_key"],keep="last")


def games(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    x=lower(schedule); x["season"]=pd.to_numeric(x.season,errors="coerce"); x["week"]=pd.to_numeric(x.week,errors="coerce")
    x=x.loc[x.season.eq(season)&x.week.between(1,18)].copy()
    if "game_type" in x.columns: x=x.loc[x.game_type.astype(str).str.upper().eq("REG")]
    gt=x.gametime.astype(str) if "gametime" in x.columns else pd.Series("13:00",index=x.index)
    z=pd.to_datetime(x.gameday.astype(str)+" "+gt,errors="coerce"); east=ZoneInfo("America/New_York")
    x["kickoff_utc"]=[pd.Timestamp(d).tz_localize(east).tz_convert("UTC") if not pd.isna(d) else pd.NaT for d in z]
    rows=[]
    for _,r in x.iterrows():
        rows.append({"season":season,"week":int(r.week),"team":tm(r.home_team),"kickoff_utc":r.kickoff_utc})
        rows.append({"season":season,"week":int(r.week),"team":tm(r.away_team),"kickoff_utc":r.kickoff_utc})
    return pd.DataFrame(rows).drop_duplicates(["season","week","team"])


def depth_tables(seasons: list[int], schedule: pd.DataFrame) -> pd.DataFrame:
    import nflreadpy as nfl
    out=[]
    # 2024 explicitly week-tagged source.
    if 2024 in seasons:
        d=lower(pdx(nfl.load_depth_charts(seasons=[2024])))
        d["season"]=pd.to_numeric(d.get("season",2024),errors="coerce").fillna(2024)
        d["week"]=pd.to_numeric(d.get("week"),errors="coerce")
        teamcol="club_code" if "club_code" in d.columns else "team"
        namecol="full_name" if "full_name" in d.columns else "football_name" if "football_name" in d.columns else "player_name"
        d["team"]=d[teamcol].map(tm); d["name_key"]=d[namecol].map(nk)
        pos=first(d,["position","depth_position"]).fillna("").astype(str).str.upper()
        d=d.loc[d.week.between(1,18)&pos.isin(RB_POS)].copy()
        d["depth_rank"]=pd.to_numeric(first(d,["depth_team","pos_rank","rank"]),errors="coerce")
        d["depth_slot"]=first(d,["depth_position","position"]).fillna("").astype(str).str.upper()
        d["depth_present"]=1.0
        out.append(d[["season","week","team","name_key","depth_rank","depth_slot","depth_present"]].drop_duplicates(["season","week","team","name_key"],keep="last"))
    if 2025 in seasons:
        d=lower(pdx(nfl.load_depth_charts(seasons=[2025])))
        d["dt_utc"]=pd.to_datetime(d.get("dt"),errors="coerce",utc=True)
        d["team"]=first(d,["team","club_code"]).map(tm)
        d["name_key"]=first(d,["player_name","full_name","football_name","player"]).map(nk)
        a=first(d,["pos_abb","position"]).fillna("").astype(str).str.upper()
        n=first(d,["pos_name"]).fillna("").astype(str).str.lower()
        g=first(d,["pos_grp"]).fillna("").astype(str).str.lower()
        d=d.loc[(a.isin(RB_POS)|n.str.contains("running back|halfback|fullback",regex=True)|g.str.contains("running back|backfield",regex=True))&d.dt_utc.notna()].copy()
        gs=games(schedule,2025); rows=[]
        for _,gme in gs.iterrows():
            q=d.loc[d.team.eq(gme.team)&d.dt_utc.lt(gme.kickoff_utc)]
            if q.empty: continue
            t=q.dt_utc.max(); z=q.loc[q.dt_utc.eq(t)].copy()
            for _,r in z.iterrows():
                rows.append({"season":2025,"week":int(gme.week),"team":gme.team,"name_key":r.name_key,
                             "depth_rank":pd.to_numeric(pd.Series([r.get("pos_rank")]),errors="coerce").iloc[0],
                             "depth_slot":str(r.get("pos_slot",r.get("pos_abb","")) or "").upper(),"depth_present":1.0})
        if rows: out.append(pd.DataFrame(rows).drop_duplicates(["season","week","team","name_key"],keep="last"))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame(columns=["season","week","team","name_key","depth_rank","depth_slot","depth_present"])


def load_snaps(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    q=lower(pdx(nfl.load_snap_counts(seasons=seasons)))
    q["season"]=pd.to_numeric(q.get("season"),errors="coerce"); q["week"]=pd.to_numeric(q.get("week"),errors="coerce")
    q=q.loc[q.season.isin(seasons)&q.week.between(1,18)].copy()
    q["team"]=first(q,["team"]).map(tm); q["name_key"]=first(q,["player","player_name"]).map(nk)
    q["offense_pct"]=pd.to_numeric(first(q,["offense_pct","offense_percentage"]),errors="coerce")
    q["offense_snaps"]=pd.to_numeric(first(q,["offense_snaps"]),errors="coerce")
    q["order"]=ordv(q.season,q.week)
    return q[["season","week","team","name_key","offense_pct","offense_snaps","order"]].drop_duplicates(["season","week","team","name_key"],keep="last")


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    try: q=normalize_historical_injuries(pdx(nfl.load_injuries(seasons=seasons)))
    except Exception as exc:
        print(f"[STACK2] injuries unavailable: {exc}"); return pd.DataFrame(columns=["season","week","team","name_key"])
    q=lower(q); q["team"]=q.team.map(tm); q["name_key"]=q.player.map(nk)
    s=q.status.fillna("").astype(str).str.lower(); p=q.practice_status.fillna("").astype(str).str.lower()
    q["injury_reported"]=1.0; q["injury_out_doubtful"]=(s.str.contains("out|doubt")).astype(float)
    q["injury_questionable"]=(s.str.contains("question")).astype(float)
    q["practice_dnp"]=(p.str.contains("did not|dnp")).astype(float); q["practice_limited"]=(p.str.contains("limit")).astype(float)
    return q[["season","week","team","name_key","injury_reported","injury_out_doubtful","injury_questionable","practice_dnp","practice_limited"]].drop_duplicates(["season","week","team","name_key"],keep="last")


def history_maps(logs: pd.DataFrame, snaps: pd.DataFrame):
    lp={k:g.sort_values("order") for k,g in logs.groupby("name_key") if k}
    sp={k:g.sort_values("order") for k,g in snaps.groupby("name_key") if k}
    # team rushing competition history
    tg=logs.groupby(["season","week","team"],as_index=False).agg(total_rush=("rushes","sum"))
    qb=logs.loc[logs.position.isin(QB_POS)].groupby(["season","week","team"],as_index=False).agg(qb_rush=("rushes","sum"))
    rr=logs.loc[logs.position.isin(RB_POS)].groupby(["season","week","team"],as_index=False).agg(rb_rush=("rushes","sum"))
    tg=tg.merge(qb,on=["season","week","team"],how="left").merge(rr,on=["season","week","team"],how="left").fillna({"qb_rush":0,"rb_rush":0})
    tg["qb_rush_share"]=np.where(tg.total_rush.gt(0),tg.qb_rush/tg.total_rush,0.0); tg["order"]=ordv(tg.season,tg.week)
    tp={k:g.sort_values("order") for k,g in tg.groupby("team") if k}
    return lp,sp,tp


def enrich_history(target: pd.DataFrame, logs: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    lp,sp,tp=history_maps(logs,snaps); rows=[]
    for _,r in target.iterrows():
        o=int(r.season)*100+int(r.week); key=r.name_key; team=r.team
        h=lp.get(key,pd.DataFrame()); h=h.loc[h.order.lt(o)].tail(5) if not h.empty else h
        s=sp.get(key,pd.DataFrame()); s=s.loc[s.order.lt(o)].tail(5) if not s.empty else s
        th=tp.get(team,pd.DataFrame()); th=th.loc[th.order.lt(o)].tail(3) if not th.empty else th
        def last(df,col,default=np.nan): return float(pd.to_numeric(df[col],errors="coerce").iloc[-1]) if len(df) and col in df else default
        def mean(df,col,n=3,default=np.nan):
            if not len(df) or col not in df: return default
            z=pd.to_numeric(df[col],errors="coerce").tail(n); return float(z.mean()) if z.notna().any() else default
        lastteam=str(h.team.iloc[-1]) if len(h) and "team" in h else ""
        rows.append({
            "prior_games":float(len(lp.get(key,pd.DataFrame()).loc[lp.get(key,pd.DataFrame()).order.lt(o)]) if key in lp else 0),
            "prior1_carries":last(h,"rushes",0.0),"prior3_carries":mean(h,"rushes",3,0.0),"prior5_carries":mean(h,"rushes",5,0.0),
            "prior1_rb_share":last(h,"rb_share",0.0),"prior3_rb_share":mean(h,"rb_share",3,0.0),"prior5_rb_share":mean(h,"rb_share",5,0.0),
            "prior1_rush_yards":last(h,"rush_yards",0.0),"prior3_rush_yards":mean(h,"rush_yards",3,0.0),
            "prior1_snap_pct":last(s,"offense_pct",np.nan),"prior3_snap_pct":mean(s,"offense_pct",3,np.nan),
            "prior1_snap_count":last(s,"offense_snaps",np.nan),"prior3_snap_count":mean(s,"offense_snaps",3,np.nan),
            "same_team_last_game":float(bool(lastteam) and lastteam==team),
            "team_prior1_qb_rush_share":last(th,"qb_rush_share",0.0),"team_prior3_qb_rush_share":mean(th,"qb_rush_share",3,0.0),
        })
    return pd.concat([target.reset_index(drop=True),pd.DataFrame(rows)],axis=1)


def add_team_competition(x: pd.DataFrame) -> pd.DataFrame:
    out=x.copy(); vals=[]
    for _,g in out.groupby(["season","week","team"],sort=False):
        idx=list(g.index)
        for i in idx:
            oth=g.loc[g.index!=i]
            shares=pd.to_numeric(oth.get("prior3_rb_share"),errors="coerce").fillna(0)
            snaps=pd.to_numeric(oth.get("prior3_snap_pct"),errors="coerce").fillna(0)
            injured=pd.to_numeric(oth.get("injury_out_doubtful"),errors="coerce").fillna(0)
            dep=pd.to_numeric(g.get("depth_present"),errors="coerce").fillna(0)
            ownshares=pd.to_numeric(g.get("prior3_rb_share"),errors="coerce").fillna(0)
            ss=float(ownshares.sum()); hhi=float(np.square(ownshares/ss).sum()) if ss>0 else 0.0
            vals.append((i,float(shares.max()) if len(shares) else 0.0,float(snaps.max()) if len(snaps) else 0.0,
                         float((shares>=.15).sum()),float(dep.sum()),hhi,float(injured.sum()),float((shares*injured).sum())))
    z=pd.DataFrame(vals,columns=["_idx","max_comp_prior3_share","max_comp_prior3_snap","credible_competitors","depth_back_count","prior_backfield_hhi","injured_comp_count","injured_comp_prior_share"])
    z=z.set_index("_idx");
    for c in z.columns: out.loc[z.index,c]=z[c]
    return out


def build_training(rosters: pd.DataFrame, logs: pd.DataFrame, depth: pd.DataFrame, snaps: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    t=rosters.loc[rosters.season.eq(2024)].copy()
    actual=logs.loc[logs.season.eq(2024)&logs.position.isin(RB_POS),["season","week","team","name_key","rushes","rush_yards","team_rb_carries"]].copy()
    actual=actual.drop_duplicates(["season","week","team","name_key"])
    t=t.merge(actual,on=["season","week","team","name_key"],how="left"); t[["rushes","rush_yards"]]=t[["rushes","rush_yards"]].fillna(0.0)
    den=actual[["season","week","team","team_rb_carries"]].drop_duplicates(["season","week","team"])
    t=t.drop(columns=["team_rb_carries"],errors="ignore").merge(den,on=["season","week","team"],how="left"); t["team_rb_carries"]=pd.to_numeric(t.team_rb_carries,errors="coerce").fillna(0.0)
    t["actual_share"]=np.where(t.team_rb_carries.gt(0),t.rushes/t.team_rb_carries,0.0)
    t=t.merge(depth,on=["season","week","team","name_key"],how="left").merge(injuries,on=["season","week","team","name_key"],how="left")
    t=enrich_history(t,logs,snaps); return add_team_competition(t)


def build_eval(m94: pd.DataFrame, rosters: pd.DataFrame, logs: pd.DataFrame, depth: pd.DataFrame, snaps: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    t=m94.copy(); t=t.loc[pd.to_numeric(t.season,errors="coerce").eq(2025)&t.position.astype(str).str.upper().isin(RB_POS)].copy()
    t["season"]=2025; t["week"]=pd.to_numeric(t.week,errors="coerce").astype(int); t["team"]=t.team.map(tm); t["name_key"]=t.player.map(nk)
    rr=rosters.loc[rosters.season.eq(2025)].copy(); cols=["season","week","team","name_key","status_raw","years_exp","rookie_year","entry_year","draft_number"]
    t=t.merge(rr[cols].drop_duplicates(["season","week","team","name_key"]),on=["season","week","team","name_key"],how="left")
    t=t.merge(depth,on=["season","week","team","name_key"],how="left").merge(injuries,on=["season","week","team","name_key"],how="left")
    t=enrich_history(t,logs,snaps); t=add_team_competition(t)
    t["actual_rush_att"]=pd.to_numeric(t.get("actual_rush_att"),errors="coerce"); t["actual_rush_yards"]=pd.to_numeric(t.get("actual_rush_yards"),errors="coerce")
    den=t.groupby(["season","week","team"])["actual_rush_att"].transform("sum")
    t["actual_share"]=np.where(den.gt(0),t.actual_rush_att/den,0.0)
    return t


def finalize_features(x: pd.DataFrame) -> pd.DataFrame:
    z=x.copy()
    z["depth_present"]=pd.to_numeric(z.get("depth_present"),errors="coerce").fillna(0.0)
    z["depth_rank"]=pd.to_numeric(z.get("depth_rank"),errors="coerce")
    z["depth_rank_missing"]=z.depth_rank.isna().astype(float); z["depth_rank"]=z.depth_rank.fillna(4.0).clip(1,8)
    slot=z.get("depth_slot",pd.Series("",index=z.index)).fillna("").astype(str).str.upper()
    z["depth_slot_rb"]=(slot.str.contains("RB|HB")).astype(float); z["depth_slot_fb"]=(slot.str.contains("FB")).astype(float)
    status=z.get("status_raw",pd.Series("",index=z.index)).fillna("").astype(str).str.upper()
    z["roster_active"]=(status.eq("ACT")).astype(float); z["roster_inactive"]=(status.eq("INA")).astype(float)
    z["rookie_flag"]=(pd.to_numeric(z.get("years_exp"),errors="coerce").fillna(99).le(0)|pd.to_numeric(z.get("rookie_year"),errors="coerce").eq(pd.to_numeric(z.get("season"),errors="coerce"))).astype(float)
    z["drafted_flag"]=pd.to_numeric(z.get("draft_number"),errors="coerce").notna().astype(float)
    z["draft_number_log"]=np.log1p(pd.to_numeric(z.get("draft_number"),errors="coerce").fillna(300).clip(1,300))
    for c in ["injury_reported","injury_out_doubtful","injury_questionable","practice_dnp","practice_limited"]:
        z[c]=pd.to_numeric(z.get(c),errors="coerce").fillna(0.0)
    return z

CORE=["depth_present","depth_rank","depth_rank_missing","depth_slot_rb","depth_slot_fb","roster_active","roster_inactive","rookie_flag","drafted_flag","draft_number_log","injury_reported","injury_out_doubtful","injury_questionable","practice_dnp","practice_limited","depth_back_count","injured_comp_count","injured_comp_prior_share"]
FULL=CORE+["prior_games","prior1_carries","prior3_carries","prior5_carries","prior1_rb_share","prior3_rb_share","prior5_rb_share","prior1_rush_yards","prior3_rush_yards","prior1_snap_pct","prior3_snap_pct","prior1_snap_count","prior3_snap_count","same_team_last_game","max_comp_prior3_share","max_comp_prior3_snap","credible_competitors","prior_backfield_hhi","team_prior1_qb_rush_share","team_prior3_qb_rush_share"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], seed=17) -> np.ndarray:
    X=train.reindex(columns=feats).apply(pd.to_numeric,errors="coerce"); y=pd.to_numeric(train.actual_share,errors="coerce")
    ok=y.notna(); X=X.loc[ok]; y=y.loc[ok].clip(0,1)
    model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=160,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.0,random_state=seed)
    model.fit(X,y)
    p=model.predict(test.reindex(columns=feats).apply(pd.to_numeric,errors="coerce")); return np.clip(p,0,1)


def normalize_team_scores(x: pd.DataFrame, score_col: str, out_col: str):
    vals=pd.Series(index=x.index,dtype=float)
    for _,g in x.groupby(["season","week","team"]):
        s=pd.to_numeric(g[score_col],errors="coerce").fillna(0).clip(lower=0); total=float(s.sum())
        if total<=0: s=pd.Series(np.ones(len(g))/len(g),index=g.index)
        else: s=s/total
        vals.loc[g.index]=s
    x[out_col]=vals


def metric(y,p):
    y=pd.to_numeric(y,errors="coerce"); p=pd.to_numeric(p,errors="coerce"); ok=y.notna()&p.notna(); y=y[ok].astype(float);p=p[ok].astype(float)
    if not len(y): return {"n":0}
    e=p-y
    return {"n":int(len(y)),"mae":float(np.abs(e).mean()),"rmse":float(np.sqrt(np.square(e).mean())),"bias":float(e.mean()),"corr":float(np.corrcoef(p,y)[0,1]) if len(y)>1 and y.std()>0 and p.std()>0 else np.nan,"actual_mean":float(y.mean()),"pred_mean":float(p.mean())}


def stack_wide(stack: pd.DataFrame) -> pd.DataFrame:
    s=stack.copy(); s["team"]=s.team.map(tm); s["name_key"]=s.get("player_clean_key",s.player).astype(str).map(nk)
    keep=s.loc[s.position.astype(str).str.upper().isin(RB_POS)&s.market.isin(["rush_att","rush_yards"])].copy()
    p=keep.pivot_table(index=["season","week","team","name_key"],columns="market",values=["ensemble_2024_frozen"],aggfunc="first")
    p.columns=["stack_att" if c[1]=="rush_att" else "stack_yards" for c in p.columns]; return p.reset_index()


def add_projection_arms(e: pd.DataFrame, stack: pd.DataFrame) -> pd.DataFrame:
    x=e.merge(stack_wide(stack),on=["season","week","team","name_key"],how="left",validate="one_to_one")
    x["m94c_att"]=pd.to_numeric(x.candidate_rush_att,errors="coerce"); x["m94c_yards"]=pd.to_numeric(x.candidate_rush_yards,errors="coerce")
    # Raw M94C share within its RB/FB opportunity pool.
    pool=x.groupby(["season","week","team"])["m94c_att"].transform("sum"); x["m94c_share"]=np.where(pool.gt(0),x.m94c_att/pool,0.0)
    # Primary enriched share is an additive 50/50 correction to M94C's existing share.
    x["enriched_share"]=.5*x.m94c_share+.5*x.alloc_full_share
    x["enriched_att"]=x.enriched_share*pool
    x["enriched_direct_att"]=x.alloc_full_share*pool
    # Preserve M94C implied player efficiency for pure allocation isolation.
    ypc=np.where(x.m94c_att.gt(.20),x.m94c_yards/x.m94c_att,np.nan); ypc=pd.Series(ypc,index=x.index).replace([np.inf,-np.inf],np.nan)
    fallback=float(pd.to_numeric(pd.Series(ypc),errors="coerce").dropna().median()) if pd.Series(ypc).notna().any() else 4.2
    x["m94c_implied_ypc"]=pd.to_numeric(ypc,errors="coerce").fillna(fallback)
    x["enriched_yards"]=x.enriched_att*x.m94c_implied_ypc; x["enriched_direct_yards"]=x.enriched_direct_att*x.m94c_implied_ypc
    # Opportunity/efficiency decomposition between the two parent systems.
    stack_ypc=np.where(pd.to_numeric(x.stack_att,errors="coerce").gt(.20),pd.to_numeric(x.stack_yards,errors="coerce")/pd.to_numeric(x.stack_att,errors="coerce"),np.nan)
    stack_ypc=pd.Series(stack_ypc,index=x.index).replace([np.inf,-np.inf],np.nan); stack_ypc=stack_ypc.fillna(x.m94c_implied_ypc)
    x["stack_implied_ypc"]=stack_ypc
    x["arch_enriched_opp_stack_eff_yards"]=x.enriched_att*x.stack_implied_ypc
    x["arch_stack_opp_m94c_eff_yards"]=pd.to_numeric(x.stack_att,errors="coerce")*x.m94c_implied_ypc
    # Precommitted point blends.
    x["blend_75s25e_yards"]=.75*pd.to_numeric(x.stack_yards,errors="coerce")+.25*x.enriched_yards
    x["blend_50s50e_yards"]=.50*pd.to_numeric(x.stack_yards,errors="coerce")+.50*x.enriched_yards
    x["blend_25s75e_yards"]=.25*pd.to_numeric(x.stack_yards,errors="coerce")+.75*x.enriched_yards
    x["blend_75s25e_att"]=.75*pd.to_numeric(x.stack_att,errors="coerce")+.25*x.enriched_att
    x["blend_50s50e_att"]=.50*pd.to_numeric(x.stack_att,errors="coerce")+.50*x.enriched_att
    x["blend_25s75e_att"]=.25*pd.to_numeric(x.stack_att,errors="coerce")+.75*x.enriched_att
    x["ypc_fallback_used"]=(pd.to_numeric(pd.Series(ypc),errors="coerce").isna()).astype(int)
    return x


def slices(x: pd.DataFrame):
    a=pd.to_numeric(x.actual_rush_att,errors="coerce"); w=pd.to_numeric(x.week,errors="coerce"); rank=pd.to_numeric(x.depth_rank,errors="coerce")
    return {"all_rb":pd.Series(True,index=x.index),"week1":w.eq(1),"weeks2_18":w.ge(2),"actual_0_5":a.le(5),"actual_6_10":a.between(6,10),"actual_11_14":a.between(11,14),"actual_15_19":a.between(15,19),"actual_20_plus":a.ge(20),"actual_25_plus":a.ge(25),"pregame_depth_rank1":rank.eq(1),"pregame_depth_rank2":rank.eq(2),"pregame_depth_rank3plus":rank.ge(3),"pregame_committee":pd.to_numeric(x.credible_competitors,errors="coerce").fillna(0).ge(1),"pregame_concentrated":pd.to_numeric(x.prior_backfield_hhi,errors="coerce").fillna(0).ge(.65)}


def score_all(x: pd.DataFrame) -> pd.DataFrame:
    yards={"STACK1_PARENT":"stack_yards","M94C_RAW":"m94c_yards","M94C_ENRICHED_ALLOCATION":"enriched_yards","M94C_ENRICHED_DIRECT_DIAG":"enriched_direct_yards","ARCH_ENRICHED_OPP_STACK_EFF":"arch_enriched_opp_stack_eff_yards","ARCH_STACK_OPP_M94C_EFF":"arch_stack_opp_m94c_eff_yards","BLEND_75_STACK_25_ENRICHED":"blend_75s25e_yards","BLEND_50_STACK_50_ENRICHED":"blend_50s50e_yards","BLEND_25_STACK_75_ENRICHED":"blend_25s75e_yards"}
    atts={"STACK1_PARENT":"stack_att","M94C_RAW":"m94c_att","M94C_ENRICHED_ALLOCATION":"enriched_att","M94C_ENRICHED_DIRECT_DIAG":"enriched_direct_att","BLEND_75_STACK_25_ENRICHED":"blend_75s25e_att","BLEND_50_STACK_50_ENRICHED":"blend_50s50e_att","BLEND_25_STACK_75_ENRICHED":"blend_25s75e_att"}
    rows=[]
    for sn,mask in slices(x).items():
        q=x.loc[mask]
        for arm,c in yards.items(): rows.append({"market":"rush_yards","slice":sn,"arm":arm,**metric(q.actual_rush_yards,q[c])})
        for arm,c in atts.items(): rows.append({"market":"rush_att","slice":sn,"arm":arm,**metric(q.actual_rush_att,q[c])})
    return pd.DataFrame(rows)


def market_score(x: pd.DataFrame, cb: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    c=cb.copy(); c["team"]=c.team.map(tm); c["name_key"]=c.player.map(nk); c["week"]=pd.to_numeric(c.week,errors="coerce")
    keys=["week","team","name_key"]; q=x.merge(c[keys+["consensus_line"]].drop_duplicates(keys),on=keys,how="inner",validate="one_to_one")
    arms={"STACK1_PARENT":"stack_yards","M94C_RAW":"m94c_yards","M94C_ENRICHED_ALLOCATION":"enriched_yards","ARCH_ENRICHED_OPP_STACK_EFF":"arch_enriched_opp_stack_eff_yards","BLEND_75_STACK_25_ENRICHED":"blend_75s25e_yards","BLEND_50_STACK_50_ENRICHED":"blend_50s50e_yards","BLEND_25_STACK_75_ENRICHED":"blend_25s75e_yards","VEGAS_CONSENSUS":"consensus_line"}
    rows=[]
    for arm,col in arms.items(): rows.append({"arm":arm,**metric(q.actual_rush_yards,q[col])})
    edges=[]
    bins=[0,2.5,5,10,np.inf]; labels=["0_2.5","2.5_5","5_10","10_plus"]
    for arm,col in arms.items():
        if arm=="VEGAS_CONSENSUS": continue
        z=q[["actual_rush_yards","consensus_line",col]].copy().dropna(); z["edge"]=pd.to_numeric(z[col],errors="coerce")-pd.to_numeric(z.consensus_line,errors="coerce"); z["abs_edge"]=z.edge.abs(); z["bucket"]=pd.cut(z.abs_edge,bins=bins,labels=labels,right=False)
        z["model_closer"]=(np.abs(pd.to_numeric(z[col],errors="coerce")-z.actual_rush_yards)<np.abs(z.consensus_line-z.actual_rush_yards)).astype(float)
        for b,g in z.groupby("bucket",observed=True): edges.append({"arm":arm,"edge_bucket":str(b),"n":len(g),"model_closer_rate":float(g.model_closer.mean()),"mean_edge":float(g.edge.mean()),"model_mae":float(np.abs(g[col]-g.actual_rush_yards).mean()),"vegas_mae":float(np.abs(g.consensus_line-g.actual_rush_yards).mean())})
    return pd.DataFrame(rows),pd.DataFrame(edges)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--stack1-root",type=Path,required=True); ap.add_argument("--market-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    import nflreadpy as nfl
    m94=one(a.m94c_root,"m94c_2025_rb_trace.csv"); stack=one(a.stack1_root,"stack1_2025_rb_trace.csv"); cb=one(a.market_root,"rb_market_casebook.csv")
    seasons=[2023,2024,2025]; logs=load_weekly_logs(seasons); rosters=load_rosters([2024,2025]); sched=lower(pdx(nfl.load_schedules(seasons=[2024,2025]))); depth=depth_tables([2024,2025],sched); snaps=load_snaps(seasons); injuries=load_injuries([2024,2025])
    tr=finalize_features(build_training(rosters,logs,depth,snaps,injuries)); ev=finalize_features(build_eval(m94,rosters,logs,depth,snaps,injuries))
    ev["alloc_core_score"]=fit_predict(tr,ev,CORE,17); ev["alloc_full_score"]=fit_predict(tr,ev,FULL,17); normalize_team_scores(ev,"alloc_core_score","alloc_core_share"); normalize_team_scores(ev,"alloc_full_score","alloc_full_share")
    x=add_projection_arms(ev,stack); scores=score_all(x); market,edges=market_score(x,cb)
    # allocation-only evidence
    alloc=[]
    for arm,col in {"M94C_RAW_SHARE":"m94c_share","DEPTH_CORE_SHARE":"alloc_core_share","FULL_ROLE_USAGE_SHARE":"alloc_full_share","ENRICHED_50ANCHOR_SHARE":"enriched_share"}.items(): alloc.append({"arm":arm,**metric(x.actual_share,x[col])})
    alloc=pd.DataFrame(alloc)
    coverage=pd.DataFrame([{"eval_rows":len(x),"depth_coverage":float(pd.to_numeric(x.depth_present,errors="coerce").fillna(0).gt(0).mean()),"snap3_coverage":float(pd.to_numeric(x.prior3_snap_pct,errors="coerce").notna().mean()),"injury_report_rows":int(pd.to_numeric(x.injury_reported,errors="coerce").fillna(0).gt(0).sum()),"roster_match_rate":float(x.status_raw.notna().mean()),"ypc_fallback_rows":int(x.ypc_fallback_used.sum()),"market_rows":int(market.loc[market.arm.eq("VEGAS_CONSENSUS"),"n"].iloc[0]) if len(market) else 0,"train_rows":len(tr)}])
    disposition=pd.DataFrame([{"disposition":"STACK2_EVIDENCE_GENERATED_REQUIRES_CAPABILITY_REVIEW","fit_season":2024,"test_season":2025,"sportsbook_upstream":0,"blend_weight_search":0,"point_blend_grid":"75/25;50/50;25/75","next":"REVIEW_ALLOCATION_COMPLEMENTARITY_THEN_M95_M96_ABLATIONS"}])
    for name,df in [("stack2_allocation_metrics.csv",alloc),("stack2_slice_metrics.csv",scores),("stack2_market_899_metrics.csv",market),("stack2_market_edge_buckets.csv",edges),("stack2_coverage.csv",coverage),("stack2_2025_casebook.csv",x),("stack2_disposition.csv",disposition)]: df.to_csv(a.out_dir/name,index=False)
    print("=== coverage ==="); print(coverage.to_string(index=False)); print("=== allocation ==="); print(alloc.to_string(index=False)); print("=== all-RB rushing yards ==="); print(scores.loc[(scores.market.eq("rush_yards"))&(scores.slice.eq("all_rb"))].to_string(index=False)); print("=== key workload yards ==="); print(scores.loc[(scores.market.eq("rush_yards"))&scores.slice.isin(["week1","actual_20_plus","actual_25_plus"])].to_string(index=False)); print("=== market ==="); print(market.to_string(index=False)); print("=== edge buckets ==="); print(edges.to_string(index=False))

if __name__=="__main__": main()
