"""M95B: compact RB offense x defense matchup engine.

Research-only. This experiment freezes the M95A leakage-safe defensive trace,
adds player/team offensive rushing priors from nflverse PBP, weekly PFR advanced
rushing, and weekly Next Gen Stats, then compares four pre-specified forward
model families:
  1) role baseline
  2) role + offense
  3) role + offense + defense
  4) full offense x defense interactions

No sportsbook inputs. No production projection code is changed.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_SEASONS = (2023, 2024, 2025)
PBP_SEASONS = (2022, 2023, 2024, 2025)
TEAM_MAP = {"ARZ":"ARI","JAC":"JAX","LA":"LAR","STL":"LAR","OAK":"LV","SD":"LAC","WSH":"WAS"}

ROLE_FEATURES = [
    "rb_games_before","rb_carries_avg1","rb_carries_avg3","rb_carries_avg5",
    "rb_rb_share_avg1","rb_rb_share_avg3","rb_rb_share_avg5",
    "rb_15plus_rate3","rb_15plus_rate5","rb_20plus_rate3","rb_20plus_rate5",
    "team_rb_pool_avg3","team_rb_pool_avg5","team_total_rush_avg3","team_total_rush_avg5",
    "team_top1_share_avg3","team_top1_share_avg5","team_rb_used_avg3","team_rb_used_avg5",
    "team_qb_rush_share_avg3","team_qb_rush_share_avg5","home",
]

DEF_FAMILIES = {
    "def_run_efficiency_score":[
        ("def_rush_ypa_allowed_avg5",1),("def_rush_epa_allowed_avg5",1),
        ("def_rush_success_allowed_avg5",1),("def_rush_first_down_rate_allowed_avg5",1),
        ("def_non_scramble_ypa_allowed_avg5",1),("def_stuff_rate_allowed_avg5",-1),
    ],
    "def_explosive_vulnerability_score":[
        ("def_explosive10_rate_allowed_avg5",1),("def_explosive15_rate_allowed_avg5",1),
        ("def_explosive20_rate_allowed_avg5",1),
    ],
    "def_resistance_weakness_score":[
        ("def_stuff_rate_allowed_avg5",-1),("def_rush_first_down_rate_allowed_avg5",1),
        ("def_short_success_allowed_avg5",1),
    ],
    "def_rb_specific_vulnerability_score":[
        ("def_rb_rush_yards_allowed_avg5",1),("def_rb_ypc_allowed_avg5",1),
        ("def_top_rb_rush_yards_allowed_avg5",1),("def_rb_over_prior5_rush_yards_allowed_avg5",1),
        ("def_rb_75plus_rate_allowed_avg5",1),("def_rb_100plus_rate_allowed_avg5",1),
    ],
    "def_redzone_vulnerability_score":[
        ("def_redzone_success_allowed_avg5",1),("def_inside10_ypa_allowed_avg5",1),
        ("def_inside5_td_rate_allowed_avg5",1),
    ],
}

OFF_FAMILIES = {
    "off_player_efficiency_score":[
        ("player_pbp_ypa_avg5",1),("player_pbp_epa_avg5",1),("player_pbp_success_avg5",1),
        ("player_pbp_first_down_rate_avg5",1),("player_pbp_stuff_rate_avg5",-1),
        ("pfr_ybc_per_att_avg5",1),("pfr_yac_per_att_avg5",1),("pfr_brk_tkl_per_att_avg5",1),
        ("ngs_ryoe_per_att_avg5",1),("ngs_rush_pct_over_expected_avg5",1),
    ],
    "off_player_explosive_score":[
        ("player_pbp_explosive10_rate_avg5",1),("player_pbp_explosive15_rate_avg5",1),
        ("player_pbp_explosive20_rate_avg5",1),("ngs_ryoe_per_att_avg5",1),
    ],
    "off_player_environment_score":[
        ("pfr_ybc_per_att_avg5",1),("ngs_expected_yards_per_att_avg5",1),
        ("ngs_percent_attempts_gte_eight_defenders_avg5",-1),("ngs_avg_time_to_los_avg5",-1),
    ],
    "off_team_rush_strength_score":[
        ("team_pbp_ypa_avg5",1),("team_pbp_epa_avg5",1),("team_pbp_success_avg5",1),
        ("team_pbp_first_down_rate_avg5",1),("team_pbp_stuff_rate_avg5",-1),
        ("team_pfr_ybc_per_att_avg5",1),("team_pfr_yac_per_att_avg5",1),
    ],
    "off_team_explosive_score":[
        ("team_pbp_explosive10_rate_avg5",1),("team_pbp_explosive20_rate_avg5",1),
    ],
    "off_team_structure_score":[
        ("team_pbp_rush_rate_avg5",1),("team_pbp_neutral_rush_rate_avg5",1),
        ("team_pbp_early_down_rush_rate_avg5",1),("team_rb_pool_avg5",1),
        ("team_top1_share_avg5",1),("team_qb_rush_share_avg5",-1),("team_rb_used_avg5",-1),
    ],
    "off_team_environment_score":[
        ("team_pbp_plays_avg5",1),("team_pbp_rush_rate_avg5",1),("team_total_rush_avg5",1),
    ],
    "off_short_redzone_score":[
        ("team_pbp_short_success_avg5",1),("team_pbp_redzone_success_avg5",1),
        ("team_pbp_inside10_ypa_avg5",1),("team_pbp_inside5_td_rate_avg5",1),
    ],
}

INTERACTIONS = [
    "mx_player_efficiency_x_def_efficiency","mx_player_explosive_x_def_explosive",
    "mx_player_environment_x_def_efficiency","mx_role_x_rb_specific",
    "mx_team_strength_x_def_efficiency","mx_team_structure_x_def_resistance",
    "mx_short_redzone_x_def_redzone","mx_team_explosive_x_def_explosive",
    "mx_directional","mx_shotgun",
]


def lower(df):
    x=df.copy(); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def num(s, default=np.nan):
    z=pd.to_numeric(s,errors="coerce"); return z.fillna(default) if np.isfinite(default) else z

def team(v):
    x=str(v).upper().strip(); return TEAM_MAP.get(x,x)

def short_name(v):
    raw=str(v).lower().strip().replace("'","")
    raw=re.sub(r"[^a-z0-9.\- ]","",raw)
    toks=[t for t in re.split(r"[\s.\-]+",raw) if t and t not in {"jr","sr","ii","iii","iv"}]
    if not toks: return ""
    if len(toks)==1: return re.sub(r"[^a-z0-9]","",toks[0])
    return re.sub(r"[^a-z0-9]","",toks[0][0]+toks[-1])

def alias(df,names):
    for n in names:
        if n in df.columns:return n
    return None


def read_trace(root:Path):
    paths=list(root.rglob("m95a_rb_game_trace.csv"))
    if not paths: raise RuntimeError(f"missing m95a_rb_game_trace.csv under {root}")
    x=lower(pd.read_csv(paths[0],low_memory=False))
    x["season"]=num(x["season"]); x["week"]=num(x["week"])
    x=x.loc[x["season"].isin(TARGET_SEASONS)].copy()
    x["season"]=x["season"].astype(int); x["week"]=x["week"].astype(int)
    x["team"]=x["team"].map(team); x["opponent"]=x["opponent"].map(team)
    x["player_short_key"]=x["player"].map(short_name)
    return x.reset_index(drop=True)


def read_pbp(root:Path):
    frames=[]
    wanted=["season","week","posteam","defteam","rush_attempt","pass_attempt","rushing_yards","epa","success",
            "down","ydstogo","yardline_100","first_down_rush","touchdown","run_location","shotgun","qb_kneel",
            "qb_scramble","rusher_player_name","score_differential"]
    for s in PBP_SEASONS:
        p=root/f"play_by_play_{s}.parquet"
        if not p.exists(): raise RuntimeError(f"missing {p}")
        cols=pd.read_parquet(p,engine="pyarrow").columns.tolist(); use=[c for c in wanted if c in cols]
        frames.append(lower(pd.read_parquet(p,columns=use,engine="pyarrow")))
    x=pd.concat(frames,ignore_index=True,sort=False)
    x["season"]=num(x["season"]);x["week"]=num(x["week"])
    x["posteam"]=x["posteam"].map(team);x["defteam"]=x["defteam"].map(team)
    return x


def player_pbp_games(pbp):
    x=pbp.copy(); rush=num(x.get("rush_attempt",pd.Series(index=x.index,dtype=float)),0).eq(1); kneel=num(x.get("qb_kneel",pd.Series(index=x.index,dtype=float)),0).eq(1)
    x=x.loc[rush&~kneel&x["posteam"].notna()].copy()
    if "rusher_player_name" not in x:return pd.DataFrame()
    x["player_short_key"]=x["rusher_player_name"].map(short_name);x=x.loc[x["player_short_key"].ne("")]
    x["rushing_yards"]=num(x.get("rushing_yards",pd.Series(index=x.index,dtype=float)));x["epa"]=num(x.get("epa",pd.Series(index=x.index,dtype=float)));x["success"]=num(x.get("success",pd.Series(index=x.index,dtype=float)))
    x["first_down_rush"]=num(x.get("first_down_rush",pd.Series(index=x.index,dtype=float)),0)
    rows=[]
    for (s,w,t,p),g in x.groupby(["season","week","posteam","player_short_key"]):
        y=g["rushing_yards"]
        rows.append({"season":int(s),"week":int(w),"team":t,"player_short_key":p,"player_pbp_att":len(g),
                     "player_pbp_ypa":y.mean(),"player_pbp_epa":g["epa"].mean(),"player_pbp_success":g["success"].mean(),
                     "player_pbp_first_down_rate":g["first_down_rush"].mean(),"player_pbp_stuff_rate":y.le(0).mean(),
                     "player_pbp_explosive10_rate":y.ge(10).mean(),"player_pbp_explosive15_rate":y.ge(15).mean(),
                     "player_pbp_explosive20_rate":y.ge(20).mean(),"actual_player_explosive20":int(y.ge(20).any())})
    return pd.DataFrame(rows)


def team_pbp_games(pbp):
    x=pbp.loc[pbp["posteam"].notna()].copy()
    for c in ["rush_attempt","pass_attempt","qb_kneel","qb_scramble","first_down_rush","touchdown","shotgun"]:x[c]=num(x.get(c,pd.Series(index=x.index,dtype=float)),0)
    x=x.loc[(x["rush_attempt"].eq(1)|x["pass_attempt"].eq(1))&x["qb_kneel"].ne(1)].copy()
    for c in ["rushing_yards","epa","success","down","ydstogo","yardline_100","score_differential"]:x[c]=num(x.get(c,pd.Series(index=x.index,dtype=float)))
    x["run_location"]=x.get("run_location",pd.Series(index=x.index,dtype=object)).astype(str).str.lower()
    rows=[]
    for (s,w,t),g in x.groupby(["season","week","posteam"]):
        r=g.loc[g["rush_attempt"].eq(1)].copy()
        if r.empty:continue
        y=r["rushing_yards"]; early=g["down"].isin([1,2]); neutral=early&g["score_differential"].abs().le(7)
        short=r["down"].isin([3,4])&r["ydstogo"].le(2); rz=r["yardline_100"].le(20);in10=r["yardline_100"].le(10);in5=r["yardline_100"].le(5);shot=r["shotgun"].eq(1);loc=r["run_location"]
        rows.append({"season":int(s),"week":int(w),"team":t,"team_pbp_plays":len(g),"team_pbp_rush_rate":len(r)/len(g),
                     "team_pbp_early_down_rush_rate":g.loc[early,"rush_attempt"].mean() if early.any() else np.nan,
                     "team_pbp_neutral_rush_rate":g.loc[neutral,"rush_attempt"].mean() if neutral.any() else np.nan,
                     "team_pbp_ypa":y.mean(),"team_pbp_epa":r["epa"].mean(),"team_pbp_success":r["success"].mean(),
                     "team_pbp_first_down_rate":r["first_down_rush"].mean(),"team_pbp_stuff_rate":y.le(0).mean(),
                     "team_pbp_explosive10_rate":y.ge(10).mean(),"team_pbp_explosive20_rate":y.ge(20).mean(),
                     "team_pbp_short_success":r.loc[short,"success"].mean() if short.any() else np.nan,
                     "team_pbp_redzone_success":r.loc[rz,"success"].mean() if rz.any() else np.nan,
                     "team_pbp_inside10_ypa":r.loc[in10,"rushing_yards"].mean() if in10.any() else np.nan,
                     "team_pbp_inside5_td_rate":r.loc[in5,"touchdown"].mean() if in5.any() else np.nan,
                     "team_pbp_shotgun_rush_rate":shot.mean(),"team_pbp_shotgun_ypa":r.loc[shot,"rushing_yards"].mean() if shot.any() else np.nan,
                     "team_pbp_qb_scramble_share":r["qb_scramble"].mean(),"team_pbp_left_share":loc.eq("left").mean(),
                     "team_pbp_middle_share":loc.eq("middle").mean(),"team_pbp_right_share":loc.eq("right").mean()})
    return pd.DataFrame(rows)


def read_pfr(root:Path):
    frames=[]
    for p in sorted(root.glob("advstats_week_rush_*.csv")):
        z=lower(pd.read_csv(p,low_memory=False));frames.append(z)
    if not frames:return pd.DataFrame()
    x=pd.concat(frames,ignore_index=True,sort=False)
    sc=alias(x,["season"]);wc=alias(x,["week"]);pc=alias(x,["player","player_name"]);tc=alias(x,["tm","team"]);ac=alias(x,["att","rush_att","carries"])
    if not all([sc,wc,pc,tc]):return pd.DataFrame()
    out=pd.DataFrame({"season":num(x[sc]),"week":num(x[wc]),"team":x[tc].map(team),"player_short_key":x[pc].map(short_name)})
    att=num(x[ac]) if ac else pd.Series(np.nan,index=x.index)
    ybc=alias(x,["ybc","yards_before_contact"]);yac=alias(x,["yac","yards_after_contact"]);br=alias(x,["brktkl","brk_tkl","broken_tackles"])
    out["pfr_att"]=att
    out["pfr_ybc_per_att"]=num(x[ybc])/att.replace(0,np.nan) if ybc else np.nan
    out["pfr_yac_per_att"]=num(x[yac])/att.replace(0,np.nan) if yac else np.nan
    out["pfr_brk_tkl_per_att"]=num(x[br])/att.replace(0,np.nan) if br else np.nan
    return out.loc[out["week"].between(1,22)].drop_duplicates(["season","week","team","player_short_key"])


def read_ngs(path:Path):
    if not path.exists():return pd.DataFrame()
    x=lower(pd.read_csv(path,low_memory=False));x=x.loc[num(x.get("week",pd.Series(index=x.index,dtype=float))).gt(0)].copy()
    pc=alias(x,["player_display_name","player_name"]);tc=alias(x,["team_abbr","team"])
    if not pc or not tc:return pd.DataFrame()
    out=pd.DataFrame({"season":num(x["season"]),"week":num(x["week"]),"team":x[tc].map(team),"player_short_key":x[pc].map(short_name)})
    amap={
        "efficiency":"ngs_efficiency","percent_attempts_gte_eight_defenders":"ngs_percent_attempts_gte_eight_defenders",
        "avg_time_to_los":"ngs_avg_time_to_los","rush_yards_over_expected_per_att":"ngs_ryoe_per_att",
        "rush_pct_over_expected":"ngs_rush_pct_over_expected","rush_yards_over_expected":"ngs_ryoe",
        "expected_rush_yards":"ngs_expected_rush_yards","rush_attempts":"ngs_rush_attempts",
    }
    for src,dst in amap.items():out[dst]=num(x[src]) if src in x else np.nan
    out["ngs_expected_yards_per_att"]=out["ngs_expected_rush_yards"]/out["ngs_rush_attempts"].replace(0,np.nan)
    return out.drop_duplicates(["season","week","team","player_short_key"])


def rolling_prior(targets,games,keys,metrics):
    hist={}
    for key,g in games.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        hist[key]=g.sort_values(["season","week"])
    rows=[]
    for r in targets.itertuples(index=False):
        key=tuple(getattr(r,k) for k in keys);g=hist.get(key,pd.DataFrame())
        prior=g.loc[(g["season"]<r.season)|((g["season"]==r.season)&(g["week"]<r.week))] if len(g) else pd.DataFrame()
        rec={"season":r.season,"week":r.week};rec.update({k:getattr(r,k) for k in keys})
        for c in metrics:
            vals=num(prior[c]).dropna() if len(prior) and c in prior else pd.Series(dtype=float)
            for n in (3,5):
                q=vals.tail(n);rec[f"{c}_avg{n}"]=q.mean() if len(q) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def add_offense(trace,pbp,pfr,ngs):
    x=trace.copy();pg=player_pbp_games(pbp);tg=team_pbp_games(pbp)
    if len(pg):
        targets=x[["season","week","player_short_key"]].drop_duplicates();metrics=[c for c in pg if c not in {"season","week","team","player_short_key","actual_player_explosive20"}]
        rp=rolling_prior(targets,pg,["player_short_key"],metrics);x=x.merge(rp,on=["season","week","player_short_key"],how="left",validate="many_to_one")
        actual=pg[["season","week","team","player_short_key","actual_player_explosive20"]].drop_duplicates();x=x.merge(actual,on=["season","week","team","player_short_key"],how="left",validate="many_to_one")
    else:x["actual_player_explosive20"]=np.nan
    targets=x[["season","week","team"]].drop_duplicates();metrics=[c for c in tg if c not in {"season","week","team"}]
    rt=rolling_prior(targets,tg,["team"],metrics);x=x.merge(rt,on=["season","week","team"],how="left",validate="many_to_one")
    if len(pfr):
        targets=x[["season","week","player_short_key"]].drop_duplicates();metrics=[c for c in pfr if c not in {"season","week","team","player_short_key"}]
        rpfr=rolling_prior(targets,pfr,["player_short_key"],metrics);x=x.merge(rpfr,on=["season","week","player_short_key"],how="left",validate="many_to_one")
        pt=pfr.groupby(["season","week","team"],as_index=False).agg(pfr_att=("pfr_att","sum"),pfr_ybc_num=("pfr_ybc_per_att",lambda s:np.nan),pfr_yac_num=("pfr_yac_per_att",lambda s:np.nan))
        # Weighted team PFR contact metrics.
        p2=pfr.copy();p2["ybc_num"]=p2["pfr_ybc_per_att"]*p2["pfr_att"];p2["yac_num"]=p2["pfr_yac_per_att"]*p2["pfr_att"]
        pt=p2.groupby(["season","week","team"],as_index=False).agg(att=("pfr_att","sum"),ybc=("ybc_num","sum"),yac=("yac_num","sum"))
        pt["team_pfr_ybc_per_att"]=pt["ybc"]/pt["att"].replace(0,np.nan);pt["team_pfr_yac_per_att"]=pt["yac"]/pt["att"].replace(0,np.nan)
        rtp=rolling_prior(x[["season","week","team"]].drop_duplicates(),pt,["team"],["team_pfr_ybc_per_att","team_pfr_yac_per_att"]);x=x.merge(rtp,on=["season","week","team"],how="left",validate="many_to_one")
    if len(ngs):
        targets=x[["season","week","player_short_key"]].drop_duplicates();metrics=[c for c in ngs if c not in {"season","week","team","player_short_key"}]
        rngs=rolling_prior(targets,ngs,["player_short_key"],metrics);x=x.merge(rngs,on=["season","week","player_short_key"],how="left",validate="many_to_one")
    return x


def pct_unique(x,col,key,sign=1):
    if col not in x:return pd.Series(np.nan,index=x.index)
    tmp=x[["season","week",key,col]].drop_duplicates(["season","week",key]).copy();tmp[col]=num(tmp[col])
    v=tmp[col] if sign>0 else -tmp[col];tmp["pct"]=v.groupby([tmp["season"],tmp["week"]]).rank(pct=True,method="average")
    idx=pd.MultiIndex.from_frame(x[["season","week",key]]);lut=tmp.set_index(["season","week",key])["pct"]
    return pd.Series(lut.reindex(idx).to_numpy(),index=x.index)


def family_score(x,specs,entity):
    parts=[]
    for col,sign in specs:
        if col in x and num(x[col]).notna().sum()>=30:parts.append(pct_unique(x,col,entity,sign))
    if not parts:return pd.Series(np.nan,index=x.index),pd.Series(0,index=x.index)
    z=pd.concat(parts,axis=1);return z.mean(axis=1,skipna=True),z.notna().sum(axis=1)


def add_scores(x):
    x=x.copy()
    for name,spec in OFF_FAMILIES.items():
        entity="player_short_key" if name.startswith("off_player") else "team";score,n=family_score(x,spec,entity);x[name]=score;x[name+"_n"]=n;x.loc[n.lt(2),name]=np.nan
    for name,spec in DEF_FAMILIES.items():
        score,n=family_score(x,spec,"opponent");x[name]=score;x[name+"_n"]=n;x.loc[n.lt(2),name]=np.nan
    role_spec=[("rb_carries_avg5",1),("rb_rb_share_avg5",1),("rb_15plus_rate5",1),("rb_20plus_rate5",1),("team_top1_share_avg5",1),("team_rb_used_avg5",-1)]
    x["off_role_opportunity_score"],x["off_role_opportunity_score_n"]=family_score(x,role_spec,"player_short_key")
    def prod(a,b):return num(x.get(a,pd.Series(index=x.index,dtype=float)))*num(x.get(b,pd.Series(index=x.index,dtype=float)))
    x["mx_player_efficiency_x_def_efficiency"]=prod("off_player_efficiency_score","def_run_efficiency_score")
    x["mx_player_explosive_x_def_explosive"]=prod("off_player_explosive_score","def_explosive_vulnerability_score")
    x["mx_player_environment_x_def_efficiency"]=prod("off_player_environment_score","def_run_efficiency_score")
    x["mx_role_x_rb_specific"]=prod("off_role_opportunity_score","def_rb_specific_vulnerability_score")
    x["mx_team_strength_x_def_efficiency"]=prod("off_team_rush_strength_score","def_run_efficiency_score")
    x["mx_team_structure_x_def_resistance"]=prod("off_team_structure_score","def_resistance_weakness_score")
    x["mx_short_redzone_x_def_redzone"]=prod("off_short_redzone_score","def_redzone_vulnerability_score")
    x["mx_team_explosive_x_def_explosive"]=prod("off_team_explosive_score","def_explosive_vulnerability_score")
    left=num(x.get("team_pbp_left_share_avg5",pd.Series(index=x.index,dtype=float)));mid=num(x.get("team_pbp_middle_share_avg5",pd.Series(index=x.index,dtype=float)));right=num(x.get("team_pbp_right_share_avg5",pd.Series(index=x.index,dtype=float)))
    dl=num(x.get("def_left_ypa_allowed_avg5",pd.Series(index=x.index,dtype=float)));dm=num(x.get("def_middle_ypa_allowed_avg5",pd.Series(index=x.index,dtype=float)));dr=num(x.get("def_right_ypa_allowed_avg5",pd.Series(index=x.index,dtype=float)))
    den=left.fillna(0)+mid.fillna(0)+right.fillna(0);x["_directional_raw"]=np.where(den.gt(0),(left.fillna(0)*dl.fillna(0)+mid.fillna(0)*dm.fillna(0)+right.fillna(0)*dr.fillna(0))/den,np.nan);x["mx_directional"]=pct_unique(x,"_directional_raw","team",1)
    x["mx_shotgun"]=num(x.get("team_pbp_shotgun_rush_rate_avg5",pd.Series(index=x.index,dtype=float)))*num(x.get("def_shotgun_ypa_allowed_avg5",pd.Series(index=x.index,dtype=float)))
    x["actual_ypc"]=np.where(num(x["actual_carries"]).gt(0),num(x["actual_rush_yards"])/num(x["actual_carries"]),np.nan)
    for c in (15,20,25):x[f"actual_carry_{c}plus"]=num(x["actual_carries"]).ge(c).astype(int)
    for c in (75,100):x[f"actual_rush_{c}plus"]=num(x["actual_rush_yards"]).ge(c).astype(int)
    return x


def ridge():return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=20.0))])
def logit():return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(C=.20,max_iter=2000,random_state=95))])

def reg_metrics(a,p):
    z=pd.DataFrame({"a":num(a),"p":pd.Series(p,index=a.index)}).dropna();
    if z.empty:return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan}
    e=z.p-z.a;return {"n":len(z),"mae":e.abs().mean(),"rmse":math.sqrt(float(np.square(e).mean())),"bias":e.mean(),"corr":z.a.corr(z.p) if z.a.nunique()>1 and z.p.nunique()>1 else np.nan}


def feature_families(x):
    role=[c for c in ROLE_FEATURES if c in x and num(x[c]).notna().sum()>=50]
    off=[c for c in OFF_FAMILIES if c in x and num(x[c]).notna().sum()>=50]
    deff=[c for c in DEF_FAMILIES if c in x and num(x[c]).notna().sum()>=50]
    inter=[c for c in INTERACTIONS if c in x and num(x[c]).notna().sum()>=50]
    return {"role_baseline":role,"role_plus_offense":role+off,"role_offense_defense":role+off+deff,"full_matchup_interactions":role+off+deff+inter}


def fit_models(x):
    fams=feature_families(x);splits=[("train_2023_test_2024",[2023],2024),("train_2023_24_test_2025",[2023,2024],2025)]
    regs=[("carries","actual_carries",0),("rush_yards","actual_rush_yards",0),("rush_rec_yards","actual_rush_rec_yards",0),("ypc","actual_ypc",3)]
    clss=[("carry_15plus_auc","actual_carry_15plus"),("carry_20plus_auc","actual_carry_20plus"),("carry_25plus_auc","actual_carry_25plus"),("rush_75plus_auc","actual_rush_75plus"),("rush_100plus_auc","actual_rush_100plus"),("explosive20_auc","actual_player_explosive20")]
    rows=[];coefs=[];predrows=[]
    for split,trseas,teseas in splits:
        tr0=x.loc[x.season.isin(trseas)&x.pregame_role.ne("unknown")].copy();te0=x.loc[x.season.eq(teseas)&x.pregame_role.ne("unknown")].copy()
        for fam,feats in fams.items():
            feats=[c for c in feats if c in tr0 and num(tr0[c]).notna().any()]
            Xtr0=tr0[feats].apply(pd.to_numeric,errors="coerce");Xte0=te0[feats].apply(pd.to_numeric,errors="coerce")
            for tn,tc,minc in regs:
                mtr=num(tr0.actual_carries).ge(minc) if minc else pd.Series(True,index=tr0.index);mte=num(te0.actual_carries).ge(minc) if minc else pd.Series(True,index=te0.index)
                tr=tr0.loc[mtr];te=te0.loc[mte];Xtr=Xtr0.loc[mtr];Xte=Xte0.loc[mte];y=num(tr[tc]);valid=y.notna()
                if valid.sum()<50 or len(te)<20:continue
                model=ridge();model.fit(Xtr.loc[valid],y.loc[valid]);pred=model.predict(Xte);met=reg_metrics(te[tc],pred)
                rows.append({"split":split,"train_seasons":",".join(map(str,trseas)),"test_season":teseas,"family":fam,"target":tn,"feature_count":len(feats),**met})
                if tn in {"carries","rush_yards","ypc"}:
                    for f,v in zip(feats,np.ravel(model.named_steps["model"].coef_)):coefs.append({"split":split,"family":fam,"target":tn,"feature":f,"standardized_coefficient":float(v),"abs_coefficient":abs(float(v))})
                if tn in {"carries","rush_yards","rush_rec_yards"}:
                    for idx,p in zip(te.index,pred):predrows.append({"split":split,"test_season":teseas,"row_index":int(idx),"family":fam,"target":tn,"prediction":float(p)})
            for tn,tc in clss:
                if tc not in tr0 or tc not in te0:continue
                ytr=num(tr0[tc]);yte=num(te0[tc]);valid=ytr.notna();validte=yte.notna()
                if valid.sum()<50 or validte.sum()<20 or ytr.loc[valid].nunique()<2 or yte.loc[validte].nunique()<2:continue
                model=logit();model.fit(Xtr0.loc[valid],ytr.loc[valid].astype(int));prob=model.predict_proba(Xte0.loc[validte])[:,1]
                rows.append({"split":split,"train_seasons":",".join(map(str,trseas)),"test_season":teseas,"family":fam,"target":tn,"feature_count":len(feats),"n":int(validte.sum()),"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":float(roc_auc_score(yte.loc[validte].astype(int),prob))})
    return pd.DataFrame(rows),pd.DataFrame(coefs),pd.DataFrame(predrows)


def incremental(results):
    rows=[]
    order=["role_baseline","role_plus_offense","role_offense_defense","full_matchup_interactions"]
    for (split,target),g in results.groupby(["split","target"]):
        for a,b in zip(order[:-1],order[1:]):
            ga=g.loc[g.family.eq(a)];gb=g.loc[g.family.eq(b)]
            if ga.empty or gb.empty:continue
            if target.endswith("_auc"):gain=float(gb.iloc[0].corr-ga.iloc[0].corr);metric="auc_gain"
            else:gain=float(ga.iloc[0].mae-gb.iloc[0].mae);metric="mae_gain"
            rows.append({"split":split,"target":target,"from_family":a,"to_family":b,"metric":metric,"incremental_gain":gain})
    return pd.DataFrame(rows)


def slices(x,preds):
    rows=[]
    base=x.copy();base["carry_slice"]=pd.cut(num(base.actual_carries),[-1,5,10,14,19,24,999],labels=["0_5","6_10","11_14","15_19","20_24","25_plus"])
    base["def_slice"]=base.get("def_bucket","unknown")
    pe=preds.loc[preds.test_season.eq(2025)&preds.target.isin(["carries","rush_yards"])].copy()
    for (fam,targ),g in pe.groupby(["family","target"]):
        mp=g.set_index("row_index")["prediction"];q=base.loc[base.index.isin(mp.index)].copy();q["pred"]=mp.reindex(q.index);actual_col="actual_carries" if targ=="carries" else "actual_rush_yards"
        groups={"all":pd.Series(True,index=q.index)}
        for c in q.carry_slice.dropna().unique():groups[f"carry_{c}"]=q.carry_slice.eq(c)
        for r in ["workhorse","strong_starter","committee","light"]:groups[f"role_{r}"]=q.pregame_role.eq(r)
        for d in ["bottom8_weak","middle16","top8_strong"]:groups[f"def_{d}"]=q.def_slice.eq(d)
        good=num(q.get("off_player_efficiency_score",pd.Series(index=q.index,dtype=float))).ge(.67);baddef=num(q.get("def_run_efficiency_score",pd.Series(index=q.index,dtype=float))).ge(.67);strongteam=num(q.get("off_team_rush_strength_score",pd.Series(index=q.index,dtype=float))).ge(.67)
        groups["good_rb_x_weak_def"]=good&baddef;groups["strong_rush_offense_x_weak_def"]=strongteam&baddef
        for name,mask in groups.items():
            z=q.loc[mask,[actual_col,"pred"]].dropna()
            if len(z)<8:continue
            e=z.pred-z[actual_col];rows.append({"family":fam,"target":targ,"slice":name,"n":len(z),"mae":e.abs().mean(),"bias":e.mean(),"corr":z[actual_col].corr(z.pred) if z[actual_col].nunique()>1 and z.pred.nunique()>1 else np.nan})
    return pd.DataFrame(rows)


def audit(x,pfr,ngs):
    rows=[
        ("Player carry/share/workload history","M95A/M91",1,"Used in role baseline."),
        ("Player rush EPA/success/explosive rates","nflverse PBP",1,"Pregame rolling 3/5."),
        ("Player yards before contact","PFR advanced weekly via nflverse",int(len(pfr)>0),"YBC/att, leakage-safe rolling."),
        ("Player yards after contact","PFR advanced weekly via nflverse",int(len(pfr)>0),"YAC/att, leakage-safe rolling."),
        ("Broken tackles","PFR advanced weekly via nflverse",int(len(pfr)>0),"Broken tackles per attempt."),
        ("RYOE / expected rush yards","NFL Next Gen Stats via nflverse",int(len(ngs)>0),"Weekly NGS; RYOE/att separates player from expected environment."),
        ("8+ defenders in box","NFL Next Gen Stats via nflverse",int(len(ngs)>0),"Player-level stacked-box frequency."),
        ("Time to line of scrimmage","NFL Next Gen Stats via nflverse",int(len(ngs)>0),"Player-level TLOS."),
        ("Team YBC/YAC rushing environment","PFR advanced weekly aggregated",int(len(pfr)>0),"Useful blocking/environment proxy; not identical to OL win rate."),
        ("Run-block win rate","ESPN tracking",0,"Public season snapshots exist, but no certified historical weekly feed was wired into M95B; using end-season values would leak."),
        ("Adjusted line yards / second-level yards","external OL analytics",0,"No certified free weekly historical feed wired yet."),
        ("Detailed run concepts","premium/manual charting",0,"PBP direction is used, but it does not truthfully identify zone/duo/power/counter."),
        ("DL/LB individual matchup and injury strength","roster/injury + personnel model",0,"Historical injuries exist elsewhere but a certified player-strength aggregation is not yet built."),
        ("FTN box/motion/RPO/backfield charting","FTN via nflverse",0,"Free 2022+ source found; reserved for next scheme/personnel extension to avoid changing this pre-specified M95B after design lock."),
    ]
    return pd.DataFrame(rows,columns=["metric_family","source","available_in_m95b","notes"])


def disposition(results,inc):
    r25=results.loc[results.split.eq("train_2023_24_test_2025")]
    def metric(f,t,col="mae"):
        q=r25.loc[r25.family.eq(f)&r25.target.eq(t)];return float(q.iloc[0][col]) if len(q) else np.nan
    rb=metric("role_baseline","rush_yards");ro=metric("role_plus_offense","rush_yards");rod=metric("role_offense_defense","rush_yards");full=metric("full_matchup_interactions","rush_yards")
    cb=metric("role_baseline","carries");cf=metric("full_matchup_interactions","carries")
    r24=results.loc[results.split.eq("train_2023_test_2024")];q24=r24.loc[r24.family.eq("full_matchup_interactions")&r24.target.eq("rush_yards")];b24=r24.loc[r24.family.eq("role_baseline")&r24.target.eq("rush_yards")]
    forward24=bool(len(q24)&len(b24)&(float(q24.iloc[0].mae)<float(b24.iloc[0].mae)))
    signal=bool(np.isfinite(full) and np.isfinite(rb) and full<rb and np.isfinite(ro) and full<ro and (not np.isfinite(cf) or not np.isfinite(cb) or cf<=cb+.10))
    advance=signal and forward24
    return pd.DataFrame([{"rush_yards_role_baseline_2025":rb,"rush_yards_role_plus_offense_2025":ro,"rush_yards_role_offense_defense_2025":rod,"rush_yards_full_matchup_2025":full,"carries_role_baseline_2025":cb,"carries_full_matchup_2025":cf,"full_beats_role_2024":int(forward24),"stable_incremental_matchup_signal":int(signal),"disposition":"ADVANCE_M95B_MATCHUP_ARCHITECTURE" if advance else "RETAIN_M95A_TRUTH_ONLY","production_change":0}])


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--m95a-root",type=Path,required=True);ap.add_argument("--pbp-root",type=Path,required=True);ap.add_argument("--pfr-root",type=Path,required=True);ap.add_argument("--ngs-file",type=Path,required=True);ap.add_argument("--out-dir",type=Path,default=Path("data/backtests/rb_m95b"));a=ap.parse_args()
    trace=read_trace(a.m95a_root);pbp=read_pbp(a.pbp_root);pfr=read_pfr(a.pfr_root);ngs=read_ngs(a.ngs_file)
    x=add_offense(trace,pbp,pfr,ngs);x=add_scores(x)
    results,coefs,preds=fit_models(x);inc=incremental(results);sl=slices(x,preds);au=audit(x,pfr,ngs);disp=disposition(results,inc)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    x.to_csv(a.out_dir/"m95b_rb_matchup_trace.csv",index=False);results.to_csv(a.out_dir/"m95b_model_comparison.csv",index=False);inc.to_csv(a.out_dir/"m95b_incremental_gain.csv",index=False);coefs.sort_values("abs_coefficient",ascending=False).to_csv(a.out_dir/"m95b_standardized_coefficients.csv",index=False);preds.to_csv(a.out_dir/"m95b_prediction_trace.csv",index=False);sl.to_csv(a.out_dir/"m95b_slice_metrics.csv",index=False);au.to_csv(a.out_dir/"m95b_feature_audit.csv",index=False);disp.to_csv(a.out_dir/"m95b_disposition.csv",index=False)
    print("[m95b] feature audit\n",au.to_string(index=False));print("\n[m95b] model comparison\n",results.to_string(index=False));print("\n[m95b] incremental gain\n",inc.to_string(index=False));print("\n[m95b] 2025 slices\n",sl.loc[sl.slice.isin(["all","carry_20_24","carry_25_plus","role_workhorse","def_bottom8_weak","def_top8_strong","good_rb_x_weak_def","strong_rush_offense_x_weak_def"])].to_string(index=False));print("\n[m95b] disposition\n",disp.to_string(index=False));return 0

if __name__=="__main__":raise SystemExit(main())
