"""M95D: rushing environment / scheme / personnel experiment.

Research-only. M95C found that leakage-safe rushing-environment information was
more stable for mean rushing yards than raw recent RB efficiency, while
runner-created metrics were more useful for upside discrimination. M95D asks
whether we can explain/predict that environment better with football structure:
FTN rush charting, participation/box context, formation/personnel and defensive
tackling information.

No sportsbook inputs. No production code changes. The model families and gates
below are fixed before the 2025 forward evaluation is inspected.
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

TARGET_SEASONS=(2023,2024,2025)
SOURCE_SEASONS=(2022,2023,2024,2025)
TEAM_MAP={"ARZ":"ARI","JAC":"JAX","LA":"LAR","STL":"LAR","OAK":"LV","SD":"LAC","WSH":"WAS"}

ROLE_FEATURES=[
 "rb_games_before","rb_carries_avg1","rb_carries_avg3","rb_carries_avg5",
 "rb_rb_share_avg1","rb_rb_share_avg3","rb_rb_share_avg5",
 "rb_15plus_rate3","rb_15plus_rate5","rb_20plus_rate3","rb_20plus_rate5",
 "team_rb_pool_avg3","team_rb_pool_avg5","team_total_rush_avg3","team_total_rush_avg5",
 "team_top1_share_avg3","team_top1_share_avg5","team_rb_used_avg3","team_rb_used_avg5",
 "team_qb_rush_share_avg3","team_qb_rush_share_avg5","home",
]

# Stable environment information carried from M95B/M95C. These are all target-
# week pregame rolling values already present in the frozen M95B trace.
M95C_ENV_CANDIDATES=[
 "pfr_ybc_per_att_avg5","ngs_expected_yards_per_att_avg5",
 "ngs_percent_attempts_gte_eight_defenders_avg5","ngs_avg_time_to_los_avg5",
 "team_pfr_ybc_per_att_avg5","team_pfr_yac_per_att_avg5",
 "team_pbp_ypa_avg5","team_pbp_epa_avg5","team_pbp_success_avg5",
 "team_pbp_first_down_rate_avg5","team_pbp_stuff_rate_avg5",
 "team_pbp_rush_rate_avg5","team_pbp_neutral_rush_rate_avg5",
 "team_pbp_early_down_rush_rate_avg5","team_pbp_plays_avg5",
 "off_player_environment_score","off_team_rush_strength_score",
 "off_team_structure_score","off_team_environment_score",
]

SCHEME_OFF=[
 "off_rush_motion_rate_avg5","off_rush_rpo_rate_avg5","off_rush_shotgun_rate_avg5",
 "off_rush_under_center_rate_avg5","off_rush_backfield_mean_avg5",
 "off_box_faced_avg5","off_heavy_box_faced_rate_avg5",
 "off_11_personnel_rush_rate_avg5","off_12_personnel_rush_rate_avg5",
 "off_singleback_rush_rate_avg5","off_shotgun_form_rush_rate_avg5",
]
SCHEME_DEF=[
 "def_box_avg5","def_heavy_box_rate_avg5","def_light_box_rate_avg5",
 "def_missed_tackle_rate_avg5","def_missed_tackles_pg_avg5",
]
INTERACTIONS=[
 "mx_env_x_def_box","mx_env_x_def_tackle","mx_motion_x_def_box",
 "mx_rpo_x_def_box","mx_ybc_x_def_box","mx_ryoe_x_def_tackle",
 "mx_heavybox_off_x_def","mx_personnel11_x_box",
]


def lower(df):
    x=df.copy();x.columns=[str(c).strip().lower() for c in x.columns];return x

def num(s,default=np.nan):
    z=pd.to_numeric(s,errors="coerce")
    return z.fillna(default) if np.isfinite(default) else z

def team(v):
    x=str(v).upper().strip();return TEAM_MAP.get(x,x)

def short_name(v):
    raw=str(v).lower().strip().replace("'","")
    raw=re.sub(r"[^a-z0-9.\- ]","",raw)
    toks=[t for t in re.split(r"[\s.\-]+",raw) if t and t not in {"jr","sr","ii","iii","iv"}]
    if not toks:return ""
    if len(toks)==1:return re.sub(r"[^a-z0-9]","",toks[0])
    return re.sub(r"[^a-z0-9]","",toks[0][0]+toks[-1])

def alias(df,names):
    for n in names:
        if n in df.columns:return n
    return None

def bool_num(s):
    if pd.api.types.is_bool_dtype(s):return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):return num(s)
    z=s.astype(str).str.strip().str.lower();o=pd.Series(np.nan,index=s.index,dtype=float)
    o.loc[z.isin(["1","true","t","yes","y"])]=1.0;o.loc[z.isin(["0","false","f","no","n"])]=0.0
    return o


def read_trace(root:Path):
    p=next(iter(root.rglob("m95b_rb_matchup_trace.csv")),None)
    if p is None:raise RuntimeError("missing frozen M95B trace")
    x=lower(pd.read_csv(p,low_memory=False));x["season"]=num(x.season).astype(int);x["week"]=num(x.week).astype(int)
    x=x.loc[x.season.isin(TARGET_SEASONS)&x.week.between(1,18)].copy();x["team"]=x.team.map(team);x["opponent"]=x.opponent.map(team)
    if "player_short_key" not in x:x["player_short_key"]=x.player.map(short_name)
    return x.reset_index(drop=True)


def read_parquets(root:Path,prefix:str):
    fs=[]
    for s in SOURCE_SEASONS:
        p=root/f"{prefix}_{s}.parquet"
        if p.exists():
            z=lower(pd.read_parquet(p));z["_source_season"]=s;fs.append(z)
    return pd.concat(fs,ignore_index=True,sort=False) if fs else pd.DataFrame()


def read_pbp(root:Path):return read_parquets(root,"play_by_play")
def read_ftn(root:Path):return read_parquets(root,"ftn_charting")
def read_part(root:Path):return read_parquets(root,"pbp_participation")


def join_ftn_pbp(ftn,pbp):
    if ftn.empty or pbp.empty:return pd.DataFrame(),pd.DataFrame([{"source":"ftn","status":"missing","rows":0,"join_rate":0}])
    gidf=alias(ftn,["nflverse_game_id","game_id"]);pidf=alias(ftn,["nflverse_play_id","play_id"])
    gidp=alias(pbp,["game_id","nflverse_game_id"]);pidp=alias(pbp,["play_id","nflverse_play_id"])
    if not all([gidf,pidf,gidp,pidp]):return pd.DataFrame(),pd.DataFrame([{"source":"ftn","status":"missing_join_keys","rows":len(ftn),"join_rate":0}])
    f=ftn.copy();p=pbp.copy();f["_gid"]=f[gidf].astype(str);f["_pid"]=num(f[pidf]);p["_gid"]=p[gidp].astype(str);p["_pid"]=num(p[pidp])
    keep=[c for c in ["_gid","_pid","posteam","defteam","rush_attempt","qb_kneel"] if c in p]
    m=f.merge(p[keep].drop_duplicates(["_gid","_pid"]),on=["_gid","_pid"],how="left",validate="many_to_one")
    jr=float(m.posteam.notna().mean()) if "posteam" in m else 0.0
    return m,pd.DataFrame([{"source":"ftn","status":"ok" if jr>=.90 else "low_join","rows":len(f),"join_rate":jr}])


def ftn_team_games(ftn_pbp):
    if ftn_pbp.empty:return pd.DataFrame()
    x=ftn_pbp.copy();x["season"]=num(x.get("season"));x["week"]=num(x.get("week"));x["posteam"]=x.get("posteam","").map(team);x["defteam"]=x.get("defteam","").map(team)
    rush=num(x.get("rush_attempt",pd.Series(index=x.index,dtype=float)),0).eq(1);kneel=num(x.get("qb_kneel",pd.Series(index=x.index,dtype=float)),0).eq(1);x=x.loc[rush&~kneel&x.posteam.ne("")].copy()
    if x.empty:return pd.DataFrame()
    motion=alias(x,["is_motion"]);rpo=alias(x,["is_rpo"]);ql=alias(x,["qb_location"]);back=alias(x,["n_offense_backfield"])
    x["_motion"]=bool_num(x[motion]) if motion else np.nan;x["_rpo"]=bool_num(x[rpo]) if rpo else np.nan
    if ql:
        q=x[ql].astype(str).str.upper();x["_shotgun"]=q.str.contains("SHOTGUN",na=False).astype(float);x["_under_center"]=(~q.str.contains("SHOTGUN",na=False)&q.ne("NAN")&q.ne("")).astype(float)
    else:x["_shotgun"]=np.nan;x["_under_center"]=np.nan
    x["_backfield"]=num(x[back]) if back else np.nan
    g=x.groupby(["season","week","posteam","defteam"],as_index=False).agg(
      rush_motion_rate=("_motion","mean"),rush_rpo_rate=("_rpo","mean"),rush_shotgun_rate=("_shotgun","mean"),
      rush_under_center_rate=("_under_center","mean"),rush_backfield_mean=("_backfield","mean"),ftn_rush_plays=("_gid","size"))
    return g.rename(columns={"posteam":"team","defteam":"opponent"})


def participation_team_games(part,pbp):
    if part.empty or pbp.empty:return pd.DataFrame(),pd.DataFrame([{"source":"participation","status":"missing","rows":0,"join_rate":0}])
    gid=alias(part,["nflverse_game_id","game_id"]);pid=alias(part,["play_id","nflverse_play_id"]);gidp=alias(pbp,["game_id","nflverse_game_id"]);pidp=alias(pbp,["play_id","nflverse_play_id"])
    if not all([gid,pid,gidp,pidp]):return pd.DataFrame(),pd.DataFrame([{"source":"participation","status":"missing_join_keys","rows":len(part),"join_rate":0}])
    q=part.copy();p=pbp.copy();q["_gid"]=q[gid].astype(str);q["_pid"]=num(q[pid]);p["_gid"]=p[gidp].astype(str);p["_pid"]=num(p[pidp])
    keep=[c for c in ["_gid","_pid","season","week","posteam","defteam","rush_attempt","qb_kneel"] if c in p]
    m=q.merge(p[keep].drop_duplicates(["_gid","_pid"]),on=["_gid","_pid"],how="left",validate="many_to_one",suffixes=("","_pbp"));jr=float(m.posteam.notna().mean()) if "posteam" in m else 0.0
    m["season"]=num(m.get("season_pbp",m.get("season")));m["week"]=num(m.get("week_pbp",m.get("week")));m["posteam"]=m.posteam.map(team);m["defteam"]=m.defteam.map(team)
    rush=num(m.get("rush_attempt",pd.Series(index=m.index,dtype=float)),0).eq(1);kneel=num(m.get("qb_kneel",pd.Series(index=m.index,dtype=float)),0).eq(1);m=m.loc[rush&~kneel&m.posteam.ne("")].copy()
    if m.empty:return pd.DataFrame(),pd.DataFrame([{"source":"participation","status":"empty_after_rush_filter","rows":len(part),"join_rate":jr}])
    box=alias(m,["defenders_in_box"]);form=alias(m,["offense_formation"]);pers=alias(m,["offense_personnel"])
    m["_box"]=num(m[box]) if box else np.nan;m["_heavy_box"]=(m._box.ge(8)).astype(float).where(m._box.notna());m["_light_box"]=(m._box.le(6)).astype(float).where(m._box.notna())
    if form:
        f=m[form].astype(str).str.upper();m["_singleback"]=f.str.contains("SINGLEBACK",na=False).astype(float);m["_shotgun_form"]=f.str.contains("SHOTGUN",na=False).astype(float)
    else:m["_singleback"]=np.nan;m["_shotgun_form"]=np.nan
    if pers:
        s=m[pers].astype(str).str.upper();m["_11p"]=(s.str.contains(r"1\s*RB",regex=True,na=False)&s.str.contains(r"1\s*TE",regex=True,na=False)&s.str.contains(r"3\s*WR",regex=True,na=False)).astype(float);m["_12p"]=(s.str.contains(r"1\s*RB",regex=True,na=False)&s.str.contains(r"2\s*TE",regex=True,na=False)&s.str.contains(r"2\s*WR",regex=True,na=False)).astype(float)
    else:m["_11p"]=np.nan;m["_12p"]=np.nan
    g=m.groupby(["season","week","posteam","defteam"],as_index=False).agg(box_faced=("_box","mean"),heavy_box_faced_rate=("_heavy_box","mean"),light_box_faced_rate=("_light_box","mean"),singleback_rush_rate=("_singleback","mean"),shotgun_form_rush_rate=("_shotgun_form","mean"),p11_rush_rate=("_11p","mean"),p12_rush_rate=("_12p","mean"),participation_rush_plays=("_gid","size"))
    return g.rename(columns={"posteam":"team","defteam":"opponent"}),pd.DataFrame([{"source":"participation","status":"ok" if jr>=.90 else "low_join","rows":len(part),"join_rate":jr}])


def read_pfr_rush(root:Path):
    fs=[]
    for p in sorted(root.glob("advstats_week_rush_*.csv")):
        z=lower(pd.read_csv(p,low_memory=False));fs.append(z)
    if not fs:return pd.DataFrame()
    x=pd.concat(fs,ignore_index=True,sort=False);sc=alias(x,["season"]);wc=alias(x,["week"]);pc=alias(x,["pfr_player_name","player","player_name"]);tc=alias(x,["team","tm"]);ac=alias(x,["carries","att","rush_att"])
    ybca=alias(x,["rushing_yards_before_contact_avg","ybc_att","ybc_per_att"]);ybct=alias(x,["rushing_yards_before_contact","ybc","yards_before_contact"])
    if not all([sc,wc,pc,tc,ac]):return pd.DataFrame()
    att=num(x[ac]);o=pd.DataFrame({"season":num(x[sc]),"week":num(x[wc]),"team":x[tc].map(team),"player_short_key":x[pc].map(short_name),"actual_pfr_att":att})
    if ybca:o["actual_ybc_per_att"]=num(x[ybca])
    elif ybct:o["actual_ybc_per_att"]=num(x[ybct])/att.replace(0,np.nan)
    else:o["actual_ybc_per_att"]=np.nan
    return o.drop_duplicates(["season","week","team","player_short_key"],keep="last")


def read_pfr_def(root:Path):
    fs=[]
    for p in sorted(root.glob("advstats_week_def_*.csv")):
        try:fs.append(lower(pd.read_csv(p,low_memory=False)))
        except Exception:pass
    if not fs:return pd.DataFrame(),pd.DataFrame([{"source":"pfr_week_def","status":"missing","rows":0,"join_rate":np.nan}])
    x=pd.concat(fs,ignore_index=True,sort=False);sc=alias(x,["season"]);wc=alias(x,["week"]);tc=alias(x,["team","tm"]);mt=alias(x,["missed_tackles","miss_tkl","mtkl","tackles_missed"]);tk=alias(x,["tackles_combined","comb","tackles","total_tackles"])
    if not all([sc,wc,tc,mt]):return pd.DataFrame(),pd.DataFrame([{"source":"pfr_week_def","status":"schema_no_missed_tackle","rows":len(x),"join_rate":np.nan}])
    x["_miss"]=num(x[mt],0);x["_tackles"]=num(x[tk],0) if tk else np.nan;x["team"]=x[tc].map(team);x["season"]=num(x[sc]);x["week"]=num(x[wc])
    g=x.groupby(["season","week","team"],as_index=False).agg(missed_tackles_pg=("_miss","sum"),tackles_pg=("_tackles","sum"))
    g["missed_tackle_rate"]=g.missed_tackles_pg/(g.tackles_pg+g.missed_tackles_pg).replace(0,np.nan)
    return g,pd.DataFrame([{"source":"pfr_week_def","status":"ok","rows":len(x),"join_rate":np.nan}])


def rolling_team_features(g,entity,cols,prefix,window=5):
    if g.empty:return pd.DataFrame()
    q=g.copy();q["season"]=num(q.season).astype(int);q["week"]=num(q.week).astype(int);q[entity]=q[entity].map(team);q=q.sort_values([entity,"season","week"]).copy()
    out=q[["season","week",entity]].copy()
    for c in cols:
        if c not in q:continue
        out[f"{prefix}{c}_avg{window}"]=q.groupby(entity,sort=False)[c].transform(lambda s:num(s).shift(1).rolling(window,min_periods=2).mean())
    return out.drop_duplicates(["season","week",entity],keep="last")


def build_structural_features(ftng,partg,pfrdef):
    # Offense-level game rows.
    off=pd.DataFrame()
    if not ftng.empty:off=ftng.copy()
    if not partg.empty:
        off=partg.copy() if off.empty else off.merge(partg,on=["season","week","team","opponent"],how="outer")
    offmap={"rush_motion_rate":"rush_motion_rate","rush_rpo_rate":"rush_rpo_rate","rush_shotgun_rate":"rush_shotgun_rate","rush_under_center_rate":"rush_under_center_rate","rush_backfield_mean":"rush_backfield_mean","box_faced":"box_faced","heavy_box_faced_rate":"heavy_box_faced_rate","p11_rush_rate":"11_personnel_rush_rate","p12_rush_rate":"12_personnel_rush_rate","singleback_rush_rate":"singleback_rush_rate","shotgun_form_rush_rate":"shotgun_form_rush_rate"}
    if not off.empty:
        for old,new in list(offmap.items()):
            if old in off and old!=new:off=off.rename(columns={old:new})
        offcols=[c for c in offmap.values() if c in off]
        roff=rolling_team_features(off,"team",offcols,"off_")
    else:roff=pd.DataFrame()
    # Defensive box rows are derived from the offense-facing participation records.
    if not partg.empty:
        d=partg.rename(columns={"opponent":"defense","team":"offense"}).copy();dcols=[c for c in ["box_faced","heavy_box_faced_rate","light_box_faced_rate"] if c in d]
        rd=rolling_team_features(d,"defense",dcols,"def_")
        rd=rd.rename(columns={"defense":"opponent","def_box_faced_avg5":"def_box_avg5","def_heavy_box_faced_rate_avg5":"def_heavy_box_rate_avg5","def_light_box_faced_rate_avg5":"def_light_box_rate_avg5"})
    else:rd=pd.DataFrame()
    if not pfrdef.empty:
        rmt=rolling_team_features(pfrdef,"team",[c for c in ["missed_tackle_rate","missed_tackles_pg"] if c in pfrdef],"def_").rename(columns={"team":"opponent"})
        rd=rmt if rd.empty else rd.merge(rmt,on=["season","week","opponent"],how="outer")
    return roff,rd


def add_interactions(x):
    x=x.copy()
    def s(c):return num(x.get(c,pd.Series(index=x.index,dtype=float)))
    # Use the stable pregame environment score if present, otherwise team YBC.
    env=s("off_team_rush_strength_score").fillna(s("team_pfr_ybc_per_att_avg5"));box=s("def_box_avg5");miss=s("def_missed_tackle_rate_avg5")
    x["mx_env_x_def_box"]=env*box;x["mx_env_x_def_tackle"]=env*miss;x["mx_motion_x_def_box"]=s("off_rush_motion_rate_avg5")*box;x["mx_rpo_x_def_box"]=s("off_rush_rpo_rate_avg5")*box;x["mx_ybc_x_def_box"]=s("team_pfr_ybc_per_att_avg5")*box;x["mx_ryoe_x_def_tackle"]=s("ngs_ryoe_per_att_avg5")*miss;x["mx_heavybox_off_x_def"]=s("off_heavy_box_faced_rate_avg5")*s("def_heavy_box_rate_avg5");x["mx_personnel11_x_box"]=s("off_11_personnel_rush_rate_avg5")*box
    return x


def ridge():return Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("model",Ridge(alpha=20.0))])
def logit():return Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("model",LogisticRegression(C=.20,max_iter=2500,random_state=95))])

def metrics(a,p):
    z=pd.DataFrame({"a":num(a),"p":pd.Series(p,index=a.index)}).dropna()
    if not len(z):return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan}
    e=z.p-z.a;return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(math.sqrt(np.square(e).mean())),"bias":float(e.mean()),"corr":float(z.a.corr(z.p)) if z.a.nunique()>1 and z.p.nunique()>1 else np.nan}


def families(x):
    role=[c for c in ROLE_FEATURES if c in x and num(x[c]).notna().sum()>=50]
    env=[c for c in M95C_ENV_CANDIDATES if c in x and num(x[c]).notna().sum()>=50]
    scheme=[c for c in SCHEME_OFF+SCHEME_DEF if c in x and num(x[c]).notna().sum()>=50]
    inter=[c for c in INTERACTIONS if c in x and num(x[c]).notna().sum()>=50]
    return {
      "role_baseline":role,
      "role_plus_m95c_environment":role+env,
      "role_plus_environment_scheme":role+env+scheme,
      "full_environment_matchup":role+env+scheme+inter,
    }


def fit_models(x):
    fam=families(x);splits=[("train_2023_test_2024",[2023],2024),("train_2023_24_test_2025",[2023,2024],2025)]
    regs=[("carries","actual_carries",0),("rush_yards","actual_rush_yards",0),("ypc_8plus","actual_ypc",8),("ybc_per_att_5plus","actual_ybc_per_att",5)]
    cls=[("rush_75plus_auc","actual_rush_75plus"),("rush_100plus_auc","actual_rush_100plus"),("explosive20_auc","actual_player_explosive20")]
    rows=[];pred=[]
    for sn,trseas,teseas in splits:
        tr0=x.loc[x.season.isin(trseas)&x.pregame_role.ne("unknown")].copy();te0=x.loc[x.season.eq(teseas)&x.pregame_role.ne("unknown")].copy()
        for fn,feats in fam.items():
            feats=[c for c in feats if c in tr0 and num(tr0[c]).notna().any()]
            if not feats:continue
            Xtr=tr0[feats].apply(pd.to_numeric,errors="coerce");Xte=te0[feats].apply(pd.to_numeric,errors="coerce")
            for tn,tc,minc in regs:
                if tc not in tr0 or tc not in te0:continue
                mtr=num(tr0.actual_carries).ge(minc) if minc else pd.Series(True,index=tr0.index);mte=num(te0.actual_carries).ge(minc) if minc else pd.Series(True,index=te0.index)
                y=num(tr0.loc[mtr,tc]);valid=y.notna();teidx=te0.index[mte]
                if valid.sum()<50 or len(teidx)<20:continue
                mod=ridge();mod.fit(Xtr.loc[mtr].loc[valid],y.loc[valid]);pp=mod.predict(Xte.loc[teidx]);met=metrics(te0.loc[teidx,tc],pp);rows.append({"split":sn,"train_seasons":",".join(map(str,trseas)),"test_season":teseas,"family":fn,"target":tn,"feature_count":len(feats),**met})
                if tn in {"carries","rush_yards","ybc_per_att_5plus"}:
                    pred.extend({"split":sn,"test_season":teseas,"family":fn,"target":tn,"row_index":int(i),"prediction":float(v)} for i,v in zip(teidx,pp))
            for tn,tc in cls:
                if tc not in tr0 or tc not in te0:continue
                ytr=num(tr0[tc]);yte=num(te0[tc]);vtr=ytr.notna();vte=yte.notna()
                if vtr.sum()<50 or vte.sum()<20 or ytr[vtr].nunique()<2 or yte[vte].nunique()<2:continue
                mod=logit();mod.fit(Xtr.loc[vtr],ytr.loc[vtr].astype(int));pr=mod.predict_proba(Xte.loc[vte])[:,1];rows.append({"split":sn,"train_seasons":",".join(map(str,trseas)),"test_season":teseas,"family":fn,"target":tn,"feature_count":len(feats),"n":int(vte.sum()),"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":float(roc_auc_score(yte.loc[vte].astype(int),pr))})
    return pd.DataFrame(rows),pd.DataFrame(pred),fam


def gain_table(r):
    rows=[];base="role_plus_m95c_environment";cand="full_environment_matchup"
    for (sp,t),g in r.groupby(["split","target"]):
        a=g.loc[g.family.eq(base)];b=g.loc[g.family.eq(cand)]
        if a.empty or b.empty:continue
        if t.endswith("_auc"):gain=float(b.iloc[0].corr-a.iloc[0].corr);metric="auc_gain"
        else:gain=float(a.iloc[0].mae-b.iloc[0].mae);metric="mae_gain"
        rows.append({"split":sp,"target":t,"baseline":base,"candidate":cand,"metric":metric,"gain":gain})
    return pd.DataFrame(rows)


def source_audit(x,ftn_j,part_j,pfrdef,fam):
    rows=[]
    for d in [ftn_j,part_j,pfrdef]:
        if isinstance(d,pd.DataFrame) and len(d):rows.extend(d.to_dict("records"))
    for name,cols in fam.items():
        cov=[float(num(x[c]).notna().mean()) for c in cols if c in x]
        rows.append({"source":f"feature_family:{name}","status":"coverage","rows":len(cols),"join_rate":float(np.mean(cov)) if cov else 0.0})
    return pd.DataFrame(rows)


def disposition(r):
    def val(split,fam,target,col="mae"):
        q=r.loc[(r.split==split)&(r.family==fam)&(r.target==target)];return float(q.iloc[0][col]) if len(q) and pd.notna(q.iloc[0][col]) else np.nan
    b="role_plus_m95c_environment";c="full_environment_matchup";s24="train_2023_test_2024";s25="train_2023_24_test_2025"
    ry24b,ry24c=val(s24,b,"rush_yards"),val(s24,c,"rush_yards");ry25b,ry25c=val(s25,b,"rush_yards"),val(s25,c,"rush_yards")
    yb24b,yb24c=val(s24,b,"ybc_per_att_5plus"),val(s24,c,"ybc_per_att_5plus");yb25b,yb25c=val(s25,b,"ybc_per_att_5plus"),val(s25,c,"ybc_per_att_5plus")
    ca25b,ca25c=val(s25,b,"carries"),val(s25,c,"carries");t25b,t25c=val(s25,b,"rush_100plus_auc","corr"),val(s25,c,"rush_100plus_auc","corr");e25b,e25c=val(s25,b,"explosive20_auc","corr"),val(s25,c,"explosive20_auc","corr")
    stable_mean=bool(np.isfinite(ry24b) and np.isfinite(ry24c) and np.isfinite(ry25b) and np.isfinite(ry25c) and ry24c<ry24b and ry25c<ry25b)
    mech=bool(np.isfinite(yb24b) and np.isfinite(yb24c) and np.isfinite(yb25b) and np.isfinite(yb25c) and yb24c<yb24b and yb25c<yb25b)
    carry_guard=bool(not(np.isfinite(ca25b) and np.isfinite(ca25c)) or ca25c<=ca25b+.10)
    tail=bool((np.isfinite(t25b) and np.isfinite(t25c) and t25c>=t25b+.005) or (np.isfinite(e25b) and np.isfinite(e25c) and e25c>=e25b+.005))
    advance=carry_guard and (stable_mean or (mech and tail))
    return pd.DataFrame([{"rush_yards_env_2024":ry24b,"rush_yards_full_2024":ry24c,"rush_yards_env_2025":ry25b,"rush_yards_full_2025":ry25c,"ybc_env_2024":yb24b,"ybc_full_2024":yb24c,"ybc_env_2025":yb25b,"ybc_full_2025":yb25c,"carries_env_2025":ca25b,"carries_full_2025":ca25c,"rush100_auc_env_2025":t25b,"rush100_auc_full_2025":t25c,"explosive_auc_env_2025":e25b,"explosive_auc_full_2025":e25c,"stable_mean_gain_both_years":int(stable_mean),"stable_ybc_mechanism_both_years":int(mech),"carry_guard":int(carry_guard),"tail_support":int(tail),"disposition":"ADVANCE_M95D_SCHEME_ENVIRONMENT_SIGNAL" if advance else "RETAIN_M95C_ENVIRONMENT_ONLY","production_change":0}])


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--m95b-root",type=Path,required=True);ap.add_argument("--pbp-root",type=Path,required=True);ap.add_argument("--ftn-root",type=Path,required=True);ap.add_argument("--participation-root",type=Path,required=True);ap.add_argument("--pfr-rush-root",type=Path,required=True);ap.add_argument("--pfr-def-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,default=Path("data/backtests/rb_m95d"));a=ap.parse_args()
    x=read_trace(a.m95b_root);pbp=read_pbp(a.pbp_root);ftn=read_ftn(a.ftn_root);part=read_part(a.participation_root)
    fj,fja=join_ftn_pbp(ftn,pbp);ftng=ftn_team_games(fj);partg,pja=participation_team_games(part,pbp);pfrdef,pda=read_pfr_def(a.pfr_def_root);roff,rdef=build_structural_features(ftng,partg,pfrdef)
    if not roff.empty:x=x.merge(roff,on=["season","week","team"],how="left",validate="many_to_one")
    if not rdef.empty:x=x.merge(rdef,on=["season","week","opponent"],how="left",validate="many_to_one")
    pfr=read_pfr_rush(a.pfr_rush_root)
    if not pfr.empty:x=x.merge(pfr,on=["season","week","team","player_short_key"],how="left",validate="many_to_one")
    if "actual_ypc" not in x:x["actual_ypc"]=np.where(num(x.actual_carries).gt(0),num(x.actual_rush_yards)/num(x.actual_carries),np.nan)
    for y in [75,100]:x[f"actual_rush_{y}plus"]=num(x.actual_rush_yards).ge(y).astype(int)
    if "actual_player_explosive20" not in x:x["actual_player_explosive20"]=num(x.actual_rush_yards).ge(20).astype(int) # fallback only; normally frozen trace contains true event label
    x=add_interactions(x);res,preds,fam=fit_models(x);gains=gain_table(res);audit=source_audit(x,fja,pja,pda,fam);disp=disposition(res)
    a.out_dir.mkdir(parents=True,exist_ok=True);x.to_csv(a.out_dir/"m95d_rb_environment_trace.csv",index=False);res.to_csv(a.out_dir/"m95d_model_comparison.csv",index=False);preds.to_csv(a.out_dir/"m95d_prediction_trace.csv",index=False);gains.to_csv(a.out_dir/"m95d_gain_vs_m95c_environment.csv",index=False);audit.to_csv(a.out_dir/"m95d_source_feature_audit.csv",index=False);disp.to_csv(a.out_dir/"m95d_disposition.csv",index=False)
    print("[m95d] disposition\n",disp.to_string(index=False));print("\n[m95d] comparison\n",res.to_string(index=False));print("\n[m95d] gains\n",gains.to_string(index=False));print("\n[m95d] source audit\n",audit.to_string(index=False));return 0

if __name__=="__main__":raise SystemExit(main())
