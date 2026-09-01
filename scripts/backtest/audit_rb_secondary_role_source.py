#!/usr/bin/env python3
"""RB STACK6 / ND3 source audit: strictly prior-game secondary-back role information.

Diagnostic only. Target-game participation is postgame truth and is NEVER used as
pregame input. This script measures whether nflverse participation + PBP can
reconstruct player-specific situational RB roles from completed prior games.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

SITUATIONS = [
    "early_down", "third_down", "third_long", "two_minute", "short_yardage",
    "red_zone", "inside10", "inside5", "shotgun", "under_center",
]


def to_pd(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def lower(x):
    z = x.copy(); z.columns = [str(c).strip().lower() for c in z.columns]; return z


def num(x): return pd.to_numeric(x, errors="coerce")


def split_list(v):
    s = str(v or "").strip()
    if not s or s.lower() in {"nan", "none", "<na>"}: return []
    return [q.strip() for q in re.split(r"[;,]", s) if q.strip()]


def clean_name(v):
    s = "" if pd.isna(v) else str(v)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", s)


def find_join_keys(part, pbp):
    for keys in (["nflverse_game_id", "play_id"], ["old_game_id", "play_id"], ["game_id", "play_id"]):
        if all(k in part.columns and k in pbp.columns for k in keys): return list(keys)
    raise RuntimeError("STACK6 no common game/play key")


def personnel_code(v):
    s = str(v or "").upper(); vals = {}
    for label in ("RB", "TE", "WR"):
        m = re.search(rf"(\d+)\s*{label}", s); vals[label] = int(m.group(1)) if m else 0
    return f"{vals['RB']}{vals['TE']}" if any(vals.values()) else ""


def load_sources(seasons):
    import nflreadpy as nfl
    return lower(to_pd(nfl.load_participation(seasons=seasons))), lower(to_pd(nfl.load_pbp(seasons=seasons)))


def derive_game_fields(x):
    z = x.copy()
    if "season" not in z or "week" not in z:
        gid = next((c for c in ["nflverse_game_id", "game_id"] if c in z), None)
        if gid:
            p = z[gid].astype("string").str.split("_", expand=True)
            if "season" not in z: z["season"] = num(p[0])
            if "week" not in z and p.shape[1] > 1: z["week"] = num(p[1])
    return z


def build_join(part, pbp):
    part = derive_game_fields(part); keys = find_join_keys(part, pbp)
    wanted = [*keys, "season", "week", "season_type", "posteam", "possession_team",
              "rush_attempt", "rusher_player_id", "rusher_id", "down", "ydstogo",
              "qtr", "quarter_seconds_remaining", "yardline_100", "shotgun"]
    right = pbp[[c for c in wanted if c in pbp.columns]].drop_duplicates(keys)
    j = part.merge(right, on=keys, how="inner", suffixes=("", "_pbp"), validate="one_to_one")
    if "season_type" in j:
        reg = j[j["season_type"].astype(str).str.upper().eq("REG")]
        if len(reg): j = reg.copy()
    team_col = "posteam" if "posteam" in j else "possession_team"
    j["team"] = j[team_col].map(canon_team)
    j = j[j.team.ne("")].copy()
    j["season"] = num(j["season"]).astype(int); j["week"] = num(j["week"]).astype(int)
    return j, keys


def explode_rb_plays(j):
    player_col = "offense_players" if "offense_players" in j else None
    pos_col = "offense_positions" if "offense_positions" in j else None
    name_col = "offense_names" if "offense_names" in j else None
    if not player_col or not pos_col: raise RuntimeError("STACK6 participation missing offense_players/offense_positions")
    rows=[]; aligned=0; eligible=0; named=0
    for _, r in j.iterrows():
        ids=split_list(r[player_col]); pos=split_list(r[pos_col]); names=split_list(r[name_col]) if name_col else []
        if ids or pos: eligible += 1
        if len(ids)!=len(pos) or not ids: continue
        aligned += 1
        if names and len(names)==len(ids): named += 1
        for k,(pid,pp) in enumerate(zip(ids,pos)):
            if str(pp).upper() not in {"RB","FB"}: continue
            nm = names[k] if names and len(names)==len(ids) else ""
            rows.append({
                "season":int(r["season"]),"week":int(r["week"]),"team":r["team"],
                "player_id":str(pid),"player_name":nm,"player_clean_key":clean_name(nm),
                "down":num(pd.Series([r.get("down")])).iloc[0],
                "ydstogo":num(pd.Series([r.get("ydstogo")])).iloc[0],
                "qtr":num(pd.Series([r.get("qtr")])).iloc[0],
                "qsec":num(pd.Series([r.get("quarter_seconds_remaining")])).iloc[0],
                "yardline_100":num(pd.Series([r.get("yardline_100")])).iloc[0],
                "rush_attempt":num(pd.Series([r.get("rush_attempt")])).fillna(0).iloc[0],
                "rusher_player_id":str(r.get("rusher_player_id", r.get("rusher_id", "")) or ""),
                "shotgun":num(pd.Series([r.get("shotgun")])).iloc[0],
                "offense_formation":str(r.get("offense_formation", "") or ""),
                "offense_personnel":str(r.get("offense_personnel", "") or ""),
            })
    out=pd.DataFrame(rows)
    parse={"eligible_plays":eligible,"aligned_plays":aligned,"aligned_rate":aligned/max(eligible,1),
           "named_aligned_plays":named,"named_rate":named/max(aligned,1)}
    return out, parse


def add_flags(x):
    z=x.copy(); d=num(z.down); y=num(z.ydstogo); q=num(z.qtr); qs=num(z.qsec); yl=num(z.yardline_100)
    z["early_down"] = d.le(2)
    z["third_down"] = d.eq(3)
    z["third_long"] = d.eq(3) & y.ge(7)
    z["two_minute"] = q.isin([2,4]) & qs.le(120)
    z["short_yardage"] = d.notna() & y.le(2)
    z["red_zone"] = yl.le(20)
    z["inside10"] = yl.le(10)
    z["inside5"] = yl.le(5)
    form=z.offense_formation.astype("string").str.upper().fillna("")
    z["shotgun"] = num(z.shotgun).eq(1) | form.str.contains("SHOTGUN",na=False)
    z["under_center"] = ~z.shotgun.astype(bool)
    z["personnel_code"] = z.offense_personnel.map(personnel_code)
    z["is_rusher"] = z.rush_attempt.eq(1) & z.player_id.eq(z.rusher_player_id)
    return z


def aggregate_roles(x):
    rows=[]
    for (s,w,t,pid),g in x.groupby(["season","week","team","player_id"],sort=True):
        team_plays = x[(x.season==s)&(x.week==w)&(x.team==t)].index.nunique()
        rec={"season":s,"week":w,"team":t,"player_id":pid,"player_name":g.player_name.replace("",np.nan).dropna().iloc[0] if g.player_name.replace("",np.nan).notna().any() else "",
             "player_clean_key":g.player_clean_key.replace("",np.nan).dropna().iloc[0] if g.player_clean_key.replace("",np.nan).notna().any() else "",
             "rb_onfield_plays":len(g),"rb_onfield_share_proxy":len(g)/max(team_plays,1),"rush_attempts_owned":int(g.is_rusher.sum())}
        for f in SITUATIONS:
            den=int(g[f].sum()); rec[f"{f}_onfield_plays"]=den
            rec[f"{f}_share_of_player_plays"]=den/max(len(g),1)
            rec[f"{f}_rushes_owned"]=int((g[f]&g.is_rusher).sum())
        for code in ["11","12","21","22"]:
            m=g.personnel_code.eq(code); rec[f"personnel_{code}_onfield_share"]=float(m.mean()) if len(g) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def add_team_rotation(pg):
    z=pg.copy(); z["team_rb_count"] = z.groupby(["season","week","team"])["player_id"].transform("nunique")
    total=z.groupby(["season","week","team"])["rb_onfield_plays"].transform("sum").replace(0,np.nan)
    z["rb_presence_share"] = z.rb_onfield_plays/total
    z["team_rb_hhi"] = z.groupby(["season","week","team"])["rb_presence_share"].transform(lambda s: float(np.square(s).sum()))
    z["team_rb_rank"] = z.groupby(["season","week","team"])["rb_onfield_plays"].rank(method="first",ascending=False)
    return z


def lag_features(pg):
    z=pg.sort_values(["team","player_id","season","week"]).copy()
    feat=[c for c in z.columns if c.endswith("_share_of_player_plays") or c.startswith("personnel_") or c in ["rb_onfield_share_proxy","rush_attempts_owned","team_rb_count","team_rb_hhi","team_rb_rank"]]
    for c in feat:
        z[f"prior1_{c}"]=z.groupby(["team","player_id"],sort=False)[c].shift(1)
        z[f"prior3_{c}"]=z.groupby(["team","player_id"],sort=False)[c].transform(lambda s: s.shift(1).rolling(3,min_periods=1).mean())
    z["target_order"] = num(z["season"]) * 100 + num(z["week"])
    z["feature_source_max_order"] = z.groupby(["team","player_id"],sort=False)["target_order"].shift(1)
    prior_order = num(z.feature_source_max_order)
    z["leakage_safe_prior_only"] = (prior_order.isna() | prior_order.lt(num(z.target_order))).astype(int)
    return z, feat


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seasons",default="2024,2025"); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True); seasons=[int(x) for x in a.seasons.split(",")]
    part,pbp=load_sources(seasons); j,keys=build_join(part,pbp); rb,parse=explode_rb_plays(j); rb=add_flags(rb); pg=add_team_rotation(aggregate_roles(rb)); lag,feat=lag_features(pg)
    part_keys=part[keys].dropna().drop_duplicates(); pbp_keys=pbp[keys].dropna().drop_duplicates(); match=len(part_keys.merge(pbp_keys,on=keys,how="inner"))/max(len(part_keys),1)
    situation_cov={f:float(rb[f].notna().mean()) for f in SITUATIONS}
    rusher_match=float(rb.loc[rb.rush_attempt.eq(1),"rusher_player_id"].astype(str).ne("").mean()) if (rb.rush_attempt.eq(1)).any() else np.nan
    source=pd.DataFrame([{
        "participation_rows":len(part),"pbp_rows":len(pbp),"joined_rows":len(j),"play_key_match_rate":match,
        **parse,"rb_play_rows":len(rb),"rb_player_games":len(pg),"rusher_id_present_rate_on_rush_plays":rusher_match,
        "strict_prior_leakage_pass_rate":float(lag.leakage_safe_prior_only.mean()),
        "live_2026_capability":"HISTORICAL_ONLY_POSTSEASON_RELEASE_REQUIRES_LIVE_EQUIVALENT",
        **{f"{k}_flag_coverage":v for k,v in situation_cov.items()}
    }])
    coverage=[]
    prior_cols=[c for c in lag.columns if c.startswith("prior1_") or c.startswith("prior3_")]
    for c in prior_cols:
        coverage.append({"feature":c,"nonnull_rate_all":float(num(lag[c]).notna().mean()),"nonnull_rate_2025":float(num(lag.loc[lag.season.eq(2025),c]).notna().mean())})
    manifest=pd.DataFrame([
        {"family":"prior_game_situational_rb_participation","source":"nflverse_participation+pbp","leakage_safe_if_lagged":1,"live_2026":0,"notes":"historical proof-of-information only; target-game participation forbidden"},
        {"family":"current_week_injury_report","source":"nflverse_injuries/other qualified pregame source","leakage_safe_if_lagged":1,"live_2026":1,"notes":"separate live-capable availability family; not target-game participation"},
    ])
    gate = int(match>=.95 and parse["aligned_rate"]>=.95 and parse["named_rate"]>=.80 and float(lag.leakage_safe_prior_only.mean())==1.0)
    disposition=pd.DataFrame([{"source_gate_pass":gate,"play_join_gate":int(match>=.95),"aligned_parse_gate":int(parse["aligned_rate"]>=.95),"name_gate_preliminary":int(parse["named_rate"]>=.80),"leakage_gate":int(float(lag.leakage_safe_prior_only.mean())==1.0),"next":"STACK6_PLAYER_JOIN_AND_SECONDARY_ROLE_MODEL" if gate else "REPAIR_SOURCE_OR_FIND_ID_BRIDGE_BEFORE_MODEL"}])
    source.to_csv(a.out_dir/"stack6_source_audit.csv",index=False); pd.DataFrame(coverage).to_csv(a.out_dir/"stack6_prior_feature_coverage.csv",index=False); manifest.to_csv(a.out_dir/"stack6_source_manifest.csv",index=False); disposition.to_csv(a.out_dir/"stack6_disposition.csv",index=False); pg.to_csv(a.out_dir/"stack6_rb_player_game_roles.csv",index=False); lag.to_csv(a.out_dir/"stack6_rb_player_game_prior_features.csv",index=False)
    print("=== source ==="); print(source.to_string(index=False)); print("=== disposition ==="); print(disposition.to_string(index=False)); print("=== top coverage ==="); print(pd.DataFrame(coverage).sort_values("nonnull_rate_2025",ascending=False).head(30).to_string(index=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
