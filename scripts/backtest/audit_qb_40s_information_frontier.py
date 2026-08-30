#!/usr/bin/env python3
"""Migration 76: QB 40s-MAE information acquisition frontier.

Diagnostic/data-contract only. M76 fits no predictive model. It consumes the
frozen canonical-v3 football-only QB snapshot, quantifies the error recovery
needed to enter the 40s, explains the 2024-vs-2025 difficulty gap, and determines
whether a genuinely new exact-personnel layer is strong enough to authorize M77.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

EXPECTED_ROWS = 884
EXPECTED_SEASONS = {2024: 444, 2025: 440}
EXPECTED_CANONICAL_SHA256 = "c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742"
MARKET_TOKENS = ("market", "spread", "moneyline", "sportsbook", "implied_total", "game_total")
GROUPS = ("OL", "WR_TE_RB", "DB", "PASS_RUSH")

SCENARIOS = [
    ("current", 0.00, 0.00),
    ("att10", .10, 0), ("att15", .15, 0), ("att20", .20, 0),
    ("att25", .25, 0), ("att30", .30, 0), ("att35", .35, 0), ("att40", .40, 0),
    ("ypa10", 0, .10), ("ypa15", 0, .15), ("ypa20", 0, .20),
    ("ypa25", 0, .25), ("ypa30", 0, .30), ("ypa40", 0, .40),
    ("both10_10", .10, .10), ("both15_15", .15, .15),
    ("both15_20", .15, .20), ("both20_15", .20, .15),
    ("both20_20", .20, .20), ("both25_25", .25, .25),
    ("perfect_att", 1, 0), ("perfect_ypa", 0, 1),
]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def clean_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def team_value(v) -> str:
    try:
        return canon_team(v)
    except Exception:
        return str(v).strip().upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metrics(actual, pred) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "tail100": 0}
    e = z.pred - z.actual
    return {
        "n": int(len(z)), "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(e.mean()),
        "corr": float(z.actual.corr(z.pred)) if len(z) > 2 else np.nan,
        "tail100": int(e.abs().ge(100).sum()),
    }


def require_canonical(path: Path) -> pd.DataFrame:
    digest = sha256_bytes(path.read_bytes())
    if digest != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical-v3 SHA drift: {digest}")
    base = lower(pd.read_csv(path, low_memory=False))
    if len(base) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} canonical rows, got {len(base)}")
    counts = {int(k): int(v) for k, v in num(base.season).value_counts().to_dict().items()}
    if counts != EXPECTED_SEASONS:
        raise RuntimeError(f"canonical season counts drifted: {counts}")
    required = {"season", "week", "team", "opponent", "pred_attempts", "actual_attempts", "pred_pass_yards", "actual_pass_yards"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise RuntimeError(f"canonical missing {missing}")
    bad = [c for c in base.columns if any(tok in c for tok in MARKET_TOKENS)]
    if bad:
        raise RuntimeError(f"football/market boundary violated: {bad}")
    base["season"] = num(base.season).astype(int)
    base["week"] = num(base.week).astype(int)
    base["team"] = base.team.map(team_value)
    base["opponent"] = base.opponent.map(team_value)
    return base


def recovered_prediction(base: pd.DataFrame, ar: float, yr: float) -> pd.Series:
    pa, aa = num(base.pred_attempts), num(base.actual_attempts)
    py, ayards = num(base.pred_pass_yards), num(base.actual_pass_yards)
    pypa = py / pa.replace(0, np.nan)
    aypa = ayards / aa.replace(0, np.nan)
    return (pa + ar * (aa - pa)) * (pypa + yr * (aypa - pypa))


def recovery_map(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("combined", base)] + [(str(int(s)), g) for s, g in base.groupby("season")]
    for name, ar, yr in SCENARIOS:
        pred = recovered_prediction(base, ar, yr)
        for label, g in groups:
            rows.append({"scenario": name, "season": label, "attempt_recovery": ar, "ypa_recovery": yr,
                         **metrics(g.actual_pass_yards, pred.loc[g.index])})
    for r in np.linspace(0, 1, 101):
        m = metrics(base.actual_pass_yards, recovered_prediction(base, float(r), float(r)))
        if m["mae"] < 50:
            rows.append({"scenario": "min_equal_recovery_below_50", "season": "combined",
                         "attempt_recovery": float(r), "ypa_recovery": float(r), **m})
            break
    return pd.DataFrame(rows)


def difficulty(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in base.groupby("season"):
        pa, aa = num(g.pred_attempts), num(g.actual_attempts)
        py, ayards = num(g.pred_pass_yards), num(g.actual_pass_yards)
        pypa = py / pa.replace(0, np.nan)
        aypa = ayards / aa.replace(0, np.nan)
        cur = metrics(ayards, py)
        oa = metrics(ayards, aa * pypa)
        oy = metrics(ayards, pa * aypa)
        rows.append({
            "season": int(season), **cur,
            "attempt_mae": float((pa-aa).abs().mean()), "attempt_bias": float((pa-aa).mean()),
            "attempt_8plus": int((pa-aa).abs().ge(8).sum()), "attempt_10plus": int((pa-aa).abs().ge(10).sum()),
            "ypa_mae": float((pypa-aypa).abs().mean()), "ypa_bias": float((pypa-aypa).mean()),
            "ypa_1p5plus": int((pypa-aypa).abs().ge(1.5).sum()), "ypa_2plus": int((pypa-aypa).abs().ge(2).sum()),
            "oracle_attempts_mae": oa["mae"], "attempt_oracle_headroom": cur["mae"]-oa["mae"],
            "oracle_ypa_mae": oy["mae"], "ypa_oracle_headroom": cur["mae"]-oy["mae"],
        })
    out = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    if len(out) == 2:
        out["mae_gap_vs_other_season"] = [float(out.iloc[0].mae-out.iloc[1].mae), float(out.iloc[1].mae-out.iloc[0].mae)]
    return out


def download_table(name: str, season: int | str, urls: list[str], snapshots: list[dict], as_of: str) -> pd.DataFrame:
    errors = []
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "m76-data-contract-audit"})
            with urlopen(req, timeout=90) as r:
                data = r.read()
                final_url = r.geturl()
            if url.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(data))
            else:
                df = pd.read_csv(io.BytesIO(data), low_memory=False)
            df = lower(df)
            if isinstance(season, int):
                if "season" in df.columns:
                    df = df.loc[num(df.season).eq(season)].copy()
                else:
                    df["season"] = season
            snapshots.append({"source": name, "season": season, "url": final_url, "rows": len(df),
                              "sha256_raw_download": sha256_bytes(data), "as_of_utc": as_of, "error": ""})
            return df
        except Exception as exc:
            errors.append(f"{url}:{type(exc).__name__}:{exc}")
    snapshots.append({"source": name, "season": season, "url": "|".join(urls), "rows": 0,
                      "sha256_raw_download": "", "as_of_utc": as_of, "error": " || ".join(errors)})
    return pd.DataFrame()


def release_urls(kind: str, season: int) -> list[str]:
    root = "https://github.com/nflverse/nflverse-data/releases/download"
    if kind == "depth":
        return [f"{root}/depth_charts/depth_charts_{season}.csv"]
    if kind == "roster":
        return [f"{root}/weekly_rosters/roster_weekly_{season}.parquet", f"{root}/weekly_rosters/roster_weekly_{season}.csv"]
    if kind == "snaps":
        return [f"{root}/snap_counts/snap_counts_{season}.parquet", f"{root}/snap_counts/snap_counts_{season}.csv"]
    if kind == "pfr":
        return [f"{root}/pfr_advstats/advstats_week_def_{season}.parquet", f"{root}/pfr_advstats/advstats_week_def_{season}.csv"]
    raise ValueError(kind)


def classify_position(value) -> str:
    s = str(value).upper().replace(" ", "").replace("-", "")
    if not s or s in {"NAN", "NONE", "<NA>"}:
        return ""
    if s in {"OL", "OT", "T", "LT", "RT", "OG", "G", "LG", "RG", "C"}:
        return "OL"
    if "WR" in s or s in {"TE", "RB", "HB", "FB"}:
        return "WR_TE_RB"
    if s in {"DB", "CB", "LCB", "RCB", "NB", "S", "FS", "SS"} or s.endswith("CB"):
        return "DB"
    if s in {"EDGE", "DE", "LDE", "RDE", "DT", "LDT", "RDT", "NT", "DL", "OLB"} or s.endswith("DE"):
        return "PASS_RUSH"
    return ""


def prepare_depth(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["season","week","team","player_id","position","group","rank","snapshot_dt"])
    if season <= 2024:
        team = first_col(df, ["club_code", "team"]); player = first_col(df, ["gsis_id"])
        pos = first_col(df, ["position", "depth_position"]); rank = first_col(df, ["depth_team"])
        week = first_col(df, ["week"])
        if not all([team, player, pos, rank, week]):
            return pd.DataFrame()
        out = pd.DataFrame({"season": season, "week": num(df[week]), "team": df[team].map(team_value),
                            "player_id": clean_id(df[player]), "position": df[pos], "rank": num(df[rank]),
                            "snapshot_dt": pd.NaT})
    else:
        team = first_col(df, ["team"]); player = first_col(df, ["gsis_id"])
        pos = first_col(df, ["pos_abb", "pos_name", "pos_grp"]); rank = first_col(df, ["pos_rank"]); dt = first_col(df, ["dt"])
        if not all([team, player, pos, rank, dt]):
            return pd.DataFrame()
        out = pd.DataFrame({"season": season, "week": np.nan, "team": df[team].map(team_value),
                            "player_id": clean_id(df[player]), "position": df[pos], "rank": num(df[rank]),
                            "snapshot_dt": pd.to_datetime(df[dt], errors="coerce", utc=True)})
    out["group"] = out.position.map(classify_position)
    return out.dropna(subset=["team", "player_id", "rank"]).reset_index(drop=True)


def prepare_roster(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    week = first_col(df, ["week"]); team = first_col(df, ["team", "club_code"]); player = first_col(df, ["gsis_id"])
    pos = first_col(df, ["position", "depth_chart_position", "ngs_position"]); status = first_col(df, ["status", "status_description_abbr"])
    if not all([week, team, player, pos, status]):
        return pd.DataFrame()
    out = pd.DataFrame({"season": season, "week": num(df[week]), "team": df[team].map(team_value),
                        "player_id": clean_id(df[player]), "position": df[pos], "status": df[status]})
    out["group"] = out.position.map(classify_position)
    return out.dropna(subset=["week", "team", "player_id"]).reset_index(drop=True)


def load_schedule(snapshots: list[dict], as_of: str) -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    try:
        req = Request(url, headers={"User-Agent": "m76-data-contract-audit"})
        with urlopen(req, timeout=90) as r:
            data = r.read(); final_url = r.geturl()
        allowed = {"season","week","game_type","season_type","gameday","gametime","home_team","away_team"}
        df = lower(pd.read_csv(io.BytesIO(data), usecols=lambda c: str(c).lower() in allowed, low_memory=False))
        snapshots.append({"source":"schedule_kickoff_only","season":"2024-2025","url":final_url,"rows":len(df),
                          "sha256_raw_download":sha256_bytes(data),"as_of_utc":as_of,"error":""})
    except Exception as exc:
        snapshots.append({"source":"schedule_kickoff_only","season":"2024-2025","url":url,"rows":0,
                          "sha256_raw_download":"","as_of_utc":as_of,"error":f"{type(exc).__name__}:{exc}"})
        return pd.DataFrame()
    if "game_type" in df: df = df.loc[df.game_type.astype(str).str.upper().eq("REG")].copy()
    elif "season_type" in df: df = df.loc[df.season_type.astype(str).str.upper().eq("REG")].copy()
    df = df.loc[num(df.season).isin([2024,2025]) & num(df.week).between(1,18)].copy()
    clock = df.gametime.astype(str) if "gametime" in df else pd.Series("13:00", index=df.index)
    naive = pd.to_datetime(df.gameday.astype(str) + " " + clock, errors="coerce")
    try:
        df["kickoff"] = naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    except Exception:
        df["kickoff"] = pd.to_datetime(naive, errors="coerce", utc=True)
    rows=[]
    for side in ["home_team","away_team"]:
        q=df[["season","week",side,"kickoff"]].copy(); q.columns=["season","week","team","kickoff"]; rows.append(q)
    out=pd.concat(rows,ignore_index=True); out["season"]=num(out.season).astype(int); out["week"]=num(out.week).astype(int); out["team"]=out.team.map(team_value)
    return out.drop_duplicates(["season","week","team"])


def target_team_weeks(base: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    a=base[["season","week","team"]].copy(); b=base[["season","week","opponent"]].rename(columns={"opponent":"team"})
    t=pd.concat([a,b],ignore_index=True).drop_duplicates(["season","week","team"]).sort_values(["season","week","team"]).reset_index(drop=True)
    if schedule.empty:
        t["kickoff"]=pd.NaT
    else:
        t=t.merge(schedule,on=["season","week","team"],how="left",validate="one_to_one")
    return t


def latest_depth_for_target(depth: pd.DataFrame, season: int, week: int, team: str, kickoff) -> pd.DataFrame:
    if depth.empty: return pd.DataFrame()
    q=depth.loc[depth.team.eq(team)].copy()
    if season <= 2024:
        return q.loc[num(q.week).eq(week)].copy()
    if pd.isna(kickoff): return pd.DataFrame()
    q=q.loc[q.snapshot_dt.notna() & q.snapshot_dt.lt(kickoff)].copy()
    if q.empty: return q
    latest=q.snapshot_dt.max()
    return q.loc[q.snapshot_dt.eq(latest)].copy()


def build_id_bridge(snap_raw: pd.DataFrame, pfr_raw: pd.DataFrame, snapshots: list[dict], as_of: str) -> tuple[pd.DataFrame,pd.DataFrame,str]:
    players=download_table("nflverse_players_bridge","current",[
        "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet",
        "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"],snapshots,as_of)
    if players.empty: return snap_raw,pfr_raw,"players_source_unavailable"
    gsis=first_col(players,["gsis_id"]); pfr=first_col(players,["pfr_id","pfr_player_id"])
    if not gsis or not pfr: return snap_raw,pfr_raw,"players_missing_gsis_pfr"
    bridge=players[[gsis,pfr]].dropna().copy(); bridge.columns=["gsis_id","pfr_id"]; bridge["gsis_id"]=clean_id(bridge.gsis_id); bridge["pfr_id"]=clean_id(bridge.pfr_id); bridge=bridge.drop_duplicates("pfr_id")
    def attach(df):
        if df.empty: return df
        if "gsis_id" in df.columns:
            out=df.copy(); out["gsis_id"]=clean_id(out.gsis_id); return out
        pc=first_col(df,["pfr_player_id","pfr_id"])
        if not pc: return df
        out=df.copy(); out[pc]=clean_id(out[pc]); return out.merge(bridge,left_on=pc,right_on="pfr_id",how="left")
    return attach(snap_raw),attach(pfr_raw),"players_gsis_pfr_bridge"


def history_frame(df: pd.DataFrame, metric_kind: str) -> tuple[pd.DataFrame,bool,list[str]]:
    if df.empty: return pd.DataFrame(),False,[]
    season=first_col(df,["season"]); week=first_col(df,["week"]); team=first_col(df,["team","team_abbr"]); player=first_col(df,["gsis_id"])
    if metric_kind=="snap":
        metrics_cols=[c for c in ["offense_snaps","offense_pct","defense_snaps","defense_pct"] if c in df.columns]
    else:
        keys=("pressure","hurr","qb_hit","sack","blitz")
        metrics_cols=[c for c in df.columns if any(k in c for k in keys) and c.startswith("def_")]
        if len(metrics_cols)<2: metrics_cols=[c for c in df.columns if any(k in c for k in keys)]
    ok=bool(all([season,week,team,player]) and len(metrics_cols)>=1)
    if not ok: return pd.DataFrame(),False,metrics_cols
    out=pd.DataFrame({"season":num(df[season]),"week":num(df[week]),"team":df[team].map(team_value),"player_id":clean_id(df[player])})
    out=out.dropna(subset=["season","week","team","player_id"]); out["season"]=out.season.astype(int); out["week"]=out.week.astype(int)
    key_nonnull=len(out)/len(df) if len(df) else 0
    return out,key_nonnull>=.90,metrics_cols


def has_prior(hist: pd.DataFrame, team: str, player: str, season: int, week: int) -> bool:
    if hist.empty: return False
    q=hist.loc[hist.team.eq(team) & hist.player_id.eq(player)]
    return bool(((q.season < season) | ((q.season == season) & (q.week < week))).any())


def summarize_gate(evidence: pd.DataFrame, depth_current_ok: bool, roster_current_ok: bool, snap_keys_ok: bool) -> pd.DataFrame:
    rows=[]
    def add(metric,value,threshold,passed): rows.append({"metric":metric,"value":float(value) if pd.notna(value) else np.nan,"threshold":threshold,"passed":bool(passed)})
    add("schedule_kickoff_coverage",evidence.kickoff.notna().mean(),">=0.99",evidence.kickoff.notna().mean()>=.99)
    for s in [2024,2025]:
        q=evidence[evidence.season.eq(s)]
        add(f"depth_game_coverage_{s}",q.depth_found.mean(),">=0.95",q.depth_found.mean()>=.95)
        add(f"roster_game_coverage_{s}",q.roster_found.mean(),">=0.95",q.roster_found.mean()>=.95)
        add(f"depth_roster_identity_bridge_{s}",q.bridge_ratio.mean(),">=0.90",q.bridge_ratio.mean()>=.90)
        add(f"strictly_prior_starter_snap_coverage_{s}",q.prior_snap_ratio.mean(),">=0.80",q.prior_snap_ratio.mean()>=.80)
        if s==2025:
            add("depth_strict_pre_kickoff_2025",q.depth_strict_pre_kickoff.mean(),">=0.95",q.depth_strict_pre_kickoff.mean()>=.95)
        for g in GROUPS:
            v=q[f"depth_group_{g}"].mean(); add(f"depth_{g}_coverage_{s}",v,">=0.90",v>=.90)
    add("current_2026_depth_release_schema",1.0 if depth_current_ok else 0.0,"==1",depth_current_ok)
    add("current_2026_weekly_roster_release_schema",1.0 if roster_current_ok else 0.0,"==1",roster_current_ok)
    add("snap_history_join_time_keys",1.0 if snap_keys_ok else 0.0,"==1",snap_keys_ok)
    return pd.DataFrame(rows)


def source_and_personnel_audit(base: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,bool]:
    as_of=datetime.now(timezone.utc).isoformat(); snapshots=[]
    schedule=load_schedule(snapshots,as_of); targets=target_team_weeks(base,schedule)

    depth_raw={s:download_table("nflverse_depth_charts",s,release_urls("depth",s),snapshots,as_of) for s in [2024,2025,2026]}
    roster_raw={s:download_table("nflverse_weekly_rosters",s,release_urls("roster",s),snapshots,as_of) for s in [2024,2025,2026]}
    snap_parts=[download_table("pfr_snap_counts",s,release_urls("snaps",s),snapshots,as_of) for s in [2023,2024,2025]]
    pfr_parts=[download_table("pfr_individual_pass_rush",s,release_urls("pfr",s),snapshots,as_of) for s in [2023,2024,2025]]
    snap_raw=pd.concat([q for q in snap_parts if not q.empty],ignore_index=True) if any(not q.empty for q in snap_parts) else pd.DataFrame()
    pfr_raw=pd.concat([q for q in pfr_parts if not q.empty],ignore_index=True) if any(not q.empty for q in pfr_parts) else pd.DataFrame()
    snap_raw,pfr_raw,bridge_detail=build_id_bridge(snap_raw,pfr_raw,snapshots,as_of)

    depth={s:prepare_depth(depth_raw[s],s) for s in [2024,2025,2026]}
    roster={s:prepare_roster(roster_raw[s],s) for s in [2024,2025,2026]}
    snap_hist,snap_keys_ok,snap_metrics=history_frame(snap_raw,"snap")
    pfr_hist,pfr_keys_ok,pfr_metrics=history_frame(pfr_raw,"pfr")

    evidence=[]
    for r in targets.itertuples(index=False):
        d=latest_depth_for_target(depth[int(r.season)],int(r.season),int(r.week),r.team,r.kickoff)
        rr=roster[int(r.season)]
        rr=rr.loc[(rr.team.eq(r.team)) & num(rr.week).eq(int(r.week))].copy() if not rr.empty else pd.DataFrame()
        dids=set(clean_id(d.player_id).dropna()) if not d.empty else set(); rids=set(clean_id(rr.player_id).dropna()) if not rr.empty else set()
        bridge_ratio=(len(dids & rids)/len(dids)) if dids else np.nan
        starters=d.loc[num(d["rank"]).le(1),"player_id"].dropna().astype(str).unique().tolist() if not d.empty else []
        snap_ratio=np.mean([has_prior(snap_hist,r.team,p,int(r.season),int(r.week)) for p in starters]) if starters else np.nan
        rushers=d.loc[(d.group.eq("PASS_RUSH")) & num(d["rank"]).le(1),"player_id"].dropna().astype(str).unique().tolist() if not d.empty else []
        pfr_ratio=np.mean([has_prior(pfr_hist,r.team,p,int(r.season),int(r.week)) for p in rushers]) if rushers else np.nan
        rec={"season":int(r.season),"week":int(r.week),"team":r.team,"kickoff":r.kickoff,
             "depth_found":bool(len(dids)),"roster_found":bool(len(rids)),"depth_player_count":len(dids),"roster_player_count":len(rids),
             "bridge_ratio":bridge_ratio,"starter_count":len(starters),"prior_snap_ratio":snap_ratio,"passrush_starter_count":len(rushers),"prior_passrush_ratio":pfr_ratio,
             "depth_snapshot_dt":d.snapshot_dt.max() if (not d.empty and "snapshot_dt" in d) else pd.NaT,
             "depth_strict_pre_kickoff":bool(int(r.season)<=2024 or (not d.empty and pd.notna(r.kickoff) and d.snapshot_dt.max()<r.kickoff))}
        for g in GROUPS: rec[f"depth_group_{g}"]=bool(not d.empty and d.group.eq(g).any())
        evidence.append(rec)
    ev=pd.DataFrame(evidence)

    d26=depth[2026]; r26=roster[2026]
    depth_current_ok=bool(not d26.empty and d26.player_id.notna().any() and d26.snapshot_dt.notna().any() and d26["rank"].notna().any() and all(d26.group.eq(g).any() for g in GROUPS))
    roster_current_ok=bool(not r26.empty and r26.player_id.notna().any() and r26.status.notna().any() and r26.position.notna().any())
    gate=summarize_gate(ev,depth_current_ok,roster_current_ok,snap_keys_ok)
    exact_gate=bool(len(gate) and gate.passed.all())

    def src_status(name):
        q=pd.DataFrame(snapshots); q=q[q.source.eq(name)]; return int(q.rows.sum()) if len(q) else 0
    contracts=pd.DataFrame([
        {"source":"nflverse_depth_charts","novel_for_qb":True,"rows":src_status("nflverse_depth_charts"),"contract_status":"QUALIFIED_GAME_LEVEL" if all(gate.loc[gate.metric.str.startswith("depth_")|gate.metric.eq("current_2026_depth_release_schema"),"passed"]) else "NOT_QUALIFIED","detail":"2024 week-specific depth records; 2025 latest dt strictly before kickoff; current 2026 release required"},
        {"source":"nflverse_weekly_rosters","novel_for_qb":True,"rows":src_status("nflverse_weekly_rosters"),"contract_status":"QUALIFIED_IDENTITY_SUPPORT" if all(gate.loc[gate.metric.str.startswith("roster_")|gate.metric.str.startswith("depth_roster_identity_bridge_")|gate.metric.eq("current_2026_weekly_roster_release_schema"),"passed"]) else "NOT_QUALIFIED","detail":"week-level roster identity/status support; not treated as timestamped target outcome"},
        {"source":"pfr_snap_counts","novel_for_qb":True,"rows":src_status("pfr_snap_counts"),"contract_status":"QUALIFIED_STRICTLY_PRIOR_ROLE" if snap_keys_ok and all(gate.loc[gate.metric.str.startswith("strictly_prior_starter_snap_coverage_"),"passed"]) else "NOT_QUALIFIED","detail":f"strictly-before-target player history only; metrics={'|'.join(snap_metrics)}; bridge={bridge_detail}"},
        {"source":"pfr_individual_pass_rush","novel_for_qb":True,"rows":src_status("pfr_individual_pass_rush"),"contract_status":"QUALIFIED_STRICTLY_PRIOR_PASS_RUSH" if pfr_keys_ok and ev.prior_passrush_ratio.mean()>=.60 else "NOT_QUALIFIED","detail":f"non-null season/week/team/player keys + strictly-prior expected-rusher history; metrics={'|'.join(pfr_metrics)}; prior_ratio={ev.prior_passrush_ratio.mean():.4f}"},
        {"source":"nflverse_injuries","novel_for_qb":False,"rows":0,"contract_status":"BROKEN_AFTER_2024","detail":"source ended after 2024; generic injury burden already tested in M67; cannot authorize M77"},
    ])
    snapshots_df=pd.DataFrame(snapshots)
    snapshots_df.to_csv(out_dir/"m76_source_snapshot_hashes.csv",index=False)
    ev.to_csv(out_dir/"m76_personnel_game_coverage.csv",index=False)
    gate.to_csv(out_dir/"m76_personnel_gate_summary.csv",index=False)
    contracts.to_csv(out_dir/"m76_source_contracts.csv",index=False)
    return contracts,gate,ev,exact_gate


def unresolved_frontiers() -> pd.DataFrame:
    return pd.DataFrame([
        {"frontier":"individual_OL_pass_block_quality_and_assignments","status":"MISSING_FREE_CONTRACT","needed_for":"true OL blocker x pass-rusher matchup"},
        {"frontier":"WR_CB_route_assignment_shadow_share","status":"MISSING_FREE_CONTRACT","needed_for":"true receiver x assigned defender matchup beyond M75 aggregate secondary"},
        {"frontier":"structured_week_specific_gameplan_reporting","status":"MISSING_STRUCTURED_HISTORY","needed_for":"why this Sunday deviates from normal DBR/role expectation"},
        {"frontier":"post_2024_injury_reports","status":"NFLVERSE_SOURCE_DEAD","needed_for":"exact practice/game availability from a replacement source"},
    ])


def no_retest_ledger() -> pd.DataFrame:
    return pd.DataFrame([
        {"family":"pass_rate_DBR_recent_history_state","migrations":"M40-M42,M64-M65,M73-M74","rule":"DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family":"aggregate_defense_pressure_EPA_coverage","migrations":"M45,M56,M69,M72","rule":"DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family":"opening_script_playcaller","migrations":"M67-M69,M74","rule":"DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family":"QB_efficiency_volatility_risk","migrations":"M70-M71","rule":"DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family":"receiver_explosive_or_tracking_x_aggregate_secondary","migrations":"M72,M75","rule":"DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family":"generic_injury_burden","migrations":"M67","rule":"DO_NOT_RETEST_WITHOUT_EXACT_PLAYER_PERSONNEL_IDENTITY"},
        {"family":"new_model_or_subset_on_same_feature_universe","migrations":"M61,M66+","rule":"PROHIBITED_AS_STANDALONE_MIGRATION"},
    ])


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--canonical",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    base=require_canonical(a.canonical)
    rec=recovery_map(base); diff=difficulty(base)
    contracts,gate,evidence,exact_gate=source_and_personnel_audit(base,a.out_dir)
    rec.to_csv(a.out_dir/"m76_recovery_map.csv",index=False); diff.to_csv(a.out_dir/"m76_season_difficulty.csv",index=False)
    unresolved_frontiers().to_csv(a.out_dir/"m76_unresolved_data_frontiers.csv",index=False); no_retest_ledger().to_csv(a.out_dir/"m76_no_retest_ledger.csv",index=False)
    cur=rec[(rec.scenario.eq("current")) & rec.season.eq("combined")].iloc[0]; eq=rec[(rec.scenario.eq("min_equal_recovery_below_50")) & rec.season.eq("combined")]
    if exact_gate:
        verdict="m76_exact_personnel_identity_discontinuity_layer_qualified"; nxt="M77_exact_personnel_discontinuity"
    else:
        verdict="m76_exact_personnel_identity_layer_not_yet_qualified"; nxt="seek_new_personnel_source_before_M77"
    interpretation=pd.DataFrame([{"canonical_rows":len(base),"baseline_mae":float(cur.mae),"mae_2024":float(diff.loc[diff.season.eq(2024),"mae"].iloc[0]),"mae_2025":float(diff.loc[diff.season.eq(2025),"mae"].iloc[0]),
        "minimum_equal_attempt_ypa_recovery_below_50":float(eq.iloc[0].attempt_recovery) if len(eq) else np.nan,"mae_at_minimum_equal_recovery":float(eq.iloc[0].mae) if len(eq) else np.nan,
        "exact_personnel_gate":exact_gate,"m76_interpretation":verdict,"next_allowed_migration":nxt,"predictive_model_fit":False,"production_actionable":False}])
    interpretation.to_csv(a.out_dir/"m76_precommitted_interpretation.csv",index=False)
    manifest={"migration":76,"canonical":"M75 Run #15 canonical-v3 football-only","canonical_sha256":EXPECTED_CANONICAL_SHA256,"expected_rows":EXPECTED_ROWS,
              "market_features":False,"predictive_models":[],"purpose":"40s recovery + season difficulty + exact personnel data-contract qualification + anti-retest ledger",
              "m77_gate":"all game-level personnel gate checks must pass"}
    (a.out_dir/"m76_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("=== M76 INTERPRETATION ==="); print(interpretation.to_string(index=False)); print("\n=== M76 PERSONNEL GATE ==="); print(gate.to_string(index=False)); print("\n=== M76 SOURCES ==="); print(contracts.to_string(index=False)); print("\n=== M76 DIFFICULTY ==="); print(diff.to_string(index=False)); print("\n=== M76 RECOVERY ==="); print(rec.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
