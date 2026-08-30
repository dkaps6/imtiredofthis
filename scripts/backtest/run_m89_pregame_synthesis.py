#!/usr/bin/env python3
"""M89 Phase 2: frozen 2023-trained QB pregame synthesis.

Two residual models are fit once on 2023:
1) football-only corrected/deployable context;
2) identical football context plus a separately labeled game-market layer.

2024-2025 are evaluation only. No postgame casebook field is read here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team

HISTORY_GAMES = 8
ALPHA = 20.0
RESIDUAL_CAP = 45.0
BOOT_N = 10000
BOOT_SEED = 89

FOOTBALL_FEATURES = [
    "base_proj", "mc_proj", "ml_proj", "state_proj",
    "component_sd", "component_range", "pred_attempts", "pred_ypa",
    "off_true_proe", "off_neutral_pace", "off_pass_rate", "off_plays",
    "qb_prior_attempts", "qb_prior_ypa",
    "def_pass_epa_allowed", "def_success_allowed", "def_ypa_allowed", "def_pass_rate_faced",
    "off_hit_sack_pressure", "def_hit_sack_pressure",
    "controlled_environment",
]
MARKET_FEATURES = [
    "market_total", "market_spread", "market_abs_spread", "market_team_implied",
    "market_opp_implied", "market_is_underdog", "market_moneyline",
]


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def canon(v) -> str:
    return canon_team(v)


def key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def n(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def normalize_trace(path: Path, season: int) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].between(1,18)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = x["team"].map(canon)
    if "opponent" in x.columns:
        x["opponent"] = x["opponent"].map(canon)
    else:
        raise RuntimeError(f"trace {path} missing opponent")
    x["player_clean_key"] = x["player_clean_key"].map(key)

    actual_col = "actual_pass_yards" if "actual_pass_yards" in x.columns else "actual"
    if actual_col not in x.columns:
        raise RuntimeError(f"trace {path} missing actual passing yards")
    x["actual_pass_yards"] = pd.to_numeric(x[actual_col], errors="coerce")
    if "ensemble_proj" not in x.columns:
        raise RuntimeError(f"trace {path} missing ensemble_proj")
    x["base_proj"] = pd.to_numeric(x["ensemble_proj"], errors="coerce")
    for c in ["mc_proj","ml_proj","state_proj"]:
        x[c] = pd.to_numeric(x.get(c), errors="coerce")
    if "pred_attempts" in x.columns:
        x["pred_attempts"] = pd.to_numeric(x["pred_attempts"], errors="coerce")
    elif "mc_expected_pass_attempts" in x.columns:
        x["pred_attempts"] = pd.to_numeric(x["mc_expected_pass_attempts"], errors="coerce")
    else:
        x["pred_attempts"] = np.nan
    if "implied_pred_ypa" in x.columns:
        x["pred_ypa"] = pd.to_numeric(x["implied_pred_ypa"], errors="coerce")
    elif "mc_rules_ypa" in x.columns:
        x["pred_ypa"] = pd.to_numeric(x["mc_rules_ypa"], errors="coerce")
    else:
        x["pred_ypa"] = x["mc_proj"] / x["pred_attempts"].replace(0, np.nan)
    comp = x[["mc_proj","ml_proj","state_proj"]]
    x["component_sd"] = comp.std(axis=1, skipna=True)
    x["component_range"] = comp.max(axis=1, skipna=True) - comp.min(axis=1, skipna=True)
    keep = [
        "season","week","team","opponent","player_clean_key","actual_pass_yards","base_proj",
        "mc_proj","ml_proj","state_proj","component_sd","component_range","pred_attempts","pred_ypa",
    ]
    return x[keep].drop_duplicates(["season","week","team","player_clean_key"])


def load_team_history(path: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon)
    return x.sort_values(["season","week","team"]).reset_index(drop=True)


def load_player_logs(path: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon)
    if "player_clean_key" in x.columns:
        x["player_clean_key"] = x["player_clean_key"].map(key)
    elif "player" in x.columns:
        x["player_clean_key"] = x["player"].map(key)
    else:
        raise RuntimeError(f"player log {path} missing player identity")
    return x.sort_values(["season","week","team","player_clean_key"]).reset_index(drop=True)


def prior_rows(df: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    q = df.loc[
        df["team"].eq(team)
        & (df["season"].lt(season) | (df["season"].eq(season) & df["week"].lt(week)))
    ].sort_values(["season","week"])
    return q.tail(HISTORY_GAMES)


def mean_col(df: pd.DataFrame, names: list[str]) -> float:
    for c in names:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return float(s.mean())
    return np.nan


def qb_prior(logs: pd.DataFrame, row: pd.Series) -> tuple[float,float,int]:
    q = logs.loc[
        logs["player_clean_key"].eq(row.player_clean_key)
        & (logs["season"].lt(row.season) | (logs["season"].eq(row.season) & logs["week"].lt(row.week)))
    ].sort_values(["season","week"]).tail(HISTORY_GAMES)
    if q.empty:
        return np.nan, np.nan, 0
    att = mean_col(q, ["pass_att","attempts","passing_attempts"])
    ypa = mean_col(q, ["ypa_game","ypa"])
    if not np.isfinite(ypa):
        py = pd.to_numeric(q.get("pass_yards"), errors="coerce") if "pass_yards" in q.columns else pd.Series(np.nan,index=q.index)
        pa = pd.to_numeric(q.get("pass_att"), errors="coerce") if "pass_att" in q.columns else pd.Series(np.nan,index=q.index)
        if pa.sum(skipna=True) > 0:
            ypa = float(py.sum(skipna=True) / pa.sum(skipna=True))
    return att, ypa, int(len(q))


def add_history_features(trace: pd.DataFrame, team_hist: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in trace.iterrows():
        oh = prior_rows(team_hist, int(r.season), int(r.week), r.team)
        dh = prior_rows(team_hist, int(r.season), int(r.week), r.opponent)
        qb_att, qb_ypa, qb_n = qb_prior(logs, r)
        rows.append({
            "off_history_games":int(len(oh)), "def_history_games":int(len(dh)), "qb_history_games":qb_n,
            "off_true_proe":mean_col(oh,["true_proe","proe"]),
            "off_neutral_pace":mean_col(oh,["neutral_pace_true","neutral_pace"]),
            "off_pass_rate":mean_col(oh,["pass_rate_off","dropback_rate"]),
            "off_plays":mean_col(oh,["plays_est"]),
            "qb_prior_attempts":qb_att, "qb_prior_ypa":qb_ypa,
            "def_pass_epa_allowed":mean_col(dh,["def_pass_epa_allowed","def_pass_epa"]),
            "def_success_allowed":mean_col(dh,["def_pass_success_allowed","success_rate_def"]),
            "def_ypa_allowed":mean_col(dh,["def_ypa_allowed"]),
            "def_pass_rate_faced":mean_col(dh,["pass_rate_faced"]),
            "off_hit_sack_pressure":mean_col(oh,["hit_sack_pressure_rate_allowed","pressure_rate_allowed"]),
            "def_hit_sack_pressure":mean_col(dh,["hit_sack_pressure_rate_generated","pressure_rate_generated"]),
        })
    return pd.concat([trace.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def load_market(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    rows = []
    for season in seasons:
        raw = nfl.load_schedules(int(season))
        s = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        s = lower(s)
        if "game_type" in s.columns:
            s = s.loc[s["game_type"].astype(str).str.upper().eq("REG")].copy()
        s["week"] = pd.to_numeric(s["week"], errors="coerce")
        s = s.loc[s["week"].between(1,18)].copy()
        for _, g in s.iterrows():
            spread = pd.to_numeric(pd.Series([g.get("spread_line")]), errors="coerce").iloc[0]
            total = pd.to_numeric(pd.Series([g.get("total_line")]), errors="coerce").iloc[0]
            for side in ["home","away"]:
                team = canon(g.get(f"{side}_team"))
                opp = canon(g.get("away_team" if side=="home" else "home_team"))
                team_spread = spread if side=="home" else (-spread if pd.notna(spread) else np.nan)
                implied = (total - team_spread)/2 if pd.notna(total) and pd.notna(team_spread) else np.nan
                opp_implied = (total + team_spread)/2 if pd.notna(total) and pd.notna(team_spread) else np.nan
                mlcol = f"{side}_moneyline"
                ml = pd.to_numeric(pd.Series([g.get(mlcol)]), errors="coerce").iloc[0] if mlcol in s.columns else np.nan
                roof = str(g.get("roof", "") or "").strip().lower()
                controlled = int(any(t in roof for t in ["dome","closed","indoor"])) if roof else np.nan
                rows.append({
                    "season":season,"week":int(g.week),"team":team,"opponent":opp,
                    "market_total":total,"market_spread":team_spread,"market_abs_spread":abs(team_spread) if pd.notna(team_spread) else np.nan,
                    "market_team_implied":implied,"market_opp_implied":opp_implied,
                    "market_is_underdog":int(team_spread>0) if pd.notna(team_spread) else np.nan,
                    "market_moneyline":ml,"controlled_environment":controlled,
                })
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["season","week","team","opponent"])


def attach_market(x: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    return x.merge(market, on=["season","week","team","opponent"], how="left", validate="many_to_one")


def model_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])


def fit_candidate(train: pd.DataFrame, features: list[str], label: str) -> tuple[Pipeline,pd.DataFrame]:
    q = train.dropna(subset=["actual_pass_yards","base_proj"]).copy()
    target = q["actual_pass_yards"] - q["base_proj"]
    model = model_pipeline()
    model.fit(q[features], target)
    # Feature names after imputation indicators are intentionally not relied on
    # for promotion; raw feature list and coverage are the audit contract.
    cov = pd.DataFrame({
        "candidate":label,
        "feature":features,
        "train_nonnull_rate":[float(pd.to_numeric(q[c],errors="coerce").notna().mean()) for c in features],
    })
    return model, cov


def predict_candidate(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray,np.ndarray]:
    raw = np.asarray(model.predict(frame[features]), dtype=float)
    capped = np.clip(raw, -RESIDUAL_CAP, RESIDUAL_CAP)
    return frame["base_proj"].to_numpy(float) + capped, capped


def metric(frame: pd.DataFrame, pred_col: str, label: str, season_label: str) -> dict:
    a = pd.to_numeric(frame["actual_pass_yards"], errors="coerce")
    p = pd.to_numeric(frame[pred_col], errors="coerce")
    z = pd.DataFrame({"a":a,"p":p}).dropna()
    e = z.p-z.a
    corr = float(z.p.corr(z.a)) if len(z)>1 and z.p.nunique()>1 and z.a.nunique()>1 else np.nan
    return {
        "season":season_label,"model":label,"n":len(z),
        "mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),
        "correlation":corr,"median_ae":float(e.abs().median()),"tails100":int(e.abs().ge(100).sum()),
        "under100":int(e.le(-100).sum()),"over100":int(e.ge(100).sum()),
    }


def bootstrap_prob(base_err: np.ndarray, cand_err: np.ndarray) -> float:
    mask = np.isfinite(base_err) & np.isfinite(cand_err)
    b = np.asarray(base_err[mask],float); c=np.asarray(cand_err[mask],float)
    if len(b)==0:
        return np.nan
    rng=np.random.default_rng(BOOT_SEED)
    wins=0
    for _ in range(BOOT_N):
        idx=rng.integers(0,len(b),len(b))
        wins += float(np.mean(b[idx]) - np.mean(c[idx]) > 0)
    return float(wins/BOOT_N)


def gate_candidate(score: pd.DataFrame, eval_frame: pd.DataFrame, candidate: str, market: bool) -> dict:
    base = score[(score.season=="COMBINED")&(score.model=="base")].iloc[0]
    cand = score[(score.season=="COMBINED")&(score.model==candidate)].iloc[0]
    gains = float(base.mae-cand.mae)
    by_season=[]; season_nonworse=True
    for s in [2024,2025]:
        b=score[(score.season==str(s))&(score.model=="base")].iloc[0]
        c=score[(score.season==str(s))&(score.model==candidate)].iloc[0]
        season_nonworse &= bool(c.mae <= b.mae)
        by_season.append(f"{s}:{b.mae:.3f}->{c.mae:.3f}")
    base_abs=(eval_frame["base_proj"]-eval_frame["actual_pass_yards"]).abs().to_numpy(float)
    cand_abs=(eval_frame[candidate]-eval_frame["actual_pass_yards"]).abs().to_numpy(float)
    boot=bootstrap_prob(base_abs,cand_abs)
    corr_gain=float(cand.correlation-base.correlation)
    market_cov=float(eval_frame[MARKET_FEATURES[:6]].notna().all(axis=1).mean()) if market else 1.0
    required_gain=1.00 if market else 0.75
    gates={
        "mae_gain":gains,
        "mae_gain_gate":bool(gains>=required_gain),
        "both_seasons_nonworse":bool(season_nonworse),
        "season_detail":" ".join(by_season),
        "rmse_nonworse":bool(cand.rmse<=base.rmse),
        "correlation_gain":corr_gain,
        "correlation_gate":True if market else bool(corr_gain>=0.01),
        "tails_nonincrease":bool(cand.tails100<=base.tails100),
        "bootstrap_p_improve":boot,
        "bootstrap_gate":bool(boot>=0.80),
        "market_coverage":market_cov,
        "market_coverage_gate":bool(market_cov>=0.90) if market else True,
    }
    gates["all_gates_pass"] = bool(all(v for k,v in gates.items() if k.endswith("_gate") or k in ["both_seasons_nonworse","rmse_nonworse","tails_nonincrease"]))
    return gates


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--trace-2023",type=Path,required=True)
    p.add_argument("--trace-2024-2025",type=Path,required=True)
    p.add_argument("--team-2023",type=Path,required=True)
    p.add_argument("--logs-2023",type=Path,required=True)
    p.add_argument("--team-2024",type=Path,required=True)
    p.add_argument("--logs-2024",type=Path,required=True)
    p.add_argument("--team-2025",type=Path,required=True)
    p.add_argument("--logs-2025",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    t23=normalize_trace(a.trace_2023,2023)
    ev=lower(pd.read_csv(a.trace_2024_2025,low_memory=False))
    t24=normalize_trace(a.trace_2024_2025,2024)
    t25=normalize_trace(a.trace_2024_2025,2025)

    f23=add_history_features(t23,load_team_history(a.team_2023),load_player_logs(a.logs_2023))
    f24=add_history_features(t24,load_team_history(a.team_2024),load_player_logs(a.logs_2024))
    f25=add_history_features(t25,load_team_history(a.team_2025),load_player_logs(a.logs_2025))
    market=load_market([2023,2024,2025])
    f23=attach_market(f23,market); f24=attach_market(f24,market); f25=attach_market(f25,market)
    train=f23; test=pd.concat([f24,f25],ignore_index=True)

    fm,fcov=fit_candidate(train,FOOTBALL_FEATURES,"football_synthesis")
    mm,mcov=fit_candidate(train,FOOTBALL_FEATURES+MARKET_FEATURES,"market_assisted")
    test["football_synthesis"],test["football_residual_correction"]=predict_candidate(fm,test,FOOTBALL_FEATURES)
    test["market_assisted"],test["market_residual_correction"]=predict_candidate(mm,test,FOOTBALL_FEATURES+MARKET_FEATURES)

    rows=[]
    for sl,part in [("2024",test[test.season.eq(2024)]),("2025",test[test.season.eq(2025)]),("COMBINED",test)]:
        rows.append(metric(part,"base_proj","base",sl))
        rows.append(metric(part,"football_synthesis","football_synthesis",sl))
        rows.append(metric(part,"market_assisted","market_assisted",sl))
    score=pd.DataFrame(rows)
    gates={
        "football_synthesis":gate_candidate(score,test,"football_synthesis",False),
        "market_assisted":gate_candidate(score,test,"market_assisted",True),
        "train_season":2023,"evaluation_seasons":[2024,2025],
        "ridge_alpha":ALPHA,"residual_cap":RESIDUAL_CAP,"bootstrap_draws":BOOT_N,"bootstrap_seed":BOOT_SEED,
        "postgame_casebook_features_used_for_prediction":False,
        "sportsbook_features_in_football_model":False,
        "market_timing":"nflverse schedule market snapshot treated as pregame/closing-style, not Wednesday information",
    }

    pd.concat([fcov,mcov],ignore_index=True).to_csv(a.out_dir/"m89_synthesis_feature_coverage.csv",index=False)
    train.to_csv(a.out_dir/"m89_2023_training_features.csv",index=False)
    test.to_csv(a.out_dir/"m89_2024_2025_synthesis_trace.csv",index=False)
    score.to_csv(a.out_dir/"m89_synthesis_scoreboard.csv",index=False)
    (a.out_dir/"m89_synthesis_gates.json").write_text(json.dumps(gates,indent=2),encoding="utf-8")
    print("=== M89 SYNTHESIS SCOREBOARD ==="); print(score.to_string(index=False))
    print("=== M89 SYNTHESIS GATES ==="); print(json.dumps(gates,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
