#!/usr/bin/env python3
"""Migration 81: FTN novel-mechanism QB predictive development screen.

Development only. 2025 target outcomes are never parsed into the modeling frame.
2023 + strictly-prior 2024 FTN/PBP history build pregame features for 2024 targets.
Four M80-qualified information families receive one frozen model architecture:
StandardScaler + Ridge(alpha=20, fit_intercept=False), separately for attempt and
YPA residuals. No model zoo, feature subset search, or post-result retuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CANON_SHA = "c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742"
EXPECTED_COUNTS = {2024: 444, 2025: 440}
HISTORY_SEASONS = (2023, 2024)
REG_WEEKS = set(range(1, 19))
RIDGE_ALPHA = 20.0
BOOT_N = 2000
BOOT_SEED = 81
MIN_RAW_FIELD_COVERAGE = 0.80
MIN_FTN_PBP_JOIN = 0.95
MIN_IDENTITY_MAP = 0.95
ATT_BOUNDS = (18.0, 48.0)
YPA_BOUNDS = (4.5, 10.5)
MARKET_TOKENS = ("market","spread","moneyline","sportsbook","implied_total","game_total","prop_line","team_total")

FAMILIES = {
    "TACTICAL_CALL_STRUCTURE": ["is_motion", "is_screen_pass", "is_rpo"],
    "PRESSURE_RESPONSE": ["n_blitzers", "is_qb_out_of_pocket", "is_throw_away", "is_qb_fault_sack"],
    "THROW_DECISION_QUALITY": ["is_interception_worthy", "read_thrown", "is_catchable_ball"],
    "RECEIVER_ERROR_ATTRIBUTION": ["is_drop"],
}

CROSSWALK = {
    "TACTICAL_CALL_STRUCTURE": "M67/M68 generic formation/opening tendency; M70 YAC/explosive decomposition",
    "PRESSURE_RESPONSE": "M9,M16,M22-M23,M45,M56,M69,M70,M72 aggregate pressure/dropback",
    "THROW_DECISION_QUALITY": "M70 completion/CPOE/interception decomposition; M71 volatility",
    "RECEIVER_ERROR_ATTRIBUTION": "M34 catch conversion; M70 completion decomposition",
}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x

def read_url_parquet(url: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": "m81-ftn-development"})
    with urlopen(req, timeout=180) as r:
        raw = r.read()
        final = r.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final, "bytes": len(raw), "sha256": sha256_bytes(raw)
    }

def load_canonical_2024(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != CANON_SHA:
        raise RuntimeError(f"canonical SHA drift: {digest}")
    text = raw.decode("utf-8")
    rdr = csv.reader(io.StringIO(text))
    header = next(rdr)
    hlow = [str(c).strip().lower() for c in header]
    if any(any(t in c for t in MARKET_TOKENS) for c in hlow):
        raise RuntimeError("market boundary violated in canonical")
    si = hlow.index("season")
    counts: dict[int, int] = {}
    rows24 = []
    for row in rdr:
        season = int(float(row[si]))
        counts[season] = counts.get(season, 0) + 1
        if season == 2024:
            rows24.append(row)
        # Deliberately do not collect or parse any 2025 outcome field.
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"canonical season-count drift: {counts}")
    x = pd.DataFrame(rows24, columns=hlow)
    for c in ["season","week","actual_pass_yards","actual_attempts","pred_pass_yards","pred_attempts"]:
        x[c] = pd.to_numeric(x[c], errors="raise")
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["actual_ypa"] = x.actual_pass_yards / x.actual_attempts.replace(0, np.nan)
    x["base_ypa"] = x.pred_pass_yards / x.pred_attempts.replace(0, np.nan)
    if x[["actual_ypa","base_ypa"]].isna().any().any():
        raise RuntimeError("canonical component null")
    return x

def bool_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    z = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[z.isin(["1","true","t","yes","y"])] = 1.0
    out.loc[z.isin(["0","false","f","no","n"])] = 0.0
    return out

def raw_coverage(s: pd.Series) -> float:
    if pd.api.types.is_bool_dtype(s):
        return float(s.notna().mean())
    if pd.api.types.is_numeric_dtype(s):
        return float(pd.to_numeric(s, errors="coerce").notna().mean())
    z = s.astype("string").str.strip()
    return float((z.notna() & z.ne("") & z.str.lower().ne("nan")).mean())

def norm_short(v: object) -> str:
    return re.sub(r"[^a-z]", "", str(v).lower())

def norm_qkey(v: object) -> str:
    z = re.sub(r"[^a-z]", "", str(v).lower())
    for suf in ("iii","ii","iv","jr","sr"):
        if z.endswith(suf) and len(z) > len(suf) + 3:
            z = z[:-len(suf)]
            break
    return z

def map_qkey_to_short(qkey: str, shorts: list[str]) -> str | None:
    q = norm_qkey(qkey)
    cand = []
    for s in shorts:
        if len(s) < 3:
            continue
        initial, last = s[0], s[1:]
        if q.startswith(initial) and q.endswith(last):
            cand.append((len(last), s))
    if not cand:
        return None
    cand.sort(reverse=True)
    return cand[0][1]

def prior_mask(df: pd.DataFrame, target_week: int) -> pd.Series:
    return (df.season < 2024) | ((df.season == 2024) & (df.week < target_week))

def hist_summary(df: pd.DataFrame, key_col: str, key: str | None, metrics: list[str], target_week: int, prefix: str) -> tuple[dict, bool]:
    prior = df.loc[prior_mask(df, target_week)].copy()
    league = prior
    ent = prior.loc[prior[key_col].eq(key)].copy() if key is not None else prior.iloc[0:0].copy()
    ent = ent.sort_values(["season","week","game_id"])
    natural = len(ent) > 0
    e8 = ent.tail(8)
    e3 = ent.tail(3)
    out = {}
    for m in metrics:
        lmean = float(pd.to_numeric(league[m], errors="coerce").mean()) if len(league) else 0.0
        if not np.isfinite(lmean):
            lmean = 0.0
        m8 = float(pd.to_numeric(e8[m], errors="coerce").mean()) if len(e8) else lmean
        m3 = float(pd.to_numeric(e3[m], errors="coerce").mean()) if len(e3) else lmean
        if not np.isfinite(m8): m8 = lmean
        if not np.isfinite(m3): m3 = lmean
        out[f"{prefix}_{m}_m8"] = m8
        out[f"{prefix}_{m}_m3"] = m3
        out[f"{prefix}_{m}_trend3v8"] = m3 - m8
    return out, natural

def prepare_sources(out: Path):
    source_rows, coverage_rows, merged_all = [], [], []
    for season in HISTORY_SEASONS:
        ftn_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        ftn, fm = read_url_parquet(ftn_url)
        pbp, pm = read_url_parquet(pbp_url)
        ftn, pbp = lower(ftn), lower(pbp)
        required_ftn = {"nflverse_game_id","nflverse_play_id","season","week"} | set(sum(FAMILIES.values(), []))
        required_pbp = {"game_id","play_id","season","week","season_type","posteam","defteam","passer_player_name","pass_attempt","sack"}
        miss_f = sorted(required_ftn - set(ftn.columns))
        miss_p = sorted(required_pbp - set(pbp.columns))
        if miss_f or miss_p:
            raise RuntimeError(f"source schema missing season={season} ftn={miss_f} pbp={miss_p}")
        ftn = ftn.loc[pd.to_numeric(ftn.week, errors="coerce").between(1,18)].copy()
        pbp = pbp.loc[pbp.season_type.astype(str).str.upper().eq("REG") & pd.to_numeric(pbp.week, errors="coerce").between(1,18)].copy()
        fw = set(pd.to_numeric(ftn.week, errors="coerce").dropna().astype(int))
        pw = set(pd.to_numeric(pbp.week, errors="coerce").dropna().astype(int))
        source_rows += [
            {"source":"FTN","season":season,"rows":len(ftn),"weeks_complete_1_18":REG_WEEKS.issubset(fw),**fm},
            {"source":"PBP","season":season,"rows":len(pbp),"weeks_complete_1_18":REG_WEEKS.issubset(pw),**pm},
        ]
        for fam, fields in FAMILIES.items():
            for fld in fields:
                coverage_rows.append({"family":fam,"season":season,"field":fld,"coverage":raw_coverage(ftn[fld])})
        ftn["join_game"] = ftn.nflverse_game_id.astype(str)
        ftn["join_play"] = pd.to_numeric(ftn.nflverse_play_id, errors="coerce")
        keep = ["game_id","play_id","season","week","posteam","defteam","passer_player_name","pass_attempt","sack"]
        p = pbp[keep].copy()
        p["join_game"] = p.game_id.astype(str)
        p["join_play"] = pd.to_numeric(p.play_id, errors="coerce")
        m = ftn.merge(p.drop(columns=["game_id","play_id","season","week"]), on=["join_game","join_play"], how="left", validate="one_to_one")
        join_rate = float(m.posteam.notna().mean())
        source_rows.append({"source":"FTN_PBP_JOIN","season":season,"rows":len(m),"weeks_complete_1_18":True,"url":"","bytes":0,"sha256":"","join_rate":join_rate})
        m["game_id"] = m.join_game
        m["season"] = pd.to_numeric(m.season, errors="coerce").astype(int)
        m["week"] = pd.to_numeric(m.week, errors="coerce").astype(int)
        merged_all.append(m)
    src = pd.DataFrame(source_rows)
    cov = pd.DataFrame(coverage_rows)
    src.to_csv(out/"m81_source_snapshot.csv", index=False)
    cov.to_csv(out/"m81_raw_field_coverage.csv", index=False)

    z = pd.concat(merged_all, ignore_index=True)
    for c in ["is_motion","is_screen_pass","is_rpo","is_qb_out_of_pocket","is_throw_away","is_qb_fault_sack","is_interception_worthy","is_catchable_ball","is_drop"]:
        z[c] = bool_num(z[c])
    z["n_blitzers"] = pd.to_numeric(z.n_blitzers, errors="coerce")
    z["read_num"] = pd.to_numeric(z.read_thrown.astype("string").str.extract(r"(\d+)", expand=False), errors="coerce")
    z["pass_mask"] = pd.to_numeric(z.pass_attempt, errors="coerce").eq(1)
    z["throw_mask"] = z.pass_mask & ~pd.to_numeric(z.sack, errors="coerce").eq(1)
    z["qb_short"] = z.passer_player_name.map(norm_short)
    z.loc[z.qb_short.eq("") | z.qb_short.eq("nan"), "qb_short"] = np.nan
    for c in ["n_blitzers","is_qb_out_of_pocket","is_throw_away","is_qb_fault_sack"]:
        z[f"p_{c}"] = z[c].where(z.pass_mask)
    for c in ["is_interception_worthy","is_catchable_ball","is_drop"]:
        z[f"t_{c}"] = z[c].where(z.throw_mask)
    z["t_read_num"] = z.read_num.where(z.throw_mask)
    z["t_read_late_rate"] = pd.Series(np.where(z.throw_mask & z.read_num.notna(), z.read_num.ge(2).astype(float), np.nan), index=z.index)

    off = z.loc[z.posteam.notna()].groupby(["season","week","game_id","posteam","defteam"], as_index=False).agg(
        motion_rate=("is_motion","mean"), screen_rate=("is_screen_pass","mean"), rpo_rate=("is_rpo","mean"), offense_drop_rate=("t_is_drop","mean")
    ).rename(columns={"posteam":"team","defteam":"opponent"})

    q = z.loc[z.qb_short.notna() & z.pass_mask].groupby(["season","week","game_id","posteam","defteam","qb_short"], as_index=False).agg(
        blitzers_mean=("p_n_blitzers","mean"), oop_rate=("p_is_qb_out_of_pocket","mean"), throwaway_rate=("p_is_throw_away","mean"), qb_fault_sack_rate=("p_is_qb_fault_sack","mean"),
        intworthy_rate=("t_is_interception_worthy","mean"), catchable_rate=("t_is_catchable_ball","mean"), read_mean=("t_read_num","mean"), read_late_rate=("t_read_late_rate","mean"), qb_drop_rate=("t_is_drop","mean")
    ).rename(columns={"posteam":"team","defteam":"opponent"})

    d = z.loc[z.defteam.notna() & z.posteam.notna()].groupby(["season","week","game_id","defteam"], as_index=False).agg(
        motion_allowed_rate=("is_motion","mean"), screen_allowed_rate=("is_screen_pass","mean"), rpo_allowed_rate=("is_rpo","mean"),
        blitzers_mean=("p_n_blitzers","mean"), oop_allowed_rate=("p_is_qb_out_of_pocket","mean"), throwaway_allowed_rate=("p_is_throw_away","mean"), qb_fault_sack_allowed_rate=("p_is_qb_fault_sack","mean"),
        intworthy_allowed_rate=("t_is_interception_worthy","mean"), catchable_allowed_rate=("t_is_catchable_ball","mean"), read_allowed_mean=("t_read_num","mean"), read_late_allowed_rate=("t_read_late_rate","mean")
    ).rename(columns={"defteam":"defense"})
    return src, cov, z, off, q, d

def build_features(canon: pd.DataFrame, off: pd.DataFrame, qb: pd.DataFrame, deff: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    shorts = sorted(qb.qb_short.dropna().astype(str).unique().tolist())
    rows, mapping = [], {}
    tactical_off = ["motion_rate","screen_rate","rpo_rate"]
    tactical_def = ["motion_allowed_rate","screen_allowed_rate","rpo_allowed_rate"]
    pressure_qb = ["blitzers_mean","oop_rate","throwaway_rate","qb_fault_sack_rate"]
    pressure_def = ["blitzers_mean","oop_allowed_rate","throwaway_allowed_rate","qb_fault_sack_allowed_rate"]
    throw_qb = ["intworthy_rate","catchable_rate","read_mean","read_late_rate"]
    throw_def = ["intworthy_allowed_rate","catchable_allowed_rate","read_allowed_mean","read_late_allowed_rate"]

    for r in canon.itertuples(index=False):
        week = int(r.week); qkey = str(r.player_clean_key); team = str(r.team); opp = str(r.opponent)
        qshort = map_qkey_to_short(qkey, shorts)
        mapping[qkey] = qshort
        base = {"season":2024,"week":week,"team":team,"opponent":opp,"player_clean_key":qkey,"qb_short":qshort,
                "actual_pass_yards":float(r.actual_pass_yards),"actual_attempts":float(r.actual_attempts),"actual_ypa":float(r.actual_ypa),
                "pred_pass_yards":float(r.pred_pass_yards),"pred_attempts":float(r.pred_attempts),"base_ypa":float(r.base_ypa)}
        a, an = hist_summary(off,"team",team,tactical_off,week,"tac_off")
        b, bn = hist_summary(deff,"defense",opp,tactical_def,week,"tac_def")
        base.update(a); base.update(b)
        for om, dm in zip(tactical_off, tactical_def):
            base[f"tac_int_{om}"] = base[f"tac_off_{om}_m8"] * base[f"tac_def_{dm}_m8"]
        base["cov_tactical"] = float(an and bn)

        a, an = hist_summary(qb,"qb_short",qshort,pressure_qb,week,"prs_qb")
        b, bn = hist_summary(deff,"defense",opp,pressure_def,week,"prs_def")
        base.update(a); base.update(b)
        base["prs_int_oop_x_blitz"] = base["prs_qb_oop_rate_m8"] * base["prs_def_blitzers_mean_m8"]
        base["prs_int_throwaway_x_blitz"] = base["prs_qb_throwaway_rate_m8"] * base["prs_def_blitzers_mean_m8"]
        base["prs_int_qbfault_x_blitz"] = base["prs_qb_qb_fault_sack_rate_m8"] * base["prs_def_blitzers_mean_m8"]
        base["prs_int_exposure_x_defblitz"] = base["prs_qb_blitzers_mean_m8"] * base["prs_def_blitzers_mean_m8"]
        base["cov_pressure"] = float(an and bn)

        a, an = hist_summary(qb,"qb_short",qshort,throw_qb,week,"tdq_qb")
        b, bn = hist_summary(deff,"defense",opp,throw_def,week,"tdq_def")
        base.update(a); base.update(b)
        for qm, dm in zip(throw_qb, throw_def):
            base[f"tdq_int_{qm}"] = base[f"tdq_qb_{qm}_m8"] * base[f"tdq_def_{dm}_m8"]
        base["cov_throw"] = float(an and bn)

        a, an = hist_summary(qb,"qb_short",qshort,["qb_drop_rate"],week,"rea_qb")
        b, bn = hist_summary(off,"team",team,["offense_drop_rate"],week,"rea_off")
        base.update(a); base.update(b)
        base["cov_receiver"] = float(an and bn)
        rows.append(base)

    f = pd.DataFrame(rows)
    return f, {"identity_map_rate":float(f.qb_short.notna().mean()), "mapping":mapping}

def family_features(df: pd.DataFrame) -> dict[str,list[str]]:
    return {
        "TACTICAL_CALL_STRUCTURE": [c for c in df if c.startswith("tac_")],
        "PRESSURE_RESPONSE": [c for c in df if c.startswith("prs_")],
        "THROW_DECISION_QUALITY": [c for c in df if c.startswith("tdq_")],
        "RECEIVER_ERROR_ATTRIBUTION": [c for c in df if c.startswith("rea_")],
    }

def mae(a,p): return float(np.mean(np.abs(np.asarray(a,float)-np.asarray(p,float))))
def rmse(a,p): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(p,float))**2)))
def corr(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float)
    return float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else np.nan

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], target: str):
    xtr=train[feats].astype(float); xte=test[feats].astype(float)
    if not np.isfinite(xtr.to_numpy()).all() or not np.isfinite(xte.to_numpy()).all():
        raise RuntimeError(f"nonfinite features target={target}")
    sc=StandardScaler(); a=sc.fit_transform(xtr); b=sc.transform(xte)
    model=Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(a, train[target].astype(float))
    co=pd.DataFrame({"feature":feats,"target":target,"coefficient":model.coef_,"scale_mean":sc.mean_,"scale_scale":sc.scale_})
    return model.predict(b), co

def evaluate(test: pd.DataFrame, attcorr: np.ndarray, ypacorr: np.ndarray):
    y=test.actual_pass_yards.to_numpy(float); base=test.pred_pass_yards.to_numpy(float)
    aa=test.actual_attempts.to_numpy(float); pa=test.pred_attempts.to_numpy(float)
    ya=test.actual_ypa.to_numpy(float); py=test.base_ypa.to_numpy(float)
    ca=np.clip(pa+attcorr,*ATT_BOUNDS); cy=np.clip(py+ypacorr,*YPA_BOUNDS); pred=ca*cy
    metrics={"n":len(test),"baseline_pass_mae":mae(y,base),"corrected_pass_mae":mae(y,pred),"pass_mae_gain":mae(y,base)-mae(y,pred),
             "baseline_pass_rmse":rmse(y,base),"corrected_pass_rmse":rmse(y,pred),"pass_rmse_gain":rmse(y,base)-rmse(y,pred),
             "baseline_pass_corr":corr(y,base),"corrected_pass_corr":corr(y,pred),"pass_corr_gain":corr(y,pred)-corr(y,base),
             "baseline_tails100":int(np.sum(np.abs(base-y)>=100)),"corrected_tails100":int(np.sum(np.abs(pred-y)>=100)),
             "attempt_baseline_mae":mae(aa,pa),"attempt_corrected_mae":mae(aa,ca),"attempt_mae_gain":mae(aa,pa)-mae(aa,ca),
             "ypa_baseline_mae":mae(ya,py),"ypa_corrected_mae":mae(ya,cy),"ypa_mae_gain":mae(ya,py)-mae(ya,cy)}
    p=test[["season","week","team","opponent","player_clean_key","actual_pass_yards","pred_pass_yards","actual_attempts","pred_attempts","actual_ypa","base_ypa"]].copy()
    p["att_correction"]=attcorr; p["ypa_correction"]=ypacorr; p["corrected_attempts"]=ca; p["corrected_ypa"]=cy; p["corrected_pass_yards"]=pred
    return metrics,p

def bootstrap(test: pd.DataFrame, pred: np.ndarray) -> dict:
    rng=np.random.default_rng(BOOT_SEED); y=test.actual_pass_yards.to_numpy(float); base=test.pred_pass_yards.to_numpy(float); gains=[]; n=len(test)
    for _ in range(BOOT_N):
        q=rng.integers(0,n,n); gains.append(mae(y[q],base[q])-mae(y[q],pred[q]))
    g=np.asarray(gains)
    return {"boot_n":BOOT_N,"p_pass_mae_gain_gt0":float(np.mean(g>0)),"gain_p10":float(np.quantile(g,.10)),"gain_p50":float(np.quantile(g,.50)),"gain_p90":float(np.quantile(g,.90))}

def gate_rows(name: str, m: dict, bt: dict, contract_ok: bool) -> pd.DataFrame:
    component_ok = (m["attempt_mae_gain"] >= 0.10) or (m["ypa_mae_gain"] >= 0.03)
    rows=[("pass_mae_gain",m["pass_mae_gain"],">=0.75",m["pass_mae_gain"]>=0.75),
          ("pass_corr_gain",m["pass_corr_gain"],">=0.015",m["pass_corr_gain"]>=0.015),
          ("pass_rmse_nonworse",m["corrected_pass_rmse"]-m["baseline_pass_rmse"],"<=0",m["corrected_pass_rmse"]<=m["baseline_pass_rmse"]),
          ("tails100_nonincrease",m["corrected_tails100"]-m["baseline_tails100"],"<=0",m["corrected_tails100"]<=m["baseline_tails100"]),
          ("component_signal",1 if component_ok else 0,"==1",component_ok),
          ("bootstrap_support",bt["p_pass_mae_gain_gt0"],">=0.70",bt["p_pass_mae_gain_gt0"]>=0.70),
          ("source_feature_contract",1 if contract_ok else 0,"==1",contract_ok)]
    return pd.DataFrame(rows,columns=["gate","value","threshold","passed"]).assign(candidate=name)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--canonical",type=Path,default=Path("data/backtests/qb_frontier_canonical_v3_football_only/qb_frontier_canonical_v3_football_only.csv")); ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); out=a.out; out.mkdir(parents=True,exist_ok=True)
    canon=load_canonical_2024(a.canonical)
    src,cov,merged,off,qb,deff=prepare_sources(out)
    source_complete=bool(src.loc[src.source.isin(["FTN","PBP"]),"weeks_complete_1_18"].fillna(False).all() and (src.loc[src.source.eq("FTN_PBP_JOIN"),"join_rate"].fillna(0)>=MIN_FTN_PBP_JOIN).all() and (cov.coverage>=MIN_RAW_FIELD_COVERAGE).all())
    feat,idmeta=build_features(canon,off,qb,deff)
    if len(feat)!=444 or set(feat.season.unique())!={2024}: raise RuntimeError("M81 target cohort drift")
    id_ok=idmeta["identity_map_rate"]>=MIN_IDENTITY_MAP
    fam_feats=family_features(feat); all_feature_cols=sorted(set(sum(fam_feats.values(),[])))
    if feat[all_feature_cols].isna().any().any():
        bad=feat[all_feature_cols].isna().sum(); raise RuntimeError(f"feature nulls: {bad[bad>0].to_dict()}")
    feat["att_resid"]=feat.actual_attempts-feat.pred_attempts; feat["ypa_resid"]=feat.actual_ypa-feat.base_ypa
    train=feat.loc[feat.week<=9].copy(); hold=feat.loc[feat.week>=10].copy()
    if len(train)<150 or len(hold)<150: raise RuntimeError(f"development split too small train={len(train)} hold={len(hold)}")

    fd=[]
    for fam,cols in fam_feats.items():
        for c in cols:
            fd.append({"family":fam,"feature":c,"closest_prior":CROSSWALK[fam],"history_rule":"2023 + strictly-prior 2024 only","transform":"fixed trailing-8, trailing-3, 3-vs-8 trend and preregistered same-family interactions"})
    pd.DataFrame(fd).to_csv(out/"m81_feature_dictionary.csv",index=False)
    feat.drop(columns=["att_resid","ypa_resid"]).to_csv(out/"m81_2024_pregame_features.csv",index=False)
    covrep=pd.DataFrame([
        {"family":"TACTICAL_CALL_STRUCTURE","natural_history_coverage":float(feat.cov_tactical.mean()),"feature_count":len(fam_feats["TACTICAL_CALL_STRUCTURE"])},
        {"family":"PRESSURE_RESPONSE","natural_history_coverage":float(feat.cov_pressure.mean()),"feature_count":len(fam_feats["PRESSURE_RESPONSE"])},
        {"family":"THROW_DECISION_QUALITY","natural_history_coverage":float(feat.cov_throw.mean()),"feature_count":len(fam_feats["THROW_DECISION_QUALITY"])},
        {"family":"RECEIVER_ERROR_ATTRIBUTION","natural_history_coverage":float(feat.cov_receiver.mean()),"feature_count":len(fam_feats["RECEIVER_ERROR_ATTRIBUTION"])},])
    covrep["identity_map_rate"]=idmeta["identity_map_rate"]; covrep["source_contract_ok"]=source_complete; covrep.to_csv(out/"m81_family_coverage.csv",index=False)
    contract={"migration":"M81","canonical_sha256":CANON_SHA,"target_season":2024,"fit_weeks":"1-9","holdout_weeks":"10-18","history_seasons":[2023,2024],"2025_target_outcomes_accessed":False,"sportsbook_features_used":False,"target_game_ftn_used":False,"ridge_alpha":RIDGE_ALPHA,"attempt_bounds":ATT_BOUNDS,"ypa_bounds":YPA_BOUNDS,"source_complete":source_complete,"identity_map_rate":idmeta["identity_map_rate"],"identity_contract_ok":id_ok,"production_actionable":False}
    (out/"m81_source_contract.json").write_text(json.dumps(contract,indent=2)+"\n")

    metrics_rows=[]; boot_rows=[]; gates=[]; preds=[]; coefs=[]; survivors=[]; family_contract_ok=bool(source_complete and id_ok)
    for fam,cols in fam_feats.items():
        ac,ca=fit_predict(train,hold,cols,"att_resid"); yc,cy=fit_predict(train,hold,cols,"ypa_resid")
        m,p=evaluate(hold,ac,yc); m["candidate"]=fam; bt=bootstrap(hold,p.corrected_pass_yards.to_numpy(float)); bt["candidate"]=fam; gd=gate_rows(fam,m,bt,family_contract_ok)
        survivor=bool(gd.passed.all()); m["development_survivor"]=survivor
        metrics_rows.append(m); boot_rows.append(bt); gates.append(gd); p["candidate"]=fam; preds.append(p); ca["candidate"]=fam; cy["candidate"]=fam; coefs += [ca,cy]
        if survivor: survivors.append(fam)

    stack_name=None
    if len(survivors)>=2:
        stack_name="SURVIVOR_STACK"; cols=sorted(set(sum([fam_feats[f] for f in survivors],[])))
        ac,ca=fit_predict(train,hold,cols,"att_resid"); yc,cy=fit_predict(train,hold,cols,"ypa_resid")
        m,p=evaluate(hold,ac,yc); m["candidate"]=stack_name; bt=bootstrap(hold,p.corrected_pass_yards.to_numpy(float)); bt["candidate"]=stack_name; gd=gate_rows(stack_name,m,bt,family_contract_ok)
        m["development_survivor"]=bool(gd.passed.all()); metrics_rows.append(m); boot_rows.append(bt); gates.append(gd); p["candidate"]=stack_name; preds.append(p); ca["candidate"]=stack_name; cy["candidate"]=stack_name; coefs += [ca,cy]

    metrics=pd.DataFrame(metrics_rows); boots=pd.DataFrame(boot_rows); gate=pd.concat(gates,ignore_index=True); predictions=pd.concat(preds,ignore_index=True); coef=pd.concat(coefs,ignore_index=True)
    metrics.to_csv(out/"m81_development_metrics.csv",index=False); boots.to_csv(out/"m81_bootstrap_summary.csv",index=False); gate.to_csv(out/"m81_gate_table.csv",index=False); predictions.to_csv(out/"m81_2024_holdout_predictions.csv",index=False); coef.to_csv(out/"m81_coefficients.csv",index=False)

    eligible=list(survivors)
    if stack_name and bool(metrics.loc[metrics.candidate.eq(stack_name),"development_survivor"].iloc[0]): eligible.append(stack_name)
    if eligible:
        rank=metrics.loc[metrics.candidate.isin(eligible)].sort_values(["corrected_pass_mae","corrected_pass_rmse","corrected_pass_corr","corrected_tails100"],ascending=[True,True,False,True]); winner=str(rank.iloc[0].candidate)
        winner_features = sorted(set(sum([fam_feats[f] for f in survivors],[]))) if winner=="SURVIVOR_STACK" else fam_feats[winner]
        decision={"migration":"M81","status":"FREEZE_M82_CANDIDATE","independent_survivors":survivors,"stack_tested":stack_name is not None,"m82_candidate":winner,"m82_feature_columns":winner_features,"m82_fit_contract":"fit same frozen architecture on all 2024 canonical target rows, using 2023 + strictly-prior 2024 FTN history; score 2025 once with no tuning","2025_target_outcomes_accessed":False,"production_actionable":False}
    else:
        decision={"migration":"M81","status":"NO_FTN_DEVELOPMENT_SURVIVOR","independent_survivors":survivors,"stack_tested":stack_name is not None,"m82_candidate":None,"m82_feature_columns":[],"2025_target_outcomes_accessed":False,"production_actionable":False}
    (out/"m81_survivor_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    print("[m81_contract]",json.dumps(contract,sort_keys=True)); print("[m81_metrics]"); print(metrics.to_string(index=False)); print("[m81_decision]",json.dumps(decision,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
