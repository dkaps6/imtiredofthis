#!/usr/bin/env python3
"""Post-M38 WR receiving-yard error decomposition.

Diagnostic only. The input component_predictions.csv MUST be rebuilt from the
exact M38 merge commit. No sportsbook data is read or accepted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

WR_POSITIONS = {"WR", "LWR", "RWR", "SWR"}
WR_MULT = (1.40, 1.14, 0.91, 0.78)
COMPONENTS = ("OPPORTUNITY", "CONVERSION", "NON_EXPLOSIVE_EFFICIENCY", "EXPLOSIVE_YARDAGE")
PSEUDO_RECEPTIONS = 20.0
EXPECTED_ALL_REC = {
    "n": 4647,
    "mae": 17.099905,
    "rmse": 25.196100,
    "bias": -5.238641,
    "correlation": 0.567946,
}


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def canon_team(v) -> str:
    s = str(v or "").upper().strip()
    return {"JAC": "JAX", "LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR"}.get(s, s)


def num(x: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in x.columns:
        return pd.Series(default, index=x.index, dtype=float)
    return pd.to_numeric(x[col], errors="coerce")


def score(actual: pd.Series, pred: pd.Series) -> dict:
    z = pd.DataFrame({"a": pd.to_numeric(actual, errors="coerce"), "p": pd.to_numeric(pred, errors="coerce")}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    e = z.p - z.a
    corr = float(z.p.corr(z.a)) if len(z) > 1 and z.p.nunique() > 1 and z.a.nunique() > 1 else np.nan
    return {"n": int(len(z)), "mae": float(e.abs().mean()), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(e.mean()), "correlation": corr}


def validate_m38_parent(cp: pd.DataFrame) -> dict:
    q = cp.loc[cp["market"].astype(str).str.lower().eq("rec_yards")].copy()
    q = q.loc[num(q, "mc_proj").notna() & num(q, "actual").notna()].copy()
    got = score(q["actual"], q["mc_proj"])
    if got["n"] != EXPECTED_ALL_REC["n"]:
        raise RuntimeError(f"M38 parent row drift: {got['n']} != {EXPECTED_ALL_REC['n']}")
    tolerances = {"mae": 0.02, "rmse": 0.03, "bias": 0.02, "correlation": 0.002}
    for k, tol in tolerances.items():
        if not np.isfinite(got[k]) or abs(got[k] - EXPECTED_ALL_REC[k]) > tol:
            raise RuntimeError(f"M38 parent {k} drift: got={got[k]:.6f}, expected={EXPECTED_ALL_REC[k]:.6f}, tol={tol}")
    return got


def sharpen_and_rank(q: pd.DataFrame) -> pd.DataFrame:
    """Recreate M38 expected target shares and pregame WR ranks."""
    out = []
    keys = ["season", "week", "event_id", "team"]
    for _, g0 in q.groupby(keys, dropna=False, sort=False):
        g = g0.copy()
        raw = num(g, "rules_tgt_share", 0.0).fillna(0.0).clip(0.0, 0.95).to_numpy(float)
        pos = g["position"].fillna("").astype(str).str.upper().to_numpy()
        wr_idx = np.flatnonzero(np.isin(pos, list(WR_POSITIONS)))
        rank = np.full(len(g), np.nan)
        adj = raw.copy()
        if len(wr_idx):
            wr = raw[wr_idx].copy()
            order = np.argsort(-wr, kind="stable")
            for r, idx in enumerate(order, start=1):
                rank[wr_idx[idx]] = r
            if len(wr_idx) > 1 and float(wr.sum()) > 0:
                mult = np.ones(len(wr), dtype=float)
                for r0, idx in enumerate(order):
                    mult[idx] = WR_MULT[min(r0, len(WR_MULT) - 1)]
                sh = wr * mult
                if float(sh.sum()) > 0:
                    sh *= float(wr.sum()) / float(sh.sum())
                    adj[wr_idx] = sh
        # Canonical target allocator caps total modeled share at 95%.
        s = float(adj.sum())
        if s > 0.95:
            adj *= 0.95 / s
        plays = float(np.nanmean(num(g, "rules_plays_est", 64.0).fillna(64.0)))
        rate = float(np.nanmean(num(g, "rules_pass_rate", 0.57).fillna(0.57)))
        plays = float(np.clip(plays, 50.0, 80.0))
        rate = float(np.clip(rate, 0.35, 0.75))
        g["m38_final_target_share"] = adj
        g["pred_targets"] = plays * rate * adj
        g["wr_rank_num"] = rank
        out.append(g)
    z = pd.concat(out, ignore_index=True)
    z["wr_rank"] = np.where(z["wr_rank_num"].eq(1), "WR1", np.where(z["wr_rank_num"].eq(2), "WR2", np.where(z["wr_rank_num"].eq(3), "WR3", "WR4+")))
    return z


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_pbp(seasons=[2024, 2025])
    x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    x = lower(x)
    if "season_type" in x.columns:
        x = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
    x["season"] = num(x, "season")
    x["week"] = num(x, "week")
    x = x.loc[x["season"].isin([2024, 2025]) & x["week"].between(1, 18)].copy()
    if "posteam" not in x.columns or "receiver_player_id" not in x.columns:
        raise RuntimeError("PBP missing posteam/receiver_player_id")
    x["team"] = x["posteam"].map(canon_team)
    complete = num(x, "complete_pass", 0).fillna(0).eq(1)
    rid = x["receiver_player_id"].fillna("").astype(str).str.strip()
    x = x.loc[complete & rid.ne("") & rid.ne("nan")].copy()
    if "receiving_yards" in x.columns:
        y = num(x, "receiving_yards", 0.0).fillna(0.0)
    else:
        y = num(x, "yards_gained", 0.0).fillna(0.0)
    x["rec_gain"] = y
    x["explosive20"] = y.ge(20).astype(int)
    x["explosive40"] = y.ge(40).astype(int)
    x["explosive_yards"] = np.where(y.ge(20), y, 0.0)
    x["nonexplosive_yards"] = np.where(y.lt(20), y, 0.0)
    x["reception"] = 1
    return x


def build_pbp_games(pbp: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    g = pbp.groupby(["season", "week", "team", "receiver_player_id"], as_index=False).agg(
        pbp_receptions=("reception", "sum"),
        pbp_rec_yards=("rec_gain", "sum"),
        explosive20_receptions=("explosive20", "sum"),
        explosive40_receptions=("explosive40", "sum"),
        explosive_yards=("explosive_yards", "sum"),
        nonexplosive_yards=("nonexplosive_yards", "sum"),
    ).rename(columns={"receiver_player_id": "player_id"})
    l = logs.copy()
    l["team"] = l["team"].map(canon_team)
    l["player_id"] = l.get("player_id", pd.Series("", index=l.index)).fillna("").astype(str).str.strip()
    l["position"] = l.get("position", pd.Series("", index=l.index)).fillna("").astype(str).str.upper().str.strip()
    pos = l[["season", "week", "team", "player_id", "position", "player_clean_key"]].drop_duplicates(["season", "week", "team", "player_id"])
    g = g.merge(pos, on=["season", "week", "team", "player_id"], how="left", validate="many_to_one")
    return g


def prior_split_for_row(target: pd.Series, pbpg: pd.DataFrame) -> tuple[float, float, dict]:
    season = int(target["season"]); week = int(target["week"]); pid = str(target.get("player_id", "")).strip()
    prior = pbpg.loc[(pbpg["season"] < season) | ((pbpg["season"] == season) & (pbpg["week"] < week))].copy()
    prior_wr = prior.loc[prior["position"].isin(WR_POSITIONS)].copy()
    league_rec = float(prior_wr["pbp_receptions"].sum())
    league_exp = float(prior_wr["explosive_yards"].sum())
    league_non = float(prior_wr["nonexplosive_yards"].sum())
    if league_rec <= 0 or league_exp + league_non <= 0:
        raise RuntimeError(f"no strict-prior league WR receiving sample for {season} W{week}")
    league_exp_ypr = league_exp / league_rec
    league_non_ypr = league_non / league_rec
    league_frac = league_exp_ypr / (league_exp_ypr + league_non_ypr)

    pp = prior.loc[prior["player_id"].astype(str).eq(pid)].sort_values(["season", "week"]).tail(8)
    prec = float(pp["pbp_receptions"].sum())
    pexp = float(pp["explosive_yards"].sum())
    pnon = float(pp["nonexplosive_yards"].sum())
    shr_exp_ypr = (pexp + PSEUDO_RECEPTIONS * league_exp_ypr) / (prec + PSEUDO_RECEPTIONS)
    shr_non_ypr = (pnon + PSEUDO_RECEPTIONS * league_non_ypr) / (prec + PSEUDO_RECEPTIONS)
    shr_frac = shr_exp_ypr / max(shr_exp_ypr + shr_non_ypr, 1e-12)
    meta = {"prior_player_receptions": prec, "prior_player_games": int(len(pp)), "prior_league_receptions": league_rec}
    return float(np.clip(league_frac, 0.0, 1.0)), float(np.clip(shr_frac, 0.0, 1.0)), meta


def prepare_casebook(cp: pd.DataFrame, logs: pd.DataFrame, pbpg: pd.DataFrame) -> pd.DataFrame:
    r = cp.loc[cp["market"].astype(str).str.lower().eq("rec_yards")].copy()
    r["position"] = r["position"].fillna("").astype(str).str.upper().str.strip()
    r = r.loc[r["position"].isin(WR_POSITIONS) & num(r, "mc_proj").notna() & num(r, "actual").notna()].copy()
    r["team"] = r["team"].map(canon_team)
    r = sharpen_and_rank(r)

    l = logs.copy(); l["team"] = l["team"].map(canon_team)
    l["player_id"] = l.get("player_id", pd.Series("", index=l.index)).fillna("").astype(str).str.strip()
    keep = ["season", "week", "team", "player_clean_key", "player_id", "targets", "receptions", "rec_yards"]
    missing = [c for c in keep if c not in l.columns]
    if missing: raise RuntimeError(f"player logs missing columns: {missing}")
    l = l[keep].drop_duplicates(["season", "week", "team", "player_clean_key"])
    z = r.merge(l, on=["season", "week", "team", "player_clean_key"], how="left", validate="many_to_one", suffixes=("", "_log"))
    if z["player_id"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("WR casebook has missing player_id after weekly-log join")

    p = pbpg[["season", "week", "team", "player_id", "pbp_receptions", "pbp_rec_yards", "explosive20_receptions", "explosive40_receptions", "explosive_yards", "nonexplosive_yards"]].copy()
    z = z.merge(p, on=["season", "week", "team", "player_id"], how="left", validate="many_to_one")
    for c in ["pbp_receptions", "pbp_rec_yards", "explosive20_receptions", "explosive40_receptions", "explosive_yards", "nonexplosive_yards"]:
        z[c] = num(z, c, 0.0).fillna(0.0)
    for c in ["targets", "receptions", "rec_yards", "mc_proj", "rules_catch_rate", "pred_targets"]:
        z[c] = num(z, c)
    # PBP may have tiny provider/stat corrections vs weekly stats. Preserve PBP split,
    # then scale its yard components to canonical weekly receiving yards so truth sums exactly.
    mismatch = (z["pbp_rec_yards"] - z["rec_yards"]).abs()
    z["pbp_yard_abs_mismatch"] = mismatch
    missing_split = z["rec_yards"].gt(0) & z["pbp_rec_yards"].le(0)
    if missing_split.any():
        sample = z.loc[missing_split, ["season", "week", "team", "player", "player_id", "rec_yards"]].head(10)
        raise RuntimeError("positive WR receiving yards missing PBP split: " + repr(sample.to_dict("records")))
    scale = np.where(z["pbp_rec_yards"].abs() > 1e-12, z["rec_yards"] / z["pbp_rec_yards"], 0.0)
    z["actual_explosive_yards"] = z["explosive_yards"] * scale
    z["actual_nonexplosive_yards"] = z["nonexplosive_yards"] * scale
    if float((z["actual_explosive_yards"] + z["actual_nonexplosive_yards"] - z["rec_yards"]).abs().max()) > 1e-8:
        raise RuntimeError("PBP yard split does not sum to canonical actual receiving yards")

    z["pred_catch_rate"] = z["rules_catch_rate"].fillna(0.64).clip(0.001, 0.999)
    z["pred_targets"] = z["pred_targets"].clip(lower=1e-8)
    z["parent_mc_yards"] = z["mc_proj"]
    z["parent_implied_ypr"] = z["parent_mc_yards"] / (z["pred_targets"] * z["pred_catch_rate"])
    z["actual_catch_rate"] = np.where(z["targets"] > 0, z["receptions"] / z["targets"], 0.0)
    z["actual_nonexpl_ypr"] = np.where(z["receptions"] > 0, z["actual_nonexplosive_yards"] / z["receptions"], 0.0)
    z["actual_expl_ypr"] = np.where(z["receptions"] > 0, z["actual_explosive_yards"] / z["receptions"], 0.0)

    league_fracs=[]; shr_fracs=[]; pg=[]; pr=[]; lr=[]
    for _, row in z.iterrows():
        lf, sf, meta = prior_split_for_row(row, pbpg)
        league_fracs.append(lf); shr_fracs.append(sf); pg.append(meta["prior_player_games"]); pr.append(meta["prior_player_receptions"]); lr.append(meta["prior_league_receptions"])
    z["league_prior_explosive_fraction"] = league_fracs
    z["player8_shrunk_explosive_fraction"] = shr_fracs
    z["prior_player_games"] = pg; z["prior_player_receptions"] = pr; z["prior_league_receptions"] = lr
    z["parent_error"] = z["parent_mc_yards"] - z["rec_yards"]
    z["parent_underprediction"] = z["rec_yards"] - z["parent_mc_yards"]
    return z


def prediction_for_mask(g: pd.DataFrame, scheme: str, mask: int) -> np.ndarray:
    frac_col = "league_prior_explosive_fraction" if scheme == "LEAGUE_PRIOR" else "player8_shrunk_explosive_fraction"
    frac = g[frac_col].to_numpy(float)
    ypr = g["parent_implied_ypr"].to_numpy(float)
    base = {
        "OPPORTUNITY": g["pred_targets"].to_numpy(float),
        "CONVERSION": g["pred_catch_rate"].to_numpy(float),
        "NON_EXPLOSIVE_EFFICIENCY": ypr * (1.0 - frac),
        "EXPLOSIVE_YARDAGE": ypr * frac,
    }
    truth = {
        "OPPORTUNITY": g["targets"].to_numpy(float),
        "CONVERSION": g["actual_catch_rate"].to_numpy(float),
        "NON_EXPLOSIVE_EFFICIENCY": g["actual_nonexpl_ypr"].to_numpy(float),
        "EXPLOSIVE_YARDAGE": g["actual_expl_ypr"].to_numpy(float),
    }
    vals={}
    for i,c in enumerate(COMPONENTS): vals[c] = truth[c] if (mask & (1 << i)) else base[c]
    return vals["OPPORTUNITY"] * vals["CONVERSION"] * (vals["NON_EXPLOSIVE_EFFICIENCY"] + vals["EXPLOSIVE_YARDAGE"])


def shapley_slice(g: pd.DataFrame, scheme: str, slice_name: str) -> list[dict]:
    if g.empty: return []
    actual = g["rec_yards"].to_numpy(float)
    mae={}
    for mask in range(16):
        pred=prediction_for_mask(g, scheme, mask)
        mae[mask]=float(np.mean(np.abs(pred-actual)))
    empty_pred=prediction_for_mask(g, scheme, 0)
    full_pred=prediction_for_mask(g, scheme, 15)
    if float(np.max(np.abs(empty_pred-g["parent_mc_yards"].to_numpy(float)))) > 1e-9:
        raise RuntimeError(f"{scheme}/{slice_name}: empty subset does not reproduce M38 MC")
    if float(np.max(np.abs(full_pred-actual))) > 1e-9:
        raise RuntimeError(f"{scheme}/{slice_name}: full subset does not reproduce actual")
    ncomp=len(COMPONENTS); rows=[]; phi={}
    for i,c in enumerate(COMPONENTS):
        val=0.0
        for mask in range(16):
            if mask & (1 << i): continue
            k=int(mask.bit_count())
            w=math.factorial(k)*math.factorial(ncomp-k-1)/math.factorial(ncomp)
            val += w*(mae[mask]-mae[mask | (1 << i)])
        phi[c]=float(val)
    if abs(sum(phi.values()) - mae[0]) > 1e-9:
        raise RuntimeError(f"{scheme}/{slice_name}: Shapley sum mismatch {sum(phi.values())} vs {mae[0]}")
    for c in COMPONENTS:
        rows.append({"scheme":scheme,"slice":slice_name,"n":len(g),"parent_mae":mae[0],"component":c,"shapley_mae_recovery":phi[c],"share_of_parent_headroom":phi[c]/mae[0] if mae[0]>0 else np.nan})
    return rows


def slices(z: pd.DataFrame) -> dict[str,pd.DataFrame]:
    out={"ALL_WR":z,"WEEK1":z.loc[z.week.eq(1)],"W2_18":z.loc[z.week.ge(2)],"W13_18":z.loc[z.week.ge(13)],"ACTUAL_100_PLUS":z.loc[z.rec_yards.ge(100)],"UNDER_25_PLUS":z.loc[z.parent_underprediction.ge(25)],"UNDER_50_PLUS":z.loc[z.parent_underprediction.ge(50)],"OVER_25_PLUS":z.loc[z.parent_error.ge(25)],"HAS_20_PLUS":z.loc[z.explosive20_receptions.ge(1)],"NO_20_PLUS":z.loc[z.explosive20_receptions.eq(0)]}
    for rank in ["WR1","WR2","WR3","WR4+"]: out[rank]=z.loc[z.wr_rank.eq(rank)]
    return out


def disposition(summary: pd.DataFrame) -> dict:
    allw=summary.loc[summary["slice"].eq("ALL_WR")].copy()
    tops={}; pcts={}
    for scheme,g in allw.groupby("scheme"):
        g=g.sort_values("shapley_mae_recovery",ascending=False); tops[scheme]=str(g.iloc[0]["component"]); pcts[scheme]={str(r.component):float(r.share_of_parent_headroom) for _,r in g.iterrows()}
    if len(set(tops.values())) != 1:
        return {"disposition":"MIXED_WR_ERROR_COMPONENTS","top_by_scheme":tops,"shares":pcts}
    top=next(iter(tops.values())); min_overall=min(pcts[s].get(top,-999) for s in pcts)
    if top=="OPPORTUNITY" and min_overall>=0.35: d="OPPORTUNITY_DOMINANT"
    elif top=="CONVERSION" and min_overall>=0.30: d="CONVERSION_DOMINANT"
    elif top=="NON_EXPLOSIVE_EFFICIENCY" and min_overall>=0.30: d="NON_EXPLOSIVE_EFFICIENCY_DOMINANT"
    elif top=="EXPLOSIVE_YARDAGE":
        tail=summary.loc[(summary["slice"].eq("UNDER_50_PLUS"))&(summary["component"].eq("EXPLOSIVE_YARDAGE"))]
        tail_min=float(tail["share_of_parent_headroom"].min()) if len(tail)==len(pcts) else -999
        d="EXPLOSIVE_YARDAGE_DOMINANT" if (min_overall>=0.30 or tail_min>=0.50) else "MIXED_WR_ERROR_COMPONENTS"
    else: d="MIXED_WR_ERROR_COMPONENTS"
    return {"disposition":d,"top_component":top,"min_overall_share":min_overall,"top_by_scheme":tops,"shares":pcts}


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    cp=lower(pd.read_csv(args.predictions,low_memory=False)); logs=lower(pd.read_csv(args.player_logs,low_memory=False))
    for c in ["season","week"]:
        cp[c]=num(cp,c); logs[c]=num(logs,c)
    parent_check=validate_m38_parent(cp)
    pbp=load_pbp(); pbpg=build_pbp_games(pbp,logs)
    z=prepare_casebook(cp,logs,pbpg)
    rows=[]
    for scheme in ["LEAGUE_PRIOR","PLAYER8_SHRUNK"]:
        for name,g in slices(z).items(): rows.extend(shapley_slice(g,scheme,name))
    s=pd.DataFrame(rows)
    d=disposition(s)
    desc=[]
    for name,g in slices(z).items():
        if g.empty: continue
        desc.append({"slice":name,"n":len(g),"parent_mae":float(g.parent_error.abs().mean()),"mean_actual_yards":float(g.rec_yards.mean()),"mean_parent_yards":float(g.parent_mc_yards.mean()),"games_with_20_plus_rate":float(g.explosive20_receptions.ge(1).mean()),"games_with_40_plus_rate":float(g.explosive40_receptions.ge(1).mean()),"mean_explosive_yards":float(g.actual_explosive_yards.mean()),"mean_nonexplosive_yards":float(g.actual_nonexplosive_yards.mean())})
    coverage={"wr_rows":int(len(z)),"player_prior_games_positive_rate":float(z.prior_player_games.gt(0).mean()),"player_prior_receptions_positive_rate":float(z.prior_player_receptions.gt(0).mean()),"pbp_yard_mismatch_mean":float(z.pbp_yard_abs_mismatch.mean()),"pbp_yard_mismatch_max":float(z.pbp_yard_abs_mismatch.max())}
    z.to_csv(args.out_dir/"wr_post_m38_casebook.csv",index=False)
    s.to_csv(args.out_dir/"wr_post_m38_shapley.csv",index=False)
    pd.DataFrame(desc).to_csv(args.out_dir/"wr_post_m38_descriptive.csv",index=False)
    with (args.out_dir/"wr_post_m38_result.json").open("w") as f: json.dump({"m38_parent_check":parent_check,"coverage":coverage,**d},f,indent=2,sort_keys=True)
    print("[wr-post-m38] parent",parent_check)
    print("[wr-post-m38] coverage",coverage)
    print(s.loc[s["slice"].isin(["ALL_WR","UNDER_50_PLUS","WR1","WR2","WR3","WR4+"])].to_string(index=False))
    print("[wr-post-m38] disposition",json.dumps(d,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
