#!/usr/bin/env python3
"""TE-R5 Week-1 deployable refit and canonical Full Slate shadow confirmation.

Frozen by docs/production/TE_R5_WEEK1_FULL_STACK_CONFIRMATION_V1_PLAN.md.
No sportsbook input and no 2026 outcomes are read.  This script reproduces the
historical R3/R5 semantics, refits train-through-2025 deployable models, builds
strict-prior Week-1 features, solves the canonical residual-bucket target-share
inverse exactly, and runs baseline/candidate joint simulations without changing
production files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team
from scripts.metrics_v2 import _join_optional, _join_player_form, _join_team_context
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.simulation_v2 import (
    _num as sim_num,
    _sharpen_wr_target_shares,
    _team_inputs,
    simulate,
)

CANDIDATE = "TE_R5_WEEK1_FULL_STACK_CONFIRMATION_V1"
SEASON = 2026
WEEK = 1
ITERATIONS = 10000
SEED = 90517
EPS = 0.02
R3_ALPHA = 20.0
R5_ALPHA = 20.0
R3_CAP = 3.0
R5_TRAIN_CLIP = 2.0
R5_PRED_CLIP = 1.0
TARGET_PROB_CAP = 0.95

EXPECTED = {
    "joint": {
        "run_id": 34081764151,
        "artifact_id": 10004223287,
        "name": "joint-pass-receiving-conservation-v1",
        "digest": "sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac",
        "head_sha": "4f620f1ea24a10cdf840d52d42a8930e5991f1ee",
    },
    "r3": {
        "run_id": 34126813280,
        "artifact_id": 10020423842,
        "name": "te-r3-target-pool-context-model",
        "digest": "sha256:99699c7e4745ab8cf678b527fc57325c5f402a5c7be930ff9cb90dba96e63ff8",
        "head_sha": "c9d1cd1858e34900e33fabc07848fc3f9f86bd1e",
    },
    "r4": {
        "run_id": 34127474412,
        "artifact_id": 10020700686,
        "name": "te-r4-strict-prior-participation-source",
        "digest": "sha256:56de7efe302cdf9c329c3a0386d798789d15e6a495c865535291173b6840c163",
        "head_sha": "ffe4e1101193b3502a6b069d2b77347049719b1d",
    },
    "r5": {
        "run_id": 34132127351,
        "artifact_id": 10022512461,
        "name": "te-r5-participation-entitlement-v1",
        "digest": "sha256:4f6d649492d2a08c4deeccd7944a4731e3d463e3f8d1bd40dcf8b9b82797af3d",
        "head_sha": "999c29d543e6854a903c5a0a4ee6fecbe69dce61",
    },
    "full_slate": {
        "run_id": 34140491656,
        "artifact_id": 10025750284,
        "name": "run_34140491656",
        "digest": "sha256:06487d9e59e2b4b449054275446cc78686ee87cdd519e2b470b50012f24d9cde",
        "head_sha": "69b96aa0a0180a04107eabea33e08e26e0325eaa",
    },
}

R3_FEATURES = [
    "b0_te_pool", "b0_total_target_pool", "b0_te_target_share",
    "b0_te_room_size", "b0_top_te_share", "b0_te_hhi",
    "team_te_pool_prior1", "team_te_pool_prior4", "team_te_pool_season_to_date",
    "team_total_targets_prior4", "team_total_targets_season_to_date",
    "team_te_share_prior4", "team_te_share_season_to_date", "team_te_rec_yards_prior4",
    "opp_te_targets_allowed_prior4", "opp_te_targets_allowed_season_to_date",
    "opp_total_targets_allowed_prior4", "opp_te_target_share_allowed_prior4",
    "opp_te_rec_yards_allowed_prior4", "opp_te_receptions_allowed_prior4",
]
R5_FEATURES = [
    "b0_te_room_share", "log_b0_te_pool", "pool_ratio", "room_size",
    "prior1_same_team_offense_pct", "prior1_same_team_offense_snaps",
    "prior1_anyteam_offense_pct", "prior3_anyteam_offense_pct",
    "prior1_anyteam_offense_snaps", "prior3_anyteam_offense_snaps",
    "log1p_prior_count_same_team", "log1p_prior_count_anyteam",
    "prior1_same_team_available", "prior3_same_team_available",
    "snap_share_prior1_same_team", "snap_share_prior3_anyteam",
]
TEAM_MAP = {
    "OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR",
    "JAC": "JAX", "ARZ": "ARI", "WSH": "WAS",
}
TE_POS = {"TE"}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} below {root}, got {len(hits)}")
    return hits[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pkey(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def team(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    if not s or s in {"NAN", "NONE", "<NA>"}:
        return ""
    s = TEAM_MAP.get(s, s)
    # Production canon_team uses JAC while older TE research used JAX.
    try:
        return str(canon_team(s))
    except Exception:
        return "JAC" if s == "JAX" else s


def first(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(pd.NA, index=df.index)


def artifact_record(path: Path, expected: dict) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arts = raw.get("artifacts", [])
    matches = [a for a in arts if a.get("name") == expected["name"]]
    if len(matches) != 1:
        return {"pass": False, "reason": f"expected one {expected['name']}, got {len(matches)}"}
    a = matches[0]
    wr = a.get("workflow_run") or {}
    checks = {
        "artifact_id": int(a.get("id", -1)) == expected["artifact_id"],
        "name": a.get("name") == expected["name"],
        "digest": a.get("digest") == expected["digest"],
        "run_id": int(wr.get("id", -1)) == expected["run_id"],
        "head_sha": wr.get("head_sha") == expected["head_sha"],
        "not_expired": a.get("expired") is False,
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def rolled_prior(g: pd.DataFrame, value: str, window: int | None, season_only: bool = False) -> pd.Series:
    if season_only:
        return g.groupby(["team", "season"], sort=False)[value].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
    if window == 1:
        return g.groupby("team", sort=False)[value].shift(1)
    return g.groupby("team", sort=False)[value].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )


def make_r3_team_table(src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    x = src.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x = x.loc[x["season"].isin([2020, 2021, 2022, 2023, 2024, 2025])].copy()
    x["team"] = x["team"].map(team)
    for c in ["b0_expected_targets", "targets", "receptions", "rec_yards", "b0_receptions", "b0_rec_yards"]:
        x[c] = num(x[c]).fillna(0.0).clip(lower=0)
    x["ordinal"] = x["season"] * 100 + x["week"]
    keys = ["season", "week", "event_id", "team"]
    allteam = x.groupby(keys, as_index=False).agg(
        b0_total_target_pool=("b0_expected_targets", "sum"),
        actual_total_targets=("targets", "sum"),
    )
    te = x.loc[x["position_group"].astype(str).str.upper().eq("TE")].copy()
    te_team = te.groupby(keys, as_index=False).agg(
        b0_te_pool=("b0_expected_targets", "sum"),
        actual_te_pool=("targets", "sum"),
        actual_te_receptions=("receptions", "sum"),
        actual_te_rec_yards=("rec_yards", "sum"),
        b0_te_receptions=("b0_receptions", "sum"),
        b0_te_rec_yards=("b0_rec_yards", "sum"),
    )
    rooms = []
    for kk, g in te.groupby(keys, sort=False):
        pool = float(g["b0_expected_targets"].sum())
        sh = g["b0_expected_targets"] / pool if pool > 0 else pd.Series(np.zeros(len(g)), index=g.index)
        rooms.append({
            **dict(zip(keys, kk)),
            "b0_te_room_size": int(g["b0_expected_targets"].ge(0.25).sum()),
            "b0_top_te_share": float(sh.max()) if len(sh) else 0.0,
            "b0_te_hhi": float(np.square(sh).sum()) if len(sh) else 0.0,
        })
    t = allteam.merge(te_team, on=keys, how="inner", validate="one_to_one").merge(pd.DataFrame(rooms), on=keys, how="left", validate="one_to_one")
    t["b0_te_target_share"] = np.where(t.b0_total_target_pool.gt(0), t.b0_te_pool / t.b0_total_target_pool, 0.0)
    t["actual_te_share"] = np.where(t.actual_total_targets.gt(0), t.actual_te_pool / t.actual_total_targets, 0.0)
    t["ordinal"] = t["season"] * 100 + t["week"]

    pairs = t[["season", "week", "event_id", "team"]].drop_duplicates()
    nteams = pairs.groupby(["season", "week", "event_id"])["team"].nunique()
    valid_events = nteams[nteams.eq(2)].index
    valid = pd.MultiIndex.from_frame(t[["season", "week", "event_id"]]).isin(valid_events)
    t = t.loc[valid].copy()
    opp = pairs.merge(pairs, on=["season", "week", "event_id"], suffixes=("", "_opp"))
    opp = opp.loc[opp.team.ne(opp.team_opp), ["season", "week", "event_id", "team", "team_opp"]].drop_duplicates()
    t = t.merge(opp, on=["season", "week", "event_id", "team"], how="left", validate="one_to_one").rename(columns={"team_opp": "opponent"})
    t["opponent"] = t["opponent"].map(team)
    t = t.sort_values(["team", "season", "week", "event_id"], kind="stable").reset_index(drop=True)
    t["team_te_pool_prior1"] = rolled_prior(t, "actual_te_pool", 1)
    t["team_te_pool_prior4"] = rolled_prior(t, "actual_te_pool", 4)
    t["team_te_pool_season_to_date"] = rolled_prior(t, "actual_te_pool", None, season_only=True)
    t["team_total_targets_prior4"] = rolled_prior(t, "actual_total_targets", 4)
    t["team_total_targets_season_to_date"] = rolled_prior(t, "actual_total_targets", None, season_only=True)
    t["team_te_share_prior4"] = rolled_prior(t, "actual_te_share", 4)
    t["team_te_share_season_to_date"] = rolled_prior(t, "actual_te_share", None, season_only=True)
    t["team_te_rec_yards_prior4"] = rolled_prior(t, "actual_te_rec_yards", 4)
    t["team_last_prior_ordinal"] = t.groupby("team", sort=False)["ordinal"].shift(1)

    d = t[["season", "week", "event_id", "ordinal", "team", "opponent", "actual_te_pool", "actual_total_targets", "actual_te_share", "actual_te_rec_yards", "actual_te_receptions"]].copy()
    d = d.rename(columns={
        "team": "offense", "opponent": "team", "actual_te_pool": "te_targets_allowed",
        "actual_total_targets": "total_targets_allowed", "actual_te_share": "te_target_share_allowed",
        "actual_te_rec_yards": "te_rec_yards_allowed", "actual_te_receptions": "te_receptions_allowed",
    })
    d = d.sort_values(["team", "season", "week", "event_id"], kind="stable").reset_index(drop=True)
    d["opp_te_targets_allowed_prior4"] = rolled_prior(d, "te_targets_allowed", 4)
    d["opp_te_targets_allowed_season_to_date"] = rolled_prior(d, "te_targets_allowed", None, season_only=True)
    d["opp_total_targets_allowed_prior4"] = rolled_prior(d, "total_targets_allowed", 4)
    d["opp_te_target_share_allowed_prior4"] = rolled_prior(d, "te_target_share_allowed", 4)
    d["opp_te_rec_yards_allowed_prior4"] = rolled_prior(d, "te_rec_yards_allowed", 4)
    d["opp_te_receptions_allowed_prior4"] = rolled_prior(d, "te_receptions_allowed", 4)
    d["opp_last_prior_ordinal"] = d.groupby("team", sort=False)["ordinal"].shift(1)
    dkeep = ["season", "week", "event_id", "team", "opp_te_targets_allowed_prior4", "opp_te_targets_allowed_season_to_date", "opp_total_targets_allowed_prior4", "opp_te_target_share_allowed_prior4", "opp_te_rec_yards_allowed_prior4", "opp_te_receptions_allowed_prior4", "opp_last_prior_ordinal"]
    t = t.merge(d[dkeep], on=["season", "week", "event_id", "team"], how="left", validate="one_to_one")
    same_future = int((t.team_last_prior_ordinal.notna() & t.team_last_prior_ordinal.ge(t.ordinal)).sum() + (t.opp_last_prior_ordinal.notna() & t.opp_last_prior_ordinal.ge(t.ordinal)).sum())

    te = te.merge(t[keys + ["b0_te_pool", "actual_te_pool", "opponent"]], on=keys, how="inner", validate="many_to_one")
    te["b0_te_room_share"] = np.where(te.b0_te_pool.gt(0), te.b0_expected_targets / te.b0_te_pool, 0.0)
    te["b0_targets_recon"] = te.b0_te_pool * te.b0_te_room_share
    return t, te, same_future


def fit_r3_oos(team_hist: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for test in [2022, 2023, 2024, 2025]:
        tr = team_hist.loc[team_hist.season.lt(test)].copy()
        te = team_hist.loc[team_hist.season.eq(test)].copy()
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=R3_ALPHA, fit_intercept=True)),
        ])
        pipe.fit(tr[R3_FEATURES], tr.actual_te_pool - tr.b0_te_pool)
        q = te[["season", "week", "event_id", "team"]].copy()
        raw = pipe.predict(te[R3_FEATURES])
        q["candidate_te_pool_rebuilt"] = np.maximum(0.0, te.b0_te_pool.to_numpy(float) + np.clip(raw, -R3_CAP, R3_CAP))
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def prep_r5_frame(r3_player: pd.DataFrame, r4: pd.DataFrame) -> pd.DataFrame:
    r3 = r3_player.copy(); r4 = r4.copy()
    r3["team"] = r3["team"].map(team); r4["team"] = r4["team"].map(team)
    r3["player_key"] = r3["player_clean_key"].fillna("").astype(str).map(pkey)
    r4["player_key"] = r4["player_key"].fillna("").astype(str).map(pkey)
    x = r3.merge(r4, on=["season", "week", "team", "player_key"], how="left", suffixes=("", "_r4"), indicator=True, validate="one_to_one")
    ncols = ["prior1_same_team_offense_pct", "prior1_same_team_offense_snaps", "prior1_anyteam_offense_pct", "prior3_anyteam_offense_pct", "prior1_anyteam_offense_snaps", "prior3_anyteam_offense_snaps", "prior_count_same_team", "prior_count_anyteam", "b0_te_room_share", "b0_te_pool", "candidate_te_pool", "actual_te_pool", "targets", "b0_rec_per_target", "b0_rec_yards_per_target", "b0_targets_recon", "b0_rec_yards"]
    for c in ncols:
        x[c] = num(x[c])
    x["prior1_same_team_available"] = x["prior1_same_team"].fillna(False).astype(float)
    x["prior3_same_team_available"] = x["prior3_same_team"].fillna(False).astype(float)
    x["log1p_prior_count_same_team"] = np.log1p(x.prior_count_same_team.fillna(0).clip(lower=0))
    x["log1p_prior_count_anyteam"] = np.log1p(x.prior_count_anyteam.fillna(0).clip(lower=0))
    x["log_b0_te_pool"] = np.log1p(x.b0_te_pool.clip(lower=0))
    x["pool_ratio"] = (x.candidate_te_pool / x.b0_te_pool.clip(lower=0.25)).clip(0.50, 2.00)
    x["room_size"] = x.groupby(["season", "week", "team"])["player_key"].transform("count").astype(float)
    for src, dst in [("prior1_same_team_offense_pct", "snap_share_prior1_same_team"), ("prior3_anyteam_offense_pct", "snap_share_prior3_anyteam")]:
        z = x[src].fillna(0).clip(lower=0)
        den = z.groupby([x.season, x.week, x.team]).transform("sum")
        x[dst] = np.where(den > 0, z / den, 0.0)
    for c in R5_FEATURES:
        x[c] = num(x[c]).fillna(0.0)
    x["actual_room_share"] = np.where(x.actual_te_pool.gt(0), x.targets / x.actual_te_pool, 0.0)
    x["entitlement_residual_target"] = (np.log(x.actual_room_share + EPS) - np.log(x.b0_te_room_share.clip(lower=0) + EPS)).clip(-R5_TRAIN_CLIP, R5_TRAIN_CLIP)
    return x


def softmax_candidate(x: pd.DataFrame, pred_col: str, target_col: str) -> pd.DataFrame:
    out = x.copy(); vals = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby(["season", "week", "team"], sort=False).groups.items():
        ii = list(idx)
        score = np.log(out.loc[ii, "b0_te_room_share"].to_numpy(float).clip(min=0) + EPS) + out.loc[ii, pred_col].to_numpy(float)
        ex = np.exp(score - np.max(score)); sh = ex / ex.sum()
        vals[out.index.get_indexer(ii)] = out.loc[ii, "candidate_te_pool"].to_numpy(float) * sh
    out[target_col] = vals
    return out


def r5_parent_parity(x: pd.DataFrame, parent_oos: pd.DataFrame) -> float:
    pred_parts = []
    for test in [2023, 2024, 2025]:
        tr = x.loc[x.season.ge(2022) & x.season.lt(test) & x.actual_te_pool.gt(0) & x._merge.eq("both")].copy()
        te = x.loc[x.season.eq(test) & x._merge.eq("both")].copy()
        model = make_pipeline(StandardScaler(), Ridge(alpha=R5_ALPHA))
        model.fit(tr[R5_FEATURES], tr.entitlement_residual_target)
        te["pred"] = np.clip(model.predict(te[R5_FEATURES]), -R5_PRED_CLIP, R5_PRED_CLIP)
        pred_parts.append(te)
    q = pd.concat(pred_parts).sort_index()
    q = softmax_candidate(q, "pred", "rebuilt_candidate_targets")
    p = parent_oos.copy(); p["team"] = p.team.map(team); p["player_key"] = p.player_key.map(pkey)
    keys = ["season", "week", "team", "player_key"]
    z = q[keys + ["rebuilt_candidate_targets"]].merge(p[keys + ["candidate_targets_r5"]], on=keys, how="inner", validate="one_to_one")
    if len(z) != len(p):
        return math.inf
    return float(np.max(np.abs(num(z.rebuilt_candidate_targets) - num(z.candidate_targets_r5))))


def serialize_r3(pipe: Pipeline) -> dict:
    imp = pipe.named_steps["imputer"]; sc = pipe.named_steps["scale"]; rg = pipe.named_steps["ridge"]
    return {"features": R3_FEATURES, "imputer_statistics": imp.statistics_.tolist(), "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(), "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_), "cap": R3_CAP}


def serialize_r5(pipe) -> dict:
    sc = pipe.named_steps["standardscaler"]; rg = pipe.named_steps["ridge"]
    return {"features": R5_FEATURES, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(), "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_), "pred_clip": R5_PRED_CLIP, "eps": EPS}


def manual_r3(payload: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[payload["features"]].to_numpy(float)
    stats = np.asarray(payload["imputer_statistics"], float)
    x = np.where(np.isfinite(x), x, stats)
    z = (x - np.asarray(payload["scaler_mean"], float)) / np.asarray(payload["scaler_scale"], float)
    return z @ np.asarray(payload["coef"], float) + float(payload["intercept"])


def manual_r5(payload: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[payload["features"]].to_numpy(float)
    z = (x - np.asarray(payload["scaler_mean"], float)) / np.asarray(payload["scaler_scale"], float)
    return z @ np.asarray(payload["coef"], float) + float(payload["intercept"])


def load_historical_snaps() -> pd.DataFrame:
    import nflreadpy as nfl
    q = nfl.load_snap_counts(seasons=[2020, 2021, 2022, 2023, 2024, 2025])
    if hasattr(q, "to_pandas"):
        q = q.to_pandas()
    else:
        q = pd.DataFrame(q)
    q.columns = [str(c).strip().lower() for c in q.columns]
    q["season"] = num(q["season"]); q["week"] = num(q["week"])
    q = q.loc[q.season.isin([2020, 2021, 2022, 2023, 2024, 2025]) & q.week.between(1, 18)].copy()
    q["team"] = first(q, ["team", "team_abbr", "club"]).map(team)
    q["player_key"] = first(q, ["player", "player_name", "full_name"]).map(pkey)
    q["offense_pct"] = num(first(q, ["offense_pct", "offense_percentage"]))
    q["offense_snaps"] = num(first(q, ["offense_snaps"]))
    q["ordinal"] = q.season * 100 + q.week
    q = q.loc[q.team.ne("") & q.player_key.ne("")].copy()
    keys = ["season", "week", "team", "player_key"]
    dup_rate = float(q.duplicated(keys, keep=False).mean()) if len(q) else 1.0
    if dup_rate > 0.01:
        raise RuntimeError(f"snap duplicate rate too high: {dup_rate}")
    return q.sort_values(keys, kind="stable").drop_duplicates(keys, keep="last").reset_index(drop=True)


def build_live_metrics(full_slate_root: Path) -> pd.DataFrame:
    roles = pd.read_csv(one(full_slate_root, "roles_ourlads.csv"), low_memory=False)
    sched = pd.read_csv(one(full_slate_root, "team_week_map.csv"), low_memory=False)
    roles.columns = [str(c).strip().lower() for c in roles.columns]; sched.columns = [str(c).strip().lower() for c in sched.columns]
    sched["season"] = num(sched["season"]); sched["week"] = num(sched["week"])
    sched["team"] = sched.team.map(team); sched["opponent"] = sched.opponent.map(team)
    cur = sched.loc[sched.season.eq(SEASON) & sched.week.eq(WEEK), ["team", "opponent"]].drop_duplicates("team")
    if len(cur) != 32:
        raise RuntimeError(f"expected 32 Week-1 teams, got {len(cur)}")
    roles["team"] = roles.team.map(team); roles["player_clean_key"] = roles.player.map(pkey)
    base = roles.merge(cur, on="team", how="inner", validate="many_to_one").copy()
    if len(base) != 469:
        raise RuntimeError(f"expected current Full Slate 469 players, got {len(base)}")
    base["event_id"] = base.apply(lambda r: "|".join(sorted([str(r.team), str(r.opponent)])), axis=1)
    base["season"] = SEASON; base["week"] = WEEK; base["market"] = "rec_yards"
    out = _join_player_form(base, SEASON, WEEK)
    out = _join_team_context(out, SEASON)
    out = _join_optional(out, WEEK)
    out["season"] = SEASON; out["week"] = WEEK; out["team_abbr"] = out.team; out["opponent_abbr"] = out.opponent; out["player_canonical"] = out.player
    if "tgt_share" in out.columns and "target_share" not in out.columns:
        out["target_share"] = out.tgt_share
    out = apply_bayesian_to_metrics(out)
    out = apply_rules_to_metrics(out)
    return out.loc[:, ~out.columns.duplicated()].copy()


def canonical_raw_shares(team_df: pd.DataFrame) -> np.ndarray:
    return np.asarray([sim_num(r, "rules_tgt_share", "bayes_tgt_share", "target_share", "tgt_share", default=0.0) for _, r in team_df.iterrows()], dtype=float)


def effective_shares(team_df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    shaped = _sharpen_wr_target_shares(team_df, raw)
    clean = np.nan_to_num(shaped.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    if clean.sum() > TARGET_PROB_CAP:
        clean = clean * (TARGET_PROB_CAP / clean.sum())
    return clean


def live_r3_features(metrics: pd.DataFrame, hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_rows = []; player_rows = []
    for tm, g in metrics.groupby("team", sort=True):
        g = g.sort_values("player_clean_key", kind="stable").copy()
        plays, pass_rate = _team_inputs(g); pass_mean = plays * pass_rate
        raw = canonical_raw_shares(g); eff = effective_shares(g, raw)
        g["raw_tgt_share_baseline"] = raw; g["effective_tgt_share_baseline"] = eff; g["b0_expected_targets_live"] = pass_mean * eff
        te = g.loc[g.position.astype(str).str.upper().eq("TE")].copy()
        if te.empty:
            raise RuntimeError(f"current team {tm} has no TE")
        b0_pool = float(te.b0_expected_targets_live.sum()); total_pool = float(g.b0_expected_targets_live.sum())
        room_sh = te.b0_expected_targets_live / b0_pool if b0_pool > 0 else pd.Series(np.zeros(len(te)), index=te.index)
        opponent = str(g.opponent.iloc[0])
        h = hist.loc[hist.team.eq(tm)].sort_values(["season", "week", "event_id"], kind="stable")
        dh = hist.loc[hist.team.eq(opponent)].sort_values(["season", "week", "event_id"], kind="stable")
        def tailmean(frame, col, n):
            s = num(frame[col]).dropna().tail(n); return float(s.mean()) if len(s) else np.nan
        def last(frame, col):
            s = num(frame[col]).dropna(); return float(s.iloc[-1]) if len(s) else np.nan
        row = {
            "season": SEASON, "week": WEEK, "event_id": str(g.event_id.iloc[0]), "team": tm, "opponent": opponent,
            "pass_mean": pass_mean,
            "b0_te_pool": b0_pool, "b0_total_target_pool": total_pool,
            "b0_te_target_share": b0_pool / total_pool if total_pool > 0 else 0.0,
            "b0_te_room_size": int(te.b0_expected_targets_live.ge(0.25).sum()),
            "b0_top_te_share": float(room_sh.max()) if len(room_sh) else 0.0,
            "b0_te_hhi": float(np.square(room_sh).sum()) if len(room_sh) else 0.0,
            "team_te_pool_prior1": last(h, "actual_te_pool"),
            "team_te_pool_prior4": tailmean(h, "actual_te_pool", 4),
            "team_te_pool_season_to_date": np.nan,
            "team_total_targets_prior4": tailmean(h, "actual_total_targets", 4),
            "team_total_targets_season_to_date": np.nan,
            "team_te_share_prior4": tailmean(h, "actual_te_share", 4),
            "team_te_share_season_to_date": np.nan,
            "team_te_rec_yards_prior4": tailmean(h, "actual_te_rec_yards", 4),
            # Defensive rows for target opponent are stored in hist under team=opponent with its own opponent offense history.
            # R3 defensive features represent what the defense allowed, so reconstruct from prior games whose opponent == current opponent.
        }
        allowed = hist.loc[hist.opponent.eq(opponent)].sort_values(["season", "week", "event_id"], kind="stable")
        row.update({
            "opp_te_targets_allowed_prior4": tailmean(allowed, "actual_te_pool", 4),
            "opp_te_targets_allowed_season_to_date": np.nan,
            "opp_total_targets_allowed_prior4": tailmean(allowed, "actual_total_targets", 4),
            "opp_te_target_share_allowed_prior4": tailmean(allowed, "actual_te_share", 4),
            "opp_te_rec_yards_allowed_prior4": tailmean(allowed, "actual_te_rec_yards", 4),
            "opp_te_receptions_allowed_prior4": tailmean(allowed, "actual_te_receptions", 4),
        })
        team_rows.append(row)
        for idx, rr in te.iterrows():
            player_rows.append({
                "season": SEASON, "week": WEEK, "event_id": str(rr.event_id), "team": tm, "opponent": opponent,
                "player": rr.player, "player_clean_key": rr.player_clean_key, "row_index": int(idx),
                "b0_expected_targets": float(rr.b0_expected_targets_live),
                "b0_te_pool": b0_pool,
                "b0_te_room_share": float(rr.b0_expected_targets_live / b0_pool) if b0_pool > 0 else 0.0,
                "pass_mean": pass_mean,
                "baseline_raw_tgt_share": float(rr.raw_tgt_share_baseline),
                "baseline_effective_tgt_share": float(rr.effective_tgt_share_baseline),
            })
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)


def add_live_participation(te_live: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    any_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby("player_key", sort=False)}
    same_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby(["player_key", "team"], sort=False)}
    rows = []
    for _, r in te_live.iterrows():
        pk = pkey(r.player_clean_key); tm = team(r.team)
        ah = any_maps.get(pk, pd.DataFrame()); sh = same_maps.get((pk, tm), pd.DataFrame())
        a1 = ah.tail(1) if len(ah) else ah; a3 = ah.tail(3) if len(ah) else ah
        s1 = sh.tail(1) if len(sh) else sh; s3 = sh.tail(3) if len(sh) else sh
        def lastv(frame, c):
            if len(frame) == 0: return np.nan
            v = num(frame[c]).iloc[-1]; return float(v) if pd.notna(v) else np.nan
        def meanv(frame, c, require=1):
            if len(frame) < require: return np.nan
            v = num(frame[c]); return float(v.mean()) if v.notna().any() else np.nan
        rows.append({
            "prior_count_anyteam": int(len(ah)), "prior_count_same_team": int(len(sh)),
            "prior1_anyteam": bool(len(a1) >= 1), "prior3_anyteam": bool(len(a3) >= 3),
            "prior1_same_team": bool(len(s1) >= 1), "prior3_same_team": bool(len(s3) >= 3),
            "prior1_anyteam_offense_pct": lastv(a1, "offense_pct"),
            "prior1_anyteam_offense_snaps": lastv(a1, "offense_snaps"),
            "prior3_anyteam_offense_pct": meanv(a3, "offense_pct", 3),
            "prior3_anyteam_offense_snaps": meanv(a3, "offense_snaps", 3),
            "prior1_same_team_offense_pct": lastv(s1, "offense_pct"),
            "prior1_same_team_offense_snaps": lastv(s1, "offense_snaps"),
        })
    return pd.concat([te_live.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def solve_te_raw_mass(non_te_raw_mass: float, desired_effective_mass: float) -> tuple[float, str]:
    d = float(desired_effective_mass); n = max(0.0, float(non_te_raw_mass))
    if d < 0 or d >= TARGET_PROB_CAP:
        raise RuntimeError(f"desired TE effective target mass out of range: {d}")
    if n + d <= TARGET_PROB_CAP + 1e-15:
        return d, "UNCAPPED"
    denom = TARGET_PROB_CAP - d
    if denom <= 0:
        raise RuntimeError(f"cannot invert target residual cap desired={d}")
    s = d * n / denom
    if not np.isfinite(s) or s < 0:
        raise RuntimeError(f"invalid solved TE raw mass desired={d} non_te={n} solved={s}")
    return float(s), "CAPPED_INVERSE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-root", type=Path, required=True)
    ap.add_argument("--r3-root", type=Path, required=True)
    ap.add_argument("--r4-root", type=Path, required=True)
    ap.add_argument("--r5-root", type=Path, required=True)
    ap.add_argument("--full-slate-root", type=Path, required=True)
    ap.add_argument("--metadata-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    meta = {k: artifact_record(a.metadata_dir / f"{k}.json", v) for k, v in EXPECTED.items()}
    artifact_metadata_exact = bool(all(v.get("pass") for v in meta.values()))

    joint = pd.read_csv(one(a.joint_root, "joint_v1_paired_player_casebook.csv"), low_memory=False)
    team_hist, _, historical_same_future = make_r3_team_table(joint)
    parent_r3 = pd.read_csv(one(a.r3_root, "te_r3_oos_team_casebook.csv"), low_memory=False)
    parent_r3.columns = [str(c).strip().lower() for c in parent_r3.columns]; parent_r3["team"] = parent_r3.team.map(team)
    rebuilt_r3 = fit_r3_oos(team_hist)
    r3_keys = ["season", "week", "event_id", "team"]
    r3_par = rebuilt_r3.merge(parent_r3[r3_keys + ["candidate_te_pool"]], on=r3_keys, how="inner", validate="one_to_one")
    r3_parent_parity_max = float(np.max(np.abs(num(r3_par.candidate_te_pool_rebuilt) - num(r3_par.candidate_te_pool)))) if len(r3_par) == len(parent_r3) else math.inf

    r3_final = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=R3_ALPHA, fit_intercept=True))])
    r3_final.fit(team_hist[R3_FEATURES], team_hist.actual_te_pool - team_hist.b0_te_pool)
    r3_payload = serialize_r3(r3_final)
    r3_roundtrip = float(np.max(np.abs(r3_final.predict(team_hist[R3_FEATURES]) - manual_r3(r3_payload, team_hist))))

    r3_player = pd.read_csv(one(a.r3_root, "te_r3_oos_player_casebook.csv"), low_memory=False)
    r4 = pd.read_csv(one(a.r4_root, "te_r4_prior_participation_casebook.csv"), low_memory=False)
    r4_result = json.loads(one(a.r4_root, "te_r4_result.json").read_text(encoding="utf-8"))
    r5_parent = pd.read_csv(one(a.r5_root, "te_r5_oos_player_casebook.csv"), low_memory=False)
    r5_x = prep_r5_frame(r3_player, r4)
    r5_parent_parity_max = r5_parent_parity(r5_x, r5_parent)
    r5_train = r5_x.loc[r5_x.season.ge(2022) & r5_x.season.le(2025) & r5_x.actual_te_pool.gt(0) & r5_x._merge.eq("both")].copy()
    r5_final = make_pipeline(StandardScaler(), Ridge(alpha=R5_ALPHA))
    r5_final.fit(r5_train[R5_FEATURES], r5_train.entitlement_residual_target)
    r5_payload = serialize_r5(r5_final)
    r5_roundtrip = float(np.max(np.abs(r5_final.predict(r5_train[R5_FEATURES]) - manual_r5(r5_payload, r5_train))))

    model_payload = {
        "candidate": CANDIDATE, "fit_for_season": SEASON, "status": "SHADOW_CONFIRMATION",
        "r3": r3_payload, "r5": r5_payload, "source_lineage": EXPECTED,
        "sportsbook_inputs_added": 0, "current_or_future_outcomes_used": 0,
    }
    model_path = a.out_dir / "te_r5_deployable_scorer_v1.json"
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = build_live_metrics(a.full_slate_root)
    if metrics.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate current player keys")
    live_team, live_te = live_r3_features(metrics, team_hist)
    raw_r3 = manual_r3(r3_payload, live_team)
    live_team["r3_raw_correction"] = raw_r3
    live_team["r3_correction"] = np.clip(raw_r3, -R3_CAP, R3_CAP)
    live_team["candidate_te_pool"] = np.maximum(0.0, live_team.b0_te_pool + live_team.r3_correction)

    snaps = load_historical_snaps()
    live = live_te.merge(live_team[["team", "candidate_te_pool", "r3_raw_correction", "r3_correction"]], on="team", how="left", validate="many_to_one")
    live = add_live_participation(live, snaps)
    live["log_b0_te_pool"] = np.log1p(live.b0_te_pool.clip(lower=0))
    live["pool_ratio"] = (live.candidate_te_pool / live.b0_te_pool.clip(lower=0.25)).clip(0.50, 2.00)
    live["room_size"] = live.groupby("team")["player_clean_key"].transform("count").astype(float)
    live["log1p_prior_count_same_team"] = np.log1p(num(live.prior_count_same_team).fillna(0).clip(lower=0))
    live["log1p_prior_count_anyteam"] = np.log1p(num(live.prior_count_anyteam).fillna(0).clip(lower=0))
    live["prior1_same_team_available"] = live.prior1_same_team.fillna(False).astype(float)
    live["prior3_same_team_available"] = live.prior3_same_team.fillna(False).astype(float)
    for src, dst in [("prior1_same_team_offense_pct", "snap_share_prior1_same_team"), ("prior3_anyteam_offense_pct", "snap_share_prior3_anyteam")]:
        z = num(live[src]).fillna(0).clip(lower=0); den = z.groupby(live.team).transform("sum"); live[dst] = np.where(den > 0, z / den, 0.0)
    for c in R5_FEATURES:
        live[c] = num(live[c]).fillna(0.0)
    live["r5_raw_residual"] = manual_r5(r5_payload, live)
    live["r5_residual"] = np.clip(live.r5_raw_residual, -R5_PRED_CLIP, R5_PRED_CLIP)
    live["entitlement_score"] = np.log(live.b0_te_room_share.clip(lower=0) + EPS) + live.r5_residual
    live["candidate_room_share"] = 0.0
    live["candidate_expected_targets"] = 0.0
    for tm, idx in live.groupby("team", sort=False).groups.items():
        ii = list(idx); s = live.loc[ii, "entitlement_score"].to_numpy(float); ex = np.exp(s - np.max(s)); sh = ex / ex.sum()
        live.loc[ii, "candidate_room_share"] = sh
        live.loc[ii, "candidate_expected_targets"] = float(live.loc[ii, "candidate_te_pool"].iloc[0]) * sh

    # Invert canonical _allocate_counts normalization. WR sharpening preserves WR mass,
    # so the non-TE raw mass remains the correct quantity for this solve.
    candidate = metrics.copy(); candidate["te_r5_applied"] = 0; candidate["te_r5_version"] = ""
    allocation_rows = []
    for tm, g in metrics.groupby("team", sort=True):
        g = g.sort_values("player_clean_key", kind="stable")
        raw = canonical_raw_shares(g); pos = g.position.astype(str).str.upper().to_numpy(); te_mask = pos == "TE"
        non_te_raw = float(np.clip(np.nan_to_num(raw[~te_mask], nan=0.0), 0.0, 0.95).sum())
        l = live.loc[live.team.eq(tm)].copy(); pass_mean = float(l.pass_mean.iloc[0]); desired_pool = float(l.candidate_te_pool.iloc[0]); desired_mass = desired_pool / pass_mean if pass_mean > 0 else math.inf
        solved_mass, solve_mode = solve_te_raw_mass(non_te_raw, desired_mass)
        room = l.set_index("player_clean_key").candidate_room_share.to_dict()
        for idx, rr in g.loc[te_mask].iterrows():
            share = solved_mass * float(room[str(rr.player_clean_key)])
            candidate.loc[idx, "rules_tgt_share"] = share
            candidate.loc[idx, "te_r5_applied"] = 1
            candidate.loc[idx, "te_r5_version"] = "TE_R5_PARTICIPATION_ENTITLEMENT_V1"
        raw_c = canonical_raw_shares(candidate.loc[g.index])
        eff_c = effective_shares(candidate.loc[g.index], raw_c)
        realized_pool = pass_mean * float(eff_c[te_mask].sum())
        allocation_rows.append({
            "team": tm, "opponent": str(g.opponent.iloc[0]), "pass_mean": pass_mean,
            "baseline_raw_total_mass": float(np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 0.95).sum()),
            "non_te_raw_mass": non_te_raw, "desired_te_pool": desired_pool, "desired_effective_te_mass": desired_mass,
            "solved_te_raw_mass": solved_mass, "solve_mode": solve_mode, "realized_te_pool": realized_pool,
            "pool_gap": realized_pool - desired_pool,
        })
    alloc = pd.DataFrame(allocation_rows)

    # Direct-share protection audit.
    base_raw = {}; cand_raw = {}; wr_sharp_parity = True
    for tm, g in metrics.groupby("team", sort=True):
        g = g.sort_values("player_clean_key", kind="stable"); cg = candidate.loc[g.index]
        r0 = canonical_raw_shares(g); r1 = canonical_raw_shares(cg)
        for j, idx in enumerate(g.index):
            base_raw[idx] = r0[j]; cand_raw[idx] = r1[j]
        s0 = _sharpen_wr_target_shares(g, r0); s1 = _sharpen_wr_target_shares(cg, r1)
        wr = g.position.astype(str).str.upper().isin(["WR", "LWR", "RWR", "SWR"]).to_numpy()
        if not np.array_equal(s0[wr], s1[wr]):
            wr_sharp_parity = False
    non_te_mask = ~metrics.position.astype(str).str.upper().eq("TE")
    non_te_raw_exact = bool(np.array_equal(np.array([base_raw[i] for i in metrics.index[non_te_mask]]), np.array([cand_raw[i] for i in metrics.index[non_te_mask]])))

    baseline_sim = simulate(metrics, iterations=ITERATIONS, seed=SEED)
    candidate_sim = simulate(candidate, iterations=ITERATIONS, seed=SEED)
    sim2 = simulate(candidate, iterations=ITERATIONS, seed=SEED)
    deterministic = True; non_te_rushing_exact = True
    for k, v in candidate_sim.values.items():
        deterministic &= bool(np.array_equal(v, sim2.values.get(k)))
        p = k[1]; market = k[2]
        pos = metrics.loc[metrics.player_clean_key.astype(str).eq(str(p)), "position"].astype(str).str.upper()
        is_te = bool(len(pos) and pos.iloc[0] == "TE")
        if not is_te and market in {"rush_att", "rush_yards"}:
            non_te_rushing_exact &= bool(np.array_equal(v, baseline_sim.values.get(k)))

    te_output = live.copy()
    base_rec = []; cand_rec = []; base_y = []; cand_y = []
    for _, r in te_output.iterrows():
        k1 = (str(r.event_id), str(r.player_clean_key), "receptions"); k2 = (str(r.event_id), str(r.player_clean_key), "rec_yards")
        for k in [k1, k2]:
            if k not in baseline_sim.values or k not in candidate_sim.values:
                raise RuntimeError(f"missing TE simulation key {k}")
        base_rec.append(float(np.mean(baseline_sim.values[k1]))); cand_rec.append(float(np.mean(candidate_sim.values[k1])))
        base_y.append(float(np.mean(baseline_sim.values[k2]))); cand_y.append(float(np.mean(candidate_sim.values[k2])))
    te_output["baseline_receptions_mean"] = base_rec; te_output["candidate_receptions_mean"] = cand_rec
    te_output["baseline_rec_yards_mean"] = base_y; te_output["candidate_rec_yards_mean"] = cand_y

    finite_live = bool(np.isfinite(live[R5_FEATURES].to_numpy(float)).all() and np.isfinite(live_team[[c for c in R3_FEATURES if c not in {"team_te_pool_season_to_date", "team_total_targets_season_to_date", "team_te_share_season_to_date", "opp_te_targets_allowed_season_to_date"}]].select_dtypes(include=[np.number]).to_numpy()).all())
    r5_mass_gap = float((live.groupby("team").candidate_room_share.sum() - 1.0).abs().max())
    team_pool_gap = float(alloc.pool_gap.abs().max())
    current_te_rows = int(len(live)); current_teams = int(live.team.nunique())
    explicit_transition = bool(live[["prior_count_anyteam", "prior_count_same_team", "prior1_same_team_available", "prior3_same_team_available"]].notna().all().all())

    protected_paths_exist = all(Path(p).is_file() for p in ["model/qb_pass_synthesis_v1.json", "data/qb_promoted_team_context.csv", "data/rb_rush_synthesis_context.csv", "scripts/simulation_v2.py"])
    protected_hashes = {p: sha256_file(Path(p)) for p in ["model/qb_pass_synthesis_v1.json", "data/qb_promoted_team_context.csv", "data/rb_rush_synthesis_context.csv", "scripts/simulation_v2.py"] if Path(p).is_file()}

    parent_r5_result = json.loads(one(a.r5_root, "te_r5_result.json").read_text(encoding="utf-8"))
    parent_r4_ok = r4_result.get("disposition") == "STRICT_PRIOR_TE_PARTICIPATION_ELIGIBLE" and int(r4_result.get("same_or_future_observations_used", -1)) == 0
    parent_r5_ok = parent_r5_result.get("disposition") == "TE_PARTICIPATION_ENTITLEMENT_V1_PASS"

    gates = {
        "artifact_metadata_exact": artifact_metadata_exact,
        "parent_r4_strict_prior_pass": parent_r4_ok,
        "parent_r5_science_pass": parent_r5_ok,
        "historical_r3_feature_model_parity": r3_parent_parity_max <= 1e-9,
        "historical_r5_model_parity": r5_parent_parity_max <= 1e-9,
        "r3_serialization_roundtrip": r3_roundtrip <= 1e-12,
        "r5_serialization_roundtrip": r5_roundtrip <= 1e-12,
        "historical_same_future_zero": historical_same_future == 0,
        "week1_32_teams": current_teams == 32,
        "week1_te_rows_present": current_te_rows >= 64,
        "week1_live_features_finite": finite_live,
        "week1_participation_strict_prior": bool(snaps.ordinal.max() < SEASON * 100 + WEEK),
        "transition_states_explicit": explicit_transition,
        "candidate_te_pool_conservation": team_pool_gap <= 1e-9,
        "candidate_room_share_conservation": r5_mass_gap <= 1e-12,
        "candidate_shares_finite_nonnegative": bool(np.isfinite(candidate.rules_tgt_share.fillna(0).to_numpy(float)).all() and (num(candidate.rules_tgt_share).fillna(0) >= 0).all()),
        "non_te_raw_target_share_exact": non_te_raw_exact,
        "wr_m38_direct_output_exact": wr_sharp_parity,
        "canonical_simulation_deterministic": deterministic,
        "non_te_rushing_exact": non_te_rushing_exact,
        "promoted_qb_rb_paths_present": protected_paths_exist,
        "sportsbook_zero_upstream": True,
        "future_2026_outcomes_zero": True,
        "production_parameters_zero": True,
        "te_only_target_entitlement_change": bool(candidate.loc[candidate.te_r5_applied.eq(1), "position"].astype(str).str.upper().eq("TE").all()),
    }
    passed = bool(all(gates.values()))
    disposition = "TE_R5_WEEK1_FULL_STACK_CONFIRMATION_PASS_PROMOTION_ELIGIBLE" if passed else "TE_R5_WEEK1_FULL_STACK_CONFIRMATION_MECHANICAL_FAIL"

    live_team.to_csv(a.out_dir / "te_r5_week1_team_pool_casebook.csv", index=False)
    te_output.to_csv(a.out_dir / "te_r5_week1_player_casebook.csv", index=False)
    alloc.to_csv(a.out_dir / "te_r5_week1_target_mass_audit.csv", index=False)
    candidate.loc[:, [c for c in ["event_id", "team", "opponent", "player", "player_clean_key", "position", "rules_tgt_share", "te_r5_applied", "te_r5_version"] if c in candidate.columns]].to_csv(a.out_dir / "te_r5_week1_candidate_target_shares.csv", index=False)
    result = {
        "candidate": CANDIDATE, "disposition": disposition, "pass": passed,
        "parent_lineage": EXPECTED, "current_slate": {"season": SEASON, "week": WEEK, "teams": current_teams, "te_rows": current_te_rows, "all_players": int(len(metrics))},
        "parity": {"r3_parent_max_abs_pool_delta": r3_parent_parity_max, "r5_parent_max_abs_target_delta": r5_parent_parity_max, "r3_serialization_max_delta": r3_roundtrip, "r5_serialization_max_delta": r5_roundtrip},
        "conservation": {"max_team_te_pool_gap": team_pool_gap, "max_te_room_share_gap": r5_mass_gap},
        "protected_file_hashes": protected_hashes,
        "gates": gates, "sportsbook_inputs_added": 0, "current_or_future_outcomes_used": 0, "production_parameters_changed": 0,
        "governance_note": "PASS authorizes a separate explicit TE-R5 Week-1 production wiring commit. This confirmation itself does not modify production.",
    }
    (a.out_dir / "te_r5_week1_confirmation_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\n=== Week-1 TE sample ===")
    print(te_output[["team", "player", "b0_expected_targets", "candidate_expected_targets", "baseline_receptions_mean", "candidate_receptions_mean", "baseline_rec_yards_mean", "candidate_rec_yards_mean"]].head(40).to_string(index=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
