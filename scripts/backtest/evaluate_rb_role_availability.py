"""M95G: pregame RB role-transition / availability engine.

Research-only. M95F showed that workload-state ranking and probability
calibration are real, but the highest-risk population remains too optimistic.
M95G asks whether truly pregame weekly roster, injury-report, and (only when
explicitly week-tagged) depth-chart information can distinguish:

- a genuine upcoming bellcow opportunity caused by a teammate vacancy/promotion;
- an established workhorse whose current-week availability/competition argues
  against assigning extreme tail mass.

The M95F raw tail scores are frozen. M95G only recalibrates those scores with
pregame role/availability information. No sportsbook input and no production
code changes.

Protocol
--------
1. Use temporal 2024 M95F OOF raw scores (Weeks 5-12) to fit candidate
   role/availability calibrators.
2. Select architecture and operating thresholds on 2024 Weeks 13-18 only.
3. Freeze architecture/thresholds, refit on all eligible 2024 temporal OOF
   rows, and evaluate once on untouched 2025.
4. M94C remains the official central carry mean throughout.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team

SEED = 9597
TARGETS = ("actual_20plus", "actual_25plus")
MAX_FLAG_MULTIPLE = 3.0
THRESHOLDS = (
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10,
    0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
)
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]
RB_POS = {"RB", "HB", "FB"}

BASE_ROLE_FEATURES = [
    "_score_logit",
    "rb_games_before",
    "rb_carries_avg1", "rb_carries_avg3", "rb_carries_avg5",
    "rb_rb_share_avg1", "rb_rb_share_avg3", "rb_rb_share_avg5",
    "role_is_workhorse", "role_is_starter_plus",
    "late_week_17plus", "week18",
]

AVAILABILITY_FEATURES = [
    "self_roster_present", "self_roster_unavailable", "self_roster_active",
    "self_injury_reported", "self_inj_out", "self_inj_doubtful",
    "self_inj_questionable", "self_practice_dnp", "self_practice_limited",
    "new_roster_entry", "returned_active", "team_change_recent",
    "reappeared_after_gap",
    "team_rb_roster_count", "team_rb_available_count", "other_rb_available_count",
    "team_rb_injury_count", "team_rb_out_count", "team_rb_doubtful_count",
    "team_rb_questionable_count", "other_rb_out_count", "other_rb_doubtful_count",
    "other_rb_questionable_count",
    "target_was_prior_top1", "target_was_prior_top2",
    "prior_top1_unavailable", "prior_top2_unavailable",
    "vacated_lead_role", "vacated_top2_role",
    "prior_top1_carries", "prior_top2_carries",
    "depth_rank", "depth_is_rb1", "depth_promotion",
]

INTERACTION_FEATURES = [
    "ix_score_x_vacancy", "ix_score_x_self_unavailable",
    "ix_score_x_prior_leader", "ix_score_x_depth1",
    "ix_vacancy_x_new_entry", "ix_vacancy_x_competition_scarcity",
]

CALIBRATOR_SPECS = (
    ("availability", 0.12),
    ("role_availability", 0.12),
    ("role_availability_interactions", 0.12),
    ("role_availability_interactions_lo", 0.05),
    ("role_availability_interactions_hi", 0.30),
)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def norm_name(v: object) -> str:
    s = str(v or "").lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    if hasattr(value, "to_dicts"):
        return pd.DataFrame(value.to_dicts())
    return pd.DataFrame(value)


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def first_col(x: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in x.columns), None)


def canon(v: object) -> str:
    try:
        return canon_team(v)
    except Exception:
        return str(v or "").strip().upper()


def load_research_inputs(m95f_root: Path, m95b_root: Path):
    oof = lower(pd.read_csv(find_one(m95f_root, "m95f_2024_temporal_oof_scores.csv"), low_memory=False))
    hold = lower(pd.read_csv(find_one(m95f_root, "m95f_2024_holdout_trace.csv"), low_memory=False))
    val = lower(pd.read_csv(find_one(m95f_root, "m95f_2025_rb_trace.csv"), low_memory=False))
    trace = lower(pd.read_csv(find_one(m95b_root, "m95b_rb_matchup_trace.csv"), low_memory=False))
    for x in (oof, hold, val, trace):
        x["season"] = num(x["season"]).astype(int)
        x["week"] = num(x["week"]).astype(int)
        x["team"] = x["team"].map(canon)
        x["player_clean_key"] = x["player_clean_key"].astype(str)
    return oof, hold, val, trace


def normalize_rosters(raw: pd.DataFrame) -> pd.DataFrame:
    cols = PLAYER_KEYS + ["position", "roster_status", "self_roster_present", "self_roster_unavailable", "self_roster_active"]
    x = lower(raw)
    if x.empty:
        return pd.DataFrame(columns=cols)
    name = first_col(x, ["full_name", "football_name", "player_name", "player", "name"])
    team = first_col(x, ["team", "club_code", "team_abbr", "team_abbreviation"])
    pos = first_col(x, ["position", "position_group"])
    status = first_col(x, ["status", "roster_status", "game_status"])
    if not name or not team or not pos or "season" not in x or "week" not in x:
        raise RuntimeError(f"weekly roster schema incomplete: {list(x.columns)}")
    out = pd.DataFrame(index=x.index)
    out["season"] = num(x["season"])
    out["week"] = num(x["week"])
    out["team"] = x[team].map(canon)
    out["player_clean_key"] = x[name].map(norm_name)
    out["position"] = x[pos].astype(str).str.upper().str.strip()
    out["roster_status"] = x[status].astype(str).str.upper().str.strip() if status else ""
    out = out.loc[out["position"].isin(RB_POS) & out["season"].notna() & out["week"].notna() & out["player_clean_key"].ne("")].copy()
    unavailable_pattern = r"\b(INA|INACTIVE|IR|RES|RESERVE|PUP|NFI|SUS|SUSPENDED)\b"
    out["self_roster_present"] = 1
    out["self_roster_unavailable"] = out["roster_status"].str.contains(unavailable_pattern, regex=True, na=False).astype(int)
    out["self_roster_active"] = (1 - out["self_roster_unavailable"]).astype(int)
    return out[cols].drop_duplicates(PLAYER_KEYS, keep="last").reset_index(drop=True)


def normalize_injuries(raw: pd.DataFrame) -> pd.DataFrame:
    cols = PLAYER_KEYS + [
        "injury_status", "practice_status", "self_injury_reported", "self_inj_out",
        "self_inj_doubtful", "self_inj_questionable", "self_practice_dnp", "self_practice_limited",
    ]
    x = lower(raw)
    if x.empty:
        return pd.DataFrame(columns=cols)
    name = first_col(x, ["full_name", "player_name", "player", "name"])
    team = first_col(x, ["team", "team_abbr", "team_abbreviation", "club"])
    status = first_col(x, ["report_status", "game_status", "status"])
    practice = first_col(x, ["practice_status", "practice_participation"])
    if not name or not team or "season" not in x or "week" not in x:
        raise RuntimeError(f"injury schema incomplete: {list(x.columns)}")
    out = pd.DataFrame(index=x.index)
    out["season"] = num(x["season"])
    out["week"] = num(x["week"])
    out["team"] = x[team].map(canon)
    out["player_clean_key"] = x[name].map(norm_name)
    out["injury_status"] = x[status].astype(str).str.upper().str.strip() if status else ""
    out["practice_status"] = x[practice].astype(str).str.upper().str.strip() if practice else ""
    out = out.loc[out["season"].notna() & out["week"].notna() & out["player_clean_key"].ne("")].copy()
    out["self_injury_reported"] = 1
    out["self_inj_out"] = out["injury_status"].str.contains(r"\bOUT\b", regex=True, na=False).astype(int)
    out["self_inj_doubtful"] = out["injury_status"].str.contains("DOUBT", na=False).astype(int)
    out["self_inj_questionable"] = out["injury_status"].str.contains("QUESTION", na=False).astype(int)
    out["self_practice_dnp"] = out["practice_status"].str.contains(r"DID NOT|\bDNP\b", regex=True, na=False).astype(int)
    out["self_practice_limited"] = out["practice_status"].str.contains("LIMIT", na=False).astype(int)
    agg = {c: "max" for c in cols if c not in PLAYER_KEYS + ["injury_status", "practice_status"]}
    last = out.sort_values(PLAYER_KEYS).drop_duplicates(PLAYER_KEYS, keep="last")
    if not agg:
        return last[cols]
    flags = out.groupby(PLAYER_KEYS, as_index=False).agg(agg)
    labels = last[PLAYER_KEYS + ["injury_status", "practice_status"]]
    return labels.merge(flags, on=PLAYER_KEYS, how="left", validate="one_to_one")[cols]


def normalize_depth(raw: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    cols = PLAYER_KEYS + ["depth_rank", "depth_is_rb1"]
    x = lower(raw)
    if x.empty or not {"season", "week"}.issubset(x.columns):
        return pd.DataFrame(columns=cols), False
    name = first_col(x, ["full_name", "football_name", "player_name", "player", "name"])
    team = first_col(x, ["club_code", "team", "team_abbr"])
    pos = first_col(x, ["position", "depth_position", "position_group"])
    rank = first_col(x, ["depth_team", "depth_rank", "rank"])
    if not name or not team or not pos or not rank:
        return pd.DataFrame(columns=cols), False
    out = pd.DataFrame(index=x.index)
    out["season"] = num(x["season"])
    out["week"] = num(x["week"])
    out["team"] = x[team].map(canon)
    out["player_clean_key"] = x[name].map(norm_name)
    out["position"] = x[pos].astype(str).str.upper().str.strip()
    out["depth_rank"] = num(x[rank])
    out = out.loc[
        out["season"].notna() & out["week"].notna() & out["player_clean_key"].ne("")
        & out["position"].str.contains("RB|HB|FB", regex=True, na=False)
    ].copy()
    out["depth_is_rb1"] = out["depth_rank"].eq(1).astype(int)
    return out[cols].drop_duplicates(PLAYER_KEYS, keep="last"), True


def load_provider_sources(seasons: list[int]):
    import nflreadpy as nfl

    roster_parts = []
    depth_parts = []
    audit = []
    for season in seasons:
        try:
            r = to_pandas(nfl.load_rosters_weekly(int(season)))
            roster_parts.append(r)
            audit.append({"source": "nflreadpy_weekly_rosters", "season": season, "rows": len(r), "status": "ok"})
        except Exception as exc:
            audit.append({"source": "nflreadpy_weekly_rosters", "season": season, "rows": 0, "status": f"error:{type(exc).__name__}:{exc}"})
        try:
            d = to_pandas(nfl.load_depth_charts(int(season)))
            depth_parts.append(d)
            audit.append({"source": "nflreadpy_depth_charts", "season": season, "rows": len(d), "status": "loaded"})
        except Exception as exc:
            audit.append({"source": "nflreadpy_depth_charts", "season": season, "rows": 0, "status": f"unavailable:{type(exc).__name__}:{exc}"})
    try:
        inj = to_pandas(nfl.load_injuries(seasons=seasons))
        audit.append({"source": "nflreadpy_injuries", "season": 0, "rows": len(inj), "status": "ok"})
    except Exception as exc:
        inj = pd.DataFrame()
        audit.append({"source": "nflreadpy_injuries", "season": 0, "rows": 0, "status": f"error:{type(exc).__name__}:{exc}"})

    if not roster_parts:
        raise RuntimeError("M95G requires leakage-safe weekly roster data")
    rosters = normalize_rosters(pd.concat(roster_parts, ignore_index=True, sort=False))
    injuries = normalize_injuries(inj)
    depth_raw = pd.concat(depth_parts, ignore_index=True, sort=False) if depth_parts else pd.DataFrame()
    depth, weekly_depth = normalize_depth(depth_raw)
    audit.append({"source": "week_tagged_depth_usable", "season": 0, "rows": len(depth), "status": "yes" if weekly_depth else "no_leakage_safe_week_field"})
    return rosters, injuries, depth, pd.DataFrame(audit)


def add_roster_transition_features(rosters: pd.DataFrame) -> pd.DataFrame:
    r = rosters.sort_values(["season", "player_clean_key", "week", "team"]).copy()
    grp = r.groupby(["season", "player_clean_key"], sort=False)
    r["_prev_team"] = grp["team"].shift(1)
    r["_prev_week"] = grp["week"].shift(1)
    r["_prev_active"] = grp["self_roster_active"].shift(1)
    r["new_roster_entry"] = r["_prev_week"].isna().astype(int)
    r["team_change_recent"] = (r["_prev_team"].notna() & r["_prev_team"].ne(r["team"])).astype(int)
    r["returned_active"] = (r["self_roster_active"].eq(1) & r["_prev_active"].eq(0)).astype(int)
    r["reappeared_after_gap"] = (num(r["week"]) - num(r["_prev_week"])).ge(2).astype(int)
    return r.drop(columns=["_prev_team", "_prev_week", "_prev_active"])


def add_depth_transition_features(depth: pd.DataFrame) -> pd.DataFrame:
    if depth.empty:
        depth = depth.copy()
        depth["depth_promotion"] = np.nan
        return depth
    d = depth.sort_values(["season", "player_clean_key", "team", "week"]).copy()
    d["_prev_depth_rank"] = d.groupby(["season", "player_clean_key", "team"])["depth_rank"].shift(1)
    d["depth_promotion"] = (
        d["depth_rank"].notna() & d["_prev_depth_rank"].notna()
        & d["depth_rank"].lt(d["_prev_depth_rank"])
    ).astype(int)
    return d.drop(columns=["_prev_depth_rank"])


def build_team_source_features(rosters: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    rt = rosters.groupby(TEAM_KEYS, as_index=False).agg(
        team_rb_roster_count=("player_clean_key", "nunique"),
        team_rb_available_count=("self_roster_active", "sum"),
        team_rb_roster_unavailable_count=("self_roster_unavailable", "sum"),
    )
    if injuries.empty:
        injt = pd.DataFrame(columns=TEAM_KEYS + ["team_rb_injury_count", "team_rb_out_count", "team_rb_doubtful_count", "team_rb_questionable_count"])
    else:
        # Restrict injury aggregation to players identified as RB/HB/FB in the
        # same weekly roster snapshot whenever possible.
        ik = injuries.merge(rosters[PLAYER_KEYS], on=PLAYER_KEYS, how="inner")
        injt = ik.groupby(TEAM_KEYS, as_index=False).agg(
            team_rb_injury_count=("self_injury_reported", "sum"),
            team_rb_out_count=("self_inj_out", "sum"),
            team_rb_doubtful_count=("self_inj_doubtful", "sum"),
            team_rb_questionable_count=("self_inj_questionable", "sum"),
        )
    return rt.merge(injt, on=TEAM_KEYS, how="left", validate="one_to_one")


def previous_team_leaders(trace: pd.DataFrame) -> pd.DataFrame:
    z = trace.copy()
    if "actual_carries" not in z.columns:
        if "actual_rush_att" in z.columns:
            z["actual_carries"] = num(z["actual_rush_att"])
        else:
            raise RuntimeError("M95B trace missing actual carry truth for prior-game leader construction")
    z = z.loc[z["season"].isin([2024, 2025])].copy()
    rows = []
    for (season, team, week), g in z.groupby(TEAM_KEYS):
        q = g.loc[num(g["actual_carries"]).notna()].copy()
        if q.empty:
            continue
        q["actual_carries"] = num(q["actual_carries"])
        q = q.sort_values(["actual_carries", "player_clean_key"], ascending=[False, True])
        rows.append({
            "season": int(season), "team": canon(team), "week": int(week),
            "game_top1_key": str(q.iloc[0]["player_clean_key"]),
            "game_top1_carries": float(q.iloc[0]["actual_carries"]),
            "game_top2_key": str(q.iloc[1]["player_clean_key"]) if len(q) > 1 else "",
            "game_top2_carries": float(q.iloc[1]["actual_carries"]) if len(q) > 1 else 0.0,
        })
    game = pd.DataFrame(rows).sort_values(["season", "team", "week"])
    grp = game.groupby(["season", "team"], sort=False)
    game["prior_top1_key"] = grp["game_top1_key"].shift(1)
    game["prior_top1_carries"] = grp["game_top1_carries"].shift(1)
    game["prior_top2_key"] = grp["game_top2_key"].shift(1)
    game["prior_top2_carries"] = grp["game_top2_carries"].shift(1)
    return game[TEAM_KEYS + ["prior_top1_key", "prior_top1_carries", "prior_top2_key", "prior_top2_carries"]]


def source_player_lookup(rosters: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    r = rosters[PLAYER_KEYS + ["self_roster_present", "self_roster_unavailable", "self_roster_active"]].copy()
    i = injuries[PLAYER_KEYS + ["self_inj_out", "self_inj_doubtful"]].copy() if not injuries.empty else pd.DataFrame(columns=PLAYER_KEYS + ["self_inj_out", "self_inj_doubtful"])
    x = r.merge(i, on=PLAYER_KEYS, how="outer")
    for c in ["self_roster_present", "self_roster_unavailable", "self_roster_active", "self_inj_out", "self_inj_doubtful"]:
        x[c] = num(x[c]).fillna(0)
    x["source_unavailable"] = (
        x["self_roster_present"].eq(0)
        | x["self_roster_unavailable"].eq(1)
        | x["self_inj_out"].eq(1)
        | x["self_inj_doubtful"].eq(1)
    ).astype(int)
    return x


def enrich_base(base: pd.DataFrame, trace: pd.DataFrame, rosters: pd.DataFrame, injuries: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    z = base.copy()
    feature_cols = [
        "rb_games_before", "rb_carries_avg1", "rb_carries_avg3", "rb_carries_avg5",
        "rb_rb_share_avg1", "rb_rb_share_avg3", "rb_rb_share_avg5",
        "role_is_workhorse", "role_is_starter_plus",
    ]
    t = trace[PLAYER_KEYS + [c for c in feature_cols if c in trace.columns]].drop_duplicates(PLAYER_KEYS)
    for c in feature_cols:
        if c in z.columns:
            z = z.drop(columns=[c])
    z = z.merge(t, on=PLAYER_KEYS, how="left", validate="one_to_one")

    rcols = PLAYER_KEYS + [
        "self_roster_present", "self_roster_unavailable", "self_roster_active",
        "new_roster_entry", "returned_active", "team_change_recent", "reappeared_after_gap",
    ]
    z = z.merge(rosters[rcols], on=PLAYER_KEYS, how="left", validate="one_to_one")
    icols = PLAYER_KEYS + [
        "self_injury_reported", "self_inj_out", "self_inj_doubtful", "self_inj_questionable",
        "self_practice_dnp", "self_practice_limited",
    ]
    if injuries.empty:
        for c in icols[len(PLAYER_KEYS):]:
            z[c] = 0
    else:
        z = z.merge(injuries[icols], on=PLAYER_KEYS, how="left", validate="one_to_one")

    team_src = build_team_source_features(rosters, injuries)
    z = z.merge(team_src, on=TEAM_KEYS, how="left", validate="many_to_one")

    if depth.empty:
        z["depth_rank"] = np.nan
        z["depth_is_rb1"] = np.nan
        z["depth_promotion"] = np.nan
    else:
        z = z.merge(depth[PLAYER_KEYS + ["depth_rank", "depth_is_rb1", "depth_promotion"]], on=PLAYER_KEYS, how="left", validate="one_to_one")

    leaders = previous_team_leaders(trace)
    z = z.merge(leaders, on=TEAM_KEYS, how="left", validate="many_to_one")
    lookup = source_player_lookup(rosters, injuries)
    for n in ("top1", "top2"):
        lk = lookup.rename(columns={"player_clean_key": f"prior_{n}_key", "source_unavailable": f"prior_{n}_unavailable"})
        keep = TEAM_KEYS + [f"prior_{n}_key", f"prior_{n}_unavailable"]
        z = z.merge(lk[keep], on=TEAM_KEYS + [f"prior_{n}_key"], how="left", validate="many_to_one")
        # A previous leader absent from the current weekly roster is itself an
        # availability signal, not missing data.
        z[f"prior_{n}_unavailable"] = num(z[f"prior_{n}_unavailable"]).fillna(z[f"prior_{n}_key"].notna().astype(int))

    fill_zero = [
        "self_roster_present", "self_roster_unavailable", "self_roster_active",
        "self_injury_reported", "self_inj_out", "self_inj_doubtful", "self_inj_questionable",
        "self_practice_dnp", "self_practice_limited", "new_roster_entry", "returned_active",
        "team_change_recent", "reappeared_after_gap", "team_rb_injury_count", "team_rb_out_count",
        "team_rb_doubtful_count", "team_rb_questionable_count",
    ]
    for c in fill_zero:
        z[c] = num(z[c]).fillna(0)
    z["team_rb_roster_count"] = num(z["team_rb_roster_count"])
    z["team_rb_available_count"] = num(z["team_rb_available_count"])
    z["other_rb_available_count"] = (z["team_rb_available_count"] - z["self_roster_active"]).clip(lower=0)
    z["other_rb_out_count"] = (z["team_rb_out_count"] - z["self_inj_out"]).clip(lower=0)
    z["other_rb_doubtful_count"] = (z["team_rb_doubtful_count"] - z["self_inj_doubtful"]).clip(lower=0)
    z["other_rb_questionable_count"] = (z["team_rb_questionable_count"] - z["self_inj_questionable"]).clip(lower=0)
    z["target_was_prior_top1"] = z["player_clean_key"].eq(z["prior_top1_key"]).astype(int)
    z["target_was_prior_top2"] = z["player_clean_key"].eq(z["prior_top2_key"]).astype(int)
    z["vacated_lead_role"] = ((z["target_was_prior_top1"].eq(0)) & num(z["prior_top1_unavailable"]).eq(1)).astype(int)
    z["vacated_top2_role"] = ((z["target_was_prior_top2"].eq(0)) & num(z["prior_top2_unavailable"]).eq(1)).astype(int)
    z["late_week_17plus"] = num(z["week"]).ge(17).astype(int)
    z["week18"] = num(z["week"]).eq(18).astype(int)
    z["competition_scarcity"] = 1.0 / (1.0 + z["other_rb_available_count"].fillna(2.0))
    return z


def add_model_features(x: pd.DataFrame, score_col: str = "raw_score") -> pd.DataFrame:
    z = x.copy()
    p = num(z[score_col]).clip(1e-5, 1 - 1e-5)
    z["_score_logit"] = np.log(p / (1 - p))
    z["ix_score_x_vacancy"] = z["_score_logit"] * num(z["vacated_lead_role"]).fillna(0)
    z["ix_score_x_self_unavailable"] = z["_score_logit"] * (
        num(z["self_roster_unavailable"]).fillna(0) + num(z["self_inj_out"]).fillna(0) + num(z["self_inj_doubtful"]).fillna(0)
    ).clip(0, 1)
    z["ix_score_x_prior_leader"] = z["_score_logit"] * num(z["target_was_prior_top1"]).fillna(0)
    z["ix_score_x_depth1"] = z["_score_logit"] * num(z["depth_is_rb1"]).fillna(0)
    z["ix_vacancy_x_new_entry"] = num(z["vacated_lead_role"]).fillna(0) * num(z["new_roster_entry"]).fillna(0)
    z["ix_vacancy_x_competition_scarcity"] = num(z["vacated_lead_role"]).fillna(0) * num(z["competition_scarcity"]).fillna(0)
    return z


def feature_columns(spec: str, x: pd.DataFrame) -> list[str]:
    if spec == "availability":
        wanted = ["_score_logit"] + AVAILABILITY_FEATURES
    elif spec == "role_availability":
        wanted = BASE_ROLE_FEATURES + AVAILABILITY_FEATURES
    else:
        wanted = BASE_ROLE_FEATURES + AVAILABILITY_FEATURES + INTERACTION_FEATURES
    return [c for c in wanted if c in x.columns and num(x[c]).notna().sum() >= 10]


def pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=3000, random_state=SEED)),
    ])


def fit_apply(train: pd.DataFrame, test: pd.DataFrame, spec: str, c: float, target: str) -> np.ndarray:
    tr = add_model_features(train)
    te = add_model_features(test)
    feats = feature_columns(spec, tr)
    if not feats:
        raise RuntimeError(f"M95G no features for {spec}")
    y = tr[target].astype(int)
    if y.nunique() < 2:
        raise RuntimeError(f"M95G one-class training for {target}")
    model = pipeline(c)
    model.fit(tr[feats], y)
    return np.clip(model.predict_proba(te[feats])[:, 1], 1e-6, 1 - 1e-6)


def ece(y, p, bins: int = 10) -> float:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if z.empty:
        return np.nan
    try:
        z["bin"] = pd.qcut(z["p"], bins, duplicates="drop")
    except ValueError:
        return abs(float(z["p"].mean() - z["y"].mean()))
    total = len(z)
    return float(sum(len(g) / total * abs(float(g["p"].mean() - g["y"].mean())) for _, g in z.groupby("bin", observed=True)))


def prob_metrics(y, p) -> dict:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    yy = z["y"].astype(int)
    pp = z["p"].clip(1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan
    return {
        "n": len(z), "base_rate": float(yy.mean()), "mean_prob": float(pp.mean()),
        "auc": auc, "brier": float(np.square(pp - yy).mean()),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])), "ece": ece(yy, pp),
    }


def threshold_grid(y, p):
    yy = num(y).astype(int)
    truth = yy.eq(1)
    actual = int(truth.sum())
    rows = []
    for t in THRESHOLDS:
        pred = num(pd.Series(p, index=yy.index)).ge(t)
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        flagged = int(pred.sum())
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "predicted_positive": flagged, "actual_positive": actual,
            "flag_multiple": flagged / max(actual, 1),
            "eligible": int(flagged <= max(int(math.ceil(MAX_FLAG_MULTIPLE * actual)), actual + 5)),
        })
    grid = pd.DataFrame(rows)
    pool = grid.loc[grid["eligible"].eq(1)].copy()
    if pool.empty:
        pool = grid.copy()
    pool = pool.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, False])
    return float(pool.iloc[0]["threshold"]), grid


def special_slice_metrics(x: pd.DataFrame, prob_col: str, target: str, scope: str) -> pd.DataFrame:
    z = x.copy()
    share_trend = num(z.get("rb_rb_share_avg1", np.nan)) - num(z.get("rb_rb_share_avg5", np.nan))
    groups = {
        "all": pd.Series(True, index=z.index),
        "stable_workhorse": num(z.get("role_is_workhorse", 0)).eq(1) & share_trend.ge(-0.10),
        "prior_top1_unavailable": num(z.get("prior_top1_unavailable", 0)).eq(1),
        "vacated_lead_role": num(z.get("vacated_lead_role", 0)).eq(1),
        "new_roster_entry": num(z.get("new_roster_entry", 0)).eq(1),
        "returned_active": num(z.get("returned_active", 0)).eq(1),
        "self_out_or_doubtful": (num(z.get("self_inj_out", 0)) + num(z.get("self_inj_doubtful", 0))).gt(0),
        "week17_18": num(z["week"]).ge(17),
    }
    rows = []
    for name, mask in groups.items():
        g = z.loc[mask]
        if g.empty:
            continue
        rows.append({
            "scope": scope, "target": target, "slice": name, "n": len(g),
            "actual_rate": float(num(g[target]).mean()),
            "predicted_rate": float(num(g[prob_col]).mean()),
            "calibration_gap": float(num(g[prob_col]).mean() - num(g[target]).mean()),
        })
    return pd.DataFrame(rows)


def operating_metrics(y, p, threshold: float) -> dict:
    yy = num(y).astype(int); pp = num(pd.Series(p, index=yy.index)); truth = yy.eq(1); pred = pp.ge(threshold)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "predicted_positive": int(pred.sum()), "actual_positive": int(truth.sum()),
        "flag_multiple": int(pred.sum()) / max(int(truth.sum()), 1),
    }


def select_candidate(train: pd.DataFrame, hold: pd.DataFrame, target: str, baseline_prob_col: str):
    base_m = prob_metrics(hold[target], hold[baseline_prob_col])
    base_s = special_slice_metrics(hold, baseline_prob_col, target, "2024_holdout_baseline")
    q = base_s.loc[base_s["slice"].eq("stable_workhorse"), "calibration_gap"]
    base_gap = abs(float(q.iloc[0])) if len(q) else np.inf
    rows = []
    predictions = {}
    threshold_tables = []
    for spec, c in CALIBRATOR_SPECS:
        p = fit_apply(train, hold, spec, c, target)
        name = f"{spec}_C{c:g}"
        predictions[name] = p
        met = prob_metrics(hold[target], p)
        tmp = hold.copy(); tmp["_candidate_prob"] = p
        sm = special_slice_metrics(tmp, "_candidate_prob", target, "2024_holdout_candidate")
        qq = sm.loc[sm["slice"].eq("stable_workhorse"), "calibration_gap"]
        gap = abs(float(qq.iloc[0])) if len(qq) else np.inf
        threshold, grid = threshold_grid(hold[target], p)
        op = operating_metrics(hold[target], p, threshold)
        grid.insert(0, "target", target); grid.insert(1, "candidate", name)
        threshold_tables.append(grid)
        eligible = int(
            met["brier"] <= base_m["brier"] + 0.002
            and met["auc"] >= base_m["auc"] - 0.005
            and gap <= base_gap + 0.01
            and op["flag_multiple"] <= MAX_FLAG_MULTIPLE + 1e-9
        )
        selection_score = (
            (base_m["brier"] - met["brier"])
            + 0.25 * (met["auc"] - base_m["auc"])
            + 0.50 * (base_gap - gap)
            - 0.002 * max(op["flag_multiple"] - 2.0, 0)
        )
        rows.append({
            "target": target, "candidate": name, "spec": spec, "C": c,
            **met, "stable_workhorse_abs_gap": gap, "baseline_brier": base_m["brier"],
            "baseline_auc": base_m["auc"], "baseline_stable_workhorse_abs_gap": base_gap,
            "threshold": threshold, "op_precision": op["precision"], "op_recall": op["recall"],
            "op_fp": op["fp"], "op_flag_multiple": op["flag_multiple"],
            "eligible": eligible, "selection_score": selection_score,
        })
    grid = pd.DataFrame(rows)
    pool = grid.loc[grid["eligible"].eq(1)].copy()
    if pool.empty:
        pool = grid.copy()
    pool = pool.sort_values(["selection_score", "brier", "auc"], ascending=[False, True, False])
    chosen = pool.iloc[0].to_dict()
    return chosen, grid.sort_values("selection_score", ascending=False), pd.concat(threshold_tables, ignore_index=True), predictions[str(chosen["candidate"])]


def calibration_bins(x: pd.DataFrame, prob_col: str, target: str, label: str):
    z = x[[prob_col, target]].dropna().copy()
    try:
        z["bin"] = pd.qcut(z[prob_col], 5, labels=False, duplicates="drop") + 1
    except ValueError:
        z["bin"] = 1
    out = z.groupby("bin", as_index=False).agg(n=(target, "size"), predicted=(prob_col, "mean"), actual=(target, "mean"))
    out.insert(0, "target", target); out.insert(1, "probability", label)
    return out


def example_tables(x: pd.DataFrame, target: str, threshold: float):
    prob = f"m95g_prob_{'20' if target == 'actual_20plus' else '25'}"
    base = f"cal_prob_{'20' if target == 'actual_20plus' else '25'}"
    cols = [c for c in ["season", "week", "team", "player_clean_key", "player", "actual_carries", "m94c_rush_att", base, prob,
                              "vacated_lead_role", "prior_top1_unavailable", "new_roster_entry", "returned_active",
                              "self_inj_out", "self_inj_doubtful", "team_rb_available_count", "other_rb_available_count"] if c in x.columns]
    fp = x.loc[num(x[prob]).ge(threshold) & num(x[target]).eq(0), cols].copy()
    fp["prob_drop_vs_m95f"] = num(fp[prob]) - num(fp[base])
    fp = fp.sort_values(prob, ascending=False).head(40)
    fn = x.loc[num(x[target]).eq(1) & num(x[prob]).lt(threshold), cols].copy()
    fn["prob_change_vs_m95f"] = num(fn[prob]) - num(fn[base])
    fn = fn.sort_values(prob, ascending=True).head(40)
    return fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95f-root", type=Path, required=True)
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m95g"))
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    oof0, hold0, val0, trace = load_research_inputs(args.m95f_root, args.m95b_root)
    rosters, injuries, depth, source_audit = load_provider_sources([2024, 2025])
    rosters = add_roster_transition_features(rosters)
    depth = add_depth_transition_features(depth)

    # Prepare one row per target from the stacked M95F temporal OOF artifact.
    oof_by_target = {}
    for target in TARGETS:
        z = oof0.loc[oof0[target].notna()].copy()
        z[target] = num(z[target]).astype(int)
        z = enrich_base(z, trace, rosters, injuries, depth)
        oof_by_target[target] = z

    hold = enrich_base(hold0, trace, rosters, injuries, depth)
    val = enrich_base(val0, trace, rosters, injuries, depth)

    selections = []
    dev_grids = []
    threshold_grids = []
    prob_audit_rows = []
    slice_audits = []
    bin_frames = []
    example_frames = []

    for target, suffix in [("actual_20plus", "20"), ("actual_25plus", "25")]:
        oof = oof_by_target[target]
        train = oof.loc[num(oof["week"]).le(12)].copy()
        baseline_col = f"cal_prob_{suffix}"
        chosen, grid, tgrid, hold_p = select_candidate(train, hold, target, baseline_col)
        dev_grids.append(grid); threshold_grids.append(tgrid)
        selections.append(chosen)

        hold[f"m95g_prob_{suffix}"] = hold_p
        for label, pcol in [("m95f", baseline_col), ("m95g", f"m95g_prob_{suffix}")]:
            prob_audit_rows.append({"scope": "2024_w13_18_holdout", "target": target, "probability": label, **prob_metrics(hold[target], hold[pcol])})
            slice_audits.append(special_slice_metrics(hold, pcol, target, f"2024_{label}"))

        # Freeze architecture and refit on all temporal 2024 OOF rows.
        spec = str(chosen["spec"]); c = float(chosen["C"])
        val_p = fit_apply(oof, val, spec, c, target)
        val[f"m95g_prob_{suffix}"] = val_p
        threshold = float(chosen["threshold"])
        for label, pcol in [("m95f", baseline_col), ("m95g", f"m95g_prob_{suffix}")]:
            met = prob_metrics(val[target], val[pcol])
            op = operating_metrics(val[target], val[pcol], threshold)
            prob_audit_rows.append({"scope": "2025_untouched_validation", "target": target, "probability": label, **met, **{f"op_{k}": v for k, v in op.items()}})
            slice_audits.append(special_slice_metrics(val, pcol, target, f"2025_{label}"))
            bin_frames.append(calibration_bins(val, pcol, target, label))

        fp, fn = example_tables(val, target, threshold)
        fp.insert(0, "target", target); fp.insert(1, "example_type", "false_positive")
        fn.insert(0, "target", target); fn.insert(1, "example_type", "false_negative")
        example_frames.extend([fp, fn])

    selection_df = pd.DataFrame(selections)
    probability_audit = pd.DataFrame(prob_audit_rows)
    slices = pd.concat([x for x in slice_audits if not x.empty], ignore_index=True, sort=False)

    # Validation gate: both targets must preserve ranking/calibration and the
    # 25+ highest-risk stable-workhorse overprediction must improve materially.
    def metric(scope, target, prob, col):
        q = probability_audit.loc[(probability_audit["scope"].eq(scope)) & (probability_audit["target"].eq(target)) & (probability_audit["probability"].eq(prob)), col]
        return float(q.iloc[0]) if len(q) else np.nan

    target_pass = {}
    for target in TARGETS:
        b0 = metric("2025_untouched_validation", target, "m95f", "brier")
        b1 = metric("2025_untouched_validation", target, "m95g", "brier")
        a0 = metric("2025_untouched_validation", target, "m95f", "auc")
        a1 = metric("2025_untouched_validation", target, "m95g", "auc")
        fm = metric("2025_untouched_validation", target, "m95g", "op_flag_multiple")
        target_pass[target] = int(b1 <= b0 + 0.001 and a1 >= a0 - 0.005 and fm <= 4.0)

    sw = slices.loc[(slices["scope"].eq("2025_m95f")) & (slices["target"].eq("actual_25plus")) & (slices["slice"].eq("stable_workhorse")), "calibration_gap"]
    sg = slices.loc[(slices["scope"].eq("2025_m95g")) & (slices["target"].eq("actual_25plus")) & (slices["slice"].eq("stable_workhorse")), "calibration_gap"]
    stable_improvement = (abs(float(sg.iloc[0])) <= 0.80 * abs(float(sw.iloc[0]))) if len(sw) and len(sg) and float(sw.iloc[0]) != 0 else False
    validation_pass = int(all(target_pass.values()) and stable_improvement)
    disposition = "ADVANCE_M95G_ROLE_AVAILABILITY_FOR_INTEGRATION_REVIEW" if validation_pass else "RETAIN_M95G_AS_DIAGNOSTIC_DO_NOT_PROMOTE"

    source_coverage = pd.DataFrame([
        {"measure": "2024_holdout_roster_present_rate", "value": float(num(hold["self_roster_present"]).mean())},
        {"measure": "2025_validation_roster_present_rate", "value": float(num(val["self_roster_present"]).mean())},
        {"measure": "2024_holdout_injury_match_rate", "value": float(num(hold["self_injury_reported"]).mean())},
        {"measure": "2025_validation_injury_match_rate", "value": float(num(val["self_injury_reported"]).mean())},
        {"measure": "week_tagged_depth_row_count", "value": float(len(depth))},
    ])

    disp = pd.DataFrame([{
        "selected_20": str(selection_df.loc[selection_df["target"].eq("actual_20plus"), "candidate"].iloc[0]),
        "selected_25": str(selection_df.loc[selection_df["target"].eq("actual_25plus"), "candidate"].iloc[0]),
        "threshold_20": float(selection_df.loc[selection_df["target"].eq("actual_20plus"), "threshold"].iloc[0]),
        "threshold_25": float(selection_df.loc[selection_df["target"].eq("actual_25plus"), "threshold"].iloc[0]),
        "2025_20_brier_m95f": metric("2025_untouched_validation", "actual_20plus", "m95f", "brier"),
        "2025_20_brier_m95g": metric("2025_untouched_validation", "actual_20plus", "m95g", "brier"),
        "2025_25_brier_m95f": metric("2025_untouched_validation", "actual_25plus", "m95f", "brier"),
        "2025_25_brier_m95g": metric("2025_untouched_validation", "actual_25plus", "m95g", "brier"),
        "2025_20_auc_m95f": metric("2025_untouched_validation", "actual_20plus", "m95f", "auc"),
        "2025_20_auc_m95g": metric("2025_untouched_validation", "actual_20plus", "m95g", "auc"),
        "2025_25_auc_m95f": metric("2025_untouched_validation", "actual_25plus", "m95f", "auc"),
        "2025_25_auc_m95g": metric("2025_untouched_validation", "actual_25plus", "m95g", "auc"),
        "stable_workhorse_25_gap_m95f": float(sw.iloc[0]) if len(sw) else np.nan,
        "stable_workhorse_25_gap_m95g": float(sg.iloc[0]) if len(sg) else np.nan,
        "target20_pass": target_pass["actual_20plus"], "target25_pass": target_pass["actual_25plus"],
        "stable_workhorse_improvement_pass": int(stable_improvement),
        "validation_pass": validation_pass, "m94c_central_mean_preserved": 1,
        "disposition": disposition, "production_change": 0,
    }])

    source_audit.to_csv(args.out_dir / "m95g_source_audit.csv", index=False)
    source_coverage.to_csv(args.out_dir / "m95g_source_coverage.csv", index=False)
    selection_df.to_csv(args.out_dir / "m95g_selected_architecture.csv", index=False)
    pd.concat(dev_grids, ignore_index=True).to_csv(args.out_dir / "m95g_2024_candidate_grid.csv", index=False)
    pd.concat(threshold_grids, ignore_index=True).to_csv(args.out_dir / "m95g_2024_threshold_grid.csv", index=False)
    probability_audit.to_csv(args.out_dir / "m95g_probability_audit.csv", index=False)
    slices.to_csv(args.out_dir / "m95g_role_slice_audit.csv", index=False)
    pd.concat(bin_frames, ignore_index=True).to_csv(args.out_dir / "m95g_2025_probability_bins.csv", index=False)
    pd.concat(example_frames, ignore_index=True, sort=False).to_csv(args.out_dir / "m95g_2025_examples.csv", index=False)
    hold.to_csv(args.out_dir / "m95g_2024_holdout_trace.csv", index=False)
    val.to_csv(args.out_dir / "m95g_2025_rb_trace.csv", index=False)
    disp.to_csv(args.out_dir / "m95g_disposition.csv", index=False)

    print("[m95g] disposition")
    print(disp.to_string(index=False))
    print("\n[m95g] source audit")
    print(source_audit.to_string(index=False))
    print("\n[m95g] source coverage")
    print(source_coverage.to_string(index=False))
    print("\n[m95g] selected architectures")
    print(selection_df.to_string(index=False))
    print("\n[m95g] probability audit")
    print(probability_audit.to_string(index=False))
    print("\n[m95g] role slice audit")
    print(slices.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
