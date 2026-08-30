#!/usr/bin/env python3
"""M87: strictly-pregame forensic atlas for M86 low-chaos catastrophic QB misses.

No predictive model is fit. Target-game postgame information is used only through
M86's already-frozen tail/component/chaos labels. Atlas features are restricted
to frozen model state, strictly-prior PBP history, target-week injury reports,
and deterministic venue architecture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.stadium_locations import STADIUM_LOCATION

EXPECTED_ROWS = 884
EXPECTED_TAIL100 = 123
EXPECTED_LOW_CHAOS = 38
EXPECTED_PRIMARY = {"VOLUME_DOMINANT": 16, "EFFICIENCY_DOMINANT": 17, "MIXED": 5}
HISTORY_GAMES = 8
MIN_HISTORY_GAMES = 4
MATCH_K = 5
CONTROL_MAX_ERROR = 50.0

TEAM_REMAP = {"JAC": "JAX", "LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR"}
MATCH_VARS = ["week", "ensemble_proj", "pred_attempts", "implied_pred_ypa"]

MODEL_FEATURES = [
    "model_ensemble_proj", "model_pred_attempts", "model_pred_ypa",
    "model_component_sd", "model_component_range", "model_ml_minus_mc",
    "model_state_minus_mc", "model_ensemble_minus_canonical",
]
OFF_FEATURES = [
    "off_pass_rate", "off_neutral_pass_rate", "off_shotgun_rate", "off_no_huddle_rate",
    "off_plays_per_game", "off_pass_epa", "off_success_rate", "off_ypa",
    "off_explosive20_rate", "off_deep_attempt_rate", "off_sack_rate", "off_scramble_rate",
]
DEF_FEATURES = [
    "def_pass_rate_faced", "def_neutral_pass_rate_faced", "def_pass_epa_allowed",
    "def_success_rate_allowed", "def_ypa_allowed", "def_explosive20_rate_allowed",
    "def_deep_attempt_rate_faced", "def_sack_rate_generated", "def_interception_rate_generated",
    "def_plays_faced_per_game",
]
OPP_OFF_FEATURES = [
    "opp_off_pass_epa", "opp_off_success_rate", "opp_off_plays_per_game",
    "opp_off_neutral_pass_rate", "opp_off_ypa",
]
CONTEXT_FEATURES = [
    "ctx_team_injury_total", "ctx_team_out_doubtful", "ctx_team_questionable",
    "ctx_opp_injury_total", "ctx_opp_out_doubtful", "ctx_opp_questionable",
    "ctx_is_home", "ctx_controlled_environment",
]
ATLAS_FEATURES = MODEL_FEATURES + OFF_FEATURES + DEF_FEATURES + OPP_OFF_FEATURES + CONTEXT_FEATURES


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def canon_team_value(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    return TEAM_REMAP.get(s, s)


def canon_team(s: pd.Series) -> pd.Series:
    return s.map(canon_team_value)


def num(x: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in x.columns:
        return pd.Series(default, index=x.index, dtype=float)
    return pd.to_numeric(x[col], errors="coerce").fillna(default)


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    z = s.astype(str).str.strip().str.lower()
    return z.isin({"1", "true", "t", "yes", "y"})


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    return aa.div(bb.where(bb.ne(0)))


def load_trace(path: Path) -> pd.DataFrame:
    q = lower(pd.read_csv(path, low_memory=False)).reset_index(drop=True)
    if len(q) != EXPECTED_ROWS:
        raise RuntimeError(f"M86 trace row drift: {len(q)}")
    required = {
        "season", "week", "team", "opponent", "actual_pass_yards", "pred_pass_yards",
        "pred_attempts", "implied_pred_ypa", "ensemble_proj", "mc_proj", "ml_proj", "state_proj",
        "ensemble_abs_error", "tail100", "tail_direction", "component_class", "high_event_chaos",
    }
    missing = required - set(q.columns)
    if missing:
        raise RuntimeError(f"M86 trace missing M87 columns: {sorted(missing)}")
    q["season"] = pd.to_numeric(q["season"], errors="raise").astype(int)
    q["week"] = pd.to_numeric(q["week"], errors="raise").astype(int)
    q["team"] = canon_team(q["team"])
    q["opponent"] = canon_team(q["opponent"])
    numeric = [
        "actual_pass_yards", "pred_pass_yards", "pred_attempts", "implied_pred_ypa",
        "ensemble_proj", "mc_proj", "ml_proj", "state_proj", "ensemble_abs_error",
    ]
    for c in numeric:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["tail100"] = bool_series(q["tail100"])
    q["high_event_chaos"] = bool_series(q["high_event_chaos"])
    q["component_class"] = q["component_class"].astype(str).str.upper().str.strip()
    q["tail_direction"] = q["tail_direction"].astype(str).str.upper().str.strip()
    q["row_id"] = np.arange(len(q), dtype=int)

    tails = q.loc[q["tail100"]]
    low = tails.loc[~tails["high_event_chaos"]]
    if len(tails) != EXPECTED_TAIL100:
        raise RuntimeError(f"M87 upstream tail-count drift: {len(tails)}")
    if len(low) != EXPECTED_LOW_CHAOS:
        raise RuntimeError(f"M87 upstream low-chaos tail drift: {len(low)}")
    counts = low["component_class"].value_counts().to_dict()
    for k, v in EXPECTED_PRIMARY.items():
        if int(counts.get(k, 0)) != v:
            raise RuntimeError(f"M87 upstream {k} count drift: {counts}")
    return q


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_pbp(seasons=[2023, 2024, 2025])
    x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    x = lower(x)
    if "season_type" in x.columns:
        x = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].isin([2023, 2024, 2025]) & x["week"].between(1, 18)].copy()
    x["team"] = canon_team(x["posteam"])
    x["opponent"] = canon_team(x["defteam"])
    if "home_team" in x.columns:
        x["home_team_canon"] = canon_team(x["home_team"])
    else:
        x["home_team_canon"] = ""
    if "away_team" in x.columns:
        x["away_team_canon"] = canon_team(x["away_team"])
    else:
        x["away_team_canon"] = ""
    return x


def aggregate_offense_games(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pbp.copy()
    pass_attempt = num(x, "pass_attempt").eq(1)
    sack = num(x, "sack").eq(1)
    scramble = num(x, "qb_scramble").eq(1)
    if "qb_dropback" in x.columns:
        dropback = num(x, "qb_dropback").eq(1)
    else:
        dropback = pass_attempt | sack | scramble
    rush = num(x, "rush_attempt").eq(1) & ~scramble
    valid = dropback | rush
    score_diff = pd.to_numeric(x["score_differential"], errors="coerce") if "score_differential" in x.columns else pd.Series(np.nan, index=x.index)
    qtr = pd.to_numeric(x["qtr"], errors="coerce") if "qtr" in x.columns else pd.Series(np.nan, index=x.index)
    neutral = valid & score_diff.abs().le(7) & qtr.le(3)
    complete = num(x, "complete_pass").eq(1)
    pass_yards = pd.to_numeric(x["passing_yards"], errors="coerce").fillna(0.0) if "passing_yards" in x.columns else pd.Series(0.0, index=x.index)
    air = pd.to_numeric(x["air_yards"], errors="coerce") if "air_yards" in x.columns else pd.Series(np.nan, index=x.index)
    epa = pd.to_numeric(x["epa"], errors="coerce") if "epa" in x.columns else pd.Series(np.nan, index=x.index)
    success = pd.to_numeric(x["success"], errors="coerce") if "success" in x.columns else pd.Series(np.nan, index=x.index)

    x["m87_valid"] = valid.astype(int)
    x["m87_pass_call"] = dropback.astype(int)
    x["m87_neutral"] = neutral.astype(int)
    x["m87_neutral_pass"] = (neutral & dropback).astype(int)
    x["m87_shotgun"] = (valid & num(x, "shotgun").eq(1)).astype(int)
    x["m87_no_huddle"] = (valid & num(x, "no_huddle").eq(1)).astype(int)
    x["m87_pass_attempt"] = pass_attempt.astype(int)
    x["m87_pass_yards"] = np.where(pass_attempt, pass_yards, 0.0)
    x["m87_exp20"] = (pass_attempt & complete & pass_yards.ge(20)).astype(int)
    x["m87_deep"] = (pass_attempt & air.ge(15)).astype(int)
    x["m87_sack"] = sack.astype(int)
    x["m87_scramble"] = scramble.astype(int)
    x["m87_int"] = num(x, "interception").eq(1).astype(int)
    x["m87_pass_epa"] = epa.where(dropback)
    x["m87_success"] = success.where(valid)

    keys = ["season", "week", "team", "opponent"]
    g = x.groupby(keys, as_index=False).agg(
        plays=("m87_valid", "sum"),
        pass_calls=("m87_pass_call", "sum"),
        neutral_plays=("m87_neutral", "sum"),
        neutral_pass_calls=("m87_neutral_pass", "sum"),
        shotgun_plays=("m87_shotgun", "sum"),
        no_huddle_plays=("m87_no_huddle", "sum"),
        pass_attempts=("m87_pass_attempt", "sum"),
        pass_yards=("m87_pass_yards", "sum"),
        explosive20=("m87_exp20", "sum"),
        deep_attempts=("m87_deep", "sum"),
        sacks=("m87_sack", "sum"),
        scrambles=("m87_scramble", "sum"),
        interceptions=("m87_int", "sum"),
        pass_epa=("m87_pass_epa", "mean"),
        success_rate=("m87_success", "mean"),
    )
    g["pass_rate"] = safe_div(g["pass_calls"], g["plays"])
    g["neutral_pass_rate"] = safe_div(g["neutral_pass_calls"], g["neutral_plays"])
    g["shotgun_rate"] = safe_div(g["shotgun_plays"], g["plays"])
    g["no_huddle_rate"] = safe_div(g["no_huddle_plays"], g["plays"])
    g["ypa"] = safe_div(g["pass_yards"], g["pass_attempts"])
    g["explosive20_rate"] = safe_div(g["explosive20"], g["pass_attempts"])
    g["deep_attempt_rate"] = safe_div(g["deep_attempts"], g["pass_attempts"])
    g["sack_rate"] = safe_div(g["sacks"], g["pass_calls"])
    g["scramble_rate"] = safe_div(g["scrambles"], g["pass_calls"])
    g["interception_rate"] = safe_div(g["interceptions"], g["pass_attempts"])

    schedule_cols = ["season", "week", "home_team_canon", "away_team_canon"]
    sched = x[schedule_cols].drop_duplicates().loc[
        lambda d: d["home_team_canon"].astype(str).ne("") & d["away_team_canon"].astype(str).ne("")
    ].copy()
    sched = sched.drop_duplicates(["season", "week", "home_team_canon", "away_team_canon"])
    return g.sort_values(["season", "week", "team"]).reset_index(drop=True), sched.reset_index(drop=True)


def build_defense_games(off: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "season": off["season"], "week": off["week"], "team": off["opponent"], "opponent": off["team"],
        "def_pass_rate_faced": off["pass_rate"],
        "def_neutral_pass_rate_faced": off["neutral_pass_rate"],
        "def_pass_epa_allowed": off["pass_epa"],
        "def_success_rate_allowed": off["success_rate"],
        "def_ypa_allowed": off["ypa"],
        "def_explosive20_rate_allowed": off["explosive20_rate"],
        "def_deep_attempt_rate_faced": off["deep_attempt_rate"],
        "def_sack_rate_generated": off["sack_rate"],
        "def_interception_rate_generated": off["interception_rate"],
        "def_plays_faced_per_game": off["plays"],
    })
    return d.sort_values(["season", "week", "team"]).reset_index(drop=True)


def prior_window(df: pd.DataFrame, team: str, season: int, week: int) -> pd.DataFrame:
    h = df.loc[
        df["team"].eq(team)
        & df["season"].ge(int(season) - 1)
        & ((df["season"].lt(int(season))) | (df["season"].eq(int(season)) & df["week"].lt(int(week))))
    ].sort_values(["season", "week"])
    return h.tail(HISTORY_GAMES)


def add_history_features(q: pd.DataFrame, off: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    out = q.copy()
    rows = []
    off_map = {
        "off_pass_rate": "pass_rate", "off_neutral_pass_rate": "neutral_pass_rate",
        "off_shotgun_rate": "shotgun_rate", "off_no_huddle_rate": "no_huddle_rate",
        "off_plays_per_game": "plays", "off_pass_epa": "pass_epa", "off_success_rate": "success_rate",
        "off_ypa": "ypa", "off_explosive20_rate": "explosive20_rate", "off_deep_attempt_rate": "deep_attempt_rate",
        "off_sack_rate": "sack_rate", "off_scramble_rate": "scramble_rate",
    }
    opp_map = {
        "opp_off_pass_epa": "pass_epa", "opp_off_success_rate": "success_rate",
        "opp_off_plays_per_game": "plays", "opp_off_neutral_pass_rate": "neutral_pass_rate", "opp_off_ypa": "ypa",
    }
    for _, r in out.iterrows():
        team = r["team"]; opp = r["opponent"]; season = int(r["season"]); week = int(r["week"])
        ho = prior_window(off, team, season, week)
        hd = prior_window(defense, opp, season, week)
        hop = prior_window(off, opp, season, week)
        rec = {"row_id": int(r["row_id"]), "off_history_games": len(ho), "def_history_games": len(hd), "opp_off_history_games": len(hop)}
        for dst, src in off_map.items():
            rec[dst] = float(pd.to_numeric(ho[src], errors="coerce").mean()) if len(ho) >= MIN_HISTORY_GAMES else np.nan
        for dst in DEF_FEATURES:
            rec[dst] = float(pd.to_numeric(hd[dst], errors="coerce").mean()) if len(hd) >= MIN_HISTORY_GAMES else np.nan
        for dst, src in opp_map.items():
            rec[dst] = float(pd.to_numeric(hop[src], errors="coerce").mean()) if len(hop) >= MIN_HISTORY_GAMES else np.nan
        rows.append(rec)
    return out.merge(pd.DataFrame(rows), on="row_id", how="left", validate="one_to_one")


def add_model_features(q: pd.DataFrame) -> pd.DataFrame:
    x = q.copy()
    comp = x[["mc_proj", "ml_proj", "state_proj"]].apply(pd.to_numeric, errors="coerce")
    x["model_ensemble_proj"] = x["ensemble_proj"]
    x["model_pred_attempts"] = x["pred_attempts"]
    x["model_pred_ypa"] = x["implied_pred_ypa"]
    x["model_component_sd"] = comp.std(axis=1, ddof=0)
    x["model_component_range"] = comp.max(axis=1) - comp.min(axis=1)
    x["model_ml_minus_mc"] = x["ml_proj"] - x["mc_proj"]
    x["model_state_minus_mc"] = x["state_proj"] - x["mc_proj"]
    x["model_ensemble_minus_canonical"] = x["ensemble_proj"] - x["pred_pass_yards"]
    return x


def load_injury_context(path: Path | None) -> pd.DataFrame:
    cols = ["season", "week", "team", "injury_total", "out_doubtful", "questionable"]
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    x = lower(pd.read_csv(path, low_memory=False))
    required = {"season", "week", "team", "status"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M87 injury history missing: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = canon_team(x["team"])
    status = x["status"].astype(str).str.strip().str.lower()
    x["out_doubtful"] = status.isin({"out", "doubtful"}).astype(int)
    x["questionable"] = status.eq("questionable").astype(int)
    g = x.groupby(["season", "week", "team"], as_index=False).agg(
        injury_total=("status", "size"), out_doubtful=("out_doubtful", "sum"), questionable=("questionable", "sum")
    )
    return g[cols]


def add_context(q: pd.DataFrame, sched: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    x = q.copy()
    sched = sched.copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    sched["home_team_canon"] = canon_team(sched["home_team_canon"])
    sched["away_team_canon"] = canon_team(sched["away_team_canon"])
    home_lookup = {(int(r.season), int(r.week), r.home_team_canon, r.away_team_canon): r.home_team_canon for r in sched.itertuples(index=False)}
    # Match by unordered pair to survive canonical home/away orientation in the QB trace.
    pair_home = {}
    for r in sched.itertuples(index=False):
        key = (int(r.season), int(r.week), tuple(sorted([r.home_team_canon, r.away_team_canon])))
        pair_home[key] = r.home_team_canon

    is_home = []
    controlled = []
    for r in x.itertuples(index=False):
        key = (int(r.season), int(r.week), tuple(sorted([r.team, r.opponent])))
        home = pair_home.get(key, "")
        is_home.append(float(home == r.team) if home else np.nan)
        meta = STADIUM_LOCATION.get(home) if home else None
        controlled.append(float(not bool(meta.get("outdoor", True))) if meta is not None else np.nan)
    x["ctx_is_home"] = is_home
    x["ctx_controlled_environment"] = controlled

    if injuries.empty:
        for c in ["ctx_team_injury_total", "ctx_team_out_doubtful", "ctx_team_questionable", "ctx_opp_injury_total", "ctx_opp_out_doubtful", "ctx_opp_questionable"]:
            x[c] = np.nan
        return x

    team_inj = injuries.rename(columns={
        "injury_total": "ctx_team_injury_total", "out_doubtful": "ctx_team_out_doubtful", "questionable": "ctx_team_questionable"
    })
    x = x.merge(team_inj, on=["season", "week", "team"], how="left", validate="many_to_one")
    opp_inj = injuries.rename(columns={
        "team": "opponent", "injury_total": "ctx_opp_injury_total", "out_doubtful": "ctx_opp_out_doubtful", "questionable": "ctx_opp_questionable"
    })
    x = x.merge(opp_inj, on=["season", "week", "opponent"], how="left", validate="many_to_one")
    # No listed injuries is a valid zero for a target game/week.
    for c in ["ctx_team_injury_total", "ctx_team_out_doubtful", "ctx_team_questionable", "ctx_opp_injury_total", "ctx_opp_out_doubtful", "ctx_opp_questionable"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    return x


def match_controls(q: pd.DataFrame, family: str) -> pd.DataFrame:
    target = q.loc[q["tail100"] & ~q["high_event_chaos"] & q["component_class"].eq(family)].copy()
    controls = q.loc[
        ~q["tail100"] & ~q["high_event_chaos"] & q["ensemble_abs_error"].lt(CONTROL_MAX_ERROR)
    ].copy()
    rows = []
    for tr in target.itertuples(index=False):
        pool = controls.loc[controls["season"].eq(int(tr.season))].dropna(subset=MATCH_VARS).copy()
        if len(pool) < MATCH_K:
            raise RuntimeError(f"M87 insufficient controls family={family} season={tr.season}: {len(pool)}")
        mu = pool[MATCH_VARS].mean()
        sd = pool[MATCH_VARS].std(ddof=0).replace(0, 1.0).fillna(1.0)
        tv = pd.Series({v: getattr(tr, v) for v in MATCH_VARS}, dtype=float)
        dist = (((pool[MATCH_VARS] - mu) / sd) - ((tv - mu) / sd)).pow(2).sum(axis=1).pow(0.5)
        sel = pool.loc[dist.nsmallest(MATCH_K).index].copy()
        for idx in sel.index:
            rows.append({
                "family": family, "target_row_id": int(tr.row_id), "control_row_id": int(pool.loc[idx, "row_id"]),
                "target_season": int(tr.season), "distance": float(dist.loc[idx]),
            })
    return pd.DataFrame(rows)


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = float(np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0))
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0 if abs(float(x.mean() - y.mean())) <= 1e-12 else np.nan
    return float((x.mean() - y.mean()) / pooled)


def feature_group(feature: str) -> str:
    if feature in MODEL_FEATURES: return "MODEL_STATE"
    if feature in OFF_FEATURES: return "TARGET_OFFENSE_HISTORY"
    if feature in DEF_FEATURES: return "OPPONENT_DEFENSE_HISTORY"
    if feature in OPP_OFF_FEATURES: return "OPPONENT_OFFENSE_HISTORY"
    if feature in CONTEXT_FEATURES: return "PREGAME_CONTEXT"
    return "OTHER"


def build_atlas(q: pd.DataFrame, matches: pd.DataFrame, family: str) -> pd.DataFrame:
    t = q.loc[q["tail100"] & ~q["high_event_chaos"] & q["component_class"].eq(family)].copy()
    m = matches.loc[matches["family"].eq(family)].copy()
    c = m.merge(q, left_on="control_row_id", right_on="row_id", how="left", validate="many_to_one", suffixes=("", "_control"))
    rows = []
    for f in ATLAS_FEATURES:
        tcov = float(t[f].notna().mean()) if len(t) else 0.0
        ccov = float(c[f].notna().mean()) if len(c) else 0.0
        s_all = smd(t[f], c[f])
        by_year = {}
        for season in (2024, 2025):
            ty = t.loc[t["season"].eq(season), f]
            cy = c.loc[c["target_season"].eq(season), f]
            by_year[season] = smd(ty, cy)
        s24, s25 = by_year[2024], by_year[2025]
        sign_agree = bool(np.isfinite(s24) and np.isfinite(s25) and s24 * s25 > 0)
        stable = bool(
            tcov >= 0.85 and ccov >= 0.85 and np.isfinite(s_all) and abs(s_all) >= 0.50
            and sign_agree and abs(s24) >= 0.20 and abs(s25) >= 0.20
        )
        rows.append({
            "family": family, "feature_group": feature_group(f), "feature": f,
            "target_n": len(t), "control_match_rows": len(c), "target_coverage": tcov, "control_coverage": ccov,
            "target_mean": float(pd.to_numeric(t[f], errors="coerce").mean()),
            "control_mean": float(pd.to_numeric(c[f], errors="coerce").mean()),
            "smd_combined": s_all, "smd_2024": s24, "smd_2025": s25,
            "season_sign_agreement": sign_agree, "stable_forensic_differentiator": stable,
        })
    z = pd.DataFrame(rows)
    z["abs_smd_combined"] = z["smd_combined"].abs()
    return z.sort_values(["stable_forensic_differentiator", "abs_smd_combined"], ascending=[False, False]).reset_index(drop=True)


def build_directional_atlas(q: pd.DataFrame, matches: pd.DataFrame, family: str) -> pd.DataFrame:
    base = q.loc[q["tail100"] & ~q["high_event_chaos"] & q["component_class"].eq(family)].copy()
    rows = []
    for direction in ["UNDERPROJECTED", "OVERPROJECTED"]:
        t = base.loc[base["tail_direction"].eq(direction)].copy()
        if len(t) < 5:
            continue
        tids = set(t["row_id"].astype(int))
        m = matches.loc[matches["family"].eq(family) & matches["target_row_id"].isin(tids)].copy()
        c = m.merge(q, left_on="control_row_id", right_on="row_id", how="left", validate="many_to_one")
        for f in ATLAS_FEATURES:
            rows.append({
                "family": family, "tail_direction": direction, "feature_group": feature_group(f), "feature": f,
                "target_n": len(t), "control_match_rows": len(c),
                "target_mean": float(pd.to_numeric(t[f], errors="coerce").mean()),
                "control_mean": float(pd.to_numeric(c[f], errors="coerce").mean()),
                "smd_combined": smd(t[f], c[f]), "advancement_eligible": False,
            })
    z = pd.DataFrame(rows)
    if len(z):
        z["abs_smd_combined"] = z["smd_combined"].abs()
        z = z.sort_values(["family", "tail_direction", "abs_smd_combined"], ascending=[True, True, False])
    return z


def model_rescue(q: pd.DataFrame, family: str) -> dict:
    t = q.loc[q["tail100"] & ~q["high_event_chaos"] & q["component_class"].eq(family)].copy()
    actual = pd.to_numeric(t["actual_pass_yards"], errors="coerce")
    errs = pd.DataFrame({
        "MC": (pd.to_numeric(t["mc_proj"], errors="coerce") - actual).abs(),
        "ML": (pd.to_numeric(t["ml_proj"], errors="coerce") - actual).abs(),
        "STATE": (pd.to_numeric(t["state_proj"], errors="coerce") - actual).abs(),
    }, index=t.index)
    oracle = errs.min(axis=1)
    best = errs.idxmin(axis=1)
    counts = best.value_counts().to_dict()
    top_model = max(counts, key=counts.get) if counts else ""
    top_share = float(counts.get(top_model, 0) / len(t)) if len(t) else 0.0
    ensemble_mae = float(t["ensemble_abs_error"].mean()) if len(t) else np.nan
    oracle_mae = float(oracle.mean()) if len(oracle) else np.nan
    rescue75 = float(oracle.lt(75).mean()) if len(oracle) else 0.0
    rescue50 = float(oracle.lt(50).mean()) if len(oracle) else 0.0
    gain = ensemble_mae - oracle_mae if np.isfinite(ensemble_mae) and np.isfinite(oracle_mae) else np.nan
    clue = bool(top_share >= 0.60 and np.isfinite(gain) and gain >= 20.0 and rescue75 >= 0.50)
    comp = t[["mc_proj", "ml_proj", "state_proj"]].apply(pd.to_numeric, errors="coerce")
    return {
        "family": family, "n": len(t), "ensemble_mae": ensemble_mae, "component_oracle_mae": oracle_mae,
        "oracle_gain_vs_ensemble": gain, "rescue_below_75_rate": rescue75, "rescue_below_50_rate": rescue50,
        "best_mc_n": int(counts.get("MC", 0)), "best_ml_n": int(counts.get("ML", 0)), "best_state_n": int(counts.get("STATE", 0)),
        "top_hindsight_model": top_model, "top_hindsight_model_share": top_share,
        "mean_component_disagreement_sd": float(comp.std(axis=1, ddof=0).mean()),
        "model_representation_clue": clue,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m86-trace", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    q = load_trace(args.m86_trace)
    q = add_model_features(q)
    pbp = load_pbp()
    off, sched = aggregate_offense_games(pbp)
    defense = build_defense_games(off)
    q = add_history_features(q, off, defense)
    injuries = load_injury_context(args.injuries)
    q = add_context(q, sched, injuries)

    matches = pd.concat([match_controls(q, "VOLUME_DOMINANT"), match_controls(q, "EFFICIENCY_DOMINANT")], ignore_index=True)
    atlases = pd.concat([
        build_atlas(q, matches, "VOLUME_DOMINANT"),
        build_atlas(q, matches, "EFFICIENCY_DOMINANT"),
    ], ignore_index=True)
    directional = pd.concat([
        build_directional_atlas(q, matches, "VOLUME_DOMINANT"),
        build_directional_atlas(q, matches, "EFFICIENCY_DOMINANT"),
    ], ignore_index=True)
    rescue = pd.DataFrame([model_rescue(q, "VOLUME_DOMINANT"), model_rescue(q, "EFFICIENCY_DOMINANT")])

    stable = atlases.loc[atlases["stable_forensic_differentiator"]].copy()
    representation = rescue.loc[rescue["model_representation_clue"]].copy()
    if len(stable):
        disposition = "STABLE_PREGAME_DIFFERENTIATORS_FOUND"
        next_allowed = True
    elif len(representation):
        disposition = "MODEL_REPRESENTATION_CLUE_ONLY"
        next_allowed = True
    else:
        disposition = "NO_STABLE_LOW_CHAOS_PREGAME_DIFFERENTIATOR"
        next_allowed = False

    low = q.loc[q["tail100"] & ~q["high_event_chaos"]].copy()
    target_cols = [
        "row_id", "season", "week", "team", "opponent", "player_clean_key" if "player_clean_key" in q.columns else "team",
        "tail_direction", "component_class", "ensemble_abs_error", "ensemble_proj", "actual_pass_yards",
    ] + ATLAS_FEATURES
    # Preserve column order while removing a possible duplicate fallback column.
    target_cols = list(dict.fromkeys([c for c in target_cols if c in low.columns]))
    low[target_cols].sort_values(["component_class", "ensemble_abs_error"], ascending=[True, False]).to_csv(args.out_dir / "m87_low_chaos_targets_with_pregame_features.csv", index=False)
    matches.to_csv(args.out_dir / "m87_matched_controls.csv", index=False)
    atlases.to_csv(args.out_dir / "m87_feature_atlas.csv", index=False)
    directional.to_csv(args.out_dir / "m87_directional_atlas_exploratory.csv", index=False)
    rescue.to_csv(args.out_dir / "m87_model_rescue.csv", index=False)

    coverage = pd.DataFrame([{
        "feature": f, "feature_group": feature_group(f), "all_884_coverage": float(q[f].notna().mean()),
        "low_chaos_tail_coverage": float(low[f].notna().mean()),
    } for f in ATLAS_FEATURES])
    coverage.to_csv(args.out_dir / "m87_feature_coverage.csv", index=False)

    decision = {
        "migration": "M87", "rows": len(q), "tail100": int(q["tail100"].sum()),
        "low_event_chaos_tail_count": len(low),
        "volume_low_chaos_n": int((low["component_class"] == "VOLUME_DOMINANT").sum()),
        "efficiency_low_chaos_n": int((low["component_class"] == "EFFICIENCY_DOMINANT").sum()),
        "mixed_low_chaos_n": int((low["component_class"] == "MIXED").sum()),
        "stable_differentiator_count": int(len(stable)),
        "stable_differentiators": stable[["family", "feature_group", "feature", "smd_combined", "smd_2024", "smd_2025"]].to_dict(orient="records"),
        "model_representation_clue_families": representation["family"].tolist(),
        "final_disposition": disposition, "next_predictive_migration_allowed": next_allowed,
        "history_window_games": HISTORY_GAMES, "minimum_history_games": MIN_HISTORY_GAMES,
        "control_match_k": MATCH_K, "control_max_abs_error": CONTROL_MAX_ERROR,
        "postgame_features_used_as_atlas_features": False, "predictive_model_fit": False,
        "sportsbook_features_used": False, "production_actionable": False,
        "anti_loop": "Forensic differentiation does not reopen a previously failed family by itself; a future predictive test requires a frozen, genuinely unresolved pregame mechanism.",
    }
    (args.out_dir / "m87_decision.json").write_text(json.dumps(decision, indent=2, default=str) + "\n")

    print("[m87_decision]")
    print(json.dumps(decision, indent=2, default=str))
    print("[m87_model_rescue]")
    print(rescue.to_string(index=False))
    print("[m87_top_volume_features]")
    print(atlases.loc[atlases["family"].eq("VOLUME_DOMINANT")].head(12).to_string(index=False))
    print("[m87_top_efficiency_features]")
    print(atlases.loc[atlases["family"].eq("EFFICIENCY_DOMINANT")].head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
