#!/usr/bin/env python3
"""WR-R18 Stage A: receiver-attributed target CPOE diagnostic.

Research only. Implements the frozen 2023 development stage from
WR_R18_RECEIVER_TARGET_CPOE_V1_PLAN.md plus the accepted reporting-only
amendments in WR_R18_CLAUDE_REVIEW_AMENDMENTS_V1.md.

This script never scores 2024.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

AUTHORITY_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
EXPECTED_ROWS = {2023: 2076, 2024: 2117}
DEV_SEASON = 2023
PRIOR_GAMES = 8
MIN_PRIOR_TARGET_GAMES = 4
MIN_VALID_CPOE_TARGETS = 16
MIN_COVERAGE = 0.60
MIN_SPEARMAN = 0.08
MIN_RESIDUAL_GAP = 5.0
MIN_TAIL_RATIO = 1.20
MIN_SLICE_N = 150
MEDIATION_MIN_SPEARMAN = 0.04
MEDIATION_MIN_RESIDUAL_GAP = 2.5


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _name_key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        return str(key) if key else ""
    except Exception:
        return ""


def _regular_only(x: pd.DataFrame) -> pd.DataFrame:
    q = x.copy()
    c = "season_type" if "season_type" in q.columns else "game_type" if "game_type" in q.columns else None
    if c:
        s = q[c].astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            q = q.loc[keep].copy()
    return q


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _first(frame: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series(default, index=frame.index)


def load_authority(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    required = {
        "variant", "team", "player_clean_key", "player", "wr_rank", "pred_targets",
        "entitlement_tgt_share", "mc_receptions", "mc_rec_yards", "season", "week",
        "actual_targets", "actual_rec_yards",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"WR-R18 authority missing columns: {missing}")
    x = x.loc[x["variant"].astype(str).eq(AUTHORITY_VARIANT)].copy()
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    counts = x.groupby("season").size().to_dict()
    if counts != EXPECTED_ROWS:
        raise RuntimeError(f"WR-R18 authority row-count parity failed: {counts} != {EXPECTED_ROWS}")
    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(10).to_dict("records")
        raise RuntimeError(f"WR-R18 duplicate authority identities: {bad}")
    x["yard_residual"] = _num(x["actual_rec_yards"]) - _num(x["mc_rec_yards"])
    return x.sort_values(keys).reset_index(drop=True)


def load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    frames = []
    for season in sorted({int(s) for s in seasons}):
        q = _to_pandas(nfl.load_pbp(seasons=[season]))
        if not q.empty:
            frames.append(_regular_only(q))
    if not frames:
        raise RuntimeError("WR-R18 historical PBP source returned zero rows")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_roster_identity(seasons: Iterable[int]) -> pd.DataFrame:
    """Identity-only, strictly-prior-capable weekly roster source."""
    import nflreadpy as nfl

    frames = []
    for season in sorted({int(s) for s in seasons}):
        raw = _to_pandas(nfl.load_rosters_weekly(season))
        if raw.empty:
            raise RuntimeError(f"WR-R18 weekly roster source returned zero rows for {season}")
        x = raw.copy()
        x.columns = [str(c).strip().lower() for c in x.columns]
        x["season"] = pd.to_numeric(_first(x, ["season"], season), errors="coerce").fillna(season).astype(int)
        x["week"] = pd.to_numeric(_first(x, ["week"]), errors="coerce")
        raw_name = _first(x, ["full_name", "football_name", "player_name", "player", "name"])
        x["player_clean_key"] = raw_name.map(_name_key)
        x["team"] = _first(x, ["team", "team_abbr", "club_code"]).map(canon_team)
        x["player_id"] = _first(x, ["gsis_id", "player_id"]).map(_clean_id)
        x = x.loc[
            x["season"].eq(season)
            & x["week"].between(1, 22, inclusive="both")
            & x["player_clean_key"].ne("")
            & x["player_id"].ne("")
        ].copy()
        if x.empty:
            raise RuntimeError(f"WR-R18 weekly roster source has zero stable identities for {season}")
        frames.append(x[["season", "week", "team", "player_clean_key", "player_id"]])
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(["season", "week", "team", "player_clean_key", "player_id"], keep="last")


def prepare_pbp_sources(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare receiver targets (including null CPOE) and all team attempts."""
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    needed = [
        "season", "week", "game_id", "posteam", "receiver_player_name",
        "receiver_player_id", "pass_attempt", "sack", "two_point_attempt",
        "cpoe", "air_yards",
    ]
    for c in needed:
        if c not in x.columns:
            x[c] = np.nan
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x["team"] = x["posteam"].map(canon_team)
    official = _num(x["pass_attempt"]).fillna(0).eq(1)
    official &= ~_num(x["sack"]).fillna(0).eq(1)
    official &= ~_num(x["two_point_attempt"]).fillna(0).eq(1)
    x = x.loc[official & x["season"].notna() & x["week"].notna() & x["team"].ne("")].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["receiver_id"] = x["receiver_player_id"].map(_clean_id)
    x["receiver_name_key"] = x["receiver_player_name"].map(_name_key)
    x["cpoe_num"] = _num(x["cpoe"])
    x["air"] = _num(x["air_yards"])
    x["pass_event_seq"] = np.arange(len(x), dtype=np.int64)

    attempts = x[["season", "week", "game_id", "team", "cpoe_num", "air", "pass_event_seq"]].copy()
    targets = x.loc[x["receiver_id"].ne("") | x["receiver_name_key"].ne("")].copy()
    targets["target_event_seq"] = np.arange(len(targets), dtype=np.int64)
    targets = targets[[
        "season", "week", "game_id", "team", "receiver_id", "receiver_name_key",
        "cpoe_num", "air", "target_event_seq",
    ]].reset_index(drop=True)
    return targets, attempts.reset_index(drop=True)


def _prior(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return frame.loc[
        (frame["season"] < int(season))
        | ((frame["season"] == int(season)) & (frame["week"] < int(week)))
    ].copy()


def _last_games(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    games = (
        frame[["season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["season", "week", "game_id"], kind="mergesort")
        .tail(int(n))
    )
    return frame.merge(games, on=["season", "week", "game_id"], how="inner", validate="many_to_one")


def resolve_roster_player_id(
    rosters: pd.DataFrame,
    authority_name_key: str,
    authority_team: str,
    season: int,
    week: int,
) -> dict:
    prior = _prior(rosters, season, week)
    exact = prior.loc[prior["player_clean_key"].eq(str(authority_name_key))].copy()
    ids = sorted({_clean_id(v) for v in exact["player_id"] if _clean_id(v)})
    out = {
        "roster_identity_mode": "unmatched",
        "roster_player_id": "",
        "roster_prior_name_rows": int(len(exact)),
        "roster_exact_name_unique_ids": int(len(ids)),
        "roster_team_disambiguated": False,
    }
    if len(ids) == 1:
        out.update({"roster_identity_mode": "id", "roster_player_id": ids[0]})
        return out
    if len(ids) > 1:
        team_rows = exact.loc[exact["team"].eq(str(authority_team))]
        team_ids = sorted({_clean_id(v) for v in team_rows["player_id"] if _clean_id(v)})
        if len(team_ids) == 1:
            out.update({
                "roster_identity_mode": "id",
                "roster_player_id": team_ids[0],
                "roster_team_disambiguated": True,
            })
        else:
            out["roster_identity_mode"] = "ambiguous"
    return out


def resolve_prior_receiver_history(
    targets: pd.DataFrame,
    rosters: pd.DataFrame,
    authority_name_key: str,
    authority_team: str,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, dict]:
    prior_targets = _prior(targets, season, week)
    roster = resolve_roster_player_id(rosters, authority_name_key, authority_team, season, week)

    if roster["roster_identity_mode"] == "ambiguous":
        return prior_targets.iloc[0:0].copy(), {
            **roster,
            "identity_mode": "ambiguous",
            "identity_source": "weekly_roster",
            "resolved_receiver_id": "",
            "fallback_event_count": 0,
        }
    if roster["roster_identity_mode"] == "id":
        rid = roster["roster_player_id"]
        by_id = prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy()
        fallback = prior_targets.loc[
            prior_targets["receiver_id"].eq("")
            & prior_targets["receiver_name_key"].eq(str(authority_name_key))
        ].copy()
        history = pd.concat([by_id, fallback], ignore_index=True, sort=False)
        return history, {
            **roster,
            "identity_mode": "id",
            "identity_source": "weekly_roster",
            "resolved_receiver_id": rid,
            "fallback_event_count": int(len(fallback)),
        }

    # Exact PBP-name fallback only when no stable prior roster anchor exists.
    alias = prior_targets.loc[prior_targets["receiver_name_key"].eq(str(authority_name_key))].copy()
    ids = sorted({_clean_id(v) for v in alias["receiver_id"] if _clean_id(v)})
    if len(ids) > 1:
        return alias.iloc[0:0].copy(), {
            **roster,
            "identity_mode": "ambiguous",
            "identity_source": "pbp_exact_name_fallback",
            "resolved_receiver_id": "",
            "fallback_event_count": 0,
        }
    if len(ids) == 1:
        rid = ids[0]
        by_id = prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy()
        fallback = prior_targets.loc[
            prior_targets["receiver_id"].eq("")
            & prior_targets["receiver_name_key"].eq(str(authority_name_key))
        ].copy()
        return pd.concat([by_id, fallback], ignore_index=True, sort=False), {
            **roster,
            "identity_mode": "id",
            "identity_source": "pbp_exact_name_fallback",
            "resolved_receiver_id": rid,
            "fallback_event_count": int(len(fallback)),
        }
    if len(alias):
        return alias, {
            **roster,
            "identity_mode": "name_fallback",
            "identity_source": "pbp_exact_name_fallback",
            "resolved_receiver_id": "",
            "fallback_event_count": int(len(alias)),
        }
    return prior_targets.iloc[0:0].copy(), {
        **roster,
        "identity_mode": "unmatched",
        "identity_source": "pbp_exact_name_fallback",
        "resolved_receiver_id": "",
        "fallback_event_count": 0,
    }


def receiver_state(
    targets: pd.DataFrame,
    rosters: pd.DataFrame,
    authority_name_key: str,
    authority_team: str,
    season: int,
    week: int,
) -> dict:
    history_all, audit = resolve_prior_receiver_history(
        targets, rosters, authority_name_key, authority_team, season, week
    )
    valid = history_all.loc[history_all["cpoe_num"].notna()].copy()
    h8_valid = _last_games(valid, PRIOR_GAMES)
    games8 = h8_valid[["season", "week", "game_id"]].drop_duplicates() if not h8_valid.empty else pd.DataFrame()
    if len(games8):
        h8_all = history_all.merge(games8, on=["season", "week", "game_id"], how="inner", validate="many_to_one")
    else:
        h8_all = history_all.iloc[0:0].copy()

    cpoe = _num(h8_valid["cpoe_num"]).dropna() if not h8_valid.empty else pd.Series(dtype=float)
    air_valid = _num(h8_valid["air"]).dropna() if not h8_valid.empty else pd.Series(dtype=float)
    null_mask = h8_all["cpoe_num"].isna() if not h8_all.empty else pd.Series(dtype=bool)
    out = {
        **audit,
        "prior_target_games": int(len(games8)),
        "prior_valid_cpoe_targets": int(len(cpoe)),
        "prior_otherwise_eligible_targets": int(len(h8_all)),
        "prior_null_cpoe_targets": int(null_mask.sum()) if len(h8_all) else 0,
        "prior_null_cpoe_rate": float(null_mask.mean()) if len(h8_all) else np.nan,
        "WR_TARGET_CPOE_MEAN8": np.nan,
        "mean_air_yards_per_target8": np.nan,
        "history_max_season": np.nan,
        "history_max_week": np.nan,
    }
    if len(games8):
        latest = games8.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["history_max_season"] = int(latest["season"])
        out["history_max_week"] = int(latest["week"])
    if audit["identity_mode"] in {"ambiguous", "unmatched"}:
        return out
    if len(games8) < MIN_PRIOR_TARGET_GAMES or len(cpoe) < MIN_VALID_CPOE_TARGETS:
        return out
    out["WR_TARGET_CPOE_MEAN8"] = float(cpoe.mean())
    out["mean_air_yards_per_target8"] = float(air_valid.mean()) if len(air_valid) else np.nan
    return out


def team_state(attempts: pd.DataFrame, team: str, season: int, week: int) -> dict:
    prior = _prior(attempts.loc[attempts["team"].eq(str(team))], season, week)
    h8 = _last_games(prior, PRIOR_GAMES)
    games8 = h8[["season", "week", "game_id"]].drop_duplicates() if not h8.empty else pd.DataFrame()
    cpoe = _num(h8["cpoe_num"]).dropna() if not h8.empty else pd.Series(dtype=float)
    out = {
        "team_prior_pass_games": int(len(games8)),
        "team_cpoe_mean8": float(cpoe.mean()) if len(cpoe) else np.nan,
        "team_history_max_season": np.nan,
        "team_history_max_week": np.nan,
    }
    if len(games8):
        latest = games8.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["team_history_max_season"] = int(latest["season"])
        out["team_history_max_week"] = int(latest["week"])
    return out


def build_development_panel(
    authority: pd.DataFrame,
    targets: pd.DataFrame,
    attempts: pd.DataFrame,
    rosters: pd.DataFrame,
) -> pd.DataFrame:
    dev = authority.loc[authority["season"].eq(DEV_SEASON)].copy()
    if len(dev) != EXPECTED_ROWS[DEV_SEASON]:
        raise RuntimeError("WR-R18 development authority count drift")
    rows = []
    for r in dev.itertuples(index=False):
        key = str(r.player_clean_key)
        display_key = _name_key(r.player)
        row = {
            "season": int(r.season),
            "week": int(r.week),
            "team": str(r.team),
            "player_clean_key": key,
            "player": str(r.player),
            "display_name_key": display_key,
            "authority_display_key_match": bool(not display_key or display_key == key),
            "wr_rank": int(r.wr_rank),
            "wr_rank_bucket": "WR1" if int(r.wr_rank) == 1 else "WR2PLUS",
            "pred_targets": float(r.pred_targets),
            "entitlement_tgt_share": float(r.entitlement_tgt_share),
            "mc_rec_yards": float(r.mc_rec_yards),
            "actual_rec_yards": float(r.actual_rec_yards),
            "yard_residual": float(r.yard_residual),
        }
        row.update(receiver_state(targets, rosters, key, row["team"], row["season"], row["week"]))
        row.update(team_state(attempts, row["team"], row["season"], row["week"]))
        rows.append(row)
    panel = pd.DataFrame(rows)

    has_receiver_hist = panel["history_max_season"].notna()
    receiver_before = (panel["history_max_season"] < panel["season"]) | (
        (panel["history_max_season"] == panel["season"])
        & (panel["history_max_week"] < panel["week"])
    )
    bad_receiver = int((has_receiver_hist & ~receiver_before).sum())
    if bad_receiver:
        raise RuntimeError(f"WR-R18 receiver target-game leakage assertion failed for {bad_receiver} rows")

    has_team_hist = panel["team_history_max_season"].notna()
    team_before = (panel["team_history_max_season"] < panel["season"]) | (
        (panel["team_history_max_season"] == panel["season"])
        & (panel["team_history_max_week"] < panel["week"])
    )
    bad_team = int((has_team_hist & ~team_before).sum())
    if bad_team:
        raise RuntimeError(f"WR-R18 team-control target-game leakage assertion failed for {bad_team} rows")
    return panel


def _spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(z) < 3 or z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan
    return float(z["a"].rank().corr(z["b"].rank()))


def _ratio(high: float, low: float) -> float:
    return float(high / low) if np.isfinite(high) and np.isfinite(low) and low > 0 else np.nan


def raw_stage_a(panel: pd.DataFrame) -> tuple[dict, dict]:
    d = panel.loc[
        _num(panel["WR_TARGET_CPOE_MEAN8"]).notna()
        & _num(panel["yard_residual"]).notna()
    ].copy()
    coverage = float(len(d) / len(panel))
    out = {
        "n": int(len(d)),
        "coverage": coverage,
        "spearman": np.nan,
        "q25": np.nan,
        "q75": np.nan,
        "q4_minus_q1_residual_gap": np.nan,
        "q4_actual100_rate": np.nan,
        "q1_actual100_rate": np.nan,
        "actual100_rate_ratio": np.nan,
        "q4_underproj30_rate": np.nan,
        "q1_underproj30_rate": np.nan,
        "underproj30_rate_ratio": np.nan,
        "wr1_n": 0,
        "wr1_gap": np.nan,
        "wr2plus_n": 0,
        "wr2plus_gap": np.nan,
        "supported_raw": False,
    }
    quartiles = {}
    if d.empty:
        return out, quartiles

    signal = "WR_TARGET_CPOE_MEAN8"
    q25 = float(d[signal].quantile(0.25, interpolation="linear"))
    q75 = float(d[signal].quantile(0.75, interpolation="linear"))
    low = d.loc[d[signal].le(q25)]
    high = d.loc[d[signal].ge(q75)]
    rho = _spearman(d[signal], d["yard_residual"])
    gap = float(high["yard_residual"].mean() - low["yard_residual"].mean())

    high100 = float(_num(high["actual_rec_yards"]).ge(100.0).mean()) if len(high) else np.nan
    low100 = float(_num(low["actual_rec_yards"]).ge(100.0).mean()) if len(low) else np.nan
    high30 = float(_num(high["yard_residual"]).ge(30.0).mean()) if len(high) else np.nan
    low30 = float(_num(low["yard_residual"]).ge(30.0).mean()) if len(low) else np.nan

    slice_vals = {}
    slice_ok = True
    for bucket in ["WR1", "WR2PLUS"]:
        s = d.loc[d["wr_rank_bucket"].eq(bucket)]
        slo = s.loc[s[signal].le(q25)]
        shi = s.loc[s[signal].ge(q75)]
        sgap = float(shi["yard_residual"].mean() - slo["yard_residual"].mean()) if len(slo) and len(shi) else np.nan
        required = len(s) >= MIN_SLICE_N
        coherent = bool(np.isfinite(sgap) and sgap > 0) if required else True
        slice_ok &= coherent
        slice_vals[bucket] = {"n": int(len(s)), "gap": sgap, "required": required, "positive": coherent}

    supported = bool(
        coverage >= MIN_COVERAGE
        and np.isfinite(rho) and rho >= MIN_SPEARMAN
        and np.isfinite(gap) and gap >= MIN_RESIDUAL_GAP
        and (
            (np.isfinite(_ratio(high100, low100)) and _ratio(high100, low100) >= MIN_TAIL_RATIO)
            or (np.isfinite(_ratio(high30, low30)) and _ratio(high30, low30) >= MIN_TAIL_RATIO)
        )
        and slice_ok
    )

    out.update({
        "spearman": rho,
        "q25": q25,
        "q75": q75,
        "q4_minus_q1_residual_gap": gap,
        "q4_actual100_rate": high100,
        "q1_actual100_rate": low100,
        "actual100_rate_ratio": _ratio(high100, low100),
        "q4_underproj30_rate": high30,
        "q1_underproj30_rate": low30,
        "underproj30_rate_ratio": _ratio(high30, low30),
        "wr1_n": slice_vals["WR1"]["n"],
        "wr1_gap": slice_vals["WR1"]["gap"],
        "wr2plus_n": slice_vals["WR2PLUS"]["n"],
        "wr2plus_gap": slice_vals["WR2PLUS"]["gap"],
        "supported_raw": supported,
    })

    # Descriptive-only quartile MAE and signed bias. Never used by gates.
    try:
        qlabels = pd.qcut(d[signal].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        qd = d.assign(_quartile=qlabels)
        for label, g in qd.groupby("_quartile", observed=True):
            quartiles[str(label)] = {
                "n": int(len(g)),
                "mae": float(_num(g["yard_residual"]).abs().mean()),
                "signed_bias": float(_num(g["yard_residual"]).mean()),
            }
    except Exception:
        quartiles = {}
    return out, quartiles


def mediation_robustness(panel: pd.DataFrame) -> dict:
    cols = [
        "WR_TARGET_CPOE_MEAN8", "team_cpoe_mean8", "mean_air_yards_per_target8",
        "entitlement_tgt_share", "wr_rank_bucket", "yard_residual",
    ]
    d = panel[cols].copy()
    for c in ["WR_TARGET_CPOE_MEAN8", "team_cpoe_mean8", "mean_air_yards_per_target8", "entitlement_tgt_share", "yard_residual"]:
        d[c] = _num(d[c])
    d = d.dropna()
    if len(d) < 10:
        return {"n": int(len(d)), "supported": False, "error": "insufficient_complete_rows"}

    y = d["WR_TARGET_CPOE_MEAN8"].to_numpy(dtype=float)
    wr1 = d["wr_rank_bucket"].eq("WR1").astype(float).to_numpy()
    X = np.column_stack([
        np.ones(len(d), dtype=float),
        d["team_cpoe_mean8"].to_numpy(dtype=float),
        d["mean_air_yards_per_target8"].to_numpy(dtype=float),
        d["entitlement_tgt_share"].to_numpy(dtype=float),
        wr1,
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    d = d.assign(receiver_specific_cpoe=residual)
    rho = _spearman(d["receiver_specific_cpoe"], d["yard_residual"])
    q25 = float(d["receiver_specific_cpoe"].quantile(0.25, interpolation="linear"))
    q75 = float(d["receiver_specific_cpoe"].quantile(0.75, interpolation="linear"))
    low = d.loc[d["receiver_specific_cpoe"].le(q25)]
    high = d.loc[d["receiver_specific_cpoe"].ge(q75)]
    gap = float(high["yard_residual"].mean() - low["yard_residual"].mean())
    supported = bool(
        np.isfinite(rho) and rho >= MEDIATION_MIN_SPEARMAN
        and np.isfinite(gap) and gap >= MEDIATION_MIN_RESIDUAL_GAP
    )
    return {
        "n": int(len(d)),
        "coefficients": {
            "intercept": float(beta[0]),
            "team_cpoe_mean8": float(beta[1]),
            "mean_air_yards_per_target8": float(beta[2]),
            "entitlement_tgt_share": float(beta[3]),
            "wr1_indicator": float(beta[4]),
        },
        "receiver_specific_spearman": rho,
        "receiver_specific_q25": q25,
        "receiver_specific_q75": q75,
        "receiver_specific_q4_minus_q1_residual_gap": gap,
        "supported": supported,
        "interpretation_caveat": (
            "A failed depth-controlled mediation check may indicate confounding or over-control of a real "
            "delivery-quality pathway; it is not causal disproof. Game-script/score-differential is unmodeled in V1."
        ),
    }


def cpoe_missingness_audit(targets: pd.DataFrame, panel: pd.DataFrame) -> dict:
    ids = sorted({str(v) for v in panel["resolved_receiver_id"].dropna().astype(str) if str(v)})
    x = targets.loc[targets["receiver_id"].isin(ids)].drop_duplicates("target_event_seq").copy()
    if x.empty:
        return {
            "resolved_wr_target_events": 0,
            "nonnull_cpoe_events": 0,
            "null_cpoe_events": 0,
            "null_cpoe_rate": np.nan,
        }
    null = x["cpoe_num"].isna()
    air = _num(x["air"])
    both = pd.DataFrame({"null": null.astype(float), "air": air}).dropna()
    corr = float(both["null"].corr(both["air"])) if len(both) >= 3 and both["null"].nunique() > 1 else np.nan
    return {
        "resolved_wr_target_events": int(len(x)),
        "nonnull_cpoe_events": int((~null).sum()),
        "null_cpoe_events": int(null.sum()),
        "null_cpoe_rate": float(null.mean()),
        "nonnull_cpoe_air_available": int((~null & air.notna()).sum()),
        "null_cpoe_air_available": int((null & air.notna()).sum()),
        "nonnull_cpoe_mean_air_yards": float(air.loc[~null].mean()) if air.loc[~null].notna().any() else np.nan,
        "null_cpoe_mean_air_yards": float(air.loc[null].mean()) if air.loc[null].notna().any() else np.nan,
        "nonnull_cpoe_median_air_yards": float(air.loc[~null].median()) if air.loc[~null].notna().any() else np.nan,
        "null_cpoe_median_air_yards": float(air.loc[null].median()) if air.loc[null].notna().any() else np.nan,
        "null_indicator_air_yards_pearson": corr,
        "note": "Descriptive selection audit only; never changes support floor, cohort, gates, or signal.",
    }


def identity_audit(panel: pd.DataFrame) -> dict:
    return {
        "development_rows": int(len(panel)),
        "expected_development_rows": EXPECTED_ROWS[DEV_SEASON],
        "identity_mode_counts": {
            str(k): int(v) for k, v in panel["identity_mode"].value_counts(dropna=False).to_dict().items()
        },
        "identity_source_counts": {
            str(k): int(v) for k, v in panel["identity_source"].value_counts(dropna=False).to_dict().items()
        },
        "rows_resolved_by_prior_roster_id": int(panel["identity_source"].eq("weekly_roster").sum()),
        "rows_team_disambiguated": int(panel["roster_team_disambiguated"].fillna(False).sum()),
        "rows_with_4plus_prior_target_games": int(panel["prior_target_games"].ge(MIN_PRIOR_TARGET_GAMES).sum()),
        "rows_with_16plus_valid_cpoe_targets": int(panel["prior_valid_cpoe_targets"].ge(MIN_VALID_CPOE_TARGETS).sum()),
        "rows_with_valid_cpoe_signal": int(panel["WR_TARGET_CPOE_MEAN8"].notna().sum()),
        "authority_display_key_mismatch_rows": int((~panel["authority_display_key_match"]).sum()),
        "target_game_leakage_rows": 0,
        "team_control_target_game_leakage_rows": 0,
        "sportsbook_inputs": 0,
        "pbp_history_seasons_loaded": [2022, 2023],
        "identity_roster_seasons_loaded": [2022, 2023],
        "holdout_2024_scored": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    authority = load_authority(args.authority)
    raw = load_pbp([2022, 2023])
    targets, attempts = prepare_pbp_sources(raw)
    rosters = load_roster_identity([2022, 2023])
    panel = build_development_panel(authority, targets, attempts, rosters)

    raw_metrics, quartile_report = raw_stage_a(panel)
    data_blocked = bool(raw_metrics["coverage"] < MIN_COVERAGE)
    mediation = None
    if data_blocked:
        disposition = "WR_RECEIVER_TARGET_CPOE_DATA_BLOCKED"
    elif not raw_metrics["supported_raw"]:
        disposition = "NO_ACTIONABLE_WR_RECEIVER_TARGET_CPOE_SIGNAL"
    else:
        mediation = mediation_robustness(panel)
        if mediation.get("supported", False):
            disposition = "WR_RECEIVER_TARGET_CPOE_DEVELOPMENT_SUPPORTED"
        else:
            disposition = "WR_TARGET_CPOE_TEAM_ROLE_MEDIATED"

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "wr_r18_stage_a_feature_panel_2023.csv", index=False)
    (out / "wr_r18_stage_a_raw_metrics_2023.json").write_text(
        json.dumps(raw_metrics, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (out / "wr_r18_stage_a_quartile_descriptives_2023.json").write_text(
        json.dumps(quartile_report, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (out / "wr_r18_stage_a_identity_audit.json").write_text(
        json.dumps(identity_audit(panel), indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (out / "wr_r18_stage_a_cpoe_missingness_audit.json").write_text(
        json.dumps(cpoe_missingness_audit(targets, panel), indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    if mediation is not None:
        (out / "wr_r18_stage_a_mediation_robustness.json").write_text(
            json.dumps(mediation, indent=2, sort_keys=True, allow_nan=True) + "\n"
        )

    result = {
        "specification": "WR_R18_RECEIVER_TARGET_CPOE_V1",
        "stage": "A_2023_DEVELOPMENT_ONLY",
        "disposition": disposition,
        "raw_development_supported": bool(raw_metrics["supported_raw"]),
        "mediation_supported": bool(mediation and mediation.get("supported", False)),
        "holdout_2024_scored": False,
        "authority_expected_rows": EXPECTED_ROWS,
        "signal": "WR_TARGET_CPOE_MEAN8",
        "frozen_direction": "positive",
        "production_change": False,
        "sportsbook_inputs": 0,
    }
    (out / "wr_r18_stage_a_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    print(pd.DataFrame([raw_metrics]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
