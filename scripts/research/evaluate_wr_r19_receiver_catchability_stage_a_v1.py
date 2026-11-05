#!/usr/bin/env python3
"""WR-R19 Stage A mechanics: receiver-specific FTN catchability.

Research only. Implements the frozen 2023 development design from
WR_R19_RECEIVER_CATCHABILITY_V1_PLAN.md. This module can be imported for
synthetic tests without loading WR outcomes. The CLI, when later authorized,
will score 2023 only and must never score 2024.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

AUTHORITY_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
EXPECTED_ROWS = {2023: 2076, 2024: 2117}
DEV_SEASON = 2023
PRIOR_GAMES = 8
MIN_PRIOR_TARGET_GAMES = 4
MIN_RECEIVER_TARGETS = 16
MIN_TEAM_GAMES = 4
MIN_TEAM_TARGETS = 40
MIN_COVERAGE = 0.60
MIN_SPEARMAN = 0.08
MIN_RESIDUAL_GAP = 5.0
MIN_TAIL_RATIO = 1.20
MIN_SLICE_N = 150
MEDIATION_MIN_SPEARMAN = 0.06
MEDIATION_MIN_RESIDUAL_GAP = 4.0


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


def _bool_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    z = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[z.isin(["1", "true", "t", "yes", "y"])] = 1.0
    out.loc[z.isin(["0", "false", "f", "no", "n"])] = 0.0
    return out


def _first(frame: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series(default, index=frame.index)


def _read_parquet(url: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": "wr-r19-receiver-catchability-v1"})
    with urlopen(req, timeout=180) as r:
        raw = r.read()
        final = r.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_authority(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    required = {
        "variant", "team", "player_clean_key", "player", "wr_rank", "pred_targets",
        "entitlement_tgt_share", "mc_rec_yards", "season", "week", "actual_rec_yards",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"WR-R19 authority missing columns: {missing}")
    x = x.loc[x["variant"].astype(str).eq(AUTHORITY_VARIANT)].copy()
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    counts = x.groupby("season").size().to_dict()
    if counts != EXPECTED_ROWS:
        raise RuntimeError(f"WR-R19 authority row-count parity failed: {counts} != {EXPECTED_ROWS}")
    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(10).to_dict("records")
        raise RuntimeError(f"WR-R19 duplicate authority identities: {bad}")
    x["yard_residual"] = _num(x["actual_rec_yards"]) - _num(x["mc_rec_yards"])
    return x.sort_values(keys).reset_index(drop=True)


def load_roster_identity(seasons: Iterable[int]) -> pd.DataFrame:
    """Identity-only weekly roster source. Target-week rows are forbidden by resolver."""
    import nflreadpy as nfl

    frames = []
    for season in sorted({int(s) for s in seasons}):
        raw = nfl.load_rosters_weekly(season)
        x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        if x.empty:
            raise RuntimeError(f"WR-R19 weekly roster source returned zero rows for {season}")
        x.columns = [str(c).strip().lower() for c in x.columns]
        x["season"] = _num(_first(x, ["season"], season)).fillna(season).astype(int)
        x["week"] = _num(_first(x, ["week"]))
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
        frames.append(x[["season", "week", "team", "player_clean_key", "player_id"]])
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(["season", "week", "team", "player_clean_key", "player_id"], keep="last")


def load_ftn_pbp_targets(seasons: Iterable[int]) -> tuple[pd.DataFrame, list[dict]]:
    """Exact FTN game/play -> PBP target receiver join; all target outcomes retained."""
    frames: list[pd.DataFrame] = []
    source_meta: list[dict] = []
    for season in sorted({int(s) for s in seasons}):
        ftn_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        ftn, fm = _read_parquet(ftn_url)
        pbp, pm = _read_parquet(pbp_url)
        ftn.columns = [str(c).strip().lower() for c in ftn.columns]
        pbp.columns = [str(c).strip().lower() for c in pbp.columns]
        req_f = {"nflverse_game_id", "nflverse_play_id", "season", "week", "is_catchable_ball"}
        req_p = {
            "game_id", "play_id", "season", "week", "season_type", "posteam",
            "pass_attempt", "sack", "two_point_attempt", "receiver_player_id",
            "receiver_player_name", "air_yards", "complete_pass",
        }
        miss_f = sorted(req_f - set(ftn.columns))
        miss_p = sorted(req_p - set(pbp.columns))
        if miss_f or miss_p:
            raise RuntimeError(f"WR-R19 source schema missing season={season}: ftn={miss_f}, pbp={miss_p}")
        ftn = ftn.loc[_num(ftn["week"]).between(1, 18)].copy()
        pbp = pbp.loc[
            pbp["season_type"].astype(str).str.upper().eq("REG")
            & _num(pbp["week"]).between(1, 18)
        ].copy()
        ftn["join_game"] = ftn["nflverse_game_id"].astype(str)
        ftn["join_play"] = _num(ftn["nflverse_play_id"])
        pbp["join_game"] = pbp["game_id"].astype(str)
        pbp["join_play"] = _num(pbp["play_id"])
        if ftn.duplicated(["join_game", "join_play"]).any():
            raise RuntimeError(f"WR-R19 duplicate FTN game/play key season={season}")
        if pbp.duplicated(["join_game", "join_play"]).any():
            raise RuntimeError(f"WR-R19 duplicate PBP game/play key season={season}")
        pcols = [
            "join_game", "join_play", "game_id", "season", "week", "posteam",
            "pass_attempt", "sack", "two_point_attempt", "receiver_player_id",
            "receiver_player_name", "air_yards", "complete_pass",
        ]
        m = ftn.merge(pbp[pcols], on=["join_game", "join_play"], how="left", validate="one_to_one", indicator=True)
        join_rate = float(m["_merge"].eq("both").mean()) if len(m) else 0.0
        if join_rate < 0.95:
            raise RuntimeError(f"WR-R19 FTN/PBP exact join below source contract season={season}: {join_rate}")
        official = _num(m["pass_attempt"]).fillna(0).eq(1)
        official &= ~_num(m["sack"]).fillna(0).eq(1)
        official &= ~_num(m["two_point_attempt"]).fillna(0).eq(1)
        rid = m["receiver_player_id"].map(_clean_id)
        t = m.loc[official & rid.ne("")].copy()
        t["receiver_id"] = t["receiver_player_id"].map(_clean_id)
        t["receiver_name_key"] = t["receiver_player_name"].map(_name_key)
        t["team"] = t["posteam"].map(canon_team)
        t["season"] = _num(t["season"]).astype(int)
        t["week"] = _num(t["week"]).astype(int)
        t["catchable"] = _bool_num(t["is_catchable_ball"])
        t["air"] = _num(t["air_yards"])
        t["complete_pass_num"] = _num(t["complete_pass"]).fillna(0)
        t["target_event_seq"] = np.arange(len(t), dtype=np.int64)
        frames.append(t[[
            "season", "week", "game_id", "team", "receiver_id", "receiver_name_key",
            "catchable", "air", "complete_pass_num", "target_event_seq",
        ]])
        source_meta.append({
            "season": season,
            "ftn_sha256": fm["sha256"], "ftn_bytes": fm["bytes"],
            "pbp_sha256": pm["sha256"], "pbp_bytes": pm["bytes"],
            "exact_join_rate": join_rate,
            "receiver_target_rows": int(len(t)),
            "catchable_coverage": float(t["catchable"].notna().mean()) if len(t) else 0.0,
            "completed_target_rows": int(t["complete_pass_num"].eq(1).sum()),
            "incomplete_target_rows": int(t["complete_pass_num"].eq(0).sum()),
        })
    if not frames:
        raise RuntimeError("WR-R19 FTN/PBP target source returned zero rows")
    return pd.concat(frames, ignore_index=True, sort=False), source_meta


def _prior(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return frame.loc[
        (frame["season"] < int(season))
        | ((frame["season"] == int(season)) & (frame["week"] < int(week)))
    ].copy()


def _last_games(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    games = (
        frame[["season", "week", "game_id"]].drop_duplicates()
        .sort_values(["season", "week", "game_id"], kind="mergesort").tail(int(n))
    )
    return frame.merge(games, on=["season", "week", "game_id"], how="inner", validate="many_to_one")


def resolve_roster_player_id(rosters: pd.DataFrame, name_key: str, team: str, season: int, week: int) -> dict:
    prior = _prior(rosters, season, week)
    exact = prior.loc[prior["player_clean_key"].eq(str(name_key))].copy()
    ids = sorted({_clean_id(v) for v in exact["player_id"] if _clean_id(v)})
    out = {
        "roster_identity_mode": "unmatched", "roster_player_id": "",
        "roster_prior_name_rows": int(len(exact)), "roster_exact_name_unique_ids": int(len(ids)),
        "roster_team_disambiguated": False,
    }
    if len(ids) == 1:
        out.update({"roster_identity_mode": "id", "roster_player_id": ids[0]})
        return out
    if len(ids) > 1:
        team_ids = sorted({_clean_id(v) for v in exact.loc[exact["team"].eq(str(team)), "player_id"] if _clean_id(v)})
        if len(team_ids) == 1:
            out.update({"roster_identity_mode": "id", "roster_player_id": team_ids[0], "roster_team_disambiguated": True})
        else:
            out["roster_identity_mode"] = "ambiguous"
    return out


def resolve_prior_receiver_history(
    targets: pd.DataFrame, rosters: pd.DataFrame, name_key: str, team: str, season: int, week: int
) -> tuple[pd.DataFrame, dict]:
    prior_targets = _prior(targets, season, week)
    roster = resolve_roster_player_id(rosters, name_key, team, season, week)
    if roster["roster_identity_mode"] == "ambiguous":
        return prior_targets.iloc[0:0].copy(), {**roster, "identity_mode": "ambiguous", "identity_source": "weekly_roster", "resolved_receiver_id": ""}
    if roster["roster_identity_mode"] == "id":
        rid = roster["roster_player_id"]
        return prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy(), {
            **roster, "identity_mode": "id", "identity_source": "weekly_roster", "resolved_receiver_id": rid,
        }
    # Exact PBP-name fallback only when no stable prior roster anchor exists.
    alias = prior_targets.loc[prior_targets["receiver_name_key"].eq(str(name_key))].copy()
    ids = sorted({_clean_id(v) for v in alias["receiver_id"] if _clean_id(v)})
    if len(ids) > 1:
        return alias.iloc[0:0].copy(), {**roster, "identity_mode": "ambiguous", "identity_source": "pbp_exact_name_fallback", "resolved_receiver_id": ""}
    if len(ids) == 1:
        rid = ids[0]
        return prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy(), {
            **roster, "identity_mode": "id", "identity_source": "pbp_exact_name_fallback", "resolved_receiver_id": rid,
        }
    return alias.iloc[0:0].copy(), {**roster, "identity_mode": "unmatched", "identity_source": "pbp_exact_name_fallback", "resolved_receiver_id": ""}


def receiver_state(
    targets: pd.DataFrame, rosters: pd.DataFrame, name_key: str, team: str, season: int, week: int
) -> dict:
    history, audit = resolve_prior_receiver_history(targets, rosters, name_key, team, season, week)
    # Literal frozen contract: choose target-bearing games first, then aggregate catchability inside them.
    h8_all = _last_games(history, PRIOR_GAMES)
    games8 = h8_all[["season", "week", "game_id"]].drop_duplicates() if not h8_all.empty else pd.DataFrame()
    catch = _num(h8_all["catchable"]).dropna() if not h8_all.empty else pd.Series(dtype=float)
    air = _num(h8_all["air"]).dropna() if not h8_all.empty else pd.Series(dtype=float)
    out = {
        **audit,
        "prior_target_games": int(len(games8)),
        "prior_target_events": int(len(h8_all)),
        "prior_valid_catchable_targets": int(len(catch)),
        "WR_TARGET_CATCHABLE_RATE8": np.nan,
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
    if len(games8) < MIN_PRIOR_TARGET_GAMES or len(catch) < MIN_RECEIVER_TARGETS:
        return out
    out["WR_TARGET_CATCHABLE_RATE8"] = float(catch.mean())
    out["mean_air_yards_per_target8"] = float(air.mean()) if len(air) else np.nan
    return out


def team_state(targets: pd.DataFrame, team: str, season: int, week: int) -> dict:
    prior = _prior(targets.loc[targets["team"].eq(str(team))], season, week)
    h8 = _last_games(prior, PRIOR_GAMES)
    games8 = h8[["season", "week", "game_id"]].drop_duplicates() if not h8.empty else pd.DataFrame()
    catch = _num(h8["catchable"]).dropna() if not h8.empty else pd.Series(dtype=float)
    usable = len(games8) >= MIN_TEAM_GAMES and len(catch) >= MIN_TEAM_TARGETS
    out = {
        "team_prior_target_games": int(len(games8)),
        "team_prior_target_events": int(len(h8)),
        "team_valid_catchable_targets": int(len(catch)),
        "TEAM_TARGET_CATCHABLE_RATE8": float(catch.mean()) if usable else np.nan,
        "team_history_max_season": np.nan,
        "team_history_max_week": np.nan,
    }
    if len(games8):
        latest = games8.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["team_history_max_season"] = int(latest["season"])
        out["team_history_max_week"] = int(latest["week"])
    return out


def build_development_panel(authority: pd.DataFrame, targets: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    dev = authority.loc[authority["season"].eq(DEV_SEASON)].copy()
    if len(dev) != EXPECTED_ROWS[DEV_SEASON]:
        raise RuntimeError("WR-R19 development authority count drift")
    rows = []
    for r in dev.itertuples(index=False):
        key = str(r.player_clean_key)
        display_key = _name_key(r.player)
        row = {
            "season": int(r.season), "week": int(r.week), "team": str(r.team),
            "player_clean_key": key, "player": str(r.player),
            "authority_display_key_match": bool(not display_key or display_key == key),
            "wr_rank": int(r.wr_rank), "wr_rank_bucket": "WR1" if int(r.wr_rank) == 1 else "WR2PLUS",
            "pred_targets": float(r.pred_targets), "entitlement_tgt_share": float(r.entitlement_tgt_share),
            "mc_rec_yards": float(r.mc_rec_yards), "actual_rec_yards": float(r.actual_rec_yards),
            "yard_residual": float(r.yard_residual),
        }
        row.update(receiver_state(targets, rosters, key, row["team"], row["season"], row["week"]))
        row.update(team_state(targets, row["team"], row["season"], row["week"]))
        rows.append(row)
    panel = pd.DataFrame(rows)
    has_r = panel["history_max_season"].notna()
    before_r = (panel["history_max_season"] < panel["season"]) | ((panel["history_max_season"] == panel["season"]) & (panel["history_max_week"] < panel["week"]))
    if int((has_r & ~before_r).sum()):
        raise RuntimeError("WR-R19 receiver target-game leakage assertion failed")
    has_t = panel["team_history_max_season"].notna()
    before_t = (panel["team_history_max_season"] < panel["season"]) | ((panel["team_history_max_season"] == panel["season"]) & (panel["team_history_max_week"] < panel["week"]))
    if int((has_t & ~before_t).sum()):
        raise RuntimeError("WR-R19 team-control target-game leakage assertion failed")
    return panel


def _spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(z) < 3 or z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan
    return float(z["a"].rank().corr(z["b"].rank()))


def _ratio(high: float, low: float) -> float:
    if not np.isfinite(high) or not np.isfinite(low) or low < 0:
        return np.nan
    if low == 0:
        return float("inf") if high > 0 else np.nan
    return float(high / low)


def _ratio_passes(value: float) -> bool:
    return bool((np.isfinite(value) or np.isposinf(value)) and value >= MIN_TAIL_RATIO)


def raw_stage_a(panel: pd.DataFrame) -> tuple[dict, dict]:
    signal = "WR_TARGET_CATCHABLE_RATE8"
    d = panel.loc[_num(panel[signal]).notna() & _num(panel["yard_residual"]).notna()].copy()
    coverage = float(len(d) / len(panel)) if len(panel) else 0.0
    out = {
        "n": int(len(d)), "coverage": coverage, "spearman": np.nan,
        "q25": np.nan, "q75": np.nan, "q4_minus_q1_residual_gap": np.nan,
        "q4_actual100_rate": np.nan, "q1_actual100_rate": np.nan, "actual100_rate_ratio": np.nan,
        "q4_underproj30_rate": np.nan, "q1_underproj30_rate": np.nan, "underproj30_rate_ratio": np.nan,
        "wr1_n": 0, "wr1_gap": np.nan, "wr2plus_n": 0, "wr2plus_gap": np.nan,
        "supported_raw": False,
    }
    quartiles = {}
    if d.empty:
        return out, quartiles
    q25 = float(d[signal].quantile(0.25, interpolation="linear"))
    q75 = float(d[signal].quantile(0.75, interpolation="linear"))
    low = d.loc[d[signal].le(q25)]
    high = d.loc[d[signal].ge(q75)]
    rho = _spearman(d[signal], d["yard_residual"])
    gap = float(high["yard_residual"].mean() - low["yard_residual"].mean())
    high100 = float(_num(high["actual_rec_yards"]).ge(100).mean()) if len(high) else np.nan
    low100 = float(_num(low["actual_rec_yards"]).ge(100).mean()) if len(low) else np.nan
    high30 = float(_num(high["yard_residual"]).ge(30).mean()) if len(high) else np.nan
    low30 = float(_num(low["yard_residual"]).ge(30).mean()) if len(low) else np.nan
    ratio100, ratio30 = _ratio(high100, low100), _ratio(high30, low30)
    slice_vals, slice_ok = {}, True
    for bucket in ["WR1", "WR2PLUS"]:
        s = d.loc[d["wr_rank_bucket"].eq(bucket)]
        slo, shi = s.loc[s[signal].le(q25)], s.loc[s[signal].ge(q75)]
        sgap = float(shi["yard_residual"].mean() - slo["yard_residual"].mean()) if len(slo) and len(shi) else np.nan
        required = len(s) >= MIN_SLICE_N
        coherent = bool(np.isfinite(sgap) and sgap > 0) if required else True
        slice_ok &= coherent
        slice_vals[bucket] = {"n": int(len(s)), "gap": sgap}
    supported = bool(
        coverage >= MIN_COVERAGE
        and np.isfinite(rho) and rho >= MIN_SPEARMAN
        and np.isfinite(gap) and gap >= MIN_RESIDUAL_GAP
        and (_ratio_passes(ratio100) or _ratio_passes(ratio30))
        and slice_ok
    )
    out.update({
        "spearman": rho, "q25": q25, "q75": q75, "q4_minus_q1_residual_gap": gap,
        "q4_actual100_rate": high100, "q1_actual100_rate": low100, "actual100_rate_ratio": ratio100,
        "q4_underproj30_rate": high30, "q1_underproj30_rate": low30, "underproj30_rate_ratio": ratio30,
        "wr1_n": slice_vals["WR1"]["n"], "wr1_gap": slice_vals["WR1"]["gap"],
        "wr2plus_n": slice_vals["WR2PLUS"]["n"], "wr2plus_gap": slice_vals["WR2PLUS"]["gap"],
        "supported_raw": supported,
    })
    try:
        qlabels = pd.qcut(d[signal].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        qd = d.assign(_quartile=qlabels)
        for label, g in qd.groupby("_quartile", observed=True):
            quartiles[str(label)] = {
                "n": int(len(g)), "mae": float(_num(g["yard_residual"]).abs().mean()),
                "signed_bias": float(_num(g["yard_residual"]).mean()),
                "actual100_rate": float(_num(g["actual_rec_yards"]).ge(100).mean()),
                "underproj30_rate": float(_num(g["yard_residual"]).ge(30).mean()),
            }
    except Exception:
        quartiles = {}
    return out, quartiles


def mediation_robustness(panel: pd.DataFrame) -> dict:
    cols = [
        "WR_TARGET_CATCHABLE_RATE8", "TEAM_TARGET_CATCHABLE_RATE8", "mean_air_yards_per_target8",
        "entitlement_tgt_share", "wr_rank_bucket", "yard_residual",
    ]
    d = panel[cols].copy()
    for c in ["WR_TARGET_CATCHABLE_RATE8", "TEAM_TARGET_CATCHABLE_RATE8", "mean_air_yards_per_target8", "entitlement_tgt_share", "yard_residual"]:
        d[c] = _num(d[c])
    d = d.dropna()
    if len(d) < 10:
        return {"n": int(len(d)), "supported": False, "error": "insufficient_complete_rows"}
    y = d["WR_TARGET_CATCHABLE_RATE8"].to_numpy(float)
    wr1 = d["wr_rank_bucket"].eq("WR1").astype(float).to_numpy()
    X = np.column_stack([
        np.ones(len(d)), d["TEAM_TARGET_CATCHABLE_RATE8"].to_numpy(float),
        d["mean_air_yards_per_target8"].to_numpy(float), d["entitlement_tgt_share"].to_numpy(float), wr1,
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    d = d.assign(receiver_specific_catchability=residual)
    rho = _spearman(d["receiver_specific_catchability"], d["yard_residual"])
    q25 = float(d["receiver_specific_catchability"].quantile(0.25, interpolation="linear"))
    q75 = float(d["receiver_specific_catchability"].quantile(0.75, interpolation="linear"))
    low = d.loc[d["receiver_specific_catchability"].le(q25)]
    high = d.loc[d["receiver_specific_catchability"].ge(q75)]
    gap = float(high["yard_residual"].mean() - low["yard_residual"].mean())
    supported = bool(np.isfinite(rho) and rho >= MEDIATION_MIN_SPEARMAN and np.isfinite(gap) and gap >= MEDIATION_MIN_RESIDUAL_GAP)
    return {
        "n": int(len(d)),
        "coefficients": {
            "intercept": float(beta[0]), "TEAM_TARGET_CATCHABLE_RATE8": float(beta[1]),
            "mean_air_yards_per_target8": float(beta[2]), "entitlement_tgt_share": float(beta[3]),
            "wr1_indicator": float(beta[4]),
        },
        "receiver_specific_spearman": rho, "receiver_specific_q25": q25, "receiver_specific_q75": q75,
        "receiver_specific_q4_minus_q1_residual_gap": gap, "supported": supported,
        "interpretation_caveat": "A failed control check may reflect confounding or over-control of a real delivery pathway; it is not causal disproof. Team-level catchability blends QB changes and is a known V1 limitation.",
    }


def identity_audit(panel: pd.DataFrame, source_meta: list[dict]) -> dict:
    return {
        "development_rows": int(len(panel)), "expected_development_rows": EXPECTED_ROWS[DEV_SEASON],
        "identity_mode_counts": {str(k): int(v) for k, v in panel["identity_mode"].value_counts(dropna=False).to_dict().items()},
        "identity_source_counts": {str(k): int(v) for k, v in panel["identity_source"].value_counts(dropna=False).to_dict().items()},
        "rows_team_disambiguated": int(panel["roster_team_disambiguated"].fillna(False).sum()),
        "rows_with_4plus_prior_target_games": int(panel["prior_target_games"].ge(MIN_PRIOR_TARGET_GAMES).sum()),
        "rows_with_16plus_valid_catchable_targets": int(panel["prior_valid_catchable_targets"].ge(MIN_RECEIVER_TARGETS).sum()),
        "rows_with_valid_catchability_signal": int(panel["WR_TARGET_CATCHABLE_RATE8"].notna().sum()),
        "team_control_available_rows": int(panel["TEAM_TARGET_CATCHABLE_RATE8"].notna().sum()),
        "authority_display_key_mismatch_rows": int((~panel["authority_display_key_match"]).sum()),
        "target_game_leakage_rows": 0, "team_control_target_game_leakage_rows": 0,
        "sportsbook_inputs": 0, "holdout_2024_scored": False,
        "source_meta": source_meta,
    }


def target_population_audit(targets: pd.DataFrame, panel: pd.DataFrame) -> dict:
    ids = sorted({str(v) for v in panel["resolved_receiver_id"].dropna().astype(str) if str(v)})
    x = targets.loc[targets["receiver_id"].isin(ids)].drop_duplicates(["season", "game_id", "target_event_seq"]).copy()
    return {
        "resolved_receiver_target_events": int(len(x)),
        "catchable_nonnull_events": int(x["catchable"].notna().sum()),
        "catchable_null_events": int(x["catchable"].isna().sum()),
        "completed_target_events": int(x["complete_pass_num"].eq(1).sum()),
        "incomplete_target_events": int(x["complete_pass_num"].eq(0).sum()),
        "both_completions_and_incompletions_present": bool(x["complete_pass_num"].eq(1).any() and x["complete_pass_num"].eq(0).any()),
        "selection_note": "Catchability is computed on all official receiver target events, including completions and incompletions; no reception-only filter exists.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    authority = load_authority(args.authority)
    targets, source_meta = load_ftn_pbp_targets([2022, 2023])
    rosters = load_roster_identity([2022, 2023])
    panel = build_development_panel(authority, targets, rosters)
    raw_metrics, quartiles = raw_stage_a(panel)
    data_blocked = bool(raw_metrics["coverage"] < MIN_COVERAGE)
    mediation = None
    if data_blocked:
        disposition = "WR_RECEIVER_CATCHABILITY_DATA_BLOCKED"
    elif not raw_metrics["supported_raw"]:
        disposition = "NO_ACTIONABLE_WR_RECEIVER_CATCHABILITY_SIGNAL"
    else:
        mediation = mediation_robustness(panel)
        disposition = "WR_RECEIVER_CATCHABILITY_DEVELOPMENT_SUPPORTED" if mediation.get("supported", False) else "WR_CATCHABILITY_TEAM_DEPTH_ROLE_MEDIATED"

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "wr_r19_stage_a_feature_panel_2023.csv", index=False)
    (out / "wr_r19_stage_a_raw_metrics_2023.json").write_text(json.dumps(raw_metrics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    (out / "wr_r19_stage_a_quartile_descriptives_2023.json").write_text(json.dumps(quartiles, indent=2, sort_keys=True, allow_nan=True) + "\n")
    (out / "wr_r19_stage_a_identity_source_audit.json").write_text(json.dumps(identity_audit(panel, source_meta), indent=2, sort_keys=True, allow_nan=True) + "\n")
    (out / "wr_r19_stage_a_target_population_audit.json").write_text(json.dumps(target_population_audit(targets, panel), indent=2, sort_keys=True, allow_nan=True) + "\n")
    if mediation is not None:
        (out / "wr_r19_stage_a_mediation_robustness.json").write_text(json.dumps(mediation, indent=2, sort_keys=True, allow_nan=True) + "\n")
    result = {
        "specification": "WR_R19_RECEIVER_CATCHABILITY_V1", "stage": "A_2023_DEVELOPMENT_ONLY",
        "disposition": disposition, "raw_development_supported": bool(raw_metrics["supported_raw"]),
        "mediation_supported": bool(mediation and mediation.get("supported", False)),
        "holdout_2024_scored": False, "authority_expected_rows": EXPECTED_ROWS,
        "signal": "WR_TARGET_CATCHABLE_RATE8", "frozen_direction": "positive",
        "production_change": False, "sportsbook_inputs": 0,
    }
    (out / "wr_r19_stage_a_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    print(pd.DataFrame([raw_metrics]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
