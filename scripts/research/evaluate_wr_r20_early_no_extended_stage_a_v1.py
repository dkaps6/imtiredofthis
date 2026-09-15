#!/usr/bin/env python3
"""WR-R20 Stage A mechanics: receiver early/no-extended progression share.

Research only. Implements the frozen 2023 development design from
WR_R20_EARLY_NO_EXTENDED_V1_PLAN.md. This module is importable by synthetic
mechanics tests without loading any WR outcomes. The CLI scores 2023 only;
2024 authority projection/outcome cells are counted/key-audited but never
parsed or materialized.
"""
from __future__ import annotations

import argparse
import csv
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
MIN_CLASSIFIABLE_TARGETS = 16
MIN_TEAM_CLASSIFIABLE_TARGETS = 40
MIN_AIR_TARGETS = 12
MIN_JOIN_RATE = 0.95
MIN_COVERAGE = 0.60
MAX_SPEARMAN = -0.08
MAX_RESIDUAL_GAP = -5.0
MIN_TAIL_RATIO = 1.20
MIN_SLICE_N = 150
MEDIATION_MAX_SPEARMAN = -0.06
MEDIATION_MAX_RESIDUAL_GAP = -4.0


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


def _first(frame: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series(default, index=frame.index)


def _read_parquet(url: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": "wr-r20-early-no-extended-v1"})
    with urlopen(req, timeout=180) as response:
        raw = response.read()
        final = response.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def normalize_progression(value, season: int) -> str:
    """Frozen V2 progression semantics; never relabel from outcomes."""
    if value is None or pd.isna(value):
        return "EARLY_NO_EXTENDED" if int(season) == 2022 else "MISSING"
    z = str(value).strip().upper()
    if z in {"0", "0.0", "DES"}:
        return "EARLY_NO_EXTENDED"
    if z in {"1", "1.0", "2", "2.0"}:
        return "EXTENDED_PROGRESS"
    if z == "CHK":
        return "CHECKDOWN"
    if z == "SD":
        return "SCRAMBLE_DRILL"
    return f"UNKNOWN:{z}"


def load_authority_development_only(path: Path) -> pd.DataFrame:
    """Count/key-audit 2023+2024 authority rows; materialize only 2023 outcomes."""
    required = {
        "variant", "team", "player_clean_key", "player", "wr_rank", "pred_targets",
        "entitlement_tgt_share", "mc_rec_yards", "season", "week", "actual_rec_yards",
    }
    counts: dict[int, int] = {}
    seen: set[tuple[int, int, str, str]] = set()
    dev_rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"WR-R20 authority missing columns: {missing}")
        for raw in reader:
            if str(raw.get("variant", "")) != AUTHORITY_VARIANT:
                continue
            season = int(float(raw["season"]))
            week = int(float(raw["week"]))
            team = canon_team(raw["team"])
            key = str(raw["player_clean_key"])
            counts[season] = counts.get(season, 0) + 1
            ident = (season, week, team, key)
            if ident in seen:
                raise RuntimeError(f"WR-R20 duplicate authority identity: {ident}")
            seen.add(ident)
            if season != DEV_SEASON:
                # Holdout seal: do not parse any 2024 projection/outcome values.
                continue
            dev_rows.append({
                "variant": raw["variant"],
                "team": team,
                "player_clean_key": key,
                "player": str(raw["player"]),
                "wr_rank": int(float(raw["wr_rank"])),
                "pred_targets": float(raw["pred_targets"]),
                "entitlement_tgt_share": float(raw["entitlement_tgt_share"]),
                "mc_rec_yards": float(raw["mc_rec_yards"]),
                "season": season,
                "week": week,
                "actual_rec_yards": float(raw["actual_rec_yards"]),
            })
    if counts != EXPECTED_ROWS:
        raise RuntimeError(f"WR-R20 authority row-count parity failed: {counts} != {EXPECTED_ROWS}")
    x = pd.DataFrame(dev_rows)
    if len(x) != EXPECTED_ROWS[DEV_SEASON]:
        raise RuntimeError(f"WR-R20 development materialization count drift: {len(x)}")
    x["yard_residual"] = _num(x["actual_rec_yards"]) - _num(x["mc_rec_yards"])
    return x.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True)


def load_roster_identity(seasons: Iterable[int]) -> pd.DataFrame:
    """Identity-only weekly roster source. Target-week evidence is forbidden."""
    import nflreadpy as nfl

    frames = []
    for season in sorted({int(s) for s in seasons}):
        raw = nfl.load_rosters_weekly(season)
        x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        if x.empty:
            raise RuntimeError(f"WR-R20 weekly roster source returned zero rows for {season}")
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


def _merge_ftn_pbp_progression_targets(
    ftn: pd.DataFrame, pbp: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, float]:
    """Pure exact-play merge used by both real source loading and synthetics."""
    ftn = ftn.copy()
    pbp = pbp.copy()
    ftn.columns = [str(c).strip().lower() for c in ftn.columns]
    pbp.columns = [str(c).strip().lower() for c in pbp.columns]
    req_f = {"nflverse_game_id", "nflverse_play_id", "season", "week", "read_thrown"}
    req_p = {
        "game_id", "play_id", "season", "week", "season_type", "posteam",
        "pass_attempt", "sack", "two_point_attempt", "receiver_player_id",
        "receiver_player_name", "air_yards",
    }
    miss_f = sorted(req_f - set(ftn.columns))
    miss_p = sorted(req_p - set(pbp.columns))
    if miss_f or miss_p:
        raise RuntimeError(f"WR-R20 source schema missing season={season}: ftn={miss_f}, pbp={miss_p}")
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
        raise RuntimeError(f"WR-R20 duplicate FTN game/play key season={season}")
    if pbp.duplicated(["join_game", "join_play"]).any():
        raise RuntimeError(f"WR-R20 duplicate PBP game/play key season={season}")
    pcols = [
        "join_game", "join_play", "game_id", "season", "week", "posteam",
        "pass_attempt", "sack", "two_point_attempt", "receiver_player_id",
        "receiver_player_name", "air_yards",
    ]
    p = pbp[pcols].rename(columns={
        "game_id": "pbp_game_id", "season": "pbp_season", "week": "pbp_week",
        "posteam": "pbp_posteam", "pass_attempt": "pbp_pass_attempt",
        "sack": "pbp_sack", "two_point_attempt": "pbp_two_point_attempt",
        "receiver_player_id": "pbp_receiver_player_id",
        "receiver_player_name": "pbp_receiver_player_name",
        "air_yards": "pbp_air_yards",
    })
    m = ftn.merge(p, on=["join_game", "join_play"], how="left", validate="one_to_one", indicator=True)
    matched = m["_merge"].eq("both")
    join_rate = float(matched.mean()) if len(m) else 0.0
    if join_rate < MIN_JOIN_RATE:
        raise RuntimeError(f"WR-R20 FTN/PBP exact join below source contract season={season}: {join_rate}")
    if matched.any():
        parity = _num(m.loc[matched, "season"]).eq(_num(m.loc[matched, "pbp_season"])) & _num(
            m.loc[matched, "week"]
        ).eq(_num(m.loc[matched, "pbp_week"]))
        if not bool(parity.all()):
            bad = m.loc[matched].loc[
                ~parity, ["join_game", "join_play", "season", "week", "pbp_season", "pbp_week"]
            ].head(10)
            raise RuntimeError(f"WR-R20 FTN/PBP season-week parity failed season={season}: {bad.to_dict('records')}")
    official = _num(m["pbp_pass_attempt"]).fillna(0).eq(1)
    official &= ~_num(m["pbp_sack"]).fillna(0).eq(1)
    official &= ~_num(m["pbp_two_point_attempt"]).fillna(0).eq(1)
    rid = m["pbp_receiver_player_id"].map(_clean_id)
    t = m.loc[matched & official & rid.ne("")].copy()
    t["game_id"] = t["pbp_game_id"].astype(str)
    t["receiver_id"] = t["pbp_receiver_player_id"].map(_clean_id)
    t["receiver_name_key"] = t["pbp_receiver_player_name"].map(_name_key)
    t["team"] = t["pbp_posteam"].map(canon_team)
    t["season"] = _num(t["season"]).astype(int)
    t["week"] = _num(t["week"]).astype(int)
    t["read_norm"] = [normalize_progression(v, int(s)) for v, s in zip(t["read_thrown"], t["season"])]
    t["progression_classifiable"] = t["read_norm"].isin(["EARLY_NO_EXTENDED", "EXTENDED_PROGRESS"])
    t["is_early_no_extended"] = t["read_norm"].eq("EARLY_NO_EXTENDED").astype(float)
    t["target_event_seq"] = np.arange(len(t), dtype=np.int64)
    return t[[
        "season", "week", "game_id", "team", "receiver_id", "receiver_name_key",
        "read_norm", "progression_classifiable", "is_early_no_extended", "target_event_seq",
    ]].reset_index(drop=True), join_rate


def load_progression_targets(seasons: Iterable[int]) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    meta: list[dict] = []
    for season in sorted({int(s) for s in seasons}):
        ftn_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        ftn, fm = _read_parquet(ftn_url)
        pbp, pm = _read_parquet(pbp_url)
        t, join_rate = _merge_ftn_pbp_progression_targets(ftn, pbp, season)
        counts = t["read_norm"].value_counts(dropna=False).to_dict()
        unknown = int(t["read_norm"].astype(str).str.startswith("UNKNOWN:").sum())
        if unknown:
            raise RuntimeError(f"WR-R20 unknown progression codes season={season}: {counts}")
        classifiable = int(t["progression_classifiable"].sum())
        frames.append(t)
        meta.append({
            "season": season,
            "ftn_sha256": fm["sha256"], "ftn_bytes": fm["bytes"],
            "pbp_sha256": pm["sha256"], "pbp_bytes": pm["bytes"],
            "exact_join_rate": join_rate, "season_week_parity": True,
            "receiver_target_rows": int(len(t)),
            "normalized_category_counts": {str(k): int(v) for k, v in counts.items()},
            "classifiable_progression_rows": classifiable,
            "classifiable_progression_rate": float(classifiable / len(t)) if len(t) else 0.0,
            "unknown_code_rows": unknown,
        })
    if not frames:
        raise RuntimeError("WR-R20 progression source returned zero rows")
    return pd.concat(frames, ignore_index=True, sort=False), meta


def load_pbp_air_targets(seasons: Iterable[int]) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    meta = []
    for season in sorted({int(s) for s in seasons}):
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        pbp, pm = _read_parquet(url)
        pbp.columns = [str(c).strip().lower() for c in pbp.columns]
        req = {
            "game_id", "season", "week", "season_type", "posteam", "pass_attempt", "sack",
            "two_point_attempt", "receiver_player_id", "air_yards",
        }
        missing = sorted(req - set(pbp.columns))
        if missing:
            raise RuntimeError(f"WR-R20 PBP air source missing season={season}: {missing}")
        x = pbp.loc[
            pbp["season_type"].astype(str).str.upper().eq("REG")
            & _num(pbp["week"]).between(1, 18)
            & _num(pbp["pass_attempt"]).fillna(0).eq(1)
            & ~_num(pbp["sack"]).fillna(0).eq(1)
            & ~_num(pbp["two_point_attempt"]).fillna(0).eq(1)
        ].copy()
        x["receiver_id"] = x["receiver_player_id"].map(_clean_id)
        x = x.loc[x["receiver_id"].ne("")].copy()
        x["season"] = _num(x["season"]).astype(int)
        x["week"] = _num(x["week"]).astype(int)
        x["game_id"] = x["game_id"].astype(str)
        x["team"] = x["posteam"].map(canon_team)
        x["air_yards"] = _num(x["air_yards"])
        frames.append(x[["season", "week", "game_id", "team", "receiver_id", "air_yards"]])
        meta.append({
            "season": season, "pbp_sha256": pm["sha256"], "target_rows": int(len(x)),
            "nonnull_air_rows": int(x["air_yards"].notna().sum()),
        })
    return pd.concat(frames, ignore_index=True, sort=False), meta


def _prior(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return frame.loc[
        (frame["season"] < int(season))
        | ((frame["season"] == int(season)) & (frame["week"] < int(week)))
    ].copy()


def _last_games(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    games = frame[["season", "week", "game_id"]].drop_duplicates().sort_values(
        ["season", "week", "game_id"], kind="mergesort"
    ).tail(int(n))
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
        return prior_targets.iloc[0:0].copy(), {
            **roster, "identity_mode": "ambiguous", "identity_source": "weekly_roster", "resolved_receiver_id": "",
        }
    if roster["roster_identity_mode"] == "id":
        rid = roster["roster_player_id"]
        return prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy(), {
            **roster, "identity_mode": "id", "identity_source": "weekly_roster", "resolved_receiver_id": rid,
        }
    alias = prior_targets.loc[prior_targets["receiver_name_key"].eq(str(name_key))].copy()
    ids = sorted({_clean_id(v) for v in alias["receiver_id"] if _clean_id(v)})
    if len(ids) > 1:
        return alias.iloc[0:0].copy(), {
            **roster, "identity_mode": "ambiguous", "identity_source": "pbp_exact_name_fallback", "resolved_receiver_id": "",
        }
    if len(ids) == 1:
        rid = ids[0]
        return prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy(), {
            **roster, "identity_mode": "id", "identity_source": "pbp_exact_name_fallback", "resolved_receiver_id": rid,
        }
    return alias.iloc[0:0].copy(), {
        **roster, "identity_mode": "unmatched", "identity_source": "pbp_exact_name_fallback", "resolved_receiver_id": "",
    }


def receiver_state(
    targets: pd.DataFrame, rosters: pd.DataFrame, name_key: str, team: str, season: int, week: int
) -> dict:
    history, audit = resolve_prior_receiver_history(targets, rosters, name_key, team, season, week)
    selected = _last_games(history, PRIOR_GAMES)
    games = selected[["season", "week", "game_id"]].drop_duplicates() if len(selected) else pd.DataFrame()
    c = selected.loc[selected["progression_classifiable"]].copy() if len(selected) else selected.copy()
    out = {
        **audit,
        "prior_target_games": int(len(games)),
        "prior_target_events": int(len(selected)),
        "prior_classifiable_progression_targets": int(len(c)),
        "prior_checkdown_targets": int(selected["read_norm"].eq("CHECKDOWN").sum()) if len(selected) else 0,
        "prior_scramble_drill_targets": int(selected["read_norm"].eq("SCRAMBLE_DRILL").sum()) if len(selected) else 0,
        "prior_missing_progression_targets": int(selected["read_norm"].eq("MISSING").sum()) if len(selected) else 0,
        "EARLY_NO_EXTENDED_SHARE8": np.nan,
        "history_max_season": np.nan,
        "history_max_week": np.nan,
    }
    if len(games):
        latest = games.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["history_max_season"] = int(latest["season"])
        out["history_max_week"] = int(latest["week"])
    if audit["identity_mode"] in {"ambiguous", "unmatched"}:
        return out
    if len(games) < MIN_PRIOR_TARGET_GAMES or len(c) < MIN_CLASSIFIABLE_TARGETS:
        return out
    out["EARLY_NO_EXTENDED_SHARE8"] = float(c["is_early_no_extended"].mean())
    return out


def team_state(targets: pd.DataFrame, team: str, season: int, week: int) -> dict:
    prior = _prior(targets.loc[targets["team"].eq(str(team))], season, week)
    selected = _last_games(prior, PRIOR_GAMES)
    games = selected[["season", "week", "game_id"]].drop_duplicates() if len(selected) else pd.DataFrame()
    c = selected.loc[selected["progression_classifiable"]].copy() if len(selected) else selected.copy()
    usable = len(c) >= MIN_TEAM_CLASSIFIABLE_TARGETS
    out = {
        "team_prior_target_games": int(len(games)),
        "team_prior_target_events": int(len(selected)),
        "team_classifiable_progression_targets": int(len(c)),
        "TEAM_EARLY_NO_EXTENDED_SHARE8": float(c["is_early_no_extended"].mean()) if usable else np.nan,
        "team_history_max_season": np.nan,
        "team_history_max_week": np.nan,
    }
    if len(games):
        latest = games.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["team_history_max_season"] = int(latest["season"])
        out["team_history_max_week"] = int(latest["week"])
    return out


def air_state(air: pd.DataFrame, receiver_id: str, season: int, week: int) -> dict:
    prior = _prior(air.loc[air["receiver_id"].eq(str(receiver_id))], season, week) if receiver_id else air.iloc[0:0].copy()
    selected = _last_games(prior, PRIOR_GAMES)
    games = selected[["season", "week", "game_id"]].drop_duplicates() if len(selected) else pd.DataFrame()
    valid = _num(selected["air_yards"]).dropna() if len(selected) else pd.Series(dtype=float)
    out = {
        "air_prior_target_games": int(len(games)),
        "valid_air_targets": int(len(valid)),
        "MEAN_AIR_YARDS_PER_TARGET8": float(valid.mean()) if len(valid) >= MIN_AIR_TARGETS else np.nan,
        "air_history_max_season": np.nan,
        "air_history_max_week": np.nan,
    }
    if len(games):
        latest = games.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["air_history_max_season"] = int(latest["season"])
        out["air_history_max_week"] = int(latest["week"])
    return out


def build_development_panel(
    authority: pd.DataFrame, targets: pd.DataFrame, rosters: pd.DataFrame, air: pd.DataFrame
) -> pd.DataFrame:
    if len(authority) != EXPECTED_ROWS[DEV_SEASON] or not authority["season"].eq(DEV_SEASON).all():
        raise RuntimeError("WR-R20 development authority count/seal drift")
    rows = []
    for r in authority.itertuples(index=False):
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
        row.update(air_state(air, row.get("resolved_receiver_id", ""), row["season"], row["week"]))
        rows.append(row)
    panel = pd.DataFrame(rows)
    for prefix in ["history", "team_history", "air_history"]:
        s_col, w_col = f"{prefix}_max_season", f"{prefix}_max_week"
        has = panel[s_col].notna()
        before = (panel[s_col] < panel["season"]) | ((panel[s_col] == panel["season"]) & (panel[w_col] < panel["week"]))
        if int((has & ~before).sum()):
            raise RuntimeError(f"WR-R20 temporal leakage assertion failed for {prefix}")
    return panel


def _spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(z) < 3 or z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan
    return float(z["a"].rank().corr(z["b"].rank()))


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 0:
        return np.nan
    if denominator == 0:
        return float("inf") if numerator > 0 else np.nan
    return float(numerator / denominator)


def _ratio_passes(value: float) -> bool:
    return bool((np.isfinite(value) or np.isposinf(value)) and value >= MIN_TAIL_RATIO)


def raw_stage_a(panel: pd.DataFrame) -> tuple[dict, dict]:
    signal = "EARLY_NO_EXTENDED_SHARE8"
    d = panel.loc[_num(panel[signal]).notna() & _num(panel["yard_residual"]).notna()].copy()
    coverage = float(len(d) / len(panel)) if len(panel) else 0.0
    out = {
        "n": int(len(d)), "coverage": coverage, "spearman": np.nan,
        "q25": np.nan, "q75": np.nan, "q4_minus_q1_residual_gap": np.nan,
        "q1_actual100_rate": np.nan, "q4_actual100_rate": np.nan, "actual100_q1_over_q4_ratio": np.nan,
        "q4_residual_le_minus30_rate": np.nan, "q1_residual_le_minus30_rate": np.nan,
        "negative30_q4_over_q1_ratio": np.nan,
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
    low100 = float(_num(low["actual_rec_yards"]).ge(100).mean()) if len(low) else np.nan
    high100 = float(_num(high["actual_rec_yards"]).ge(100).mean()) if len(high) else np.nan
    high_neg30 = float(_num(high["yard_residual"]).le(-30).mean()) if len(high) else np.nan
    low_neg30 = float(_num(low["yard_residual"]).le(-30).mean()) if len(low) else np.nan
    ratio100 = _ratio(low100, high100)
    ratio30 = _ratio(high_neg30, low_neg30)
    slice_vals, slice_ok = {}, True
    for bucket in ["WR1", "WR2PLUS"]:
        s = d.loc[d["wr_rank_bucket"].eq(bucket)]
        slo, shi = s.loc[s[signal].le(q25)], s.loc[s[signal].ge(q75)]
        sgap = float(shi["yard_residual"].mean() - slo["yard_residual"].mean()) if len(slo) and len(shi) else np.nan
        required = len(s) >= MIN_SLICE_N
        coherent = bool(np.isfinite(sgap) and sgap < 0) if required else True
        slice_ok &= coherent
        slice_vals[bucket] = {"n": int(len(s)), "gap": sgap}
    supported = bool(
        coverage >= MIN_COVERAGE
        and np.isfinite(rho) and rho <= MAX_SPEARMAN
        and np.isfinite(gap) and gap <= MAX_RESIDUAL_GAP
        and (_ratio_passes(ratio100) or _ratio_passes(ratio30))
        and slice_ok
    )
    out.update({
        "spearman": rho, "q25": q25, "q75": q75, "q4_minus_q1_residual_gap": gap,
        "q1_actual100_rate": low100, "q4_actual100_rate": high100, "actual100_q1_over_q4_ratio": ratio100,
        "q4_residual_le_minus30_rate": high_neg30, "q1_residual_le_minus30_rate": low_neg30,
        "negative30_q4_over_q1_ratio": ratio30,
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
                "residual_le_minus30_rate": float(_num(g["yard_residual"]).le(-30).mean()),
            }
    except Exception:
        quartiles = {}
    return out, quartiles


def mediation_robustness(panel: pd.DataFrame) -> dict:
    cols = [
        "EARLY_NO_EXTENDED_SHARE8", "TEAM_EARLY_NO_EXTENDED_SHARE8", "MEAN_AIR_YARDS_PER_TARGET8",
        "entitlement_tgt_share", "wr_rank_bucket", "yard_residual",
    ]
    d = panel[cols].copy()
    for c in [
        "EARLY_NO_EXTENDED_SHARE8", "TEAM_EARLY_NO_EXTENDED_SHARE8", "MEAN_AIR_YARDS_PER_TARGET8",
        "entitlement_tgt_share", "yard_residual",
    ]:
        d[c] = _num(d[c])
    d = d.dropna()
    if len(d) < 10:
        return {"n": int(len(d)), "supported": False, "error": "insufficient_complete_rows"}
    y = d["EARLY_NO_EXTENDED_SHARE8"].to_numpy(float)
    wr1 = d["wr_rank_bucket"].eq("WR1").astype(float).to_numpy()
    X = np.column_stack([
        np.ones(len(d)), d["TEAM_EARLY_NO_EXTENDED_SHARE8"].to_numpy(float),
        d["MEAN_AIR_YARDS_PER_TARGET8"].to_numpy(float), d["entitlement_tgt_share"].to_numpy(float), wr1,
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    d = d.assign(receiver_specific_early_no_extended=residual)
    rho = _spearman(d["receiver_specific_early_no_extended"], d["yard_residual"])
    q25 = float(d["receiver_specific_early_no_extended"].quantile(0.25, interpolation="linear"))
    q75 = float(d["receiver_specific_early_no_extended"].quantile(0.75, interpolation="linear"))
    low = d.loc[d["receiver_specific_early_no_extended"].le(q25)]
    high = d.loc[d["receiver_specific_early_no_extended"].ge(q75)]
    gap = float(high["yard_residual"].mean() - low["yard_residual"].mean())
    supported = bool(
        np.isfinite(rho) and rho <= MEDIATION_MAX_SPEARMAN
        and np.isfinite(gap) and gap <= MEDIATION_MAX_RESIDUAL_GAP
    )
    return {
        "n": int(len(d)),
        "coefficients": {
            "intercept": float(beta[0]),
            "TEAM_EARLY_NO_EXTENDED_SHARE8": float(beta[1]),
            "MEAN_AIR_YARDS_PER_TARGET8": float(beta[2]),
            "entitlement_tgt_share": float(beta[3]),
            "wr1_indicator": float(beta[4]),
        },
        "receiver_specific_spearman": rho,
        "receiver_specific_q25": q25,
        "receiver_specific_q75": q75,
        "receiver_specific_q4_minus_q1_residual_gap": gap,
        "supported": supported,
        "interpretation_caveat": (
            "A failed robustness gate may reflect genuine confounding or over-control of a real pathway; "
            "it is not causal disproof, but it blocks R20 holdout exposure."
        ),
    }


def identity_source_audit(panel: pd.DataFrame, source_meta: list[dict], air_meta: list[dict]) -> dict:
    supported = panel["EARLY_NO_EXTENDED_SHARE8"].notna()
    return {
        "development_rows": int(len(panel)), "expected_development_rows": EXPECTED_ROWS[DEV_SEASON],
        "identity_mode_counts": {str(k): int(v) for k, v in panel["identity_mode"].value_counts(dropna=False).to_dict().items()},
        "identity_source_counts": {str(k): int(v) for k, v in panel["identity_source"].value_counts(dropna=False).to_dict().items()},
        "rows_team_disambiguated": int(panel["roster_team_disambiguated"].fillna(False).sum()),
        "rows_with_4plus_prior_target_games": int(panel["prior_target_games"].ge(MIN_PRIOR_TARGET_GAMES).sum()),
        "rows_with_16plus_classifiable_targets": int(panel["prior_classifiable_progression_targets"].ge(MIN_CLASSIFIABLE_TARGETS).sum()),
        "rows_with_valid_primary_signal": int(supported.sum()),
        "team_control_available_rows": int(panel["TEAM_EARLY_NO_EXTENDED_SHARE8"].notna().sum()),
        "air_control_available_rows": int(panel["MEAN_AIR_YARDS_PER_TARGET8"].notna().sum()),
        "authority_display_key_mismatch_rows": int((~panel["authority_display_key_match"]).sum()),
        "selected_checkdown_targets": int(panel["prior_checkdown_targets"].sum()),
        "selected_scramble_drill_targets": int(panel["prior_scramble_drill_targets"].sum()),
        "selected_missing_progression_targets": int(panel["prior_missing_progression_targets"].sum()),
        "target_game_leakage_rows": 0,
        "team_control_target_game_leakage_rows": 0,
        "air_control_target_game_leakage_rows": 0,
        "sportsbook_inputs": 0,
        "holdout_2024_scored": False,
        "holdout_2024_projection_or_outcome_fields_parsed": False,
        "source_meta": source_meta,
        "air_meta": air_meta,
        "semantic_note": (
            "CHK and SD are excluded from the binary denominator because they are separately charted states that do not "
            "cleanly identify numbered progression position; this choice was frozen before outcomes."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    authority = load_authority_development_only(args.authority)
    targets, source_meta = load_progression_targets([2022, 2023])
    rosters = load_roster_identity([2022, 2023])
    air, air_meta = load_pbp_air_targets([2022, 2023])
    panel = build_development_panel(authority, targets, rosters, air)
    raw_metrics, quartiles = raw_stage_a(panel)
    mediation = None
    if not raw_metrics["supported_raw"]:
        disposition = "NO_ACTIONABLE_WR_EARLY_NO_EXTENDED_SIGNAL"
    else:
        mediation = mediation_robustness(panel)
        disposition = (
            "WR_EARLY_NO_EXTENDED_DEVELOPMENT_SUPPORTED"
            if mediation.get("supported", False)
            else "WR_EARLY_NO_EXTENDED_TEAM_DEPTH_ROLE_MEDIATED"
        )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "wr_r20_stage_a_feature_panel_2023.csv", index=False)
    (out / "wr_r20_stage_a_raw_metrics_2023.json").write_text(
        json.dumps(raw_metrics, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (out / "wr_r20_stage_a_quartile_descriptives_2023.json").write_text(
        json.dumps(quartiles, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (out / "wr_r20_stage_a_identity_source_audit.json").write_text(
        json.dumps(identity_source_audit(panel, source_meta, air_meta), indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    if mediation is not None:
        (out / "wr_r20_stage_a_mediation_robustness.json").write_text(
            json.dumps(mediation, indent=2, sort_keys=True, allow_nan=True) + "\n"
        )
    result = {
        "specification": "WR_R20_EARLY_NO_EXTENDED_V1",
        "stage": "A_2023_DEVELOPMENT_ONLY",
        "disposition": disposition,
        "raw_development_supported": bool(raw_metrics["supported_raw"]),
        "mediation_evaluated": bool(mediation is not None),
        "mediation_supported": bool(mediation and mediation.get("supported", False)),
        "holdout_2024_scored": False,
        "holdout_2024_projection_or_outcome_fields_parsed": False,
        "authority_expected_rows": EXPECTED_ROWS,
        "signal": "EARLY_NO_EXTENDED_SHARE8",
        "frozen_direction": "negative",
        "production_change": False,
        "sportsbook_inputs": 0,
    }
    (out / "wr_r20_stage_a_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    print(pd.DataFrame([raw_metrics]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
