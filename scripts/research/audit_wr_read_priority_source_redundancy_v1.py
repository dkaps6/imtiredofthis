#!/usr/bin/env python3
"""Source-only post-R19 audit for receiver read priority.

No WR outcomes are loaded or scored. The exact WR-R15 artifact is used only for
2023 identity/opportunity fields (entitlement_tgt_share, pred_targets, wr_rank).
2024 authority rows are counted/key-audited but no 2024 projection/outcome field
is parsed or materialized.
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
MIN_RECEIVER_TARGETS = 16
MIN_JOIN_RATE = 0.95
MIN_INTERPRETABLE = 0.95
MIN_COVERAGE = 0.60
MAX_ABS_ENTITLEMENT_SPEARMAN = 0.75
MAX_REDUNDANCY_R2 = 0.60


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
    req = Request(url, headers={"User-Agent": "wr-read-priority-source-audit-v1"})
    with urlopen(req, timeout=180) as r:
        raw = r.read()
        final = r.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_authority_opportunity_only(path: Path) -> tuple[pd.DataFrame, dict]:
    """Stream exact authority; materialize only 2023 non-outcome fields."""
    required = {
        "variant", "season", "week", "team", "player_clean_key", "player",
        "wr_rank", "pred_targets", "entitlement_tgt_share",
    }
    counts: dict[int, int] = {}
    seen: set[tuple[int, int, str, str]] = set()
    dev_rows: list[dict] = []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"authority missing required non-outcome columns: {missing}")
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
                raise RuntimeError(f"duplicate authority identity: {ident}")
            seen.add(ident)

            # Holdout protection: do not parse any 2024 projection/outcome field.
            if season != DEV_SEASON:
                continue
            dev_rows.append({
                "season": season,
                "week": week,
                "team": team,
                "player_clean_key": key,
                "player": str(raw["player"]),
                "wr_rank": int(float(raw["wr_rank"])),
                "pred_targets": float(raw["pred_targets"]),
                "entitlement_tgt_share": float(raw["entitlement_tgt_share"]),
            })

    if counts != EXPECTED_ROWS:
        raise RuntimeError(f"authority row-count parity failed: {counts} != {EXPECTED_ROWS}")
    x = pd.DataFrame(dev_rows)
    if len(x) != EXPECTED_ROWS[DEV_SEASON]:
        raise RuntimeError(f"development materialization count drift: {len(x)}")
    return x.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True), {
        "authority_expected_rows": EXPECTED_ROWS,
        "authority_materialized_seasons": [DEV_SEASON],
        "outcome_fields_parsed": False,
        "holdout_2024_projection_or_outcome_fields_parsed": False,
    }


def load_roster_identity(seasons: Iterable[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    frames = []
    for season in sorted({int(s) for s in seasons}):
        raw = nfl.load_rosters_weekly(season)
        x = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        if x.empty:
            raise RuntimeError(f"weekly roster source returned zero rows for {season}")
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


def normalize_read(value, season: int) -> str:
    """Frozen official nflverse/FTN semantics for read_thrown."""
    if value is None or pd.isna(value):
        return "PRIMARY_READ" if int(season) == 2022 else "MISSING"
    z = str(value).strip().upper()
    if z in {"0", "0.0"}:
        return "PRIMARY_READ"
    if z in {"1", "1.0"}:
        return "SECOND_READ"
    if z in {"2", "2.0"}:
        return "THIRD_PLUS"
    if z == "CHK":
        return "CHECKDOWN"
    if z == "DES":
        return "DESIGNED"
    if z == "SD":
        return "SCRAMBLE_DRILL"
    return f"UNKNOWN:{z}"


def load_ftn_pbp_targets(seasons: Iterable[int]) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    meta: list[dict] = []
    for season in sorted({int(s) for s in seasons}):
        ftn_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        ftn, fm = _read_parquet(ftn_url)
        pbp, pm = _read_parquet(pbp_url)
        ftn.columns = [str(c).strip().lower() for c in ftn.columns]
        pbp.columns = [str(c).strip().lower() for c in pbp.columns]
        req_f = {"nflverse_game_id", "nflverse_play_id", "season", "week", "read_thrown"}
        req_p = {
            "game_id", "play_id", "season", "week", "season_type", "posteam",
            "pass_attempt", "sack", "two_point_attempt", "receiver_player_id", "receiver_player_name",
        }
        miss_f = sorted(req_f - set(ftn.columns))
        miss_p = sorted(req_p - set(pbp.columns))
        if miss_f or miss_p:
            raise RuntimeError(f"source schema missing season={season}: ftn={miss_f}, pbp={miss_p}")

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
            raise RuntimeError(f"duplicate FTN game/play key season={season}")
        if pbp.duplicated(["join_game", "join_play"]).any():
            raise RuntimeError(f"duplicate PBP game/play key season={season}")

        pcols = [
            "join_game", "join_play", "game_id", "season", "week", "posteam",
            "pass_attempt", "sack", "two_point_attempt", "receiver_player_id", "receiver_player_name",
        ]
        p = pbp[pcols].rename(columns={
            "game_id": "pbp_game_id", "season": "pbp_season", "week": "pbp_week",
            "posteam": "pbp_posteam", "pass_attempt": "pbp_pass_attempt",
            "sack": "pbp_sack", "two_point_attempt": "pbp_two_point_attempt",
            "receiver_player_id": "pbp_receiver_player_id", "receiver_player_name": "pbp_receiver_player_name",
        })
        m = ftn.merge(p, on=["join_game", "join_play"], how="left", validate="one_to_one", indicator=True)
        matched = m["_merge"].eq("both")
        join_rate = float(matched.mean()) if len(m) else 0.0
        if join_rate < MIN_JOIN_RATE:
            raise RuntimeError(f"FTN/PBP exact join below source contract season={season}: {join_rate}")
        if matched.any():
            parity = _num(m.loc[matched, "season"]).eq(_num(m.loc[matched, "pbp_season"])) & _num(m.loc[matched, "week"]).eq(_num(m.loc[matched, "pbp_week"]))
            if not bool(parity.all()):
                raise RuntimeError(f"FTN/PBP season-week parity failed season={season}")

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
        t["read_raw"] = t["read_thrown"].astype("object")
        t["read_norm"] = [normalize_read(v, season) for v in t["read_raw"]]
        t["is_primary_read"] = t["read_norm"].eq("PRIMARY_READ").astype(float)
        t["target_event_seq"] = np.arange(len(t), dtype=np.int64)

        norm_counts = t["read_norm"].value_counts(dropna=False).to_dict()
        interpretable = ~t["read_norm"].astype(str).str.startswith("UNKNOWN:") & ~t["read_norm"].eq("MISSING")
        interpretable_rate = float(interpretable.mean()) if len(t) else 0.0
        frames.append(t[[
            "season", "week", "game_id", "team", "receiver_id", "receiver_name_key",
            "read_norm", "is_primary_read", "target_event_seq",
        ]])
        meta.append({
            "season": season,
            "ftn_sha256": fm["sha256"], "ftn_bytes": fm["bytes"],
            "pbp_sha256": pm["sha256"], "pbp_bytes": pm["bytes"],
            "exact_join_rate": join_rate,
            "season_week_parity": True,
            "receiver_target_rows": int(len(t)),
            "interpretable_read_rate": interpretable_rate,
            "normalized_category_counts": {str(k): int(v) for k, v in norm_counts.items()},
            "primary_read_share_all_targets": float(t["is_primary_read"].mean()) if len(t) else None,
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
    games = frame[["season", "week", "game_id"]].drop_duplicates().sort_values(["season", "week", "game_id"], kind="mergesort").tail(int(n))
    return frame.merge(games, on=["season", "week", "game_id"], how="inner", validate="many_to_one")


def resolve_roster_player_id(rosters: pd.DataFrame, name_key: str, team: str, season: int, week: int) -> dict:
    prior = _prior(rosters, season, week)
    exact = prior.loc[prior["player_clean_key"].eq(str(name_key))].copy()
    ids = sorted({_clean_id(v) for v in exact["player_id"] if _clean_id(v)})
    out = {"mode": "unmatched", "player_id": "", "prior_name_rows": int(len(exact)), "team_disambiguated": False}
    if len(ids) == 1:
        out.update({"mode": "id", "player_id": ids[0]})
        return out
    if len(ids) > 1:
        team_ids = sorted({_clean_id(v) for v in exact.loc[exact["team"].eq(str(team)), "player_id"] if _clean_id(v)})
        if len(team_ids) == 1:
            out.update({"mode": "id", "player_id": team_ids[0], "team_disambiguated": True})
        else:
            out["mode"] = "ambiguous"
    return out


def resolve_prior_receiver_history(targets: pd.DataFrame, rosters: pd.DataFrame, name_key: str, team: str, season: int, week: int) -> tuple[pd.DataFrame, dict]:
    prior_targets = _prior(targets, season, week)
    roster = resolve_roster_player_id(rosters, name_key, team, season, week)
    if roster["mode"] == "ambiguous":
        return prior_targets.iloc[0:0].copy(), {**roster, "identity_source": "weekly_roster"}
    if roster["mode"] == "id":
        rid = roster["player_id"]
        return prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy(), {**roster, "identity_source": "weekly_roster"}
    alias = prior_targets.loc[prior_targets["receiver_name_key"].eq(str(name_key))].copy()
    ids = sorted({_clean_id(v) for v in alias["receiver_id"] if _clean_id(v)})
    if len(ids) > 1:
        return alias.iloc[0:0].copy(), {**roster, "mode": "ambiguous", "identity_source": "pbp_exact_name_fallback"}
    if len(ids) == 1:
        rid = ids[0]
        return prior_targets.loc[prior_targets["receiver_id"].eq(rid)].copy(), {**roster, "mode": "id", "player_id": rid, "identity_source": "pbp_exact_name_fallback"}
    return alias.iloc[0:0].copy(), {**roster, "mode": "unmatched", "identity_source": "pbp_exact_name_fallback"}


def pearson(a: pd.Series, b: pd.Series) -> float:
    x = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(x) < 3 or x["a"].nunique() < 2 or x["b"].nunique() < 2:
        return float("nan")
    return float(x["a"].corr(x["b"]))


def spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(x) < 3 or x["a"].nunique() < 2 or x["b"].nunique() < 2:
        return float("nan")
    return float(x["a"].rank(method="average").corr(x["b"].rank(method="average")))


def regression_r2(frame: pd.DataFrame) -> float:
    cols = ["primary_read_share8", "entitlement_tgt_share", "pred_targets", "wr1_indicator"]
    x = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(x) < 10 or x["primary_read_share8"].nunique() < 2:
        return float("nan")
    y = x["primary_read_share8"].to_numpy(dtype=float)
    X = np.column_stack([
        np.ones(len(x)),
        x["entitlement_tgt_share"].to_numpy(dtype=float),
        x["pred_targets"].to_numpy(dtype=float),
        x["wr1_indicator"].to_numpy(dtype=float),
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def audit(authority: pd.DataFrame, targets: pd.DataFrame, rosters: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    identity_counts: dict[str, int] = {}
    for r in authority.itertuples(index=False):
        hist, ident = resolve_prior_receiver_history(targets, rosters, r.player_clean_key, r.team, r.season, r.week)
        identity_counts[ident["mode"]] = identity_counts.get(ident["mode"], 0) + 1
        selected = _last_games(hist, PRIOR_GAMES)
        n_games = int(selected[["season", "week", "game_id"]].drop_duplicates().shape[0]) if len(selected) else 0
        n_targets = int(len(selected))
        supported = ident["mode"] == "id" and n_games >= MIN_PRIOR_TARGET_GAMES and n_targets >= MIN_RECEIVER_TARGETS
        share = float(selected["is_primary_read"].mean()) if supported and len(selected) else float("nan")
        rows.append({
            "season": int(r.season), "week": int(r.week), "team": r.team,
            "player_clean_key": r.player_clean_key, "player": r.player,
            "wr_rank": int(r.wr_rank), "wr1_indicator": int(int(r.wr_rank) == 1),
            "pred_targets": float(r.pred_targets), "entitlement_tgt_share": float(r.entitlement_tgt_share),
            "identity_mode": ident["mode"], "identity_source": ident.get("identity_source", ""),
            "resolved_receiver_id": ident.get("player_id", ""),
            "prior_target_games8": n_games, "prior_targets8": n_targets,
            "primary_read_share8": share, "supported": bool(supported),
        })
    panel = pd.DataFrame(rows)
    supported = panel.loc[panel["supported"] & panel["primary_read_share8"].notna()].copy()
    coverage = float(len(supported) / len(panel)) if len(panel) else 0.0
    ent_p = pearson(supported["primary_read_share8"], supported["entitlement_tgt_share"])
    ent_s = spearman(supported["primary_read_share8"], supported["entitlement_tgt_share"])
    tgt_p = pearson(supported["primary_read_share8"], supported["pred_targets"])
    tgt_s = spearman(supported["primary_read_share8"], supported["pred_targets"])
    r2 = regression_r2(supported)

    quartiles = []
    if len(supported):
        q = pd.qcut(supported["entitlement_tgt_share"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        for label, g in supported.assign(entitlement_quartile=q).groupby("entitlement_quartile", observed=True):
            s = g["primary_read_share8"]
            quartiles.append({
                "quartile": str(label), "n": int(len(g)), "mean": float(s.mean()),
                "sd": float(s.std(ddof=0)), "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            })

    summary = {
        "authority_rows_2023": int(len(panel)),
        "supported_rows": int(len(supported)),
        "coverage": coverage,
        "identity_mode_counts": identity_counts,
        "primary_vs_entitlement_pearson": ent_p,
        "primary_vs_entitlement_spearman": ent_s,
        "primary_vs_pred_targets_pearson": tgt_p,
        "primary_vs_pred_targets_spearman": tgt_s,
        "primary_explained_by_entitlement_predtargets_wr1_r2": r2,
        "entitlement_quartile_dispersion": quartiles,
    }
    return panel, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    authority, authority_meta = load_authority_opportunity_only(args.authority)
    targets, source_meta = load_ftn_pbp_targets([2022, 2023, 2024])
    rosters = load_roster_identity([2022, 2023])
    panel, redundancy = audit(authority, targets.loc[targets["season"].isin([2022, 2023])].copy(), rosters)

    source_ok = all(float(x["exact_join_rate"]) >= MIN_JOIN_RATE and float(x["interpretable_read_rate"]) >= MIN_INTERPRETABLE for x in source_meta)
    coverage_ok = float(redundancy["coverage"]) >= MIN_COVERAGE
    ent_s = redundancy["primary_vs_entitlement_spearman"]
    r2 = redundancy["primary_explained_by_entitlement_predtargets_wr1_r2"]
    nonredundant = bool(np.isfinite(ent_s) and abs(float(ent_s)) < MAX_ABS_ENTITLEMENT_SPEARMAN and np.isfinite(r2) and float(r2) < MAX_REDUNDANCY_R2)
    disposition = "READ_PRIORITY_R20_PLAN_ELIGIBLE" if source_ok and coverage_ok and nonredundant else "READ_PRIORITY_SOURCE_REDUNDANCY_BLOCKED"

    result = {
        "specification": "WR_POST_R19_READ_PRIORITY_SOURCE_REDUNDANCY_AUDIT_V1",
        "disposition": disposition,
        "source_ok": source_ok,
        "coverage_ok": coverage_ok,
        "nonredundant_under_frozen_screen": nonredundant,
        "thresholds": {
            "min_join_rate": MIN_JOIN_RATE,
            "min_interpretable_read_rate": MIN_INTERPRETABLE,
            "min_supported_coverage": MIN_COVERAGE,
            "max_abs_entitlement_spearman": MAX_ABS_ENTITLEMENT_SPEARMAN,
            "max_redundancy_r2": MAX_REDUNDANCY_R2,
        },
        "authority_meta": authority_meta,
        "source_meta": source_meta,
        "redundancy": redundancy,
        "wr_outcomes_loaded": False,
        "sportsbook_inputs": 0,
        "production_change": False,
    }

    panel.to_csv(args.out_dir / "wr_read_priority_source_feature_panel_2023.csv", index=False)
    (args.out_dir / "wr_read_priority_source_redundancy_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "wr_read_priority_source_meta.json").write_text(json.dumps(source_meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "disposition": disposition,
        "coverage": redundancy["coverage"],
        "primary_vs_entitlement_spearman": ent_s,
        "primary_vs_pred_targets_spearman": redundancy["primary_vs_pred_targets_spearman"],
        "redundancy_r2": r2,
        "wr_outcomes_loaded": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
