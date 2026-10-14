#!/usr/bin/env python3
"""Player Personnel-Conditioned Opportunity V1.

Research-only player-first diagnostic. Uses historical nflverse participation/PBP
to ask whether a named WR/TE's own strictly-prior target propensity by offensive
personnel, combined with his team's strictly-prior personnel mix, explains his
next promoted-authority opportunity residual.

No target-game personnel is used as a feature. No coefficient is fit.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

VERSION = "PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_V1"
WR_SEASONS = (2023, 2024)
TE_SEASONS = (2024, 2025)
LOAD_SEASONS = (2023, 2024, 2025)
PLAYER_HISTORY_GAMES = 8
TEAM_HISTORY_GAMES = 3
MIN_PLAYER_GAMES = 3
MIN_TEAM_GAMES = 2
MIN_PLAYER_DB = 40
MIN_BUCKET_DB = 10
BOOT_REPS = 5000
BOOT_SEED = 20261014
EXPECTED_WR_ROWS = 3224
EXPECTED_TE_ROWS = 2005
PERSONNEL_BUCKETS = ("11", "12", "OTHER")

SUPPORT = {
    "WR": {"season_rows": 400, "season_players": 75, "pooled_rows": 1000, "pooled_players": 120},
    "TE": {"season_rows": 200, "season_players": 35, "pooled_rows": 500, "pooled_players": 60},
}


def to_pd(v) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    return pd.DataFrame(v)


def lower(x: pd.DataFrame) -> pd.DataFrame:
    q = x.copy()
    q.columns = [str(c).strip().lower() for c in q.columns]
    return q


def num(v):
    return pd.to_numeric(v, errors="coerce")


def text(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "<na>"} else s


def split_list(v) -> list[str]:
    s = text(v)
    if not s:
        return []
    return [z.strip() for z in re.split(r"[;,]", s) if z.strip()]


def canon_player_key(name) -> str:
    raw = text(name)
    if not raw:
        return ""
    try:
        cname, key = canonicalize_player_name_safe(raw)
    except Exception:
        cname, key = "", ""
    cname = text(cname) or raw
    return text(key) or "".join(ch.lower() for ch in cname if ch.isalnum())


def personnel_code(v) -> str:
    s = text(v).upper()
    vals = {}
    for lab in ("RB", "TE", "WR"):
        m = re.search(rf"(\d+)\s*{lab}", s)
        vals[lab] = int(m.group(1)) if m else 0
    if not any(vals.values()):
        return "OTHER"
    code = f"{vals['RB']}{vals['TE']}"
    return code if code in {"11", "12"} else "OTHER"


def join_keys(part: pd.DataFrame, pbp: pd.DataFrame) -> list[str]:
    for keys in (
        ["nflverse_game_id", "play_id"],
        ["old_game_id", "play_id"],
        ["game_id", "play_id"],
    ):
        if all(k in part.columns and k in pbp.columns for k in keys):
            return list(keys)
    raise RuntimeError("participation/PBP have no shared game+play key")


def first(df: pd.DataFrame, cols: list[str], default="") -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index)


def read_parent(path: Path, pos: str) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    need = {"season", "week", "team", "player_clean_key", "current_opportunity_error"}
    miss = need - set(x.columns)
    if miss:
        raise RuntimeError(f"{pos} parent missing columns {sorted(miss)}")
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    x["current_opportunity_error"] = num(x["current_opportunity_error"])
    x["position_family"] = pos
    seasons = WR_SEASONS if pos == "WR" else TE_SEASONS
    x = x.loc[
        x["season"].isin(seasons)
        & x["week"].notna()
        & x["team"].ne("")
        & x["player_clean_key"].ne("")
        & x["current_opportunity_error"].notna()
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    expected = EXPECTED_WR_ROWS if pos == "WR" else EXPECTED_TE_ROWS
    if len(x) != expected:
        raise RuntimeError(f"{pos} parent row drift {len(x)} != {expected}")
    if x.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError(f"{pos} duplicate parent identity")
    return x.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True)


def load_roster_identity() -> tuple[dict[tuple[int, str, str], str], dict]:
    import nflreadpy as nfl

    rows = []
    for s in LOAD_SEASONS:
        q = lower(to_pd(nfl.load_rosters_weekly(int(s))))
        if q.empty:
            continue
        q["season"] = num(first(q, ["season"], s)).fillna(s).astype(int)
        q["team"] = first(q, ["team", "team_abbr", "club_code"]).map(canon_team)
        q["position"] = first(q, ["position", "pos"]).fillna("").astype(str).str.upper().str.strip()
        q = q.loc[q["position"].isin(["WR", "TE"])].copy()
        q["player_name"] = first(q, ["full_name", "football_name", "player_name", "player", "name"]).fillna("").astype(str)
        q["player_clean_key"] = q["player_name"].map(canon_player_key)
        q["gsis_id"] = first(q, ["gsis_id", "player_id"]).fillna("").astype(str).str.strip()
        q = q.loc[q["team"].ne("") & q["player_clean_key"].ne("") & q["gsis_id"].ne("")].copy()
        rows.append(q[["season", "team", "player_clean_key", "gsis_id"]])
    if not rows:
        raise RuntimeError("weekly roster identity source empty")
    allr = pd.concat(rows, ignore_index=True).drop_duplicates()
    mapping: dict[tuple[int, str, str], str] = {}
    ambiguous = 0
    for key, g in allr.groupby(["season", "team", "player_clean_key"], sort=False):
        ids = sorted(set(g["gsis_id"].astype(str)))
        if len(ids) == 1:
            mapping[(int(key[0]), str(key[1]), str(key[2]))] = ids[0]
        else:
            ambiguous += 1
    audit = {
        "roster_rows": int(len(allr)),
        "unique_mappings": int(len(mapping)),
        "ambiguous_name_team_season_keys": int(ambiguous),
    }
    return mapping, audit


def load_play_state() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import nflreadpy as nfl

    pbp = lower(to_pd(nfl.load_pbp(seasons=list(LOAD_SEASONS))))
    part = lower(to_pd(nfl.load_participation(seasons=list(LOAD_SEASONS))))
    if pbp.empty or part.empty:
        raise RuntimeError("PBP or participation source empty")

    keys = join_keys(part, pbp)
    keep = keys + [
        c for c in [
            "season", "week", "season_type", "posteam", "qb_dropback",
            "pass_attempt", "receiver_player_id"
        ] if c in pbp.columns and c not in keys
    ]
    right = pbp[keep].drop_duplicates(keys)
    x = part.merge(right, on=keys, how="inner", suffixes=("", "_pbp"), validate="one_to_one")
    if x.empty:
        raise RuntimeError("participation/PBP join empty")

    season_col = "season_pbp" if "season_pbp" in x.columns else "season"
    week_col = "week_pbp" if "week_pbp" in x.columns else "week"
    season_type_col = "season_type_pbp" if "season_type_pbp" in x.columns else "season_type"
    team_col = "posteam_pbp" if "posteam_pbp" in x.columns else ("posteam" if "posteam" in x.columns else "possession_team")
    qb_col = "qb_dropback_pbp" if "qb_dropback_pbp" in x.columns else "qb_dropback"
    rec_col = "receiver_player_id_pbp" if "receiver_player_id_pbp" in x.columns else "receiver_player_id"

    x["_season"] = num(x[season_col])
    x["_week"] = num(x[week_col])
    if season_type_col in x.columns:
        x = x.loc[x[season_type_col].astype(str).str.upper().eq("REG")].copy()
    x["_team"] = x[team_col].map(canon_team)
    x["_dbr"] = num(x.get(qb_col, 0)).fillna(0).eq(1)
    x["_receiver"] = x.get(rec_col, pd.Series("", index=x.index)).fillna("").astype(str).str.strip()
    x["_bucket"] = x.get("offense_personnel", pd.Series("", index=x.index)).map(personnel_code)
    x = x.loc[
        x["_season"].isin(LOAD_SEASONS)
        & x["_week"].between(1, 18)
        & x["_team"].ne("")
        & x["_dbr"]
    ].copy()
    x["_season"] = x["_season"].astype(int)
    x["_week"] = x["_week"].astype(int)

    team_game = (
        x.groupby(["_season", "_week", "_team", "_bucket"], as_index=False)
        .size()
        .rename(columns={"_season": "season", "_week": "week", "_team": "team", "_bucket": "bucket", "size": "dropbacks"})
    )

    rows = []
    bad_parallel = 0
    for r in x[["_season", "_week", "_team", "_bucket", "_receiver", "offense_players", "offense_positions"]].itertuples(index=False, name=None):
        season, week, team, bucket, receiver, op, opos = r
        ids = split_list(op)
        pos = [p.upper().strip() for p in split_list(opos)]
        if len(ids) != len(pos):
            bad_parallel += 1
            continue
        seen = set()
        for pid, pp in zip(ids, pos):
            if pp not in {"WR", "TE"}:
                continue
            pid = text(pid)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            rows.append({
                "season": int(season),
                "week": int(week),
                "team": str(team),
                "gsis_id": pid,
                "position": pp,
                "bucket": str(bucket),
                "target": int(pid == text(receiver)),
            })
    if not rows:
        raise RuntimeError("expanded WR/TE participation rows empty")
    exp = pd.DataFrame(rows)
    player_game = (
        exp.groupby(["season", "week", "team", "gsis_id", "position", "bucket"], as_index=False)
        .agg(on_field_dropbacks=("target", "size"), targets=("target", "sum"))
    )
    audit = {
        "pbp_rows": int(len(pbp)),
        "participation_rows": int(len(part)),
        "joined_dropback_plays": int(len(x)),
        "expanded_wrte_dropback_rows": int(len(exp)),
        "player_game_bucket_rows": int(len(player_game)),
        "team_game_bucket_rows": int(len(team_game)),
        "parallel_list_mismatch_plays_skipped": int(bad_parallel),
    }
    return player_game, team_game, audit


def build_features(parent: pd.DataFrame, roster_map, player_game: pd.DataFrame, team_game: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    mapped = 0
    history_future_violations = 0

    pg_groups = {
        k: g.sort_values("week").copy()
        for k, g in player_game.groupby(["season", "team", "gsis_id"], sort=False)
    }
    tg_groups = {
        k: g.sort_values("week").copy()
        for k, g in team_game.groupby(["season", "team"], sort=False)
    }

    for r in parent.itertuples(index=False):
        identity = (int(r.season), str(r.team), str(r.player_clean_key))
        pid = roster_map.get(identity)
        if not pid:
            continue
        mapped += 1

        pg = pg_groups.get((int(r.season), str(r.team), str(pid)))
        tg = tg_groups.get((int(r.season), str(r.team)))
        if pg is None or tg is None:
            continue

        prior_pg = pg.loc[pg["week"].lt(int(r.week))].copy()
        prior_weeks = sorted(prior_pg["week"].unique())[-PLAYER_HISTORY_GAMES:]
        prior_pg = prior_pg.loc[prior_pg["week"].isin(prior_weeks)].copy()

        prior_tg = tg.loc[tg["week"].lt(int(r.week))].copy()
        prior_team_weeks = sorted(prior_tg["week"].unique())[-TEAM_HISTORY_GAMES:]
        prior_tg = prior_tg.loc[prior_tg["week"].isin(prior_team_weeks)].copy()

        if len(prior_pg) and int(prior_pg["week"].max()) >= int(r.week):
            history_future_violations += 1
        if len(prior_tg) and int(prior_tg["week"].max()) >= int(r.week):
            history_future_violations += 1

        n_player_games = len(prior_weeks)
        n_team_games = len(prior_team_weeks)
        if n_player_games < MIN_PLAYER_GAMES or n_team_games < MIN_TEAM_GAMES:
            continue

        ptotal_db = float(prior_pg["on_field_dropbacks"].sum())
        ptotal_tgt = float(prior_pg["targets"].sum())
        if ptotal_db < MIN_PLAYER_DB:
            continue
        unconditional = ptotal_tgt / ptotal_db

        pby = prior_pg.groupby("bucket", as_index=True)[["on_field_dropbacks", "targets"]].sum()
        tby = prior_tg.groupby("bucket", as_index=True)["dropbacks"].sum()
        team_total = float(tby.sum())
        if team_total <= 0:
            continue

        rates = {}
        mixes = {}
        for b in PERSONNEL_BUCKETS:
            db = float(pby.loc[b, "on_field_dropbacks"]) if b in pby.index else 0.0
            tgt = float(pby.loc[b, "targets"]) if b in pby.index else 0.0
            rates[b] = (tgt / db) if db >= MIN_BUCKET_DB else unconditional
            mixes[b] = (float(tby.get(b, 0.0)) / team_total)

        conditional = sum(mixes[b] * rates[b] for b in PERSONNEL_BUCKETS)
        lift = conditional - unconditional

        rows.append({
            "position_family": str(r.position_family),
            "season": int(r.season),
            "week": int(r.week),
            "team": str(r.team),
            "player_clean_key": str(r.player_clean_key),
            "gsis_id": str(pid),
            "prior_player_games": int(n_player_games),
            "prior_team_games": int(n_team_games),
            "prior_player_dropbacks": float(ptotal_db),
            "unconditional_target_rate": float(unconditional),
            "conditional_target_rate": float(conditional),
            "personnel_lift": float(lift),
            "mix_11": float(mixes["11"]),
            "mix_12": float(mixes["12"]),
            "mix_other": float(mixes["OTHER"]),
            "rate_11": float(rates["11"]),
            "rate_12": float(rates["12"]),
            "rate_other": float(rates["OTHER"]),
            "current_opportunity_error": float(r.current_opportunity_error),
            "realized_opportunity_gain": float(-r.current_opportunity_error),
        })

    out = pd.DataFrame(rows)
    audit = {
        "parent_rows": int(len(parent)),
        "identity_mapped_rows": int(mapped),
        "identity_coverage": float(mapped / len(parent)) if len(parent) else 0.0,
        "eligible_rows": int(len(out)),
        "same_future_history_violations": int(history_future_violations),
    }
    return out, audit


def corr(a, b) -> float:
    q = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(q) < 3 or q["a"].nunique() < 2 or q["b"].nunique() < 2:
        return float("nan")
    return float(spearmanr(q["a"].to_numpy(float), q["b"].to_numpy(float)).statistic)


def summarize(q: pd.DataFrame) -> dict:
    if q.empty:
        return {"rows": 0, "players": 0, "rho": None, "q4_q1_gap": None}
    rho = corr(q["personnel_lift"], q["realized_opportunity_gain"])
    q25 = float(np.quantile(q["personnel_lift"], 0.25, method="linear"))
    q75 = float(np.quantile(q["personnel_lift"], 0.75, method="linear"))
    lo = q.loc[q["personnel_lift"].le(q25), "realized_opportunity_gain"]
    hi = q.loc[q["personnel_lift"].ge(q75), "realized_opportunity_gain"]
    return {
        "rows": int(len(q)),
        "players": int(q["player_clean_key"].nunique()),
        "rho": float(rho) if np.isfinite(rho) else None,
        "personnel_lift_q25": q25,
        "personnel_lift_q75": q75,
        "q1_mean_realized_opportunity_gain": float(lo.mean()) if len(lo) else None,
        "q4_mean_realized_opportunity_gain": float(hi.mean()) if len(hi) else None,
        "q4_q1_gap": float(hi.mean() - lo.mean()) if len(lo) and len(hi) else None,
    }


def bootstrap_player_rho(q: pd.DataFrame) -> dict:
    players = sorted(q["player_clean_key"].astype(str).unique())
    groups = {p: q.loc[q["player_clean_key"].astype(str).eq(p)].copy() for p in players}
    rng = np.random.default_rng(BOOT_SEED)
    vals = []
    for _ in range(BOOT_REPS):
        sampled = rng.choice(players, size=len(players), replace=True)
        z = pd.concat([groups[p] for p in sampled], ignore_index=True)
        v = corr(z["personnel_lift"], z["realized_opportunity_gain"])
        if np.isfinite(v):
            vals.append(v)
    a = np.asarray(vals, dtype=float)
    return {
        "reps": BOOT_REPS,
        "valid_reps": int(len(a)),
        "seed": BOOT_SEED,
        "observed_rho": corr(q["personnel_lift"], q["realized_opportunity_gain"]),
        "p_positive": float((a > 0).mean()) if len(a) else None,
        "ci_low": float(np.quantile(a, 0.025)) if len(a) else None,
        "ci_high": float(np.quantile(a, 0.975)) if len(a) else None,
    }


def position_result(q: pd.DataFrame, pos: str, parent_audit: dict) -> dict:
    seasons = WR_SEASONS if pos == "WR" else TE_SEASONS
    cfg = SUPPORT[pos]
    z = q.loc[q["position_family"].eq(pos)].copy()
    by = {str(s): summarize(z.loc[z["season"].eq(s)].copy()) for s in seasons}
    pooled = summarize(z)
    boot = bootstrap_player_rho(z) if len(z) else {
        "reps": BOOT_REPS, "valid_reps": 0, "seed": BOOT_SEED,
        "observed_rho": None, "p_positive": None, "ci_low": None, "ci_high": None
    }
    season_identity = {}
    for s in seasons:
        p = parent_audit[(pos, s)]
        season_identity[str(s)] = p

    support = (
        all(
            by[str(s)]["rows"] >= cfg["season_rows"]
            and by[str(s)]["players"] >= cfg["season_players"]
            and season_identity[str(s)]["identity_coverage"] >= 0.60
            for s in seasons
        )
        and pooled["rows"] >= cfg["pooled_rows"]
        and pooled["players"] >= cfg["pooled_players"]
    )

    science = (
        support
        and all(by[str(s)]["rho"] is not None and by[str(s)]["rho"] > 0 for s in seasons)
        and pooled["rho"] is not None and pooled["rho"] > 0
        and boot["ci_low"] is not None and boot["ci_low"] > 0
        and pooled["q4_q1_gap"] is not None and pooled["q4_q1_gap"] > 0
    )
    return {
        "support": bool(support),
        "confirmed": bool(science),
        "by_season": by,
        "pooled": pooled,
        "player_cluster_bootstrap": boot,
        "identity_by_season": season_identity,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-rows", type=Path, required=True)
    ap.add_argument("--te-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    wr = read_parent(a.wr_rows, "WR")
    te = read_parent(a.te_rows, "TE")
    parent = pd.concat([wr, te], ignore_index=True)

    roster_map, roster_audit = load_roster_identity()
    player_game, team_game, play_audit = load_play_state()

    features, overall_audit = build_features(parent, roster_map, player_game, team_game)

    parent_audit = {}
    for pos, seasons in (("WR", WR_SEASONS), ("TE", TE_SEASONS)):
        for s in seasons:
            p = parent.loc[parent["position_family"].eq(pos) & parent["season"].eq(s)].copy()
            mapped = sum((int(r.season), str(r.team), str(r.player_clean_key)) in roster_map for r in p.itertuples(index=False))
            e = features.loc[features["position_family"].eq(pos) & features["season"].eq(s)].copy()
            parent_audit[(pos, s)] = {
                "parent_rows": int(len(p)),
                "identity_mapped_rows": int(mapped),
                "identity_coverage": float(mapped / len(p)) if len(p) else 0.0,
                "eligible_rows": int(len(e)),
                "eligible_players": int(e["player_clean_key"].nunique()) if len(e) else 0,
            }

    wr_result = position_result(features, "WR", parent_audit)
    te_result = position_result(features, "TE", parent_audit)

    integrity = (
        overall_audit["same_future_history_violations"] == 0
        and len(wr) == EXPECTED_WR_ROWS
        and len(te) == EXPECTED_TE_ROWS
        and not features.duplicated(["position_family", "season", "week", "team", "player_clean_key"]).any()
        and np.isfinite(features["personnel_lift"]).all()
        and np.isfinite(features["realized_opportunity_gain"]).all()
    )

    if not integrity:
        disposition = "PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_INTEGRITY_FAILURE"
    elif not wr_result["support"] and not te_result["support"]:
        disposition = "PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_SOURCE_LIMITED"
    elif wr_result["confirmed"] and te_result["confirmed"]:
        disposition = "WR_TE_PERSONNEL_CONDITIONED_OPPORTUNITY_CONFIRMED"
    elif wr_result["confirmed"]:
        disposition = "WR_PERSONNEL_CONDITIONED_OPPORTUNITY_CONFIRMED"
    elif te_result["confirmed"]:
        disposition = "TE_PERSONNEL_CONDITIONED_OPPORTUNITY_CONFIRMED"
    else:
        disposition = "NO_ACTIONABLE_PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY"

    result = {
        "version": VERSION,
        "disposition": disposition,
        "wr_source_run": 37634044900,
        "wr_source_artifact": 11487359185,
        "te_source_run": 37631011196,
        "te_source_artifact": 11486082927,
        "wr": wr_result,
        "te": te_result,
        "roster_audit": roster_audit,
        "play_audit": play_audit,
        "overall_audit": overall_audit,
        "candidate_models_fit": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_read": 0,
        "target_game_personnel_used": 0,
        "prior_residual_fed_to_projection": 0,
        "production_changed": False,
    }

    features.to_csv(a.out_dir / "player_personnel_conditioned_opportunity_rows.csv", index=False)
    (a.out_dir / "player_personnel_conditioned_opportunity_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
