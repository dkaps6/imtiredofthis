#!/usr/bin/env python3
"""Build the frozen R27D strict-prior xYAC/YACOE dataset.

Scientific boundary:
- Source PBP: REG 2019-2025 only.
- Every feature for (season, week) uses source weeks strictly earlier than that week.
- Current-week catches are added to history only AFTER all current-week features/labels are materialized.
- Target-game PBP is used only for the historical training label, never its own predictors.
- No sportsbook information and no candidate projection are created here.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.pbp import get_pbp

RB_POS = {"RB", "FB", "HB", "TB"}
PLAYER_K = 12.0
TEAM_K = 30.0
OPP_K = 30.0
FEATURES = [
    "player_relative_yacoe_prior",
    "player_expected_yac_prior_relative_to_league",
    "team_rb_relative_yacoe_prior",
    "team_rb_expected_yac_prior_relative_to_league",
    "opp_rb_relative_yacoe_allowed_prior",
    "opp_rb_expected_yac_allowed_prior_relative_to_league",
    "week1",
    "prior_xyac_reception_support_log1p",
]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def key_name(v) -> str:
    try:
        _, k = canonicalize_player_name_safe(v)
        if k:
            return str(k)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def nonblank(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip().ne("")


def load_weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    q = _normalize_weekly(_to_pandas(raw), int(season)).copy()
    q["week"] = num(q["week"])
    q["team"] = q["team"].map(canon_team)
    q["player_id_norm"] = q.get("player_id", "").astype("string").fillna("").str.strip()
    q["player_clean_key"] = q.get("player_clean_key", q.get("player", "")).astype("string").fillna("").map(key_name)
    q["position"] = q.get("position", "").astype("string").fillna("").str.upper().str.strip()
    return q


def load_rb_xyac_games(season: int) -> tuple[pd.DataFrame, dict]:
    p = get_pbp(int(season), min_rows=1).copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
    if "season_type" in p.columns:
        reg = p[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            p = reg

    name_col = first_col(p, ["receiver_player_name", "receiver_name", "receiver"])
    id_col = first_col(p, ["receiver_player_id", "receiver_id"])
    required = ["week", "posteam", "defteam", "complete_pass", "yards_after_catch", "xyac_mean_yardage"]
    missing = [c for c in required if c not in p.columns]
    if name_col is None and id_col is None:
        missing.append("receiver_identity")
    if missing:
        raise RuntimeError(f"{season}: required R27D PBP fields unavailable: {missing}")

    p["season"] = int(season)
    p["week"] = num(p["week"])
    p["team"] = p["posteam"].map(canon_team)
    p["opponent"] = p["defteam"].map(canon_team)
    p["receiver_id_norm"] = p[id_col].astype("string").fillna("").str.strip() if id_col else ""
    p["receiver_name_norm"] = p[name_col].astype("string").fillna("").str.strip() if name_col else ""
    p["receiver_name_key"] = p["receiver_name_norm"].map(key_name)
    targeted = (
        (nonblank(p["receiver_id_norm"]) | nonblank(p["receiver_name_norm"]))
        & p["week"].notna()
        & p["team"].ne("")
        & p["opponent"].ne("")
    )
    t = p.loc[targeted].copy()

    weekly = load_weekly(int(season))
    id_map = weekly.loc[nonblank(weekly["player_id_norm"]), ["week", "team", "player_id_norm", "player_clean_key", "position"]].drop_duplicates(
        ["week", "team", "player_id_norm"]
    ).rename(columns={"player_id_norm": "receiver_id_norm", "player_clean_key": "id_player_key", "position": "position_by_id"})
    name_map = weekly.loc[nonblank(weekly["player_clean_key"]), ["week", "team", "player_clean_key", "position"]].drop_duplicates(
        ["week", "team", "player_clean_key"]
    ).rename(columns={"player_clean_key": "receiver_name_key", "position": "position_by_name"})

    t = t.merge(id_map, on=["week", "team", "receiver_id_norm"], how="left", validate="many_to_one")
    t = t.merge(name_map, on=["week", "team", "receiver_name_key"], how="left", validate="many_to_one")
    t["player_clean_key"] = t["id_player_key"].replace("", pd.NA).combine_first(t["receiver_name_key"].replace("", pd.NA)).fillna("")
    t["receiver_position"] = t["position_by_id"].replace("", pd.NA).combine_first(t["position_by_name"]).fillna("")
    rb = t[t["receiver_position"].astype(str).str.upper().isin(RB_POS) & t["player_clean_key"].ne("")].copy()
    rb["complete_num"] = num(rb["complete_pass"]).fillna(0.0)
    rb["yac"] = num(rb["yards_after_catch"])
    rb["xyac"] = num(rb["xyac_mean_yardage"])
    obs = rb[rb["complete_num"].eq(1) & rb["yac"].notna() & rb["xyac"].notna()].copy()
    obs["yacoe"] = obs["yac"] - obs["xyac"]

    keys = ["season", "week", "team", "opponent", "player_clean_key"]
    games = obs.groupby(keys, dropna=False).agg(
        game_xyac_observed_receptions=("yacoe", "size"),
        game_yacoe_sum=("yacoe", "sum"),
        game_expected_yac_sum=("xyac", "sum"),
    ).reset_index()
    games["game_yacoe_per_reception"] = games["game_yacoe_sum"] / games["game_xyac_observed_receptions"]
    games["game_expected_yac_per_reception"] = games["game_expected_yac_sum"] / games["game_xyac_observed_receptions"]

    meta = {
        "season": int(season),
        "rb_target_rows": int(len(rb)),
        "xyac_observed_completed_rb_catches": int(len(obs)),
        "xyac_game_rows": int(len(games)),
        "receiver_position_resolution_rate": float(t["receiver_position"].ne("").mean()) if len(t) else 0.0,
    }
    return games, meta


def state():
    return [0.0, 0.0, 0.0]  # n, yacoe_sum, expected_yac_sum


def add_state(bucket: dict, k, n: float, ysum: float, xsum: float) -> None:
    cur = bucket[k]
    cur[0] += float(n)
    cur[1] += float(ysum)
    cur[2] += float(xsum)


def feature_row(season: int, week: int, player_key: str, team: str, opponent: str,
                league: list[float], players: dict, teams: dict, opps: dict) -> dict:
    ln, lys, lxs = league
    if ln <= 0:
        return {**{c: np.nan for c in FEATURES[:-2]}, "week1": int(week == 1), "prior_xyac_reception_support_log1p": 0.0,
                "league_prior_available": 0, "league_prior_xyac_receptions": 0.0,
                "league_rb_yacoe_prior": np.nan, "league_rb_expected_yac_prior": np.nan}
    ly = lys / ln
    lx = lxs / ln

    def rel(bucket: dict, k, K: float) -> tuple[float, float, float]:
        n, ys, xs = bucket[k]
        sy = (ys + K * ly) / (n + K)
        sx = (xs + K * lx) / (n + K)
        return sy - ly, sx - lx, n

    py, px, pn = rel(players, player_key, PLAYER_K)
    ty, tx, _ = rel(teams, team, TEAM_K)
    oy, ox, _ = rel(opps, opponent, OPP_K)
    return {
        "player_relative_yacoe_prior": py,
        "player_expected_yac_prior_relative_to_league": px,
        "team_rb_relative_yacoe_prior": ty,
        "team_rb_expected_yac_prior_relative_to_league": tx,
        "opp_rb_relative_yacoe_allowed_prior": oy,
        "opp_rb_expected_yac_allowed_prior_relative_to_league": ox,
        "week1": int(week == 1),
        "prior_xyac_reception_support_log1p": float(math.log1p(max(pn, 0.0))),
        "league_prior_available": 1,
        "league_prior_xyac_receptions": float(ln),
        "league_rb_yacoe_prior": float(ly),
        "league_rb_expected_yac_prior": float(lx),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-predictions", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    parent = pd.read_csv(args.parent_predictions).copy()
    required_parent = [
        "season", "week", "player_clean_key", "team", "opponent", "role",
        "vacancy_active", "vacancy_incumbent", "baseline_targets", "candidate_targets",
        "baseline_receptions", "candidate_receptions", "production_implied_ypr",
        "production_ypt", "production_catch_rate", "b0_rec_yards", "b1_rec_yards",
        "actual_rec_yards",
    ]
    missing_parent = [c for c in required_parent if c not in parent.columns]
    if missing_parent:
        raise RuntimeError(f"R27D exact parent is missing required columns: {missing_parent}")
    parent["parent_row_id"] = np.arange(len(parent), dtype=int)
    parent["season"] = num(parent["season"]).astype(int)
    parent["week"] = num(parent["week"]).astype(int)
    parent["team"] = parent["team"].map(canon_team)
    parent["opponent"] = parent["opponent"].map(canon_team)
    parent["player_clean_key"] = parent["player_clean_key"].astype(str).map(key_name)

    source_parts, source_meta = [], []
    for season in range(2019, 2026):
        g, m = load_rb_xyac_games(season)
        source_parts.append(g)
        source_meta.append(m)
        print("[r27d-source]", json.dumps(m, sort_keys=True))
    source = pd.concat(source_parts, ignore_index=True)
    source["season"] = num(source["season"]).astype(int)
    source["week"] = num(source["week"]).astype(int)
    source["team"] = source["team"].map(canon_team)
    source["opponent"] = source["opponent"].map(canon_team)
    source["player_clean_key"] = source["player_clean_key"].astype(str).map(key_name)

    league = state()
    players = defaultdict(state)
    teams = defaultdict(state)
    opps = defaultdict(state)
    hist_rows: list[dict] = []
    parent_features: dict[int, dict] = {}
    source_by_week = {(int(s), int(w)): g.copy() for (s, w), g in source.groupby(["season", "week"], sort=True)}
    parent_by_week = {(int(s), int(w)): g.copy() for (s, w), g in parent.groupby(["season", "week"], sort=True)}
    all_weeks = sorted(set(source_by_week) | set(parent_by_week))

    for season, week in all_weeks:
        current_source = source_by_week.get((season, week), pd.DataFrame())
        current_parent = parent_by_week.get((season, week), pd.DataFrame())

        # Freeze all features/labels against history ending strictly before this week.
        if len(current_source):
            for r in current_source.itertuples(index=False):
                f = feature_row(season, week, r.player_clean_key, r.team, r.opponent, league, players, teams, opps)
                label = (
                    float(r.game_yacoe_per_reception) - float(f["league_rb_yacoe_prior"])
                    if f["league_prior_available"] == 1 else np.nan
                )
                hist_rows.append({
                    "season": season,
                    "week": week,
                    "team": r.team,
                    "opponent": r.opponent,
                    "player_clean_key": r.player_clean_key,
                    "game_xyac_observed_receptions": int(r.game_xyac_observed_receptions),
                    "game_yacoe_per_reception": float(r.game_yacoe_per_reception),
                    "game_expected_yac_per_reception": float(r.game_expected_yac_per_reception),
                    "relative_game_yacoe": label,
                    **f,
                })

        if len(current_parent):
            for r in current_parent.itertuples(index=False):
                parent_features[int(r.parent_row_id)] = feature_row(
                    season, week, r.player_clean_key, r.team, r.opponent, league, players, teams, opps
                )

        # Only after the entire current week is materialized may it enter future priors.
        if len(current_source):
            for r in current_source.itertuples(index=False):
                n = float(r.game_xyac_observed_receptions)
                ys = float(r.game_yacoe_sum)
                xs = float(r.game_expected_yac_sum)
                league[0] += n; league[1] += ys; league[2] += xs
                add_state(players, r.player_clean_key, n, ys, xs)
                add_state(teams, r.team, n, ys, xs)
                add_state(opps, r.opponent, n, ys, xs)

    hist = pd.DataFrame(hist_rows)
    pf = pd.DataFrame.from_dict(parent_features, orient="index").rename_axis("parent_row_id").reset_index()
    enriched = parent.merge(pf, on="parent_row_id", how="left", validate="one_to_one")

    # Frozen study evaluates 2020-2025; every evaluation row must have a legal league prior.
    eval_rows = enriched[enriched["season"].between(2020, 2025)].copy()
    if len(eval_rows) != len(parent):
        raise RuntimeError(f"parent evaluation universe drift: parent={len(parent)} eval={len(eval_rows)}")
    if not eval_rows["league_prior_available"].eq(1).all():
        bad = eval_rows.loc[~eval_rows["league_prior_available"].eq(1), ["season", "week", "player_clean_key"]].head(20)
        raise RuntimeError(f"evaluation rows without legal prior league xYAC state:\n{bad}")

    # Training rows with undefined pre-history (e.g. earliest 2019 week) are intentionally not fit.
    trainable = hist[hist["relative_game_yacoe"].notna()].copy()
    cutoff_violations = int(((trainable["season"] == trainable["season"]) & (trainable["league_prior_available"] != 1)).sum())
    if cutoff_violations:
        raise RuntimeError("strict-prior training label materialization violation")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist.to_csv(args.out_dir / "r27d_historical_yacoe_training_rows.csv", index=False)
    enriched.to_csv(args.out_dir / "r27d_parent_rows_with_strict_prior_features.csv", index=False)
    pd.DataFrame(source_meta).to_csv(args.out_dir / "r27d_source_meta_by_season.csv", index=False)

    summary = {
        "study": "RB_R27D_STRICT_PRIOR_YACOE_RESIDUAL_V1",
        "parent_rows": int(len(parent)),
        "historical_xyac_player_games": int(len(hist)),
        "trainable_historical_rows": int(len(trainable)),
        "evaluation_rows_with_legal_league_prior": int(eval_rows["league_prior_available"].eq(1).sum()),
        "feature_names": FEATURES,
        "player_k": PLAYER_K,
        "team_k": TEAM_K,
        "opponent_k": OPP_K,
        "sportsbook_inputs_used": 0,
        "candidate_projection_created": False,
        "target_week_features_used": 0,
        "future_features_used": 0,
        "chronology_rule": "materialize full season/week before updating any accumulator with that week",
    }
    (args.out_dir / "r27d_dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
