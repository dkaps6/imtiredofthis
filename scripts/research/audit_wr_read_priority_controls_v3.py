#!/usr/bin/env python3
"""Source-only control feasibility audit for prospective WR-R20.

No WR receiving-yard outcomes are loaded or scored. Uses the already-frozen V2
EARLY_NO_EXTENDED_SHARE8 semantics and tests whether the two controls required
by Claude's adversarial review are available without collapsing coverage or
showing obvious redundancy:
  - TEAM_EARLY_NO_EXTENDED_SHARE8
  - MEAN_AIR_YARDS_PER_TARGET8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import audit_wr_read_priority_source_redundancy_v1 as base
from scripts.research import audit_wr_read_priority_source_redundancy_v2 as v2
from scripts._opponent_map import canon_team

MIN_TEAM_CLASSIFIABLE = 40
MIN_AIR_TARGETS = 12
MIN_CONTROL_COVERAGE = 0.95
MAX_ABS_CONTROL_SPEARMAN = 0.75
MAX_EXPANDED_R2 = 0.60


def load_pbp_air_targets(seasons):
    frames = []
    meta = []
    for season in sorted({int(s) for s in seasons}):
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        pbp, pm = base._read_parquet(url)
        pbp.columns = [str(c).strip().lower() for c in pbp.columns]
        req = {"game_id","season","week","season_type","posteam","pass_attempt","sack","two_point_attempt","receiver_player_id","air_yards"}
        missing = sorted(req - set(pbp.columns))
        if missing:
            raise RuntimeError(f"PBP air source missing season={season}: {missing}")
        x = pbp.loc[
            pbp["season_type"].astype(str).str.upper().eq("REG")
            & pd.to_numeric(pbp["week"], errors="coerce").between(1,18)
            & pd.to_numeric(pbp["pass_attempt"], errors="coerce").fillna(0).eq(1)
            & ~pd.to_numeric(pbp["sack"], errors="coerce").fillna(0).eq(1)
            & ~pd.to_numeric(pbp["two_point_attempt"], errors="coerce").fillna(0).eq(1)
        ].copy()
        x["receiver_id"] = x["receiver_player_id"].map(base._clean_id)
        x = x.loc[x["receiver_id"].ne("")].copy()
        x["season"] = pd.to_numeric(x["season"], errors="coerce").astype(int)
        x["week"] = pd.to_numeric(x["week"], errors="coerce").astype(int)
        x["game_id"] = x["game_id"].astype(str)
        x["team"] = x["posteam"].map(canon_team)
        x["air_yards"] = pd.to_numeric(x["air_yards"], errors="coerce")
        frames.append(x[["season","week","game_id","team","receiver_id","air_yards"]])
        meta.append({"season": season, "pbp_sha256": pm["sha256"], "target_rows": int(len(x)), "nonnull_air_rows": int(x["air_yards"].notna().sum())})
    return pd.concat(frames, ignore_index=True), meta


def expanded_r2(frame: pd.DataFrame) -> float:
    cols = ["early_no_extended_share8","entitlement_tgt_share","pred_targets","wr1_indicator","team_early_no_extended_share8","mean_air_yards_per_target8"]
    x = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(x) < 20 or x["early_no_extended_share8"].nunique() < 2:
        return float("nan")
    y = x["early_no_extended_share8"].to_numpy(float)
    X = np.column_stack([np.ones(len(x))] + [x[c].to_numpy(float) for c in cols[1:]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y-pred)**2)); ss_tot = float(np.sum((y-y.mean())**2))
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    authority, authority_meta = base.load_authority_opportunity_only(args.authority)
    targets, source_meta = v2.load_targets_v2([2022,2023,2024])
    rosters = base.load_roster_identity([2022,2023])
    air, air_meta = load_pbp_air_targets([2022,2023])
    history_targets = targets.loc[targets["season"].isin([2022,2023])].copy()

    rows = []
    for r in authority.itertuples(index=False):
        hist, ident = base.resolve_prior_receiver_history(history_targets, rosters, r.player_clean_key, r.team, r.season, r.week)
        selected = base._last_games(hist, base.PRIOR_GAMES)
        n_games = int(selected[["season","week","game_id"]].drop_duplicates().shape[0]) if len(selected) else 0
        c = selected.loc[selected["progression_classifiable"]].copy() if len(selected) else selected.copy()
        n_class = int(len(c))
        primary_supported = ident["mode"] == "id" and n_games >= base.MIN_PRIOR_TARGET_GAMES and n_class >= v2.MIN_CLASSIFIABLE_TARGETS
        early = float(c["is_early_no_extended"].mean()) if primary_supported and n_class else float("nan")

        # Team scheme control: same offense, strictly prior, last 8 target-bearing games.
        team_hist = base._prior(history_targets.loc[history_targets["team"].eq(r.team)], r.season, r.week)
        team_sel = base._last_games(team_hist, base.PRIOR_GAMES)
        team_c = team_sel.loc[team_sel["progression_classifiable"]].copy() if len(team_sel) else team_sel.copy()
        team_n = int(len(team_c))
        team_share = float(team_c["is_early_no_extended"].mean()) if team_n >= MIN_TEAM_CLASSIFIABLE else float("nan")

        # Receiver target-depth control: exact receiver GSIS, strictly prior, last 8 target-bearing games.
        rid = ident.get("player_id", "") if ident["mode"] == "id" else ""
        ah = base._prior(air.loc[air["receiver_id"].eq(rid)], r.season, r.week) if rid else air.iloc[0:0].copy()
        asel = base._last_games(ah, base.PRIOR_GAMES)
        valid_air = asel["air_yards"].dropna() if len(asel) else pd.Series(dtype=float)
        air_mean = float(valid_air.mean()) if len(valid_air) >= MIN_AIR_TARGETS else float("nan")

        rows.append({
            "season": int(r.season), "week": int(r.week), "team": r.team, "player_clean_key": r.player_clean_key,
            "wr_rank": int(r.wr_rank), "wr1_indicator": int(int(r.wr_rank)==1),
            "pred_targets": float(r.pred_targets), "entitlement_tgt_share": float(r.entitlement_tgt_share),
            "identity_mode": ident["mode"], "resolved_receiver_id": rid,
            "prior_target_games8": n_games, "prior_classifiable_progression_targets8": n_class,
            "early_no_extended_share8": early, "primary_supported": bool(primary_supported),
            "team_classifiable_progression_targets8": team_n, "team_early_no_extended_share8": team_share,
            "valid_air_targets8": int(len(valid_air)), "mean_air_yards_per_target8": air_mean,
        })

    panel = pd.DataFrame(rows)
    s = panel.loc[panel["primary_supported"] & panel["early_no_extended_share8"].notna()].copy()
    team_cov = float(s["team_early_no_extended_share8"].notna().mean()) if len(s) else 0.0
    air_cov = float(s["mean_air_yards_per_target8"].notna().mean()) if len(s) else 0.0
    complete = s.dropna(subset=["team_early_no_extended_share8","mean_air_yards_per_target8"]).copy()
    team_s = base.spearman(complete["early_no_extended_share8"], complete["team_early_no_extended_share8"])
    air_s = base.spearman(complete["early_no_extended_share8"], complete["mean_air_yards_per_target8"])
    r2 = expanded_r2(complete)
    controls_ok = bool(team_cov >= MIN_CONTROL_COVERAGE and air_cov >= MIN_CONTROL_COVERAGE and np.isfinite(team_s) and abs(team_s) < MAX_ABS_CONTROL_SPEARMAN and np.isfinite(air_s) and abs(air_s) < MAX_ABS_CONTROL_SPEARMAN and np.isfinite(r2) and r2 < MAX_EXPANDED_R2)
    disposition = "READ_PRIORITY_R20_CONTROL_SET_ELIGIBLE" if controls_ok else "READ_PRIORITY_R20_CONTROL_SET_BLOCKED"

    result = {
        "specification": "WR_POST_R19_READ_PRIORITY_CONTROL_FEASIBILITY_V3",
        "disposition": disposition,
        "feature": "EARLY_NO_EXTENDED_SHARE8",
        "primary_supported_rows": int(len(s)),
        "team_control_coverage": team_cov,
        "air_control_coverage": air_cov,
        "complete_control_rows": int(len(complete)),
        "early_vs_team_early_spearman": team_s,
        "early_vs_mean_air_spearman": air_s,
        "expanded_redundancy_r2": r2,
        "thresholds": {"min_team_classifiable": MIN_TEAM_CLASSIFIABLE, "min_valid_air_targets": MIN_AIR_TARGETS, "min_control_coverage": MIN_CONTROL_COVERAGE, "max_abs_control_spearman": MAX_ABS_CONTROL_SPEARMAN, "max_expanded_r2": MAX_EXPANDED_R2},
        "authority_meta": authority_meta,
        "source_meta": source_meta,
        "air_meta": air_meta,
        "wr_outcomes_loaded": False,
        "holdout_2024_projection_or_outcome_fields_parsed": False,
        "sportsbook_inputs": 0,
        "production_change": False,
    }
    panel.to_csv(args.out_dir / "wr_read_priority_v3_control_panel_2023.csv", index=False)
    (args.out_dir / "wr_read_priority_v3_control_result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k: result[k] for k in ["disposition","primary_supported_rows","team_control_coverage","air_control_coverage","early_vs_team_early_spearman","early_vs_mean_air_spearman","expanded_redundancy_r2","wr_outcomes_loaded"]}, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
