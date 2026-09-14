#!/usr/bin/env python3
"""WR-R17 Stage A, v1c: prior-roster GSIS identity bridge.

This is a pre-valid-result mechanical repair after run 34897984434 showed
0/2,076 authority rows could be bridged from full WR names directly to the
abbreviated PBP receiver-name field. No scientific signal was evaluable.

Frozen football science is unchanged. This implementation uses nflverse
weekly rosters as identity-only evidence, strictly prior to each target row,
to resolve the WR-R15 authority name to a stable GSIS/player ID. Prior PBP
history is then selected by that stable ID. Name fallback remains exact and
audited only when no stable ID is available. 2024 remains unscored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research import evaluate_wr_r17_target_depth_distribution_stage_a_v1b as base


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


def load_roster_identity(seasons: Iterable[int]) -> pd.DataFrame:
    """Load identity-only weekly roster rows with stable GSIS IDs.

    These rows contribute no football feature. Only season/week, canonical
    full name, team, and stable ID are retained for temporal identity mapping.
    """
    import nflreadpy as nfl

    frames = []
    for season in sorted({int(s) for s in seasons}):
        raw = _to_pandas(nfl.load_rosters_weekly(int(season)))
        if raw.empty:
            raise RuntimeError(f"WR-R17 weekly roster identity source returned zero rows for {season}")
        x = raw.copy()
        x.columns = [str(c).strip().lower() for c in x.columns]
        x["season"] = pd.to_numeric(_first(x, ["season"], season), errors="coerce").fillna(season).astype(int)
        x["week"] = pd.to_numeric(_first(x, ["week"]), errors="coerce")
        raw_name = _first(x, ["full_name", "football_name", "player_name", "player", "name"])
        x["player_clean_key"] = raw_name.map(base._name_key)
        x["team"] = _first(x, ["team", "team_abbr", "club_code"]).map(canon_team)
        x["player_id"] = _first(x, ["gsis_id", "player_id"]).map(base._clean_id)
        x = x.loc[
            x["season"].eq(season)
            & x["week"].between(1, 22, inclusive="both")
            & x["player_clean_key"].ne("")
            & x["player_id"].ne("")
        ].copy()
        if x.empty:
            raise RuntimeError(f"WR-R17 weekly roster source has zero stable identities for {season}")
        frames.append(x[["season", "week", "team", "player_clean_key", "player_id"]])
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(["season", "week", "team", "player_clean_key", "player_id"], keep="last")


def _prior(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return frame.loc[
        (frame["season"] < int(season))
        | ((frame["season"] == int(season)) & (frame["week"] < int(week)))
    ].copy()


def resolve_roster_player_id(
    rosters: pd.DataFrame,
    authority_name_key: str,
    authority_team: str,
    season: int,
    week: int,
) -> dict:
    """Resolve an exact authority full-name key to a prior stable GSIS ID.

    Exact name is used only to find the stable ID in the independent roster
    identity source. If multiple historical IDs share the exact name key, the
    authority team may disambiguate only when it leaves exactly one stable ID.
    Otherwise the row fails closed as ambiguous.
    """
    prior = _prior(rosters, season, week)
    exact = prior.loc[prior["player_clean_key"].eq(str(authority_name_key))].copy()
    ids = sorted({base._clean_id(v) for v in exact["player_id"] if base._clean_id(v)})
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
        team_ids = sorted({base._clean_id(v) for v in team_rows["player_id"] if base._clean_id(v)})
        if len(team_ids) == 1:
            out.update({
                "roster_identity_mode": "id",
                "roster_player_id": team_ids[0],
                "roster_team_disambiguated": True,
            })
        else:
            out["roster_identity_mode"] = "ambiguous"
        return out
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
        # Missing-ID PBP rows may accompany the ID-backed history only through
        # an exact authority-name alias. They are explicitly counted.
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

    # If no prior roster GSIS anchor exists, retain the frozen exact-name
    # fallback behavior. This still fails closed on multiple stable PBP IDs.
    history, pbp_audit = base.resolve_prior_receiver_history(
        targets, authority_name_key, season, week
    )
    return history, {
        **roster,
        **pbp_audit,
        "identity_source": "pbp_exact_name_fallback",
    }


def receiver_state(
    targets: pd.DataFrame,
    rosters: pd.DataFrame,
    authority_name_key: str,
    authority_team: str,
    season: int,
    week: int,
) -> dict:
    history, audit = resolve_prior_receiver_history(
        targets, rosters, authority_name_key, authority_team, season, week
    )
    h8 = base._last_games(history, base.PRIOR_GAMES)
    games8 = h8[["season", "week", "game_id"]].drop_duplicates() if not h8.empty else pd.DataFrame()
    air = base._num(h8["air"]).dropna() if not h8.empty else pd.Series(dtype=float)
    out = {
        **audit,
        "prior_target_games": int(len(games8)),
        "prior_target_events": int(len(air)),
        "DEPTH_SD8": np.nan,
        "DEPTH_IQR8": np.nan,
        "DEEP15_TARGET_SHARE8": np.nan,
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
    if len(games8) < base.MIN_PRIOR_TARGET_GAMES or len(air) < base.MIN_PRIOR_TARGET_EVENTS:
        return out
    out["DEPTH_SD8"] = float(air.std(ddof=0))
    out["DEPTH_IQR8"] = float(
        air.quantile(0.75, interpolation="linear")
        - air.quantile(0.25, interpolation="linear")
    )
    out["DEEP15_TARGET_SHARE8"] = float(air.ge(15.0).mean())
    out["mean_air_yards_per_target8"] = float(air.mean())
    return out


def build_development_panel(
    authority: pd.DataFrame,
    targets: pd.DataFrame,
    rosters: pd.DataFrame,
) -> pd.DataFrame:
    dev = authority.loc[authority["season"].eq(base.DEV_SEASON)].copy()
    if len(dev) != base.EXPECTED_ROWS[base.DEV_SEASON]:
        raise RuntimeError("WR-R17 development authority count drift")
    rows = []
    for r in dev.itertuples(index=False):
        key = str(r.player_clean_key)
        display_key = base._name_key(r.player)
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
        rows.append(row)
    panel = pd.DataFrame(rows)
    has_hist = panel["history_max_season"].notna()
    before = (panel["history_max_season"] < panel["season"]) | (
        (panel["history_max_season"] == panel["season"])
        & (panel["history_max_week"] < panel["week"])
    )
    bad = int((has_hist & ~before).sum())
    if bad:
        raise RuntimeError(f"WR-R17 target-game leakage assertion failed for {bad} rows")
    return panel


def identity_audit(panel: pd.DataFrame) -> dict:
    return {
        "development_rows": int(len(panel)),
        "expected_development_rows": base.EXPECTED_ROWS[base.DEV_SEASON],
        "identity_mode_counts": {
            str(k): int(v) for k, v in panel["identity_mode"].value_counts(dropna=False).to_dict().items()
        },
        "identity_source_counts": {
            str(k): int(v) for k, v in panel["identity_source"].value_counts(dropna=False).to_dict().items()
        },
        "rows_resolved_by_prior_roster_id": int(panel["identity_source"].eq("weekly_roster").sum()),
        "rows_team_disambiguated": int(panel["roster_team_disambiguated"].fillna(False).sum()),
        "rows_with_4plus_prior_target_games": int(panel["prior_target_games"].ge(base.MIN_PRIOR_TARGET_GAMES).sum()),
        "rows_with_12plus_prior_target_events": int(panel["prior_target_events"].ge(base.MIN_PRIOR_TARGET_EVENTS).sum()),
        "rows_with_valid_depth_signal": int(panel["DEPTH_SD8"].notna().sum()),
        "authority_display_key_mismatch_rows": int((~panel["authority_display_key_match"]).sum()),
        "target_game_leakage_rows": 0,
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

    authority = base.load_authority(args.authority)
    targets = base.prepare_target_events(base.load_pbp([2022, 2023]))
    rosters = load_roster_identity([2022, 2023])
    panel = build_development_panel(authority, targets, rosters)
    metrics, thresholds, advancing = base.score_development(panel)

    max_coverage = float(metrics["coverage"].max()) if len(metrics) else 0.0
    data_blocked = bool(max_coverage < base.MIN_COVERAGE)
    robustness = None
    if data_blocked:
        disposition = "WR_TARGET_DEPTH_DISTRIBUTION_DATA_BLOCKED"
        advancing = None
    elif advancing is None:
        disposition = "NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL"
    else:
        row = metrics.loc[metrics["signal"].eq(advancing)].iloc[0]
        robustness = base.role_endogeneity_robustness(
            panel, advancing, int(row["expected_direction"])
        )
        disposition = "WR_TARGET_DEPTH_DISTRIBUTION_DEVELOPMENT_SUPPORTED"

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "wr_r17_stage_a_feature_panel_2023.csv", index=False)
    metrics.to_csv(out / "wr_r17_stage_a_metrics_2023.csv", index=False)
    (out / "wr_r17_stage_a_thresholds_2023.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n"
    )
    (out / "wr_r17_stage_a_identity_audit.json").write_text(
        json.dumps(identity_audit(panel), indent=2, sort_keys=True) + "\n"
    )
    if robustness is not None:
        (out / "wr_r17_stage_a_role_endogeneity_robustness.json").write_text(
            json.dumps(robustness, indent=2, sort_keys=True) + "\n"
        )
    result = {
        "specification": "WR_R17_TARGET_DEPTH_DISTRIBUTION_V1",
        "implementation": "v1c_prior_roster_gsis_bridge",
        "stage": "A_2023_DEVELOPMENT_ONLY",
        "disposition": disposition,
        "advancing_signal": advancing,
        "max_signal_coverage": max_coverage,
        "signal_priority": base.SIGNAL_PRIORITY,
        "holdout_2024_scored": False,
        "role_mediated_warning": bool(robustness and robustness["role_mediated_warning"]),
        "authority_expected_rows": base.EXPECTED_ROWS,
        "supersedes_invalid_zero_coverage_run": 34897984434,
    }
    (out / "wr_r17_stage_a_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("=== WR-R17 STAGE A / 2023 ONLY / v1c identity bridge ===")
    print(metrics.to_string(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(identity_audit(panel), indent=2, sort_keys=True))
    print("2024 holdout was not scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
