#!/usr/bin/env python3
"""R26J source-only audit of 2020 Week-1 RB-room comparability.

No target-game outcomes/errors are loaded. The audit compares pregame room-transition,
strict-prior exited-player receiving state, and frozen pregame model-state across
2020-2025 Week-1 vacancy rooms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = tuple(range(2020, 2026))
EPS = 1e-12
RELATIVE_REFERENCE_FLOOR = 0.05

R26_USECOLS = [
    "season", "week", "team", "player_clean_key",
    "vacancy_active", "room_exits_n", "room_entrants_n",
    "continuing_same_team", "new_to_team_veteran", "no_prior_nfl_roster",
    "prior_depth_available", "baseline_room_share", "candidate_room_share",
    "r9_reliability", "sportsbook_inputs_used", "future_outcomes_used",
]

# Section / rate declarations are frozen implementation state.
FEATURE_SECTION = {
    # A room continuity / turnover
    "current_room_n": "A",
    "prior_room_n": "A",
    "exits_n": "A",
    "entrants_n": "A",
    "balanced_turnover_flag": "A",
    "abs_room_size_change": "A",
    "continuing_n": "A",
    "continuing_share": "A",
    "entrant_share": "A",
    # B entrant composition
    "veteran_entry_n": "B",
    "veteran_entry_share": "B",
    "no_prior_entry_n": "B",
    "no_prior_entry_share": "B",
    "both_entry_classes_flag": "B",
    "unresolved_entry_room_flag": "B",
    # C vacated receiving significance
    "exit_history_coverage": "C",
    "exit_positive_history_n": "C",
    "max_exit_prior_targets_pg": "C",
    "sum_exit_prior_targets_pg": "C",
    "max_exit_prior_rb_room_share": "C",
    "sum_exit_prior_rb_room_share": "C",
    "max_exit_last8_targets_pg": "C",
    "sum_exit_last8_targets_pg": "C",
    "meaningful_exit_n": "C",
    "multiple_meaningful_exit_flag": "C",
    # D returning/model-state structure
    "baseline_room_hhi": "D",
    "baseline_top_room_share": "D",
    "incumbent_n": "D",
    "r9_reliability": "D",
    "r26_allocation_shift_l1": "D",
    # E source quality diagnostics
    "prior_depth_room_coverage": "E",
    "entrant_state_resolution": "E",
}

RATE_FEATURES = {
    "balanced_turnover_flag", "continuing_share", "entrant_share",
    "veteran_entry_share", "no_prior_entry_share", "both_entry_classes_flag",
    "unresolved_entry_room_flag", "exit_history_coverage",
    "multiple_meaningful_exit_flag", "prior_depth_room_coverage",
    "entrant_state_resolution",
}


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def r26_files(root: Path) -> list[Path]:
    out = sorted(root.rglob("r26_predictions.csv"))
    if len(out) != 6:
        raise RuntimeError(f"expected exactly 6 R26 prediction files, found {len(out)}")
    return out


def read_r26_pregame(root: Path) -> pd.DataFrame:
    parts = []
    for p in r26_files(root):
        header = pd.read_csv(p, nrows=0)
        missing = [c for c in R26_USECOLS if c not in header.columns]
        if missing:
            raise RuntimeError(f"R26 pregame columns missing in {p}: {missing}")
        # usecols physically prevents actual_* outcome fields from entering memory/frame.
        x = pd.read_csv(p, usecols=R26_USECOLS, low_memory=False)
        if any(str(c).startswith("actual_") for c in x.columns):
            raise RuntimeError("R26J loaded prohibited actual_* field")
        parts.append(x)
    x = pd.concat(parts, ignore_index=True, sort=False)
    x["season"] = num(x.season).astype(int)
    x["week"] = num(x.week).astype(int)
    for c in [
        "vacancy_active", "room_exits_n", "room_entrants_n", "continuing_same_team",
        "new_to_team_veteran", "no_prior_nfl_roster", "prior_depth_available",
        "sportsbook_inputs_used", "future_outcomes_used",
    ]:
        x[c] = num(x[c]).fillna(0)
    for c in ["baseline_room_share", "candidate_room_share", "r9_reliability"]:
        x[c] = num(x[c])
    return x


def read_one_csv(root: Path, name: str) -> pd.DataFrame:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return pd.read_csv(hits[0], low_memory=False)


def read_one_json(root: Path, name: str) -> dict:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return json.loads(hits[0].read_text())


def build_r26_rooms(pred: pd.DataFrame) -> pd.DataFrame:
    x = pred.loc[pred.season.isin(SEASONS) & pred.week.eq(1) & pred.vacancy_active.eq(1)].copy()
    if x.empty:
        raise RuntimeError("R26J found zero Week-1 vacancy rows")
    keys = ["season", "week", "team"]
    state_cols = ["room_exits_n", "room_entrants_n", "vacancy_active", "r9_reliability"]
    nu = x.groupby(keys)[state_cols].nunique(dropna=False)
    if (nu > 1).any().any():
        raise RuntimeError("R26J inconsistent room-state values inside team-week")

    rows = []
    for k, g in x.groupby(keys, sort=True):
        cur_n = int(len(g))
        continuing_n = int(num(g.continuing_same_team).eq(1).sum())
        entrants_n = int(num(g.room_entrants_n).iloc[0])
        vet_n = int(num(g.new_to_team_veteran).eq(1).sum())
        nop_n = int(num(g.no_prior_nfl_roster).eq(1).sum())
        resolved_entry_n = min(entrants_n, vet_n + nop_n)
        bshare = num(g.baseline_room_share).fillna(0.0).clip(lower=0.0)
        cshare = num(g.candidate_room_share).fillna(0.0).clip(lower=0.0)
        rows.append({
            "season": int(k[0]), "week": int(k[1]), "team": str(k[2]),
            "current_room_n": cur_n,
            "exits_n_r26": int(num(g.room_exits_n).iloc[0]),
            "entrants_n": entrants_n,
            "balanced_turnover_flag": int(int(num(g.room_exits_n).iloc[0]) == entrants_n),
            "continuing_n": continuing_n,
            "continuing_share": continuing_n / max(cur_n, 1),
            "entrant_share": entrants_n / max(cur_n, 1),
            "veteran_entry_n": vet_n,
            "veteran_entry_share": vet_n / max(cur_n, 1),
            "no_prior_entry_n": nop_n,
            "no_prior_entry_share": nop_n / max(cur_n, 1),
            "both_entry_classes_flag": int(vet_n > 0 and nop_n > 0),
            "unresolved_entry_room_flag": int(entrants_n > 0 and resolved_entry_n == 0),
            "entrant_state_resolution": resolved_entry_n / max(entrants_n, 1),
            "baseline_room_hhi": float(np.square(bshare.to_numpy(float)).sum()),
            "baseline_top_room_share": float(bshare.max()) if len(bshare) else np.nan,
            "incumbent_n": continuing_n,
            "r9_reliability": float(num(g.r9_reliability).iloc[0]),
            "r26_allocation_shift_l1": float(np.abs(cshare.to_numpy(float) - bshare.to_numpy(float)).sum()),
            "sportsbook_inputs_used": int(num(g.sportsbook_inputs_used).fillna(0).sum()),
            "future_outcomes_used": int(num(g.future_outcomes_used).fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def meaningful_exit_count(exits: pd.DataFrame) -> pd.DataFrame:
    z = exits.copy()
    for c in ["season", "week", "prior_games", "prior_targets_pg", "prior_rb_room_share"]:
        z[c] = num(z[c])
    z = z.loc[z.season.isin(SEASONS) & z.week.eq(1)].copy()
    z["positive_history"] = z.prior_games.gt(0).astype(int)
    z["meaningful_exit"] = (
        z.prior_games.gt(0)
        & (z.prior_targets_pg.gt(1.0) | z.prior_rb_room_share.ge(.25))
    ).astype(int)
    keys = ["season", "week", "team"]
    out = z.groupby(keys, as_index=False).agg(
        exit_player_rows=("player_key", "count"),
        exit_positive_history_n=("positive_history", "sum"),
        meaningful_exit_n=("meaningful_exit", "sum"),
        prior_depth_room_coverage=("prior_depth_available", lambda s: float(num(s).fillna(0).eq(1).mean())),
    )
    out["multiple_meaningful_exit_flag"] = out.meaningful_exit_n.ge(2).astype(int)
    return out


def build_rooms(r26: pd.DataFrame, tw: pd.DataFrame, exits: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    room = build_r26_rooms(r26)
    tw = tw.copy()
    tw["season"] = num(tw.season).astype(int)
    tw["week"] = num(tw.week).astype(int)
    tw = tw.loc[tw.season.isin(SEASONS) & tw.week.eq(1)].copy()
    keys = ["season", "week", "team"]
    if tw.duplicated(keys).any():
        raise RuntimeError("R26C teamweek aggregate has duplicate Week-1 keys")

    parent_keys = set(map(tuple, room[keys].astype({"season": int, "week": int, "team": str}).to_numpy()))
    src_keys = set(map(tuple, tw[keys].astype({"season": int, "week": int, "team": str}).to_numpy()))
    reconstruction = len(parent_keys & src_keys) / max(len(parent_keys), 1)

    want = [
        *keys,
        "prior_rb_room_n", "current_rb_room_n", "exits_n", "exit_history_coverage",
        "max_exit_prior_targets_pg", "sum_exit_prior_targets_pg",
        "max_exit_prior_rb_room_share", "sum_exit_prior_rb_room_share",
        "max_exit_last8_targets_pg", "sum_exit_last8_targets_pg",
    ]
    missing = [c for c in want if c not in tw.columns]
    if missing:
        raise RuntimeError(f"R26C teamweek fields missing: {missing}")
    room = room.merge(tw[want], on=keys, how="left", validate="one_to_one")

    ex = meaningful_exit_count(exits)
    room = room.merge(ex, on=keys, how="left", validate="one_to_one")
    if room.prior_rb_room_n.isna().any():
        raise RuntimeError("R26J missing R26C source state for R26 vacancy room")

    room["prior_room_n"] = num(room.prior_rb_room_n)
    room["abs_room_size_change"] = (num(room.current_room_n) - room.prior_room_n).abs()
    room["exits_n"] = num(room.exits_n)
    # exact source/state consistency check
    if not num(room.exits_n_r26).eq(room.exits_n).all():
        raise RuntimeError("R26 vs R26C exits_n mismatch")
    room["exit_history_coverage"] = num(room.exit_history_coverage)
    room["exit_positive_history_n"] = num(room.exit_positive_history_n).fillna(0)
    room["meaningful_exit_n"] = num(room.meaningful_exit_n).fillna(0)
    room["multiple_meaningful_exit_flag"] = num(room.multiple_meaningful_exit_flag).fillna(0)
    room["prior_depth_room_coverage"] = num(room.prior_depth_room_coverage).fillna(0)
    return room, reconstruction


def season_summary(room: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, section in FEATURE_SECTION.items():
        if feature not in room.columns:
            raise RuntimeError(f"R26J missing predeclared feature: {feature}")
        for season in SEASONS:
            v = num(room.loc[room.season.eq(season), feature]).dropna()
            rows.append({
                "section": section,
                "feature": feature,
                "is_rate": feature in RATE_FEATURES,
                "season": season,
                "n": int(len(v)),
                "mean": float(v.mean()) if len(v) else np.nan,
                "median": float(v.median()) if len(v) else np.nan,
                "p25": float(v.quantile(.25)) if len(v) else np.nan,
                "p75": float(v.quantile(.75)) if len(v) else np.nan,
            })
    return pd.DataFrame(rows)


def distinctions(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, g in summary.groupby("feature", sort=True):
        g20 = g.loc[g.season.eq(2020)]
        gp = g.loc[g.season.ge(2021)]
        if len(g20) != 1 or len(gp) != 5:
            raise RuntimeError(f"R26J incomplete season summary for {feature}")
        v20 = float(g20.iloc[0]["mean"])
        vals = num(gp["mean"]).dropna().to_numpy(float)
        ref = float(np.mean(vals)) if len(vals) else np.nan
        lo = float(np.min(vals)) if len(vals) else np.nan
        hi = float(np.max(vals)) if len(vals) else np.nan
        section = str(g20.iloc[0]["section"])
        is_rate = bool(g20.iloc[0]["is_rate"])
        range_sep = bool(np.isfinite(v20) and np.isfinite(lo) and (v20 < lo - EPS or v20 > hi + EPS))
        rel_shift = bool(
            np.isfinite(v20) and np.isfinite(ref)
            and abs(ref) >= RELATIVE_REFERENCE_FLOOR
            and abs(v20 - ref) / abs(ref) >= .25
        )
        rate_shift = bool(is_rate and np.isfinite(v20) and np.isfinite(ref) and abs(v20 - ref) >= .15)
        source_integrity_shift = bool(section == "E" and np.isfinite(v20) and np.isfinite(ref) and abs(v20 - ref) >= .10)
        distinct = bool(range_sep or rel_shift or rate_shift or source_integrity_shift)
        rows.append({
            "section": section,
            "feature": feature,
            "is_rate": is_rate,
            "mean_2020": v20,
            "mean_2021_2025": ref,
            "min_2021_2025": lo,
            "max_2021_2025": hi,
            "absolute_difference": float(v20 - ref) if np.isfinite(v20) and np.isfinite(ref) else np.nan,
            "relative_difference": float((v20 - ref) / abs(ref)) if np.isfinite(v20) and np.isfinite(ref) and abs(ref) >= RELATIVE_REFERENCE_FLOOR else np.nan,
            "range_separation": range_sep,
            "large_relative_shift": rel_shift,
            "large_absolute_rate_shift": rate_shift,
            "source_integrity_shift": source_integrity_shift,
            "structurally_distinct": distinct,
            "counts_toward_A_D_disposition": bool(distinct and section in {"A", "B", "C", "D"}),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--r26c-root", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    if not a.protected_clean_marker.exists() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26J protected-production clean marker missing")

    r26 = read_r26_pregame(a.r26_root)
    cdisp = read_one_json(a.r26c_root, "r26c_source_disposition.json")
    if cdisp.get("disposition") != "EXIT_SIGNIFICANCE_SOURCE_READY":
        raise RuntimeError("R26J R26C parent source disposition mismatch")
    if cdisp.get("allowed_roster_status") != ["ACT", "INA"]:
        raise RuntimeError("R26J R26C roster-status contract mismatch")
    if cdisp.get("same_week_historical_depth_used") is not False:
        raise RuntimeError("R26J same-week historical depth contract violated")
    if int(cdisp.get("target_game_outcomes_selected_or_used", 1)) != 0:
        raise RuntimeError("R26J R26C outcome-free contract violated")
    if int(cdisp.get("target_game_participation_selected_or_used", 1)) != 0:
        raise RuntimeError("R26J R26C participation-free contract violated")
    if int(cdisp.get("sportsbook_inputs_used", 1)) != 0:
        raise RuntimeError("R26J R26C sportsbook-free contract violated")

    tw = read_one_csv(a.r26c_root, "r26c_vacancy_teamweek_aggregate.csv")
    exits = read_one_csv(a.r26c_root, "r26c_exited_player_source_state.csv")
    room, reconstruction = build_rooms(r26, tw, exits)
    if not room.week.eq(1).all() or not room.season.isin(SEASONS).all():
        raise RuntimeError("R26J non-Week-1/target-season room survived")

    # Explicit leakage/upstream checks from only loaded fields.
    selected_actual_fields = [c for c in room.columns if str(c).startswith("actual_")]
    sportsbook = int(num(r26.sportsbook_inputs_used).fillna(0).sum())
    future = int(num(r26.future_outcomes_used).fillna(0).sum())
    duplicate_room_keys = int(room.duplicated(["season", "week", "team"]).sum())

    summary = season_summary(room)
    dist = distinctions(summary)
    ad = dist.loc[dist.counts_toward_A_D_disposition].copy()
    independent_dims = int(ad.feature.nunique())
    independent_sections = int(ad.section.nunique())

    integrity_gates = {
        "01_parent_digests_verified_by_workflow": True,
        "02_week1_vacancy_population_only": bool(room.week.eq(1).all()),
        "03_zero_selected_actual_fields": len(selected_actual_fields) == 0,
        "04_target_game_outcome_participation_zero": bool(
            int(cdisp.get("target_game_outcomes_selected_or_used", 1)) == 0
            and int(cdisp.get("target_game_participation_selected_or_used", 1)) == 0
            and future == 0
        ),
        "05_sportsbook_zero": sportsbook == 0,
        "06_same_week_historical_depth_false": cdisp.get("same_week_historical_depth_used") is False,
        "07_canonical_act_ina_contract": cdisp.get("allowed_roster_status") == ["ACT", "INA"],
        "08_room_keys_unique": duplicate_room_keys == 0,
        "09_r26c_vacancy_reconstruction_ge_99pct": reconstruction >= .99,
        "10_protected_production_files_clean": True,
    }
    integrity_ok = bool(all(integrity_gates.values()))
    distinct = bool(integrity_ok and independent_dims >= 3 and independent_sections >= 2)
    disposition = (
        "2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP"
        if distinct else "2020_SOURCE_REGIME_NOT_DISTINCT_ENOUGH_FOR_EXEMPTION"
    )

    result = {
        "candidate": "RB_R26J_2020_WEEK1_COMPARABILITY_SOURCE_AUDIT_V1",
        "scientific_label": "SOURCE_ONLY_COMPARABILITY_AUDIT_NO_OUTCOMES",
        "disposition": disposition,
        "integrity_gates": integrity_gates,
        "all_integrity_gates_pass": integrity_ok,
        "week1_vacancy_rooms": int(len(room)),
        "season_room_counts": {str(s): int(room.season.eq(s).sum()) for s in SEASONS},
        "r26c_reconstruction_rate": float(reconstruction),
        "selected_actual_fields": selected_actual_fields,
        "sportsbook_inputs_used": sportsbook,
        "future_outcome_features_used": future,
        "independent_structurally_distinct_A_D_dimensions": independent_dims,
        "independent_A_D_sections_with_distinction": independent_sections,
        "distinct_A_D_features": ad[["section", "feature"]].drop_duplicates().to_dict("records"),
        "outcome_mechanism_followup_authorized": distinct,
        "exclude_2020_authorized": False,
        "prospective_shadow_authorized": False,
        "production_promotion_authorized": False,
        "production_parameters_changed": False,
        "r9_refit": False,
        "predictions_regenerated": False,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    room.to_csv(a.out_dir / "r26j_week1_vacancy_room_source_state.csv", index=False)
    summary.to_csv(a.out_dir / "r26j_season_source_summary.csv", index=False)
    dist.to_csv(a.out_dir / "r26j_2020_distinction_atlas.csv", index=False)
    (a.out_dir / "r26j_source_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(summary.to_csv(index=False))
    print("=== 2020 distinctions ===")
    print(dist.to_csv(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
