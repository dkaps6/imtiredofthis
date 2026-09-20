#!/usr/bin/env python3
"""Outcome-free qualification for OL Roster Pairwise Cohesion V1.

Frozen plan:
docs/research/OL_ROSTER_PAIRWISE_COHESION_QUALIFICATION_V1.md

This script intentionally reuses the already-qualified OL weekly-roster identity
contract, but tests a different mechanism: accumulated shared roster history
among current OL pairs over up to 20 strictly prior scheduled team-games.

No predictive outcome, sportsbook field, target-game snap or participation is
used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.football_context import qualify_ol_roster_continuity_v1 as cont

CANDIDATE = "ol_roster_pairwise_cohesion_prior_share"
LOOKBACK_GAMES = 20
MIN_COVERAGE = 0.80
MIN_ELIGIBLE = 500
MIN_STABLE_ID = 0.99
MIN_STABILITY_PAIRS = 500
MIN_STABILITY_SPEARMAN = 0.50
REDUNDANCY_TRAIN_MIN = 1000
REDUNDANCY_HOLDOUT_MIN = 500
REDUNDANCY_HIGH = 0.90
REDUNDANCY_REVIEW = 0.75
PRIOR_TEAM_METRICS = list(cont.PRIOR_TEAM_METRICS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame({
        "a": pd.to_numeric(a, errors="coerce"),
        "b": pd.to_numeric(b, errors="coerce"),
    }).dropna()
    if len(z) < 3 or z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan
    return float(z["a"].corr(z["b"], method="spearman"))


def materialize_pairwise_cohesion(
    schedule: pd.DataFrame,
    roster_sets: dict[tuple[int, int, str], set[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    temporal_violations = 0

    ordered = schedule.sort_values(["team", "season", "week"]).copy()
    for team, g in ordered.groupby("team", sort=True):
        keys = [
            (int(r.season), int(r.week), str(team))
            for r in g.sort_values(["season", "week"]).itertuples(index=False)
        ]
        for idx, key in enumerate(keys):
            season, week, team_name = key
            current = roster_sets.get(key, set())
            prior_keys = keys[max(0, idx - LOOKBACK_GAMES):idx]
            if any((ps > season) or (ps == season and pw >= week) for ps, pw, _ in prior_keys):
                temporal_violations += 1

            available_prior = [
                roster_sets[k]
                for k in prior_keys
                if roster_sets.get(k, set())
            ]

            rec: dict[str, object] = {
                "season": season,
                "week": week,
                "team": team_name,
                "current_ol_roster_count": int(len(current)),
                "current_ol_pair_count": int(len(current) * (len(current) - 1) // 2),
                "prior_scheduled_games_in_window": int(len(prior_keys)),
                "prior_roster_games_available": int(len(available_prior)),
                "mean_prior_corostered_games": np.nan,
                "median_prior_corostered_games": np.nan,
                "zero_prior_coroster_pair_share": np.nan,
                CANDIDATE: np.nan,
                "cohesion_state": "",
            }

            if len(current) < 2:
                rec["cohesion_state"] = "UNKNOWN_CURRENT_ROSTER_LT2"
            elif not prior_keys:
                rec["cohesion_state"] = "UNKNOWN_NO_PRIOR_HISTORY"
            elif not available_prior:
                rec["cohesion_state"] = "UNKNOWN_PRIOR_ROSTER_HISTORY"
            else:
                pairs = list(itertools.combinations(sorted(current), 2))
                pair_counts = []
                for a, b in pairs:
                    n = sum(1 for prior in available_prior if a in prior and b in prior)
                    pair_counts.append(float(n))
                counts = np.asarray(pair_counts, dtype=float)
                shares = counts / float(len(available_prior))
                rec["mean_prior_corostered_games"] = float(counts.mean())
                rec["median_prior_corostered_games"] = float(np.median(counts))
                rec["zero_prior_coroster_pair_share"] = float(np.mean(counts == 0))
                rec[CANDIDATE] = float(shares.mean())
                rec["cohesion_state"] = "KNOWN"

            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["season", "team", "week"]).reset_index(drop=True)
    return out, {
        "published_duplicate_team_week_rows": int(
            out.duplicated(["season", "week", "team"], keep=False).sum()
        ),
        "chronology_violations": int(temporal_violations),
        "schedule_join_fanout": 0,
        "target_game_pbp_used_in_cohesion": False,
        "target_game_snap_or_participation_used": False,
        "future_week_roster_used": False,
    }


def attach_immediate_continuity(
    cohesion: pd.DataFrame,
    schedule: pd.DataFrame,
    roster_sets: dict[tuple[int, int, str], set[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    immediate, integrity = cont.materialize_continuity(schedule, roster_sets)
    keep = immediate[
        ["season", "week", "team", "prior_game_week", cont.CANDIDATE]
    ].copy()
    before = len(cohesion)
    out = cohesion.merge(
        keep,
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )
    return out, {
        "immediate_continuity_join_fanout": int(len(out) - before),
        "immediate_continuity_chronology_violations": int(integrity["chronology_violations"]),
    }


def stability_evidence(cohesion: pd.DataFrame) -> pd.DataFrame:
    z = cohesion[["season", "week", "team", CANDIDATE]].copy()
    z = z.sort_values(["team", "season", "week"])
    z["prior_value"] = z.groupby("team")[CANDIDATE].shift(1)
    valid = z[[CANDIDATE, "prior_value"]].dropna()
    delta = (valid[CANDIDATE] - valid["prior_value"]).abs()
    rho = _safe_spearman(valid["prior_value"], valid[CANDIDATE])
    passed = bool(
        len(valid) >= MIN_STABILITY_PAIRS
        and np.isfinite(rho)
        and rho >= MIN_STABILITY_SPEARMAN
    )
    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "stability_applicability": "HARD_GATE_ACCUMULATED_COHESION_STATE",
        "adjacent_game_pairs": int(len(valid)),
        "adjacent_game_spearman": float(rho) if np.isfinite(rho) else np.nan,
        "median_absolute_adjacent_change": float(delta.median()) if len(delta) else np.nan,
        "min_adjacent_game_pairs": MIN_STABILITY_PAIRS,
        "min_adjacent_game_spearman": MIN_STABILITY_SPEARMAN,
        "stability_gate_passed": passed,
    }])


def support_table(cohesion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", cohesion)]
    for season in sorted(cont.SEASONS):
        groups.append((f"SEASON_{season}", cohesion.loc[cohesion["season"].eq(season)]))
    for label, g in groups:
        known = g.loc[g[CANDIDATE].notna()].copy()
        rows.append({
            "slice": label,
            "eligible_team_games": int(len(g)),
            "known_rows": int(len(known)),
            "pregame_coverage": float(len(known) / len(g)) if len(g) else np.nan,
            "median_current_ol_pair_count": float(
                pd.to_numeric(known["current_ol_pair_count"], errors="coerce").median()
            ) if len(known) else np.nan,
            "median_prior_roster_games_available": float(
                pd.to_numeric(known["prior_roster_games_available"], errors="coerce").median()
            ) if len(known) else np.nan,
            "median_mean_prior_corostered_games": float(
                pd.to_numeric(known["mean_prior_corostered_games"], errors="coerce").median()
            ) if len(known) else np.nan,
            "median_zero_prior_coroster_pair_share": float(
                pd.to_numeric(known["zero_prior_coroster_pair_share"], errors="coerce").median()
            ) if len(known) else np.nan,
            "unknown_no_prior_history": int(g["cohesion_state"].eq("UNKNOWN_NO_PRIOR_HISTORY").sum()),
            "unknown_current_roster_lt2": int(g["cohesion_state"].eq("UNKNOWN_CURRENT_ROSTER_LT2").sum()),
            "unknown_prior_roster_history": int(g["cohesion_state"].eq("UNKNOWN_PRIOR_ROSTER_HISTORY").sum()),
        })
    return pd.DataFrame(rows)


def attach_prior_team_state(
    frame: pd.DataFrame,
    team_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    # Reuse the qualified V1 prior-team-state contract. Week 1 has no same-season
    # prior_game_week and remains missing here; train-only imputation handles it.
    out, integrity = cont.attach_prior_team_state(frame, team_weekly)
    return out, integrity


def _redundancy_design(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    prior_inputs = [f"prior_{m}" for m in PRIOR_TEAM_METRICS]
    cols = [
        "season", "week", "team", CANDIDATE, cont.CANDIDATE, *prior_inputs
    ]
    z = frame[cols].copy()
    z[CANDIDATE] = pd.to_numeric(z[CANDIDATE], errors="coerce")
    z = z.dropna(subset=[CANDIDATE])

    train = z.loc[z["season"].between(2019, 2023)].copy()
    holdout = z.loc[z["season"].between(2024, 2025)].copy()
    ntr, nte = len(train), len(holdout)
    if ntr < REDUNDANCY_TRAIN_MIN or nte < REDUNDANCY_HOLDOUT_MIN:
        return (
            np.empty((0, 0)), np.empty(0), np.empty((0, 0)), np.empty(0),
            int(ntr), int(nte),
        )

    train["team"] = train["team"].fillna("").astype(str).replace("", "__MISSING__")
    holdout["team"] = holdout["team"].fillna("").astype(str).replace("", "__MISSING__")
    teams = sorted(train["team"].unique().tolist())

    numeric = ["week", cont.CANDIDATE, *prior_inputs]
    for col in numeric:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        holdout[col] = pd.to_numeric(holdout[col], errors="coerce")
        med = float(train[col].median()) if train[col].notna().any() else 0.0
        train[col] = train[col].fillna(med)
        holdout[col] = holdout[col].fillna(med)

    def matrix(q: pd.DataFrame) -> np.ndarray:
        pieces = [np.ones((len(q), 1), dtype=float), q[numeric].to_numpy(float)]
        for team in teams:
            pieces.append(q["team"].eq(team).astype(float).to_numpy()[:, None])
        return np.hstack(pieces)

    return (
        matrix(train), train[CANDIDATE].to_numpy(float),
        matrix(holdout), holdout[CANDIDATE].to_numpy(float),
        int(ntr), int(nte),
    )


def redundancy_audit(frame: pd.DataFrame) -> pd.DataFrame:
    Xtr, ytr, Xte, yte, ntr, nte = _redundancy_design(frame)
    r2 = np.nan
    if ntr >= REDUNDANCY_TRAIN_MIN and nte >= REDUNDANCY_HOLDOUT_MIN:
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        pred = Xte @ beta
        sst = float(np.sum((yte - yte.mean()) ** 2))
        r2 = 1.0 - float(np.sum((yte - pred) ** 2)) / sst if sst > 0 else np.nan

    if not np.isfinite(r2):
        disposition = "REDUNDANCY_UNRESOLVED_SOURCE_THIN"
    elif r2 >= REDUNDANCY_HIGH:
        disposition = "HIGHLY_RECONSTRUCTIBLE_REDUNDANT"
    elif r2 >= REDUNDANCY_REVIEW:
        disposition = "REDUNDANCY_REVIEW"
    else:
        disposition = "INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE"

    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "reconstruction_inputs": (
            "ol_roster_continuity_share_prev_game|prior_pressure_rate_allowed|"
            "prior_success_rate_off|prior_dropback_rate|prior_plays_est|prior_proe|"
            "team_onehot|target_week"
        ),
        "train_seasons": "2019-2023",
        "holdout_seasons": "2024-2025",
        "train_rows": int(ntr),
        "holdout_rows": int(nte),
        "holdout_reconstructibility_r2": float(r2) if np.isfinite(r2) else np.nan,
        "redundancy_disposition": disposition,
        "target_game_pbp_used": False,
        "sportsbook_read": False,
    }])


def qualify(
    cohesion: pd.DataFrame,
    identity: dict[str, object],
    integrity: dict[str, int],
    stability: pd.DataFrame,
    redundancy: pd.DataFrame,
) -> pd.DataFrame:
    eligible = int(len(cohesion))
    known = int(cohesion[CANDIDATE].notna().sum())
    coverage = float(known / eligible) if eligible else 0.0
    stable = stability.iloc[0]
    rd = redundancy.iloc[0]

    hard_integrity = bool(
        float(identity["stable_id_coverage"]) >= MIN_STABLE_ID
        and int(identity["ambiguous_same_week_gsis_team_conflicts"]) == 0
        and int(integrity["published_duplicate_team_week_rows"]) == 0
        and int(integrity["chronology_violations"]) == 0
        and int(integrity["schedule_join_fanout"]) == 0
        and int(integrity["immediate_continuity_join_fanout"]) == 0
        and int(integrity["immediate_continuity_chronology_violations"]) == 0
        and int(integrity["redundancy_join_fanout"]) == 0
    )
    red = str(rd["redundancy_disposition"])

    if not hard_integrity:
        disposition = "REJECTED_INTEGRITY"
        reason = "stable identity, duplicate, fanout or chronology integrity gate failed"
    elif eligible < MIN_ELIGIBLE or coverage < MIN_COVERAGE:
        disposition = "ENGINEERING_READY_SOURCE_THIN"
        reason = "broad scheduled-team-game coverage or sample support below frozen floor"
    elif not bool(stable["stability_gate_passed"]):
        disposition = "DESCRIPTIVE_ONLY"
        reason = "accumulated cohesion did not pass frozen adjacent-game stability gate"
    elif red == "REDUNDANCY_UNRESOLVED_SOURCE_THIN":
        disposition = "ENGINEERING_READY_SOURCE_THIN"
        reason = "redundancy audit lacks frozen train/holdout support"
    elif red in {"HIGHLY_RECONSTRUCTIBLE_REDUNDANT", "REDUNDANCY_REVIEW"}:
        disposition = "DESCRIPTIVE_ONLY"
        reason = "candidate is too reconstructible from immediate continuity plus prior team state"
    else:
        disposition = "READY_FOR_FROZEN_EXPERIMENT"
        reason = "identity, coverage, stability, mechanism and redundancy gates satisfied"

    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "family": "OL_ROSTER_PAIRWISE_COHESION_V1",
        "grain": "scheduled_team_game",
        "seasons_available": "2019-2025",
        "eligible_rows": eligible,
        "observed_rows": known,
        "pregame_coverage": coverage,
        "stable_id_coverage": float(identity["stable_id_coverage"]),
        "unknown_rate": float(1.0 - coverage),
        "duplicate_key_count": int(integrity["published_duplicate_team_week_rows"]),
        "fanout_count": int(
            integrity["schedule_join_fanout"]
            + integrity["immediate_continuity_join_fanout"]
            + integrity["redundancy_join_fanout"]
        ),
        "stability_stat": "adjacent_game_spearman",
        "stability_value": stable["adjacent_game_spearman"],
        "stability_gate_passed": bool(stable["stability_gate_passed"]),
        "intended_component": (
            "OL accumulated cohesion / protection and run-blocking environment uncertainty"
        ),
        "mechanism_note": (
            "mean prior co-roster share of every current OL pair over up to 20 strictly "
            "prior scheduled team-games; distinct from last-game turnover counts"
        ),
        "source_class": "NFLVERSE_WEEKLY_ROSTER_CANONICAL_EXISTING_SOURCE",
        "redundancy_disposition": red,
        "holdout_reconstructibility_r2": rd["holdout_reconstructibility_r2"],
        "qualification_disposition": disposition,
        "qualification_reason": reason,
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--git-sha", required=True)
    args = ap.parse_args()

    schedule = cont.normalize_schedule(pd.read_csv(args.schedule, low_memory=False))
    roster_raw = pd.read_csv(args.roster, low_memory=False)
    team_weekly = pd.read_csv(args.team_weekly, low_memory=False)

    roster_sets, identity = cont.build_roster_sets(roster_raw, schedule)
    cohesion, cohesion_integrity = materialize_pairwise_cohesion(schedule, roster_sets)
    with_cont, immediate_integrity = attach_immediate_continuity(cohesion, schedule, roster_sets)
    enriched, red_integrity = attach_prior_team_state(with_cont, team_weekly)
    integrity = {**cohesion_integrity, **immediate_integrity, **red_integrity}

    stability = stability_evidence(cohesion)
    support = support_table(cohesion)
    redundancy = redundancy_audit(enriched)
    inventory = qualify(cohesion, identity, integrity, stability, redundancy)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    support.to_csv(out / "ol_pairwise_cohesion_support_v1.csv", index=False)
    stability.to_csv(out / "ol_pairwise_cohesion_stability_v1.csv", index=False)
    redundancy.to_csv(out / "ol_pairwise_cohesion_redundancy_v1.csv", index=False)
    inventory.to_csv(out / "ol_pairwise_cohesion_qualification_inventory_v1.csv", index=False)

    manifest = {
        "qualification_version": "OL_ROSTER_PAIRWISE_COHESION_QUALIFICATION_V1",
        "frozen_pre_result_disposition": "OL_ROSTER_PAIRWISE_COHESION_QUALIFICATION_V1_FROZEN_PRE_RESULT",
        "git_sha": args.git_sha,
        "candidate": CANDIDATE,
        "lookback_scheduled_team_games": LOOKBACK_GAMES,
        "formula": (
            "mean_current_ol_pair(prior_games_both_on_team_ol_roster/"
            "available_strictly_prior_team_games)"
        ),
        "schedule_sha256": _sha256(args.schedule),
        "weekly_roster_sha256": _sha256(args.roster),
        "team_weekly_sha256": _sha256(args.team_weekly),
        "identity": identity,
        "integrity": integrity,
        "seasons": sorted(cont.SEASONS),
        "backups_retained": True,
        "cross_season_prior_history_allowed": True,
        "target_game_snap_or_participation_used": False,
        "target_game_pbp_used": False,
        "future_week_roster_used": False,
        "predictive_outcomes_scored": False,
        "sportsbook_read": False,
        "production_changed": False,
        "issue_535_touched": False,
        "qualification_min_coverage": MIN_COVERAGE,
        "qualification_min_rows": MIN_ELIGIBLE,
        "qualification_min_stable_id_coverage": MIN_STABLE_ID,
        "stability_min_pairs": MIN_STABILITY_PAIRS,
        "stability_min_spearman": MIN_STABILITY_SPEARMAN,
        "redundancy_train_seasons": "2019-2023",
        "redundancy_holdout_seasons": "2024-2025",
        "redundancy_high_r2": REDUNDANCY_HIGH,
        "redundancy_review_r2": REDUNDANCY_REVIEW,
        "qualification_disposition": str(inventory.iloc[0]["qualification_disposition"]),
    }
    (out / "ol_pairwise_cohesion_qualification_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("QUALIFICATION")
    print(inventory.to_string(index=False))
    print("\nBROAD SUPPORT")
    print(support.loc[support["slice"].eq("ALL")].to_string(index=False))
    print("\nSTABILITY")
    print(stability.to_string(index=False))
    print("\nREDUNDANCY")
    print(redundancy.to_string(index=False))
    print("\nMANIFEST")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
