#!/usr/bin/env python3
"""Outcome-free qualification for OL Roster Continuity V1.

Frozen plan: docs/research/OL_ROSTER_CONTINUITY_QUALIFICATION_V1.md

The primary candidate is current pregame OL-roster identity overlap with the same
team's previous scheduled regular-season game. Target-game snaps, participation,
PBP and predictive outcomes are forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

SEASONS = set(range(2019, 2026))
OL_POSITIONS = {"C", "G", "T", "OT", "OG", "OL"}
OL_DEPTH_POSITIONS = {"LT", "LG", "C", "RG", "RT", "OT", "OG", "OL", "G", "T"}
CANDIDATE = "ol_roster_continuity_share_prev_game"
MIN_COVERAGE = 0.80
MIN_ELIGIBLE = 500
MIN_STABLE_ID = 0.99
REDUNDANCY_TRAIN_MIN = 1000
REDUNDANCY_HOLDOUT_MIN = 500
REDUNDANCY_HIGH = 0.90
REDUNDANCY_REVIEW = 0.75
PRIOR_TEAM_METRICS = [
    "pressure_rate_allowed",
    "success_rate_off",
    "dropback_rate",
    "plays_est",
    "proe",
]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _norm_pos(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z]", "", str(value).upper().strip())


def _norm_gsis(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


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
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b, method="spearman"))


def normalize_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    s = _lower(schedule)
    required = {"season", "week", "team"}
    missing = required - set(s.columns)
    if missing:
        raise RuntimeError(f"schedule missing columns {sorted(missing)}")
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s["team"] = s["team"].map(canon_team)
    s = s.loc[s["season"].isin(SEASONS) & s["week"].gt(0)].copy()
    out = s[["season", "week", "team"]].drop_duplicates().copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate scheduled team-week rows")
    return out.sort_values(["season", "team", "week"]).reset_index(drop=True)


def build_roster_sets(
    roster: pd.DataFrame,
    schedule: pd.DataFrame,
) -> tuple[dict[tuple[int, int, str], set[str]], dict[str, object]]:
    r = _lower(roster)
    required = ["season", "week"]
    for c in required:
        if c not in r.columns:
            raise RuntimeError(f"weekly roster missing {c}")
    team_col = _first_col(r, ["team", "team_abbr", "club_code"])
    gsis_col = _first_col(r, ["gsis_id", "player_id", "player_gsis_id"])
    pos_col = _first_col(r, ["position", "pos"])
    depth_col = _first_col(r, ["depth_chart_position", "depth_position"])
    if not team_col or not gsis_col or not pos_col:
        raise RuntimeError("weekly roster missing team/GSIS/position")

    r["season"] = pd.to_numeric(r["season"], errors="coerce")
    r["week"] = pd.to_numeric(r["week"], errors="coerce")
    r["team"] = r[team_col].map(canon_team)
    r["gsis_id"] = r[gsis_col].map(_norm_gsis)
    r["position_norm"] = r[pos_col].map(_norm_pos)
    r["depth_position_norm"] = r[depth_col].map(_norm_pos) if depth_col else ""
    eligible = (
        r["position_norm"].isin(OL_POSITIONS)
        | r["depth_position_norm"].isin(OL_DEPTH_POSITIONS)
    )
    r = r.loc[
        r["season"].isin(SEASONS)
        & r["week"].gt(0)
        & eligible
    ].copy()

    sched_keys = schedule[["season", "week", "team"]].drop_duplicates()
    before = len(r)
    r = r.merge(sched_keys, on=["season", "week", "team"], how="inner", validate="many_to_one")
    source_rows_on_scheduled_games = int(len(r))

    stable = r["gsis_id"].ne("")
    stable_id_coverage = float(stable.mean()) if len(r) else 0.0

    valid = r.loc[stable].copy()
    conflicts = (
        valid.groupby(["season", "week", "gsis_id"])["team"]
        .nunique()
    )
    ambiguous_same_week_gsis = int((conflicts > 1).sum())

    # Duplicated source rows for the same team/week/stable person do not create
    # fanout because the state is a set of identities.
    duplicate_identity_rows = int(
        valid.duplicated(["season", "week", "team", "gsis_id"], keep=False).sum()
    )
    valid = valid.drop_duplicates(["season", "week", "team", "gsis_id"])

    sets: dict[tuple[int, int, str], set[str]] = {}
    for (season, week, team), g in valid.groupby(["season", "week", "team"], sort=False):
        sets[(int(season), int(week), str(team))] = set(g["gsis_id"].astype(str))

    return sets, {
        "source_ol_rows_before_schedule_filter": int(before),
        "source_ol_rows_on_scheduled_games": source_rows_on_scheduled_games,
        "stable_id_coverage": stable_id_coverage,
        "ambiguous_same_week_gsis_team_conflicts": ambiguous_same_week_gsis,
        "duplicate_source_identity_rows_collapsed_to_set": duplicate_identity_rows,
        "depth_chart_position_available": bool(depth_col),
        "target_game_snap_or_participation_used": False,
        "name_fallback_used": False,
    }


def materialize_continuity(
    schedule: pd.DataFrame,
    roster_sets: dict[tuple[int, int, str], set[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    temporal_violations = 0
    for (season, team), g in schedule.groupby(["season", "team"], sort=True):
        weeks = sorted(g["week"].astype(int).tolist())
        prior_week = None
        for week in weeks:
            current = roster_sets.get((int(season), int(week), str(team)), set())
            rec = {
                "season": int(season),
                "week": int(week),
                "team": str(team),
                "prior_game_week": int(prior_week) if prior_week is not None else np.nan,
                "current_ol_roster_count": int(len(current)),
                "prior_game_ol_roster_count": np.nan,
                "returning_ol_count": np.nan,
                "added_ol_count": np.nan,
                "departed_ol_count": np.nan,
                CANDIDATE: np.nan,
                "continuity_state": "",
            }
            if prior_week is None:
                rec["continuity_state"] = "UNKNOWN_NO_PRIOR_GAME"
            else:
                if int(prior_week) >= int(week):
                    temporal_violations += 1
                prior = roster_sets.get((int(season), int(prior_week), str(team)), set())
                rec["prior_game_ol_roster_count"] = int(len(prior))
                if not current:
                    rec["continuity_state"] = "UNKNOWN_CURRENT_ROSTER"
                elif not prior:
                    rec["continuity_state"] = "UNKNOWN_PRIOR_ROSTER"
                else:
                    returning = len(current & prior)
                    rec["returning_ol_count"] = int(returning)
                    rec["added_ol_count"] = int(len(current - prior))
                    rec["departed_ol_count"] = int(len(prior - current))
                    rec[CANDIDATE] = float(returning / len(current))
                    rec["continuity_state"] = "KNOWN"
            rows.append(rec)
            prior_week = week

    out = pd.DataFrame(rows).sort_values(["season", "team", "week"]).reset_index(drop=True)
    dup = int(out.duplicated(["season", "week", "team"], keep=False).sum())
    return out, {
        "published_duplicate_team_week_rows": dup,
        "chronology_violations": int(temporal_violations),
        "schedule_join_fanout": 0,
        "target_game_pbp_used_in_continuity": False,
        "future_week_roster_used": False,
    }


def stability_diagnostic(continuity: pd.DataFrame) -> pd.DataFrame:
    z = continuity[["season", "week", "team", CANDIDATE]].copy()
    z = z.sort_values(["season", "team", "week"])
    z["prior_value"] = z.groupby(["season", "team"])[CANDIDATE].shift(1)
    valid = z[[CANDIDATE, "prior_value"]].dropna()
    delta = (valid[CANDIDATE] - valid["prior_value"]).abs()
    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "stability_applicability": "DIAGNOSTIC_ONLY_DIRECT_PREGAME_CHANGE_STATE",
        "adjacent_game_pairs": int(len(valid)),
        "adjacent_game_spearman": _safe_spearman(valid["prior_value"], valid[CANDIDATE]),
        "median_absolute_adjacent_change": float(delta.median()) if len(delta) else np.nan,
        "hard_qualification_gate": False,
    }])


def attach_prior_team_state(
    continuity: pd.DataFrame,
    team_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    t = _lower(team_weekly)
    required = {"season", "week", "team", *PRIOR_TEAM_METRICS}
    missing = required - set(t.columns)
    if missing:
        raise RuntimeError(f"team weekly state missing columns: {sorted(missing)}")
    t["season"] = pd.to_numeric(t["season"], errors="coerce")
    t["week"] = pd.to_numeric(t["week"], errors="coerce")
    t["team"] = t["team"].map(canon_team)
    if t.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("team weekly state has duplicate team-week rows")

    prior = t[["season", "week", "team", *PRIOR_TEAM_METRICS]].copy()
    prior = prior.rename(
        columns={
            "week": "prior_game_week",
            **{m: f"prior_{m}" for m in PRIOR_TEAM_METRICS},
        }
    )
    before = len(continuity)
    out = continuity.merge(
        prior,
        on=["season", "prior_game_week", "team"],
        how="left",
        validate="many_to_one",
    )
    return out, {
        "redundancy_join_fanout": int(len(out) - before),
        "target_game_team_state_used": False,
    }


def _redundancy_design(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    inputs = [f"prior_{m}" for m in PRIOR_TEAM_METRICS]
    z = frame[["season", "week", "team", CANDIDATE, *inputs]].copy()
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

    numeric = ["week", *inputs]
    for col in numeric:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        holdout[col] = pd.to_numeric(holdout[col], errors="coerce")
        med = float(train[col].median()) if train[col].notna().any() else 0.0
        train[col] = train[col].fillna(med)
        holdout[col] = holdout[col].fillna(med)

    def matrix(q: pd.DataFrame) -> np.ndarray:
        pieces = [np.ones((len(q), 1), dtype=float)]
        pieces.append(q[numeric].to_numpy(float))
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
            "prior_pressure_rate_allowed|prior_success_rate_off|prior_dropback_rate|"
            "prior_plays_est|prior_proe|team_onehot|target_week"
        ),
        "train_seasons": "2019-2023",
        "holdout_seasons": "2024-2025",
        "train_rows": ntr,
        "holdout_rows": nte,
        "holdout_reconstructibility_r2": float(r2) if np.isfinite(r2) else np.nan,
        "redundancy_disposition": disposition,
        "target_game_pbp_used": False,
        "sportsbook_read": False,
    }])


def coverage_table(continuity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", continuity)]
    for season in sorted(SEASONS):
        groups.append((f"SEASON_{season}", continuity.loc[continuity["season"].eq(season)]))
    for week in sorted(continuity["week"].unique()):
        groups.append((f"WEEK_{int(week)}", continuity.loc[continuity["week"].eq(week)]))
    for label, g in groups:
        known = int(g[CANDIDATE].notna().sum())
        rows.append({
            "slice": label,
            "eligible_team_games": int(len(g)),
            "known_rows": known,
            "pregame_coverage": float(known / len(g)) if len(g) else np.nan,
            "unknown_no_prior_game": int(g["continuity_state"].eq("UNKNOWN_NO_PRIOR_GAME").sum()),
            "unknown_current_roster": int(g["continuity_state"].eq("UNKNOWN_CURRENT_ROSTER").sum()),
            "unknown_prior_roster": int(g["continuity_state"].eq("UNKNOWN_PRIOR_ROSTER").sum()),
        })
    return pd.DataFrame(rows)


def qualify(
    continuity: pd.DataFrame,
    identity: dict[str, object],
    integrity: dict[str, int],
    redundancy: pd.DataFrame,
) -> pd.DataFrame:
    eligible = int(len(continuity))
    known = int(continuity[CANDIDATE].notna().sum())
    coverage = float(known / eligible) if eligible else 0.0
    rd = redundancy.iloc[0]
    hard_integrity = bool(
        float(identity["stable_id_coverage"]) >= MIN_STABLE_ID
        and int(identity["ambiguous_same_week_gsis_team_conflicts"]) == 0
        and int(integrity["published_duplicate_team_week_rows"]) == 0
        and int(integrity["chronology_violations"]) == 0
        and int(integrity["schedule_join_fanout"]) == 0
        and int(integrity["redundancy_join_fanout"]) == 0
    )
    red = str(rd["redundancy_disposition"])
    if not hard_integrity:
        disposition = "REJECTED_INTEGRITY"
        reason = "stable identity, duplicate, fanout or chronology integrity gate failed"
    elif eligible < MIN_ELIGIBLE or coverage < MIN_COVERAGE:
        disposition = "ENGINEERING_READY_SOURCE_THIN"
        reason = "broad scheduled-team-game coverage or sample support below frozen floor"
    elif red == "REDUNDANCY_UNRESOLVED_SOURCE_THIN":
        disposition = "ENGINEERING_READY_SOURCE_THIN"
        reason = "redundancy audit lacks frozen train/holdout support"
    elif red == "HIGHLY_RECONSTRUCTIBLE_REDUNDANT":
        disposition = "DESCRIPTIVE_ONLY"
        reason = "candidate highly reconstructible from canonical strict-prior team state"
    else:
        disposition = "READY_FOR_FROZEN_EXPERIMENT"
        reason = "identity, integrity, broad coverage, mechanism and redundancy gates satisfied"

    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "family": "OL_ROSTER_PERSONNEL_CONTINUITY_V1",
        "grain": "scheduled_team_game",
        "seasons_available": "2019-2025",
        "eligible_rows": eligible,
        "observed_rows": known,
        "pregame_coverage": coverage,
        "stable_id_coverage": float(identity["stable_id_coverage"]),
        "unknown_rate": float(1.0 - coverage),
        "duplicate_key_count": int(integrity["published_duplicate_team_week_rows"]),
        "fanout_count": int(integrity["schedule_join_fanout"] + integrity["redundancy_join_fanout"]),
        "stability_stat": "NOT_HARD_GATE_DIRECT_PREGAME_CHANGE_STATE",
        "stability_value": np.nan,
        "intended_component": (
            "OL protection/run-blocking continuity and uncertainty; QB/rushing efficiency context"
        ),
        "mechanism_note": (
            "current pregame OL personnel continuity versus the same team's previous game "
            "may identify changed blocking environments without target-game participation"
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

    schedule_raw = pd.read_csv(args.schedule, low_memory=False)
    roster = pd.read_csv(args.roster, low_memory=False)
    team_weekly = pd.read_csv(args.team_weekly, low_memory=False)

    schedule = normalize_schedule(schedule_raw)
    roster_sets, identity = build_roster_sets(roster, schedule)
    continuity, base_integrity = materialize_continuity(schedule, roster_sets)
    enriched, red_integrity = attach_prior_team_state(continuity, team_weekly)
    integrity = {**base_integrity, **red_integrity}

    stability = stability_diagnostic(continuity)
    redundancy = redundancy_audit(enriched)
    coverage = coverage_table(continuity)
    inventory = qualify(continuity, identity, integrity, redundancy)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out / "ol_roster_continuity_coverage_v1.csv", index=False)
    stability.to_csv(out / "ol_roster_continuity_stability_v1.csv", index=False)
    redundancy.to_csv(out / "ol_roster_continuity_redundancy_v1.csv", index=False)
    inventory.to_csv(out / "ol_roster_continuity_qualification_inventory_v1.csv", index=False)

    manifest = {
        "qualification_version": "OL_ROSTER_CONTINUITY_QUALIFICATION_V1",
        "frozen_pre_result_disposition": "OL_ROSTER_CONTINUITY_QUALIFICATION_V1_FROZEN_PRE_RESULT",
        "git_sha": args.git_sha,
        "schedule_sha256": _sha256(args.schedule),
        "weekly_roster_sha256": _sha256(args.roster),
        "team_weekly_sha256": _sha256(args.team_weekly),
        "identity": identity,
        "integrity": integrity,
        "seasons": sorted(SEASONS),
        "candidate": CANDIDATE,
        "formula": "intersection(current_ol_ids,prior_game_ol_ids)/count(current_ol_ids)",
        "week1_state": "UNKNOWN_NO_PRIOR_GAME",
        "backups_retained": True,
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
        "redundancy_train_seasons": "2019-2023",
        "redundancy_holdout_seasons": "2024-2025",
        "redundancy_high_r2": REDUNDANCY_HIGH,
        "redundancy_review_r2": REDUNDANCY_REVIEW,
        "qualification_disposition": str(inventory.iloc[0]["qualification_disposition"]),
    }
    (out / "ol_roster_continuity_qualification_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("QUALIFICATION")
    print(inventory.to_string(index=False))
    print("\nBROAD COVERAGE")
    print(coverage.loc[coverage["slice"].eq("ALL")].to_string(index=False))
    print("\nSTABILITY DIAGNOSTIC")
    print(stability.to_string(index=False))
    print("\nREDUNDANCY")
    print(redundancy.to_string(index=False))
    print("\nMANIFEST")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
