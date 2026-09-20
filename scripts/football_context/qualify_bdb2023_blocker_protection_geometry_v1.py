#!/usr/bin/env python3
"""Outcome-free qualification for BDB2023 blocker protection geometry V1.

Frozen plan:
docs/research/BDB2023_BLOCKER_PROTECTION_GEOMETRY_QUALIFICATION_V1.md

The qualifier uses only certified strict-prior blocker geometry, stable identity
crosswalks and weekly pregame roster metadata. It never uses target-week tracking
geometry, target-game participation/snap counts, PFF pressure outcomes, sportsbook
data or production outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = {
    "hist_blocker_snap_distance_median_yards": {
        "raw": "blocker_target_snap_distance_yards",
        "support": "snap_sample_count",
    },
    "hist_blocker_min_distance_median_yards": {
        "raw": "blocker_target_min_distance_yards",
        "support": "min_distance_sample_count",
    },
    "hist_blocker_time_to_min_distance_median_seconds": {
        "raw": "blocker_target_time_to_min_distance_seconds",
        "support": "time_to_min_sample_count",
    },
}
SOURCE_HASH = "1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182"
SOURCE_SEASON = 2021
TARGET_WEEKS = set(range(1, 9))
OL_POSITIONS = {"C", "G", "T", "OT", "OG", "OL"}
OL_DEPTH_POSITIONS = {"LT", "LG", "C", "RG", "RT", "OT", "OG", "OL", "G", "T"}
MIN_SOURCE_SUPPORT = 10
MIN_COVERAGE = 0.80
MIN_ELIGIBLE_ROWS = 500
MIN_STABLE_ID_COVERAGE = 0.99
REDUNDANCY_HIGH = 0.90
REDUNDANCY_REVIEW = 0.75
REDUNDANCY_TRAIN_MIN = 500
REDUNDANCY_HOLDOUT_MIN = 200


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _norm_numeric_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def _norm_gsis(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def _norm_position(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z]", "", str(value).upper().strip())


def _height_inches(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        number = float(text)
        if np.isfinite(number):
            return float(number)
    except (TypeError, ValueError):
        pass
    m = re.match(r"^(\d+)\s*[-']\s*(\d+)", text)
    if m:
        return float(int(m.group(1)) * 12 + int(m.group(2)))
    return np.nan


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
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


def build_direct_identity_bridge(
    raw_interactions: pd.DataFrame,
    players: pd.DataFrame,
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """BDB blocker nfl_id -> nflverse nfl_id -> gsis_id -> weekly roster gsis."""
    raw = _lower(raw_interactions)
    p = _lower(players)
    r = _lower(roster)

    if "blocker_nfl_id" not in raw.columns:
        raise RuntimeError("BDB interactions missing blocker_nfl_id")

    nfl_col = _first_col(p, ["nfl_id", "nfl_player_id", "nflid"])
    gsis_col = _first_col(p, ["gsis_id", "player_id", "player_gsis_id"])
    roster_gsis_col = _first_col(r, ["gsis_id", "player_id", "player_gsis_id"])
    if not nfl_col or not gsis_col or not roster_gsis_col:
        raise RuntimeError(
            "stable-ID bridge columns unavailable: "
            f"players={sorted(p.columns.tolist())} roster={sorted(r.columns.tolist())}"
        )

    bdb_ids = sorted(set(raw["blocker_nfl_id"].map(_norm_numeric_id)) - {""})

    cross = p[[nfl_col, gsis_col]].copy()
    cross["bdb_nfl_id"] = cross[nfl_col].map(_norm_numeric_id)
    cross["gsis_id"] = cross[gsis_col].map(_norm_gsis)
    cross = cross.loc[
        cross["bdb_nfl_id"].ne("") & cross["gsis_id"].ne(""),
        ["bdb_nfl_id", "gsis_id"],
    ].drop_duplicates()

    bdb_amb = cross.groupby("bdb_nfl_id")["gsis_id"].nunique()
    ambiguous_bdb_ids = sorted(bdb_amb[bdb_amb.gt(1)].index.tolist())
    cross = cross.loc[~cross["bdb_nfl_id"].isin(ambiguous_bdb_ids)]
    cross = cross.drop_duplicates("bdb_nfl_id")

    rr = r.copy()
    rr["gsis_id"] = rr[roster_gsis_col].map(_norm_gsis)

    # GSIS is the stable identity authority. Name drift under the same GSIS ID is
    # diagnostic only and must not invalidate a direct stable-ID bridge.
    name_col = _first_col(rr, ["full_name", "football_name", "player_name", "player", "name"])
    roster_name_variation_gsis: list[str] = []
    if name_col:
        names = rr.loc[rr["gsis_id"].ne(""), ["gsis_id", name_col]].copy()
        names["_name"] = (
            names[name_col].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        )
        nuniq = names.loc[names["_name"].ne("")].groupby("gsis_id")["_name"].nunique()
        roster_name_variation_gsis = sorted(nuniq[nuniq.gt(1)].index.tolist())

    # A real roster ambiguity is the same stable GSIS identity resolving to
    # conflicting teams in the same target week. Repeated weekly rows and name
    # variants are not identity ambiguity.
    week_col = _first_col(rr, ["week"])
    team_col = _first_col(rr, ["team", "team_abbr", "club_code"])
    ambiguous_roster_gsis: list[str] = []
    if week_col and team_col:
        rr["_week"] = pd.to_numeric(rr[week_col], errors="coerce")
        rr["_team"] = rr[team_col].astype(str).str.upper().str.strip()
        weekly = rr.loc[
            rr["gsis_id"].ne("") & rr["_week"].notna() & rr["_team"].ne("")
        ]
        conflicts = weekly.groupby(["_week", "gsis_id"])["_team"].nunique()
        ambiguous_roster_gsis = sorted(set(
            conflicts[conflicts.gt(1)].index.get_level_values("gsis_id").tolist()
        ))

    roster_ids = set(rr["gsis_id"]) - {""}
    safe = cross.loc[
        cross["gsis_id"].isin(roster_ids)
        & ~cross["gsis_id"].isin(ambiguous_roster_gsis)
    ].copy()

    mapped_ids = set(safe["bdb_nfl_id"])
    coverage = float(
        sum(v in mapped_ids for v in bdb_ids) / len(bdb_ids)
    ) if bdb_ids else 0.0

    report = {
        "players_nfl_id_column": nfl_col,
        "players_gsis_id_column": gsis_col,
        "roster_gsis_id_column": roster_gsis_col,
        "bdb_unique_blocker_ids": int(len(bdb_ids)),
        "bdb_ids_mapped_to_weekly_roster_gsis": int(sum(v in mapped_ids for v in bdb_ids)),
        "direct_stable_id_bridge_coverage": coverage,
        "ambiguous_bdb_nfl_id_count": int(len(ambiguous_bdb_ids)),
        "ambiguous_roster_gsis_count": int(len(ambiguous_roster_gsis)),
        "roster_name_variation_gsis_count": int(len(roster_name_variation_gsis)),
        "roster_name_variation_is_diagnostic_only": True,
        "name_fallback_used": False,
    }
    return safe[["bdb_nfl_id", "gsis_id"]].reset_index(drop=True), report


def build_broad_ol_universe(roster: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    r = _lower(roster)
    season_col = _first_col(r, ["season"])
    week_col = _first_col(r, ["week"])
    gsis_col = _first_col(r, ["gsis_id", "player_id", "player_gsis_id"])
    team_col = _first_col(r, ["team", "team_abbr", "club_code"])
    pos_col = _first_col(r, ["position", "pos"])
    depth_col = _first_col(r, ["depth_chart_position", "depth_position"])
    if not all([season_col, week_col, gsis_col, team_col, pos_col]):
        raise RuntimeError(
            "weekly roster missing required season/week/GSIS/team/position columns"
        )

    r["season"] = pd.to_numeric(r[season_col], errors="coerce")
    r["week"] = pd.to_numeric(r[week_col], errors="coerce")
    r["gsis_id"] = r[gsis_col].map(_norm_gsis)
    r["team"] = r[team_col].astype(str).str.upper().str.strip()
    r["position_norm"] = r[pos_col].map(_norm_position)
    r["depth_position_norm"] = (
        r[depth_col].map(_norm_position) if depth_col else ""
    )

    eligible = (
        r["position_norm"].isin(OL_POSITIONS)
        | r["depth_position_norm"].isin(OL_DEPTH_POSITIONS)
    )
    u = r.loc[
        r["season"].eq(SOURCE_SEASON)
        & r["week"].isin(TARGET_WEEKS)
        & r["gsis_id"].ne("")
        & eligible
    ].copy()

    # Target-game participation/snap fields are intentionally ignored. Roster
    # presence plus position metadata alone defines this broad denominator.
    key = ["season", "week", "gsis_id"]
    duplicate_rows = int(u.duplicated(key, keep=False).sum())
    if duplicate_rows:
        # Preserve only if rows are semantically identical across the fields used
        # here; otherwise the published grain would be ambiguous.
        used = ["team", "position_norm", "depth_position_norm"]
        conflicts = 0
        for _, g in u.loc[u.duplicated(key, keep=False)].groupby(key, dropna=False):
            if any(g[c].astype(str).nunique(dropna=False) > 1 for c in used):
                conflicts += 1
        if conflicts:
            raise RuntimeError(
                f"weekly roster has ambiguous duplicate OL player-week keys: {conflicts}"
            )
        u = u.drop_duplicates(key, keep="last").copy()

    height_col = _first_col(u, ["height"])
    weight_col = _first_col(u, ["weight"])
    years_col = _first_col(u, ["years_exp", "years_experience", "experience"])
    u["height_inches"] = u[height_col].map(_height_inches) if height_col else np.nan
    u["weight_num"] = pd.to_numeric(u[weight_col], errors="coerce") if weight_col else np.nan
    u["years_exp_num"] = pd.to_numeric(u[years_col], errors="coerce") if years_col else np.nan

    out = u[[
        "season", "week", "team", "gsis_id", "position_norm",
        "depth_position_norm", "height_inches", "weight_num", "years_exp_num",
    ]].copy()
    out = out.sort_values(["week", "team", "gsis_id"]).reset_index(drop=True)

    report = {
        "eligible_rostered_ol_player_weeks": int(len(out)),
        "eligible_rostered_ol_players": int(out["gsis_id"].nunique()),
        "week1_rows_in_denominator": int(out["week"].eq(1).sum()),
        "published_duplicate_player_week_rows": int(out.duplicated(key, keep=False).sum()),
        "target_game_snap_or_participation_used_for_eligibility": False,
        "depth_chart_position_available": bool(depth_col),
        "height_available": bool(height_col),
        "weight_available": bool(weight_col),
        "years_experience_available": bool(years_col),
    }
    return out, report


def map_blocker_history(
    snapshots: pd.DataFrame,
    bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    s = _lower(snapshots)
    required = {
        "target_week", "blocker_nfl_id", "history_max_source_week",
        *[spec["support"] for spec in CANDIDATES.values()],
        *CANDIDATES.keys(),
    }
    missing = required - set(s.columns)
    if missing:
        raise RuntimeError(f"blocker history snapshots missing columns: {sorted(missing)}")

    s["bdb_nfl_id"] = s["blocker_nfl_id"].map(_norm_numeric_id)
    s["target_week"] = pd.to_numeric(s["target_week"], errors="coerce").astype("Int64")
    s["history_max_source_week"] = pd.to_numeric(
        s["history_max_source_week"], errors="coerce"
    ).astype("Int64")
    out = s.merge(bridge, on="bdb_nfl_id", how="left", validate="many_to_one")

    comparable = out["target_week"].notna() & out["history_max_source_week"].notna()
    chronology_violations = int(
        (
            out.loc[comparable, "history_max_source_week"]
            >= out.loc[comparable, "target_week"]
        ).sum()
    )
    support_violations = 0
    for feature, spec in CANDIDATES.items():
        support = pd.to_numeric(out[spec["support"]], errors="coerce")
        support_violations += int((out[feature].notna() & support.lt(MIN_SOURCE_SUPPORT)).sum())

    mapped = out.loc[out["gsis_id"].notna() & out["gsis_id"].astype(str).ne("")].copy()
    duplicate_rows = int(
        mapped.duplicated(["target_week", "gsis_id"], keep=False).sum()
    )
    return mapped, {
        "chronology_violations": chronology_violations,
        "materializer_support_threshold_violations": int(support_violations),
        "mapped_snapshot_duplicate_rows": duplicate_rows,
    }


def attach_history(
    universe: pd.DataFrame,
    mapped: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    keep = [
        "target_week", "gsis_id", "history_max_source_week",
        *[spec["support"] for spec in CANDIDATES.values()],
        *CANDIDATES.keys(),
    ]
    snap = mapped[keep].copy().rename(columns={"target_week": "week"})
    if snap.duplicated(["week", "gsis_id"]).any():
        raise RuntimeError("blocker snapshot join would fan out rostered OL player-weeks")
    before = len(universe)
    out = universe.merge(
        snap, on=["week", "gsis_id"], how="left", validate="one_to_one"
    )
    return out, int(len(out) - before)


def source_stability(raw_interactions: pd.DataFrame) -> pd.DataFrame:
    raw = _lower(raw_interactions)
    required = {"week", "blocker_nfl_id", *[v["raw"] for v in CANDIDATES.values()]}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"raw blocker interactions missing stability columns: {sorted(missing)}")
    raw["week"] = pd.to_numeric(raw["week"], errors="coerce")
    rows = []
    for feature, spec in CANDIDATES.items():
        metric = spec["raw"]
        q = raw[["blocker_nfl_id", "week", metric]].copy()
        q[metric] = pd.to_numeric(q[metric], errors="coerce")
        early = (
            q.loc[q["week"].between(1, 4)]
            .dropna(subset=[metric])
            .groupby("blocker_nfl_id")[metric]
            .agg(["median", "count"]).reset_index()
            .rename(columns={"median": "early_value", "count": "early_n"})
        )
        late = (
            q.loc[q["week"].between(5, 8)]
            .dropna(subset=[metric])
            .groupby("blocker_nfl_id")[metric]
            .agg(["median", "count"]).reset_index()
            .rename(columns={"median": "late_value", "count": "late_n"})
        )
        z = early.merge(late, on="blocker_nfl_id", how="inner")
        z = z.loc[
            z["early_n"].ge(MIN_SOURCE_SUPPORT)
            & z["late_n"].ge(MIN_SOURCE_SUPPORT)
        ].copy()
        rows.append({
            "feature_name": feature,
            "raw_geometry_metric": metric,
            "stability_stat": "blocker_early_late_median_spearman_weeks1_4_vs5_8",
            "stability_value": _safe_spearman(z["early_value"], z["late_value"]),
            "stability_pairs": int(len(z)),
            "minimum_observations_each_half": MIN_SOURCE_SUPPORT,
            "pff_pressure_outcomes_read": False,
        })
    return pd.DataFrame(rows)


def coverage_by_week_position(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slices: list[tuple[str, pd.DataFrame]] = [("ALL", joined)]
    for week in sorted(TARGET_WEEKS):
        slices.append((f"WEEK_{week}", joined.loc[joined["week"].eq(week)]))
    for pos in sorted(set(joined["position_norm"].astype(str))):
        slices.append((f"POSITION_{pos}", joined.loc[joined["position_norm"].eq(pos)]))
        for week in sorted(TARGET_WEEKS):
            slices.append((
                f"POSITION_{pos}_WEEK_{week}",
                joined.loc[
                    joined["position_norm"].eq(pos) & joined["week"].eq(week)
                ],
            ))
    for label, g in slices:
        for feature in CANDIDATES:
            observed = int(g[feature].notna().sum()) if len(g) else 0
            rows.append({
                "slice": label,
                "eligible_rows": int(len(g)),
                "feature_name": feature,
                "observed_rows": observed,
                "pregame_coverage": float(observed / len(g)) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def _design_train_holdout(
    frame: pd.DataFrame, feature: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    cols = [
        "week", feature, "position_norm", "depth_position_norm",
        "height_inches", "weight_num", "years_exp_num",
    ]
    z = frame[cols].copy()
    z[feature] = pd.to_numeric(z[feature], errors="coerce")
    z = z.dropna(subset=[feature])
    train = z.loc[z["week"].between(1, 4)].copy()
    holdout = z.loc[z["week"].between(5, 8)].copy()
    if len(train) < REDUNDANCY_TRAIN_MIN or len(holdout) < REDUNDANCY_HOLDOUT_MIN:
        return (
            np.empty((0, 0)), np.empty(0),
            np.empty((0, 0)), np.empty(0),
            int(len(train)), int(len(holdout)),
        )

    cat_cols = ["position_norm", "depth_position_norm"]
    numeric_cols = ["height_inches", "weight_num", "years_exp_num"]
    for col in cat_cols:
        train[col] = train[col].fillna("").astype(str).replace("", "__MISSING__")
        holdout[col] = holdout[col].fillna("").astype(str).replace("", "__MISSING__")

    categories = {
        col: sorted(train[col].unique().tolist())
        for col in cat_cols
    }
    medians = {}
    for col in numeric_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        holdout[col] = pd.to_numeric(holdout[col], errors="coerce")
        med = float(train[col].median()) if train[col].notna().any() else 0.0
        medians[col] = med
        train[col] = train[col].fillna(med)
        holdout[col] = holdout[col].fillna(med)

    def matrix(df: pd.DataFrame) -> np.ndarray:
        pieces = [np.ones((len(df), 1), dtype=float)]
        for col in numeric_cols:
            pieces.append(df[[col]].to_numpy(float))
        for col in cat_cols:
            for cat in categories[col]:
                pieces.append((df[col].eq(cat).astype(float).to_numpy()[:, None]))
        return np.hstack(pieces)

    return (
        matrix(train), train[feature].to_numpy(float),
        matrix(holdout), holdout[feature].to_numpy(float),
        int(len(train)), int(len(holdout)),
    )


def redundancy_audit(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CANDIDATES:
        Xtr, ytr, Xte, yte, ntr, nte = _design_train_holdout(joined, feature)
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
        rows.append({
            "feature_name": feature,
            "reconstruction_inputs": (
                "position_onehot|depth_chart_position_onehot|height|weight|years_experience"
            ),
            "train_weeks": "2021 W1-W4",
            "holdout_weeks": "2021 W5-W8",
            "reconstruction_train_rows": ntr,
            "reconstruction_holdout_rows": nte,
            "holdout_reconstructibility_r2": float(r2) if np.isfinite(r2) else np.nan,
            "redundancy_disposition": disposition,
            "target_game_participation_used": False,
            "pff_pressure_outcomes_read": False,
            "sportsbook_read": False,
        })
    return pd.DataFrame(rows)


def build_inventory(
    joined: pd.DataFrame,
    stability: pd.DataFrame,
    redundancy: pd.DataFrame,
    identity: dict[str, object],
    integrity: dict[str, object],
) -> pd.DataFrame:
    st = stability.set_index("feature_name")
    rd = redundancy.set_index("feature_name")
    eligible_rows = int(len(joined))
    hard_integrity = (
        float(identity["direct_stable_id_bridge_coverage"]) >= MIN_STABLE_ID_COVERAGE
        and int(identity["ambiguous_bdb_nfl_id_count"]) == 0
        and int(identity["ambiguous_roster_gsis_count"]) == 0
        and int(integrity["chronology_violations"]) == 0
        and int(integrity["materializer_support_threshold_violations"]) == 0
        and int(integrity["mapped_snapshot_duplicate_rows"]) == 0
        and int(integrity["published_duplicate_player_week_rows"]) == 0
        and int(integrity["join_fanout_rows"]) == 0
    )
    rows = []
    for feature, spec in CANDIDATES.items():
        observed = joined[feature].notna()
        coverage = float(observed.mean()) if eligible_rows else 0.0
        support = pd.to_numeric(
            joined.loc[observed, spec["support"]], errors="coerce"
        ).dropna()
        stability_value = st.loc[feature, "stability_value"]
        stability_pairs = int(st.loc[feature, "stability_pairs"])
        red = str(rd.loc[feature, "redundancy_disposition"])

        if not hard_integrity:
            disposition = "REJECTED_INTEGRITY"
            reason = "stable identity, chronology, support, duplicate or fanout gate failed"
        elif eligible_rows < MIN_ELIGIBLE_ROWS or coverage < MIN_COVERAGE:
            disposition = "ENGINEERING_READY_SOURCE_THIN"
            reason = "broad rostered-OL pregame coverage or sample support below frozen qualification floor"
        elif pd.isna(stability_value) or stability_pairs <= 0:
            disposition = "ENGINEERING_READY_SOURCE_THIN"
            reason = "source temporal stability evidence unavailable"
        elif red == "REDUNDANCY_UNRESOLVED_SOURCE_THIN":
            disposition = "ENGINEERING_READY_SOURCE_THIN"
            reason = "outcome-free redundancy audit lacks frozen train/holdout support"
        elif red == "HIGHLY_RECONSTRUCTIBLE_REDUNDANT":
            disposition = "DESCRIPTIVE_ONLY"
            reason = "candidate highly reconstructible from basic pregame roster metadata"
        else:
            disposition = "READY_FOR_FROZEN_EXPERIMENT"
            reason = "integrity, broad support, stability, mechanism and redundancy gates satisfied"

        rows.append({
            "feature_name": feature,
            "family": "BDB2023_BLOCKER_PROTECTION_GEOMETRY_V1",
            "grain": "2021_rostered_OL_player_week",
            "seasons_available": "2021",
            "eligible_rows": eligible_rows,
            "observed_rows": int(observed.sum()),
            "pregame_coverage": coverage,
            "stable_id_coverage": float(identity["direct_stable_id_bridge_coverage"]),
            "unknown_rate": float(1.0 - coverage),
            "duplicate_key_count": int(integrity["published_duplicate_player_week_rows"])
                + int(integrity["mapped_snapshot_duplicate_rows"]),
            "fanout_count": int(integrity["join_fanout_rows"]),
            "prior_support_median": float(support.median()) if len(support) else np.nan,
            "stability_stat": st.loc[feature, "stability_stat"],
            "stability_value": float(stability_value) if pd.notna(stability_value) else np.nan,
            "stability_pairs": stability_pairs,
            "intended_component": (
                "blocker/protection archetype, protection continuity/uncertainty, "
                "QB efficiency or scramble environment"
            ),
            "mechanism_note": (
                "strict-prior blocker interaction geometry may describe stable protection "
                "structure without asserting universal blocker-rusher assignment"
            ),
            "source_class": "CERTIFIED_BDB2023_RESEARCH_ACCESS",
            "redundancy_disposition": red,
            "holdout_reconstructibility_r2": rd.loc[feature, "holdout_reconstructibility_r2"],
            "reconstruction_train_rows": int(rd.loc[feature, "reconstruction_train_rows"]),
            "reconstruction_holdout_rows": int(rd.loc[feature, "reconstruction_holdout_rows"]),
            "qualification_disposition": disposition,
            "qualification_reason": reason,
        })
    return pd.DataFrame(rows)


def run(
    roster_path: Path,
    players_path: Path,
    materialized_dir: Path,
    out_dir: Path,
    *,
    active_sha: str,
    materializer_sha: str,
) -> dict[str, object]:
    private = materialized_dir / "private"
    sanitized = materialized_dir / "sanitized"
    raw_path = private / "bdb2023_protection_interactions_v1.csv"
    hist_path = private / "bdb2023_blocker_history_snapshots_v1.csv"
    qa_path = sanitized / "bdb2023_protection_geometry_advanced_feature_qa_v1.json"
    for path in [roster_path, players_path, raw_path, hist_path, qa_path]:
        if not path.exists():
            raise RuntimeError(f"required qualification input missing: {path}")

    qa = json.loads(qa_path.read_text())
    observed_hash = str(qa.get("source_hash_observed", ""))
    if observed_hash != SOURCE_HASH or not qa.get("source_hash_verified", False):
        raise RuntimeError(f"BDB2023 source hash mismatch: {observed_hash}")
    temporal = qa.get("temporal_audit", {})
    if int(temporal.get("blocker_history", {}).get("violations", -1)) != 0:
        raise RuntimeError("certified materializer reports blocker-history chronology violation")
    if int(temporal.get("target_game_rows_used", -1)) != 0:
        raise RuntimeError("certified materializer reports target-game history use")

    roster = pd.read_csv(roster_path, low_memory=False)
    players = pd.read_csv(players_path, low_memory=False)
    raw = pd.read_csv(raw_path, low_memory=False)
    snapshots = pd.read_csv(hist_path, low_memory=False)

    bridge, identity = build_direct_identity_bridge(raw, players, roster)
    universe, universe_meta = build_broad_ol_universe(roster)
    mapped, temporal_meta = map_blocker_history(snapshots, bridge)
    joined, fanout = attach_history(universe, mapped)
    integrity = {
        **universe_meta,
        **temporal_meta,
        "join_fanout_rows": int(fanout),
    }

    stability = source_stability(raw)
    coverage = coverage_by_week_position(joined)
    redundancy = redundancy_audit(joined)
    inventory = build_inventory(joined, stability, redundancy, identity, integrity)

    out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out_dir / "bdb2023_blocker_geometry_coverage_v1.csv", index=False)
    stability.to_csv(out_dir / "bdb2023_blocker_geometry_stability_v1.csv", index=False)
    redundancy.to_csv(out_dir / "bdb2023_blocker_geometry_redundancy_v1.csv", index=False)
    inventory.to_csv(out_dir / "bdb2023_blocker_geometry_qualification_inventory_v1.csv", index=False)

    manifest = {
        "qualification_version": "BDB2023_BLOCKER_PROTECTION_GEOMETRY_QUALIFICATION_V1",
        "frozen_pre_result_disposition": "BDB2023_BLOCKER_PROTECTION_GEOMETRY_QUALIFICATION_V1_FROZEN_PRE_RESULT",
        "active_context_sha": active_sha,
        "certified_materializer_sha": materializer_sha,
        "source_hash_expected": SOURCE_HASH,
        "source_hash_observed": observed_hash,
        "source_hash_verified": True,
        "roster_sha256": _sha256_file(roster_path),
        "players_crosswalk_sha256": _sha256_file(players_path),
        "raw_interactions_ephemeral_sha256": _sha256_file(raw_path),
        "blocker_history_ephemeral_sha256": _sha256_file(hist_path),
        "identity": identity,
        "integrity": integrity,
        "materializer_temporal_audit": temporal,
        "broad_target_season": SOURCE_SEASON,
        "broad_target_weeks": "1-8",
        "broad_week1_retained": True,
        "backups_retained": True,
        "materializer_min_prior_interactions": MIN_SOURCE_SUPPORT,
        "qualification_min_pregame_coverage": MIN_COVERAGE,
        "qualification_min_eligible_rows": MIN_ELIGIBLE_ROWS,
        "qualification_min_stable_id_coverage": MIN_STABLE_ID_COVERAGE,
        "redundancy_train": "2021 W1-W4",
        "redundancy_holdout": "2021 W5-W8",
        "redundancy_high_r2": REDUNDANCY_HIGH,
        "redundancy_review_r2": REDUNDANCY_REVIEW,
        "pff_pressure_outcomes_read_for_qualification": False,
        "target_game_participation_used_for_eligibility": False,
        "target_game_geometry_used_pregame": False,
        "universal_blocker_rusher_assignment_claimed": False,
        "sportsbook_read": False,
        "production_changed": False,
        "issue_535_touched": False,
        "raw_competition_files_uploaded": False,
        "derived_per_row_bdb_files_uploaded": False,
        "candidate_dispositions": dict(zip(
            inventory["feature_name"], inventory["qualification_disposition"]
        )),
        "any_candidate_ready": bool(
            inventory["qualification_disposition"].eq("READY_FOR_FROZEN_EXPERIMENT").any()
        ),
        "all_candidates_ready": bool(
            len(inventory)
            and inventory["qualification_disposition"].eq("READY_FOR_FROZEN_EXPERIMENT").all()
        ),
    }
    (out_dir / "bdb2023_blocker_geometry_qualification_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("QUALIFICATION INVENTORY")
    print(inventory.to_string(index=False))
    print("\nSTABILITY")
    print(stability.to_string(index=False))
    print("\nREDUNDANCY")
    print(redundancy.to_string(index=False))
    print("\nBROAD COVERAGE")
    print(coverage.loc[coverage["slice"].eq("ALL")].to_string(index=False))
    print("\nMANIFEST")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--players", type=Path, required=True)
    ap.add_argument("--materialized-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--active-sha", required=True)
    ap.add_argument("--materializer-sha", required=True)
    args = ap.parse_args()
    run(
        args.roster,
        args.players,
        args.materialized_dir,
        args.out_dir,
        active_sha=args.active_sha,
        materializer_sha=args.materializer_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
