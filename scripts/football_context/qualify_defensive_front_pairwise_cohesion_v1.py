#!/usr/bin/env python3
"""Outcome-free Defensive Front Pairwise Cohesion Qualification V1."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.football_context import qualify_ol_roster_continuity_v1 as ol

SEASONS = set(range(2019, 2026))
FRONT_POSITIONS = {"DE", "DT", "NT", "DL", "EDGE", "OLB", "ILB", "LB"}
CANDIDATE = "def_front_pairwise_cohesion_prior_share"
IMMEDIATE = "def_front_roster_continuity_share_prev_game"
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
PRIOR_DEF_METRICS = [
    "pressure_rate_generated",
    "success_rate_def",
    "def_pass_epa",
    "explosive_play_rate_allowed",
]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


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


def _norm_alt_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def _semantic_gsis_collision_ids(roster: pd.DataFrame) -> tuple[set[str], list[str]]:
    """Find GSIS values positively proven to represent multiple source people.

    ESB ID and Smart ID are upstream person identifiers carried by the same weekly
    roster source. A GSIS is quarantined only when one of those authorities has
    multiple distinct nonblank values for that GSIS. Same-person multi-team rows
    are deliberately not resolved here and remain subject to the frozen ambiguity
    gate.
    """
    alt_cols = [c for c in ["esb_id", "smart_id"] if c in roster.columns]
    if not alt_cols:
        return set(), []

    q = roster.loc[
        roster["season"].isin(SEASONS)
        & roster["week"].gt(0)
        & roster["gsis_id"].ne("")
    ].copy()
    collision_ids: set[str] = set()
    for col in alt_cols:
        z = q[["gsis_id", col]].copy()
        z["_alt"] = z[col].map(_norm_alt_id)
        z = z.loc[z["_alt"].ne("")]
        if z.empty:
            continue
        counts = z.groupby("gsis_id")["_alt"].nunique()
        collision_ids.update(counts.loc[counts.gt(1)].index.astype(str).tolist())
    return collision_ids, alt_cols


def build_front_sets(
    roster: pd.DataFrame,
    schedule: pd.DataFrame,
) -> tuple[dict[tuple[int, int, str], set[str]], dict[str, object]]:
    r = _lower(roster)
    team_col = _first_col(r, ["team", "team_abbr", "club_code"])
    gsis_col = _first_col(r, ["gsis_id", "player_id", "player_gsis_id"])
    pos_col = _first_col(r, ["position", "pos"])
    depth_col = _first_col(r, ["depth_chart_position", "depth_position"])
    if not {"season", "week"}.issubset(r.columns) or not team_col or not gsis_col or not pos_col:
        raise RuntimeError("weekly roster missing season/week/team/GSIS/position")

    r["season"] = pd.to_numeric(r["season"], errors="coerce")
    r["week"] = pd.to_numeric(r["week"], errors="coerce")
    r["team"] = r[team_col].map(canon_team)
    r["gsis_id"] = r[gsis_col].map(ol._norm_gsis)
    r["position_norm"] = r[pos_col].map(ol._norm_pos)
    r["depth_position_norm"] = r[depth_col].map(ol._norm_pos) if depth_col else ""
    semantic_collision_ids, semantic_authority_cols = _semantic_gsis_collision_ids(r)
    eligible = (
        r["position_norm"].isin(FRONT_POSITIONS)
        | r["depth_position_norm"].isin(FRONT_POSITIONS)
    )
    r = r.loc[
        r["season"].isin(SEASONS) & r["week"].gt(0) & eligible
    ].copy()

    before = int(len(r))
    keys = schedule[["season", "week", "team"]].drop_duplicates()
    r = r.merge(keys, on=["season", "week", "team"], how="inner", validate="many_to_one")
    on_schedule = int(len(r))

    raw_nonblank = r["gsis_id"].ne("")
    semantic_collision = r["gsis_id"].isin(semantic_collision_ids)
    stable = raw_nonblank & ~semantic_collision
    raw_nonblank_cov = float(raw_nonblank.mean()) if len(r) else 0.0
    stable_cov = float(stable.mean()) if len(r) else 0.0
    quarantined_rows = int((raw_nonblank & semantic_collision).sum())
    valid = r.loc[stable].copy()
    conflicts = valid.groupby(["season", "week", "gsis_id"])["team"].nunique()
    ambiguous = int((conflicts > 1).sum())
    duplicates = int(valid.duplicated(["season", "week", "team", "gsis_id"], keep=False).sum())
    valid = valid.drop_duplicates(["season", "week", "team", "gsis_id"])

    sets = {
        (int(s), int(w), str(t)): set(g["gsis_id"].astype(str))
        for (s, w, t), g in valid.groupby(["season", "week", "team"], sort=False)
    }
    return sets, {
        "source_front_rows_before_schedule_filter": before,
        "source_front_rows_on_scheduled_games": on_schedule,
        "raw_nonblank_gsis_coverage": raw_nonblank_cov,
        "stable_id_coverage": stable_cov,
        "semantic_gsis_collision_ids": sorted(semantic_collision_ids),
        "semantic_gsis_collision_id_count": int(len(semantic_collision_ids)),
        "semantic_gsis_collision_rows_quarantined": quarantined_rows,
        "semantic_gsis_collision_authority_columns": semantic_authority_cols,
        "semantic_gsis_collision_rule": "quarantine_gsis_if_esb_id_or_smart_id_has_multiple_nonblank_values",
        "ambiguous_same_week_gsis_team_conflicts": ambiguous,
        "duplicate_source_identity_rows_collapsed_to_set": duplicates,
        "depth_chart_position_available": bool(depth_col),
        "name_fallback_used": False,
        "target_game_snap_or_participation_used": False,
    }


def materialize(
    schedule: pd.DataFrame,
    front_sets: dict[tuple[int, int, str], set[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    chronology = 0
    for team, g in schedule.sort_values(["team", "season", "week"]).groupby("team", sort=True):
        keys = [
            (int(r.season), int(r.week), str(team))
            for r in g.sort_values(["season", "week"]).itertuples(index=False)
        ]
        for idx, key in enumerate(keys):
            season, week, team_name = key
            current = front_sets.get(key, set())
            prior_key = keys[idx - 1] if idx else None
            prior = front_sets.get(prior_key, set()) if prior_key else set()
            hist_keys = keys[max(0, idx - LOOKBACK_GAMES):idx]
            if any((ps > season) or (ps == season and pw >= week) for ps, pw, _ in hist_keys):
                chronology += 1
            available = [front_sets[k] for k in hist_keys if front_sets.get(k, set())]

            rec = {
                "season": season,
                "week": week,
                "team": team_name,
                "current_front_roster_count": int(len(current)),
                "current_front_pair_count": int(len(current) * (len(current) - 1) // 2),
                "prior_scheduled_games_in_window": int(len(hist_keys)),
                "prior_roster_games_available": int(len(available)),
                "mean_prior_corostered_games": np.nan,
                "median_prior_corostered_games": np.nan,
                "zero_prior_coroster_pair_share": np.nan,
                IMMEDIATE: np.nan,
                CANDIDATE: np.nan,
                "cohesion_state": "",
            }
            if current and prior:
                rec[IMMEDIATE] = float(len(current & prior) / len(current))

            if len(current) < 2:
                rec["cohesion_state"] = "UNKNOWN_CURRENT_ROSTER_LT2"
            elif not hist_keys:
                rec["cohesion_state"] = "UNKNOWN_NO_PRIOR_HISTORY"
            elif not available:
                rec["cohesion_state"] = "UNKNOWN_PRIOR_ROSTER_HISTORY"
            else:
                pairs = list(itertools.combinations(sorted(current), 2))
                counts = np.asarray([
                    sum(1 for prev in available if a in prev and b in prev)
                    for a, b in pairs
                ], dtype=float)
                rec["mean_prior_corostered_games"] = float(counts.mean())
                rec["median_prior_corostered_games"] = float(np.median(counts))
                rec["zero_prior_coroster_pair_share"] = float(np.mean(counts == 0))
                rec[CANDIDATE] = float(np.mean(counts / len(available)))
                rec["cohesion_state"] = "KNOWN"
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["season", "team", "week"]).reset_index(drop=True)
    return out, {
        "published_duplicate_team_week_rows": int(out.duplicated(["season", "week", "team"], keep=False).sum()),
        "chronology_violations": int(chronology),
        "schedule_join_fanout": 0,
        "target_game_pbp_used_in_cohesion": False,
        "future_week_roster_used": False,
    }


def stability_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    z = frame[["season", "week", "team", CANDIDATE]].sort_values(["team", "season", "week"]).copy()
    z["prior"] = z.groupby("team")[CANDIDATE].shift(1)
    q = z[[CANDIDATE, "prior"]].dropna()
    rho = _safe_spearman(q["prior"], q[CANDIDATE])
    delta = (q[CANDIDATE] - q["prior"]).abs()
    passed = bool(
        len(q) >= MIN_STABILITY_PAIRS
        and np.isfinite(rho)
        and rho >= MIN_STABILITY_SPEARMAN
    )
    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "adjacent_game_pairs": int(len(q)),
        "adjacent_game_spearman": float(rho) if np.isfinite(rho) else np.nan,
        "median_absolute_adjacent_change": float(delta.median()) if len(delta) else np.nan,
        "min_adjacent_game_pairs": MIN_STABILITY_PAIRS,
        "min_adjacent_game_spearman": MIN_STABILITY_SPEARMAN,
        "stability_gate_passed": passed,
    }])


def support_table(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [("ALL", frame)]
    for season in sorted(SEASONS):
        groups.append((f"SEASON_{season}", frame.loc[frame["season"].eq(season)]))
    rows = []
    for label, g in groups:
        known = g.loc[g[CANDIDATE].notna()]
        rows.append({
            "slice": label,
            "eligible_team_games": int(len(g)),
            "known_rows": int(len(known)),
            "pregame_coverage": float(len(known) / len(g)) if len(g) else np.nan,
            "median_current_front_roster_count": float(pd.to_numeric(known["current_front_roster_count"], errors="coerce").median()) if len(known) else np.nan,
            "median_current_front_pair_count": float(pd.to_numeric(known["current_front_pair_count"], errors="coerce").median()) if len(known) else np.nan,
            "median_prior_roster_games_available": float(pd.to_numeric(known["prior_roster_games_available"], errors="coerce").median()) if len(known) else np.nan,
            "median_mean_prior_corostered_games": float(pd.to_numeric(known["mean_prior_corostered_games"], errors="coerce").median()) if len(known) else np.nan,
            "median_zero_prior_coroster_pair_share": float(pd.to_numeric(known["zero_prior_coroster_pair_share"], errors="coerce").median()) if len(known) else np.nan,
            "unknown_no_prior_history": int(g["cohesion_state"].eq("UNKNOWN_NO_PRIOR_HISTORY").sum()),
            "unknown_current_roster_lt2": int(g["cohesion_state"].eq("UNKNOWN_CURRENT_ROSTER_LT2").sum()),
            "unknown_prior_roster_history": int(g["cohesion_state"].eq("UNKNOWN_PRIOR_ROSTER_HISTORY").sum()),
        })
    return pd.DataFrame(rows)


def attach_prior_def_state(
    frame: pd.DataFrame,
    team_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    tw = _lower(team_weekly)
    required = {"season", "week", "team", *PRIOR_DEF_METRICS}
    missing = required - set(tw.columns)
    if missing:
        raise RuntimeError(f"team weekly missing {sorted(missing)}")
    tw["season"] = pd.to_numeric(tw["season"], errors="coerce")
    tw["week"] = pd.to_numeric(tw["week"], errors="coerce")
    tw["team"] = tw["team"].map(canon_team)
    if tw.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("team weekly duplicate team-week rows")
    lookup = {
        (int(r.season), int(r.week), str(r.team)): r
        for r in tw.dropna(subset=["season", "week"]).itertuples(index=False)
    }

    prior_rows = []
    chronology = 0
    for team, g in frame.sort_values(["team", "season", "week"]).groupby("team", sort=True):
        keys = [(int(r.season), int(r.week), str(team)) for r in g.itertuples(index=False)]
        prev = None
        for key in keys:
            s, w, t = key
            rec = {"season": s, "week": w, "team": t}
            if prev is not None:
                ps, pw, _ = prev
                if (ps > s) or (ps == s and pw >= w):
                    chronology += 1
                src = lookup.get(prev)
            else:
                src = None
            for m in PRIOR_DEF_METRICS:
                rec[f"prior_{m}"] = getattr(src, m) if src is not None else np.nan
            prior_rows.append(rec)
            prev = key
    prior = pd.DataFrame(prior_rows)
    before = len(frame)
    out = frame.merge(prior, on=["season", "week", "team"], how="left", validate="one_to_one")
    return out, {
        "prior_def_state_chronology_violations": int(chronology),
        "redundancy_join_fanout": int(len(out) - before),
        "target_game_team_state_used": False,
    }


def _redundancy_design(frame: pd.DataFrame):
    prior = [f"prior_{m}" for m in PRIOR_DEF_METRICS]
    cols = ["season", "week", "team", CANDIDATE, IMMEDIATE, *prior]
    z = frame[cols].copy()
    z[CANDIDATE] = pd.to_numeric(z[CANDIDATE], errors="coerce")
    z = z.dropna(subset=[CANDIDATE])
    tr = z.loc[z["season"].between(2019, 2023)].copy()
    te = z.loc[z["season"].between(2024, 2025)].copy()
    ntr, nte = len(tr), len(te)
    if ntr < REDUNDANCY_TRAIN_MIN or nte < REDUNDANCY_HOLDOUT_MIN:
        return np.empty((0,0)), np.empty(0), np.empty((0,0)), np.empty(0), ntr, nte

    tr["team"] = tr["team"].fillna("").astype(str).replace("", "__MISSING__")
    te["team"] = te["team"].fillna("").astype(str).replace("", "__MISSING__")
    teams = sorted(tr["team"].unique().tolist())
    numeric = ["week", IMMEDIATE, *prior]
    for c in numeric:
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        te[c] = pd.to_numeric(te[c], errors="coerce")
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)

    def mat(q):
        pieces = [np.ones((len(q),1)), q[numeric].to_numpy(float)]
        for t in teams:
            pieces.append(q["team"].eq(t).astype(float).to_numpy()[:,None])
        return np.hstack(pieces)
    return mat(tr), tr[CANDIDATE].to_numpy(float), mat(te), te[CANDIDATE].to_numpy(float), ntr, nte


def redundancy_audit(frame: pd.DataFrame) -> pd.DataFrame:
    Xtr,ytr,Xte,yte,ntr,nte = _redundancy_design(frame)
    r2 = np.nan
    if ntr >= REDUNDANCY_TRAIN_MIN and nte >= REDUNDANCY_HOLDOUT_MIN:
        beta,*_ = np.linalg.lstsq(Xtr,ytr,rcond=None)
        pred=Xte@beta
        sst=float(np.sum((yte-yte.mean())**2))
        r2=1-float(np.sum((yte-pred)**2))/sst if sst>0 else np.nan
    if not np.isfinite(r2):
        disp="REDUNDANCY_UNRESOLVED_SOURCE_THIN"
    elif r2>=REDUNDANCY_HIGH:
        disp="HIGHLY_RECONSTRUCTIBLE_REDUNDANT"
    elif r2>=REDUNDANCY_REVIEW:
        disp="REDUNDANCY_REVIEW"
    else:
        disp="INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE"
    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "reconstruction_inputs": (
            "def_front_roster_continuity_share_prev_game|prior_pressure_rate_generated|"
            "prior_success_rate_def|prior_def_pass_epa|prior_explosive_play_rate_allowed|"
            "team_onehot|target_week"
        ),
        "train_rows": int(ntr),
        "holdout_rows": int(nte),
        "holdout_reconstructibility_r2": float(r2) if np.isfinite(r2) else np.nan,
        "redundancy_disposition": disp,
        "target_game_pbp_used": False,
        "sportsbook_read": False,
    }])


def qualify(frame, identity, integrity, stability, redundancy):
    eligible=len(frame); known=int(frame[CANDIDATE].notna().sum())
    coverage=known/eligible if eligible else 0
    st=stability.iloc[0]; rd=redundancy.iloc[0]
    clean=(
        identity["stable_id_coverage"]>=MIN_STABLE_ID
        and identity["ambiguous_same_week_gsis_team_conflicts"]==0
        and integrity["published_duplicate_team_week_rows"]==0
        and integrity["chronology_violations"]==0
        and integrity["schedule_join_fanout"]==0
        and integrity["prior_def_state_chronology_violations"]==0
        and integrity["redundancy_join_fanout"]==0
    )
    red=str(rd["redundancy_disposition"])
    if not clean:
        disp="REJECTED_INTEGRITY"; reason="identity/chronology/duplicate/fanout gate failed"
    elif eligible<MIN_ELIGIBLE or coverage<MIN_COVERAGE:
        disp="ENGINEERING_READY_SOURCE_THIN"; reason="coverage/sample floor failed"
    elif not bool(st["stability_gate_passed"]):
        disp="DESCRIPTIVE_ONLY"; reason="accumulated front cohesion failed frozen stability gate"
    elif red=="REDUNDANCY_UNRESOLVED_SOURCE_THIN":
        disp="ENGINEERING_READY_SOURCE_THIN"; reason="redundancy support unresolved"
    elif red in {"HIGHLY_RECONSTRUCTIBLE_REDUNDANT","REDUNDANCY_REVIEW"}:
        disp="DESCRIPTIVE_ONLY"; reason="too reconstructible from immediate front continuity and prior defense state"
    else:
        disp="READY_FOR_FROZEN_EXPERIMENT"; reason="all frozen qualification gates passed"
    return pd.DataFrame([{
        "feature_name": CANDIDATE,
        "family": "DEFENSIVE_FRONT_PAIRWISE_COHESION_V1",
        "grain": "scheduled_team_game",
        "seasons_available": "2019-2025",
        "eligible_rows": int(eligible),
        "observed_rows": known,
        "pregame_coverage": float(coverage),
        "stable_id_coverage": float(identity["stable_id_coverage"]),
        "unknown_rate": float(1-coverage),
        "stability_value": st["adjacent_game_spearman"],
        "stability_gate_passed": bool(st["stability_gate_passed"]),
        "holdout_reconstructibility_r2": rd["holdout_reconstructibility_r2"],
        "redundancy_disposition": red,
        "qualification_disposition": disp,
        "qualification_reason": reason,
    }])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--roster",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--git-sha",required=True)
    a=ap.parse_args()

    schedule=ol.normalize_schedule(pd.read_csv(a.schedule,low_memory=False))
    roster=pd.read_csv(a.roster,low_memory=False)
    team_weekly=pd.read_csv(a.team_weekly,low_memory=False)
    sets,identity=build_front_sets(roster,schedule)
    frame,base_integrity=materialize(schedule,sets)
    enriched,red_integrity=attach_prior_def_state(frame,team_weekly)
    integrity={**base_integrity,**red_integrity}
    support=support_table(frame)
    stability=stability_evidence(frame)
    redundancy=redundancy_audit(enriched)
    inventory=qualify(frame,identity,integrity,stability,redundancy)

    out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    support.to_csv(out/"def_front_cohesion_support_v1.csv",index=False)
    stability.to_csv(out/"def_front_cohesion_stability_v1.csv",index=False)
    redundancy.to_csv(out/"def_front_cohesion_redundancy_v1.csv",index=False)
    inventory.to_csv(out/"def_front_cohesion_qualification_inventory_v1.csv",index=False)
    manifest={
        "qualification_version":"DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1",
        "frozen_pre_result_disposition":"DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1_FROZEN_PRE_RESULT",
        "git_sha":a.git_sha,
        "candidate":CANDIDATE,
        "immediate_candidate":IMMEDIATE,
        "front_positions":sorted(FRONT_POSITIONS),
        "lookback_scheduled_team_games":LOOKBACK_GAMES,
        "schedule_sha256":_sha256(a.schedule),
        "weekly_roster_sha256":_sha256(a.roster),
        "team_weekly_sha256":_sha256(a.team_weekly),
        "identity":identity,
        "integrity":integrity,
        "qualification_disposition":str(inventory.iloc[0]["qualification_disposition"]),
        "predictive_outcomes_scored":False,
        "sportsbook_read":False,
        "production_changed":False,
        "issue_535_touched":False,
        "target_game_pbp_used":False,
        "target_game_snap_or_participation_used":False,
    }
    (out/"def_front_cohesion_manifest_v1.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("QUALIFICATION"); print(inventory.to_string(index=False))
    print("\nBROAD SUPPORT"); print(support.loc[support["slice"].eq("ALL")].to_string(index=False))
    print("\nSTABILITY"); print(stability.to_string(index=False))
    print("\nREDUNDANCY"); print(redundancy.to_string(index=False))
    print("\nMANIFEST"); print(json.dumps(manifest,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
