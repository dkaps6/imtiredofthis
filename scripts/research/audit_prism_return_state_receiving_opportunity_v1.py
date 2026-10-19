#!/usr/bin/env python3
"""PRISM Return-State Receiving Opportunity V1 source audit.

Maps source-certified restored-return episodes to exact same-player receiving
opportunity evidence and finite team target volume. Source/lineage only:
no target-restoration relationship is scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "930907dd91c1b4139f2aa6be52db23a730c7a96b"
SEASONS = (2024, 2025)
FAMILIES = {"RB", "WR", "TE"}
RETURN_RESTORED = "RETURN_PARTICIPATION_RESTORED"
CLEAN_STATE = "NO_REPORT_PARTICIPATED"
ALLOWED_STATES = {
    "NO_REPORT_PARTICIPATED",
    "REPORTED_INJURY_PARTICIPATED",
    "REPORTED_INJURY_NO_OFFENSIVE_SNAPS",
    "NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED",
}
MAX_PRIOR = 4
MIN_PRIOR = 3
TOL = 1e-10

EP_REQ = {
    "season","team","stable_id","family","episode_number",
    "episode_start_week","pre_episode_snap_pct","return_week","return_state",
}
REC_REQ = {
    "season","week","team","stable_id","family","positive_participation",
    "offense_snaps","offense_pct","observation_state",
    "playerform_weekly_row_present","targets","team_targets","target_share_game",
}
HEALTH_REQ = {
    "season","week","team","stable_id","family","positive_participation",
    "observation_state",
}


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true":True,"false":False,"1":True,"0":False})
        .fillna(False).astype(bool)
    )


def load_episodes(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = EP_REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"episode parent missing {sorted(miss)}")

    for c in ("season","episode_number","episode_start_week","return_week"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["pre_episode_snap_pct"] = pd.to_numeric(
        x["pre_episode_snap_pct"], errors="coerce"
    )
    x["stable_id"] = x["stable_id"].map(text)
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["return_state"] = x["return_state"].astype("string").fillna("").str.strip()

    x = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].isin(FAMILIES)
        & x["stable_id"].ne("")
        & x["return_state"].eq(RETURN_RESTORED)
    ].copy()
    x["season"] = x["season"].astype(int)
    x["episode_number"] = x["episode_number"].astype(int)
    x["episode_start_week"] = x["episode_start_week"].astype(int)

    return x.sort_values(
        ["season","team","stable_id","episode_start_week","episode_number"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_receiving(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REC_REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"receiving parent missing {sorted(miss)}")

    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    for c in ("offense_snaps","offense_pct","targets","team_targets","target_share_game"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["stable_id"] = x["stable_id"].map(text)
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["observation_state"] = (
        x["observation_state"].astype("string").fillna("").str.strip()
    )
    x["positive_participation"] = as_bool(x["positive_participation"])
    x["playerform_weekly_row_present"] = as_bool(x["playerform_weekly_row_present"])

    x = x.loc[
        x["season"].isin(SEASONS)
        & x["stable_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)

    keys = ["season","week","team","stable_id"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate receiving identities: {bad}")

    return x.sort_values(
        ["season","team","stable_id","week"], kind="mergesort"
    ).reset_index(drop=True)



def load_health(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = HEALTH_REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"health parent missing {sorted(miss)}")

    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["stable_id"] = x["stable_id"].map(text)
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["observation_state"] = (
        x["observation_state"].astype("string").fillna("").str.strip()
    )
    x["positive_participation"] = as_bool(x["positive_participation"])

    x = x.loc[
        x["season"].isin(SEASONS)
        & x["stable_id"].ne("")
        & x["family"].isin(FAMILIES)
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)

    keys = ["season","week","team","stable_id"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate health identities: {bad}")

    return x.sort_values(
        ["season","team","stable_id","week"], kind="mergesort"
    ).reset_index(drop=True)

def valid_opportunity_row(r: pd.Series) -> bool:
    return bool(
        r["positive_participation"]
        and r["playerform_weekly_row_present"]
        and pd.notna(r["targets"])
        and float(r["targets"]) >= 0
        and pd.notna(r["team_targets"])
        and float(r["team_targets"]) > 0
        and pd.notna(r["target_share_game"])
        and np.isfinite(float(r["target_share_game"]))
    )


def build_membership(
    episodes: pd.DataFrame,
    receiving: pd.DataFrame,
    health: pd.DataFrame,
) -> tuple[pd.DataFrame,pd.DataFrame,set[tuple[int,int,str]],dict]:
    rec_groups = {
        (int(s),str(t),str(pid)): g.sort_values("week", kind="mergesort").reset_index(drop=True)
        for (s,t,pid),g in receiving.groupby(["season","team","stable_id"], sort=False)
    }
    health_groups = {
        (int(s),str(t),str(pid)): g.sort_values("week", kind="mergesort").reset_index(drop=True)
        for (s,t,pid),g in health.groupby(["season","team","stable_id"], sort=False)
    }

    membership = []
    prior_rows = []
    team_weeks: set[tuple[int,int,str]] = set()
    chronology_violations = 0
    skipped_next_violations = 0

    for e in episodes.to_dict("records"):
        season = int(e["season"])
        team = str(e["team"])
        pid = str(e["stable_id"])
        family = str(e["family"])
        start_week = int(e["episode_start_week"])
        return_week = int(e["return_week"]) if pd.notna(e["return_week"]) else -1
        pre_snap = float(e["pre_episode_snap_pct"])

        if return_week <= start_week:
            chronology_violations += 1

        g = rec_groups.get((season,team,pid))
        if g is None:
            g = pd.DataFrame(columns=receiving.columns)
        hg = health_groups.get((season,team,pid))
        if hg is None:
            hg = pd.DataFrame(columns=health.columns)

        prior = g.loc[
            (g["week"] < start_week)
            & g["positive_participation"]
        ].tail(MAX_PRIOR).copy()

        prior_chron_ok = bool(
            len(prior) == 0 or prior["week"].lt(start_week).all()
        )
        if not prior_chron_ok:
            chronology_violations += 1

        prior_rows_valid = [
            valid_opportunity_row(r) for _,r in prior.iterrows()
        ]
        prior_support_ok = bool(
            len(prior) >= MIN_PRIOR
            and all(prior_rows_valid)
        )

        for _,r in prior.iterrows():
            team_weeks.add((season,int(r["week"]),team))
            prior_rows.append({
                "season":season,
                "team":team,
                "stable_id":pid,
                "family":family,
                "episode_number":int(e["episode_number"]),
                "episode_start_week":start_week,
                "return_week":return_week,
                "prior_week":int(r["week"]),
                "targets":float(r["targets"]) if pd.notna(r["targets"]) else np.nan,
                "team_targets":float(r["team_targets"]) if pd.notna(r["team_targets"]) else np.nan,
                "target_share_game":float(r["target_share_game"]) if pd.notna(r["target_share_game"]) else np.nan,
                "offense_snaps":float(r["offense_snaps"]) if pd.notna(r["offense_snaps"]) else np.nan,
                "offense_pct":float(r["offense_pct"]) if pd.notna(r["offense_pct"]) else np.nan,
                "observation_state":str(r["observation_state"]),
                "playerform_weekly_row_present":bool(r["playerform_weekly_row_present"]),
                "opportunity_row_valid":bool(valid_opportunity_row(r)),
            })

        ret = g.loc[g["week"].eq(return_week)].copy()
        return_unique = len(ret) == 1
        return_valid = False
        return_targets = np.nan
        return_team_targets = np.nan
        return_share = np.nan
        return_state = ""
        return_pf = False
        if return_unique:
            rr = ret.iloc[0]
            return_valid = valid_opportunity_row(rr)
            return_targets = float(rr["targets"]) if pd.notna(rr["targets"]) else np.nan
            return_team_targets = float(rr["team_targets"]) if pd.notna(rr["team_targets"]) else np.nan
            return_share = float(rr["target_share_game"]) if pd.notna(rr["target_share_game"]) else np.nan
            return_state = str(rr["observation_state"])
            return_pf = bool(rr["playerform_weekly_row_present"])
            team_weeks.add((season,return_week,team))

        # The immutable health-state parent—not the positive-participation
        # receiving panel—defines the exact immediately next scheduled same-team
        # observation after the restored return.
        hnext = hg.loc[hg["week"] > return_week].sort_values("week", kind="mergesort")
        next_present = len(hnext) > 0
        next_week = np.nan
        next_state = ""
        next_positive = False
        next_pf = False
        next_targets = np.nan
        next_team_targets = np.nan
        next_share = np.nan
        next_receiving_row_present = False
        next_valid_if_positive = True
        if next_present:
            hr = hnext.iloc[0]
            next_week = int(hr["week"])
            if next_week <= return_week:
                chronology_violations += 1
            next_state = str(hr["observation_state"])
            next_positive = bool(hr["positive_participation"])

            first_later_week = int(hg.loc[hg["week"] > return_week, "week"].min())
            if next_week != first_later_week:
                skipped_next_violations += 1

            exact = g.loc[g["week"].eq(next_week)].copy()
            next_receiving_row_present = len(exact) == 1
            if next_receiving_row_present:
                nr = exact.iloc[0]
                next_pf = bool(nr["playerform_weekly_row_present"])
                next_targets = float(nr["targets"]) if pd.notna(nr["targets"]) else np.nan
                next_team_targets = float(nr["team_targets"]) if pd.notna(nr["team_targets"]) else np.nan
                next_share = float(nr["target_share_game"]) if pd.notna(nr["target_share_game"]) else np.nan
                team_weeks.add((season,next_week,team))
                if next_positive:
                    next_valid_if_positive = valid_opportunity_row(nr)
            elif next_positive:
                next_valid_if_positive = False

        membership.append({
            "season":season,
            "team":team,
            "stable_id":pid,
            "family":family,
            "episode_number":int(e["episode_number"]),
            "episode_start_week":start_week,
            "return_week":return_week,
            "pre_episode_snap_pct":pre_snap,
            "prior_rows":int(len(prior)),
            "prior_support_ok":prior_support_ok,
            "return_row_unique":return_unique,
            "return_opportunity_valid":bool(return_valid),
            "return_observation_state":return_state,
            "return_playerform_weekly_row_present":return_pf,
            "return_targets":return_targets,
            "return_team_targets":return_team_targets,
            "return_target_share_game":return_share,
            "immediate_next_present":next_present,
            "immediate_next_week":next_week,
            "immediate_next_state":next_state,
            "immediate_next_positive_participation":next_positive,
            "immediate_next_receiving_row_present":next_receiving_row_present,
            "immediate_next_playerform_weekly_row_present":next_pf,
            "immediate_next_opportunity_valid_if_positive":bool(next_valid_if_positive),
            "immediate_next_targets":next_targets,
            "immediate_next_team_targets":next_team_targets,
            "immediate_next_target_share_game":next_share,
            "immediate_next_clean":bool(next_state == CLEAN_STATE),
        })

    mem = pd.DataFrame(membership)
    pri = pd.DataFrame(prior_rows)
    audit = {
        "chronology_violations":int(chronology_violations),
        "skipped_immediate_next_violations":int(skipped_next_violations),
    }
    return mem,pri,team_weeks,audit

def finite_opportunity_audit(
    receiving: pd.DataFrame,
    represented: set[tuple[int,int,str]],
) -> tuple[pd.DataFrame,dict]:
    rows = []
    unique_violations = 0
    nonfinite_team_weeks = 0
    target_sum_violations = 0
    negative_residual_violations = 0

    for season,week,team in sorted(represented):
        g = receiving.loc[
            receiving["season"].eq(season)
            & receiving["week"].eq(week)
            & receiving["team"].eq(team)
        ].copy()
        vals = pd.to_numeric(g["team_targets"], errors="coerce").dropna().unique()
        unique_ok = len(vals) == 1
        if not unique_ok:
            unique_violations += 1
            team_targets = np.nan
        else:
            team_targets = float(vals[0])

        finite_positive = bool(np.isfinite(team_targets) and team_targets > 0)
        if not finite_positive:
            nonfinite_team_weeks += 1

        skill = g.loc[g["family"].isin(FAMILIES)].copy()
        skill_targets = float(
            pd.to_numeric(skill["targets"], errors="coerce").fillna(0).sum()
        )
        residual = (
            float(team_targets - skill_targets)
            if finite_positive else np.nan
        )
        sum_ok = bool(
            finite_positive and skill_targets <= team_targets + TOL
        )
        residual_ok = bool(
            finite_positive and residual >= -TOL
        )
        if not sum_ok:
            target_sum_violations += 1
        if not residual_ok:
            negative_residual_violations += 1

        rows.append({
            "season":season,
            "week":week,
            "team":team,
            "team_targets":team_targets,
            "rb_wr_te_targets":skill_targets,
            "residual_team_targets":residual,
            "team_targets_unique":unique_ok,
            "team_targets_finite_positive":finite_positive,
            "target_sum_within_team_total":sum_ok,
            "residual_nonnegative":residual_ok,
        })

    out = pd.DataFrame(rows)
    audit = {
        "represented_team_weeks":int(len(out)),
        "team_target_unique_violations":int(unique_violations),
        "nonfinite_or_nonpositive_team_target_weeks":int(nonfinite_team_weeks),
        "rb_wr_te_target_sum_exceeds_team_total_weeks":int(target_sum_violations),
        "negative_residual_team_target_weeks":int(negative_residual_violations),
    }
    return out,audit


def summarize(
    mem: pd.DataFrame,
    pri: pd.DataFrame,
    finite: pd.DataFrame,
    season: int,
    duplicate_episode_identities: int,
    duplicate_receiving_identities: int,
    audit: dict,
) -> dict:
    s = mem.loc[mem["season"].eq(season)].copy()
    p = pri.loc[pri["season"].eq(season)].copy()
    f = finite.loc[finite["season"].eq(season)].copy()

    next_states = (
        s.loc[s["immediate_next_present"], "immediate_next_state"]
        .value_counts(dropna=False)
        .to_dict()
    )

    positive_next = s.loc[
        s["immediate_next_present"]
        & s["immediate_next_positive_participation"]
    ].copy()

    families = []
    for fam,g in s.groupby("family", sort=True):
        families.append({
            "family":str(fam),
            "restored_episodes":int(len(g)),
            "named_players":int(g["stable_id"].nunique()),
            "prior_support_episodes":int(g["prior_support_ok"].sum()),
            "return_valid_episodes":int(g["return_opportunity_valid"].sum()),
            "immediate_next_present":int(g["immediate_next_present"].sum()),
            "immediate_next_clean":int(g["immediate_next_clean"].sum()),
        })

    restored = int(len(s))
    players = int(s["stable_id"].nunique())
    prior_support_share = float(s["prior_support_ok"].mean()) if restored else 0.0
    return_coverage = float(s["return_row_unique"].mean()) if restored else 0.0
    next_coverage = float(s["immediate_next_present"].mean()) if restored else 0.0

    prior_valid = bool(
        len(p) == 0
        or (
            p["playerform_weekly_row_present"].all()
            and p["targets"].notna().all()
            and p["targets"].ge(0).all()
            and p["team_targets"].notna().all()
            and p["team_targets"].gt(0).all()
            and p["target_share_game"].notna().all()
        )
    )

    ret_admitted = s.loc[s["return_row_unique"]].copy()
    ret_valid_fields = bool(
        len(ret_admitted) > 0
        and ret_admitted["return_opportunity_valid"].all()
    )

    next_positive_valid = bool(
        len(positive_next) == 0
        or positive_next["immediate_next_opportunity_valid_if_positive"].all()
    )

    state_vocab_ok = bool(
        set(str(v) for v in next_states.keys() if str(v))
        <= ALLOWED_STATES
    )

    gates = {
        "restored_episodes_ge_100":restored >= 100,
        "restored_named_players_ge_90":players >= 90,
        "stable_id_coverage_eq_100":bool(s["stable_id"].ne("").all()),
        "duplicate_episode_identities_zero":duplicate_episode_identities == 0,
        "duplicate_receiving_identities_zero":duplicate_receiving_identities == 0,
        "return_after_episode_start_all":bool((s["return_week"] > s["episode_start_week"]).all()),
        "prior_rows_strictly_before_start_all":bool(
            len(p) == 0 or (p["prior_week"] < p["episode_start_week"]).all()
        ),
        "immediate_next_after_return_all":bool(
            (
                ~s["immediate_next_present"]
                | (pd.to_numeric(s["immediate_next_week"], errors="coerce") > s["return_week"])
            ).all()
        ),
        "skipped_immediate_next_zero":audit["skipped_immediate_next_violations"] == 0,
        "prior_support_share_ge_095":prior_support_share >= 0.95,
        "admitted_prior_playerform_present_all":prior_valid,
        "admitted_prior_targets_finite_nonnegative_all":prior_valid,
        "admitted_prior_team_targets_finite_positive_all":prior_valid,
        "admitted_prior_target_share_finite_all":prior_valid,
        "return_row_coverage_ge_099":return_coverage >= 0.99,
        "return_rows_positive_participation_all":ret_valid_fields,
        "return_rows_playerform_present_all":ret_valid_fields,
        "return_targets_finite_nonnegative_all":ret_valid_fields,
        "return_team_targets_finite_positive_all":ret_valid_fields,
        "return_target_share_finite_all":ret_valid_fields,
        "immediate_next_coverage_ge_075":next_coverage >= 0.75,
        "immediate_next_state_vocab_source_certified":state_vocab_ok,
        "immediate_next_clean_nonzero":bool(s["immediate_next_clean"].sum() > 0),
        "positive_immediate_next_playerform_present_all":next_positive_valid,
        "team_target_unique_all":bool(
            len(f) > 0 and f["team_targets_unique"].all()
        ),
        "represented_team_targets_finite_positive_all":bool(
            len(f) > 0 and f["team_targets_finite_positive"].all()
        ),
        "rb_wr_te_target_sum_within_team_total_all":bool(
            len(f) > 0 and f["target_sum_within_team_total"].all()
        ),
        "residual_team_targets_nonnegative_all":bool(
            len(f) > 0 and f["residual_nonnegative"].all()
        ),
    }

    return {
        "season":season,
        "restored_episodes":restored,
        "restored_named_players":players,
        "prior_support_episodes":int(s["prior_support_ok"].sum()),
        "prior_support_share":prior_support_share,
        "exact_return_row_episodes":int(s["return_row_unique"].sum()),
        "exact_return_row_coverage":return_coverage,
        "immediate_next_observation_episodes":int(s["immediate_next_present"].sum()),
        "immediate_next_observation_coverage":next_coverage,
        "immediate_next_clean_episodes":int(s["immediate_next_clean"].sum()),
        "immediate_next_state_counts":{str(k):int(v) for k,v in next_states.items()},
        "positive_immediate_next_rows":int(len(positive_next)),
        "positive_immediate_next_valid_rows":int(
            positive_next["immediate_next_opportunity_valid_if_positive"].sum()
        ) if len(positive_next) else 0,
        "represented_team_weeks":int(len(f)),
        "family_summary":families,
        "gates":gates,
        "source_ready":bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--receiving", type=Path, required=True)
    ap.add_argument("--health", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(a.episodes)
    receiving = load_receiving(a.receiving)
    health = load_health(a.health)

    ep_dup = int(
        episodes.duplicated(["season","team","stable_id","episode_number"]).sum()
    )
    rec_dup = int(
        receiving.duplicated(["season","week","team","stable_id"]).sum()
    )

    mem,pri,represented,audit = build_membership(episodes,receiving,health)
    finite,finite_audit = finite_opportunity_audit(receiving,represented)

    summaries = [
        summarize(mem,pri,finite,s,ep_dup,rec_dup,audit)
        for s in SEASONS
    ]

    integrity = {
        "chronology_violations_zero":audit["chronology_violations"] == 0,
        "predictive_target_relationships_scored_zero":True,
        "target_restoration_labels_created_zero":True,
        "coefficients_fit_zero":True,
        "thresholds_searched_zero":True,
        "routes_inferred_from_snaps_false":True,
        "outcomes_2026_zero":True,
        "sportsbook_oddsapi_zero":True,
        "production_changed_false":True,
    }

    ready = bool(
        all(s["source_ready"] for s in summaries)
        and all(integrity.values())
    )

    result = {
        "version":VERSION,
        "freeze_commit":FREEZE_COMMIT,
        "return_parent_run":39162437467,
        "return_parent_artifact":12149417234,
        "return_parent_digest":"sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e",
        "receiving_parent_run":39161711741,
        "receiving_parent_artifact":12149181200,
        "receiving_parent_digest":"sha256:6cae427b667db4a120a1308747d00b82b6770fe0adbc46d3efc68fce3e1db240",
        "health_parent_run":39154145666,
        "health_parent_artifact":12146625382,
        "health_parent_digest":"sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb",
        "disposition":(
            "PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_READY"
            if ready else
            "PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_BLOCKED"
        ),
        "source_ready":ready,
        "audit":audit,
        "finite_opportunity_audit":finite_audit,
        "season_summaries":summaries,
        "integrity_gates":integrity,
        "predictive_target_relationships_scored":0,
        "target_restoration_labels_created":0,
        "coefficients_fit":0,
        "thresholds_searched":0,
        "routes_inferred_from_snaps":False,
        "outcomes_2026_used":0,
        "sportsbook_inputs_used":0,
        "oddsapi_calls":0,
        "production_changed":False,
    }

    mem.to_csv(
        a.out_dir/"return_state_receiving_opportunity_membership.csv",
        index=False,float_format="%.12f"
    )
    pri.to_csv(
        a.out_dir/"return_state_receiving_opportunity_prior_rows.csv",
        index=False,float_format="%.12f"
    )
    finite.to_csv(
        a.out_dir/"return_state_receiving_opportunity_finite_team_audit.csv",
        index=False,float_format="%.12f"
    )
    pd.DataFrame([
        dict(season=s["season"], **f)
        for s in summaries for f in s["family_summary"]
    ]).to_csv(
        a.out_dir/"return_state_receiving_opportunity_family_summary.csv",
        index=False
    )
    (a.out_dir/"return_state_receiving_opportunity_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
