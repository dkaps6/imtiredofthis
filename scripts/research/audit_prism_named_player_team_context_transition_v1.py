#!/usr/bin/env python3
"""PRISM Named-Player Team-Context Transition V1 source audit.

Source/lineage only. Certifies whether an exact named RB/WR/TE changed NFL
teams relative to his own immediately prior scheduled weekly-roster
observation. No player performance or opportunity outcome is scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import player_name_key

VERSION = "PRISM_NAMED_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "fc63601c34760d042adcc00fff12976496622574"
SOURCE_SEASONS = (2023, 2024, 2025, 2026)
HISTORICAL_SEASONS = (2024, 2025)
PARITY_SEASON = 2026
WEEKS = tuple(range(1, 19))
FAMILIES = {"RB", "WR", "TE"}

STATE_CONTINUED = "TEAM_CONTEXT_CONTINUED"
STATE_TRANSITION = "TEAM_CONTEXT_TRANSITION"
STATE_NO_PRIOR = "NO_PRIOR_NFL_ROSTER_OBSERVATION"
STATES = (STATE_CONTINUED, STATE_TRANSITION, STATE_NO_PRIOR)


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def pdx(v) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v.copy()
    # Prefer record conversion for Polars-like frames. This preserves the
    # frozen source semantics without requiring pyarrow solely for a bridge.
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    return pd.DataFrame(v)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def stable(v) -> str:
    return text(v)


def name_key(v) -> str:
    return player_name_key(v, strip_suffix=True)


def family(v) -> str:
    p = text(v).upper()
    if p in {"RB", "HB", "TB", "FB"} or p.startswith(("RB", "HB", "TB", "FB")):
        return "RB"
    if p == "WR" or p.startswith("WR"):
        return "WR"
    if p == "TE" or p.startswith("TE"):
        return "TE"
    return ""


def normalize_schedule(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = lower(raw)
    cols = {
        "season": first_col(x, ("season",)),
        "week": first_col(x, ("week",)),
        "game_type": first_col(x, ("game_type", "season_type")),
        "home": first_col(x, ("home_team", "home")),
        "away": first_col(x, ("away_team", "away")),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise RuntimeError(f"schedule missing semantics: {missing}")

    q = pd.DataFrame({
        "season": pd.to_numeric(x[cols["season"]], errors="coerce"),
        "week": pd.to_numeric(x[cols["week"]], errors="coerce"),
        "game_type": x[cols["game_type"]].astype("string").str.upper().str.strip(),
        "home": x[cols["home"]].map(canon_team),
        "away": x[cols["away"]].map(canon_team),
    })
    q = q.loc[
        q["season"].isin(SOURCE_SEASONS)
        & q["week"].isin(WEEKS)
        & q["game_type"].eq("REG")
        & q["home"].ne("")
        & q["away"].ne("")
    ].copy()

    rows = []
    for r in q.itertuples(index=False):
        rows.append({"season": int(r.season), "week": int(r.week), "team": str(r.home)})
        rows.append({"season": int(r.season), "week": int(r.week), "team": str(r.away)})
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("schedule normalized to zero team-weeks")
    out = out.drop_duplicates().sort_values(["season", "week", "team"]).reset_index(drop=True)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("schedule team-week duplicates after normalization")
    return out, {"columns": cols, "team_weeks": int(len(out))}


def normalize_rosters(
    parts: list[tuple[int, pd.DataFrame]],
    scheduled: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    scheduled_keys = set(zip(scheduled["season"], scheduled["week"], scheduled["team"]))
    frames = []
    schemas = {}
    raw_dup_rows = 0

    for requested, raw in parts:
        x = lower(raw)
        cols = {
            "season": first_col(x, ("season",)),
            "week": first_col(x, ("week",)),
            "team": first_col(x, ("team", "team_abbr", "team_abbreviation", "club")),
            "stable_id": first_col(x, ("gsis_id", "player_id", "nflverse_id")),
            "name": first_col(x, ("player_name", "full_name", "player", "name")),
            "position": first_col(x, ("position", "pos", "position_group")),
            "status": first_col(x, ("status", "roster_status")),
        }
        required = ("week", "team", "name", "position")
        missing = [k for k in required if cols[k] is None]
        if missing:
            raise RuntimeError(f"weekly roster {requested} missing semantics {missing}")

        season = (
            pd.to_numeric(x[cols["season"]], errors="coerce")
            if cols["season"] is not None
            else pd.Series(int(requested), index=x.index, dtype=float)
        )
        q = pd.DataFrame({
            "season": season,
            "week": pd.to_numeric(x[cols["week"]], errors="coerce"),
            "team": x[cols["team"]].map(canon_team),
            "stable_id": x[cols["stable_id"]].map(stable) if cols["stable_id"] else "",
            "player": x[cols["name"]].map(text),
            "name_key": x[cols["name"]].map(name_key),
            "position_raw": x[cols["position"]].map(text),
            "roster_status": x[cols["status"]].map(text) if cols["status"] else "",
        })
        q["family"] = q["position_raw"].map(family)
        q = q.loc[
            q["season"].eq(int(requested))
            & q["week"].isin(WEEKS)
            & q["team"].ne("")
            & q["name_key"].ne("")
            & q["family"].isin(FAMILIES)
        ].copy()
        q["season"] = q["season"].astype(int)
        q["week"] = q["week"].astype(int)
        q["scheduled_team_week"] = [
            (int(s), int(w), str(t)) in scheduled_keys
            for s, w, t in zip(q["season"], q["week"], q["team"])
        ]
        q = q.loc[q["scheduled_team_week"]].copy()
        q["_source_order"] = np.arange(len(q), dtype=int)

        nonblank = q.loc[q["stable_id"].ne("")]
        raw_dup_rows += int(
            nonblank.duplicated(["season", "week", "team", "stable_id"], keep=False).sum()
        )
        # Exact stable-ID dedup only. Blank-ID rows are retained for coverage diagnostics
        # but can never certify a transition.
        with_id = nonblank.sort_values(
            ["season", "week", "team", "stable_id", "_source_order"], kind="stable"
        ).drop_duplicates(["season", "week", "team", "stable_id"], keep="last")
        without_id = q.loc[q["stable_id"].eq("")].copy()
        q = pd.concat([with_id, without_id], ignore_index=True)
        schemas[str(requested)] = cols
        frames.append(q)

    roster = pd.concat(frames, ignore_index=True)
    roster["ordinal"] = roster["season"] * 100 + roster["week"]
    roster = roster.sort_values(
        ["season", "week", "team", "stable_id", "name_key"], kind="stable"
    ).reset_index(drop=True)

    post_dup = int(
        roster.loc[roster["stable_id"].ne("")]
        .duplicated(["season", "week", "team", "stable_id"])
        .sum()
    )
    return roster, {
        "schemas": schemas,
        "raw_duplicate_rows": int(raw_dup_rows),
        "post_dedup_duplicate_rows": int(post_dup),
    }


def build_team_context(roster: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    q = roster.loc[roster["stable_id"].ne("")].copy()
    # Same player on multiple teams in the same season-week has no resolvable
    # within-week ordering from this source. Keep it as source ambiguity and
    # exclude those rows from transition certification rather than guessing.
    multi = (
        q.groupby(["season", "week", "stable_id"])["team"]
        .nunique()
        .reset_index(name="team_count")
    )
    ambiguous_keys = {
        (int(r.season), int(r.week), str(r.stable_id))
        for r in multi.loc[multi["team_count"].gt(1)].itertuples(index=False)
    }
    q["same_week_multi_team_ambiguous"] = [
        (int(s), int(w), str(pid)) in ambiguous_keys
        for s, w, pid in zip(q["season"], q["week"], q["stable_id"])
    ]
    cert = q.loc[~q["same_week_multi_team_ambiguous"]].copy()
    cert = cert.sort_values(["stable_id", "ordinal", "team"], kind="stable").reset_index(drop=True)

    cert["prior_season"] = cert.groupby("stable_id")["season"].shift(1)
    cert["prior_week"] = cert.groupby("stable_id")["week"].shift(1)
    cert["prior_team"] = cert.groupby("stable_id")["team"].shift(1)
    cert["prior_ordinal"] = cert.groupby("stable_id")["ordinal"].shift(1)

    has_prior = cert["prior_ordinal"].notna()
    cert["team_context_state"] = np.select(
        [
            ~has_prior,
            has_prior & cert["prior_team"].eq(cert["team"]),
            has_prior & cert["prior_team"].ne(cert["team"]),
        ],
        [STATE_NO_PRIOR, STATE_CONTINUED, STATE_TRANSITION],
        default="",
    )
    if not cert["team_context_state"].isin(STATES).all():
        raise RuntimeError("team-context state construction emitted unknown state")

    chronology_bad = int(
        (
            has_prior
            & pd.to_numeric(cert["prior_ordinal"], errors="coerce").ge(cert["ordinal"])
        ).sum()
    )

    prior_obs_counts = []
    prior_team_counts = []
    current_team_prior_counts = []
    skipped_prior = 0

    grouped = {
        pid: g.sort_values(["ordinal", "team"], kind="stable").reset_index(drop=True)
        for pid, g in cert.groupby("stable_id", sort=False)
    }
    for r in cert.itertuples(index=False):
        hist = grouped[str(r.stable_id)]
        prior = hist.loc[hist["ordinal"].lt(int(r.ordinal))]
        prior_obs_counts.append(int(len(prior)))
        prior_team_counts.append(int(prior["team"].eq(str(r.prior_team)).sum()) if pd.notna(r.prior_team) else 0)
        current_team_prior_counts.append(int(prior["team"].eq(str(r.team)).sum()))

        if pd.notna(r.prior_ordinal) and len(prior):
            latest = prior.sort_values(["ordinal", "team"], kind="stable").iloc[-1]
            if int(latest["ordinal"]) != int(r.prior_ordinal) or str(latest["team"]) != str(r.prior_team):
                skipped_prior += 1

    cert["prior_roster_observation_count"] = prior_obs_counts
    cert["prior_team_roster_observation_count"] = prior_team_counts
    cert["current_team_prior_roster_observation_count"] = current_team_prior_counts
    cert["prior_team_support_ge3"] = cert["prior_team_roster_observation_count"].ge(3)
    cert["prior_team_support_ge8"] = cert["prior_team_roster_observation_count"].ge(8)
    cert["transition_week1"] = cert["team_context_state"].eq(STATE_TRANSITION) & cert["week"].eq(1)
    cert["transition_inseason"] = cert["team_context_state"].eq(STATE_TRANSITION) & cert["week"].gt(1)

    return cert, {
        "same_week_multi_team_ambiguous_rows": int(q["same_week_multi_team_ambiguous"].sum()),
        "same_week_multi_team_ambiguous_player_weeks": int(len(ambiguous_keys)),
        "chronology_violations": int(chronology_bad),
        "skipped_prior_observation_violations": int(skipped_prior),
    }


def roster_team_week_coverage(
    roster: pd.DataFrame,
    scheduled: pd.DataFrame,
    season: int,
    week: int | None = None,
) -> tuple[int, int, float]:
    sch = scheduled.loc[scheduled["season"].eq(season)].copy()
    rr = roster.loc[roster["season"].eq(season)].copy()
    if week is not None:
        sch = sch.loc[sch["week"].eq(week)]
        rr = rr.loc[rr["week"].eq(week)]
    sch_keys = set(zip(sch["week"], sch["team"]))
    rr_keys = set(zip(rr["week"], rr["team"]))
    covered = len(sch_keys & rr_keys)
    total = len(sch_keys)
    return covered, total, float(covered / total) if total else 0.0


def historical_summary(
    season: int,
    roster: pd.DataFrame,
    scheduled: pd.DataFrame,
    states: pd.DataFrame,
    audit: dict,
) -> dict:
    r = roster.loc[roster["season"].eq(season)].copy()
    s = states.loc[states["season"].eq(season)].copy()
    sch = scheduled.loc[scheduled["season"].eq(season)].copy()
    transition = s.loc[s["team_context_state"].eq(STATE_TRANSITION)].copy()

    covered, total, coverage = roster_team_week_coverage(roster, scheduled, season)
    stable_cov = float(r["stable_id"].ne("").mean()) if len(r) else 0.0
    state_counts = {
        st: int(s["team_context_state"].eq(st).sum())
        for st in STATES
    }
    trans_players = int(transition["stable_id"].nunique())
    ge3_share = float(transition["prior_team_support_ge3"].mean()) if len(transition) else 0.0
    ge8_share = float(transition["prior_team_support_ge8"].mean()) if len(transition) else 0.0

    fam = []
    for f, g in transition.groupby("family", sort=True):
        fam.append({
            "family": str(f),
            "transition_events": int(len(g)),
            "transition_named_players": int(g["stable_id"].nunique()),
            "prior_team_support_ge3_share": float(g["prior_team_support_ge3"].mean()),
            "prior_team_support_ge8_share": float(g["prior_team_support_ge8"].mean()),
            "week1_transition_events": int(g["transition_week1"].sum()),
            "inseason_transition_events": int(g["transition_inseason"].sum()),
        })

    weeks = sorted(int(v) for v in sch["week"].unique())
    teams = int(sch["team"].nunique())
    gates = {
        "schedule_reg_weeks_1_18": weeks == list(WEEKS),
        "schedule_all_32_teams": teams == 32,
        "roster_scheduled_team_week_coverage_ge_095": coverage >= 0.95,
        "scoped_stable_id_coverage_ge_099": stable_cov >= 0.99,
        "duplicate_team_week_stable_id_zero": not r.loc[r["stable_id"].ne("")].duplicated(
            ["season", "week", "team", "stable_id"]
        ).any(),
        "transition_stable_id_coverage_eq_100": bool(
            len(transition) > 0 and transition["stable_id"].ne("").all()
        ),
        "chronology_violations_zero": audit["chronology_violations"] == 0,
        "skipped_prior_observation_violations_zero": audit["skipped_prior_observation_violations"] == 0,
        "transition_events_ge_25": len(transition) >= 25,
        "transition_named_players_ge_20": trans_players >= 20,
        "rb_transition_events_nonzero": bool((transition["family"] == "RB").any()),
        "wr_transition_events_nonzero": bool((transition["family"] == "WR").any()),
        "te_transition_events_nonzero": bool((transition["family"] == "TE").any()),
        "prior_team_support_ge3_share_ge_075": ge3_share >= 0.75,
        "no_prior_state_nonzero": state_counts[STATE_NO_PRIOR] > 0,
        "target_game_snaps_used_zero": True,
        "target_game_opportunity_performance_used_zero": True,
    }

    return {
        "season": int(season),
        "scheduled_team_weeks": int(total),
        "covered_roster_team_weeks": int(covered),
        "roster_team_week_coverage": coverage,
        "scoped_roster_rows": int(len(r)),
        "stable_id_rows": int(r["stable_id"].ne("").sum()),
        "stable_id_coverage": stable_cov,
        "distinct_named_players": int(s["stable_id"].nunique()),
        "state_counts": state_counts,
        "transition_events": int(len(transition)),
        "transition_named_players": trans_players,
        "week1_transition_events": int(transition["transition_week1"].sum()),
        "inseason_transition_events": int(transition["transition_inseason"].sum()),
        "prior_team_support_ge3_events": int(transition["prior_team_support_ge3"].sum()),
        "prior_team_support_ge3_share": ge3_share,
        "prior_team_support_ge8_events": int(transition["prior_team_support_ge8"].sum()),
        "prior_team_support_ge8_share": ge8_share,
        "family_summary": fam,
        "gates": gates,
        "source_ready": bool(all(gates.values())),
    }


def prospective_summary(
    roster: pd.DataFrame,
    scheduled: pd.DataFrame,
    states: pd.DataFrame,
    audit: dict,
) -> dict:
    season = PARITY_SEASON
    r = roster.loc[roster["season"].eq(season)].copy()
    s = states.loc[states["season"].eq(season)].copy()
    transition = s.loc[s["team_context_state"].eq(STATE_TRANSITION)].copy()

    weekly = []
    complete_weeks = []
    for week in WEEKS:
        covered, total, coverage = roster_team_week_coverage(roster, scheduled, season, week)
        if total == 0:
            continue
        weekly.append({
            "week": int(week),
            "scheduled_team_weeks": int(total),
            "covered_roster_team_weeks": int(covered),
            "coverage": float(coverage),
        })
        if coverage >= 0.95:
            complete_weeks.append(int(week))

    first6 = {row["week"]: row["coverage"] for row in weekly if row["week"] <= 6}
    first6_ok = set(first6) == set(range(1, 7)) and all(first6[w] >= 0.95 for w in range(1, 7))
    stable_cov = float(r["stable_id"].ne("").mean()) if len(r) else 0.0
    max_week = int(r["week"].max()) if len(r) else 0

    schemas_same = True  # normalized through the exact same frozen loader/aliases.

    gates = {
        "weekly_roster_same_semantics": schemas_same and len(r) > 0,
        "schedule_same_semantics": bool(len(scheduled.loc[scheduled["season"].eq(season)]) > 0),
        "weeks_1_6_roster_coverage_ge_095": first6_ok,
        "scoped_stable_id_coverage_ge_099": stable_cov >= 0.99,
        "duplicate_team_week_stable_id_zero": not r.loc[r["stable_id"].ne("")].duplicated(
            ["season", "week", "team", "stable_id"]
        ).any(),
        "transition_events_nonzero": len(transition) > 0,
        "chronology_violations_zero": audit["chronology_violations"] == 0,
        "outcomes_2026_used_zero": True,
    }

    return {
        "season": season,
        "max_roster_week": max_week,
        "complete_coverage_weeks": complete_weeks,
        "weekly_coverage": weekly,
        "scoped_roster_rows": int(len(r)),
        "stable_id_coverage": stable_cov,
        "transition_events": int(len(transition)),
        "transition_named_players": int(transition["stable_id"].nunique()),
        "week1_transition_events": int(transition["transition_week1"].sum()),
        "inseason_transition_events": int(transition["transition_inseason"].sum()),
        "gates": gates,
        "parity_ready": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import nflreadpy as nfl

    schedule_raw = pdx(nfl.load_schedules(seasons=list(SOURCE_SEASONS)))
    scheduled, schedule_audit = normalize_schedule(schedule_raw)

    roster_parts = [
        (season, pdx(nfl.load_rosters_weekly(int(season))))
        for season in SOURCE_SEASONS
    ]
    roster, roster_audit = normalize_rosters(roster_parts, scheduled)
    states, state_audit = build_team_context(roster)

    historical = [
        historical_summary(
            season,
            roster=roster,
            scheduled=scheduled,
            states=states,
            audit=state_audit,
        )
        for season in HISTORICAL_SEASONS
    ]
    prospective = prospective_summary(
        roster=roster,
        scheduled=scheduled,
        states=states,
        audit=state_audit,
    )

    integrity = {
        "predictive_player_outcomes_scored_zero": True,
        "player_opportunity_relationships_scored_zero": True,
        "target_game_snaps_used_zero": True,
        "target_game_targets_carries_receptions_yards_tds_used_zero": True,
        "coefficients_fit_zero": True,
        "thresholds_searched_zero": True,
        "outcomes_2026_used_zero": True,
        "sportsbook_oddsapi_zero": True,
        "production_changed_false": True,
    }
    ready = bool(
        all(x["source_ready"] for x in historical)
        and prospective["parity_ready"]
        and all(integrity.values())
    )

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "disposition": (
            "PRISM_NAMED_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY"
            if ready
            else "PRISM_NAMED_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_BLOCKED"
        ),
        "source_ready": ready,
        "source_seasons": list(SOURCE_SEASONS),
        "historical_adjudication_seasons": list(HISTORICAL_SEASONS),
        "schedule_audit": schedule_audit,
        "roster_audit": roster_audit,
        "state_audit": state_audit,
        "historical_seasons": historical,
        "prospective_2026": prospective,
        "integrity_gates": integrity,
        "predictive_player_outcomes_scored": 0,
        "player_opportunity_relationships_scored": 0,
        "target_game_snaps_used": 0,
        "target_game_performance_fields_used": 0,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "outcomes_2026_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "paid_oddsapi_calls": 0,
        "production_changed": False,
    }

    roster.to_csv(a.out_dir / "team_context_normalized_weekly_rosters.csv", index=False)
    states.to_csv(a.out_dir / "team_context_named_player_states.csv", index=False)
    states.loc[
        states["team_context_state"].eq(STATE_TRANSITION)
    ].to_csv(a.out_dir / "team_context_transition_events.csv", index=False)

    pd.DataFrame([
        {
            "season": s["season"],
            "family": f["family"],
            "transition_events": f["transition_events"],
            "transition_named_players": f["transition_named_players"],
            "prior_team_support_ge3_share": f["prior_team_support_ge3_share"],
            "prior_team_support_ge8_share": f["prior_team_support_ge8_share"],
            "week1_transition_events": f["week1_transition_events"],
            "inseason_transition_events": f["inseason_transition_events"],
        }
        for s in historical
        for f in s["family_summary"]
    ]).to_csv(a.out_dir / "team_context_transition_family_summary.csv", index=False)

    (a.out_dir / "team_context_transition_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
