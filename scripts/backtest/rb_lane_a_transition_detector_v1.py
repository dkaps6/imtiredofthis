"""RB Lane A -- transition detector (identities/features only, no outcomes).

Implements "Transition definition" from
``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V1_PLAN.md`` (Amendment 3):
splits the broad ``detected_transition`` disclosure population from the narrow
``scored_v1_transition`` population (loss/vacancy events only).

STRICT SCOPE: this module identifies WHICH team-weeks are transitions and WHY.
It computes NO candidate rush-yard output, touches NO actual/outcome columns,
and is not itself a candidate mechanism. Per GPT-5.6's required execution
order (Issue #535 comment `5702132088`), this population must exist and pass
Gate-0.3 requirements 5/6 (``gate03_event_report`` in ``rb_lane_a_gate0_v1.py``)
BEFORE any candidate rushing-yard output is computed or inspected.

Inputs are Gate 0's own harmonized frames (roster_state from Gate 0.3, weeks
1-18 only; injury_state from Gate 0.2) -- no new source is introduced.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

UNAVAILABLE_STATUSES = {"OUT", "DOUBTFUL", "IR", "PUP"}


def _name_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def _restrict_reg_season(roster_state: pd.DataFrame) -> pd.DataFrame:
    return roster_state.loc[roster_state["week"].between(1, 18)].copy()


def build_player_week_status(roster_state: pd.DataFrame, injury_state: pd.DataFrame) -> pd.DataFrame:
    """RB-room player-week rows with an `unavailable` flag joined from injury_state.

    Only players already in the Gate-0.3 RB-room roster universe are scored --
    injury_state itself carries no position, so this join is what restricts it
    to RB/FB/HB players. A player with no injury-report row for a given week is
    implicitly available (no OUT/DOUBTFUL/IR/PUP designation), matching the
    frozen plan's own default.
    """
    roster = _restrict_reg_season(roster_state).copy()
    if roster.empty:
        return roster.assign(unavailable=pd.Series(dtype=bool))

    inj = injury_state.copy()
    if inj.empty:
        roster["unavailable"] = False
        return roster

    inj = inj.copy()
    inj["name_key"] = _name_key(inj["player"])
    inj["_unavailable"] = inj["status"].fillna("").astype(str).str.upper().isin(UNAVAILABLE_STATUSES)
    inj_flag = (
        inj.groupby(["season", "week", "team", "name_key"], as_index=False)["_unavailable"]
        .max()
        .rename(columns={"_unavailable": "unavailable"})
    )

    merged = roster.merge(
        inj_flag, on=["season", "week", "team", "name_key"], how="left", validate="many_to_one"
    )
    merged["unavailable"] = merged["unavailable"].fillna(False)
    return merged


def build_detected_transitions(roster_state: pd.DataFrame, injury_state: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team, week) with detected/scored transition flags.

    `prior_season`/`prior_week` identify the immediately preceding resolvable
    state actually used (never assumed to be `week - 1`), per Gate-0.3
    requirement 5's "current and immediately previous resolvable state"
    language.
    """
    status = build_player_week_status(roster_state, injury_state)
    if status.empty:
        return pd.DataFrame(
            columns=[
                "season", "week", "team", "prior_season", "prior_week",
                "detected_transition", "scored_v1_transition",
                "trigger_status_onset_loss", "trigger_status_onset_return",
                "trigger_membership_shrink", "trigger_membership_gain",
            ]
        )

    rows = []
    for (season, team), g in status.groupby(["season", "team"], sort=False):
        weeks = sorted(g["week"].unique())
        for i in range(1, len(weeks)):
            week, prior_week = weeks[i], weeks[i - 1]
            cur = g.loc[g.week == week]
            prev = g.loc[g.week == prior_week]

            cur_room = set(cur["player_key"])
            prev_room = set(prev["player_key"])
            departed = prev_room - cur_room
            added = cur_room - prev_room

            cur_avail = {r.player_key: not r.unavailable for r in cur.itertuples()}
            prev_avail = {r.player_key: not r.unavailable for r in prev.itertuples()}
            common = cur_room & prev_room
            onset_loss = {p for p in common if prev_avail.get(p) and not cur_avail.get(p)}
            onset_return = {p for p in common if not prev_avail.get(p) and cur_avail.get(p)}

            trigger_membership_shrink = len(departed) > 0
            trigger_membership_gain = len(added) > 0
            trigger_status_onset_loss = len(onset_loss) > 0
            trigger_status_onset_return = len(onset_return) > 0

            detected = (
                trigger_membership_shrink
                or trigger_membership_gain
                or trigger_status_onset_loss
                or trigger_status_onset_return
            )
            # Scored V1 (Amendment 3): loss/vacancy events only -- the "return"/
            # "addition" directions are detected but never scored.
            scored = trigger_status_onset_loss or trigger_membership_shrink

            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "team": team,
                    "prior_season": int(season),
                    "prior_week": int(prior_week),
                    "detected_transition": detected,
                    "scored_v1_transition": scored,
                    "trigger_status_onset_loss": trigger_status_onset_loss,
                    "trigger_status_onset_return": trigger_status_onset_return,
                    "trigger_membership_shrink": trigger_membership_shrink,
                    "trigger_membership_gain": trigger_membership_gain,
                }
            )

    return pd.DataFrame(rows)


def build_scored_v1_event_population(detected_transitions: pd.DataFrame) -> pd.DataFrame:
    """Narrow to the scored V1 (loss/vacancy) rows only.

    This is the exact `event_population` shape `gate03_event_report()` expects:
    season, week, team, prior_season, prior_week -- and deliberately NO outcome
    column, so the requirement-6 structural check has nothing to trip on.
    """
    scored = detected_transitions.loc[detected_transitions["scored_v1_transition"]].copy()
    return scored[["season", "week", "team", "prior_season", "prior_week"]].reset_index(drop=True)


def disclosure_report(detected_transitions: pd.DataFrame, seasons: Iterable[int]) -> dict:
    """Per-season rates for the disclosure population vs. the scored V1 population."""
    out = {}
    for season in sorted({int(s) for s in seasons}):
        s = detected_transitions.loc[detected_transitions.season == season]
        n = int(len(s))
        out[str(season)] = {
            "team_weeks_evaluated": n,
            "detected_transition_count": int(s["detected_transition"].sum()) if n else 0,
            "scored_v1_transition_count": int(s["scored_v1_transition"].sum()) if n else 0,
            "trigger_status_onset_loss_count": int(s["trigger_status_onset_loss"].sum()) if n else 0,
            "trigger_membership_shrink_count": int(s["trigger_membership_shrink"].sum()) if n else 0,
            "trigger_status_onset_return_count_disclosure_only": int(s["trigger_status_onset_return"].sum()) if n else 0,
            "trigger_membership_gain_count_disclosure_only": int(s["trigger_membership_gain"].sum()) if n else 0,
        }
    return out
