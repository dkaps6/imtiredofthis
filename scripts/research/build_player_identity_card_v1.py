#!/usr/bin/env python3
"""Player Identity Card V1 — strictly-prior per-player, per-week football state.

Implements the descriptive substrate specified in
`docs/research/PLAYER_IDENTITY_ROLE_STATE_ARCHITECTURE_V1.md`:

    For every player, maintain a strictly-prior historical identity state.

This is **state construction, not a predictive experiment**. It fits nothing,
promotes nothing, and never reads the target week's outcomes to build that
week's card. Predictive use is a separate, preregistered question gated by the
anti-retest rules agreed in control-room comment 6420198257.

Why this exists
---------------
The production board carries 79 columns and almost no football context: no
target share, snap share, route participation, red-zone usage, teammate
competition or situational role. The per-player evidence the pipeline does
build (`player_form`, entitlement traces) is run-artifact only and expires. So
a named player's identity has had nowhere to accumulate across a season.

Everything here is rebuilt from published nflverse sources, so the whole season
is recoverable and no expired artifact is required.

The central distinction
-----------------------
Opportunity is not one number. Fifteen carries between the twenties is not ten
carries with three inside the five. Six targets is not six targets if one
player draws schemed red-zone looks and another draws underneath checkdowns. So
quantity and **quality/location/purpose** are tracked separately, per the
architecture's "Opportunity Quality / Topology" section.

Corrections applied after audit 6420504608
------------------------------------------
That audit blocked V1 for five player-first correctness defects. All are fixed
here, and the fixes share one spine: **weekly rosters are the authority**, not
play-by-play inference.

(A) Snap counts are now genuinely consumed. They define the participation
    denominator rather than merely stamping provenance.

(B) Active zero-opportunity weeks are preserved as real zeros. A player who
    took offensive snaps and got no carry or target no longer has that week
    silently dropped — which previously biased level, trajectory and stability
    upward. Weeks with no participation are *excluded* from the denominator
    rather than coerced to zero, so a DNP never masquerades as a quiet game.

(C) Target-week team comes from the weekly roster authority for that week, not
    from the mode of historical play-by-play. Historical teams are reported
    separately so identity and current state are never conflated.

(D) The universe is every player on a target-week roster, not only players
    with prior opportunity. Rookies and sparse-role players get a card with an
    explicit `history_state` instead of being dropped.

(E) The snap/PBP identity bridge is `gsis_id` <-> `pfr_id` from weekly
    rosters. No name joining. Bridge coverage is reported, and the run fails
    closed if the bridge cannot be established.

Strict-prior contract
---------------------
A card for (player, season, week=w) is built **only** from weeks < w of that
season, optionally extended by prior seasons. Week w never contributes to its
own card, which makes the output leakage-safe for evaluating any week-w
hypothesis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _nflv():
    """nflverse loaders, imported on first use.

    Deliberately lazy: data acquisition needs nflreadpy, the state arithmetic
    below does not. Keeping the import at the boundary lets the card logic be
    unit-tested on fixtures without reaching for the network library.
    """
    try:
        import nflreadpy
    except ImportError:  # pragma: no cover - environment guard
        print("FATAL: nflreadpy is required to load nflverse data",
              file=sys.stderr)
        raise
    return nflreadpy


# High-value opportunity zones, by distance to the opponent goal line.
# Situational columns the opportunity-quality topology is derived from. Absent
# any of these the card cannot distinguish "15 carries between the 20s" from
# "10 carries with 3 inside the 5", which is the distinction it exists to carry.
_QUALITY_COLS = ("yardline_100", "down", "goal_to_go", "half_seconds_remaining")

# Positions that can plausibly be allocated a carry or a target. Cards are
# still emitted for everyone on the roster -- fix (D) forbids narrowing the
# universe -- but a punter's "no rushing history" is not a finding, and
# without this flag it is 69% of the rows.
_OPPORTUNITY_POSITIONS = frozenset({"QB", "RB", "FB", "HB", "WR", "TE"})

# PLAYER-STATE ANOMALY INVESTIGATION GATE (AGENTS.md).
#
# A week's observation may not enter trajectory or stability logic until it has
# been state-labelled, and an unresolved cause must fail closed rather than
# become affirmative evidence. Fix (B) was right that an active zero-touch week
# is a real zero and must not be dropped -- but an active zero is not ONE
# thing. A healthy back schemed out is signal; a back who took eight snaps and
# left with a hamstring is contamination. Both were being written into the same
# trajectory and stability fields.
#
# What the loaded sources can support today:
#   DNP_INACTIVE       - no offensive participation that week
#   HEALTHY_FULL       - snaps consistent with the player's own prior baseline
#   UNRESOLVED_CONTEXT - materially abnormal snaps, or a return from absence,
#                        with no loaded source that explains the cause
#   NO_BASELINE        - too little prior participation to judge abnormality
#
# INJURY_LIMITED, PARTIAL_GAME and RETURN_FROM_INJURY require an injury /
# participation-detail source that this builder does not yet load. Until it
# does, those cases land in UNRESOLVED_CONTEXT, which is precisely the
# fail-closed behaviour the gate prescribes -- not a silent HEALTHY_FULL.
#
# A SCREENING threshold, not a fitted parameter: a week at or below this
# fraction of the player's own prior median offensive snaps is treated as
# materially abnormal and sent for explanation.
PARTIAL_PARTICIPATION_RATIO = 0.5
# Minimum prior participation weeks before abnormality can be judged at all.
MIN_BASELINE_WEEKS = 2

STATE_DNP = "DNP_INACTIVE"
STATE_HEALTHY = "HEALTHY_FULL"
STATE_UNRESOLVED = "UNRESOLVED_CONTEXT"
STATE_NO_BASELINE = "NO_BASELINE"
STATE_INJURY_LIMITED = "INJURY_LIMITED"
STATE_RETURN_FROM_INJURY = "RETURN_FROM_INJURY"

# PARTIAL_GAME is deliberately absent. Establishing that a player left a game
# early needs per-play participation, and nflverse load_participation covers
# 2016-2025 only -- it raises "Season must be between 2016 and 2025" for 2026.
# Play-by-play is not a substitute: it names only players who touched the ball,
# so a sparse-usage player's last appearance says nothing about when he left.
# For 2026 this state is SOURCE BLOCKED; such weeks resolve to INJURY_LIMITED
# when a designation explains them and stay UNRESOLVED_CONTEXT otherwise.
PARTIAL_GAME_SOURCE_STATE = "SOURCE_BLOCKED_NO_2026_PARTICIPATION_DATA"

# An injury designation explains an abnormal week. It does not make that week
# representative evidence of the player's healthy role, so an explained week is
# still withheld from trajectory/efficiency -- withheld WITH a reason rather
# than as an unknown. Only these two dispositions exist.
DISPOSITION_USABLE = "USABLE"
DISPOSITION_WITHHELD = "WITHHELD"

# report_status / practice_status values that explain a limited or missed week.
_INJURY_REPORT_LIMITING = frozenset({"out", "doubtful", "questionable"})
_INJURY_PRACTICE_LIMITING = frozenset({
    "did not participate in practice", "limited participation in practice"})
_INJURY_ABSENT_CAUSE = frozenset({"out", "doubtful"})

RED_ZONE = 20
INSIDE_10 = 10
INSIDE_5 = 5
TWO_MINUTE_SECONDS = 120


def _to_pandas(df):
    try:
        return df.to_pandas()
    except AttributeError:
        return df


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _first_col(df: pd.DataFrame, candidates) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _num_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Numeric column as a Series aligned to df.index, or a default-filled one.

    `df.get(col, 0)` returns the bare scalar when the column is missing, which
    then has no Series methods. Always returning an aligned Series keeps the
    callers safe against upstream schema drift.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def load_weekly_rosters(seasons: list[int]) -> pd.DataFrame:
    """Authority for target-week team, player universe, and the id bridge.

    Fixes (C), (D) and (E) together: weekly rosters carry both `gsis_id` and
    `pfr_id`, so they are simultaneously the team-of-record per week, the list
    of players who should get a card, and the bridge between play-by-play
    (gsis) and snap counts (pfr).
    """
    nflv = _nflv()
    last = None
    for fn in ("load_rosters_weekly", "load_weekly_rosters", "load_rosters"):
        if not hasattr(nflv, fn):
            continue
        try:
            df = _lower(_to_pandas(getattr(nflv, fn)(seasons=seasons)))
            if "week" not in df.columns and fn == "load_rosters":
                # Season-level rosters cannot anchor a target week.
                last = f"{fn} has no week column"
                continue
            print(f"[identity] roster authority: {fn}, rows={len(df)}")
            return df
        except Exception as exc:
            last = f"{fn}: {type(exc).__name__}: {exc}"
    raise RuntimeError(
        "no weekly roster authority available; cannot establish target-week "
        f"team, player universe, or the gsis<->pfr identity bridge ({last})"
    )


def load_snaps(seasons: list[int]) -> pd.DataFrame:
    try:
        df = _lower(_to_pandas(_nflv().load_snap_counts(seasons=seasons)))
        print(f"[identity] snap counts rows={len(df)}")
        return df
    except Exception as exc:
        print(f"[identity] WARNING snap counts unavailable: "
              f"{type(exc).__name__}: {exc}")
        return pd.DataFrame()


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    """Weekly injury report. Optional enrichment: fails soft but visibly.

    Keyed on gsis_id directly, so no identity bridge is involved. A week-w
    report is published before week w is played, so using it to explain week w
    introduces no look-ahead, and cards only ever consume weeks < target.
    """
    try:
        df = _lower(_to_pandas(_nflv().load_injuries(seasons=seasons)))
        print(f"[identity] injury report rows={len(df)}")
        return df
    except Exception as exc:
        print(f"[identity] WARNING injuries unavailable: "
              f"{type(exc).__name__}: {exc}")
        return pd.DataFrame()


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    return _lower(_to_pandas(_nflv().load_pbp(seasons=seasons)))


def build_participation(
    rosters: pd.DataFrame, snaps: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Per (season, week, gsis_id) offensive participation.

    This is fix (A): snaps stop being decorative and become the denominator.
    A player-week appears here only if he actually took offensive snaps, which
    is what lets (B) distinguish "active, no touches" from "did not play".
    """
    diag = {"snap_rows": int(len(snaps))}
    if snaps.empty:
        diag["bridge_state"] = "unavailable_no_snaps"
        return pd.DataFrame(), diag

    r_gsis = _first_col(rosters, ["gsis_id", "player_id", "gsis"])
    r_pfr = _first_col(rosters, ["pfr_id", "pfr_player_id"])
    s_pfr = _first_col(snaps, ["pfr_player_id", "pfr_id"])
    s_off = _first_col(snaps, ["offense_snaps", "offense_snap", "off_snaps"])
    # Needed to tell a bye week from a genuine absence: without it every
    # player on a team is flagged the week after that team is idle.
    s_team = _first_col(snaps, ["team", "recent_team", "club_code",
                                "team_abbr", "posteam"])
    if not (r_gsis and r_pfr and s_pfr and s_off):
        diag["bridge_state"] = "unavailable_missing_columns"
        diag["bridge_columns"] = {
            "roster_gsis": r_gsis, "roster_pfr": r_pfr,
            "snap_pfr": s_pfr, "snap_offense": s_off,
        }
        return pd.DataFrame(), diag

    raw = (rosters[["season", "week", r_gsis, r_pfr]]
           .rename(columns={r_gsis: "gsis_id", r_pfr: "pfr_id"}))
    # Why a bridge miss happens matters as much as how often. A roster row can
    # be missing outright, or present with no pfr_id to join on -- the second
    # is a roster completeness problem, not a snap-counts problem, and only
    # this breakdown tells them apart.
    diag["roster_rows"] = int(len(raw))
    diag["roster_rows_with_gsis"] = int(raw["gsis_id"].notna().sum())
    diag["roster_rows_with_pfr"] = int(raw["pfr_id"].notna().sum())
    diag["roster_rows_bridgeable"] = int(
        (raw["gsis_id"].notna() & raw["pfr_id"].notna()).sum())

    # pfr_id <-> gsis_id is a PLAYER-level invariant, not a weekly fact: a
    # player's PFR id does not change by week. Keying the bridge on
    # (season, week, pfr_id) therefore discarded every week whose roster row
    # happens to omit pfr_id, which is 25% of them. Build the map at player
    # level from any row carrying both, and fail closed on ambiguity rather
    # than pick a winner.
    pairs = (raw.dropna(subset=["gsis_id", "pfr_id"])[["gsis_id", "pfr_id"]]
             .drop_duplicates())
    per_pfr = pairs.groupby("pfr_id")["gsis_id"].nunique()
    ambiguous = sorted(per_pfr.loc[per_pfr > 1].index.astype(str))
    diag["bridge_ambiguous_pfr_ids"] = len(ambiguous)
    if ambiguous:
        # One PFR id mapping to several gsis ids means the identity map itself
        # is unsound; guessing here would silently merge two players.
        diag["bridge_state"] = "unavailable_ambiguous_identity_map"
        diag["bridge_ambiguous_examples"] = ambiguous[:10]
        return pd.DataFrame(), diag
    bridge = pairs.drop_duplicates("pfr_id")
    diag["bridge_players_mapped"] = int(len(bridge))
    keep = ["season", "week", s_pfr, s_off] + ([s_team] if s_team else [])
    s = snaps[keep].rename(
        columns={s_pfr: "pfr_id", s_off: "offense_snaps",
                 **({s_team: "team"} if s_team else {})})
    diag["bye_detection"] = "available" if s_team else "unavailable_no_team_col"
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")

    merged = s.merge(bridge, on=["pfr_id"], how="left")
    matched = merged["gsis_id"].notna()
    diag["bridge_matched"] = int(matched.sum())
    diag["bridge_unmatched"] = int((~matched).sum())
    diag["bridge_match_rate"] = (float(matched.mean())
                                 if len(merged) else float("nan"))
    diag["bridge_state"] = "ok" if matched.any() else "unavailable_no_matches"

    merged["offense_snaps"] = pd.to_numeric(merged["offense_snaps"],
                                            errors="coerce").fillna(0)

    # A bridge miss is only harmful where it drops an OFFENSIVE player-week for
    # a player who bridges in some other week: his participation denominator is
    # then understated, which inflates every per-week rate built on it. That is
    # the same upward bias fix (B) exists to remove, re-entering by another
    # door, so measure it rather than trust the overall match rate.
    off = merged.loc[merged["offense_snaps"] > 0]
    off_unmatched = off.loc[off["gsis_id"].isna()]
    bridged_ids = set(off.loc[off["gsis_id"].notna(), "pfr_id"].dropna())
    partial = off_unmatched.loc[off_unmatched["pfr_id"].isin(bridged_ids)]
    diag["offensive_player_weeks_total"] = int(len(off))
    diag["offensive_player_weeks_unmatched"] = int(len(off_unmatched))
    # Never bridges at all -> no card denominator exists, so the card fails
    # closed to NaN. Harmless to rates, but it means the player is invisible.
    never = off_unmatched.loc[~off_unmatched["pfr_id"].isin(bridged_ids)]
    diag["unbridged_only_players"] = int(never["pfr_id"].nunique())
    # Of those, how many are known to the roster at all? A pfr_id the roster
    # never lists is a different failure from one it lists without a gsis_id.
    roster_pfr_ever = set(raw.loc[raw["pfr_id"].notna(), "pfr_id"])
    never_ids = set(never["pfr_id"].dropna())
    diag["unbridged_absent_from_roster"] = int(
        len(never_ids - roster_pfr_ever))
    diag["unbridged_on_roster_without_gsis"] = int(
        len(never_ids & roster_pfr_ever))
    # Bridges sometimes -> denominator understated, rates inflated. This is the
    # number that matters, and it must stay at or near zero.
    diag["partial_bridge_player_weeks"] = int(len(partial))
    diag["partial_bridge_players"] = int(partial["pfr_id"].nunique())
    # Resolve the affected players to gsis so the card layer can fail their
    # rates closed, the same way it already does for a player with no
    # participation at all. An inflated rate is worse than a missing one.
    partial_gsis = sorted(
        off.loc[off["pfr_id"].isin(set(partial["pfr_id"].dropna()))
                & off["gsis_id"].notna(), "gsis_id"].unique().tolist())
    diag["partial_bridge_gsis_ids"] = partial_gsis

    agg = {"offense_snaps": "max"}
    if "team" in off.columns:
        agg["team"] = "first"
    part = (off.loc[off["gsis_id"].notna()]
            .groupby(["season", "week", "gsis_id"], as_index=False)
            .agg(agg))
    diag["participation_player_weeks"] = int(len(part))
    return part, diag


def _injury_flags(injuries: pd.DataFrame) -> tuple[dict, dict]:
    """(season, week, gsis_id) -> limiting designation, and -> absence cause."""
    if injuries.empty:
        return {}, {}
    need = {"season", "week", "gsis_id"}
    if not need <= set(injuries.columns):
        return {}, {}
    inj = injuries.copy()
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    rep = (inj["report_status"].astype(str).str.strip().str.lower()
           if "report_status" in inj.columns else pd.Series("", index=inj.index))
    prac = (inj["practice_status"].astype(str).str.strip().str.lower()
            if "practice_status" in inj.columns
            else pd.Series("", index=inj.index))

    limiting, absent = {}, {}
    keys = list(zip(inj["season"], inj["week"], inj["gsis_id"]))
    for k, r, pr in zip(keys, rep, prac):
        if any(pd.isna(x) for x in k[:2]) or k[2] is None:
            continue
        kk = (int(k[0]), int(k[1]), k[2])
        if r in _INJURY_REPORT_LIMITING or pr in _INJURY_PRACTICE_LIMITING:
            # Keep the strongest designation seen for the week.
            limiting[kk] = r if r in _INJURY_REPORT_LIMITING else pr
        if r in _INJURY_ABSENT_CAUSE or pr == "did not participate in practice":
            absent[kk] = r if r in _INJURY_ABSENT_CAUSE else pr
    return limiting, absent


def label_player_week_states(participation: pd.DataFrame,
                             injuries: pd.DataFrame | None = None,
                             partial_ratio: float | None = None,
                             ) -> pd.DataFrame:
    """State-label every participation week before it can become a trend.

    Required by the PLAYER-STATE ANOMALY INVESTIGATION GATE. Each week is
    judged only against the player's own EARLIER weeks, so a given week's
    label is stable no matter which target week's card is being built.

    Returns the participation frame plus: baseline_snaps, snap_ratio,
    returned_from_absence, player_state.
    """
    if participation.empty:
        return participation.assign(
            baseline_snaps=pd.Series(dtype=float),
            snap_ratio=pd.Series(dtype=float),
            returned_from_absence=pd.Series(dtype=bool),
            absence_bye_explained=pd.Series(dtype=bool),
            player_state=pd.Series(dtype=object),
            injury_source_state=pd.Series(dtype=object),
            injury_designation=pd.Series(dtype=object),
            partial_game_source_state=pd.Series(dtype=object),
            gate_disposition=pd.Series(dtype=object))

    pw = participation.sort_values(["gsis_id", "season", "week"]).copy()
    pw["offense_snaps"] = pd.to_numeric(pw["offense_snaps"], errors="coerce")

    # group_keys=False keeps apply from prepending the group key to the index,
    # which is pandas 1.5 default behaviour but a FutureWarning: being explicit
    # makes this survive the 2.x change rather than silently misalign.
    g = pw.groupby("gsis_id", sort=False, group_keys=False)["offense_snaps"]
    # Expanding median of EARLIER weeks only: shift(1) keeps the week being
    # judged out of its own baseline.
    pw["baseline_snaps"] = g.apply(lambda x: x.shift(1).expanding().median())
    pw["baseline_weeks"] = g.apply(lambda x: x.shift(1).expanding().count())
    pw["snap_ratio"] = np.where(
        pw["baseline_snaps"] > 0,
        pw["offense_snaps"] / pw["baseline_snaps"], np.nan)

    # A gap in a player's own participation sequence: he played, then did not,
    # then played again. The gate treats the return as needing explanation --
    # but ONLY if his team actually played in the intervening weeks. A bye
    # week is a gap for every player on the roster and explains itself; left
    # uncorrected it flags whole teams and makes the gate look like it is
    # firing when it is mostly reading the schedule.
    prev_week = pw.groupby("gsis_id", sort=False)["week"].shift(1)
    same_season = pw.groupby("gsis_id", sort=False)["season"].shift(1).eq(
        pw["season"])
    gap = same_season & prev_week.notna() & (pw["week"] - prev_week > 1)

    if "team" in pw.columns and pw["team"].notna().any():
        # A (season, week, team) present in participation at all means that
        # team took offensive snaps that week, i.e. it played.
        played = set(map(tuple, pw.loc[pw["team"].notna(),
                                       ["season", "week", "team"]]
                         .dropna().to_numpy().tolist()))

        def _team_played_during_gap(row) -> bool:
            lo, hi = int(row["_prev_week"]) + 1, int(row["week"])
            return any((row["season"], w, row["team"]) in played
                       for w in range(lo, hi))

        pw["_prev_week"] = prev_week
        idx = pw.index[gap.fillna(False)]
        real = pd.Series(False, index=pw.index)
        if len(idx):
            real.loc[idx] = pw.loc[idx].apply(_team_played_during_gap, axis=1)
        pw["returned_from_absence"] = real
        pw["absence_bye_explained"] = gap.fillna(False) & ~real
        pw = pw.drop(columns=["_prev_week"])
    else:
        # No team column: cannot distinguish a bye from an absence, so keep
        # the stricter reading rather than silently calling the week clean.
        pw["returned_from_absence"] = gap.fillna(False)
        pw["absence_bye_explained"] = False

    enough = pw["baseline_weeks"].fillna(0) >= MIN_BASELINE_WEEKS
    ratio = (PARTIAL_PARTICIPATION_RATIO if partial_ratio is None
             else float(partial_ratio))
    abnormal = enough & pw["snap_ratio"].le(ratio)

    pw["player_state"] = np.where(
        ~enough, STATE_NO_BASELINE,
        np.where(abnormal | pw["returned_from_absence"],
                 STATE_UNRESOLVED, STATE_HEALTHY))

    # ---- source-qualified resolution of the unexplained weeks -------------
    inj = pd.DataFrame() if injuries is None else injuries
    limiting, absent_cause = _injury_flags(inj)
    pw["injury_source_state"] = ("available" if limiting or absent_cause
                                 else "unavailable")
    pw["injury_designation"] = ""
    pw["partial_game_source_state"] = PARTIAL_GAME_SOURCE_STATE
    pw["partial_participation_ratio"] = ratio

    if limiting or absent_cause:
        prev_w = pw.groupby("gsis_id", sort=False, group_keys=False)["week"] \
            .shift(1)
        states, desig = [], []
        for i, (sea, wk, pid, st, ret) in enumerate(zip(
                pw["season"], pw["week"], pw["gsis_id"], pw["player_state"],
                pw["returned_from_absence"])):
            if st != STATE_UNRESOLVED:
                states.append(st)
                desig.append("")
                continue
            key = (int(sea), int(wk), pid)
            if bool(ret):
                # Was the gap itself injury-explained? Look only at the weeks
                # between his last appearance and this one -- all earlier than
                # this week, so no look-ahead.
                pv = prev_w.iloc[i]
                cause = ""
                if pd.notna(pv):
                    for w in range(int(pv) + 1, int(wk)):
                        c = absent_cause.get((int(sea), w, pid))
                        if c:
                            cause = c
                            break
                if cause:
                    states.append(STATE_RETURN_FROM_INJURY)
                    desig.append(cause)
                    continue
            d = limiting.get(key)
            if d:
                states.append(STATE_INJURY_LIMITED)
                desig.append(d)
            else:
                states.append(STATE_UNRESOLVED)
                desig.append("")
        pw["player_state"] = states
        pw["injury_designation"] = desig

    # An explained week is still not representative of the healthy player, so
    # explanation changes the REASON a week is withheld, never the fact.
    pw["gate_disposition"] = np.where(
        pw["player_state"].isin([STATE_HEALTHY, STATE_NO_BASELINE]),
        DISPOSITION_USABLE, DISPOSITION_WITHHELD)
    return pw.drop(columns=["baseline_weeks"])


def player_opportunity_rows(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, opportunity) with its situational flags.

    Rushing and receiving involvement are emitted separately because a single
    play gives a carry to one player and a target to another.
    """
    # Opportunity quality is the point of this artifact, and every quality flag
    # is derived from field position / situation. A missing column here would
    # read as "no red-zone work" rather than "unknown", so refuse instead.
    missing = [c for c in _QUALITY_COLS if c not in pbp.columns]
    if missing:
        raise RuntimeError(
            "play-by-play lacks the situational columns opportunity quality is "
            f"derived from: {missing}. Without them every quality flag would "
            "silently read as absent work rather than unknown work.")

    frames = []
    base = [c for c in ("season", "week", "posteam", "defteam", "yardline_100",
                        "down", "goal_to_go", "half_seconds_remaining",
                        "yards_gained") if c in pbp.columns]

    if {"rusher_player_id", "rush_attempt"} <= set(pbp.columns):
        r = pbp.loc[pd.to_numeric(pbp["rush_attempt"], errors="coerce").eq(1)]
        r = r[r["rusher_player_id"].notna()]
        o = r[base].copy()
        o["gsis_id"] = r["rusher_player_id"]
        o["pbp_name"] = (r["rusher_player_name"]
                         if "rusher_player_name" in r.columns else None)
        o["opportunity_kind"] = "carry"
        o["yards"] = _num_series(r, "yards_gained")
        o["complete"] = np.nan
        o["air_yards"] = np.nan
        frames.append(o)

    if {"receiver_player_id", "pass_attempt"} <= set(pbp.columns):
        p = pbp.loc[pd.to_numeric(pbp["pass_attempt"], errors="coerce").eq(1)]
        p = p[p["receiver_player_id"].notna()]
        o = p[base].copy()
        o["gsis_id"] = p["receiver_player_id"]
        o["pbp_name"] = (p["receiver_player_name"]
                         if "receiver_player_name" in p.columns else None)
        o["opportunity_kind"] = "target"
        o["yards"] = _num_series(p, "yards_gained")
        # Present-but-null means an incomplete pass (nflverse encodes 0/1), so
        # fill it. Column absent entirely means unknown, and must stay NaN:
        # a 0.0 here would feed catch_rate a confident "catches nothing".
        o["complete"] = (_num_series(p, "complete_pass").fillna(0.0)
                         if "complete_pass" in p.columns else np.nan)
        o["air_yards"] = _num_series(p, "air_yards")
        frames.append(o)

    if not frames:
        return pd.DataFrame()
    ev = pd.concat(frames, ignore_index=True, sort=False)
    ev["season"] = pd.to_numeric(ev["season"], errors="coerce")
    ev["week"] = pd.to_numeric(ev["week"], errors="coerce")

    yl = _num_series(ev, "yardline_100")
    down = _num_series(ev, "down")
    g2g = _num_series(ev, "goal_to_go", 0.0).fillna(0.0)
    half = _num_series(ev, "half_seconds_remaining")
    ev["is_red_zone"] = yl.le(RED_ZONE)
    ev["is_inside_10"] = yl.le(INSIDE_10)
    ev["is_inside_5"] = yl.le(INSIDE_5)
    ev["is_goal_to_go"] = g2g.eq(1)
    ev["is_third_down"] = down.eq(3)
    ev["is_two_minute"] = half.le(TWO_MINUTE_SECONDS)
    ev["is_early_down"] = down.isin([1, 2])
    return ev


def _week_stats(wk_counts: pd.Series, part_weeks: list) -> dict:
    """Level, stability and trajectory over an explicit set of weeks.

    Numerator and denominator are drawn from the SAME week set, which the
    previous inline version did not do: it summed every current-season
    opportunity over a participation-week count, so a player with opportunity
    in a week his participation record omitted produced a rate that could not
    be reconstructed from its own inputs.

    A week in `part_weeks` with no opportunity contributes a genuine zero --
    that is fix (B). A week not in `part_weeks` contributes nothing at all.
    """
    weeks = sorted({int(w) for w in part_weeks})
    n = len(weeks)
    if n == 0:
        return {"per_week": np.nan, "sd": np.nan, "recent": np.nan,
                "earlier": np.nan, "weeks": 0, "opp": 0.0}
    counts = (wk_counts.reindex(weeks).fillna(0).astype(float)
              if len(wk_counts) else pd.Series(0.0, index=weeks))
    total = float(counts.sum())
    recent_weeks = weeks[-2:]
    n_recent = float(counts.reindex(recent_weeks).fillna(0).sum())
    den_earlier = n - len(recent_weeks)
    return {
        "per_week": total / n,
        "sd": float(counts.std()) if n > 1 else np.nan,
        "recent": n_recent / len(recent_weeks),
        "earlier": ((total - n_recent) / den_earlier
                    if den_earlier > 0 else np.nan),
        "weeks": n,
        "opp": total,
    }


def _rate(num, den):
    den = float(den)
    return float(num) / den if den > 0 else np.nan


def build_cards(
    pbp: pd.DataFrame,
    rosters: pd.DataFrame,
    participation: pd.DataFrame,
    season: int,
    weeks: list[int],
    prior_seasons: list[int],
    untrusted_denominator_ids: set | None = None,
) -> tuple[pd.DataFrame, dict]:
    ev = player_opportunity_rows(pbp)
    if ev.empty:
        raise RuntimeError("no player-involvement rows derived from PBP")

    r_gsis = _first_col(rosters, ["gsis_id", "player_id", "gsis"])
    r_team = _first_col(rosters, ["team", "recent_team", "club_code"])
    r_name = _first_col(rosters, ["player_name", "full_name", "player_display_name"])
    r_pos = _first_col(rosters, ["position", "depth_chart_position"])
    if not (r_gsis and r_team):
        raise RuntimeError("roster authority lacks gsis/team columns")
    ros = rosters.copy()
    ros["season"] = pd.to_numeric(ros["season"], errors="coerce")
    ros["week"] = pd.to_numeric(ros["week"], errors="coerce")

    have_part = not participation.empty
    # Apply the anomaly gate HERE rather than trusting a caller to have done
    # it: a forgotten call would silently publish ungated trajectory fields,
    # which is the failure mode the gate exists to prevent.
    if have_part and "gate_disposition" not in participation.columns:
        participation = label_player_week_states(participation)
    # Players whose participation weeks are only partially bridged. Their
    # denominator is short, so any rate built on it is inflated; fail those
    # rates closed per player rather than publish a confident overstatement.
    untrusted = set(untrusted_denominator_ids or ())
    inj_state = (participation["injury_source_state"].iloc[0]
                 if have_part and "injury_source_state" in participation.columns
                 and len(participation) else "unavailable")
    kinds = ("carry", "target")
    rows = []

    for wk in weeks:
        # (D) universe = every player on a roster for the TARGET week.
        uni = ros.loc[ros["season"].eq(season) & ros["week"].eq(wk)].copy()
        uni = uni.dropna(subset=[r_gsis]).drop_duplicates(r_gsis)
        if uni.empty:
            print(f"[identity] WARNING no roster rows for {season} wk{wk}; skipping")
            continue

        # STRICT PRIOR history window.
        hist = ev.loc[(ev["season"].eq(season) & ev["week"].lt(wk))
                      | (ev["season"].isin(prior_seasons))]
        cur = hist.loc[hist["season"].eq(season)]

        # (B) denominator = participation weeks strictly before wk.
        if have_part:
            pw = participation.loc[participation["season"].eq(season)
                                   & participation["week"].lt(wk)]
            part_weeks = pw.groupby("gsis_id")["week"].nunique()
            # PLAYER-STATE ANOMALY GATE: weeks whose cause is unexplained may
            # not become trajectory/stability evidence. Only UNRESOLVED_CONTEXT
            # is withheld -- a healthy active week with zero touches stays in,
            # which is fix (B), and withholding it here would re-introduce the
            # very upward bias (B) removed.
            weeks_all = pw.groupby("gsis_id")["week"].apply(
                lambda x: sorted(set(x.astype(int)))).to_dict()
            # Withhold by DISPOSITION, not by the single unresolved label: an
            # injury-explained week is explained, not representative, so it
            # stays withheld. Keying off the label alone would have silently
            # promoted every week the resolver explained into usable evidence.
            clean = pw.loc[pw["gate_disposition"].eq(DISPOSITION_USABLE)]
            weeks_clean = clean.groupby("gsis_id")["week"].apply(
                lambda x: sorted(set(x.astype(int)))).to_dict()
            state_counts = (pw.groupby(["gsis_id", "player_state"]).size()
                            .unstack(fill_value=0))
            # The gate names EFFICIENCY as well as trajectory. A carry taken
            # in an unexplained week is a real carry, but its yardage is not
            # representative evidence of what this player is, so the same
            # weeks are withheld from the efficiency aggregates. Prior seasons
            # are labelled too, so this is not current-season-only.
            unres = participation.loc[
                participation["gate_disposition"].eq(DISPOSITION_WITHHELD),
                ["gsis_id", "season", "week"]]
            withheld_pw = set(map(tuple, unres.to_numpy().tolist()))
            w = pw.loc[pw["gate_disposition"].eq(DISPOSITION_WITHHELD)]
            wh_expl = (w.loc[w["player_state"].isin(
                [STATE_INJURY_LIMITED, STATE_RETURN_FROM_INJURY])]
                .groupby("gsis_id").size().to_dict())
            wh_unexpl = (w.loc[w["player_state"].eq(STATE_UNRESOLVED)]
                         .groupby("gsis_id").size().to_dict())
        else:
            part_weeks = pd.Series(dtype=int)
            weeks_all, weeks_clean = {}, {}
            state_counts = pd.DataFrame()
            withheld_pw = set()
            wh_expl, wh_unexpl = {}, {}

        # Week-grained team totals, so a share can be built over the SAME
        # accepted player-week set as its numerator. Gating only the numerator
        # while leaving the team denominator on all weeks would mechanically
        # depress every share -- a worse error than the one being fixed.
        team_wk = cur.groupby(["posteam", "opportunity_kind", "week"]).size()
        team_wk_rz = cur.loc[cur["is_red_zone"]].groupby(
            ["posteam", "opportunity_kind", "week"]).size()
        team_wk_i5 = cur.loc[cur["is_inside_5"]].groupby(
            ["posteam", "opportunity_kind", "week"]).size()

        team_tot = cur.groupby(["posteam", "opportunity_kind"]).size()
        team_rz = cur.loc[cur["is_red_zone"]].groupby(
            ["posteam", "opportunity_kind"]).size()
        team_i5 = cur.loc[cur["is_inside_5"]].groupby(
            ["posteam", "opportunity_kind"]).size()

        hist_by = dict(tuple(hist.groupby(["gsis_id", "opportunity_kind"])))
        cur_by = dict(tuple(cur.groupby(["gsis_id", "opportunity_kind"])))

        for _, pr in uni.iterrows():
            pid = pr[r_gsis]
            team_now = pr[r_team]  # (C) target-week team authority
            n_part = int(part_weeks.get(pid, 0)) if have_part else 0
            trusted = have_part and pid not in untrusted

            for kind in kinds:
                g = hist_by.get((pid, kind), ev.iloc[0:0])
                gc = cur_by.get((pid, kind), ev.iloc[0:0])
                n, n_cur = len(g), len(gc)

                if n == 0 and n_cur == 0 and n_part == 0:
                    # No history and no participation: still emit a card (D),
                    # but say so rather than implying a measured zero.
                    history_state = "no_history"
                elif n_cur > 0 and len(g) > n_cur:
                    history_state = "prior+current"
                elif n_cur > 0:
                    history_state = "current_only"
                elif n > 0:
                    history_state = "prior_only"
                else:
                    history_state = "participation_only_zero_opportunity"

                # (A)(B) rates divide by PARTICIPATION weeks, so an active week
                # with zero touches counts as a real zero. Without a
                # participation source these fail closed to NaN rather than
                # silently reverting to opportunity-observed weeks.
                # Gate-clean opportunity rows, for the EFFICIENCY aggregates.
                if n and withheld_pw:
                    keep = ~pd.Series(
                        list(zip(g["gsis_id"], g["season"], g["week"])),
                        index=g.index).isin(withheld_pw)
                    g_clean = g.loc[keep]
                else:
                    g_clean = g
                n_clean = len(g_clean)

                wk_counts = (gc.groupby("week").size()
                             if n_cur else pd.Series(dtype=float))
                obs = sorted(wk_counts.index.astype(int).tolist()) if n_cur else []
                if trusted and n_part > 0:
                    # Canonical fields are the GATE-COMPLIANT ones, computed on
                    # weeks whose state is explained. The all-weeks figures stay
                    # visible as diagnostics but have to be named to be used.
                    gated = _week_stats(wk_counts, weeks_clean.get(pid, []))
                    ungated = _week_stats(wk_counts, weeks_all.get(pid, []))
                else:
                    gated = ungated = _week_stats(pd.Series(dtype=float), [])
                per_week, sd = gated["per_week"], gated["sd"]
                per_week_recent, per_week_earlier = (gated["recent"],
                                                    gated["earlier"])

                tn = team_tot.get((team_now, kind), 0)
                trz = team_rz.get((team_now, kind), 0)
                ti5 = team_i5.get((team_now, kind), 0)

                # Gated shares: numerator and denominator over the identical
                # accepted week set. Empty set -> NaN, never a fabricated 0.
                acc = weeks_clean.get(pid, []) if trusted else []
                def _team_sum(idx, weeks=acc, t=team_now, k=kind):
                    return float(sum(idx.get((t, k, int(w)), 0) for w in weeks))
                g_tn = _team_sum(team_wk)
                g_trz = _team_sum(team_wk_rz)
                g_ti5 = _team_sum(team_wk_i5)
                if acc and n_cur:
                    acc_set = set(acc)
                    gca = gc.loc[gc["week"].astype(int).isin(acc_set)]
                else:
                    gca = gc.iloc[0:0]
                g_num = float(len(gca))
                g_num_rz = float(gca["is_red_zone"].sum()) if len(gca) else 0.0
                g_num_i5 = float(gca["is_inside_5"].sum()) if len(gca) else 0.0

                rows.append({
                    "season": season,
                    "target_week": wk,
                    "gsis_id": pid,
                    "player": (pr[r_name] if r_name and pd.notna(pr.get(r_name))
                               else (g["pbp_name"].dropna().iloc[0]
                                     if n and g["pbp_name"].notna().any() else "")),
                    "roster_position": pr[r_pos] if r_pos else "",
                    "can_receive_opportunity": bool(
                        r_pos and str(pr[r_pos]).strip().upper()
                        in _OPPORTUNITY_POSITIONS),
                    "team_target_week": team_now,
                    "teams_in_history": ",".join(
                        sorted({str(t) for t in g["posteam"].dropna().unique()})),
                    "opportunity_kind": kind,
                    "history_state": history_state,
                    # --- participation (A)(B) -----------------------------
                    "participation_weeks_prior": n_part,
                    "participation_source_state": (
                        "unavailable" if not have_part
                        else "available_untrusted_partial_bridge"
                        if pid in untrusted else "available"),
                    # --- opportunity QUANTITY -----------------------------
                    "opp_total_all_sources": n,
                    "opp_total_current_season": n_cur,
                    "opp_weeks_observed_current": len(obs),
                    "active_zero_opportunity_weeks": (
                        max(0, n_part - len(obs)) if trusted else np.nan),
                    # n_part < observed weeks is impossible on consistent
                    # inputs: it means the participation denominator is
                    # provably short, so flag it rather than let max(0, ...)
                    # hide it behind a clean-looking zero.
                    "participation_denominator_contradicted": (
                        bool(trusted and n_part < len(obs))),
                    "opp_per_participation_week": per_week,
                    # Gated (canonical) and ungated, plus the exact numerator
                    # and denominator used, so the share is reconstructible.
                    "team_share_current": _rate(g_num, g_tn),
                    "team_share_current_all_weeks": _rate(n_cur, tn),
                    "team_opp_in_gate_weeks": g_tn,
                    "player_opp_in_gate_weeks_current": g_num,
                    # --- trajectory / stability ---------------------------
                    "opp_per_week_recent2": per_week_recent,
                    "opp_per_week_earlier": per_week_earlier,
                    "opp_week_to_week_sd": sd,
                    # --- PLAYER-STATE ANOMALY GATE -------------------------
                    # The canonical trajectory/stability fields above are
                    # computed on explained weeks only. These make the gate's
                    # effect auditable: what it withheld, and what the figures
                    # would have been without it.
                    "gate_weeks_used": gated["weeks"],
                    "gate_weeks_withheld": max(
                        0, ungated["weeks"] - gated["weeks"]),
                    "gate_withheld_explained": int(
                        wh_expl.get(pid, 0)) if have_part else 0,
                    "gate_withheld_unexplained": int(
                        wh_unexpl.get(pid, 0)) if have_part else 0,
                    "injury_source_state": inj_state,
                    "partial_game_source_state": PARTIAL_GAME_SOURCE_STATE,
                    "gate_state_counts": ";".join(
                        f"{k}={int(v)}" for k, v in sorted(
                            state_counts.loc[pid].items()) if v
                    ) if (not state_counts.empty
                          and pid in state_counts.index) else "",
                    "opp_per_participation_week_all_weeks": ungated["per_week"],
                    "opp_week_to_week_sd_all_weeks": ungated["sd"],
                    "opp_per_week_recent2_all_weeks": ungated["recent"],
                    "opp_per_week_earlier_all_weeks": ungated["earlier"],
                    # Numerator actually used, so the rate is reconstructible
                    # from its own published inputs.
                    "opp_in_gate_weeks": gated["opp"],
                    # --- opportunity QUALITY / topology -------------------
                    "red_zone_opp": int(g["is_red_zone"].sum()) if n else 0,
                    "inside_10_opp": int(g["is_inside_10"].sum()) if n else 0,
                    "inside_5_opp": int(g["is_inside_5"].sum()) if n else 0,
                    "goal_to_go_opp": int(g["is_goal_to_go"].sum()) if n else 0,
                    "third_down_opp": int(g["is_third_down"].sum()) if n else 0,
                    "two_minute_opp": int(g["is_two_minute"].sum()) if n else 0,
                    "early_down_opp": int(g["is_early_down"].sum()) if n else 0,
                    "red_zone_rate_of_own_opp": _rate(
                        g["is_red_zone"].sum() if n else 0, n),
                    "inside_5_rate_of_own_opp": _rate(
                        g["is_inside_5"].sum() if n else 0, n),
                    "team_red_zone_share_current": _rate(g_num_rz, g_trz),
                    "team_inside_5_share_current": _rate(g_num_i5, g_ti5),
                    "team_red_zone_share_current_all_weeks": _rate(
                        gc["is_red_zone"].sum() if n_cur else 0, trz),
                    "team_inside_5_share_current_all_weeks": _rate(
                        gc["is_inside_5"].sum() if n_cur else 0, ti5),
                    "team_rz_opp_in_gate_weeks": g_trz,
                    "team_i5_opp_in_gate_weeks": g_ti5,
                    "player_rz_opp_in_gate_weeks": g_num_rz,
                    "player_i5_opp_in_gate_weeks": g_num_i5,
                    # --- efficiency identity ------------------------------
                    # Gate-compliant: opportunities taken in weeks whose state
                    # is unexplained are excluded, because their yardage is not
                    # representative evidence of what this player is. The
                    # ungated figures stay below under _all_weeks.
                    "yards_per_opp": (float(g_clean["yards"].mean())
                                      if n_clean and g_clean["yards"].notna().any()
                                      else np.nan),
                    "yards_per_opp_sd": (float(g_clean["yards"].std())
                                         if n_clean and g_clean["yards"].notna().sum() > 1
                                         else np.nan),
                    "catch_rate": (float(g_clean["complete"].mean())
                                   if kind == "target" and n_clean
                                   and g_clean["complete"].notna().any() else np.nan),
                    "mean_air_yards": (float(g_clean["air_yards"].mean())
                                       if kind == "target" and n_clean
                                       and g_clean["air_yards"].notna().any() else np.nan),
                    "efficiency_opp_used": n_clean,
                    "efficiency_opp_withheld": n - n_clean,
                    "yards_per_opp_all_weeks": (float(g["yards"].mean())
                                                if n and g["yards"].notna().any()
                                                else np.nan),
                    "catch_rate_all_weeks": (float(g["complete"].mean())
                                             if kind == "target" and n
                                             and g["complete"].notna().any() else np.nan),
                    # --- provenance ---------------------------------------
                    "strict_prior": True,
                    "history_seasons": ",".join(
                        str(int(s)) for s in sorted(g["season"].dropna().unique())
                    ) if n else "",
                    "sportsbook_inputs_used": 0,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no identity cards produced")
    diag = {
        "cards": int(len(out)),
        "players": int(out["gsis_id"].nunique()),
        "teams": int(out["team_target_week"].nunique()),
        "history_state_counts": {
            k: int(v) for k, v in out["history_state"].value_counts().items()},
        # Composition, so "27,928 no_history" is not read as a finding when
        # most of it is linemen and defensive backs with no rushing history.
        "opportunity_eligible_cards": int(out["can_receive_opportunity"].sum()),
        "history_state_counts_opportunity_eligible": {
            k: int(v) for k, v in
            out.loc[out["can_receive_opportunity"], "history_state"]
            .value_counts().items()},
        "untrusted_denominator_cards": int(
            (out["participation_source_state"]
             == "available_untrusted_partial_bridge").sum()),
        "denominator_contradicted_cards": int(
            out["participation_denominator_contradicted"].sum()),
        # PLAYER-STATE ANOMALY GATE effect, so it is never invisible.
        "gate_cards_with_withheld_weeks": int(
            (pd.to_numeric(out["gate_weeks_withheld"], errors="coerce")
             .fillna(0) > 0).sum()),
        "gate_weeks_withheld_total": int(
            pd.to_numeric(out["gate_weeks_withheld"], errors="coerce")
            .fillna(0).sum()),
        "player_week_state_counts": {
            k: int(v) for k, v in
            (participation["player_state"].value_counts().items()
             if have_part and "player_state" in participation.columns else [])},
        "anomaly_gate_version": (
            "PLAYER_STATE_ANOMALY_GATE_V2_INJURY_RESOLVED"
            if inj_state == "available"
            else "PLAYER_STATE_ANOMALY_GATE_V1_PARTICIPATION_ONLY"),
        "injury_source_state": inj_state,
        "partial_game_source_state": PARTIAL_GAME_SOURCE_STATE,
        # How much of the withholding is now EXPLAINED rather than mysterious.
        # The withheld set itself is unchanged by resolution: an explained week
        # is not a representative one.
        "withheld_explained_cards": int(
            (pd.to_numeric(out["gate_withheld_explained"], errors="coerce")
             .fillna(0) > 0).sum()),
        "withheld_unexplained_cards": int(
            (pd.to_numeric(out["gate_withheld_unexplained"], errors="coerce")
             .fillna(0) > 0).sum()),
    }
    out = out.sort_values(
        ["season", "target_week", "team_target_week", "opportunity_kind",
         "opp_total_current_season"],
        ascending=[True, True, True, True, False]).reset_index(drop=True)
    return out, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", required=True, help="e.g. 1-7 or 3,4,5")
    ap.add_argument("--prior-seasons", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--status-out", type=Path, required=True)
    ap.add_argument("--require-participation", action="store_true",
                    help="fail closed if the snap/participation bridge is "
                         "unavailable, instead of emitting NaN rate fields")
    a = ap.parse_args()

    if "-" in a.weeks:
        lo, hi = a.weeks.split("-", 1)
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in a.weeks.split(",") if w.strip()]
    priors = [int(s) for s in a.prior_seasons.split(",") if s.strip()]
    seasons = sorted({a.season, *priors})

    print(f"[identity] seasons={seasons} target_weeks={weeks}")
    rosters = load_weekly_rosters(seasons)
    snaps = load_snaps(seasons)
    participation, bridge = build_participation(rosters, snaps)
    print(f"[identity] bridge: {json.dumps(bridge, sort_keys=True)}")

    if a.require_participation and bridge.get("bridge_state") != "ok":
        raise RuntimeError(
            "participation bridge unavailable and --require-participation set; "
            f"refusing to emit a card whose denominators cannot be trusted: {bridge}"
        )

    # Source-qualified player-state resolution, per GPT audit 6425282770
    # priority 1. Optional enrichment: if it fails the gate simply stays
    # fail-closed with every abnormal week UNRESOLVED_CONTEXT.
    injuries = load_injuries(seasons)
    participation = label_player_week_states(participation, injuries)
    if not participation.empty:
        print("[identity] player-week states: "
              + json.dumps(participation["player_state"]
                           .value_counts().to_dict(), sort_keys=True))
        print("[identity] injury source: "
              f"{participation['injury_source_state'].iloc[0]}; "
              f"PARTIAL_GAME: {PARTIAL_GAME_SOURCE_STATE}")

    pbp = load_pbp(seasons)
    print(f"[identity] pbp rows={len(pbp)}")

    cards, diag = build_cards(
        pbp, rosters, participation, a.season, weeks, priors,
        untrusted_denominator_ids=set(bridge.get("partial_bridge_gsis_ids", [])))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cards.to_csv(a.out, index=False)

    status = {
        "version": "PLAYER_IDENTITY_CARD_V1",
        "scientific_boundary": "DESCRIPTIVE_STATE_ONLY_NO_FIT_NO_PROMOTION",
        "audit_corrections_applied": "6420504608_A_through_E",
        "season": a.season,
        "target_weeks": weeks,
        "prior_seasons": priors,
        "strict_prior": True,
        "identity_bridge": bridge,
        "sportsbook_inputs_used": 0,
        **diag,
    }
    a.status_out.parent.mkdir(parents=True, exist_ok=True)
    a.status_out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    print("\n=== PLAYER IDENTITY CARD V1 ===")
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"\nwrote {a.out}")

    # Legibility check: the architecture's own point is that volume and
    # high-value opportunity are different things. Show where they diverge.
    last = cards.loc[cards["target_week"].eq(max(weeks))
                     & cards["opportunity_kind"].eq("carry")
                     & cards["opp_total_current_season"].ge(20)].copy()
    if not last.empty:
        last["quality_gap"] = (last["team_inside_5_share_current"]
                               - last["team_share_current"])
        last = last.sort_values("quality_gap", ascending=False)
        print(f"\ninside-5 team share MINUS overall carry share, week {max(weeks)}")
        print("(positive = owns the valuable touches beyond his volume)")
        cols = ["player", "team_target_week", "opp_total_current_season",
                "participation_weeks_prior", "active_zero_opportunity_weeks",
                "team_share_current", "team_inside_5_share_current", "quality_gap"]
        print(last[[c for c in cols if c in last.columns]].head(12)
              .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
