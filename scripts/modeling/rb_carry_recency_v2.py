"""RB_CARRY_SHARE_RECENCY_V2: recent carry share for RBs, with the anomaly gate.

Shared by production PlayerForm (scripts/player_form_v2.py) and the validation
(scripts/research/evaluate_rb_recency_end_to_end_v2.py), so both run the same code.

For an RB/FB whose last three games with his slate team were all played healthy and in
full, the current-season carry share fed to the Bayesian baseline becomes

    0.15 x season-to-date share (production's existing value) + 0.85 x last-3 share

Otherwise the season-to-date share is left exactly as it was. No other position, metric or
game count changes.

What V2 fixes in V1 (PR review, 2026-11-24):
- the last-3 window counts every game he took an offensive snap, with carries filled as 0.
  V1 used PlayerForm's logs, which drop zero-touch games, so the window reached back to
  older and busier games;
- AGENTS.md anomaly gate. Each game in the window is state-labelled from snap counts and the
  official injury report. Recency applies only when all three are HEALTHY_FULL. Otherwise
  it fails closed to season-to-date;
- only RB/FB rows change. V1's validation changed every position's share.

Game states (a game is judged against the median offense snap share of his OTHER games with
the same team; the strictly-prior history only):
- RETURN_FROM_INJURY: he missed his team's previous game(s), and this game's snap share is
  below 0.75 x his baseline;
- INJURY_LIMITED: he was on that week's injury report (game status, or limited / no
  practice), and his snap share is below 0.75 x his baseline;
- PARTIAL_GAME: his snap share fell below 0.5 x baseline (and at least 0.20 lower), and he
  was on the following week's report, so the likely cause is an in-game injury;
- UNRESOLVED_CONTEXT: the same fall with no injury record (an early exit or a benching that
  no source explains), or a spike above 2 x baseline (and at least 0.20 higher), which a
  teammate's absence or exit can cause;
- HEALTHY_FULL: none of the above.

A persistent new level is not an anomaly. Once it lasts, it becomes most of the baseline and
stops being flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "RB_CARRY_SHARE_RECENCY_V2"
WEIGHT = 0.85
MIN_GAMES = 3
POSITIONS = frozenset({"RB", "FB"})
LIMITED_RATIO = 0.75
DROP_RATIO = 0.5
SPIKE_RATIO = 2.0
MIN_ABS_GAP = 0.20
HEALTHY = "HEALTHY_FULL"

_LISTED_REPORT = {"questionable", "doubtful", "out"}
_LISTED_PRACTICE = {"limited participation in practice", "did not participate in practice"}


def injury_listed(injuries: pd.DataFrame) -> pd.DataFrame:
    """(key, week) pairs where the player carried a game status or a limited/no practice."""
    if injuries is None or injuries.empty:
        return pd.DataFrame(columns=["key", "week"])
    rep = injuries.get("report_status", pd.Series("", index=injuries.index)).astype("string").fillna("").str.strip().str.lower()
    pra = injuries.get("practice_status", pd.Series("", index=injuries.index)).astype("string").fillna("").str.strip().str.lower()
    hit = rep.isin(_LISTED_REPORT) | pra.isin(_LISTED_PRACTICE)
    out = injuries.loc[hit, ["key", "week"]].copy()
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    return out.dropna().drop_duplicates()


def label_games(games: pd.DataFrame, team_weeks: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """State-label every snap-qualified game.

    games: one row per (key, team, week) in which the player took >= 1 offensive snap for
        that team, with offense_pct (0-1), carries (0 when none) and team_carries.
    team_weeks: (team, week) of every game each team has played in the history.
    injuries: (key, week, report_status, practice_status) from the official report.
    """
    g = games.copy()
    g["week"] = pd.to_numeric(g["week"], errors="coerce")
    g = g.sort_values(["key", "team", "week"]).reset_index(drop=True)
    pct = pd.to_numeric(g["offense_pct"], errors="coerce")

    # baseline = median snap share of his OTHER games with the same team
    grp = g.groupby(["key", "team"], sort=False)
    base = pd.Series(np.nan, index=g.index)
    n_other = grp["week"].transform("size") - 1
    for _, idx in grp.groups.items():
        v = pct.loc[idx].to_numpy(dtype=float)
        if len(v) < 3:
            continue
        for j, i in enumerate(idx):
            base.at[i] = float(np.median(np.delete(v, j)))
    g["baseline_pct"] = base
    g["n_other_games"] = n_other

    listed = injury_listed(injuries)
    listed_set = set(zip(listed["key"], listed["week"].astype(int)))
    g["listed_this_week"] = [(k, int(w)) in listed_set for k, w in zip(g["key"], g["week"])]
    g["listed_next_week"] = [(k, int(w) + 1) in listed_set for k, w in zip(g["key"], g["week"])]

    # missed the team's previous game: a team game between his previous appearance and this one
    tw = team_weeks[["team", "week"]].drop_duplicates().copy()
    tw["week"] = pd.to_numeric(tw["week"], errors="coerce")
    team_sched = {t: np.sort(x["week"].to_numpy(dtype=float)) for t, x in tw.groupby("team")}
    prev_week = grp["week"].shift(1)
    missed = []
    for t, w, pw in zip(g["team"], g["week"], prev_week):
        if pd.isna(pw):
            missed.append(False)
            continue
        s = team_sched.get(t, np.array([]))
        missed.append(bool(((s > pw) & (s < w)).any()))
    g["missed_previous_team_game"] = missed

    b = g["baseline_pct"]
    limited = pct < LIMITED_RATIO * b
    drop = (pct < DROP_RATIO * b) & ((b - pct) >= MIN_ABS_GAP)
    spike = (pct > SPIKE_RATIO * b) & ((pct - b) >= MIN_ABS_GAP)
    state = np.select(
        [
            b.isna(),
            g["missed_previous_team_game"] & limited,
            g["listed_this_week"] & limited,
            drop & g["listed_next_week"],
            drop | spike,
        ],
        ["NO_BASELINE", "RETURN_FROM_INJURY", "INJURY_LIMITED", "PARTIAL_GAME", "UNRESOLVED_CONTEXT"],
        default=HEALTHY,
    )
    g["game_state"] = state
    return g


def last3(labeled: pd.DataFrame, slate_team: pd.DataFrame) -> pd.DataFrame:
    """Per player: last-3 share with his slate team, and whether the window is clean.

    slate_team: (key, team) for the slate. Returns one row per key with rush_share_last3,
    window_games, window_states, window_weeks and window_clean.
    """
    h = labeled.merge(slate_team[["key", "team"]].drop_duplicates("key"), on=["key", "team"], how="inner")
    h = h.sort_values(["key", "week"])
    w = h.groupby("key", sort=False).tail(MIN_GAMES)
    agg = w.groupby("key").agg(
        l3_carries=("carries", "sum"),
        l3_team=("team_carries", "sum"),
        window_games=("week", "nunique"),
        window_states=("game_state", lambda s: "|".join(map(str, s))),
        window_weeks=("week", lambda s: "|".join(str(int(x)) for x in s)),
    )
    agg["window_clean"] = agg["window_states"].map(lambda s: all(x == HEALTHY for x in s.split("|")))
    ok = (agg["window_games"] >= MIN_GAMES) & (agg["l3_team"] > 0)
    agg["rush_share_last3"] = np.where(ok, agg["l3_carries"] / agg["l3_team"].where(agg["l3_team"] > 0), np.nan)
    return agg[["rush_share_last3", "window_games", "window_states", "window_weeks", "window_clean"]]


def apply(form: pd.DataFrame, windows: pd.DataFrame | None, *, key_col: str = "player_identity_key",
          unavailable_reason: str = "") -> pd.DataFrame:
    """Write the blended share for eligible RB/FB rows and the audit columns for every row.

    form must carry key_col, position, current_games and rush_share_current (season-to-date).
    windows is last3() output indexed by key, or None when the snap / injury sources are
    unavailable, in which case nothing changes and every RB row records why.
    """
    out = form.copy()
    std = pd.to_numeric(out.get("rush_share_current"), errors="coerce")
    out["rush_share_current_season"] = std
    out["rush_share_last3"] = np.nan
    out["rb_carry_recency_window_states"] = ""
    out["rb_carry_recency_window_weeks"] = ""
    out["rb_carry_recency_applied"] = 0
    out["rb_carry_recency_status"] = ""
    out["rb_carry_recency_version"] = VERSION
    pos = out.get("position", pd.Series("", index=out.index)).astype("string").fillna("").str.upper()
    is_rb = pos.isin(POSITIONS)
    out.loc[~is_rb, "rb_carry_recency_status"] = "NOT_RB"
    if windows is None:
        out.loc[is_rb, "rb_carry_recency_status"] = f"SOURCE_UNAVAILABLE:{unavailable_reason}"[:200]
        return out
    k = out[key_col]
    l3s = pd.to_numeric(k.map(windows["rush_share_last3"]), errors="coerce")
    clean = k.map(windows["window_clean"]).fillna(False).astype(bool)
    out["rush_share_last3"] = l3s
    out["rb_carry_recency_window_states"] = k.map(windows["window_states"]).fillna("")
    out["rb_carry_recency_window_weeks"] = k.map(windows["window_weeks"]).fillna("")
    cg = pd.to_numeric(out.get("current_games"), errors="coerce").fillna(0)
    status = np.select(
        [~is_rb, cg.lt(MIN_GAMES) | l3s.isna(), std.isna(), ~clean],
        ["NOT_RB", "FEWER_THAN_3_GAMES", "NO_SEASON_SHARE", "WITHHELD_ANOMALY"],
        default="APPLIED",
    )
    out["rb_carry_recency_status"] = status
    eligible = out["rb_carry_recency_status"].eq("APPLIED")
    out.loc[eligible, "rush_share_current"] = (1.0 - WEIGHT) * std[eligible] + WEIGHT * l3s[eligible]
    out.loc[eligible, "rb_carry_recency_applied"] = 1
    return out


def snap_games(snaps: pd.DataFrame, pfr_to_key: dict, logs: pd.DataFrame, before_week: int) -> pd.DataFrame:
    """Snap-qualified (key, team, week) games before `before_week`, with carries and team carries.

    snaps: nflverse snap counts (pfr_player_id, team, week, offense_snaps, offense_pct).
    pfr_to_key: pfr id -> identity key.
    logs: PlayerForm weekly logs for the season (key, team, week, rushes, team_rushes).
    Team carries come from the logs' team totals; a snap row for a team-week with no log
    team total is dropped.
    """
    s = snaps.copy()
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s = s[(s["week"] < int(before_week)) & (pd.to_numeric(s["offense_snaps"], errors="coerce").fillna(0) > 0)]
    s["key"] = s["pfr_player_id"].map(pfr_to_key)
    s = s.dropna(subset=["key"])
    s = s[["key", "team", "week", "offense_pct"]].drop_duplicates(["key", "team", "week"])
    car = logs[["key", "team", "week", "rushes"]].groupby(["key", "team", "week"], as_index=False)["rushes"].sum()
    tc = logs.groupby(["team", "week"], as_index=False)["team_rushes"].first().rename(columns={"team_rushes": "team_carries"})
    g = s.merge(car, on=["key", "team", "week"], how="left").merge(tc, on=["team", "week"], how="inner")
    g["carries"] = pd.to_numeric(g["rushes"], errors="coerce").fillna(0.0)
    return g.drop(columns="rushes")
