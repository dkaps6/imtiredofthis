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
    if not (r_gsis and r_pfr and s_pfr and s_off):
        diag["bridge_state"] = "unavailable_missing_columns"
        diag["bridge_columns"] = {
            "roster_gsis": r_gsis, "roster_pfr": r_pfr,
            "snap_pfr": s_pfr, "snap_offense": s_off,
        }
        return pd.DataFrame(), diag

    bridge = (rosters[["season", "week", r_gsis, r_pfr]]
              .rename(columns={r_gsis: "gsis_id", r_pfr: "pfr_id"})
              .dropna(subset=["gsis_id", "pfr_id"])
              .drop_duplicates(["season", "week", "pfr_id"]))
    s = snaps[["season", "week", s_pfr, s_off]].rename(
        columns={s_pfr: "pfr_id", s_off: "offense_snaps"})
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    bridge["season"] = pd.to_numeric(bridge["season"], errors="coerce")
    bridge["week"] = pd.to_numeric(bridge["week"], errors="coerce")

    merged = s.merge(bridge, on=["season", "week", "pfr_id"], how="left")
    matched = merged["gsis_id"].notna()
    diag["bridge_matched"] = int(matched.sum())
    diag["bridge_unmatched"] = int((~matched).sum())
    diag["bridge_match_rate"] = (float(matched.mean())
                                 if len(merged) else float("nan"))
    diag["bridge_state"] = "ok" if matched.any() else "unavailable_no_matches"

    part = merged.loc[matched].copy()
    part["offense_snaps"] = pd.to_numeric(part["offense_snaps"],
                                          errors="coerce").fillna(0)
    part = (part.loc[part["offense_snaps"] > 0]
            .groupby(["season", "week", "gsis_id"], as_index=False)
            ["offense_snaps"].max())
    diag["participation_player_weeks"] = int(len(part))
    return part, diag


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
        o["complete"] = _num_series(p, "complete_pass", 0.0).fillna(0.0)
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
        else:
            part_weeks = pd.Series(dtype=int)

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
                if have_part and n_part > 0:
                    per_week = _rate(n_cur, n_part)
                    wk_counts = (gc.groupby("week").size()
                                 if n_cur else pd.Series(dtype=int))
                    zeros = max(0, n_part - int(wk_counts.shape[0]))
                    series = pd.concat(
                        [wk_counts, pd.Series([0] * zeros, dtype=int)]
                    ) if zeros else wk_counts
                    sd = float(series.std()) if len(series) > 1 else np.nan
                    obs = sorted(wk_counts.index.tolist()) if n_cur else []
                    recent_weeks = sorted(
                        pw.loc[pw["gsis_id"].eq(pid), "week"].unique()
                    )[-2:] if n_part else []
                    n_recent = int(wk_counts.reindex(recent_weeks).fillna(0).sum()) \
                        if n_cur and recent_weeks else 0
                    per_week_recent = _rate(n_recent, len(recent_weeks)) \
                        if recent_weeks else np.nan
                    n_earlier = n_cur - n_recent
                    den_earlier = n_part - len(recent_weeks)
                    per_week_earlier = _rate(n_earlier, den_earlier) \
                        if den_earlier > 0 else np.nan
                else:
                    per_week = sd = per_week_recent = per_week_earlier = np.nan
                    obs = sorted(gc["week"].unique().tolist()) if n_cur else []

                tn = team_tot.get((team_now, kind), 0)
                trz = team_rz.get((team_now, kind), 0)
                ti5 = team_i5.get((team_now, kind), 0)

                rows.append({
                    "season": season,
                    "target_week": wk,
                    "gsis_id": pid,
                    "player": (pr[r_name] if r_name and pd.notna(pr.get(r_name))
                               else (g["pbp_name"].dropna().iloc[0]
                                     if n and g["pbp_name"].notna().any() else "")),
                    "roster_position": pr[r_pos] if r_pos else "",
                    "team_target_week": team_now,
                    "teams_in_history": ",".join(
                        sorted({str(t) for t in g["posteam"].dropna().unique()})),
                    "opportunity_kind": kind,
                    "history_state": history_state,
                    # --- participation (A)(B) -----------------------------
                    "participation_weeks_prior": n_part,
                    "participation_source_state": (
                        "available" if have_part else "unavailable"),
                    # --- opportunity QUANTITY -----------------------------
                    "opp_total_all_sources": n,
                    "opp_total_current_season": n_cur,
                    "opp_weeks_observed_current": len(obs),
                    "active_zero_opportunity_weeks": (
                        max(0, n_part - len(obs)) if have_part else np.nan),
                    "opp_per_participation_week": per_week,
                    "team_share_current": _rate(n_cur, tn),
                    # --- trajectory / stability ---------------------------
                    "opp_per_week_recent2": per_week_recent,
                    "opp_per_week_earlier": per_week_earlier,
                    "opp_week_to_week_sd": sd,
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
                    "team_red_zone_share_current": _rate(
                        gc["is_red_zone"].sum() if n_cur else 0, trz),
                    "team_inside_5_share_current": _rate(
                        gc["is_inside_5"].sum() if n_cur else 0, ti5),
                    # --- efficiency identity ------------------------------
                    "yards_per_opp": (float(g["yards"].mean())
                                      if n and g["yards"].notna().any() else np.nan),
                    "yards_per_opp_sd": (float(g["yards"].std())
                                         if n and g["yards"].notna().sum() > 1 else np.nan),
                    "catch_rate": (float(g["complete"].mean())
                                   if kind == "target" and n
                                   and g["complete"].notna().any() else np.nan),
                    "mean_air_yards": (float(g["air_yards"].mean())
                                       if kind == "target" and n
                                       and g["air_yards"].notna().any() else np.nan),
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

    pbp = load_pbp(seasons)
    print(f"[identity] pbp rows={len(pbp)}")

    cards, diag = build_cards(pbp, rosters, participation, a.season, weeks, priors)
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
