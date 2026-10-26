#!/usr/bin/env python3
"""Player Identity Card V1 — strictly-prior per-player, per-week football state.

Implements the descriptive substrate specified in
`docs/research/PLAYER_IDENTITY_ROLE_STATE_ARCHITECTURE_V1.md`:

    For every player, maintain a strictly-prior historical identity state.

This is **state construction, not a predictive experiment**. It fits nothing,
promotes nothing, and never reads the target week's outcomes to build that
week's card. Any predictive *use* of this state is a separate, preregistered
question.

Why this exists
---------------
The production board carries 79 columns and almost no football context: no
target share, snap share, route participation, red-zone usage, teammate
competition or situational role. The per-player evidence the pipeline does
build (`player_form`, entitlement traces) is run-artifact only and expires.
So there has been no durable substrate from which a named player's identity
could accumulate across a season.

Everything here is rebuilt from published nflverse sources, so the whole
season is recoverable and no expired artifact is required.

The central distinction
-----------------------
Opportunity is not one number. Fifteen carries between the twenties is not ten
carries with three inside the five. Six targets is not six targets if one
player draws schemed red-zone looks and another draws underneath checkdowns.
So quantity and **quality/location/purpose** are tracked separately, per the
architecture's "Opportunity Quality / Topology" section.

Strict-prior contract
---------------------
A card for (player, season, week=w) is built **only** from plays in weeks < w
of that season, optionally extended by prior seasons. Week w never contributes
to its own card. That makes the output leakage-safe for evaluating any
week-w hypothesis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nflreadpy as nflv
except ImportError:  # pragma: no cover - environment guard
    print("FATAL: nflreadpy is required", file=sys.stderr)
    raise

# High-value opportunity zones, by distance to the opponent goal line.
RED_ZONE = 20
INSIDE_10 = 10
INSIDE_5 = 5
TWO_MINUTE_SECONDS = 120


def _to_pandas(df):
    try:
        return df.to_pandas()
    except AttributeError:
        return df


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    df = _to_pandas(nflv.load_pbp(seasons=seasons))
    df.columns = [str(c).lower() for c in df.columns]
    return df


def load_snaps(seasons: list[int]) -> pd.DataFrame:
    """Snap counts give participation, which is the base of opportunity."""
    try:
        df = _to_pandas(nflv.load_snap_counts(seasons=seasons))
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception as exc:
        print(f"WARNING: snap counts unavailable ({type(exc).__name__}: {exc}); "
              "participation fields will be labelled unavailable")
        return pd.DataFrame()


def _rate(num, den):
    """Share with an explicit NaN when there is no denominator to divide by."""
    den = float(den)
    return float(num) / den if den > 0 else np.nan


def _player_rows(pbp: pd.DataFrame) -> pd.DataFrame:
    """Explode plays into one row per (player, role-in-play).

    A single play can give a carry to one player and a target to another, so
    rushing and receiving involvement are emitted separately and then
    aggregated per player.
    """
    frames = []
    base = [
        "season", "week", "posteam", "defteam", "yardline_100", "down",
        "goal_to_go", "half_seconds_remaining", "yards_gained",
    ]
    base = [c for c in base if c in pbp.columns]

    if {"rusher_player_id", "rush_attempt"} <= set(pbp.columns):
        r = pbp.loc[pd.to_numeric(pbp["rush_attempt"], errors="coerce").eq(1)].copy()
        r = r[r["rusher_player_id"].notna()]
        out = r[base].copy()
        out["player_id"] = r["rusher_player_id"]
        out["player"] = r.get("rusher_player_name")
        out["opp_kind"] = "carry"
        out["yards"] = pd.to_numeric(r["yards_gained"], errors="coerce")
        frames.append(out)

    if {"receiver_player_id", "pass_attempt"} <= set(pbp.columns):
        p = pbp.loc[pd.to_numeric(pbp["pass_attempt"], errors="coerce").eq(1)].copy()
        p = p[p["receiver_player_id"].notna()]
        out = p[base].copy()
        out["player_id"] = p["receiver_player_id"]
        out["player"] = p.get("receiver_player_name")
        out["opp_kind"] = "target"
        out["yards"] = pd.to_numeric(p["yards_gained"], errors="coerce")
        out["complete"] = pd.to_numeric(
            p.get("complete_pass", 0), errors="coerce"
        ).fillna(0)
        out["air_yards"] = pd.to_numeric(p.get("air_yards"), errors="coerce")
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ("complete", "air_yards"):
        if c not in df.columns:
            df[c] = np.nan
    return df


def _flag_situations(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the high-value-opportunity flags that define quality."""
    yl = pd.to_numeric(df.get("yardline_100"), errors="coerce")
    down = pd.to_numeric(df.get("down"), errors="coerce")
    g2g = pd.to_numeric(df.get("goal_to_go"), errors="coerce").fillna(0)
    half = pd.to_numeric(df.get("half_seconds_remaining"), errors="coerce")
    df = df.copy()
    df["is_red_zone"] = yl.le(RED_ZONE)
    df["is_inside_10"] = yl.le(INSIDE_10)
    df["is_inside_5"] = yl.le(INSIDE_5)
    df["is_goal_to_go"] = g2g.eq(1)
    df["is_third_down"] = down.eq(3)
    df["is_two_minute"] = half.le(TWO_MINUTE_SECONDS)
    df["is_early_down"] = down.isin([1, 2])
    return df


def build_cards(
    pbp: pd.DataFrame,
    snaps: pd.DataFrame,
    season: int,
    weeks: list[int],
    prior_seasons: list[int],
) -> pd.DataFrame:
    """One card per (player, season, target_week), strictly prior."""
    ev = _flag_situations(_player_rows(pbp))
    if ev.empty:
        raise RuntimeError("no player-involvement rows derived from PBP")
    ev["season"] = pd.to_numeric(ev["season"], errors="coerce")
    ev["week"] = pd.to_numeric(ev["week"], errors="coerce")

    snap_ok = not snaps.empty and {"pfr_player_id", "week", "season"} <= set(snaps.columns)
    if not snap_ok and not snaps.empty:
        snap_ok = {"player", "week", "season", "offense_pct"} <= set(snaps.columns)

    rows = []
    for wk in weeks:
        # STRICT PRIOR: this season's earlier weeks, plus whole prior seasons.
        hist = ev.loc[
            (ev["season"].eq(season) & ev["week"].lt(wk))
            | (ev["season"].isin(prior_seasons))
        ].copy()
        cur = hist.loc[hist["season"].eq(season)]
        if hist.empty:
            continue

        # Team denominators for share, computed on the same prior window.
        team_tot = cur.groupby(["posteam", "opp_kind"]).size().rename("team_n")
        team_rz = (
            cur.loc[cur["is_red_zone"]].groupby(["posteam", "opp_kind"]).size()
            .rename("team_rz_n")
        )
        team_i5 = (
            cur.loc[cur["is_inside_5"]].groupby(["posteam", "opp_kind"]).size()
            .rename("team_i5_n")
        )

        for (pid, kind), g in hist.groupby(["player_id", "opp_kind"], dropna=True):
            gc = g.loc[g["season"].eq(season)]
            team = (gc["posteam"].mode().iat[0] if len(gc)
                    else g["posteam"].mode().iat[0])
            n = len(g)
            n_cur = len(gc)
            wks = sorted(gc["week"].dropna().unique().tolist())

            # Trajectory: most recent two prior weeks vs the rest of the season.
            recent = gc.loc[gc["week"].isin(wks[-2:])] if wks else gc.iloc[0:0]
            earlier = gc.loc[~gc["week"].isin(wks[-2:])] if wks else gc.iloc[0:0]
            per_wk = gc.groupby("week").size()

            tn = team_tot.get((team, kind), 0)
            trz = team_rz.get((team, kind), 0)
            ti5 = team_i5.get((team, kind), 0)

            rec = {
                "season": season,
                "target_week": wk,
                "player_id": pid,
                "player": (g["player"].dropna().mode().iat[0]
                           if g["player"].notna().any() else ""),
                "team": team,
                "opportunity_kind": kind,
                # --- opportunity QUANTITY -------------------------------
                "prior_weeks_observed": len(wks),
                "opp_total_all_sources": n,
                "opp_total_current_season": n_cur,
                "opp_per_week_current": _rate(n_cur, len(wks)) if wks else np.nan,
                "team_share_current": _rate(n_cur, tn),
                # --- trajectory / stability -----------------------------
                "opp_per_week_recent2": _rate(len(recent), min(2, len(wks))) if wks else np.nan,
                "opp_per_week_earlier": _rate(len(earlier), max(0, len(wks) - 2)) if len(wks) > 2 else np.nan,
                "opp_week_to_week_sd": float(per_wk.std()) if len(per_wk) > 1 else np.nan,
                # --- opportunity QUALITY / topology ---------------------
                "red_zone_opp": int(g["is_red_zone"].sum()),
                "inside_10_opp": int(g["is_inside_10"].sum()),
                "inside_5_opp": int(g["is_inside_5"].sum()),
                "goal_to_go_opp": int(g["is_goal_to_go"].sum()),
                "third_down_opp": int(g["is_third_down"].sum()),
                "two_minute_opp": int(g["is_two_minute"].sum()),
                "early_down_opp": int(g["is_early_down"].sum()),
                "red_zone_rate_of_own_opp": _rate(g["is_red_zone"].sum(), n),
                "inside_5_rate_of_own_opp": _rate(g["is_inside_5"].sum(), n),
                # the share of the TEAM's high-value work this player owns
                "team_red_zone_share_current": _rate(
                    gc["is_red_zone"].sum() if len(gc) else 0, trz),
                "team_inside_5_share_current": _rate(
                    gc["is_inside_5"].sum() if len(gc) else 0, ti5),
                # --- efficiency identity --------------------------------
                "yards_per_opp": float(g["yards"].mean()) if g["yards"].notna().any() else np.nan,
                "yards_per_opp_sd": float(g["yards"].std()) if g["yards"].notna().sum() > 1 else np.nan,
                "catch_rate": (float(g["complete"].mean())
                               if kind == "target" and g["complete"].notna().any() else np.nan),
                "mean_air_yards": (float(g["air_yards"].mean())
                                   if kind == "target" and g["air_yards"].notna().any() else np.nan),
                # --- provenance -----------------------------------------
                "strict_prior": True,
                "history_seasons": ",".join(
                    str(int(s)) for s in sorted(g["season"].dropna().unique())),
                "snap_source_state": "available" if snap_ok else "unavailable",
                "sportsbook_inputs_used": 0,
            }
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no identity cards produced")
    return out.sort_values(
        ["season", "target_week", "team", "opportunity_kind", "opp_total_current_season"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", required=True, help="e.g. 1-7 or 3,4,5")
    ap.add_argument("--prior-seasons", default="",
                    help="comma list of whole prior seasons to include as history")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--status-out", type=Path, required=True)
    a = ap.parse_args()

    if "-" in a.weeks:
        lo, hi = a.weeks.split("-", 1)
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in a.weeks.split(",") if w.strip()]
    priors = [int(s) for s in a.prior_seasons.split(",") if s.strip()]

    seasons = sorted({a.season, *priors})
    print(f"[identity] loading PBP for {seasons}")
    pbp = load_pbp(seasons)
    print(f"[identity] pbp rows={len(pbp)}")
    snaps = load_snaps(seasons)
    print(f"[identity] snap rows={len(snaps)}")

    cards = build_cards(pbp, snaps, a.season, weeks, priors)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cards.to_csv(a.out, index=False)

    status = {
        "version": "PLAYER_IDENTITY_CARD_V1",
        "scientific_boundary": "DESCRIPTIVE_STATE_ONLY_NO_FIT_NO_PROMOTION",
        "season": a.season,
        "target_weeks": weeks,
        "prior_seasons": priors,
        "cards": int(len(cards)),
        "players": int(cards["player_id"].nunique()),
        "teams": int(cards["team"].nunique()),
        "strict_prior": True,
        "snap_source_state": str(cards["snap_source_state"].iloc[0]),
        "sportsbook_inputs_used": 0,
    }
    a.status_out.parent.mkdir(parents=True, exist_ok=True)
    a.status_out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    print("\n=== PLAYER IDENTITY CARD V1 ===")
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"\nwrote {a.out}")

    # A legibility check: the architecture's own example is that volume and
    # high-value opportunity are different things. Show the players where they
    # diverge most, since that divergence is the whole point.
    last = cards.loc[cards["target_week"].eq(max(weeks))].copy()
    car = last.loc[last["opportunity_kind"].eq("carry")]
    car = car.loc[car["opp_total_current_season"].ge(20)]
    if not car.empty:
        car = car.assign(
            quality_gap=car["team_inside_5_share_current"] - car["team_share_current"]
        ).sort_values("quality_gap", ascending=False)
        print(f"\ninside-5 share MINUS overall carry share, week {max(weeks)} "
              "(positive = owns the valuable touches beyond his volume):")
        cols = ["player", "team", "opp_total_current_season",
                "team_share_current", "team_inside_5_share_current", "quality_gap"]
        print(car[cols].head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
