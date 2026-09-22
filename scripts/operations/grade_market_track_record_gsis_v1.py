#!/usr/bin/env python3
"""Corrected actual-outcome grading for the market track record.

Fixes three real bugs found grading the real Week 1 2026 board against
nflreadpy:

1. `load_actual_stats()` in `grade_market_track_record_v1.py` reuses
   `player_form_v2._normalize_weekly()`, which drops any player-week row
   where `targets + rushes + pass_att == 0`. That filter is correct for its
   original purpose (usage-rate features have nothing to model at zero
   usage), but wrong here: a rostered player with zero recorded involvement
   in a completed game is a real, gradable outcome (a legitimate UNDER),
   not missing data. This loader keeps every row regardless of usage.

2. The original matcher joins bets to actuals on
   (season, week, team, player_clean_key) -- a bare name-derived key, which
   silently drops any row where the live PlayerForm/roster source spells a
   name differently than nflreadpy's own player_display_name (found: Brian
   Robinson Jr./Michael Pittman Jr./Travis Etienne Jr. -- nflreadpy drops
   the suffix; Chris Godwin -- nflreadpy adds it). This resolver instead
   builds a per-team alias->GSIS index from nflreadpy's own weekly stats
   table (exact full name and suffix-stripped name, both team-scoped and
   globally-unique fallbacks), matching the same deterministic, fail-closed,
   no-fuzzy-matching hierarchy used for the WR-R15 historical research
   tonight, then joins actuals by (season, week, resolved GSIS) instead of
   by name key.

3. A player absent from the weekly stats table may either have participated
   with zero box-score usage or may have been inactive/DNP. Roster presence
   proves identity, not sportsbook action. The grader therefore joins
   postgame PFR snap counts: any positive offense/defense/special-teams snap
   confirms participation and permits a verified zero; explicit `INA` with
   no participation is settled VOID for the captured DraftKings/FanDuel
   player-prop books; missing/ambiguous participation evidence fails closed.

Source-only. No production change; this does not modify player_form_v2 or
the live pricing pipeline, only how already-priced boards get graded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.operations.grade_market_track_record_v1 import (
    MARKET_STAT_COLUMNS, load_boards, select_model_bet, num, outcome_side,
    american_profit, edge_bucket,
)
from scripts.utils.canonical_names import canonicalize_player_name_safe

TG = ["season", "week", "team"]
DNP_VOID_BOOKS = {"draftkings", "fanduel"}
ACTIVE_ROSTER_STATUSES = {"ACT", "ACTIVE"}
INACTIVE_ROSTER_STATUSES = {"INA", "INACTIVE", "DNP"}


def _book_key(value) -> str:
    return str(value or "").strip().lower()




def _to_pandas(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def _suffix_strip(key: str) -> str:
    # Longest-first is required: a III key also ends with II, and an IV key
    # also ends with V. Checking the shorter token first corrupts the base key.
    for suf in ("iii", "jr", "sr", "ii", "iv", "v"):
        if key.endswith(suf) and len(key) > len(suf):
            return key[: -len(suf)]
    return key


def load_actual_stats_unfiltered(season: int, weeks: list[int] | None) -> pd.DataFrame:
    """Same source as load_actual_stats() but WITHOUT the usage>0 filter, and
    with GSIS player_id carried through for identity resolution."""
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    x = _to_pandas(raw)
    x.columns = [str(c).strip().lower() for c in x.columns]
    s = pd.to_numeric(x.get("season"), errors="coerce")
    x = x.loc[s.eq(int(season))].copy()
    x["week"] = pd.to_numeric(x.get("week"), errors="coerce")
    x = x.loc[x["week"].notna()].copy()
    x["week"] = x["week"].astype(int)
    if weeks:
        x = x.loc[x["week"].isin(weeks)].copy()
    x["season"] = int(season)
    x["gsis_id"] = x.get("player_id", x.get("gsis_id", "")).astype("string").fillna("").str.strip()
    raw_name = x.get("player_display_name", x.get("player_name", "")).astype("string").fillna("").str.strip()
    canon = raw_name.map(canonicalize_player_name_safe)
    x["player"] = canon.map(lambda t: t[0])
    x["player_clean_key"] = canon.map(lambda t: t[1])
    team_raw = x.get("recent_team", x.get("team", "")).astype("string").fillna("").str.strip()
    x["team"] = team_raw.map(canon_team)
    for col in ["targets", "receptions", "rushing_attempts", "carries"]:
        pass
    x["receptions"] = pd.to_numeric(x.get("receptions"), errors="coerce").fillna(0.0)
    x["rec_yards"] = pd.to_numeric(x.get("receiving_yards"), errors="coerce").fillna(0.0)
    x["rush_yards"] = pd.to_numeric(x.get("rushing_yards"), errors="coerce").fillna(0.0)
    x["pass_yards"] = pd.to_numeric(x.get("passing_yards"), errors="coerce").fillna(0.0)
    x["position"] = x.get("position", x.get("position_group", "")).astype("string").fillna("").str.strip().str.upper()
    x = x.loc[x["team"].astype(str).ne("") & x["gsis_id"].astype(str).ne("")].copy()
    keep = ["season", "week", "team", "gsis_id", "player", "player_clean_key", "position",
            "receptions", "rec_yards", "rush_yards", "pass_yards"]
    out = x[keep].drop_duplicates(["season", "week", "gsis_id"], keep="last")
    return out


def load_roster_identity(season: int, weeks: list[int] | None) -> pd.DataFrame:
    """(season, week, team, gsis_id, player_clean_key) identity evidence from
    nflreadpy's weekly rosters -- broader than the stats table, since it
    includes players who were rostered but recorded nothing that week
    (including explicit inactive status), which the stats table omits
    entirely rather than zero-filling."""
    import nflreadpy as nfl

    raw = nfl.load_rosters_weekly(int(season))
    x = _to_pandas(raw)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = pd.to_numeric(x.get("season", season), errors="coerce").fillna(season).astype(int)
    x["week"] = pd.to_numeric(x.get("week"), errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].notna()].copy()
    x["week"] = x["week"].astype(int)
    if weeks:
        x = x.loc[x["week"].isin(weeks)].copy()
    team_col = next((c for c in ("team", "team_abbr", "club_code") if c in x.columns), None)
    x["team"] = x[team_col].astype("string").fillna("").str.strip().map(canon_team) if team_col else ""
    x["gsis_id"] = x.get("gsis_id", x.get("player_id", "")).astype("string").fillna("").str.strip()
    name_col = next((c for c in ("full_name", "football_name", "player_name", "player") if c in x.columns), None)
    raw_name = x[name_col].astype("string").fillna("").str.strip() if name_col else pd.Series("", index=x.index)
    canon = raw_name.map(canonicalize_player_name_safe)
    x["player_clean_key"] = canon.map(lambda t: t[1])
    x["status"] = x.get("status", "").astype("string").fillna("")
    x["position"] = x.get("position", x.get("depth_chart_position", "")).astype("string").fillna("").str.strip().str.upper()
    x = x.loc[x["team"].astype(str).ne("") & x["gsis_id"].astype(str).ne("") & x["player_clean_key"].astype(str).ne("")]
    return x[["season", "week", "team", "gsis_id", "player_clean_key", "status", "position"]].drop_duplicates()


def load_snap_participation(season: int, weeks: list[int] | None) -> pd.DataFrame:
    """Postgame participation evidence from PFR snap counts.

    Any positive offense, defense, or special-teams snap proves the player
    participated in the event. Rows are team/name keyed because the PFR snap
    source does not expose GSIS ids.
    """
    import nflreadpy as nfl

    raw = nfl.load_snap_counts(seasons=[int(season)])
    x = _to_pandas(raw)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = pd.to_numeric(x.get("season", season), errors="coerce")
    x["week"] = pd.to_numeric(x.get("week"), errors="coerce")
    x = x.loc[
        x["season"].eq(int(season))
        & x["week"].notna()
        & x["week"].between(1, 18)
    ].copy()
    x["week"] = x["week"].astype(int)
    if weeks:
        x = x.loc[x["week"].isin(weeks)].copy()

    team_col = next((c for c in ("team", "team_abbr", "club") if c in x.columns), None)
    name_col = next((c for c in ("player", "player_name", "full_name") if c in x.columns), None)
    if team_col is None or name_col is None:
        raise RuntimeError("snap participation source missing team/player")

    x["team"] = (
        x[team_col].astype("string").fillna("").str.strip().map(canon_team)
    )
    canon = (
        x[name_col].astype("string").fillna("").str.strip()
        .map(canonicalize_player_name_safe)
    )
    x["player_clean_key"] = canon.map(lambda t: t[1])

    snap_cols = [c for c in ("offense_snaps", "defense_snaps", "st_snaps") if c in x.columns]
    if not snap_cols:
        raise RuntimeError("snap participation source missing snap-count columns")
    total = pd.Series(0.0, index=x.index)
    for c in snap_cols:
        total = total + pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x["snap_participated"] = total.gt(0)

    x = x.loc[
        x["team"].astype(str).ne("")
        & x["player_clean_key"].astype(str).ne("")
    ].copy()
    return (
        x[["season", "week", "team", "player_clean_key", "snap_participated"]]
        .groupby(["season", "week", "team", "player_clean_key"], as_index=False)
        .agg(snap_participated=("snap_participated", "max"))
    )


def _status_map(roster: pd.DataFrame) -> dict:
    out = {}
    for key, q in roster.groupby(["season", "week", "team", "gsis_id"], dropna=False):
        vals = sorted({
            str(v).strip().upper()
            for v in q["status"].tolist()
            if str(v).strip()
        })
        out[key] = vals[0] if len(vals) == 1 else ("AMBIGUOUS:" + "|".join(vals) if vals else "")
    return out


def _snap_participation_sets(snaps: pd.DataFrame) -> tuple[set, set]:
    exact = set()
    base = set()
    for r in snaps.loc[snaps["snap_participated"]].itertuples(index=False):
        exact.add((int(r.season), int(r.week), r.team, r.player_clean_key))
        base.add((int(r.season), int(r.week), r.team, _suffix_strip(r.player_clean_key)))
    return exact, base


def apply_postgame_settlement(detail: pd.DataFrame) -> pd.DataFrame:
    """Classify actual/void/unresolved outcomes without turning DNP into zero."""
    out = detail.copy()
    resolved = out["identity_status"].eq("RESOLVED_GSIS")
    has_stat_row = out["actual"].notna()
    participated = out.get("snap_participated", False)
    if not isinstance(participated, pd.Series):
        participated = pd.Series(bool(participated), index=out.index)
    participated = participated.fillna(False).astype(bool)

    status = (
        out.get("roster_status", pd.Series("", index=out.index))
        .astype("string").fillna("").str.strip().str.upper()
    )
    book = (
        out.get("book", pd.Series("", index=out.index))
        .astype("string").fillna("").str.strip().str.lower()
    )
    rostered = out.get(
        "roster_confirmed_this_team_week",
        pd.Series(False, index=out.index),
    ).fillna(False).astype(bool)

    missing_stat = resolved & ~has_stat_row
    participated_zero = missing_stat & participated
    explicit_dnp = (
        missing_stat
        & rostered
        & ~participated
        & status.isin(INACTIVE_ROSTER_STATUSES)
        & book.isin(DNP_VOID_BOOKS)
    )

    out.loc[participated_zero, "actual"] = 0.0
    out["actual_source"] = np.select(
        [resolved & has_stat_row, participated_zero, explicit_dnp],
        ["stats_table", "snap_confirmed_verified_zero", "sportsbook_void_dnp"],
        default="unresolved",
    )
    out["settlement_status"] = np.select(
        [
            out["actual_source"].isin(["stats_table", "snap_confirmed_verified_zero"]),
            out["actual_source"].eq("sportsbook_void_dnp"),
        ],
        ["SETTLED", "VOID"],
        default="UNRESOLVED",
    )
    out["has_verified_actual"] = out["settlement_status"].eq("SETTLED")
    return out


def build_alias_index(actual: pd.DataFrame, roster: pd.DataFrame) -> dict:
    """team-scoped exact/suffix-stripped alias -> set of GSIS ids, plus a
    global (any-team) index for the fallback tiers, built from the UNION of
    stats-table and roster-table identity evidence so a genuinely-inactive
    player (present on roster, absent from stats) can still resolve.
    Ambiguous buckets (more than one GSIS under the same alias) are kept
    as-is so the resolver can fail closed on them."""
    idx = {"team_exact": {}, "team_base": {}, "global_exact": {}, "global_base": {}}
    for source in (actual, roster):
        for r in source.itertuples(index=False):
            key_exact = r.player_clean_key
            key_base = _suffix_strip(key_exact)
            idx["team_exact"].setdefault((r.team, key_exact), set()).add(r.gsis_id)
            idx["team_base"].setdefault((r.team, key_base), set()).add(r.gsis_id)
            idx["global_exact"].setdefault(key_exact, set()).add(r.gsis_id)
            idx["global_base"].setdefault(key_base, set()).add(r.gsis_id)
    return idx


def resolve_gsis(player_clean_key: str, team: str, idx: dict) -> tuple[str, str]:
    """Returns (gsis_id_or_empty, status). Deterministic hierarchy, fail
    closed on ambiguity, no fuzzy matching."""
    key_base = _suffix_strip(player_clean_key)
    for tier, k in [
        ("team_exact", (team, player_clean_key)),
        ("team_base", (team, key_base)),
        ("global_exact", player_clean_key),
        ("global_base", key_base),
    ]:
        cands = idx[tier].get(k)
        if not cands:
            continue
        if len(cands) > 1:
            return "", "AMBIGUOUS_IDENTITY"
        return next(iter(cands)), "RESOLVED_GSIS"
    return "", "UNRESOLVED_IDENTITY"


def grade(season: int, weeks: list[int], detail_out: Path | None = None) -> dict:
    board = load_boards(season, weeks)
    bets = select_model_bet(board)
    bets["team"] = bets["team"].map(canon_team)
    actual = load_actual_stats_unfiltered(season, weeks)
    roster = load_roster_identity(season, weeks)
    snaps = load_snap_participation(season, weeks)
    idx = build_alias_index(actual, roster)
    roster_confirmed = set(zip(roster["season"], roster["week"], roster["team"], roster["gsis_id"]))
    roster_status = _status_map(roster)
    snap_exact, snap_base = _snap_participation_sets(snaps)

    resolved = []
    for r in bets.itertuples(index=False):
        gsis, status = resolve_gsis(r.player_clean_key, r.team, idx)
        resolved.append({"gsis_id": gsis, "identity_status": status})
    res_df = pd.concat([bets.reset_index(drop=True), pd.DataFrame(resolved)], axis=1)
    res_df["roster_confirmed_this_team_week"] = [
        (s, w, t, g) in roster_confirmed
        for s, w, t, g in zip(res_df["season"], res_df["week"], res_df["team"], res_df["gsis_id"])
    ]
    res_df["roster_status"] = [
        roster_status.get((s, w, t, g), "")
        for s, w, t, g in zip(res_df["season"], res_df["week"], res_df["team"], res_df["gsis_id"])
    ]
    res_df["snap_participated"] = [
        (
            (int(s), int(w), t, k) in snap_exact
            or (int(s), int(w), t, _suffix_strip(k)) in snap_base
        )
        for s, w, t, k in zip(
            res_df["season"], res_df["week"], res_df["team"], res_df["player_clean_key"]
        )
    ]

    detail_parts = []
    for market, stat_col in [
        ("pass_yards", "pass_yards"), ("rush_yards", "rush_yards"),
        ("rec_yards", "rec_yards"), ("receptions", "receptions"),
    ]:
        m = res_df.loc[res_df["market"].eq(market)].copy()
        if m.empty:
            continue
        a = actual[["season", "week", "gsis_id", stat_col]].rename(columns={stat_col: "actual"})
        d = m.merge(a, on=["season", "week", "gsis_id"], how="left", validate="many_to_one")
        detail_parts.append(d)
    rr = res_df.loc[res_df["market"].eq("rush_rec_yards")].copy()
    if len(rr):
        a = actual[["season", "week", "gsis_id", "rec_yards", "rush_yards"]].copy()
        a["actual"] = a["rec_yards"] + a["rush_yards"]
        d = rr.merge(a[["season", "week", "gsis_id", "actual"]], on=["season", "week", "gsis_id"], how="left", validate="many_to_one")
        detail_parts.append(d)
    detail = pd.concat(detail_parts, ignore_index=True, sort=False) if detail_parts else pd.DataFrame()

    # Position is identity, not outcome: it classifies who the bet was on so
    # the board can be read per position. It never feeds pricing or selection.
    pos_map: dict[str, str] = {}
    for source in (roster, actual):
        for gid, pos in zip(source["gsis_id"], source["position"]):
            if str(pos).strip():
                pos_map[gid] = str(pos).strip().upper()
    detail["position"] = detail["gsis_id"].map(pos_map).fillna("UNKNOWN")

    detail = apply_postgame_settlement(detail)
    unresolved = detail.loc[detail["settlement_status"].eq("UNRESOLVED")]
    graded = detail.loc[~detail["settlement_status"].eq("UNRESOLVED")].copy()

    graded["vegas_line"] = num(graded["vegas_line"])
    graded["model_proj"] = num(graded["model_proj"])
    graded["model_error"] = graded["model_proj"] - graded["actual"]
    graded["vegas_error"] = graded["vegas_line"] - graded["actual"]
    graded["model_closer_than_vegas"] = graded["model_error"].abs() < graded["vegas_error"].abs()
    graded["actual_side"] = [
        outcome_side(a, l) if pd.notna(a) else "VOID"
        for a, l in zip(graded["actual"], graded["vegas_line"])
    ]
    graded["bet_result"] = np.select(
        [
            graded["settlement_status"].eq("VOID"),
            graded["actual_side"].eq("PUSH"),
            graded["side"].astype(str).str.upper().eq(graded["actual_side"]),
        ],
        ["VOID", "PUSH", "WIN"],
        default="LOSS",
    )
    graded["unit_result"] = np.where(
        graded["bet_result"].eq("WIN"), [american_profit(o) for o in graded["vegas_odds"]],
        np.where(graded["bet_result"].eq("LOSS"), -1.0, 0.0),
    )
    decided = graded.loc[graded["bet_result"].isin(["WIN", "LOSS"])]
    summary = {
        "status": "graded",
        "archived_bet_rows": int(len(detail)),
        "selected_settlement_rows": int(len(graded)),
        "verified_actual_rows": int(graded["has_verified_actual"].sum()),
        "verified_via_stats_table": int((graded["actual_source"] == "stats_table").sum()),
        "verified_via_snap_confirmed_zero": int((graded["actual_source"] == "snap_confirmed_verified_zero").sum()),
        "void_dnp_rows": int((graded["actual_source"] == "sportsbook_void_dnp").sum()),
        "still_unresolved_rows": int(len(unresolved)),
        "unresolved_identity_status_counts": unresolved["identity_status"].value_counts().to_dict(),
        "decided_bets": int(len(decided)),
        "wins": int(decided["bet_result"].eq("WIN").sum()),
        "losses": int(decided["bet_result"].eq("LOSS").sum()),
        "win_rate": float(decided["bet_result"].eq("WIN").mean()) if len(decided) else np.nan,
        "units": float(decided["unit_result"].sum()) if len(decided) else np.nan,
        "roi_per_unit": float(decided["unit_result"].mean()) if len(decided) else np.nan,
        "model_mae": float(graded["model_error"].abs().mean()),
        "vegas_mae": float(graded["vegas_error"].abs().mean()),
        "model_closer_than_vegas_rate": float(graded["model_closer_than_vegas"].mean()),
    }
    if len(unresolved):
        print("still-unresolved rows after GSIS fix + zero-usage-preserving loader:")
        cols = [c for c in ["player", "team", "opponent", "market", "identity_status"] if c in unresolved.columns]
        print(unresolved[cols].drop_duplicates().to_string(index=False))
    if detail_out is not None:
        detail_out.parent.mkdir(parents=True, exist_ok=True)
        graded.to_csv(detail_out, index=False)
        print(f"graded detail rows written: {len(graded)} -> {detail_out}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", required=True)
    ap.add_argument("--detail-out", type=Path, default=None,
                    help="optional path to write the per-bet graded detail rows")
    args = ap.parse_args()
    weeks = [int(w) for w in args.weeks.split(",") if w.strip()]
    result = grade(args.season, weeks, detail_out=args.detail_out)
    print("\n=== CORRECTED MARKET TRACK RECORD GRADING (GSIS identity + zero-usage-preserving actuals) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
