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

3. A player who was genuinely inactive/DNP that week has NO row at all in
   nflreadpy's weekly stats table (not even a zero row), so bug #2's fix
   alone can't identity-resolve them -- their name never appears in a
   stats-table-derived alias index (found: Odell Beckham Jr., Calvin
   Ridley, Jalen Tolbert). The alias index is extended with weekly roster
   data (which does have them, including Tolbert's explicit `INA` status),
   so identity can resolve from roster presence alone; once resolved, a
   confirmed-rostered GSIS with no stats-table row is graded as a verified
   zero for every stat column, since a real inactive player is a real,
   gradable 0-yard/0-reception outcome, not missing data.

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
    idx = build_alias_index(actual, roster)
    roster_confirmed = set(zip(roster["season"], roster["week"], roster["team"], roster["gsis_id"]))

    resolved = []
    for r in bets.itertuples(index=False):
        gsis, status = resolve_gsis(r.player_clean_key, r.team, idx)
        resolved.append({"gsis_id": gsis, "identity_status": status})
    res_df = pd.concat([bets.reset_index(drop=True), pd.DataFrame(resolved)], axis=1)
    res_df["roster_confirmed_this_team_week"] = [
        (s, w, t, g) in roster_confirmed
        for s, w, t, g in zip(res_df["season"], res_df["week"], res_df["team"], res_df["gsis_id"])
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

    # A resolved GSIS with a real stats-table row: use it. A resolved GSIS
    # confirmed on that team's roster that week but absent from the stats
    # table: a genuine inactive/zero-involvement outcome, so it's a real
    # verified zero, not missing data -- fill it in rather than drop it.
    resolved_ok = detail["identity_status"].eq("RESOLVED_GSIS")
    has_stat_row = detail["actual"].notna()
    verified_zero = resolved_ok & ~has_stat_row & detail["roster_confirmed_this_team_week"]
    detail.loc[verified_zero, "actual"] = 0.0
    detail["actual_source"] = np.select(
        [resolved_ok & has_stat_row, verified_zero],
        ["stats_table", "roster_confirmed_verified_zero"], default="unresolved",
    )
    detail["has_verified_actual"] = detail["actual_source"].ne("unresolved")
    unresolved = detail.loc[~detail["has_verified_actual"]]
    graded = detail.loc[detail["has_verified_actual"]].copy()

    graded["vegas_line"] = num(graded["vegas_line"])
    graded["model_proj"] = num(graded["model_proj"])
    graded["model_error"] = graded["model_proj"] - graded["actual"]
    graded["vegas_error"] = graded["vegas_line"] - graded["actual"]
    graded["model_closer_than_vegas"] = graded["model_error"].abs() < graded["vegas_error"].abs()
    graded["actual_side"] = [outcome_side(a, l) for a, l in zip(graded["actual"], graded["vegas_line"])]
    graded["bet_result"] = np.select(
        [graded["actual_side"].eq("PUSH"), graded["side"].astype(str).str.upper().eq(graded["actual_side"])],
        ["PUSH", "WIN"], default="LOSS",
    )
    graded["unit_result"] = np.where(
        graded["bet_result"].eq("WIN"), [american_profit(o) for o in graded["vegas_odds"]],
        np.where(graded["bet_result"].eq("LOSS"), -1.0, 0.0),
    )
    decided = graded.loc[graded["bet_result"].isin(["WIN", "LOSS"])]
    summary = {
        "status": "graded",
        "archived_bet_rows": int(len(detail)),
        "verified_actual_rows": int(len(graded)),
        "verified_via_stats_table": int((graded["actual_source"] == "stats_table").sum()),
        "verified_via_roster_confirmed_zero": int((graded["actual_source"] == "roster_confirmed_verified_zero").sum()),
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
