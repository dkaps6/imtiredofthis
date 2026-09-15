#!/usr/bin/env python3
"""Corrected actual-outcome grading for the market track record.

Fixes two real bugs found grading the real Week 1 2026 board against
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
    for suf in ("jr", "sr", "ii", "iii", "iv", "v"):
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
    x = x.loc[x["team"].astype(str).ne("") & x["gsis_id"].astype(str).ne("")].copy()
    keep = ["season", "week", "team", "gsis_id", "player", "player_clean_key",
            "receptions", "rec_yards", "rush_yards", "pass_yards"]
    out = x[keep].drop_duplicates(["season", "week", "gsis_id"], keep="last")
    return out


def build_alias_index(actual: pd.DataFrame) -> dict:
    """team-scoped exact/suffix-stripped alias -> set of GSIS ids, plus a
    global (any-team) index for the fallback tiers. Ambiguous buckets (more
    than one GSIS under the same alias) are kept as-is so the resolver can
    fail closed on them."""
    idx = {"team_exact": {}, "team_base": {}, "global_exact": {}, "global_base": {}}
    for r in actual.itertuples(index=False):
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


def grade(season: int, weeks: list[int]) -> dict:
    board = load_boards(season, weeks)
    bets = select_model_bet(board)
    bets["team"] = bets["team"].map(canon_team)
    actual = load_actual_stats_unfiltered(season, weeks)
    idx = build_alias_index(actual)

    resolved = []
    for r in bets.itertuples(index=False):
        gsis, status = resolve_gsis(r.player_clean_key, r.team, idx)
        resolved.append({"gsis_id": gsis, "identity_status": status})
    res_df = pd.concat([bets.reset_index(drop=True), pd.DataFrame(resolved)], axis=1)

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

    detail["has_verified_actual"] = detail["identity_status"].eq("RESOLVED_GSIS") & detail["actual"].notna()
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
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", required=True)
    args = ap.parse_args()
    weeks = [int(w) for w in args.weeks.split(",") if w.strip()]
    result = grade(args.season, weeks)
    print("\n=== CORRECTED MARKET TRACK RECORD GRADING (GSIS identity + zero-usage-preserving actuals) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
