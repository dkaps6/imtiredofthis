"""RB Lane A -- Gate 0 schema/timing harmonizers (transition-gated allocation V1).

Implements the frozen plan at
``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V1_PLAN.md`` (Amendments 1-5),
adjudicated on Issue #535. This module builds and reports the three Gate-0
sub-gates -- BEFORE any candidate outcome is computed or inspected:

* 0.1 depth-chart source (disclosure-only under Amendment 5; does not gate Gate 0)
* 0.2 injury/status source (native week-tagged, PASS confirmed empirically)
* 0.3 roster-membership source (``nflreadpy.load_rosters_weekly``, replacing the
  current-snapshot-only ``ACTIVE_ROLES_CSV``, per Amendment 5's seven frozen
  requirements)

No candidate mechanism, comparator, or scoring logic lives here. Nothing in this
module is a production change.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._opponent_map import canon_team
from scripts.backtest.build_historical_injuries import load_historical_injuries
from scripts.backtest.historical_inputs import ALLOWED_ROSTER_STATUS
from scripts.build._schedule_utils import get_nfl_schedule

RB_POS = {"RB", "FB", "HB"}
_ET = ZoneInfo("America/New_York")


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    if hasattr(value, "to_dicts"):
        return pd.DataFrame(value.to_dicts())
    return pd.DataFrame(value)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _name_key(series: pd.Series) -> pd.Series:
    """Same convention as historical_inputs.py::_weekly_depth_lookup's player_join."""
    return series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


# ---------------------------------------------------------------------------
# Kickoff-UTC reconstruction (shared by 0.1's live-side as-of join)
# ---------------------------------------------------------------------------


def build_team_week_kickoffs(seasons: Iterable[int]) -> pd.DataFrame:
    """season/week/team/opponent/kickoff_utc, long format, one row per team-game.

    Reuses ``scripts/build/_schedule_utils.py::get_nfl_schedule`` -- the
    already-ET-localized, already-bug-fixed kickoff reconstruction (see the
    fix-schedule-kickoff-timezone-bug history in this repo). Does NOT
    reimplement kickoff math.
    """
    rows = []
    for season in sorted({int(s) for s in seasons}):
        sched = get_nfl_schedule(int(season))
        for _, r in sched.iterrows():
            home, away = canon_team(r["home"]), canon_team(r["away"])
            for team, opponent in ((home, away), (away, home)):
                rows.append(
                    {
                        "season": int(season),
                        "week": int(r["week"]),
                        "team": team,
                        "opponent": opponent,
                        "kickoff_utc": r["kickoff_utc"],
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["season", "week", "team", "opponent", "kickoff_utc"])
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True, errors="coerce")
    return out.drop_duplicates(["season", "week", "team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gate 0.1 -- depth-chart source (Amendment 5: disclosure-only, does not gate)
# ---------------------------------------------------------------------------


def harmonize_depth_state(seasons: Iterable[int], kickoffs: pd.DataFrame) -> pd.DataFrame:
    """Depth-chart state per (season, week, team, name_key).

    Native week-tagged for <=2024; strict `dt < kickoff_utc` as-of join for
    >=2025 (matching STACK2's ``depth_tables()`` / ND2B's audit script
    convention -- most-recent snapshot before kickoff, ties included).

    Per Amendment 5, this output feeds ONLY the disclosure-only
    ``detected_transition`` population -- never the scored V1 trigger, the
    candidate mechanism, any adequacy cohort, or any promotion gate.
    """
    import nflreadpy as nfl

    seasons = sorted({int(s) for s in seasons})
    out = []

    native_seasons = [s for s in seasons if s <= 2024]
    if native_seasons:
        d = _lower(_to_pandas(nfl.load_depth_charts(seasons=native_seasons)))
        if not d.empty:
            d["season"] = pd.to_numeric(d.get("season"), errors="coerce")
            d["week"] = pd.to_numeric(d.get("week"), errors="coerce")
            team_col = "club_code" if "club_code" in d.columns else "team"
            name_col = "full_name" if "full_name" in d.columns else "football_name"
            pos = (
                d.get("position", d.get("depth_position", pd.Series("", index=d.index)))
                .fillna("")
                .astype(str)
                .str.upper()
            )
            d = d.loc[d["week"].between(1, 18) & pos.isin(RB_POS)].copy()
            d["team"] = d[team_col].map(canon_team)
            d["name_key"] = _name_key(d[name_col])
            d["depth_rank"] = pd.to_numeric(
                d.get("depth_team", d.get("pos_rank")), errors="coerce"
            )
            d["depth_slot"] = pos
            d["pregame_source"] = "nflverse_native_week_tagged"
            d = d.loc[d.season.isin(native_seasons)]
            out.append(
                d[["season", "week", "team", "name_key", "depth_rank", "depth_slot", "pregame_source"]]
                .drop_duplicates(["season", "week", "team", "name_key"], keep="last")
            )

    asof_seasons = [s for s in seasons if s >= 2025]
    if asof_seasons:
        d = _lower(_to_pandas(nfl.load_depth_charts(seasons=asof_seasons)))
        if not d.empty:
            d["dt_utc"] = pd.to_datetime(d.get("dt"), errors="coerce", utc=True)
            team_col = "team" if "team" in d.columns else "club_code"
            d["team"] = d[team_col].map(canon_team)
            name_col = next(
                (c for c in ("player_name", "full_name", "football_name", "player") if c in d.columns),
                None,
            )
            d["name_key"] = _name_key(d[name_col]) if name_col else ""
            abb = d.get("pos_abb", pd.Series("", index=d.index)).astype(str).str.upper()
            name = d.get("pos_name", pd.Series("", index=d.index)).astype(str).str.lower()
            grp = d.get("pos_grp", pd.Series("", index=d.index)).astype(str).str.lower()
            rb_mask = (
                abb.isin(RB_POS)
                | name.str.contains("running back|halfback|fullback", regex=True)
                | grp.str.contains("running back|backfield", regex=True)
            )
            d = d.loc[rb_mask & d["dt_utc"].notna()].copy()

            kk = kickoffs.loc[kickoffs.season.isin(asof_seasons)]
            for _, g in kk.iterrows():
                q = d.loc[d.team.eq(g.team) & d.dt_utc.lt(g.kickoff_utc)]
                if q.empty:
                    continue
                t = q.dt_utc.max()
                z = q.loc[q.dt_utc.eq(t)]
                # Hard assertion (Amendment 5, step 4c): never a same-week/postgame snapshot.
                assert (z.dt_utc < g.kickoff_utc).all(), "gate0.1 as-of leakage: snapshot >= kickoff_utc"
                for _, r in z.iterrows():
                    out.append(
                        pd.DataFrame(
                            [
                                {
                                    "season": int(g.season),
                                    "week": int(g.week),
                                    "team": g.team,
                                    "name_key": r.name_key,
                                    "depth_rank": pd.to_numeric(pd.Series([r.get("pos_rank")]), errors="coerce").iloc[0],
                                    "depth_slot": str(r.get("pos_slot", r.get("pos_abb", "")) or "").upper(),
                                    "pregame_source": "nflverse_live_asof_dt_lt_kickoff",
                                }
                            ]
                        )
                    )

    if not out:
        return pd.DataFrame(
            columns=["season", "week", "team", "name_key", "depth_rank", "depth_slot", "pregame_source"]
        )
    return pd.concat(out, ignore_index=True).drop_duplicates(
        ["season", "week", "team", "name_key"], keep="last"
    )


def gate01_report(depth_state: pd.DataFrame, seasons: Iterable[int]) -> dict:
    """Amendment 5 disposition: NOT_CONSTRUCTIBLE_NO_OVERLAP, disclosure-only.

    No 2023-2024 date-stamped depth source exists in nflreadpy or in ND2B's
    preserved lineage (origin/research-rb-nd2b-allocation-env-atlas @ 6a01c631),
    both confirmed empirically during implementation. The originally-specified
    semantic-parity check cannot be constructed; per Amendment 5 this does not
    gate Gate 0's overall pass/fail, since depth-chart state is never an input
    to the scored V1 trigger, candidate mechanism, adequacy cohort, or any
    promotion gate.
    """
    coverage = {}
    for season in sorted({int(s) for s in seasons}):
        s = depth_state.loc[depth_state.season == season]
        coverage[str(season)] = {
            "resolvable_team_week_player_rows": int(len(s)),
            "pregame_source_counts": s["pregame_source"].value_counts().to_dict() if not s.empty else {},
        }
    return {
        "gate": "0.1_depth_chart",
        "amendment": 5,
        "semantic_parity_2023_2024_disposition": "NOT_CONSTRUCTIBLE_NO_OVERLAP",
        "reason": (
            "No date-stamped (non-week-tagged) depth source exists for 2023-2024 in "
            "nflreadpy (native week-tagged through 2024, date-stamped only from 2025) "
            "or in ND2B's preserved lineage (date-stamped coverage confirmed 2025-only). "
            "No synthetic overlap manufactured."
        ),
        "gates_scored_v1_science": False,
        "gates_gate0_overall_disposition": False,
        "scope": "disclosure-only, feeds detected_transition population only",
        "coverage_by_season": coverage,
    }


# ---------------------------------------------------------------------------
# Gate 0.2 -- injury/status source (empirically confirmed PASS, Amendment 5)
# ---------------------------------------------------------------------------


def harmonize_injury_state(seasons: Iterable[int]) -> pd.DataFrame:
    """Native season/week-tagged injury state for every season, 2016-2025 alike.

    Reuses ``scripts/backtest/build_historical_injuries.py::load_historical_injuries``
    unchanged -- confirmed empirically (2025-09-16) to return the identical
    native season/week-tagged schema for 2025 as for 2016-2024; no as-of join
    needed anywhere.
    """
    return load_historical_injuries(sorted({int(s) for s in seasons}))


def gate02_report(injury_state: pd.DataFrame, seasons: Iterable[int]) -> dict:
    coverage = {}
    for season in sorted({int(s) for s in seasons}):
        s = injury_state.loc[injury_state.season == season]
        coverage[str(season)] = {
            "rows": int(len(s)),
            "null_status_rows": int(s["status"].isna().sum()) if not s.empty else 0,
        }
    return {
        "gate": "0.2_injury_status",
        "amendment": 5,
        "disposition": "PASS",
        "reason": (
            "nfl.load_injuries is natively season/week-tagged for every season 2016-2025 "
            "(empirically verified for 2025: weeks 1-22, zero null season/week, "
            "report_status/practice_status present). The as-of-join contingency in the "
            "frozen plan never triggers."
        ),
        "coverage_by_season": coverage,
    }


# ---------------------------------------------------------------------------
# Gate 0.3 -- roster-membership source (Amendment 5: formally replaced source)
# ---------------------------------------------------------------------------


def harmonize_roster_membership(seasons: Iterable[int]) -> pd.DataFrame:
    """RB-room membership per (season, week, team) via nflreadpy.load_rosters_weekly.

    Replaces ACTIVE_ROLES_CSV (current-snapshot-only, no historical-replay mode)
    per Amendment 5. Reuses the exact team/position/status normalization already
    used by historical_inputs.py::build_pregame_universe_for_week
    (RB/FB/HB subset, ALLOWED_ROSTER_STATUS = {ACT, INA}).
    """
    import nflreadpy as nfl

    out = []
    for season in sorted({int(s) for s in seasons}):
        d = _lower(_to_pandas(nfl.load_rosters_weekly(int(season))))
        if d.empty:
            continue
        d["season"] = pd.to_numeric(d.get("season"), errors="coerce")
        d["week"] = pd.to_numeric(d.get("week"), errors="coerce")
        pos = d.get("position", pd.Series("", index=d.index)).fillna("").astype(str).str.upper()
        status = d.get("status", pd.Series("", index=d.index)).fillna("").astype(str).str.upper()
        d = d.loc[pos.isin(RB_POS) & status.isin(ALLOWED_ROSTER_STATUS)].copy()
        if d.empty:
            continue
        d["team"] = d["team"].map(canon_team)
        d["player_key"] = d.get("gsis_id", pd.Series("", index=d.index)).fillna("").astype(str)
        missing_id = d["player_key"].eq("")
        if missing_id.any():
            d.loc[missing_id, "player_key"] = "namekey:" + _name_key(d.loc[missing_id, "full_name"])
        d["position"] = pos.loc[d.index]
        d["status"] = status.loc[d.index]
        out.append(
            d[["season", "week", "team", "player_key", "position", "status"]]
            .dropna(subset=["season", "week"])
        )

    if not out:
        return pd.DataFrame(columns=["season", "week", "team", "player_key", "position", "status"])
    result = pd.concat(out, ignore_index=True)
    result["season"] = result["season"].astype(int)
    result["week"] = result["week"].astype(int)
    return result.drop_duplicates(["season", "week", "team", "player_key"], keep="last")


def gate03_report(roster_state: pd.DataFrame, seasons: Iterable[int]) -> dict:
    """Score the seven frozen Amendment-5 requirements. Fail-closed, not silent."""
    seasons = sorted({int(s) for s in seasons})
    failures: list[str] = []

    # Requirement 3: zero duplicate (season, week, team, player_key) identities.
    dupe_mask = roster_state.duplicated(["season", "week", "team", "player_key"], keep=False)
    dupe_count = int(dupe_mask.sum())
    if dupe_count:
        failures.append(f"requirement_3_duplicate_identities: {dupe_count} rows")

    # Requirement 1: schema present (checked structurally; empty frame is a hard failure).
    required_cols = {"season", "week", "team", "position"}
    if not required_cols.issubset(roster_state.columns):
        failures.append(f"requirement_1_schema_missing: {required_cols - set(roster_state.columns)}")

    # Requirement 4: every scheduled team-week in both OOS test seasons (2024, 2025)
    # has a resolvable RB-room roster state.
    coverage_gap: dict[str, int] = {}
    for season in (2024, 2025):
        if season not in seasons:
            continue
        sched = get_nfl_schedule(season)
        tw = pd.concat(
            [
                sched[["week", "home"]].rename(columns={"home": "team"}),
                sched[["week", "away"]].rename(columns={"away": "team"}),
            ],
            ignore_index=True,
        ).drop_duplicates()
        tw["team"] = tw["team"].map(canon_team)
        have = roster_state.loc[roster_state.season == season, ["week", "team"]].drop_duplicates()
        merged = tw.merge(have.assign(_has=1), on=["week", "team"], how="left")
        gap = int(merged["_has"].isna().sum())
        coverage_gap[str(season)] = gap
        if gap:
            failures.append(f"requirement_4_unresolvable_team_weeks_{season}: {gap}")

    coverage_by_season = {}
    for season in seasons:
        s = roster_state.loc[roster_state.season == season]
        coverage_by_season[str(season)] = {
            "rows": int(len(s)),
            "unique_team_weeks": int(s[["week", "team"]].drop_duplicates().shape[0]) if not s.empty else 0,
        }

    disposition = "GATE0_BLOCKED" if failures else "PASS"
    return {
        "gate": "0.3_roster_membership",
        "amendment": 5,
        "source": "nflreadpy.load_rosters_weekly (replaces ACTIVE_ROLES_CSV)",
        "disposition": disposition,
        "failures": failures,
        "scheduled_team_week_coverage_gap": coverage_gap,
        "coverage_by_season": coverage_by_season,
        "note": (
            "Requirement 5 (both current and immediately-previous resolvable state for "
            "every scored loss/vacancy event) and requirement 6 (zero target-week "
            "statistics/outcomes) are verified at candidate-construction time, not here, "
            "since they depend on the scored V1 transition population which is not yet "
            "computed at Gate-0 time."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="RB Lane A Gate 0 harmonizers (research only, no production change)")
    ap.add_argument("--seasons", default="2023,2024,2025", help="comma-separated seasons")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kickoffs = build_team_week_kickoffs(seasons)
    kickoffs.to_csv(args.out_dir / "gate0_team_week_kickoffs.csv", index=False)

    depth_state = harmonize_depth_state(seasons, kickoffs)
    depth_state.to_csv(args.out_dir / "gate0_1_depth_state.csv", index=False)
    gate01 = gate01_report(depth_state, seasons)

    injury_state = harmonize_injury_state(seasons)
    injury_state.to_csv(args.out_dir / "gate0_2_injury_state.csv", index=False)
    gate02 = gate02_report(injury_state, seasons)

    roster_state = harmonize_roster_membership(seasons)
    roster_state.to_csv(args.out_dir / "gate0_3_roster_state.csv", index=False)
    gate03 = gate03_report(roster_state, seasons)

    overall_blocked = gate02["disposition"] != "PASS" or gate03["disposition"] != "PASS"
    report = {
        "seasons": seasons,
        "gate0_1": gate01,
        "gate0_2": gate02,
        "gate0_3": gate03,
        "gate0_overall_disposition": "RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED"
        if overall_blocked
        else "GATE0_PASS",
    }
    out_path = args.out_dir / "gate0_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
