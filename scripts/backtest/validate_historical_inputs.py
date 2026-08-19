"""Validate historical walk-forward input artifacts before model backtesting.

This is deliberately strict: malformed schedule/team history or any missing/
implausible weekly pregame universe should stop the backtest before projections
are generated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

EXPECTED_REG_WEEKS = tuple(range(1, 19))
OFF_POSITIONS = {"QB", "RB", "FB", "HB", "WR", "LWR", "RWR", "SWR", "TE"}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    if out.empty:
        raise RuntimeError(f"{label} has zero rows: {path}")
    return out


def validate_schedule(schedule: pd.DataFrame, season: int, weeks: list[int]) -> dict:
    required = {"season", "week", "team", "opponent"}
    missing = required - set(schedule.columns)
    if missing:
        raise RuntimeError(f"schedule history missing columns: {sorted(missing)}")
    s = schedule.copy()
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    if s[["team", "opponent"]].eq("").any().any():
        raise RuntimeError("schedule history contains unresolved team/opponent identity")
    if s.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("schedule history contains duplicate season/week/team rows")

    target = s.loc[s["season"].eq(int(season)) & s["week"].isin(weeks)].copy()
    if target.empty:
        raise RuntimeError(f"schedule has no rows for season {season}")
    missing_weeks = [w for w in weeks if not target["week"].eq(w).any()]
    if missing_weeks:
        raise RuntimeError(f"schedule missing requested weeks: {missing_weeks}")

    # NFL bye weeks mean scheduled-team counts vary, but every listed team must
    # have a reciprocal opponent row for the same week.
    keyed = set((int(r.season), int(r.week), str(r.team), str(r.opponent)) for r in target.itertuples())
    bad_recip = []
    for season_v, week_v, team, opp in keyed:
        if (season_v, week_v, opp, team) not in keyed:
            bad_recip.append((season_v, week_v, team, opp))
    if bad_recip:
        raise RuntimeError(f"schedule contains non-reciprocal matchup rows: {bad_recip[:5]}")

    return {
        "rows": int(len(target)),
        "teams": int(target["team"].nunique()),
        "weeks": int(target["week"].nunique()),
    }


def validate_team_history(team_weekly: pd.DataFrame, season: int, prior_season: int) -> dict:
    required = {"season", "week", "team"}
    missing = required - set(team_weekly.columns)
    if missing:
        raise RuntimeError(f"team weekly history missing columns: {sorted(missing)}")
    t = team_weekly.copy()
    t["season"] = pd.to_numeric(t["season"], errors="coerce")
    t["week"] = pd.to_numeric(t["week"], errors="coerce")
    t["team"] = t["team"].map(canon_team)
    if t["team"].eq("").any():
        raise RuntimeError("team weekly history contains unresolved team identity")
    if t.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("team weekly history contains duplicate season/week/team rows")

    counts = {}
    for s in (int(prior_season), int(season)):
        part = t.loc[t["season"].eq(s)]
        if part.empty:
            raise RuntimeError(f"team weekly history has no rows for {s}")
        if part["team"].nunique() < 30:
            raise RuntimeError(f"team weekly history has only {part['team'].nunique()} teams for {s}")
        counts[s] = {"rows": int(len(part)), "teams": int(part["team"].nunique()), "weeks": int(part["week"].nunique())}
    return counts


def validate_universe(frame: pd.DataFrame, schedule: pd.DataFrame, season: int, week: int) -> dict:
    required = {"player", "team", "opponent", "position", "season", "week"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"pregame universe {season} W{week:02d} missing columns: {sorted(missing)}")
    u = frame.copy()
    u["season"] = pd.to_numeric(u["season"], errors="coerce")
    u["week"] = pd.to_numeric(u["week"], errors="coerce")
    if not u["season"].eq(int(season)).all() or not u["week"].eq(int(week)).all():
        raise RuntimeError(f"pregame universe {season} W{week:02d} contains wrong season/week rows")
    u["team"] = u["team"].map(canon_team)
    u["opponent"] = u["opponent"].map(canon_team)
    u["position"] = u["position"].astype(str).str.upper().str.strip()
    if u[["player", "team", "opponent"]].fillna("").astype(str).eq("").any().any():
        raise RuntimeError(f"pregame universe {season} W{week:02d} has blank identity values")
    if u.duplicated(["team", "player"]).any():
        raise RuntimeError(f"pregame universe {season} W{week:02d} has duplicate player/team rows")
    bad_pos = sorted(set(u.loc[~u["position"].isin(OFF_POSITIONS), "position"].astype(str)))
    if bad_pos:
        raise RuntimeError(f"pregame universe {season} W{week:02d} has non-offensive positions: {bad_pos[:10]}")

    sched = schedule.copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    sched["team"] = sched["team"].map(canon_team)
    sched["opponent"] = sched["opponent"].map(canon_team)
    sched = sched.loc[sched["season"].eq(int(season)) & sched["week"].eq(int(week))]
    opp_map = dict(zip(sched["team"], sched["opponent"]))
    bad_matchups = u.loc[u.apply(lambda r: opp_map.get(r["team"]) != r["opponent"], axis=1), ["team", "opponent"]]
    if not bad_matchups.empty:
        raise RuntimeError(f"pregame universe {season} W{week:02d} has schedule-opponent mismatches: {bad_matchups.head().to_dict('records')}")

    scheduled_teams = set(sched["team"])
    universe_teams = set(u["team"])
    missing_teams = sorted(scheduled_teams - universe_teams)
    if missing_teams:
        raise RuntimeError(f"pregame universe {season} W{week:02d} missing scheduled teams: {missing_teams}")
    if len(u) < 300:
        raise RuntimeError(f"pregame universe {season} W{week:02d} implausibly small: {len(u)} players")

    return {
        "players": int(len(u)),
        "teams": int(u["team"].nunique()),
        "qbs": int(u["position"].eq("QB").sum()),
        "rbs": int(u["position"].isin(["RB", "FB", "HB"]).sum()),
        "wrs": int(u["position"].isin(["WR", "LWR", "RWR", "SWR"]).sum()),
        "tes": int(u["position"].eq("TE").sum()),
    }


def validate_all(*, data_dir: Path, season: int, prior_season: int, weeks: list[int]) -> pd.DataFrame:
    schedule = _read(data_dir / "schedule_history.csv", "schedule history")
    team_weekly = _read(data_dir / "team_weekly_history.csv", "team weekly history")
    schedule_summary = validate_schedule(schedule, season, weeks)
    team_summary = validate_team_history(team_weekly, season, prior_season)

    rows = []
    for week in weeks:
        path = data_dir / "pregame_universe" / f"{season}_week_{week:02d}.csv"
        universe = _read(path, f"pregame universe {season} W{week:02d}")
        summary = validate_universe(universe, schedule, season, week)
        rows.append({"season": season, "week": week, **summary})
        print(
            f"[backtest_validate] {season} W{week:02d}: "
            f"teams={summary['teams']} players={summary['players']} QB={summary['qbs']} "
            f"RB={summary['rbs']} WR={summary['wrs']} TE={summary['tes']}"
        )

    report = pd.DataFrame(rows)
    print(f"[backtest_validate] schedule rows={schedule_summary['rows']} weeks={schedule_summary['weeks']}")
    print(f"[backtest_validate] team history={team_summary}")
    print(f"[backtest_validate] PASS weeks={len(report)} player_rows={int(report['players'].sum())}")
    return report


def _parse_weeks(value: str) -> list[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/backtests"))
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--report", type=Path, default=Path("data/backtests/historical_input_validation.csv"))
    args = p.parse_args()
    report = validate_all(
        data_dir=args.data_dir,
        season=args.season,
        prior_season=args.prior_season,
        weeks=_parse_weeks(args.weeks),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    print(f"[backtest_validate] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
