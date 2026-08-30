"""Historical input builders for the 2025 walk-forward backtest.

Player universes come from nflverse weekly roster snapshots and, when a true
week-tagged depth snapshot is available, depth-chart enrichment. Target-week box
scores are never used to decide who is in the pregame universe. Team-week
features are built from historical PBP and later lagged by
``historical_context.py`` before a prediction week.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.providers.build_schedule import build_or_get_schedule
from scripts.utils.pbp import get_pbp

OFF_POSITIONS = {"QB", "RB", "FB", "HB", "WR", "LWR", "RWR", "SWR", "TE"}
ALLOWED_ROSTER_STATUS = {"ACT", "INA"}


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _load_schedule(season: int) -> pd.DataFrame:
    """Prefer nflreadpy's maintained schedule loader; retain legacy fallback."""
    try:
        import nflreadpy as nfl

        return _to_pandas(nfl.load_schedules(int(season)))
    except Exception:
        return build_or_get_schedule(int(season))


def build_schedule_history(seasons: Iterable[int]) -> pd.DataFrame:
    rows = []
    for season in sorted(set(int(s) for s in seasons)):
        sched = _lower(_load_schedule(season))
        if "game_type" in sched.columns:
            sched = sched.loc[sched["game_type"].astype(str).str.upper().eq("REG")].copy()
        elif "season_type" in sched.columns:
            sched = sched.loc[sched["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not {"season", "week", "home_team", "away_team"}.issubset(sched.columns):
            raise RuntimeError(f"schedule {season} missing required columns")
        if "game_id" not in sched.columns:
            sched["game_id"] = ""
        for _, r in sched.iterrows():
            home, away = canon_team(r["home_team"]), canon_team(r["away_team"])
            base = {"season": int(r["season"]), "week": int(r["week"]), "game_id": str(r.get("game_id", "") or "")}
            rows.append({**base, "team": home, "opponent": away, "home_away": "home"})
            rows.append({**base, "team": away, "opponent": home, "home_away": "away"})
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("historical schedule builder produced zero rows")
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("historical schedule has duplicate season/week/team rows")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def build_team_weekly_from_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    """Build long-form team-week observations from completed historical games.

    ``qb_dropback`` is a dropback opportunity, not an official pass attempt.
    nflverse PBP's ``pass_attempt`` indicator includes sacks, so official NFL pass
    attempts are represented here by ``pass_attempt == 1`` with ``sack != 1``.
    Scrambles are already excluded from ``pass_attempt`` but remain dropbacks.
    We preserve the team's dropback share for game-script modeling and separately
    record ``pass_attempts_per_dropback`` so the historical MC can convert the
    projected dropback workload into official passing attempts without leakage.

    These rows are observations, not pregame features. The historical-context
    layer is responsible for using only rows strictly before each target week.
    """
    all_rows = []
    for season in sorted(set(int(s) for s in seasons)):
        x = _lower(get_pbp(season, min_rows=1))
        if "season_type" in x.columns:
            reg = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                x = reg
        if not {"week", "posteam", "defteam"}.issubset(x.columns):
            raise RuntimeError(f"PBP {season} missing week/posteam/defteam")
        x["posteam"] = x["posteam"].map(canon_team)
        x["defteam"] = x["defteam"].map(canon_team)
        for c in ("qb_dropback", "rush_attempt", "pass_attempt", "sack", "qb_hit", "success", "epa", "yards_gained", "game_seconds_remaining", "play_id"):
            x[c] = _num(x, c)
        x["official_pass_attempt"] = (
            x["pass_attempt"].fillna(0).eq(1) & ~x["sack"].fillna(0).eq(1)
        ).astype(int)
        x["off_play"] = (x["qb_dropback"].fillna(0).eq(1) | x["rush_attempt"].fillna(0).eq(1)).astype(int)
        x["pressure"] = (x["sack"].fillna(0).eq(1) | x["qb_hit"].fillna(0).eq(1)).astype(int)
        x["explosive"] = x["yards_gained"].fillna(0).ge(20).astype(int)

        league = x.loc[x["off_play"].eq(1)].groupby("week").agg(league_pass_rate=("qb_dropback", "mean"))

        for (week, team), g in x.loc[x["off_play"].eq(1) & x["posteam"].ne("")].groupby(["week", "posteam"]):
            dg = x.loc[x["off_play"].eq(1) & x["defteam"].eq(team) & x["week"].eq(week)]
            drop = g["qb_dropback"].fillna(0).eq(1)
            opp_drop = dg["qb_dropback"].fillna(0).eq(1)
            pass_rate = float(drop.mean()) if len(g) else np.nan
            pass_attempts_per_dropback = (
                float(g.loc[drop, "official_pass_attempt"].fillna(0).mean()) if drop.any() else np.nan
            )
            league_pass = float(league.loc[week, "league_pass_rate"]) if week in league.index else np.nan

            neutral_pace = np.nan
            if "game_id" in g.columns and g["game_seconds_remaining"].notna().any():
                dclock = g.sort_values(["game_id", "play_id"]).copy()
                dclock["prev_clock"] = dclock.groupby("game_id")["game_seconds_remaining"].shift(1)
                delta = dclock["prev_clock"] - dclock["game_seconds_remaining"]
                delta = delta.loc[(delta >= 0) & (delta < 90)]
                if not delta.empty:
                    neutral_pace = float(delta.mean())

            all_rows.append({
                "season": int(season),
                "week": int(week),
                "team": canon_team(team),
                "success_rate_off": float(g["success"].mean()) if g["success"].notna().any() else np.nan,
                "success_rate_def": float(dg["success"].mean()) if dg["success"].notna().any() else np.nan,
                "pressure_rate_allowed": float(g.loc[drop, "pressure"].mean()) if drop.any() else np.nan,
                "pressure_rate_generated": float(dg.loc[opp_drop, "pressure"].mean()) if opp_drop.any() else np.nan,
                "neutral_pace": neutral_pace,
                "plays_est": float(len(g)),
                "dropback_rate": pass_rate,
                "pass_attempts_per_dropback": pass_attempts_per_dropback,
                "proe": pass_rate - league_pass if np.isfinite(pass_rate) and np.isfinite(league_pass) else np.nan,
                "explosive_play_rate_allowed": float(dg["explosive"].mean()) if len(dg) else np.nan,
                "def_pass_epa": float(dg.loc[opp_drop, "epa"].mean()) if opp_drop.any() else np.nan,
                "def_rush_epa": float(dg.loc[dg["rush_attempt"].fillna(0).eq(1), "epa"].mean()) if dg["rush_attempt"].fillna(0).eq(1).any() else np.nan,
            })
    out = pd.DataFrame(all_rows)
    if out.empty:
        raise RuntimeError("historical team-week PBP builder produced zero rows")
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("historical team-week features contain duplicates")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def _load_nflreadpy_weekly_sources(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    rosters = _lower(_to_pandas(nfl.load_rosters_weekly(int(season))))
    try:
        depth = _lower(_to_pandas(nfl.load_depth_charts(int(season))))
    except Exception:
        depth = pd.DataFrame()
    return rosters, depth


def _weekly_depth_lookup(depth_charts: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Return a leakage-safe depth lookup only when source is explicitly week-tagged.

    New nflverse depth-chart releases can be date-based rather than week-based.
    Without a reliable pregame date cutoff in this builder, those rows are not
    merged: using an arbitrary/full-season date snapshot could leak future depth
    information. Weekly roster data remains sufficient to define the universe.
    """
    empty = pd.DataFrame(columns=["team", "player_join", "depth_position", "depth_team"])
    d = _lower(depth_charts)
    if d.empty or not {"season", "week"}.issubset(d.columns):
        return empty

    team_col = "club_code" if "club_code" in d.columns else "team" if "team" in d.columns else None
    dname_col = "full_name" if "full_name" in d.columns else "football_name" if "football_name" in d.columns else None
    if not team_col or not dname_col:
        return empty

    d = d.loc[
        pd.to_numeric(d["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(d["week"], errors="coerce").eq(int(week))
    ].copy()
    if d.empty:
        return empty

    d["team"] = d[team_col].map(canon_team)
    d["player_join"] = d[dname_col].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    if "depth_position" not in d.columns:
        d["depth_position"] = d["position"] if "position" in d.columns else ""
    if "depth_team" not in d.columns:
        d["depth_team"] = ""
    return d[["team", "player_join", "depth_position", "depth_team"]].drop_duplicates(["team", "player_join"])


def build_pregame_universe_for_week(
    *, season: int, week: int, schedule_history: pd.DataFrame,
    rosters_weekly: pd.DataFrame, depth_charts: pd.DataFrame,
) -> pd.DataFrame:
    """Create one historical pregame player universe without target-week results."""
    sched = _lower(schedule_history)
    sched = sched.loc[
        pd.to_numeric(sched["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(sched["week"], errors="coerce").eq(int(week))
    ]
    if sched.empty:
        raise RuntimeError(f"schedule has no rows for {season} W{week}")
    matchup = {canon_team(r["team"]): (canon_team(r["opponent"]), str(r.get("game_id", "") or "")) for _, r in sched.iterrows()}

    r = _lower(rosters_weekly)
    if not {"season", "week", "team", "position"}.issubset(r.columns):
        raise RuntimeError("weekly rosters missing season/week/team/position")
    name_col = "full_name" if "full_name" in r.columns else "football_name" if "football_name" in r.columns else None
    if name_col is None:
        raise RuntimeError("weekly rosters missing full_name/football_name")
    r = r.loc[
        pd.to_numeric(r["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(r["week"], errors="coerce").eq(int(week))
    ].copy()
    r["team"] = r["team"].map(canon_team)
    r["position"] = r["position"].astype(str).str.upper().str.strip()
    r = r.loc[r["team"].isin(matchup) & r["position"].isin(OFF_POSITIONS)]
    if "status" in r.columns:
        keep = r["status"].astype(str).str.upper().isin(ALLOWED_ROSTER_STATUS)
        if keep.any():
            r = r.loc[keep]

    r["player"] = r[name_col].astype(str).str.strip()
    r["player_join"] = r["player"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    depth_lookup = _weekly_depth_lookup(depth_charts, int(season), int(week))
    if not depth_lookup.empty:
        r = r.merge(depth_lookup, on=["team", "player_join"], how="left", validate="many_to_one")
        source_name = "nflverse_weekly_roster+week_tagged_depth_chart"
    else:
        r["depth_position"] = ""
        r["depth_team"] = ""
        source_name = "nflverse_weekly_roster"

    rows = []
    for _, p in r.iterrows():
        opp, game_id = matchup[p["team"]]
        role = str(p.get("depth_position", "") or "")
        if pd.notna(p.get("depth_team")) and str(p.get("depth_team", "")).strip():
            rank = str(p.get("depth_team")).strip()
            role = f"{role}{rank}" if role else f"DEPTH{rank}"
        rows.append({
            "player": p["player"], "team": p["team"], "opponent": opp,
            "position": p["position"], "role": role, "game_id": game_id,
            "season": int(season), "week": int(week), "pregame_source": source_name,
        })
    out = pd.DataFrame(rows).drop_duplicates(["team", "player"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"pregame universe produced zero players for {season} W{week}")
    return out


def build_all_historical_inputs(
    *, season: int = 2025, prior_season: int = 2024,
    out_dir: Path = Path("data/backtests"), weeks: Iterable[int] = range(1, 19),
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    universe_dir = out_dir / "pregame_universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    seasons = [int(prior_season), int(season)]
    schedule = build_schedule_history(seasons)
    schedule_path = out_dir / "schedule_history.csv"
    schedule.to_csv(schedule_path, index=False)

    team_weekly = build_team_weekly_from_pbp(seasons)
    team_path = out_dir / "team_weekly_history.csv"
    team_weekly.to_csv(team_path, index=False)

    rosters, depth = _load_nflreadpy_weekly_sources(int(season))
    for week in sorted(set(int(w) for w in weeks)):
        u = build_pregame_universe_for_week(
            season=season, week=week, schedule_history=schedule,
            rosters_weekly=rosters, depth_charts=depth,
        )
        u.to_csv(universe_dir / f"{season}_week_{week:02d}.csv", index=False)

    return {"schedule": schedule_path, "team_weekly": team_path, "universe_dir": universe_dir}
