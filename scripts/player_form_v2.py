#!/usr/bin/env python3
"""Authoritative PlayerForm v2 with Player Identity v3.

Design goals:
- player_game_logs.csv is a true season/week/player/team observation table;
- nflverse/GSIS player IDs are the primary historical identity when available;
- names remain provider aliases/display fields, never the primary history join;
- current slate identity/opponent comes from the current props + schedule layer;
- previous-season data is an explicit prior, not something that leaks into the
  active season by accident;
- route_rate and YPRR are only populated when a real routes field exists.
  Targets/dropbacks are never mislabeled as routes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_prior_season, resolve_season, resolve_slate_date, resolve_week
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.player_identity_v3 import (
    attach_historical_identity,
    build_identity_registry,
    resolve_slate_identities,
)

DATA = Path("data")
OUTPUTS = Path("outputs")
ROLES = DATA / "roles_ourlads.csv"
TEAM_WEEK_MAP = DATA / "team_week_map.csv"
OPPONENT_MAP = DATA / "opponent_map_from_props.csv"
PROPS = OUTPUTS / "props_raw.csv"


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _load_weekly(season: int) -> pd.DataFrame:
    errors: list[str] = []
    try:
        import nflreadpy as nflv
        raw = nflv.load_player_stats(seasons=[int(season)], summary_level="week")
        df = _to_pandas(raw)
        if not df.empty:
            return df
    except Exception as exc:
        errors.append(f"nflreadpy: {exc}")
    try:
        import nfl_data_py as nfl
        df = nfl.import_weekly_data([int(season)], downcast=True)
        if df is not None and not df.empty:
            return _to_pandas(df)
    except Exception as exc:
        errors.append(f"nfl_data_py: {exc}")
    raise RuntimeError(f"Unable to load weekly player stats for {season}: {' | '.join(errors)}")


def _canon_name(value) -> tuple[str, str]:
    try:
        name, key = canonicalize_player_name_safe(value)
    except Exception:
        name, key = "", ""
    raw = "" if value is None else str(value).strip()
    name = (name or raw).strip()
    key = (key or "").strip()
    if not key:
        key = "".join(ch.lower() for ch in name if ch.isalnum())
    return name, key


def _first(frame: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    for c in candidates:
        if c in frame.columns:
            return frame[c]
    return pd.Series(default, index=frame.index)


def _num(frame: pd.DataFrame, candidates: Iterable[str], default=0.0) -> pd.Series:
    return pd.to_numeric(_first(frame, candidates, default), errors="coerce").fillna(default)


def _normalize_weekly(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "season" in x.columns:
        s = pd.to_numeric(x["season"], errors="coerce")
        x = x.loc[s.eq(int(season))].copy()
    x["season"] = int(season)
    x["week"] = pd.to_numeric(_first(x, ["week"]), errors="coerce").astype("Int64")
    x = x.loc[x["week"].notna()].copy()

    x["player_id"] = (
        _first(x, ["player_id", "gsis_id", "player_gsis_id"], "")
        .astype("string").fillna("").str.strip()
    )
    raw_name = (
        _first(x, ["player_display_name", "player_name", "display_name", "name"], "")
        .astype("string").fillna("").str.strip()
    )
    canon = raw_name.map(_canon_name)
    x["player"] = canon.map(lambda t: t[0])
    x["player_clean_key"] = canon.map(lambda t: t[1])
    team_raw = _first(x, ["recent_team", "team", "posteam"], "").astype("string").fillna("").str.strip()
    x["team"] = team_raw.map(canon_team)
    x["position"] = _first(x, ["position", "position_group", "pos"], pd.NA).astype("string").str.upper().str.strip()

    # Stable ID becomes the primary historical identity.  The name keys remain
    # available for provider bridging and audit only.
    x = attach_historical_identity(x, id_col="player_id", name_col="player", team_col="team")

    x["targets"] = _num(x, ["targets"])
    x["receptions"] = _num(x, ["receptions"])
    x["rec_yards"] = _num(x, ["receiving_yards", "rec_yards"])
    x["rushes"] = _num(x, ["rushing_attempts", "carries", "rush_att"])
    x["rush_yards"] = _num(x, ["rushing_yards", "rush_yards"])
    x["pass_att"] = _num(x, ["attempts", "passing_attempts", "pass_attempts"])
    x["pass_yards"] = _num(x, ["passing_yards", "pass_yards"])

    # Only honor a route metric if the source actually supplies routes.
    if "routes" in x.columns:
        x["routes"] = pd.to_numeric(x["routes"], errors="coerce")
    elif "routes_run" in x.columns:
        x["routes"] = pd.to_numeric(x["routes_run"], errors="coerce")
    else:
        x["routes"] = np.nan

    usage = x[["targets", "rushes", "pass_att"]].sum(axis=1)
    x = x.loc[
        (usage > 0)
        & x["player_identity_key"].astype(str).ne("")
        & x["team"].astype(str).ne("")
    ].copy()

    team_keys = ["season", "week", "team"]
    den = x.groupby(team_keys, dropna=False).agg(
        team_targets=("targets", "sum"),
        team_rushes=("rushes", "sum"),
        team_dropbacks=("pass_att", "sum"),
        team_routes=("routes", "sum"),
    ).reset_index()
    x = x.merge(den, on=team_keys, how="left")
    x["tgt_share_game"] = np.where(x["team_targets"] > 0, x["targets"] / x["team_targets"], np.nan)
    x["rush_share_game"] = np.where(x["team_rushes"] > 0, x["rushes"] / x["team_rushes"], np.nan)
    x["route_rate_game"] = np.where(x["team_routes"] > 0, x["routes"] / x["team_routes"], np.nan)
    x["ypt_game"] = np.where(x["targets"] > 0, x["rec_yards"] / x["targets"], np.nan)
    x["ypc_game"] = np.where(x["rushes"] > 0, x["rush_yards"] / x["rushes"], np.nan)
    x["ypa_game"] = np.where(x["pass_att"] > 0, x["pass_yards"] / x["pass_att"], np.nan)
    x["catch_rate_game"] = np.where(x["targets"] > 0, x["receptions"] / x["targets"], np.nan)
    x["yprr_game"] = np.where(x["routes"] > 0, x["rec_yards"] / x["routes"], np.nan)
    return x


def _load_schedule() -> pd.DataFrame:
    if not TEAM_WEEK_MAP.exists() or TEAM_WEEK_MAP.stat().st_size == 0:
        raise RuntimeError("data/team_week_map.csv is required")
    s = pd.read_csv(TEAM_WEEK_MAP)
    s.columns = [str(c).lower() for c in s.columns]
    for c in ("season", "week"):
        s[c] = pd.to_numeric(s[c], errors="coerce").astype("Int64")
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    if "game_id" not in s.columns:
        s["game_id"] = pd.NA
    return s[["season", "week", "team", "opponent", "game_id"]].drop_duplicates()


def _attach_schedule(logs: pd.DataFrame) -> pd.DataFrame:
    sched = _load_schedule()
    out = logs.merge(sched, on=["season", "week", "team"], how="left")
    if out["game_id"].isna().any():
        missing = out["game_id"].isna()
        out.loc[missing, "game_id"] = (
            out.loc[missing, "season"].astype("Int64").astype(str)
            + "_" + out.loc[missing, "week"].astype("Int64").astype(str).str.zfill(2)
            + "_" + out.loc[missing, "team"].astype(str)
            + "_" + out.loc[missing, "opponent"].astype(str)
        )
    return out


def _season_totals(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame()
    if "player_identity_key" not in logs.columns:
        raise RuntimeError("PlayerForm totals require player_identity_key")

    # A player keeps the same history through spelling changes, suffix variance,
    # and trades because GSIS identity—not name—is the grouping grain.
    g = logs.groupby(["season", "player_identity_key"], dropna=False)
    totals = g.agg(
        player_id=("player_id", "last"),
        player=("player", "last"),
        player_clean_key=("player_clean_key", "last"),
        identity_full_name_key=("identity_full_name_key", "last"),
        identity_base_name_key=("identity_base_name_key", "last"),
        historical_team=("team", "last"),
        historical_position=("position", "last"),
        games=("week", "nunique"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        rec_yards=("rec_yards", "sum"),
        rushes=("rushes", "sum"),
        rush_yards=("rush_yards", "sum"),
        pass_att=("pass_att", "sum"),
        pass_yards=("pass_yards", "sum"),
        routes=("routes", lambda s: s.sum(min_count=1)),
        team_targets=("team_targets", "sum"),
        team_rushes=("team_rushes", "sum"),
        team_dropbacks=("team_dropbacks", "sum"),
        team_routes=("team_routes", lambda s: s.sum(min_count=1)),
    ).reset_index()
    totals["tgt_share"] = np.where(totals["team_targets"] > 0, totals["targets"] / totals["team_targets"], np.nan)
    totals["rush_share"] = np.where(totals["team_rushes"] > 0, totals["rushes"] / totals["team_rushes"], np.nan)
    totals["route_rate"] = np.where(totals["team_routes"] > 0, totals["routes"] / totals["team_routes"], np.nan)
    totals["ypt"] = np.where(totals["targets"] > 0, totals["rec_yards"] / totals["targets"], np.nan)
    totals["ypc"] = np.where(totals["rushes"] > 0, totals["rush_yards"] / totals["rushes"], np.nan)
    totals["ypa"] = np.where(totals["pass_att"] > 0, totals["pass_yards"] / totals["pass_att"], np.nan)
    totals["receptions_per_target"] = np.where(totals["targets"] > 0, totals["receptions"] / totals["targets"], np.nan)
    totals["yprr"] = np.where(totals["routes"] > 0, totals["rec_yards"] / totals["routes"], np.nan)
    return totals


def _load_roles() -> pd.DataFrame:
    if not ROLES.exists() or ROLES.stat().st_size == 0:
        raise RuntimeError("data/roles_ourlads.csv is required")
    r = pd.read_csv(ROLES)
    r.columns = [str(c).lower() for c in r.columns]
    if not {"player", "team"}.issubset(r.columns):
        raise RuntimeError("roles_ourlads.csv missing player/team")
    r["team"] = r["team"].map(canon_team)
    canon = r["player"].map(_canon_name)
    r["display_name"] = canon.map(lambda t: t[0])
    r["player_clean_key"] = canon.map(lambda t: t[1])
    for c in ("position", "role"):
        if c not in r.columns:
            r[c] = pd.NA
    order = [c for c in ("team", "player_clean_key", "display_name", "position", "role") if c in r.columns]
    return r[order].drop_duplicates(["team", "player_clean_key"])


def _load_slate_universe(season: int, week: int) -> pd.DataFrame:
    roles = _load_roles()
    base = pd.DataFrame()
    if PROPS.exists() and PROPS.stat().st_size > 0:
        p = pd.read_csv(PROPS)
        p.columns = [str(c).lower() for c in p.columns]
        if "player" in p.columns:
            canon = p["player"].map(_canon_name)
            p["player"] = canon.map(lambda t: t[0])
            p["player_clean_key"] = canon.map(lambda t: t[1])
            team = _first(p, ["team_abbr", "team", "player_team_abbr"], "")
            p["team"] = team.map(canon_team)
            opp = _first(p, ["opponent_abbr", "opponent", "opponent_team_abbr"], "")
            p["opponent"] = opp.map(canon_team)
            base = p[["player", "player_clean_key", "team", "opponent"]].drop_duplicates()
    if base.empty:
        base = roles.rename(columns={"display_name": "player"})[["player", "player_clean_key", "team"]].copy()
        base["opponent"] = pd.NA

    # Authoritative schedule fills/overrides missing opponent.
    sched = _load_schedule()
    cur = sched.loc[
        (sched["season"] == int(season)) & (sched["week"] == int(week)),
        ["team", "opponent"],
    ].drop_duplicates("team")
    base = base.merge(cur.rename(columns={"opponent": "schedule_opponent"}), on="team", how="left")
    base["opponent"] = base["opponent"].replace("", pd.NA).combine_first(base["schedule_opponent"])
    base.drop(columns=["schedule_opponent"], inplace=True)
    base = base.merge(roles, on=["team", "player_clean_key"], how="left", suffixes=("", "_role"))
    base["player"] = base["player"].replace("", pd.NA).combine_first(base.get("display_name"))
    base["season"] = int(season)
    base["week"] = int(week)
    return base.drop_duplicates(["team", "player_clean_key"])


def _attach_roles_by_identity(universe: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Repair role joins that name-only matching cannot safely make.

    Ourlads intentionally drops suffixes such as Jr/II/III.  Resolve the Ourlads
    row against stable history, then merge role/position on player_identity_key.
    """
    roles = _load_roles().rename(columns={"display_name": "role_display_name"})
    if roles.empty:
        return universe
    role_names = roles.rename(columns={"role_display_name": "player"})
    resolved = resolve_slate_identities(
        role_names,
        registry,
        name_col="player",
        team_col="team",
        strict_ambiguous=True,
        allow_temporary=True,
    )
    keep = [c for c in ("team", "player_identity_key", "role", "position", "player") if c in resolved.columns]
    role_map = resolved[keep].copy().rename(
        columns={"role": "identity_role", "position": "identity_position", "player": "identity_role_player"}
    )
    role_map = role_map.sort_values(["team", "player_identity_key"]).drop_duplicates(
        ["team", "player_identity_key"], keep="first"
    )
    out = universe.merge(role_map, on=["team", "player_identity_key"], how="left", validate="many_to_one")
    if "role" not in out.columns:
        out["role"] = out["identity_role"]
    else:
        out["role"] = out["role"].replace("", pd.NA).combine_first(out["identity_role"])
    if "position" not in out.columns:
        out["position"] = out["identity_position"]
    else:
        out["position"] = out["position"].replace("", pd.NA).combine_first(out["identity_position"])
    out.drop(columns=["identity_role", "identity_position", "identity_role_player"], inplace=True, errors="ignore")
    return out


def _dedupe_resolved_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """Collapse harmless alias duplicates while rejecting conflicting events."""
    if universe.empty:
        return universe
    key = ["team", "player_identity_key"]
    dup = universe.duplicated(key, keep=False)
    if not dup.any():
        return universe
    for _, part in universe.loc[dup].groupby(key, dropna=False):
        if "opponent" in part.columns and part["opponent"].dropna().astype(str).nunique() > 1:
            raise RuntimeError(f"Resolved player identity has conflicting opponents: {part.to_dict('records')[:10]}")
        if "event_id" in part.columns and part["event_id"].dropna().astype(str).nunique() > 1:
            raise RuntimeError(f"Resolved player identity has conflicting event IDs: {part.to_dict('records')[:10]}")
    out = universe.copy()
    out["_identity_rank"] = pd.to_numeric(out.get("identity_confidence"), errors="coerce").fillna(0.0)
    out = out.sort_values(["_identity_rank"], ascending=False, kind="mergesort")
    out = out.drop_duplicates(key, keep="first").drop(columns=["_identity_rank"])
    return out


def _blend(prior: pd.DataFrame, current: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["tgt_share", "rush_share", "route_rate", "yprr", "ypt", "ypc", "ypa", "receptions_per_target"]
    join_key = "player_identity_key"
    if join_key not in universe.columns:
        raise RuntimeError("PlayerForm universe must be resolved to player_identity_key before blending")

    p = (
        prior[[join_key, "games", *metric_cols]].copy()
        if not prior.empty
        else pd.DataFrame(columns=[join_key, "games", *metric_cols])
    )
    c = (
        current[[join_key, "games", *metric_cols]].copy()
        if not current.empty
        else pd.DataFrame(columns=[join_key, "games", *metric_cols])
    )
    p = p.rename(columns={"games": "prior_games", **{m: f"{m}_prior" for m in metric_cols}})
    c = c.rename(columns={"games": "current_games", **{m: f"{m}_current" for m in metric_cols}})
    out = universe.merge(p, on=join_key, how="left", validate="many_to_one").merge(
        c, on=join_key, how="left", validate="many_to_one"
    )
    out["prior_games"] = pd.to_numeric(out.get("prior_games"), errors="coerce").fillna(0)
    out["current_games"] = pd.to_numeric(out.get("current_games"), errors="coerce").fillna(0)
    # Four games of prior-season equivalent pseudo-sample gives meaningful Week 1
    # priors while allowing current evidence to take over quickly.
    w_cur = out["current_games"] / (out["current_games"] + 4.0)
    for m in metric_cols:
        pv = pd.to_numeric(out.get(f"{m}_prior"), errors="coerce")
        cv = pd.to_numeric(out.get(f"{m}_current"), errors="coerce")
        both = pv.notna() & cv.notna()
        value = pv.copy()
        value.loc[cv.notna() & pv.isna()] = cv.loc[cv.notna() & pv.isna()]
        value.loc[both] = (1.0 - w_cur.loc[both]) * pv.loc[both] + w_cur.loc[both] * cv.loc[both]
        out[m] = value
    out["target_share"] = out["tgt_share"]
    out["yprr_proxy"] = out["yprr"]
    out["rz_share"] = np.nan
    out["rz_tgt_share"] = np.nan
    out["rz_rush_share"] = np.nan
    out["rz_carry_share"] = np.nan
    return out


def build(season: int, prior_season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for yr in (prior_season, season):
        try:
            raw = _load_weekly(yr)
            logs = _attach_schedule(_normalize_weekly(raw, yr))
            frames.append(logs)
            print(f"[player_form_v2] weekly season={yr} rows={len(logs)}")
        except Exception as exc:
            if yr == prior_season:
                raise
            print(f"[player_form_v2] current-season weekly unavailable for {yr}: {exc}")
    all_logs = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if all_logs.empty:
        raise RuntimeError("No historical player game logs available")

    # No same-week leakage: active-season evidence must precede the slate week.
    prior_logs = all_logs.loc[all_logs["season"].eq(int(prior_season))].copy()
    current_logs = all_logs.loc[
        all_logs["season"].eq(int(season)) & all_logs["week"].lt(int(week))
    ].copy()
    prior_totals = _season_totals(prior_logs)
    current_totals = _season_totals(current_logs)

    # Identity registry itself is built only from information available before the
    # target week.  Week 1 rookies therefore remain explicit temporary identities
    # until a maintained source exposes their stable NFL ID.
    eligible_identity_logs = pd.concat([prior_logs, current_logs], ignore_index=True, sort=False)
    registry = build_identity_registry(eligible_identity_logs)

    universe = _load_slate_universe(season, week)
    universe = resolve_slate_identities(
        universe,
        registry,
        name_col="player",
        team_col="team",
        strict_ambiguous=True,
        allow_temporary=True,
    )
    universe = _attach_roles_by_identity(universe, registry)
    universe = _dedupe_resolved_universe(universe)

    form = _blend(prior_totals, current_totals, universe)
    if form.empty:
        raise RuntimeError("PlayerForm v2 produced 0 slate players")

    # Preserve current roster role/position; historical position only fills gaps.
    hist_pos = (
        prior_totals[["player_identity_key", "historical_position"]]
        .drop_duplicates("player_identity_key")
        if not prior_totals.empty
        else pd.DataFrame()
    )
    if not hist_pos.empty:
        form = form.merge(hist_pos, on="player_identity_key", how="left", validate="many_to_one")
        if "position" not in form.columns:
            form["position"] = form["historical_position"]
        else:
            form["position"] = form["position"].replace("", pd.NA).combine_first(form["historical_position"])
    form["position"] = form.get("position", pd.Series(pd.NA, index=form.index)).astype("string").str.upper()
    form["role"] = form.get("role", pd.Series(pd.NA, index=form.index)).astype("string").str.upper()
    form["team_abbr"] = form["team"]
    form["opponent_abbr"] = form["opponent"]
    form["player_canonical"] = form["player"]
    form["display_name"] = form["player"]

    # Season totals artifact includes both prior and active histories with real season labels.
    totals = _season_totals(all_logs)
    return all_logs, totals, form


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--prior-season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    season = int(args.season if args.season is not None else resolve_season())
    prior = int(args.prior_season if args.prior_season is not None else resolve_prior_season())
    slate_date = (args.date if args.date is not None else resolve_slate_date()) or ""
    week = int(args.week if args.week is not None else resolve_week(season=season, slate_date=slate_date))
    if prior >= season:
        raise RuntimeError(f"prior_season={prior} must be earlier than season={season}")

    logs, totals, form = build(season, prior, week)
    DATA.mkdir(parents=True, exist_ok=True)
    logs.to_csv(DATA / "player_game_logs.csv", index=False)
    totals.to_csv(DATA / "player_season_totals.csv", index=False)
    form.to_csv(DATA / "player_form.csv", index=False)
    form.to_csv(DATA / "player_form_consensus.csv", index=False)

    eligible = logs.loc[
        logs["season"].eq(int(prior))
        | (logs["season"].eq(int(season)) & logs["week"].lt(int(week)))
    ].copy()
    registry = build_identity_registry(eligible)
    registry.to_csv(DATA / "player_identity_registry.csv", index=False)
    form[[c for c in (
        "player", "player_clean_key", "player_identity_key", "player_id", "team", "opponent",
        "identity_resolution", "identity_confidence", "identity_registry_player",
    ) if c in form.columns]].to_csv(DATA / "player_identity_slate.csv", index=False)

    temporary = int(form["player_identity_key"].astype(str).str.startswith("temp:").sum()) if "player_identity_key" in form.columns else 0
    print(
        f"[player_form_v2] wrote logs={len(logs)} totals={len(totals)} slate_form={len(form)} "
        f"season={season} week={week} stable_identities={len(form)-temporary} temporary_identities={temporary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
