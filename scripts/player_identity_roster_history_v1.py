#!/usr/bin/env python3
"""Build identity-only NFL roster history for current-slate player resolution.

This source is deliberately separated from the football modeling prior. Historical
weekly roster rows may contribute stable GSIS identity/name/team aliases, but they
never contribute targets, carries, yards, routes, scoring, or any projection
feature. That lets a returning/traded veteran retain the correct person identity
without broadening the validated PlayerForm prior window.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.player_identity_v3 import (
    attach_historical_identity,
    build_identity_registry,
    player_name_key,
)

DATA = Path("data")
AUDIT = DATA / "player_identity_roster_history_audit.json"
SNAPSHOT = DATA / "player_identity_roster_history.csv"
ALIASES = DATA / "player_identity_aliases.csv"
SKILL_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _first(frame: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series(default, index=frame.index)


def _position(value) -> str:
    text = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if text in {"HB", "TB"}:
        return "RB"
    if text.startswith("WR") or text in {"LWR", "RWR", "SWR"}:
        return "WR"
    if text.startswith("RB"):
        return "RB"
    if text.startswith("TE"):
        return "TE"
    if text.startswith("QB"):
        return "QB"
    if text.startswith("FB"):
        return "FB"
    return text


def _load_one(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = _to_pandas(nfl.load_rosters_weekly(int(season)))
    if raw.empty:
        raise RuntimeError(f"weekly roster identity source returned zero rows for season={season}")
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = pd.to_numeric(_first(x, ["season"], season), errors="coerce").fillna(season).astype(int)
    x["week"] = pd.to_numeric(_first(x, ["week"]), errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].between(1, 22, inclusive="both")].copy()
    if x.empty:
        raise RuntimeError(f"weekly roster identity source has no dated rows for season={season}")

    raw_name = _first(x, ["full_name", "football_name", "player_name", "player", "name"]).astype("string").fillna("").str.strip()
    canon = raw_name.map(canonicalize_player_name_safe)
    x["player"] = canon.map(lambda pair: pair[0])
    x["player_clean_key"] = canon.map(lambda pair: pair[1])
    x["team"] = _first(x, ["team", "team_abbr", "club_code"]).map(canon_team)
    x["position"] = _first(x, ["position", "pos"]).map(_position)
    x["player_id"] = _first(x, ["gsis_id", "player_id"]).astype("string").fillna("").str.strip()

    # Only stable GSIS-backed skill identities are useful here. Rows without IDs
    # cannot improve identity certainty and must never manufacture a person key.
    x = x.loc[
        x["position"].isin(SKILL_POSITIONS)
        & x["player"].astype(str).str.len().gt(0)
        & x["team"].astype(str).str.len().gt(0)
        & x["player_id"].astype(str).str.len().gt(0)
    ].copy()
    if x.empty:
        raise RuntimeError(f"weekly roster identity source has zero stable skill identities for season={season}")

    x = attach_historical_identity(x, id_col="player_id", name_col="player", team_col="team")
    return x[[
        "season", "week", "team", "position", "player", "player_clean_key",
        "player_id", "player_identity_key", "identity_full_name_key", "identity_base_name_key",
    ]].drop_duplicates(["season", "week", "team", "player_identity_key"], keep="last")


def _load_current_depth_charts(season: int) -> tuple[pd.DataFrame, dict]:
    """Current-season depth-chart identity: who is on which team RIGHT NOW.

    Weekly rosters are the historical identity spine, but they lag an in-season
    signing and they carry the player under his RELEASING club. The depth chart
    carries him under his new one, with the same stable GSIS id, which is the
    only source that can resolve a mid-season move.

    The defect this closes: Josh Johnson (QB, gsis 00-0026300, 18 seasons) was
    cut by Cincinnati and signed by Arizona on 2026-11-10. His 2021-2025 weekly
    roster teams are BAL/SF/WAS, never ARI, so the team-scoped lookup missed, the
    resolver fell through to the global name, and it found two people -- him and
    a 2021-2022 Seattle RB of the same name who has not been on a roster since.
    resolve_slate_identities then correctly refused to guess, and every Full
    Slate run from 2026-11-10 died at PlayerForm on "Ambiguous player identity:
    team=ARI player=Josh Johnson". nflverse's own depth chart resolves the
    Arizona player to 00-0026300 unambiguously.

    Identity metadata only, exactly as this module's contract requires: gsis id,
    team, name and position. No targets, carries, yards, routes or scoring, and
    stamped week 0 like a verified alias so it can never be read as game
    evidence or widen the model prior.

    Only UNAMBIGUOUS facts are added. A (team, name) that the depth chart itself
    maps to more than one GSIS id is dropped rather than inserted, so this source
    can resolve an ambiguity but never manufacture one.
    """
    import nflreadpy as nfl

    stats = {"available": False, "rows": 0, "teams": 0, "dropped_ambiguous": 0}
    try:
        raw = _to_pandas(nfl.load_depth_charts(seasons=[int(season)]))
    except Exception as exc:  # pragma: no cover - provider/network dependent
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), stats
    if raw.empty:
        stats["error"] = "provider returned zero rows"
        return pd.DataFrame(), stats

    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["team"] = _first(x, ["team", "team_abbr", "club_code"]).map(canon_team)
    x["player_id"] = _first(x, ["gsis_id", "player_id"]).astype("string").fillna("").str.strip()
    x["position"] = _first(x, ["pos_abb", "position", "pos_name", "pos"]).map(_position)
    raw_name = _first(x, ["player_name", "full_name", "football_name", "player", "name"]).astype("string").fillna("").str.strip()
    canon = raw_name.map(canonicalize_player_name_safe)
    x["player"] = canon.map(lambda pair: pair[0])
    x["player_clean_key"] = canon.map(lambda pair: pair[1])

    x = x.loc[
        x["position"].isin(SKILL_POSITIONS)
        & x["player"].astype(str).str.len().gt(0)
        & x["team"].astype(str).str.len().gt(0)
        & x["player_id"].astype(str).str.len().gt(0)
    ].copy()
    if x.empty:
        stats["error"] = "no stable skill identities in depth charts"
        return pd.DataFrame(), stats

    # Keep only the most recent snapshot per team. The provider accumulates every
    # published chart for the season, so without this the frame is dominated by
    # stale team placements that would reintroduce the very ambiguity this fixes.
    if "dt" in x.columns:
        order = pd.to_datetime(x["dt"], errors="coerce", utc=True)
        x = x.assign(_dt=order).dropna(subset=["_dt"])
        if x.empty:
            stats["error"] = "depth charts carry no parseable dt"
            return pd.DataFrame(), stats
        latest = x.groupby("team")["_dt"].transform("max")
        x = x.loc[x["_dt"].eq(latest)].drop(columns=["_dt", "dt"])

    x = x.drop_duplicates(["team", "player_id"], keep="last")

    # Never manufacture an ambiguity: drop any (team, name) the chart maps to more
    # than one person.
    name_key = x["player"].map(lambda v: player_name_key(v))
    grp = x.assign(_k=name_key).groupby(["team", "_k"])["player_id"].transform("nunique")
    ambiguous = grp.gt(1)
    stats["dropped_ambiguous"] = int(ambiguous.sum())
    x = x.loc[~ambiguous].copy()
    if x.empty:
        stats["error"] = "every depth-chart identity was ambiguous"
        return pd.DataFrame(), stats

    x["season"] = int(season)
    x["week"] = 0
    x = attach_historical_identity(x, id_col="player_id", name_col="player", team_col="team")
    out = x[[
        "season", "week", "team", "position", "player", "player_clean_key",
        "player_id", "player_identity_key", "identity_full_name_key", "identity_base_name_key",
    ]].drop_duplicates(["season", "team", "player_identity_key"], keep="last")
    stats.update(available=True, rows=int(len(out)), teams=int(out["team"].nunique()))
    return out, stats


def _apply_verified_aliases(history: pd.DataFrame, current_season: int) -> tuple[pd.DataFrame, int]:
    """Append current identity aliases only after verifying their historical GSIS anchor.

    Alias rows contain person identity metadata only. They are deliberately stamped
    as current-season week 0 and never carry usage/performance fields, so they can
    change name/team resolution without broadening the model prior.
    """
    if not ALIASES.exists() or ALIASES.stat().st_size == 0:
        return history, 0
    aliases = pd.read_csv(ALIASES, dtype="string").fillna("")
    required = {
        "current_name", "historical_name", "player_id", "current_team", "position",
        "reason", "verified_source", "verified_date",
    }
    missing = required - set(aliases.columns)
    if missing:
        raise RuntimeError(f"identity alias config missing columns: {sorted(missing)}")
    if aliases.empty:
        return history, 0
    if aliases[list(required)].apply(lambda s: s.str.strip().eq("")).any().any():
        raise RuntimeError("identity alias config contains blank required values")
    if aliases["player_id"].duplicated().any() or aliases["current_name"].map(
        lambda v: player_name_key(v, strip_suffix=True)
    ).duplicated().any():
        raise RuntimeError("identity alias config contains duplicate player IDs or current names")

    rows: list[pd.DataFrame] = []
    for rec in aliases.to_dict("records"):
        pid = str(rec["player_id"]).strip()
        current_name = str(rec["current_name"]).strip()
        historical_name = str(rec["historical_name"]).strip()
        team = canon_team(rec["current_team"])
        position = _position(rec["position"])
        source_rows = history.loc[history["player_id"].astype(str).eq(pid)].copy()
        if source_rows.empty:
            raise RuntimeError(f"verified identity alias player_id not found in historical roster source: {pid}")
        hist_base = player_name_key(historical_name, strip_suffix=True)
        source_bases = set(source_rows["player"].map(lambda v: player_name_key(v, strip_suffix=True)))
        if hist_base not in source_bases:
            raise RuntimeError(
                f"verified identity alias historical name does not match GSIS source: "
                f"player_id={pid} historical_name={historical_name} source_names={sorted(source_rows['player'].astype(str).unique())[:10]}"
            )
        source_positions = set(source_rows["position"].map(_position))
        if position not in source_positions:
            raise RuntimeError(
                f"verified identity alias position mismatch player_id={pid} alias={position} source={sorted(source_positions)}"
            )
        canon_name, clean_key = canonicalize_player_name_safe(current_name)
        alias = pd.DataFrame([{
            "season": int(current_season),
            "week": 0,
            "team": team,
            "position": position,
            "player": canon_name,
            "player_clean_key": clean_key,
            "player_id": pid,
        }])
        alias = attach_historical_identity(alias, id_col="player_id", name_col="player", team_col="team")
        if str(alias.iloc[0]["player_identity_key"]) != f"gsis:{pid}":
            raise RuntimeError(f"verified identity alias failed GSIS round trip: {current_name} -> {pid}")
        rows.append(alias[[
            "season", "week", "team", "position", "player", "player_clean_key",
            "player_id", "player_identity_key", "identity_full_name_key", "identity_base_name_key",
        ]])

    out = pd.concat([history, *rows], ignore_index=True, sort=False)
    return out, len(rows)


def load_identity_roster_history(seasons: Iterable[int]) -> pd.DataFrame:
    requested = sorted({int(s) for s in seasons})
    if not requested:
        raise RuntimeError("identity roster history requires at least one historical season")
    frames: list[pd.DataFrame] = []
    season_rows: dict[str, int] = {}
    for season in requested:
        frame = _load_one(season)
        frames.append(frame)
        season_rows[str(season)] = int(len(frame))
    out = pd.concat(frames, ignore_index=True, sort=False)
    if out.empty:
        raise RuntimeError("identity roster history produced zero rows")

    current_season = max(requested) + 1

    # Current-season team placement. This is an identity ENRICHMENT: without it
    # the registry resolves exactly as it did before, so a provider failure is
    # visible in the audit rather than fatal. With it, an in-season signing whose
    # name collides with another person resolves instead of killing the slate.
    depth, depth_stats = _load_current_depth_charts(current_season)
    if not depth.empty:
        out = pd.concat([out, depth], ignore_index=True, sort=False)

    out, alias_rows = _apply_verified_aliases(out, current_season)

    # Stable ID must remain one person identity. Multiple historical/current teams
    # are expected and are exactly what allows offseason-trade resolution.
    collisions = out.groupby("player_id")["player_identity_key"].nunique()
    bad = collisions.loc[collisions.gt(1)]
    if not bad.empty:
        raise RuntimeError(f"identity roster history has GSIS collisions: {bad.head(20).to_dict()}")

    DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(SNAPSHOT, index=False)
    registry = build_identity_registry(out)
    result = {
        "disposition": "IDENTITY_ONLY_ROSTER_HISTORY_READY",
        "seasons": requested,
        "rows": int(len(out)),
        "stable_player_ids": int(out["player_id"].nunique()),
        "registry_rows": int(len(registry)),
        "season_rows": season_rows,
        "verified_alias_rows": int(alias_rows),
        "current_depth_chart_identity": depth_stats,
        "model_feature_columns_supplied": [],
        "identity_only": True,
        "snapshot": str(SNAPSHOT),
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[identity_roster_history] " + json.dumps(result, sort_keys=True))
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, required=True)
    args = parser.parse_args()
    load_identity_roster_history(args.seasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
