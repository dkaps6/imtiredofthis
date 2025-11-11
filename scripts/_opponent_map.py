# scripts/_opponent_map.py
import logging
import os
from typing import Iterable, Optional, Union

from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("opponent_map")
CENTRAL_TZ = ZoneInfo("America/Chicago")

# --- Team name/abbr normalization used by fetchers and mappers ---

# Books sometimes use non-standard three-letter codes.
TEAM_ABBR_NORMALIZER = {
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    # defensively support a few common alternates we’ve seen
    "JAC": "JAX", "LA": "LAR", "WSH": "WAS",
    "LVG": "LV",  "ARZ": "ARI", "TAM": "TB",
}


def normalize_team_abbr(s: str) -> str:
    """Uppercase and map any odd sportsbook abbreviations to model abbrs."""
    if not s:
        return s
    s = s.strip().upper()
    return TEAM_ABBR_NORMALIZER.get(s, s)


# Normalize many site/book/team variants into model's canonical 2–3 letter codes.
TEAM_ALIASES = {
    # user-specified weird site codes
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    # common alternates
    "ARZ": "ARI",
    "JAC": "JAX",
    "JAX": "JAX",
    "LA": "LAR",
    "STL": "LAR",
    "OAK": "LV",
    "WSH": "WAS",
    "WFT": "WAS",
    # Expanded aliases from legacy pipelines
    "BAL RAVENS": "BAL",
    "CLEVELAND BROWNS": "CLE",
    "HOUSTON TEXANS": "HOU",
    "COMMANDERS": "WAS",
    "NEP": "NE",
    "N.E.": "NE",
    "GNB": "GB",
    "G.B.": "GB",
    "SFO": "SF",
    "S.F.": "SF",
    "KCC": "KC",
    "K.C.": "KC",
    "SD": "LAC",
    "S.D.": "LAC",
    "LA CHARGERS": "LAC",
    "LA RAMS": "LAR",
    "LAR": "LAR",
    "N.O.": "NO",
    "NOR": "NO",
    "NOS": "NO",
    "T.B.": "TB",
    "TAM": "TB",
    "N.Y. JETS": "NYJ",
    "NY JETS": "NYJ",
    "N.Y. GIANTS": "NYG",
    "NY GIANTS": "NYG",
}


def _canon_team_str(s: str) -> str | None:
    if not s:
        return None
    t = str(s).strip().upper()
    if t == "" or t in {"NA", "NONE"}:
        return None
    t = TEAM_ALIASES.get(t, t)
    if t in {"NAN", "<NA>"}:
        return None
    return TEAM_ALIASES.get(t, t)


def normalize_team(
    x: Union[pd.Series, list, tuple, str, None]
) -> Union[pd.Series, str, None]:
    """
    Vector-safe normalizer. If x is a Series/array-like, return a Series with
    alias mapping + trim + upper. If x is a scalar, do the same and return str|None.
    """

    if isinstance(x, pd.Series):
        series = x.astype("string").fillna("").map(_canon_team_str)
        return series.astype("string")
    if isinstance(x, (list, tuple)):
        s = pd.Series(list(x), dtype="string").fillna("").map(_canon_team_str)
        return s.astype("string")
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, pd.api.extensions.ExtensionArray):
        s = pd.Series(x).astype("string").fillna("").map(_canon_team_str)
        return s.astype("string")
    if pd.isna(x):  # type: ignore[arg-type]
        return None
    return _canon_team_str(x)


def map_normalize_team(series: pd.Series) -> pd.Series:
    """Backwards-compat wrapper: expects a Series, returns Series."""

    normalized = normalize_team(series)
    if isinstance(normalized, pd.Series):
        return normalized.astype("string")
    return pd.Series(normalized, index=series.index, dtype="string")


def map_normalize_opponent_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any opponent/team columns we use in props/opponent mapping.
    Expected optional columns: 'team', 'team_abbr', 'opponent', 'opponent_abbr', 'home_team', 'away_team'
    """

    out = df.copy()
    for col in (
        "team",
        "team_abbr",
        "opponent",
        "opponent_abbr",
        "home_team",
        "away_team",
    ):
        if col in out.columns:
            out[col] = map_normalize_team(out[col].astype("string"))
    return out

CANON_SET = set([
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
    "IND","JAX","KC","LV","LAC","LA","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
])


def normalize_team_series(vals: Union[pd.Series, Iterable]) -> pd.Series:
    if isinstance(vals, pd.Series):
        return map_normalize_team(vals)
    return map_normalize_team(pd.Series(list(vals), dtype="string"))


def _build_from_odds_game(paths: Optional[list[str]] = None) -> tuple[pd.DataFrame, list[dict]]:
    if paths is None:
        paths = ["outputs/odds_game.csv", "data/odds_game.csv"]

    frame = pd.DataFrame()
    for path in paths:
        if os.path.exists(path):
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    break
            except Exception as exc:
                logger.warning("[opponent_map] failed reading %s: %s", path, exc)
    if frame.empty:
        return pd.DataFrame(columns=["event_id", "team", "opponent", "game_date", "origin"]), []

    unresolved: list[dict] = []
    rows: list[dict] = []
    frame.columns = [c.lower() for c in frame.columns]
    for record in frame.to_dict(orient="records"):
        event_id = str(record.get("event_id", "")).strip()
        home = normalize_team(record.get("home_team"))
        away = normalize_team(record.get("away_team"))
        commence = record.get("commence_time") or record.get("kickoff_ts")
        game_date = None
        if commence:
            dt = pd.to_datetime(commence, utc=True, errors="coerce")
            if not pd.isna(dt):
                dt = dt.tz_convert(CENTRAL_TZ) if dt.tzinfo else dt.tz_localize("UTC").tz_convert(CENTRAL_TZ)
                game_date = dt.date().isoformat()
        if not event_id:
            unresolved.append({"source": "odds_game", "reason": "missing event_id", "home": home, "away": away})
            continue
        if not home or not away:
            unresolved.append({"source": "odds_game", "event_id": event_id, "reason": "team not recognized"})
            continue
        rows.append({
            "event_id": event_id,
            "team": home,
            "opponent": away,
            "game_date": game_date,
            "origin": "odds_game",
        })
        rows.append({
            "event_id": event_id,
            "team": away,
            "opponent": home,
            "game_date": game_date,
            "origin": "odds_game",
        })
    return pd.DataFrame(rows), unresolved


def _build_from_team_week_map(
    path: str = "data/team_week_map.csv",
    *,
    week: Optional[int] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    if not os.path.exists(path):
        logger.warning("[opponent_map] %s not found", path)
        return pd.DataFrame(columns=["event_id", "team", "opponent", "season", "week", "origin"]), [
            {"source": "team_week_map", "reason": "file missing"}
        ]

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        logger.error("[opponent_map] failed to read %s: %s", path, exc)
        return pd.DataFrame(columns=["event_id", "team", "opponent", "season", "week", "origin"]), [
            {"source": "team_week_map", "reason": f"read error: {exc}"}
        ]

    if frame.empty:
        return pd.DataFrame(columns=["event_id", "team", "opponent", "season", "week", "origin"]), []

    unresolved: list[dict] = []
    working = frame.copy()
    if week is not None and "week" in working.columns:
        working = working[working["week"] == week]

    for col in ("team", "opponent"):
        if col in working.columns:
            working[col] = normalize_team_series(working[col])

    rows: list[dict] = []
    for record in working.to_dict(orient="records"):
        team = record.get("team")
        opponent = record.get("opponent")
        if not team or team == "BYE":
            continue
        if not opponent or opponent == "BYE":
            unresolved.append({
                "source": "team_week_map",
                "team": team,
                "week": record.get("week"),
                "reason": "opponent missing",
            })
            continue
        rows.append({
            "event_id": str(record.get("event_id", "")).strip(),
            "team": team,
            "opponent": opponent,
            "season": record.get("season"),
            "week": record.get("week"),
            "origin": "team_week_map",
        })
    return pd.DataFrame(rows), unresolved

def build_opponent_map(week: Optional[int] = 10, team_map_path: str = "data/team_week_map.csv") -> pd.DataFrame:
    """Merge opponent mapping data from odds_game and team_week_map sources."""

    odds_df, odds_unresolved = _build_from_odds_game()
    tw_df, tw_unresolved = _build_from_team_week_map(team_map_path, week=week)

    combined = pd.concat([odds_df, tw_df], ignore_index=True, sort=False)
    if combined.empty:
        result = pd.DataFrame(
            columns=["event_id", "team", "opponent", "game_date", "season", "week", "origin"]
        )
    else:
        combined["event_id"] = combined.get("event_id", "").fillna("").astype(str)
        combined["team"] = normalize_team_series(combined.get("team", ""))
        combined["opponent"] = normalize_team_series(combined.get("opponent", ""))
        combined["_priority"] = combined.get("origin").map({"odds_game": 0, "team_week_map": 1}).fillna(1)
        combined["_dedupe_key"] = combined.apply(
            lambda row: (
                row.get("event_id", ""),
                row.get("week"),
                row.get("team", ""),
            ),
            axis=1,
        )
        combined = (
            combined.sort_values("_priority")
            .drop_duplicates(subset="_dedupe_key", keep="first")
            .drop(columns=["_dedupe_key", "_priority"], errors="ignore")
        )
        result_cols = [
            "event_id",
            "team",
            "opponent",
            "game_date",
            "season",
            "week",
            "origin",
        ]
        for col in result_cols:
            if col not in combined.columns:
                combined[col] = pd.NA
        result = combined[result_cols]

    unresolved = odds_unresolved + tw_unresolved
    if unresolved:
        unresolved_df = pd.DataFrame(unresolved)
    else:
        unresolved_df = pd.DataFrame(columns=["source", "reason"])

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "opponent_map_from_props.csv")
    result.to_csv(out_path, index=False)
    logger.info("[OpponentMap] wrote %s rows → %s", len(result), out_path)

    unresolved_path = os.path.join("data", "opponent_map_unresolved.csv")
    unresolved_df.to_csv(unresolved_path, index=False)
    if not unresolved_df.empty:
        logger.info(
            "[OpponentMap] unresolved entries=%s → %s",
            len(unresolved_df),
            unresolved_path,
        )

    return result

def attach_opponent(
    df: pd.DataFrame,
    team_col: str = "team",
    coverage_path: str = "data/team_week_map.csv",
    opponent_col: str = "opponent",
    inplace: bool = True,
    week: Optional[int] = None,
) -> pd.DataFrame:
    """
    Attach opponent by (season, week, team) using team_week_map.csv;
    fall back to any pre-built data/opponent_map_from_props.csv when available.
    """
    if df is None or df.empty:
        return df
    target = df if inplace else df.copy()
    if team_col not in target.columns:
        return target

    target[team_col] = normalize_team_series(target[team_col])
    if opponent_col not in target.columns:
        target[opponent_col] = pd.NA

    sched = build_opponent_map(week=week, team_map_path=coverage_path)
    if not sched.empty:
        join_cols = []
        for col in ("season","week","team"):
            if col in target.columns and col in sched.columns:
                join_cols.append(col)
        # event_id optional improvement
        extra = [c for c in ("event_id",) if c in target.columns and c in sched.columns]
        sel = list(set(join_cols + extra + ["opponent"]))
        over = sched[sel].drop_duplicates()
        over = over.rename(columns={"opponent": "__schedule_opp"})
        target = target.merge(over, on=join_cols + extra, how="left")
        if "__schedule_opp" in target.columns:
            target[opponent_col] = target[opponent_col].fillna(target["__schedule_opp"])
            target.drop(columns=["__schedule_opp"], inplace=True)

    # If props-built map exists, allow it to fill remaining holes.
    props_path = "data/opponent_map_from_props.csv"
    if os.path.exists(props_path):
        try:
            om = pd.read_csv(props_path)
        except Exception:
            om = pd.DataFrame()
        if not om.empty:
            for c in ("team","opponent"):
                if c in om.columns:
                    om[c] = normalize_team_series(om[c])
            if "team" in target.columns and "team" in om.columns:
                over2 = om[["team","opponent"]].dropna().drop_duplicates().rename(columns={"opponent":"__props_opp"})
                target = target.merge(over2, on=["team"], how="left")
                if "__props_opp" in target.columns:
                    target[opponent_col] = target[opponent_col].fillna(target["__props_opp"])
                    target.drop(columns=["__props_opp"], inplace=True)

    # Log any unresolved mappings for debugging
    miss = target[target[opponent_col].isna()].copy()
    if not miss.empty:
        os.makedirs("data/_debug", exist_ok=True)
        miss.to_csv("data/_debug/opponent_map_unresolved.csv", index=False)
        print(f"[OpponentMap] WARNING unresolved opponent rows: {len(miss)} → data/_debug/opponent_map_unresolved.csv")

    return target

# --- Legacy API shims for backward compatibility with existing scripts ---
# Some scripts still import: map_normalize_team, team_map
# Keep these thin wrappers so we don’t have to touch the callers.

def map_normalize_team(x):
    """Legacy alias → normalize a single team token to canonical code."""
    return normalize_team(x)


def map_normalize_team_series(vals):
    """Legacy alias → normalize a pandas Series/iterable of team tokens."""
    return normalize_team_series(vals)


def dump_norm_debug(df: pd.DataFrame, path: str) -> None:
    try:
        sel = df.copy()
        for col in ("team", "opponent", "team_abbr", "opponent_abbr"):
            if col in sel.columns:
                sel[col] = map_normalize_team(sel[col].astype("string"))
        sel.head(200).to_csv(path, index=False)
    except Exception:
        pass


def team_map(week: int = 10, team_map_path: str = "data/team_week_map.csv"):
    """Legacy alias → returns the schedule/opponent map for a given week."""
    return build_opponent_map(week=week, team_map_path=team_map_path)

# Be explicit about public surface (helps static tools/linters).
__all__ = [
    "TEAM_ALIASES", "CANON_SET",
    "normalize_team", "normalize_team_series",
    "map_normalize_team", "map_normalize_team_series",
    "map_normalize_opponent_map", "dump_norm_debug",
    "build_opponent_map", "attach_opponent",
    # legacy:
    "team_map",
]
