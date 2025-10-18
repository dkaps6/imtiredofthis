# scripts/make_player_form.py
"""
Create ``player_form.csv``: per-player usage & efficiency metrics.

Inputs
======
* Play-by-play via ``nflreadpy`` (preferred) with automatic fallbacks to
  ``nfl_data_py`` when the lightweight mirror is unavailable.
* Optional enrichers (ESPN, MySportsFeeds, API-Sports, NFLGSIS) seeded in
  ``data/``.

Outputs
=======
* ``data/player_form.csv`` (legacy location for downstream joins)
* ``metrics/player_form.csv`` (mirrors the legacy location so newer jobs read
  from the metrics folder first)

Key features
============
* Robust handling of the nflverse helpers.  Different versions of
  ``nflreadpy`` (and its dependency ``nfl_data_py``) expose slightly different
  keyword signatures.  The loader now retries with progressively smaller
  argument sets and gracefully removes unsupported keywords so that both the
  2025 mirrors and older cached wheels succeed.
* Automatic CSV caching.  Whenever we successfully pull play-by-play or
  participation we persist copies to ``data/`` and ``external/nflverse_bundle``
  (including ``…/outputs``).  Subsequent runs read these cached mirrors so the
  pipeline can operate in offline environments such as GitHub Actions
  reruns.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

def _import_nflverse():
    try:
        import nflreadpy as nflv
        return nflv, "nflreadpy"
    except ModuleNotFoundError:
        print(
            "[make_player_form] ⚠️ nflreadpy missing; falling back to nfl_data_py. "
            "Install nflreadpy so the 2025 feeds remain available.",
            file=sys.stderr,
        )
    except ImportError as exc:
        if "original_mlq" in str(exc):
            print(
                "[make_player_form] ⚠️ nflreadpy requires nfl_data_py>=0.3.4 (missing `original_mlq`). "
                "Upgrade nfl_data_py to keep using the nflreadpy mirror. Falling back to nfl_data_py.",
                file=sys.stderr,
            )
        else:
            print(
                f"[make_player_form] ⚠️ nflreadpy import failed ({exc}); falling back to nfl_data_py.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(
            f"[make_player_form] ⚠️ nflreadpy import failed ({exc}); falling back to nfl_data_py.",
            file=sys.stderr,
        )
    import nfl_data_py as nflv  # fallback
    return nflv, "nfl_data_py"

NFLV, NFLPKG = _import_nflverse()

DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "player_form.csv")

METRICS_DIR = "metrics"
METRICS_OUTPATH = os.path.join(METRICS_DIR, "player_form.csv")

METRICS_DIR = "metrics"
METRICS_OUTPATH = os.path.join(METRICS_DIR, "player_form.csv")
_CACHE_DIRS = [
    DATA_DIR,
    os.path.join("external", "nflverse_bundle"),
    os.path.join("external", "nflverse_bundle", "outputs"),
]

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _ensure_pandas(obj) -> pd.DataFrame:
    """Coerce any tabular object returned by nflverse helpers into pandas."""

    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


def _cache_csv(kind: str, df: pd.DataFrame, season: int) -> None:
    """Persist successfully fetched tables so future runs can reuse them."""

    if df.empty:
        return
    for base in _CACHE_DIRS:
        if not base:
            continue
        try:
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, f"{kind}_{season}.csv")
            df.to_csv(path, index=False)
        except Exception:
            continue


def _call_nflverse(func, **kwargs):
    """Invoke an nflverse helper while stripping unsupported kwargs on the fly."""

    attempt = {k: v for k, v in kwargs.items() if v is not None}
    while True:
        try:
            return func(**attempt)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if match:
                bad = match.group(1)
                if bad in attempt:
                    attempt.pop(bad)
                    if attempt:
                        continue
            raise

def safe_div(n, d):
    n = n.astype(float); d = d.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, np.nan, n / d)

def _load_cached_csv(kind: str, season: int) -> Tuple[pd.DataFrame, str]:
    for base in _CACHE_DIRS:
        path = os.path.join(base, f"{kind}_{season}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        return df, path
    return pd.DataFrame(), ""


def _validate_season(df: pd.DataFrame, season: int, label: str) -> None:
    if df.empty or "season" not in df.columns:
        return
    try:
        seasons = (
            pd.to_numeric(df["season"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
    except Exception:
        return
    if len(seasons) == 0:
        return
    if any(int(s) != int(season) for s in seasons):
        raise RuntimeError(
            f"{label} spans seasons {sorted(map(int, seasons))}; expected {season}"
        )


def load_pbp(season:int)->pd.DataFrame:
    cached, source_path = _load_cached_csv("pbp", season)
    if not cached.empty:
        _validate_season(cached, season, "cached pbp")
        print(f"[make_player_form] ℹ️ Using cached pbp_{season}.csv from {source_path}")
        return cached
    df = pd.DataFrame()
    errors: list[str] = []

    if NFLPKG == "nflreadpy":
        load_fn = getattr(NFLV, "load_pbp", None)
        if load_fn is None:
            errors.append("nflreadpy.load_pbp missing")
        else:
            variants = [
                {"seasons": [season], "season_type": "REG"},
                {"seasons": [season], "season_types": ["REG"]},
                {"seasons": [season]},
            ]
            for kwargs in variants:
                try:
                    df = _ensure_pandas(_call_nflverse(load_fn, **kwargs))
                except Exception as exc:  # pragma: no cover - dependency specific
                    errors.append(str(exc))
                    continue
                if not df.empty:
                    break

    if df.empty:
        try:
            df = _ensure_pandas(
                NFLV.import_pbp_data([season], downcast=True, season_type="REG")
            )
        except TypeError as exc:
            # older nfl_data_py builds expose ``season_type`` as ``season_types``
            try:
                df = _ensure_pandas(
                    NFLV.import_pbp_data([season], downcast=True, season_types=["REG"])
                )
            except Exception:
                errors.append(str(exc))
                df = _ensure_pandas(NFLV.import_pbp_data([season], downcast=True))
        except Exception as exc:  # pragma: no cover - dependency specific
            errors.append(str(exc))

    df = _ensure_pandas(df)
    df.columns = [c.lower() for c in df.columns]
    if df.empty and errors:
        raise RuntimeError("; ".join(errors))

    _cache_csv("pbp", df, season)
    return df


def _load_required_pbp(season: int) -> tuple[pd.DataFrame, int]:
    """Load play-by-play strictly for ``season`` or raise."""
    cached, cache_path = _load_cached_csv("pbp", season)
    if not cached.empty:
        _validate_season(cached, season, "cached pbp")
        print(
            f"[make_player_form] ℹ️ Loaded cached pbp_{season}.csv from {cache_path}"
        )
        return cached, season

    errors: list[str] = []
    try:
        df = load_pbp(season)
    except Exception as err:
        errors.append(f"{type(err).__name__}: {err}")
    else:
        if not df.empty:
            _validate_season(df, season, "pbp feed")
            return df, season
        errors.append("empty dataframe")

    raise RuntimeError(
        "PBP unavailable for requested season. "
        + "; ".join(errors) if errors else ""
    )


def _load_pbp_with_optional_fallback(
    season: int, allow_fallback: bool
) -> tuple[pd.DataFrame, int]:
    """Load PBP for ``season``; optionally walk back to prior seasons."""

    if not allow_fallback:
        return _load_required_pbp(season)

    earliest_season = 1999
    candidates = list(range(season, earliest_season - 1, -1))
    errors: list[str] = []
    for candidate in candidates:
        try:
            pbp, source_season = _load_required_pbp(candidate)
        except RuntimeError as err:
            errors.append(f"{candidate}: {err}")
            continue

        if candidate != season:
            print(
                "[make_player_form] ⚠️ Requested season "
                f"{season} unavailable; falling back to {candidate}."
            )
        return pbp, source_season

    tried = ", ".join(str(s) for s in candidates)
    suffix = "; ".join(errors)
    message = f"PBP unavailable for requested season; tried {tried}."
    if suffix:
        message += f" {suffix}"
    raise RuntimeError(message)


def load_participation(season:int)->pd.DataFrame:
    cached, source_path = _load_cached_csv("participation", season)
    if not cached.empty:
        _validate_season(cached, season, "participation cache")
        print(
            f"[make_player_form] ℹ️ Using cached participation_{season}.csv from {source_path}"
        )
        return cached
    try:
        p = pd.DataFrame()
        if NFLPKG == "nflreadpy":
            part_fn = getattr(NFLV, "load_participation", None)
            if part_fn is not None:
                variants = [
                    {"seasons": [season], "season_type": "REG"},
                    {"seasons": [season], "season_types": ["REG"]},
                    {"seasons": [season]},
                ]
                for kwargs in variants:
                    try:
                        p = _ensure_pandas(_call_nflverse(part_fn, **kwargs))
                    except AttributeError:
                        break
                    except Exception:
                        continue
                    if not p.empty:
                        break
        if p.empty:
            try:
                p = _ensure_pandas(NFLV.import_participation([season]))  # type: ignore[attr-defined]
            except Exception:
                p = pd.DataFrame()
        p.columns = [c.lower() for c in p.columns]
        _validate_season(p, season, "participation feed")
        _cache_csv("participation", p, season)
        return p
    except Exception:
        return pd.DataFrame()

def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_player_usage(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute target/rush shares, RZ shares, and simple efficiency."""
    df = pbp.copy()
    df.columns = [c.lower() for c in df.columns]

    # require an offense team column
    off_col = "posteam" if "posteam" in df else ("offense_team" if "offense_team" in df else None)
    if off_col is None:
        return pd.DataFrame()

    # col variants we see in mirrors
    rec_name_col = _first_present(df, ["receiver_player_name", "receiver", "receiver_name"])
    rush_name_col = _first_present(df, ["rusher_player_name", "rusher", "rusher_name"])
    if rec_name_col is None and rush_name_col is None:
        # nothing to aggregate
        return pd.DataFrame()

    is_pass = df.get("pass", pd.Series(False, index=df.index)).astype(bool)
    is_rush = df.get("rush", pd.Series(False, index=df.index)).astype(bool)

    # Receiving: targets, RZ targets, YPT
    rec = pd.DataFrame()
    if rec_name_col is not None:
        rec = df.loc[is_pass & df[rec_name_col].notna(),
                     [off_col, rec_name_col, "yardline_100", "yards_gained"]].copy()
        rec["is_rz"] = pd.to_numeric(rec["yardline_100"], errors="coerce") <= 20
        tgt = rec.groupby([off_col, rec_name_col]).size().rename("targets")
        rz_tgt = rec.loc[rec["is_rz"]].groupby([off_col, rec_name_col]).size().rename("rz_targets")
        ypt = rec.groupby([off_col, rec_name_col])["yards_gained"].mean().rename("ypt")
        rec_tbl = pd.concat([tgt, rz_tgt, ypt], axis=1).reset_index().rename(
            columns={off_col:"team", rec_name_col:"player"}
        )
    else:
        rec_tbl = pd.DataFrame(columns=["team","player","targets","rz_targets","ypt"])

    # Rushing: attempts, RZ carries, YPC
    rush = pd.DataFrame()
    if rush_name_col is not None:
        rush = df.loc[is_rush & df[rush_name_col].notna(),
                      [off_col, rush_name_col, "yardline_100", "yards_gained"]].copy()
        rush["is_rz"] = pd.to_numeric(rush["yardline_100"], errors="coerce") <= 20
        att = rush.groupby([off_col, rush_name_col]).size().rename("rush_att")
        rz_carry = rush.loc[rush["is_rz"]].groupby([off_col, rush_name_col]).size().rename("rz_carries")
        ypc = rush.groupby([off_col, rush_name_col])["yards_gained"].mean().rename("ypc")
        rush_tbl = pd.concat([att, rz_carry, ypc], axis=1).reset_index().rename(
            columns={off_col:"team", rush_name_col:"player"}
        )
    else:
        rush_tbl = pd.DataFrame(columns=["team","player","rush_att","rz_carries","ypc"])

    # combine
    if rec_tbl.empty and rush_tbl.empty:
        return pd.DataFrame()

    if "rush_att" not in rec_tbl: rec_tbl["rush_att"] = 0
    if "targets" not in rush_tbl: rush_tbl["targets"] = 0
    base = pd.concat([rec_tbl, rush_tbl], axis=0, ignore_index=True)
    base = base.groupby(["team","player"], dropna=False).sum().reset_index()

    # shares (team denominators)
    base["target_share"]   = safe_div(base["targets"],   base.groupby("team")["targets"].transform("sum").fillna(0))
    base["rush_share"]     = safe_div(base["rush_att"],  base.groupby("team")["rush_att"].transform("sum").fillna(0))
    base["rz_tgt_share"]   = safe_div(base["rz_targets"],base.groupby("team")["rz_targets"].transform("sum").fillna(0))
    base["rz_carry_share"] = safe_div(base["rz_carries"],base.groupby("team")["rz_carries"].transform("sum").fillna(0))

    # simple efficiency
    base["yprr_proxy"] = np.where(base["targets"] > 0, base.get("ypt", np.nan), np.nan)
    base["route_rate"] = np.nan  # to be enriched later
    return base

def enrich_with_participation(base: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Add route_rate from participation if available."""
    if part.empty:
        return base
    p = part.copy()
    p.columns = [c.lower() for c in p.columns]
    team_col = "offense_team" if "offense_team" in p.columns else ("posteam" if "posteam" in p.columns else None)
    if team_col is None or "player" not in p.columns:
        return base
    if "routes_run" in p.columns and "plays" in p.columns:
        rr = (
            p.groupby([team_col, "player"])
             .apply(lambda x: safe_div(x["routes_run"].sum(), x["plays"].sum()))
             .rename("route_rate")
             .reset_index()
             .rename(columns={team_col: "team"})
        )
        base = base.merge(rr, on=["team","player"], how="left", suffixes=("","_part"))
        base["route_rate"] = base["route_rate"].combine_first(base.get("route_rate_part"))
        if "route_rate_part" in base.columns:
            base.drop(columns=["route_rate_part"], inplace=True)
    return base

def _player_enricher_paths() -> list[str]:
    """Return candidate CSV filenames that may contain player usage enrichments."""
    return [
        "espn_player_form.csv",
        "espn_player.csv",
        "msf_player_form.csv",
        "msf_player.csv",
        "apisports_player_form.csv",
        "apisports_player.csv",
        "nflgsis_player_form.csv",
        "gsis_player.csv",
        "pfr_player_enrich.csv",
    ]


_PLAYER_COLUMN_ALIASES = {
    "player_name": "player",
    "name": "player",
    "athlete": "player",
    "athlete_display_name": "player",
    "full_name": "player",
    "team_abbr": "team",
    "team_code": "team",
    "posteam": "team",
    "offense_team": "team",
    "team_name": "team",
    "target_pct": "target_share",
    "targets_share": "target_share",
    "tgt_share": "target_share",
    "rush_pct": "rush_share",
    "rushing_share": "rush_share",
    "carry_share": "rush_share",
    "routes_share": "route_rate",
    "route_pct": "route_rate",
    "routes": "route_rate",
    "routes_run_share": "route_rate",
    "yards_per_target": "ypt",
    "ypt_avg": "ypt",
    "yards_per_carry": "ypc",
    "rushing_yards_per_attempt": "ypc",
    "yprr": "yprr_proxy",
    "yards_per_route_run": "yprr_proxy",
}


def _standardize_player_enricher(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so they align with player_form expectations."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # apply alias map where canonical column missing
    for src, dest in _PLAYER_COLUMN_ALIASES.items():
        if src in df.columns and dest not in df.columns:
            df = df.rename(columns={src: dest})

    # ensure required keys exist
    if "team" not in df.columns:
        for cand in ["team", "team_abbreviation", "team_short", "club"]:
            if cand in df.columns:
                df["team"] = df[cand]
                break
    if "player" not in df.columns:
        for cand in ["player", "player_short", "player_id", "gsis_name"]:
            if cand in df.columns:
                df["player"] = df[cand]
                break

    if not {"team", "player"}.issubset(df.columns):
        return pd.DataFrame()

    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["player"] = df["player"].astype(str).str.strip()

    keep_cols = [
        c
        for c in [
            "team",
            "player",
            "target_share",
            "rush_share",
            "route_rate",
            "ypt",
            "ypc",
            "yprr_proxy",
            "rz_tgt_share",
            "rz_carry_share",
        ]
        if c in df.columns
    ]
    if not keep_cols:
        return pd.DataFrame()
    return df[keep_cols].drop_duplicates()


def fallback_from_external(base: pd.DataFrame) -> pd.DataFrame:
    """Merge optional enrichers without overwriting existing values."""
    for filename in _player_enricher_paths():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            enrich = pd.read_csv(path)
        except Exception:
            continue
        std = _standardize_player_enricher(enrich)
        if std.empty:
            continue
        base = base.merge(std, on=["team", "player"], how="left", suffixes=("", "_ext"))
        for col in [c for c in std.columns if c not in {"team", "player"}]:
            ext_col = f"{col}_ext"
            if ext_col in base.columns:
                base[col] = base[col].combine_first(base[ext_col])
                base.drop(columns=[ext_col], inplace=True)
    return base

def build_player_form(season:int, allow_fallback: bool=False)->tuple[pd.DataFrame, int]:
    print(f"[make_player_form] Loading PBP for {season} ({NFLPKG}) ...")
    pbp, source_season = _load_pbp_with_optional_fallback(season, allow_fallback)
    if pbp.empty:
        raise RuntimeError("Empty PBP.")
    base = compute_player_usage(pbp)
    part = load_participation(source_season)
    base = enrich_with_participation(base, part)
    base = fallback_from_external(base)
    base["season"] = season
    base["source_season"] = source_season
    order = ["player","team","season","source_season","target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate"]
    for c in order:
        if c not in base.columns:
            base[c] = np.nan
    base = base[order].sort_values(["team","player"]).reset_index(drop=True)
    return base, source_season

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Permit using prior seasons when requested season data is unavailable",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Compatibility flag used by CI. "
            "Player form already clamps to the requested season, so this is ignored."
        ),
    )
    args = parser.parse_args()
    _safe_mkdir(DATA_DIR)
    _safe_mkdir(METRICS_DIR)
    try:
        df, source_season = build_player_form(
            args.season,
            allow_fallback=args.allow_fallback,
        )
    except Exception as e:
        print(f"[make_player_form] ERROR: {e}", file=sys.stderr)
        df = pd.DataFrame(columns=["player","team","season","source_season","target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate"])
        source_season = args.season
    df.to_csv(OUTPATH, index=False)
    df.to_csv(METRICS_OUTPATH, index=False)
    print(
        f"[make_player_form] Wrote {len(df)} rows → {OUTPATH} and {METRICS_OUTPATH}"
    )

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
