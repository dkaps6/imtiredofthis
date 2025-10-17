# scripts/make_player_form.py
"""
Create player_form.csv: per-player usage & efficiency metrics.

Inputs:
- Play-by-play via nflreadpy (preferred) or nfl_data_py (fallback)
- Optional enrichers (ESPN, MySportsFeeds, API-Sports, NFLGSIS)
Output:
- data/player_form.csv
"""

from __future__ import annotations
import argparse, os, sys, warnings
import pandas as pd, numpy as np
from typing import Tuple

def _import_nflverse():
    try:
        import nflreadpy as nflv
        return nflv, "nflreadpy"
    except Exception:
        import nfl_data_py as nflv  # fallback
        return nflv, "nfl_data_py"

NFLV, NFLPKG = _import_nflverse()
DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "player_form.csv")
_CACHE_DIRS = [
    DATA_DIR,
    os.path.join("external", "nflverse_bundle"),
]

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

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


def load_pbp(season:int)->pd.DataFrame:
    cached, source_path = _load_cached_csv("pbp", season)
    if not cached.empty:
        print(f"[make_player_form] ℹ️ Using cached pbp_{season}.csv from {source_path}")
        return cached
    if NFLPKG=="nflreadpy":
        df = NFLV.load_pbp(seasons=[season])
    else:
        df = NFLV.import_pbp_data([season], downcast=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def _load_required_pbp(season: int) -> tuple[pd.DataFrame, int]:
    """Load play-by-play strictly for ``season`` or raise."""
    cached, cache_path = _load_cached_csv("pbp", season)
    if not cached.empty:
        print(
            f"[make_player_form] ℹ️ Loaded cached pbp_{season}.csv from {cache_path}"
        )
        return cached, season

    errors: list[str] = []
    try:
        df = load_pbp(season)
    except Exception as err:
        errors.append(str(err))
    else:
        if not df.empty:
            return df, season
        errors.append("empty dataframe")

    raise RuntimeError(
        "PBP unavailable for requested season. "
def _load_pbp_with_fallback(
    season: int,
    *,
    allow_prior_seasons: bool,
    max_lookback: int = 5,
) -> tuple[pd.DataFrame, int]:
    """
    Return the first available season ≤ ``season`` with play-by-play data.

    When ``allow_prior_seasons`` is False we will raise instead of silently
    substituting older data; this keeps 2025 runs from drifting to 2024 metrics
    when live pulls fail.
    """
def _load_pbp_with_fallback(season: int, max_lookback: int = 5) -> tuple[pd.DataFrame, int]:
    """Return the first available season ≤ ``season`` with play-by-play data."""
    errors: list[str] = []
    for offset in range(0, max_lookback + 1):
        candidate = season - offset
        if candidate < 2000:
            break
        try:
            df = load_pbp(candidate)
        except Exception as err:
            errors.append(f"season {candidate}: {err}")
            continue
        if not df.empty:
            if candidate == season:
                return df, candidate

            if allow_prior_seasons:
                print(
                    f"[make_player_form] ⚠️ No PBP for {season}; using {candidate} instead"
                )
                return df, candidate

            errors.append(
                f"season {candidate}: available but fallback disabled"
            )
            continue
            if candidate != season:
                print(f"[make_player_form] ⚠️ No PBP for {season}; using {candidate} instead")
            return df, candidate
        errors.append(f"season {candidate}: empty dataframe")
    raise RuntimeError(
        "PBP unavailable for requested season and fallbacks. "
        + "; ".join(errors) if errors else ""
    )

def load_participation(season:int)->pd.DataFrame:
    cached, source_path = _load_cached_csv("participation", season)
    if not cached.empty:
        print(
            f"[make_player_form] ℹ️ Using cached participation_{season}.csv from {source_path}"
        )
        return cached
    try:
        if NFLPKG=="nflreadpy":
            p = NFLV.load_participation(seasons=[season])
        else:
            return pd.DataFrame()
        p.columns = [c.lower() for c in p.columns]
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

def build_player_form(season:int)->tuple[pd.DataFrame, int]:
    print(f"[make_player_form] Loading PBP for {season} ({NFLPKG}) ...")
    pbp, source_season = _load_required_pbp(season)
    if pbp.empty:
        raise RuntimeError("Empty PBP.")
    base = compute_player_usage(pbp)
    print(f"[make_player_form] Loading PBP for {season} ({NFLPKG}) ...")
    pbp, source_season = _load_required_pbp(season)
    if pbp.empty:
        raise RuntimeError("Empty PBP.")
    base = compute_player_usage(pbp)
def build_player_form(season:int, *, allow_fallback: bool)->tuple[pd.DataFrame, int]:
    print(f"[make_player_form] Loading PBP for {season} ({NFLPKG}) ...")
    pbp, source_season = _load_pbp_with_fallback(
        season, allow_prior_seasons=allow_fallback
    )
    if pbp.empty:
        raise RuntimeError("Empty PBP.")
    base = compute_player_usage(pbp)
    if base.empty and source_season > 2000 and allow_fallback:
    pbp, source_season = _load_pbp_with_fallback(season)
    if pbp.empty:
        raise RuntimeError("Empty PBP.")
    base = compute_player_usage(pbp)
    if base.empty and source_season > 2000:
        # Try progressively earlier seasons when usage extraction fails (rare)
        for fallback in range(source_season - 1, max(source_season - 5, 1999), -1):
            try:
                alt_pbp = load_pbp(fallback)
            except Exception:
                continue
            if alt_pbp.empty:
                continue
            tmp = compute_player_usage(alt_pbp)
            if not tmp.empty:
                print(f"[make_player_form] ⚠️ Usage empty; falling back to {fallback}")
                base = tmp
                source_season = fallback
                pbp = alt_pbp
                break
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
    args = parser.parse_args()
    _safe_mkdir(DATA_DIR)
    try:
        df, source_season = build_player_form(
            args.season
        )
            args.season, allow_fallback=args.allow_fallback
        )
        if source_season != args.season:
            print(
                f"[make_player_form] ℹ️ Using {source_season} metrics as proxy for {args.season}"
            )
    except Exception as e:
        print(f"[make_player_form] ERROR: {e}", file=sys.stderr)
        df = pd.DataFrame(columns=["player","team","season","source_season","target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate"])
        source_season = args.season
    df.to_csv(OUTPATH, index=False)
    print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
