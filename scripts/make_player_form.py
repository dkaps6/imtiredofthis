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

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_div(n, d):
    n = n.astype(float); d = d.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, np.nan, n / d)

def load_pbp(season:int)->pd.DataFrame:
    if NFLPKG=="nflreadpy":
        df = NFLV.load_pbp(seasons=[season])
    else:
        df = NFLV.import_pbp_data([season], downcast=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def load_participation(season:int)->pd.DataFrame:
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

def fallback_from_external(base: pd.DataFrame) -> pd.DataFrame:
    """Merge optional enrichers without overwriting existing values."""
    enrichers = ["espn_player_form.csv","msf_player_form.csv","apisports_player_form.csv","nflgsis_player_form.csv"]
    for f in enrichers:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            continue
        try:
            e = pd.read_csv(path)
        except Exception:
            continue
        e.columns = [c.lower() for c in e.columns]
        keep = [c for c in ["team","player","target_share","rush_share","route_rate","ypt","ypc","yprr_proxy"] if c in e.columns]
        if not keep:
            continue
        e = e[keep]
        base = base.merge(e, on=["team","player"], how="left", suffixes=("","_ext"))
        for c in [k for k in keep if k not in ["team","player"]]:
            base[c] = base[c].combine_first(base.get(f"{c}_ext"))
            dropcol = f"{c}_ext"
            if dropcol in base.columns:
                base.drop(columns=[dropcol], inplace=True)
    return base

def build_player_form(season:int)->pd.DataFrame:
    print(f"[make_player_form] Loading PBP for {season} ({NFLPKG}) ...")
    pbp = load_pbp(season)
    if pbp.empty:
        raise RuntimeError("Empty PBP.")
    base = compute_player_usage(pbp)
    if base.empty:
        # Try prior season as a safety net (preseason weeks often sparse)
        prev = load_pbp(season-1)
        if not prev.empty:
            base = compute_player_usage(prev)
    part = load_participation(season)
    base = enrich_with_participation(base, part)
    base = fallback_from_external(base)
    base["season"] = season
    order = ["player","team","season","target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate"]
    for c in order:
        if c not in base.columns:
            base[c] = np.nan
    base = base[order].sort_values(["team","player"]).reset_index(drop=True)
    return base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    _safe_mkdir(DATA_DIR)
    try:
        df = build_player_form(args.season)
    except Exception as e:
        print(f"[make_player_form] ERROR: {e}", file=sys.stderr)
        df = pd.DataFrame(columns=["player","team","season","target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate"])
    df.to_csv(OUTPATH, index=False)
    print(f"[make_player_form] Wrote {len(df)} rows → {OUTPATH}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
