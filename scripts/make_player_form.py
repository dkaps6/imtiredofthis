#!/usr/bin/env python3
# make_player_form.py  — restored to your original with tiny, surgical additions only.
from __future__ import annotations

import os  # >>> ADD: needed for env handling
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/player_form.csv")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
ERR_LOG = LOG_DIR / "nfl_pbp_error.txt"

# ---------------------- helpers (ONLY additions are clearly marked) ----------------------

def _safe_read_csv(p: str | Path) -> pd.DataFrame:
    p = Path(p)
    if not p.exists() or p.stat().st_size < 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# >>> ADD: tiny, non-destructive “fill missing from provider” helper
def _merge_missing_player(df: pd.DataFrame, add: pd.DataFrame,
                          on: tuple[str, str] | list[str],
                          mapping: dict[str, str] | None = None,
                          tag: str = "") -> pd.DataFrame:
    """Only fills NaNs for mapped columns; never overwrites non-null values."""
    if add is None or add.empty:
        return df
    m = add.copy()
    if mapping:
        m = m.rename(columns=mapping)
    on_cols = list(on) if isinstance(on, (list, tuple)) else [on]
    keep_cols = on_cols + [c for c in (mapping.values() if mapping else []) if c in m.columns]
    keep_cols = [c for c in keep_cols if c in m.columns]
    if len(keep_cols) <= len(on_cols):
        return df
    merged = df.merge(m[keep_cols], on=on_cols, how="left", suffixes=("", "__prov"))
    for col in keep_cols:
        if col in on_cols:
            continue
        prov = f"{col}__prov"
        if prov in merged.columns:
            mask = merged[col].isna() & merged[prov].notna()
            if mask.any():
                print(f"[player_form] filled {mask.sum()} rows for {col} from {tag}")
                merged.loc[mask, col] = merged.loc[mask, prov]
            merged.drop(columns=[prov], inplace=True, errors="ignore")
    return merged
# <<< ADD

def _safe_write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        # write a stub to keep pipeline moving
        pd.DataFrame(columns=[
            "player","team","position","role",
            "target_share","rush_share","route_rate",
            "rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","qb_ypa","ypt"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

# ---------------------- data fetch (unchanged pattern, 2025-safe) ----------------------

def _fetch_pbp(season: int, *, allow_fallback: bool = False, tries: int = 3, wait: float = 1.0) -> pd.DataFrame:
    """
    Play-by-play: prefer nflreadpy, fallback to nfl_data_py.
    If allow_fallback=True, try season-1 when requested season returns empty (does not raise).
    """
    last_err = None
    try:
        import nflreadpy as nfr
        use_readpy = True
    except Exception:
        use_readpy = False
        try:
            import nfl_data_py as nfl  # noqa: F401
        except Exception:
            pass

    seasons_to_try = [season] if not allow_fallback else [season, season - 1]

    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== PBP fetch season={season} allow_fallback={allow_fallback} ===\n")

    for s in seasons_to_try:
        for k in range(1, tries + 1):
            try:
                if use_readpy:
                    import nflreadpy as nfr
                    pf = nfr.load_pbp([s])
                    df = pf.to_pandas()
                else:
                    import nfl_data_py as nfl
                    df = nfl.import_pbp_data([s])

                if isinstance(df, pd.DataFrame) and len(df):
                    with open(ERR_LOG, "a", encoding="utf-8") as f:
                        f.write(f"season {s} try {k}: OK rows={len(df)} via {'nflreadpy' if use_readpy else 'nfl_data_py'}\n")
                    if s != season:
                        print(f"[player_form] NOTE: using season {s} as fallback for {season}")
                    return df
                raise RuntimeError("empty dataframe")
            except Exception as e:
                last_err = e
                with open(ERR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"season {s} try {k}: {type(e).__name__}: {e}\n{traceback.format_exc()}\n")
                time.sleep(wait)

    raise RuntimeError(f"PBP fetch failed: {type(last_err).__name__}: {last_err}")

# ---------------------- builder (restored + tiny additions) ----------------------

def build_player_form(season: int, *, allow_fallback: bool = False) -> pd.DataFrame:
    pbp = _fetch_pbp(season, allow_fallback=allow_fallback)
    if pbp is None or pbp.empty:
        raise RuntimeError("no pbp returned")

    # Normalize
    for c in ("posteam", "defteam", "receiver", "rusher", "passer"):
        if c in pbp.columns:
            pbp[c] = pbp[c].astype(str)

    # Flags as bools (support multiple column names from libs)
    for c in ("is_pass", "is_rush", "pass", "rush"):
        if c in pbp.columns:
            pbp[c] = pbp[c].astype(str).str.lower().isin(["1","true","t","yes"])
    pbp["is_pass"] = pbp.get("is_pass", pbp.get("pass", False)).astype(bool)
    pbp["is_rush"] = pbp.get("is_rush", pbp.get("rush", False)).astype(bool)

    # Red-zone helper
    if "yardline_100" in pbp.columns:
        pbp["in_rz"] = (pbp["yardline_100"] <= 20).astype(bool)
    else:
        pbp["in_rz"] = False

    # Receiving tallies
    rec = pbp.loc[pbp["is_pass"] & (pbp["receiver"] != ""), ["game_id", "posteam", "receiver"]].copy()
    rec["tgt"] = 1
    ply_tgts = rec.groupby(["game_id", "posteam", "receiver"])["tgt"].sum().rename("player_targets")
    team_tgts = rec.groupby(["game_id", "posteam"])["tgt"].sum().rename("team_targets")
    rec_yds = pbp.loc[pbp["receiver"] != ""].groupby(
        ["game_id", "posteam", "receiver"]
    )["yards_gained"].sum().rename("rec_yards")

    # Rushing tallies
    rush = pbp.loc[pbp["is_rush"] & (pbp["rusher"] != ""), ["game_id", "posteam", "rusher"]].copy()
    rush["att"] = 1
    ply_rush = rush.groupby(["game_id", "posteam", "rusher"])["att"].sum().rename("player_rush_att")
    team_rush = rush.groupby(["game_id", "posteam"])["att"].sum().rename("team_rush_att")
    rush_yds = pbp.loc[pbp["rusher"] != ""].groupby(
        ["game_id", "posteam", "rusher"]
    )["yards_gained"].sum().rename("rush_yards")

    # Red-zone
    rz_targs = pbp.loc[pbp["is_pass"] & pbp["in_rz"] & (pbp["receiver"] != ""), ["game_id", "posteam", "receiver"]].copy()
    rz_targs["rz_tgt"] = 1
    ply_rz_tgt = rz_targs.groupby(["game_id", "posteam", "receiver"])["rz_tgt"].sum().rename("player_rz_tgt")
    team_rz_tgt = rz_targs.groupby(["game_id", "posteam"])["rz_tgt"].sum().rename("team_rz_tgt")

    rz_rush = pbp.loc[pbp["is_rush"] & pbp["in_rz"] & (pbp["rusher"] != ""), ["game_id", "posteam", "rusher"]].copy()
    rz_rush["rz_carry"] = 1
    ply_rz_carry = rz_rush.groupby(["game_id", "posteam", "rusher"])["rz_carry"].sum().rename("player_rz_carry")
    team_rz_carry = rz_rush.groupby(["game_id", "posteam"])["rz_carry"].sum().rename("team_rz_carry")

    # Base player rows
    rec_level = ply_tgts.reset_index().rename(columns={"receiver": "player"})
    rush_level = ply_rush.reset_index().rename(columns={"rusher": "player"})
    players = pd.concat(
        [
            rec_level[["game_id", "posteam", "player", "player_targets"]],
            rush_level[["game_id", "posteam", "player", "player_rush_att"]],
        ],
        ignore_index=True,
    ).fillna(0)

    # Merge all aggregates
    merges = [
        team_tgts.reset_index(),
        team_rush.reset_index(),
        ply_rz_tgt.reset_index().rename(columns={"receiver": "player"}),
        team_rz_tgt.reset_index(),
        ply_rz_carry.reset_index().rename(columns={"rusher": "player"}),
        team_rz_carry.reset_index(),
        rec_yds.reset_index().rename(columns={"receiver": "player"}),
        rush_yds.reset_index().rename(columns={"rusher": "player"}),
    ]
    for m in merges:
        on_cols = [c for c in ["game_id", "posteam", "player"] if c in m.columns]
        players = players.merge(m, on=on_cols, how="left")

    # Safe fill numeric zeros
    for c in [
        "player_targets","team_targets","player_rush_att","team_rush_att",
        "player_rz_tgt","team_rz_tgt","player_rz_carry","team_rz_carry",
        "rec_yards","rush_yards"
    ]:
        if c not in players.columns:
            players[c] = 0
        players[c] = players[c].fillna(0)

    # Season aggregate
    agg = players.groupby(["posteam", "player"], as_index=False).sum(numeric_only=True)
    agg["team"] = agg["posteam"].astype(str).str.upper()
    agg["target_share"] = (agg["player_targets"] / agg["team_targets"]).replace([np.inf, -np.inf], np.nan)
    agg["rush_share"]   = (agg["player_rush_att"] / agg["team_rush_att"]).replace([np.inf, -np.inf], np.nan)
    agg["rz_tgt_share"] = (agg["player_rz_tgt"] / agg["team_rz_tgt"]).replace([np.inf, -np.inf], np.nan)
    agg["rz_carry_share"] = (agg["player_rz_carry"] / agg["team_rz_carry"]).replace([np.inf, -np.inf], np.nan)
    agg["ypt"] = (agg["rec_yards"] / agg["player_targets"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    agg["ypc"] = (agg["rush_yards"] / agg["player_rush_att"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Position guess (original behavior)
    agg["position"] = np.where(agg["target_share"].fillna(0) > 0.20, "WR",
                        np.where(agg["rush_share"].fillna(0) > 0.20, "RB", "TE"))
    agg["role"] = np.where(agg["position"].eq("WR"), "WR1",
                    np.where(agg["position"].eq("RB"), "RB1", "TE1"))

    # >>> ADD: ensure QB rows exist (QB1 per team) with qb_ypa for downstream QB props
    try:
        pass_rows = pbp.loc[pbp["is_pass"] & (pbp["passer"] != ""), ["posteam", "passer", "yards_gained"]].copy()
        pass_rows["att"] = 1
        qb_by_team = pass_rows.groupby(["posteam", "passer"], as_index=False).agg(
            att=("att", "sum"),
            pass_yards=("yards_gained", "sum")
        )
        qb_top = qb_by_team.sort_values(["posteam", "att", "pass_yards"],
                                        ascending=[True, False, False]).groupby("posteam", as_index=False).first()
        qb_top["team"] = qb_top["posteam"].astype(str).str.upper()
        qb_top["player"] = qb_top["passer"]
        qb_top["qb_ypa"] = (qb_top["pass_yards"] / qb_top["att"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        qb_top["position"] = "QB"
        qb_top["role"] = "QB1"
        qb_top["target_share"] = 0.0
        qb_top["rush_share"] = 0.0
        qb_top["route_rate"] = 0.0
        qb_top["rz_tgt_share"] = 0.0
        qb_top["rz_carry_share"] = 0.0
        qb_top["yprr_proxy"] = 0.0
        qb_top["ypc"] = 0.0
        qb_top["ypt"] = np.nan

        qb_cols = ["player","team","position","role","target_share","rush_share","route_rate",
                   "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","qb_ypa","ypt"]
        for col in qb_cols:
            if col not in agg.columns:
                agg[col] = np.nan
        qb_out = qb_top.reindex(columns=["player","team","position","role",
                                         "target_share","rush_share","route_rate",
                                         "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","qb_ypa","ypt"])
        # concat to agg later after we finish providers fill
    except Exception as _e:
        qb_out = pd.DataFrame()
        print(f"[player_form] WARN: QB enrich skipped: {_e}")
    # <<< ADD

    # Prepare base output
    out = agg.copy()
    for c in ["route_rate","yprr_proxy","target_share","rush_share","rz_tgt_share","rz_carry_share","ypc","ypt","qb_ypa"]:
        if c not in out.columns:
            out[c] = np.nan
    out = out[[
        "player","team","position","role",
        "target_share","rush_share","route_rate",
        "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","qb_ypa","ypt"
    ]]

    # ===================== FALLBACK PROVIDERS (NO PFR) =====================

    # ESPN
    espn = _safe_read_csv("data/espn_player.csv")
    out = _merge_missing_player(out, espn, ("player","team"), {
        "routes_per_dropback":"route_rate","routes_db":"route_rate","route_rate":"route_rate",
        "target_share":"target_share","targets_share":"target_share",
        "rush_share":"rush_share",
        "yprr":"yprr_proxy","yards_per_route_run":"yprr_proxy",
        "ypc":"ypc","yards_per_carry":"ypc",
        "ypa":"qb_ypa","yards_per_attempt":"qb_ypa",
        "rz_tgt_share":"rz_tgt_share","rz_carry_share":"rz_carry_share",
    }, tag="ESPN")

    # MySportsFeeds
    msf = _safe_read_csv("data/msf_player.csv")
    out = _merge_missing_player(out, msf, ("player","team"), {
        "route_rate":"route_rate","routes_per_dropback":"route_rate",
        "target_share":"target_share","tgt_share":"target_share",
        "rush_share":"rush_share",
        "yprr":"yprr_proxy",
        "ypc":"ypc",
        "ypa":"qb_ypa",
        "rz_tgt_share":"rz_tgt_share","rz_tgt_pct":"rz_tgt_share",
        "rz_carry_share":"rz_carry_share",
    }, tag="MySportsFeeds")

    # API-Sports
    apis = _safe_read_csv("data/apisports_player.csv")
    out = _merge_missing_player(out, apis, ("player","team"), {
        "routes_per_dropback":"route_rate","route_pct":"route_rate",
        "tgt_share":"target_share","target_share":"target_share",
        "rush_share":"rush_share",
        "yprr":"yprr_proxy",
        "ypc":"ypc",
        "ypa":"qb_ypa",
        "rz_tgt_share":"rz_tgt_share",
        "rz_carry_share":"rz_carry_share",
    }, tag="API-Sports")

    # NFLGSIS
    gsis = _safe_read_csv("data/gsis_player.csv")
    out = _merge_missing_player(out, gsis, ("player","team"), {
        "routes_per_dropback":"route_rate","route_rate":"route_rate",
        "target_share":"target_share",
        "rush_share":"rush_share",
        "yprr":"yprr_proxy",
        "ypc":"ypc",
        "ypa":"qb_ypa",
        "rz_tgt_share":"rz_tgt_share",
        "rz_carry_share":"rz_carry_share",
    }, tag="NFLGSIS")

    # >>> ADD: now append QB rows (if any) so QB props can merge smoothly
    if not qb_out.empty:
        out = pd.concat([out, qb_out[out.columns]], ignore_index=True)
    # <<< ADD

    # Numeric sanitize
    for c in ("target_share","rush_share","route_rate","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","qb_ypa","ypt"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# ---------------------- CLI wrapper (unchanged) ----------------------

def cli(season: int, *, allow_fallback: bool = False) -> int:
    try:
        df = build_player_form(season, allow_fallback=allow_fallback)
        if df is None or df.empty:
            raise RuntimeError("[player_form] looks empty/stub — check logs/nfl_pbp_error.txt")
    except Exception as e:
        print(f"[player_form] fatal error: {type(e).__name__}: {e}")
        df = pd.DataFrame()
    _safe_write(df, OUT)
    print(f"[player_form] wrote rows={len(df)} → {OUT}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--allow-fallback", action="store_true",
                    help="Also try season-1 if the requested season returns empty PBP.")
    args = ap.parse_args()
    allow_fb = args.allow_fallback or os.getenv("ALLOW_PBP_FALLBACK", "").strip() in ("1","true","TRUE","yes","YES")
    sys.exit(cli(args.season, allow_fallback=allow_fb))
