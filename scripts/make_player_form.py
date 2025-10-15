#!/usr/bin/env python3
from __future__ import annotations
import sys, os, time, traceback
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/player_form.csv")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
ERR_LOG = LOG_DIR / "nfl_pbp_error.txt"

# ---------------------- utils ----------------------
def _safe_read_csv(p: str | Path) -> pd.DataFrame:
    p = Path(p)
    if not p.exists() or (hasattr(p, "stat") and p.stat().st_size < 5):
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def _safe_write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame(columns=[
            "player","team","position","role",
            "target_share","rush_share","route_rate",
            "rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","qb_ypa",
            # NEW: explicit ypt + route_share alias for downstream if needed
            "ypt"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

def _merge_missing_player(df: pd.DataFrame, add: pd.DataFrame, on: tuple[str,str] | list[str], mapping: dict[str, str] | None = None, tag: str = "") -> pd.DataFrame:
    if add.empty:
        return df
    src = add.copy()
    if mapping:
        src = src.rename(columns=mapping)
    on_cols = list(on) if isinstance(on, (list, tuple)) else [on]
    cols = on_cols + [c for c in (mapping.values() if mapping else []) if c in src.columns]
    cols = [c for c in cols if c in src.columns]
    if len(cols) <= len(on_cols):
        return df
    merged = df.merge(src[cols], on=on_cols, how="left", suffixes=("","__prov"))
    for col in cols:
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

# ---------------------- sources ----------------------
def _fetch_pbp(season: int, tries: int = 3, wait: float = 1.0) -> pd.DataFrame:
    last = None
    seasons_to_try = [season, season-1]

    # FIX: initialize use_readpy just like in team file
    try:
        import nflreadpy as nfr
        use_readpy = True
    except Exception:
        use_readpy = False
        import nfl_data_py as nfl  # noqa: F401

    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== PBP fetch wanted season={season} ===\n")

    for s in seasons_to_try:
        for i in range(1, tries + 1):
            try:
                if use_readpy:
                    pf = nfr.load_pbp([s])
                    df = pf.to_pandas()
                else:
                    df = nfl.import_pbp_data([s])

                if isinstance(df, pd.DataFrame) and len(df):
                    with open(ERR_LOG, "a", encoding="utf-8") as f:
                        f.write(f"season {s} try {i}: OK rows={len(df)} via "
                                f"{'nflreadpy' if use_readpy else 'nfl_data_py'}\n")
                    if s != season:
                        print(f"[player_form] NOTE: using season {s} as fallback for {season}", flush=True)
                    return df
                raise RuntimeError("empty dataframe")
            except Exception as e:
                last = e
                with open(ERR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"season {s} try {i}: {type(e).__name__}: {e}\n{traceback.format_exc()}\n")
                if (not use_readpy) and ("HTTP Error 404" in str(e)):
                    break
                time.sleep(wait)
    raise RuntimeError(f"PBP fetch failed: {type(last).__name__}: {last}")

# ---------------------- builder ----------------------
def build_player_form(season: int) -> pd.DataFrame:
    pbp = _fetch_pbp(season)
    if pbp is None or pbp.empty:
        raise RuntimeError("no pbp")

    # Normalize strings for join safety
    for c in ("posteam","defteam","receiver","rusher","passer"):
        if c in pbp.columns:
            pbp[c] = pbp[c].astype(str)

    # Flags
    for c in ("is_pass","is_rush","pass","rush"):
        if c in pbp.columns:
            pbp[c] = pbp[c].astype(str).str.lower().isin(["1","true","t","yes"])
    pbp["pass_flag"] = pbp.get("is_pass", pbp.get("pass", False)).astype(bool)
    pbp["rush_flag"] = pbp.get("is_rush", pbp.get("rush", False)).astype(bool)

    # Receiving: player + team level targets & yards
    rec = pbp.loc[pbp["pass_flag"]==True, ["game_id","posteam","receiver","yards_gained"]].copy()
    rec["tgt"] = 1
    rec_tgts = rec.groupby(["game_id","posteam","receiver"]).agg(player_targets=("tgt","sum")).astype(int)
    team_tgts = rec.groupby(["game_id","posteam"]).agg(team_targets=("tgt","sum")).astype(int)
    rec_yards = rec.groupby(["game_id","posteam","receiver"]).agg(rec_yards=("yards_gained","sum"))

    # Rushing: player + team carries & yards
    rush = pbp.loc[pbp["rush_flag"]==True, ["game_id","posteam","rusher","yards_gained"]].copy()
    rush["att"] = 1
    ply_rush = rush.groupby(["game_id","posteam","rusher"]).agg(player_rush_att=("att","sum")).astype(int)
    team_rush = rush.groupby(["game_id","posteam"]).agg(team_rush_att=("att","sum")).astype(int)
    ply_rush_yds = rush.groupby(["game_id","posteam","rusher"]).agg(rush_yards=("yards_gained","sum"))

    # Red-zone targets/carries
    rbp = pbp.copy()
    if "yardline_100" in rbp.columns:
        rbp["is_rz_play"] = (rbp["yardline_100"] <= 20).astype(int)
    else:
        rbp["is_rz_play"] = 0
    rz_pass = rbp.loc[rbp["pass_flag"]==True, ["game_id","posteam","receiver","is_rz_play"]]
    rz_rush = rbp.loc[rbp["rush_flag"]==True, ["game_id","posteam","rusher","is_rz_play"]]
    ply_rz_tgt  = rz_pass.groupby(["game_id","posteam","receiver"]).agg(player_rz_tgt=("is_rz_play","sum"))
    team_rz_tgt = rz_pass.groupby(["game_id","posteam"]).agg(team_rz_tgt=("is_rz_play","sum"))
    ply_rz_carry  = rz_rush.groupby(["game_id","posteam","rusher"]).agg(player_rz_carry=("is_rz_play","sum"))
    team_rz_carry = rz_rush.groupby(["game_id","posteam"]).agg(team_rz_carry=("is_rz_play","sum"))

    # Collect per-game player rows then aggregate season
    rec_level  = rec_tgts.reset_index().rename(columns={"receiver":"player"})
    rush_level = ply_rush.reset_index().rename(columns={"rusher":"player"})
    players = pd.concat(
        [rec_level[["game_id","posteam","player","player_targets"]],
         rush_level[["game_id","posteam","player","player_rush_att"]]],
        ignore_index=True
    ).fillna(0)

    merges = [
        team_tgts.reset_index(),
        team_rush.reset_index(),
        ply_rz_tgt.reset_index().rename(columns={"receiver":"player"}),
        team_rz_tgt.reset_index(),
        ply_rz_carry.reset_index().rename(columns={"rusher":"player"}),
        team_rz_carry.reset_index(),
        rec_yards.reset_index().rename(columns={"receiver":"player"}),
        ply_rush_yds.reset_index().rename(columns={"rusher":"player"}),
    ]
    for m in merges:
        players = players.merge(m, on=[c for c in ["game_id","posteam","player"] if c in m.columns], how="left")

    for c in ["player_targets","team_targets","player_rush_att","team_rush_att",
              "player_rz_tgt","team_rz_tgt","player_rz_carry","team_rz_carry",
              "rec_yards","rush_yards"]:
        if c not in players.columns: players[c]=0
        players[c] = players[c].fillna(0)

    agg = players.groupby(["posteam","player"]).sum(numeric_only=True).reset_index()
    agg["team"] = agg["posteam"].astype(str).str.upper()
    agg["target_share"] = (agg["player_targets"] / agg["team_targets"]).replace([np.inf,-np.inf], np.nan)
    agg["rush_share"]   = (agg["player_rush_att"] / agg["team_rush_att"]).replace([np.inf,-np.inf], np.nan)
    agg["rz_tgt_share"] = (agg["player_rz_tgt"] / agg["team_rz_tgt"]).replace([np.inf,-np.inf], np.nan)
    agg["rz_carry_share"] = (agg["player_rz_carry"] / agg["team_rz_carry"]).replace([np.inf,-np.inf], np.nan)
    agg["ypt"] = (agg["rec_yards"] / agg["player_targets"].replace(0,np.nan)).replace([np.inf,-np.inf], np.nan)
    agg["ypc"] = (agg["rush_yards"] / agg["player_rush_att"].replace(0,np.nan)).replace([np.inf,-np.inf], np.nan)

    # Default positions: WR/RB/TE guess from usage; refined later if you ingest roles.csv
    agg["position"] = np.where(agg["target_share"].fillna(0) > 0.20, "WR",
                        np.where(agg["rush_share"].fillna(0) > 0.20, "RB", "TE"))
    agg["role"] = np.where(agg["position"].eq("WR"), "WR1",
                     np.where(agg["position"].eq("RB"), "RB1", "TE1"))

    # Participation-based route proxy (optional)
    pass_snaps_cols = pd.DataFrame()
    try:
        import nflreadpy as nfr
        pf = nfr.load_participation([season])  # polars -> pandas
        pf = pf.to_pandas()
        if pf is not None and not pf.empty and {"game_id","play_id","offense_players"}.issubset(pf.columns):
            _off = pbp.loc[pbp["posteam"].notna()].copy()
            _off["team"] = _off["posteam"].str.upper()
            p_pass = _off.loc[_off["is_pass"]==True, ["game_id","play_id","team"]].copy()
            pp = p_pass.merge(pf[["game_id","play_id","offense_players"]], on=["game_id","play_id"], how="left")
            pp["offense_players"] = pp["offense_players"].fillna("")
            pp = pp.loc[pp["offense_players"].ne("")]
            pp = pp.assign(player_name_list=pp["offense_players"].str.split(";")).explode("player_name_list")
            pp["player"] = pp["player_name_list"].str.replace(r"\(.*\)","", regex=True).str.strip()
            pp["player_norm"] = pp["player"].str.lower().str.replace(r"[.'’]", "", regex=True).str.strip()
            team_pass_snaps = pp.groupby("team", as_index=False).agg(team_pass_snaps=("play_id","count"))
            ply_pass_snaps  = pp.groupby(["team","player_norm"], as_index=False).agg(pass_snaps=("play_id","count"))
            pass_snaps_cols = ply_pass_snaps.merge(team_pass_snaps, on="team", how="left")
    except Exception as e:
        print(f"[player_form] participation route proxy skipped: {type(e).__name__}: {e}", flush=True)

    # Merge participation-based route proxy
    out = agg.copy()
    out["player_norm"] = out["player"].astype(str).str.lower().str.replace(r"[.'’]", "", regex=True).str.strip()
    if not pass_snaps_cols.empty:
        out = out.merge(pass_snaps_cols, on=["team","player_norm"], how="left")
        # route_rate if missing: pass_snaps / team_pass_snaps
        rr = (out["pass_snaps"] / out["team_pass_snaps"]).replace([np.inf,-np.inf], np.nan)
        out["route_rate"] = out["route_rate"].where(out["route_rate"].notna(), rr)
        # yprr_proxy real: rec_yards / pass_snaps (falls back to ypt when snaps=0)
        yprr_real = (out["rec_yards"] / np.where(out["pass_snaps"].fillna(0).eq(0), np.nan, out["pass_snaps"])).replace([np.inf,-np.inf], np.nan)
        out["yprr_proxy"] = out["yprr_proxy"].where(out["yprr_proxy"].notna(), yprr_real)
    out.drop(columns=["player_norm","pass_snaps","team_pass_snaps"], inplace=True, errors="ignore")

    # --- ADD: make sure critical columns exist BEFORE selecting subset (prevents KeyError when participation missing)
    for c in ["route_rate","yprr_proxy","target_share","rush_share","rz_tgt_share","rz_carry_share","ypc","ypt"]:
        if c not in out.columns:
            out[c] = np.nan

    out = out.rename(columns={"player":"player"})[[
        "player","team","position","role",
        "target_share","rush_share","route_rate",
        "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt"
    ]]

    # ESPN/PFR enrich (optional, non-destructive)
    try:
        pfr = _safe_read_csv("data/pfr_player_enrich.csv")
        if not pfr.empty:
            pfr = pfr.rename(columns={
                "player":"player",
                "team":"team",
                "routes_per_dropback":"route_rate_enrich"
            })
            out = out.merge(pfr[["player","team","route_rate_enrich"]], on=["player","team"], how="left")
            if "route_rate_enrich" in out.columns and "route_rate" in out.columns:
                out["route_rate"] = out["route_rate"].fillna(out["route_rate_enrich"])
    except Exception as e:
        print(f"[player_form] PFR enrich skipped: {type(e).__name__}: {e}", flush=True)

    # --- Ensure QB rows exist (QB1 per team) so QB props can merge cleanly
    try:
        pass_att = pbp.loc[pbp["pass_flag"]==True, ["posteam","passer","yards_gained"]].copy()
        pass_att["att"] = 1
        qb_by_team = pass_att.groupby(["posteam","passer"], as_index=False).agg(
            att=("att","sum"),
            pass_yards=("yards_gained","sum")
        )
        # pick top passer per team by attempts
        qb_top = qb_by_team.sort_values(["posteam","att","pass_yards"], ascending=[True, False, False]).groupby("posteam", as_index=False).first()
        qb_top["team"] = qb_top["posteam"].astype(str).str.upper()
        qb_top["player"] = qb_top["passer"]
        qb_top["qb_ypa"] = (qb_top["pass_yards"] / qb_top["att"].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)

        # schema columns with safe defaults
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
        qb_cols = ["player","team","position","role","target_share","rush_share","route_rate","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","qb_ypa","ypt"]
        # align to out columns, append
        for col in qb_cols:
            if col not in out.columns:
                out[col] = np.nan
        qb_out = qb_top.reindex(columns=qb_cols)
        out = pd.concat([out, qb_out[out.columns]], ignore_index=True)
    except Exception as _e:
        print(f"[player_form] WARN: QB rows enrich skipped: {_e}")

    # 4) External provider fallbacks (fill only remaining NaNs) — NO NUKES, try multiple common header names

    # ESPN
    espn = _safe_read_csv("data/espn_player.csv")
    out = _merge_missing_player(out, espn, ("player","team"), {
        # route %
        "routes_db":"route_rate", "routes_per_dropback":"route_rate", "route_rate":"route_rate",
        # shares
        "target_share":"target_share", "targets_share":"target_share",
        "rush_share":"rush_share",
        # efficiencies
        "yprr":"yprr_proxy", "yards_per_route_run":"yprr_proxy",
        "ypc":"ypc", "yards_per_carry":"ypc",
        # QB efficiency if present
        "ypa":"qb_ypa", "yards_per_attempt":"qb_ypa",
    }, tag="ESPN")

    # MySportsFeeds
    msf = _safe_read_csv("data/msf_player.csv")
    out = _merge_missing_player(out, msf, ("player","team"), {
        "route_rate":"route_rate", "routes_per_dropback":"route_rate",
        "target_share":"target_share", "tgt_share":"target_share",
        "rush_share":"rush_share",
        "yprr":"yprr_proxy",
        "ypc":"ypc",
        "ypa":"qb_ypa",
    }, tag="MySportsFeeds")

    # NFLGSIS
    gsis = _safe_read_csv("data/gsis_player.csv")
    out = _merge_missing_player(out, gsis, ("player","team"), {
        "route_rate":"route_rate", "routes_per_dropback":"route_rate",
        "target_share":"target_share",
        "rush_share":"rush_share",
        "yprr":"yprr_proxy",
        "ypc":"ypc",
        "ypa":"qb_ypa",
    }, tag="NFLGSIS")

    # --- ADD: API-Sports
    apis = _safe_read_csv("data/apisports_player.csv")
    out = _merge_missing_player(out, apis, ("player","team"), {
        "routes_per_dropback":"route_rate", "route_pct":"route_rate",
        "tgt_share":"target_share", "target_share":"target_share",
        "rush_share":"rush_share",
        "yprr":"yprr_proxy",
        "ypc":"ypc",
        "ypa":"qb_ypa",
    }, tag="API-Sports")

    # Final sanitize
    for c in ("target_share","rush_share","route_rate","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","qb_ypa","ypt"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# ---------------------- CLI ----------------------
def cli(season: int) -> int:
    try:
        df = build_player_form(season)
        if df is None or df.empty:
            raise RuntimeError("[player_form] looks empty/stub — check logs/nfl_pbp_error.txt")
    except Exception as e:
        print(f"[player_form] fatal error: {type(e).__name__}: {e}", flush=True)
        df = pd.DataFrame()
    _safe_write(df, OUT)
    print(f"[player_form] wrote rows={len(df)} → {OUT}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    a = ap.parse_args()
    sys.exit(cli(a.season))
