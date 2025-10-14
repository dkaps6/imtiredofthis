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

def _merge_missing_player(df: pd.DataFrame, add: pd.DataFrame, on=("player","team"), mapping: dict[str, str] | None = None, tag: str = "") -> pd.DataFrame:
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
            merged.drop(columns=[prov], inplace=True)
    return merged

def _fill_prior_season_missing(df: pd.DataFrame, season: int, cols: list[str]) -> pd.DataFrame:
    prior = _safe_read_csv(Path("outputs/season_cache") / f"player_form_{season-1}.csv")
    if prior.empty:
        prior = _safe_read_csv(Path("data") / f"player_form_{season-1}.csv")
    if prior.empty or not {"player","team"}.issubset(prior.columns):
        return df
    prior = prior.copy()
    prior["team"] = prior["team"].astype(str).str.upper()
    merged = df.merge(prior[["player","team"] + [c for c in cols if c in prior.columns]],
                      on=["player","team"], how="left", suffixes=("","__prior"))
    for c in cols:
        if c in merged.columns and f"{c}__prior" in merged.columns:
            mask = merged[c].isna() & merged[f"{c}__prior"].notna()
            if mask.any():
                print(f"[player_form] prior-season fallback filled {mask.sum()} rows for {c}")
                merged.loc[mask, c] = merged.loc[mask, f"{c}__prior"]
            merged.drop(columns=[f"{c}__prior"], inplace=True)
    return merged

# --- PBP fetcher: prefer nflreadpy (2025), fall back to nfl_data_py; 404 -> prior season ---
def _fetch_pbp_with_retry(season: int, tries: int = 3, wait: int = 4) -> pd.DataFrame:
    seasons_to_try = [season, season - 1, season - 2]
    last = None
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
def build_from_nflverse(season: int) -> pd.DataFrame:
    pbp = _fetch_pbp_with_retry(season)
    pbp = pbp.loc[pbp["season"]==season].copy()

    pbp["posteam"] = pbp["posteam"].astype(str).str.upper()
    pbp["receiver"] = pbp.get("receiver_player_name","").fillna("").astype(str)
    pbp["rusher"]   = pbp.get("rusher_player_name","").fillna("").astype(str)
    pbp["passer"]   = pbp.get("passer_player_name","").fillna("").astype(str)

    pbp["is_pass"] = (pbp.get("pass",0)==1) | (pbp.get("pass_attempt",0)==1) | (pbp.get("play_type","")=="pass")
    pbp["is_rush"] = (pbp.get("rush",0)==1) | (pbp.get("play_type","")=="run")
    pbp["in_rz"]   = (pbp.get("yardline_100").fillna(100) <= 20)

    targs = pbp.loc[pbp["is_pass"] & (pbp["receiver"]!=""), ["game_id","posteam","receiver"]].copy()
    targs["targets"] = 1
    team_tgts = targs.groupby(["game_id","posteam"])["targets"].sum().rename("team_targets")
    rec_tgts  = targs.groupby(["game_id","posteam","receiver"])["targets"].sum().rename("player_targets")

    rush = pbp.loc[pbp["is_rush"] & (pbp["rusher"]!=""), ["game_id","posteam","rusher","yards_gained"]].copy()
    rush["rush_att"] = 1
    team_rush = rush.groupby(["game_id","posteam"])["rush_att"].sum().rename("team_rush_att")
    ply_rush  = rush.groupby(["game_id","posteam","rusher"])["rush_att"].sum().rename("player_rush_att")
    ply_rush_yds = rush.groupby(["game_id","posteam","rusher"])["yards_gained"].sum().rename("rush_yards")

    rec_yards = pbp.loc[pbp["receiver"]!=""].groupby(["game_id","posteam","receiver"])["yards_gained"].sum().rename("rec_yards")

    rz_targs = pbp.loc[pbp["is_pass"] & pbp["in_rz"] & (pbp["receiver"]!=""), ["game_id","posteam","receiver"]].copy()
    rz_targs["rz_tgt"] = 1
    ply_rz_tgt = rz_targs.groupby(["game_id","posteam","receiver"])["rz_tgt"].sum().rename("player_rz_tgt")
    team_rz_tgt = rz_targs.groupby(["game_id","posteam"])["rz_tgt"].sum().rename("team_rz_tgt")

    rz_rush = pbp.loc[pbp["is_rush"] & pbp["in_rz"] & (pbp["rusher"]!=""), ["game_id","posteam","rusher"]].copy()
    rz_rush["rz_carry"] = 1
    ply_rz_carry = rz_rush.groupby(["game_id","posteam","rusher"])["rz_carry"].sum().rename("player_rz_carry")
    team_rz_carry = rz_rush.groupby(["game_id","posteam"])["rz_carry"].sum().rename("team_rz_carry")

    pass_att = pbp.loc[pbp["is_pass"] & (pbp["passer"]!=""), ["game_id","posteam","passer","yards_gained"]].copy()
    pass_att["att"] = 1
    team_pass_yds = pass_att.groupby(["game_id","posteam"])["yards_gained"].sum().rename("team_pass_yds")
    team_atts = pass_att.groupby(["game_id","posteam"])["att"].sum().rename("team_pass_att")
    team_ypa = (team_pass_yds.groupby("posteam").sum() / team_atts.groupby("posteam").sum()).rename("qb_ypa_all")

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

    # Core shares
    agg["target_share"]   = (agg["player_targets"] / agg["team_targets"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rush_share"]     = (agg["player_rush_att"] / agg["team_rush_att"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rz_tgt_share"]   = (agg["player_rz_tgt"] / agg["team_rz_tgt"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["rz_carry_share"] = (agg["player_rz_carry"] / agg["team_rz_carry"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # Efficiency
    # ypt (explicit)
    agg["ypt"]        = (agg["rec_yards"] / np.where(agg["player_targets"]>0, agg["player_targets"], np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    agg["ypc"]        = (agg["rush_yards"] / np.where(agg["player_rush_att"]>0, agg["player_rush_att"], np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    # yprr_proxy will be overwritten below when pass_snaps are available, else remains yds/targets fallback
    agg["yprr_proxy"] = agg["ypt"].copy()

    # Position heuristic (kept)
    agg["position"] = np.where(agg["player_rush_att"] > agg["player_targets"], "RB", "WR")
    agg["route_rate"] = np.nan
    agg["role"] = ""

    team_ypa = team_ypa.reset_index().rename(columns={"posteam":"team","qb_ypa_all":"qb_ypa"})
    out = agg.merge(team_ypa, on="team", how="left")
    out["qb_ypa"] = out["qb_ypa"].fillna(out["qb_ypa"].mean() if np.isfinite(out["qb_ypa"].mean()) else 6.9)

    # --- NEW: Build a real routes proxy using participation pass snaps (if available)
    pass_snaps_cols = pd.DataFrame(columns=["team","player_norm","pass_snaps","team_pass_snaps"])
    try:
        # Try both stacks you already support in team_form
        try:
            import nflreadpy as nfr
            pf = nfr.load_participation([season]).to_pandas()
        except Exception:
            pf = None
        if pf is None or pf.empty:
            import nfl_data_py as nfl
            for fn in ("import_participation", "import_participation_data", "import_ngs_participation"):
                if hasattr(nfl, fn):
                    pf = getattr(nfl, fn)([season])
                    break
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
    out["player_norm"] = out["player"].astype(str).str.lower().str.replace(r"[.'’]", "", regex=True).str.strip()
    if not pass_snaps_cols.empty:
        out = out.merge(pass_snaps_cols, on=["team","player_norm"], how="left")
        # route_rate if missing: pass_snaps / team_pass_snaps
        rr = (out["pass_snaps"] / out["team_pass_snaps"]).replace([np.inf,-np.inf], np.nan)
        out["route_rate"] = out["route_rate"].where(out["route_rate"].notna(), rr)
        # yprr_proxy real: rec_yards / pass_snaps (falls back to ypt when snaps=0)
        yprr_real = (out["rec_yards"] / np.where(out["pass_snaps"]>0, out["pass_snaps"], np.nan)).replace([np.inf,-np.inf], np.nan)
        out["yprr_proxy"] = out["yprr_proxy"].where(out["yprr_proxy"].notna(), yprr_real)
    out.drop(columns=["player_norm","pass_snaps","team_pass_snaps"], inplace=True, errors="ignore")

    out = out.rename(columns={"player":"player"})[[
        "player","team","position","role",
        "target_share","rush_share","route_rate",
        "rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","qb_ypa","ypt"
    ]].sort_values(["team","player"]).reset_index(drop=True)

    # Optional: PFR/mirrors enrich for route/adot if provided
    try:
        enrich_path = Path("data/pfr_player_enrich.csv")
        if enrich_path.exists():
            enrich = pd.read_csv(enrich_path)
            rename_map = {"team_abbr":"team","routes_db":"routes_per_dropback","aDOT":"aDOT"}
            enrich = enrich.rename(columns={k:v for k,v in rename_map.items() if k in enrich.columns})
            keep = ["player","team","routes_per_dropback","route_rate","aDOT","slot_rate","snap_share"]
            enrich = enrich[[c for c in keep if c in enrich.columns]].copy()
            for col in ["routes_per_dropback","route_rate","slot_rate","snap_share"]:
                if col in enrich.columns:
                    enrich[col] = pd.to_numeric(enrich[col], errors="coerce")
            out = out.merge(enrich, on=["player","team"], how="left", suffixes=("","_enrich"))
            if "routes_per_dropback" in out.columns:
                out["route_rate"] = out["route_rate"].fillna(out["routes_per_dropback"])
            if "route_rate_enrich" in out.columns:
                out["route_rate"] = out["route_rate"].fillna(out["route_rate_enrich"])
    except Exception as e:
        print(f"[player_form] PFR enrich skipped: {type(e).__name__}: {e}", flush=True)

    # 4) External provider fallbacks (fill only remaining NaNs) — preserved
    espn = _safe_read_csv("data/espn_player.csv")
    out = _merge_missing_player(out, espn, ("player","team"), {
        "routes_db":"route_rate","route_rate":"route_rate",
        "target_share":"target_share","targets_share":"target_share",
        "rush_share":"rush_share"
    }, tag="ESPN")

    msf = _safe_read_csv("data/msf_player.csv")
    out = _merge_missing_player(out, msf, ("player","team"), {
        "route_rate":"route_rate",
        "target_share":"target_share",
        "rush_share":"rush_share"
    }, tag="MySportsFeeds")

    gsis = _safe_read_csv("data/gsis_player.csv")
    out = _merge_missing_player(out, gsis, ("player","team"), {
        "route_rate":"route_rate",
        "target_share":"target_share",
        "rush_share":"rush_share"
    }, tag="NFLGSIS")

    apis = _safe_read_csv("data/apisports_player.csv")
    out = _merge_missing_player(out, apis, ("player","team"), {
        "route_rate":"route_rate",
        "target_share":"target_share",
        "rush_share":"rush_share"
    }, tag="API-Sports")

    # 5) Last-resort prior season (only if still NaN) — preserved
    out = _fill_prior_season_missing(out, season, ["route_rate","target_share","rush_share","ypt","yprr_proxy"])

    return out

# ---------------------- CLI ----------------------
def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
        if os.getenv("NFL_FORM_STRICT") == "1":
            if df is None or df.empty or df["team"].nunique() < 8 or len(df) < 50:
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
