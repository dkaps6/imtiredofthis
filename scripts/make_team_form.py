#!/usr/bin/env python3
from __future__ import annotations
import sys, os, time, traceback
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/team_form.csv")
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
            "team","def_pass_epa","def_rush_epa","def_sack_rate",
            "pace","proe","light_box_rate","heavy_box_rate",
            "def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","proe_z","light_box_rate_z","heavy_box_rate_z"
        ]).to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0: return pd.Series(0.0, index=s.index)
    return (s - mu) / (sd + 1e-9)

def _merge_missing_team(df: pd.DataFrame, add: pd.DataFrame, on="team", mapping: dict[str, str] | None = None, tag: str = "") -> pd.DataFrame:
    if add.empty:
        return df
    src = add.copy()
    if mapping:
        src = src.rename(columns=mapping)
    cols = [on] + [c for c in (mapping.values() if mapping else []) if c in src.columns]
    cols = [c for c in cols if c in src.columns]
    if len(cols) <= 1:
        return df
    merged = df.merge(src[cols], on=on, how="left", suffixes=("","__prov"))
    for col in cols:
        if col == on: 
            continue
        prov = f"{col}__prov"
        if prov in merged.columns:
            mask = merged[col].isna() & merged[prov].notna()
            if mask.any():
                print(f"[team_form] filled {mask.sum()} rows for {col} from {tag}")
                merged.loc[mask, col] = merged.loc[mask, prov]
            merged.drop(columns=[prov], inplace=True)
    return merged

def _fill_prior_season_missing(df: pd.DataFrame, season: int, cols: list[str]) -> pd.DataFrame:
    """
    Last resort only. Uses last season outputs if available; fills ONLY NaNs.
    Looks in outputs/season_cache/team_form_{season-1}.csv or data/team_form_{season-1}.csv
    """
    prior = _safe_read_csv(Path("outputs/season_cache") / f"team_form_{season-1}.csv")
    if prior.empty:
        prior = _safe_read_csv(Path("data") / f"team_form_{season-1}.csv")
    if prior.empty or "team" not in prior.columns:
        return df
    prior = prior.copy()
    prior["team"] = prior["team"].astype(str).str.upper()
    merged = df.merge(prior[["team"] + [c for c in cols if c in prior.columns]],
                      on="team", how="left", suffixes=("","__prior"))
    for c in cols:
        if c in merged.columns and f"{c}__prior" in merged.columns:
            mask = merged[c].isna() & merged[f"{c}__prior"].notna()
            if mask.any():
                print(f"[team_form] prior-season fallback filled {mask.sum()} rows for {c}")
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
                        print(f"[team_form] NOTE: using season {s} as fallback for {season}", flush=True)
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

def _try_load_participation(season: int) -> pd.DataFrame:
    try:
        import nflreadpy as nfr
        pf = nfr.load_participation([season])  # polars
        return pf.to_pandas()
    except Exception:
        pass
    try:
        import nfl_data_py as nfl
        for fn in ("import_participation", "import_participation_data", "import_ngs_participation"):
            if hasattr(nfl, fn):
                return getattr(nfl, fn)([season])
    except Exception:
        pass
    return pd.DataFrame()

# ---------------------- builder ----------------------
def build_from_nflverse(season: int) -> pd.DataFrame:
    # 1) PBP (real data)
    pbp = _fetch_pbp_with_retry(season)
    pbp = pbp.loc[pbp["season"] == season].copy()

    pbp["defteam"] = pbp["defteam"].astype(str).str.upper()
    pbp["posteam"] = pbp["posteam"].astype(str).str.upper()

    pbp["is_pass"]     = (pbp.get("pass",0)==1) | (pbp.get("pass_attempt",0)==1) | (pbp.get("play_type","")=="pass")
    pbp["is_rush"]     = (pbp.get("rush",0)==1) | (pbp.get("play_type","")=="run")
    pbp["is_dropback"] = (pbp.get("dropback",0)==1)
    pbp["is_sack"]     = (pbp.get("sack",0)==1)

    # --- Defensive EPA (pass / rush)
    def_pass = pbp.loc[pbp["is_pass"]].groupby("defteam")["epa"].mean().rename("def_pass_epa")
    def_rush = pbp.loc[pbp["is_rush"]].groupby("defteam")["epa"].mean().rename("def_rush_epa")

    # --- Defensive sack rate from opponent dropbacks
    drop  = pbp.loc[pbp["is_dropback"]].groupby("defteam")["is_dropback"].count().rename("db")
    sacks = pbp.loc[pbp["is_sack"]].groupby("defteam")["is_sack"].count().rename("sacks")
    sack_rate = (sacks / drop).replace([np.inf,-np.inf], np.nan).rename("def_sack_rate")

    # --- Neutral PACE (sec/snap) per OFFENSE team using inter-snap deltas
    neutral = pbp.loc[
        (pbp["qtr"] <= 3) &
        (pbp["down"].between(1, 3, inclusive="both")) &
        (pbp["score_differential"].between(-7, 7, inclusive="both"))
    ].copy()

    # use game clock: higher to lower -> negative diff; take absolute
    neutral.sort_values(["game_id","posteam","qtr","game_seconds_remaining"],
                        ascending=[True, True, True, False], inplace=True)
    neutral["delta"] = neutral.groupby(["game_id","posteam"])["game_seconds_remaining"].diff(-1).abs()
    neutral["delta"] = neutral["delta"].clip(lower=5, upper=90).fillna(40)
    pace_off = neutral.groupby("posteam")["delta"].mean().rename("pace")  # seconds per snap (lower = faster)

    # --- PROE from neutral: mean(pass - xpass) if xpass available; else delta vs league neutral PR
    if "xpass" in pbp.columns:
        nu = neutral.copy()
        for c in ["is_pass","xpass"]:
            nu[c] = pd.to_numeric(nu[c], errors="coerce")
        proe_off = (nu["is_pass"] - nu["xpass"]).groupby(nu["posteam"]).mean().rename("proe")
    else:
        nu = neutral.copy()
        team_pr   = nu.groupby("posteam")["is_pass"].mean().rename("team_neutral_pr")
        league_pr = float(team_pr.mean()) if len(team_pr) else 0.55
        proe_off  = (team_pr - league_pr).rename("proe")

    # Build frame keyed by team (uppercase)
    pace = pace_off.reset_index().rename(columns={"posteam":"team"})
    proe = proe_off.reset_index().rename(columns={"posteam":"team"})
    pace["team"] = pace["team"].astype(str).str.upper()
    proe["team"] = proe["team"].astype(str).str.upper()

    df = pd.concat([def_pass, def_rush, sack_rate], axis=1).reset_index().rename(columns={"defteam":"team"})
    df["team"] = df["team"].astype(str).str.upper()
    df = df.merge(pace, on="team", how="left").merge(proe, on="team", how="left")

    # 2) Participation → light/heavy box (best-effort)
    try:
        part = _try_load_participation(season)
        needed = {"game_id","play_id","defenders_in_box"}
        if isinstance(part, pd.DataFrame) and len(part) and needed.issubset(set(part.columns)):
            p = part[["game_id","play_id","defenders_in_box"]].copy()
            p = p.merge(pbp[["game_id","play_id","defteam","is_rush"]], on=["game_id","play_id"], how="inner")
            p = p[p["is_rush"]==True].copy()
            p["defenders_in_box"] = pd.to_numeric(p["defenders_in_box"], errors="coerce")
            p["light"] = (p["defenders_in_box"] <= 6).astype(float)
            p["heavy"] = (p["defenders_in_box"] >= 8).astype(float)
            box = p.groupby("defteam")[["light","heavy"]].mean(numeric_only=True).rename(columns={
                "light":"light_box_rate","heavy":"heavy_box_rate"
            })
            df = df.merge(box.reset_index().rename(columns={"defteam":"team"}), on="team", how="left")
    except Exception as e:
        print(f"[team_form] participation enrich skipped: {type(e).__name__}: {e}", flush=True)

    # 3) PFR team enrich (observed box rates etc) if present
    try:
        enrich_path = Path("data/pfr_team_enrich.csv")
        if enrich_path.exists():
            ten = pd.read_csv(enrich_path)
            ten = ten.rename(columns={
                "team_abbr":"team",
                "light_box_rate":"light_box_rate_obs",
                "heavy_box_rate":"heavy_box_rate_obs",
                "prwr":"pass_rush_win_rate",
                "press_rate":"pressure_rate",
                "rsr":"run_stop_rate",
                "man_rate":"man_coverage_rate",
                "zone_rate":"zone_coverage_rate",
            })
            keep = ["team","light_box_rate_obs","heavy_box_rate_obs",
                    "pass_rush_win_rate","pressure_rate","run_stop_rate",
                    "man_coverage_rate","zone_coverage_rate"]
            ten = ten[[c for c in keep if c in ten.columns]].copy()
            for c in ten.columns:
                if c != "team":
                    ten[c] = pd.to_numeric(ten[c], errors="coerce")
            df = df.merge(ten, on="team", how="left")
            if "light_box_rate_obs" in df.columns:
                df["light_box_rate"] = df["light_box_rate"].fillna(df["light_box_rate_obs"])
            if "heavy_box_rate_obs" in df.columns:
                df["heavy_box_rate"] = df["heavy_box_rate"].fillna(df["heavy_box_rate_obs"])
    except Exception as e:
        print(f"[team_form] PFR enrich skipped: {type(e).__name__}: {e}", flush=True)

    # 4) External provider fallbacks (fill only remaining NaNs)
    espn = _safe_read_csv("data/espn_team.csv")
    df = _merge_missing_team(df, espn, "team", {
        "def_sack_rate":"def_sack_rate",
        "pace":"pace", "pace_sec_per_snap":"pace",
        "proe":"proe",
        "light_box_rate":"light_box_rate",
        "heavy_box_rate":"heavy_box_rate",
    }, tag="ESPN")

    msf = _safe_read_csv("data/msf_team.csv")
    df = _merge_missing_team(df, msf, "team", {
        "def_sack_rate":"def_sack_rate",
        "pace":"pace","pace_sec_per_snap":"pace",
        "proe":"proe",
        "light_box_rate":"light_box_rate",
        "heavy_box_rate":"heavy_box_rate",
    }, tag="MySportsFeeds")

    gsis = _safe_read_csv("data/gsis_team.csv")
    df = _merge_missing_team(df, gsis, "team", {
        "def_sack_rate":"def_sack_rate",
        "pace":"pace",
        "proe":"proe",
        "light_box_rate":"light_box_rate",
        "heavy_box_rate":"heavy_box_rate",
    }, tag="NFLGSIS")

    apis = _safe_read_csv("data/apisports_team.csv")
    df = _merge_missing_team(df, apis, "team", {
        "def_sack_rate":"def_sack_rate",
        "pace":"pace",
        "proe":"proe",
        "light_box_rate":"light_box_rate",
        "heavy_box_rate":"heavy_box_rate",
    }, tag="API-Sports")

    # 5) Last-resort from prior season (only if still NaN)
    df = _fill_prior_season_missing(df, season, [
        "def_sack_rate","pace","proe","light_box_rate","heavy_box_rate"
    ])

    # 6) z-scores
    for c in ["def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","light_box_rate","heavy_box_rate"]:
        if c in df.columns:
            df[f"{c}_z"] = _z(df[c])

    # 7) ensure schema stability
    cols = ["team","def_pass_epa","def_rush_epa","def_sack_rate",
            "pace","proe","light_box_rate","heavy_box_rate",
            "def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","proe_z","light_box_rate_z","heavy_box_rate_z"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

# ---------------------- CLI ----------------------
def cli(season: int) -> int:
    try:
        df = build_from_nflverse(season)
        if os.getenv("NFL_FORM_STRICT") == "1":
            if df is None or df.empty or df["team"].nunique() < 8:
                raise RuntimeError("[team_form] looks empty/stub — check logs/nfl_pbp_error.txt")
    except Exception as e:
        print(f"[team_form] fatal error: {type(e).__name__}: {e}", flush=True)
        df = pd.DataFrame()
    _safe_write(df, OUT)
    print(f"[team_form] wrote rows={len(df)} → {OUT}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    a = ap.parse_args()
    sys.exit(cli(a.season))
