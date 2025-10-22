# scripts/enrich_player_form.py
"""
Enrich player_form.csv with depth-chart roles, positions, and safe fallbacks.

Goals
- Guarantee player_form.csv is non-empty and has the expected schema
- Add/repair team, position, and role (WR1/WR2/SLOT/RB1/TE1, etc.)
- Fill missing route_rate / yprr_proxy when possible
- Never overwrite a non-null value with a fallback (non-destructive)

Inputs (best-effort, all optional except player_form.csv):
- data/player_form.csv                  # base built by make_player_form.py
- outputs/props_raw.csv                 # bootstrap if player_form is empty
- data/roles.csv                        # user-maintained roles (preferred)
- data/depth_chart_espn.csv             # ESP(N) depth charts
- data/depth_chart_ourlads.csv          # OurLads depth charts
- data/pfr_player_positions.csv         # fallback positions (if present)

Output:
- data/player_form.csv                  # enriched in-place
"""

from __future__ import annotations
import os, sys, warnings
import pandas as pd, numpy as np

def _try_load_weekly_rosters(season: int = 2025) -> pd.DataFrame:
    """
    Best-effort weekly roster loader using nflreadpy (preferred) or nfl_data_py.
    Returns normalized columns: player, team, season
    """
    try:
        try:
            import nflreadpy as nflv
            df = nflv.load_weekly_rosters(seasons=[season])
        except Exception:
            import nfl_data_py as nflv  # type: ignore
            df = nflv.import_weekly_rosters([season])  # type: ignore
        if df is None:
            return pd.DataFrame()
        pdf = pd.DataFrame(df).copy()
        pdf.columns = [c.lower() for c in pdf.columns]
        name_col = "player_name" if "player_name" in pdf.columns else ("name" if "name" in pdf.columns else None)
        team_col = "team" if "team" in pdf.columns else ("team_abbr" if "team_abbr" in pdf.columns else None)
        if not name_col or not team_col:
            return pd.DataFrame()
        out = pdf[[name_col, team_col]].rename(columns={name_col: "player", team_col: "team"}).dropna()
        out["player"] = out["player"].astype(str).str.replace(".", "", regex=False).str.strip()
        out["team"]   = out["team"].astype(str).str.upper().str.strip()
        out["season"] = int(season)
        return out.drop_duplicates(subset=["player","team"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

DATA_DIR = "data"
OUTPATH   = os.path.join(DATA_DIR, "player_form.csv")

BASE_COLS = [
    "player","team","season",
    "target_share","rush_share","rz_tgt_share","rz_carry_share",
    "ypt","ypc","yprr_proxy","route_rate",
    "position","role"
]

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def _write_csv(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in BASE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    # preserve additional columns (denominators, etc.)
    return out


def non_destructive_merge(base: pd.DataFrame, add: pd.DataFrame, on, mapping=None) -> pd.DataFrame:
    """
    Merge 'add' into 'base' without overwriting existing non-null values.
    mapping: optional dict {base_col: add_col}
    """
    if add.empty:
        return base
    if mapping is None:
        mapping = {}
    add = add.copy()
    add.columns = [c.lower() for c in add.columns]
    # standardize keys
    if isinstance(on, str):
        on = [on]
    for k in on:
        if k not in base.columns or k not in add.columns:
            return base
    # select only needed columns
    add_cols = on + list(mapping.values() if mapping else [])
    add = add[[c for c in add_cols if c in add.columns]].drop_duplicates()
    merged = base.merge(add, on=on, how="left", suffixes=("","_new"))
    # fill
    for bcol, acol in mapping.items():
        if acol not in merged.columns:
            continue
        newcol = f"{acol}"
        fillcol = f"{acol}"
        # write into base column bcol, but only when base is null
        merged[bcol] = merged[bcol].combine_first(merged[fillcol])
        if fillcol in merged.columns and fillcol not in BASE_COLS:
            # keep the base schema; drop extra
            pass
    # drop any *_new suffix artifacts
    drop_cols = [c for c in merged.columns if c.endswith("_new") and c not in BASE_COLS]
    if drop_cols:
        merged.drop(columns=drop_cols, inplace=True, errors="ignore")
    return merged

def bootstrap_from_props_if_empty(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and df["player"].notna().any():
        return df
    props = _read_csv(os.path.join("outputs","props_raw.csv"))
    if props.empty:
        # nothing to bootstrap with
        return df
    # Try to infer player/team from props_raw
    candidates = []
    # unified player column from receiver/rusher/passer fields
    for pcol in ["player","receiver","rusher","passer","name"]:
        if pcol in props.columns:
            candidates.append(pcol)
    if not candidates:
        return df
    # heuristics: pick first present as "player"
    pcol = candidates[0]
    team_col = None
    for tcol in ["team","posteam","home_team","away_team","team_name"]:
        if tcol in props.columns:
            team_col = tcol; break
    boot = props[[c for c in [pcol,team_col] if c]].dropna().copy()
    boot = boot.rename(columns={pcol:"player", team_col:"team"}) if team_col else boot.rename(columns={pcol:"player"})
    boot["player"] = boot["player"].astype(str)
    if "team" in boot:
        boot["team"] = boot["team"].astype(str)
    boot = boot.drop_duplicates(subset=["player"] if "team" not in boot else ["player","team"])
    # minimal schema
    boot["season"] = boot.get("season", pd.Series(np.nan, index=boot.index))
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","ypt","ypc","yprr_proxy","route_rate","position","role"]:
        boot[c] = np.nan
    boot = boot[BASE_COLS]
    return boot

def load_roles_priority() -> pd.DataFrame:
    """
    Load roles from most trustworthy source to least:
    1) data/roles.csv               (user-authored)
    2) data/depth_chart_espn.csv    (parsed from ESPN depth)
    3) data/depth_chart_ourlads.csv (parsed from OurLads)
    Returns: DataFrame with columns at least ['player','team','position','role']
    """
    # 1) roles.csv
    roles = _read_csv(os.path.join(DATA_DIR,"roles.csv"))
    if not roles.empty and {"player","team","role"}.issubset(roles.columns):
        # try to keep a 'position' column if present
        if "position" not in roles.columns:
            roles["position"] = np.nan
        # normalize role strings
        roles["role"] = roles["role"].astype(str).str.upper()
        return roles[["player","team","position","role"]].drop_duplicates()

    # 2) espn depth
    espn = _read_csv(os.path.join(DATA_DIR,"depth_chart_espn.csv"))
    if not espn.empty:
        # expected columns: player, team, position, depth, slot_flag?
        # Construct a role from position + depth (heuristic)
        espn["position"] = espn.get("position", np.nan)
        espn["depth"] = espn.get("depth", np.nan)
        def _role_from_row(r):
            pos = str(r.get("position") or "").upper()
            dep = r.get("depth")
            if pos.startswith("WR"):
                if dep == 1: return "WR1"
                if dep == 2: return "WR2"
                if dep == 3: return "WR3"
                return "WR"
            if pos.startswith("RB"):
                if dep == 1: return "RB1"
                if dep == 2: return "RB2"
                return "RB"
            if pos.startswith("TE"):
                if dep == 1: return "TE1"
                if dep == 2: return "TE2"
                return "TE"
            if pos.startswith("QB"):
                return "QB1" if dep == 1 else "QB"
            return pos or np.nan
        espn["role"] = espn.apply(_role_from_row, axis=1)
        espn["role"] = espn["role"].astype(str).str.upper()
        espn = espn[["player","team","position","role"]].drop_duplicates()
        return espn

    # 3) ourlads
    ol = _read_csv(os.path.join(DATA_DIR,"depth_chart_ourlads.csv"))
    if not ol.empty:
        ol["position"] = ol.get("position", np.nan)
        ol["depth"] = ol.get("depth", np.nan)
        ol["role"] = ol.apply(lambda r: (str(r.get("position") or "").upper() + str(r.get("depth") or "")), axis=1)
        # normalize to common roles
        def _normalize(role):
            role = str(role).upper()
            if role.startswith("WR1"): return "WR1"
            if role.startswith("WR2"): return "WR2"
            if role.startswith("WR3"): return "WR3"
            if role.startswith("RB1"): return "RB1"
            if role.startswith("RB2"): return "RB2"
            if role.startswith("TE1"): return "TE1"
            if role.startswith("TE2"): return "TE2"
            if role.startswith("QB1"): return "QB1"
            if role.startswith("WR"):  return "WR"
            if role.startswith("RB"):  return "RB"
            if role.startswith("TE"):  return "TE"
            if role.startswith("QB"):  return "QB"
            return role
        ol["role"] = ol["role"].apply(_normalize)
        ol = ol[["player","team","position","role"]].drop_duplicates()
        return ol

    # empty
    return pd.DataFrame(columns=["player","team","position","role"])

def fill_route_rate_and_yprr(df: pd.DataFrame) -> pd.DataFrame:
    """
    When route_rate is missing:
    - WR/TE: estimate from target share as a crude proxy (cap 0.95)
    - RB: use min(target_share*0.6, 0.65)
    - QB: ignore
    Fill yprr_proxy from ypt for receivers if missing.
    """
    out = df.copy()

    # normalize position for logic
    pos = out.get("position", pd.Series(np.nan, index=out.index)).astype(str).str.upper()

    # route_rate rules
    rr = out.get("route_rate", pd.Series(np.nan, index=out.index)).astype(float)
    tshare = out.get("target_share", pd.Series(0.0, index=out.index)).astype(float)

    # WR/TE heuristic
    mask_wrte = pos.str.startswith("WR") | pos.str.startswith("TE")
    rr_wrte = tshare.mul(1.15).clip(lower=0.05, upper=0.95)  # 5%–95% clamp
    rr = rr.mask(mask_wrte & rr.isna(), rr_wrte)

    # RB heuristic
    mask_rb = pos.str.startswith("RB")
    rr_rb = tshare.mul(0.6).clip(upper=0.65)
    rr = rr.mask(mask_rb & rr.isna(), rr_rb)

    out["route_rate"] = rr

    # yprr_proxy fallback for WR/TE if missing but ypt exists
    yprr = out.get("yprr_proxy", pd.Series(np.nan, index=out.index)).astype(float)
    ypt   = out.get("ypt", pd.Series(np.nan, index=out.index)).astype(float)
    yprr  = yprr.mask(mask_wrte & yprr.isna() & ypt.notna(), ypt)
    out["yprr_proxy"] = yprr

    return out

def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pf = _read_csv(OUTPATH)
        if pf.empty or pf["player"].dropna().empty:
            print("[enrich_player_form] player_form empty → bootstrapping from props_raw.csv ...")
            pf = bootstrap_from_props_if_empty(pf)

        # Ensure base schema
        pf = ensure_schema(pf)

    # harmonize names with make_player_form (no nukes)
    if "target_share" not in pf.columns and "tgt_share" in pf.columns:
        pf["target_share"] = pf["tgt_share"]
    if "yprr_proxy" not in pf.columns and "yprr" in pf.columns:
        pf["yprr_proxy"] = pf["yprr"]
    if "rz_carry_share" not in pf.columns and "rz_rush_share" in pf.columns:
        pf["rz_carry_share"] = pf["rz_rush_share"]

        # Merge roles/positions from priority sources (non-destructive)
        roles = load_roles_priority()
        if not roles.empty:
            pf = non_destructive_merge(
                pf, roles, on=["player","team"],
                mapping={
                    "position":"position",
                    "role":"role"
                }
            )

        # Fallback positions from PFR if present
        pfr_pos = _read_csv(os.path.join(DATA_DIR,"pfr_player_positions.csv"))
        if not pfr_pos.empty and {"player","position"}.issubset(pfr_pos.columns):
            pf = non_destructive_merge(
                pf, pfr_pos, on=["player"],
                mapping={"position":"position"}
            )

        # Fill route_rate and yprr proxies if still missing
        pf = fill_route_rate_and_yprr(pf)
        # --- A) Participation shares (non-destructive; optional) ---
        try:
            # pick a season for the query (fallback to 2025 if unknown)
            season_guess = int(pf["season"].dropna().iloc[0]) if "season" in pf.columns and pf["season"].notna().any() else 2025

            import nflreadpy as _nflv   # will be used if installed; otherwise we silently skip
            part = _nflv.load_participation(seasons=[season_guess])
            part.columns = [c.lower() for c in part.columns]

            team_col = "posteam" if "posteam" in part.columns else ("offense_team" if "offense_team" in part.columns else None)
            if team_col is not None and "player_name" in part.columns:
                p = part.rename(columns={team_col: "team", "player_name": "player"})
                p["team"]   = p["team"].astype(str).str.upper().str.strip()
                p["player"] = p["player"].astype(str).str.replace(".", "", regex=False).str.strip()

                # sum snaps/routes across the season (or available rows)
                g = p.groupby(["team","player"], dropna=False).agg(
                    off_snaps=("offense", "sum") if "offense" in p.columns else ("onfield", "sum"),
                    routes   =("route",   "sum") if "route"   in p.columns else ("routes",  "sum") if "routes" in p.columns else ("onfield","sum")
                ).reset_index()

                # team totals for shares
                tt = g.groupby("team", dropna=False).agg(team_off_snaps=("off_snaps","sum"),
                                                         team_routes   =("routes","sum")).reset_index()
                g = g.merge(tt, on="team", how="left")
                g["snap_share"] = np.where(g["team_off_snaps"] > 0, g["off_snaps"] / g["team_off_snaps"], np.nan)
                g["route_rate"] = np.where(g["team_routes"]   > 0, g["routes"]    / g["team_routes"],   np.nan)

                # merge into pf without overwriting existing non-null values
                pf = pf.merge(g[["team","player","snap_share","route_rate"]],
                              on=["team","player"], how="left", suffixes=("","_part"))
                for col in ["snap_share","route_rate"]:
                    ext = col + "_part"
                    if ext in pf.columns:
                        pf[col] = pf[col].combine_first(pf[ext])
                        pf.drop(columns=[ext], inplace=True)
        except Exception:
            # optional; never fail on participation fetch
            pass
        # --- END A ---
        # --- B) Resolve placeholder roles like WR1/RB1/TE1/QB1 to actual names (optional) ---
        try:
            if "role" in pf.columns:
                tmp = pf.copy()
                for c in ["snap_share","route_rate","target_share"]:
                    if c not in tmp.columns:
                        tmp[c] = np.nan
                # ranking signal to pick WR1/WR2... etc
                tmp["rk"] = tmp["route_rate"].fillna(0)*1.0 + tmp["snap_share"].fillna(0)*0.7 + tmp["target_share"].fillna(0)*0.5
                tmp["posu"] = tmp.get("position", np.nan).astype(str).str.upper()

                mask = tmp["role"].astype(str).str.fullmatch(r"(WR|RB|TE|QB)\d", case=False, na=False)
                for idx, row in tmp.loc[mask].iterrows():
                    fam = str(row["role"])[:2].upper()   # WR/RB/TE/QB
                    try:
                        k = int(str(row["role"])[2:])    # the number (1/2/3…)
                    except Exception:
                        k = 1
                    sub = tmp[tmp["team"] == row["team"]]
                    if fam in ("WR","TE","RB","QB"):
                        sub = sub[sub["posu"].str.startswith(fam)]
                    sub = sub.sort_values("rk", ascending=False)
                    if len(sub) > 0:
                        name = sub["player"].iloc[min(k-1, len(sub)-1)]
                        # only set when current player is missing or still a placeholder
                        if pd.isna(pf.at[idx, "player"]) or str(pf.at[idx, "player"]).upper().startswith((fam)):
                            pf.at[idx, "player"] = name
        except Exception:
            # optional; never fail on mapping
            pass
        # --- END B ---

        # Final tidy: ensure ALL-opponent consensus rows survive de-duplication
        if {"player","team","season"}.issubset(pf.columns) and "opponent" in pf.columns:
            try:
                normalized_opp = (
                    pf.get("opponent", pd.Series([], dtype=object))
                      .fillna("")
                      .astype(str)
                      .str.strip()
                )
                consensus_mask = normalized_opp.eq("") | normalized_opp.str.upper().eq("ALL")
                pf["_opp_priority"] = (~consensus_mask).astype(int)
                pf["_orig_order"] = np.arange(len(pf))
                sort_keys = [
                    c for c in ["player", "team", "season", "_opp_priority", "_orig_order"] if c in pf.columns
                ]
                pf = pf.sort_values(sort_keys, kind="mergesort")
            except Exception:
                pf.drop(columns=[c for c in ["_opp_priority", "_orig_order"] if c in pf.columns], inplace=True, errors="ignore")
        # Final tidy: keep unique player-team-season rows
        pf = pf.drop_duplicates(subset=["player","team","season"], keep="first")
        if "opponent" in pf.columns:
            try:
                normalized_opp = (
                    pf.get("opponent", pd.Series([], dtype=object))
                      .fillna("")
                      .astype(str)
                      .str.strip()
                )
                consensus_mask = normalized_opp.eq("") | normalized_opp.str.upper().eq("ALL")
                if consensus_mask.any():
                    pf.loc[consensus_mask, "opponent"] = "ALL"
            except Exception:
                pass
        if "_opp_priority" in pf.columns:
            pf.drop(columns=["_opp_priority"], inplace=True, errors="ignore")
        if "_orig_order" in pf.columns:
            pf.drop(columns=["_orig_order"], inplace=True, errors="ignore")
# --- ADD: persist a stable player_key for robust downstream joins ---
        try:
            pf["player_key"] = (
                pf.get("player", pd.Series([], dtype=object))
                  .fillna("").astype(str)
                  .str.lower()
                  .str.replace(r"[^a-z0-9]", "", regex=True)
            )
        except Exception:
            pass
# --- END ADD ---

        # Write back
    # Persist a stable player_key for robust downstream joins
    try:
        pf["player_key"] = (
            pf.get("player", pd.Series([], dtype=object))
              .fillna("").astype(str)
              .str.lower()
              .str.replace(r"[^a-z0-9]", "", regex=True)
        )
    except Exception:
        pass

        _write_csv(OUTPATH, pf)
        print(f"[enrich_player_form] Wrote {len(pf)} rows → {OUTPATH}")

if __name__ == "__main__":
    main()
