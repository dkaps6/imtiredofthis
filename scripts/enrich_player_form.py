# scripts/enrich_player_form.py
"""
Enrich player_form.csv with depth-chart roles/positions and safe fallbacks.

Surgical guarantees:
- Keep existing columns and per-opponent rows (no nukes).
- Never overwrite a non-null with a fallback.
- Force all share/rate metrics (esp. RZ) to 0.0 instead of NaN.
- Merge roles/positions using normalized player_key across sources.
- Drop rows with blank names (these tank validator coverage).
- Filter obvious non-skill positions (OL/DEF) only when position is known.
- Emit a post-enrich unmatched report for manual curation.

Outputs:
- data/player_form.csv                       (enriched)
- data/unmatched_roles_after_enrich.csv      (remaining gaps)
"""

from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPATH  = os.path.join(DATA_DIR, "player_form.csv")
UNMATCHED_OUT = os.path.join(DATA_DIR, "unmatched_roles_after_enrich.csv")

# -------------------- IO --------------------

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

# -------------------- Name normalization --------------------

_SUFFIXES = {"jr": "jr", "sr": "sr", "ii": "ii", "iii": "iii", "iv": "iv"}

def normalize_name(s: str) -> str:
    """Lowercase; remove punctuation/whitespace; keep suffix tokens at end."""
    if not isinstance(s, str):
        return ""
    x = s.strip().lower()
    for ch in [".", ",", "'", "’", "-", "–", "—", "(", ")", "/"]:
        x = x.replace(ch, " ")
    toks = [t for t in x.split() if t]
    if not toks:
        return ""
    suf = ""
    if toks[-1] in _SUFFIXES:
        suf = toks[-1]
        toks = toks[:-1]
    key = "".join(toks + ([suf] if suf else []))
    return key

def add_player_key(df: pd.DataFrame, player_col: str = "player") -> pd.DataFrame:
    out = df.copy()
    if player_col not in out.columns:
        out[player_col] = ""
    out["player_key"] = out[player_col].fillna("").astype(str).apply(normalize_name)
    return out

# -------------------- Schema helpers --------------------

BASE_COLS = [
    "player","team","season","opponent",
    "target_share","rush_share","route_rate","yprr_proxy",
    "ypt","ypc","ypa","receptions_per_target",
    "rz_share","rz_tgt_share","rz_rush_share","rz_carry_share",
    "position","role","player_key"
]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in BASE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out

def alias_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # target share
    if "target_share" not in out.columns and "tgt_share" in out.columns:
        out["target_share"] = pd.to_numeric(out["tgt_share"], errors="coerce")
    if "tgt_share" not in out.columns and "target_share" in out.columns:
        out["tgt_share"] = pd.to_numeric(out["target_share"], errors="coerce")
    # yprr proxy
    if "yprr_proxy" not in out.columns and "yprr" in out.columns:
        out["yprr_proxy"] = pd.to_numeric(out["yprr"], errors="coerce")
    if "yprr" not in out.columns:
        if "yprr_proxy" in out.columns:
            out["yprr"] = pd.to_numeric(out["yprr_proxy"], errors="coerce")
        elif "ypt" in out.columns:
            out["yprr"] = pd.to_numeric(out["ypt"], errors="coerce")
        else:
            out["yprr"] = np.nan
    # RZ carry/rush naming
    if "rz_carry_share" not in out.columns and "rz_rush_share" in out.columns:
        out["rz_carry_share"] = pd.to_numeric(out["rz_rush_share"], errors="coerce")
    if "rz_rush_share" not in out.columns and "rz_carry_share" in out.columns:
        out["rz_rush_share"] = pd.to_numeric(out["rz_carry_share"], errors="coerce")
    return out

def zero_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Always 0.0 default for RZ
    for c in ["rz_share","rz_tgt_share","rz_rush_share","rz_carry_share"]:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    # Other share/rate fields
    rate_cols = [
        "target_share","tgt_share","route_rate","snap_share","rush_share",
        "yprr_proxy","yprr","ypt","ypc","ypa","receptions_per_target"
    ]
    for c in rate_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    # Clamp proportions
    prop_cols = ["target_share","tgt_share","route_rate","snap_share","rush_share",
                 "rz_share","rz_tgt_share","rz_rush_share","rz_carry_share","receptions_per_target"]
    for c in prop_cols:
        if c in out.columns:
            out[c] = out[c].clip(0.0, 1.0)
    return out

# -------------------- Non-destructive merge --------------------

def nondestructive_merge(base: pd.DataFrame, add: pd.DataFrame, on, mapping=None) -> pd.DataFrame:
    if add.empty:
        return base
    if mapping is None:
        mapping = {}
    if isinstance(on, str):
        on = [on]
    for k in on:
        if k not in base.columns or k not in add.columns:
            return base
    add = add.copy()
    add.columns = [c.lower() for c in add.columns]
    select_cols = on + list(mapping.values())
    add = add[[c for c in select_cols if c in add.columns]].drop_duplicates()
    merged = base.merge(add, on=on, how="left", suffixes=("","_new"))
    for bcol, acol in mapping.items():
        if acol in merged.columns:
            merged[bcol] = merged[bcol].combine_first(merged[acol])
    drop_cols = [c for c in merged.columns if c.endswith("_new")]
    if drop_cols:
        merged.drop(columns=drop_cols, inplace=True, errors="ignore")
    return merged

# -------------------- Roles/positions loaders --------------------

def load_roles_priority() -> pd.DataFrame:
    # roles.csv
    roles = _read_csv(os.path.join(DATA_DIR, "roles.csv"))
    if not roles.empty and {"player","team","role"}.issubset(roles.columns):
        roles = add_player_key(roles, "player")
        if "position" not in roles.columns:
            roles["position"] = np.nan
        roles["role"] = roles["role"].astype(str).str.upper()
        return roles[["player","player_key","team","position","role"]].drop_duplicates()

    # ESPN
    espn = _read_csv(os.path.join(DATA_DIR, "depth_chart_espn.csv"))
    if not espn.empty and {"player","team"}.issubset(espn.columns):
        espn = add_player_key(espn, "player")
        espn["position"] = espn.get("position", np.nan)
        espn["depth"]    = espn.get("depth", np.nan)

        def _role_from_row(r):
            pos = str(r.get("position") or "").upper()
            dep = r.get("depth")
            if pos.startswith("WR"):
                return "WR1" if dep == 1 else "WR2" if dep == 2 else "WR3" if dep == 3 else "WR"
            if pos.startswith("RB"):
                return "RB1" if dep == 1 else "RB2" if dep == 2 else "RB"
            if pos.startswith("TE"):
                return "TE1" if dep == 1 else "TE2" if dep == 2 else "TE"
            if pos.startswith("QB"):
                return "QB1" if dep == 1 else "QB"
            return pos or np.nan

        espn["role"] = espn.apply(_role_from_row, axis=1).astype(str).str.upper()
        return espn[["player","player_key","team","position","role"]].drop_duplicates()

    # OurLads
    ol = _read_csv(os.path.join(DATA_DIR, "depth_chart_ourlads.csv"))
    if not ol.empty and {"player","team"}.issubset(ol.columns):
        ol = add_player_key(ol, "player")
        ol["position"] = ol.get("position", np.nan)
        ol["depth"]    = ol.get("depth", np.nan)

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

        ol["role"] = (ol.get("position","").astype(str).str.upper() + ol.get("depth","").astype(str)).apply(_normalize)
        return ol[["player","player_key","team","position","role"]].drop_duplicates()

    return pd.DataFrame(columns=["player","player_key","team","position","role"])

# -------------------- Heuristic fills --------------------

def fill_route_rate_and_yprr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pos = out.get("position", pd.Series(np.nan, index=out.index)).astype(str).str.upper()
    rr = pd.to_numeric(out.get("route_rate", np.nan), errors="coerce")
    tshare = pd.to_numeric(out.get("target_share", 0.0), errors="coerce").fillna(0.0)

    mask_wrte = pos.str.startswith("WR") | pos.str.startswith("TE")
    rr_wrte = (tshare * 1.15).clip(0.05, 0.95)
    rr = rr.mask(mask_wrte & rr.isna(), rr_wrte)

    mask_rb = pos.str.startswith("RB")
    rr_rb = (tshare * 0.6).clip(upper=0.65)
    rr = rr.mask(mask_rb & rr.isna(), rr_rb)

    out["route_rate"] = rr.fillna(0.0)

    yprr = pd.to_numeric(out.get("yprr_proxy", np.nan), errors="coerce")
    ypt  = pd.to_numeric(out.get("ypt", np.nan), errors="coerce")
    yprr = yprr.mask(mask_wrte & yprr.isna() & ypt.notna(), ypt)
    out["yprr_proxy"] = yprr.fillna(0.0)
    return out

# -------------------- Main --------------------

def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pf = _read_csv(OUTPATH)
        if pf.empty:
            print("[enrich_player_form] WARNING: data/player_form.csv is empty before enrich.")
            return

        pf = add_player_key(pf, "player")
        pf = ensure_schema(pf)
        pf = alias_metrics(pf)
        pf = zero_and_clip(pf)

        roles = load_roles_priority()
        if not roles.empty:
            roles["team"] = roles["team"].astype(str).str.upper().str.strip()
            pf["team"] = pf["team"].astype(str).str.upper().str.strip()

            # primary: by key
            pf = nondestructive_merge(
                pf, roles.rename(columns={"position":"position_key","role":"role_key"}),
                on=["team","player_key"],
                mapping={"position":"position_key","role":"role_key"}
            )
            # fallback: raw player string
            pf = nondestructive_merge(
                pf, roles.rename(columns={"position":"position_raw","role":"role_raw"}),
                on=["player","team"],
                mapping={"position":"position_raw","role":"role_raw"}
            )

        # participation-derived proxies (best-effort; non-fatal)
        try:
            season_guess = int(pf["season"].dropna().iloc[0])
            import nflreadpy as _nflv  # type: ignore
            part = _nflv.load_participation(seasons=[season_guess])
            part.columns = [c.lower() for c in part.columns]
            team_col = "posteam" if "posteam" in part.columns else ("offense_team" if "offense_team" in part.columns else None)
            if team_col is not None and "player_name" in part.columns:
                p = part.rename(columns={team_col: "team", "player_name": "player"})
                p["team"] = p["team"].astype(str).str.upper().str.strip()
                p = add_player_key(p, "player")
                g = p.groupby(["team","player_key"], dropna=False).agg(
                    off_snaps=("offense","sum") if "offense" in p.columns else ("onfield","sum"),
                    routes=("route","sum") if "route" in p.columns else ("routes","sum") if "routes" in p.columns else ("onfield","sum"),
                ).reset_index()
                tt = g.groupby("team", dropna=False).agg(team_off_snaps=("off_snaps","sum"),
                                                         team_routes=("routes","sum")).reset_index()
                g = g.merge(tt, on="team", how="left")
                g["snap_share"] = np.where(g["team_off_snaps"]>0, g["off_snaps"]/g["team_off_snaps"], 0.0)
                g["route_rate_part"] = np.where(g["team_routes"]>0, g["routes"]/g["team_routes"], 0.0)

                pf = pf.merge(g[["team","player_key","snap_share","route_rate_part"]],
                              on=["team","player_key"], how="left")
                if "snap_share" in pf.columns:
                    pf["snap_share"] = pd.to_numeric(pf["snap_share"], errors="coerce").fillna(0.0)
                else:
                    pf["snap_share"] = 0.0
                pf["route_rate"] = pd.to_numeric(pf["route_rate"], errors="coerce").fillna(0.0).combine_first(
                    pd.to_numeric(pf["route_rate_part"], errors="coerce").fillna(0.0)
                )
                pf.drop(columns=["route_rate_part"], inplace=True, errors="ignore")
        except Exception:
            pass  # optional

        # heuristic fills for route_rate/yprr_proxy
        pf = fill_route_rate_and_yprr(pf)

        # drop rows with blank names (these hurt coverage)
        blank_mask = pf["player"].isna() | (pf["player"].astype(str).str.strip().isin(["", "nan", "None", "..."]))
        if blank_mask.any():
            pf = pf.loc[~blank_mask].copy()

        # filter obvious non-skill positions (only if we know the position)
        if "position" in pf.columns:
            nonskill = {"C","G","OG","T","OT","OL","LT","LG","RT","RG",
                        "DT","DE","EDGE","DL","NT","LB","ILB","OLB",
                        "CB","DB","FS","SS","S","K","P","LS","FB"}
            pf = pf.loc[~pf["position"].astype(str).str.upper().isin(nonskill)].copy()

        # preserve opponent rows; only drop exact dups
        if {"player","team","season"}.issubset(pf.columns) and "opponent" in pf.columns:
            pf["_ord"] = np.arange(len(pf))
            pf = pf.sort_values(["player","team","season","opponent","_ord"], kind="mergesort")
            pf = pf.drop_duplicates(subset=["player","team","season","opponent"], keep="first")
            pf.drop(columns=["_ord"], inplace=True, errors="ignore")
            pf["opponent"] = (
                pf["opponent"].astype(str).str.strip()
                  .replace({"": "ALL", "nan": "ALL", "None": "ALL", "...": "ALL"})
                  .str.upper()
            )

        # rebuild/ensure player_key after any player edits
        pf = add_player_key(pf, "player")

        # emit post-enrich unmatched report
        unmatched = pf.loc[pd.isna(pf.get("position")) | (pf.get("position").astype(str).str.strip()=="") |
                           pd.isna(pf.get("role")) | (pf.get("role").astype(str).str.strip()=="")][
            ["player","player_key","team","season","position","role"]
        ].drop_duplicates()
        _write_csv(UNMATCHED_OUT, unmatched)

        # final zero/clip pass
        pf = zero_and_clip(pf)

        _write_csv(OUTPATH, pf)
        print(f"[enrich_player_form] Wrote {len(pf)} rows -> {OUTPATH}")
        print(f"[enrich_player_form] Unmatched roles report -> {UNMATCHED_OUT}  (rows={len(unmatched)})")

if __name__ == "__main__":
    main()
