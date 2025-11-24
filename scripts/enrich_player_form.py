# scripts/enrich_player_form.py
"""
Enrich player_form.csv with depth-chart roles, positions, and safe fallbacks.

Goals
- Guarantee player_form.csv is non-empty and has the expected schema
- Add/repair team, position, and role (WR1/WR2/SLOT/RB1/TE1, etc.)
- Fill missing route_rate / yprr_proxy when possible
- Preserve per-opponent rows (no collapsing) and make validator-friendly aliases
- Default missing share/rate metrics (incl. RZ) to 0.0; keep raw counts untouched

Inputs (best-effort, all optional except player_form.csv):
- data/player_form.csv                  # base built by make_player_form.py
- outputs/props_raw.csv                 # bootstrap if player_form is empty
- data/roles.csv                        # user-maintained roles (preferred)
- data/depth_chart_espn.csv             # ESPN depth charts
- data/depth_chart_ourlads.csv          # OurLads depth charts
- data/pfr_player_positions.csv         # fallback positions (if present)

Output:
- data/player_form.csv                  # enriched in-place
"""

from __future__ import annotations
from scripts._opponent_map import attach_opponent
from scripts.utils.canonical_names import (
    canonicalize_player_name as _canonicalize_with_utils,
    log_unmapped_variant,
)
import os
import re
import warnings
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPATH = os.path.join(DATA_DIR, "player_form.csv")

BASE_COLS = [
    "player", "team", "season",
    "target_share", "rush_share", "rz_tgt_share", "rz_carry_share",
    "ypt", "ypc", "yprr_proxy", "route_rate",
    "position", "role"
]

CONSENSUS_RATE_COLUMNS = [
    "tgt_share",
    "target_share",
    "route_rate",
    "rush_share",
    "yprr",
    "yprr_proxy",
    "ypt",
    "ypc",
    "ypa",
    "rz_share",
    "rz_tgt_share",
    "rz_rush_share",
    "receptions_per_target",
    "snap_share",
]

# -------------------- IO helpers --------------------

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
    return out


def _canonical_identity_fields(raw) -> dict:
    raw_str = "" if raw is None else str(raw)
    canon_tuple = _canonicalize_with_utils(raw_str)
    if isinstance(canon_tuple, (list, tuple)):
        canonical = canon_tuple[0] if canon_tuple else ""
        clean_key = canon_tuple[1] if len(canon_tuple) > 1 else None
    else:
        canonical = canon_tuple
        clean_key = None
    canonical = (canonical or "").strip()
    clean_key = (clean_key or "").strip()

    if raw_str and canonical and canonical == clean_key:
        try:
            log_unmapped_variant(
                "enrich_player_form",
                raw_str,
                {"stage": "canonical_identity_fields"},
            )
        except Exception:
            pass

    if not canonical:
        canonical = raw_str.upper().strip()

    canonical_lower = re.sub(r"[^a-z0-9 ]+", "", canonical.lower()).strip()
    clean_slug = re.sub(r"\s+", "_", canonical_lower)

    return {
        "player_name_canonical": canonical,
        "player_canonical": canonical_lower,
        "player_clean_key": clean_slug,
    }


def add_canonical_columns(df: pd.DataFrame, player_col: str = "player") -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    if player_col not in out.columns:
        if "player_name_canonical" not in out.columns:
            out["player_name_canonical"] = ""
        if "player_canonical" not in out.columns:
            out["player_canonical"] = ""
        if "player_clean_key" not in out.columns:
            out["player_clean_key"] = ""
        return out

    canonical_df = out[player_col].apply(
        lambda nm: pd.Series(_canonical_identity_fields(nm))
    )
    out["player_name_canonical"] = (
        canonical_df["player_name_canonical"].astype(str).str.strip().str.upper()
    )
    out["player_canonical"] = canonical_df["player_canonical"].astype(str)
    out["player_clean_key"] = canonical_df["player_clean_key"].astype(str)
    return out


def merge_opponent_from_props(df: pd.DataFrame) -> pd.DataFrame:
    om = _read_csv(os.path.join(DATA_DIR, "opponent_map_from_props.csv"))
    if om.empty:
        return df

    om = add_canonical_columns(om)
    if "team" in om.columns:
        om["team"] = om["team"].astype(str).str.upper().str.strip()
    if "week" in om.columns:
        om["week"] = pd.to_numeric(om["week"], errors="coerce")
    if "opponent" in om.columns:
        om["opponent"] = om["opponent"].astype(str).str.upper().str.strip()

    base = df.copy()
    if "team" in base.columns:
        base["team"] = base["team"].astype(str).str.upper().str.strip()
    if "week" in base.columns:
        base["week"] = pd.to_numeric(base["week"], errors="coerce")
    if "opponent" not in base.columns:
        base["opponent"] = pd.NA

    merge_keys = ["player_name_canonical"]
    if "team" in base.columns and "team" in om.columns:
        merge_keys.append("team")
    if "week" in base.columns and "week" in om.columns:
        merge_keys.append("week")

    merge_cols = merge_keys + [c for c in ["opponent"] if c in om.columns]
    merge_cols = list(dict.fromkeys(merge_cols))

    merged = base.merge(
        om[merge_cols].drop_duplicates(),
        on=merge_keys,
        how="left",
        suffixes=("", "_props"),
    )

    if "opponent_props" in merged.columns:
        merged["opponent"] = merged.get("opponent").combine_first(
            merged["opponent_props"]
        )
        merged = merged.drop(columns=["opponent_props"])

    return merged


def merge_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    weather = _read_csv(os.path.join(DATA_DIR, "weather_week.csv"))
    if weather.empty:
        return df

    weather = weather.copy()
    weather.columns = [c.lower() for c in weather.columns]

    base = df.copy()
    merge_keys: list[str] = []

    if "game_id" in base.columns and "game_id" in weather.columns:
        merge_keys.append("game_id")

    if "team" in base.columns and "team" in weather.columns:
        base["team"] = base["team"].astype(str).str.upper().str.strip()
        weather["team"] = weather["team"].astype(str).str.upper().str.strip()
        merge_keys.append("team")

    if "week" in base.columns and "week" in weather.columns:
        base["week"] = pd.to_numeric(base["week"], errors="coerce")
        weather["week"] = pd.to_numeric(weather["week"], errors="coerce")
        if "week" not in merge_keys:
            merge_keys.append("week")

    if not merge_keys:
        return base

    subset_cols = merge_keys + [c for c in weather.columns if c not in merge_keys]
    subset = weather[subset_cols].drop_duplicates()

    merged = base.merge(
        subset,
        on=merge_keys,
        how="left",
        suffixes=("", "_weather"),
    )

    weather_cols = [c for c in subset.columns if c not in merge_keys]
    for col in weather_cols:
        weather_col = f"{col}_weather"
        if weather_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[weather_col])
                merged.drop(columns=[weather_col], inplace=True)
            else:
                merged.rename(columns={weather_col: col}, inplace=True)

    return merged


def _first_non_empty(series: pd.Series) -> str:
    if series is None:
        return ""
    for val in series:
        if isinstance(val, str) and val.strip():
            return val
    cleaned = series.dropna()
    if not cleaned.empty:
        return str(cleaned.iloc[0])
    return ""


def build_player_form_consensus(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["player_name_canonical", "player"])

    metrics = [col for col in CONSENSUS_RATE_COLUMNS if col in df.columns]
    agg_map = {col: "mean" for col in metrics}

    grouped = (
        df.groupby("player_name_canonical", dropna=False)
        .agg(agg_map)
        .reset_index()
    )

    if "player" in df.columns:
        first_player = (
            df.groupby("player_name_canonical", dropna=False)["player"]
            .agg(_first_non_empty)
            .reset_index()
        )
        grouped = grouped.merge(first_player, on="player_name_canonical", how="left")

    if "player_clean_key" in df.columns:
        first_key = (
            df.groupby("player_name_canonical", dropna=False)["player_clean_key"]
            .agg(_first_non_empty)
            .reset_index()
        )
        grouped = grouped.merge(first_key, on="player_name_canonical", how="left")

    ordered = ["player_name_canonical"]
    if "player" in grouped.columns:
        ordered.append("player")
    if "player_clean_key" in grouped.columns:
        ordered.append("player_clean_key")
    ordered.extend([col for col in metrics if col in grouped.columns])
    remaining = [col for col in grouped.columns if col not in ordered]
    consensus = grouped[ordered + remaining]

    if metrics:
        consensus[metrics] = (
            consensus[metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        )

    return consensus


def _should_backfill(df: pd.DataFrame, target: str, source: str) -> bool:
    """Return True when an alias backfill should run."""
    if source not in df.columns:
        return False
    if target not in df.columns:
        return True
    try:
        series = pd.to_numeric(df[target], errors="coerce")
    except KeyError:
        return False
    if series.notna().any() and (series.abs() > 0).any():
        return False
    return True


def _apply_alias_backfills(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if _should_backfill(out, "target_share", "tgt_share"):
        out["target_share"] = pd.to_numeric(out["tgt_share"], errors="coerce")
    if _should_backfill(out, "yprr_proxy", "yprr"):
        out["yprr_proxy"] = pd.to_numeric(out["yprr"], errors="coerce")
    if _should_backfill(out, "rz_carry_share", "rz_rush_share"):
        out["rz_carry_share"] = pd.to_numeric(out["rz_rush_share"], errors="coerce")
    return out

# -------------------- Bootstrap if empty --------------------

def bootstrap_from_props_if_empty(df: pd.DataFrame) -> pd.DataFrame:
    """If player_form is empty, build a minimal skeleton from outputs/props_raw.csv."""
    if not df.empty and df["player"].notna().any():
        return df
    props = _read_csv(os.path.join("outputs", "props_raw.csv"))
    if props.empty:
        return df

    candidates = [c for c in ["player", "receiver", "rusher", "passer", "name"] if c in props.columns]
    if not candidates:
        return df
    pcol = candidates[0]

    team_col = None
    for tcol in ["team", "posteam", "home_team", "away_team", "team_name"]:
        if tcol in props.columns:
            team_col = tcol
            break

    boot = props[[c for c in [pcol, team_col] if c]].dropna().copy()
    boot = boot.rename(columns={pcol: "player", team_col: "team"}) if team_col else boot.rename(columns={pcol: "player"})
    boot["player"] = boot["player"].astype(str)
    if "team" in boot:
        boot["team"] = boot["team"].astype(str)
    subset = ["player", "team"] if "team" in boot else ["player"]
    boot = boot.drop_duplicates(subset=subset)

    boot["season"] = boot.get("season", pd.Series(np.nan, index=boot.index))
    for c in ["target_share", "rush_share", "rz_tgt_share", "rz_carry_share", "ypt", "ypc", "yprr_proxy", "route_rate", "position", "role"]:
        boot[c] = np.nan
    boot = boot[BASE_COLS]
    return boot

# -------------------- Roles / positions --------------------

def non_destructive_merge(base: pd.DataFrame, add: pd.DataFrame, on, mapping=None) -> pd.DataFrame:
    """Merge 'add' into 'base' without overwriting existing non-null values."""
    if add.empty:
        return base
    if mapping is None:
        mapping = {}
    add = add.copy()
    add.columns = [c.lower() for c in add.columns]

    if isinstance(on, str):
        on = [on]
    for k in on:
        if k not in base.columns or k not in add.columns:
            return base

    add_cols = on + list(mapping.values() if mapping else [])
    add = add[[c for c in add_cols if c in add.columns]].drop_duplicates()
    merged = base.merge(add, on=on, how="left", suffixes=("", "_new"))

    for bcol, acol in mapping.items():
        if acol in merged.columns:
            merged[bcol] = merged[bcol].combine_first(merged[acol])

    drop_cols = [c for c in merged.columns if c.endswith("_new") and c not in BASE_COLS]
    if drop_cols:
        merged.drop(columns=drop_cols, inplace=True, errors="ignore")
    return merged

def load_roles_priority() -> pd.DataFrame:
    """1) roles.csv, 2) depth_chart_espn.csv, 3) depth_chart_ourlads.csv."""
    # 1) roles.csv
    roles = _read_csv(os.path.join(DATA_DIR, "roles.csv"))
    if not roles.empty and {"player", "team", "role"}.issubset(roles.columns):
        if "position" not in roles.columns:
            roles["position"] = np.nan
        roles["role"] = roles["role"].astype(str).str.upper()
        return roles[["player", "team", "position", "role"]].drop_duplicates()

    # 2) ESPN
    espn = _read_csv(os.path.join(DATA_DIR, "depth_chart_espn.csv"))
    if not espn.empty:
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

        espn["role"] = espn.apply(_role_from_row, axis=1).astype(str).str.upper()
        return espn[["player", "team", "position", "role"]].drop_duplicates()

    # 3) OurLads
    ol = _read_csv(os.path.join(DATA_DIR, "depth_chart_ourlads.csv"))
    if not ol.empty:
        ol["position"] = ol.get("position", np.nan)
        ol["depth"] = ol.get("depth", np.nan)

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

        ol["role"] = (ol.get("position", "") + ol.get("depth", "").astype(str)).apply(_normalize)
        return ol[["player", "team", "position", "role"]].drop_duplicates()

    return pd.DataFrame(columns=["player", "team", "position", "role"])

# -------------------- Metric coercion helpers --------------------

def _enforce_rz_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Default RZ metrics to 0.0 (not NaN) when no events present."""
    out = df.copy()
    for c in ["rz_share", "rz_tgt_share", "rz_rush_share"]:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def _coerce_metric_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill share/rate metrics; leave raw counts untouched."""
    metric_cols = [
        "target_share", "tgt_share", "route_rate", "yprr", "yprr_proxy", "ypt", "receptions_per_target", "snap_share",
        "rush_share", "ypc", "ypa",
        "rz_share", "rz_tgt_share", "rz_rush_share"
    ]
    out = df.copy()
    for c in metric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

# -------------------- Fill proxies --------------------

def fill_route_rate_and_yprr(df: pd.DataFrame) -> pd.DataFrame:
    """
    route_rate heuristic:
      WR/TE: ~ target_share * 1.15 (clamped 5%-95%)
      RB:    ~ target_share * 0.6  (clamped <=65%)
    yprr_proxy fallback to ypt for WR/TE if missing.
    """
    out = df.copy()
    pos = out.get("position", pd.Series(np.nan, index=out.index)).astype(str).str.upper()

    rr = out.get("route_rate", pd.Series(np.nan, index=out.index)).astype(float)
    tshare = out.get("target_share", pd.Series(0.0, index=out.index)).astype(float)

    mask_wrte = pos.str.startswith("WR") | pos.str.startswith("TE")
    rr_wrte = tshare.mul(1.15).clip(lower=0.05, upper=0.95)
    rr = rr.mask(mask_wrte & rr.isna(), rr_wrte)

    mask_rb = pos.str.startswith("RB")
    rr_rb = tshare.mul(0.6).clip(upper=0.65)
    rr = rr.mask(mask_rb & rr.isna(), rr_rb)
    out["route_rate"] = rr

    yprr = out.get("yprr_proxy", pd.Series(np.nan, index=out.index)).astype(float)
    ypt  = out.get("ypt", pd.Series(np.nan, index=out.index)).astype(float)
    yprr = yprr.mask(mask_wrte & yprr.isna() & ypt.notna(), ypt)
    out["yprr_proxy"] = yprr

    return out

# -------------------- Finalizer (strict validator-friendly) --------------------

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _clip_01(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0.0, upper=1.0)
    return df

def finalize_player_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make validate_metrics happy:
      - numeric dtypes, no NaNs in core metrics
      - shares/rates in [0,1]
      - keys normalized
    """
    numeric_cols = [
        "tgt_share", "route_rate", "rush_share",
        "rz_share", "rz_tgt_share", "rz_rush_share", "rz_carry_share",
        "yprr", "ypt", "ypc", "ypa", "receptions_per_target",
        "yprr_proxy", "ay_per_att", "ay_per_att_z",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = _coerce_numeric(df, numeric_cols)
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    clip_01_cols = [
        "tgt_share", "route_rate", "rush_share",
        "rz_share", "rz_tgt_share", "rz_rush_share", "rz_carry_share",
        "receptions_per_target",
    ]
    clip_01_cols = [c for c in clip_01_cols if c in df.columns]
    df = _clip_01(df, clip_01_cols)

    if "player" in df.columns:
        df["player"] = df["player"].fillna("").astype(str).str.strip()
    if "team" in df.columns:
        df["team"] = df["team"].fillna("").astype(str).str.upper()
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].fillna("").astype(str).str.upper()

    # guarantee these exist
    for need in ["yprr_proxy", "rz_carry_share"]:
        if need in df.columns:
            df[need] = pd.to_numeric(df[need], errors="coerce").fillna(0.0)
        else:
            df[need] = 0.0

    return df

# -------------------- Main --------------------

def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pf = _read_csv(OUTPATH)
        if pf.empty or pf["player"].dropna().empty:
            print("[enrich_player_form] player_form empty → bootstrapping from props_raw.csv ...")
            pf = bootstrap_from_props_if_empty(pf)

        pf = _apply_alias_backfills(pf)
        pf = ensure_schema(pf)

    # validator-friendly aliases kept alongside originals
    if "tgt_share" not in pf.columns and "target_share" in pf.columns:
        pf["tgt_share"] = pd.to_numeric(pf["target_share"], errors="coerce")
    if "yprr" not in pf.columns:
        if "yprr_proxy" in pf.columns:
            pf["yprr"] = pd.to_numeric(pf["yprr_proxy"], errors="coerce")
        elif "ypt" in pf.columns:
            pf["yprr"] = pd.to_numeric(pf["ypt"], errors="coerce")
        else:
            pf["yprr"] = np.nan
    if "rz_rush_share" not in pf.columns and "rz_carry_share" in pf.columns:
        pf["rz_rush_share"] = pd.to_numeric(pf["rz_carry_share"], errors="coerce")
    if "rz_carry_share" not in pf.columns and "rz_rush_share" in pf.columns:
        pf["rz_carry_share"] = pd.to_numeric(pf["rz_rush_share"], errors="coerce")

    # zero defaults for RZ metrics + other rate/share fields
    pf = _enforce_rz_zero(pf)
    pf = _coerce_metric_zeros(pf)

    pf = add_canonical_columns(pf)
    if "team" in pf.columns:
        pf["team"] = pf["team"].astype(str).str.upper().str.strip()
    if "week" in pf.columns:
        pf["week"] = pd.to_numeric(pf["week"], errors="coerce")

    # roles/positions
    roles = load_roles_priority()
    if not roles.empty:
        roles = add_canonical_columns(roles)
        if "team" in roles.columns:
            roles["team"] = roles["team"].astype(str).str.upper().str.strip()
        merge_keys = ["player_name_canonical"]
        if "team" in roles.columns and "team" in pf.columns:
            merge_keys.append("team")
        pf = non_destructive_merge(
            pf,
            roles,
            on=merge_keys,
            mapping={"position": "position", "role": "role"},
        )

    # optional PFR position fallback
    pfr_pos = _read_csv(os.path.join(DATA_DIR, "pfr_player_positions.csv"))
    if not pfr_pos.empty and {"player", "position"}.issubset(pfr_pos.columns):
        pfr_pos = add_canonical_columns(pfr_pos)
        pf = non_destructive_merge(
            pf,
            pfr_pos,
            on=["player_name_canonical"],
            mapping={"position": "position"},
        )

    # participation-derived proxies (best-effort; non-fatal if missing libs)
    try:
        season_guess = int(pf["season"].dropna().iloc[0]) if "season" in pf.columns and pf["season"].notna().any() else 2025
        import nflreadpy as _nflv  # noqa: F401
        part = _nflv.load_participation(seasons=[season_guess])
        part.columns = [c.lower() for c in part.columns]
        team_col = "posteam" if "posteam" in part.columns else ("offense_team" if "offense_team" in part.columns else None)
        if team_col is not None and "player_name" in part.columns:
            p = part.rename(columns={team_col: "team", "player_name": "player"})
            p["team"] = p["team"].astype(str).str.upper().str.strip()
            p["player"] = p["player"].astype(str).str.replace(".", "", regex=False).str.strip()

            g = p.groupby(["team", "player"], dropna=False).agg(
                off_snaps=("offense", "sum") if "offense" in p.columns else ("onfield", "sum"),
                routes=("route", "sum") if "route" in p.columns else ("routes", "sum") if "routes" in p.columns else ("onfield", "sum")
            ).reset_index()

            tt = g.groupby("team", dropna=False).agg(team_off_snaps=("off_snaps", "sum"),
                                                     team_routes=("routes", "sum")).reset_index()
            g = g.merge(tt, on="team", how="left")
            g["snap_share"] = np.where(g["team_off_snaps"] > 0, g["off_snaps"] / g["team_off_snaps"], 0.0)
            g["route_rate"] = np.where(g["team_routes"] > 0, g["routes"] / g["team_routes"], 0.0)

            pf = pf.merge(g[["team", "player", "snap_share", "route_rate"]],
                          on=["team", "player"], how="left", suffixes=("", "_part"))
            for col in ["snap_share", "route_rate"]:
                ext = col + "_part"
                if ext in pf.columns:
                    pf[col] = pd.to_numeric(pf[col], errors="coerce").fillna(0.0).combine_first(
                        pd.to_numeric(pf[ext], errors="coerce").fillna(0.0)
                    )
                    pf.drop(columns=[ext], inplace=True)
    except Exception:
        pass  # optional enrichment only

    pf = merge_opponent_from_props(pf)
    pf = merge_weather_data(pf)

    # preserve opponent dimension; only drop exact duplicates including opponent
    if {"player", "team", "season"}.issubset(pf.columns) and "opponent" in pf.columns:
        try:
            pf["_orig_order"] = np.arange(len(pf))
            pf = pf.sort_values(["player", "team", "season", "opponent", "_orig_order"], kind="mergesort")
            pf = pf.drop_duplicates(subset=["player", "team", "season", "opponent"], keep="first")
            pf.drop(columns=["_orig_order"], inplace=True, errors="ignore")

            pf["opponent"] = (
                pf["opponent"]
                .astype(str)
                .str.strip()
                .replace({"": "ALL", "nan": "ALL", "None": "ALL", "...": "ALL"})
                .str.upper()
            )
        except Exception:
            for c in ["_orig_order"]:
                if c in pf.columns:
                    pf.drop(columns=[c], inplace=True, errors="ignore")

    # ensure a stable player_key
    try:
        pf["player_key"] = (
            pf.get("player", pd.Series([], dtype=object))
            .fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        )
    except Exception:
        pass

    # ---- FINALIZE (coerce types, fillna 0.0, clamp shares) ----
    # Best-effort schedule-based opponent backfill. Any opponent data already
    # merged from props (merge_opponent_from_props) is preserved.
    try:
        pf = attach_opponent(
            pf,
            season_col="season",
            week_col="week",
            team_col="team",
            out_col="opponent",
            schedule_path=os.path.join(DATA_DIR, "team_week_map.csv"),
        )
    except Exception:
        # Non-fatal: leave existing opponent values as-is so the validator can
        # still run even if the schedule file is missing or mismatched.
        pass

    pf = finalize_player_form(pf)

    consensus = build_player_form_consensus(pf)

    _write_csv(OUTPATH, pf)
    _write_csv(os.path.join(DATA_DIR, "player_form_consensus.csv"), consensus)
    print(f"[enrich_player_form] Wrote {len(pf)} rows → {OUTPATH}")
    print(
        f"[enrich_player_form] Wrote {len(consensus)} rows → {os.path.join(DATA_DIR, 'player_form_consensus.csv')}"
    )

if __name__ == "__main__":
    main()
