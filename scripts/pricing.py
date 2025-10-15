# scripts/pricing.py
# Prices props with model μ/σ, de-vig, blend, Kelly, tiers, plus:
#   - Coverage-by-route adjustments (press/man/zone vs route_profile_*)
#   - SGP pair-aware dynamic: QB→WR small lift in same-game edges
#   - Market-specific sigmas learned via EMA in data/market_sigmas.json

from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- normalize market keys (aliases) ---
# INTERNAL canonical keys we use elsewhere in pricing/base_mu():
#   player_pass_yds, player_rec_yds, player_rush_yds, player_receptions, player_rush_rec_yds, player_anytime_td
MARKET_ALIASES = {
    # receiving yards (map all variants to our internal 'player_rec_yds')
    "player_reception_yds": "player_rec_yds",       # Odds API canonical
    "player_receiving_yards": "player_rec_yds",
    "player_receiving_yds": "player_rec_yds",
    "player_rec_yds": "player_rec_yds",

    # rush+rec (map to our internal 'player_rush_rec_yds')
    "player_rush_reception_yds": "player_rush_rec_yds",
    "player_rush_rec_yds": "player_rush_rec_yds",
    "player_rush_and_receive_yards": "player_rush_rec_yds",
    "player_rush_and_receive_yds": "player_rush_rec_yds",
    "rushing_plus_receiving_yards": "player_rush_rec_yds",
    "rush_rec_yards": "player_rush_rec_yds",
    "rush_rec": "player_rush_rec_yds",

    # passthrough for others
    "player_pass_yds": "player_pass_yds",
    "player_rush_yds": "player_rush_yds",
    "player_receptions": "player_receptions",
    "player_anytime_td": "player_anytime_td",
}

def normalize_market_key(k: str) -> str:
    k2 = str(k).strip().lower()
    return MARKET_ALIASES.get(k2, k2)

# ---------- Odds helpers ----------
def american_to_prob(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)

def prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1 or np.isnan(p):
        return np.nan
    q = 1.0 - p
    if p >= q:
        # favorite
        return -100.0 * p / q
    else:
        # dog
        return 100.0 * q / p

def devig_two_way(p_over_vig: pd.Series, p_under_vig: pd.Series) -> tuple[pd.Series,pd.Series]:
    # fair = p / (p_over + p_under)
    denom = (p_over_vig + p_under_vig).replace(0, np.nan)
    p_over_fair  = (p_over_vig / denom).clip(0,1)
    p_under_fair = (p_under_vig / denom).clip(0,1)
    return p_over_fair, p_under_fair

def _extract_odds(df: pd.DataFrame) -> pd.Series:
    """
    Robust odds extractor:
      - Prefer explicit over/under if present
      - Else side + single price column
      - Else fallbacks ('price', 'american', 'price_american', etc.).
    Never returns a scalar; always a Series aligned to df.index.
    """
    idx = df.index

    # Already provided
    if "odds" in df.columns:
        return pd.to_numeric(df["odds"], errors="coerce")

    # Common Odds API shape with side + split columns
    if {"side", "over_odds", "under_odds"}.issubset(df.columns):
        side = df["side"].astype(str).str.lower()
        series = np.where(side.eq("over"), df["over_odds"], df["under_odds"])
        return pd.to_numeric(series, errors="coerce")

    # Single-column odds from some sources
    for col in ("price_american", "american", "price"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    # Fallback: all-NaN
    return pd.Series(np.nan, index=idx, dtype="float64")

# ---------- Sigma memory ----------
SIGMA_PATH = Path("data/market_sigmas.json")
SIGMA_DEFAULTS = {
    "player_pass_yds": 48.0,
    # include both internal canonical & a couple externals so lookups never miss
    "player_rec_yds": 26.0,
    "player_reception_yds": 26.0,
    "player_receiving_yards": 26.0,
    "player_rush_yds": 24.0,
    "player_receptions": 1.8,
    "player_rush_rec_yds": 36.0,
}
def load_sigmas() -> dict[str,float]:
    if SIGMA_PATH.exists():
        try:
            return json.loads(SIGMA_PATH.read_text())
        except Exception:
            pass
    return dict(SIGMA_DEFAULTS)

def save_sigmas(sigmas: dict[str,float]) -> None:
    SIGMA_PATH.write_text(json.dumps(sigmas, indent=2))

def ema_update(old: float, new: float, alpha: float = 0.2) -> float:
    if np.isnan(new): return old
    return alpha*new + (1.0-alpha)*old

# ---------- Coverage boosts/penalties ----------
# route_profile_* are shares; press/man/zone rates from opponent.
def coverage_multiplier(row: pd.Series) -> float:
    # weights can be tuned; keep conservative
    press_w = 0.95  # slight downshift vs press if profile heavily press
    man_w   = 0.97
    zone_w  = 1.02
    rp_press = float(row.get("route_profile_press", 0.33) or 0.33)
    rp_man   = float(row.get("route_profile_man", 0.33) or 0.33)
    rp_zone  = float(row.get("route_profile_zone", 0.33) or 0.33)

    opp_press = float(row.get("press_rate_opp", 0.33) or 0.33)
    opp_man   = float(row.get("man_rate_opp",   0.50) or 0.50)
    opp_zone  = float(row.get("zone_rate_opp",  0.50) or 0.50)

    # expected multiplier = sum(profile * opp_rate * weight) normalized
    # normalize opp rates man/zone to 1 baseline
    mz_den = max(1e-9, opp_man + opp_zone)
    opp_man_n  = opp_man / mz_den
    opp_zone_n = opp_zone / mz_den

    mult = (
        rp_press * opp_press * press_w +
        rp_man   * opp_man_n * man_w   +
        rp_zone  * opp_zone_n * zone_w
    )
    # normalize around 1.0
    return max(0.90, min(1.08, mult / max(1e-9, rp_press + rp_man + rp_zone)))

# ---------- Base μ ----------
def base_mu(row: pd.Series) -> float:
    """
    A conservative, context-aware μ for each market.
    """
    mkt = normalize_market_key(row.get("market",""))
    line = float(row.get("line", np.nan)) if pd.notna(row.get("line", np.nan)) else np.nan

    # opponent context
    def_pass_pen = float(row.get("def_pass_epa_opp", 0.0) or 0.0)
    def_rush_pen = float(row.get("def_rush_epa_opp", 0.0) or 0.0)
    pressure_z   = float(row.get("def_sack_rate_opp", 0.0) or 0.0)

    # QB baseline & pressure
    qb_ypa = float(row.get("qb_ypa", np.nan))
    if np.isnan(qb_ypa):
        qb_ypa = float(row.get("ypt_qb", 6.8) or 6.8)
    qb_pen = (1.0 - 0.35 * max(0.0, pressure_z)) * (1.0 - 0.25 * max(0.0, def_pass_pen))
    qb_pen = max(0.75, min(1.05, qb_pen))

    # coverage multiplier
    cov_mult = coverage_multiplier(row)

    # weather nudges
    weather_m = 1.0
    wind = row.get("wind_mph", np.nan)
    precip = str(row.get("precip", "")).lower()
    try:
        wind = float(wind)
    except Exception:
        wind = np.nan
    if not np.isnan(wind) and wind >= 15:
        if mkt in {"player_pass_yds","player_rec_yds","player_rush_rec_yds"}:
            weather_m *= 0.94
    if precip in {"rain","snow"}:
        if mkt in {"player_rec_yds","player_rush_rec_yds"}:
            weather_m *= 0.97
        if mkt in {"player_rush_yds"}:
            weather_m *= 1.02

    if mkt == "player_pass_yds":
        # FIX: use qb_ypa (we compute it above) instead of undefined variable
        return max(0.0, qb_ypa * 28.0 * qb_pen * weather_m)  # ~28 attempts baseline

    # ★ enrich: receiving-volume enrichments
    # Prefer route_rate if present; else routes_per_dropback (both 0..1)
    route_rate = row.get("route_rate", np.nan)
    routes_per_db = float(row.get("routes_per_dropback", np.nan))
    if not np.isnan(route_rate):
        rvol_factor = 0.75 + 0.5 * max(0.0, min(1.0, float(route_rate)))   # 0.75x..1.25x
    elif not np.isnan(routes_per_db):
        rvol_factor = 0.75 + 0.5 * max(0.0, min(1.0, routes_per_db))       # 0.75x..1.25x
    else:
        rvol_factor = 1.0

    # modest slot boost when opponent plays more zone
    slot_rate     = float(row.get("slot_rate", np.nan))     # team-level slot target share (0..1)
    opp_zone      = float(row.get("zone_rate_opp", 0.50) or 0.50)
    slot_mult = 1.0
    if not np.isnan(slot_rate):
        slot_mult *= (1.0 + 0.06 * (slot_rate - 0.33) * opp_zone)  # ≈ ±3–4% typical
        slot_mult = max(0.94, min(1.06, slot_mult))
        cov_mult *= slot_mult

    # ★ patch: TE bump from 12p_rate (team-level)
    twelp = float(row.get("12p_rate", np.nan))
    te12_mult = 1.0
    if not np.isnan(twelp) and twelp > 0.18 and str(row.get("position","")).upper() == "TE":
        te12_mult = 1.03

    # Team air-yards sanity check (downshift when scheme is shallow)
    ay_att = float(row.get("ay_per_att", np.nan))
    ay_mult = 1.0
    if not np.isnan(ay_att) and ay_att <= 6.0 and mkt in {"player_rec_yds","player_rush_rec_yds"}:
        ay_mult = 0.92

    # Combine multipliers
    cov_mult = max(0.90, min(1.10, cov_mult * te12_mult * ay_mult))

    if mkt == "player_rec_yds":
        ypt = float(row.get("ypt", np.nan))
        if np.isnan(ypt):
            ypt = float(row.get("yprr_proxy", 1.8)) * 1.6  # soft fallback
        targets = float(row.get("target_share", np.nan))
        if np.isnan(targets): targets = 0.18
        plays = float(row.get("pace", 28.0) or 28.0)
        pass_rate = 0.56 + float(row.get("proe", 0.0) or 0.0)
        mu = ypt * (targets * plays * pass_rate) * cov_mult * rvol_factor * weather_m
        return max(0.0, mu)

    if mkt == "player_receptions":
        rrate = float(row.get("route_rate", np.nan))
        if np.isnan(rrate): rrate = 0.65
        catch_rate = float(row.get("catch_rate", 0.62))
        plays = float(row.get("pace", 28.0) or 28.0)
        pass_rate = 0.56 + float(row.get("proe", 0.0) or 0.0)
        mu = (rrate * plays * pass_rate) * catch_rate * cov_mult * rvol_factor * weather_m
        return max(0.0, mu)

    if mkt == "player_rush_yds":
        ypc = float(row.get("ypc", np.nan))
        if np.isnan(ypc): ypc = 4.2
        rush_share = float(row.get("rush_share", 0.45))
        plays = float(row.get("pace", 28.0) or 28.0)
        run_rate = 1.0 - (0.56 + float(row.get("proe", 0.0) or 0.0))
        box_z = float(row.get("heavy_box_rate_opp", 0.0))
        box_mult = 1.0 - 0.06 * max(0.0, box_z)
        mu = ypc * (rush_share * plays * run_rate) * box_mult * weather_m
        return max(0.0, mu)

    if mkt == "player_rush_rec_yds":
        # simple sum of the above two mus
        row_pass = row.copy(); row_pass["market"] = "player_rec_yds"
        row_rush = row.copy(); row_rush["market"] = "player_rush_yds"
        return max(0.0, base_mu(row_pass) + base_mu(row_rush))

    # default safe fallback (anchor to line if available)
    return float(line) if pd.notna(line) else np.nan

# ---------- σ logic ----------
def sigma_for_market(mkt: str, sigmas: dict[str,float], volatility_flag: float = 0.0) -> float:
    base = sigmas.get(mkt, SIGMA_DEFAULTS.get(mkt, 30.0))
    s = float(base)
    if volatility_flag and s > 0:
        s *= (1.0 + 0.15 * min(1.0, volatility_flag))
    return max(1e-6, s)

# ---------- Pricing ----------
def _ensure_over_under_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If df lacks over_odds/under_odds but has side + price, synthesize them
    by grouping on (event_id, player, market, line, book [if present])."""
    if {"over_odds","under_odds"}.issubset(df.columns):
        return df
    side_col = None
    for c in ("side","name"):
        if c in df.columns:
            side_col = c
            break
    price_col = None
    for c in ("price_american","price","american","odds"):
        if c in df.columns:
            price_col = c
            break
    if side_col is None or price_col is None:
        return df
    key_candidates = ["event_id","player","market","line","book"]
    keys = [k for k in key_candidates if k in df.columns]
    if not keys:
        return df
    tmp = df[keys + [side_col, price_col]].copy()
    tmp[side_col] = tmp[side_col].astype(str).str.upper()
    piv = (tmp
           .dropna(subset=[side_col, price_col])
           .groupby(keys + [side_col], as_index=False)[price_col]
           .first()
           .pivot_table(index=keys, columns=side_col, values=price_col, aggfunc="first"))
    if isinstance(piv, pd.DataFrame):
        piv = piv.rename(columns={"OVER":"over_odds","UNDER":"under_odds"})
        for c in ("over_odds","under_odds"):
            if c not in piv.columns:
                piv[c] = np.nan
        piv = piv.reset_index()
        df = df.merge(piv[keys + ["over_odds","under_odds"]], on=keys, how="left", suffixes=("","__p"))
        for c in ("over_odds","under_odds"):
            if c + "__p" in df.columns:
                df[c] = pd.to_numeric(df.get(c), errors="coerce")
                df[c] = df[c].where(df[c].notna(), pd.to_numeric(df[c + "__p"], errors="coerce"))
                df.drop(columns=[c + "__p"], inplace=True, errors=True)
    return df

def main(props_path: str) -> None:
    props = pd.read_csv(props_path) if Path(props_path).exists() else pd.DataFrame()
    if props.empty:
        print("[pricing] no props found at", props_path); return

    # Normalize market keys so downstream logic matches our internal names
    if "market" in props.columns:
        props["market"] = props["market"].astype(str).map(normalize_market_key)
    # Build over/under columns if missing (from side + price)
    props = _ensure_over_under_columns(props)

    # Metrics (tolerate empty/missing)
    from pandas.errors import EmptyDataError
    m_path = Path("data/metrics_ready.csv")
    if m_path.exists() and m_path.stat().st_size > 0:
        try:
            met = pd.read_csv(m_path)
        except EmptyDataError:
            print("[pricing] WARNING: data/metrics_ready.csv exists but is empty; continuing without metrics.")
            met = pd.DataFrame()
    else:
        print("[pricing] WARNING: data/metrics_ready.csv missing or empty; continuing without metrics.")
        met = pd.DataFrame()

    # Coerce merge keys to string to avoid object/float mismatches
    for _df in (props, met):
        for k in ("event_id","player","book","market","team","opponent"):
            if k in _df.columns:
                _df[k] = _df[k].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # Merge
    keys = ["event_id","player"] if "event_id" in props.columns else ["player"]
    df = props.merge(met, on=keys, how="left", suffixes=("","_m"))

    # --- Odds normalization & market de-vig (robust) ---
    if {"over_odds", "under_odds"}.issubset(df.columns):
        # Two-way present: remove vig from both sides
        df["over_odds"]  = pd.to_numeric(df["over_odds"], errors="coerce")
        df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
        p_over_vig  = pd.to_numeric(df["over_odds"],  errors="coerce").apply(american_to_prob)
        p_under_vig = pd.to_numeric(df["under_odds"], errors="coerce").apply(american_to_prob)
        df["mkt_over_fair"], df["mkt_under_fair"] = devig_two_way(p_over_vig, p_under_vig)
        # Also define unified 'odds' for Kelly using Over side
        df["odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    else:
        side = df.get("side", "OVER").astype(str).str.upper()
        df["odds"] = _extract_odds(df)
        p_vig = df["odds"].apply(american_to_prob)
        df["mkt_over_fair"]  = np.where(side.eq("OVER"), p_vig, 1.0 - p_vig)
        df["mkt_under_fair"] = 1.0 - df["mkt_over_fair"]

        # Defensive guard: if we could not extract any odds, don’t crash the run
        if df["odds"].isna().all():
            print("[pricing] WARN: could not find odds columns for any rows; writing empty output.")
            Path("outputs").mkdir(exist_ok=True)
            df.head(0).to_csv("outputs/props_priced_clean.csv", index=False)
            return

    # Model μ/σ and Over% at the posted line
    sigmas = load_sigmas()
    df["model_proj"] = df.apply(base_mu, axis=1)
    # Last-ditch: if model_proj is NaN or 0 but we have a posted line, anchor to line
    if "line" in df.columns:
        msk = df["model_proj"].replace(0, np.nan).isna() & pd.to_numeric(df["line"], errors="coerce").notna()
        if msk.any():
            df.loc[msk, "model_proj"] = pd.to_numeric(df.loc[msk, "line"], errors="coerce")


    # ★ enrich: more robust volatility flag using pressure enrich if present
    # If you have pressure mismatch columns from metrics, average them; else safe heuristic
    if {"press_rate_opp","def_sack_rate_opp"}.issubset(df.columns):
        vol_parts = []
        for c in ("press_rate_opp","def_sack_rate_opp"):
            v = pd.to_numeric(df[c], errors="coerce")
            v = (v - v.mean(skipna=True)) / (v.std(ddof=0, skipna=True) + 1e-9)
            vol_parts.append(v.clip(lower=0))
        df["vol_flag"] = pd.concat(vol_parts, axis=1).mean(axis=1)
    else:
        df["vol_flag"] = (df.get("def_sack_rate_opp",0).fillna(0).clip(lower=0) + np.maximum(0, df.get("def_pass_epa_opp",0).fillna(0))) / 2.0

    df["model_sd"] = df.apply(lambda r: sigma_for_market(str(r.get("market","")), sigmas, r.get("vol_flag",0)), axis=1)

    # Over probability under Normal(μ,σ)
    from scipy.stats import norm
    line = df.get("line", np.nan).astype(float)
    mu   = df["model_proj"].astype(float)
    sd   = df["model_sd"].astype(float).replace(0,1e-6)
    z    = (line - mu) / sd
    df["model_over_pct"] = (1.0 - norm.cdf(z)).clip(0,1)

    # Blend with market anchor (65/35)
    df["p_over_blend"] = 0.65*df["model_over_pct"] + 0.35*df["mkt_over_fair"]
    df["fair_over_odds"] = df["p_over_blend"].apply(prob_to_american)

    # Edge & Kelly
    df["edge_abs"] = df["p_over_blend"] - df["mkt_over_fair"]

    def kelly_fraction(p, o):
        # o is American price for OVER
        if np.isnan(p) or np.isnan(o): return 0.0
        b = (100/o) if o>0 else (-o/100.0)
        q = 1.0 - p
        frac = (b*p - q) / b
        return max(0.0, min(frac, 0.05))  # cap 5%

    over_price = df["odds"]  # unified source
    df["kelly_frac"] = [kelly_fraction(p, o) for p, o in zip(df["p_over_blend"], over_price)]

    # Tiers
    def tier(e):
        if e >= 0.06: return "ELITE"
        if e >= 0.04: return "GREEN"
        if e >= 0.01: return "AMBER"
        return "RED"
    df["tier"] = df["edge_abs"].apply(tier)

    # ---------- SGP pair-aware dynamic (QB→WR small lift) ----------
    sgp_threshold = 0.05      # trigger threshold on QB edge
    sgp_max_lift  = 0.015     # cap per WR
    qb_to_wr_corr = 0.60      # same-game base correlation

    df["edge_bonus_sgp"] = 0.0
    if "event_id" in df.columns and "position" in df.columns:
        # NOTE: keep conservative; apply tiny lift to WR if QB on same event has ELITE edge
        qb_mask = (df["position"].astype(str).str.upper() == "QB") & (df["edge_abs"] >= sgp_threshold)
        if qb_mask.any():
            elite_events = set(df.loc[qb_mask, "event_id"].astype(str).tolist())
            wr_mask = (df["position"].astype(str).str.upper() == "WR") & (df["event_id"].astype(str).isin(elite_events))
            df.loc[wr_mask, "edge_bonus_sgp"] = np.minimum(sgp_max_lift, qb_to_wr_corr * df.loc[wr_mask, "edge_abs"] * 0.05)

    df["edge_total"] = (df["edge_abs"] + df["edge_bonus_sgp"]).clip(-1, 1)

    out_cols = [
        "event_id","player","team","opponent","position","market","line",
        "over_odds","under_odds","mkt_over_fair",
        "model_proj","model_sd","model_over_pct",
        "p_over_blend","fair_over_odds",
        "edge_abs","edge_bonus_sgp","edge_total","kelly_frac","tier"
    ]
    for c in out_cols:
        if c not in df.columns: df[c] = np.nan

    Path("outputs").mkdir(exist_ok=True)
    df[out_cols].to_csv("outputs/props_priced_clean.csv", index=False)
    print(f"[pricing] ✅ wrote outputs/props_priced_clean.csv rows={len(df)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", default="outputs/props_raw.csv")
    args = ap.parse_args()
    main(args.props)
