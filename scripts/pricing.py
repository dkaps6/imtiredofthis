# scripts/pricing.py
# Prices props with model μ/σ, de-vig, blend, Kelly, tiers, plus:
#   - Coverage-by-route adjustments (press/man/zone vs route_profile_*)
#   - SGP pair-aware dynamic: QB→WR small lift in same-game edges
#   - Market-specific sigmas learned via EMA in data/market_sigmas.json

from __future__ import annotations
import json, math
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
    if p <= 0: return np.nan
    if p >= 1: return np.nan
    return (100.0 * p / (1.0 - p)) if p < 0.5 else (-100.0 * (1.0 - p) / p)

def devig_two_way(p_over_vig: pd.Series, p_under_vig: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = (p_over_vig + p_under_vig).replace(0, np.nan)
    p_over_fair = p_over_vig / s
    p_under_fair = p_under_vig / s
    return p_over_fair, p_under_fair

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
    "player_rush_rec_yds": 30.0,
    "player_rush_reception_yds": 30.0,
}

def load_sigmas() -> dict[str, float]:
    if SIGMA_PATH.exists():
        try:
            return json.loads(SIGMA_PATH.read_text())
        except Exception:
            pass
    return SIGMA_DEFAULTS.copy()

def save_sigmas(sigmas: dict[str, float]) -> None:
    SIGMA_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    # scale into a modest [0.94, 1.06] band
    return max(0.94, min(1.06, mult * 1.06))

# ---------- μ builders ----------
def base_mu(row: pd.Series) -> float:
    mkt = str(row.get("market",""))
    # volumes & priors
    tgt = float(row.get("target_share", 0.18))
    rsh = float(row.get("rush_share",   0.30))
    yprr= float(row.get("yprr_proxy",   1.7))
    ypc = float(row.get("ypc",          4.2))
    ypa = float(row.get("qb_ypa",       6.9))
    wp  = float(row.get("team_wp",      0.5))

    # opponent context (Z-like scales are already baked into priors)
    pass_e = float(row.get("def_pass_epa_opp",   0.0))
    rush_e = float(row.get("def_rush_epa_opp",   0.0))
    sack_r = float(row.get("def_sack_rate_opp",  0.0))
    lbx    = float(row.get("light_box_rate_opp", 0.5))
    hbx    = float(row.get("heavy_box_rate_opp", 0.5))
    pace   = float(row.get("pace_opp",           0.0))
    proe   = float(row.get("proe_opp",           0.0))

    # coverage/CB/weather
    cov_mult = coverage_multiplier(row)
    cb_pen = float(row.get("cb_penalty", 0.0) or 0.0)  # 0..0.25
    wind   = float(row.get("wind_mph",   np.nan))
    precip = str(row.get("precip","")).lower().strip()

    # script nudges (RB favored when wp↑; QB volume ↓ modest vs elite pass D/sacks)
    run_bonus = 1.0 + 0.04 * (wp - 0.5)  # +2..4% RB attempts when favored
    qb_pen    = 1.0 - 0.15 * max(0.0, sack_r) - 0.12 * max(0.0, pass_e)

    # weather
    weather_m = 1.0
    if not np.isnan(wind) and wind >= 15:
        if mkt in {"player_pass_yds","player_rec_yds","player_receptions","player_rush_rec_yds"}:
            weather_m *= 0.94
    if precip in {"rain","snow"}:
        if mkt in {"player_rec_yds","player_rush_rec_yds"}:
            weather_m *= 0.97
        if mkt in {"player_rush_yds"}:
            weather_m *= 1.02

    if mkt == "player_pass_yds":
        return max(0.0, ypa * 28.0 * qb_pen * weather_m)  # ~28 attempts baseline
    if mkt == "player_rec_yds":
        # WR/TE receiving yards from YPRR * routes, coverage penalty & CB
        mu = yprr * (35.0 * tgt) * cov_mult * (1.0 - cb_pen) * weather_m
        # sanity downshift for very low YPRR teams
        team_ay_att = float(row.get("team_ay_per_att", 0.0))
        if team_ay_att <= -0.8:  # low air-yards profile
            mu = min(mu, 0.8 * max(20.0, mu))
        return max(0.0, mu)
    if mkt == "player_receptions":
        cr = 0.62 * cov_mult * (1.0 - 0.5*cb_pen)  # catch rate proxy
        return max(0.0, cr * (32.0 * tgt) * weather_m)
    if mkt == "player_rush_yds":
        ypc_mod = ypc * ((1.0 + 0.07*lbx) * (1.0 - 0.06*hbx)) * run_bonus * weather_m
        return max(0.0, ypc_mod * (14.0 * rsh))
    if mkt == "player_rush_rec_yds":
        return max(0.0, base_mu(row.copy().assign(market="player_rush_yds")) +
                         0.7*base_mu(row.copy().assign(market="player_rec_yds")))
    # default
    return np.nan

def sigma_for_market(market: str, sigmas: dict[str,float], volatility_flag: float = 0.0) -> float:
    s = float(sigmas.get(market, SIGMA_DEFAULTS.get(market, 25.0)))
    if volatility_flag > 0:
        s *= (1.0 + 0.15 * min(1.0, volatility_flag))
    return max(1e-6, s)

# ---------- Pricing ----------
def main(props_path: str) -> None:
    props = pd.read_csv(props_path) if Path(props_path).exists() else pd.DataFrame()
    if props.empty:
        print("[pricing] no props found at", props_path); return

    # Normalize market keys so downstream logic matches our internal names
    if "market" in props.columns:
        props["market"] = props["market"].astype(str).map(normalize_market_key)

    met = pd.read_csv("data/metrics_ready.csv") if Path("data/metrics_ready.csv").exists() else pd.DataFrame()
    if met.empty:
        print("[pricing] WARNING: metrics_ready.csv is empty; proceeding with minimal features.")

    # Coerce merge keys to string to avoid object/float mismatches
    for _df in (props, met):
        for k in ("event_id","player","book","market","team","opponent"):
            if k in _df.columns:
                _df[k] = _df[k].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # Merge
    keys = ["event_id","player"]
    if "event_id" not in props.columns: keys = ["player"]
    df = props.merge(met, on=keys, how="left", suffixes=("","_m"))

    # De-vig market fair Over/Under
    if "over_odds" in df.columns and "under_odds" in df.columns:
        p_over_vig  = df["over_odds"].apply(american_to_prob)
        p_under_vig = df["under_odds"].apply(american_to_prob)
        df["mkt_over_fair"], df["mkt_under_fair"] = devig_two_way(p_over_vig, p_under_vig)
    else:
        side = df.get("side","OVER").astype(str).str.upper()
        # normalize odds column name and coerce to numeric
    if "odds" not in df.columns:
        # common cases from Odds API
        if "price_american" in df.columns:
            df["odds"] = df["price_american"]
        elif {"over_odds", "under_odds"} <= set(df.columns) and "side" in df.columns:
            # pick the correct side if you melted earlier and kept a 'side' column
            df["odds"] = np.where(
                df["side"].str.lower().eq("over"), df["over_odds"],
                np.where(df["side"].str.lower().eq("under"), df["under_odds"], np.nan)
            )
        else:
            # fall back: create an aligned NaN series
            df["odds"] = np.nan

    odds = pd.to_numeric(df["odds"], errors="coerce")

        p_vig = odds.apply(american_to_prob)
        df["mkt_over_fair"]  = np.where(side=="OVER", p_vig, 1.0-p_vig)
        df["mkt_under_fair"] = 1.0 - df["mkt_over_fair"]

    # Model μ/σ and Over% at the posted line
    sigmas = load_sigmas()
    df["model_proj"] = df.apply(base_mu, axis=1)
    # volatility flag: strong pass rush vs weak OL proxy — use sack rate opp + pass epa opp
    df["vol_flag"] = (df.get("def_sack_rate_opp",0).fillna(0) + np.maximum(0, df.get("def_pass_epa_opp",0).fillna(0))) / 2.0
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
    df["kelly_frac"] = [kelly_fraction(p, o) for p,o in zip(df["p_over_blend"], df["over_odds"])]

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
        for eid, g in df.groupby("event_id"):
            qb_rows = g[(g["position"].str.upper()=="QB") & (g["market"]=="player_pass_yds")]
            if qb_rows.empty:
                continue
            qb_edge = float(qb_rows["edge_abs"].max())
            if qb_edge < sgp_threshold:
                continue
            qb_team = qb_rows.iloc[0].get("team","")
            wr_mask = (g["position"].str.upper().isin(["WR","TE"])) & \
                      (g["team"]==qb_team) & \
                      (g["market"].isin(["player_rec_yds","player_receptions"]))
            lift = min(sgp_max_lift, qb_to_wr_corr * 0.25 * max(0.0, qb_edge-0.03))
            df.loc[wr_mask.index[wr_mask], "edge_bonus_sgp"] += lift

    df["edge_total"] = (df["edge_abs"] + df["edge_bonus_sgp"]).clip(-1, 1)

    # ---------- Persist sigma learning (EMA) ----------
    for mkt, sub in df.groupby("market"):
        try:
            se = ((sub["line"].astype(float) - sub["model_proj"].astype(float))**2).mean()
            est_sigma = float(np.sqrt(se)) if not np.isnan(se) else np.nan
            if not np.isnan(est_sigma) and est_sigma>1e-3:
                sigmas[mkt] = round(ema_update(sigmas.get(mkt, SIGMA_DEFAULTS.get(mkt,25.0)), est_sigma), 4)
        except Exception:
            pass
    save_sigmas(sigmas)

    # Output
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
