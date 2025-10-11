from __future__ import annotations
import argparse, math, json, warnings, os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
DATA_DIR     = Path("data")
OUT_DIR      = Path("outputs")
METRICS_DIR  = Path("outputs/metrics")

PROPS_RAW    = OUT_DIR / "props_raw.csv"
TEAM_FORM1   = METRICS_DIR / "team_form.csv"
TEAM_FORM2   = DATA_DIR / "team_form.csv"
PLAYER_FORM1 = METRICS_DIR / "player_form.csv"
PLAYER_FORM2 = DATA_DIR / "player_form.csv"
METRICS_READY= DATA_DIR / "metrics_ready.csv"     # <— NEW primary source
GAME_LINES   = OUT_DIR / "odds_game.csv"
WEATHER      = DATA_DIR / "weather.csv"
COVERAGE     = DATA_DIR / "coverage.csv"
CB_ASSIGN    = DATA_DIR / "cb_assignments.csv"
INJURIES     = DATA_DIR / "injuries.csv"
ROLES        = DATA_DIR / "roles.csv"
CALIB_JSON   = Path("metrics/calibration.json")

PROPS_OUT    = OUT_DIR / "props_priced_clean.csv"
ANCHOR_OUT   = OUT_DIR / "props_market_anchor.csv"

# ----------------------------
# Config knobs
# ----------------------------
BLEND_MODEL = 0.65
BLEND_MKT   = 0.35

SIGMA_DEFAULTS = {
    "rec_yards"     : 26.0,
    "receptions"    : 1.8,
    "rush_yards"    : 23.0,
    "rush_att"      : 3.0,
    "pass_yards"    : 48.0,
    "rush_rec_yards": 32.0,
}
VOLATILITY_WIDEN = 0.15  # 15%

TIERS = [("ELITE", 0.06), ("GREEN", 0.04), ("AMBER", 0.01)]
KELLY_CAP = {"straight": 0.05, "alt": 0.025, "sgp": 0.01}

# ----------------------------
# Helpers
# ----------------------------
def _safe_load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 5:
        print(f"[pricing] WARNING: {path} is empty; using stub.")
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        print(f"[pricing] WARNING: {path} had no columns; using stub.")
        return pd.DataFrame()

def weather_multiplier(wind: float|None, precip: str|None, market: str) -> float:
    m = 1.0
    try:
        w = float(wind) if wind is not None and wind == wind else np.nan
    except Exception:
        w = np.nan
    if not pd.isna(w) and w >= 15:
        if market in {"pass_yards", "rec_yards", "rush_rec_yards"}:
            m *= 0.94
    p = (precip or "").lower()
    if p in {"rain", "snow"}:
        if market in {"rec_yards", "rush_rec_yards"}:
            m *= 0.97
        if market in {"rush_yards", "rush_att"}:
            m *= 1.02
    return m

def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def prob_over_normal(line: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-8:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    return 1.0 - _phi(z)

def american_to_prob(odds: float) -> float:
    if odds > 0:  return 100.0 / (odds + 100.0)
    else:         return (-odds) / ((-odds) + 100.0)

def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1-1e-6)
    if p >= 0.5:  return - (p / (1-p)) * 100.0
    else:         return ((1-p) / p) * 100.0

def devig_two_way(p_over: float, p_under: float) -> tuple[float,float]:
    denom = p_over + p_under
    if denom <= 0: return (0.5, 0.5)
    return (p_over/denom, p_under/denom)

def kelly_fraction(p: float, price: float) -> float:
    dec = 1.0 + (price/100.0 if price>0 else 100.0/(-price))
    b = dec - 1.0
    q = 1.0 - p
    f = (b*p - q) / b if b > 0 else 0.0
    return max(0.0, f)

# ----------------------------
# Loaders
# ----------------------------
def load_team_form() -> pd.DataFrame:
    df1 = _safe_load_csv(str(TEAM_FORM1)); df2 = _safe_load_csv(str(TEAM_FORM2))
    df = df1 if not df1.empty else df2
    if df.empty: return df
    df["team"] = df["team"].astype(str).str.upper()
    return df

def load_player_form() -> pd.DataFrame:
    df1 = _safe_load_csv(str(PLAYER_FORM1)); df2 = _safe_load_csv(str(PLAYER_FORM2))
    df = df1 if not df1.empty else df2
    if df.empty: return df
    df["team"] = df["team"].astype(str).str.upper()
    return df

def load_metrics_ready() -> pd.DataFrame:
    df = _safe_load_csv(str(METRICS_READY))
    if not df.empty and "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper()
    return df

def load_context() -> dict[str,pd.DataFrame]:
    return {
        "lines"   : _safe_load_csv(str(GAME_LINES)),
        "wx"      : _safe_load_csv(str(WEATHER)),
        "coverage": _safe_load_csv(str(COVERAGE)),
        "cb"      : _safe_load_csv(str(CB_ASSIGN)),
        "inj"     : _safe_load_csv(str(INJURIES)),
        "roles"   : _safe_load_csv(str(ROLES)),
    }

# ----------------------------
# μ (volume×efficiency) — now reads opponent from the row (_opp cols)
# ----------------------------
def sigma_for_market(market: str, volatility_flag: bool) -> float:
    base = SIGMA_DEFAULTS.get(market, 25.0)
    if volatility_flag: base *= (1.0 + VOLATILITY_WIDEN)
    return base

def project_mu(row: pd.Series) -> float:
    mk   = str(row.get("market",""))
    line = float(row.get("line", 0.0))

    # team context
    plays_est = float(row.get("plays_est", 120.0))
    proe      = float(row.get("proe", 0.0))

    # opponent context (already merged into row by make_metrics)
    def_pass  = float(row.get("def_pass_epa_opp", 0.0))
    def_rush  = float(row.get("def_rush_epa_opp", 0.0))
    sack_rate = float(row.get("def_sack_rate_opp", 0.0))
    light_box = float(row.get("light_box_rate_opp", 0.0))
    heavy_box = float(row.get("heavy_box_rate_opp", 0.0))

    # player priors
    tgt_share = float(row.get("target_share", 0.17))
    rush_share= float(row.get("rush_share", 0.35))
    ypt       = float(row.get("ypt", 7.8))
    ypc       = float(row.get("ypc", 4.2))
    qb_ypa    = float(row.get("qb_ypa", 6.9))
    yprr      = float(row.get("yprr_proxy", 1.6))

    # script & weather & coverage
    winp      = float(row.get("team_wp", row.get("win_prob", 0.5)))
    status    = (row.get("status") or "").lower()
    coverage_tag = str(row.get("coverage_tags","")).lower()
    cb_penalty   = float(row.get("cb_penalty", 0.0))
    wx_mult   = weather_multiplier(row.get("wind_mph"), row.get("precip"), mk)

    # pressure effects
    qb_pressure_mult = (1.0 - 0.35 * sack_rate) * (1.0 - 0.25 * def_pass)
    qb_pressure_mult = max(0.6, min(1.2, qb_pressure_mult))

    # funnels
    pass_share_team = min(max(0.58 + proe, 0.35), 0.70)
    run_share_team  = 1.0 - pass_share_team
    if (def_rush >= 0.4) and (def_pass <= -0.3):
        pass_share_team *= 0.97; run_share_team = 1.0 - pass_share_team
    elif (def_pass >= 0.4) and (def_rush <= -0.3):
        pass_share_team *= 1.03; run_share_team = 1.0 - pass_share_team

    # box counts → YPC mod
    ypc_mult = 1.0
    if light_box >= 0.60: ypc_mult *= 1.07
    if heavy_box >= 0.60: ypc_mult *= 0.94

    # script escalator
    rb_attempts_bump = 3.0 if winp >= 0.55 else 0.0

    # coverage / CB shadow
    wr_target_mult = 1.0
    wr_ypt_mult    = 1.0
    if "top_shadow" in coverage_tag or cb_penalty >= 0.05:
        wr_target_mult *= 0.92
        wr_ypt_mult    *= 0.94
    if "heavy_zone" in coverage_tag:
        wr_target_mult *= 1.03

    if ("out" in status) or ("doubt" in status):
        tgt_share *= 0.33

    # build μ
    if mk == "rec_yards":
        team_targets = plays_est * pass_share_team
        targets = team_targets * tgt_share * wr_target_mult
        mu = (targets * ypt * wr_ypt_mult) * wx_mult
    elif mk == "receptions":
        team_targets = plays_est * pass_share_team
        targets = team_targets * tgt_share * wr_target_mult
        catch_rate = min(0.85, max(0.45, (ypt / 11.0)))
        mu = (targets * catch_rate) * wx_mult
    elif mk == "rush_yards":
        team_carries = plays_est * run_share_team + rb_attempts_bump
        carries = team_carries * rush_share
        mu = (carries * ypc * ypc_mult) * wx_mult
    elif mk == "rush_att":
        team_carries = plays_est * run_share_team + rb_attempts_bump
        carries = team_carries * rush_share
        mu = carries * wx_mult
    elif mk == "pass_yards":
        attempts = plays_est * pass_share_team
        mu = (attempts * qb_ypa * qb_pressure_mult) * wx_mult
    elif mk == "rush_rec_yards":
        team_targets = plays_est * pass_share_team
        team_carries = plays_est * run_share_team + rb_attempts_bump
        mu_rec  = (team_targets * tgt_share * wr_target_mult) * (ypt * wr_ypt_mult)
        mu_rush = (team_carries * rush_share) * (ypc * ypc_mult)
        mu = (mu_rec + mu_rush) * wx_mult
    else:
        mu = float(line)

    if (mk in {"rec_yards","rush_rec_yards"}) and (float(row.get("yprr_proxy",1.6)) < 1.0):
        mu *= 0.92

    return max(0.0, mu)

# ----------------------------
# Pricing
# ----------------------------
def price_props(props: pd.DataFrame, metrics: pd.DataFrame, ctx: dict[str,pd.DataFrame]) -> pd.DataFrame:
    df = props.copy()

    # Merge metrics (team + player + opponent already baked in)
    # Prefer match on (event_id, player); fallback to player only if no event_id in props
    if not metrics.empty:
        if "event_id" in df.columns and "event_id" in metrics.columns:
            df = df.merge(metrics, on=["event_id","player"], how="left", suffixes=("","_m"))
        else:
            df = df.merge(metrics.drop_duplicates(subset=["player"]), on="player", how="left", suffixes=("","_m"))

    # If win prob missing from metrics, fallback to odds_game merge
    if "team_wp" not in df.columns:
        lines = ctx["lines"]
        if not lines.empty and "event_id" in df.columns and "event_id" in lines.columns:
            df = df.merge(lines[["event_id","home_wp","away_wp","home_team","away_team"]],
                          on="event_id", how="left")
            def _winp(r):
                tm = str(r.get("team","")).upper()
                return float(r.get("home_wp", np.nan)) if tm == str(r.get("home_team","")).upper() else float(r.get("away_wp", np.nan))
            df["team_wp"] = df.apply(_winp, axis=1).fillna(0.5)
        else:
            df["team_wp"] = 0.5

    # Weather (keep if you maintain data/weather.csv keyed by event_id)
    wx = ctx["wx"]
    if not wx.empty and "event_id" in df.columns and "event_id" in wx.columns:
        df = df.merge(wx[["event_id","wind_mph","temp_f","precip","altitude_ft","dome"]], on="event_id", how="left")
    else:
        for c in ["wind_mph","temp_f","precip","altitude_ft","dome"]:
            if c not in df.columns: df[c] = np.nan

    # Coverage & CB
    cov = ctx["coverage"]
    if not cov.empty:
        cov_g = cov.groupby("defense_team")["tag"].apply(lambda s: ",".join(sorted(set(str(x) for x in s)))).reset_index()
        df = df.merge(cov_g.rename(columns={"defense_team":"opp_team","tag":"coverage_tags"}),
                      on="opp_team", how="left")
    else:
        if "coverage_tags" not in df.columns:
            df["coverage_tags"] = ""

    cb = ctx["cb"]
    if not cb.empty and "receiver" in cb.columns:
        cb = cb.rename(columns={"receiver":"player","defense_team":"opp_team"})
        if "penalty" not in cb.columns: cb["penalty"] = 0.0
        df = df.merge(cb[["opp_team","player","penalty"]], on=["opp_team","player"], how="left")
        df["cb_penalty"] = df["penalty"].fillna(0.0)
    else:
        if "cb_penalty" not in df.columns:
            df["cb_penalty"] = 0.0

    # Injuries (if not already in metrics)
    if "status" not in df.columns:
        inj = ctx["inj"]
        if not inj.empty:
            df = df.merge(inj[["player","team","status"]], on=["player","team"], how="left")
        else:
            df["status"] = ""

    # volatility flag uses *_opp columns now
    def _vol_flag(r):
        mk = r.get("market","")
        if mk in {"pass_yards","rec_yards"} and float(r.get("def_sack_rate_opp",0.0)) > 0.08:
            return True
        return False
    df["vol_flag"] = df.apply(_vol_flag, axis=1)

    # μ and σ
    df["model_proj"]  = df.apply(project_mu, axis=1)
    df["model_sigma"] = df.apply(lambda r: sigma_for_market(str(r.get("market","")), bool(r.get("vol_flag", False))), axis=1)

    # Market anchor (de-vig)
    if "over_odds" in df.columns and "under_odds" in df.columns:
        p_over_vig  = df["over_odds"].apply(american_to_prob).astype(float)
        p_under_vig = df["under_odds"].apply(american_to_prob).astype(float)
        p_over_fair, p_under_fair = devig_two_way(p_over_vig, p_under_vig)
        df["mkt_over_fair"]  = p_over_fair
        df["mkt_under_fair"] = p_under_fair
    else:
        side = df.get("side","OVER").astype(str).str.upper()
        odds = df.get("odds", np.nan).astype(float)
        p_vig = odds.apply(american_to_prob)
        df["mkt_over_fair"]  = np.where(side=="OVER", p_vig, 1.0-p_vig)
        df["mkt_under_fair"] = 1.0 - df["mkt_over_fair"]

    # Model Over prob
    df["model_over_pct"] = df.apply(lambda r: prob_over_normal(float(r.get("line",0.0)),
                                                              float(r.get("model_proj",0.0)),
                                                              float(r.get("model_sigma",25.0))), axis=1)

    # Calibration
    if CALIB_JSON.exists():
        try:
            calib = json.loads(CALIB_JSON.read_text())
            def _adj_mu(r):
                mk = r.get("market","")
                s  = float(calib.get(mk,{}).get("mu_shrink", 1.0))
                return float(r["model_proj"]) * s
            df["model_proj"] = df.apply(_adj_mu, axis=1)
            df["model_over_pct"] = df.apply(lambda r: prob_over_normal(float(r["line"]),
                                                                       float(r["model_proj"]),
                                                                       float(r["model_sigma"])), axis=1)
        except Exception as e:
            warnings.warn(f"Failed reading calibration.json: {e}")

    # Blend + fair odds + edge
    df["p_over_blend"]   = BLEND_MODEL*df["model_over_pct"] + BLEND_MKT*df["mkt_over_fair"]
    df["p_under_blend"]  = 1.0 - df["p_over_blend"]
    df["fair_over_odds"] = df["p_over_blend"].apply(prob_to_american)
    df["fair_under_odds"]= df["p_under_blend"].apply(prob_to_american)
    df["edge_abs"]       = (df["p_over_blend"] - df["mkt_over_fair"]).abs()
    df["bet_side"]       = np.where(df["p_over_blend"] >= df["mkt_over_fair"], "OVER", "UNDER")

    def _kelly(r):
        p = float(r["p_over_blend"]) if r["bet_side"]=="OVER" else float(r["p_under_blend"])
        price = float(r["over_odds"] if r["bet_side"]=="OVER" else r.get("under_odds", r.get("odds", -110)))
        f = kelly_fraction(p, price)
        cap = KELLY_CAP["straight"];  cap *= 0.5 if bool(r.get("vol_flag", False)) else 1.0
        return max(0.0, min(cap, f))
    df["kelly"] = df.apply(_kelly, axis=1)

    def _tier(e):
        for name, th in TIERS:
            if e >= th: return name
        return "RED"
    df["tier"] = df["edge_abs"].apply(_tier)

    # Ensure output columns exist
    out_cols = [
        "player","team","opp_team","event_id","market","line",
        "over_odds","under_odds","mkt_over_fair",
        "model_proj","model_sigma","model_over_pct",
        "p_over_blend","fair_over_odds","edge_abs","bet_side","kelly","tier",
        "team_wp","wind_mph","precip","coverage_tags","cb_penalty"
    ]
    for c in out_cols:
        if c not in df.columns: df[c] = np.nan

    return df[out_cols]

# ----------------------------
# CLI
# ----------------------------
def _write_empty_priced():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROPS_OUT.write_text(
        "player,team,opp_team,event_id,market,line,over_odds,under_odds,mkt_over_fair,"
        "model_proj,model_sigma,model_over_pct,p_over_blend,fair_over_odds,edge_abs,bet_side,kelly,tier,"
        "team_wp,wind_mph,precip,coverage_tags,cb_penalty\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", default=str(PROPS_RAW))
    ap.add_argument("--write", default=str(PROPS_OUT))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    props_path = Path(args.props)
    if (not props_path.exists()) or (props_path.stat().st_size == 0):
        print(f"[pricing] WARNING: props file missing/empty: {props_path}")
        _write_empty_priced();  return

    props = _safe_load_csv(str(props_path))
    if props.empty:
        print("[pricing] WARNING: no props rows; writing empty")
        _write_empty_priced();  return

    metrics = load_metrics_ready()
    ctx     = load_context()

    if metrics.empty:
        warnings.warn("metrics_ready.csv is empty — opponent features may be missing.")

    priced = price_props(props, metrics, ctx)

    # Optional: export market anchor for debugging
    try:
        keep = ["player","team","market","line","over_odds","under_odds","mkt_over_fair"]
        priced[keep].to_csv(ANCHOR_OUT, index=False)
    except Exception:
        pass

    priced.to_csv(PROPS_OUT, index=False)
    print(f"[pricing] ✅ wrote {len(priced)} rows → {PROPS_OUT}")

if __name__ == "__main__":
    main()
