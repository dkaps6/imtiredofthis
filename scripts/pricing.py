# scripts/pricing.py
from __future__ import annotations
import argparse, math, json, warnings
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import os

def _safe_load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 5:
        print(f"[pricing] WARNING: {path} is empty, writing stub frame.")
        return pd.DataFrame(columns=["event_id","commence_time","book","market","player","line","odds"])
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        print(f"[pricing] WARNING: {path} had no columns, writing stub frame.")
        return pd.DataFrame(columns=["event_id","commence_time","book","market","player","line","odds"])

# ----------------------------
# Paths (keep these as-is unless you changed locations)
# ----------------------------
DATA_DIR     = Path("data")
OUT_DIR      = Path("outputs")
METRICS_DIR  = Path("outputs/metrics")

PROPS_RAW    = OUT_DIR / "props_raw.csv"
TEAM_FORM1   = METRICS_DIR / "team_form.csv"
TEAM_FORM2   = DATA_DIR / "team_form.csv"
PLAYER_FORM1 = METRICS_DIR / "player_form.csv"
PLAYER_FORM2 = DATA_DIR / "player_form.csv"
FEATURES     = DATA_DIR / "features_external.csv"
GAME_LINES   = OUT_DIR / "odds_game.csv"        # note: your engine writes odds_game.csv
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
def _read_first(*paths: Path) -> pd.DataFrame:
    for p in paths:
        if p.exists():
            try:    return pd.read_csv(p)
            except: pass
    return pd.DataFrame()

def load_team_form() -> pd.DataFrame:
    df = _read_first(TEAM_FORM1, TEAM_FORM2)
    if df.empty: return df
    need = ["team","def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","light_box_rate","heavy_box_rate"]
    for c in need:
        if c not in df.columns: df[c] = 0.0
    df["team"] = df["team"].astype(str).str.upper()
    return df

def load_player_form() -> pd.DataFrame:
    df = _read_first(PLAYER_FORM1, PLAYER_FORM2)
    if df.empty: return df
    need = ["player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","ypt","qb_ypa"]
    for c in need:
        if c not in df.columns: df[c] = 0.0
    df["team"] = df["team"].astype(str).str.upper()
    return df

def load_context() -> dict[str,pd.DataFrame]:
    return {
        "features": _read_first(FEATURES),
        "lines"   : _read_first(GAME_LINES),
        "wx"      : _read_first(WEATHER),
        "coverage": _read_first(COVERAGE),
        "cb"      : _read_first(CB_ASSIGN),
        "inj"     : _read_first(INJURIES),
        "roles"   : _read_first(ROLES),
    }

# ----------------------------
# μ (volume×efficiency) builder
# ----------------------------
def project_mu(row: pd.Series, team: pd.DataFrame, pform: pd.DataFrame, wx: pd.DataFrame) -> float:
    mk   = row.get("market", "")
    line = float(row.get("line", 0.0))
    player = str(row.get("player", ""))
    team_abbr = str(row.get("team", "")).upper()
    opp_abbr  = str(row.get("opp_team", "")).upper()

    pf = pform[(pform["player"]==player) & (pform["team"]==team_abbr)]
    if pf.empty:
        pf = pform[pform["player"].str.contains(player.split()[-1], case=False, na=False) & (pform["team"]==team_abbr)]

    tf = team[team["team"]==team_abbr]
    if tf.empty:
        plays_est = 120.0; proe = 0.0
        def_pass_z = def_rush_z = sack_z = light_box = heavy_box = 0.0
    else:
        plays_est = float(tf["plays_est"].mean()) if "plays_est" in tf.columns else 120.0
        proe = float(tf["proe"].mean())
        def_pass_z = float(tf["def_pass_epa"].mean())
        def_rush_z = float(tf["def_rush_epa"].mean())
        sack_z     = float(tf["def_sack_rate"].mean())
        light_box  = float(tf["light_box_rate"].mean())
        heavy_box  = float(tf["heavy_box_rate"].mean())

    pass_share_team = min(max(0.58 + proe, 0.35), 0.70)
    run_share_team  = 1.0 - pass_share_team

    tgt_share = float(pf["target_share"].mean()) if not pf.empty else 0.17
    rush_share= float(pf["rush_share"].mean())   if not pf.empty else 0.35
    ypt       = float(pf["ypt"].mean())         if not pf.empty else 7.8
    ypc       = float(pf["ypc"].mean())         if not pf.empty else 4.2
    qb_ypa    = float(pf["qb_ypa"].mean())      if not pf.empty else 6.9
    yprr      = float(pf["yprr_proxy"].mean())  if not pf.empty else 1.6

    qb_pressure_mult = (1.0 - 0.35 * (sack_z)) * (1.0 - 0.25 * (def_pass_z))
    qb_pressure_mult = max(0.6, min(1.2, qb_pressure_mult))

    if (def_rush_z >= 0.4) and (def_pass_z <= -0.3):
        pass_share_team *= 0.97; run_share_team = 1.0 - pass_share_team
    elif (def_pass_z >= 0.4) and (def_rush_z <= -0.3):
        pass_share_team *= 1.03; run_share_team = 1.0 - pass_share_team

    ypc_mult = 1.0
    if light_box >= 0.60: ypc_mult *= 1.07
    if heavy_box >= 0.60: ypc_mult *= 0.94

    winp = float(row.get("win_prob", 0.5))
    rb_attempts_bump = 3.0 if winp >= 0.55 else 0.0

    coverage_tag = str(row.get("coverage_tags","")).lower()
    cb_penalty   = float(row.get("cb_penalty", 0.0))
    wr_target_mult = 1.0
    wr_ypt_mult    = 1.0
    if "top_shadow" in coverage_tag or cb_penalty >= 0.05:
        wr_target_mult *= 0.92
        wr_ypt_mult    *= 0.94
    if "heavy_zone" in coverage_tag:
        wr_target_mult *= 1.03

    wx_mult = weather_multiplier(row.get("wind_mph"), row.get("precip"), mk)

    status = (row.get("status") or "").lower()
    if ("out" in status) or ("doubt" in status):
        tgt_share *= 0.33

    mu = 0.0
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

    if (mk in {"rec_yards","rush_rec_yards"}) and (yprr < 1.0):
        mu *= 0.92

    return max(0.0, mu)

def sigma_for_market(market: str, volatility_flag: bool) -> float:
    base = SIGMA_DEFAULTS.get(market, 25.0)
    if volatility_flag: base *= (1.0 + VOLATILITY_WIDEN)
    return base

# ----------------------------
# Pricing
# ----------------------------
def price_props(props: pd.DataFrame, team: pd.DataFrame, pform: pd.DataFrame, ctx: dict[str,pd.DataFrame]) -> pd.DataFrame:
    df = props.copy()

    # WP / lines
    lines = ctx["lines"]
    if not lines.empty and "event_id" in df.columns and "event_id" in lines.columns:
        df = df.merge(lines[["event_id","home_wp","away_wp","home_team","away_team"]],
                      on="event_id", how="left")
        def _winp(r):
            tm = str(r.get("team","")).upper()
            return float(r.get("home_wp", np.nan)) if tm == str(r.get("home_team","")).upper() else float(r.get("away_wp", np.nan))
        df["win_prob"] = df.apply(_winp, axis=1).fillna(0.5)
    else:
        df["win_prob"] = 0.5

    # weather
    wx = ctx["wx"]
    if not wx.empty and "event_id" in df.columns and "event_id" in wx.columns:
        df = df.merge(wx[["event_id","wind_mph","temp_f","precip","altitude_ft","dome"]], on="event_id", how="left")
    else:
        for c in ["wind_mph","temp_f","precip","altitude_ft","dome"]:
            if c not in df.columns: df[c] = np.nan

    # coverage
    cov = ctx["coverage"]
    if not cov.empty and "opp_team" in df.columns:
        cov = cov.groupby("defense_team")["tag"].apply(lambda s: ",".join(sorted(set(str(x) for x in s)))).reset_index()
        df = df.merge(cov.rename(columns={"defense_team":"opp_team","tag":"coverage_tags"}),
                      on="opp_team", how="left")
    else:
        df["coverage_tags"] = ""

    cb = ctx["cb"]
    if not cb.empty and "receiver" in cb.columns:
        cb["penalty"] = cb.get("penalty", 0.0)
        df = df.merge(cb.rename(columns={"receiver":"player","defense_team":"opp_team"})[["opp_team","player","penalty"]],
                      on=["opp_team","player"], how="left")
        df["cb_penalty"] = df["penalty"].fillna(0.0)
    else:
        df["cb_penalty"] = 0.0

    inj = ctx["inj"]
    if "status" not in df.columns:
        if not inj.empty:
            df = df.merge(inj[["player","team","status"]], on=["player","team"], how="left")
        else:
            df["status"] = ""
            
if "opp_team" not in df.columns:
    # derive from matchup or skip merge safely
    print("[pricing] warning: opp_team missing; skipping defensive merge")
else:
    df = df.merge(
        opp_t[["opp_team","def_sack_rate","def_pass_epa","def_rush_epa","light_box_rate","heavy_box_rate"]],
        on="opp_team", how="left"
    )

    opp_t = team.rename(columns={"team":"opp_team"})
    df = df.merge(opp_t[["opp_team","def_sack_rate","def_pass_epa","def_rush_epa","light_box_rate","heavy_box_rate"]],
                  on="opp_team", how="left")

    def _vol_flag(r):
        mk = r.get("market","")
        if mk in {"pass_yards","rec_yards"} and float(r.get("def_sack_rate",0.0)) > 0.08:
            return True
        return False
    df["vol_flag"] = df.apply(_vol_flag, axis=1)

    # Build μ and σ
    mus, sigs = [], []
    for _, r in df.iterrows():
        mu = project_mu(r, team, pform, wx)
        sigma = sigma_for_market(str(r.get("market","")), bool(r.get("vol_flag", False)))
        mus.append(mu); sigs.append(sigma)
    df["model_proj"]  = mus
    df["model_sigma"] = sigs

    # Market anchor (de-vig)
    if "over_odds" in df.columns and "under_odds" in df.columns:
        p_over_vig  = df["over_odds"].apply(american_to_prob).astype(float)
        p_under_vig = df["under_odds"].apply(american_to_prob).astype(float)
        p_over_fair, p_under_fair = devig_two_way(p_over_vig, p_under_vig)
        df["mkt_over_fair"]  = p_over_fair
        df["mkt_under_fair"] = p_under_fair
    else:
        side = df.get("side","OVER").str.upper()
        odds = df.get("odds", np.nan).astype(float)
        p_vig = odds.apply(american_to_prob)
        df["mkt_over_fair"]  = np.where(side=="OVER", p_vig, 1.0-p_vig)
        df["mkt_under_fair"] = 1.0 - df["mkt_over_fair"]

    # Model Over prob
    df["model_over_pct"] = df.apply(lambda r: prob_over_normal(float(r["line"]), float(r["model_proj"]), float(r["model_sigma"])), axis=1)

    # Calibration
    if CALIB_JSON.exists():
        try:
            calib = json.loads(CALIB_JSON.read_text())
            def _adj_mu(r):
                mk = r.get("market","")
                s  = float(calib.get(mk,{}).get("mu_shrink", 1.0))
                return float(r["model_proj"]) * s
            df["model_proj"] = df.apply(_adj_mu, axis=1)
            df["model_over_pct"] = df.apply(lambda r: prob_over_normal(float(r["line"]), float(r["model_proj"]), float(r["model_sigma"])), axis=1)
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

    keep_trace = ["market","line","over_odds","under_odds","mkt_over_fair","model_proj","model_sigma",
                  "model_over_pct","p_over_blend","fair_over_odds","edge_abs","bet_side","kelly","tier",
                  "win_prob","wind_mph","precip","coverage_tags","cb_penalty"]
    for c in keep_trace:
        if c not in df.columns: df[c] = np.nan

    return df

# ----------------------------
# CLI
# ----------------------------
def _write_empty_priced():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROPS_OUT.write_text(
        "player,team,opp_team,event_id,market,line,over_odds,under_odds,mkt_over_fair,"
        "model_proj,model_sigma,model_over_pct,p_over_blend,fair_over_odds,edge_abs,bet_side,kelly,tier,"
        "win_prob,wind_mph,precip,coverage_tags,cb_penalty\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", default=str(PROPS_RAW))
    ap.add_argument("--write", default=str(PROPS_OUT))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Safety: no props → write empty and exit cleanly
    props_path = Path(args.props)
    if (not props_path.exists()) or (props_path.stat().st_size == 0):
        print(f"[pricing] WARNING: props file missing/empty: {props_path}")
        _write_empty_priced();  return

    props = _safe_load_csv(props_path)
    if props.empty:
        print("[pricing] WARNING: no props rows; writing empty")
        _write_empty_priced();  return

    team  = load_team_form()
    pform = load_player_form()
    ctx   = load_context()

    if team.empty or pform.empty:
        warnings.warn("Missing team_form or player_form; pricing will be weak or fail.")

    df = price_props(props, team, pform, ctx)

    # Optional: export market anchor for debugging
    try:
        df[["player","team","market","line","over_odds","under_odds","mkt_over_fair"]].to_csv(ANCHOR_OUT, index=False)
    except Exception:
        pass

    out_cols = [
        "player","team","opp_team","event_id","market","line",
        "over_odds","under_odds","mkt_over_fair",
        "model_proj","model_sigma","model_over_pct",
        "p_over_blend","fair_over_odds","edge_abs","bet_side","kelly","tier",
        "win_prob","wind_mph","precip","coverage_tags","cb_penalty"
    ]
    for c in out_cols:
        if c not in df.columns: df[c] = np.nan

    df[out_cols].to_csv(PROPS_OUT, index=False)
    print(f"[pricing] ✅ wrote {len(df)} rows → {PROPS_OUT}")

if __name__ == "__main__":
    main()
