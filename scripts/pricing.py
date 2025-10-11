# scripts/pricing.py
from __future__ import annotations
import math
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

# ---------- Odds helpers ----------

def american_to_prob(odds):
    o = pd.to_numeric(odds, errors="coerce")
    return np.where(o > 0, 100 / (o + 100.0), -o / (-o + 100.0))

def prob_to_american(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    dec = 1.0 / p
    am = np.where(dec >= 2.0, (dec - 1.0) * 100.0, -100.0 / (dec - 1.0))
    return am

def devig_two_way(p_over, p_under) -> Tuple[float, float]:
    a = (pd.to_numeric(p_over, errors="coerce")).astype(float)
    b = (pd.to_numeric(p_under, errors="coerce")).astype(float)
    s = a + b
    a_fair = np.where(s > 1e-9, a / s, np.nan)
    b_fair = np.where(s > 1e-9, b / s, np.nan)
    return a_fair, b_fair

# ---------- Distribution helpers ----------

SQRT2 = math.sqrt(2.0)
def _phi(z):    # standard normal CDF
    return 0.5 * (1.0 + math.erf(z / SQRT2))

def p_over_normal(line, mean, sigma):
    if not np.isfinite(line) or not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 1e-9:
        return np.nan
    z = (line - mean) / float(sigma)
    return 1.0 - _phi(z)

# ---------- Safe getters ----------

def _get(row, col, default=0.0):
    v = row.get(col)
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default

# ---------- Feature multipliers (+ SGP hooks) ----------

def apply_multipliers(row: pd.Series, market: str, base_mean: float) -> tuple[float, float]:
    qb_pressure_mult = _get(row, "qb_pressure_mult", 1.0)
    sack_elasticity  = _get(row, "sack_elasticity", 1.0)
    proe_z           = _get(row, "proe_z", 0.0)
    run_funnel       = _get(row, "run_funnel", 0.0)
    pass_funnel      = _get(row, "pass_funnel", 0.0)
    wx_pass          = _get(row, "wx_mult_pass", 1.0)
    wx_rec           = _get(row, "wx_mult_rec", 1.0)
    wx_rush          = _get(row, "wx_mult_rush", 1.0)
    wx_ru_re         = _get(row, "wx_mult_rush_rec", 1.0)

    rb_attempts_escalator = _get(row, "rb_attempts_escalator", 0.0)
    wr_shadow_penalty     = _get(row, "wr_shadow_penalty", 0.0)
    slot_zone_boost       = _get(row, "slot_zone_boost", 0.0)
    vol                   = _get(row, "volatility_flag", 0.0)

    sgp_qb_wr_boost       = _get(row, "sgp_qb_wr_boost", 0.0)  # 0..0.25

    # Base sigma heuristic per market
    if market in ("player_pass_yds",):
        sigma = max(18.0, 0.45 * math.sqrt(max(base_mean, 40.0)))
    elif market in ("player_rush_yds",):
        sigma = max(12.0, 0.60 * math.sqrt(max(base_mean, 20.0)))
    elif market in ("player_rec_yds",):
        sigma = max(14.0, 0.55 * math.sqrt(max(base_mean, 25.0)))
    elif market in ("player_receptions",):
        sigma = max(1.8, 0.38 * math.sqrt(max(base_mean, 1.0)))
    elif market in ("player_anytime_td",):
        sigma = 0.27
    else:
        sigma = max(10.0, 0.5 * math.sqrt(max(base_mean, 10.0)))

    mean = base_mean

    if market == "player_pass_yds":
        mean *= qb_pressure_mult
        mean *= sack_elasticity
        mean *= (1.0 + 0.030 * proe_z)
        mean *= (1.0 + 0.040 * pass_funnel - 0.030 * run_funnel)
        mean *= wx_pass

    elif market == "player_rec_yds":
        mean *= (1.0 + wr_shadow_penalty + slot_zone_boost)
        mean *= (1.0 + 0.022 * proe_z + 0.035 * pass_funnel - 0.020 * run_funnel)
        mean *= wx_rec
        # SGP tilt: bump WR rec yds if he’s the paired WR and game is pass-leaning
        mean *= (1.0 + 0.08 * sgp_qb_wr_boost)

    elif market == "player_receptions":
        mean *= (1.0 + wr_shadow_penalty + slot_zone_boost)
        mean *= (1.0 + 0.025 * proe_z + 0.030 * pass_funnel - 0.015 * run_funnel)
        mean *= wx_rec
        mean *= (1.0 + 0.06 * sgp_qb_wr_boost)

    elif market == "player_rush_yds":
        mean *= (1.0 + 0.050 * run_funnel - 0.025 * pass_funnel)
        mean += rb_attempts_escalator
        mean *= wx_rush

    elif market == "player_anytime_td":
        pass

    if vol >= 1.0:
        sigma *= 1.18

    return mean, sigma

# ---------- Role priors → baseline mean ----------

def choose_baseline_mean(row: pd.Series, market: str, fallback_line: float) -> float:
    base = float(fallback_line) if np.isfinite(fallback_line) else 0.0

    # Enrich using priors if present (routes/yprr, ypc/att, ypa/att, targets)
    if market == "player_rec_yds":
        yprr   = row.get("yprr")
        routes = row.get("routes")
        if pd.notna(yprr) and pd.notna(routes):
            base = 0.55 * base + 0.45 * float(yprr) * float(routes)

    elif market == "player_rush_yds":
        ypc = row.get("ypc")
        att = row.get("rush_att_proj")
        if pd.notna(ypc) and pd.notna(att):
            base = 0.55 * base + 0.45 * float(ypc) * float(att)

    elif market == "player_pass_yds":
        ypa = row.get("qb_ypa")
        att = row.get("qb_att_proj")
        if pd.notna(ypa) and pd.notna(att):
            base = 0.55 * base + 0.45 * float(ypa) * float(att)

    elif market == "player_receptions":
        tgs = row.get("targets_proj")
        if pd.notna(tgs):
            base = 0.65 * base + 0.35 * float(tgs) * 0.70  # 70% catch prior

    base *= 1.005
    return base

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props",   default="outputs/props_raw.csv")
    ap.add_argument("--metrics", default="data/metrics_ready.csv")
    ap.add_argument("--out",     default="outputs/props_priced_clean.csv")
    ap.add_argument("--top",     default="outputs/props_priced_top.csv")
    args = ap.parse_args()

    Path("outputs").mkdir(parents=True, exist_ok=True)

    props = pd.read_csv(args.props)
    if props.empty:
        print(f"[pricing] No props found at {args.props}")
        pd.DataFrame().to_csv(args.out, index=False)
        return

    metrics = pd.read_csv(args.metrics) if Path(args.metrics).exists() else pd.DataFrame()

    # Normalize props columns
    for col in ["event_id","date","book","market","player","side","line","price_american","commence_time"]:
        if col not in props.columns:
            props[col] = np.nan

    # Pivot into O/U pairs
    key = ["event_id","date","book","market","player","line"]
    over = props[props["side"].str.lower() == "over"].copy()
    under = props[props["side"].str.lower() == "under"].copy()

    over  = over.groupby(key, as_index=False).agg({"price_american":"first","commence_time":"first"}).rename(columns={"price_american":"over_american"})
    under = under.groupby(key, as_index=False).agg({"price_american":"first"}).rename(columns={"price_american":"under_american"})
    book  = over.merge(under, on=key, how="outer")
    book  = book.dropna(subset=["over_american","under_american"], how="any")

    book["p_over"]  = american_to_prob(book["over_american"])
    book["p_under"] = american_to_prob(book["under_american"])
    book["p_over_fair"], book["p_under_fair"] = devig_two_way(book["p_over"], book["p_under"])

    # Merge metrics (event_id+player) → (date+player) → (player)
    m = metrics.copy()
    for c in ["event_id","date","player"]:
        if c not in m.columns:
            m[c] = np.nan

    merged = book.merge(m, on=["event_id","player"], how="left", suffixes=("",""))
    need_date = merged["qb_pressure_mult"].isna() & merged["date"].notna()
    if need_date.any():
        fallback = book[need_date].merge(m, on=["date","player"], how="left", suffixes=("",""))
        merged.loc[need_date, merged.columns] = fallback.reindex(columns=merged.columns).values

    need_player = merged["qb_pressure_mult"].isna()
    if need_player.any():
        fallback = book[need_player].merge(m.drop(columns=["event_id","date"], errors="ignore"), on=["player"], how="left")
        merged.loc[need_player, merged.columns] = fallback.reindex(columns=merged.columns).values

    # Model probabilities
    rows = []
    for _, r in merged.iterrows():
        market = str(r.get("market") or "")
        line   = pd.to_numeric(r.get("line"), errors="coerce")
        if not np.isfinite(line):
            continue

        base = choose_baseline_mean(r, market, line)
        mean_adj, sigma_adj = apply_multipliers(r, market, base)

        if market == "player_anytime_td":
            p_fair = r.get("p_over_fair")
            p_prior = 0.45 if pd.isna(p_fair) else float(p_fair)
            logit = math.log(p_prior / max(1e-6, 1 - p_prior))
            logit += 0.25 * _get(r, "run_funnel", 0.0)
            logit -= 0.18 * _get(r, "pass_funnel", 0.0)
            logit += 0.20 * (_get(r, "rb_attempts_escalator", 0.0) / 6.0)
            logit += 0.15 * _get(r, "proe_z", 0.0)
            model_p_over = 1 / (1 + math.exp(-logit))
        else:
            model_p_over = p_over_normal(float(line), float(mean_adj), float(sigma_adj))

        rows.append({
            "event_id": r.get("event_id"),
            "date": r.get("date"),
            "commence_time": r.get("commence_time"),
            "book": r.get("book"),
            "market": market,
            "player": r.get("player"),
            "line": float(line),
            "over_american": r.get("over_american"),
            "under_american": r.get("under_american"),
            "p_over_fair": r.get("p_over_fair"),
            "p_under_fair": r.get("p_under_fair"),
            "model_p_over": model_p_over,
            "model_mean": mean_adj,
            "model_sigma": sigma_adj,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[pricing] No priceable props after merging and modeling.")
        out.to_csv(args.out, index=False)
        return

    out["edge_over"] = out["model_p_over"] - out["p_over_fair"]
    # Kelly on American odds (very rough, safeguards applied)
    dec = np.abs(pd.to_numeric(out["over_american"], errors="coerce")) / 100.0
    kelly = (out["model_p_over"] * (dec + 1) - 1) / dec
    out["kelly_frac"] = kelly.replace([np.inf, -np.inf], np.nan).clip(lower=0)

    keep = [
        "event_id","date","commence_time","book","market","player","line",
        "over_american","under_american",
        "p_over_fair","p_under_fair","model_p_over","edge_over",
        "model_mean","model_sigma","kelly_frac",
    ]
    out = out[keep].sort_values(["market","edge_over"], ascending=[True, False])

    out.to_csv(args.out, index=False)
    print(f"[pricing] wrote → {args.out} rows={len(out)}")

    top = (
        out.assign(rank=out.groupby("market")["edge_over"].rank(method="first", ascending=False))
           .query("rank <= 10")
           .sort_values(["market","rank"])
    )
    top.to_csv(args.top, index=False)
    print(f"[pricing] wrote top picks → {args.top}")

if __name__ == "__main__":
    main()
