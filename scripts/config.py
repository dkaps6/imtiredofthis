# scripts/config.py
# Central configuration for the pipeline: paths, providers, markets/books,
# modeling knobs, correlations, defaults. Reads secrets from env.

from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, List, Set

# ----------------------------
# Paths & directories
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root
DIR = {
    "data": ROOT / "data",
    "logs": ROOT / "logs",
    "outputs": ROOT / "outputs",
    "tmp_props": ROOT / "outputs" / "_tmp_props",
    "models_out": ROOT / "outputs" / "models",
    "metrics_out": ROOT / "outputs" / "metrics",
}

FILES = {
    "team_form": DIR["data"] / "team_form.csv",
    "player_form": DIR["data"] / "player_form.csv",
    "metrics_ready": DIR["data"] / "metrics_ready.csv",
    "props_raw": DIR["outputs"] / "props_raw.csv",
    "props_raw_wide": DIR["outputs"] / "props_raw_wide.csv",
    "odds_game": DIR["outputs"] / "odds_game.csv",
    "props_priced": DIR["outputs"] / "props_priced_clean.csv",
    "slate_predictions": DIR["models_out"] / "slate_predictions.parquet",
    "slate_diagnostics": DIR["models_out"] / "slate_diagnostics.json",
    "calibration": DIR["data"] / "calibration.json",
    "market_sigmas": DIR["data"] / "market_sigmas.json",
}

def ensure_dirs() -> None:
    for p in [*DIR.values()]:
        p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Providers & Secrets (env)
# ----------------------------
# These are read at runtime; do NOT hardcode values here.
ENV = {
    "ODDS_API_KEY": os.getenv("ODDS_API_KEY", "").strip(),
    "NFLGSIS_USER": os.getenv("NFLGSIS_USER", "").strip(),
    "NFLGSIS_PASS": os.getenv("NFLGSIS_PASS", "").strip(),
    "MSF_API_KEY": os.getenv("MSF_API_KEY", "").strip(),
    "API_SPORTS_KEY": os.getenv("API_SPORTS_KEY", "").strip(),
}

def require_env(key: str) -> str:
    v = ENV.get(key, "")
    if not v:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return v

# ----------------------------
# Odds API (v4) settings
# ----------------------------
ODDS = {
    "sport_key": "americanfootball_nfl",
    "region": os.getenv("ODDS_API_REGION", "us"),
    "base_url": "https://api.the-odds-api.com/v4",
    # We use sport-level bulk for certain player markets (per v4 quirks)
    "bulk_only_markets": {"player_reception_yds"},  # receiving yards canonical in our repo
}

# Books to query by default (normalized to odds API names: lowercase, spaces→underscores)
BOOKS_DEFAULT: List[str] = [
    b.strip() for b in os.getenv("BOOKS_DEFAULT", "draftkings,fanduel,betmgm,caesars").split(",") if b.strip()
]

# ----------------------------
# Markets (canonical + aliases)
# ----------------------------
# Canonical keys (the ones your code expects AFTER alias normalization)
# Receiving yards canonical = player_reception_yds (not player_receiving_yards)
MARKETS_DEFAULT: List[str] = [
    # game lines are handled separately, kept here for reference
    # "h2h", "spreads", "totals",
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",        # receiving yards (canonical for v4 fetcher)
    "player_rush_reception_yds",   # combo (rush+rec)
    "player_anytime_td",
]

# Aliases we accept on input (CLI, engine, or external callers) → canonical
MARKET_ALIASES: Dict[str, str] = {
    # receiving yards variants
    "player_rec_yds": "player_reception_yds",
    "player_receiving_yards": "player_reception_yds",
    "player_receiving_yds": "player_reception_yds",
    "player_reception_yds": "player_reception_yds",
    "player_receiving_yards_ou": "player_reception_yds",
    "player_receiving_yds_over_under": "player_reception_yds",

    # rush+rec combo variants
    "player_rush_rec_yds": "player_rush_reception_yds",
    "rush_rec": "player_rush_reception_yds",

    # passing
    "player_passing_yards": "player_pass_yds",

    # rushing
    "player_rushing_yards": "player_rush_yds",
}

def normalize_market(m: str) -> str:
    return MARKET_ALIASES.get(m, m)

# ----------------------------
# Modeling knobs (global)
# ----------------------------
# Blend of model vs market (closing-line anchoring)
BLEND = {
    "model_weight": float(os.getenv("BLEND_MODEL_W", 0.65)),
    "market_weight": float(os.getenv("BLEND_MARKET_W", 0.35)),
}

# Default sigmas per market (used if not learned/available)
# You can override live by writing JSON to FILES["market_sigmas"] = {"player_pass_yds": 48.0, ...}
DEFAULT_SIGMAS: Dict[str, float] = {
    "player_pass_yds": 48.0,
    "player_reception_yds": 26.0,
    "player_receptions": 1.8,
    "player_rush_yds": 24.0,
    "player_rush_reception_yds": 35.0,
    "player_anytime_td": 0.0,  # Bernoulli; sigma not used
}

# Volatility widening factors (pressure mismatch, QB inconsistency, etc.)
VOLATILITY = {
    "mild": 1.10,
    "mod": 1.15,
    "high": 1.20,
}

# Kelly staking caps
KELLY_CAPS = {
    "straight": 0.05,   # ≤ 5%
    "alt": 0.025,       # ≤ 2.5%
    "sgp": 0.01,        # ≤ 1%
}

# SGP correlations (base)
CORR = {
    ("QB_pass_yds", "WR_rec_yds"): 0.60,
    ("QB_pass_yds", "RB_rush_yds"): -0.35,
    ("WR1_rec_yds", "WR2_rec_yds"): 0.20,
}

# Monte Carlo settings
MC = {
    "iterations": int(os.getenv("MC_ITERATIONS", "25000")),  # You asked for higher precision
    "seed": int(os.getenv("MC_SEED", "42")),
    "tail_sd_expand_pct": 0.15,  # apply if volatility flagged
}

# Weather heuristics (applied in pricing/builders)
WEATHER = {
    "wind_downshift_threshold_mph": 15,
    "wind_ypr_mult": 0.94,   # pass/rec penalties for deep aDOT
    "rain_yac_mult": 0.97,   # YAC downshift
    "rush_rate_bump_rain": 1.02,
}

# Script escalators & injury caps (also referenced in elite_rules.py if present)
RULES = {
    "qb_pressure_mult": -0.35,   # QB baseline mult per Z(pressure)
    "qb_opp_pass_epa_mult": -0.25,
    "sack_to_att_elasticity": -0.15,
    "alpha_wr_injury_cap_pctile": (0.33, 0.50),
    "injury_redistribution": {"WR2": 0.60, "SLOT_TE": 0.30, "RB": 0.10},
    "boxcount_ypp_light": 1.07,
    "boxcount_ypp_heavy": 0.94,
    "pace_smoothing": 0.5,
    "script_rb_attempts_fav": (2, 4),
}

# Calibration (week-to-week shrinkage)
CALIBRATION = {
    "apply": True,
    "alpha": 0.10,            # mu <- 0.9*mu + 0.1*mu_market
    "brier_threshold": None,  # reserved for future automated toggling
}

# ----------------------------
# Helpers for other modules
# ----------------------------
def books_from_env() -> List[str]:
    raw = os.getenv("BOOKS", "")
    if not raw:
        return BOOKS_DEFAULT
    return [b.strip() for b in raw.split(",") if b.strip()]

def markets_from_env() -> List[str]:
    raw = os.getenv("MARKETS", "")
    if not raw:
        return MARKETS_DEFAULT
    vals = [normalize_market(m.strip()) for m in raw.split(",") if m.strip()]
    # drop duplicates but keep order
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def dump_config_summary() -> str:
    """Short text summary for logging."""
    d = {
        "books": books_from_env(),
        "markets": markets_from_env(),
        "region": ODDS["region"],
        "blend": BLEND,
        "mc": {"iterations": MC["iterations"], "seed": MC["seed"]},
        "bulk_only_markets": sorted(ODDS["bulk_only_markets"]),
        "paths": {k: str(v) for k, v in FILES.items()},
    }
    return json.dumps(d, indent=2)

# Ensure directories exist at import-time (safe no-op if already created)
ensure_dirs()
