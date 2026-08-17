#!/usr/bin/env python3
"""Independent player-prop pricing engine.

Unlike the legacy pricing module, the model projection is not initialized at
the sportsbook line.  The line is used only after an independent projection is
built from team volume, player opportunity shares, efficiency, environment,
and matchup context.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.config import DEFAULT_SIGMAS
from scripts.runtime_context import resolve_season

DATA = Path("data")
OUTPUTS = Path("outputs")
METRICS = DATA / "metrics_ready.csv"
PLAYER_FORM = DATA / "player_form_consensus.csv"
OUT = OUTPUTS / "props_priced_clean.csv"

MARKET_MAP = {
    "player_pass_yds": "pass_yards", "player_passing_yards": "pass_yards", "pass_yards": "pass_yards",
    "player_rush_yds": "rush_yards", "player_rushing_yards": "rush_yards", "rush_yards": "rush_yards",
    "player_reception_yds": "rec_yards", "player_rec_yds": "rec_yards", "player_receiving_yards": "rec_yards", "rec_yards": "rec_yards",
    "player_receptions": "receptions", "receptions": "receptions",
    "player_rush_att": "rush_att", "rush_att": "rush_att",
    "player_rush_reception_yds": "rush_rec_yards", "player_rush_rec_yds": "rush_rec_yards", "rush_rec_yards": "rush_rec_yards",
    "player_anytime_td": "anytime_td", "anytime_td": "anytime_td", "atd": "anytime_td",
}
SIGMA = {
    "pass_yards": float(DEFAULT_SIGMAS.get("player_pass_yds", 48.0)),
    "rush_yards": float(DEFAULT_SIGMAS.get("player_rush_yds", 24.0)),
    "rec_yards": float(DEFAULT_SIGMAS.get("player_reception_yds", 26.0)),
    "receptions": float(DEFAULT_SIGMAS.get("player_receptions", 1.8)),
    "rush_rec_yards": float(DEFAULT_SIGMAS.get("player_rush_reception_yds", 35.0)),
    "rush_att": 3.2,
}


def _prob_from_american(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)


def _fair_market_prob(over_odds, under_odds) -> tuple[float, float]:
    po = _prob_from_american(over_odds)
    pu = _prob_from_american(under_odds)
    if pd.notna(po) and pd.notna(pu) and po + pu > 0:
        over = po / (po + pu)
        return over, 1.0 - over
    if pd.notna(po):
        return po, 1.0 - po
    if pd.notna(pu):
        return 1.0 - pu, pu
    return np.nan, np.nan


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob_over(mu: float, sigma: float, line: float) -> float:
    if not all(np.isfinite(v) for v in (mu, sigma, line)) or sigma <= 0:
        return np.nan
    return float(1.0 - _norm_cdf((line - mu) / sigma))


def _fair_odds(p: float) -> float:
    if pd.isna(p):
        return np.nan
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return -(100.0 * p / (1.0 - p)) if p >= 0.5 else 100.0 * (1.0 - p) / p


def _num(row: pd.Series, *names, default=np.nan) -> float:
    for name in names:
        if name in row.index:
            try:
                v = float(row.get(name))
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return float(default)


def _team_volume(row: pd.Series) -> dict[str, float]:
    pace = _num(row, "pace", "neutral_pace")
    plays = _num(row, "plays_est")
    if not np.isfinite(plays):
        # pace is offensive seconds/snap; a team has roughly half of regulation
        # possession time, so 1800/pace is a sane team-play estimate.
        plays = 1800.0 / pace if np.isfinite(pace) and pace > 0 else 64.0
    plays = float(np.clip(plays, 50.0, 80.0))

    proe = _num(row, "proe", "pass_rate_over_expected", default=0.0)
    pass_rate = 0.58 + (proe if np.isfinite(proe) else 0.0)
    wp = _num(row, "team_wp", "team_win_prob")
    if np.isfinite(wp):
        if wp >= 0.60:
            pass_rate -= 0.02
        elif wp <= 0.40:
            pass_rate += 0.02
    pass_rate = float(np.clip(pass_rate, 0.35, 0.75))
    return {
        "plays": plays,
        "pass_rate": pass_rate,
        "pass_att": plays * pass_rate,
        "rush_att": plays * (1.0 - pass_rate),
    }


def _matchup_multiplier(row: pd.Series, market: str) -> float:
    mult = 1.0
    pass_epa = _num(row, "def_pass_epa_opp", "opp_def_pass_epa", "def_pass_epa")
    rush_epa = _num(row, "def_rush_epa_opp", "opp_def_rush_epa", "def_rush_epa")
    if market in {"pass_yards", "rec_yards", "receptions", "rush_rec_yards"} and np.isfinite(pass_epa):
        mult *= 1.0 + float(np.clip(pass_epa, -0.30, 0.30)) * 0.25
    if market in {"rush_yards", "rush_att", "rush_rec_yards"} and np.isfinite(rush_epa):
        mult *= 1.0 + float(np.clip(rush_epa, -0.30, 0.30)) * 0.25

    wind = _num(row, "wind_mph")
    if np.isfinite(wind) and wind >= 15 and market in {"pass_yards", "rec_yards", "rush_rec_yards"}:
        mult *= 0.94
    precip = str(row.get("precip", "") or "").lower()
    if any(token in precip for token in ("rain", "snow")):
        if market in {"rec_yards", "rush_rec_yards"}:
            mult *= 0.97
        if market in {"rush_yards", "rush_att"}:
            mult *= 1.02
    return float(np.clip(mult, 0.75, 1.25))


def _projection(row: pd.Series, market: str) -> float:
    tv = _team_volume(row)
    target_share = _num(row, "target_share", "tgt_share")
    rush_share = _num(row, "rush_share")
    route_rate = _num(row, "route_rate")
    ypt = _num(row, "ypt")
    yprr = _num(row, "yprr", "yprr_proxy")
    ypc = _num(row, "ypc")
    ypa = _num(row, "ypa", "ypa_prior")
    catch_rate = _num(row, "receptions_per_target", "catch_rate", default=0.64)

    targets = tv["pass_att"] * target_share if np.isfinite(target_share) else np.nan
    rushes = tv["rush_att"] * rush_share if np.isfinite(rush_share) else np.nan
    rec_yards_target = targets * ypt if np.isfinite(targets) and np.isfinite(ypt) else np.nan
    rec_yards_route = tv["pass_att"] * route_rate * yprr if all(np.isfinite(v) for v in (route_rate, yprr)) else np.nan
    if np.isfinite(rec_yards_target) and np.isfinite(rec_yards_route):
        rec_yards = 0.5 * rec_yards_target + 0.5 * rec_yards_route
    else:
        rec_yards = rec_yards_target if np.isfinite(rec_yards_target) else rec_yards_route
    rush_yards = rushes * ypc if np.isfinite(rushes) and np.isfinite(ypc) else np.nan

    if market == "pass_yards":
        mu = tv["pass_att"] * ypa if np.isfinite(ypa) else np.nan
    elif market == "rec_yards":
        mu = rec_yards
    elif market == "receptions":
        mu = targets * catch_rate if np.isfinite(targets) and np.isfinite(catch_rate) else np.nan
    elif market == "rush_yards":
        mu = rush_yards
    elif market == "rush_att":
        mu = rushes
    elif market == "rush_rec_yards":
        mu = rec_yards + rush_yards if np.isfinite(rec_yards) and np.isfinite(rush_yards) else np.nan
    else:
        return np.nan
    return float(mu * _matchup_multiplier(row, market)) if np.isfinite(mu) else np.nan


def _load() -> pd.DataFrame:
    if not METRICS.exists() or METRICS.stat().st_size == 0:
        raise RuntimeError("data/metrics_ready.csv missing or empty")
    df = pd.read_csv(METRICS)
    df.columns = [str(c).lower() for c in df.columns]

    # Recover player efficiency fields that legacy make_metrics may not carry.
    if PLAYER_FORM.exists() and PLAYER_FORM.stat().st_size > 0:
        pf = pd.read_csv(PLAYER_FORM)
        pf.columns = [str(c).lower() for c in pf.columns]
        key_options = [
            ["season", "week", "player_clean_key"],
            ["player_clean_key", "team"],
            ["player", "team"],
        ]
        keys = next((k for k in key_options if all(c in df.columns and c in pf.columns for c in k)), None)
        if keys:
            extras = [c for c in ["tgt_share", "target_share", "rush_share", "route_rate", "yprr", "ypt", "ypc", "ypa", "ypa_prior", "receptions_per_target"] if c in pf.columns and c not in keys]
            right = pf[keys + extras].drop_duplicates(keys, keep="last")
            df = df.merge(right, on=keys, how="left", suffixes=("", "_pf"))
            for c in extras:
                alt = f"{c}_pf"
                if alt in df.columns:
                    if c in df.columns:
                        df[c] = df[c].combine_first(df[alt])
                    else:
                        df[c] = df[alt]
                    df.drop(columns=[alt], inplace=True)
    return df


def price(season: int) -> pd.DataFrame:
    df = _load()
    if "season" in df.columns:
        s = pd.to_numeric(df["season"], errors="coerce")
        df = df.loc[s.eq(int(season))].copy()
    if df.empty:
        raise RuntimeError(f"metrics_ready has no rows for season={season}")

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        market_raw = str(row.get("market", "") or "").lower()
        market = MARKET_MAP.get(market_raw, market_raw)
        if market == "anytime_td":
            # Do not manufacture an independent TD projection without a valid
            # scoring-rate/RZ model.  A future TD model can plug in here.
            skipped += 1
            continue
        if market not in SIGMA:
            skipped += 1
            continue
        try:
            line = float(row.get("line"))
        except Exception:
            skipped += 1
            continue
        mu = _projection(row, market)
        if not np.isfinite(mu):
            skipped += 1
            continue
        sigma = SIGMA[market]
        p_over = _prob_over(mu, sigma, line)
        p_under = 1.0 - p_over if pd.notna(p_over) else np.nan
        over_odds = row.get("over_odds")
        under_odds = row.get("under_odds")
        mkt_over, mkt_under = _fair_market_prob(over_odds, under_odds)

        common = {
            "event_id": row.get("event_id"), "player": row.get("player"), "team": row.get("team"),
            "opponent": row.get("opponent"), "market": market, "source_market": market_raw,
            "vegas_line": line, "model_proj": mu, "model_sd": sigma,
            "vegas_over_odds": over_odds, "vegas_under_odds": under_odds,
            "season": int(season), "week": row.get("week"),
        }
        for side, prob, market_prob, vegas_odds in (
            ("OVER", p_over, mkt_over, over_odds),
            ("UNDER", p_under, mkt_under, under_odds),
        ):
            edge = prob - market_prob if pd.notna(prob) and pd.notna(market_prob) else np.nan
            rec = dict(common)
            rec.update({
                "side": side, "fair_prob": prob, "market_prob": market_prob,
                "vegas_odds": vegas_odds, "fair_odds": _fair_odds(prob),
                "edge_pct": edge, "edge_abs": abs(edge) if pd.notna(edge) else np.nan,
            })
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Independent pricing produced 0 priceable rows")
    print(f"[pricing_v2] priced_rows={len(out)} skipped_source_rows={skipped}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    out = price(season)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[pricing_v2] wrote {len(out)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
