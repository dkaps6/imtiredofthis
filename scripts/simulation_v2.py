"""Joint Monte Carlo simulation for player-prop outcomes.

Each simulated game shares team plays, pass rate, and efficiency shocks across
players. Player target/carry opportunities are allocated with multinomial draws,
so same-team outcomes compete for finite volume instead of being simulated as
independent normal distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from scripts.config import MC

MARKET_MAP = {
    "player_pass_yds": "pass_yards", "player_passing_yards": "pass_yards", "pass_yards": "pass_yards",
    "player_rush_yds": "rush_yards", "player_rushing_yards": "rush_yards", "rush_yards": "rush_yards",
    "player_reception_yds": "rec_yards", "player_rec_yds": "rec_yards", "player_receiving_yards": "rec_yards", "rec_yards": "rec_yards",
    "player_receptions": "receptions", "receptions": "receptions",
    "player_rush_att": "rush_att", "rush_att": "rush_att",
    "player_rush_reception_yds": "rush_rec_yards", "player_rush_rec_yds": "rush_rec_yards", "rush_rec_yards": "rush_rec_yards",
    "player_anytime_td": "anytime_td", "anytime_td": "anytime_td", "atd": "anytime_td",
}


@dataclass
class SimulationResult:
    values: Dict[tuple[str, str, str], np.ndarray]
    iterations: int


def _num(row: pd.Series, *names, default=np.nan) -> float:
    for name in names:
        if name in row.index:
            try:
                value = float(row.get(name))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return float(default)


def _clip_prob(value: float, default: float) -> float:
    if not np.isfinite(value):
        value = default
    return float(np.clip(value, 0.001, 0.999))


def _team_inputs(team_rows: pd.DataFrame) -> tuple[float, float]:
    row = team_rows.iloc[0]
    plays = _num(row, "plays_est", "pbp_plays_offense")
    if not np.isfinite(plays):
        pace = _num(row, "pace", "neutral_pace")
        plays = 1800.0 / pace if np.isfinite(pace) and pace > 0 else 64.0
    plays = float(np.clip(plays, 50.0, 80.0))
    proe = _num(row, "proe", "pass_rate_over_expected", default=0.0)
    pass_rate = 0.58 + (proe if np.isfinite(proe) else 0.0)
    team_wp = _num(row, "team_wp")
    if np.isfinite(team_wp):
        pass_rate += -0.02 if team_wp >= 0.60 else 0.02 if team_wp <= 0.40 else 0.0
    return plays, float(np.clip(pass_rate, 0.35, 0.75))


def _allocate_counts(rng: np.random.Generator, totals: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Allocate integer opportunities across modeled players plus a residual bucket."""
    n_iter = len(totals)
    n_players = len(shares)
    if n_players == 0:
        return np.empty((n_iter, 0), dtype=int)
    clean = np.nan_to_num(shares.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    total_share = clean.sum()
    if total_share > 0.95:
        clean *= 0.95 / total_share
    residual = max(0.0, 1.0 - clean.sum())
    probs = np.append(clean, residual)
    probs = probs / probs.sum()
    allocations = np.empty((n_iter, n_players), dtype=int)
    for i, total in enumerate(totals.astype(int)):
        allocations[i] = rng.multinomial(max(0, int(total)), probs)[:n_players]
    return allocations


def _player_key(row: pd.Series) -> str:
    value = row.get("player_clean_key")
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    return "".join(ch.lower() for ch in str(row.get("player", "")) if ch.isalnum())


def simulate(metrics: pd.DataFrame, *, iterations: int | None = None, seed: int | None = None) -> SimulationResult:
    iterations = int(iterations or MC.get("iterations", 25000))
    seed = int(MC.get("seed", 42) if seed is None else seed)
    rng = np.random.default_rng(seed)
    values: Dict[tuple[str, str, str], np.ndarray] = {}

    if metrics.empty:
        return SimulationResult(values, iterations)
    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(_player_key, axis=1)
    game_key = "event_id" if "event_id" in frame.columns and frame["event_id"].notna().any() else None
    if game_key is None:
        frame["_game_key"] = frame.apply(lambda r: "|".join(sorted([str(r.get("team", "")), str(r.get("opponent", ""))])), axis=1)
        game_key = "_game_key"

    # Props contain multiple lines/books for the same player. Collapse to one
    # player input row before simulation, then reuse the outcome array for every line.
    player_cols = [game_key, "team", "player_clean_key"]
    players = frame.sort_values(player_cols).drop_duplicates(player_cols, keep="last")

    for game, game_df in players.groupby(game_key, dropna=False):
        # One shared game pace shock affects both teams, creating cross-team correlation.
        game_pace_shock = rng.normal(0.0, 2.0, iterations)
        for team, team_df in game_df.groupby("team", dropna=False):
            if pd.isna(team) or not str(team).strip():
                continue
            plays_mean, pass_rate_mean = _team_inputs(team_df)
            plays = np.rint(np.clip(rng.normal(plays_mean, 3.5, iterations) + game_pace_shock, 45, 85)).astype(int)
            pass_rate = np.clip(rng.normal(pass_rate_mean, 0.035, iterations), 0.25, 0.82)
            pass_att = rng.binomial(plays, pass_rate)
            rush_att = plays - pass_att

            # Shared efficiency shocks create realistic QB/receiver co-movement.
            pass_eff_shock = np.clip(rng.normal(1.0, 0.09, iterations), 0.65, 1.35)
            rush_eff_shock = np.clip(rng.normal(1.0, 0.10, iterations), 0.60, 1.40)

            target_shares = np.array([_num(r, "target_share", "tgt_share", default=0.0) for _, r in team_df.iterrows()])
            rush_shares = np.array([_num(r, "rush_share", default=0.0) for _, r in team_df.iterrows()])
            targets = _allocate_counts(rng, pass_att, target_shares)
            carries = _allocate_counts(rng, rush_att, rush_shares)

            for j, (_, row) in enumerate(team_df.iterrows()):
                pkey = _player_key(row)
                if not pkey:
                    continue
                role = str(row.get("model_role", row.get("role", "")) or "").upper()
                position = str(row.get("position", "") or "").upper()
                catch_rate = _clip_prob(_num(row, "receptions_per_target", "catch_rate", default=0.64), 0.64)
                receptions = rng.binomial(targets[:, j], catch_rate)

                ypt = _num(row, "ypt")
                if not np.isfinite(ypt) or ypt <= 0:
                    ypt = 7.5
                # Receiving-yard expectation follows targets while shared pass
                # efficiency creates same-game correlation. Noise scales with opportunity.
                rec_mu = targets[:, j] * ypt * pass_eff_shock
                rec_sd = np.maximum(6.0, np.sqrt(np.maximum(targets[:, j], 1)) * ypt * 0.55)
                rec_yards = np.clip(rng.normal(rec_mu, rec_sd), 0.0, None)

                ypc = _num(row, "ypc")
                if not np.isfinite(ypc) or ypc <= 0:
                    ypc = 4.2
                rush_mu = carries[:, j] * ypc * rush_eff_shock
                rush_sd = np.maximum(3.0, np.sqrt(np.maximum(carries[:, j], 1)) * ypc * 0.65)
                rush_yards = np.clip(rng.normal(rush_mu, rush_sd), 0.0, None)

                values[(str(game), pkey, "receptions")] = receptions.astype(float)
                values[(str(game), pkey, "rec_yards")] = rec_yards
                values[(str(game), pkey, "rush_att")] = carries.astype(float)
                values[(str(game), pkey, "rush_yards")] = rush_yards
                values[(str(game), pkey, "rush_rec_yards")] = rush_yards + rec_yards

                if position == "QB" or role.startswith("QB"):
                    ypa = _num(row, "ypa", "ypa_prior")
                    if not np.isfinite(ypa) or ypa <= 0:
                        ypa = 7.0
                    qb_noise = np.clip(rng.normal(1.0, 0.07, iterations), 0.72, 1.28)
                    pass_yards = np.clip(pass_att * ypa * pass_eff_shock * qb_noise, 0.0, None)
                    values[(str(game), pkey, "pass_yards")] = pass_yards

                # ATD uses empirical offensive-TD rate with red-zone share as a
                # bounded role adjustment. This is independent of the sportsbook price.
                td_rate = _num(row, "offensive_td_rate")
                if np.isfinite(td_rate) and td_rate >= 0:
                    rz = _num(row, "rz_share", default=np.nan)
                    rz_mult = 1.0
                    if np.isfinite(rz):
                        rz_mult = float(np.clip(0.75 + rz, 0.75, 1.35))
                    wp = _num(row, "team_wp")
                    script_mult = 1.0 + (0.08 * (wp - 0.5) if np.isfinite(wp) else 0.0)
                    lam = max(0.0, td_rate * rz_mult * script_mult)
                    p_td = float(np.clip(1.0 - np.exp(-lam), 0.001, 0.95))
                    # Shared team scoring shock gives teammate TD outcomes a common environment.
                    team_scoring_shock = np.clip(rng.normal(1.0, 0.12, iterations), 0.65, 1.35)
                    p_iter = np.clip(1.0 - np.exp(-lam * team_scoring_shock), 0.001, 0.98)
                    values[(str(game), pkey, "anytime_td")] = rng.binomial(1, p_iter).astype(float)

    return SimulationResult(values, iterations)


def lookup(result: SimulationResult, row: pd.Series, market: str) -> np.ndarray | None:
    game = row.get("event_id")
    if pd.isna(game) or not str(game).strip():
        game = "|".join(sorted([str(row.get("team", "")), str(row.get("opponent", ""))]))
    pkey = _player_key(row)
    return result.values.get((str(game), pkey, MARKET_MAP.get(str(market).lower(), str(market).lower())))
