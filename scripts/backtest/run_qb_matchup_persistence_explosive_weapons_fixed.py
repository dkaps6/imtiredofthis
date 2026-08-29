#!/usr/bin/env python3
"""Authoritative M72 entrypoint with stable-primary QB history integrity.

The frozen M72 hypotheses, families, models and gates are unchanged. This wrapper
prevents backup/relief passers from becoming independent historical defense-matchup
observations. Historical QB baselines and defense residual persistence use only the
primary passer in a team-game when that passer owns at least 80% of the team's
official pass attempts, matching the stable-QB research population used by the
canonical frontier.
"""
from __future__ import annotations

import pandas as pd

import scripts.backtest.audit_qb_matchup_persistence_explosive_weapons as m

_orig_hist = m.build_historical_qb_residual_games
_orig_defense = m.defense_residual_features


def stable_primary_passer_games(passer_games: pd.DataFrame) -> pd.DataFrame:
    q = passer_games.copy()
    keys = ["season", "week", "game_id", "team"]
    q["_team_attempts"] = q.groupby(keys)["attempts"].transform("sum")
    q["_primary_attempts"] = q.groupby(keys)["attempts"].transform("max")
    q["_attempt_share"] = m.num(q.attempts) / m.num(q._team_attempts)
    keep = m.num(q.attempts).eq(m.num(q._primary_attempts)) & m.num(q._attempt_share).ge(0.80)
    return q[keep].drop(columns=["_team_attempts", "_primary_attempts", "_attempt_share"]).copy()


def build_historical_qb_residual_games_stable(passer_games):
    return _orig_hist(stable_primary_passer_games(passer_games))


def defense_residual_features_stable(hist_qb, passer_games, target_pid, season, week, defense):
    return _orig_defense(
        hist_qb,
        stable_primary_passer_games(passer_games),
        target_pid,
        season,
        week,
        defense,
    )


m.build_historical_qb_residual_games = build_historical_qb_residual_games_stable
m.defense_residual_features = defense_residual_features_stable


if __name__ == "__main__":
    raise SystemExit(m.main())
