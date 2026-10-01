"""Semantic-RNG QB C2 apply function for the isolation production candidate.

Selector logic is intentionally outside this module and remains the frozen
production authority. This module only generates the mean-neutral C2 candidate
passing-yard arrays from already-preserved team states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.simulation_c2_qb_candidate import (
    C2_RESIDUAL_CATCH_RATE,
    C2_RESIDUAL_YPT,
    C2_YPR_MAX,
    C2_YPR_MIN,
    PASS_CATCHER_POSITIONS,
    StateSimulationResult,
    _primary_qb_row,
)
from scripts.simulation_rng_isolation_v1 import (
    _hierarchical_targets,
    _prepare,
    _rng,
)
from scripts.simulation_v2 import _clip_prob, _num, _player_key


def apply_c2(
    base: StateSimulationResult,
    metrics: pd.DataFrame,
    *,
    anchor_map: dict[tuple[str, str], float],
    seed: int = 5601,
) -> StateSimulationResult:
    values = {k: np.asarray(v, dtype=float).copy() for k, v in base.values.items()}
    frame, plan = _prepare(metrics)
    if frame.empty:
        return StateSimulationResult(values, base.iterations, base.team_states)

    game_key = (
        "event_id"
        if "event_id" in frame.columns and frame["event_id"].notna().any()
        else None
    )
    if game_key is None:
        frame["_game_key"] = frame.apply(
            lambda r: "|".join(
                sorted([str(r.get("team", "")), str(r.get("opponent", ""))])
            ),
            axis=1,
        )
        game_key = "_game_key"

    players = (
        frame.sort_values([game_key, "team", "player_clean_key"], kind="mergesort")
        .drop_duplicates([game_key, "team", "player_clean_key"], keep="last")
    )

    for game, game_df in players.groupby(game_key, dropna=False, sort=True):
        gs = str(game)
        for team, team_df in game_df.groupby("team", dropna=False, sort=True):
            if pd.isna(team) or not str(team).strip():
                continue
            ts = str(team)
            team_df = team_df.sort_values(
                "player_clean_key", kind="mergesort"
            ).reset_index(drop=True)

            pass_att = np.asarray(
                base.team_states[(gs, ts, "pass_att")], dtype=int
            )
            pass_eff = np.asarray(
                base.team_states[(gs, ts, "pass_eff_shock")], dtype=float
            )

            positions = (
                team_df.get("position", pd.Series("", index=team_df.index))
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )
            catcher_mask = positions.isin(PASS_CATCHER_POSITIONS).to_numpy()
            catcher_df = team_df.loc[catcher_mask].reset_index(drop=True)
            if catcher_df.empty:
                raise RuntimeError(
                    f"RNG-isolated C2 found no pass catchers game={gs} team={ts}"
                )

            targets = _hierarchical_targets(
                game=gs,
                team=ts,
                team_df=catcher_df,
                pass_att=pass_att,
                plan=plan,
                base_seed=int(seed),
            )
            residual_targets = np.maximum(0, pass_att - targets.sum(axis=1))

            receiving_arrays = []
            for j, (_, row) in enumerate(catcher_df.iterrows()):
                pkey = _player_key(row)
                if not pkey:
                    continue
                catch = _clip_prob(
                    _num(
                        row,
                        "rules_catch_rate",
                        "bayes_receptions_per_target",
                        "receptions_per_target",
                        "catch_rate",
                        default=C2_RESIDUAL_CATCH_RATE,
                    ),
                    C2_RESIDUAL_CATCH_RATE,
                )
                recs = _rng(seed, "C2_CATCH", gs, ts, pkey).binomial(
                    targets[:, j], catch
                )

                ypt = _num(row, "rules_ypt", "bayes_ypt", "ypt")
                ypt = (
                    C2_RESIDUAL_YPT
                    if not np.isfinite(ypt) or ypt <= 0
                    else float(ypt)
                )
                ypr = float(np.clip(ypt / catch, C2_YPR_MIN, C2_YPR_MAX))
                vol = float(
                    np.clip(
                        _num(row, "rules_volatility_mult", default=1.0),
                        0.75,
                        1.50,
                    )
                )
                mu = recs.astype(float) * ypr * pass_eff
                sd = (
                    np.maximum(
                        3.0,
                        np.sqrt(np.maximum(recs, 1)) * ypr * 0.55,
                    )
                    * vol
                )
                yards = np.clip(
                    _rng(seed, "C2_REC_YARDS", gs, ts, pkey).normal(mu, sd),
                    0.0,
                    None,
                )
                receiving_arrays.append(np.where(recs > 0, yards, 0.0))

            residual_recs = _rng(
                seed, "C2_RESIDUAL_CATCH", gs, ts
            ).binomial(residual_targets, C2_RESIDUAL_CATCH_RATE)
            residual_ypr = C2_RESIDUAL_YPT / C2_RESIDUAL_CATCH_RATE
            residual_mu = residual_recs.astype(float) * residual_ypr * pass_eff
            residual_sd = np.maximum(
                3.0,
                np.sqrt(np.maximum(residual_recs, 1)) * residual_ypr * 0.55,
            )
            residual_yards = np.where(
                residual_recs > 0,
                np.clip(
                    _rng(seed, "C2_RESIDUAL_YARDS", gs, ts).normal(
                        residual_mu, residual_sd
                    ),
                    0.0,
                    None,
                ),
                0.0,
            )

            raw_total = (
                np.sum(np.vstack(receiving_arrays), axis=0)
                if receiving_arrays
                else np.zeros(base.iterations)
            ) + residual_yards
            raw_mean = float(np.mean(raw_total)) if len(raw_total) else np.nan
            anchor = float(anchor_map.get((gs, ts), np.nan))
            if not np.isfinite(anchor) or anchor <= 0:
                raise RuntimeError(
                    f"RNG-isolated C2 invalid anchor game={gs} team={ts} anchor={anchor}"
                )
            if not np.isfinite(raw_mean) or raw_mean <= 0:
                raise RuntimeError(
                    f"RNG-isolated C2 invalid raw mean game={gs} team={ts} mean={raw_mean}"
                )

            qb_row = _primary_qb_row(team_df)
            if qb_row is None:
                raise RuntimeError(
                    f"RNG-isolated C2 primary QB missing game={gs} team={ts}"
                )
            qb_key = _player_key(qb_row)
            if not qb_key:
                raise RuntimeError(
                    f"RNG-isolated C2 blank primary QB key game={gs} team={ts}"
                )

            qb_yards = raw_total * (anchor / raw_mean)
            key = (gs, qb_key, "pass_yards")
            canonical = np.asarray(base.values.get(key), dtype=float)
            if canonical.shape != qb_yards.shape or not np.isfinite(qb_yards).all():
                raise RuntimeError(
                    f"RNG-isolated C2 invalid primary QB array key={key}"
                )
            mean_gap = float(qb_yards.mean() - canonical.mean())
            if abs(mean_gap) > 1e-10:
                raise RuntimeError(
                    f"RNG-isolated C2 mean neutrality failed key={key} gap={mean_gap}"
                )
            values[key] = qb_yards

    return StateSimulationResult(values, base.iterations, base.team_states)
