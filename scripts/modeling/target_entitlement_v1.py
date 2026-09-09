"""Explicit team target-entitlement materialization.

This module does NOT introduce a new football model. It lifts the exact receiving
allocation transformation that simulation_v2 already performs out of the hidden
Monte Carlo boundary and makes it auditable:

1. take rule-adjusted player target shares;
2. apply the already-promoted M38 WR hierarchy while preserving WR mass;
3. conserve modeled player target mass at the existing 0.95 cap;
4. leave the remaining probability in the simulator's residual receiver bucket.

The output is intended to be projection-neutral relative to the pre-existing
simulator. Position-specific entitlement research can later redistribute mass
inside this conserved state without creating opportunity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.simulation_v2 import _sharpen_wr_target_shares

TARGET_MASS_CAP = 0.95
# Keep the explicit player sum one floating-point step below 0.95. This prevents
# simulation_v2's legacy defensive ``> 0.95`` guard from re-scaling an already-
# conserved explicit state solely because summation lands at 0.9500000000000001.
ALLOCATOR_SAFE_CAP = float(np.nextafter(TARGET_MASS_CAP, 0.0))


def _game_column(frame: pd.DataFrame) -> str:
    if "event_id" in frame.columns and frame["event_id"].notna().any():
        return "event_id"
    raise RuntimeError("explicit target entitlement requires event_id")


def materialize_target_entitlement(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics is None or metrics.empty:
        raise RuntimeError("cannot materialize target entitlement from empty metrics")
    out = metrics.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {"team", "player", "player_clean_key", "rules_tgt_share", "position"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"target entitlement missing columns: {sorted(missing)}")
    game_col = _game_column(out)

    # Projection-neutrality requires matching simulation_v2's exact player order
    # BEFORE M38 is applied. M38 uses stable share ranking, so tied target-share
    # priors (common for position-prior players) inherit the dataframe order as
    # the tie breaker. The canonical simulator first sorts by game/team/player
    # key; materializing in roster/source order would assign WR1/WR2/WR3/WR4
    # multipliers to different tied players and silently change projections.
    player_cols = [game_col, "team", "player_clean_key"]
    dup = out.duplicated(player_cols, keep=False)
    if dup.any():
        sample = out.loc[dup, player_cols + ["player", "position"]].head(20).to_dict("records")
        raise RuntimeError(f"explicit target entitlement requires one football row per player/game/team: {sample}")
    out = out.sort_values(player_cols).copy()

    out["entitlement_tgt_share"] = np.nan
    out["entitlement_raw_team_sum"] = np.nan
    out["entitlement_post_m38_team_sum"] = np.nan
    out["entitlement_team_scale"] = np.nan
    out["entitlement_residual_share"] = np.nan
    out["entitlement_version"] = "TEAM_TARGET_ENTITLEMENT_V1_PROJECTION_NEUTRAL"
    trace: list[dict] = []

    for (game, team), idx in out.groupby([game_col, "team"], dropna=False, sort=False).groups.items():
        group = out.loc[idx].copy()
        raw = pd.to_numeric(group["rules_tgt_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, TARGET_MASS_CAP)
        sharpened = _sharpen_wr_target_shares(group, raw)
        raw_sum = float(raw.sum())
        post_m38_sum = float(sharpened.sum())
        if not np.isfinite(post_m38_sum) or post_m38_sum < 0:
            raise RuntimeError(f"invalid post-M38 target mass game={game} team={team}: {post_m38_sum}")
        scale = ALLOCATOR_SAFE_CAP / post_m38_sum if post_m38_sum > TARGET_MASS_CAP else 1.0
        entitlement = sharpened * scale
        final_sum = float(entitlement.sum())
        if final_sum > TARGET_MASS_CAP:
            # Floating summation can still land one ulp high. Apply one final
            # projection-neutral guard, never a football/model retune.
            entitlement *= ALLOCATOR_SAFE_CAP / final_sum
            final_sum = float(entitlement.sum())
        residual = max(0.0, 1.0 - final_sum)
        if final_sum > TARGET_MASS_CAP + 1e-12:
            raise RuntimeError(f"explicit target entitlement exceeds cap game={game} team={team}: {final_sum}")
        if (entitlement < -1e-12).any() or not np.isfinite(entitlement).all():
            raise RuntimeError(f"invalid explicit player target entitlement game={game} team={team}")

        out.loc[idx, "entitlement_tgt_share"] = entitlement
        out.loc[idx, "entitlement_raw_team_sum"] = raw_sum
        out.loc[idx, "entitlement_post_m38_team_sum"] = post_m38_sum
        out.loc[idx, "entitlement_team_scale"] = scale
        out.loc[idx, "entitlement_residual_share"] = residual
        for j, row_idx in enumerate(idx):
            row = out.loc[row_idx]
            trace.append({
                "event_id": str(game),
                "team": str(team),
                "player": row.get("player"),
                "player_clean_key": row.get("player_clean_key"),
                "position": row.get("position"),
                "rules_tgt_share": float(raw[j]),
                "post_m38_tgt_share": float(sharpened[j]),
                "entitlement_tgt_share": float(entitlement[j]),
                "raw_team_sum": raw_sum,
                "post_m38_team_sum": post_m38_sum,
                "team_scale": float(scale),
                "modeled_player_sum": final_sum,
                "residual_share": residual,
                "entitlement_version": "TEAM_TARGET_ENTITLEMENT_V1_PROJECTION_NEUTRAL",
            })

    if out["entitlement_tgt_share"].isna().any():
        raise RuntimeError("explicit target entitlement left missing player rows")
    trace_df = pd.DataFrame(trace)
    team = trace_df.drop_duplicates(["event_id", "team"])
    if not np.allclose(
        team["modeled_player_sum"].to_numpy(float) + team["residual_share"].to_numpy(float),
        1.0, rtol=0, atol=1e-10,
    ):
        raise RuntimeError("explicit target entitlement does not conserve probability mass")
    return out, trace_df
