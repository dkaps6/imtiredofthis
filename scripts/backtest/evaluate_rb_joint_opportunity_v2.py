"""M94D v2 runner: fixes duplicate team-trace merge columns from Run #1."""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.backtest.evaluate_rb_joint_opportunity as m


def _apply_joint_fixed(
    rb: pd.DataFrame,
    team_trace: pd.DataFrame,
    scores: pd.DataFrame,
    config: dict[str, float | str],
) -> pd.DataFrame:
    out = rb.copy()
    optional = [
        "candidate_team_rush_att", "baseline_team_rush_att",
        "pred_lead_play_share", "pred_trail_play_share", "pred_off_plays",
    ]
    # _player_candidate already carries candidate_team_rush_att. Merge only
    # fields not already present so pandas does not suffix the canonical column.
    add_cols = [c for c in optional if c in team_trace.columns and c not in out.columns]
    if add_cols:
        out = out.merge(
            team_trace[m.TEAM_KEYS + add_cols].drop_duplicates(m.TEAM_KEYS),
            on=m.TEAM_KEYS, how="left", validate="many_to_one",
        )
    out = out.merge(
        scores[m.TEAM_KEYS + ["concentration_probability", "lead_key"]],
        on=m.TEAM_KEYS, how="left", validate="many_to_one",
    )
    out["concentration_probability"] = pd.to_numeric(
        out["concentration_probability"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["joint_is_lead"] = out["player_clean_key"].astype(str).eq(out["lead_key"].astype(str))

    rb_pool = out.groupby(m.TEAM_KEYS)["m94c_rush_att"].transform(lambda s: s.sum(min_count=1))
    base_pool = out.groupby(m.TEAM_KEYS)["base_rush_att"].transform(lambda s: s.sum(min_count=1))
    base_share = np.where(base_pool.gt(0), out["base_rush_att"] / base_pool, 0.0)
    out["m94c_rb_pool"] = rb_pool
    out["base_rb_pool_share"] = base_share

    team_rush = pd.to_numeric(out["candidate_team_rush_att"], errors="coerce")
    if "baseline_team_rush_att" in out.columns:
        base_team = pd.to_numeric(out["baseline_team_rush_att"], errors="coerce")
    else:
        base_team = pd.to_numeric(out["base_team"], errors="coerce")
    base_team = base_team.replace(0, np.nan)
    out["team_volume_ratio"] = (team_rush / base_team).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    mode = str(config["mode"])
    if mode == "gate":
        active = (
            out["concentration_probability"].ge(float(config["conc_gate"]))
            & team_rush.ge(float(config["team_rush_gate"]))
        )
        gamma = np.where(active, float(config["gamma"]), 1.0)
        joint_score = out["concentration_probability"] * m._sigmoid(
            (team_rush - float(config["team_rush_gate"])) / 3.0
        )
    elif mode == "continuous":
        vf = m._sigmoid((team_rush - float(config["center"])) / 3.0)
        joint_score = out["concentration_probability"] * vf
        gamma = 1.0 + float(config["strength"]) * joint_score
        active = gamma > 1.05
    else:
        raise ValueError(f"unknown M94D mode: {mode}")

    out["joint_score"] = np.asarray(joint_score, dtype=float)
    out["joint_active"] = np.asarray(active, dtype=bool)
    out["joint_gamma"] = np.asarray(gamma, dtype=float)
    raw = np.power(
        np.clip(np.asarray(base_share, dtype=float), 1e-12, None),
        out["joint_gamma"].to_numpy(dtype=float),
    )
    out["_raw_share"] = raw
    denom = out.groupby(m.TEAM_KEYS)["_raw_share"].transform("sum")
    out["candidate_rb_pool_share"] = np.where(
        denom.gt(0), out["_raw_share"] / denom, base_share
    )
    out["candidate_rush_att"] = out["candidate_rb_pool_share"] * rb_pool

    ypc = np.where(
        pd.to_numeric(out["base_rush_att"], errors="coerce").gt(0.5),
        pd.to_numeric(out["base_rush_yards"], errors="coerce")
        / pd.to_numeric(out["base_rush_att"], errors="coerce"),
        np.nan,
    )
    ypc = pd.Series(ypc, index=out.index).clip(lower=0.0, upper=12.0)
    out["candidate_rush_yards"] = np.where(
        ypc.notna(), out["candidate_rush_att"] * ypc, out["base_rush_yards"]
    )
    out["candidate_rush_rec_yards"] = (
        out["base_rush_rec_yards"] + out["candidate_rush_yards"] - out["base_rush_yards"]
    )
    return out.drop(columns=["_raw_share"])


m._apply_joint = _apply_joint_fixed

if __name__ == "__main__":
    raise SystemExit(m.main())
