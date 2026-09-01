"""M96A mechanics-only complete-truth wrapper.

The first green M96A run exposed that the M95F research trace has 53 missing
rushing-yard truth values for low-volume RB/FB rows, while the authoritative
M94C trace has complete truth for all 1,393 player-games.  This wrapper changes
only the truth source: M94C supplies actual rushing yards; M95F/M95I continue
to supply their frozen workload outputs.  Candidate formulas and routing gates
are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.backtest.evaluate_rb_m96a_opportunity_efficiency_attribution as m


def load_inputs_complete_truth(
    m94c_root: Path, m95f_root: Path, m95i_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m94 = m.prep(pd.read_csv(m.find_one(m94c_root, "m94c_2025_rb_trace.csv"), low_memory=False))
    f = m.prep(pd.read_csv(m.find_one(m95f_root, "m95f_2025_rb_trace.csv"), low_memory=False))
    i = m.prep(pd.read_csv(m.find_one(m95i_root, "m95i_2025_trace.csv"), low_memory=False))

    for name, x in [("m94c", m94), ("m95f", f), ("m95i", i)]:
        if x.duplicated(m.PLAYER_KEYS).any():
            raise RuntimeError(f"{name} has duplicate M96A player-game keys")

    f_keep = m.PLAYER_KEYS + [
        "player_clean_key", "actual_carries", "actual_rush_yards", "m94c_rush_att",
        "m95f_mix_mean", "m95f_p50", "m95f_p75", "m95f_p90", "m95f_p95",
        "cal_prob_20", "cal_prob_25", "role_is_workhorse", "rb_rb_share_avg1", "rb_rb_share_avg5",
    ]
    i_keep = m.PLAYER_KEYS + [
        "m95i_rush_att", "m95i_tail_uplift", "m95i_tail_eligible",
        "prior_top1_unavailable", "p20_joint", "p25_joint",
    ]
    m_keep = m.PLAYER_KEYS + [
        "candidate_rush_att", "candidate_rush_yards", "actual_rush_att", "actual_rush_yards"
    ]

    f2 = f[[c for c in f_keep if c in f.columns]].copy().rename(
        columns={"actual_rush_yards": "actual_rush_yards_m95f"}
    )
    i2 = i[[c for c in i_keep if c in i.columns]].copy()
    # M94C is the complete authoritative rushing-yard truth source for M96A.
    m2 = m94[[c for c in m_keep if c in m94.columns]].copy()

    fi = f2.merge(i2, on=m.PLAYER_KEYS, how="inner", validate="one_to_one")
    all3 = fi.merge(m2, on=m.PLAYER_KEYS, how="inner", validate="one_to_one")

    base_n = len(f2)
    source_rows = []
    for name, x in [("m95f", f2), ("m95i", i2), ("m94c", m2), ("m95f_m95i", fi), ("all_three", all3)]:
        source_rows.append({
            "source": name,
            "rows": int(len(x)),
            "coverage_vs_m95f": float(len(x) / base_n) if base_n else np.nan,
        })
    source = pd.DataFrame(source_rows)

    if len(fi) / max(base_n, 1) < 0.995 or len(all3) / max(base_n, 1) < 0.995:
        raise RuntimeError(f"M96A source join coverage below 99.5%: {source.to_dict(orient='records')}")

    carry_diff = np.abs(m.num(all3["m94c_rush_att"]) - m.num(all3["candidate_rush_att"]))
    actual_carry_diff = np.abs(m.num(all3["actual_carries"]) - m.num(all3["actual_rush_att"]))
    f_yards = m.num(all3["actual_rush_yards_m95f"])
    c_yards = m.num(all3["actual_rush_yards"])
    shared_yard_mask = f_yards.notna() & c_yards.notna()
    yard_diff = np.abs(f_yards.loc[shared_yard_mask] - c_yards.loc[shared_yard_mask])

    source["max_m94c_carry_parity_diff"] = float(carry_diff.max())
    source["max_actual_carry_parity_diff"] = float(actual_carry_diff.max())
    source["m94c_yard_truth_nonmissing"] = int(c_yards.notna().sum())
    source["m95f_yard_truth_nonmissing"] = int(f_yards.notna().sum())
    source["max_shared_actual_yard_parity_diff"] = float(yard_diff.max()) if len(yard_diff) else np.nan

    if c_yards.notna().sum() != len(all3):
        raise RuntimeError("M96A complete M94C rushing-yard truth gate failed")
    if carry_diff.max() > 1e-6 or actual_carry_diff.max() > 1e-6:
        raise RuntimeError("M96A frozen carry-trace parity gate failed")
    if len(yard_diff) and yard_diff.max() > 1e-6:
        raise RuntimeError("M96A shared rushing-yard truth parity gate failed")

    return all3, source


m.load_inputs = load_inputs_complete_truth

if __name__ == "__main__":
    raise SystemExit(m.main())
