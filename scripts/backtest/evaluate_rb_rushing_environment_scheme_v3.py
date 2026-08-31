"""M95D authoritative control-alignment shim.

This keeps the pre-specified M95D scheme/personnel features, interactions,
models, temporal splits and advancement gates unchanged. It corrects the
research control so `role_plus_m95c_environment` is the *exact* M95C
`role_plus_environment` feature family rather than a broader environment set.

The correction is required for an apples-to-apples incremental test against the
validated M95C result and is not 2025-driven feature selection.
"""
import numpy as np
import pandas as pd

import scripts.backtest.evaluate_rb_rushing_environment_scheme_v2 as v2
import scripts.backtest.evaluate_rb_rushing_environment_scheme as m

EXACT_M95C_ENVIRONMENT = [
    "pfr_ybc_per_att_avg3", "pfr_ybc_per_att_avg5",
    "ngs_expected_yards_per_att_avg3", "ngs_expected_yards_per_att_avg5",
    "ngs_percent_attempts_gte_eight_defenders_avg3", "ngs_percent_attempts_gte_eight_defenders_avg5",
    "ngs_avg_time_to_los_avg3", "ngs_avg_time_to_los_avg5",
    "team_pfr_ybc_per_att_avg3", "team_pfr_ybc_per_att_avg5",
    "team_pbp_stuff_rate_avg3", "team_pbp_stuff_rate_avg5",
    "rel_ybc_vs_team_avg3", "rel_ybc_vs_team_avg5",
]

# Exact control contract from M95C.
m.M95C_ENV_CANDIDATES = EXACT_M95C_ENVIRONMENT

_base_read_trace = m.read_trace

def read_trace_exact(root):
    x = _base_read_trace(root)
    for n in (3, 5):
        p = f"pfr_ybc_per_att_avg{n}"
        t = f"team_pfr_ybc_per_att_avg{n}"
        x[f"rel_ybc_vs_team_avg{n}"] = (
            m.num(x[p]) - m.num(x[t]) if p in x.columns and t in x.columns
            else np.nan
        )
    return x

m.read_trace = read_trace_exact

# Re-apply the two implementation-only fixes from v2 explicitly because v2
# imports/patches the base module when loaded, while this file is the executable.
m.gain_table = v2.gain_table_fixed
m.read_pfr_def = v2.read_pfr_def_fixed

if __name__ == "__main__":
    raise SystemExit(m.main())
