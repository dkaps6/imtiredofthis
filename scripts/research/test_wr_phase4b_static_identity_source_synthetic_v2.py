#!/usr/bin/env python3
"""Synthetic mechanics for Phase 4B static-identity/PBP source amendment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import audit_wr_phase4b_prior_roster_gsis_alias_v2 as strict
from scripts.research import audit_wr_phase4b_identity_temporal_ablation_v3 as static
from scripts.utils.player_identity_v3 import player_name_key


def main() -> int:
    # Two aliases: Alpha has strictly-prior evidence; Beta first appears target week.
    alpha = player_name_key("Alpha Receiver")
    beta = player_name_key("Beta Receiver")
    brandon = player_name_key("Brandon Smith")
    full_index = {
        alpha: [(2023, 1, "AAA", "00-001")],
        beta: [(2023, 5, "BBB", "00-002")],
        brandon: [(2024, 15, "NYJ", "00-010"), (2024, 15, "NYJ", "00-011")],
    }
    base_index = {
        player_name_key("Alpha Receiver", strip_suffix=True): [(2023, 1, "AAA", "00-001")],
        player_name_key("Beta Receiver", strip_suffix=True): [(2023, 5, "BBB", "00-002")],
        player_name_key("Brandon Smith", strip_suffix=True): [(2024, 15, "NYJ", "00-010"), (2024, 15, "NYJ", "00-011")],
    }
    counts = pd.DataFrame([
        {"season": 2023, "week": 4, "team": "AAA", "receiver_id": "00-001", "pbp_targets": 7.0},
        {"season": 2023, "week": 5, "team": "BBB", "receiver_id": "00-002", "pbp_targets": 3.0},
    ])
    games = pd.DataFrame([
        {"season": 2023, "week": 4, "team": "AAA"},
        {"season": 2023, "week": 5, "team": "BBB"},
        {"season": 2024, "week": 15, "team": "NYJ"},
    ])
    frame = pd.DataFrame([
        {"season": 2023, "week": 4, "team": "AAA", "player_clean_key": "alpha", "player": "Alpha Receiver"},
        {"season": 2023, "week": 5, "team": "BBB", "player_clean_key": "beta", "player": "Beta Receiver"},
        {"season": 2024, "week": 15, "team": "NYJ", "player_clean_key": "brandonsmith", "player": "Brandon Smith"},
    ])

    old = strict.resolve_frame(frame, full_index, base_index, counts, games)
    new = static.resolve_static_frame(frame, full_index, base_index, counts, games)

    a_old = old.loc[old["player_clean_key"].eq("alpha")].iloc[0]
    a_new = new.loc[new["player_clean_key"].eq("alpha")].iloc[0]
    assert a_old.identity_status == "RESOLVED_GSIS"
    assert a_new.identity_status == "RESOLVED_GSIS"
    assert a_old.resolved_player_id == a_new.resolved_player_id == "00-001"
    assert float(a_old.pbp_targets) == float(a_new.pbp_targets) == 7.0

    b_old = old.loc[old["player_clean_key"].eq("beta")].iloc[0]
    b_new = new.loc[new["player_clean_key"].eq("beta")].iloc[0]
    assert b_old.identity_status == "UNRESOLVED_IDENTITY"
    assert b_new.identity_status == "RESOLVED_GSIS"
    assert b_new.resolved_player_id == "00-002"
    assert float(b_new.pbp_targets) == 3.0

    br = new.loc[new["player_clean_key"].eq("brandonsmith")].iloc[0]
    assert br.identity_status == "AMBIGUOUS_IDENTITY"
    assert br.resolved_player_id == ""
    assert np.isnan(br.pbp_targets)

    # Verified zero: resolved identity + present team-game + no target row.
    zero_frame = pd.DataFrame([
        {"season": 2023, "week": 4, "team": "AAA", "player_clean_key": "alpha", "player": "Alpha Receiver"}
    ])
    zero_counts = counts.loc[~counts["receiver_id"].eq("00-001")].copy()
    z = static.resolve_static_frame(zero_frame, full_index, base_index, zero_counts, games).iloc[0]
    assert z.identity_status == "RESOLVED_GSIS"
    assert float(z.pbp_targets) == 0.0

    print("Phase 4B static identity source synthetic: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
