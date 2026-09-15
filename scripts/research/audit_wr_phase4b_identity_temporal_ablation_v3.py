#!/usr/bin/env python3
"""Source-only Phase 4B identity temporal ablation.

Compares the existing strict-prior roster-alias -> GSIS resolver against an
identity-only resolver allowed to use all audited 2022-2024 weekly roster alias
evidence.  The PBP target-count rule is unchanged.  No receiving-yard fields
are loaded and no attribution outcome is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import audit_wr_phase4b_prior_roster_gsis_alias_v2 as base
from scripts.utils.player_identity_v3 import clean_player_id, player_name_key


def _all_ids(index, key: str, team: str | None) -> set[str]:
    vals = index.get(str(key), [])
    ids: set[str] = set()
    for _season, _week, tm, pid in vals:
        if team is not None and tm != str(team):
            continue
        pid = clean_player_id(pid)
        if pid:
            ids.add(pid)
    return ids


def resolve_static_gsis(player: str, team: str, full_index, base_index) -> dict:
    full = player_name_key(player)
    base_key = player_name_key(player, strip_suffix=True)
    criteria = [
        ("team_exact_full_alias", full_index, full, str(team)),
        ("team_suffix_insensitive_alias", base_index, base_key, str(team)),
        ("global_unique_full_alias", full_index, full, None),
        ("global_unique_suffix_alias", base_index, base_key, None),
    ]
    for method, index, key, team_filter in criteria:
        if not key:
            continue
        ids = _all_ids(index, key, team_filter)
        if len(ids) == 1:
            return {
                "identity_status": "RESOLVED_GSIS",
                "identity_method": method,
                "resolved_player_id": next(iter(ids)),
                "identity_candidate_count": 1,
                "identity_full_key": full,
                "identity_base_key": base_key,
            }
        if len(ids) > 1:
            return {
                "identity_status": "AMBIGUOUS_IDENTITY",
                "identity_method": f"ambiguous_{method}",
                "resolved_player_id": "",
                "identity_candidate_count": int(len(ids)),
                "identity_full_key": full,
                "identity_base_key": base_key,
            }
    return {
        "identity_status": "UNRESOLVED_IDENTITY",
        "identity_method": "no_roster_alias",
        "resolved_player_id": "",
        "identity_candidate_count": 0,
        "identity_full_key": full,
        "identity_base_key": base_key,
    }


def resolve_static_frame(frame, full_index, base_index, target_counts, pbp_team_games):
    rows = []
    for r in frame.itertuples(index=False):
        rec = {c: getattr(r, c) for c in frame.columns}
        rec.update(resolve_static_gsis(str(rec["player"]), str(rec["team"]), full_index, base_index))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out.merge(
        pbp_team_games.assign(pbp_team_game_present=True),
        on=base.TG, how="left", validate="many_to_one",
    )
    out["pbp_team_game_present"] = out["pbp_team_game_present"].fillna(False).astype(bool)
    c = target_counts.rename(columns={"receiver_id": "resolved_player_id"})
    out = out.merge(c, on=base.TG + ["resolved_player_id"], how="left", validate="many_to_one")
    resolved = out["identity_status"].eq("RESOLVED_GSIS")
    out.loc[resolved & out["pbp_team_game_present"] & out["pbp_targets"].isna(), "pbp_targets"] = 0.0
    out.loc[~resolved, "pbp_targets"] = np.nan
    return out


def recovery_table(strict: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    keys = base.IDENT
    a = strict[keys + ["player", "identity_status", "identity_method", "resolved_player_id", "pbp_targets"]].copy()
    b = static[keys + ["identity_status", "identity_method", "resolved_player_id", "pbp_targets"]].copy()
    a = a.rename(columns={
        "identity_status": "strict_status", "identity_method": "strict_method",
        "resolved_player_id": "strict_player_id", "pbp_targets": "strict_pbp_targets",
    })
    b = b.rename(columns={
        "identity_status": "static_status", "identity_method": "static_method",
        "resolved_player_id": "static_player_id", "pbp_targets": "static_pbp_targets",
    })
    out = a.merge(b, on=keys, how="inner", validate="one_to_one")
    out["recovered_by_static_identity"] = (
        ~out["strict_status"].eq("RESOLVED_GSIS") & out["static_status"].eq("RESOLVED_GSIS")
    )
    return out


def parity_summary(cand: pd.DataFrame) -> dict:
    x = cand.loc[cand["identity_status"].eq("RESOLVED_GSIS") & cand["pbp_targets"].notna()].copy()
    x["target_delta"] = pd.to_numeric(x["pbp_targets"], errors="raise") - pd.to_numeric(x["actual_targets"], errors="raise")
    parity = x["target_delta"].abs().le(1e-12)
    return {
        "rows": int(len(cand)),
        "resolved_rows": int(len(x)),
        "exact_target_parity_rows": int(parity.sum()),
        "target_parity_fail_rows": int((~parity).sum()),
        "max_abs_target_delta": float(x["target_delta"].abs().max()) if len(x) else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    pred, feat, canonical = base._load_structures(args.predictions, args.features)
    aliases = base.load_roster_aliases([2022, 2023, 2024])
    full_index = base._index_aliases(aliases, "full_key")
    base_index = base._index_aliases(aliases, "base_key")
    counts, team_games = base.load_pbp_targets([2023, 2024])

    l4_input = feat[base.IDENT + ["player"]].copy()
    l23_input = canonical[base.IDENT + ["player"]].copy()
    cand_input = pred[base.IDENT + ["player", "actual_targets", "wr_rank"]].copy()

    strict_l4 = base.resolve_frame(l4_input, full_index, base_index, counts, team_games)
    strict_l23 = base.resolve_frame(l23_input, full_index, base_index, counts, team_games)
    strict_cand = base.resolve_frame(cand_input, full_index, base_index, counts, team_games)

    static_l4 = resolve_static_frame(l4_input, full_index, base_index, counts, team_games)
    static_l23 = resolve_static_frame(l23_input, full_index, base_index, counts, team_games)
    static_cand = resolve_static_frame(cand_input, full_index, base_index, counts, team_games)

    rec_l4 = recovery_table(strict_l4, static_l4)
    rec_l23 = recovery_table(strict_l23, static_l23)
    rec_cand = recovery_table(strict_cand, static_cand)

    result = {
        "specification": "WR_PHASE4B_IDENTITY_TEMPORAL_ABLATION_V3",
        "identity_only_temporal_question": True,
        "identity_evidence_seasons": [2022, 2023, 2024],
        "static_identity_rule": "all_audited_weekly_roster_alias_evidence_allowed_for_alias_to_gsis_only",
        "predictive_feature_temporal_rules_changed": False,
        "target_source": "nflverse_pbp_receiver_player_id",
        "pbp_target_rule": "REG week1-18 AND pass_attempt==1 AND two_point_attempt!=1 AND no_play!=1 AND receiver_player_id nonnull",
        "fuzzy_matching": False,
        "strict_prior_layer4": base._summary(strict_l4),
        "static_identity_layer4": base._summary(static_l4),
        "strict_prior_layer23": base._summary(strict_l23),
        "static_identity_layer23": base._summary(static_l23),
        "layer4_recovered_rows": int(rec_l4["recovered_by_static_identity"].sum()),
        "layer23_recovered_rows": int(rec_l23["recovered_by_static_identity"].sum()),
        "candidate_recovered_rows": int(rec_cand["recovered_by_static_identity"].sum()),
        "strict_prior_candidate_parity": parity_summary(strict_cand),
        "static_identity_candidate_parity": parity_summary(static_cand),
        "receiving_yard_fields_loaded": False,
        "attribution_outcomes_run": False,
        "sportsbook_inputs": 0,
        "zero_imputation_performed": False,
        "challenger_model_authorized": False,
        "production_change": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec_l4.to_csv(args.out_dir / "layer4_temporal_ablation.csv", index=False)
    rec_l23.to_csv(args.out_dir / "layer23_temporal_ablation.csv", index=False)
    rec_cand.to_csv(args.out_dir / "candidate_temporal_ablation.csv", index=False)
    static_cand.loc[~static_cand["identity_status"].eq("RESOLVED_GSIS")].to_csv(
        args.out_dir / "static_identity_candidate_unresolved.csv", index=False
    )
    (args.out_dir / "identity_temporal_ablation_v3.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
