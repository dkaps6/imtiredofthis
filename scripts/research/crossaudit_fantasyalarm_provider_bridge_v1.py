"""Cross-audit the FantasyAlarm provider-ID bridge for anchor-set isolation.

Independent check of PR #665's source-quality audit, requested on issue #535.
Consumes that audit's own row-level output -- it re-acquires nothing and
re-scrapes nothing, so it cannot disturb the archive lane it is auditing.

The question: `_provider_bridge()` learns FantasyAlarm-ID -> GSIS-ID mappings
from rows filtered only on `schedule_match`. It does not require the row to be
PRE_KICKOFF. Identity is not a football feature, so this is not outcome
leakage -- but `stable_identity_ready` gates `source_quality_row_ready`, so if
any mapping rests on a row the audit itself quarantines as AFTER_KICKOFF, then
the headline pre-kickoff "ready" count is not reproducible under the quarantine
it claims to enforce.

This script reconstructs the pre-bridge identity state (recoverable exactly:
a row was unresolved iff its identity method is the bridge sentinel), rebuilds
the mapping under the audit's own rule and under a PRE_KICKOFF-only rule, and
reports the delta. A zero delta closes the concern cheaply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BRIDGE = "FANTASYALARM_STABLE_ID_BRIDGE"
SIDES = [
    ("wr", "wr_source_player_id", "wr_gsis_id", "wr_identity_method"),
    ("cb", "cb_source_player_id", "cb_gsis_id", "cb_identity_method"),
]


def _s(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame[col].astype("string").fillna("").str.strip()


def _bridge(anchor: pd.DataFrame, src: str, gsis: str) -> tuple[dict, set]:
    """Reproduce _provider_bridge() exactly: unique GSIS per provider ID wins."""
    pairs = anchor[[src, gsis]].drop_duplicates()
    grouped = pairs.groupby(src)[gsis].agg(
        lambda z: tuple(sorted({str(v) for v in z if str(v)}))
    )
    collisions = {str(k) for k, ids in grouped.items() if len(ids) != 1}
    mapping = {str(k): ids[0] for k, ids in grouped.items() if len(ids) == 1}
    return mapping, collisions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    x = pd.read_csv(args.rows, low_memory=False)
    timing = _s(x, "publication_timing_status")
    sched = x["schedule_match"].astype(str).str.lower().eq("true") \
        if "schedule_match" in x.columns else pd.Series(False, index=x.index)
    pre = timing.eq("PRE_KICKOFF")

    report: dict = {
        "contract": "WR_CB_PROVIDER_BRIDGE_CROSSAUDIT_V1",
        "source_rows": int(len(x)),
        "timing_breakdown": timing.value_counts().to_dict(),
        "schedule_match_rows": int(sched.sum()),
        "reacquired_archive": False,
        "model_candidates_scored": 0,
        "parameters_fit": 0,
        "sides": {},
    }

    for side, src_col, gsis_col, method_col in SIDES:
        src = _s(x, src_col)
        gsis = _s(x, gsis_col)
        method = _s(x, method_col)
        was_bridged = method.eq(BRIDGE)
        # Exactly recoverable: bridged rows were empty before the bridge ran.
        pre_gsis = gsis.mask(was_bridged, "")

        anchor_base = pd.DataFrame({src_col: src, gsis_col: pre_gsis})
        as_built = sched & src.str.len().gt(0) & pre_gsis.str.len().gt(0)
        strict = as_built & pre

        map_built, coll_built = _bridge(anchor_base.loc[as_built], src_col, gsis_col)
        map_strict, coll_strict = _bridge(anchor_base.loc[strict], src_col, gsis_col)

        # Which provider IDs owe their mapping to a quarantined anchor?
        only_after = sorted(set(map_built) - set(map_strict))
        changed = sorted(k for k in set(map_built) & set(map_strict)
                         if map_built[k] != map_strict[k])
        new_coll = sorted(coll_strict - coll_built)
        lost_coll = sorted(coll_built - coll_strict)

        # Rows that would lose their identity under the strict anchor rule.
        eligible_built = was_bridged
        rows_lost = int((eligible_built & ~src.isin(set(map_strict))).sum())
        # Bridge applied to rows that are NOT schedule-consistent: these inflate
        # stable_identity_ready without ever being gate-eligible.
        bridged_off_schedule = int((was_bridged & ~sched).sum())
        bridged_after_kick = int((was_bridged & ~pre).sum())

        report["sides"][side] = {
            "bridged_rows": int(was_bridged.sum()),
            "anchor_rows_as_built": int(as_built.sum()),
            "anchor_rows_pre_kickoff_only": int(strict.sum()),
            "anchor_rows_quarantined_but_used": int((as_built & ~pre).sum()),
            "mapped_ids_as_built": len(map_built),
            "mapped_ids_pre_kickoff_only": len(map_strict),
            "ids_resting_only_on_quarantined_anchors": only_after,
            "ids_whose_target_changes_under_strict_rule": changed,
            "collisions_as_built": len(coll_built),
            "collisions_pre_kickoff_only": len(coll_strict),
            "collisions_revealed_by_strict_rule": new_coll,
            "collisions_hidden_by_strict_rule": lost_coll,
            "bridged_rows_that_lose_identity_under_strict_rule": rows_lost,
            "bridged_rows_not_schedule_consistent": bridged_off_schedule,
            "bridged_rows_not_pre_kickoff": bridged_after_kick,
        }

    # Does the gate number itself move?
    if "source_quality_row_ready" in x.columns:
        ready = x["source_quality_row_ready"].astype(str).str.lower().eq("true")
        touched = pd.Series(False, index=x.index)
        for side, _s_col, _g_col, method_col in SIDES:
            touched |= _s(x, method_col).eq(BRIDGE)
        report["source_quality_row_ready_rows"] = int(ready.sum())
        report["ready_rows_depending_on_bridge"] = int((ready & touched).sum())

    if "wr_identity_method" in x.columns:
        report["wr_identity_method_mix"] = _s(x, "wr_identity_method").value_counts().to_dict()
        report["cb_identity_method_mix"] = _s(x, "cb_identity_method").value_counts().to_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
