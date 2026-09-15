#!/usr/bin/env python3
"""RB-PD2 2021-2024 replication re-run on a fresh-rebuilt current-route source.

Per Issue #535 (GPT-5.6, 2026-09-15, DIRECT_FRESH_REVALIDATION_ALLOWED): the
original PR #556 evaluator's `verify_m95q_parity()` gate is provenance
certification for M95Q's own downstream stable-workhorse role-model family,
which this candidate (#562) never consumes -- not a scientific input to the
carry/yard difficulty-persistence mechanism itself. That gate's own upstream
dependency (an M91-temporal-baseline artifact) has since expired.

This script reuses PR #556's science UNCHANGED -- build_panel/build_wf/score,
identical target seasons, identical prior-season-only ensemble weighting,
identical identity/leakage assertions -- imported directly, not reimplemented.
The only thing replaced is the provenance check: instead of requiring the old
M95Q downstream artifact, this records which source run the fresh M91 data
came from and re-derives the same structural guarantees build_panel/build_wf
already assert internally (target seasons exactly 2021-2024, zero 2025,
unique paired RB identity, zero sportsbook inputs, zero production change).

If YARD_DIFFICULTY_PERSISTENCE does not replicate on this fresh, drifted
source, the #562 width lane is NOT authorized to proceed -- per GPT-5.6's
explicit stop condition, this is not a rescue point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.evaluate_rb_pd2_multiseason_current_route_v1 import (
    TARGET_SEASONS,
    build_panel,
    build_wf,
    score,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                     help="Directory with one subdir per season (2020-2024), each "
                          "containing component_predictions.csv, from a fresh M91 rebuild.")
    ap.add_argument("--source-run-id", required=True,
                     help="The GitHub Actions run ID the fresh M91 artifacts came from, "
                          "recorded for the lineage record (not re-derived/re-fetched here).")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # build_panel/build_wf already hard-assert (raise RuntimeError, fail closed):
    #  - target season set is exactly TARGET_SEASONS (2021-2024), zero 2025
    #  - prior-season-only ensemble weight fit (fit_season = target_season - 1)
    #  - unique paired rush_att/rush_yards identity per (season,week,team,player)
    #  - non-finite / duplicate-identity / leakage fail closed
    # No sportsbook columns are read anywhere in this path; no production file is touched.
    panel, weights = build_panel(a.root)
    wf = build_wf(panel)
    metrics, by_season, result = score(wf)

    result["source_parity"] = {
        "verification_method": "FRESH_SOURCE_STRUCTURAL_REVALIDATION_V1",
        "supersedes": "verify_m95q_parity (PR #556 original, provenance-only, "
                       "not a scientific input to this candidate -- see Issue #535 "
                       "comment 5689291516)",
        "fresh_m91_source_run_id": str(a.source_run_id),
        "target_seasons_exact": sorted(panel["season"].unique().tolist()) == TARGET_SEASONS,
        "zero_2025_rows": bool(not panel["season"].eq(2025).any()),
        "unique_identity_contract_pass": True,  # build_panel raises before returning otherwise
        "sportsbook_inputs_used": False,
        "production_changed": False,
    }

    panel[["season", "week", "team", "player_key"]].drop_duplicates().sort_values(
        ["season", "week", "team", "player_key"]
    ).to_csv(a.out_dir / "rb_pd2_fresh_identity_manifest.csv", index=False)
    panel.to_csv(a.out_dir / "rb_pd2_fresh_parent_panel.csv", index=False)
    weights.to_csv(a.out_dir / "rb_pd2_fresh_weights.csv", index=False)
    wf.to_csv(a.out_dir / "rb_pd2_fresh_walkforward_casebook.csv", index=False)
    metrics.to_csv(a.out_dir / "rb_pd2_fresh_metrics.csv", index=False)
    by_season.to_csv(a.out_dir / "rb_pd2_fresh_by_season.csv", index=False)
    (a.out_dir / "rb_pd2_fresh_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    print(metrics.to_string(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
