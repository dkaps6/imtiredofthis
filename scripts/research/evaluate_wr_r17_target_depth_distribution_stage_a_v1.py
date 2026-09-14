#!/usr/bin/env python3
"""Compatibility entrypoint for frozen WR-R17 Stage A.

The real implementation lives in v1b. That suffix records a pre-result
mechanical correction: preserve every targeted-pass event, including repeated
same-depth targets within a game. No frozen science, cohort, feature, or gate
changed, and no real-data WR-R17 result existed before this correction.
"""
from scripts.research.evaluate_wr_r17_target_depth_distribution_stage_a_v1b import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
