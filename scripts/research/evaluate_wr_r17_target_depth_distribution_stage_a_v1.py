#!/usr/bin/env python3
"""Compatibility entrypoint for frozen WR-R17 Stage A.

Implementation lineage:
- v1b preserved every target event before any valid scientific result;
- v1c adds a strictly-prior weekly-roster GSIS identity bridge after run
  34897984434 was data-blocked at 0% coverage because PBP receiver names were
  abbreviated. Neither repair changes the frozen football science or gates.
"""
from scripts.research.evaluate_wr_r17_target_depth_distribution_stage_a_v1c import main


if __name__ == "__main__":
    raise SystemExit(main())
