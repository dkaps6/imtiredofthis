#!/usr/bin/env python3
"""Canonical market-model lineage entrypoint for the Week-1 V4 stack.

The former V2 implementation is preserved in `audit_market_model_lineage_v2_core.py`.
Existing Full Slate wiring now routes through the R22-aware V3 lineage audit.
Current-availability production runs first apply the frozen downstream coverage
seam so governance cardinalities match the certified eligible football universe.
"""
from scripts.operations.apply_current_availability_downstream_certification_seam_v1 import (
    main as apply_current_scope_certification_seam,
)

apply_current_scope_certification_seam()

from scripts.audit_market_model_lineage_v3 import main


if __name__ == "__main__":
    raise SystemExit(main())
