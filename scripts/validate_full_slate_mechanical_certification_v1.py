#!/usr/bin/env python3
"""Run the post-pricing audit and narrow its durable certification semantics.

The underlying validator proves exact offer reconciliation, data/identity integrity,
component consumption, promoted QB/RB routing, quarantine preservation, and
provider limitations.  Those are execution/mechanical guarantees.  They do not
prove that every market currently consumes the best specialist research model or
that all football-model science gates have passed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.validate_full_slate_post_pricing_v1 import audit

OUT = Path("outputs/paid_full_slate_replay_result.json")
AUDIT = Path("data/full_slate_post_pricing_audit.csv")


def main() -> int:
    result = audit()
    blockers = int(result.get("certification_blockers", 0))
    result["legacy_post_pricing_disposition"] = result.get("disposition")
    result["certification_scope"] = "MECHANICAL_EXECUTION_DATA_IDENTITY_PRICING_AND_COMPONENT_ROUTING_ONLY"
    result["does_not_certify_all_market_science"] = True
    result["disposition"] = (
        "PAID_FULL_SLATE_REPLAY_MECHANICAL_EXECUTION_CERTIFIED"
        if blockers == 0
        else "PAID_FULL_SLATE_REPLAY_MECHANICAL_EXECUTION_NOT_CERTIFIED"
    )
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if AUDIT.exists() and AUDIT.stat().st_size > 0:
        frame = pd.read_csv(AUDIT)
        frame = pd.concat(
            [frame, pd.DataFrame([{
                "check": "certification_scope",
                "status": "PASS" if blockers == 0 else "BLOCKED",
                "detail": result["certification_scope"],
            }])],
            ignore_index=True,
        )
        frame.to_csv(AUDIT, index=False)

    print("[full_slate_mechanical_certification] " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
