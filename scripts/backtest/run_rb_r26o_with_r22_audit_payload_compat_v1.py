#!/usr/bin/env python3
"""R26P-authorized evidence-wiring wrapper for frozen R26O.

Preserves the already-audited R22 identity dtype compatibility seam and changes
only the return-object order exposed to the unchanged frozen R26O evaluator:
protected R22 `(adapted, trace, payload)` -> R26O-local `(adapted, payload, trace)`.
No football arrays, trace rows, payload values, gates, thresholds, seeds, or
iterations are modified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Importing this module activates the already-frozen dtype-only compatibility
# patch for R22's strict-prior identity merge seam.
import scripts.backtest.run_rb_r26o_with_r22_identity_dtype_compat_v1 as _dtype_compat  # noqa:F401
import scripts.backtest.evaluate_rb_r26o_2026_week1_receptions_shadow_integration_v1 as r26o

_ORIGINAL_APPLY = r26o.apply_rb_receiving_tail_production
EXPECTED_DISPOSITION = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS"


def _apply_with_audit_payload_exposed(*args, **kwargs):
    adapted, trace, payload = _ORIGINAL_APPLY(*args, **kwargs)

    if not isinstance(trace, pd.DataFrame):
        raise RuntimeError(f"R26O Gate15 repair expected R22 trace DataFrame, got {type(trace)!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"R26O Gate15 repair expected R22 audit dict, got {type(payload)!r}")
    if payload.get("disposition") != EXPECTED_DISPOSITION:
        raise RuntimeError(f"R26O Gate15 repair unexpected R22 disposition: {payload.get('disposition')}")
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise RuntimeError("R26O Gate15 repair R22 payload missing gates dict")
    if not isinstance(gates.get("mean_parity"), (bool, np.bool_)):
        raise RuntimeError("R26O Gate15 repair R22 payload missing boolean gates.mean_parity")
    if not isinstance(gates.get("receptions_exact"), (bool, np.bool_)):
        raise RuntimeError("R26O Gate15 repair R22 payload missing boolean gates.receptions_exact")
    try:
        max_mean_delta = float(payload.get("max_mean_delta"))
    except Exception as exc:
        raise RuntimeError("R26O Gate15 repair R22 payload missing numeric max_mean_delta") from exc
    if not np.isfinite(max_mean_delta):
        raise RuntimeError("R26O Gate15 repair R22 payload max_mean_delta is nonfinite")

    print(
        "R26O_GATE15_AUDIT_PAYLOAD_COMPAT_PASS "
        f"disposition={payload['disposition']} "
        f"mean_parity={bool(gates['mean_parity'])} "
        f"max_mean_delta={max_mean_delta:.17g} "
        f"receptions_exact={bool(gates['receptions_exact'])} "
        f"trace_rows={len(trace)}"
    )

    # The frozen evaluator assigned its second return value to `r22_audit`.
    # Expose the already-produced protected payload there; preserve trace as the
    # third return. No value inside adapted/trace/payload is modified.
    return adapted, payload, trace


r26o.apply_rb_receiving_tail_production = _apply_with_audit_payload_exposed


if __name__ == "__main__":
    raise SystemExit(r26o.main())
