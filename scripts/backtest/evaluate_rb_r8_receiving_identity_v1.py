"""Production compatibility shim for the frozen R8 identity runtime used by R22.

This is intentionally not the historical R8 backtest implementation. The Week-1
production branch keeps the exact four runtime objects required by the certified
R22 scorer in `scripts.modeling.rb_receiving_identity_runtime_v1`, while preserving
the historical import path expected by the already-certified R22 adapter.
"""
from scripts.modeling.rb_receiving_identity_runtime_v1 import (
    EPS,
    FEATURES,
    attach_identity as _attach_identity,
    identity_atlas as _identity_atlas,
)

__all__ = ["EPS", "FEATURES", "_attach_identity", "_identity_atlas"]
