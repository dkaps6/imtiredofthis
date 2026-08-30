#!/usr/bin/env python3
"""Deprecated legacy orchestration entry point.

The canonical NFL production pipeline is `.github/workflows/full-slate.yml`.
This module is intentionally non-runnable because the previous engine invoked a
separate 2025-era builder/pricing lineage that no longer matches the promoted
2026 production architecture.

Keeping a hard failure here is safer than silently producing projections from a
second, stale model path. Historical research/backtest scripts are unaffected.
"""
from __future__ import annotations


DEPRECATION_MESSAGE = (
    "engine/engine.py is retired. The only canonical production pipeline is "
    ".github/workflows/full-slate.yml. Do not use the legacy engine for 2026 projections."
)


def run_pipeline(*args, **kwargs):
    """Fail closed for callers that still import the legacy production engine."""
    raise RuntimeError(DEPRECATION_MESSAGE)


def main() -> int:
    print(DEPRECATION_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
