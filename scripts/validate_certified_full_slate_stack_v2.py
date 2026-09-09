#!/usr/bin/env python3
"""Canonical certified Full Slate stack validator for the Week-1 V4 stack.

The former V2 implementation is preserved in
`validate_certified_full_slate_stack_v2_core.py`. Existing Full Slate workflow
wiring now routes through the R22-aware V3 validator.
"""
from scripts.validate_certified_full_slate_stack_v3 import main


if __name__ == "__main__":
    raise SystemExit(main())
