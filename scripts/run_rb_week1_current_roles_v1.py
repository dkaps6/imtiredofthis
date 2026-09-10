#!/usr/bin/env python3
"""Run protected Week-1 RB P3 with reconciled current roles.

The wrapper redirects only the Ourlads-role read. Schedule, PlayerForm, model
artifacts, P3/R26/R22 parameters and all other inputs remain untouched.
"""
from __future__ import annotations

from pathlib import Path

from scripts.utils.current_roles_v1 import resolve_current_roles_path
import scripts.run_rb_week1_no_odds as rb


def main() -> int:
    active = resolve_current_roles_path(require_active=True)
    original_read = rb._read
    raw_roles = rb.DATA / "roles_ourlads.csv"

    def _read(path: Path, label: str):
        if Path(path) == raw_roles:
            print(f"[current_roles_v1] RB P3 roles={active}")
            return original_read(active, "reconciled active roles")
        return original_read(path, label)

    rb._read = _read
    return int(rb.main())


if __name__ == "__main__":
    raise SystemExit(main())
