#!/usr/bin/env python3
"""Run the protected PlayerForm loader against an explicit current-role artifact.

Only the role input path changes. PlayerForm identity/history/blend mechanics are
unchanged. Set ACTIVE_ROLES_CSV in the candidate Full Slate workflow.
"""
from __future__ import annotations

from scripts.utils.current_roles_v1 import resolve_current_roles_path
import scripts.run_player_form_v2_loader as loader


def main() -> int:
    path = resolve_current_roles_path(require_active=True)
    loader.runner.pf.ROLES = path
    print(f"[current_roles_v1] PlayerForm roles={path}")
    return int(loader.main())


if __name__ == "__main__":
    raise SystemExit(main())
