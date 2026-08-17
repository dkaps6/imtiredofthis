#!/usr/bin/env python3
"""Static repository audit for the production Full Slate path."""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

from scripts.artifact_contracts import CONTRACTS, ROOT

PRODUCTION_SCRIPTS = (
    "scripts/config.py",
    "scripts/runtime_context.py",
    "scripts/providers/ourlads_depth.py",
    "scripts/utils/make_team_week_map.py",
    "scripts/fetch_props_oddsapi.py",
    "scripts/build/build_opponent_map_from_props.py",
    "scripts/providers/sharpfootball_pull.py",
    "scripts/run_team_form_context.py",
    "scripts/build/build_weather_week.py",
    "scripts/build/build_injuries_weekly.py",
    "scripts/build/pbp_features.py",
    "scripts/player_form_v2.py",
    "scripts/run_player_form_v2.py",
    "scripts/enrich_player_scoring_v2.py",
    "scripts/metrics_v2.py",
    "scripts/metrics_enrichment_v2.py",
    "scripts/run_metrics_context.py",
    "scripts/metrics_ready.py",
    "scripts/pricing_v2.py",
    "scripts/simulation_v2.py",
    "scripts/run_pricing_v2.py",
    "scripts/validate_build_integrity.py",
    "scripts/artifact_contracts.py",
)
WORKFLOWS = (
    ".github/workflows/full-slate.yml",
    ".github/workflows/audit-only.yml",
    ".github/workflows/repo-ci.yml",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _syntax_errors() -> list[str]:
    errors = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", ".venv", "venv"} for part in path.parts):
            continue
        try:
            ast.parse(_read(path), filename=str(path))
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    return errors


def _workflow_errors() -> list[str]:
    errors = []
    if yaml is None:
        return ["PyYAML is not installed; cannot validate workflow YAML"]
    for rel in WORKFLOWS:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing workflow: {rel}")
            continue
        try:
            yaml.safe_load(_read(path))
        except Exception as exc:
            errors.append(f"invalid workflow {rel}: {exc}")
    return errors


def _presence_errors() -> list[str]:
    return [f"missing production script: {rel}" for rel in PRODUCTION_SCRIPTS if not (ROOT / rel).exists()]


def _contract_errors() -> list[str]:
    errors = []
    seen_paths: dict[Path, str] = {}
    for key, contract in CONTRACTS.items():
        path = contract.path.resolve(strict=False)
        if path in seen_paths:
            errors.append(f"duplicate artifact path: {key} and {seen_paths[path]} -> {path}")
        else:
            seen_paths[path] = key
        if contract.min_rows < 0:
            errors.append(f"contract {key} has invalid min_rows={contract.min_rows}")
        if len(set(contract.required_columns)) != len(contract.required_columns):
            errors.append(f"contract {key} contains duplicate required columns")
    return errors


def _stale_literal_errors() -> list[str]:
    errors = []
    patterns = (
        re.compile(r"==\s*2025"),
        re.compile(r"\[\s*2025\s*\]"),
        re.compile(r"season\s*=\s*2025"),
        re.compile(r"default\s*=\s*2025"),
        re.compile(r"WEEK\s*=\s*\d+"),
    )
    for rel in PRODUCTION_SCRIPTS:
        text = _read(ROOT / rel)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "PRIOR_SEASON" in line or "prior_season" in line:
                continue
            if any(p.search(line) for p in patterns):
                errors.append(f"stale runtime literal in {rel}:{line_no}: {line.strip()}")
    return errors


def _workflow_contract_errors() -> list[str]:
    path = ROOT / ".github/workflows/full-slate.yml"
    if not path.exists():
        return []
    text = _read(path)
    required_tokens = (
        "scripts/run_player_form_v2.py",
        "scripts/enrich_player_scoring_v2.py",
        "scripts/run_metrics_context.py",
        "scripts/validate_build_integrity.py",
        "scripts/metrics_ready.py",
        "scripts/run_pricing_v2.py",
        "scripts/utils/audit_repo.py --strict",
    )
    return [f"full-slate workflow does not invoke {token}" for token in required_tokens if token not in text]


def run_audit() -> list[str]:
    checks = {
        "presence": _presence_errors(),
        "python syntax": _syntax_errors(),
        "workflow yaml": _workflow_errors(),
        "artifact contracts": _contract_errors(),
        "runtime literals": _stale_literal_errors(),
        "workflow wiring": _workflow_contract_errors(),
    }
    failures = []
    print("\n======== Repo Audit — Production Full Slate ========")
    for label, issues in checks.items():
        if issues:
            print(f"[FAIL] {label}: {len(issues)}")
            for issue in issues:
                print("  -", issue)
                failures.append(f"{label}: {issue}")
        else:
            print(f"[OK] {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    failures = run_audit()
    if failures and args.strict:
        print(f"[AUDIT] strict mode failed with {len(failures)} issue(s)")
        return 1
    print(f"[AUDIT] {'passed' if not failures else f'completed with {len(failures)} warning issue(s)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
