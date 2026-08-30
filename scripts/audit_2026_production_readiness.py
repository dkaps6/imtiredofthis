#!/usr/bin/env python3
"""Static 2026 production-readiness audit for the canonical Full Slate path.

This is intentionally separate from scripts/utils/audit_repo.py while the 2026
overhaul is in progress. It is allowed to report known blockers on main without
breaking ordinary Repo CI. Once the listed P0/P1 issues are closed, this audit
should be promoted into the required Full Slate/CI gate.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FULL_SLATE = ROOT / ".github/workflows/full-slate.yml"
ENSEMBLE_WEIGHTS = ROOT / "data/model_ensemble_weights.csv"
QB_SYNTHESIS = ROOT / "model/qb_pass_synthesis_v1.json"

# Files that can directly execute or materially feed the canonical Full Slate
# path. Backtest-only files are intentionally excluded from stale-runtime scans.
PRODUCTION_RUNTIME_FILES = (
    "scripts/config.py",
    "scripts/runtime_context.py",
    "scripts/providers/ourlads_depth.py",
    "scripts/utils/build_team_week_map_v2.py",
    "scripts/utils/make_team_week_map.py",
    "scripts/fetch_props_oddsapi.py",
    "scripts/providers/sharpfootball_pull.py",
    "scripts/team_form_prior_bridge.py",
    "scripts/run_team_form_context.py",
    "scripts/make_team_form.py",
    "scripts/run_qb_promoted_context.py",
    "scripts/build/build_weather_week_v2.py",
    "scripts/build/build_weather_week.py",
    "scripts/build/build_injuries_weekly.py",
    "scripts/run_coverage_v2.py",
    "scripts/build/build_coverage_v2.py",
    "scripts/fantasypoints_wr_cb_scraper.py",
    "scripts/build/pbp_features.py",
    "scripts/player_stats_loader_v2.py",
    "scripts/player_form_v2.py",
    "scripts/slate_universe_v2.py",
    "scripts/run_player_form_v2_loader.py",
    "scripts/enrich_player_scoring_v2.py",
    "scripts/modeling/context_bridge.py",
    "scripts/modeling/bayesian_v2.py",
    "scripts/modeling/ml_v2.py",
    "scripts/modeling/state_v2.py",
    "scripts/modeling/ensemble_v2.py",
    "scripts/modeling/rules_v2.py",
    "scripts/modeling/simulation_rules.py",
    "scripts/modeling/qb_pass_synthesis_v1.py",
    "scripts/simulation_v2.py",
    "scripts/metrics_v2.py",
    "scripts/metrics_enrichment_v2.py",
    "scripts/run_metrics_context.py",
    "scripts/run_pricing_v2.py",
)

# Explicitly allowed 2025 references in 2026 production: prior-season defaults,
# documentation/comments describing historical validation, and provider history.
# Any executable current-season assignment/default must be reviewed separately.
RUNTIME_2025_PATTERNS = (
    re.compile(r"\bseason\s*=\s*2025\b", re.I),
    re.compile(r"default\s*=\s*2025\b", re.I),
    re.compile(r"datetime\.date\(\s*2025\s*,", re.I),
)


def _read(rel: str | Path) -> str:
    path = rel if isinstance(rel, Path) else ROOT / rel
    return path.read_text(encoding="utf-8")


def _finding(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _workflow_findings() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not FULL_SLATE.exists():
        return [_finding("P0", "workflow_missing", "canonical .github/workflows/full-slate.yml is missing")]
    text = _read(FULL_SLATE)
    for token in (
        'default: "2026"',
        'default: "2025"',
        "scripts/utils/build_team_week_map_v2.py",
        "scripts/run_qb_promoted_context.py",
        "scripts/run_player_form_v2_loader.py",
        "scripts/run_pricing_v2.py",
    ):
        if token not in text:
            out.append(_finding("P0", "workflow_contract", f"Full Slate missing required token: {token}"))
    return out


def _runtime_literal_findings() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for rel in PRODUCTION_RUNTIME_FILES:
        path = ROOT / rel
        if not path.exists():
            out.append(_finding("P0", "production_file_missing", f"missing production dependency: {rel}"))
            continue
        for line_no, line in enumerate(_read(path).splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Legitimate prior-season defaults are explicitly allowed.
            if "PRIOR_SEASON" in line or "prior_season" in line:
                continue
            for pattern in RUNTIME_2025_PATTERNS:
                if pattern.search(line):
                    severity = "P1" if "fantasypoints_wr_cb_scraper.py" in rel else "P0"
                    out.append(_finding(
                        severity,
                        "stale_2025_runtime_literal",
                        f"{rel}:{line_no}: {stripped}",
                    ))
                    break
    return out


def _ensemble_findings() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not ENSEMBLE_WEIGHTS.exists() or ENSEMBLE_WEIGHTS.stat().st_size == 0:
        out.append(_finding(
            "P0",
            "ensemble_not_promoted",
            "data/model_ensemble_weights.csv is absent; a fresh Full Slate checkout falls back to MC-only even though M89/M90 requires the calibrated base ensemble",
        ))
        return out
    try:
        with ENSEMBLE_WEIGHTS.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:
        return [_finding("P0", "ensemble_weights_unreadable", f"unable to parse {ENSEMBLE_WEIGHTS}: {exc}")]
    pass_rows = [r for r in rows if str(r.get("market", "")).strip().lower() == "pass_yards"]
    if not pass_rows:
        out.append(_finding("P0", "qb_ensemble_missing", "promoted ensemble weights contain no pass_yards row"))
    else:
        r = pass_rows[-1]
        try:
            weights = [float(r[k]) for k in ("mc_weight", "ml_weight", "state_weight")]
            total = sum(weights)
            if any(v < 0 for v in weights) or abs(total - 1.0) > 1e-6:
                raise ValueError(f"weights={weights} sum={total}")
        except Exception as exc:
            out.append(_finding("P0", "qb_ensemble_invalid", f"invalid pass_yards ensemble weights: {exc}"))
    return out


def _promotion_findings() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not QB_SYNTHESIS.exists() or QB_SYNTHESIS.stat().st_size == 0:
        out.append(_finding("P0", "qb_synthesis_missing", "model/qb_pass_synthesis_v1.json is missing"))
    pricing = ROOT / "scripts/run_pricing_v2.py"
    if pricing.exists():
        text = _read(pricing)
        for token in ("predict_qb_synthesis", "qb_synthesis_applied", "qb_attempt_conversion"):
            if token not in text:
                out.append(_finding("P0", "qb_synthesis_wiring", f"production pricing missing promoted QB token: {token}"))
    return out


def _legacy_authority_findings() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    engine = ROOT / "engine/engine.py"
    if engine.exists():
        text = _read(engine)
        if "2025" in text and ("run_pipeline" in text or "default=2025" in text):
            out.append(_finding(
                "P0",
                "legacy_production_authority",
                "engine/engine.py remains a runnable 2025-era orchestration path; Full Slate must be the sole canonical production authority",
            ))
    agents = ROOT / "AGENTS.md"
    if agents.exists():
        text = _read(agents)
        if "engine/engine.py" in text and "Canonical" in text:
            out.append(_finding(
                "P2",
                "stale_production_docs",
                "AGENTS.md still points to the legacy engine as canonical production",
            ))
    return out


def run() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    findings += _workflow_findings()
    findings += _runtime_literal_findings()
    findings += _ensemble_findings()
    findings += _promotion_findings()
    findings += _legacy_authority_findings()
    rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    findings.sort(key=lambda x: (rank.get(x["severity"], 9), x["code"], x["message"]))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit non-zero for any P0/P1 finding")
    args = parser.parse_args()
    findings = run()
    print("\n======== 2026 Production Readiness Audit ========")
    if not findings:
        print("[READY] no findings")
        return 0
    counts: dict[str, int] = {}
    for item in findings:
        counts[item["severity"]] = counts.get(item["severity"], 0) + 1
        print(f"[{item['severity']}] {item['code']}: {item['message']}")
    print("summary:", counts)
    blockers = [f for f in findings if f["severity"] in {"P0", "P1"}]
    if args.strict and blockers:
        print(f"[NOT READY] strict 2026 audit failed with {len(blockers)} P0/P1 finding(s)")
        return 1
    print("[AUDIT] findings recorded; strict mode not requested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
