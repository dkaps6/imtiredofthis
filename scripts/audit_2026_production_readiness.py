#!/usr/bin/env python3
"""Static 2026 production-readiness audit for the canonical Full Slate path.

Runtime provider health is validated separately by validate_2026_provider_artifacts.
This audit protects the repository wiring so the canonical v3 identity/context/provider
contracts cannot be accidentally bypassed by a later refactor.

Important: this audit follows the *canonical production entrypoints*.  Legacy helper
modules may retain historical literals when they are encapsulated by a production
wrapper that injects runtime season/week and repairs provenance.  We validate that
wrapper contract explicitly instead of flagging every historical literal in a helper.
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

# Files that directly own canonical runtime behavior.  Intentionally excluded:
# - scripts/make_team_form.py: legacy builder encapsulated by run_team_form_context.py
# - scripts/fantasypoints_wr_cb_scraper.py: standalone legacy scraper; not invoked by
#   the canonical Coverage v2 runner.
PRODUCTION_RUNTIME_FILES = (
    "scripts/config.py",
    "scripts/runtime_context.py",
    "scripts/providers/ourlads_depth.py",
    "scripts/utils/build_team_week_map_v2.py",
    "scripts/utils/make_team_week_map.py",
    "scripts/fetch_props_oddsapi.py",
    "scripts/providers/sharpfootball_pull.py",
    "scripts/run_sharpfootball_v2.py",
    "scripts/team_form_prior_bridge.py",
    "scripts/run_team_form_context.py",
    "scripts/run_qb_promoted_context.py",
    "scripts/team_context_v3.py",
    "scripts/build/build_weather_week_v2.py",
    "scripts/build/build_weather_week.py",
    "scripts/build/build_injuries_weekly.py",
    "scripts/run_coverage_v2.py",
    "scripts/build/build_coverage_v2.py",
    "scripts/build/pbp_features.py",
    "scripts/player_stats_loader_v2.py",
    "scripts/player_form_v2.py",
    "scripts/slate_universe_v2.py",
    "scripts/run_player_form_v2_loader.py",
    "scripts/enrich_player_scoring_v2.py",
    "scripts/utils/player_identity_v3.py",
    "scripts/validate_player_identity_v3.py",
    "scripts/validate_2026_provider_artifacts.py",
    "scripts/modeling/context_bridge.py",
    "scripts/modeling/context_bridge_v3.py",
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
    "scripts/run_model_context_bridge.py",
    "scripts/run_pricing_v2.py",
)

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
        "scripts/run_team_form_context.py",
        "scripts/run_qb_promoted_context.py",
        "scripts/run_player_form_v2_loader.py",
        "scripts/run_model_context_bridge.py",
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
            if "PRIOR_SEASON" in line or "prior_season" in line:
                continue
            for pattern in RUNTIME_2025_PATTERNS:
                if pattern.search(line):
                    out.append(_finding(
                        "P0",
                        "stale_2025_runtime_literal",
                        f"{rel}:{line_no}: {stripped}",
                    ))
                    break
    return out


def _team_form_wrapper_findings() -> list[dict[str, str]]:
    """Protect the runtime wrapper around the legacy TeamForm builder."""
    out: list[dict[str, str]] = []
    wrapper = ROOT / "scripts/run_team_form_context.py"
    if not wrapper.exists():
        return [_finding("P0", "team_form_wrapper_missing", str(wrapper.relative_to(ROOT)))]
    text = _read(wrapper)
    for token in (
        "resolve_season",
        "resolve_prior_season",
        "resolve_week",
        "_install_pbp_season_guard",
        "_repair_success_explosive_context",
        "_stamp_provenance",
        '"--season"',
        "make_team_form.main()",
    ):
        if token not in text:
            out.append(_finding(
                "P0",
                "team_form_wrapper_contract",
                f"run_team_form_context.py missing required runtime/provenance token: {token}",
            ))

    smoke = ROOT / ".github/workflows/2026-full-slate-smoke.yml"
    if smoke.exists() and "--box-backfill-prev" not in _read(smoke):
        out.append(_finding(
            "P0",
            "team_form_preseason_contract",
            "2026 no-credit smoke must explicitly enable prior-season box backfill during preseason",
        ))
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


def _v3_contract_findings() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    checks = {
        "scripts/run_model_context_bridge.py": (
            "run_provider_readiness",
            "materialize_team_context",
            "context_bridge_v3",
        ),
        "scripts/modeling/qb_pass_synthesis_v1.py": (
            "data/team_context_v3.csv",
            "TEAM_CONTEXT_V3",
        ),
        "scripts/run_player_form_v2_loader.py": (
            "validate_player_identity_v3",
        ),
        "scripts/validate_player_identity_v3.py": (
            "player_identity_validation.csv",
        ),
        "scripts/build/build_injuries_weekly.py": (
            "injuries_source_status.json",
            "provider_outage",
            "no_official_report",
        ),
    }
    for rel, tokens in checks.items():
        path = ROOT / rel
        if not path.exists():
            out.append(_finding("P0", "v3_contract_file_missing", rel))
            continue
        text = _read(path)
        for token in tokens:
            if token not in text:
                out.append(_finding("P0", "v3_contract_wiring", f"{rel} missing required v3 token: {token}"))
    return out


def _legacy_authority_findings() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    engine = ROOT / "engine/engine.py"
    if engine.exists():
        text = _read(engine)
        retired_tokens = (
            ".github/workflows/full-slate.yml",
            "raise RuntimeError(DEPRECATION_MESSAGE)",
            "intentionally non-runnable",
        )
        if not all(token in text for token in retired_tokens):
            out.append(_finding(
                "P0",
                "legacy_production_authority",
                "engine/engine.py is not demonstrably retired/fail-closed; Full Slate must be the sole canonical production authority",
            ))

    agents = ROOT / "AGENTS.md"
    if agents.exists():
        text = _read(agents)
        docs_ok = (
            ".github/workflows/full-slate.yml" in text
            and "The old `engine/engine.py` path is retired" in text
            and "only canonical production orchestration path" in text
        )
        if not docs_ok:
            out.append(_finding(
                "P2",
                "stale_production_docs",
                "AGENTS.md does not clearly establish Full Slate as sole canonical production and retire engine/engine.py",
            ))
    return out


def run() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    findings += _workflow_findings()
    findings += _runtime_literal_findings()
    findings += _team_form_wrapper_findings()
    findings += _ensemble_findings()
    findings += _promotion_findings()
    findings += _v3_contract_findings()
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
