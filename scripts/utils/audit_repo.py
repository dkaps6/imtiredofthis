#!/usr/bin/env python3
"""Static repository audit for the canonical production Full Slate path."""
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

# This list intentionally includes direct workflow entry points AND material
# imported dependencies. The old audit only listed wrappers, which allowed a
# hidden runtime season literal inside make_team_form.py to escape CI.
PRODUCTION_SCRIPTS = (
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
    "scripts/build/pbp_features.py",
    "scripts/player_form_v2.py",
    "scripts/player_stats_loader_v2.py",
    "scripts/slate_universe_v2.py",
    "scripts/run_player_form_v2.py",
    "scripts/run_player_form_v2_loader.py",
    "scripts/run_player_form_current_roles_v1.py",
    "scripts/enrich_player_scoring_v2.py",
    "scripts/modeling/context_bridge.py",
    "scripts/modeling/bayesian_v2.py",
    "scripts/modeling/ml_v2.py",
    "scripts/modeling/state_v2.py",
    "scripts/modeling/ensemble_v2.py",
    "scripts/modeling/rules_v2.py",
    "scripts/modeling/simulation_rules.py",
    "scripts/modeling/qb_pass_synthesis_v1.py",
    "scripts/run_model_context_bridge.py",
    "scripts/run_model_bayesian_bridge.py",
    "scripts/run_model_ml_bridge.py",
    "scripts/run_model_state_bridge.py",
    "scripts/run_model_ensemble_bridge.py",
    "scripts/run_model_rules_bridge.py",
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


def _syntax_errors():
    errors = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", ".venv", "venv"} for part in path.parts):
            continue
        try:
            ast.parse(_read(path), filename=str(path))
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    return errors


def _workflow_errors():
    if yaml is None:
        return ["PyYAML is not installed; cannot validate workflow YAML"]
    errors = []
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


def _presence_errors():
    return [
        f"missing production script: {rel}"
        for rel in PRODUCTION_SCRIPTS
        if not (ROOT / rel).exists()
    ]


def _contract_errors():
    errors = []
    seen = {}
    for key, c in CONTRACTS.items():
        path = c.path.resolve(strict=False)
        if path in seen:
            errors.append(f"duplicate artifact path: {key} and {seen[path]} -> {path}")
        else:
            seen[path] = key
        if c.min_rows < 0:
            errors.append(f"contract {key} has invalid min_rows={c.min_rows}")
        if len(set(c.required_columns)) != len(c.required_columns):
            errors.append(f"contract {key} contains duplicate required columns")
    return errors


def _legacy_team_form_guarded() -> bool:
    """Return True only while the legacy TeamForm 2025 literal is neutralized."""
    wrapper = ROOT / "scripts/run_team_form_context.py"
    if not wrapper.exists():
        return False
    text = _read(wrapper)
    return (
        "redirected stale PBP season request" in text
        and "_repair_success_explosive_context" in text
        and "success_explosive_source_season" in text
        and "week.lt(int(target_week))" in text
    )


def _stale_literal_errors():
    errors = []
    patterns = (
        re.compile(r"==\s*2025"),
        re.compile(r"\[\s*2025\s*\]"),
        re.compile(r"season\s*=\s*2025"),
        re.compile(r"default\s*=\s*2025"),
        re.compile(r"WEEK\s*=\s*\d+"),
    )
    legacy_guarded = _legacy_team_form_guarded()
    for rel in PRODUCTION_SCRIPTS:
        for n, line in enumerate(_read(ROOT / rel).splitlines(), 1):
            if "PRIOR_SEASON" in line or "prior_season" in line:
                continue
            if not any(p.search(line) for p in patterns):
                continue
            if rel == "scripts/make_team_form.py" and legacy_guarded:
                continue
            errors.append(f"stale runtime literal in {rel}:{n}: {line.strip()}")
    return errors


def _promoted_pricing_chain_errors(workflow_text: str) -> list[str]:
    """Fail closed on the exact promoted Week-1 V4 pricing chain.

    Full Slate no longer invokes run_pricing_v2.py directly.  The governed public
    compatibility entrypoint routes to V4, which preserves the frozen V3 core and
    then applies the exact R22 receiving-tail adapter.  The legacy pricing module
    remains a material imported dependency and is audited separately below.
    """
    errors: list[str] = []
    public = ROOT / "scripts/run_pricing_with_full_roster_universe_v3.py"
    v4 = ROOT / "scripts/run_pricing_with_full_roster_universe_v4_production.py"
    v3_core = ROOT / "scripts/run_pricing_with_full_roster_universe_v3_core.py"
    r22 = ROOT / "scripts/modeling/rb_receiving_tail_production_adapter_v1.py"
    r19_model = ROOT / "data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json"
    r19_pools = ROOT / "data/models/rb_r19_production_v1/rb_r19_residual_pools_v1.npz"

    for path, label in (
        (public, "public V3 compatibility entrypoint"),
        (v4, "V4 production entrypoint"),
        (v3_core, "preserved V3 core"),
        (r22, "R22 production adapter"),
        (r19_model, "R19 scorer asset"),
        (r19_pools, "R19 residual-pools asset"),
    ):
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"promoted pricing chain missing {label}: {path.relative_to(ROOT)}")

    if errors:
        return errors

    if "scripts/run_pricing_with_full_roster_universe_v3.py" not in workflow_text:
        errors.append("full-slate workflow does not invoke promoted public V3/V4 pricing entrypoint")

    public_text = _read(public)
    if "from scripts.run_pricing_with_full_roster_universe_v4_production import main" not in public_text:
        errors.append("public V3 compatibility entrypoint does not route exactly to V4 production")

    v4_text = _read(v4)
    for token, msg in (
        ("run_pricing_with_full_roster_universe_v3_core", "V4 does not preserve the certified V3 core"),
        ("apply_rb_receiving_tail_production", "V4 does not apply the R22 receiving-tail adapter"),
        ("data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json", "V4 does not pin the committed R19 scorer asset"),
        ("data/models/rb_r19_production_v1/rb_r19_residual_pools_v1.npz", "V4 does not pin the committed R19 residual-pools asset"),
        ("RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS", "V4 does not enforce R22 pricing lineage"),
    ):
        if token not in v4_text:
            errors.append(msg)

    r22_text = _read(r22)
    for token, msg in (
        ("EXPECTED_MODEL_SHA256", "R22 adapter does not pin scorer SHA256"),
        ("EXPECTED_POOLS_SHA256", "R22 adapter does not pin residual-pools SHA256"),
        ("sportsbook_inputs_added", "R22 adapter lacks sportsbook-separation contract"),
        ("current_or_future_outcomes_used", "R22 adapter lacks future-outcome contract"),
        ("receptions_exact", "R22 adapter lacks receptions protection gate"),
        ("non_rb_exact", "R22 adapter lacks non-RB protection gate"),
    ):
        if token not in r22_text:
            errors.append(msg)
    return errors


def _player_form_wrapper_errors(workflow_text: str) -> list[str]:
    """Protect the certified current-role PlayerForm wrapper and its legacy authority."""
    errors: list[str] = []
    wrapper_rel = "scripts/run_player_form_current_roles_v1.py"
    wrapper = ROOT / wrapper_rel
    if wrapper_rel not in workflow_text:
        errors.append(f"full-slate workflow does not invoke {wrapper_rel}")
        return errors
    if not wrapper.exists() or wrapper.stat().st_size == 0:
        errors.append(f"certified PlayerForm current-role wrapper missing: {wrapper_rel}")
        return errors
    text = _read(wrapper)
    for token, msg in (
        ("import scripts.run_player_form_v2_loader as loader", "PlayerForm current-role wrapper does not delegate to protected loader"),
        ("loader.main()", "PlayerForm current-role wrapper does not execute protected loader"),
        ("resolve_current_roles_path", "PlayerForm current-role wrapper does not resolve explicit certified current roles"),
        ("strict_prior_logs", "PlayerForm current-role wrapper lacks strict-prior history filter"),
        ("publish_strict_prior_history", "PlayerForm current-role wrapper lacks strict-prior publication step"),
        ("w.ge(week)", "PlayerForm current-role wrapper lacks target/future-week publication guard"),
    ):
        if token not in text:
            errors.append(msg)
    return errors


def _workflow_contract_errors():
    path = ROOT / ".github/workflows/full-slate.yml"
    if not path.exists():
        return []
    text = _read(path)
    required = (
        "scripts/utils/build_team_week_map_v2.py",
        "scripts/run_team_form_context.py",
        "scripts/run_qb_promoted_context.py",
        "scripts/run_player_form_current_roles_v1.py",
        "scripts/enrich_player_scoring_v2.py",
        "scripts/run_model_context_bridge.py",
        "scripts/run_model_bayesian_bridge.py",
        "scripts/run_model_ml_bridge.py",
        "scripts/run_model_state_bridge.py",
        "scripts/run_model_ensemble_bridge.py",
        "scripts/run_model_rules_bridge.py",
        "scripts/run_metrics_context.py",
        "scripts/validate_build_integrity.py",
        "scripts/metrics_ready.py",
        "scripts/run_pricing_with_full_roster_universe_v3.py",
        "scripts/utils/audit_repo.py --strict",
    )
    errors = [f"full-slate workflow does not invoke {t}" for t in required if t not in text]
    errors.extend(_player_form_wrapper_errors(text))
    errors.extend(_promoted_pricing_chain_errors(text))

    # The old run_pricing_v2 implementation remains a material dependency of the
    # preserved football-first stack, so continue verifying its core projection
    # contracts even though Full Slate no longer invokes it directly.
    pricing = ROOT / "scripts/run_pricing_v2.py"
    if pricing.exists():
        p = _read(pricing)
        for token, msg in (
            ("apply_ml_to_metrics", "production pricing does not attach canonical ML v2 projection"),
            ("apply_state_to_metrics", "production pricing does not attach canonical state v2 projection"),
            ("apply_bayesian_to_metrics", "production pricing does not apply canonical Bayesian baseline"),
            ("apply_rules_to_metrics", "production pricing does not apply canonical empirical rules"),
            ("apply_ensemble", "production pricing does not apply canonical ensemble interface"),
            ("mc_proj", "production pricing does not preserve Monte Carlo component projection"),
            ("ensemble_proj", "production pricing does not preserve ensemble projection"),
            ("predict_qb_synthesis", "production pricing does not apply promoted QB synthesis"),
            ("qb_attempt_conversion", "production pricing does not preserve QB official-attempt conversion"),
        ):
            if token not in p:
                errors.append(msg)

    qb_model = ROOT / "model/qb_pass_synthesis_v1.json"
    ensemble_weights = ROOT / "data/model_ensemble_weights.csv"
    if qb_model.exists():
        if not ensemble_weights.exists() or ensemble_weights.stat().st_size == 0:
            errors.append("promoted QB synthesis exists without committed pass_yards ensemble weights")
        else:
            try:
                weights = __import__("pandas").read_csv(ensemble_weights)
                q = weights.loc[weights["market"].astype(str).str.lower().eq("pass_yards")]
                if len(q) != 1:
                    errors.append("promoted QB synthesis requires exactly one pass_yards ensemble-weight row")
                else:
                    vals = [float(q.iloc[0][c]) for c in ("mc_weight", "ml_weight", "state_weight")]
                    if any(v < 0 for v in vals) or abs(sum(vals) - 1.0) > 1e-6 or vals[1] == 0 or vals[2] == 0:
                        errors.append(f"promoted QB pass_yards ensemble weights invalid: {vals}")
            except Exception as exc:
                errors.append(f"unable to validate promoted QB ensemble weights: {exc}")

    legacy_ml = ROOT / "scripts/models/ml_ensemble.py"
    if legacy_ml.exists() and "ML fallback 0.5" in _read(legacy_ml):
        errors.append("legacy ML placeholder still silently returns 0.5")
    legacy_state = ROOT / "scripts/models/markov.py"
    if legacy_state.exists() and "Markov fallback" in _read(legacy_state):
        errors.append("legacy pseudo-Markov model still silently returns 0.5")
    legacy_ensemble = ROOT / "scripts/models/ensemble.py"
    if legacy_ensemble.exists():
        etext = _read(legacy_ensemble)
        if "0.25" in etext or "65/35" in etext or "p_market_fair" in etext:
            errors.append("legacy fixed-weight/market-blended ensemble remains active")
    ensemble = ROOT / "scripts/modeling/ensemble_v2.py"
    if ensemble.exists():
        etext = _read(ensemble)
        if "positive=True" not in etext or "uncalibrated_mc_only" not in etext:
            errors.append("canonical ensemble does not enforce nonnegative calibrated weights and explicit MC fallback")

    engine = ROOT / "engine/engine.py"
    if engine.exists() and "The only canonical production pipeline" not in _read(engine):
        errors.append("legacy engine exists but is not explicitly retired in favor of Full Slate")
    return errors


def run_audit():
    checks = {
        "presence": _presence_errors(),
        "python syntax": _syntax_errors(),
        "workflow yaml": _workflow_errors(),
        "artifact contracts": _contract_errors(),
        "runtime literals": _stale_literal_errors(),
        "workflow wiring": _workflow_contract_errors(),
    }
    fail = []
    print("\n======== Repo Audit — Production Full Slate ========")
    for label, issues in checks.items():
        if issues:
            print(f"[FAIL] {label}: {len(issues)}")
            for issue in issues:
                print("  -", issue)
                fail.append(f"{label}: {issue}")
        else:
            print(f"[OK] {label}")
    return fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    fail = run_audit()
    if fail and args.strict:
        print(f"[AUDIT] strict mode failed with {len(fail)} issue(s)")
        return 1
    print(f"[AUDIT] {'passed' if not fail else f'completed with {len(fail)} warning issue(s)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
