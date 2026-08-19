#!/usr/bin/env python3
"""Static repository audit for the production Full Slate path."""
from __future__ import annotations
import argparse, ast, re
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None
from scripts.artifact_contracts import CONTRACTS, ROOT

PRODUCTION_SCRIPTS = (
    "scripts/config.py","scripts/runtime_context.py","scripts/providers/ourlads_depth.py","scripts/utils/make_team_week_map.py",
    "scripts/fetch_props_oddsapi.py","scripts/build/build_opponent_map_from_props.py","scripts/providers/sharpfootball_pull.py",
    "scripts/run_team_form_context.py","scripts/build/build_weather_week.py","scripts/build/build_injuries_weekly.py","scripts/build/pbp_features.py",
    "scripts/player_form_v2.py","scripts/player_stats_loader_v2.py","scripts/run_player_form_v2.py","scripts/run_player_form_v2_loader.py","scripts/enrich_player_scoring_v2.py",
    "scripts/modeling/context_bridge.py","scripts/modeling/bayesian_v2.py","scripts/modeling/ml_v2.py","scripts/modeling/state_v2.py","scripts/modeling/rules_v2.py","scripts/modeling/simulation_rules.py",
    "scripts/run_model_context_bridge.py","scripts/run_model_bayesian_bridge.py","scripts/run_model_ml_bridge.py","scripts/run_model_state_bridge.py","scripts/run_model_rules_bridge.py",
    "scripts/metrics_v2.py","scripts/metrics_enrichment_v2.py","scripts/run_metrics_context.py","scripts/metrics_ready.py","scripts/pricing_v2.py","scripts/simulation_v2.py","scripts/run_pricing_v2.py",
    "scripts/validate_build_integrity.py","scripts/artifact_contracts.py",
)
WORKFLOWS=(".github/workflows/full-slate.yml",".github/workflows/audit-only.yml",".github/workflows/repo-ci.yml")

def _read(path:Path)->str:return path.read_text(encoding="utf-8")
def _syntax_errors():
    errors=[]
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git",".venv","venv"} for part in path.parts):continue
        try:ast.parse(_read(path),filename=str(path))
        except Exception as exc:errors.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    return errors
def _workflow_errors():
    if yaml is None:return ["PyYAML is not installed; cannot validate workflow YAML"]
    errors=[]
    for rel in WORKFLOWS:
        path=ROOT/rel
        if not path.exists():errors.append(f"missing workflow: {rel}");continue
        try:yaml.safe_load(_read(path))
        except Exception as exc:errors.append(f"invalid workflow {rel}: {exc}")
    return errors
def _presence_errors():return [f"missing production script: {rel}" for rel in PRODUCTION_SCRIPTS if not (ROOT/rel).exists()]
def _contract_errors():
    errors=[];seen={}
    for key,c in CONTRACTS.items():
        path=c.path.resolve(strict=False)
        if path in seen:errors.append(f"duplicate artifact path: {key} and {seen[path]} -> {path}")
        else:seen[path]=key
        if c.min_rows<0:errors.append(f"contract {key} has invalid min_rows={c.min_rows}")
        if len(set(c.required_columns))!=len(c.required_columns):errors.append(f"contract {key} contains duplicate required columns")
    return errors
def _stale_literal_errors():
    errors=[];patterns=(re.compile(r"==\s*2025"),re.compile(r"\[\s*2025\s*\]"),re.compile(r"season\s*=\s*2025"),re.compile(r"default\s*=\s*2025"),re.compile(r"WEEK\s*=\s*\d+"))
    for rel in PRODUCTION_SCRIPTS:
        for n,line in enumerate(_read(ROOT/rel).splitlines(),1):
            if "PRIOR_SEASON" in line or "prior_season" in line:continue
            if any(p.search(line) for p in patterns):errors.append(f"stale runtime literal in {rel}:{n}: {line.strip()}")
    return errors
def _workflow_contract_errors():
    path=ROOT/".github/workflows/full-slate.yml"
    if not path.exists():return []
    text=_read(path)
    required=("scripts/run_player_form_v2_loader.py","scripts/enrich_player_scoring_v2.py","scripts/run_model_context_bridge.py","scripts/run_model_bayesian_bridge.py","scripts/run_model_ml_bridge.py","scripts/run_model_state_bridge.py","scripts/run_model_rules_bridge.py","scripts/run_metrics_context.py","scripts/validate_build_integrity.py","scripts/metrics_ready.py","scripts/run_pricing_v2.py","scripts/utils/audit_repo.py --strict")
    errors=[f"full-slate workflow does not invoke {t}" for t in required if t not in text]
    pricing=ROOT/"scripts/run_pricing_v2.py"
    if pricing.exists():
        p=_read(pricing)
        for token,msg in (("apply_ml_to_metrics","production pricing does not attach canonical ML v2 projection"),("apply_state_to_metrics","production pricing does not attach canonical state v2 projection"),("apply_bayesian_to_metrics","production pricing does not apply canonical Bayesian baseline"),("apply_rules_to_metrics","production pricing does not apply canonical empirical rules"),("ml_proj","production pricing does not preserve ML v2 projection"),("state_proj","production pricing does not preserve state v2 projection")):
            if token not in p:errors.append(msg)
    legacy_ml=ROOT/"scripts/models/ml_ensemble.py"
    if legacy_ml.exists() and "ML fallback 0.5" in _read(legacy_ml):errors.append("legacy ML placeholder still silently returns 0.5")
    legacy_state=ROOT/"scripts/models/markov.py"
    if legacy_state.exists() and "Markov fallback" in _read(legacy_state):errors.append("legacy pseudo-Markov model still silently returns 0.5")
    return errors
def run_audit():
    checks={"presence":_presence_errors(),"python syntax":_syntax_errors(),"workflow yaml":_workflow_errors(),"artifact contracts":_contract_errors(),"runtime literals":_stale_literal_errors(),"workflow wiring":_workflow_contract_errors()};fail=[]
    print("\n======== Repo Audit — Production Full Slate ========")
    for label,issues in checks.items():
        if issues:
            print(f"[FAIL] {label}: {len(issues)}")
            for issue in issues:print("  -",issue);fail.append(f"{label}: {issue}")
        else:print(f"[OK] {label}")
    return fail
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--strict",action="store_true");args=parser.parse_args();fail=run_audit()
    if fail and args.strict:print(f"[AUDIT] strict mode failed with {len(fail)} issue(s)");return 1
    print(f"[AUDIT] {'passed' if not fail else f'completed with {len(fail)} warning issue(s)'}");return 0
if __name__=="__main__":raise SystemExit(main())
