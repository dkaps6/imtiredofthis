#!/usr/bin/env python3
import os, sys, ast, importlib.utils, json, re, glob
from typing import List, Dict, Tuple

try:
    import yaml  # pip install pyyaml if needed
except Exception:
    yaml = None

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
REPO = os.path.abspath(os.path.join(ROOT, ".."))

# Where inputs should live (your new convention)
DATA_DIR = os.path.join(REPO, "data")
OUTPUTS_DIR = os.path.join(REPO, "outputs")

# Expected data inputs for your new builders
EXPECTED_DATA = [
    "player_form_consensus.csv",
    "cb_coverage_team.csv",
    "cb_coverage_player.csv",
    "weather.csv",
    "injuries.csv",
    "qb_scramble_rates.csv",
    "qb_designed_runs.csv",
    "wr_cb_exposure.csv",
    "play_volume_splits.csv",
    "volatility_widening.csv",
    "run_pass_funnel.csv",
    "coverage_penalties.csv",
    "script_escalators.csv",
    "opponent_map_from_props.csv",
]

# Key scripts we expect to exist
EXPECTED_SCRIPTS = [
    "scripts/make_player_form.py",
    "scripts/make_team_form.py",
    "scripts/make_metrics.py",
    "scripts/pricing.py",
    "scripts/enrich_player_form.py",
    "scripts/enrich_team_form.py",
    "scripts/validate_metrics.py",
    "scripts/build/build_cb_coverage_team.py",
    "scripts/build/build_cb_coverage_player.py",
    "scripts/build/build_weather_week.py",
    "scripts/build/build_injuries_weekly.py",
    "scripts/build/build_qb_run_metrics.py",
    "scripts/build/build_wr_cb_exposure.py",
    "scripts/build/build_play_volume_splits.py",
    "scripts/build/build_volatility_widening.py",
    "scripts/build/build_run_pass_funnel.py",
    "scripts/build/build_coverage_penalties.py",
    "scripts/build/build_script_escalators.py",
    "scripts/build/build_opponent_map_from_props.py",
    "scripts/utils/merge_opponent_into_player_form.py",
    "scripts/run_all_builds.py",
]

def headline(msg): print("\n" + "="*8 + " " + msg + " " + "="*8)
def warn(msg): print(f"[WARN] {msg}")
def ok(msg): print(f"[OK] {msg}")

def find_py_files(base) -> List[str]:
    files = []
    for root, _, fnames in os.walk(base):
        for f in fnames:
            if f.endswith(".py"):
                files.append(os.path.join(root, f))
    return sorted(files)

def syntax_check(py_path: str) -> Tuple[bool, str]:
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            src = f.read()
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}:{e.offset}"
    except Exception as e:
        return False, f"ParseError: {type(e).__name__}: {e}"

def safe_importable(path: str) -> bool:
    # Try to import only repo-local modules under scripts/ or engine/ or model/
    if not ("/scripts/" in path or "/engine/" in path or "/model/" in path):
        return False
    if path.endswith("__init__.py"):
        return False
    # Heuristic: skip heavy entrypoints (pricing, CLI) to avoid side effects
    skip_names = ("pricing.py","engine.py","run_predictors.py")
    if os.path.basename(path) in skip_names:
        return False
    return True

def import_check(py_path: str) -> Tuple[bool, str]:
    if not safe_importable(py_path):
        return True, ""
    try:
        spec = importlib.utils.spec_from_file_location("audit_tmp_module", py_path)
        mod = importlib.utils.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return True, ""
    except Exception as e:
        return False, f"ImportError: {type(e).__name__}: {e}"

def load_yaml(path: str) -> Tuple[bool, str]:
    if yaml is None:
        return False, "pyyaml not installed"
    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
        return True, ""
    except Exception as e:
        return False, f"YAMLError: {e}"

def load_json(path: str) -> Tuple[bool, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True, ""
    except Exception as e:
        return False, f"JSONError: {e}"

def check_files_exist(paths: List[str]) -> List[str]:
    missing = []
    for p in paths:
        if not os.path.exists(os.path.join(REPO, p)):
            missing.append(p)
    return missing

def detect_path_mismatches() -> List[str]:
    issues = []
    # Any expected data living in outputs/ instead of data/?
    for name in EXPECTED_DATA:
        if os.path.exists(os.path.join(OUTPUTS_DIR, name)) and not os.path.exists(os.path.join(DATA_DIR, name)):
            issues.append(f"Data file {name} exists in outputs/ but not in data/ (should move to data/).")
    return issues

def scan_workflows():
    wf_dir = os.path.join(REPO, ".github", "workflows")
    results = []
    if not os.path.isdir(wf_dir):
        return ["No .github/workflows directory found."]
    for y in glob.glob(os.path.join(wf_dir, "*.yml")) + glob.glob(os.path.join(wf_dir, "*.yaml")):
        ok_y, msg = load_yaml(y)
        results.append(f"{'OK' if ok_y else 'FAIL'} workflow {os.path.basename(y)} {'' if ok_y else '('+msg+')'}")
    return results

def simple_schema_hints() -> Dict[str, List[str]]:
    # Very light schema expectations to catch obvious column mismatches
    return {
        "player_form_consensus.csv": ["player","team","week"],
        "opponent_map_from_props.csv": ["player","team","week","opponent"],
        "cb_coverage_team.csv": ["team"],
        "weather.csv": ["team","opponent","week"],
        "injuries.csv": ["player","team","week"],
        "qb_scramble_rates.csv": ["player","week"],
        "qb_designed_runs.csv": ["player","week"],
        "wr_cb_exposure.csv": ["player","opponent","week"],
        "play_volume_splits.csv": ["team","week"],
        "volatility_widening.csv": ["team","week"],
        "run_pass_funnel.csv": ["team","week"],
        "coverage_penalties.csv": ["team","week"],
        "script_escalators.csv": ["team","week"],
    }

def check_csv_columns():
    import pandas as pd
    findings = []
    for name, cols in simple_schema_hints().items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            findings.append(f"[MISS] data/{name} not found")
            continue
        try:
            df = pd.read_csv(path, nrows=5)
            missing_cols = [c for c in cols if c not in df.columns]
            if missing_cols:
                findings.append(f"[SCHEMA] data/{name} missing columns: {missing_cols}")
            else:
                findings.append(f"[OK] data/{name} columns OK for quick check")
        except Exception as e:
            findings.append(f"[FAIL] data/{name} unreadable: {e}")
    return findings

def main():
    headline("Repo Audit — Structure & Scripts")

    # 1) Expected scripts present?
    miss = check_files_exist(EXPECTED_SCRIPTS)
    if miss:
        for m in miss: warn(f"Missing script: {m}")
    else:
        ok("All expected scripts found.")

    # 2) Python syntax & import checks
    py_files = find_py_files(REPO)
    syntax_fail = []
    import_fail = []
    for p in py_files:
        ok_s, msg = syntax_check(p)
        if not ok_s: syntax_fail.append((p, msg))
        ok_i, msg2 = import_check(p)
        if not ok_i and msg2: import_fail.append((p, msg2))

    if syntax_fail:
        headline("Syntax/Indentation Issues")
        for p, m in syntax_fail: print(f"- {p}: {m}")
    else:
        ok("No syntax/indentation errors detected via ast.parse")

    if import_fail:
        headline("Import Errors (repo-local modules)")
        for p, m in import_fail: print(f"- {p}: {m}")
    else:
        ok("No import errors for repo-local modules (skipped heavy entrypoints)")

    # 3) Workflow checks
    headline("GitHub Workflows")
    for line in scan_workflows():
        print("-", line)

    # 4) Data location & schema checks
    headline("Data Location")
    for issue in detect_path_mismatches() or ["No path mismatches detected (data/ vs outputs/)"]:
        print("-", issue)

    headline("Data Schema Quick Checks")
    for line in check_csv_columns():
        print("-", line)

    # 5) Communication — who consumes what? (lightweight hints)
    headline("Communication Map (light)")
    consumers = {
        "scripts/make_metrics.py": ["data/player_form_consensus.csv","data/cb_coverage_team.csv","data/weather.csv"],
        "scripts/enrich_player_form.py": ["data/player_form_consensus.csv","data/opponent_map_from_props.csv"],
        "scripts/enrich_team_form.py": ["data/make_team_form.csv","data/run_pass_funnel.csv","data/coverage_penalties.csv"],
        "scripts/pricing.py": ["data/player_form_consensus.csv","data/props_raw.csv"],
    }
    for k, v in consumers.items():
        exists = os.path.exists(os.path.join(REPO, k))
        print(f"- {k}: {'present' if exists else 'missing'}; expects {', '.join(v)}")

    print("\nDone. Paste this report to your assistant for targeted fixes.")

if __name__ == "__main__":
    main()
