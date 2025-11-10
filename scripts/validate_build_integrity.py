"""
Build Validation Script (with CI enforcement)
Auto-runs after metrics generation and fails builds if validation fails.
"""

import pandas as pd
import os
import sys
from datetime import datetime

DATA_PATH = "data"
LOG_PATH = os.path.join(DATA_PATH, "logs")
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE = os.path.join(LOG_PATH, "validate_build.log")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[VALIDATE] {ts} | {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

EXPECTED_FILES = {
    "roles_ourlads.csv": {"min_rows": 350, "must_have": ["player", "team", "role", "position"]},
    "opponent_map.csv": {"min_rows": 32, "must_have": ["team", "opponent"]},
    "player_form.csv": {"min_rows": 350, "must_have": ["player", "team", "week", "season"]},
    "props_raw.csv": {"min_rows": 150, "must_have": ["player", "team_abbr", "opponent_abbr", "market_type"]},
    "weather.csv": {"min_rows": 20, "must_have": ["team_abbr", "week"]},
    "make_metrics_output.csv": {"min_rows": 350, "must_have": ["player", "team", "week"]}
}

def validate_file(fname, meta):
    path = os.path.join(DATA_PATH, fname)
    if not os.path.exists(path):
        log(f"❌ Missing file: {fname}")
        return False
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"❌ Could not read {fname}: {e}")
        return False

    # Row count check
    if len(df) < meta["min_rows"]:
        log(f"⚠️ {fname} has only {len(df)} rows (expected ≥ {meta['min_rows']})")

    # Column check
    missing_cols = [c for c in meta["must_have"] if c not in df.columns]
    if missing_cols:
        log(f"❌ {fname} missing required columns: {missing_cols}")
        return False

    log(f"✅ {fname} validated ({len(df)} rows, {len(df.columns)} cols)")
    return True


def run_core_validation():
    log("=" * 60)
    log("STARTING BUILD VALIDATION")
    log("=" * 60)
    passed = True

    for f, meta in EXPECTED_FILES.items():
        ok = validate_file(f, meta)
        passed &= ok

    # 2025 season-only check
    pf = os.path.join(DATA_PATH, "player_form.csv")
    if os.path.exists(pf):
        df = pd.read_csv(pf)
        if "season" in df.columns:
            non_2025 = len(df[df["season"] != 2025])
            if non_2025 > 0:
                log(f"⚠️ {non_2025} players not from 2025 season found in player_form.csv")
                passed = False

    # Opponent bidirectional consistency
    om = os.path.join(DATA_PATH, "opponent_map.csv")
    if os.path.exists(om):
        df = pd.read_csv(om)
        if {"team", "opponent"}.issubset(df.columns):
            tset, oset = set(df.team), set(df.opponent)
            missing = tset - oset
            if missing:
                log(f"⚠️ Missing reverse opponent mappings for: {', '.join(missing)}")
                passed = False

    # Missing player names in final metrics
    mm = os.path.join(DATA_PATH, "make_metrics_output.csv")
    if os.path.exists(mm):
        df = pd.read_csv(mm)
        if "player" in df.columns and df["player"].isna().any():
            log(f"⚠️ {df['player'].isna().sum()} missing player names in make_metrics_output.csv")
            passed = False

    log("=" * 60)
    if passed:
        log("✅ VALIDATION PASSED SUCCESSFULLY")
    else:
        log("❌ VALIDATION FOUND ISSUES — BUILD WILL FAIL")
    log("=" * 60)

    return passed


if __name__ == "__main__":
    ok = run_core_validation()
    sys.exit(0 if ok else 1)
