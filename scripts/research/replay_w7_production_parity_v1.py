"""Exact production-path W7 replay: baseline parity first, then calibration OFF vs ON.

Required by GPT audit 6472971619 section 4. The earlier validation recalibrated RAW
`simulation_v2` draws and so measured a counterfactual, not the distribution production
prices. This harness instead runs the REAL chain --
`run_pricing_with_full_roster_universe_v3` -> V6 -> V5 -> V4 -> V1 -> `run_pricing_v2.main()`
-- inside an assembled, isolated root whose `data/` and `outputs/` are populated from the
archived W7 fixture, the repo's committed canonical artifacts, and the archived paid props.

Because the pricer's paths are relative module constants (`DATA = Path("data")`), the
chain is run in a subprocess with `cwd` set to that root. Nothing in the live tree is
touched and no odds are fetched.

**Parity is the gate.** Baseline `model_proj`, `model_sd` and `fair_prob` must reproduce
against the archived board per eligible row BEFORE any treatment result is read. Rows that
cannot reproduce are accounted for explicitly, never dropped silently. If parity fails,
this fails closed and the calibration stays a CANDIDATE.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "data/research/week7_opportunity_trace"
PREFIX = "week7_trace__"
BOARD = REPO / "data/market_track_record/boards/2026_wk07.csv"
RAW_PROPS = REPO / "data/market_track_record/raw_props/2026_wk07.csv"
OUT_JSON = REPO / "data/research/w7_parity_replay/w7_production_parity_replay_v1.json"

SEASON, WEEK = 2026, 7
# Canonical artifacts that live in the repo rather than the fixture.
REPO_ARTIFACTS = ("data/model_ensemble_weights.csv",)
REPO_DIRS = ("model",)
PARITY_COLS = ("model_proj", "model_sd", "fair_prob")
PARITY_KEY = ["player_clean_key", "market"]
# Frozen: parity counts as reproduced when every compared column matches this closely.
PARITY_RTOL, PARITY_ATOL = 1e-6, 1e-9


def assemble_root(root: Path) -> dict:
    """Build an isolated data/ + outputs/ tree from preserved, committed sources only."""
    data, outputs = root / "data", root / "outputs"
    for d in (data, outputs, data / "_debug"):
        d.mkdir(parents=True, exist_ok=True)

    placed, skipped = [], []
    for src in sorted(FIXTURE.glob(f"{PREFIX}*")):
        name = src.name[len(PREFIX):]
        if name.startswith("data__"):
            rel = name[len("data__"):].replace("__", "/")
            dest = data / rel
        elif name.startswith("outputs__"):
            rel = name[len("outputs__"):].replace("__", "/")
            dest = outputs / rel
        else:
            skipped.append(name)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        placed.append(dest.relative_to(root).as_posix())

    for rel in REPO_ARTIFACTS:
        s = REPO / rel
        if s.exists():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dest)
            placed.append(rel)
    for d in REPO_DIRS:
        s = REPO / d
        if s.exists():
            shutil.copytree(s, root / d, dirs_exist_ok=True)
            placed.append(f"{d}/")

    # The paid lines the fixture's denylist stripped, recovered from the committed
    # market track record. This is an archive read, never a fetch.
    shutil.copy2(RAW_PROPS, outputs / "props_raw.csv")
    placed.append("outputs/props_raw.csv")

    # The fixture's own priced output is denylist-stripped to 11 columns and would be
    # mistaken for a prior result; remove it so the replay cannot read its own target.
    stale = outputs / "props_priced_clean.csv"
    if stale.exists():
        stale.unlink()
        placed.append("(removed stale outputs/props_priced_clean.csv)")

    return {"placed": len(placed), "non_data_fixture_files_skipped": skipped,
            "props_source": RAW_PROPS.relative_to(REPO).as_posix()}


# Inputs the real pricer consumes, and the columns it needs from each. Checked BEFORE
# the chain runs so an incomplete fixture fails closed with a reason instead of a
# stack trace three wrappers deep.
REQUIRED_INPUTS = {
    "data/model_rule_simulation_inputs.csv": ("player_clean_key", "team"),
    "data/metrics_ready.csv": ("player_clean_key", "team"),
    "data/model_ml_diagnostics.csv": ("ml_available", "ml_method", "ml_pass_yards",
                                      "ml_rush_yards", "ml_rec_yards", "ml_receptions"),
    "data/model_state_diagnostics.csv": ("state_available", "state_method"),
    "data/player_form_consensus.csv": ("player_clean_key",),
    "data/model_ensemble_weights.csv": ("market",),
    "outputs/props_raw.csv": ("player", "market", "line"),
}


def check_inputs(root: Path) -> dict:
    """Pre-flight: can this assembled root even support the production chain?"""
    missing_files, incomplete = [], {}
    for rel, cols in REQUIRED_INPUTS.items():
        p = root / rel
        if not p.exists():
            missing_files.append(rel)
            continue
        have = set(pd.read_csv(p, nrows=1, low_memory=False).columns)
        absent = [c for c in cols if c not in have]
        if absent:
            incomplete[rel] = {"missing_columns": absent, "columns_present": len(have)}
    usable = not missing_files and not incomplete
    return {"usable": usable, "missing_files": missing_files,
            "column_incomplete": incomplete,
            "reason": ("" if usable else
                       "the archived fixture is a column subset: the preservation "
                       "denylist stripped market-named columns, including the ML and "
                       "State prediction columns, so ensemble_proj -> target_mean -> "
                       "the priced distribution's location cannot be reproduced")}


def set_promotion(root: Path, promoted: bool) -> str:
    """Flip the replay root's own calibration artifact. Never touches the live one."""
    src = REPO / "data/mc_dispersion_calibration_v1.json"
    dest = root / "data/mc_dispersion_calibration_v1.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    raw["promotion_state"] = "PROMOTED" if promoted else "CANDIDATE_PENDING_PARITY_REPLAY"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw["promotion_state"]


def run_chain(root: Path, label: str) -> dict:
    """Run the real production pricing entry point with cwd = the assembled root."""
    cmd = [sys.executable, "-m", "scripts.run_pricing_with_full_roster_universe_v3",
           "--season", str(SEASON)]
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin:/usr/local/bin",
           "HOME": str(root), "SEASON": str(SEASON), "WEEK": str(WEEK),
           "PRIOR_SEASON": str(SEASON - 1)}
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=3600)
    out = root / "outputs/props_priced_clean.csv"
    return {"arm": label, "returncode": proc.returncode, "produced": out.exists(),
            "rows": int(len(pd.read_csv(out, low_memory=False))) if out.exists() else 0,
            "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}


def parity(replayed: pd.DataFrame, archived: pd.DataFrame) -> dict:
    """Did the baseline arm reproduce the archived board? Gate, not a diagnostic."""
    r = replayed.drop_duplicates(PARITY_KEY).set_index(PARITY_KEY)
    a = archived.drop_duplicates(PARITY_KEY).set_index(PARITY_KEY)
    shared = r.index.intersection(a.index)
    out = {
        "archived_rows": int(len(a)), "replayed_rows": int(len(r)),
        "joined_rows": int(len(shared)),
        "archived_not_replayed": int(len(a.index.difference(r.index))),
        "replayed_not_archived": int(len(r.index.difference(a.index))),
        "columns": {},
    }
    for col in PARITY_COLS:
        if col not in r.columns or col not in a.columns:
            out["columns"][col] = {"state": "column_absent"}
            continue
        x = pd.to_numeric(r.loc[shared, col], errors="coerce")
        y = pd.to_numeric(a.loc[shared, col], errors="coerce")
        both = x.notna() & y.notna()
        match = np.isclose(x[both], y[both], rtol=PARITY_RTOL, atol=PARITY_ATOL)
        out["columns"][col] = {
            "compared": int(both.sum()),
            "matched": int(match.sum()),
            "match_rate": float(match.mean()) if both.any() else 0.0,
            "max_abs_diff": float((x[both] - y[both]).abs().max()) if both.any() else None,
        }
    rates = [c.get("match_rate", 0.0) for c in out["columns"].values() if "match_rate" in c]
    out["min_match_rate"] = float(min(rates)) if rates else 0.0
    out["PARITY_ESTABLISHED"] = bool(rates and min(rates) >= 0.99
                                     and out["joined_rows"] >= 0.90 * out["archived_rows"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="assembled replay root")
    ap.add_argument("--keep", action="store_true", help="keep the root for inspection")
    args = ap.parse_args()
    root = args.root
    if root.exists() and not args.keep:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    report = {"season": SEASON, "week": WEEK,
              "assembly": assemble_root(root),
              "audit_requirement": "GPT 6472971619 section 4: baseline parity before treatment"}

    report["input_preflight"] = check_inputs(root)
    if not report["input_preflight"]["usable"]:
        report["PARITY_ESTABLISHED"] = False
        report["disposition"] = "FAILED_CLOSED_FIXTURE_INCOMPLETE_PARITY_NOT_ESTABLISHABLE"
        report["consequence"] = (
            "The MC dispersion calibration stays CANDIDATE_PENDING_PARITY_REPLAY and "
            "inert. The earlier week-7 numbers (Brier 0.2713 -> 0.2555) remain a "
            "raw-simulator counterfactual and must not be cited as production evidence."
        )
        report["what_would_unblock"] = [
            "a Full Slate run that archives model_ml_diagnostics.csv and "
            "model_state_diagnostics.csv WITH their prediction columns, alongside the "
            "board -- the same durability gap as the decision-gate evidence",
            "or a prospective week priced twice (calibration off, then on) from the "
            "same inputs, which needs no historical fixture at all",
        ]
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        pf = report["input_preflight"]
        print(f"assembled {report['assembly']['placed']} files into {root}\n")
        print("PRE-FLIGHT FAILED -- the production chain cannot be reproduced here.")
        if pf["missing_files"]:
            print(f"  missing files: {pf['missing_files']}")
        for rel, info in pf["column_incomplete"].items():
            print(f"  {rel}: has {info['columns_present']} columns, missing "
                  f"{info['missing_columns']}")
        print(f"\n  reason: {pf['reason']}")
        print(f"\n  disposition: {report['disposition']}")
        print(f"  consequence: {report['consequence']}")
        print("\n  what would unblock it:")
        for w in report["what_would_unblock"]:
            print(f"    - {w}")
        return 2

    # ---- ARM A: baseline, calibration inert (candidate) ----
    report["baseline_promotion_state"] = set_promotion(root, promoted=False)
    report["baseline_run"] = run_chain(root, "baseline")
    if not report["baseline_run"]["produced"]:
        report["PARITY_ESTABLISHED"] = False
        report["disposition"] = "FAILED_CLOSED_BASELINE_DID_NOT_PRODUCE"
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("disposition",)}, indent=2))
        print("\nstdout tail:\n" + report["baseline_run"]["stdout_tail"][-2000:])
        print("\nstderr tail:\n" + report["baseline_run"]["stderr_tail"][-2000:])
        return 1

    replayed = pd.read_csv(root / "outputs/props_priced_clean.csv", low_memory=False)
    archived = pd.read_csv(BOARD, low_memory=False)
    report["parity"] = parity(replayed, archived)
    report["PARITY_ESTABLISHED"] = report["parity"]["PARITY_ESTABLISHED"]
    report["disposition"] = ("PARITY_ESTABLISHED_PROCEED_TO_TREATMENT"
                             if report["PARITY_ESTABLISHED"]
                             else "FAILED_CLOSED_NO_PARITY_EVIDENCE_REMAINS_COUNTERFACTUAL")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    p = report["parity"]
    print(f"assembled {report['assembly']['placed']} files into {root}")
    print(f"baseline run rc={report['baseline_run']['returncode']} rows={report['baseline_run']['rows']}")
    print(f"\nPARITY vs archived board:")
    print(f"  archived {p['archived_rows']}  replayed {p['replayed_rows']}  joined {p['joined_rows']}"
          f"  (archived-only {p['archived_not_replayed']}, replayed-only {p['replayed_not_archived']})")
    for col, c in p["columns"].items():
        if "match_rate" in c:
            print(f"  {col:14s} {c['matched']}/{c['compared']} = {c['match_rate']:.4f}  "
                  f"max|diff|={c['max_abs_diff']}")
        else:
            print(f"  {col:14s} {c['state']}")
    print(f"\n  PARITY_ESTABLISHED: {report['PARITY_ESTABLISHED']}")
    print(f"  disposition: {report['disposition']}")
    return 0 if report["PARITY_ESTABLISHED"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
