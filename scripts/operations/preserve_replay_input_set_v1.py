"""Preserve the model-side inputs a pricing replay needs, from a Full Slate artifact.

WHY THIS EXISTS. Four separate validations have now been lost to the same gap:

  1. weeks 3-5 priced boards expired at `retention-days: 7`;
  2. production decision-gate evidence absent for weeks 6-7, so they cannot be
     bet-graded;
  3. the week-7 explanatory trace has `model_ml_diagnostics.csv` and
     `model_state_diagnostics.csv` stripped to six identity columns, so the exact
     production pricing path cannot be replayed at all;
  4. run 40551229604's artifact carries the ML/State columns but no pricing boundary.

The archive and the explanatory trace each hold half of what a replay needs, and
neither holds the other half.

WHAT THIS ARCHIVES, AND WHY IT IS NOT DENYLIST-STRIPPED. The explanatory trace exists
for *research* and correctly strips every sportsbook and decision column (AGENTS.md §3),
which is exactly what destroys the model-side prediction columns. This set exists for
*replay*: to re-run the production pricing chain you need the inputs production actually
had, unmodified. Nothing new is exposed -- the paid boundary is already archived beside
it as `raw_props/` and `game_odds/`.

**`replay_inputs/` is therefore NOT research-safe and must never be used to construct a
projection.** It is a reproduction fixture. The guard below refuses to write a file whose
required columns are missing, so a future stripping regression is caught on the run that
causes it rather than months later when a replay is attempted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

VERSION = "REPLAY_INPUT_SET_V1"

# Each entry: artifact-relative path -> columns that MUST survive for a replay to work.
# The column lists are the ones whose absence actually broke the week-7 parity attempt.
REQUIRED = {
    "data/model_ml_diagnostics.csv": ("ml_available", "ml_method", "ml_pass_yards",
                                      "ml_rush_yards", "ml_rec_yards", "ml_receptions"),
    "data/model_state_diagnostics.csv": ("state_available", "state_method"),
    "data/metrics_ready.csv": ("player_clean_key", "team"),
    "data/model_rule_simulation_inputs.csv": ("player_clean_key", "team"),
    "data/player_form_consensus.csv": ("player_clean_key",),
}
# Useful for a replay but not fatal if a given run did not produce them.
OPTIONAL = (
    "data/model_ensemble_weights.csv",
    "data/model_ensemble_diagnostics.csv",
    "data/weather_week.csv",
    "data/team_week_map.csv",
    "data/qb_promoted_team_context.csv",
    "data/rb_rush_synthesis_context.csv",
)

README = """# replay_inputs — reproduction fixtures, NOT research inputs

Each subdirectory is one slate's model-side pricing inputs, gzipped, exactly as
production had them. They exist so the production pricing chain can be re-run later.

**These files are deliberately NOT denylist-stripped.** Unlike `../explanatory/`, which
is scrubbed of every sportsbook and decision column for research use, a replay needs the
inputs production actually consumed. Do not use anything here to construct a projection.

`manifest.json` in each directory carries a sha256 per file, so a replay can prove it ran
on the same bytes.
"""


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preserve(artifact_dir: Path, out_dir: Path, stem: str, run_id: str,
             git_sha: str = "", week: str = "") -> dict:
    dest = out_dir / stem
    dest.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    entries, missing, stripped = {}, [], {}
    for rel, cols in REQUIRED.items():
        src = artifact_dir / rel
        if not src.exists():
            missing.append(rel)
            continue
        have = set(pd.read_csv(src, nrows=1, low_memory=False).columns)
        absent = [c for c in cols if c not in have]
        if absent:
            # Loud: this is the exact regression that cost the week-7 replay.
            stripped[rel] = {"missing_columns": absent, "columns_present": len(have)}
            continue
        entries[rel] = _write(src, dest)

    for rel in OPTIONAL:
        src = artifact_dir / rel
        if src.exists():
            entries[rel] = _write(src, dest)

    manifest = {
        "version": VERSION, "stem": stem, "week": week,
        "source_run_id": run_id, "source_git_sha": git_sha,
        "files": entries,
        "required_missing": missing,
        "required_column_stripped": stripped,
        "complete": not missing and not stripped,
        "warning": ("NOT denylist-stripped; reproduction fixture only, never a "
                    "projection input"),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return manifest


def _write(src: Path, dest: Path) -> dict:
    out = dest / (src.name + ".gz")
    with src.open("rb") as fi, gzip.open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return {"bytes_raw": src.stat().st_size, "bytes_gz": out.stat().st_size,
            "sha256_raw": _digest(src), "archived_as": out.name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stem", required=True, help="e.g. 2026_wk08")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--week", default="")
    args = ap.parse_args()

    m = preserve(args.artifact_dir, args.out_dir, args.stem, args.run_id,
                 args.git_sha, args.week)
    total = sum(f["bytes_gz"] for f in m["files"].values())
    print(f"{VERSION}: archived {len(m['files'])} files for {args.stem} "
          f"({total/1024:.0f} KB gzipped)")
    if m["required_missing"]:
        print(f"::warning::replay set INCOMPLETE -- absent from artifact: {m['required_missing']}")
    for rel, info in m["required_column_stripped"].items():
        print(f"::warning::{rel} archived NOTHING: it has {info['columns_present']} columns "
              f"and is missing {info['missing_columns']}. A pricing replay of this slate "
              f"will be impossible. This is the week-7 failure repeating.")
    if not m["complete"]:
        # Loud but non-fatal: the paid boundary is the archive's first duty and must
        # never be jeopardised by this enrichment (AGENTS.md §4).
        print("::warning::replay input set is INCOMPLETE for " + args.stem)
        return 0
    print(f"replay set COMPLETE for {args.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
