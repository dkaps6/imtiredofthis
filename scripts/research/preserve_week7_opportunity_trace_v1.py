#!/usr/bin/env python3
"""Preserve the pre-sportsbook Week-7 production opportunity trace, durably.

Required by audit response `6428042413`. The explanatory inputs needed to
audit production's own position-prior fallback -- `prior_games`,
`current_games`, `bayes_rush_share`, `bayes_tgt_share`, post-rule opportunity
shares, rush top-five admission, room normalisation and final target
entitlement -- exist only inside the paid Week-7 run artifact `12456298985`
(run `39849371821`, head `4dc178b5`). No run commits them, so when that
artifact expires on 2027-01-20 the ability to audit the fallback route goes
with it. This script extracts them into git while the window is open.

Two hard rules, both from AGENTS.md §3 and both enforced rather than promised:

  1. The artifact also contains the priced sportsbook board. Every
     sportsbook/decision column is EXCLUDED from the durable trace by an
     explicit denylist, and the script fails if one survives into the output.
     The trace is pre-sportsbook explanatory state only.

  2. No outcome, grade or result column is carried either. This is pregame
     football state, not a postgame casebook.

The artifact's internal layout is not assumed. The script scans every CSV it
contains, scores each by how many required explanatory columns it carries, and
reports what was found and what was missing rather than silently emitting a
partial trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

# Identity / role context, and the explanatory fields the audit response named.
_WANTED = (
    # identity & role
    "player", "player_clean_key", "player_display_name", "team", "recent_team",
    "opponent", "position", "pos", "role", "rules_role", "depth_chart_position",
    "season", "week",
    # evidence state
    "prior_games", "current_games", "bayes_evidence_state", "bayes_applied",
    "bayes_position_group",
    # posterior opportunity shares (the values the audit cited)
    "bayes_rush_share", "bayes_tgt_share", "bayes_target_share",
    "rush_share_prior", "tgt_share_prior", "rush_share_current",
    "tgt_share_current", "rush_share", "tgt_share", "target_share",
    # post-rule opportunity
    "rules_applied", "rules_rush_share", "rules_tgt_share",
    # allocator route
    "sim_selected_rush_share", "rush_top5_admitted", "top_n_admitted",
    "rush_allocation_prob", "rush_room_scale", "room_normalization_factor",
    "final_rush_probability", "analytic_expected_carries",
    "target_entitlement", "final_target_entitlement", "expected_targets",
    # provenance / fallback reason
    "source", "fallback_reason", "evidence_state", "provenance",
    "sportsbook_inputs_used",
)

# Columns that must never reach the durable trace. A sportsbook variable
# upstream of a football projection is the one thing §3 forbids outright, and
# the artifact legitimately contains the priced board, so exclusion is active
# rather than assumed.
_DENY_TOKENS = frozenset({
    "odds", "vegas", "line", "price", "edge", "implied", "payout", "clv",
    "book", "bookmaker", "sportsbook", "prob", "fair", "proj", "sd",
    "graded", "result", "hit", "won", "profit", "roi", "stake", "ticket",
})
# Narrow allowlist of names that contain a denied token but are provenance
# attestations or football state rather than market data.
_DENY_EXEMPT = frozenset({
    "sportsbook_inputs_used", "bayes_evidence_state", "evidence_state",
})


def _denied(col: str) -> bool:
    n = str(col).strip().lower()
    if n in _DENY_EXEMPT:
        return False
    return bool(_DENY_TOKENS & set(n.split("_")))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
                "sha256": _sha256(p),
            })
    return rows


def extract(root: Path, week: int) -> tuple[dict, dict]:
    """Return {source_path: trace_df} and a diagnostic dict."""
    wanted = {c.lower() for c in _WANTED}
    found, diag = {}, {"scanned": [], "required_columns_seen": {}}

    for p in sorted(root.rglob("*.csv")):
        try:
            head = pd.read_csv(p, nrows=0)
        except Exception as exc:
            diag["scanned"].append({"path": str(p.relative_to(root)),
                                    "error": f"{type(exc).__name__}: {exc}"})
            continue
        cols = [c for c in head.columns if str(c).strip().lower() in wanted]
        keep = [c for c in cols if not _denied(c)]
        diag["scanned"].append({
            "path": str(p.relative_to(root)),
            "columns": int(len(head.columns)),
            "explanatory_columns_matched": len(keep),
            "matched": sorted(str(c) for c in keep),
        })
        if len(keep) < 3:
            continue
        df = pd.read_csv(p, usecols=keep, low_memory=False)
        if "week" in {str(c).lower() for c in df.columns}:
            wcol = next(c for c in df.columns if str(c).lower() == "week")
            w = pd.to_numeric(df[wcol], errors="coerce")
            if w.notna().any():
                df = df.loc[w.eq(week)]
        if df.empty:
            continue
        found[str(p.relative_to(root))] = df
        for c in keep:
            diag["required_columns_seen"].setdefault(str(c), []).append(
                str(p.relative_to(root)))

    missing = sorted(
        c for c in _WANTED
        if not _denied(c) and c not in diag["required_columns_seen"])
    diag["required_columns_missing_from_artifact"] = missing
    return found, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--week", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-id", default="39849371821")
    ap.add_argument("--artifact-id", default="12456298985")
    ap.add_argument("--source-head", default="4dc178b59da8708dbea978b8e54e753b1e2f75c5")
    # Lets the Full Slate archive reuse this extractor for the forward-looking
    # contract instead of a second implementation of the §3 denylist.
    ap.add_argument("--stem-prefix", default=None,
                    help="output filename prefix; defaults to week<N>")
    a = ap.parse_args()

    if not a.artifact_dir.exists():
        raise RuntimeError(f"artifact dir {a.artifact_dir} not found")

    prefix = a.stem_prefix or f"week{a.week}"
    man = manifest(a.artifact_dir)
    found, diag = extract(a.artifact_dir, a.week)
    if not found:
        raise RuntimeError(
            "no file in the artifact carried the explanatory columns; refusing "
            f"to emit an empty trace. scanned: {json.dumps(diag['scanned'])}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for src, df in found.items():
        leaked = [c for c in df.columns if _denied(c)]
        if leaked:
            raise RuntimeError(
                f"sportsbook/decision columns survived into the trace from "
                f"{src}: {leaked}; §3 forbids a sportsbook variable upstream "
                "of a football projection")
        safe = src.replace("/", "__").replace(".csv", "")
        out = a.out_dir / f"{prefix}_trace__{safe}.csv"
        df.to_csv(out, index=False)
        written.append({"source_in_artifact": src, "durable_file": out.name,
                        "rows": int(len(df)),
                        "columns": sorted(str(c) for c in df.columns)})

    status = {
        "version": "WEEK7_PRODUCTION_OPPORTUNITY_TRACE_V1",
        "authorized_by": "GPT_AUDIT_RESPONSE_6428042413",
        "purpose": ("preserve the pre-sportsbook explanatory state needed to "
                    "audit production's position-prior fallback route before "
                    "artifact 12456298985 expires 2027-01-20"),
        "scientific_boundary": "PRE_SPORTSBOOK_EXPLANATORY_STATE_ONLY",
        "sportsbook_columns_excluded": True,
        "outcomes_inspected": False,
        "source_run_id": a.run_id,
        "source_artifact_id": a.artifact_id,
        "source_head": a.source_head,
        "week": a.week,
        "artifact_manifest": man,
        "artifact_file_count": len(man),
        "durable_files": written,
        "extraction_diagnostics": diag,
    }
    (a.out_dir / f"{prefix}_opportunity_trace_v1_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"=== WEEK {a.week} PRODUCTION OPPORTUNITY TRACE V1 ===")
    print(f"artifact files: {len(man)}")
    print(f"durable trace files written: {len(written)}")
    for w in written:
        print(f"  {w['durable_file']}: {w['rows']} rows, "
              f"{len(w['columns'])} explanatory columns "
              f"(from {w['source_in_artifact']})")
    if diag["required_columns_missing_from_artifact"]:
        print("\nrequested columns NOT present anywhere in the artifact:")
        for c in diag["required_columns_missing_from_artifact"]:
            print("  -", c)
    print("\nsportsbook/decision columns were excluded by denylist, not "
          "assumed absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
