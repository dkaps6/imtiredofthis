"""Reconcile seven Week-3 zero-snap rows against the canonical postmortem voids.

Requested on issue #535. My Week-3 DK subset priced seven players who recorded
no Week-3 stat line. The canonical Weeks1-3 postmortem reports 20 voids across
1,260 selected rows and states Week-3 settlement closed with zero unresolved
rows. The question is purely one of provenance: were these same seven rows
already voided there, or were they settled as decided bets?

Read-only. Consumes the postmortem's own artifact. No regrade, no refit, no
Week-3 science reopened -- this only reports which bucket each row landed in.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DNP = [
    "Darius Slayton", "Elijah Arroyo", "Xavier Legette", "Adonai Mitchell",
    "Blake Whiteheart", "Erick All", "Devin Singletary",
]
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$")


def _norm(v: str) -> str:
    s = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", SUFFIX.sub("", s.strip())).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    csvs = sorted(args.evidence.rglob("*.csv"))
    print("artifact inventory:")
    for c in csvs:
        print(f"  {c.relative_to(args.evidence)}  ({c.stat().st_size} bytes)")

    targets = {_norm(n): n for n in DNP}
    report = {
        "contract": "WEEK3_DNP_PROVENANCE_RECONCILIATION_V1",
        "regraded_week3": False,
        "refit": False,
        "files_scanned": [str(c.relative_to(args.evidence)) for c in csvs],
        "matches": [],
        "files_with_player_column": [],
    }

    for c in csvs:
        try:
            df = pd.read_csv(c, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {c.name}: {exc}")
            continue
        pcol = next((k for k in ["player", "player_name", "player_display_name",
                                 "player_clean_key", "name"] if k in df.columns), None)
        if pcol is None:
            continue
        report["files_with_player_column"].append(str(c.relative_to(args.evidence)))
        wk = df["week"].astype(str) if "week" in df.columns else None
        norm = df[pcol].map(_norm)
        hit = norm.isin(targets)
        if wk is not None:
            hit &= wk.eq("3")
        if not hit.any():
            continue
        keep = [k for k in ["season", "week", "player", "player_name", "team", "market",
                            "side", "bet_result", "result", "unit_result", "void",
                            "void_reason", "settlement_status", "participation_status",
                            "vegas_line", "model_proj", "actual", "fair_prob"]
                if k in df.columns]
        for _, r in df.loc[hit, keep].iterrows():
            report["matches"].append({"file": str(c.relative_to(args.evidence)),
                                      **{k: (None if pd.isna(r[k]) else str(r[k])) for k in keep}})

    found = {_norm(m.get("player") or m.get("player_name") or "") for m in report["matches"]}
    report["dnp_players_found_in_postmortem"] = sorted(
        targets[f] for f in found if f in targets
    )
    report["dnp_players_absent_from_postmortem"] = sorted(
        n for k, n in targets.items() if k not in found
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
