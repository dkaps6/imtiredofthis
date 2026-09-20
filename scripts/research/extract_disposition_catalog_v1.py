#!/usr/bin/env python3
"""Extract every research disposition in the repo into one reviewable table.

Motivation: dispositions are currently free-text strings scattered across ~70
research/migration docs, hundreds of Issue #535 comments and ephemeral CI
artifacts. Nobody can answer "what has actually helped" by querying anything,
which is how a positive finding like RB_PLAYER_ERROR_PERSISTENCE_DETECTED can
sit unintegrated while adjacent lanes keep getting re-litigated.

This script does extraction only -- it finds disposition tokens and their
surrounding context and emits them for classification. It deliberately does NOT
classify, because deciding whether a lane failed on its own target or on an
unrelated secondary gate requires reading the plan's hypothesis, and a regex
that guessed at that would produce confident garbage.

Read-only. Touches no model, projection, pricing or production path.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# Disposition tokens in this repo are SCREAMING_SNAKE_CASE and tend to end in a
# verdict word, but plenty do not (e.g. M87_REGIMES_NOT_REPLICATED), so match
# the shape and let the reviewer filter rather than over-constraining here.
# The first segment must allow two characters: most dispositions here are
# position-prefixed (RB_/QB_/WR_/TE_), and requiring three silently drops them.
TOKEN = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,})\b")
MIN_TOKEN_LEN = 12

VERDICT_HINTS = (
    "NOT_ACTIONABLE", "FAILED", "FAIL", "NOT_QUALIFIED", "NOT_REPLICATED",
    "CLOSED", "QUALIFIED", "SUPPORTED", "DETECTED", "PASS", "MIXED",
    "NO_INTEGRATION", "READY", "ELIGIBLE", "MAPPED", "PROSPECTIVE_ONLY",
    "INSUFFICIENT", "HOLD", "DISCREPANCY", "NOT_PRODUCTION",
)

# Tokens that are artifact/column/contract names, not verdicts.
NOISE_PREFIXES = ("GITHUB_", "PYTHON", "PIP_", "LD_", "PKG_")


def looks_like_disposition(token: str) -> bool:
    if token.startswith(NOISE_PREFIXES) or len(token) < MIN_TOKEN_LEN:
        return False
    return any(h in token for h in VERDICT_HINTS)


def scan(paths: list[Path], context: int) -> list[dict]:
    rows = []
    for path in sorted(paths):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            for token in TOKEN.findall(line):
                if not looks_like_disposition(token):
                    continue
                lo = max(0, i - context)
                hi = min(len(lines), i + context + 1)
                blob = " ".join(x.strip() for x in lines[lo:hi])
                rows.append({
                    "disposition": token,
                    "file": str(path),
                    "line": i + 1,
                    "context": blob[:900],
                })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=["docs"])
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    paths: list[Path] = []
    for root in a.roots:
        paths.extend(Path(root).rglob("*.md"))

    rows = scan(paths, a.context)

    # One row per (disposition, file): the same token repeated in a doc is one fact.
    seen, deduped = set(), []
    for r in rows:
        key = (r["disposition"], r["file"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["disposition", "file", "line", "context"])
        w.writeheader()
        w.writerows(deduped)

    distinct = sorted({r["disposition"] for r in deduped})
    print(f"docs scanned:        {len(paths)}")
    print(f"disposition rows:    {len(deduped)}")
    print(f"distinct tokens:     {len(distinct)}")
    print(f"written to:          {a.out}")
    print("\n--- distinct dispositions ---")
    for d in distinct:
        print(f"  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
