#!/usr/bin/env python3
from pathlib import Path

src = Path("scripts/backtest/evaluate_qb_first_down_score_state_decomp_v1.py")
dst = Path("scripts/backtest/evaluate_qb_first_down_score_state_decomp_v1_mechfix.py")
text = src.read_text(encoding="utf-8")
old = "            & ~two & ~nop & ~kneel\n"
new = "            & ~two & ~nop\n"
count = text.count(old)
if count != 1:
    raise SystemExit(f"mechanical repair assertion failed: expected 1 kneel-filter boundary, found {count}")
text = text.replace(old, new, 1)
dst.write_text(text, encoding="utf-8")
print("MECHANICAL PARITY REPAIR ONLY: restored parent first-down target universe by retaining kneels; frozen score-state definitions/routing/gates unchanged.")
