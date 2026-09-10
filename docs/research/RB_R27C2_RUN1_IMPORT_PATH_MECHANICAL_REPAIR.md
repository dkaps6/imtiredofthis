# RB R27C2 Run1 — Import-Path Mechanical Repair

Status: `MECHANICAL FAILURE ONLY / NO FORENSIC CONCLUSION`

## Failed run preserved

- Workflow run: `34431105624`
- Job: `102726619356`
- Head / implementation lock: `aa9406750e23efa77f8186c3054512d55a7ff316`
- Frozen plan blob: `57e26c75f8f551c28c86d49c55375ef4f590b6e9`
- Frozen diagnostic script blob: `179cabe788afc9d995a827b597ba269fb7dcdded`
- Frozen workflow blob at Run1: `cc1b20cd04a7134153e9ba9fe6be8e685cbe3fc0`

All pre-execution integrity checks passed before failure:
- frozen plan hash PASS
- diagnostic code hash PASS
- protected production boundary PASS
- exact R27B V2 artifact ID/digest PASS
- exact R27C artifact ID/digest PASS

The diagnostic script then failed immediately on import before any PBP reconstruction, cohort calculation, or forensic output:

`ModuleNotFoundError: No module named 'scripts'`

The failing invocation executed the file path directly without adding repository root to `PYTHONPATH`.

## Minimum value-neutral repair

Change only the workflow invocation from:

`python scripts/backtest/forensic_rb_r27c2_realized_target_quality_v1.py ...`

to:

`PYTHONPATH=. python scripts/backtest/forensic_rb_r27c2_realized_target_quality_v1.py ...`

No diagnostic code, cohort, feature, metric, threshold, parent artifact, scientific question, or production file may change.

Run1 remains a preserved mechanical failure and carries no scientific/forensic interpretation.
