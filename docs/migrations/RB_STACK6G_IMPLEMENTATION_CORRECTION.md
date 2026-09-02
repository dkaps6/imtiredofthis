# RB STACK6G — Mechanical Implementation Correction

## Status

Mechanical correction only. The frozen STACK6G hypotheses, variables, historical scope, failure bins, support thresholds, integrity rules, and dispositions are unchanged.

## Failed execution

- Workflow: `RB STACK6G Regime Change Source Audit`
- Run: `33631495970`
- Job: `100251786735`
- Tested SHA: `bc6d5c694f50b723d2da6aa55d54b641c57aaafc`

The run compiled successfully and downloaded the frozen STACK6/P3 casebook successfully. It then failed before any STACK6G scientific output was written.

Traceback location:

`playcaller_table()` attempted:

```python
int(changed or (caller and tenure in (2, 3) and prev == caller))
```

For seasons with no M68 playcaller mapping, `caller == ""`; Python's boolean expression therefore returned the empty string rather than a boolean, and `int("")` raised `ValueError`.

## Authorized correction

Coerce the expression to boolean before integer conversion:

```python
int(bool(changed or (bool(caller) and tenure in (2, 3) and prev == caller)))
```

This changes no football definition. It only makes the already-frozen `playcaller_recent_change` flag evaluate to `0` when no playcaller mapping exists.

## Frozen-contract statement

No outcome metric, source-coverage result, 2025 forensic separation, or support-gate result was exposed before this correction. No threshold, feature, hypothesis, season rule, or model logic may change in the corrected rerun.