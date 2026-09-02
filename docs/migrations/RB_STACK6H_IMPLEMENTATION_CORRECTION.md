# RB STACK6H — Mechanical Implementation Correction

## Failed execution

- Workflow run: `33632461834`
- Job: `100255027362`
- Tested SHA: `c2c23f5a223f7f3a23a14fa9db3338f454b9a044`

The run compiled and downloaded both frozen STACK6 and M94C artifacts successfully. It failed before any oracle score/output was written.

## Failure

The frozen variable for actual total team rush attempts is named column `T`. The first implementation used pandas attribute syntax `t.T`, which is reserved for DataFrame transpose rather than resolving the column. The denominator check therefore attempted to compare an entire mixed-type transposed DataFrame with integer zero and raised `TypeError`.

## Authorized correction

Use explicit bracket access for the already-frozen `T` column everywhere:

```python
t["T"]
```

instead of:

```python
t.T
```

No data definition, population, oracle formula, attribution threshold, or integrity gate changes. No scientific STACK6H outputs were exposed before this repair.