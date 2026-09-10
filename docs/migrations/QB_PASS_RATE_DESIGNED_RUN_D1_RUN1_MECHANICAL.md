# QB Pass-Opportunity Rate Designed-Run D1 — Run 1 Mechanical Disposition

## Disposition

`MECHANICAL_SCHEMA_CASE_MISMATCH_NO_SCIENCE`

Run 1 did not reach scientific evaluation and is preserved as a mechanical/no-decision run.

## Canonical lineage

- Branch: `research-qb-pass-rate-designed-run-d1`
- Frozen plan commit: `d2b512951c45bc8d617ebb5e0fcab451717a17f8`
- Evaluator commit: `7908aed52da9a25070f2d0a012caee8f712d1430`
- Tested/workflow head: `99c2e8abf359f3fc52f60fd7d0ad205cd2411236`
- Run: `34538083332`
- Job: `103074236226`
- Failure step: `Run frozen 2024 development screen`
- Evidence artifact: `10176183646` (`qb-pass-rate-designed-run-d1`)
- Artifact digest: `sha256:07a8a86bdd12be8632938127499b34bb1d5e05dbafb3656ca068ea2aeb7d9bcb`

## Exact failure

The immutable parent `qb_opportunity_chain_casebook.csv` stores the factor columns as:

- `pred_C`
- `pred_S`

The D1 reader requested lowercase `pred_c` and `pred_s` in `pandas.read_csv(usecols=...)` before its later column normalization step, causing:

`ValueError: Usecols do not match columns, columns expected but not found: ['pred_s', 'pred_c']`

## Scientific boundary

- candidate metrics were not calculated;
- no result JSON was created;
- no development disposition was reached;
- no 2025 confirmation metric was scored;
- frozen candidate formula and all pass/fail gates remain unchanged.

## Frozen repair

The only authorized repair is to request the immutable source columns using their exact case (`pred_C`, `pred_S`) and then normalize them to lowercase inside the evaluator as originally intended.

No candidate definition, coefficient, history window, weekly center, metric, threshold, cohort, or stopping rule may change.
