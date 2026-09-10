# Current Player Availability V1 Run1 — Shallow Checkout Mechanical Repair

Status: `MECHANICAL WORKFLOW FAILURE / FIXTURE SEMANTICS PASSED`

- Run: `34436894543`
- Job: `102743747393`
- Head: `75a02a4f14d9dcce69970130c9cf01138a0cbb90`

The implementation compiled and all 8 frozen availability fixture tests passed (`8 passed in 0.59s`).

The final production-unwired assertion failed before comparing any file because `actions/checkout@v4` used its default shallow `fetch-depth: 1`, so the pinned parent commit `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e` was not present locally:

`fatal: bad object 99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`

Minimum repair: set checkout `fetch-depth: 0`. No availability precedence, role logic, fixture, source contract, production file, or validation criterion changes.
