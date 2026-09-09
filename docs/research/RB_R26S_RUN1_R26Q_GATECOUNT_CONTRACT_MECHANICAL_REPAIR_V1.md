# R26S Run 1 — R26Q Gate-Count Contract Mechanical Repair V1

Status: **FROZEN BEFORE REPAIR IMPLEMENTATION / RERUN**

## Preserved first R26S execution

- run: `34405689393`
- job: `102647981880`
- head: `51806c896fdd79f1d1f9a56f7e3bfa3b0ce4b065`
- artifact: `10125223220`
- artifact digest: `sha256:6130a64872739436a0626ee45eb249a06edb3281a8f750755c8b4483af6aa9bd`
- emitted disposition: `R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_FAIL_NO_PROMOTION`
- emitted gate count: 22
- emitted pass count: 12
- scientific evaluation ready: false
- primary evaluable rows: 0

## Why this is mechanical rather than scientific

The first execution occurred with no valid Week 1 outcome/participation dataset:
- nflverse weekly 2026 stats URL returned 404 / unavailable;
- `nflreadpy.load_snap_counts(seasons=[2026])` reported current snap-count support only through 2025;
- all scientific accuracy gates 11–16 were explicitly `NOT_EVALUATED_SOURCE_OR_SAMPLE_INCOMPLETE`.

The evaluator nevertheless emitted FAIL because its internal `structural_parent_ok` check expected these two keys inside `r26q_disposition.json`:
- `gate_count == 28`
- `gate_pass_count == 28`

Canonical R26Q does **not** store those fields in its disposition JSON. Canonical R26Q stores the 28/28 evidence in `r26q_gate_matrix.csv`, while `r26q_disposition.json` separately stores `all_seal_gates_pass: true` and the exact PASS disposition.

The R26S gate matrix itself already proved:
- `01_exact_r26q_parent_contract = true`
- 28 R26Q gate-matrix rows were present and all passed;
- exact R26Q disposition was present;
- sealed 107 arrays and hashes verified exactly.

Therefore the first R26S FAIL is a **metadata-contract wiring defect before scientific evidence**, not evidence against the R26 candidate.

## Authorized repair only

Do **not** modify the frozen R26S scientific plan, evaluator, thresholds, cohorts, bootstrap, or parent artifacts.

Authorize only a compatibility staging wrapper that:
1. copies the exact downloaded R26Q artifact to an isolated staged directory;
2. verifies the original exact R26Q PASS disposition;
3. verifies `all_seal_gates_pass == true`;
4. verifies the original `r26q_gate_matrix.csv` contains exactly 28 rows and all 28 pass;
5. writes `gate_count: 28` and `gate_pass_count: 28` into the **staged compatibility copy only** of `r26q_disposition.json`;
6. changes no other JSON field or sealed football file;
7. records original/staged disposition hashes and a field-level audit;
8. runs the original locked R26S evaluator byte-identically against the staged compatibility copy.

The authoritative R26Q artifact remains unchanged and must still be verified by exact artifact ID, digest, and head before staging.

## Scientific rules remain frozen

No changes are authorized to:
- R26Q candidate values or arrays;
- R26R market snapshot;
- primary population or `>=50` evaluability threshold;
- actual-reception MAE gate;
- 10,000 team-cluster bootstrap / seed 42;
- `-0.05` lower-CI non-inferiority threshold;
- within-room share-MAE gate;
- `>=0.50 receptions` large-mover threshold;
- role-cohort 10% worsening cap;
- DNP/zero-snap handling;
- sportsbook downstream-only boundary;
- production authority ceiling.

## Expected repaired behavior with no outcome data

If the same authoritative outcome sources remain unavailable and all immutable parent checks pass, the unchanged evaluator should emit:

`R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION`

That is the frozen scientific contract for insufficient outcome/participation evidence.
