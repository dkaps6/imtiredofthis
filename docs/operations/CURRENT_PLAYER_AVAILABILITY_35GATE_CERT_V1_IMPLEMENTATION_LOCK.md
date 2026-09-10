# Current Player Availability — 35-Gate Certification V1 Implementation Lock

Status: `LOCKED_BEFORE_FIRST_35_GATE_EXECUTION`

## Frozen authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Frozen Full Slate integration plan commit: `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- Frozen integration plan blob: `54eb4629c48062fcaef3153918b2069238584d0a`
- Eligible-team seam frozen plan commit: `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- Eligible-team seam regression result commit: `30558db704f81b6336514024e3ed0c5c75291fcc`
- Seam regression run/job/artifact: `34453027002` / `102792905910` / `10142304020`
- Seam regression digest: `sha256:071f791c916d5c17c655b62d6858ca2adfbfd276009878281898608c2d9d3cc0`

## Immutable candidate input

The certification MUST consume the already-successful sportsbook-independent candidate and MUST NOT regenerate the live roster/availability snapshot:

- candidate branch/head: `ops-current-player-availability-full-slate-v1` / `9800254f3ab208ab42501faac83a0d6e5fe3b93d`
- run: `34447900206`
- job: `102776660124`
- artifact: `10140425929`
- artifact name: `current-player-availability-full-slate-candidate-v1`
- digest: `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- frozen live state: 15 eligible games / 30 eligible teams / 437 production-eligible active-role rows / 1 withheld game / sportsbook inputs 0.

## Exact certification implementation blobs

- football-stack runner `scripts/operations/run_current_availability_football_stack_cert_v1.py`: `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`
- fixture builder `scripts/operations/build_current_availability_certification_fixtures_v1.py`: `dbe9f41e4d175b503bf0cd8caf51c1dc8d0da95d`
- gates 1-34 evaluator `scripts/operations/evaluate_current_availability_35gate_v1.py`: `3bc2bb390ab6762f1c20e775152812d3f8e9729e`
- gate-35 finalizer `scripts/operations/finalize_current_availability_35gate_v1.py`: `a5302186eadb748863c70b175421d064885dde60`
- eligible-team helper `scripts/utils/eligible_team_set_v1.py`: `77b591e431378ec984c51e8a032262e673d4c843`
- eligible-team seam transformer `scripts/operations/apply_current_availability_eligible_team_seam_v1.py`: `b64ec5ccd59728121a250433e40e77e3e1013a05`
- certification workflow `.github/workflows/ops-current-player-availability-35gate-cert-v1.yml`: `461c00c6ac5fdc7a289ebcedfb274ba717895be2`

Protected source anchors transformed only ephemerally inside the certification workspace:
- full-universe source blob `f8429ea5b6dd730f054460493facde4ab21b0998`
- R26 source blob `0c7528a3ca9e750d3b9ef2f08ef9721949b3e7fc`

## Execution semantics

The baseline certification uses synthetic football-only lookup market rows with canonical game/player identities and NO line, odds, book, market probability, sportsbook event identity or market-derived win probability. These lookup rows exist only to exercise the protected full-roster football wrapper without live pricing.

The real promoted football path is preserved as implemented by production code:
- M38 finite target entitlement;
- TE-R5P inside the conserved TE room;
- WR-R15 inside the conserved WR2+ room with M38 WR1 anchor preserved;
- QB C2 distribution selector;
- R22 RB receiving-yard tail distribution;
- R26 RB receptions refinement;
- outer P3 rush+receiving conservation.

The eligible-team seam may change only current-slate coverage validation from legacy 32-team expectation to the exact certified 30-team set. No model coefficient, trained asset, vacancy definition, R22/R26 mechanism, target entitlement mathematics, QB selector parameters, or sportsbook boundary may change.

## Frozen fixtures

The first execution also must run three isolated fixture candidates from copies of the immutable candidate evidence:
1. RB1 definitive OUT: old RB1 absent/zero, prior eligible successor becomes RB1 and receives positive current P3 opportunity.
2. QB1 inactive: old QB1 absent/zero, exactly one eligible successor QB1 survives in reconciled roles and PlayerForm/current football universe.
3. WR2+/TE1 unavailable: removed players have zero entitlement and the existing TE-R5P and WR-R15 room/team conservation invariants still pass.

Fixture mutations are never written into the baseline candidate evidence and are executed in isolated worktrees.

## Result immutability and mechanical-failure policy

- The 35 gates are exactly those frozen in the integration plan; no threshold, definition or exception may be changed after seeing a valid result.
- Gates 1-34 are evaluated before evidence upload.
- Gate 35 is finalized only from the actual uploaded evidence artifact ID/digest plus branch/head/run/job lineage.
- The first scientifically/integration-valid 35-gate result is immutable.
- A failure caused before valid gate evaluation by path/import/schema/workflow/plumbing defects is `MECHANICAL_FAILURE_NO_DECISION`; preserve it and repair only the minimum value-neutral defect before retrying.
- A valid result with any failed frozen gate is `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_FAIL_NO_PROMOTION` and may not be retuned, subset-routed or partially promoted.
- Production promotion is authorized only by exact `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION` with 35/35 gates, followed by a separately frozen production-promotion implementation and post-promotion Full Slate verification.

## Production boundary

This lock and its certification workflow do not mutate protected production. Sportsbook inputs used to define availability, role, opportunity or football simulation remain exactly zero. Historical research artifacts remain immutable.
