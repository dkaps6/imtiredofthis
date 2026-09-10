# Current Player Availability — 35-Gate Certification V1 Implementation Lock

Status: `RELOCKED_AFTER_RUN1_MECHANICAL_REPAIR / NO_VALID_GATE_RESULT_YET`

## Frozen authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Frozen Full Slate integration plan commit: `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- Frozen integration plan blob: `54eb4629c48062fcaef3153918b2069238584d0a`
- Eligible-team seam frozen plan commit: `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- Eligible-team seam regression result commit: `30558db704f81b6336514024e3ed0c5c75291fcc`
- Seam regression run/job/artifact: `34453027002` / `102792905910` / `10142304020`
- Seam regression digest: `sha256:071f791c916d5c17c655b62d6858ca2adfbfd276009878281898608c2d9d3cc0`

## Immutable candidate input

- candidate branch/head: `ops-current-player-availability-full-slate-v1` / `9800254f3ab208ab42501faac83a0d6e5fe3b93d`
- run/job/artifact: `34447900206` / `102776660124` / `10140425929`
- artifact name: `current-player-availability-full-slate-candidate-v1`
- digest: `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- frozen live state: 15 eligible games / 30 eligible teams / 437 production-eligible active-role rows / 1 withheld game / sportsbook inputs 0.

## Exact certification implementation blobs

- football-stack runner: `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`
- fixture builder: `dbe9f41e4d175b503bf0cd8caf51c1dc8d0da95d`
- gates 1-34 evaluator: `3bc2bb390ab6762f1c20e775152812d3f8e9729e`
- gate-35 finalizer: `a5302186eadb748863c70b175421d064885dde60`
- eligible-team helper: `77b591e431378ec984c51e8a032262e673d4c843`
- full-universe/R26 seam transformer: `b64ec5ccd59728121a250433e40e77e3e1013a05`
- QB C2 eligible-team seam transformer: `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- certification workflow after Run1 repair: `130ed21db8d23575a38e84f1cfe0cc865498f001`

Protected source anchors transformed only ephemerally inside certification workspaces:
- full-universe source: `f8429ea5b6dd730f054460493facde4ab21b0998`
- R26 source: `0c7528a3ca9e750d3b9ef2f08ef9721949b3e7fc`
- QB C2 source: `7b677470b27b6776055c75c924a0ddf22d724a44`

## Preserved Run1 mechanical failure

- original lock head: `eb7f37fa6e2a9019ed0ec8f3bbe6fe73be202699`
- run/job: `34459655725` / `102814178762`
- disposition: `MECHANICAL_FAILURE_NO_DECISION`
- frozen gates evaluated: 0/35
- exact exception: `RuntimeError: QB C2 starter authority must cover 32 teams, got rows=30`
- record: `docs/operations/CURRENT_PLAYER_AVAILABILITY_35GATE_RUN1_MECHANICAL_FAILURE_RECORD.md`

Run1 had already passed immutable candidate staging and had successfully materialized M38 -> TE-R5P -> WR-R15 for the 30-team certified universe before the legacy QB coverage assertion stopped execution.

## Frozen Run1 repair

- repair plan commit: `760fbcafec3df636b671b49c16a6d1d04c134162`
- repair lock commit: `3825235af0442e7868f474e905ea11ad1214432e`
- repair regression run/job: `34460044387` / `102815456345`
- regression conclusion: SUCCESS

The repair replaces only the QB C2 starter-authority current-team coverage check with the already-frozen availability-aware team-set validator. Legacy/no-explicit-availability behavior remains exactly 32 teams; explicit availability behavior requires exact equality to the certified eligible set. The QB C2 32-team state-context source-integrity assertion remains unchanged. Starter selection/authority precedence/Ourlads fallback/C2 parameters and distribution logic are unchanged.

Because Run1 never evaluated gates, the next execution remains eligible to become the first valid immutable 35-gate result.

## Execution semantics

The baseline certification uses synthetic football-only lookup rows containing canonical football identity and market labels but NO book, line, odds, market probability, sportsbook event identity or market-derived win probability. The actual promoted football path is exercised as implemented:
- M38 finite target entitlement;
- TE-R5P;
- WR-R15;
- QB C2;
- R22;
- R26;
- outer P3 rush+receiving conservation.

The eligible-team seams affect current-slate coverage validation only. No scientific coefficient, trained asset, vacancy definition, target-entitlement math, QB selection logic, R22/R26 mechanism or sportsbook boundary may change.

## Frozen fixtures

1. RB1 definitive OUT: old RB1 absent/zero; prior eligible successor becomes RB1 and receives positive P3 opportunity.
2. QB1 inactive: old QB1 absent/zero; exactly one eligible successor QB1 survives roles and PlayerForm/current universe.
3. WR2+/TE1 unavailable: removed players receive zero entitlement and TE-R5P/WR-R15 conservation remains intact.

Fixtures run in isolated worktrees and never overwrite baseline evidence.

## Result immutability

- The 35 gates remain exactly those frozen at plan commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`; none has changed.
- Gates 1-34 are evaluated before evidence upload.
- Gate 35 is finalized only from actual uploaded artifact lineage.
- Pre-evaluator plumbing failure remains `MECHANICAL_FAILURE_NO_DECISION` and permits only minimal value-neutral repair.
- Any valid gate failure is `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_FAIL_NO_PROMOTION` and is immutable.
- Exact 35/35 is required for `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`, after which production still requires a separately frozen promotion implementation and verification.

## Production boundary

This certification does not mutate production. Sportsbook inputs used to define availability, role, opportunity or football simulation remain zero. Historical research remains immutable.
