# Current Player Availability 35-Gate Run2 Mechanical Failure Record

Status: `MECHANICAL_FAILURE_NO_DECISION`

## Immutable execution

- Branch: `ops-current-player-availability-35gate-cert-v1`
- Head: `1d38953995842aa0236edda7124a79665d0b8628`
- Run: `34460227422`
- Job: `102816052394`
- Immutable candidate artifact: `10140425929`
- Candidate digest: `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`

## Boundary reached

Run2 passed:
- frozen implementation / protected-science verification;
- exact immutable candidate digest verification and staging;
- availability-aware 30-team full-universe / R26 seams;
- first QB C2 starter-audit seam;
- certification fixture construction;
- M38 -> TE-R5P -> WR-R15 entitlement materialization and conservation on 30 teams.

It failed inside baseline football-stack execution before any fixture execution and before `Evaluate frozen gates 1 through 34`.

Therefore zero frozen gates were evaluated and no integration disposition exists.

## Exact exception

GitHub Actions Job `102816052394` ended at `scripts/modeling/qb_c2_production_adapter_v1.py`, `apply_qb_c2_selector()` with:

`RuntimeError: QB C2 production adapter did not resolve exactly one primary QB per team`

The immediately preceding M38 / TE-R5P / WR-R15 audit reported `teams: 30` and all frozen entitlement-conservation flags true.

## Diagnosis

This is the second legacy 32-team current-slate coverage assertion in QB C2. The prior mechanical repair correctly made `annotate_primary_qbs()` starter-audit coverage availability-aware. Run2 then advanced past that point and failed at the downstream `primary` coverage assertion, which still requires exactly 32 primary QBs even when the explicit certified active-role universe contains 30 eligible teams.

The 32-team `qb_distribution_state_context.csv` source-integrity assertion did not fail and remains scientifically appropriate: it validates the complete football context source, not the currently eligible production subset.

## Disposition

`MECHANICAL_FAILURE_NO_DECISION`

The first valid 35-gate result remains unburned. Any repair must be frozen before execution, must affect only current-team coverage validation, must retain legacy 32-team behavior when explicit availability is absent, and must leave QB starter ranking, starter authority priority, state-context source integrity, C2 selector features/parameters/distributions, R26, R22, receiver models and sportsbook boundaries unchanged.
