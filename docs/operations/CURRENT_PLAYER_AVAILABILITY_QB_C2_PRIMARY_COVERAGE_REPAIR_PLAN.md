# Current Player Availability — QB C2 Primary-Coverage Mechanical Repair Plan

Status: `FROZEN BEFORE IMPLEMENTATION / MECHANICAL ONLY / NO GATE RESULT`

## Parent evidence

- 35-gate Run2 head: `1d38953995842aa0236edda7124a79665d0b8628`
- Run2: `34460227422`
- Job: `102816052394`
- Run2 mechanical-failure record commit: `8de180a05dc8561d222db0f0c5e0706534764580`
- Immutable availability candidate artifact: `10140425929`
- Candidate digest: `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- Protected QB C2 source blob before runtime seams: `7b677470b27b6776055c75c924a0ddf22d724a44`
- First QB starter-audit seam transformer blob: `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`

## Exact Run2 failure

Run2 advanced through the first availability-aware QB starter-audit coverage seam and then failed inside `apply_qb_c2_selector()` at the next legacy current-slate assertion:

```python
primary = frame.loc[pd.to_numeric(frame["qb_projection_eligible"], errors="coerce").eq(1)].copy()
if len(primary) != 32 or primary["team"].nunique() != 32:
    raise RuntimeError("QB C2 production adapter did not resolve exactly one primary QB per team")
```

The baseline certification universe is intentionally 30 teams because NE-SEA is already kicked off and withheld by the frozen T-75 contract. Gates 1-34 were never reached.

## Frozen permitted repair

After the first QB starter-audit transformer has been applied, replace ONLY the downstream `primary` current-team coverage assertion above with the already-frozen shared availability-aware validator:

```python
from scripts.utils.eligible_team_set_v1 import validate_current_team_set
validate_current_team_set(primary["team"].astype(str), label="QB C2 primary QB coverage")
```

Semantics:
- legacy / no explicit `ACTIVE_ROLES_CSV`: still requires exactly the full 32-team current slate;
- explicit availability mode: requires exactly the certified eligible team set derived from `ACTIVE_ROLES_CSV`;
- missing or extra teams fail closed.

## Explicitly forbidden changes

This repair MUST NOT alter:
- Ourlads QB depth ranking or ambiguity handling;
- official-team starter-authority priority or identity matching;
- the existing complete 32-team `qb_distribution_state_context.csv` source-integrity assertion;
- C2 selector features, artifact, thresholds, parameters, random seed or generated distributions;
- M89/M90 point-mean authority;
- M38, WR-R15, TE-R5P;
- R26, R22 or P3;
- availability T-75 timing or current-role derivation;
- sportsbook boundaries;
- any of the frozen 35 gate definitions/evaluator/finalizer.

## Regression requirements before 35-gate retry

A dedicated mechanical regression must prove:
1. the protected QB adapter starts at exact protected blob and both frozen runtime transformers apply cleanly;
2. legacy mode accepts exact 32 teams and rejects 30;
3. explicit availability mode accepts the exact certified 30-team set;
4. explicit availability mode rejects one missing team;
5. explicit availability mode rejects one extra team;
6. the complete 32-team state-context source-integrity guard remains byte-visible after transformation;
7. the repaired adapter compiles;
8. no gate evaluator/finalizer executes and no 35-gate disposition is produced.

Only after this regression passes may the 35-gate workflow be relocked with the new transformer blob and retried. Run1 and Run2 remain preserved mechanical failures; the first valid immutable 35-gate result remains unburned.
