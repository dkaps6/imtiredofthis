# RB R26F — 2020 Week 1 Vacancy Failure Forensic Atlas V1 Result

Status: **WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED**
Date: 2026-09-09
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research head: `2ac9633108ef853258107325f27c39e6b819dca2`

## Authoritative lineage

- Workflow: `RB R26F 2020 Week 1 Vacancy Forensic Atlas V1`
- Run: `34365496225`
- Job: `102513015357`
- Artifact: `10109658591`
- Artifact digest: `sha256:bce3f1a81ff516f962109dcadc9ad1e6efa2273c3a96851e94abbe9e4c332022`
- Parent R26 artifact: `10106271075`
- Parent R26E artifact: `10109398212`
- Mechanical conclusion: success
- Scientific disposition: **WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED**
- Child-candidate design authorized: `true`
- Prospective shadow authorized: `false`
- Production promotion authorized: `false`

R26E remains `WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW`.

## Full forensic conclusion

The 2020 Week-1 failure is primarily an **individual allocation failure**, not an RB-room receiving-volume failure.

For 2020 Week-1 vacancy rooms:
- room-total reception absolute-error delta sum: `-0.796046693475847` — the frozen R26 candidate **improved the room total**;
- summed player reception absolute-error delta sum: `+3.3146046536415` — the candidate **worsened who received that volume**.

Frozen classification:

`WITHIN_ROOM_ALLOCATION_DOMINANT`

This is strong evidence to preserve R26's vacancy/pool logic and change only the player-allocation activation state where supported.

## Replicated harmful states

Two predeclared states met the frozen cross-season replication rule.

### 1. Balanced turnover: `EXITS_EQ_ENTRANTS`

2020:
- `n=26`
- receptions MAE worsened about `7.93%`
- mean player absolute-error delta `+0.10594`
- accounted for about `51.22%` of 2020 net worsening

Replication:
- same harmful direction in `2021` and `2023`
- combined supporting `n=25`

Interpretation:

A room where the number of RB/FB exits equals the number of entrants is plausibly a **replacement/churn state**, not necessarily true net-open receiving capacity. Applying the full vacancy redistribution blindly in such a room can over-reallocate receiving work even though the room's total receiving volume is reasonable.

This is the most football-grounded replicated state and is eligible for a separately frozen child-candidate test.

### 2. R9 absolute residual magnitude Q3

Source-only quartile edges for absolute R9 residual were approximately:
- Q1: `0.0014–0.295`
- Q2: `0.295–0.641`
- Q3: `0.641–0.966`
- Q4: `0.966–1.0`

Q3 in 2020:
- `n=11`
- receptions MAE worsened about `20.55%`
- accounted for about `56.27%` of net worsening

Replication:
- same harmful direction in `2021`, `2023`, and `2025`
- combined supporting `n=23`

Interpretation:

This state is statistically replicated but is less football-natural as a first routing rule. Do not build the first child candidate around an arbitrary residual-magnitude quartile when a pregame room-structure mechanism also replicated.

## Material 2020 states that did not replicate sufficiently

Several other states localized 2020 damage but failed the predeclared cross-season replication requirement and therefore may not be used as first-child guards:

- one-or-more entrant
- exits greater than entrants
- one exit
- two-plus exits
- prior depth available
- R9 residual positive
- reception projection increase
- largest reception-movement quartile
- RB1
- no target-leader flip
- target projection increase
- Q3/Q4 target-movement quartiles

These remain diagnostic evidence only.

## Supported components to preserve

Under `RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`, preserve:

1. broad Week-1 vacancy is a useful parent signal;
2. R26's fixed RB receiving pool / conservation mechanics;
3. R9 receiving identity as the parent allocation mechanism outside a separately supported guard state;
4. production-exact non-vacancy rooms;
5. non-RB exactness;
6. receiving-yard mean exactness;
7. R22 authority;
8. sportsbook separation;
9. strict-prior leakage protections;
10. all 17 passing R26E Week-1 gates and their demonstrated improvements.

## Failed / unsupported component

Blindly applying full R26 redistribution to **balanced-turnover rooms** is now a replicated failure mechanism eligible for a child-candidate repair.

The R9 Q3 residual state is also replicated, but it is not selected as the first child delta because it is a less football-grounded, quantile-defined intervention and risks overfitting a diagnostic magnitude bin.

## Next child-candidate delta

Authorize design of **R26G — Week-1 Balanced-Turnover Guard V1**.

First child hypothesis:

- Week 1, non-vacancy room → production baseline exactly.
- Week 1, vacancy room with `room_exits_n != room_entrants_n` → preserve original R26 candidate exactly.
- Week 1, vacancy room with `room_exits_n == room_entrants_n` → fall back the **entire RB room** to the production baseline target/reception entitlement rather than applying R26 redistribution.

Why the entire room falls back:
- R26 is room-conserving;
- the failure is within-room allocation;
- mixing baseline/candidate player-by-player inside the same guarded room could violate finite-room conservation;
- selecting one complete conserved room state preserves structural integrity.

R26G must retain the exact R26E 18-gate Week-1 qualification contract and should add preservation gates ensuring the child does not erase the pooled gains of the original R26 Week-1 component.

No R26G result may retroactively change R26, R26D, R26E, or R26F dispositions.

## Do-not-change list

- R9 identity formula, fit, reliability, and clipping;
- R26 candidate outside balanced-turnover Week-1 rooms;
- R26 pool conservation;
- production baseline definition;
- non-RB entitlement;
- receiving-yard means;
- R22;
- sportsbook separation;
- strict-prior timing contract;
- 2020 negative evidence;
- original frozen gates.
