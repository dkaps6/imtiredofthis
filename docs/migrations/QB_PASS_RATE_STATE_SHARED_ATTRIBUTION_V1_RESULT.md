# QB Pass-Rate State Shared Attribution V1 — Result

## Canonical execution

- Branch: `research-qb-pass-rate-state-shared-attribution-v1`
- Frozen plan commit: `857fe90cd7c3954063deeb9de7d341bba37e5b3e`
- Evaluator commit: `461dd07aeb4562221390ffecb7e3d902d47bcb59`
- Canonical execution head: `c5d8b8dc5bcfb2a8a6eaea001d308a424fc4a013`
- Run: `34544457497`
- Job: `103093958683`
- Artifact: `10178499888`
- Artifact name: `qb-pass-rate-state-shared-attribution-v1`
- Digest: `sha256:db401ff605820441605c985bf2262cb6114e1b06c7468605bc6b0a0708c84f4d`
- Disposition: `FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC`
- Production actionable: `false`

## Integrity

All frozen integrity gates passed:

- exact 884 parent rows;
- exact 444/440 season split;
- exact 440/884 shared receiver cohorts;
- no duplicate canonical keys;
- all eight state contributions finite;
- per-state contribution sum reconciled parent `WITHIN_STATE_RATE` within `1e-10`;
- three-group sum reconciled parent `WITHIN_STATE_RATE` within `1e-10`;
- shared receiver joins preserved exact cohort sizes;
- zero sportsbook inputs;
- zero model fitting;
- zero new target-game PBP loading;
- zero production changes.

## Frozen group routing result

Pooled 2024-2025, all 884 games:

| Group | Mean abs contribution | Group absolute-mass share | Dominant row rate |
|---|---:|---:|---:|
| `FIRST_DOWN` | **0.044837** | **44.43%** | **53.17%** |
| `SECOND_DOWN` | 0.034951 | 34.64% | 33.37% |
| `LATE_DOWN` | 0.021121 | 20.93% | 13.46% |

First-down magnitude was essentially identical by season:

- 2024: `0.044827`
- 2025: `0.044846`

Second down remained meaningful but lower:

- 2024: `0.036184`
- 2025: `0.033707`

Late downs were materially smaller:

- 2024: `0.021587`
- 2025: `0.020650`

First down cleared every frozen `SHARED_PRIMARY` gate:

1. largest pooled mean-absolute group contribution — PASS;
2. season stability — PASS;
3. at least 20% larger than second-largest pooled group — PASS;
4. 2025 WR-target absolute Spearman >=0.30 — PASS;
5. pooled WR-reception absolute Spearman >=0.20 — PASS;
6. leave-one-group-out correlation drop >=0.05 in at least one receiver view — PASS.

## Shared receiver linkage

### 2025 WR target-mass residual (n=440)

Parent total within-state propensity:

- Spearman: `0.495654`

Group contributions:

- `FIRST_DOWN`: **`0.419535`**
- `SECOND_DOWN`: `0.389076`
- `LATE_DOWN`: `0.307120`

Leave-one-group-out absolute-Spearman drop:

- remove `FIRST_DOWN`: **`0.069378`**
- remove `SECOND_DOWN`: `0.040275`
- remove `LATE_DOWN`: `0.010890`

### Pooled 2024-2025 WR reception-mass residual (n=884)

Parent total within-state propensity:

- Spearman: `0.379216`

Group contributions:

- `FIRST_DOWN`: `0.313202`
- `SECOND_DOWN`: **`0.333278`**
- `LATE_DOWN`: `0.190769`

Leave-one-group-out absolute-Spearman drop:

- remove `FIRST_DOWN`: `0.040553`
- remove `SECOND_DOWN`: **`0.053426`**
- remove `LATE_DOWN`: `-0.001037`

Interpretation: first and second down both carry shared receiver opportunity information, but first down is the only group that satisfies the complete frozen primary-routing contract. Late downs are not the main shared opportunity problem.

## Tail behavior

For absolute fixed-0.57 rate misses >=0.08 (n=401 pooled):

- first-down mass share: `45.89%`;
- second-down mass share: `35.50%`;
- late-down mass share: `18.61%`;
- first-down dominant-row rate: `57.86%`.

For absolute fixed-0.57 rate misses >=0.12 (n=232 pooled):

- first-down mass share: `47.27%`;
- second-down mass share: `35.06%`;
- late-down mass share: `17.67%`;
- first-down dominant-row rate: `61.64%`;
- first-down sign agreement with the fixed-0.57 residual: `97.41%`.

The first-down mechanism becomes more dominant as the pass-rate miss becomes more extreme.

## Individual state detail

2025 WR target-mass Spearman by individual state contribution:

- `D1`: **`0.419535`**
- `D2_LONG`: `0.301871`
- `D3_LONG`: `0.251129`
- `D2_SHORT`: `0.231982`
- `D2_MEDIUM`: `0.231263`
- `D3_SHORT`: `0.189955`
- `D3_MEDIUM`: `0.076552`
- `D4`: `0.069032`

Pooled WR reception-mass Spearman:

- `D1`: **`0.313202`**
- `D2_LONG`: `0.253416`
- `D2_MEDIUM`: `0.226353`
- `D2_SHORT`: `0.170049`
- `D3_SHORT`: `0.164894`
- `D3_LONG`: `0.115974`
- `D3_MEDIUM`: `0.029689`
- `D4`: `0.027853`

## Scientific interpretation

The shared opportunity chain is now localized as:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION`

Second down remains a material secondary channel, particularly for receiver reception mass, but first down is the primary shared mechanism under the frozen routing rule.

This is not evidence to add a first-down correction blindly. It identifies the physical question the next pregame source must answer:

> What information available before kickoff tells us that a team will choose pass-origin plays on first down more or less often than its strictly-prior offense/opponent-defense expectation?

## Next-work boundary

A subsequent source audit must target genuinely new first-down play-selection intent or incentive information.

Do not reopen:

- generic team pass-rate / PROE history;
- M56 static defense/pass-funnel context;
- M64/M65 score-state, pace, possession or occupancy models;
- M67 situational DBR / formation / no-huddle / shotgun;
- M68 opening script / playcaller / leverage;
- M77-M79 generic personnel/inactive corrections;
- M81 tactical call history;
- M83 adaptive defensive gameplan;
- M87/M88 pass-funnel threshold regimes;
- failed 0.59 anchor calibration.

Sportsbook remains downstream only.
