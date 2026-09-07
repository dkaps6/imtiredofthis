# RB-R3 Dynamic Workload Allocation — Result

## Canonical evidence
- Branch: `research-rb-r3-dynamic-workload-allocation`
- Run: `34072849006`
- Job: `101593288171`
- Tested SHA: `5b1cbc3cced96c19c7f34457ca6e84b608d1c27b`
- Artifact: `10001031461` (`rb-r3-dynamic-workload-allocation`)
- Artifact SHA256: `dedc19233cb2e16ac036963edee379272a03d473e39225d6076d1759f1b9f015`
- Sportsbook inputs used: **false**
- Model fitting used: **false**
- Production changed: **false**

## Cohort
- RB-R1 source rows: **1393**
- RB-R1 qualifying profiles: **86**
- CARRIES-dominant / INDIVIDUAL_ALLOCATION players: **17**
- Conditioned rows: **212**

## Results

| Signal | Valid N | Coverage | Spearman | Allocation gap | Carry-residual gap | Underalloc tail enrich | W2-18 gap | W13-18 gap | Within-player positive rate | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PRIOR1_ROOM_CARRY_SHARE | 194 | .915094 | **.136515** | **+1.774976** | **+2.318288** | **1.749407x** | **+1.774976** | **+.609375** | **.117647** | No |
| ROOM_CARRY_SHARE_ACCEL_1V4 | 160 | .754717 | -.178632 | -1.714715 | -1.594734 | .736842x | -1.714715 | -2.103907 | .133333 | No |
| PRIOR1_CARRIES | 194 | .915094 | .095287 | +.705292 | +.422587 | 1.353488x | +.705292 | -.425474 | .117647 | No |
| CARRIES_ACCEL_1V4 | 160 | .754717 | -.096893 | -1.087903 | -1.328649 | .802005x | -1.087903 | -1.831856 | .333333 | No |

## Frozen disposition
**`NO_ACTIONABLE_RB_DYNAMIC_WORKLOAD_ALLOCATION_SIGNAL`**

No exact signal passed every preregistered gate.

## Important pooled-vs-within-player finding
`PRIOR1_ROOM_CARRY_SHARE` passed **9 of 10** frozen gates:
- N/coverage passed;
- Spearman **.136515** passed;
- Q4-Q1 allocation component **+1.775 carries** passed;
- Q4-Q1 carry residual **+2.318 carries** passed;
- under-allocation tail enrichment **1.749x** passed;
- W2-18 and W13-18 were both positive;
- all 17 players had >=6 valid games.

But the required within-player consistency failed dramatically: only **2 of 17 players (11.76%)** had a positive computable within-player association, far below the frozen 60% gate.

This is not a near-miss eligible for rescue. The discrepancy is the scientific result.

## Interpretation
The strong pooled signal is primarily **between-player structural heterogeneity**, not a reusable week-to-week recency effect. Higher-workload backs as a class tend to be the players for whom the model is more likely to under-allocate carries, but a high share in one player's immediately prior game does not reliably imply that the model should increase that same player's next-game allocation.

The acceleration tests reinforce that conclusion. Both room-share acceleration and raw-carry acceleration were negative in aggregate and late-season slices. Therefore simple recency/momentum is not the missing carry-allocation mechanism.

This is directly relevant to the 2026 Week-1 concern: the evidence points away from blindly making the latest depth chart or latest-game usage more authoritative. The next hypothesis should distinguish a player's **stable workload/role class** from short-term fluctuation.

## Authorized next step
No production change and no last-game/acceleration retry. A new experiment may test a stable pregame workload tier / participation role built only from earlier evidence (for example rolling prior workload share and, if timestamp-safe sources are available, offensive snaps / down-and-distance / route or third-down role). It must be frozen separately and evaluated temporally so the full-season mechanism label is not used as a future-information shortcut.
