# RB RECEIVING ROOM SHARE IMPACT V1 — W1-W4 RESULT

Date: 2026-10-07  
Branch: `research-rb-receiving-room-share-impact-v1`  
Successful head: `bbd4eb25af62461045d88142dcba2d1f565f9278`  
GitHub Actions run: `37703522415` — **SUCCESS**  
Artifact: `11518826839`  
Artifact digest: `sha256:10a1bf06ad19953b3e07512c5f5197afd7eba50126ec0ac68b5bad1252178448`

## Disposition

`RETROSPECTIVE_RB_RECEIVING_SHARE_IMPACT_CONFIRMED__PROSPECTIVE_VALIDATION_STILL_REQUIRED`

The frozen no-fit RB receiving-room share redistribution improves individual RB
receiving opportunity and receiving-output projections over the completed 2026
Weeks 1-4 retrospective replay.

This answers the retrospective impact question:

> Yes — if this mechanism had been applied during Weeks 1-4, the model's
> individual RB receiving projections would have been closer to the realized
> results on average.

This is **not** clean out-of-sample promotion evidence because Weeks 1-4 were
also used to discover/localize the mechanism. Week 5+ remains the prospective
validation boundary.

## Frozen mechanism

- state: strict-prior `prior_rb_room_share`
- parameters fit: **0**
- threshold search: **none**
- total RB/FB target-entitlement mass preserved
- missing-history RBs preserve current canonical room share
- WR entitlement unchanged
- TE entitlement unchanged
- team target mass unchanged
- rushing arrays and rush-yards point projections hard-locked to baseline
- sportsbook inputs upstream: **false**
- paid OddsAPI: **false**

The transform applied to **127 / 128** team receiving rooms across Weeks 1-4.

Maximum conservation gaps were floating-point only:
- team entitlement: <= ~1.11e-16
- RB entitlement: <= ~5.55e-17
- non-RB entitlement: exactly 0

Rush-yards projection max absolute change: **0.0**

## Individual target results

436 RB/FB player-games.

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Target MAE | 1.3863 | **1.2810** | **-7.60%** |
| Target median AE | 1.1485 | **1.0443** | improved |
| Target RMSE | 1.9008 | **1.8000** | improved |

Player-by-player:
- candidate closer: **242**
- baseline closer: **193**
- ties: **1**

### Per-week target MAE

- Week 1: 1.3946 -> **1.2855** (**7.82% better**)
- Week 2: 1.2284 -> **1.1024** (**10.26% better**)
- Week 3: 1.4025 -> **1.3258** (**5.47% better**)
- Week 4: 1.5213 -> **1.4118** (**7.20% better**)

The target mechanism improves MAE in **all four completed weeks**.

## Final receiving-yards projections

436 individual RB/FB player-games.

- baseline MAE: **10.7223 yards**
- candidate MAE: **10.4121 yards**
- improvement: **0.3102 yards / 2.89%**

Player-by-player:
- candidate closer: **251**
- baseline closer: **185**
- ties: 0

Per week:
- W1: 11.0500 -> **10.5831** (**4.23% better**)
- W2: 10.1645 -> **9.7986** (**3.60% better**)
- W3: 10.6231 -> **10.4688** (**1.45% better**)
- W4: 11.0636 -> **10.8062** (**2.33% better**)

Receiving-yard MAE improves in **all four weeks**.

## Final receptions projections

436 individual RB/FB player-games.

- baseline MAE: **1.1906 receptions**
- candidate MAE: **1.1451**
- improvement: **3.83%**

Player-by-player:
- candidate closer: **247**
- baseline closer: **189**

Per week:
- W1: **4.76% better**
- W2: **6.83% better**
- W3: **1.49% better**
- W4: **2.49% better**

Receptions MAE improves in **all four weeks**.

## Final rush+receiving-yard projections

436 player-games.

- baseline MAE: **24.0169**
- candidate MAE: **23.6176**
- improvement: **1.66%**

Player-by-player:
- candidate closer: **248**
- baseline closer: **188**

Per-week MAE:
- W1: **2.82% better**
- W2: **2.04% better**
- W3: **1.26% better**
- W4: **0.43% better**

The total-yard improvement is smaller because the rushing component is
deliberately unchanged.

## Rushing control

RB rush-yards projections are exact baseline controls:

- 436 / 436 ties
- baseline MAE: 18.9067
- candidate MAE: 18.9067
- max point projection difference: **0.0**

This certifies that the receiving-share mechanism did not accidentally improve
or harm rushing through RNG drift.

## High receiving-workload players

For player-games with **6+ realized targets** (30 rows), the benefit is larger.

### Targets
- baseline MAE: 5.0094
- candidate MAE: **4.4826**
- improvement: **10.52%**
- baseline bias: -5.0094
- candidate bias: **-4.4826**
- candidate closer: 23
- baseline closer: 7

### Receiving yards
- baseline MAE: 26.2954
- candidate MAE: **24.9492**
- improvement: **5.12%**
- baseline bias: -25.0702
- candidate bias: **-22.9994**
- candidate closer: 21
- baseline closer: 9

### Receptions
- baseline MAE: 3.4231
- candidate MAE: **3.2156**
- improvement: **6.06%**
- candidate closer: 22
- baseline closer: 8

This is directionally aligned with the original problem: focal receiving backs
were being under-allocated, and the room-share redistribution moves them closer.

## Interpretation

The mechanism is not merely improving an intermediate room-share statistic.

It propagates into **actual individual-player projection improvement**:

1. target allocations move closer;
2. reception projections move closer;
3. receiving-yard projections move closer;
4. rush+receiving projections move closer;
5. rushing remains exactly unchanged.

The magnitude is modest at the final yardage layer because:
- receiving share is only one component of receiving yards;
- the ensemble still retains unchanged ML/state components;
- efficiency is intentionally untouched;
- total team and RB receiving pools are intentionally conserved.

That is the desired behavior for a narrow player-individualization mechanism.

## Why prospective Week-5 shadow still matters

Weeks 1-4 are useful and were used here.

But they served two roles before this result:
- they revealed the workload-compression problem;
- they helped identify RB room-share history as the promising mechanism family.

Therefore Weeks 1-4 cannot honestly be called untouched validation data.

The correct interpretation is:

- **W1-4:** retrospective evidence that the mechanism would have improved real
  individual-player projections;
- **W5+:** prospective evidence that the same frozen rule continues to work
  without changing it after seeing outcomes.

The Week-5 prospective lock is therefore a confirmation boundary, not a reason
to ignore the four weeks of information we already have.

## Next action

Freeze the exact same mechanism prospectively for Week 5 before outcomes:

`RB_RECEIVING_ROOM_SHARE_SHADOW_V1_WEEK5_LOCK`

No changes to:
- history field
- redistribution formula
- fallback behavior
- position scope
- conservation rules

Then evaluate Week 5 only after results settle.

No automatic production promotion yet.
