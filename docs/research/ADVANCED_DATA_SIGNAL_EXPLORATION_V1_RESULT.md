# Advanced Data Signal Exploration V1 — Initial Reconnaissance Result

**Status:** COMPLETED INITIAL RECONNAISSANCE — NO PRODUCTION CHANGE.

**Canonical run:** `35289753485`

**Canonical source SHA:** `199ba5ae4019135923820d1523b503dc71966f67`

All jobs:
- exploration helper tests: PASS
- BDB 2021 route geometry: PASS
- BDB 2023 protection geometry: PASS
- BDB 2026 throw-window geometry: PASS

No sportsbook inputs were used. No production projection was read or modified. No outcome-model tournament was run.

## Artifacts

- BDB 2021: artifact `10525667870`, digest `sha256:974df05465b2c2a42044413952f2309bd717c1eeb7af1da410b0966d6dc9d1ba`
- BDB 2023: artifact `10525797510`, digest `sha256:b6c86b73a73a91214b3eb8c620f9af18f2504569d9465dda8fd5950da8555a94`
- BDB 2026: artifact `10525194365`, digest `sha256:8c2af5622dce039b042ef0c8a6af198b35828a3836f4ee7ef12b049cb23ed527`

# Executive result

The new data are **not merely descriptive**.

Several advanced geometry families demonstrate meaningful temporal persistence and strict-prior ability to anticipate later same-family geometry.

This does not establish that they improve yards/receptions/rushing projections. It establishes that multiple families behave like stable or partially stable football traits rather than pure play-level noise.

## Highest-priority result — BDB 2026 targeted-receiver release geometry

### Player-level early/late persistence

188 targeted receivers qualified with at least 8 observations in each half.

| Feature | Spearman early vs late | Pearson |
|---|---:|---:|
| release nearest-defender distance | **0.705** | 0.881 |
| release second-defender distance | **0.614** | 0.821 |
| defenders within 2 yd count | -0.039 | -0.037 |
| defenders within 3 yd count | **0.322** | 0.322 |

Interpretation:

- nearest-defender release spacing is a strong persistent receiver/environment profile;
- second-defender spacing is also materially persistent;
- raw 2-yard crowding count is not stable as a simple player-level median trait;
- 3-yard crowding has modest persistence.

### Receiver × route persistence

280 receiver×route profiles qualified with at least 5 observations in each half.

| Feature | Spearman early vs late |
|---|---:|
| route-conditioned release nearest-defender distance | **0.730** |
| route-conditioned release second-defender distance | **0.637** |

This is one of the clearest findings in the new frontier work.

Route context materially sharpens receiver-spacing identity.

### Strict-prior history vs later target-week geometry

2,580 receiver-week matches had qualified strict-prior history.

| Prior feature | Later same-family geometry Spearman | MAE |
|---|---:|---:|
| prior median nearest release distance | **0.482** | 1.609 yd |
| prior median second-defender distance | **0.399** | 1.954 yd |
| prior 2-yard crowding rate | **0.384** | 0.209 rate |
| prior 3-yard crowding rate | **0.406** | 0.243 rate |

Route-conditioned strict-prior history:

- 3,361 matched receiver×route target-week rows
- prior route-conditioned nearest release spacing vs later same-route spacing:
  - Spearman: **0.537**
  - Pearson: **0.649**
  - MAE: **1.712 yd**

Target-game route remains unknown pregame. This result therefore validates route-conditioned history as a real trait but does not authorize using realized target-game route.

### Coverage / route structure

The source shows large and coherent geometric differences between man and zone within the same route family.

Examples:

- GO median release nearest:
  - man: **1.33 yd**
  - zone: **1.90 yd**
- CROSS:
  - man: **2.39 yd**
  - zone: **4.56 yd**
- HITCH:
  - man: **2.55 yd**
  - zone: **4.08 yd**
- OUT:
  - man: **2.18 yd**
  - zone: **3.76 yd**
- FLAT:
  - man: **5.00 yd**
  - zone: **8.42 yd**

These are descriptive geometry differences, not coverage-responsibility claims.

### Release to post-release geometry

Retrospective-only validation:

- release nearest distance vs terminal nearest predicted defender:
  - Spearman **0.641**
- release nearest distance vs minimum post-release nearest predicted defender:
  - Spearman **0.771**

This supports the idea that release-window spatial state contains meaningful information about the later throw-window geometry.

It remains retrospective validation, not a pregame outcome result.

### Coverage growth through season

At the frozen 8-observation threshold:

- Week 2 qualified targeted-receiver history: 6.8%
- Week 6: 48.4%
- Week 12: 58.9%
- Week 18: 68.1%
- end-of-season target Week 19 snapshot: 69.2%

This implies a real early-season cold-start problem. Role-regime / partial-pooling logic will be especially important early in the year and for new-team/new-role players.

## BDB 2021 route geometry

### Player × route persistence

1,513 player×route profiles had at least 5 observations in each temporal half.

| Feature | Spearman early vs late |
|---|---:|
| throw nearest-defender distance | **0.593** |
| throw second-defender distance | **0.571** |
| throw first-to-second spacing gap | **0.401** |

### Strict-prior history vs later geometry

16,961 qualifying future comparisons:

| Feature | Spearman overall | Spearman HIGH support |
|---|---:|---:|
| nearest-defender throw spacing | **0.411** | **0.492** |
| second-defender throw spacing | **0.375** | **0.481** |
| first-to-second spacing gap | 0.246 | **0.392** |

Absolute error also improves as historical support increases.

Interpretation:

BDB2021 route geometry contains repeatable route/player structure.

Important limitation:

The source does not semantically identify the targeted receiver, so this lab is best suited to route/archetype/spacing research and analog construction rather than directly claiming targeted-receiver matchup quality.

### Coverage at the 5-observation player×route threshold

- Week 2: 19.1%
- Week 6: 49.8%
- Week 14: 61.6%
- Week 18: 64.7%

Again, cold-start/sparse-route histories are material.

## BDB 2023 protection geometry

### Blocker temporal persistence

246 blockers qualified with at least 10 interactions per temporal half.

| Feature | Spearman early vs late |
|---|---:|
| blocker-target snap distance | **0.847** |
| time to minimum distance | **0.575** |
| minimum interaction distance | **0.487** |

Snap separation is extremely persistent, likely reflecting stable alignment/position/interaction structure.

Time-to-minimum and minimum distance also contain repeatable blocker-level structure.

### Strict-prior history vs later blocker geometry

1,553 qualifying blocker-week comparisons:

| Prior feature | Spearman overall | HIGH-support Spearman | HIGH-support MAE |
|---|---:|---:|---:|
| snap distance | **0.820** | 0.788 | 0.248 yd |
| minimum distance | 0.319 | **0.462** | 0.064 yd |
| time to minimum | **0.436** | **0.454** | 0.171 sec |

History becomes substantially more precise with greater support.

### Retrospective PFF outcome relationship

Single-interaction geometry by itself is not a strong outcome classifier.

Examples:

- beaten interactions have median minimum distance **0.644 yd** vs **0.772 yd** when not beaten;
- hurry-allowed interactions: **0.610 yd** vs **0.774 yd**;
- sack-allowed interactions: **0.626 yd** vs **0.768 yd**.

The point-biserial relationships are small.

Protection-window duration shows some relationship to pressure outcomes:
- hurry allowed median window: **3.5 sec** vs 2.9 sec;
- sack allowed median window: **4.1 sec** vs 2.9 sec.

Interpretation:

The geometry captures meaningful interaction state, but no single raw geometry metric should be treated as a sack/hurry predictor.

Its strongest immediate value is as:
- blocker archetype;
- personnel continuity context;
- protection interaction profile;
- input to a later jointly specified OL/front matchup model.

### Coverage at 10-interaction history threshold

- Week 2: 51.6%
- Week 5: 63.5%
- Week 8: 67.8%
- Week 9 snapshot: 68.6%

## Initial family dispositions

### High priority for a later frozen predictive test

**BDB2026 targeted-receiver release nearest-defender history**
- direct targeted-receiver semantics;
- strong temporal persistence;
- strict-prior future-geometry signal;
- legal pregame historical transform.

**BDB2026 targeted-receiver second-defender spacing**
- materially persistent;
- distinct from nearest distance;
- direct target semantics.

These should eventually be tested against WR/TE opportunity/efficiency under a separately preregistered position-specific experiment.

### Research-worthy / conditional

**BDB2026 route-conditioned spacing**
- very strong persistence;
- strongest receiver geometry result;
- deployment requires a pregame-safe route tendency/scenario mechanism because realized route is unknown.

**BDB2026 3-yard crowding history**
- modest persistence / useful future-rate relationship;
- likely better as context than standalone feature.

**BDB2021 player×route geometry**
- repeatable and useful for route archetypes/analogs;
- semantic target limitation prevents direct target-mismatch claims.

**BDB2023 blocker geometry**
- strong persistent player/archetype information;
- best suited to trench/personnel context and later interaction model;
- not justified as a standalone pressure predictor.

### Lower standalone priority

**BDB2026 raw 2-yard crowding count**
- near-zero simple early/late player persistence;
- prior crowding rate still has moderate future-rate relation, so it should not be discarded, but it is not a first-choice standalone trait.

## Important interaction with the new football-context program

The new advanced-data results strengthen the case for the other program work:

1. geometry histories have cold-start problems;
2. new-team/new-role players will often have stale or unavailable current-regime history;
3. role/environment regime evidence is therefore needed to decide when old player geometry/history should be trusted;
4. historical analogs can provide partial pooling when a current player lacks same-regime samples;
5. personnel continuity can tell us when blocker/receiver historical profiles are more or less representative.

These workstreams are complementary, not competing.

## What this does NOT prove

This run does not prove:
- receiving-yards lift;
- receptions lift;
- QB passing-yards lift;
- rushing-yards lift;
- betting edge;
- true CB responsibility;
- true live blocker-rusher assignment.

## Next allowed advanced-data step

Before any production use:

1. freeze one WR/TE candidate using strict-prior BDB2026 receiver release spacing;
2. separately freeze a trench/personnel candidate using blocker profiles;
3. build route-tendency engineering before route-conditioned spacing can be used pregame;
4. add the advanced history families to the historical matchup-analog state;
5. do not combine all families into one model before individual mechanism tests.

Final disposition:

`ADVANCED_DATA_SIGNAL_EXPLORATION_V1_INITIAL_RECONNAISSANCE_PASS_MULTIPLE_PERSISTENT_FAMILIES`
