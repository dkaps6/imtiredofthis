# 2026 Week 4 Production Postmortem — Initial Canonical Findings V1

Status: **CORE FULL-BOARD BACKTEST COMPLETE; FROZEN RESEARCH-LANE CLOSURE IN PROGRESS**  
Date: 2026-10-06  
Research branch: `research-week4-postmortem-execution-v1`

## Canonical Week-4 authority

The postmortem grades the already-frozen recovered Week-4 board:
- recovery run `36935917903`
- artifact `11197776900`
- digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`
- 3,460 published side rows
- 1,730 priced offers
- 873 player-market rows
- pricing status CURRENT

No new OddsAPI acquisition was used.

Canonical successful postmortem execution:
- run `37482840038`
- head `8c6225c4aae9876d6f3220d38424c57685613140`
- all Week-4 source-completeness gates passed
- all 16 games final
- player stats / roster / snap coverage: 32 / 32 teams
- final settlement unresolved rows: 0

## Settlement integrity

The first execution correctly failed closed on four Jadarian Price DraftKings rows rather than treating nonparticipation as zero.

Exact official late-scratch/nonparticipant evidence was then added through the existing exact-row manual-settlement mechanism. Those four rows are VOID, not UNDER wins.

Final Week-4 settlement:
- selected settlement rows: 440
- decided: 427
- voids: 13
- wins/losses: **220-207**
- hit rate: **51.52%**
- units: **-3.38u**
- ROI per unit: **-0.79%**

## Projection quality

- model MAE: **18.89**
- selected market-line MAE: **18.32**
- model closer than selected line: **46.84%**
- model signed bias: **-6.34**
- selected line signed bias: **-3.83**

Week 4 improved hit rate versus the first three weeks, but the model still lost to the selected market line on absolute error and retained a material low-mean bias.

## Week-4 position record

- QB: 27-24, +0.02u, 52.9%
- RB: 78-75, -3.74u, 51.0%
- WR: 78-78, -2.73u, 50.0%
- TE: 36-29, +3.15u, 55.4%

No position result alone is promotion evidence.

## Week-4 market record

- pass_yards: 12-12, -1.36u, 50.0%
- rush_yards: 45-36, +4.06u, 55.6%
- rec_yards: 75-71, -4.31u, 51.4%
- receptions: 71-70, +1.19u, 50.4%
- rush_rec_yards: 17-18, -2.95u, 48.6%

## Calibration

Mean stated fair probability across decided Week-4 bets: **66.48%**
Realized win rate: **51.52%**

70-100% stated-confidence band:
- n=151
- mean stated probability: **79.06%**
- realized: **59.60%**

Overconfidence remains material even though the highest-confidence band performed better than in Weeks 1-3.

## Raw-edge concentration follow-up

The pregame Week-4 note identified a systematic RB rushing/rush+receiving under tilt.

Postgame:
- RB rush_yards UNDER: 18-16, approximately flat units
- RB rush+rec UNDER: 15-12, +1.31u
- RB rec_yards UNDER: 10-9, approximately flat units

The large systematic RB model-market disagreements were therefore **not** a simple collection of independent slam-dunk UNDERs. The negative RB mean bias remained visible.

The largest-edge Week-4 strata did perform well descriptively, including the 20+ native-edge bucket, but the frozen clustered slice analysis tested 42 eligible slices and **zero survived BH-FDR q=.10**.

Do not rescue the terminal-null selector studies with a Week-4 top-N, 20+, or UNDER filter.

## Cumulative Weeks 1-4

Weeks 1-3 were not regraded. The exact frozen Weeks-1-3 graded artifact was combined with the newly graded Week-4 rows.

- W1: 204-205, -20.92u
- W2: 192-198, -22.06u
- W3: 233-208, +2.26u
- W4: 220-207, -3.38u

Cumulative:
- decided: **1,667**
- record: **849-818**
- hit rate: **50.93%**
- units: **-44.10u**
- ROI: **-2.65%**
- model MAE: **17.89**
- selected line MAE: **16.86**
- model closer than line: **46.07%**

Cumulative 70-100% stated-confidence band:
- n=653
- mean stated probability: **80.52%**
- realized: **53.60%**

Cumulative clustered slice analysis:
- 57 eligible slices
- **0 BH-FDR survivors**

Weeks 1-4 still do not authorize a categorical market/position/side/edge carveout.

## Manual user-facing selection lesson

Thursday's four-leg PIT-CLE ticket and Monday's three-leg ATL-NO ticket both hit, but that success must not be confused with selector validation.

The prospectively recorded larger Sunday process-backed pool went only **10-12 (45.5%)**. This is strong evidence against retroactively declaring the subjective “clean football disagreement” curation process a validated selector.

The correct lesson is:
- Thursday/Monday showed individual good reads exist;
- Sunday showed human/model curation still needs a frozen, prospectively testable selection contract before it can be called reliable.

## Frozen research-lane state

Projection-authority move direction:
- Week-4 point estimates are directionally consistent with discovery;
- support remains 1/8 weeks, 111/400 strengthened, 227/400 weakened;
- disposition `FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT`.

GSIS RB successor:
- corrected Week-4 allocation lock is valid;
- required pregame three-arm projection lock was not persisted;
- Week-4 scientific score must fail closed rather than be reconstructed postgame;
- no PASS/FAIL.

RB-PD2:
- no identifiable Week-4 prospective lock/capture exists in GitHub Actions, the active research branches, the recovered Week-4 artifact, or the authorized Library search;
- do not manufacture Observation #2 retrospectively;
- existing HOLD remains unchanged.

## Current conclusion

Week 4 was better than Weeks 1-2 and close to breakeven, but it does **not** solve the project's core betting-selection/calibration problem.

The strongest conclusions remain:
1. raw model-vs-book edge is not validated confidence;
2. stated probabilities remain overconfident;
3. the market line still beats the model on aggregate error;
4. no tested simple slice survives clustered multiple-comparison control;
5. prospective frozen research must continue without outcome-driven rescue.
