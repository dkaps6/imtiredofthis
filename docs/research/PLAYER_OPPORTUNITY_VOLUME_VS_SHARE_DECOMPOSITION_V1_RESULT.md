# PLAYER OPPORTUNITY VOLUME VS SHARE DECOMPOSITION V1 — RESULT

Date: 2026-10-07  
Branch: `research-player-opportunity-volume-vs-share-decomposition-v1`  
Successful head: `3c557e846e731acb403e2df81255ec8cff403304`  
GitHub Actions run: `37694474836` — **SUCCESS**  
Artifact: `11515041385`  
Artifact digest: `sha256:ea989763090af01230c94c76b63e7648f5afaa40e3a42f00b929ff56d360e16c`

## Disposition

`QB_TEAM_VOLUME_DOMINANT__RB_WR_TE_PLAYER_SHARE_DOMINANT`

This decomposition provides a clean mechanistic split after historical
availability parity:

- **QB pass attempts:** team-volume error is the dominant source.
- **RB carries:** player-share allocation error is dominant.
- **RB targets:** player-share allocation error is dominant.
- **TE targets:** player-share allocation error is dominant.
- **WR targets:** player-share allocation error is dominant.

No production change is authorized by this result.

## Mechanical validity

Run `37694474836` passed:

- decomposition unit tests;
- frozen ACT-only parent artifact recovery;
- volume/share decomposition;
- certification boundary;
- strict repository audit.

Boundary:
- rows: **2,106**
- unique player-weeks: **1,670**
- parameters fit: **0**
- automatic promotion: **false**
- sportsbook inputs upstream: **false**
- paid OddsAPI: **false**
- max deterministic expectation parity gap: **7.11e-15**
- max sampled-allocation parity gap: **0.0895**
- max full identity gap: **1.78e-15**

The comparator uses exact simulator-matched team-volume semantics:
- QB official pass attempts;
- target allocation on dropback-side team volume;
- carry allocation on canonical non-dropback team volume.

## Core result

| Position | Opportunity | Model MAE | MAE with actual team volume | MAE with actual player share | Error removed by team volume | Error removed by player share |
|---|---|---:|---:|---:|---:|---:|
| QB | pass attempts | 8.235 | **3.269** | 6.199 | **60.3%** | 24.7% |
| RB | carries | 4.301 | 4.098 | **1.525** | 4.7% | **64.5%** |
| RB | targets | 1.386 | 1.347 | **0.325** | 2.8% | **76.5%** |
| TE | targets | 1.795 | 1.786 | **0.412** | 0.5% | **77.1%** |
| WR | targets | 2.062 | 1.986 | **0.648** | 3.7% | **68.6%** |

The split is not subtle.

## QB — team-volume dominant

Model pass-attempt MAE: **8.235**

Supplying realized team pass volume while preserving the model QB share reduces
MAE to **3.269**, removing **60.3%** of error.

Supplying realized QB share while preserving modeled team volume reduces MAE
only to **6.199**, removing **24.7%**.

Additional diagnostics:
- predicted team official pass attempts mean: **29.45**
- realized team official pass attempts mean: **32.94**
- team-volume MAE: **6.67 attempts**
- predicted QB share mean: **0.961**
- realized QB share mean: **0.922**
- share-error vs individual opportunity-error Pearson: **0.681**
- team-volume-error vs individual opportunity-error Pearson: **0.689**

### High-volume QB games

For QB games with 41+ realized attempts:

- model bias: **-17.62 attempts**
- with actual team volume: **-0.57**
- with actual QB share: **-17.26**

This is decisive: the focal high-attempt QB miss is overwhelmingly a **team
pass-volume** miss, not a player-share miss.

Do not reopen generic QB efficiency/mean science from this result.

## RB carries — player-share dominant

Model carry MAE: **4.301**

- actual team rush/non-dropback volume diagnostic: **4.098 MAE**
  - removes only **4.7%**
- actual RB carry share diagnostic: **1.525 MAE**
  - removes **64.5%**

Diagnostics:
- share-error vs opportunity-error Pearson: **0.879**
- team-volume-error vs opportunity-error Pearson: **0.322**

### Focal RB rushing games

15+ realized carries:

- model bias: **-9.86 carries**
- actual team-volume diagnostic bias: **-8.85**
- actual player-share diagnostic bias: **-1.49**

The dominant issue is who gets the carries, not the team's total rushing
opportunity.

## RB targets — player-share dominant

Model target MAE: **1.386**

- actual team dropback volume diagnostic: **1.347**
  - removes **2.8%**
- actual RB target share diagnostic: **0.325**
  - removes **76.5%**

Diagnostics:
- share-error vs opportunity-error Pearson: **0.941**
- team-volume-error vs opportunity-error Pearson: **0.288**

9+ realized targets:

- model bias: **-8.07**
- actual team-volume diagnostic bias: **-7.09**
- actual player-share diagnostic bias: **-2.70**

Again, player share dominates.

## TE targets — player-share dominant

Model target MAE: **1.795**

- actual team dropback volume diagnostic: **1.786**
  - removes **0.5%**
- actual TE target share diagnostic: **0.412**
  - removes **77.1%**

Diagnostics:
- share-error vs opportunity-error Pearson: **0.930**
- team-volume-error vs opportunity-error Pearson: **0.213**

9+ realized targets:

- model bias: **-6.63**
- actual team-volume diagnostic bias: **-5.36**
- actual player-share diagnostic bias: **-2.18**

The remaining TE opportunity issue is almost entirely individual target-share
allocation.

## WR targets — player-share dominant

Model target MAE: **2.062**

- actual team dropback volume diagnostic: **1.986**
  - removes **3.7%**
- actual WR target share diagnostic: **0.648**
  - removes **68.6%**

Diagnostics:
- share-error vs opportunity-error Pearson: **0.908**
- team-volume-error vs opportunity-error Pearson: **0.289**

9+ realized targets:

- model bias: **-4.77**
- actual team-volume diagnostic bias: **-3.46**
- actual player-share diagnostic bias: **-1.71**

The residual WR problem is individual target-share state.

## Scientific interpretation

The all-player replay's workload compression is not one generic model problem.

It separates cleanly:

### QB
The model is not generating enough team passing opportunity in high-volume games.
The next QB question, if legally novel relative to existing closed QB work, is a
**pregame team pass-attempt/dropback volume state** question.

This does not authorize reopening QB YPA/efficiency mean science.

### RB / WR / TE
The model's team opportunity totals are not the main failure.

The dominant issue is **individual player allocation within the team**:
- carry share for RB rushing;
- target share for RB receiving;
- target share for WR;
- target share for TE.

That is exactly the player-centric mechanism family currently being pursued.

## Reconciliation with already-frozen prospective player state

Before opening any new player-share science, reconcile this result against the
already-frozen prospective mechanisms:

### WR / TE
`PLAYER_TARGET_SHARE_TRAJECTORY_V1` is historically CONFIRMED and the Week-5
prospective shadow is already frozen.

Its exact rule is not legally eligible in Weeks 1-4 because it requires four
prior same-season team games. Do not weaken that eligibility after seeing these
outcomes.

This decomposition strengthens the rationale for evaluating that Week-5
trajectory shadow; it does not authorize a retrospective substitute.

### RB
`RB_PLAYER_STATE_ALLOCATION_SHADOW_V1` is already frozen prospectively for
Week 5 and directly targets individual room allocation using strict-prior carry
share / snap participation.

Do not retrofit it into Weeks 1-4.

## Next action

Do **not** open a second WR/TE or RB share model merely because this
decomposition is strong.

First perform a bounded crosswalk:

`PLAYER_OPPORTUNITY_DECOMPOSITION_TO_EXISTING_SHADOWS_V1`

Required questions:

1. Does the frozen Week-5 WR/TE target-share trajectory shadow directly operate
   on the failure mechanism identified here?
2. Does the frozen Week-5 RB player-state allocation shadow directly operate on
   the carry/target allocation mechanism identified here?
3. Are there uncovered player-share mechanisms after accounting for those
   existing shadows?
4. Is QB team pass-volume a genuinely novel open lane, or has an equivalent
   pass-volume mechanism already been tested/closed elsewhere in the repo?

Only after that crosswalk may another scientific candidate be opened.

No paid OddsAPI. No production promotion. No threshold fitting.
