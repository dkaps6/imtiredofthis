# PLAYER OPPORTUNITY VOLUME VS SHARE DECOMPOSITION V1 — RESULT

Date: 2026-10-07  
Branch: `research-player-opportunity-volume-share-decomposition-v1`  
Successful head: `825c661838737c11027db2581ebbff893b4c8456`  
GitHub Actions run: `37689289742` — **SUCCESS**  
Artifact: `11512980697`  
Artifact digest: `sha256:9844527f59f632ddb52b93eb6f206b0ce64ad37641701300b2ab588f78c6c237`

## Disposition

`MIXED_VOLUME_AND_SHARE__QB_TEAM_VOLUME_DOMINANT__RB_WR_TE_PLAYER_SHARE_DOMINANT`

The ACT-only individual-player decomposition identifies two different residual
mechanisms:

- **QB pass attempts:** team pass-volume error is dominant.
- **RB carries and RB/WR/TE targets:** individual player share/allocation error
  is dominant.

This is exactly the distinction the player-centric replay needed. The remaining
skill-position problem is not primarily generic team volume and not a generic
position mean. It is the amount of finite team opportunity allocated to the
specific player.

No production change is authorized by this result.

## Scientific boundary

Frozen parent:
- historical availability parity run `37687979574`
- ACT-only parent artifact `11511927864`
- parent digest `sha256:6bcbf63f8dfc5b2b0362e16fbd612653d6f813e5c2d9aec79e9d49c98cb0303b`

Decomposition run:
- rows: **2,106**
- unique player-weeks: **1,670**
- Weeks 1-4 only
- no sportsbook input
- no paid OddsAPI
- parameters fit: **0**
- automatic promotion: **false**
- actual team volume loaded only after prediction parent freeze: **true**
- max parent arithmetic gap: **7.11e-15**
- max full-oracle identity gap: **1.78e-15**

## Overall decomposition

### QB — pass attempts

Baseline opportunity MAE: **8.235 attempts**

- oracle team volume MAE: **3.352**
  - improvement: **4.883**
  - removes **59.3%** of baseline MAE
- oracle QB share MAE: **6.281**
  - improvement: **1.954**
  - removes **23.7%** of baseline MAE

On QB rows with actual attempts >0:

- baseline MAE: **7.514**
- oracle team volume MAE: **2.271**
  - removes **69.8%**
- oracle QB share MAE: **6.536**
  - removes **13.0%**

On 41+ attempt games:

- baseline MAE: **17.625**
- oracle team volume MAE: **0.714**
  - removes **95.9%**
- oracle QB share MAE: **17.322**
  - removes only **1.7%**

**Interpretation:** the residual QB attempt problem is overwhelmingly team pass
volume, especially in true high-volume passing games. Do not turn this result
into a new QB player-share lane.

### RB — carries

Baseline carry MAE: **4.301 carries**

- oracle team rush volume MAE: **4.031**
  - removes **6.3%**
- oracle RB carry share MAE: **1.463**
  - removes **66.0%**

Active/nonzero carry rows:

- baseline MAE: **4.449**
- oracle team volume removes **10.0%**
- oracle player share removes **52.5%**

15+ carry games:

- baseline MAE: **9.861**
- oracle team volume MAE: **8.294**
  - removes **15.9%**
- oracle player share MAE: **3.955**
  - removes **59.9%**

**Interpretation:** the residual RB rushing problem is principally individual
carry-share allocation, not team rush volume.

### RB — targets

Baseline target MAE: **1.386 targets**

- oracle team pass volume MAE: **1.321**
  - removes **4.6%**
- oracle RB target share MAE: **0.329**
  - removes **76.2%**

Active target rows:

- baseline MAE: **1.463**
- oracle team volume removes **3.5%**
- oracle player share removes **61.5%**

9+ target RB games:

- baseline MAE: **8.067**
- oracle team volume removes **8.9%**
- oracle player share removes **54.0%**

**Interpretation:** individual RB receiving opportunity share is the dominant
mechanism.

### TE — targets

Baseline target MAE: **1.795 targets**

- oracle team pass volume MAE: **1.694**
  - removes **5.6%**
- oracle TE target share MAE: **0.427**
  - removes **76.2%**

Active target rows:

- baseline MAE: **1.670**
- oracle team volume removes **8.5%**
- oracle player share removes **58.3%**

9+ target TE games:

- baseline MAE: **6.630**
- oracle team volume removes **14.6%**
- oracle player share removes **67.5%**

**Interpretation:** individual TE target-share allocation is dominant.

### WR — targets

Baseline target MAE: **2.062 targets**

- oracle team pass volume MAE: **1.957**
  - removes **5.1%**
- oracle WR target share MAE: **0.676**
  - removes **67.2%**

Active target rows:

- baseline MAE: **2.057**
- oracle team volume removes **6.3%**
- oracle player share removes **56.4%**

9+ target WR games:

- baseline MAE: **4.767**
- oracle team volume removes **13.4%**
- oracle player share removes **58.0%**

**Interpretation:** individual WR target-share allocation is dominant.

## Error transmission remains player-specific

The frozen ACT-only opportunity errors remain strongly related to the player's
downstream prop error:

- QB attempt error -> pass-yard error: **0.761**
- RB carry error -> rush-yard error: **0.816**
- RB target error -> rec-yard error: **0.782**
- RB target error -> reception error: **0.920**
- TE target error -> rec-yard error: **0.778**
- TE target error -> reception error: **0.898**
- WR target error -> rec-yard error: **0.692**
- WR target error -> reception error: **0.841**

Thus player-share error is not merely a bookkeeping discrepancy. It materially
transmits into individual player yard/reception projection misses.

## Important nuance: high-volume games

Team volume is not irrelevant.

Among high-volume focal-player games, predicted team volume is often also low:

- QB 41+ attempts: team-volume bias **-17.39 attempts**
- RB 15+ carries: team-rush-volume bias **-4.89**
- RB 9+ targets: team-pass-volume bias **-11.55**
- TE 9+ targets: team-pass-volume bias **-7.40**
- WR 9+ targets: team-pass-volume bias **-3.97**

But for skill positions, replacing player share still removes far more
individual opportunity error than replacing team volume.

This supports a split conclusion:
- QB remains a team-volume problem.
- RB/WR/TE remain player-allocation problems, with team volume a secondary
  contributor in the extreme tail.

## What this does NOT authorize

Do not:

- fit realized share back into production;
- create an outcome-selected share multiplier;
- reopen generic WR/TE/RB mean calibration;
- back-apply the Week-5 RB room-allocation shadow to Weeks 1-4;
- weaken Target Share Trajectory V1 eligibility;
- reopen closed QB science merely because team volume is imperfect;
- introduce sportsbook lines into opportunity allocation.

## Next authorized step

For skill positions, the next bounded research lane is:

`PLAYER_SHARE_INPUT_COVERAGE_AND_RESIDUAL_AUDIT_V1`

Purpose: determine **why** the current strictly-pregame player-share machinery
underallocates focal RB/WR/TE players before proposing another mechanism.

The audit must inventory, per ACT-only player-game:

### RB
- current predicted carry share;
- current predicted target share;
- strict-prior same-team and any-team carry/target history already available to
  the stack;
- current room/rank/state inputs already consumed;
- Week-5 room-allocation shadow must remain prospective only.

### WR / TE
- baseline M38 entitlement;
- final post-TE-R5P / post-WR-R15 entitlement;
- strict-prior same-team and any-team prior1/prior3 opportunity evidence already
  consumed by the specialists;
- exact specialist-applied flags;
- current position-room/rank;
- Target Share Trajectory V1 eligibility flag remains false for W1-4.

Questions:
1. Are focal misses concentrated where strictly-prior player history is missing?
2. Or does the stack have strong prior player evidence but still shrink the
   player's share toward the room/team mean?
3. How much does each specialist move the player's share toward or away from the
   realized focal role?
4. Is RB carry-share error the same structural problem as receiver target-share
   error, or a separate allocation mechanism?

No new player-share formula may be fitted until this audit answers those
questions.

For QB, retain the existing protected QB player science and record the
team-pass-volume finding separately; do not mix it into the skill-position
player-share lane.
