# PRISM 2026 Week-5 Stage-1/Stage-2 Retrospective Replay V1 — Certified Result

Date: 2026-10-17

Status: **FROZEN INDEPENDENT MECHANISM REPLAY COMPLETE / DIRECTIONALLY SUPPORTIVE / NO PROMOTION**

Disposition:

`WEEK5_STAGE12_CURRENT_SEASON_REPLAY_DIRECTIONALLY_SUPPORTIVE`

Parent:
`docs/research/PRISM_2026_WEEK5_STAGE12_RETROSPECTIVE_REPLAY_V1_PLAN.md`

## 1. Scientific question

The Stage-1 active-role and Stage-2 conditional-allocation mechanisms were frozen and qualified from independent 2024/2025 evidence with zero 2026 outcomes.

This replay asks only:

> If that exact already-frozen player-first mechanism had been applied unchanged to the preserved Week-5 pregame football state, would named-player opportunity allocation have improved?

This is retrospective current-season QA, not a Week-5 prospective claim and not a new qualification.

## 2. Certified execution

Canonical green run:

- run `39057189966` — **SUCCESS**;
- exact head `25b95767f3aeb1a52c28e0cbb07e6375f36a44ee`;
- artifact `12109199205`;
- digest `sha256:ba7fb4c8851feb3becb4fdfd805ab1e5f3530102b10ec929d9347fecf7568a7f`;
- candidate digest `sha256:3c962fb747d7954418d09fd187a91e79635ac140549d90e9d68da0a157157a0a`;
- strict repository audit — **PASS**.

The build phase completed and hash-froze the candidate **before** Week-5 outcomes were downloaded:

- candidate rows: **302**;
- candidate players: **200**;
- rooms: **90**;
- named-player Stage-1 rows: **266**;
- position fallback rows: **36**;
- prior-history max week: **4**;
- target-week participation columns consumed: **0**;
- target-week outcomes read by builder: **0**;
- max allocation mass gap: **2.22e-16**;
- RB target identity coverage: **99.0%**;
- TE target identity coverage: **99.02%**.

No model, threshold, window, entitlement, or active-set law changed.

## 3. Cell-level result

### RB rushing

- scored rows: **73**;
- players: **73**;
- rooms: **25**;
- named-player Stage-1 support: **93.15%**;
- original Week-5 opportunity MAE: **4.0752 carries**;
- frozen Stage-2 parent MAE: **3.9363**;
- frozen Stage-1/Stage-2 candidate MAE: **3.8472**;
- candidate improvement vs original: **+0.2279 carries**;
- candidate improvement vs parent: **+0.0890 carries**;
- share of rows improved vs original: **45.21%**;
- share improved vs parent: **45.21%**.

The mean improves even though fewer than half of player rows improve. Therefore the gain is not a universal per-player improvement; it is concentrated in some named-player cases.

### RB receiving

- scored rows: **73**;
- original MAE: **1.3614 targets**;
- candidate MAE: **1.3568**;
- improvement vs original and parent: **+0.00465 targets**;
- share improved: **45.21%**.

This is directionally positive but practically tiny on Week 5.

### TE receiving

- scored rows: **101**;
- rooms: **30**;
- named-player Stage-1 support: **87.13%**;
- original MAE: **1.6197 targets**;
- candidate MAE: **1.6005**;
- improvement vs original and parent: **+0.0192 targets**;
- share improved: **60.40%**.

Direction is supportive, magnitude is modest.

## 4. Named-player Week-5 forensic linkage

The frozen opportunity-dominant casebook legally linked four RB rushing cases to the scored replay.

### Jahmyr Gibbs — clear repair direction

- actual carries: **25**;
- original preserved Week-5 state estimate: **12.88**;
- frozen Stage-2 parent: **15.19**;
- Stage-1/Stage-2 candidate: **17.35**;
- candidate shift: **+2.16 carries** from parent;
- candidate improved vs parent: **yes**;
- candidate improved vs original: **yes**.

The frozen independent PRISM mechanism moved materially toward Gibbs' realized role without seeing Week-5 outcomes.

It still remained well below 25 carries, so it did not fully solve the role-regime problem.

### Aaron Jones — small improvement vs original, not parent

- actual carries: **25**;
- original preserved state estimate: **8.56**;
- Stage-2 parent: **9.89**;
- Stage-1/Stage-2 candidate: **9.79**.

The candidate improved over the old original state but slightly worsened the parent.

The major Week-5 allocation miss remains unresolved.

### Ollie Gordon II — small improvement vs original, not parent

- actual carries: **18**;
- original preserved state estimate: **5.40**;
- Stage-2 parent: **6.03**;
- candidate: **5.96**.

Again the candidate is directionally better than the old original estimate but does not explain the large realized role.

### Braelon Allen — source-score blocked

Braelon was present in the frozen candidate:
- canonical rushing q: **20.12%**;
- candidate expected q: **21.84%**.

But the frozen Week-5 outcome authority used by this replay did not mark his row scoreable, so the replay correctly excluded him rather than bridging to another outcome source after seeing the result.

Disposition:
`FROZEN_WEEK5_OUTCOME_AUTHORITY_NOT_SCOREABLE`.

This preserves source integrity but leaves one of the largest allocation cases unresolved in this exact replay.

## 5. Brock Bowers — important negative control

Bowers was not opportunity-dominant in the frozen failure casebook, so he is not part of the four-row opportunity-dominant linkage.

The all-player TE replay nonetheless contains him:

- actual targets: **10**;
- frozen canonical Stage-2 target entitlement: **16.78%**;
- candidate expected q: **15.93%**;
- original/canonical expected targets: **5.99**;
- candidate expected targets: **5.68**.

The Stage-1/Stage-2 candidate moved **away** from Bowers' realized target volume.

This matters because the separate Week-4/5 forensic replay found a different Bowers seam:
- frozen Week-5 rules target share: **27.77%**;
- final target probability: **20.63%**;
- realized target share: **27.78%**.

Therefore Bowers is not evidence that Stage-1/Stage-2 active-role conditioning solves every allocation problem. His case exposes a distinct production-transmission / entitlement-path issue.

## 6. Interpretation

The exact frozen Stage-1/Stage-2 mechanism is **directionally supportive**, especially for RB rushing, and it repaired part of the Gibbs miss.

But it does not close the broader PRISM role-state frontier:

- Aaron Jones remains badly underallocated;
- Ollie Gordon remains badly underallocated;
- Braelon Allen is source-score blocked in this replay;
- Bowers moves the wrong way under this Stage-2 entitlement path;
- RB receiving gain is nearly zero;
- TE receiving gain is small.

Therefore the Week-4/5 forensic finding survives:

> the unresolved problem is not simply “turn on Stage 2.” PRISM still needs to identify which named-player role regime is active now and preserve the correct player-specific allocation path through the final market projection.

## 7. Protected conclusions

This replay does **not**:
- refit Stage 1;
- refit Stage 2;
- reopen raw recent share;
- reopen QB Stage 2;
- manufacture WR Stage 2;
- weaken source gates;
- change production.

Week-6 prospective Stage-2 shadow remains the prospective authority and remains immutable/ungraded.

## 8. Integrity

- candidate frozen before outcomes: **yes**;
- candidate row digest verified: **yes**;
- Week-6 outcomes read: **0**;
- sportsbook predictors: **0**;
- OddsAPI calls: **0**;
- models fit: **0**;
- thresholds searched: **0**;
- production changed: **false**.

Final interpretation:
`WEEK5_STAGE12_CURRENT_SEASON_REPLAY_DIRECTIONALLY_SUPPORTIVE__ROLE_STATE_FRONTIER_STILL_OPEN`.

