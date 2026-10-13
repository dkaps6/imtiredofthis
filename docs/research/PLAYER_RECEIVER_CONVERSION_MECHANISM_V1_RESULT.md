# PLAYER RECEIVER CONVERSION MECHANISM V1 — HISTORICAL RESULT

Date: 2026-10-13

Status: **NOT CONFIRMED / FAIL CLOSED**

Frozen parent contracts:
- `docs/research/PLAYER_RECEIVER_CONVERSION_STATE_V1_SOURCE_ANTIRETEST.md`
- `docs/research/PLAYER_RECEIVER_CONVERSION_STATE_V1_SOURCE_RESULT.md`
- `docs/research/PLAYER_RECEIVER_CONVERSION_MECHANISM_V1_FREEZE.md`

## Canonical authority

- workflow: `Research Player Receiver Conversion Mechanism V1`
- run: `38411614849` — **SUCCESS**
- exact head: `c59942fad8e75817f395cb1db6097b1d84de464d`
- artifact: `11822441707`
- digest:
  `sha256:a9dc093124e56507e191983a6a0c5fe6fabf9b0461aba45fdf990b69a75ae25b`
- strict repository audit: **PASS**
- frozen integrity certification: **PASS**

The earlier red attempts were mechanical only:
- source/environment and stable-identity plumbing;
- same-name roster ambiguity outside the WR/TE scoring cells.

The final repair resolves receiver identity by stable GSIS ID inside the frozen
WR/TE position family. It did not alter features, seasons, support floors,
model architecture, gates, or thresholds.

## Frozen question

Does strictly-prior receiver-level charted conversion state add incremental
pregame information beyond the canonical player catch-rate mean?

Frozen features:
- receiver drop rate conditional on charted catchable targets;
- receiver catchability relative to same-team offense catchability.

Development:
- 2023 only.

Untouched confirmation:
- 2024;
- 2025.

No sportsbook/OddsAPI information entered the experiment.

## Panel / integrity

- panel rows: **5,913**
- distinct players: **370**
- chronology violations: **0**
- same-team-history violations: **0**
- models fit: **2** (one WR, one TE)
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- 2026 outcomes: **0**
- threshold searches: **0**
- production changes: **0**

## WR result

Development support:
- 1,362 rows
- 164 players
- PASS

### 2024 confirmation

Catch-rate MAE:
- canonical: **0.224812**
- candidate: **0.224990**
- improvement: **-0.000178**

Oracle-target reception MAE:
- canonical: **0.907435**
- candidate: **0.906403**
- improvement: **+0.001033**

Mean catch-rate shift:
- **-0.001422**

Disposition:
**FAIL**. The candidate worsened catch-rate MAE and therefore did not clear the
frozen confirmation gate.

### 2025 confirmation

Catch-rate MAE:
- canonical: **0.229410**
- candidate: **0.230135**
- improvement: **-0.000725**

Oracle-target reception MAE:
- canonical: **0.844002**
- candidate: **0.847163**
- improvement: **-0.003161**

Mean catch-rate shift:
- **+0.001488**

Disposition:
**FAIL**.

Final WR disposition:

`PLAYER_RECEIVER_CONVERSION_MECHANISM_NOT_CONFIRMED`

## TE result

Development support:
- 636 rows
- 85 players
- PASS

### 2024 confirmation

Catch-rate MAE:
- canonical: **0.249681**
- candidate: **0.247826**
- improvement: **+0.001855**

Oracle-target reception MAE:
- canonical: **0.715518**
- candidate: **0.710783**
- improvement: **+0.004735**

Mean catch-rate shift:
- **-0.002754**

Despite positive point estimates, the frozen clustered-confidence requirements
did not produce a position-level confirmation.

### 2025 confirmation

Catch-rate MAE:
- canonical: **0.231457**
- candidate: **0.232630**
- improvement: **-0.001172**

Oracle-target reception MAE:
- canonical: **0.700205**
- candidate: **0.702906**
- improvement: **-0.002701**

Mean catch-rate shift:
- **+0.000559**

Disposition:
**FAIL**.

Final TE disposition:

`PLAYER_RECEIVER_CONVERSION_MECHANISM_NOT_CONFIRMED`

## Overall disposition

`PLAYER_RECEIVER_CONVERSION_MECHANISM_NOT_CONFIRMED`

Neither position confirmed unchanged in both 2024 and 2025.

## Interpretation

The receiver-level source itself is excellent and remains useful evidence:
stable GSIS identity, charted drop attribution, and catchability are available
historically with strong coverage.

But this exact simple mechanism does **not** provide stable incremental
prediction beyond the canonical catch-rate mean.

That matters: the result separates **data availability** from **predictive
value**.

The model should not alter a WR/TE catch-rate mean simply because the receiver
has recently seen more catchable throws or has a different charted drop rate.

## Stop rule

This exact family is closed.

Do not rescue with:
- alternate 3/5/10-game windows;
- different Ridge alpha;
- WR1/WR2/slot/TE1 slices;
- pooled 2024+2025 confirmation;
- thresholded “high-drop” or “high-catchability” groups;
- extra FTN variables added after seeing this result;
- nonlinear model substitution;
- target-depth interaction added to this same failed candidate;
- post-hoc coefficient sign constraints.

Any future receiver-efficiency lane must introduce materially different
pregame football information or mechanism state.

## Protected science

This result does not modify:
- WR-R15;
- TE-R5P;
- M38 WR1;
- QB M89/M90;
- QB C2;
- RB receiving-room improvement;
- RB R22;
- target-share trajectory;
- joint Dirichlet RB-carry / WR-target opportunity uncertainty.

Week-6 opportunity shadow remains separately source-gated by exact Week-6 roster
authority.

No production promotion is authorized.
