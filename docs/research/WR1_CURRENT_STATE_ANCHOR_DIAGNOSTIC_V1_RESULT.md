# WR1 Current-State Anchor Diagnostic V1 — Result

Date: 2026-09-25

Disposition: **WR1_CURRENT_STATE_ANCHOR_DIAGNOSTIC_V1_COMPLETE — CANDIDATE NOT JUSTIFIED**

This is a discovery-only diagnostic result. No production change is authorized.

## Authority

- branch: `research-wr1-current-state-anchor-diagnostic-v1`
- authoritative run: `36171050355`
- job: `108190378923`
- tested head: `d92ebaf1053a5a0feca05d43247591b96e9eacbf`
- artifact: `10880365550`
- digest: `sha256:73faf54df34e4809c5d80fbef71d410fd4ac248e67c5bb726ee8925d88ef89d6`
- discovery seasons: 2022-2023 only
- 2024-2025 confirmation seasons inspected: **none**
- parameters fit: 0
- candidate variants scored: 0
- sportsbook inputs: 0
- production changed: false

Two earlier runs stopped mechanically before producing metrics:
- `36170632540`: entitlement frame did not carry stable identity;
- `36170890991`: merge validation rejected repeated ineligible blank identities.

Both were repaired without changing the frozen scientific question or thresholds.

## Frozen verdict

The frozen diagnostic rule required all of:
1. blend-4 WR1 team-share MAE < M38 MAE;
2. positive state-gap vs needed-correction Spearman;
3. state-gap sign agreement >50%;
4. state-normalized WR1 within-room MAE < M38 within-room MAE;
5. same-direction signal in the targetable-hurts cohort.

Criterion 4 failed.

Therefore:

`candidate_justified = false`

No WR1-only state-update candidate is authorized.

## Pooled 2022-2023 — absolute/team target share

M38 WR1:
- n: **848**
- MAE: **0.082427**
- bias: **-0.051290**
- correlation: **0.2818**

Validated blend-4 current-season state:
- MAE: **0.071169**
- bias: **-0.002258**
- correlation: **0.3465**

Prior-only:
- MAE: **0.073396**

Current-only:
- MAE: **0.074384**

So the already-validated blend-4 state materially outperformed the immutable M38
WR1 anchor as an estimate of **absolute team target share** in discovery.

State gap versus needed M38 correction:
- Spearman: **+0.1943**
- Pearson: **+0.2046**
- sign agreement: **70.40%** over 848 rows

The signal was positive in both discovery seasons:
- 2022 Spearman: **+0.1458**
- 2023 Spearman: **+0.2462**

## Early-season behavior

Blend-4 beat M38 WR1 MAE in every current-games bucket:

- 1 completed game: `0.08806 -> 0.06963`
- 2 completed games: `0.09647 -> 0.06970`
- 3 completed games: `0.08689 -> 0.08112`
- 4 completed games: `0.09344 -> 0.08514`
- 5-8 games: `0.08345 -> 0.06958`
- 9+ games: `0.07623 -> 0.06876`

This supports the earlier Current-Season State Persistence V1 finding that WR
target-share role state updates quickly.

## Targetable-volume interaction

Rows where targetable-dropback thinning **hurt** WR1 target accuracy:
- rows: **553**
- mean state gap (blend-4 minus M38): **+0.05264**
- positive state-gap rate: **97.11%**
- state-gap vs needed-correction Spearman: **+0.1453**
- correction-direction sign agreement: **89.69%**

Rows where targetable thinning helped:
- rows: 295
- positive state-gap rate: 92.20%
- state-gap vs needed-correction Spearman: +0.0938
- sign agreement: 34.24%

The failure mode is therefore strongly associated with a pre-existing M38 WR1
absolute-share shortfall.

## Why the WR1-only candidate is still rejected

The frozen conserved-WR-room test did not improve.

On the subset with complete strictly-prior state for every modeled WR:

M38 WR1 within-room share:
- n: **60**
- MAE: **0.127663**
- bias: **-0.051126**

State-normalized WR1 within-room share:
- MAE: **0.152165**
- bias: **-0.104174**

Room-share state-gap sign agreement:
- **35.0%**

Therefore the strong absolute WR1 state signal cannot be cleanly interpreted as
a request to redistribute a fixed WR room toward WR1.

## Scientific interpretation

This diagnostic closes the narrow hypothesis:

> keep total WR-room mass fixed and update only the WR1 anchor from current-season state.

That mechanism is not supported.

The result instead reinforces the upstream room-composition diagnosis:

- current stack underallocates WR room mass;
- M38 WR1 absolute team share is materially low;
- WR2+ is slightly overallocated inside the existing WR room;
- targetable-dropback team-volume correction improves WR2+ while worsening WR1;
- TE and RB/FB room mass improved under targetable thinning while WR room mass worsened.

The missing WR1 opportunity therefore appears to cross the **room boundary**,
not merely the WR1-vs-WR2+ boundary.

This does **not** reopen C1's old team-history position-group calibration.

A future room-composition lane is allowed only if it uses genuinely new
strictly-prior information not present in C1, such as active-roster
player-level current-state evidence already shown to persist.

## Closure

No M38 multiplier retune.
No WR1-only blend replacement.
No within-WR-room state rescue.
No 2024-2025 confirmation run for this rejected mechanism.

The next question must be a distinct active-roster / room-composition hypothesis.
