# Receiver Targetable-Dropback V1 — Player Full-Stack Result

Date: 2026-09-25

Disposition: **RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_FAILED_CLOSED**

This is a scientific failure under the frozen player-level gates, not a
mechanical/provenance failure.

The parent team-opportunity result remains valid and unchanged: the exact
cumulative-count targetable-dropback formula improved team receiver-target
forecasting independently in 2022, 2023, 2024 and 2025.

## Authority

- branch: `research-receiver-targetable-dropback-v1-full-stack`
- frozen plan:
  `docs/research/RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_PLAN.md`
- freeze commit: `c5471e179b3e55b8c02fcac88974a7e61536b87c`
- authoritative run: `36147357028`
- job: `108111725439`
- tested head: `474a54b0407b98187bbfd5e4f1c4bb570821e577`
- artifact: `10869364930`
- digest:
  `sha256:981c56190e4bfdf629a93cee380335d0138108a4fef167fde372fe3a91c93ce6`
- seasons: 2024 and 2025
- MC draws: 5000
- parameters fit: 0
- candidate variants scored: 1
- sportsbook inputs: 0
- production changed: false

## Exact candidate tested

Strict-prior targetable-dropback rate remained unchanged:

`R_T = sum(prior team targets) / sum(prior team dropbacks)`

At the player simulation seam:

`candidate target probability_i = R_T * current entitlement_i`

The candidate preserved:
- team dropback state;
- M38;
- TE-R5P;
- WR-R15;
- catch rate;
- YPT;
- QB M89/M90 and C2;
- all rushing arrays;
- ATD;
- ML/State components;
- ensemble weights;
- RB Rush+Receiving Conservation V2 formula.

## Mechanical / integrity result

All frozen integrity gates passed.

- baseline explicit-entitlement parity: PASS
- strict-prior targetable-rate provenance: PASS
- league fallback <=1%: PASS
- finite targetable rates in [0,1]: PASS
- target entitlement unchanged: PASS
- TE-R5P conservation: PASS
- WR-R15 historical conservation: PASS
- QB pass-yards arrays bit-identical: PASS
- rush-att arrays bit-identical: PASS
- rush-yards arrays bit-identical: PASS
- ATD arrays bit-identical: PASS
- raw rush+receiving identity: PASS
- RB V2 pathwise identity: PASS
- sportsbook inputs: 0
- target-game outcomes upstream: 0
- parameters fit: 0
- variants scored: 1

Therefore the player-level result is scientifically interpretable.

## Receptions

Pooled WR/TE/RB macro MAE:

`1.290256 -> 1.290744`

Pooled macro p90 AE:

`2.764796 -> 2.801976`

Pooled macro absolute bias:

`0.194655 -> 0.356983`

By position pooled MAE:

- WR: `1.440661 -> 1.451920`
- TE: `1.285236 -> 1.290029`
- RB: `1.144872 -> 1.130283`

So RB receptions improved, but WR and TE worsened and the macro protection gates
failed.

## Receiving yards

Pooled WR/TE/RB macro MAE:

`16.237196 -> 16.196135`

This is a real average improvement.

But pooled macro p90 AE worsened:

`34.211309 -> 35.543674`

Pooled macro absolute bias worsened:

`2.511945 -> 4.489480`

By position pooled MAE:

- WR: `21.972878 -> 21.981999`
- TE: `15.599941 -> 15.640045`
- RB: `11.138769 -> 10.966360`

The strongest player-level benefit was again RB receiving yards. WR and TE were
slightly worse.

## High-entitlement protection

Q4 receiving-yard MAE:

`26.542664 -> 26.865304`

Q4 p90 AE:

`55.510605 -> 58.121353`

This repeats the earlier pattern from One-Pass / hierarchical reconciliation:
a broad team-level correction can improve average signal while damaging
high-authority receiver tails.

## RB rush+receiving protection

- 2024 MAE: `25.975085 -> 26.048348`
- 2025 MAE: `25.172820 -> 25.227087`
- pooled p90 AE: `55.528455 -> 56.388650`

All three protected RB combo gates failed.

## Raw-MC diagnostic

The final ensemble did **not** create the failure. The raw MC candidate itself
already moved in the wrong direction for WR/TE.

Pooled raw-MC MAE:

### Receptions
- WR: `1.484202 -> 1.565120`
- TE: `1.318318 -> 1.395013`
- RB: `1.146085 -> 1.176187`

### Receiving yards
- WR: `22.185703 -> 22.604410`
- TE: `15.711398 -> 16.156069`
- RB: `10.778706 -> 10.669338`

The existing ensemble actually dampened much of the WR/TE damage.

Raw-MC receiving-yard bias was already negative before the candidate:

- WR: `-6.676 -> -11.338`
- TE: `-5.934 -> -8.894`
- RB: `-2.130 -> -3.487`

Final ensemble bias was less negative:

- WR: `-3.952 -> -7.032`
- TE: `-3.544 -> -5.499`
- RB: `-0.040 -> -0.938`

## Semantics check

The player-share layer was not double-corrected.

Canonical PlayerForm defines:

`tgt_share = player targets / team targets`

Therefore multiplying conditional target entitlement by

`R_T = team targets / team dropbacks`

is mathematically valid for converting a conditional target share into a
dropback-level player target probability.

The full-stack failure therefore cannot be dismissed as a denominator mistake.

## Interpretation

Two facts now coexist:

1. **Team targetable-dropback volume is real and replicated.**
   The same strict-prior cumulative-count formula improved team receiver-target
   prediction in four separate seasons.
2. **Uniformly translating that team-volume correction through the existing
   player entitlement layer is not production-safe.**

The evidence says the loss occurs downstream of the team-volume signal.

Historical ecosystem evidence already shows persistent room-mass imbalance:
- WR room target share underallocated;
- TE room target share overallocated;
- RB/FB room target share overallocated.

At the same time, current WR/TE receiving-yard MC is already negatively biased,
so reducing opportunity uniformly can worsen yardage even when team target
volume becomes more correct.

This creates a plausible compensation structure:
legacy opportunity inflation may be partially offsetting room-mass and/or
efficiency underprojection at the player level.

That is a diagnosis to test, not a license to preserve a known semantic error.

## Frozen stopping rule applied

No rescue is authorized for this candidate.

Do **not** try:
- WR/TE/RB carveouts;
- Q4 exemptions;
- targetable-rate caps/floors;
- partial application of R_T;
- shrinkage;
- recency weighting;
- alternate history windows;
- position-specific targetable rates;
- C2 routing;
- hierarchical reconciliation;
- catch-rate or YPT retuning;
- sportsbook routing;
- 2026 outcome fitting.

## Next research

The next lane should be diagnostic-only:

**CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1**

Question:

Where does the valid team-level target-volume improvement get lost in the
current player stack?

Decompose, on the current promoted historical stack:
1. team target-volume error;
2. WR / TE / RB room-mass error;
3. within-room player entitlement error;
4. catch-rate / reception conversion error;
5. YPT / receiving-yard efficiency error.

The audit should reuse the concepts from M34/M35 and the old ecosystem audit,
but must not rerun closed C1 group-mass calibration or generic historical
allocator experiments.

Only a new, structured, leakage-safe mechanism isolated by that audit may earn
a separately frozen candidate.
