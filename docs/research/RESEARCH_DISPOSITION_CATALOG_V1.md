# Research Disposition Catalog V1

**STATUS: DESCRIPTIVE META-AUDIT. NO RE-TESTING. NO NEW EVIDENCE CLAIMED.**

## Why this exists

Dispositions in this repository are free-text `SCREAMING_SNAKE_CASE` strings
scattered across ~100 research/migration docs, 300+ Issue #535 comments, and
ephemeral CI artifacts. There is no queryable record of what has been tried,
what passed, what failed, and *why* it failed. The practical consequence is
that a positive finding can sit unexploited for weeks while adjacent lanes are
re-litigated, and nobody — user, GPT-5.6, or Claude — can answer "what has
actually helped" without re-reading the whole tree.

This catalog is the missing index.

## What this is, and what it is explicitly NOT

This is **hypothesis generation**. It is not evidence.

The invalid version of this exercise — going back through closed lanes,
cherry-picking whichever sub-metric happened to improve, and stacking those
into a combined improvement claim — is selection-on-outcome and would
manufacture a result that cannot replicate. That is not a hypothetical
concern: on 2026-09-20 a 22-slice analysis of 438 real graded bets
(run `35520068364`) produced cells at 67-70% win rate and +37% ROI **on
synthetic data drawn at a flat 51.1% win rate**, and zero survived FDR
correction.

So the rule for anything this catalog surfaces:

> A catalog entry tells us **what to test next**. It never substitutes for
> the test. Any candidate it promotes requires one new preregistered
> confirmation with a correctly-scoped gate, on data not used to generate
> the hypothesis.

## Method

`scripts/research/extract_disposition_catalog_v1.py` scans every `.md` under
`docs/` for disposition-shaped tokens with surrounding context and emits them
for review. Extraction is automated; **classification below is by reading**,
because deciding whether a lane failed on its own target or on an unrelated
secondary gate requires reading the plan's hypothesis, and a regex that
guessed at that would produce confident garbage.

101 docs scanned, 107 disposition rows, 64 distinct tokens.

One extraction caveat worth recording: the same token appears both in **plan**
documents (as a hypothetical branch, e.g. "if any state passes:
`QB_INTERNAL_RELIABILITY_STATE_SUPPORTED`") and in **result** documents (as an
actual verdict). The token alone does not tell you which. This catalog records
the result-document verdict only.

## Classification scheme

| Bucket | Meaning | Action |
|---|---|---|
| `NULL_EFFECT` | Failed on its own hypothesis target. Effect absent or indistinguishable from zero. | Stays closed. No further interest. |
| `OVER_GATED` | Passed on its own target with a real effect size; died only on gates for *different* targets. | Candidate — requires a re-scoped fresh confirmation. |
| `PROCESS_FAIL` | Died on integrity, cohort, coverage or execution error — not on evidence. | Potentially revivable once the defect is repaired. |
| `NEVER_ATTEMPTED` | Diagnostic passed; the authorized follow-up candidate was never built. | Candidate — highest value, no re-test needed to justify. |
| `PROMOTED` | Passed and reached production, or passed and is under forward confirmation. | Live. |

### Component-level status is required, not optional

A single bucket per lane is insufficient, and V1 got a headline finding wrong by
forcing one. A multi-arm experiment can simultaneously contain a failed
architecture gate, a `PROCESS_FAIL` parity component, and a submechanism whose
own diagnostics improved — and collapsing that into one verdict destroys
exactly the information this catalog exists to preserve. WR-R3 below is the
worked example. **Record per-component status, and record which components were
never independently dispositioned.**

## Catalog

### The player-error-persistence family — the key cross-position pattern

| Pos | Lane | Diagnostic disposition | What happened next | Bucket |
|---|---|---|---|---|
| RB | PD2 | `RB_PLAYER_ERROR_PERSISTENCE_DETECTED` — 4/4 gates pass | PD3/PD4/PD5 applied it to the **mean**; all three failed. PD6 blocked by `RB_PD5_COHORT_EXECUTION_DISCREPANCY`. Then the **width** application qualified: `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`, 28/28 gates, pooled CRPS +1.218%, high-difficulty quartile +2.624%. | `PROMOTED` (forward confirmation pending, Lane A) |
| WR | R3 | `WR_PLAYER_ERROR_PERSISTENCE_DETECTED` — 3/3 gates pass (Spearman .08889, +9.55 signed-yard quartile gap, 6/6 positive seasons, 10,675 scoreable rows / 478 players) | **The combined candidate WAS built and executed** — `research-wr-r3-combined-calibration@3d5ae543`, run `34726509088`, terminal disposition `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`. Its Lane B *was* a mean-neutral uncertainty-widening arm using the identical `1 + 0.30*clip((score−0.50)/0.50,0,1)` form. See component breakdown below. | **`ATTEMPTED`** — see components |
| QB | PD2 | `NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE` | Legitimately null. Redirect to PD3 also null (`NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE`, all 4 states failed). | `NULL_EFFECT` |
| TE | R1 | `TE_MECHANISM_DECOMPOSITION_ACTIONABLE` — 5/5 gates pass | Confirmed targets/entitlement dominate TE error (45.2%); validated the R2→R5 path that became TE-R5P. Remaining efficiency gap quantified at ~55% of error mass — unaddressed. | `PROMOTED` (partial; efficiency gap open) |

### RB chain

| Disposition | Bucket | Note |
|---|---|---|
| `RB_PLAYER_ERROR_PERSISTENCE_DETECTED` | see above | Origin of the whole productive RB line |
| `RB_PD2_MULTISEASON_INTEGRITY_FAILURE` | `PROCESS_FAIL` | Integrity, not evidence |
| `RB_PD3_PLAYER_RESIDUAL_CALIBRATION_FAIL` | `NULL_EFFECT` | Mean application |
| `RB_PD4_ROLE_STABILITY_GATED_CALIBRATION_FAIL` | `NULL_EFFECT` | Mean application |
| `RB_PD5_CARRY_ONLY_RESIDUAL_CALIBRATION_FAIL` | `NULL_EFFECT` | Mean application |
| `RB_PD5_COHORT_EXECUTION_DISCREPANCY` | `PROCESS_FAIL` | Blocked PD6 entirely; PD6 never implemented |
| `RB_YARD_DIFFICULTY_WIDTH_INTEGRITY_FAILURE` | `PROCESS_FAIL` | Repaired; superseded by the qualified run |
| `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` | `PROMOTED` | Lane A forward confirmation |
| `FOOTBALL_RESEARCH_CHAMPION_NOT_PRODUCTION_QUALIFIED` | `NULL_EFFECT` | P3 Weeks 2-18; failed tail gate |
| `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION` | `NULL_EFFECT` | 4/13 gates, aggregate MAE **worse**, 3/6 seasons. Despite the "MIXED" label this failed on its own target — not over-gated. |
| `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS` | `PROMOTED` | Week 1 only |
| `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_FAIL_NO_PROMOTION` **and** `..._PASS_READY_FOR_PROMOTION` | — | **Both dispositions exist for the same lane.** Chronology resolves it, but this is a concrete example of why the index was needed. |

### WR / TE / QB / cross-cutting

| Disposition | Bucket | Note |
|---|---|---|
| `WR_PLAYER_ERROR_PERSISTENCE_DETECTED` | `NEVER_ATTEMPTED` | See key finding below |
| `WR_INDIVIDUAL_MECHANISMS_MAPPED` (226/337 WRs) | `NEVER_ATTEMPTED` | Descriptive mapping, never applied |
| `WR_TARGET_CATCH_YPR_INDIVIDUAL_MECHANISMS_MAPPED` | `NEVER_ATTEMPTED` | Same |
| `WR_NGS_TARGET_MODEL_FAIL` | `NULL_EFFECT` | |
| `TE_PARTICIPATION_ENTITLEMENT_V1_PASS` | `PROMOTED` | TE-R5P |
| `TE_TARGET_POOL_CONTEXT_MODEL_FAIL` | `NULL_EFFECT` | |
| `M87_REGIMES_NOT_REPLICATED` | `NULL_EFFECT` | Honest 2023 holdout |
| `M88_SOURCE_OR_COHORT_FAILURE` | `PROCESS_FAIL` | |
| `NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY` | `NULL_EFFECT` | 2025 supported ROI −9.56% vs −3.23% baseline |
| `CONSERVATION_ONLY_SUPPORTED` | `PROMOTED` (partial) | |
| `STRICT_PRIOR_NGS_FEATURES_ELIGIBLE` | `NEVER_ATTEMPTED` | Eligible, not applied |
| `HOLD_SOURCE_BLOCKED_NEW_INFORMATION` | `PROCESS_FAIL` | Access, not science |

## Key findings

### 1. CORRECTED — the width idea originated in WR, not RB

**V1 of this catalog asserted that the WR-R3 combined candidate was never built
and that no WR width lane existed anywhere in the repo. Both claims were false.**
GPT-5.6 caught it (Issue #535, comment `5751273851`); the correction is verified
below and the original claim is preserved here so the error stays in the record.

What is actually true:

- `research-wr-r3-combined-calibration@3d5ae543` exists and executed. Run
  `34726509088`, job `103641446982`, artifact `10309091155`. Terminal
  disposition `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`.
- `docs/migrations/WR_R3_COMBINED_CALIBRATION_PLAN.md` §"Lane B —
  player-specific mean-neutral uncertainty widening" defines
  `width_mult = 1.0 + 0.30 * clip((uncertainty_score - 0.50) / 0.50, 0.0, 1.0)`.
- **The lineage runs the other way.**
  `RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_AMENDMENT3.md:43` states verbatim: *"The
  `0.30` width coefficient and `0.50` onset come from the prior WR-R3 Lane-B
  uncertainty precedent in `docs/migrations/WR_R3_COMBINED_CALIBRATION_PLAN.md`
  / `scripts/backtest/evaluate_wr_r3_combined_calibration.py`; they were not fit
  to this RB candidate."*

So WR width was first, and the qualified RB mechanism explicitly inherited its
constants from it. The V1 finding inverted the actual history.

#### Component-level status for WR-R3 (the correct record)

| Component | Status |
|---|---|
| `WR_PLAYER_ERROR_PERSISTENCE_DETECTED` diagnostic | positive, 3/3 gates |
| Combined architecture (Lane A mean + Lane B width) | **failed its own gates** — 2025 MAE 22.0073 → 21.7978 (~0.95%, below the frozen ≥1% bar); pooled miss20 worsened |
| Exact-parent parity / cohort reproduction | `PROCESS_FAIL` — `parent_mae_exact`, `parent_rmse_exact`, `parent_bias_exact`, `parent_correlation_exact`, `wr_2025_rows_exact` all drifted from frozen constants |
| Lane B uncertainty submechanism | diagnostics **improved and were not among the failed gates** — eligible 80% coverage .61313 → .61878; high-uncertainty 80% coverage .57992 → .60246; signed mean shift +0.0142 yd. **Never independently dispositioned.** |

#### What remains genuinely open, narrowly

Per GPT-5.6, the executed WR Lane B is *not* identical to the later qualified RB
mechanism: it modified pre-MC `rules_volatility_mult` (clipped to the
simulator's volatility range), its non-negative floor permitted small mechanical
mean drift, it was entangled with a Lane A mean correction, and **it was never
graded by CRPS on a pure post-simulation exact-mean-neutral arm**.

So a **pure post-MC, exact-mean-neutral, CRPS-graded WR width mechanism** may
still be scientifically open. But it is emphatically **not** a fresh discovery:
it carries prior WR outcome exposure from run `34726509088` plus the RB
qualification. Any such study requires an explicit anti-retest distinction from
WR-R3 Lane B, a new preregistered mechanism statement, **no** tuning of `0.30`,
`0.50`, the history window or score construction against the observed WR result,
and genuinely fresh confirmation data. Re-running a post-MC width transform on
the same 2020-2025 WR cohort and calling it a blind discovery would not be
acceptable.

#### Why the V1 search missed it — recorded as a method failure

Two greps, both too narrow, produced a **universal negative**:

- branch filter `wr.*(width|uncertain|crps)` → 0 matches. The branch is
  `research-wr-r3-combined-calibration`, which contains none of those words.
- docs filter `WR.*(MC_WIDTH|WIDTH|uncertainty calib)` → 0 matches **in the WR
  width plan document itself**, because it required `WR` and the width term on
  the same line, and the Lane B heading reads "player-specific mean-neutral
  uncertainty widening" with no `WR` on that line.

The lesson is not "write better regexes." It is: **a universal negative ("no
such lane exists anywhere") requires exhaustive verification, not two keyword
greps.** A bounded claim ("I did not find one via these two searches") would
have been true and would have invited exactly the check that caught it.

### 2. Mean applications fail; width applications do better (survives the correction)

This finding stands, and the correction above actually strengthens it.

Every *mean* application of player-error persistence has failed: RB PD3, PD4,
PD5, and WR-R3's Lane A. Meanwhile the RB width application qualified 28/28,
and WR's Lane B width diagnostics **improved and were not among the failed
gates** — the WR architecture died on its combined mean/tail gates and on
exact-parent parity, not on its width arm.

So across two positions and four failed mean corrections, the width arm is the
component that has never been the thing that broke. That is consistent with a
persistence-of-*absolute*-error statistic carrying information about
**uncertainty** rather than **level**, which is what it should carry
mechanistically.

Caveat that must travel with this finding: the WR width arm was never
independently dispositioned, so "did not fail" is not "passed."

### 3. Process failures are a large share of closures

`RB_PD2_MULTISEASON_INTEGRITY_FAILURE`, `RB_PD5_COHORT_EXECUTION_DISCREPANCY`,
`RB_YARD_DIFFICULTY_WIDTH_INTEGRITY_FAILURE`, `M88_SOURCE_OR_COHORT_FAILURE`,
`HOLD_SOURCE_BLOCKED_NEW_INFORMATION` — five closures that carry no scientific
information at all. PD6 was never implemented purely because of one of them.
These should be tracked separately from evidence-based closures, because they
are repairable and currently look identical to real failures in the record.

### 4. No `OVER_GATED` case was found

The originating hypothesis for this catalog was that mechanisms were being
killed by gates on unrelated targets. **On review, no clear instance of that
was found.** The closest candidate by name — `R27D_..._MIXED_OR_FAIL` — failed
on its own target (aggregate MAE worse, 4/13 gates, 3/6 seasons).

This is worth recording as a negative result about our own process: the frozen
multi-gate plans have generally been scoped to their own hypothesis. The real
leakage of value has been `NEVER_ATTEMPTED` and `PROCESS_FAIL`, not
over-gating.

## Recommended process change

Separate **scientific** gates from **deployment** gates in future frozen plans.
If a hypothesis is "X improves WR receiving yards," the scientific gate belongs
on receiving yards; "must not degrade receptions" is a deployment constraint —
measure it, record it, but do not let it close the science. Conflating the two
means a real mechanism and an unacceptable tradeoff produce the same
disposition, and the record cannot distinguish them later.

Additionally: every closure should record **which bucket above it falls in**,
so `PROCESS_FAIL` and `NEVER_ATTEMPTED` stop being indistinguishable from
`NULL_EFFECT` in the ledger.

## Lineage

Extractor: `scripts/research/extract_disposition_catalog_v1.py`
Generated: 2026-09-20. Issue #535.
No production path, model, projection or pricing logic is touched by this document.
