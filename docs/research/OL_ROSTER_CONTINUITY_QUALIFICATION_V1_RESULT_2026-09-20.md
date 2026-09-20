# OL Roster Continuity Qualification V1 — Result — 2026-09-20

**Status:** CLOSED QUALIFICATION RESULT  
**Frozen plan:** `docs/research/OL_ROSTER_CONTINUITY_QUALIFICATION_V1.md`  
**Implementation commit:** `d5ce896c4ff34fe48e76b6bcb8f87c29b55e9848`  
**Production changes authorized:** false  
**Predictive outcomes inspected:** false

## Canonical execution

- workflow: `OL Roster Continuity Qualification V1`
- run: `35514546250`
- job: `106088156234`
- artifact: `10606771062`
- artifact digest: `sha256:e7688b9abd6ac683ca19674f412b11f18dc77109e785d15420d60609c03d33ee`
- run head: `d5ce896c4ff34fe48e76b6bcb8f87c29b55e9848`
- workflow conclusion: `success`

## Frozen candidate

`ol_roster_continuity_share_prev_game`

Formula:

`|current OL stable IDs ∩ prior scheduled same-team game OL IDs| / |current OL stable IDs|`

Backups remained in the broad roster state. Week 1 / no prior same-season scheduled game remained explicit
`UNKNOWN_NO_PRIOR_GAME`.

## Qualification result

Final qualification disposition:

`READY_FOR_FROZEN_EXPERIMENT`

Evidence:

- eligible scheduled team-games: **3,742**
- known rows: **3,518**
- broad pregame coverage: **0.940139** (**94.0139%**)
- unknown rows: **224**
- unknown because no prior scheduled game: **224**
- unknown current roster: **0**
- unknown prior roster: **0**
- stable-ID coverage: **0.9999086440956679** (**99.9909%**)
- source OL rows before schedule filter: **57,357**
- source OL rows on scheduled games: **54,731**

Identity/integrity:

- ambiguous same-week GSIS/team conflicts: **0**
- duplicate source identity rows collapsed to set: **0**
- duplicate published team-week rows: **0**
- schedule join fanout: **0**
- redundancy join fanout: **0**
- chronology violations: **0**
- name fallback used: **false**
- future-week roster used: **false**
- target-game snap/participation used: **false**
- target-game PBP used in continuity: **false**
- target-game team state used: **false**

Stability diagnostic only, because this candidate is a directly observed change state:

- adjacent-game pairs: **3,294**
- adjacent-game Spearman: **0.052769**
- median absolute adjacent change: **0.0**
- hard stability gate: **false**

Outcome-free redundancy:

- train seasons: **2019–2023**
- holdout seasons: **2024–2025**
- train rows: **2,494**
- holdout rows: **1,024**
- holdout reconstructibility R2: **0.005216**
- disposition: `INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE`

The continuity state is therefore almost entirely unreconstructible from the frozen prior-team-state redundancy inputs.

## Input fingerprints

- schedule SHA256: `60db4d57a7132b4f00d7f51996dab19b4d171e8e90393f3f95c8fa8b19b14f04`
- weekly roster SHA256: `f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a`
- team weekly SHA256: `ab61d6aeff466aa53b6347829a8bf4796896a76797d78b59e6fd1fdb17f5639a`

## Anti-retest predictive-authorization audit

Qualification does **not** automatically authorize a predictive experiment.

Before opening any QB/RB efficiency outcome, the repository's prior research was audited.

### M77 overlap

M77 (`scripts/backtest/test_qb_exact_personnel_discontinuity.py`) already performed one frozen QB predictive test using exact pregame personnel discontinuity features, including:

- `off_ol_turnover`
- `off_ol_added_ratio`
- `off_ol_replacement_deficit`
- `off_ol_role_delta`

M77 separately corrected attempt residual and YPA residual, evaluated passing-yards MAE/correlation/tail behavior, and was rejected. The no-retest ledger requires a **new mechanism beyond discontinuity counts** before reopening that family.

The V1 roster-continuity candidate is broader than M77's starter-depth representation and has excellent source coverage, but scientifically it remains a last-game personnel-overlap count/share. Using it as another QB point-mean correction would therefore repeat the closed M77 mechanism.

### M71 overlap

M71 (`scripts/backtest/audit_qb_efficiency_uncertainty.py`) already tested QB efficiency uncertainty/tail-risk prediction using frozen 2024 training and untouched 2025 evaluation. Its feature families included QB intrinsic volatility, offense ecosystem volatility, opponent volatility, week-specific context and structural model disagreement. The family was closed as negative under the QB no-retest ledger.

The ledger allows reopening QB efficiency volatility/risk only with materially new information. Although OL roster continuity is new source information relative to M71, it is still a personnel-discontinuity count/share and is not sufficiently distinct from the separately failed M77 depth-role discontinuity family to justify spending a new QB holdout.

## Predictive authorization decision

For QB outcomes:

`PREDICTIVE_AUTHORIZATION_WITHHELD_QB_ANTI_RETEST`

No QB predictive target was inspected.

RB research remains pinned/paused under the current user scope, so no RB predictive experiment is opened from this result either.

The candidate remains a qualified engineering/context asset. A future predictive use requires a separately frozen plan whose mechanism is demonstrably distinct from the closed discontinuity-count and QB-volatility families.

## Boundaries preserved

- no production change
- no sportsbook teacher
- no paid odds pull
- no Issue #535 touch
- no failed-family rescue
- no target outcome inspection during qualification or this authorization audit

## Final disposition

`OL_ROSTER_CONTINUITY_QUALIFICATION_V1_READY_PREDICTIVE_AUTHORIZATION_WITHHELD_QB_ANTI_RETEST`
