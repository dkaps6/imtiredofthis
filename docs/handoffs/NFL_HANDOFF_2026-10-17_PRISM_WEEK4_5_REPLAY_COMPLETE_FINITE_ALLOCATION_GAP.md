# NFL HANDOFF — 2026-10-17 — PRISM WEEK-4/5 REPLAY COMPLETE / FINITE ALLOCATION COHERENCE GAP CONFIRMED

**Status:** CURRENT / SUPERSEDES THE EARLIER WEEK-4/5 REPLAY-ACTIVE HANDOFF  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Primary issue:** #535

## 1. NON-NEGOTIABLE MODEL OBJECT

PRISM remains an **individual-player projection model**.

The scientific object is the named player in his actual pregame football situation:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE remains downstream QA only.

The J.K. Dobbins “Construct NFL Tickets” discussion remains binding model intuition: understand why this named player should receive and convert the opportunity relevant to the exact market. Carries, targets, receptions, yards and TDs are different football questions.

Ticket order remains:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

## 2. WEEK-4/5 PLAYER-FIRST FORENSIC REPLAY — COMPLETE

Frozen protocol:
`docs/research/PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_V1_PROTOCOL.md`

Protocol freeze:
`b4bc97a484a9c71676bdd6f6694983a3b822f33b`

Canonical all-row execution:
- run `39043393975` — SUCCESS;
- exact head `c77a31900c8f99692201453114685cd15796e568`;
- artifact `12105086305`;
- artifact name `prism-week4-5-all-row-player-first-replay-v1-39043393975`;
- digest `sha256:653e58cd37511dc2ddae499c0b123a564a5317b9912d556a2c427cb05148a363`;
- Issue #535 checkpoint comment `6235459292`.

Disposition:

`PRISM_WEEK4_5_ALL_ROW_PLAYER_FIRST_REPLAY_COMPLETE__DIAGNOSTIC_ONLY__NO_PROMOTION`.

Preserved replay universe:
- 836 named player-game-market rows;
- Week 4: 440;
- Week 5: 396;
- settled: 819;
- non-settled/void/DNP: 17;
- 236 unique named players.

Player-first diagnostic outputs:
- informative casebook: 100 rows;
- matched correct controls: 100 rows;
- repeat same-player + market informative registry: 20 rows / 10 identities;
- exact/preserved opportunity-efficiency component rows: 405;
  - Week 4: 365;
  - Week 5 exact frozen casebook: 40.

Integrity:
- Week-4 Shapley identity max gap `5.684e-14`;
- Week-5 WR exact four-prior-current-season reliability eligibility retained where legal;
- Week-4 exact WR dual score manufactured: 0;
- model fits: 0;
- threshold searches: 0;
- sportsbook upstream/refetch: 0;
- OddsAPI calls: 0;
- Week-6 outcomes read: 0;
- Week-6 locks changed: false;
- production changed: false.

Important source boundary:
the original Week-5 pregame Full Slate artifact `11585973062` is no longer retrievable. Therefore exact Week-5 final component attribution outside the already-frozen top-40 casebook is **not reconstructed or approximated**. The broader Week-5 rows retain legitimate player-state evidence only.

## 3. PLAYER-LEVEL ATTRIBUTION FROM THE 100-CASE INFORMATIVE CASEBOOK

Primary attribution counts:
- `OPPORTUNITY_QUANTITY_MISREAD`: 40;
- `SOURCE_BLOCKED_OR_UNAVAILABLE`: 28;
- `EFFICIENCY_MISREAD`: 20;
- `MIXED`: 12.

By week:
- Week 4: opportunity 29 / source-blocked 11 / efficiency 5 / mixed 5;
- Week 5: source-blocked 17 / efficiency 15 / opportunity 11 / mixed 7.

Thus opportunity or mixed opportunity/efficiency is involved in **52/100** informative player cases.

Matched-control contrast:
- informative-case median absolute error: approximately **67.55**;
- matched-control median absolute error: approximately **3.85**;
- informative-case prior same-player MAE median: approximately **27.92**;
- matched-control prior same-player MAE median: approximately **21.57**.

These are diagnostic contrasts only, not selectors or fitted thresholds.

## 4. REPLAY-GENERATED HYPOTHESES — RANKED AFTER PLAYER ATTRIBUTION

The replay ranked these questions, not promoted mechanisms:

1. Can PRISM arbitrate a real named-player current-role regime change without raw recent-share replacement?
2. Where does strong upstream named-player role knowledge get attenuated before final market opportunity?
3. Can exact pregame teammate availability + finite room competition identify the active role state?
4. Can opponent information be transmitted only through the named player’s actual opportunity subtype?
5. Can named-player trust be reduced when the player’s historical failure mechanism is unstable week to week?

The replay also preserved upstream-but-not-necessarily-transmitted fields including:
- RB/TE depth-role string;
- offensive PROE;
- lead/neutral/trailing game-script state;
- opponent pass EPA;
- opponent rush EPA;
- explosive-play rate allowed.

This ledger is descriptive. It does not authorize a predictor.

Source-blocked/missing objects remain:
- route-rate parity;
- YPRR parity;
- WR-CB assignments;
- defender-injury parity;
- exact historical row-level inactive room identity;
- player-conditioned role-subtype matchup authority.

## 5. CURRENT-ROLE STATE ARBITRATION ANTI-RETEST BOUNDARY

Canonical audit:
`docs/research/PRISM_CURRENT_ROLE_STATE_ARBITRATION_FRONTIER_V1_AUDIT.md`

Disposition:

`PRISM_CURRENT_ROLE_STATE_ARBITRATION_FRONTIER_OPEN__SOURCE_AUDIT_ONLY`.

Meaning:
- role-state arbitration remains a legitimate PRISM frontier;
- a new predictive state rule is **not** authorized from the Week-4/5 outcomes;
- first prove a genuinely new leakage-safe pregame state object;
- do not relabel raw recent share, WR trajectory, RB carry/snap, Stage-1 participation or vacancy transfer as “new.”

Protected:
- raw recent-share replacement CLOSED;
- Opportunity Authority Priority CLOSED;
- opportunity-state disagreement uncertainty CLOSED;
- active-role Stage 1 QUALIFIED ALL;
- Stage 2 QUALIFIED RB rushing / RB receiving / TE receiving;
- QB exact Stage 2 CLOSED;
- WR Stage 2 source-blocked/not scored;
- RB carry/snap protected;
- WR/TE trajectory protected;
- role-elasticity V1 source-blocked.

## 6. RB/TE DISCRETE DEPTH-ROLE TRANSITION SOURCE AUDIT — BLOCKED

Frozen plan:
`docs/research/PRISM_RB_TE_DEPTH_ROLE_TRANSITION_V1_SOURCE_PLAN.md`

Freeze commit:
`dd274610e2545a490c458577e0ed0aee624e9880`

Canonical execution:
- run `39043968618` — SUCCESS;
- exact head `a57b6160a7fea6c31f88a9ea4be3eb198d0349d8`;
- artifact `12105236525`;
- digest `sha256:016ef7db56cd9aa077d39613b9d52062927c122e8c5b6d01df36acc35a4eb492`;
- Issue #535 comment `6235543134`.

Disposition:

`PRISM_RB_TE_DEPTH_ROLE_TRANSITION_SOURCE_BLOCKED`.

Historical source findings:
- RB 2024: role/rank coverage 85.63%, prior-state W3+ 84.31%, 89 transitions; failed identity, conflict and transition-support gates;
- TE 2024: role/rank coverage 89.58%, prior-state W3+ 88.31%, 69 transitions; failed identity/conflict gates;
- RB 2025: no eligible explicit week-tagged depth source;
- TE 2025: no eligible explicit week-tagged depth source.

2026 prospective source probe:
- explicit week-tagged depth source: false;
- latest eligible week: none;
- prospective parity: false.

Do not rescue by:
- lowering source gates;
- using target-game snaps;
- substituting raw recent carries/targets;
- using arbitrary date-only depth snapshots;
- treating current Ourlads roles as historically identical;
- turning the result into vacancy transfer.

## 7. FINITE ALLOCATION TRANSMISSION — ARCHITECTURE GAP CONFIRMED

Frozen diagnostic plan:
`docs/research/PRISM_WEEK5_FINITE_ALLOCATION_TRANSMISSION_V1_PLAN.md`

Canonical latest execution:
- run `39044043517` — SUCCESS;
- exact head `189c23a063c58c74473de27f9033b7a4b411a9cc`;
- artifact `12105037700`;
- digest `sha256:fcdbe6732df9aa1803aa235680ee151dd23c3067221ebfb5dd74bc8a78801da8`;
- Issue #535 comment `6235554464`.

Disposition:

`PRISM_WEEK5_FINITE_ALLOCATION_TRANSMISSION_AUDIT_COMPLETE__ARCHITECTURE_GAP_CONFIRMED__NO_MODEL_CHANGE`.

This is the most important new architecture finding from the replay.

Week-5 preserved player-state snapshot:
- 30 team rooms;
- **30/30 target rooms** exceeded the canonical 0.95 modeled-player target mass cap before conservation;
- raw target mass mean **1.5052**;
- raw target mass range **1.2409–1.7392**;
- mean target scale factor after conservation **0.6352**.

Rushing:
- after canonical top-five selection, **30/30 rush rooms** still exceeded 0.95;
- retained top-five rush mass mean **1.2682**;
- mean rush scale factor **0.7614**.

Exact frozen casebook validation:
- clean non-WR target identities: 6;
- compressed: 6/6;
- clean retained rush identities: 10;
- compressed: 9/10.

### Brock Bowers — exact architecture example

Frozen Week-5:
- upstream `rules_tgt_share = 0.27766642611`;
- final target probability `0.206264830495`;
- clean TE global scale factor approximately **0.742851**;
- relative attenuation approximately **25.7%**;
- implied modeled team target mass before the 0.95 cap `1.278856430209`.

This proves a specific architecture failure mode:

> independently constructed player claims can collectively overbook finite team opportunity, forcing the conservation safeguard to reduce even an upstream named-player role estimate that was already strong/correct.

The conservation cap itself is **not** the problem. Finite team opportunity must be conserved.

Do **not**:
- remove the 0.95 cap;
- permit total player opportunity probability >1;
- boost Bowers or any player post hoc;
- use raw recent share;
- fit a Week-5 normalization repair.

The scientific target is upstream **joint/coherent player entitlement/allocation before conservation**.

## 8. HOW THIS RELATES TO ALREADY-QUALIFIED STAGE 2

Canonical Stage-2 result:
`docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_RESULT.md`

Authority:
- mechanism frozen before scoring at `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`;
- source run `38955368643`, artifact `12070806828`;
- scored run `38957575645`;
- exact scored head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- disposition `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Qualified cells:
- RB rushing;
- RB receiving;
- TE receiving.

Stage-2 mechanics already solve the correct class of problem historically:
1. estimate named-player active-role probability;
2. preserve the existing canonical entitlement `q_i`;
3. condition on the active set;
4. renormalize jointly with an explicit anonymous complement;
5. conserve finite opportunity exactly.

Historical qualification:
- RB rushing strongly improved 2024 development and 2025 independent confirmation;
- RB receiving qualified with smaller/heterogeneous player benefit;
- TE receiving qualified in both seasons;
- QB failed closed;
- WR was not scored because legitimate 2025 WR-R15 independent OOS authority is unavailable.

This is **not production promotion**.

## 9. IMPORTANT SOURCE BLOCK — DO NOT FAKE THE W4/5 STAGE-2 COUNTERFACTUAL

It would be useful to ask:

> If the later-qualified Stage-2 active-set allocation had been applied to the exact Week-4/5 pregame state, would it have prevented the finite-pool attenuation seen in the forensic replay?

That exact retrospective transport is currently:

`SOURCE_BLOCKED_EXACT_PREGAME_Q_AND_AVAILABILITY_ARTIFACTS_EXPIRED__DO_NOT_APPROXIMATE`.

Do not:
- reconstruct target-week q from outcomes;
- substitute later files;
- reverse-engineer availability from target-game snaps;
- claim that Stage 2 would have fixed Bowers/Gibbs/Jones without exact source authority.

The architecture relationship is strong; the exact Week-4/5 counterfactual is not source-legal.

## 10. OLDER CONSERVATION FAMILY REMAINS CLOSED WHERE IT FAILED

Do not confuse this finding with a generic conservation retest.

Older Joint Pass/Receiving Conservation V1:
- run `34081764151`;
- disposition `CONSERVATION_ONLY_SUPPORTED`.

The conservation principle survived, but broad group-mass/joint-combination candidates did not automatically improve individual-player projections.

Receiving attempt-semantics and other generic pool repairs remain closed.

The new finding is narrower and player-first:
**independent player entitlements are jointly incoherent before a valid conservation step**.

Any next mechanism must be explicitly tied to the qualified named-player active-set allocation architecture rather than generic room-mass recalibration.

## 11. EXACT NEXT ACTION

Do not stop at “the cap compresses players.”

The next legal move is a **source/lineage audit for an independent validation bridge from qualified Stage-2 opportunity into market-specific PRISM outcomes**.

Preferred first question:

> Can the exact frozen Stage-1 + canonical q_i + Stage-2 joint allocation semantics be reconstructed on an untouched/pre-authority historical season (preferably 2023) with valid OOS entitlement provenance for RB rushing, RB receiving and TE receiving?

Source audit first.

If exact 2023 source/OOS parity exists:
- freeze one market-specific Stage-2 translation test before scoring;
- preserve qualified Stage-2 mechanics unchanged;
- preserve the existing efficiency authority unchanged;
- test whether the coherent opportunity state improves named-player carries/targets first, then market-specific receptions/yards only under a separately frozen bridge;
- no threshold/window/weight tuning.

If exact source parity does **not** exist:
- fail closed;
- do not score a reconstructed proxy;
- move to prospective mechanics for future weeks without touching the immutable Week-6 locks.

WR remains separate/source-blocked. QB remains closed for exact Stage 2.

## 12. PROTECTED PARALLEL SCIENCE

Still immutable:
- Week-6 Sunday/Monday Stage-2 shadow;
- Week-6 WR opportunity trust lock;
- Week-6 WR efficiency trust lock;
- Week-6 TE trust lock;
- no Week-6 WR dual backfill;
- Weeks 7-10 WR dual-reliability prospective protocol;
- no early Week-6 grading.

Still prohibited:
- paid OddsAPI without explicit user authorization;
- sportsbook upstream;
- raw recent-share replacement;
- generic DvP;
- gate weakening;
- production mutation from forensic evidence.

## 13. CURRENT SCIENTIFIC INTERPRETATION

The Week-4/5 replay did what it was supposed to do: it moved PRISM from a vague “opportunity misses exist” diagnosis to a concrete player-first architecture failure.

For some named players, the model may already possess meaningful role information upstream, but independent player claims are not jointly coherent before entering the finite team opportunity allocator. Conservation then necessarily compresses them.

That is not a position-average problem.

It is a **named-player state + teammate competition + finite allocation coherence problem**, exactly in the center of the PRISM architecture.

Any repair must preserve:
- the player as the scientific object;
- teammate competition;
- finite team opportunity;
- market-specific opportunity;
- independent validation.

No Week-4/5 outcome may be used to fit the repair.
