# RB Workhorse Transition Gate V1

**FROZEN AFTER COUNTS-ONLY ADEQUACY CENSUS, BEFORE ANY CLASSIFIER FITTING OR FEATURE/OUTCOME INSPECTION. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## 1. Why this experiment exists

RB Lane-A V2 is terminal as a production candidate. Its frozen 2024/2025 evaluation showed that applying the HHI-dampened reallocation to every scored loss/vacancy transition worsened overall transition and whole-season rushing-yard accuracy, while the same mechanism produced large descriptive improvements inside the already-preregistered heavy-workload slices (`actual carries >= 20` and `actual rush_yards >= 100`).

That result motivates a new hypothesis rather than a V2 rescue:

> The frozen V2 allocation mechanism may be useful only when a pregame loss/vacancy transition is likely to create a genuinely concentrated/workhorse backfield state.

This V1 gate therefore asks only whether a scored transition is likely, before kickoff, to produce a workhorse workload. It does not change V2's allocation formula, recipient universe, HHI exponent, conservation pool, rush-yard translation, or safety gates.

Authoritative lineage before this plan freeze:

- Lane-A V2 final outcome run: `35169355472`
- V2 outcome head: `9abae2f5f407dea00c489d797aa4af2d57cb223e`
- V2 artifact: `10476248072`
- V2 artifact digest: `sha256:6cf9826e7014e0f8b6c88946d10abd6138da39f1c969337175591310652c7b8f`
- V2 final disposition: `RB_LANE_A_TRANSITION_ALLOCATION_INSUFFICIENT_EVIDENCE`
- Counts-only adequacy commit: `2b64ac2cfc333a4fa1eed714916b0f0d50fa0da3`
- Counts-only Issue #535 comment: `5714642809`
- Prospective design review comments: `5707262504`, `5707304196`, `5707483934`, `5707523675`, `5714574380`
- Initial prospective plan freeze: `7fac65c960422bd87d40f45cff85ba5f4fc55c85`
- Implementation-readiness ambiguity review: Issue #535 comment `5714817311`
- Chronology two-rotation amendment (supersedes Section 4 and its downstream year-references only; see Section 16): Issue #535 comments `5715606827`, `5715781636`, `5715821516`

As of this plan freeze, the prospective implementation clarifications in Section 15, and the chronology supersession in Section 16, no Workhorse-Gate classifier has been fit; no event probabilities have been generated; no feature/outcome relationship has been inspected; no cutoff has been selected; V2 has not been rerun; and 2024-2025 outcomes have not been reopened for this gate experiment.

## 2. Scientific question

Among the same narrow, leakage-safe scored V1/V2 **loss/vacancy transition events**, can a strictly pregame, low-capacity classifier identify workhorse transitions with enough precision and recall to justify using the already-frozen V2 allocation mechanism only when the gate fires?

The gate is not trained or evaluated on the broader `detected_transition` disclosure population. Returns/gains and any transition class on which V2 would never deploy are outside the Workhorse-Gate target population.

## 3. Frozen event population and target

### 3.1 Event population

Use the exact scored V1/V2 loss/vacancy population produced by the already-audited transition machinery (`build_detected_transitions` -> `build_scored_v1_event_population`), **intersected with canonical scheduled target-game team-weeks per Section 17** before any Gate-0.3 event check or feature construction.

The post-transition active room must use the same Gate-0 roster/status semantics as V1/V2, including the same unavailable-status filtering. Do not redefine active-room membership for this experiment.

### 3.2 Binary target

For each scored loss/vacancy transition team-week:

`WORKHORSE_EVENT = 1`

if and only if at least one post-transition active RB/FB/HB records **>=20 actual rushing attempts** in that game; otherwise `0`.

The >=20 boundary is inherited from the already-preregistered Lane-A protected-workload cohort. It was not selected by searching the V2 result.

Actual rushing attempts are target/evaluation data only. They may not enter transition detection, feature construction, scaling, fitting, cutoff selection inputs other than each rotation's own cutoff-year binary labels (Section 16), or any pregame production value.

Do not use `actual rush_yards >= 100` as a training target. That slice mixes workload and efficiency and remains evaluation-only.

## 4. Frozen temporal design (two-rotation, see Section 16)

The chronology uses two independent historical rotations rather than a single fit/calibrate/confirm sequence, so a production-integration review does not have to wait on a full forward 2026 season for its only confirmation evidence (Section 16).

- **2018:** history seed only, for both rotations. Never scored as a gate-development target.

**Rotation A**
- **2019-2020:** classifier fit/development sample only.
- **2021:** probability-cutoff calibration sample only. No coefficient refit on 2021.
- **2022:** one untouched confirmation. No feature, target, model-family, coefficient, scaling, cutoff-grid, threshold-selection rule, or gate may change after 2022 is opened for Rotation A.

**Rotation B**
- **2019-2021:** classifier fit/development sample only.
- **2022:** probability-cutoff calibration sample only. No coefficient refit on 2022.
- **2023:** one untouched confirmation. No feature, target, model-family, coefficient, scaling, cutoff-grid, threshold-selection rule, or gate may change after 2023 is opened for Rotation B.

No architecture, feature, target, metric, cutoff-grid, confirmation-gate, or fail-closed rule may differ between Rotation A and Rotation B.

- **2024-2025:** no-tuning transport/disclosure only, and only if BOTH Rotation A's 2022 confirmation and Rotation B's 2023 confirmation independently pass every gate. These years already informed the hypothesis through Lane-A V2 and are not independent confirmation evidence for either rotation.
- **2026:** eligible for a separate production-integration review once both rotations confirm and the 2024-2025 transport passes its safety stack -- not conditioned on first accumulating a full forward season of live evidence. If the architecture is later promoted under that separate review, permanent 2026 shadow monitoring is mandatory (Section 13).

No year substitution is allowed after either rotation's classifier result is exposed. No rescue or selective use of the surviving rotation if the other rotation fails.

## 5. Counts-only adequacy already disclosed before fitting

The authorized census exposed only event counts, positive counts/prevalence, and source/timing integrity. Those counts may not be used to alter this plan after freeze.

- 2019: 98 scored events, 24 positive (24.5%)
- 2020: 154 scored events, 25 positive (16.2%)
- 2021: 167 scored events, 34 positive (20.4%)
- 2022: 113 scored events, 25 positive (22.1%)
- 2023: 123 scored events, 23 positive (18.7%)

Rotation-pooled fit samples (Section 16):

- Rotation A fit (2019-2020): 252 events, 49 positive.
- Rotation B fit (2019-2021): 419 events, 83 positive.

Frozen adequacy floors: each rotation's own cutoff year and confirmation year individually require >=30 scored events and >=10 positive events; each rotation's pooled fit sample requires both classes present and >=30 positive events total. All six checks -- Rotation A fit/2021-cutoff/2022-confirm and Rotation B fit/2022-cutoff/2023-confirm -- pass on the counts above. This does not qualify the science; it only authorizes implementation.

## 6. Source/timing contract and mandatory Gate-0 repair

All source features must be available strictly before kickoff and must reuse already-audited source/harmonizer paths rather than independently reimplementing roster, injury, depth, history, or walk-forward semantics.

The counts-only census established:

- Gate 0.2 injury integrity: PASS.
- Gate 0.3 roster structural checks available for the earlier seasons with zero observed structural failures.
- Gate 0.3 event requirements 5/6 (current+prior resolvable state; zero outcome columns in event population): PASS.

One mechanical gap is known and must be repaired **before any fit**: Gate 0.3 schedule-coverage requirement 4 is hardcoded to seasons `(2024, 2025)`. Generalize that check to the actual evaluated earlier seasons (2019-2023) without changing its scientific meaning. It must fail closed if schedule coverage is incomplete.

That generalization is plumbing, not a new scientific feature. The implementation review must verify it before the first model fit. The preferred implementation is an additive optional evaluated-season parameter whose default preserves existing V1/V2 behavior; no existing V1/V2 call-site semantics may change.

Raw MC/ML/state component generation for the earlier seasons must use the existing walk-forward path. The audit verified that `component_predictions.py::predict_week` retrains the ML/state tiers from history cutoff to the target season/week and does not load a hidden 2024-fitted bundle.

Sportsbook lines, odds, implied probabilities, closing information, and market-derived features are forbidden upstream.

## 7. Frozen 13-feature contract

Exactly these event-level pregame features are permitted:

1. `active_top_prior3_rb_share`
2. `active_second_prior3_rb_share`
3. `pre_transition_backfield_hhi`
4. `departed_room_prior3_share_sum`
5. `departed_room_max_prior3_share`
6. `active_rb_room_size`
7. `prior_rb_room_size`
8. `mc_projected_plays`
9. `mc_dropback_rate`
10. `raw_mc_team_rush_volume`
11. `historical_rb_room_rush_share`
12. `raw_mc_top_active_rush_att`
13. `raw_mc_second_active_rush_att`

No feature additions, substitutions, feature-selection loop, or outcome-driven deletions are permitted after fitting starts.

The previously proposed `active_out_doubtful_rb_count` is deliberately excluded because the V1/V2 unavailable filter removes OUT/DOUBTFUL/IR/PUP players before the active-room definition, making the proposed feature effectively degenerate.

The following exact derived redundancies are deliberately excluded: `active_top_minus_second_prior3_rb_share`, `rb_room_contraction`, and `raw_mc_top_minus_second_rush_att`.

Feature semantics:

- `prior3_rb_share` and `pre_transition_backfield_hhi` must reuse the existing STACK2/V1/V2 strictly-prior history machinery unchanged.
- `active_top_prior3_rb_share` and `active_second_prior3_rb_share` are the largest and second-largest existing-helper `prior3_rb_share` values among members of the post-transition active room. If the active room contains exactly one member, the second value is structurally `0.0`. If the post-transition active room contains zero members, the entire experiment fails closed before fitting/scoring. Existing helper behavior for a player with no prior history remains unchanged; this clarification does not introduce a new missing-history imputation.
- `departed_room_*` is computed only from players present in the immediately prior resolvable pre-transition room but absent/unavailable from the post-transition active room for the scored loss/vacancy event, using the transition detector's own audited current/prior states.
- `active_rb_room_size` and `prior_rb_room_size` are counts under those same room definitions.
- `mc_projected_plays` and `mc_dropback_rate` are the raw pregame MC team-week values emitted by the historical walk-forward **canonical Build A**. Build B exists only for same-job authority/parity verification and may not be used as a feature source.
- `raw_mc_team_rush_volume = mc_projected_plays * (1 - mc_dropback_rate)`.
- `historical_rb_room_rush_share` reuses the V1/V2 strictly-prior trailing-3 team RB-room rush-share calculation unchanged.
- `raw_mc_top_active_rush_att` and `raw_mc_second_active_rush_att` are the largest and second-largest raw canonical Build-A `mc_proj` values for `market == rush_att` among exact-identity matched members of the post-transition active room. If exactly one matched active player exists, the second value is structurally `0.0`. If no active player can be matched to a raw MC rush-att row, the **entire Workhorse-Gate V1 experiment fails closed** before classifier fitting/scoring; do not impute a projection and do not drop only that event.
- Existing strictly-prior history helper behavior for players with no prior history must be reused unchanged. Do not introduce a new missing-history imputation rule for this gate.
- The existing role-weight/HHI helper may be extended mechanically to expose the already-computed per-player strictly-prior role-share frame needed for the two `departed_room_*` features. Such an extension must not alter any existing role-share or HHI formula, event membership, or V1/V2 output.

Every scored event in the frozen event population must produce one complete finite 13-feature row before any fit/calibration/confirmation step. A source, identity, coverage, or timing defect in **any** scored event is a pre-outcome integrity failure for the whole V1 experiment. Event-level exclusion is forbidden. The failing event keys and exact reasons must be preserved in the evidence artifact.

## 8. Frozen classifier family

Use exactly one classifier family, fit independently once per rotation (Section 16):

- `StandardScaler` fit on the rotation's own fit years only (Rotation A: 2019-2020; Rotation B: 2019-2021);
- L2 logistic regression only;
- `C = 1.0`;
- `class_weight = 'balanced'`;
- `solver = 'liblinear'`;
- `fit_intercept = True`;
- `max_iter = 1000`;
- `random_state = 42` where applicable.

Each rotation's scaler and classifier coefficients are fit once, on that rotation's own fit years only, and remain frozen for that rotation's own cutoff year, that rotation's own confirmation year, and -- if both rotations confirm -- the shared 2024-2025 transport and any 2026 shadow application. Rotation A's fitted model is never applied to Rotation B's years or vice versa outside the shared 2024-2025 transport step.

No trees, boosting, neural networks, splines, polynomial expansion, interaction search, alternate regularization sweep, alternate C search, model-family tournament, isotonic/Platt recalibration, or feature-selection loop is permitted in either rotation.

## 9. Frozen cutoff protocol (per rotation)

After a rotation's fit-year scaler/classifier is frozen, generate that rotation's own cutoff-year probabilities once (Rotation A: 2021; Rotation B: 2022).

Candidate probability cutoffs are exactly:

`{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}`.

Select the cutoff that maximizes **F0.5** on that rotation's cutoff year, weighting precision more heavily than recall because V2's broad failure mode was false-positive activation of the aggressive allocation mechanism. The grid, metric, and tie rule are identical for both rotations.

For any candidate cutoff producing zero predicted positives, record precision, recall, and F0.5 as `0.0` for cutoff-selection purposes (`zero_division=0` semantics). If multiple cutoffs tie exactly on F0.5, choose the higher cutoff, including a possible all-zero tie. Do not replace the frozen tie rule after seeing either rotation's cutoff-year probabilities.

Do not refit or rescale either rotation's model using its cutoff year. Do not change the grid or optimization metric after seeing either rotation's cutoff-year probabilities. Preserve the complete cutoff table for both rotations in the evidence artifact.

## 10. Frozen confirmation gates (per rotation)

Open each rotation's confirmation year exactly once, using that rotation's own unchanged fit-year model and its own cutoff-year-selected cutoff (Rotation A: confirm on 2022; Rotation B: confirm on 2023).

All gates below must pass **independently in each rotation**:

1. Source/timing/leakage integrity PASS, including generalized earlier-season schedule coverage and complete finite 13-feature constructibility for every scored event in that rotation's confirmation year.
2. Adequacy: >=30 scored loss/vacancy transition events and >=10 positive `WORKHORSE_EVENT` events. Already known to pass for both (Rotation A 2022: 113/25; Rotation B 2023: 123/23) and may not be used to alter any other rule.
3. Precision exceeds that confirmation year's unconditional workhorse prevalence by **>=10 percentage points**.
4. Precision **>=0.60**.
5. Recall **>=0.25**.
6. ROC-AUC **>0.60**.
7. PR-AUC **> positive-class prevalence**.
8. `sportsbook_inputs_used = 0`.

If a rotation's frozen cutoff produces zero predicted positives on its confirmation year, precision is treated as undefined for scientific confirmation and gates 3 and 4 fail closed for that rotation; record disposition detail `PRECISION_UNDEFINED_NO_PREDICTED_POSITIVES`. Do not rescue by changing that rotation's cutoff.

If either rotation fails any required gate, final V1 gate disposition is `RB_WORKHORSE_TRANSITION_GATE_V1_NOT_QUALIFIED` (or `INSUFFICIENT_EVIDENCE` only for a true adequacy failure) and the experiment stops before 2024-2025 transport. A pre-outcome constructibility/source/timing failure must retain its specific integrity failure label rather than being relabeled as scientific nonqualification. No rescue fitting, feature change, threshold change, alternate classifier, event exclusion, rotation substitution, or selective use of only the surviving rotation is permitted.

Only if BOTH Rotation A and Rotation B independently pass every gate above does disposition become `RB_WORKHORSE_TRANSITION_GATE_V1_CONFIRMED_EARLY_OOS`, which authorizes only the frozen 2024-2025 transport/disclosure described below. It does not authorize production.

## 11. Frozen downstream router if both rotations confirm

For each scored loss/vacancy transition event:

- `gate = NO` -> exact current promotion comparator / production path, with no Lane-A modification;
- `gate = YES` -> apply the already-frozen Lane-A V2 production-scoreable recipient universe and HHI-dampened allocation mechanism unchanged.

The workhorse gate may only route between those two already-defined arms. It may not alter V2's formula, exponent, pool, recipient filtering, incumbent-YPC translation, or any prediction after the gate decision.

## 12. 2024-2025 transport/disclosure

Only after both Rotation A's 2022 confirmation and Rotation B's 2023 confirmation independently pass every gate may the fully frozen gate+router be transported onto 2024 and 2025 once.

Those years are hypothesis-generating/exposed data from Lane-A V2. They therefore cannot serve as independent qualification evidence and may not be used to retune anything.

The transport must report the existing Lane-A safety table unchanged, including at minimum transition rushing-yard MAE, protected workload/tail slices, p90/catastrophic error, player-cluster and crossed bootstrap diagnostics, stable-row identity, conservation, per-season results, and whole-season deployable safety.

2024-2025 transport may be labeled supportive or unsupportive descriptively, but a supportive result cannot promote the gate. An unsupportive result closes this V1 architecture without rescue.

## 13. Production boundary / 2026

No part of this plan changes production.

If both rotations confirm and the 2024-2025 transport is supportive, the architecture may become eligible for a separately approved production-integration review during 2026. That eligibility is not conditioned on first accumulating a full forward 2026 season of live evidence -- Section 16 supersedes the earlier single-rotation requirement that such forward evidence precede any production-integration proposal. The gate, cutoff, feature list, and V2 router must remain unchanged entering that review.

If the architecture is promoted under that separate review, permanent 2026 shadow monitoring is mandatory afterward.

No sportsbook information may enter the football-side gate or allocation mechanism. Market comparison, if any, remains downstream evaluation only.

## 14. Stop rules

- V1/V2 historical results remain terminal and are never rewritten.
- No post-result rescue of Workhorse-Gate V1.
- No feature additions/deletions after fit begins.
- No cutoff-grid or F0.5 rule change after either rotation's cutoff-year probabilities are exposed.
- No parameter/model-family changes after either rotation's confirmation year is opened.
- No scored-event exclusion for feature/source/identity defects.
- No rotation substitution after either rotation's confirmation year is opened.
- No rescue or selective use of a surviving rotation if the other rotation fails.
- No 2024/2025 tuning.
- No production mutation from this research branch.
- Mechanical defects may be repaired only when the scientific meaning is unchanged, with the defect, repair, before/after lineage, and rerun authority documented explicitly.

## 15. Prospective implementation clarifications frozen before fitting

These clarifications resolve the ambiguities raised in Issue #535 comment `5714817311`. They were committed before classifier fitting, before probability generation, before cutoff selection, and before any 2023 model result was exposed. They do not use model outcomes and therefore do not constitute post-result rescue tuning.

1. **Single-member active room:** for the prior3-share top/second pair, exactly one active member means second share is structurally `0.0`, mirroring the already-frozen raw-MC pair. Zero active members fail the entire experiment.
2. **Meaning of fail closed:** any scored 2019-2023 event that cannot produce the complete finite 13-feature vector terminates V1 before scientific scoring. No event is silently or explicitly excluded to make the model run.
3. **Raw component authority:** all raw MC feature values come from canonical same-job Build A. Build B remains parity verification only.
4. **Role-share exposure helper:** exposing already-computed per-player prior3 shares from the existing role/HHI machinery is permitted as a mechanical implementation extension only; formulas and semantics remain unchanged.
5. **Zero predicted positives:** 2022 cutoff-table metrics use zero-division=0 semantics; on 2023, zero predicted positives is a fail-closed scientific confirmation state with `PRECISION_UNDEFINED_NO_PREDICTED_POSITIVES`.
6. **Gate-0 schedule repair:** generalize requirement 4 with an additive evaluated-season parameter/default-preserving implementation before fitting; existing V1/V2 semantics must remain unchanged.
7. **Redundant prevalence gate:** the already-known 2023 prevalence makes the `precision >= prevalence + 10pp` gate weaker than the fixed `precision >= 0.60` gate in this specific year. Both remain frozen exactly as written; no gate is removed or weakened.
8. **Authorization boundary:** after this clarification commit, implementation may be reviewed and coded, but classifier fitting remains unauthorized until the reviewer returns `IMPLEMENTATION_PLAN_REVIEW_PASS` against this amended plan.

## 16. Chronology supersession (two-rotation amendment)

This section supersedes ONLY the single-rotation chronology and its downstream year-references in Sections 3.2, 4, 5, 8, 9, 10, 11, 12, 13, and 14 as amended above. The event population (Section 3.1), 13-feature contract (Section 7), ambiguity resolutions (Section 15), classifier hyperparameters, cutoff grid/F0.5 rule, source/timing rules, sportsbook prohibition, and unchanged V2 router are NOT altered by this section.

Authoritative lineage for this amendment:

- Superseding chronology comment: Issue #535 `5715606827`
- State-mismatch hold (correctly declined to guess rather than silently redo or silently override): Issue #535 `5715781636`
- Reconciliation confirming supersession scope: Issue #535 `5715821516`

Rationale: the single-rotation design's original Section 13 treated a genuine forward 2026 season as a hard prerequisite for any production-integration proposal, which risked converting this research lane into a live-season-only path inconsistent with the `live production authority + permanent shadow laboratory` operating model. Two independent historical rotations -- Rotation A (fit 2019-2020 / cutoff 2021 / confirm 2022) and Rotation B (fit 2019-2021 / cutoff 2022 / confirm 2023) -- provide two independent out-of-sample confirmations without waiting on 2026 to accumulate first. 2024-2025 remain non-independent transport/disclosure only, exactly as before. 2026 becomes eligible for a separate production-integration review once both rotations confirm and the 2024-2025 transport passes, rather than a precondition for even proposing that review; permanent 2026 shadow monitoring remains mandatory if the architecture is later promoted.

No rescue or selective use of a surviving rotation if the other fails. No rotation substitution after either rotation's confirmation year is opened. This section does not authorize any fitting; it only amends the chronology contract prospectively, before any classifier has been fit, any probability generated, or any cutoff selected.

## 17. Schedule-domain correction to the scored event population

Amends ONLY Section 3.1's event-population definition, by adding a filtering step. It does not alter `build_detected_transitions`/`build_scored_v1_event_population` (the frozen V1/V2 transition trigger logic), the binary target (Section 3.2), the 13-feature contract (Section 7), the classifier family (Section 8), the cutoff protocol (Section 9), the confirmation gates (Section 10), the router (Section 11), or the two-rotation chronology (Section 16).

**Defect found**: the first real two-rotation-evaluation CI run (`35243854637`, head `e8f5532e`) reached `WORKHORSE_GATE_V1_WHOLE_EXPERIMENT_FAIL_CLOSED` -- 5 of 655 scored events (2019-2023) could not produce a complete finite 13-feature row, triggering the whole-experiment fail-closed rule (Section 15 point 2) before any outcome was opened. Auditing the artifact (Issue #535 comment `5718356931`) showed all 5 were non-game team-weeks, not a genuine feature-construction gap:

- 2019 W18 NO, 2020 W18 BUF, 2020 W18 NO, 2020 W18 LAR -- 2019 and 2020 regular seasons ended at Week 17; these are fictitious post-season-end roster-snapshot rows.
- 2020 W5 DEN -- a COVID-postponed, bye-shifted week with no scheduled game.

The frozen transition detector (Section 3.1's `build_detected_transitions`) can flag a membership/status change on any roster-snapshot week in its 1-18 range; it does not itself verify the target `(season, week, team)` is an actual scheduled game, because Lane-A V1/V2 only ever evaluated 2024/2025, both full 18-week seasons with no such gaps. A gate that predicts whether an RB gets 20+ carries "in that game" cannot be scored against a team-week with no game, and Build-A production MC inputs correctly do not exist for one -- which is exactly why those 5 rows failed feature construction.

**Correction**: before Gate-0.3 event checks (`gate03_event_report`) and before feature construction, intersect the scored loss/vacancy population with canonical scheduled target-game team-weeks, built by reusing (unchanged) `rb_lane_a_gate0_v1.build_team_week_kickoffs`/`get_nfl_schedule` -- the same already-audited schedule path used elsewhere in Gate 0. Implemented in `scripts/backtest/rb_workhorse_gate_v1_event_population.py::filter_scored_events_to_scheduled_games`. Every excluded row is preserved with reason `NO_SCHEDULED_TARGET_GAME` in a disclosure artifact -- never silently dropped -- and an explicit integrity assertion proves every retained row is a scheduled game team-week.

**Corrected counts-only census** (2019-2023, run before any fitting, per Section 5's discipline): 655 scored events pre-correction -> 650 retained (5 excluded, matching the diagnosis exactly). Per-season counts after correction: 2019 n=97/pos=24 (24.7%), 2020 n=150/pos=25 (16.7%), 2021 n=167/pos=34 (20.4%), 2022 n=113/pos=25 (22.1%), 2023 n=123/pos=23 (18.7%). All seasons remain far above the adequacy floor (>=30 events, >=10 positive); the correction does not threaten adequacy in either rotation.

Lineage: Issue #535 comments `5718111356` (Run 3 fail-closed result reported), `5718356931` (GPT-5.6's artifact audit, root cause, and authorization of this correction), this amendment.

This section does not authorize any fitting. The two-rotation classifier run remains on hold pending review of this amendment, per the same comment.
