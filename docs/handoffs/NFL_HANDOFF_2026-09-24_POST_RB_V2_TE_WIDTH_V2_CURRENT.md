# NFL HANDOFF — 2026-09-24 — POST-RB-V2 / TE WIDTH V2 CURRENT

GitHub is canonical over chat memory.

This handoff is intentionally memory-efficient. A new chat should **not** load the whole project history.

## Read only

1. `AGENTS.md`
2. top active checkpoint of `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. Issue #535 from comment `5815680099` onward
5. live state of:
   - `main`
   - `research-te-live-entitlement-efficiency-v1`
   - TE Width V2 run `36040911515`

Do not open older handoffs unless this file explicitly sends you there.

---

# 1. CURRENT PRODUCTION MAIN

Verified current `main`:

`28fd7b4b9a4e4f70a45787d4e09d34939d62e775`

This includes:
- PR #628 — RB Rush+Receiving Conservation V2 production promotion
- PR #629 — continuity-only post-merge handoff update

PR #628 is **MERGED/CLOSED**. Do not reopen.

## RB Rush+Receiving Conservation V2 is production-active

Stable Full Slate public pricing entrypoint:
`scripts/run_pricing_with_full_roster_universe_v3.py`

It now routes to:
`scripts/run_pricing_with_full_roster_universe_v6_production.py`

Canonical Full Slate therefore consumes:
`RB_RUSH_REC_CONSERVATION_V2`

Scope:
- position = RB only
- week != 1
- market = `rush_rec_yards`

Formula:
`rush_rec_yards = final standalone rush_yards mean + final standalone rec_yards mean`

Draw identity:
`combo_draw[i] = final-mean-aligned rush_draw[i] + final-mean-aligned rec_draw[i]`

Explicit no-op:
- FB
- Week 1
- standalone rushing yards
- standalone receiving yards
- receptions
- rush attempts
- QB / WR / TE
- sportsbook inputs upstream

Historical qualification:
- run `36005675177`
- artifact `10809812602`
- n = 2,787 RB player-games, 2024-2025
- MAE **27.6853 -> 25.5718**
- RMSE **39.8437 -> 36.4030**
- bias **+14.3388 -> +9.3891**
- p90 AE **64.7742 -> 57.2604**
- 30+ yard misses **841 -> 768**
- 2024 and 2025 both improved independently

Exact Week-2 observational confirmation:
- MAE **30.3620 -> 28.9014**
- underprojection bias **+20.1339 -> +9.8670**

Production certification:
- final stable-entrypoint cert run `36009823313` = SUCCESS
- preserved paid Full Slate PR replay `36010438890` = SUCCESS
- Repo CI `36010438820` = SUCCESS
- merge SHA `e26fcedade9a94a9634f6ba74558775968573818`

Important interpretation:
**If Full Slate is run from current main for Week 3+, this RB rush+receiving science is in the live model.**

Do not retest or reopen this lane absent a concrete production defect or genuinely new prospective evidence.

---

# 2. ACTIVE SCIENCE TASK — TE-R5P RECEIVING-YARDS WIDTH V2

Active branch:
`research-te-live-entitlement-efficiency-v1`

Current physical branch head:
`a9c7a93ae375f75e926139a0a29675aab89d2fdc`

Active frozen validation run:
`36040911515`

At handoff creation, the run is **IN PROGRESS** and has already passed:
- setup
- dependency install
- frozen mechanics/replay tests
- canonical empirical baseline download
- WR-R15 authority download
- TE-R5P authority download
- exact PR #549 compact replay authority download
- exact PR #549 replay-only source restore
- frozen fold authority verification
- independent 2024/2025 historical input build
- combined schedule/team history
- canonical historical player logs

It is currently rebuilding canonical baseline component traces.

Do not start another duplicate run while `36040911515` is active.

## What is already solved

The old provider-drift blocker is **solved**.

Run `35954272152` proved:
- canonical baseline projection reproduction: PASS
- 51,197 frozen rows = 51,197 reproduced rows
- missing frozen rows = 0
- extra rows = 0
- frozen-row `mc_proj` drift = 0
- reconstructed fold-safe TE-R5P arrays: PASS
- specialist full-stack trace parity against frozen PR #549 authority: PASS

The prior parity failures were mechanical provenance drift, not Width V2 science.

Two exact replay-only source differences caused the drift:
1. `data/manual_name_overrides.csv` had post-#549 suffix/name additions
2. `data/model_ensemble_weights.csv` had post-#549 rec_yards/receptions promoted weights

The Width V2 workflow now restores the PR #549 versions of those two files **inside the Actions workspace only** before reproduction.

Do not undo current production identity/weight improvements globally.

## Precious PR #549 authority

Compact replay authority:
- run `34722725629`
- artifact `10307242156`
- digest `sha256:d5a991bd76df5b053e6411e9873b12bdaada1458592c2586e3c1416c6fe37044`

Original raw-draw artifact `10306649017` is expired.

Exact deterministic replay now regenerates the 2,000-draw specialist arrays. Do not approximate them.

---

# 3. CURRENT WIDTH V2 MECHANICAL REPAIR

After exact baseline + specialist parity passed, run `35954272152` failed only inside the blind candidate validator:

`KeyError: 'game_id'`

Location:
`scripts/research/te_r5p_rec_yards_width_v2.py::_secondary_market_eval()`

Root cause:
- distribution identity `KEYS` intentionally excludes `game_id`
- `evaluate_season()` rebuilt blind detail rows only from those keys
- secondary historical market evaluation legitimately requires `game_id` to join the fixed sportsbook archive
- the projection source already contains `game_id`; the validator simply dropped it

This is a mechanical schema-carry defect, not science.

Repair commits:
- `06bed4ddcad43f014b542fa1f3a4806ba30ccf0c` — carry existing `game_id` into blind detail rows and fail closed if absent
- `a9c7a93ae375f75e926139a0a29675aab89d2fdc` — regression test proving `game_id` carry-through

No change to:
- k formula
- factor search policy
- cohort
- football mean
- projection rows
- sportsbook role
- frozen qualification gates
- parity tolerances
- TE-R5P entitlement
- PR #549 authority

Issue #535 checkpoint:
`5819778386`

---

# 4. WIDTH V2 FROZEN SCIENCE — DO NOT ALTER

Scientific question:

Does one football-only TE receiving-yard width factor fitted on one historical season improve the exact TE-R5P specialist distribution in the other season?

Both directions required:
- fit 2024 -> blind 2025
- fit 2025 -> blind 2024

For fit season:
`k = SD(actual - final_projection) / mean(row_MC_SD)`

For test season:
`widened = final_mean + k * (aligned_draw - final_mean)`

Then re-anchor to exact final mean.

No k search.

No global SD rescale.

No 2026 outcome fitting.

Sportsbook data enters only after k is frozen for secondary calibration evaluation.

## Frozen gates

Candidate qualifies only if all remain true:
1. point MAE invariant <= 1e-10 in both directions
2. max row mean shift <= 1e-8 in both directions
3. mean CRPS strictly improves in both directions
4. 80% coverage gap improves in both directions
5. 90% coverage gap improves in both directions
6. pooled Brier non-worse
7. pooled log loss non-worse
8. sportsbook inputs used to fit k = 0
9. exact PR #549 replay/fold/conservation authority passes

If a primary football-only gate fails:
- exact Width V2 candidate closes
- no k search
- no cap rescue
- no alternate factor search
- no sportsbook-conditioned width
- no global TE receiving-yard SD multiplier rescue

If Width V2 qualifies:
- it is **not automatically production**
- use the predeclared pooled 2024+2025 formula for `k_future`
- build a separate production-integration validation
- certify exact mean invariance, non-TE protection, sportsbook separation, and Full Slate lineage before any merge

---

# 5. EXACT NEXT ACTION

First:
inspect run `36040911515`.

### If SUCCESS
Read:
- uploaded `summary.json`
- `RESULT.md`
- blind row detail
- blind market detail
- fit factors

Then classify strictly from frozen gates:

A. `TE_R5P_REC_YARDS_WIDTH_V2_QUALIFIED`
- document exact metrics/run/artifact/digest in Issue #535
- freeze a separate production-integration plan before touching production
- no 2026 result tuning

B. `TE_R5P_REC_YARDS_WIDTH_V2_FAILED_CLOSED`
- document scientific failure
- no rescue/search
- immediately move to next genuinely-new-information lane

### If FAILURE before science output
Diagnose the exact exception once.

Only repair if it is:
- bounded
- obviously mechanical
- does not change any frozen scientific choice

If it becomes another open-ended plumbing/provenance loop:
- document
- park Width V2
- move immediately to model-improvement research

The user's priority is **model improvement this week**, not infrastructure perfection.

---

# 6. LIVE / PROSPECTIVE RB VACANCY LANE

Separate branch:
`research-rb-vacancy-opportunity-v1`

Do not merge it into Width V2 work.

Frozen V1 hypothesis:
definitive unavailable backfield teammate -> vacated rushing opportunity -> deterministic successor transfer using strict-prior information.

Locked mechanics:
- vacancy requires canonical `definitive_unavailable == 1` / `UNAVAILABLE_*`
- DOUBTFUL / QUESTIONABLE are not vacancies
- unavailable player's vacated share = most recent same-team strict-prior `rush_share_game`
- successor weights = normalized most recent same-team strict-prior `offense_pct`
- opportunity/carries only
- YPC unchanged
- missing prior evidence fails closed
- no sportsbook fields
- no target-game usage features

Week-2 preserved pregame authorities contain **zero qualifying definitive-unavailable RB/FB events**.

Therefore:
`NO_QUALIFYING_PRESERVED_VACANCY_EVENT_YET_PROSPECTIVE_CAPTURE_REQUIRED`

This is not a scientific failure.

Latest clean harness:
run `35996942852` = SUCCESS.

Next legitimate use:
freeze the next 2026 slate **pregame**. If a real definitive-unavailable RB/FB event exists and strict-prior evidence exists, lock the no-outcome state before kickoff and grade it once afterward.

Do not reconstruct historical absences from postgame participation.

Do not broaden to DOUBTFUL/QUESTIONABLE just to create a cohort.

---

# 7. OTHER CURRENT SCIENCE DISPOSITIONS

## TE target-quality / NGS efficiency V1

Source was excellent/dense.

Blind global mean correction failed.

Run:
`36002008896`

Disposition:
`TE_TARGET_QUALITY_EFFICIENCY_V1_FAIL`

Important nuance:
- residual correlation was positive
- RMSE/bias/p90 often improved
- MAE worsened in both blind directions

Do not rescue with exposed alpha search, feature subsets, caps, or high-error routers.

The NGS source remains potentially useful for a genuinely different precommitted reliability/distribution hypothesis, not another global mean correction.

## Public coach / beat-writer intent

The idea remains scientifically plausible and aligned with the user's desire for weekly new information.

Existing V1/V1B source family already exists.

V1B automation result:
`RETRIEVAL_AUTOMATION_NOT_QUALIFIED`

Reason:
generic unauthenticated search-engine HTML was not a reliable scalable retrieval transport.

This is **not** a scientific rejection of coach/beat-writer/public-intent information.

Do not cycle through more generic HTML search engines.

A future restart requires:
- materially different source/search connector, OR
- prospective current-week structured capture

Potential structured semantics:
- planned run/pass emphasis
- committee/workload intent
- player feature intent
- personnel/package change
- tempo
- protection-driven plan
- explicit matchup exploitation

Any signal must be timestamped pre-kickoff and attributable to coach/team/local beat source.

## Practice trajectory

Historical maintained injury feed is not dense enough for Wed/Thu/Fri sequences.

Disposition:
`PRACTICE_TRAJECTORY_SOURCE_NOT_DENSE`

Do not relabel a single weekly DNP/LIMITED field as a trajectory.

A future trajectory lane requires a genuinely daily source.

---

# 8. CLOSED / PROTECTED LANES

Do not reopen:

- PR #625
- PR #626
- PR #628
- PR #629
- duplicate PR #627
- TE-R1
- exposed retrospective M96 RB router variants
- M96A / M96E
- STACK6 team-rush-context slicing
- another historical YPR/YPT/YAC/xYAC/YACOE RB receiving-mean transformation
- QB pass-yards mean retuning

QB passing yards is frozen/prospective.

No global SD rescale.

No sportsbook line upstream.

No paid OddsAPI pull without explicit user approval.

Do not redo Claude's original paid W1/W2 acquisition/backtest.

---

# 9. IMPORTANT RESEARCH PHILOSOPHY FROM THIS SESSION

The useful breakthrough came from checking whether the model obeyed a football identity, not from throwing another feature family at a learner.

RB rush+receiving V2 found:
- a structural projection contradiction
- a parameter-free football correction
- independent 2024 and 2025 improvement
- Week-2 observational confirmation
- exact non-target protection
- clean production promotion

Continue to look for:
- football conservation identities
- incompatible component projections
- structurally impossible states
- missing current-week role/injury/personnel information
- calibrated uncertainty where means are already reasonable
- genuinely new pregame information

Avoid:
- repeated micro-tuning on exposed folds
- feature-search rescues
- scoreboard busywork
- open-ended provider archaeology

---

# 10. CONTINUITY RULE

Every hypothesis, test, mechanical failure, scientific result, rejection, promotion decision, run, artifact, digest, commit, PR and next action must be written to GitHub / Issue #535.

Verify physical remote state after every write.

Distinguish:
- mechanical failure
- source/provenance failure
- scientific failure
- scientific qualification
- production certification

Do not let one be mislabeled as another.
