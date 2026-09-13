# NFL STUFF — CURRENT RESEARCH HANDOFF — 2026-09-13

## READ THIS FIRST IN THE NEXT CHAT

Repository: `dkaps6/imtiredofthis`

GitHub is canonical over chat memory. Do not make the user re-explain prior work and do not restart closed research.

Read in this order:
1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. Issue #535 (`GPT-5.6 + Claude joint research control room`)
5. PR #561 (`VEGAS_CONFIRMATION_LIKELIHOOD_V1`)

Current `main` at handoff creation: `2d55e512abb562714a86d761123a973928ab2178` (merge of PR #560).

Active research branch: `research-vegas-confirmation-likelihood-v1`
Current branch head: `6db58b14dea6570de7db52c080e4f1857b6b6d7d`
Active PR: #561, open, mergeable, research-only.

## OPERATING CONTRACT

- Football prediction first. Sportsbook/market data is downstream benchmark or explicitly isolated game-script research only; it must not silently teach upstream player projections.
- One authoritative production projection per player/market.
- Freeze hypotheses, cohorts, gates, routing, and source boundaries before viewing results.
- Preserve failures. No post-hoc rescue, threshold tuning, or coefficient retuning after seeing outcomes.
- No target-game outcomes/PBP as pregame features.
- No silent production mutation.
- No extra paid live OddsAPI calls merely for debugging/research.
- Do not invent the old chat label `M108 = 26/26 PASS`; no authoritative repo lineage for that gate was recovered.
- Production science stays frozen unless a prospectively frozen research plan earns promotion.
- GPT-5.6 and Claude collaborate through Issue #535 and GitHub artifacts/PRs, not by unrecorded chat claims.
- Isolated branches; no direct writes to `main`; no force updates.

## PRODUCTION AUTHORITIES — UNCHANGED

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing: `RB_P3_SYNTHESIS_V1` for its qualified Week-1 route
- RB receptions/opportunity: R26
- RB receiving-yard tail/distribution: R22 using frozen R19 assets
- availability-first roles
- sportsbook remains downstream

Architecture remains:
`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MC -> PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`.

## WEEK-1 PRODUCTION INCIDENT — CLOSED

Do not restart.

- PR #523 merged at `f84c6242da02b1804b4b9675c3a3a3e679838e10`.
- Paid Full Slate run `34650067599` got through Steps 1-28; Step 29 exposed the stale invariant.
- Exact paid artifact was replayed through repaired Steps 29-31 with no extra OddsAPI call.
- Post-merge Repo CI `34656484043` PASS.
- No-live Full Slate `34656483980` PASS.
- This is mechanical certification only, never profitability evidence.

## HISTORICAL BENCHMARK / RESEARCH LEDGER — IMPORTANT CLOSED RESULTS

### Identity-clean historical BASE benchmark

PR #541 clean rebuild:
- artifact `historical-benchmark-clean-rebuild-v1`
- 51,197 projection rows
- 32,092 prop rows
- 17,715 graded rows
- identity mismatches = 0

Correct interpretation only:
**Vegas beat the identity-clean historical BASE ensemble benchmark in every tested market/cell. This was not an end-to-end replay of the full current 2026 promoted production stack.**

BASE STRONG examples:
- rec_yards n=5,355 ROI -3.3642%; model MAE 21.602740 vs Vegas 19.999346
- receptions n=5,349 ROI -0.8683%; model MAE 1.706857 vs Vegas 1.546177
- rush_rec n=2,448 ROI -4.4241%; model MAE 33.040482 vs Vegas 27.214461
- rush_yards n=2,449 ROI -3.1452%; model MAE 21.461767 vs Vegas 19.591874
- pass_yards n=721 ROI -5.4737%; model MAE 60.213971 vs Vegas 56.874480

### Fair-probability reconstruction / distribution work

PR #546: 17,715 rows, exactly 2,000 draws/row. Empirical MC improved calibration and ROI versus the legacy Normal translator, but ROI remained negative.

PR #548: prior-season distribution widening improved held-out calibration; research candidate only. Exact widening factors are not automatically transportable to current C2/WR/TE production distributions.

### WR/TE production-order replay

PR #549, run `34722725629`, artifact `10307242156`.

Actual production order replayed M38 -> finite targets -> TE-R5P -> WR-R15 -> joint MC on historically legitimate seasons.

Football accuracy improved on fixed rows, but raw fixed-row ROI did not broadly improve. Final claim:
**production-order WR/TE replay improves football accuracy and probability calibration; the unchanged STRONG rule becomes more selective and less loss-making, but raw betting ROI does not improve across the identical authorized row set.**

`rush_rec` remains a standing problem area.

## WR-R3 PLAYER-HISTORY LANE — CLOSED EXACT CANDIDATE, SIGNAL SURVIVES

Original WR-R3 authority detected three strict-prior signals across 12,396 rows:
- signed bias persistence
- player difficulty / abs-error persistence
- extreme-miss persistence

Combined calibration candidate PR #552:
- run `34726509088`
- artifact `10309091155`
- disposition `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`

2025 primary cohort:
- baseline MAE 22.007275862
- combined MAE 21.797773346
- improvement 0.9519693%, below frozen >=1.00% gate
- miss20 worsened 0.378226 -> 0.385265
- miss30/miss40/under50 improved
- all six strict-prior seasons improved, but frozen primary gate still failed
- tiny one-row parent parity drift remains an UNVERIFIED provenance issue, not a reason to rescue the candidate

Correct taxonomy:
- underlying WR-R3 player-history signal: REPRODUCED, weak but consistent
- exact combined candidate: FALSIFIED by frozen gates
- no production change

Corrected GBM-vs-standardized-Ridge follow-up PR #554 also failed. Checkpoint 44 in Issue #535: both scaled Ridge and GBM worsened 2025 MAE versus M38 baseline despite improving RMSE. Do not reopen by hyperparameter sweeping.

## EMPIRICAL-MC STRONG GATE AUDIT

GPT-5.6 PR #555 diagnosed the remaining empirical-MC decision gate after the legacy `component_sd` defect was separated out.

Critical finding:
- on all 17,715 graded rows, every row satisfying EV >= 5% also satisfied `prob_edge >= 3pp`
- 15,324 EV-qualified rows; the probability-edge half of STRONG rejected zero of them
- therefore STRONG effectively collapsed to a single EV gate on this cohort
- mean empirical-model probability on STRONG rows was ~75.36% versus ~52.04% realized success

This is diagnosis, not authorization to tune a new threshold.

Claude PR #557 later tested per-market isotonic calibration of the gate and obtained `STRONG_GATE_CALIBRATION_NO_IMPROVEMENT`: calibration reduced STRONG coverage but worsened held-out performance in both directions. Do not deploy.

## RB MULTI-SEASON PREREQUISITE — PASSED, BUT PD6 LITERAL P3 BLOCKER REMAINS SEPARATE

GPT-5.6 PR #556:
- canonical run `34733536116`
- artifact `10310481521`
- reconstructed 2021-2024 prior-season-fitted canonical ensemble route and reran original PD2-style strict-prior persistence diagnostics
- all four persistence signals replicated and were positive in every 2021, 2022, 2023, 2024 season

Strong pooled examples:
- carry-error difficulty Spearman ~0.321
- yard-error difficulty Spearman ~0.378
- Q4-vs-Q1 next-game carry-error gap ~+2.62 carries
- Q4-vs-Q1 next-game yard-error gap ~+18.14 yards

Meaning:
- carry uncertainty-width research lane is legitimately unlocked for a newly frozen experiment
- yard uncertainty-width research lane is legitimately unlocked for a newly frozen experiment
- this did NOT resolve the separate literal `PD6 P3-equivalent` blocker; P3 is actually promoted only for its Week-1 route
- do not relabel the current-route replication as literal multi-season P3 validation

## GAME-SCRIPT / VEGAS RESEARCH LANE — MERGED PREDECESSORS

All research-only; none changed production.

### PR #558 — market-implied game script

Corrected result: no meaningful incremental value from tested spread/total representation over a fitted prior-8 rolling baseline for team plays/dropback rate. This is a team-level null only; it did not by itself close position-specific player-usage hypotheses.

### PR #559 — Vegas line calibration

Merged `091651ce76598ba0d2dad8ba0fda89cd7470d4ec`.

2023-2025, 816 games:
- spread/total have moderate direct correlation with realized margin/total
- roughly 10-point MAE
- monotonic bucket ordering
- no useful OOS linear or median/L1 recalibration bias

### PR #560 — confirmed player usage

Merged `2d55e512abb562714a86d761123a973928ab2178`.

After same-side confirmation bug was fixed:
- realized margin -> RB rush-volume effect is large and stable, around Cohen's d ~1.0
- realized total -> WR/TE-volume effect is large, around d ~0.6-0.8
- raw unconditional pregame Vegas labels capture only roughly one-third of that effect
- hindsight restriction to games where the Vegas line was later confirmed within T=7 moves both effects materially toward the ground-truth ceiling

But confirmation uses actual outcome, so #560 is a ceiling diagnostic, not a pregame actionable feature.

## ACTIVE/FINAL PR #561 — VEGAS_CONFIRMATION_LIKELIHOOD_V1

This is the exact point where the previous chat timed out.

### Frozen design

Prospectively frozen in Issue #535 checkpoint 34 and approved by Claude before result exposure.

- train: 2023 + 2024 regular season
- blind test: 2025 regular season
- one game/classification row
- primary confirmation tolerance T=7; T=3 sensitivity only
- frozen model: `StandardScaler -> L2 LogisticRegression`
- selector: Q75 of 2023-24 predicted probabilities; no 2025 threshold tuning
- base features:
  1. signed home spread
  2. absolute spread
  3. posted total
  4. distance to nearest key margin `{3,7,10,14}`
- optional moneyline block only if >=80% complete two-sided coverage in BOTH train and test:
  5. no-vig probability of the spread-favored team
  6. train-only spread-vs-moneyline consistency residual
- primary classification gates:
  - AUC > .55
  - Brier better than train-prevalence constant
  - selected confirmation lift >=10pp
  - adequate class support
- primary downstream gates:
  - margin hypothesis -> RB rush attempts
  - total hypothesis -> WR/TE targets
  - adequate selected support
  - selected effect must beat unconditional Vegas effect AND move closer to ground-truth effect

No line-history/cross-book-consensus feature was claimed because the repo does not contain historical per-book line movement/dispersion data for this period and no new paid historical source was authorized.

### Mechanical lineage

- plan freeze commit: `27e3baa924ca730d6d3a84b0aa1018356e810049`
- evaluator initially committed after freeze
- Codex/implementation review required two fidelity fixes before accepting the full result:
  - moneyline residual must use preregistered favorite moneyline probability
  - downstream gate must explicitly require movement toward ground truth
- Claude then added the blocked test/workflow files on branch head `6db58b14dea6570de7db52c080e4f1857b6b6d7d`
- 10 focused tests were added

### AUTHORITATIVE CI RESULT — COMPLETE

Workflow: `Vegas Confirmation Likelihood V1`
Run: `34759462334`
Head: PR merge ref containing branch `6db58b14dea6570de7db52c080e4f1857b6b6d7d`
Test job: PASS
Diagnose job: PASS
Repo CI at branch head: run `34759462313` PASS
Artifact: `vegas-confirmation-likelihood-v1`
Artifact ID: `10318561868`
Artifact SHA256: `fd684e1619a2829944c6bcd8487f1e747ba6611ae61f37ba12eede38cf68dd58`

Moneyline coverage was 100% in train and test, so all six frozen features were used.

#### Margin confirmation T=7

- train rows 544; test rows 272
- test prevalence 0.345588
- AUC **0.528030** — FAIL AUC > .55
- Brier **0.235380** vs prevalence baseline **0.230536** — FAIL
- selected rows 100
- selected confirmation rate 0.380000
- lift **+0.034412** — FAIL required +0.10
- only class-support gate passed

T=3 sensitivity:
- AUC 0.501346
- selected lift -0.003373
- FAIL

#### Total confirmation T=7

- train rows 544; test rows 272
- test prevalence 0.327206
- AUC **0.508043** — FAIL
- Brier **0.222259** vs baseline **0.220142** — FAIL
- selected rows 63
- selected confirmation rate 0.317460
- lift **-0.009746** — FAIL
- only class-support gate passed

T=3 sensitivity:
- AUC 0.484441
- selected lift -0.003676
- FAIL

#### Downstream player-usage check also goes the wrong way

Margin / RB rush attempts, blind 2025:
- unconditional Vegas Cohen's d = **0.284058**
- selected-high-confirmation d = **0.237685**
- ground-truth d = **1.031372**
- selection made the primary RB signal weaker, not stronger

Total / WR+TE targets, blind 2025:
- unconditional Vegas d = **0.271527**
- selected-high-confirmation d = **0.168246**
- ground-truth d = **0.284567**
- selection again made the primary effect weaker

### FINAL DISPOSITION

- margin: `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE`
- total: `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE`
- overall: `NO_COMBINED_PROMOTION`

This is a clean fail under the prospectively frozen gates. Do NOT rescue it by changing classifier family, probability cutoff, tolerance, feature transforms, or post-hoc subsets from this result.

Important timeout correction: immediately before the old chat hit its limit, GPT-5.6 had independently seen a four-base-feature local preview near chance (roughly margin AUC .525 and total AUC .512). That preview was not canonical. The full six-feature CI run above is canonical and independently confirms the null.

PR #561 is still OPEN at this handoff; no production change is authorized.

## IMMEDIATE NEXT STEPS FOR THE NEW CHAT

1. Refetch `main`, Issue #535, and PR #561 first; do not assume PR state has not changed.
2. Independently cross-audit #561's authoritative artifact/logs against the frozen plan and confirm no gate/formula mismatch.
3. If clean, post the final #561 verdict to Issue #535 and either merge the research-only PR or request correction if an actual implementation defect is found. Do not reinterpret the null.
4. Once #561 is dispositioned, choose the next research lane from surviving evidence rather than reopening failed Vegas-confirmation selection:
   - RB difficulty -> MC-width candidate, now research-authorized by PR #556 but requiring a newly frozen experiment; or
   - another explicitly approved lane from Issue #535.
5. Keep the WR-R1 one-row parent-classification drift as low-urgency provenance backlog unless it blocks a result.
6. Do not restart the Week-1 production repair, WR-R3 combined calibration, corrected WR GBM/Ridge, STRONG isotonic calibration, #558, #559, #560, or #561 after final disposition.

## USER EXPECTATION

The user expects autonomous continuation. Do not ask them to re-explain what happened. Use GitHub as the paper trail, keep Claude/GPT-5.6 coordination in Issue #535, and only stop for a genuinely consequential decision.