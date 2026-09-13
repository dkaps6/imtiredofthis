# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## CURRENT CHECKPOINT — 2026-09-13

GitHub is canonical over chat memory. Do not make the user re-explain prior work and do not restart closed production repair or failed research lanes.

Read in this order:

1. `AGENTS.md`
2. this file
3. `docs/handoffs/NFL_HANDOFF_2026-09-13_CONFIRMATION_LIKELIHOOD_V1_COMPLETE_CURRENT.md` on commit `b506c9caef51c92fa840ece9473db1f23fe3dc5f` / branch `research-vegas-confirmation-likelihood-v1`
4. Issue #535 — `GPT-5.6 + Claude joint research control room`
5. merged PR #561 — `VEGAS_CONFIRMATION_LIKELIHOOD_V1`
6. active draft PR #562 — `RB-PD2 Yard-Difficulty MC-Width V1`

Verified `main` before opening #562:
- `902c19fba5b432e5099b00fd12251d6b8b8349d1`
- this is the merge commit for PR #561

Active research branch:
- `research-rb-pd2-yard-difficulty-mc-width-v1`
- frozen-plan commit: `222c78cb0314e79e0f9ce6e147ab127a30f68f10`
- draft PR #562

## OPERATING CONTRACT

- Football prediction first. Sportsbook data stays downstream unless an explicitly isolated research plan says otherwise.
- Freeze hypothesis, cohort, formula, gates, routing, and source boundaries before result exposure.
- Preserve failures. No post-hoc rescue, threshold tuning, coefficient retuning, or subset fishing.
- No target-game outcomes/PBP as pregame features.
- No silent production mutation.
- No paid live-odds call merely for debugging/research.
- GPT-5.6 and Claude collaborate through Issue #535 and GitHub artifacts/PRs.
- Use isolated branches/PRs; no direct writes to `main`; no force updates.
- Do not invent or require the old chat-only label `M108 = 26/26 PASS` absent concrete GitHub lineage.

## PRODUCTION AUTHORITIES — UNCHANGED

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing: `RB_P3_SYNTHESIS_V1` for its qualified Week-1 route
- RB receptions/opportunity: R26
- RB receiving-yard tail/distribution: R22 using frozen R19 assets
- availability-first roles
- sportsbook downstream only

Architecture remains:
`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MC -> PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`.

No work below changes production.

## WEEK-1 PRODUCTION INCIDENT — CLOSED

Do not restart.

- PR #523 merged at `f84c6242da02b1804b4b9675c3a3a3e679838e10`.
- Exact paid artifact from run `34650067599` was replayed through repaired Steps 29-31 without another OddsAPI acquisition.
- Post-merge Repo CI `34656484043` PASS.
- No-live Full Slate `34656483980` PASS.

## VEGAS / GAME-SCRIPT LANE — FINAL STATUS

Research-only predecessors remain dispositioned:

- PR #557: `STRONG_GATE_CALIBRATION_NO_IMPROVEMENT` — do not deploy/retry isotonic rescue.
- PR #558: tested game-market representation adds no meaningful value over fitted prior-8 team baseline for plays/dropback rate.
- PR #559: Vegas spread/total moderately track realized game state but no useful OOS recalibration emerged.
- PR #560: realized game script strongly affects player usage; hindsight Vegas-confirmed games move toward that ceiling, but confirmation uses actual outcomes and is not pregame actionable.

### PR #561 — COMPLETE, MERGED, INDEPENDENTLY CROSS-AUDITED

PR #561 merged to `main` at:
- `902c19fba5b432e5099b00fd12251d6b8b8349d1`

Authoritative Actions evidence:
- workflow: `Vegas Confirmation Likelihood V1`
- run: `34759462334`
- head: `6db58b14dea6570de7db52c080e4f1857b6b6d7d`
- `test`: PASS
- `diagnose`: PASS
- artifact: `vegas-confirmation-likelihood-v1`
- artifact ID: `10318561868`
- digest: `sha256:fd684e1619a2829944c6bcd8487f1e747ba6611ae61f37ba12eede38cf68dd58`

GPT-5.6 independently downloaded the exact artifact after merge, recomputed the digest, audited the evaluator against the frozen plan, and verified the two pre-result Codex fidelity corrections:

1. moneyline consistency residual uses train-only `abs_spread -> favorite_ml_prob`;
2. downstream qualification requires both a stronger selected effect and movement closer to ground truth.

Canonical blind-2025 result:

**Margin confirmation T=7**
- AUC `0.528030` — FAIL `>0.55`
- Brier `0.235380` vs baseline `0.230536` — FAIL
- selected lift `+0.034412` — FAIL required `+0.10`

**Total confirmation T=7**
- AUC `0.508043` — FAIL
- Brier `0.222259` vs baseline `0.220142` — FAIL
- selected lift `-0.009746` — FAIL

T=3 sensitivity also failed to support either hypothesis.

Downstream primary effects weakened after selection:
- margin -> RB rush attempts: unconditional `d=0.284058`, selected `d=0.237685`, ground truth `d=1.031372`;
- total -> WR/TE targets: unconditional `d=0.271527`, selected `d=0.168246`, ground truth `d=0.284567`.

Final #561 disposition:
- margin: `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE`
- total: `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE`
- overall: `NO_COMBINED_PROMOTION`

This is a clean prospective falsification. Do not reopen it with another classifier, tolerance, cutoff, transform, or post-hoc subset.

GPT-5.6 posted the independent final audit to Issue #535 comment `5653566015`.

## RB PLAYER-DIFFICULTY LINEAGE — SURVIVING POSITIVE EVIDENCE

Original RB-PD2 found strict-prior same-player persistence in both direction and difficulty and explicitly authorized:

- prior carry bias as a possible carry-mean calibration input;
- prior yard bias only after carry effects at the efficiency/yardage layer;
- prior carry/yard **difficulty as uncertainty/MC-width calibration rather than a blind mean offset**.

Repository reconciliation `docs/research/overnight/RB_PD_CHAIN_STATUS.md` confirms:

- PD3, PD4, PD5 all tested mean corrections and failed their frozen gates;
- PD6 was frozen but never implemented/launched because of the 2025 cohort discrepancy;
- **none of PD3/PD4/PD5/PD6 implemented the difficulty -> MC-width leg**.

PR #556 then reproduced the original persistence signal on the non-2025 current-production-equivalent 2021-2024 route:

- canonical run `34733536116`
- artifact `10310481521`
- scoreable rows `4,652`
- yard-difficulty Spearman `0.377996`
- yard-difficulty Q4-Q1 next-game absolute-error gap `+18.143 yd`
- positive direction in all 4 target seasons
- carry difficulty also replicated

Disposition: `MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE_REPRODUCED`.

The carry-width and yard-width research lanes are legitimately unlocked for newly frozen experiments. This still does **not** resolve the separate literal PD6 multi-season P3-equivalent blocker.

## ACTIVE DRAFT PR #562 — RB-PD2 YARD-DIFFICULTY MC-WIDTH V1

Frozen before any candidate distribution result:
- branch `research-rb-pd2-yard-difficulty-mc-width-v1`
- draft PR #562
- plan commit `222c78cb0314e79e0f9ce6e147ab127a30f68f10`
- plan `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md`

### Deliberate scope

V1 tests **rushing-yard width only**.

Carry width remains separately authorized but is deferred because carry dispersion is generated by finite-team multinomial allocation; a valid carry-width candidate must preserve team-rush conservation and is a different architecture hypothesis.

### Frozen yard-width hypothesis

Holding the current-production-equivalent rushing-yard mean exactly fixed, widen only the RB rushing-yard Monte Carlo distribution for players with high strictly-prior same-player yard difficulty.

Core contract:

- exact M95Q run `33450395426` / PR #556 temporal lineage;
- qualifying seasons 2021-2024 only; 2025 forbidden;
- reconstruct exact canonical historical MC: 2,000 draws, seed `42+week`, raw mean must reproduce source `mc_proj` within `1e-8`;
- align canonical MC shape to prior-season-fitted `ensemble_proj` with the same multiplicative mean-alignment contract production pricing uses;
- last8/min4 `prior8_yard_mae`;
- strictly-prior empirical difficulty percentile, minimum 100 earlier reference rows;
- fixed conservative width mapping borrowed prospectively from the repo's prior WR-R3 uncertainty plan: lower half unchanged, top half widens monotonically to max `1.30x`;
- widen around the already-aligned mean, floor at zero, then re-anchor exactly to baseline mean;
- no carry, YPC, share, team volume, mean, receiving, sportsbook, or production change.

Frozen hard gates include:
- exact lineage / no leakage / no sportsbook;
- exact mean neutrality and point-MAE identity;
- pooled CRPS improvement >=0.5% with paired-bootstrap P(improve)>=0.95;
- top-difficulty-quartile CRPS improvement >=1.0%;
- top-quartile 80% and 90% coverage gaps both strictly improve;
- pooled 80/90 coverage gaps non-worse, at least one strictly better;
- 100-yard exceedance Brier strictly better; 50/75-yard Brier non-worse;
- pooled CRPS improves in >=3/4 seasons, 2023/2024 non-worse, high-difficulty CRPS improves in >=3/4 seasons.

Positive disposition would be `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`, still research-only and requiring a separate production-shadow/forward-confirmation plan before any promotion.

Failure is `NO_ACTIONABLE_RB_YARD_DIFFICULTY_MC_WIDTH` and closes this exact mapping with no coefficient/cutoff/subset rescue. Carry width remains separately unanswered.

### Adversarial review gate before implementation/result exposure

GPT-5.6 posted the frozen plan to Issue #535 comment `5653596342` and asked Claude to attack leakage, production alignment, mean neutrality, stop-rule compatibility, and gate validity before implementation.

PR #562 comment `5653596905` separately asks Codex for pre-result plan review.

**Do not expose a candidate #562 result until substantive preregistration defects raised by this pre-result review are resolved prospectively.** Mechanical implementation work may proceed only if it cannot trigger/inspect the candidate output; safest default is to disposition the pre-result design review first.

## CLOSED / DO-NOT-RESTART LANES

- Week-1 Full Slate production repair / identity / rematch / alias incident work
- WR-R3 exact combined candidate
- corrected WR GBM/Ridge rescue
- STRONG isotonic calibration
- PR #558
- PR #559
- PR #560
- PR #561
- RB PD3/PD4/PD5 mean-correction candidates
- PD6 on the already-observed 2025 cohort
- STACK6 team-rush-context slicing
- M95T retrospective rushing-tail tuning family
- R23-R27D historical receiving-efficiency transformations

## IMMEDIATE NEXT STEP

1. Refetch Issue #535 and PR #562 review state.
2. Resolve any genuine pre-result plan defect prospectively without looking at candidate output.
3. If the plan survives review, implement the frozen evaluator/tests/workflow exactly as written.
4. Run once, preserve exact artifact/logs, independently cross-audit all hard gates, and post the disposition to Issue #535.
5. No production change unless a later separately frozen production-shadow/forward-confirmation experiment earns it.
