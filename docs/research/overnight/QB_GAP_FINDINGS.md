# QB Gap Findings — Overnight Research Survey

**STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.**
Nothing in this document has touched production code, workflows, or data. It is a landscape map plus proposals for you to review.

## What's already in production

- M89/M90 QB pass-yard synthesis (`RB... ` — no, `QB_PASS_SYNTHESIS_V1`, Ridge alpha=20.0, 21 features) is frozen and runs **every week**, not week-1-gated.
- QB C2 (`scripts/modeling/qb_c2_production_adapter_v1.py`) refines distribution shape at pricing time, also every week.
- Documented benchmark (`docs/production/QB_PASS_SYNTHESIS_V1.md`): base ensemble MAE ~57.64 → synthesis MAE ~55.06 yards. Vegas benchmark exists at `data/backtests/historical_market_vegas_benchmark_v1/` and `full_stack_vegas_benchmark_v1/`.

## The landscape: a single, very disciplined diagnostic chain (Sept 6–11, 2026)

Unlike RB, the unmerged QB research (24 branches) is **not** a scattershot of independent ideas — it's one continuous, frozen-protocol diagnostic chain, each step routing to the next, hunting for the source of the residual M89 pass-opportunity miss. Every step is a "diagnostic routing result only" (explicitly not a model change) until the final predictive test. In order:

1. **`research-qb-opportunity-chain-decomposition-v1`** → routes to `TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC` (team dropbacks × attempt conversion × primary-QB share dominates).
2. **`research-qb-team-pass-opportunity-play-rate-decomp-v1`** → routes to `PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`.
3. **`research-qb-pass-rate-state-shared-attribution-v1`** → routes to `FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC` (first down specifically drives the shared miss).
4. **`research-qb-pass-rate-down-distance-decomp-v1`** → `WITHIN_STATE_PASS_PROPENSITY_PRIMARY_DIAGNOSTIC`: it's a **within-down/distance play-selection** problem, not an occupancy problem.
5. **`research-qb-first-down-field-position-decomp-v1`** → field position does **not** explain it; the mechanism is team pass-origin propensity **within** the same field-position zone.
6. **`research-qb-first-down-score-state-decomp-v1`** → score-state occupancy has some shared signal but fails the frozen primary-routing gate. Conclusion: **"week-specific first-down pass/run choice uncertainty inside otherwise comparable football states."**
7. **`research-qb-first-down-choice-economics-source-v1`** → source-eligibility gate `QUALIFIED` for one predictive test: an EPA/success-based `CHOICE_EDGE` signal.
8. **`research-qb-first-down-choice-economics-d1`** → **`FIRST_DOWN_CHOICE_ECONOMICS_D1_FAIL_NO_CONFIRMATION`**. The EPA/success economics signal did not predict the choice. Valid scientific failure, not a mechanical one.
9. **`research-qb-first-down-public-intent-source-v1` / `-v1b`** (most recent, Sept 11) — the next attempt after the EPA-economics failure: a **manual/semi-automated crawl of pregame public reporting** (beat-writer/coach-intent language) as a non-sportsbook signal for first-down play-calling intent. **This is UNFINISHED, not concluded.** V1 paused mid-manual-crawl; V1B is testing whether the same frozen source family can be collected at practical scale via automation. Strict contamination rules apply (no outcomes, no odds, no player props, no residuals). This is a real live thread — do not restart it from scratch; it has a frozen protocol and partial collection already (`docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1_COLLECTION_PREFIX.csv` on that branch).

**Separately**, a player-reliability lane:

- **`research-qb-pd2-player-error-persistence`** → **`NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE`** (explicit reject). None of directional-bias, individual-difficulty, or synthesis-reliability persistence cleared frozen gates across 2024/2025. Explicit do-not-repeat list: *do not feed full-sample QB MAE/bias into the 2026 projection; do not search nearby prior windows; do not lower persistence thresholds; do not create QB-specific corrections from Week-1 market discrepancies.* It explicitly redirects future reliability work toward **"component disagreement, synthesis-correction magnitude/cap behavior, and whether those pregame internal states predict historical out-of-sample error."**
- **`research-qb-pd3-internal-disagreement-reliability`** — launched Sept 7 as the direct follow-up to that redirect (plan frozen → diagnostic implemented → workflow launched), but **the branch stops there — no RESULT.md was ever committed.** This looks like unfinished/interrupted work, not a rejected result. This is the single most "shovel-ready" item in the whole QB backlog: the protocol is already frozen and the workflow already exists, it just never finished executing and reporting.
- **`research-qb-r1-player-context-mechanism-router`** — 2/5 frozen gates passed (N≥300 PASS, W2-18 Spearman>0 PASS; Spearman≥0.15 FAIL, quartile gaps FAIL). Inconclusive/weak, feeds into the PD2 rejection above.

**Also explicitly failed** (do not repeat): `research-qb-pass-rate-anchor-semantic-recalibration-a1` (scientific fail), `research-qb-pass-rate-designed-run-d1` (scientific fail), `research-qb-pass-rate-directional-personnel-source-audit-v1` (no-go), `research-qb-synthesis-opportunity-reparameterization-a1` (failed), `research-qb-team-pass-opportunity-pbp-d2` (no-survivor), `research-qb-team-pass-opportunity-schedule-rest-d1` (failed).

## Explicit stop-rules (verbatim)

From PD2: "Do not: feed full-sample QB MAE/bias into the 2026 projection; search nearby prior windows after this result; lower the persistence thresholds; create QB-specific corrections from the Week-1 market discrepancies."

From the diagnostic chain generally: field position, score-state occupancy, down/distance occupancy, and EPA/success choice-economics are all **ruled out** as explanations for the residual first-down shared miss. Do not re-test these without a genuinely different signal.

## Current open frontier (their own words)

The unresolved QB mechanism is **first-down pass/run play-selection uncertainty that is irreducible from every football-statistical signal tried so far** (occupancy, field position, score state, EPA/success economics). The only channel still open and unconcluded is non-sportsbook **public reporting/intent language** (in progress, paused at automation-feasibility). No numeric predictive signal has yet cleared a frozen gate for this specific mechanism.

## Proposed new research directions (not duplicating the above)

1. **Finish PD3 first.** Not "new," but it's frozen, scoped, half-built, and directly responds to PD2's own redirect. Lowest-risk, highest-alignment next step: does pregame *component disagreement* (MC vs ML vs State vs Bayes spread) or *synthesis-correction magnitude* predict out-of-sample error? This is a genuinely different question from PD2 (which was about player-level persistence) — it's about whether the model's own internal uncertainty is informative, which is a calibration question, not a new mean-feature hunt, so it doesn't violate PD2's stop-rules.
2. **Reframe the frontier as a variance/calibration problem, not a mean-signal problem.** Every mean-signal attempt at explaining first-down choice (occupancy, field position, score state, EPA economics) has failed. Rather than continuing to search for a point-estimate feature, treat "week-specific first-down choice uncertainty in comparable states" as **irreducible aleatoric uncertainty for that specific population** and widen/recalibrate the predictive interval (not the mean) for QBs whose team sits in that diagnosed state. This is validated differently than the prior tests: check whether current 2026 pricing already implicitly captures this via QB C2's distribution-state work, or whether the point synthesis is being priced with a distribution too narrow for exactly this diagnosed population — testable against the existing canonical cohort's calibration (coverage of realized outcomes inside stated percentile bands) without a full walk-forward re-run.
3. **Team/coordinator-identity persistence of the *choice* mechanism (not the error).** PD2 killed *player error* persistence. Nobody has tested whether a team's own historical first-down pass/run choice-volatility (not accuracy, just consistency of tendency) is stable across a coordinator's tenure — a different, narrower claim than what score-state/down-distance decomposition already ruled out (those were about within-game occupancy/zone effects, not cross-season identity of the choice-uncertainty itself). Cheap to check: does the frozen 2023-2025 casebook already have coordinator-tenure fields? If not, this needs new data plumbing and is a bigger lift — flag as medium effort, not free.
4. **Let the public-intent thread run its course before starting anything else in that lane** — it's the only unconcluded numeric-adjacent signal left, and duplicating it would waste the manual collection work already banked on that branch.

## Open questions for you

- Do you want PD3 resumed/finished as the first move (cheapest, most aligned with your own prior redirect)?
- Is the public-intent (V1/V1B) crawl still active, or should it be considered abandoned/paused indefinitely? I did not attempt to continue it — it requires exact protocol adherence and live web research at a scale that deserves your explicit go-ahead given how strict its contamination rules are.
- Proposal #2 (variance/calibration reframe) is the one I'd personally prioritize if you want me to actually build and test something next — it's the only angle here that isn't "try yet another mean-feature and probably fail again."
