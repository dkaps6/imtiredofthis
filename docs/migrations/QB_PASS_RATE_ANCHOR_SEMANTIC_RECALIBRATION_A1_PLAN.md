# QB Pass-Rate Anchor Semantic Recalibration A1 — Frozen Plan

## Status

Frozen before any A1 candidate is scored.

This is a narrow architecture/semantic reconciliation experiment. It is **not** a generic QB mean-feature search, not a replay of M42 dynamic pass-rate history, and not a production change.

## Why this lane is reopened

The current promoted stack uses a fixed `0.57` team pass-opportunity/dropback-rate anchor.

Prior work established:

- M20/M21 selected/promoted `0.57` under the historical Monte Carlo path.
- M42 later tested trailing team pass-rate redistribution around `0.57` while explicitly recentering every candidate back to a league mean near `0.57`; that family is closed and will not be repeated here.
- M89 corrected official passing-attempt semantics. nflverse PBP `pass_attempt` includes sacks; official attempts exclude sacks, while scrambles remain dropbacks but not official attempts. Production therefore now uses `official attempts / (official attempts + sacks + QB scrambles)` as `pass_attempts_per_dropback`.
- The fixed `0.57` anchor was not re-swept after that M89 semantic correction.
- The authoritative play/rate decomposition found pass-opportunity rate is a larger unresolved opportunity component than total offensive plays, and the corrected 2024 M89 trace shows systematic under-centering from a fixed `0.57` baseline.

This is sufficient architecture-specific evidence to test whether the **league anchor itself** remains calibrated under the corrected M89 attempt-conversion contract.

## Anti-reinvention boundary

This experiment does **not** test:

- trailing 3-/5-game pass rate;
- PROE;
- score-state / lead-trail modifiers;
- coach/game-script heuristics;
- personnel/injury adjustments;
- schedule/rest;
- QB designed-run adjustments;
- pressure, explosives, receiver matchup, or any other new feature;
- sportsbook lines/odds;
- fitted coefficients or model families.

M42-style within-league redistribution remains closed. A1 changes only the single league anchor.

## Immutable parent evidence

### Play/rate decomposition

- Run: `34535405829`
- Head: `36f100799c81769d1ade998c7304d932e5914b44`
- Artifact: `10175200512`
- Name: `qb-team-pass-opportunity-play-rate-decomp-v1`
- Digest: `sha256:f2a79b330c7cc6f24d6b83479806478e5534afd2bd09961d37304a2c247b6bb4`

### Opportunity-chain decomposition

- Run: `34523313743`
- Head: `8916d1d4537c6d6f9470ee4d97a94fdde816eb6d`
- Artifact: `10170531084`
- Name: `qb-opportunity-chain-decomposition-v1`
- Digest: `sha256:75cd32198caf7d9cbf193d5769e5762acce53352f2b2626cd92a9b117e75dafd`

No parent sportsbook artifact is permitted.

## Development / confirmation split

- Development: **2024 only**, exactly the 444 corrected M89 QB-game rows.
- Confirmation: **2025 only**, exactly the 440 corrected M89 QB-game rows, and remains sealed unless every A1 development advance gate passes.
- If any A1 development gate fails, 2025 is not scored or summarized for this candidate family.

## Frozen candidate grid

The only candidates are constant league dropback/pass-opportunity anchors:

`0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63`

Rationale for one-directional grid: the M89 semantic correction weakly lowers official attempts per dropback relative to the pre-correction nflverse `pass_attempt` conversion because sacks are removed from official attempts. Therefore a reconciliation of the old calibrated workload, if required, should not require an anchor below the already-promoted `0.57`. `0.63` is included only to bracket the correction; a development winner at the upper boundary is **not** confirmation-eligible.

No interpolation, coefficient fitting, post-result grid expansion, or per-team adjustment is allowed inside A1.

## Frozen formulas

For each 2024 row and candidate anchor `r`:

- `candidate_rate = r`
- `candidate_D = pred_plays * r`
- `candidate_qb_attempts = candidate_D * pred_C * pred_S`
- `candidate_mechanics_pass_yards = candidate_qb_attempts * pred_ypa`

Where:

- `pred_plays` comes from the immutable play/rate decomposition;
- `pred_C` is the corrected strict-prior M89 official-attempts-per-dropback conversion;
- `pred_S` is the frozen QB attempt share;
- `pred_ypa` is the frozen pregame YPA component from the opportunity-chain artifact.

The promoted M89/M90 `football_synthesis` passing-yard mean is **not changed, refit, or used to select the anchor in A1**.

## Frozen selection rule

Rank candidates lexicographically by:

1. lowest 2024 pass-rate MAE;
2. lowest 2024 team pass-opportunity (`D`) MAE;
3. lowest 2024 QB-attempt MAE;
4. lower anchor if still exactly tied.

The baseline is always `0.57`.

## Metrics

For baseline and every candidate report:

- pass-rate MAE, RMSE, bias, correlation, p90 absolute error;
- team pass-opportunity MAE, RMSE, bias, correlation, p90 absolute error;
- QB-attempt MAE, RMSE, bias, correlation, p90 absolute error;
- QB-attempt 8+ and 10+ absolute-miss rates;
- deterministic mechanics pass-yard MAE, RMSE, bias, correlation, p90 absolute error;
- anchor adjustment versus `0.57`;
- paired bootstrap gain distributions for pass-rate MAE, team-D MAE, and QB-attempt MAE using 5,000 draws and fixed seed `5701`.

## Development advance gates

Every gate must pass. No discretionary rescue.

### Integrity

1. Exactly 444 2024 rows and unique `(season, week, team, player_clean_key)` keys.
2. 2025 candidate outcomes are not scored or summarized.
3. Zero sportsbook inputs.
4. Zero model fitting.
5. Zero production changes.
6. Candidate grid exactly equals the frozen seven anchors.
7. Existing baseline identities reproduce within `1e-9`:
   - `pred_D == pred_plays * 0.57`;
   - `pred_attempts == pred_D * pred_C * pred_S`.
8. `football_synthesis` is read only for identity/audit if needed and is never modified or used for candidate selection.

### Scientific

For the selected development winner versus `0.57`:

1. Winner anchor must be `> 0.57`.
2. Winner anchor must be `< 0.63` (upper-boundary winner means bracket unresolved; no confirmation).
3. Pass-rate MAE gain must be `>= 0.0040`.
4. Team pass-opportunity MAE gain must be `>= 0.20` opportunities.
5. QB-attempt MAE gain must be `>= 0.15` attempts.
6. Absolute pass-rate bias must improve by at least 25%.
7. Absolute team-D bias must improve by at least 25%.
8. Absolute QB-attempt bias must improve by at least 20%.
9. Pass-rate p90 absolute error must be non-worse.
10. Team-D p90 absolute error must be non-worse.
11. QB-attempt p90 absolute error must be non-worse.
12. QB 10+ attempt miss rate must be non-worse.
13. Mechanics pass-yard MAE may worsen by at most `0.25` yards.
14. Mechanics pass-yard p90 absolute error may worsen by at most `1.0` yard.
15. Paired bootstrap `P(MAE gain > 0)` must be `>= 0.95` for pass rate.
16. Paired bootstrap `P(MAE gain > 0)` must be `>= 0.95` for team D.
17. Paired bootstrap `P(MAE gain > 0)` must be `>= 0.90` for QB attempts.
18. At least one immediately adjacent grid anchor must also improve both pass-rate MAE and team-D MAE versus `0.57`; this prevents a knife-edge winner.

If any gate fails, disposition is `QB_PASS_RATE_ANCHOR_SEMANTIC_A1_FAIL_NO_CONFIRMATION` and the anchor remains `0.57` in production.

## Confirmation rule if development passes

Only after a full A1 development pass:

1. Freeze the exact selected anchor in a separate confirmation plan/commit.
2. Score exactly the 440 2025 corrected M89 rows with no retuning.
3. Require positive pass-rate, team-D, and QB-attempt MAE gains; non-worse p90/tail guardrails; improved absolute bias; and paired-bootstrap support before any production proposal.
4. A production change would still require a separate integration experiment through the current M89/M90 + receiver-conservation stack. A1 alone can never promote production.

## Decision interpretation

A1 is testing a semantic calibration defect, not trying to beat M89/M90 passing-yard means directly.

- If `0.57` still wins or gates fail, the fixed anchor survives corrected semantics and research returns to genuinely new pass-opportunity information.
- If a higher anchor passes and confirms, the next task is to integrate that corrected team opportunity pool through QB attempts and shared receiver opportunity, then revalidate M89/M90 and cross-position conservation without using sportsbook information.
