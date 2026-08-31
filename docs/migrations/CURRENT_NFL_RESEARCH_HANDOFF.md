# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from `NEXT MIGRATION`.

## Repository / current state

- Repo: `dkaps6/imtiredofthis`
- Current research branch: `research-rb-m95k-feed-tendency-carry-ceiling`
- Stable continuity ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No M91-M95K RB research has been promoted to production.
- Phase A production/data cleanup Waves 1-4 is complete.

## Non-negotiable modeling rules

1. Predict real football first.
2. Sportsbook/player-prop lines are downstream benchmark/decision inputs only; never feed them upstream into football projections.
3. No fake/synthetic sportsbook lines.
4. Do not waive production gates because one slice improves.
5. Do not manually boost tail coefficients after seeing validation.
6. Mechanical/source-contract fixes may not change scientific candidate grids or validation gates after exposure.
7. Preserve ordinary-game performance while fixing tails.
8. 2025 has been inspected repeatedly during M91-M95K and is no longer a pristine final confirmation set for new hypotheses. Report it honestly, but use a sealed temporal confirmation before any production promotion.

## Sportsbook / 2026 operational state

- ordinary pushes use `FETCH_LIVE_ODDS=false`;
- explicit live-odds workflows only;
- sportsbook remains downstream;
- 2026 Week 1 live player-prop acceptance is not yet considered fully exercised because preseason no-credit rehearsals could not validate normal live prop inventory.

## QB state — frozen, future audit flagged

Broad QB mean research is frozen after M90.

M90 headline:

- MAE ~`60.63 -> 56.56`
- RMSE `75.63 -> 69.63`
- corr `.173 -> .243`
- 100+ yard misses `81 -> 64`

Frozen PR #498 ensemble:

- MC `.208753`
- ML `.267121`
- State `.524126`

Future narrow QB residual-quality audit is flagged, not active. After RB work, test explicit intrinsic QB quality after controlling for frozen M90: CPOE/accuracy over expectation, EPA/dropback, pressure-adjusted efficiency, sack avoidance, deep-ball/air-yard creation, YAC dependence, turnover tendency, red-zone efficiency, and separation of QB-created ability from protection/receivers/play calling/opponent environment. Do not reopen QB during RB work.

## Research sequence

1. Production/data cleanup — complete.
2. RB refinement — current.
3. WR refinement — after RB closure.
4. Dedicated TE research.

# RB research — durable findings

## M91 / M92

2025 RB-only M91:

- carry MAE `3.494731`
- rush-yard MAE `21.018907`
- rush+rec MAE `25.352140`

M92 oracle decomposition proved opportunity architecture is the primary RB failure. Correct team volume, player share and carries unlock large gains, especially on 20+ workloads.

Never confuse the legacy all-player rushing scoreboard (~7.76-8 yards) with RB-only rush-yard MAE.

## M93 / M93B

Backfield concentration is real. Universal sharpening helps the extreme tail but damages middle slices. Role-aware concentration is useful but insufficient alone.

## M94 / M94B / M94C

Explicit football game-state decomposition improved team rushing opportunity. M94C is the current central carry/opportunity reference during tail research.

2025 M94C:

- team rush MAE `5.812091`
- RB carry MAE `3.411003`
- 20+ carry MAE `7.876590`
- 25+ carry MAE `11.954550`

Legacy all-player rush-yard guard:

- baseline `7.758864`
- M94C `7.762069`
- gain `-0.003205`

Tiny failure; do not waive.

M94D confirmed that sharpening lead-RB share of a carry pool that is itself too small cannot generate realistic 25+ absolute carries.

## M95A-D — offense / defense / quality / environment

M95A validated the user's core football hypothesis: established workhorses perform materially better against weak pregame run defenses than strong ones across 2023-25. A giant correlated defensive feature soup failed, so matchup data must be compact/interpretable.

M95B validated RB + his offense × opponent defense architecture and recovered advanced rushing data including YBC, YAC, broken tackles, expected rushing yards, RYOE, 8+ box rate and time to LOS.

M95C found blocking/environment signals more stable for the mean while runner-created ability is more useful for upside/tails.

M95D recovered FTN motion/RPO/formation/box charting, nflverse participation/personnel and historical defensive missed tackles. Mean projection did not improve, but 100+ rushing-yard discrimination improved in both forward seasons. Retain these as upside context.

## M95E / M95F — workload regime

M95E showed workload-state ranking is strong even when deterministic carry mean remains compressed.

M95F calibrated the class-balanced tail scores:

2025:

- 20+ AUC ~`.846`
- 25+ AUC ~`.844`
- 25+ actual base `1.72%`
- raw score mean `22.51%`
- calibrated mean `3.06%`

Mixture distributions improved true high-workload games but damaged ordinary games. Stable workhorses remained overconfident.

## M95G / M95H — current role and entitlement

M95G proved roster/injury/depth/role information improves 20+ workload discrimination but established that **a vacancy is not a successor**.

M95H tested exact lead identity, >=60% share and >=70% share. Exact lead and >=60% did not validate. >=70% share did:

- AUC `.903118 -> .919599`
- Brier `.096200 -> .090868`

Vacancy >=70% ranking jumped `.784 -> .865`, but probability was too high.

## M95I — deep-concentration + tail integration

Authoritative:

- run `33402566592`
- job `99522191259`
- artifact `9761827238`
- SHA256 `2ec133f4a97b2207d678544e7bde11c98e2d93701e1977536122c5104180fb46`
- disposition `RETAIN_M95I_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

M95I materially improved vacancy calibration and identified a major regime split.

Vacancy 25+:

- AUC `.721739 -> .939130`
- Brier `.008840 -> .008445`
- logloss `.048953 -> .040330`

Stable workhorses remained too bullish. Their unresolved problem is not ownership; it is which specific week converts known lead status into extreme workload.

## M95J — generic stable-week conversion failed

Authoritative:

- run `33405821436`
- job `99533036053`
- artifact `9763096005`
- SHA256 `048bdcbfd1d39659c9a058b35e76ec291622ce4f64e8dd8e77301e4265667b3a`
- disposition `RETAIN_M95J_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

Generic week-specific script/matchup/competition variables looked strong on 2024 selection but failed in 2025 stable workhorses. This suggested player/team/coaching **feed tendency and workload ceiling** differ persistently across workhorses.

25+ frequency in 2025:

- 24 / 1,393 RB player-games = `1.72%`
- 24 / 544 team-games = `4.41%`
- 24 / 272 NFL games = `8.82%`
- roughly one 25+ RB workload every `11.33` NFL games

Thus 25+ is rare for an individual RB but too common league-wide to ignore.

# Latest completed migration: M95K — feed tendency / carry ceiling

## Authoritative run

- Workflow: `M95K RB Feed-Tendency Carry-Ceiling`
- Run: **`33411719023`**
- Job: **`99552629521`**
- Tested SHA: **`daa39544bd895084223532073b5cb9aa2eb4e872`**
- Branch: `research-rb-m95k-feed-tendency-carry-ceiling`
- Artifact: `migration-95k-rb-feed-tendency-carry-ceiling`
- Artifact ID: **`9765397828`**
- Artifact SHA256: **`fd559c2c1cf691f0af9d46c9ce48375be73daca57ade0bfbf11b399612d44671`**
- Artifact size: `1,726,498` bytes
- Execution: success
- Scientific disposition: **`ADVANCE_M95K_TAIL_ARCHITECTURE_TO_SEALED_CONFIRMATION`**
- Production change: `0`
- Sportsbook inputs: `0`
- M94C central carry mean preserved: `1`

Full results: `docs/migrations/M95K_RB_FEED_TENDENCY_CARRY_CEILING_RESULTS.md`.

## M95K architecture

M95K created strictly pregame empirical-Bayes player/team feed priors:

- prior 20+/25+ frequency;
- p90/p95 carry ceiling;
- player career and current-season history;
- team lead-RB multi-season and current-season history;
- sample depth;
- current M94C opportunity and compact game-environment context.

Selected with 2024 only:

- shrink `k=4`
- spec `feed_compact_env`
- logistic `C=.03`

Critical safeguard: **mass-preserving rerank**. Stable-workhorse aggregate M95F 20+/25+ probability mass is held exactly constant; M95K only reallocates it toward the backs/weeks with stronger carry-ceiling/feed evidence.

Vacancy branch stays frozen M95I. Other RBs stay frozen M95F. Central carries remain M94C.

## M95K 2025 research-validation result

Stable-workhorse population: `237` games.

### Stable 20+

M95F:

- actual `21.94%`
- mean probability `29.60%`
- AUC `.581185`
- Brier `.186593`
- logloss `.554301`

M95K:

- mean probability **exactly `29.60%` preserved**
- AUC **`.641164`**
- Brier **`.171528`**
- logloss **`.527166`**

Gains:

- AUC **`+0.059979`**
- Brier **`+0.015065`**

### Stable 25+

M95F:

- actual `4.64%`
- mean `11.10%`
- AUC `.591714`
- Brier `.053017`

M95K:

- mean **exactly `11.10%` preserved**
- AUC **`.612631`**
- Brier **`.051386`**

Gains:

- AUC **`+0.020917`**
- Brier **`+0.001630`**

### Strong individual ceiling signals

In stable workhorses:

- player current-season p95 carries: 25+ AUC **`.7170`**
- player current-season p90 carries: **`.6840`**
- team current-season lead-RB p95 carries: **`.6549`**
- team current-season lead-RB p90 carries: **`.6335`**

This is strong evidence that not all workhorses/teams have the same real workload ceiling.

### Full population

20+:

- AUC `.846474 -> .851667`
- Brier `.062636 -> .059739`
- logloss `.208788 -> .202694`

25+:

- AUC `.844321 -> .851351`
- Brier `.017985 -> .017675`
- logloss `.078526 -> .077260`

M95K passed all pre-specified scientific tail gates:

- stable20 `1`
- stable25 `1`
- all20 `1`
- all25 `1`
- vacancy25 preserved `1`
- stable probability mass preserved `1`
- scientific pass **`1`**

Production gate is still `0` because the inherited M94C legacy all-player rush-yard guard remains `-0.003205`. Do not waive.

## Current scientific interpretation

The RB workload-tail architecture now has two distinct mechanisms that both show value:

1. **Vacancy / role transition:** recipient-specific deep concentration + workload-tail signal from M95I.
2. **Stable incumbent workhorse:** persistent player/team feed tendency and carry ceiling + current football environment from M95K.

M95K is the first stable-workhorse experiment in this sequence to materially improve both ranking and probability scoring without making the group more bullish overall.

This is a probability/distribution architecture, not a reason to manually inflate central carry means.

# NEXT MIGRATION — M95L

Name: **M95L — Sealed Temporal Confirmation of M95K Regime Tail Architecture**

Primary question:

> Does the frozen M95K two-regime tail architecture reproduce its stable-workhorse and full-population 20+/25+ gains on a genuinely independent temporal validation period?

Required freezes:

- M95K `feed_compact_env` feature family;
- empirical-Bayes shrink `k=4`;
- logistic `C=.03`;
- mass-preserving 20+ rerank;
- conditional-ratio + mass-anchor 25+ method;
- frozen M95I vacancy mechanism;
- M94C central carry reference;
- no sportsbook inputs;
- no new feature/weight search.

Preferred confirmation design:

- reconstruct a leakage-safe 2023 late-season stable-workhorse test using earlier-2023 history;
- recover 2023 weekly roster/injury/depth context and the football-environment fields required by the frozen M95K spec;
- fit coefficients only on an earlier temporal window and evaluate a later sealed window;
- do not use 2025 to change M95K after this point;
- if a faithful 2023 reconstruction is impossible, establish another genuinely sealed temporal protocol rather than weakening the test.

Required diagnostics:

- stable 20+ AUC/Brier/logloss;
- stable 25+ AUC/Brier/logloss;
- vacancy 25+ preservation where sample permits;
- full-population 20+/25+ metrics;
- probability-mass preservation;
- central carry unchanged;
- legacy guard status;
- false-positive / false-negative examples;
- no production promotion unless the sealed confirmation passes and production guards are separately resolved.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from the `NEXT MIGRATION` section. Do not reconstruct the project from unrelated memories or restart old research. Preserve all modeling and validation rules in the handoff file.
