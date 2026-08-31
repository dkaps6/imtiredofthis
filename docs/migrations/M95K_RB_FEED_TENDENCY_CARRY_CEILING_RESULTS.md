# M95K — Stable Workhorse Feed-Tendency / Carry-Ceiling Results

## Authoritative run

- Workflow: `M95K RB Feed-Tendency Carry-Ceiling`
- Run: `33411719023`
- Job: `99552629521`
- Tested SHA: `daa39544bd895084223532073b5cb9aa2eb4e872`
- Branch: `research-rb-m95k-feed-tendency-carry-ceiling`
- Artifact: `migration-95k-rb-feed-tendency-carry-ceiling`
- Artifact ID: `9765397828`
- Artifact SHA256: `fd559c2c1cf691f0af9d46c9ce48375be73daca57ade0bfbf11b399612d44671`
- Artifact size: `1,726,498` bytes
- Execution: success
- Scientific disposition: `ADVANCE_M95K_TAIL_ARCHITECTURE_TO_SEALED_CONFIRMATION`
- Production change: `0`
- Sportsbook inputs: `0`
- M94C central carry estimate preserved: `1`

## Hypothesis

M95J showed that generic week-specific football context does not reliably distinguish monster-workload weeks for established workhorses. M95K tested a different hypothesis: some players, teams and coaching environments have a persistent, leakage-safe propensity to feed a lead back into the 20+/25+ carry range, and that historical carry ceiling can help rerank otherwise-similar workhorse weeks.

## Pregame feed / carry-ceiling features

All historical features were snapshotted strictly before the current season-week and updated only after the entire week received its pregame snapshot.

Empirical-Bayes families included:

- player career prior 20+ / 25+ frequency;
- player current-season prior 20+ / 25+ frequency;
- player prior p90 / p95 carry ceiling;
- team lead-RB prior 20+ / 25+ frequency;
- team current-season lead-RB 20+ / 25+ frequency;
- team lead-RB prior p90 / p95 carry ceiling;
- sample depth / confidence;
- current M94C team rush opportunity;
- projected offensive plays;
- projected lead/trail game-state share;
- neutral / leading rush tendency;
- QB rush siphoning;
- RB1 concentration and number of RBs used.

Shrinkage values `4`, `8`, and `12` were tested using 2024 only.

## Critical architecture safeguard — probability-mass preservation

M95F/M95I/M95J already established that stable-workhorse probabilities were too high in aggregate. M95K therefore was **not allowed to create additional aggregate stable-workhorse 20+/25+ probability mass**.

The selected feed model only reranks the existing M95F stable-workhorse tail probability:

1. fit the feed / ceiling model;
2. shift its 20+ probabilities so the stable-workhorse mean exactly equals the frozen M95F stable-workhorse mean;
3. use the relative 20+ reranking to modify M95F's conditional 25+ risk;
4. shift 25+ probabilities so their stable-workhorse mean also exactly equals frozen M95F.

Thus M95K tests **who deserves the existing tail mass**, not whether to make the entire group more bullish.

## 2024 selection

Training: Weeks 13-15.

Selection: Weeks 16-18.

Selected architecture:

- empirical-Bayes shrink `k = 4`
- spec `feed_compact_env`
- logistic `C = .03`
- stable 25+ method: `M95F conditional ratio + mass anchor`
- vacancy branch: frozen M95I joint probabilities
- other RBs: frozen M95F
- central carry mean: M94C unchanged

The selected feature set was compact rather than the full raw history table: baseline 20+ logit, composite 20+/25+ feed rates, composite p90/p95 carry ceilings, sample depth, M94C team rush opportunity, offensive plays, lead/trail state, neutral/lead rushing tendency, QB rush share, RB1 concentration and RB count.

## 2025 research validation — stable workhorses

This is a reused research validation season, **not a pristine final confirmation set** after the long M91-M95K sequence.

Stable-workhorse population: `237` RB-games.

### 20+ carries

M95F:

- actual rate: `21.94%`
- mean probability: `29.60%`
- AUC: `.581185`
- Brier: `.186593`
- log loss: `.554301`

M95K:

- mean probability: **`29.60%` exactly preserved**
- AUC: **`.641164`**
- Brier: **`.171528`**
- log loss: **`.527166`**

Gains:

- AUC **`+0.059979`**
- Brier **`+0.015065`**

This is the first large improvement in distinguishing high-workload weeks among already-established workhorses without simply increasing their aggregate probability.

### 25+ carries

M95F:

- actual rate: `4.64%`
- mean probability: `11.10%`
- AUC: `.591714`
- Brier: `.053017`
- log loss: `.218378`

M95K:

- mean probability: **`11.10%` exactly preserved**
- AUC: **`.612631`**
- Brier: **`.051386`**
- log loss: **`.215156`**

Gains:

- AUC **`+0.020917`**
- Brier **`+0.001630`**

The 25+ gain is smaller than the 20+ gain, but both ranking and probability scoring improve without increasing total stable-workhorse 25+ mass.

## Individual feed-signal audit

The strongest leakage-safe carry-ceiling signals inside the 2025 stable-workhorse population included:

- player current-season p95 carry ceiling: 25+ AUC **`.7170`**;
- player current-season p90 carry ceiling: 25+ AUC **`.6840`**;
- team current-season p95 lead-RB carry ceiling: 25+ AUC **`.6549`**;
- team current-season p90 lead-RB carry ceiling: 25+ AUC **`.6335`**.

For 20+ workloads, composite and ceiling signals were also positive, generally around `.60-.63` AUC individually.

This directly supports the M95K hypothesis: **not all workhorses / teams have the same demonstrated upper workload ceiling.**

## Vacancy branch remains preserved

M95K did not reopen the successful role-transition mechanism.

Vacancy 25+:

- M95F AUC `.721739`
- frozen M95I/M95K AUC **`.939130`**
- Brier `.008840 -> .008445`
- log loss `.048953 -> .040330`

Thus the two-regime architecture remains intact:

- vacancy / transition: recipient-specific deep concentration + workload tail;
- stable workhorse: persistent feed tendency / carry ceiling + current environment.

## Full-population 2025 result

### 20+

M95F:

- AUC `.846474`
- Brier `.062636`
- log loss `.208788`

M95K regime architecture:

- AUC **`.851667`**
- Brier **`.059739`**
- log loss **`.202694`**

Gains:

- AUC `+0.005193`
- Brier `+0.002897`

### 25+

M95F:

- AUC `.844321`
- Brier `.017985`
- log loss `.078526`

M95K regime architecture:

- AUC **`.851351`**
- Brier **`.017675`**
- log loss **`.077260`**

Gains:

- AUC `+0.007031`
- Brier `+0.000310`

For the first time in this tail sequence, the regime architecture improves both 20+ and 25+ ranking and probability scoring at the full-population level while also materially improving stable workhorses and preserving the vacancy signal.

## Gates

- stable20 pass: `1`
- stable25 pass: `1`
- all20 pass: `1`
- all25 pass: `1`
- vacancy25 preserved: `1`
- stable probability mass preserved: `1`
- core scientific pass: **`1`**

However:

- inherited M94C legacy all-player rushing guard gain: `-0.003205`
- production gate: `0`

The legacy guard is not waived.

## Scientific interpretation

M95K is a meaningful breakthrough in the workload-tail research.

M95J failed because universal weekly-context coefficients treated workhorses too similarly. M95K adds a persistent workload-ceiling prior and uses current environment only after acknowledging that individual players and teams differ in their historical willingness / ability to reach extreme carry volume.

The result also supports the broader architectural lesson from the RB sequence: useful football signals often need the correct mathematical role. Carry-ceiling history is valuable as a **probability reranker**, not as a universal mean-carry boost.

M95K should **not** be promoted yet. The architecture now deserves a sealed temporal confirmation because 2025 has been inspected repeatedly throughout this research program.

## Recommended next migration — M95L

**M95L — Sealed Temporal Confirmation of the M95K Regime Tail Architecture**

Primary question:

> Does the frozen M95K two-regime tail architecture reproduce its gains on a validation period that was not used to choose its feed families, shrinkage, regularization or probability-mass-preserving mechanism?

Requirements:

- freeze M95K spec `feed_compact_env`, shrink `4`, C `.03`;
- freeze the mass-preserving 20+/25+ reranking method;
- freeze M95I vacancy mechanism;
- no new coefficient or feature search;
- construct a prior temporal rotation using only information available before each game;
- preferably rebuild a 2023 late-season confirmation using earlier-2023 history and leakage-safe roster/injury/depth information, while reproducing the required M94C football-environment variables;
- if a sufficiently faithful 2023 reconstruction cannot be made, do not weaken the test: establish another genuinely sealed temporal protocol before production consideration;
- score stable 20+/25+, vacancy 25+, all 20+/25+, calibration, and ordinary central-carry preservation;
- no sportsbook input;
- no production change during confirmation.
