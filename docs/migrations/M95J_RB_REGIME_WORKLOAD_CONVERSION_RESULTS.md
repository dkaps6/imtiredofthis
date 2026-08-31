# M95J — Regime-Specific Workload Conversion Results

## Authoritative run

- Workflow: `M95J RB Regime Workload Conversion`
- Run: `33405821436`
- Job: `99533036053`
- Tested SHA: `632f9c27e1adc2658d01fca40071a43497fafc4d`
- Branch: `research-rb-m95j-regime-workload-conversion`
- Artifact: `migration-95j-rb-regime-workload-conversion`
- Artifact ID: `9763096005`
- Artifact SHA256: `048bdcbfd1d39659c9a058b35e76ec291622ce4f64e8dd8e77301e4265667b3a`
- Artifact size: `63,232` bytes
- Execution: success
- Scientific disposition: `RETAIN_M95J_AS_DIAGNOSTIC_DO_NOT_PROMOTE`
- Production change: `0`
- Sportsbook inputs: `0`
- M94C central carry estimate preserved: `1`

## 25+ workload frequency

In 2025 there were **24 RB performances with at least 25 carries**.

- 24 / 1,393 RB player-games = **1.7229%**
- 24 / 544 team-games = **4.4118%**
- those 24 events occurred in 24 distinct NFL games
- 24 / 272 NFL games = **8.8235%**
- equivalently, about **one 25+ RB workload every 11.33 NFL games**

Thus 25+ is rare at the individual-player level but not negligible at the league/game level. It is too common to ignore, but too rare to justify distorting the entire RB mean model.

## Scientific design

M95I showed that vacancy/role-transition backs and stable workhorses are different problems. M95J therefore used three regimes:

1. **Vacancy / role transition:** preserve frozen M95I joint workload probabilities.
2. **Stable incumbent workhorse:** fit a new week-specific 20+ conversion model from 2024 only.
3. **Other RBs:** preserve frozen M95F probabilities.

Stable-workhorse definition required:

- established workhorse role;
- prior top RB still available;
- player was the prior top RB;
- recent share had not collapsed by more than 10 percentage points versus the five-game baseline;
- player was not OUT or doubtful.

The stable model tested week-specific football variables rather than additional role-identity evidence: projected team rush attempts, projected offensive plays, predicted lead/trail game-state shares, neutral rush tendency, QB rush siphoning, backfield competition, recent carry/share trend, injury/practice limitation, and depth context. Matchup variants also tested defensive rushing vulnerability.

Training used 2024 Weeks 13-15 and selection used Weeks 16-18. The selected model was refit on all 2024 Weeks 13-18 and then evaluated on 2025.

Because 25+ is very rare, M95J did not fit a separate high-dimensional 25+ stable model. It preserved M95F's conditional 25|20 ratio and rescaled it with the new stable 20+ probability.

## 2024 development result — looked promising

Selected stable architecture:

- spec: `script_competition`
- C: `0.10`

On the 2024 selection slice:

- baseline 20+ AUC: `.619048`
- M95J AUC: **`.743590`**
- baseline Brier: `.247201`
- M95J Brier: **`.215228`**

The development result therefore looked meaningfully positive and passed the pre-specified selection rule.

## 2025 stable-workhorse validation — failed to generalize

Stable-workhorse population: `237` RB-games.

### 20+ carries

Actual rate: **21.94%**

M95F:

- mean probability: `29.60%`
- AUC: `.581185`
- Brier: `.186593`
- log loss: `.554301`

M95J stable model:

- mean probability: **35.01%**
- AUC: `.576715`
- Brier: `.196940`
- log loss: `.580709`

Changes:

- AUC: `-0.004470`
- Brier gain: `-0.010347`

The model became more overconfident and did not improve ranking.

### 25+ carries

Actual rate: **4.64%**

M95F:

- mean probability: `11.10%`
- AUC: `.591714`
- Brier: `.053017`

M95J stable model:

- mean probability: **13.47%**
- AUC: `.561545`
- Brier: `.057708`

Changes:

- AUC: `-0.030169`
- Brier gain: `-0.004692`

This is a clear failure. Generic week-specific football variables, as encoded here, looked strong in the small 2024 development slice but did not generalize to 2025 stable workhorses.

## Vacancy branch remains strongly useful

M95J deliberately preserved the frozen M95I vacancy mechanism.

### Vacancy 20+

- M95F AUC `.884956` -> M95I/M95J `.870206`
- Brier `.029865` -> **`.025854`**
- log loss `.123561` -> **`.105825`**

So vacancy 20+ benefits mainly from improved calibration rather than ranking.

### Vacancy 25+

- AUC `.721739` -> **`.939130`**
- Brier `.008840` -> **`.008445`**
- log loss `.048953` -> **`.040330`**

This remains one of the strongest isolated RB tail signals found in the sequence. The role-transition/vacancy problem should remain separate from the stable-incumbent problem.

## Overall regime combination

Because the stable branch failed, the combined population also failed.

### 20+

- AUC `.846474` -> `.846324`
- Brier `.062636` -> `.064062`

### 25+

- AUC `.844321` -> `.844503`
- Brier `.017985` -> `.018750`

The tiny 25+ ranking gain does not compensate for worsened calibration.

## Carry mean

M95J intentionally preserved M94C central carries. No carry-mean uplift was applied because the experiment's purpose was regime-specific conversion probability and the new stable branch did not validate.

Therefore all carry MAE slices remain exactly M94C in M95J.

## Scientific interpretation

M95J strengthens, rather than weakens, the regime-split conclusion:

- **Vacancy/role-transition:** we have a legitimately strong recipient-specific tail signal.
- **Stable workhorse:** the unresolved variable is not role identity, and a generic cross-player weekly-context model is not stable enough.

The 2024-to-2025 reversal suggests the stable-workhorse problem may require **persistent player/team/coach-specific workload-ceiling behavior**, not just universal game-state and matchup coefficients.

A coach/team may have a different willingness to feed an RB 25+ times even when two games look statistically similar. Likewise, some workhorses have demonstrated a much higher historical carry ceiling than others. These tendencies should be estimated leakage-safely and partially pooled rather than treated as identical across all workhorses.

## Recommended next migration — M95K

**M95K — Stable Workhorse Feed-Tendency / Carry-Ceiling Model**

Primary question:

> Among already-established stable workhorses, can leakage-safe player/team/coach-specific high-workload propensity identify which backs truly have a 20+/25+ carry ceiling, and then interact that propensity with the current week's football environment without overconfidence?

Candidate pregame families:

- player prior 20+/25+ carry frequency;
- player prior maximum / p90 / p95 carries;
- team lead-RB prior 20+/25+ frequency;
- team lead-RB carry maximum / upper quantiles;
- coaching/team run-persistence when leading or neutral;
- lead-RB share when team is leading late;
- fourth-quarter closeout rushing tendency;
- persistence after successful early rushing;
- team-specific rather than league-global workload conversion rates;
- opponent ability to keep game competitive as a counterweight;
- current M94C projected team rush opportunity;
- current matchup, QB siphoning, RB2 competition, injury/practice context;
- hierarchical / empirical-Bayes shrinkage so small player/team samples cannot dominate.

Do not reopen the successful vacancy branch. Do not manually inflate carry means. No sportsbook inputs.
