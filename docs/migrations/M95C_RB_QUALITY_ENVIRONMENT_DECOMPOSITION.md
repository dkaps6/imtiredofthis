# M95C — RB Quality vs Blocking/Environment Decomposition

## Status

Research only. No production change. No sportsbook or game-market variables.

M95B established a stable forward signal from RB/offensive context and a smaller but useful matchup signal. It also recovered weekly historical advanced rushing data that had not been present in the older frozen RB baseline: yards before contact, yards after contact, broken tackles, Next Gen Stats expected rushing yards, RYOE, rush percentage over expected, stacked-box frequency, and time to line of scrimmage.

M95C asks the next narrow question: **can we distinguish what the rushing environment gives the RB from what the RB creates himself, and does that decomposition improve forward prediction beyond raw rushing efficiency?**

## Frozen source

M95C consumes the successful M95B artifact from workflow run `33357785600` (`migration-95b-rb-offense-defense-matchup`). It does not redownload or rebuild historical football data. This freezes all identity joins and leakage-safe rolling features before M95C model fitting.

## Pregame feature groups

Every family contains the same M95B role/opportunity controls so the test is about efficiency attribution rather than rediscovering who gets carries.

### Raw rushing efficiency

Outcome-level historical measures:

- rushing yards per attempt;
- rush EPA;
- rush success rate;
- rushing first-down rate;
- stuff rate;
- 10+, 15+, and 20+ explosive-run rates.

### Environment / blocking opportunity

Pregame measures intended to describe what is available to the runner:

- PFR yards before contact per attempt;
- NGS expected rushing yards per attempt;
- percentage of attempts against 8+ defenders in the box;
- average time to line of scrimmage;
- team yards before contact per attempt;
- team stuff rate;
- player-vs-team relative yards-before-contact measures.

These are useful rushing-environment proxies, not a claim that we have a certified historical weekly offensive-line run-block win-rate feed.

### Runner-created value

Pregame measures intended to describe value created beyond the environment:

- PFR yards after contact per attempt;
- broken tackles per attempt;
- NGS rushing yards over expected per attempt;
- NGS rush percentage over expected;
- player-vs-team relative yards-after-contact measures.

RYOE is especially important because its expectation is already context-sensitive.

## Pre-specified model families

1. `role_baseline`
2. `role_plus_raw_efficiency`
3. `role_plus_environment`
4. `role_plus_created`
5. `role_plus_decomposition`
6. `role_plus_decomposition_and_raw`

The purpose is not to pick the best model from a broad zoo. It is to determine whether explicit environment-vs-created decomposition adds stable information beyond raw rushing outcomes.

## Forward validation

Exactly two forward tests:

- train 2023 -> test 2024;
- train 2023+2024 -> test 2025.

No 2025 feature selection or hyperparameter tuning.

Model forms are frozen to the M95B regularized linear setup:

- Ridge for continuous outcomes;
- Logistic Regression for tail-event AUCs.

## Scored outcomes

Continuous:

- carries (guardrail, not the primary M95C target);
- rushing yards;
- YPC among games with 3+ carries;
- YPC among games with 8+ carries.

Tail discrimination:

- 75+ rushing yards;
- 100+ rushing yards;
- at least one 20+ yard rush.

Additional 2025 diagnostics compare backs with poor pregame YBC but high pregame RYOE against backs with strong YBC but low RYOE. These are descriptive slices only and are not used for model selection.

## Advancement gate

The final decomposition+raw family advances only if all of the following are true:

1. rushing-yard MAE beats raw efficiency in both forward seasons;
2. 2025 YPC (8+ carries) is no worse than raw efficiency;
3. 2025 carry MAE does not regress by more than 0.05 carries;
4. at least one of 100+ rushing-yard AUC or 20+ explosive-run AUC is no worse than raw efficiency.

Passing this gate means `ADVANCE_M95C_QUALITY_ENVIRONMENT_DECOMPOSITION` for continued RB research. It does **not** authorize a production promotion.

Failing the gate means `RETAIN_M95B_OFFENSE_PROFILE` and the advanced measures remain useful diagnostics rather than a justified new architecture.

## Football interpretation

The causal distinction M95C is trying to recover is:

`blocking / box / expected lane quality -> yards available before contact`

versus

`runner tackle-breaking / after-contact / over-expectation ability -> yards created by the RB`

That decomposition can later be combined with the separately researched workload architecture (team rush volume + backfield concentration). M95C deliberately does not try to solve the known 25+ carry underprojection problem by itself.
