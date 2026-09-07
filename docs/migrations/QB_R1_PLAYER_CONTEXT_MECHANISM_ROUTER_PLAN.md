# QB-R1 Player + Context Mechanism Router — Frozen Plan

## Why this is new
Prior QB work already tested different questions:
- M63 classified unusually high/low **attempt-volume surprises** from pregame context.
- M70/M71 studied directional YPA mechanisms and **efficiency uncertainty**.
- the recent QB individual-mechanism stability audit showed that player identity has meaningful persistence, but a QB's dominant mechanism changes with context.

QB-R1 does **not** retry those failed mean-feature searches. It asks a new routing question:

**Before the game, can player-specific recent mechanism history plus legitimate football context predict whether this game's mechanical projection error will be more ATTEMPT-driven or YPA-driven?**

This diagnostic predicts *which mechanism matters*, not a passing-yard correction. No production change is authorized.

## Frozen evidence
- Exact QB individual-mechanism run: `34065369449`, artifact name `qb-individual-mechanism-decomposition`.
- Exact M89 football evidence run: `33331073376`, artifact name `m89-qb-data-integrity-casebook-synthesis`.
- Mechanism casebook expected rows: **884** (2024-2025).
- Sportsbook/game-market features prohibited.

## Target
From exact QB mechanism casebook:

`attempt_mechanism_share = abs(attempt_component) / (abs(attempt_component) + abs(ypa_component))`

Rows with zero denominator are unscoreable for this routing target.

Diagnostic discrete states:
- `ATTEMPTS_DOMINANT` if `abs(attempt_component) >= 1.25 * abs(ypa_component)`;
- `YPA_DOMINANT` if `abs(ypa_component) >= 1.25 * abs(attempt_component)`;
- otherwise `MIXED`.

No other dominance ratio will be tried.

## Player-specific pregame signal
`prior4_player_attempt_share` = mean target mechanism share over the QB's previous up-to-4 scored games, minimum 3. The current and future game are never included.

This is a prior/context signal only. It is not a fixed player correction.

## Explicit football context feature set
Only already-established leakage-safe M89 synthesis-trace fields may be used:
- `qb_prior_attempts`
- `qb_prior_ypa`
- `off_true_proe`
- `off_neutral_pace`
- `def_pass_epa_allowed`
- `def_pass_success_allowed`
- `def_ypa_allowed`
- `off_pass_epa`
- `off_pass_success`
- `off_ypa`
- `off_plays`
- `off_pass_rate`
- `def_pass_rate_faced`

No spread, total, prop, closing line, realized target-game PBP feature, or postgame feature is allowed.

## Frozen models
One algorithm only: Ridge regression, alpha = **10.0**, with median imputation and standardization fitted on training data only.

Three preregistered feature views:
1. `PLAYER_ONLY`: `prior4_player_attempt_share`.
2. `CONTEXT_ONLY`: the 13 explicit football-context fields above.
3. `COMBINED`: player signal + all 13 context fields.

No hyperparameter sweep, feature selection, alternate windows, or model zoo.

## Temporal split
- Train: 2024 rows with scoreable target and required model inputs after imputation rules.
- Test: untouched 2025 rows.
- Rolling player signal for every row uses only chronologically earlier casebook games.

## Primary OOS metrics
For 2025 report for all three views:
- N;
- Pearson and Spearman between predicted and actual mechanism share;
- Q4-minus-Q1 actual mechanism-share gap by predicted score;
- top-quartile `ATTEMPTS_DOMINANT` rate;
- bottom-quartile `ATTEMPTS_DOMINANT` rate;
- top-minus-bottom attempts-dominant rate gap;
- W2-18 Spearman;
- W13-18 Spearman.

Also report `prior4_player_attempt_share` raw Spearman against the current mechanism share in 2025.

## Frozen discovery gate
Disposition is `QB_PLAYER_CONTEXT_MECHANISM_ROUTER_DISCOVERY_PASS` only if **COMBINED** satisfies every condition:
1. 2025 N >= 300;
2. Spearman >= 0.15;
3. Q4-Q1 actual mechanism-share gap >= 0.10;
4. top-minus-bottom `ATTEMPTS_DOMINANT` rate gap >= 0.12;
5. W2-18 Spearman > 0;
6. W13-18 Spearman > 0;
7. and at least one individual-value condition holds:
   - COMBINED Spearman >= CONTEXT_ONLY Spearman + 0.03; or
   - raw prior4 player signal Spearman >= 0.10.

Otherwise disposition is `NO_ACTIONABLE_QB_PLAYER_CONTEXT_MECHANISM_ROUTER`.

A discovery pass still does not alter production. It would authorize a separately frozen simulation-layer experiment in which uncertainty/variance is routed differently between attempt and YPA mechanics while the promoted QB mean remains untouched.
