# PRISM RB Short-Yardage Carry-Type Identity V1 — Result

Date: 2026-10-18

Status: **CLOSED / RESEARCH ONLY / NO PRODUCTION CHANGE**

## Canonical execution

- run: `39097257642` — SUCCESS
- exact scored head: `c366ef479b66ab4ee99907964748430002904351`
- job: `117586172786`
- artifact: `12125077660`
- digest: `sha256:56c5eb81a7f90f0c401baa53883852026771eb859e6d1617dbea6f08bd091bc3`
- source authority: run `39094323921`, head `5dbfe88491a7398bb9ce34450a5dffb22bda7a84`, artifact `12123703725`
- frozen mechanism: `docs/research/PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_V1_MECHANISM_FREEZE.md`

Disposition:

`RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_CLOSED`

## Frozen question

Conditional on the named RB receiving a carry, does his own strictly-prior
same-team tendency to receive a frozen short-yardage carry improve prediction
of carry type over the same team's strictly-prior RB carry-type baseline?

Frozen short-yardage event:

`down in {3,4} AND 0 < ydstogo <= 2`.

The candidate did not alter:
- team rush volume;
- named-player total carry entitlement;
- teammate carry allocation;
- conservation;
- rushing yards;
- touchdown probability;
- production.

## Frozen estimator

Team baseline:

`p_team = (prior_team_RB_short + 0.5) / (prior_team_RB_carries + 1)`.

Named-player candidate:
- minimum 10 prior same-team carries;
- fixed 20 team-baseline pseudo-carries;

`p_player = (prior_player_short + 20*p_team) / (prior_player_carries + 20)`.

No coefficient, window, support threshold, subgroup or event definition was
chosen after scoring.

## 2024 development

- carry rows: **11,069**
- player-games: **1,147**
- players: **109**
- carry Brier: `0.05980966 -> 0.05955101`
- carry Brier gain: **+0.00025865**
- carry log-loss gain: **+0.00061863**
- conditional-game Brier: `0.20645586 -> 0.20105910`
- conditional-game Brier gain: **+0.00539676**
- conditional-game log-loss gain: **+0.01125373**
- equal-player carry Brier gain: **+0.00241897**
- median player carry gain: **-0.00009210**
- positive-player share: **43.12%**

Both frozen development gates passed.

## 2025 independent confirmation

- carry rows: **11,190**
- player-games: **1,136**
- players: **115**
- carry Brier: `0.06471472 -> 0.06415518`
- carry Brier gain: **+0.00055954**
- carry log-loss gain: **+0.00279814**
- conditional-game Brier: `0.21509368 -> 0.20821590`
- conditional-game Brier gain: **+0.00687777**
- conditional-game log-loss gain: **+0.01992709**
- equal-player carry Brier gain: **+0.00334769**
- median player carry gain: **+0.00001248**
- positive-player share: **50.43%**

Carry-level 95% intervals:
- player-cluster: **[+0.00003286, +0.00121715]**
- target-game-cluster: **[+0.00022467, +0.00091296]**
- equal-player: **[+0.00042213, +0.00713377]**

Conditional-game 95% intervals:
- target-game-cluster: **[+0.00035029, +0.01363806]**
- player-cluster: **[-0.00139111, +0.01590455]**

## Frozen gate adjudication

Ten of eleven gates passed.

The exact failing gate was:

`confirm_2025_game_player_ci_low_gt_0`

because the conditional-game player-cluster 95% CI crossed zero.

Under the frozen AND-gate contract, this is a scientific close, not a near-pass
that may be rescued.

## Interpretation

There is real descriptive evidence that named RBs differ in their short-yardage
carry-type composition:

- carry-level performance improved in both seasons;
- all 2025 carry-level dependence-aware intervals were strictly positive;
- the average conditional-game score also improved;
- the target-game-cluster interval for conditional-game scoring was positive.

However, the exact mechanism did **not** establish sufficiently stable
same-player conditional-game improvement across RB identities. The player-level
heterogeneity protection gate was intentionally stricter than pooled average
improvement, and it failed.

Therefore PRISM must **not** encode this exact direct persistent short-yardage
carry-type identity.

This does not mean short-yardage role is irrelevant. It means this exact
strictly-prior same-team player-history scalar, with the frozen direct
transmission, did not clear the player-stability bar required for promotion.

## Anti-rescue

Do not rescue with:
- another pseudo-carry strength;
- different minimum prior carries;
- recent windows;
- all-team player history;
- second-and-short;
- one-yard-only / three-yard cutoffs;
- goal-line or field-position subsets;
- lead-back/starter carveouts;
- score/clock/game-script subsets;
- opponent interactions;
- yardage or TD translation;
- sportsbook/OddsAPI;
- 2026 outcomes.

Any future short-yardage research requires a materially new football object,
not a reparameterization of this direct player-history mechanism.

## Integrity

- 2026 outcomes read: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- fitted coefficients: **0**
- threshold search: **0**
- production mutation: **false**
