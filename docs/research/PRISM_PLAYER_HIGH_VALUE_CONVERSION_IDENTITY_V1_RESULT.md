# PRISM Player High-Value Conversion Identity V1 — Predictive Result

Date: 2026-10-17

Frozen mechanism:
`docs/research/PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_V1_MECHANISM_FREEZE.md`

Source authority:
- run `39093918886` — SUCCESS;
- source disposition `PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_SOURCE_READY_PARTIAL`;
- RB receiving remained source-blocked and was not scored.

Canonical scored authority:
- run `39094077341` — **SUCCESS**
- exact scored head `659455070c5a9b6d61ae5f23141672fede1ed30f`
- artifact `12123792855`
- digest `sha256:0b2399a6d78762b2c5ee1059af6d224bffe6fbfcc001e296fbc9a495591e4b26`

Overall disposition:

`PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_CLOSED_ALL`

Both source-qualified cells failed the frozen development and independent
confirmation gates. No production promotion is authorized.

## What was tested

Conditional on the named player receiving the frozen scoring-area opportunity:

- baseline = strictly-prior Jeffreys-smoothed channel conversion rate;
- candidate = same-player strictly-prior conversion rate shrunk by exactly
  four channel-baseline pseudo-opportunities;
- minimum player history = 3 prior high-value opportunities;
- 2024 development;
- 2025 independent confirmation;
- no 2026 outcomes;
- no sportsbook/OddsAPI;
- no opportunity-entitlement mutation.

This was conversion efficiency **after opportunity exists**, not a TD-outcome
rescue of the closed scoring-area-share model.

## RB rushing inside 10 — CLOSED

Disposition:
`RB_RUSH_INSIDE10_PLAYER_CONVERSION_IDENTITY_CLOSED`.

### 2024 development

- event rows: 1,030
- player-games: 513
- named players: 84
- event Brier gain: **-0.005053**
- event log-loss gain: **-0.013900**
- conditional player-game Brier gain: **-0.005812**
- conditional player-game log-loss gain: **-0.019137**
- equal-player event Brier gain: **-0.007655**
- median player gain: **-0.004560**
- positive-player share: **34.52%**

The candidate was already worse in development.

### 2025 independent confirmation

- event rows: 988
- player-games: 509
- named players: 86
- event Brier gain: **-0.005372**
- event log-loss gain: **-0.015430**
- conditional player-game Brier gain: **-0.008072**
- conditional player-game log-loss gain: **-0.019743**
- equal-player event Brier gain: **-0.005378**
- median player gain: **-0.005829**
- positive-player share: **31.40%**

2025 event Brier CIs:
- player-cluster: **[-0.008668, -0.001996]**
- game-cluster: **[-0.008995, -0.001713]**
- equal-player: **[-0.015018, +0.006978]**

2025 conditional-game Brier CIs:
- player-cluster: **[-0.016139, +0.000700]**
- game-cluster: **[-0.015520, -0.000605]**

Every frozen qualification gate failed.

## TE receiving red zone — CLOSED

Disposition:
`TE_RECEIVING_REDZONE_PLAYER_CONVERSION_IDENTITY_CLOSED`.

### 2024 development

- event rows: 543
- player-games: 372
- named players: 76
- event Brier gain: **-0.001291**
- event log-loss gain: **-0.003689**
- conditional player-game Brier gain: **-0.002203**
- conditional player-game log-loss gain: **-0.003779**
- equal-player event Brier gain: **-0.002712**
- median player gain: **-0.005460**
- positive-player share: **38.16%**

The candidate was already worse in development.

### 2025 independent confirmation

- event rows: 599
- player-games: 407
- named players: 90
- event Brier gain: **-0.008033**
- event log-loss gain: **-0.022883**
- conditional player-game Brier gain: **-0.015628**
- conditional player-game log-loss gain: **-0.035120**
- equal-player event Brier gain: **-0.014771**
- median player gain: **-0.005986**
- positive-player share: **35.56%**

2025 event Brier CIs:
- player-cluster: **[-0.012999, -0.002930]**
- game-cluster: **[-0.014745, -0.001336]**
- equal-player: **[-0.029204, -0.002101]**

2025 conditional-game Brier CIs:
- player-cluster: **[-0.024531, -0.006829]**
- game-cluster: **[-0.025321, -0.006066]**

Every frozen qualification gate failed.

## Interpretation

The result is stronger than “not enough evidence.”

For both qualified channels, direct persistent same-player scoring-area
conversion history made the next conversion probability **worse** than the
strictly-prior channel baseline in both 2024 and 2025.

Therefore PRISM should **not** encode a generic persistent “goal-line finisher”
or “red-zone converter” identity from this direct historical conversion rate.

That does not mean the touchdown market is random or that the player does not
matter. It means the player-specific TD path should be expressed primarily
through football state that changes the *chance to receive and quality of the
opportunity*—role, teammate competition, scoring-area entitlement, team
creation, play/opportunity type and valid player-conditioned context—rather
than a generic same-player conversion-rate memory.

## Binding anti-rescue

Do not reopen this direct persistent conversion family with:
- another shrinkage strength;
- another history minimum;
- inside-5/inside-20 variants;
- end-zone-target replacement;
- recent-window conversion;
- same-team only conversion;
- lead-back/TE1 subsets;
- pooled RB/TE;
- opponent or teammate add-ons to rescue this exact scalar;
- sportsbook/market information.

RB receiving remains source-blocked from run `39093918886`; do not lower its
support gate.

## ATD boundary

Generic ATD remains uncertified.

This result does not authorize betting or production mutation. It narrows the
architecture: TD science still needs a coherent chain from team scoring
opportunity through named-player entitlement/opportunity type before
market-specific probability calibration.

## Integrity

- 2026 outcomes read: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- production mutation: false
