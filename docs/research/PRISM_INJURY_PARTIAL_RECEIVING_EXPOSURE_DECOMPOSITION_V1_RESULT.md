# PRISM Injury-Partial Receiving Exposure Decomposition V1 — Result

Date: 2026-10-18

Status: **CLOSED / EXACT DECOMPOSITION DID NOT QUALIFY / NO PRODUCTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen contract:
`docs/research/PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1_FREEZE.md`

Freeze:
`653eb5e4e46a49f0b575c119259d2b11eb845380`

Canonical scored execution:
- run `39162113968` — SUCCESS;
- exact head `f71db18c3fc992793cd8e9f374256e4bcfd7dc31`;
- first run `39162055249` was mechanical test plumbing only and produced no science;
- source run `39161711741`;
- source artifact `12149181200`;
- source digest
  `sha256:6cae427b667db4a120a1308747d00b82b6770fe0adbc46d3efc68fce3e1db240`.

Disposition:

`PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1_CLOSED`

## Question

For a named RB/WR/TE whose latest four same-team positive-participation games
contained one injury-corroborated participation collapse, could PRISM improve
the next comparable target-share estimate by separating:

- ordinary expected offensive snap exposure; and
- target earning conditional on actual snap exposure?

No position coefficient was used.

## 2024 development

Affected:
- target games: **211**;
- named players: **66**.

Named-player result:
- equal-player closeness gain: **-0.002285**;
- median player gain: **-0.001424**;
- improved: **46.97%**;
- worsened: **53.03%**.

Family diagnostics:
- RB: 56 games / 20 players, equal-player **-0.008173**;
- TE: 44 / 14, **+0.000341**;
- WR: 111 / 32, **+0.000247**.

The frozen overall player-level development gates failed.

## 2025 independent confirmation

Affected:
- target games: **193**;
- named players: **64**.

Named-player result:
- equal-player closeness gain: **-0.000473**;
- median player gain: **-0.001121**;
- improved: **43.75%**;
- tied: **1.56%**;
- worsened: **54.69%**;
- player bootstrap 95% interval:
  **[-0.003408, +0.002614]**.

Family diagnostics:
- RB: 50 games / 16 players, equal-player **+0.000030**;
- TE: 58 / 18, **-0.002117**;
- WR: 85 / 30, **+0.000246**.

The frozen overall confirmation gates failed.

No family result rescues the contract.

## Interpretation

The exact formula is closed.

It is not sufficient to estimate:

`median comparable snap exposure × recent snap-conditioned target intensity`

as the general receiving-role correction after an injury-partial observation.

This does not restore the old interpretation.

Still forbidden:
- injury-partial game = ordinary regression evidence;
- DNP = zero performance;
- unexplained participation collapse = role decline.

The failure instead says the missing object is not captured by this simple
exposure × earning arithmetic.

## Next genuinely distinct player-state frontier

Move upstream to **return-from-injury role restoration**.

Question:

> After a player's participation is interrupted by an injury-corroborated
> partial game or injury-related no-snap observation, does his first subsequent
> positive-participation game restore his prior role state, remain reduced, or
> remain unresolved?

Define this state from participation/injury chronology only before asking what
it predicts for targets/yards.

This is a current-role/regime-state object, not another transformation of the
same target-share history.

## Anti-retest

Do not rescue this V1 with:
- another snap multiplier;
- another window;
- mean vs median;
- position-specific coefficients;
- WR-only cherry-pick;
- partial weighting of the same formula;
- route claims from snap data;
- sportsbook inputs;
- 2026 outcomes.

## Integrity

- first execution failure: mechanical tests only, no scoring;
- canonical science run: SUCCESS;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.
