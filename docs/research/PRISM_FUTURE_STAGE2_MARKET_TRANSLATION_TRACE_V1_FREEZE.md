# PRISM Future Stage-2 Market Translation Trace V1 — Mechanics Freeze

Date: 2026-10-17

Status: **MECHANICS FROZEN / NO FUTURE-WEEK SCIENTIFIC LOCK YET / NO WEEK-6 BACKFILL / RESEARCH ONLY**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_FUTURE_STAGE2_MARKET_TRANSLATION_TRACE_V1`

## 1. Scientific purpose

The player-first replay identified named-player finite-opportunity allocation as the dominant repeated seam in major direct skill-player misses.

The Stage-2 active-role conditional-allocation bridge is independently qualified for:
- RB rushing;
- RB receiving;
- TE receiving.

The 2023 independent translation source audit is source-blocked because exact Stage-1 official-inactive parity is unavailable.

Therefore the next legal path is prospective.

The future trace asks:

> Before kickoff, if the already-qualified Stage-2 architecture changes this named player's coherent expected carries/targets while leaving his existing pregame efficiency authority unchanged, what exact market-specific football mean follows?

This is a **translation trace**, not a new opportunity model and not a new efficiency model.

## 2. No current target lock

At this mechanics freeze:
- target future week: **none**;
- Week-6 lock rows altered: **0**;
- Week-6 market-translation rows created: **0**;
- future target outcomes read: **0**.

Do not backfill the immutable Week-6 Stage-2 shadow with this new market translation.

Implementation may be built and mechanically tested with synthetic fixtures now.

A real scientific lock requires a separately authorized future-week pregame authority and valid chronology.

## 3. Required upstream row contract

The future lock will consume an immutable Stage-2 opportunity row with:

- season;
- week;
- event_id;
- team;
- cell;
- family;
- player;
- player_clean_key;
- player_id;
- `p_active`;
- Stage-1 probability source;
- canonical `q_i`;
- candidate coherent expected `q_i`;
- team opportunity mean;
- baseline expected opportunities;
- candidate expected opportunities;
- opportunity shift;
- availability authority;
- final availability state.

Eligible cells only:
- `RB_RUSHING`;
- `RB_RECEIVING`;
- `TE_RECEIVING`.

No QB.
No WR.

The Stage-2 row must already satisfy its own source/identity/conservation gates.
This translation layer may not repair an invalid Stage-2 row.

## 4. Existing efficiency authority

Join the exact same pregame football-universe player identity for the target game.

Allowed efficiency fields:

### RB rushing
- `rules_ypc`.

### RB receiving
- `rules_catch_rate`;
- `rules_ypt`.

### TE receiving
- `rules_catch_rate`;
- `rules_ypt`.

These fields are held fixed between baseline and candidate.

No new efficiency coefficient.
No outcome-derived efficiency update.
No position-average efficiency replacement.
No generic opponent multiplier.
No sportsbook-derived efficiency.

If the required player-specific efficiency field is unavailable/non-finite, that market translation fails closed for that row.

## 5. Frozen market translations

### RB rushing

Opportunity market:
- baseline carries = `baseline_expected_opportunities`;
- candidate carries = `candidate_expected_opportunities`.

Rushing-yards football mean:
- baseline = baseline carries × `rules_ypc`;
- candidate = candidate carries × `rules_ypc`.

Candidate-minus-baseline rushing-yards shift:
`(candidate carries - baseline carries) * rules_ypc`.

### RB receiving

Opportunity market:
- baseline targets = `baseline_expected_opportunities`;
- candidate targets = `candidate_expected_opportunities`.

Receptions football mean:
- baseline = baseline targets × `rules_catch_rate`;
- candidate = candidate targets × `rules_catch_rate`.

Receiving-yards football mean:
- baseline = baseline targets × `rules_ypt`;
- candidate = candidate targets × `rules_ypt`.

### TE receiving

Use the exact same target -> reception / receiving-yard transform as RB receiving:
- existing `rules_catch_rate`;
- existing `rules_ypt`;
- no TE-specific new efficiency adjustment.

TE efficiency-trust research remains CLOSED. This trace does not reopen it; it merely holds the existing pregame mean-conversion authority fixed while changing the upstream coherent target count.

## 6. Market rows emitted

For each eligible Stage-2 player row, emit:

RB rushing:
- `rush_attempts`;
- `rush_yards`.

RB receiving:
- `targets`;
- `receptions`;
- `rec_yards`.

TE receiving:
- `targets`;
- `receptions`;
- `rec_yards`.

Every emitted row retains:
- player identity;
- source Stage-2 cell;
- baseline and candidate opportunity;
- exact efficiency field/value used;
- baseline football mean;
- candidate football mean;
- mean shift;
- Stage-1 / Stage-2 provenance;
- availability provenance.

## 7. Invariants

Required:
1. player/event/team identity is one-to-one between Stage-2 row and football universe;
2. eligible cells only;
3. baseline/candidate expected opportunity finite and >=0;
4. canonical/candidate q finite and in [0,1];
5. required efficiency finite;
6. `rules_ypc > 0` for rushing-yards rows;
7. `0 <= rules_catch_rate <= 1`;
8. `rules_ypt >= 0`;
9. opportunity row translation has exact identity;
10. candidate-minus-baseline market shift equals opportunity shift × held-fixed efficiency within 1e-12;
11. no sportsbook columns accepted;
12. no target-game actual/outcome columns accepted;
13. no production write.

This layer does not renormalize shares.
Finite-pool conservation is upstream Stage 2's responsibility.

## 8. Reliability boundary

Reliability remains downstream of the football projection.

This trace does not:
- alter a football mean based on trust;
- create an over/under;
- select a bet;
- choose an alt line.

Later decision order remains:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

## 9. Future scientific lock

A future target lock must be frozen before target outcomes.

At minimum it must certify:
- target roster/identity authority;
- target availability semantics;
- Stage-1 p_active source;
- canonical q source;
- Stage-2 conservation;
- player-specific efficiency attach;
- zero sportsbook upstream;
- zero target outcomes;
- exact row hashes.

Do not create a Week-7 scientific lock until Week 6 is legitimately complete and the target pregame chronology is valid.

Implementation readiness is not a scientific lock.

## 10. Future grading hierarchy

When a future locked week settles, grade in this order:

1. opportunity:
   - carries or targets;
2. direct translated market:
   - rush yards;
   - receptions;
   - receiving yards;
3. named-player casebook;
4. matched controls;
5. aggregate position/market QA last.

This preserves the diagnostic question:
> did Stage 2 improve this player's market mean because it improved this player's finite opportunity allocation?

Do not interpret a yards gain as proof of an efficiency mechanism.

## 11. Integrity

At mechanics freeze:
- future target outcomes read: **0**;
- Week-6 outcomes read: **0**;
- Week-6 locks changed: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- models fit: **0**;
- thresholds searched: **0**;
- production changes: **0**.

Disposition:

`PRISM_FUTURE_STAGE2_MARKET_TRANSLATION_TRACE_V1_MECHANICS_FROZEN__NO_TARGET_LOCK`.
