# PRISM Future Stage-2 Market Translation Trace V1 — Mechanics Result

Date: 2026-10-17

Status: **MECHANICS READY / NO SCIENTIFIC TARGET LOCK / NO WEEK-6 BACKFILL / RESEARCH ONLY**

Model:
**PRISM — Player Role & Individual Situation Model**

Disposition:

`PRISM_FUTURE_STAGE2_MARKET_TRANSLATION_TRACE_V1_MECHANICS_READY__NO_TARGET_LOCK`

Frozen mechanics contract:

`docs/research/PRISM_FUTURE_STAGE2_MARKET_TRANSLATION_TRACE_V1_FREEZE.md`

Freeze commit:

`4ad5ab802f68b67c2358f8e8bc6942ebfedb5eaa`

## 1. Canonical validation

Green mechanics run:

- run `39058313282` — **SUCCESS**;
- exact head `fe30cd1f7e00fe3fafecd5b4190d69412491d4ca`;
- artifact `12110143626`;
- digest `sha256:b6d1cd80c9254c970420cf13e7d96ed56767e40a7abaf2c2800bf2dfbe9712cb`;
- focused tests: **7 / 7 PASS**;
- strict repository audit: **PASS**.

Implementation:

`scripts/research/build_prism_future_stage2_market_translation_trace_v1.py`

Focused tests:

`tests/test_prism_future_stage2_market_translation_trace_v1.py`

Workflow:

`.github/workflows/research-prism-future-stage2-market-translation-trace-v1.yml`

## 2. What this mechanics layer does

The layer starts only after a valid, already-coherent Stage-2 player opportunity row exists.

It does not construct or repair:
- player identity;
- Stage-1 `p_active`;
- canonical `q_i`;
- Stage-2 active-set allocation;
- teammate room state;
- team opportunity volume;
- player efficiency.

It translates the frozen coherent named-player opportunity amount into the exact market-specific football mean while holding the existing pregame efficiency authority fixed.

This isolates the question:

> If PRISM improves this named player's finite carries/targets allocation, what exact market mean follows before any sportsbook comparison?

## 3. Eligible cells

Only historically qualified Stage-2 cells:

- `RB_RUSHING`;
- `RB_RECEIVING`;
- `TE_RECEIVING`.

Not authorized:
- QB Stage 2 — CLOSED;
- WR Stage 2 — source-blocked.

## 4. Exact translations

### RB rushing

Opportunity:
- baseline carries = frozen baseline expected opportunities;
- candidate carries = frozen coherent Stage-2 expected opportunities.

Rushing yards:
- baseline mean = baseline carries × existing `rules_ypc`;
- candidate mean = candidate carries × the **same** `rules_ypc`.

### RB receiving

Targets:
- baseline/candidate targets = baseline/candidate expected opportunities.

Receptions:
- baseline = baseline targets × existing `rules_catch_rate`;
- candidate = candidate targets × the **same** `rules_catch_rate`.

Receiving yards:
- baseline = baseline targets × existing `rules_ypt`;
- candidate = candidate targets × the **same** `rules_ypt`.

### TE receiving

Uses the identical target translation:
- targets;
- receptions via existing `rules_catch_rate`;
- receiving yards via existing `rules_ypt`.

No TE efficiency model is reopened.

## 5. Identity / conservation protections

Before any translation, the implementation requires exact Stage-2 identities:

- `candidate_expected_q - canonical_q = share_shift`;
- `team_opportunity_mean × canonical_q = baseline_expected_opportunities`;
- `team_opportunity_mean × candidate_expected_q = candidate_expected_opportunities`;
- `candidate_expected_opportunities - baseline_expected_opportunities = expected_opportunity_shift`.

Each market row then requires:

`candidate_market_mean - baseline_market_mean = expected_opportunity_shift × held_fixed_efficiency`

within `1e-12`.

The player/event/team identity must join one-to-one to the pregame football universe through:
- event;
- team;
- stable player ID;
- canonical player key parity.

This translation layer performs no new renormalization. Finite-pool conservation remains Stage 2's responsibility upstream.

## 6. Fail-closed protections

The mechanics explicitly reject:

- any 2026 Week-6-or-earlier input;
- sportsbook / bookmaker / prop-line / odds / no-vig / market-implied columns;
- target-game actual / realized / settled / outcome columns;
- missing or non-finite required player efficiency;
- invalid `rules_ypc`;
- invalid catch-rate probability;
- negative YPT;
- player identity mismatch;
- Stage-2 share/opportunity identity drift;
- multiple target weeks in one scientific input.

This is intentional. The implementation is not permitted to repair a bad Stage-2 lock.

## 7. Focused validation

Synthetic Week-7 fixtures verify:

- exact RB carries translation;
- exact RB rushing-yard translation;
- exact RB target translation;
- exact RB receptions translation;
- exact RB receiving-yard translation;
- exact TE target/reception/receiving-yard translation;
- fixed-efficiency identity;
- Week-6 rejection;
- sportsbook rejection;
- target-outcome rejection;
- missing-efficiency rejection;
- player-identity rejection;
- Stage-2 opportunity-identity rejection.

Focused test result:

**7 passed in 0.41s**.

## 8. Scientific-lock boundary

At mechanics certification:

- target scientific lock: **false**;
- target lock rows: **0**;
- Week-6 rows created: **0**;
- Week-6 backfill: **false**;
- Week-6 outcomes read: **0**;
- future target outcomes read: **0**.

The mechanics being ready is not permission to create a Week-7 scientific lock before chronology is valid.

A future lock still requires:
- Week 6 legitimately complete;
- strict-prior history complete through target_week - 1;
- target-week roster/slate identity;
- valid availability authority;
- Stage-1 `p_active`;
- canonical `q_i`;
- Stage-2 coherent allocation;
- player-specific efficiency attach;
- immutable row hashes;
- zero target outcomes.

## 9. Relation to the Week-4/5 forensic result

The Week-4/5 replay found the dominant repeated direct-skill seam in large opportunity misses was named-player allocation inside a finite team pool.

The newer finite-allocation transmission audit independently showed why individual player claims cannot simply be boosted independently: raw player claims overbook the finite team pool and conservation then compresses them.

This translation layer therefore does **not** bypass conservation.

It waits for coherent Stage-2 opportunity allocation first, then asks the exact market question for the named player.

That preserves the PRISM order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

## 10. Integrity

Certified mechanics:

- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- target outcomes: **0**;
- Week-6 outcomes: **0**;
- models fit: **0**;
- thresholds searched: **0**;
- efficiency authorities changed: **0**;
- production changed: **false**.

Final disposition:

`PRISM_FUTURE_STAGE2_MARKET_TRANSLATION_TRACE_V1_MECHANICS_READY__NO_TARGET_LOCK`.

## 11. Exact next action

Do not grade or mutate Week 6.

After Week 6 is final and strict-prior target-week authority is valid, create the first outcome-blind future Stage-2 + market-translation scientific lock using these frozen mechanics.

Until then, implementation may prepare the generic future lock orchestration and integrity checks, but it may not create scientific Week-7 rows early.
