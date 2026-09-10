# QB Pass-Opportunity Rate Directional Personnel Source Audit V1 — Result

## Disposition

`SOURCE_INELIGIBLE_IDENTITY_OR_USAGE_COVERAGE`

The directional personnel consequence family does **not** advance to predictive testing under the frozen V1 gates. Production is unchanged.

## Canonical lineage

- Branch: `research-qb-pass-rate-directional-personnel-source-audit-v1`
- Frozen plan: `6fd86f8ceade4c55db5c672e55f8610c65a357ba`
- Evaluator commit: `18a94d97274d8d33cd15deba6a5c90e0942e3c8e`
- Tested/workflow head: `2ccbf68a5f03c48659838c428e671fd7167fcd9d`
- Run: `34537299384`
- Job: `103071783186`
- Artifact: `10175912101` (`qb-pass-rate-directional-personnel-source-v1`)
- Artifact digest: `sha256:fb2c646b1ba446a2e33bb6c1f5f94a89794b82a2ce68308753ac4ca503164da3`
- Parent play/rate result commit: `7dbbc93e42eae68e6032ec6cb2299d357be6cb20`
- Parent disposition: `PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`

## Source integrity

The corrected M78 inactive authority was reproduced exactly:

- CSV SHA256: `d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`
- team-week rows: `1088`
- total inactive tokens: `6851`
- sportsbook inputs: `0`
- target outcomes read: `0`
- model fitting: `0`
- production changes: `0`

The weekly player-stat schema was clean and stable across 2023-2025:

- carries field: `carries`
- targets field: `targets`
- receptions field: `receptions`
- player identity: `player_id`
- team: `team`
- minimum carries populated coverage: `100%`
- minimum targets populated coverage: `100%`
- Week-1 prior-season-only construction: PASS

## Identity result

### Backfield run channel
- relevant inactive RB/FB/HB events: `520`
- exact normalized unique identity resolution: `96.9231%`
- frozen requirement: `>=95%` — PASS

### Receiving channel
- relevant inactive WR/TE/RB/FB/HB events: `1707`
- exact normalized unique identity resolution: `97.4224%`
- frozen requirement: `>=95%` — PASS

Therefore identity mapping was **not** the blocker.

## Strict-prior usage result

### Backfield run capacity
Among mapped backfield inactive events:

- >=1 prior carry-history row: `82.9365%`
- frozen gate: `>=90%` — FAIL
- >=3 prior carry-history rows: `73.8095%`
- frozen gate: `>=75%` — FAIL

### Receiving capacity
Among mapped receiving-skill inactive events:

- >=1 prior target-history row: `83.2832%`
- frozen gate: `>=90%` — FAIL
- >=3 prior target-history rows: `74.0830%`
- frozen gate: `>=75%` — FAIL

Week 1 itself was mechanically safe: `50` mapped Week-1 events had prior history and every such history row came from a season strictly before the target season.

## Scientific/source meaning

The exact information distinction remains conceptually different from M77/M79: those studies grouped offensive skill personnel together, while V1 separated unavailable backfield rushing opportunity from unavailable receiving opportunity.

However, the historical opportunity-history coverage is not high enough under the frozen contract to support a clean population-level predictive test. The missing-history problem is concentrated in players without sufficient prior NFL usage, which means treating missing usage as zero would silently conflate **unknown NFL opportunity** with **known zero opportunity**.

That distinction is scientifically important and is why the source gate was frozen before results.

## Stopping rule honored

- no pass-opportunity-rate residual was joined;
- no QB attempt or passing-yard result was inspected;
- no WR residual was inspected;
- no predictive model was fit;
- no missing-history event was imputed as zero to rescue coverage;
- no 90% / 75% threshold was relaxed after results;
- no one-channel backfield-only or receiving-only test is authorized.

## Next research direction

Preserve this family as source-blocked in its current form. Do not retest it by loosening history requirements.

The next pass-opportunity-rate investigation must use another materially independent football mechanism. A candidate architecture linkage worth source-auditing separately is **QB designed-run burden**: the production stack already estimates player-specific designed QB rushing opportunity downstream, while the upstream team pass/rush split remains fixed at 57/43. This may represent a conservation/architecture seam rather than a generic team pass-rate-history retest, but it requires its own anti-reinvention and source audit before any target residual is inspected.
