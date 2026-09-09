# RB R22 Week 1 Receiving-Tail Production Integration V1 — Result

Date: 2026-09-08/09 UTC
Branch: `research-cross-position-catastrophic-casebook-v1`
Frozen plan: `docs/migrations/RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_V1_PLAN.md`
Final replay head: `4d0690f9827466d0792be42752bcc6b6d8a03f96`
Run: `34298516960`
Job: `102300245980`
Artifact: `10084118525`
Artifact name: `rb-r22-week1-production-integration-v2`
Artifact digest: `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
Disposition: `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
PASS: true

## Scope

R22 productionizes the already-supported R19/R17/R18 receiving-yard tail mechanism for **2026 Week 1 only** on top of the certified V3 Full Slate stack.

Allowed changes:

- RB `rec_yards` distribution shape;
- mathematically linked RB `rush_rec_yards` distribution shape.

Forbidden and confirmed unchanged:

- RB receiving-yard mean;
- RB targets/receptions;
- RB rushing outputs;
- FB outputs;
- QB/WR/TE outputs;
- every pricing row outside adapted RB `rec_yards` / `rush_rec_yards` probability outputs.

Sportsbook inputs to football distribution generation: 0.
Current/future 2026 outcomes used: 0.
Production mean parameters changed: 0.

## Production adapter result

Disposition: `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`

- football RB rows: 94
- adapted RB rows: 94
- FB rows: 13
- max CONTROL-vs-adapted receiving mean delta: `5.329070518200751e-15` yards
- minimum rank Spearman: `0.9999999999999999`
- max R9 shadow RB-pool conservation gap: `2.7755575615628914e-17`
- p30 range: `0.01695685242394029 .. 0.2685044337528834`
- p50 range: `0.0027874569735999778 .. 0.12554110558631817`
- state probability range: `0.0 .. 0.6036812328937262`

All adapter gates passed:

- all 94 RBs adapted
- exact R19 assets
- strict-prior history
- no current/future outcomes
- no sportsbook inputs upstream
- deterministic replay
- finite/nonnegative receiving yards
- mean parity
- rank preservation
- exact R9 shadow pool conservation
- non-RB exact
- FB exact
- receptions exact
- RB nonreceiving markets exact
- `rush_rec_yards = rush_yards + rec_yards` identity preserved
- qualified Week 1 route

Exact R19 production assets remain hash-pinned:

- scorer model SHA256: `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual-pools file SHA256: `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

Source R19 run: `34288244770`, artifact `10080377483`, digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`.

## Pricing lineage result

Disposition: `RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS`

- adapted player keys: 94
- priced rows stamped: 204
- RB receiving-yard side rows stamped: 138
- RB rush+receiving-yard side rows stamped: 66
- RB reception side rows explicitly present and unadapted: 130
- provider alias rows in priced output: 168
- adapted provider alias rows: 62
- sportsbook inputs to adapter: 0
- production mean parameters changed: 0

The provider identity bridge is lookup-only after football simulation. Sportsbook event/player identities do not define the football universe or football distributions.

## Same-checkout V3 -> V4 differential certification

Comparison contract: `SAME_CHECKOUT_SAME_INPUTS_SAME_SEED_V3_VS_V4`.

- control rows: 3,244
- candidate rows: 3,244
- adapted RB football keys: 94
- allowed probability-change rows observed: 202
- unexpected probability-change rows: 0

Point/mean outputs remained exact to numerical tolerance:

- `model_proj` all equal, max abs delta `2.842170943040401e-14`
- `mc_proj` all equal, max abs delta `2.842170943040401e-14`
- `ensemble_proj` all equal, max abs delta `2.842170943040401e-14`
- `ml_proj` all equal, max abs delta `0.0`
- `state_proj` all equal, max abs delta `0.0`

Every frozen differential gate passed, including:

- exact row universe
- adapter integration/mean/rank integrity
- exact 94-RB scope
- receptions exact
- all nonallowed pricing rows exact
- zero unexpected probability changes
- RB receiving probability changes reach pricing
- existing QB stack valid
- existing TE stack valid
- existing WR stack valid
- sportsbook zero to adapter
- outcomes zero to adapter
- suffix-safe provider identity consistent and unambiguous

## Mechanical identity repair

A prior R22 replay failed only because the pricing-lineage/differential audit compared provider-form suffix keys to the canonical football keys for two players:

- Kenneth Walker III -> canonical football key `kennethwalker`
- Travis Etienne Jr. -> canonical football key `travisetienne`

The actual post-simulation distribution lookup was already suffix-safe and correctly served the adapted arrays. The repair changed only the audit/stamping identity bridge to use the same governed suffix-safe contract, with additional ambiguity/consistency guards.

No football feature, model coefficient, residual pool, seed, distribution rule, scientific gate, projection, or probability-generation rule was changed to obtain the final PASS.

## Existing certified stack retained

The R22 replay revalidated the complete V3 stack before the R22 differential gate:

- 16 games / 32 teams
- 469 sportsbook-independent football players
- 393 players with priced offers
- 76 simulated football players without priced offers
- explicit finite target-entitlement pool active
- TE-R5P production entitlement active
- M38 + WR-R15 production entitlement active
- M89/M90 QB mean authority active
- QB C2 mean-neutral distribution selector active
- RB P3 Week-1 rushing authority active
- sportsbook rows used to define football universe: 0
- sportsbook inputs used for football distributions: false

## Governance / production boundary

R22 is now **production-integration certified for 2026 Week 1 RB receiving-yard distribution shape**.

This does **not** claim RB receptions or RB receiving target entitlement are solved. R22 intentionally leaves receptions/targets and the receiving-yard mean unchanged.

The prospective R21 CONTROL/SHADOW lock remains valuable and should be graded after Week 1 as independent evidence; it does not block Week-1 use of the now-certified R22 distribution adapter.

Next production action: make V4 (certified V3 stack + R22) the canonical Week-1 Full Slate pricing path, update the certified-stack/market-lineage audit to recognize R22 explicitly, then run canonical Full Slate validation before the final live Week-1 slate/odds refresh.
