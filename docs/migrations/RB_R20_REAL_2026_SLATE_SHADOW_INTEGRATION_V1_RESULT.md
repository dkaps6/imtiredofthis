# RB R20 — Real 2026 Slate Shadow Integration V1 — RESULT

Date: 2026-09-08
Branch: `research-cross-position-catastrophic-casebook-v1`
Repository: `dkaps6/imtiredofthis`

## Disposition

**PASS — SHADOW ONLY**

`RB_R20_REAL_2026_SLATE_SHADOW_INTEGRATION_PASS_SHADOW_ONLY`

R20 proves that the immutable R19 2026 tail scorer and frozen R17/R18 distribution adapter can score the real governed 2026 Week-1 full-slate football surface, fail closed on lineage/data quality, preserve the canonical Monte Carlo mean/dependence contract, and leave existing certified production outputs unchanged. It does **not** promote the RB receiving-tail lane into production.

## Immutable execution lineage

### R20
- Evaluator: `scripts/backtest/evaluate_rb_r20_real_slate_shadow_integration_v1.py`
- Evaluator commit: `db46dc0b10c14baaaa0883ecd6543b4eb9b1907d`
- Workflow: `.github/workflows/research-rb-r20-real-slate-shadow-integration-v1.yml`
- Workflow commit/head SHA: `587bf2a89ca16f11361016df3915361390289a7e`
- Actions run: `34291433027`
- Job: `102278616103`
- Artifact: `10081502774`
- Artifact name: `rb-r20-real-slate-shadow-integration-v1`
- Artifact digest: `sha256:853c0d3aea971c058ae6cc3b80c99ae2a5a0f681fae642835bbfa544da8283ca`

### Immutable R19 scorer
- Run: `34288244770`
- Artifact: `10080377483`
- Artifact digest: `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- Head SHA: `6ac1342f737f142acac6a3e4b459f442faf1442a`
- Model JSON SHA256: `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- Residual-pool NPZ SHA256: `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

### Immutable governed Week-1 replay
- Replay run: `34243241733`
- Artifact: `10062978930`
- Artifact digest: `sha256:31351365f3afefbf7f73a4df653499e84ce212d367e77a74ea6a12744c8a4f5f`
- Head SHA: `c2ffc633cc63ef9a110c16d2ce10ef3de07890e6`
- Raw paid source run: `34152868136`
- Raw source artifact: `10030344451`
- Raw source digest: `sha256:c19bd303a0eb7ca58a3484117e28b5e5144459b74459bd1032970873cae6d035`
- `odds_api_refetched: false`

R20 consumed the already-paid immutable replay only. It did not perform a new OddsAPI fetch.

## Frozen execution

- Season/week: `2026 / 1`
- Games: `16`
- Teams: `32`
- Full football universe: `469` players
- RB/FB scorer rows: `107`
- RB rows: `94`
- FB rows: `13`
- Canonical MC iterations: `10,000`
- Canonical simulation seed: `92020`
- Tail-adapter seed: `918`
- Strict-prior identity history: `2013-2025`
- Latest history state: `2025 W18` (`202518`), strictly before current slate `202601`
- Strict-prior state rows: `18,791`

## R19/R17 lineage integrity

All immutable artifact IDs, run IDs, head SHAs and artifact digests matched exactly.

R19 residual pools reproduced exactly:
- non-tail: 3,890 rows; SHA256 `f677e91cd25cdbd6db044e9decccec6312943b99f8ffc25a827527e54d4d7b1d`
- 30-49 tail: 174 rows; SHA256 `3da7bf656fcab3c111c8d5fb60e38fb7f735d7fcce639dabad899431f613c225`
- 50+ tail: 79 rows; SHA256 `ee203d3ffa01b8687e7774d817d0dce6cdc8b56ca62321a5583d5287dfd5ee43`

## Real-slate shadow scorer outputs

### R9 shadow feature
- reliability: `1.0`
- exact RB-room conservation max gap: `2.7755575615628914e-17`
- shadow target delta range: `-1.2485154947618782` to `+2.1204713220632616` targets
- production target entitlement remained unmodified.

### R11 state probability
- TOP20 identity rows: `22`
- REST80 rows: `85`
- REST80 probability: exactly `0`
- state-probability range: `0.0` to `0.6036812328937262`

### R16 tail probability
- p30 range: `0.016941397176149082` to `0.26847795088684917`
- p50 range: `0.0027410534089245767` to `0.1252831166144702`
- all probabilities finite and in [0,1].

## Canonical R17/R18 adapter parity on the real slate

Frozen adapter acted on all `94` true RB rows. FB rows remained exact under the existing R17/R18 position contract.

- maximum RB receiving-yard mean delta: `3.552713678800501e-15` yards
- minimum canonical/adapted Spearman rank correlation: `0.9999999999999999`
- canonical mean parity: PASS
- non-RB arrays exact: PASS
- other RB component arrays exact: PASS
- `rush_rec_yards = rush_yards + adapted rec_yards`: PASS
- allocation trace exact: PASS
- adapted receiving yards nonnegative: PASS
- all draws finite: PASS
- deterministic replay: PASS
- canonical result object unmutated: PASS
- canonical `scripts/simulation_v2.py` blob unchanged: `887e9c776ab112276ec8281195b0fed790ea0551`

## Full-slate and production-isolation checks

The current certified full-slate validator passed both **before** and **after** shadow scoring.

The immutable governed replay retained the current certified composition:
- QB: M38/C2 governed path
- WR: `WR_R15_PRODUCTION_MODEL_V1`
- TE: TE-R5P
- RB rushing: promoted P3 authority
- RB receiving-tail scorer: **shadow only**

Critical governed replay source hashes matched exactly. Pristine source bytes remained unchanged, and the working production projection/pricing files remained byte-identical before vs after R20.

- sportsbook inputs added to shadow scorer: `0`
- current/future 2026 outcomes used: `0`
- production parameters changed: `0`

## Frozen gates

Every R20 gate passed, including:
- exact R19 artifact metadata/model/pool hashes and lineage;
- exact governed replay + raw paid artifact metadata;
- exact governed source hashes and replay contract;
- certified validator pass before/after;
- exact Week-1 real-slate shape;
- unique live player/team/event keys;
- complete finite RB live inputs;
- strict-prior R8 history and exact serialized feature order;
- exact R9 RB-room conservation;
- finite R11/R16 probabilities and REST80 zero rule;
- complete 107-row player shadow casebook;
- R18 mean/non-RB/component/rush-rec/allocation/rank/determinism parity;
- canonical simulation unchanged;
- pristine and working production bytes unchanged;
- zero sportsbook inputs to the shadow scorer;
- zero 2026 outcomes;
- zero production parameter changes;
- shadow-only status.

## Evidence artifact files

- `rb_r20_result.json`
- `rb_r20_shadow_casebook.csv`
- `rb_r20_adapter_player_audit.csv`
- `rb_r20_frozen_adapter_audit.csv`
- `rb_r20_source_hash_audit.csv`

## Production boundary

Do **not** describe R20/R19/R17/R16 as active production RB receiving-tail code. R20 certifies deployability and mechanical parity on a real 2026 slate in shadow mode only.

No production RB receiving mean, target entitlement, canonical simulation, pricing output, or sportsbook role changed in R20.

## Exact next step

The clean scientific next step is **R21: prospective 2026 shadow grading**, not immediate silent promotion.

Freeze the R21 grading protocol **before** using 2026 Week-1 outcomes. Once the scored Week-1 slate is complete and outcomes are available, grade the already-frozen R20 player casebook without refitting or changing R19/R20 probabilities/distributions. The prospective evaluation should test calibration/discrimination of p30/p50 and distribution quality (CRPS, tail Brier, q90/q95 pinball/coverage) against the untouched canonical comparator, while separately auditing mean preservation and production isolation.

Until a separately governed promotion decision is made, R20 remains `PASS_SHADOW_ONLY`.
