# NE–SEA Pregame Current-Stack Counterfactual V1 — Run5 Unsupported-Market Replay Compatibility Repair

## Status
This document preserves Run5 exactly as observed and freezes the minimum diagnostic-only repair for the second historical-replay compatibility seam. It does not alter any football projection, player role, availability decision, scientific parameter, sportsbook line/price, promoted production code, or production workflow.

## Preserved Run5
- Run: `34510166458`
- Job: `102982031566`
- Head: `b60df0fe4b86720f0e345d2c61d0c58a9a6b7c85`
- Artifact: `10165619124`
- Artifact digest: `sha256:3751426d212330a04240e4fd185f5e11ea617216eeef342b3b2d05cf97f3aec6`
- Pregame personnel-state gate: PASS
- Strict-prior football-stack rebuild: PASS
- Immutable Sep-7 sportsbook artifact download/staging: PASS
- Run3 opponent-map mechanical repair: PASS
- Metrics context: PASS, 103 rows
- `metrics_ready.py`: PASS, including `data/opponent_map_from_props.csv` with 22 rows
- Final promoted pricing entrypoint: ENTERED, then stopped on unsupported historical replay offers
- Disposition: `NE_SEA_COUNTERFACTUAL_RUN5_MECHANICAL_UNSUPPORTED_MARKET_REPLAY_COMPATIBILITY_NO_FINAL_PRICING`

## Scientific evidence preserved from Run5
At diagnostic `asof_utc=2026-09-09T19:00:00Z` the frozen availability and strict-prior football reconstruction remained unchanged and valid:
- NE–SEA production eligible at the frozen pregame clock;
- Rhamondre Stevenson active `RB1` and opportunity eligible;
- TreVeyon Henderson definitive unavailable and opportunity ineligible;
- Sam Darnold active `QB1` and opportunity eligible;
- A.J. Brown eligible;
- sportsbook inputs to availability/football generation: zero;
- target-game/current Week-1 history excluded from PlayerForm publication;
- target-game PBP enrichment intentionally not run;
- Rhamondre Stevenson P3 rush-yard mean: `30.790770`;
- Jadarian Price P3 rush-yard mean: `18.159696`.

Immediately before the failure, current promoted pricing initialization also reached the promoted QB synthesis, RB P3, target-entitlement + TE-R5P + WR-R15 layer, and QB C2 distribution integration. The QB C2 integration reported its production distribution gate PASS. No scientific/model failure was observed.

## Exact Run5 failure
The preserved Sep-7 NE/SEA staging contained 162 side rows across 22 players and seven historical provider markets. The Run5 opponent-map repair successfully materialized 22 NE/SEA opponent-map rows, and `metrics_ready.py` validated all required inputs.

The current full-roster pricing simulator then failed because two historical `player_pass_tds` player/market keys did not have canonical pre-generated football distributions:
- Drake Maye, NE, `player_pass_tds`;
- Sam Darnold, SEA, `player_pass_tds`.

The immutable source artifact contains exactly 8 `player_pass_tds` side rows: Drake Maye and Sam Darnold, each at line 1.5, DraftKings and FanDuel, OVER and UNDER. Those eight rows are part of the provider snapshot, but they are not part of the current production model-facing supported-market contract.

## Production contract audit
The current production boundary `scripts/materialize_live_props_for_model_v1.py` explicitly treats provider output as a superset of model capability and quarantines unsupported markets before modeling. Its `SUPPORTED_MARKETS` are:
- `player_pass_yds`
- `player_rush_yds`
- `player_reception_yds`
- `player_receptions`
- `player_rush_reception_yds`
- `player_anytime_td`

`player_pass_tds` is intentionally absent. The downstream production adapter `scripts/materialize_pricing_offers_v1.py` declares the same supported-market set. Therefore passing historical `player_pass_tds` rows directly to pricing is not faithful to the current production sportsbook boundary.

Run5 bypassed the live compact/materialization adapters by design because the replay source is an immutable historical side-row artifact rather than a current compact `offers_json` snapshot. That bypass preserved sportsbook values correctly but also allowed an unsupported provider market to reach the simulator.

## Frozen minimum repair
In the diagnostic staging step only, after reconstructing the exact 162 active NE/SEA historical side rows and before writing the model-facing `outputs/props_raw.csv`:
1. import/read the current production `SUPPORTED_MARKETS` authority from `scripts/materialize_live_props_for_model_v1.py`;
2. partition the exact staged rows into supported vs unsupported markets without altering any row values;
3. require the unsupported partition to be exactly 8 rows, exactly market `player_pass_tds`, exactly players Drake Maye and Sam Darnold;
4. write those 8 rows to a diagnostic quarantine artifact for preservation;
5. write only the supported 154 side rows to the model-facing replay `outputs/props_raw.csv`;
6. derive the replay opponent map from those supported rows and the already-frozen Week-1 schedule as in Run5;
7. rerun the unchanged current metrics/pricing scripts.

The 162-row immutable sportsbook source remains the authoritative historical capture. The 154-row model-facing replay is only the current production-supported subset, mirroring the live production market-capability boundary. No line, side, book, price, player identity, role, football projection, distribution, or model parameter may be changed.

Do not reconstruct compact `offers_json` or invoke live acquisition in this replay: doing so would add unnecessary transformation surface and could refetch/mutate provenance. The repair is only the deterministic supported-market boundary that Run5 omitted.

If another historical-replay compatibility seam appears, stop and classify it before any further repair.

## Interpretation boundary
This is diagnostic plumbing only. It does not authorize adding passing-TD science, changing QB distributions, teaching from sportsbook values, or modifying production. The missing `player_pass_tds` canonical distribution is expected under the current production-supported market contract and is not evidence that the promoted QB model failed.
