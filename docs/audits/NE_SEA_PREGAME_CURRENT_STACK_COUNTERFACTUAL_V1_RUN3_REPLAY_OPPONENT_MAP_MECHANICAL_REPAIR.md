# NE–SEA Pregame Current-Stack Counterfactual V1 — Run3 Replay Opponent-Map Mechanical Repair

## Status
This document preserves the corrected-clock Run3 failure and freezes the minimum replay-only repair. It does not alter any football projection, player role, availability decision, scientific parameter, sportsbook price, or production file.

## Preserved corrected-clock Run3
- Run: `34509425408`
- Job: `102979570511`
- Head: `3beb00ee518b8076873a5f54dffed51ee2009e5e`
- Artifact: `10165290158`
- Artifact digest: `sha256:41c0e81b489f67d4b57bb98a249865e019fa644c04b4cf891b51734203e068c5`
- Pregame personnel-state gate: PASS
- Strict-prior football-stack rebuild: PASS
- Historical sportsbook download/staging: PASS
- Final pricing replay: MECHANICAL FAILURE BEFORE PRICING
- Disposition: `NE_SEA_COUNTERFACTUAL_RUN3_MECHANICAL_REPLAY_OPPONENT_MAP_MISSING_NO_FINAL_PRICING`

## Scientifically useful pre-pricing evidence preserved from Run3
At diagnostic `asof_utc=2026-09-09T19:00:00Z`:
- all 16 Week-1 games were `NOT_YET_REQUIRED` and production-eligible;
- Rhamondre Stevenson was active `RB1`;
- TreVeyon Henderson was definitive unavailable and opportunity-ineligible;
- Sam Darnold was active `QB1`;
- A.J. Brown was active/eligible;
- sportsbook inputs to availability/football were zero;
- current/future 2026 Week-1 history was dropped from PlayerForm publication;
- target-game PBP enrichment was intentionally not run.

Current promoted P3 rebuilt successfully before replay failure. Relevant diagnostic means:
- Rhamondre Stevenson: stack_att `7.301058`, rush-yard mean `30.790770`;
- Corey Kiner: stack_att `5.909515`, rush-yard mean `22.776669`;
- Jadarian Price: stack_att `3.138717`, rush-yard mean `18.159696`;
- George Holani: stack_att `4.310829`, rush-yard mean `21.151877`.
These are valid counterfactual football outputs from the frozen current stack, but Run3 produced no final sportsbook pricing rows.

## Exact mechanical failure
The diagnostic staged 162 immutable Sep-7 NE/SEA sportsbook side rows into `outputs/props_raw.csv` after football generation, then called the current metrics/pricing path.

`run_metrics_context.py` ran and reported 103 metric rows. `metrics_ready.py` then failed on its canonical required-input contract because `data/opponent_map_from_props.csv` had not been materialized:

`RuntimeError: Required artifact missing or empty: .../data/opponent_map_from_props.csv`

The current Full Slate live path normally materializes this sportsbook-derived opponent mapping before metrics readiness. The diagnostic replay bypassed the live acquisition path by design and therefore omitted that downstream compatibility artifact.

## Frozen minimum repair
After staging immutable historical NE/SEA props, write `data/opponent_map_from_props.csv` deterministically from those same staged rows and the already-frozen Week-1 schedule context. The map must contain, at minimum, canonical contract fields `player`, `team`, `opponent`, `season`, `week`, and may include `player_clean_key`, `event_id`, and `game_timestamp` for traceability.

The map is allowed to contain only teams NE and SEA, only their mutual opponent pairing, season 2026, week 1, and the same preserved sportsbook event IDs. It must not fetch any data, infer roles, modify opportunity, or alter sportsbook values.

Then rerun the unchanged current metrics/pricing scripts. If another historical-replay compatibility artifact is missing, stop and classify before any additional repair.

## Interpretation boundary
This repair exists solely to replay preserved historical lines through the current downstream pricing contract. It cannot affect the already-valid P3/PlayerForm/QB/WR/TE football reconstruction and cannot authorize production/scientific changes.
