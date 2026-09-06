# QB Individual Mechanism Stability Result

## Canonical run

- Run: `34066616942`
- Job: `101576382095`
- Tested SHA: `82aae80bf547ba809e5a9e36ce7d5de29fb581f6`
- Artifact: `qb-individual-mechanism-stability`
- Artifact ID: `9999139159`
- Artifact SHA256: `9657f549e465ce5da340bfa079d3d84626a85f05f5c86bfb9547219a51c5e50e`
- Conclusion: `success`

## Frozen disposition

`QB_PLAYER_MECHANISMS_REQUIRE_CONTEXT_REGIMES`

The frozen useful-stability gate did not pass because attempt-component share lacked the required cross-season rank stability.

## Metrics

- canonical rows: `884`
- qualifying cross-season QBs: `22`
- same dominant-component rate: `0.6363636363636364`
- attempt-share Pearson: `0.3023469916557545`
- attempt-share Spearman: `0.20609824957651046` — failed frozen `0.30` gate
- YPA-share Pearson: `0.2625114420656069`
- YPA-share Spearman: `0.33370976849237727`
- passing-yard MAE Pearson across seasons: `0.32751708909848765`
- passing-yard MAE Spearman: `0.3709768492377189`
- attempt-bias sign persistence: `0.7727272727272727`
- YPA-bias sign persistence: `0.4090909090909091`

## Interpretation

There is meaningful player-specific structure — 63.6% retained the same dominant mechanism and attempt-bias direction persisted for 77.3% — but the relative amount of attempt-driven error is not stable enough to treat each QB as having one fixed correction archetype.

The correct next independent-QB direction is therefore **player + game-context regime**, not player-only correction. A QB's own history remains important, but whether a specific game is volume-driven or efficiency-driven must be informed by timestamp-safe pregame context such as expected play/pass environment, opponent, protection/pressure environment, receiver availability/quality, game script and related football states already researched in prior QB migrations.

This result does not reopen broad M56-M88 feature hunting and does not authorize per-QB constants.

- Sportsbook inputs used: `false`
- Model fitting used: `false`
- Production changed: `false`