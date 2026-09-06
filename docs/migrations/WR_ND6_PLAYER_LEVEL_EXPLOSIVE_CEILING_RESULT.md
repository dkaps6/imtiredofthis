# WR-ND6 — Player-Level Explosive Ceiling Diagnostic Result

## Status

Canonical WR-ND6 diagnostic completed successfully. Scientific disposition: `NO_ACTIONABLE_EXPLOSIVE_CEILING_SIGNAL`.

No production change is authorized by this result.

## Canonical lineage

- Branch: `research-wr-nd6-player-level-explosive-ceiling`
- Tested SHA: `ce17a3d8c6f5cd9ed1bb63249e61e72540199f8a`
- Exact M38 parent: `b98518d97b3038f471aee9ae3201009b2c70bb29`
- GitHub Actions run: `34056970854`
- Job: `101550525907`
- Artifact ID: `9996440756`
- Artifact name: `wr-nd6-player-level-explosive-ceiling`
- Artifact digest: `sha256:2382554e2c723589ae502a1bee7bb0df08e4f811cf4b83d1c8b7e380269edc81`

## Integrity checks

- M38 receiving-yard rows: `4647`
- M38 receiving-yard MC MAE: `17.099904733366`
- M38 RMSE: `25.196099510685915`
- M38 bias: `-5.238640833494836`
- M38 correlation: `0.5679458508349821`
- WR evaluation rows: `2130`
- Target reconstruction MAE: `2.076010432545868`
- Fixed history window: last `8` eligible prior games
- Sportsbook inputs used: `false`
- Postseason participation used: `false`
- WR-CB assignment used: `false`
- Player-defense interactions tested: `false`
- Model fitting used: `false`
- Production changed: `false`

## Frozen gate

A candidate required all of the following:

- coverage >= `0.75`
- Spearman with receiving-yard residual >= `+0.08`
- high-vs-low receiving-yard residual gap >= `+4.0` yards
- `UNDER25` enrichment >= `1.25x`
- at least one of `UNDER50` or `ACTUAL100` enrichment >= `1.25x`
- W2-18 residual gap > `0`
- W13-18 residual gap > `0`
- positive residual gap in at least 2 of WR1/WR2/WR3

No threshold was changed after results were visible.

## Signal results

| Signal | Side | Coverage | Spearman | Residual gap | UNDER25 | UNDER50 | ACTUAL100 | W2-18 gap | W13-18 gap | Role positives | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `PLAYER_EXP20_PER_TARGET_PRIOR8` | player | 0.973239 | -0.022804 | 0.247116 | 0.992200x | 1.030024x | 1.008895x | 0.620076 | 2.416496 | 3 | No |
| `PLAYER_EXP40_PER_TARGET_PRIOR8` | player | 0.973239 | 0.029233 | 3.172645 | 1.165074x | 1.146775x | 1.263657x | 3.774125 | 1.807620 | 3 | No |
| `PLAYER_YAC_PER_RECEPTION_PRIOR8` | player | 0.967606 | 0.015009 | 1.041191 | 0.928286x | 0.924969x | 0.829562x | 0.984824 | 0.013592 | 1 | No |
| `PLAYER_AIR_PER_TARGET_PRIOR8` | player | 0.973239 | -0.014118 | -0.804860 | 0.965092x | 0.833275x | 0.642743x | -0.545915 | 0.575965 | 1 | No |
| `DEF_EXP20_PER_ATT_ALLOWED_PRIOR8` | defense | 1.000000 | 0.004347 | -1.036273 | 0.985051x | 1.019477x | 0.947618x | -0.915354 | 0.044095 | 0 | No |
| `DEF_EXP40_PER_ATT_ALLOWED_PRIOR8` | defense | 1.000000 | 0.014472 | 0.617412 | 1.044609x | 0.929319x | 1.001282x | 0.464959 | 1.232337 | 2 | No |
| `DEF_YAC_PER_COMPLETION_ALLOWED_PRIOR8` | defense | 1.000000 | -0.015596 | -1.970503 | 0.949602x | 0.872210x | 0.915342x | -1.867287 | -3.342498 | 1 | No |
| `DEF_AIR_PER_ATT_ALLOWED_PRIOR8` | defense | 1.000000 | -0.011054 | -1.288608 | 0.880529x | 0.899679x | 1.045172x | -1.318366 | 0.356259 | 2 | No |

## Interpretation

The strongest descriptive player-side near-signal was `PLAYER_EXP40_PER_TARGET_PRIOR8`: it had a positive 3.17-yard high-low residual gap, positive W2-18 and W13-18 gaps, positive WR1/WR2/WR3 slices, and `ACTUAL100` enrichment of 1.264x. It nevertheless failed the frozen gate materially because Spearman was only `0.0292`, the residual gap was below `4.0`, and `UNDER25` enrichment was only `1.165x`. It is not eligible for integration or threshold retuning.

The simple defense explosive-vulnerability families were clearly non-actionable. No defense-side candidate established independent evidence.

Because no player-side and no defense-side signal passed, the frozen ND6 plan does **not** authorize a player × defense interaction test. Do not combine the failed ND6 candidates after seeing these results.

## Disposition

`NO_ACTIONABLE_EXPLOSIVE_CEILING_SIGNAL`

The simple prior-8 player explosive-rate and opponent explosive-vulnerability families are closed under this specification. Further WR work should move to materially new pregame information rather than nearby history-window or threshold searches.
