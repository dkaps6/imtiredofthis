# NFL Research Active Ledger

Purpose: preserve the currently authorized production anchors, unresolved concerns, active experiments, failed experiments, and future lanes so a new architecture experiment never silently abandons prior evidence.

Last updated: 2026-09-06/07 project session.

## Permanent objective

Build leakage-safe football models whose independent player/game distributions are more accurate than the betting market. Sportsbook data remains downstream only. Full Slate remains canonical production authority.

## QB

### Production anchor
- Mean architecture: `QB_PASS_SYNTHESIS_V1` / M89-M90.
- M89 canonical run: `33331073376`.
- M90 prospective run: `33333730480`.
- Frozen mean model: Ridge residual correction, alpha 20, correction cap +/-45, 21 football-only features.
- M90 2023 prospective MAE: 60.632751 -> 56.559869; RMSE 75.634635 -> 69.629921; bias -27.248932 -> -6.606555.
- QB mean feature hunting remains closed this cycle unless genuinely new information authorizes a new migration.

### Supported cross-position evidence
- QB-WR shared pass-volume audit run `34066549394`: `STRONG_SHARED_PASS_VOLUME_MECHANISM`.
- 2025 WR target-mass residual vs QB attempt residual: Pearson 0.697305, Spearman 0.670648, same-sign 0.731818.
- Receiving ecosystem audit run `34075068419`: `MATERIAL_PASS_RECEIVING_CONSERVATION_GAP`.

### Open QB work
- 2026 W1 no-odds promoted projections still require final freeze/export before attaching the 32 manually supplied sportsbook passing-yard lines.
- QB distribution-shape architecture is now being tested through Joint Pass/Receiving Conservation V1; M89/M90 remains the mean anchor.

## WR

### Production anchor
- M38 within-WR target hierarchy multipliers: `1.40 / 1.14 / 0.91 / 0.78`.
- Exact M38 ref: `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- Exact M37 comparison parent: `ba83fd05412a36309822cac6aa9cc5003388b073`.
- Full-stack integration protocol: `docs/migrations/WR_FULL_STACK_INTEGRATION_PROTOCOL.md`.

### Multi-season confirmation — PRESERVED
WR-R1 canonical run `34058453941`, job `101554556794`, artifact `9997412312`, digest `sha256:d75bf1233bdd48d119508c0631a0f904e0256239e471bb72804d705d73ed6cfe`.
- Disposition: `M38_MULTISEASON_CONFIRMED`.
- WR rec-yard rows 2020-2025: 12,396.
- M38 beat M37 in all 6 seasons.
- pooled 2020-2025 MAE: 24.215219 -> 23.247553, improvement 0.967666 yards.
- pooled 2024-2025 MAE: 23.220444 -> 22.366178, improvement 0.854267 yards.
- 2025 exact all-receiver parity: n=4,647, MAE=17.099904733366.
This result is not superseded by the receiving-ecosystem work; the joint architecture explicitly preserves M38 within-WR hierarchy unless a separately frozen full-stack replacement beats it.

### Other preserved research
- ND1-ND6 dispositions remain evidence; failed gates are not to be lowered or repackaged.
- WR-R9 NGS source audit run `34073245454`: `NGS_RECEIVING_SOURCE_PARTIAL_ONLY`.
  - pooled join coverage 0.457406, below frozen 0.60 full-backbone gate;
  - source remains legitimate for a separately frozen selected-cohort diagnostic and is not discarded.

### Open WR work
- WR continues inside the joint receiving ecosystem through group target mass, M38 within-room hierarchy, catch conversion, efficiency, and distribution tails.
- No WR production replacement is authorized merely by a diagnostic result.

## RB

### Production anchor
- RB P3 W1 promoted main commit: `754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`.
- Full Slate production run: `33993831181`.
- Artifact: `9977465711`.
- P3 parity: 1,393 rows; rushing-yards MAE 19.949524; carry MAE 3.357494.

### Current-role carry audit — COMPLETED
Canonical branch: `audit-rb-2026-w1-current-role-carry-authority`.
- Result commit: `f27eccbe3007ae25ffe55c3f44e22ccaa00961c4`.
- Run: `34060593280`.
- Job: `101560300616`.
- Artifact: `9997365508`.
- Digest: `sha256:8c856deab95fdb61358754eacbe6a2c0bd6a7aedd8a63b3f8109e2638425ffd9`.
- Disposition: `CURRENT_ROLE_PRESERVED_BUT_NOT_DIRECT_ALLOCATION_INPUT`.
- 107 rows / 32 teams / 100% Ourlads-to-P3 identity coverage / P3 parent parity max abs diff 0.0.
- Depth-role vs model-role mismatches: 26 rows.
- Depth-role vs projected carry-order mismatches: 45 rows.
- Current depth-rank-1 back not projected carry leader: 7 teams (ARI, CLE, GB, HOU, KC, NE, SEA).

### Direct depth remaps — REJECTED, NEVER TO BE REPACKAGED
Role-Order Remap V1:
- run `34063904515`, job `101569200535`, artifact `9998334300`.
- disposition `ROLE_ORDER_REMAP_V1_NOT_ACTIONABLE`.
- baseline carry MAE 3.482576 -> candidate 4.106428; rushing-yard MAE 20.424163 -> 22.836638.

Role-Order Remap V2 Individual:
- canonical run `34065137207`, job `101572457042`, artifact `9998696792`, digest `sha256:ea37c8f18152605b26b0bc91ed1279d1461ee33c60dc6a302c1a85bd2c732f10`.
- disposition `ROLE_ORDER_REMAP_V2_REJECTED`.
- baseline carry MAE 3.482576 -> candidate 4.509012; rushing-yard MAE 20.424163 -> 24.374262.
The conclusion is specific: current depth information matters, but hard-remapping current depth order onto carry magnitudes is not a valid correction.

### Individual-player persistence diagnostic — SUPPORTED
RB-PD2 canonical run `34064637295`, job `101571138337`, artifact `9998549410`, digest `sha256:3ab28410b055ef2dcf35635e86dd93a9f45cbeba971764bad313df13080996c2`.
- result commit `a4173a7a06cd72c93c1064cd377977c9fd404c16`.
- disposition `RB_PLAYER_ERROR_PERSISTENCE_DETECTED`.
- 1,393 canonical rows; 882 scoreable rows; 148 players; last 8 strictly prior same-player games, minimum 4.
- carry directional persistence: Spearman .15433, quartile gap +1.507 carries, sign agreement 62.52%.
- carry difficulty persistence: Spearman .24127, abs-error quartile gap +1.789 carries.
- yard directional persistence: Spearman .14358, quartile gap +12.432 yards, sign agreement 60.43%.
- yard difficulty persistence: Spearman .33056, abs-error quartile gap +16.696 yards.
All four frozen diagnostics passed. This supports a separately frozen conservative player-residual calibration/uncertainty test; it does not itself authorize a player correction.

### Receiving baseline established
Receiving ecosystem audit, 2020-2025 RB rows: 13,282.
- targets MAE 1.41995
- receptions MAE 1.15459
- receiving-yards MAE 10.25671
- rushing-yards MAE 20.09053
- rush+receiving-yards MAE 26.62563

### Open RB work
- Next rushing lane: separately frozen conservative use of strictly prior player-error persistence at its natural layers; no further naive depth remap.
- Dedicated RB receiving lane: role/routes/checkdowns/screens/two-minute/third-down/pressure/game-script/YAC mechanisms.
- Rushing and receiving mechanics remain separately auditable until the final joint RB distribution.
- Final RB output must include carries, rushing yards, targets, receptions, receiving yards, and rush+receiving yards.

## TE

### Baseline established
Receiving ecosystem audit, 2020-2025 TE rows: 11,423.
- targets MAE 1.82171
- receptions MAE 1.35664
- receiving-yards MAE 15.52938

### Open TE work
- Dedicated TE research lane is authorized.
- Reuse the WR scientific process, not WR coefficients.
- TE-specific mechanisms include role/route participation, inline/slot deployment, target hierarchy, catch conversion, YPR/YAC, middle-of-field/coverage behavior, red-zone usage, and tail behavior.

## Receiving ecosystem / joint architecture

### Frozen audit result
Run `34075068419`, job `101599465910`, artifact `10001852203`, digest `sha256:caf27f748cf4bf4acffffd5e87fb5b6f18391e227ae90b49786d6310ead72960`.

Historical accounting integrity passed: 3,230 aligned team-games, all-passer pass yards vs all-receiver receiving yards MAE 0.0120743.

Dispositions:
- `POSITION_GROUP_MISALLOCATION_SUPPORTED`.
- `MATERIAL_PASS_RECEIVING_CONSERVATION_GAP`.

Promoted QB vs modeled receiver sum, 2024-2025 n=884:
- MAE 17.45731
- median absolute gap 13.74895
- p90 absolute gap 36.27954

### Active Joint V1
Branch: `research-joint-pass-receiving-conservation-v1`.
Frozen plan commit: `6148f6ff3165f0c103059976d20b14eb06c853c5`.
Frozen implementation-spec commit: `4eb809a51f90bf0e741c9e5e6598be0c68f5ce27`.
Scientific run: `34076564092`, job `101603723220`.

Predeclared factorial architectures:
- B0 current
- C1 group target-mass calibration only
- C2 pass/receiving conservation only
- C3 joint

No production change is authorized from V1 without passing the frozen full-stack gates and then a separate production-integration migration.

## Receiving pass-volume attempt semantics — ACTIVE PARALLEL LANE

The historical component path explicitly documents that `rules_pass_rate` represents dropbacks/plays and separately derives `pass_attempts_per_dropback` for official QB attempts. Current `simulation_v2` still uses its team pass-count draw directly for receiver target allocation. This means receiver targets may currently be allocated from dropbacks rather than official pass attempts.

This was **not folded into Joint V1 after its plan was frozen**. A separate isolated candidate has been frozen and launched:
- Branch: `research-receiving-attempt-semantics-v1`.
- Frozen plan commit: `db53721150ea0d7e46b732792e577282b3fcc5bd`.
- Workflow head: `056d8ba5ecd5b985d38317f48291e84328264f1a`.
- Scientific run: `34076777066`, job `101604330317`.
- C4 converts dropbacks to official attempts before receiver target allocation while leaving RB rushing mechanics untouched.

## Future shared game models — queued, not abandoned

Once the player-level joint game state is sufficiently coherent and validated, build from the same simulator rather than separate disconnected models:
- anytime / first / last touchdown scorer;
- moneyline / win probability;
- game total;
- spread / margin distribution.

The existing player, team, pace, PROE, pressure, defense, injury, weather, matchup, usage, efficiency, and MC work should become inputs/building blocks for these game-level distributions. Future market odds remain downstream for evaluation/pricing, not football-model inputs.

## Non-negotiable research rules

- Full Slate only is production authority.
- Walk-forward, leakage-safe historical evaluation.
- Sportsbook data downstream only.
- Freeze hypothesis/method/gates before results.
- Mechanical repairs may restore frozen semantics; they may not alter scientific gates.
- Failed candidates remain documented evidence.
- No production anchor is replaced because a newer architecture sounds more realistic; replacement requires paired full-stack evidence.
