# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from `NEXT MIGRATION`.

## Repository / current state

- Repo: `dkaps6/imtiredofthis`
- Current research branch: `research-rb-m95p-dynamic-workload-regime-audit`
- Stable continuity ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No M91-M95P RB research has been promoted to production.
- Phase A production/data cleanup Waves 1-4 is complete.
- M94C remains the RB central carry/opportunity reference during tail research.
- M95F remains the safer stable-workhorse tail baseline after M95L/M95O.
- M95I vacancy/role-transition mechanism remains a promising separate diagnostic branch, not production-promoted.
- M95K was a strong 2024/2025 research result but failed sealed temporal confirmation in M95L.
- M95N established micro-regime dependence.
- M95O showed a fixed agreement gate does not transfer because the workload population prior/probability scale moves across seasons.
- M95P expanded workload-regime context to 2018-2025, showed 2023 was not a unique RB outlier, and found modest pregame-only dynamic-regime signal.

## Non-negotiable modeling rules

1. Predict real football first.
2. Sportsbook/player-prop lines are downstream comparison/decision data only; never feed them upstream into football projections.
3. No fake/synthetic sportsbook lines.
4. Never weaken integrity/source/production gates to force green results.
5. Distinguish mechanical/data/source failures from scientific/model failures.
6. Mechanical fixes may not change candidate logic or validation gates after outcomes are exposed.
7. Preserve ordinary-game performance while fixing tails.
8. Do not manually boost tail coefficients after seeing validation.
9. Failed experiments remain evidence; do not erase or silently rewrite them.
10. 2025 was inspected repeatedly during M91-M95K and is not pristine confirmation.
11. 2023 W13-18 was opened in M95L and is not pristine for derivative candidates.
12. 2024 was used for M95K development/selection and is not independent confirmation.
13. Broad 2018-2025 M95P lead-RB census is contextual evidence, not an exact substitute for M95K stable-workhorse labels.
14. Do not build one model per player or hand-pick an expert for a matchup after seeing the game. Conditional experts/gates must be pregame-defined and temporally validated.
15. A derivative candidate after M95L-N-O-P still needs genuinely prospective/untouched confirmation before promotion.
16. Broad QB mean research remains frozen after M90 while RB work is active.

## Production architecture / sportsbook rules

- `.github/workflows/full-slate.yml` is the only canonical production orchestration.
- `engine/engine.py` is retired/fails closed.
- Production defaults: `SEASON=2026`, `PRIOR_SEASON=2025`.
- NFL week comes from authoritative schedule mapping, never ISO-week arithmetic.
- Ordinary pushes use `FETCH_LIVE_ODDS=false`; explicit live-odds workflows only.
- Sportsbook/player lines remain downstream comparison/decision data.
- 2026 Week 1 live player-prop acceptance has not yet been fully exercised in normal regular-season inventory.

## QB state — frozen after M90

M90 broad mean research was promoted/frozen after corrected M89/M90 semantics.

M90 headline:

- MAE ~`60.63 -> 56.56`
- RMSE `75.63 -> 69.63`
- corr `.173 -> .243`
- 100+ yard misses `81 -> 64`

Frozen PR #498 ensemble:

- MC `.208753`
- ML `.267121`
- State `.524126`

Separate parked hypothesis for later QB/WR research: explosive WR/TE/RB matchup probability may explain some very large QB passing-yard overs, while uniformly poor receiver matchups may suppress pass volume/yardage. This is not yet validated.

## Research sequence

1. Production/data cleanup — complete.
2. RB refinement — current.
3. WR refinement — after RB closure.
4. Dedicated TE research.

# RB durable findings

## M91-M94C — central opportunity foundation

- M91 2025 RB-only carry MAE `3.494731`, rush-yard MAE `21.018907`, rush+rec MAE `25.352140`.
- M92 oracle decomposition proved opportunity architecture is the main RB failure.
- M93/M93B: backfield concentration is real; universal sharpening helps the extreme tail but harms the middle.
- M94/M94B/M94C: explicit game-state decomposition improved team rush opportunity.

M94C 2025:

- team rush MAE `5.812091`
- RB carry MAE `3.411003`
- 20+ carry MAE `7.876590`
- 25+ carry MAE `11.954550`

Legacy all-player rush-yard guard:

- baseline `7.758864`
- M94C `7.762069`
- gain `-0.003205`

Do not waive. M94D showed that sharpening lead-RB share of an undersized carry pool cannot create realistic 25+ absolute workloads.

## M95A-D — matchup / quality / environment

- M95A: established workhorses perform better against weak pregame run defenses than strong ones across 2023-25; giant correlated defensive feature soup failed.
- M95B: validated RB + offense × opponent-defense architecture; recovered YBC/YAC, broken tackles, expected rushing yards, RYOE, 8+ box and time-to-LOS context.
- M95C: blocking/environment signals are more stable for mean projection; runner-created ability is more useful for upside/tails.
- M95D: motion/RPO/formation/box, participation/personnel and missed-tackle data did not improve mean projection but improved 100+ rushing-yard discrimination. Retain as upside context.

## M95E/F — workload-state tail baseline

M95F 2025:

- 20+ AUC ~`.846`
- 25+ AUC ~`.844`
- raw tail scores were badly overconfident before calibration
- stable workhorses remained overconfident even after calibration

M95F remains the safer stable-workhorse tail baseline after later failures.

## M95G/H/I — role / vacancy regime

M95G established **a vacancy is not a successor**. M95H validated only the >=70% RB-share entitlement target:

- AUC `.903118 -> .919599`
- Brier `.096200 -> .090868`

M95I authoritative run `33402566592`, job `99522191259`, artifact `9761827238`.

Vacancy 25+:

- AUC `.721739 -> .939130`
- Brier `.008840 -> .008445`
- logloss `.048953 -> .040330`

M95I remains diagnostic/not promoted.

## M95J — generic stable-week conversion failed

Run `33405821436`, job `99533036053`, artifact `9763096005`.

Generic current-week script/matchup/competition variables selected on 2024 failed to generalize to 2025 stable workhorses, motivating persistent player/team feed tendency and workload ceiling.

## M95K — feed tendency / carry ceiling

Authoritative:

- run `33411719023`
- job `99552629521`
- SHA `daa39544bd895084223532073b5cb9aa2eb4e872`
- artifact `9765397828`
- disposition `ADVANCE_M95K_TAIL_ARCHITECTURE_TO_SEALED_CONFIRMATION`

Frozen selected architecture:

- `feed_compact_env`
- shrink `k=4`
- logistic `C=.03`
- mass-preserving stable 20+ rerank
- conditional-ratio + mass-anchor 25+
- vacancy frozen M95I
- other RBs frozen M95F
- central carries M94C unchanged

2024 was the development/selection year: fit W13-15, select W16-18. On 2024 W16-18 stable workhorses, 20+ AUC `.619048 -> .725275`, Brier `.247201 -> .225884`.

2025 stable workhorses (`n=237`):

- 20+ AUC `.581185 -> .641164`, Brier `.186593 -> .171528`, mean probability exactly preserved at `29.60%`
- 25+ AUC `.591714 -> .612631`, Brier `.053017 -> .051386`, mean exactly preserved at `11.10%`
- player current-season p95/p90 25+ AUCs `.7170` / `.6840`

Because 2025 had been repeatedly inspected, K correctly advanced to sealed confirmation rather than promotion.

## M95L — sealed temporal confirmation FAILED

Authoritative:

- run `33429747106`
- job `99611940386`
- SHA `caa9401eb50f6980e2a2c35ddd8e54467f57cbef`
- artifact `9772316395`
- disposition `M95K_SEALED_TEMPORAL_CONFIRMATION_FAILED`

Mechanical source repair moved M94C player join `95.585% -> 100.000%` with verified GSIS-backed aliases; frozen `97%` gate was not lowered. A duplicate-tail merge was separately repaired before sealed metrics were exposed.

2023 W13-18 stable workhorses (`n=73`):

- 20+ M95F AUC `.727041`, frozen K/L `.545068`; Brier `.233221 -> .244446`
- 25+ M95F AUC `.533333`, frozen K/L `.442857`; Brier `.123614 -> .126356`

M95K therefore failed its predetermined sealed confirmation.

## M95M — cross-season postmortem

Authoritative run `33433593731`, job `99624596080`, SHA `52b537dcd52561a1545a8c87b381c1ea5fca63da`, artifact `9773558859`.

Key findings:

- 2025 W13-18 M95K 20+ AUC gain `+0.086066` vs 2023 W13-18 `-0.181973`: genuine cross-season nonstationarity, not merely late-season timing.
- 25+ unstable even in 2025 late (`-0.084388`) and 2023 (`-0.090476`).
- player current-season p95 25+ AUC: 2025 full `.717015`, 2025 late `.622363`, 2023 `.481746`.
- some broader feed/team ceiling signals remained useful in 2023.
- sample depth did not rescue K.

## M95N — conditional player-game micro-regime audit

Authoritative run `33435092627`, job `99629424342`, SHA `13f86f95e548a4675d2030340b7a9e2caf6e5172`, artifact `9774088423`.

Pregame-only regimes based on current-context rank and historical-feed score:

2023 W13-18 20+ rates:
- aligned-high `57.14%`
- aligned-low `16.67%`
- context-only `43.75%`
- history-only `11.11%`

2025 W13-18:
- aligned-high `43.33%`
- aligned-low `13.33%`
- context-only `15.38%`
- history-only `41.67%`

Findings:

- aligned-high > aligned-low was stable across seasons;
- preferred side of disagreement flipped by season;
- `micro_regime_dependence_supported = 1`;
- secondary splits by volume, concentration, matchup and role momentum did not yield a clean sufficiently sampled resolver;
- 25+ remained too sparse.

## M95O — fixed agreement gate FAILED

Full results: `docs/migrations/M95O_RB_AGREEMENT_GATED_TAIL_RESULTS.md`.

Authoritative:

- run `33437157679`
- job `99636245739`
- SHA `d194a69a4d8939067b1c7d495de438c3062822eb`
- artifact `9774831420`
- disposition `RETAIN_M95O_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

A precommitted 2024-reference agreement gate partially reduced M95K's 2023 damage but did not solve the instability and lost much of K's 2025 value.

20+ AUCs:

- 2024 W16-18: M95F `.619048`, M95K `.725275`, M95O `.659341`
- 2025 full: `.581185`, `.641164`, `.589605`
- 2025 W13-18: `.646858`, `.732923`, `.635246`
- 2023 W13-18: `.727041`, `.545068`, `.568027`

Major M95O finding: absolute context/feed score meaning shifts by season because stable-workhorse population priors/calibration move. Fixed 2024 thresholds classified 2023 very differently and failed prospectively.

## Latest completed migration: M95P — dynamic workload-regime / population-prior audit

Full results: `docs/migrations/M95P_RB_DYNAMIC_WORKLOAD_REGIME_RESULTS.md`.

Authoritative:

- workflow `M95P RB Dynamic Workload Regime Audit`
- run **`33437999902`**
- job **`99639034733`**
- tested SHA **`a5b36ecb11e4eafabc2d4874e32b22fd9256bc2d`**
- artifact **`9775138348`**
- artifact SHA256 **`01eec3d8a80aee77d1c17fd7c4c56065762b0148e79a9c1b57d877035301a04e`**
- execution success
- disposition **`M95P_AUDIT_COMPLETE_NO_PRODUCTION_CHANGE`**
- feature search `0`; coefficient search `0`; new model fit `0`; sportsbook `0`; production change `0`

### M95P broadened historical context to 2018-2025

Broad population: each team's lead RB in each team-week, derived from nflverse weekly player stats. This is workload-regime context, not the exact M95K stable-workhorse label.

2023 full regular season:

- lead-RB 20+ rate `18.75%`
- lead-RB 25+ rate `5.51%`
- mean lead carries `15.08`

2023 Weeks 13-18:

- lead-RB 20+ `23.91%`
- lead-RB 25+ `7.61%`
- late-season z-scores only `+0.854` / `+0.708`
- both around the 75th percentile of the eight-season sample
- `2023_broad_workload_extreme_1p5sd = 0`

Therefore **2023 was not a uniquely weird RB workload year**. 2021 had a higher late-season 20+ rate (`26.09%`), and 2024 had higher late 20+ (`24.19%`) and 25+ (`9.14%`) rates.

### Exact stable-workhorse model trace still only 2023-2025

Late-season exact stable-workhorse 20+ rates:

- 2023 `32.88%`
- 2024 `35.06%`
- 2025 `28.24%`

Range only `6.83 pp`.

But M95F calibration gaps (actual minus predicted 20+ probability):

- 2023 `+16.69 pp`
- 2024 `+7.69 pp`
- 2025 late `-1.20 pp`
- 2025 full `-7.66 pp`

Late calibration-gap range = **`17.88 pp`**.

Interpretation: the instability is not simply that 2023 had impossible RB workloads; the model probability scale is failing to adapt to the contemporaneous workload prior.

### Pregame-only regime signal

Across exact 2023-2025 stable-workhorse rows (`n=371`):

- team prior-four 25+ lead-RB rate vs upcoming 20+ event: Spearman `.2064`
- team prior-four mean lead carries vs upcoming 20+ event: `.2037`
- team prior-four 20+ rate vs upcoming 20+ event: `.1913`
- season-to-date league lead-RB 20+ rate vs M95F calibration error: `.1823`

League prior-four lead20 quartiles:

- Q1 low: actual 20+ `17.53%`, M95F `27.86%`, gap `-10.33 pp`
- Q2: actual `24.18%`, M95F `24.37%`, gap `-0.19 pp`
- Q3: actual `34.34%`, M95F `23.68%`, gap `+10.66 pp`
- Q4 high: actual `30.95%`, M95F `28.94%`, gap `+2.01 pp`

The effect is not perfectly monotonic, so M95P does not justify a production formula. But `pregame_regime_structure_detected_diagnostic = 1` and the dynamic-prior hypothesis remains worth pursuing.

### Backtesting-width conclusion

For rare-event tail architecture, the exact 2023-2025 panel is too narrow to treat three seasons as enough evidence. M95P expanded broad workload context to eight seasons and showed why more seasons matter. However, the exact comparable M95K/M95F stable-workhorse model trace remains only 2023-2025.

Do not use the broad lead-RB census as if it were the exact model backtest. The next migration should expand the exact leakage-safe historical reconstruction.

# Current scientific interpretation

The system already individualizes player-week football inputs. The current problem is not simply one-size-fits-all vs one-model-per-player. The evidence now points to three interacting levels:

1. **Global backbone** — M94C central carries / M95F stable-workhorse tail baseline.
2. **Role regime** — vacancy/transition is structurally different from stable incumbent workload (M95I).
3. **Population/game regime** — the meaning/calibration of historical feed and current context changes with contemporaneous league/team workload state (M95N-O-P).

M95K remains a legitimate 2024/2025 research success but is not robust enough for promotion after M95L. M95O is not promoted. M95P is diagnostic only.

The immediate priority is now **more exact historical temporal depth**, not another coefficient search.

# NEXT MIGRATION — M95Q

Name: **M95Q — Expanded Historical Stable-Workhorse Backtest Reconstruction**

Primary question:

> Can we reconstruct comparable leakage-safe stable-workhorse model traces for additional modern seasons so future dynamic-prior / conditional-tail candidates are evaluated across more than 2023-2025?

Required design:

- target at least 2020-2022 first; include older modern seasons only if source contracts remain comparable;
- preserve M94C/M95F/M95G semantics rather than replacing them with M95P's broad lead-RB proxy;
- source roster/injury/depth/workload information from records available before each target game;
- keep player identity joins stable-ID-backed and auditable;
- document source coverage by season and fail closed if a historical year is not comparable;
- reconstruct central/team environment, role-state and calibrated 20+/25+ baseline outputs without using postgame target-week information;
- rerun baseline calibration/workload diagnostics across the expanded exact panel;
- do not fit a new dynamic-prior candidate until the reconstruction is validated;
- no sportsbook input;
- no production change;
- preserve all failed M95K/L/O evidence.

If exact historical reconstruction proves feasible, the migration after M95Q should precommit a dynamic population-prior candidate using the expanded panel. Any eventual promotion still requires genuinely prospective/untouched confirmation.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from the `NEXT MIGRATION` section. Preserve all modeling/validation rules and do not restart old research.