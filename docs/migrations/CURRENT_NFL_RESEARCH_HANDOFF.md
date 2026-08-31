# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from `NEXT MIGRATION`.

## Repository / current state

- Repo: `dkaps6/imtiredofthis`
- Current research branch: `research-rb-m95q-expanded-historical-stable-workhorse`
- Stable continuity ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No M91-M95Q RB research has been promoted to production.
- Phase A production/data cleanup Waves 1-4 is complete.
- M94C remains the RB central carry/opportunity reference during tail research.
- M95F remains the safer stable-workhorse tail baseline after M95L/M95O.
- M95I vacancy/role-transition remains a separate promising diagnostic regime, not production-promoted.
- M95K was a strong 2024/2025 research result but failed sealed temporal confirmation in M95L.
- M95N established micro-regime dependence.
- M95O showed a fixed agreement gate does not transfer because workload priors/probability scale move across seasons.
- M95P found pregame dynamic workload-regime signal and showed 2023 was not a uniquely abnormal broad RB season.
- **M95Q successfully expanded exact comparable stable-workhorse temporal depth to 2020-2022 and passed its 2024 reconstruction controls.**

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
13. M95Q 2020-2022 exact reconstructions are now research/development data once used; they are not future pristine confirmation.
14. Broad M95P 2018-2025 lead-RB census is contextual evidence, not an exact substitute for M95K/M95F stable-workhorse labels.
15. Do not build one model per player or hand-pick an expert for a matchup after seeing the game. Conditional experts/gates must be pregame-defined and temporally validated.
16. Any derivative candidate still needs genuinely prospective/untouched confirmation before production promotion.
17. Broad QB mean research remains frozen after M90 while RB work is active.

## Production architecture / sportsbook rules

- `.github/workflows/full-slate.yml` is the only canonical production orchestration.
- `engine/engine.py` is retired/fails closed.
- Production defaults: `SEASON=2026`, `PRIOR_SEASON=2025`.
- NFL week comes from authoritative schedule mapping, never ISO-week arithmetic or a hardcoded opening date.
- Ordinary pushes use `FETCH_LIVE_ODDS=false`; explicit live-odds workflows only.
- Sportsbook/player lines remain downstream comparison/decision data.
- Required production validation remains compileall + strict repo audit + production-readiness audit + pytest.
- Do not weaken truth/integrity gates.

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

Separate parked hypothesis for later QB/WR research: explosive WR/TE/RB matchup probability may explain some very large QB passing-yard overs, while uniformly poor receiver matchups may suppress pass volume/yardage. This is not validated yet.

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

Frozen architecture:

- `feed_compact_env`
- shrink `k=4`
- logistic `C=.03`
- mass-preserving stable 20+ rerank
- conditional-ratio + mass-anchor 25+
- vacancy frozen M95I
- other RBs frozen M95F
- central carries M94C unchanged

2025 stable workhorses (`n=237`):

- 20+ AUC `.581185 -> .641164`, Brier `.186593 -> .171528`, mean probability exactly preserved at `29.60%`
- 25+ AUC `.591714 -> .612631`, Brier `.053017 -> .051386`, mean exactly preserved at `11.10%`

Because 2025 had been repeatedly inspected, K correctly advanced to sealed confirmation rather than promotion.

## M95L — sealed temporal confirmation FAILED

Authoritative:

- run `33429747106`
- job `99611940386`
- SHA `caa9401eb50f6980e2a2c35ddd8e54467f57cbef`
- artifact `9772316395`
- disposition `M95K_SEALED_TEMPORAL_CONFIRMATION_FAILED`

Mechanical source repair moved M94C player join `95.585% -> 100.000%` with verified aliases; the frozen gate was not lowered. A duplicate-tail merge was separately repaired before sealed metrics were exposed.

2023 W13-18 stable workhorses (`n=73`):

- 20+ M95F AUC `.727041`, frozen K/L `.545068`; Brier `.233221 -> .244446`
- 25+ M95F AUC `.533333`, frozen K/L `.442857`; Brier `.123614 -> .126356`

Exact M95K failed sealed confirmation. Do not retune it against opened 2023 labels.

## M95M — cross-season postmortem

Authoritative run `33433593731`, job `99624596080`, SHA `52b537dcd52561a1545a8c87b381c1ea5fca63da`, artifact `9773558859`.

- 2025 W13-18 M95K 20+ AUC gain `+0.086066` vs 2023 W13-18 `-0.181973`: true cross-season nonstationarity.
- 25+ unstable even in 2025 late (`-0.084388`) and 2023 (`-0.090476`).
- player current-season p95 25+ AUC: 2025 full `.717015`, 2025 late `.622363`, 2023 `.481746`.
- some broader feed/team signals survived 2023.
- sample depth did not explain the failure.

## M95N — conditional player-game micro-regime audit

Authoritative run `33435092627`, job `99629424342`, SHA `13f86f95e548a4675d2030340b7a9e2caf6e5172`, artifact `9774088423`.

Pregame-only 20+ regimes:

2023 W13-18:
- aligned-high `57.14%`
- aligned-low `16.67%`
- context-only `43.75%`
- history-only `11.11%`

2025 W13-18:
- aligned-high `43.33%`
- aligned-low `13.33%`
- context-only `15.38%`
- history-only `41.67%`

Aligned-high > aligned-low was stable, but the disagreement side flipped by season. `micro_regime_dependence_supported = 1`; no sufficiently sampled pregame resolver emerged. 25+ remained sparse.

## M95O — fixed agreement gate FAILED

Authoritative:

- run `33437157679`
- job `99636245739`
- SHA `d194a69a4d8939067b1c7d495de438c3062822eb`
- artifact `9774831420`
- disposition `RETAIN_M95O_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

20+ AUCs:

- 2024 W16-18: M95F `.619048`, M95K `.725275`, M95O `.659341`
- 2025 full: `.581185`, `.641164`, `.589605`
- 2025 W13-18: `.646858`, `.732923`, `.635246`
- 2023 W13-18: `.727041`, `.545068`, `.568027`

A fixed 2024 agreement gate partially reduced K's 2023 damage but did not solve instability and discarded much of its 2025 value. Absolute context/feed score meaning shifts by season.

## M95P — dynamic workload-regime / population-prior audit

Authoritative:

- run `33437999902`
- job `99639034733`
- SHA `a5b36ecb11e4eafabc2d4874e32b22fd9256bc2d`
- artifact `9775138348`
- disposition `M95P_AUDIT_COMPLETE_NO_PRODUCTION_CHANGE`

M95P broad 2018-2025 lead-RB census showed 2023 was not a uniquely weird season. 2023 late lead-RB 20+ was `23.91%` and 25+ `7.61%`; 2021/2024 had equal or higher broad late-tail rates on relevant measures.

Exact 2023-2025 stable-workhorse 20+ rates were relatively close, but M95F calibration gaps varied substantially. Pregame dynamic-regime features showed modest signal, especially team prior-four workload and league workload-state measures.

Important M95P rule: the broad lead-RB census is context only, not an exact M95F/M95K backtest substitute. This motivated M95Q.

# Latest completed migration: M95Q — expanded exact historical stable-workhorse reconstruction

Full results: `docs/migrations/M95Q_RB_EXPANDED_HISTORICAL_STABLE_WORKHORSE_RESULTS.md`.

Authoritative:

- workflow `M95Q RB Expanded Historical Stable Workhorse`
- run **`33450395426`**
- run number **7**
- results job **`99680228542`**
- tested SHA **`92bd39f624fca6ab79f328bd856c11197d4894e0`**
- artifact **`9779790912`**
- artifact SHA256 **`ed2ae370f9832ac8509b8a85a7359334c66329eb57c726818cdbe8776a227062`**
- execution success
- disposition **`M95Q_EXPANDED_PANEL_READY`**
- feature search `0`; coefficient search `0`; sportsbook `0`; production change `0`

### M95Q mechanical repair

Initial reconstruction failures were historical-calendar mechanics, not scientific failures:

- 2019/2020 regular seasons end at Week 17, not Week 18;
- nflverse weekly data can expose postseason observations under Week 18 for pre-2021 seasons.

The final repair used season-specific regular-season calendars and a Q-only historical player-log adapter. No model logic or comparability gate changed.

### 2024 exact reconstruction controls

M91 RB universe:

- exact rows `1394`
- rebuilt rows `1394`
- overlap `1394` = **100.000%**
- carry-truth MAE **0.000000**
- exact-only `0`, rebuilt-only `0`
- gate **PASS**

Downstream M95F/M95G parity:

- exact rows `479`, recon rows `479`, overlap `466`
- overlap `97.2860%` vs precommitted >=95%
- workhorse-role agreement **100.000%**
- stable-mask agreement **100.000%**
- p20 corr **`.962355`**, p20 MAE **`.017481`**
- p25 corr **`.988349`**, p25 MAE **`.002663`**
- downstream parity **PASS**

The 13 exact-only / 13 recon-only downstream rows are explicitly retained in the audit. They were not waived; the exact M91 universe is perfect and the precommitted downstream parity requirements were satisfied.

### New comparable exact years

All three attempted years passed comparability:

- 2020: roster join `100%`, 43 raw 20+ features, stable `n=44`
- 2021: roster join `100%`, 43 features, stable `n=61`
- 2022: roster join `100%`, 43 features, stable `n=72`

Thus `new_exact_years_attempted=3`, `new_exact_years_comparable=3`.

### Historical M95F stable baseline diagnostics

2020:
- actual 20+ `27.27%`, predicted `25.15%`, gap `+2.13 pp`, AUC `.776042`, Brier `.156351`
- actual 25+ `11.36%`, AUC `.702564`

2021:
- actual 20+ `36.07%`, predicted `31.72%`, gap `+4.34 pp`, AUC `.497669`, Brier `.259780`
- actual 25+ `13.11%`, AUC `.679245`

2022:
- actual 20+ `29.17%`, predicted `26.71%`, gap `+2.46 pp`, AUC `.690009`, Brier `.188409`
- actual 25+ `2.78%`, AUC `.971429`, but only **2** 25+ positives so this AUC is not robust

2024 control:
- actual 20+ `34.18%`, predicted `29.03%`, gap `+5.14 pp`, AUC `.701567`
- actual 25+ `15.19%`, AUC `.568408`

### M95Q conclusion

M95Q answers its reconstruction question **yes**. The exact stable-workhorse panel can be expanded backward without replacing M95F/M95G semantics with a broad proxy. We now have materially more temporal diversity for testing the population/game-regime hypothesis.

The new panel also reinforces nonstationarity: M95F 20+ ranking ranges from strong in 2020 to nearly random in 2021, while calibration gaps and 25+ event density vary by season. This supports testing a dynamic pregame population prior, not another universal historical-ceiling multiplier.

M95Q does **not** rescue M95K, promote M95O, or justify a production formula.

# Current scientific interpretation

The system already individualizes player-week football inputs. The evidence points to three interacting levels:

1. **Global backbone** — M94C central carries / M95F stable-workhorse tail baseline.
2. **Role regime** — vacancy/transition is structurally different from stable incumbent workload (M95I).
3. **Population/game regime** — the meaning/calibration of historical feed and current context changes with contemporaneous league/team workload state (M95N-O-P-Q).

The main bottleneck is no longer lack of exact historical depth. M95Q solved that sufficiently for the next research candidate. The next step is to precommit a dynamic population-prior architecture and evaluate it across the expanded panel with strict temporal ordering.

# NEXT MIGRATION — M95R

Name: **M95R — Expanded-Panel Dynamic Population-Prior Candidate**

Primary question:

> Can a pregame workload-population prior adapt M95F stable-workhorse 20+ probabilities across seasons and game environments without recreating M95K's cross-season instability or harming the M94C/M95F backbone?

Required design:

- **Precommit the candidate before fitting**; do not search feature combinations after seeing results.
- Use the exact comparable stable-workhorse panel now available from M95Q plus the already-authoritative 2023-2025 traces.
- Keep M94C central carries unchanged.
- M95F is the stable-workhorse tail baseline.
- M95I vacancy/role-transition remains separate; do not mix vacancy rows into the stable candidate.
- Candidate inputs must be pregame-only and limited to population/workload-regime signals supported by M95P/M95N, such as:
  - league season-to-date / recent lead-RB 20+ workload rate;
  - team recent lead-RB 20+/25+ workload rates;
  - team recent mean lead-RB carries;
  - current pregame role/context state already available before kickoff.
- No sportsbook input.
- No target-week/postgame variables in the prior or gate.
- Use strict rolling/leave-future-out temporal evaluation; no random pooled split.
- Evaluate calibration first (Brier/logloss/mean-probability gap) and discrimination second (AUC), with 20+ as the primary stable-workhorse target.
- Treat 25+ as a secondary pooled diagnostic unless event counts support a predeclared valid gate; do not let a tiny seasonal 25+ sample drive selection.
- Include season-by-season and pooled results, regime slices, event counts, and a casebook of meaningful probability changes.
- Fail closed if gains depend on one season or if ordinary/global baseline behavior regresses materially.
- Do not retune exact M95K on opened 2023 labels.
- No production change in M95R itself.

Any architecture that survives M95R remains research-only until it receives genuinely prospective/untouched confirmation.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from the `NEXT MIGRATION` section. Preserve all modeling/validation rules and do not restart old research.
