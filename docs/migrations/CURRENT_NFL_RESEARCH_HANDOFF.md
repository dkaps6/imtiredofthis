# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from `NEXT MIGRATION`.

## Repository / current state

- Repo: `dkaps6/imtiredofthis`
- Current research branch: `research-rb-nd1-forensic-atlas`
- Stable continuity ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No M91-M96E RB research or downstream RB market benchmark has been promoted to production.
- Phase A production/data cleanup Waves 1-4 is complete.
- M94C remains the RB central carry/opportunity reference during tail research.
- M95F remains the safer stable-workhorse tail baseline after M95L/M95O.
- M95I vacancy/role-transition remains a separate promising diagnostic regime, not production-promoted.
- M95K was a strong 2024/2025 research result but failed sealed temporal confirmation in M95L.
- M95N established micro-regime dependence.
- M95O showed a fixed agreement gate does not transfer because workload priors/probability scale move across seasons.
- M95P found pregame dynamic workload-regime signal and showed 2023 was not a uniquely abnormal broad RB season.
- **M95Q successfully expanded exact comparable stable-workhorse temporal depth to 2020-2022 and passed its 2024 reconstruction controls.**
- **M95R tested a precommitted rolling dynamic population-prior residual layer; it helped 2023, was neutral in 2024, but badly overcorrected 2025 and is not promoted.**
- **M95S separated population-mass calibration from player ranking: M95R overcorrection was visible in 2025 pregame context, while frozen M95K ranking helped 2025 but harmed 2023. The next step is one final constrained historical candidate, M95T.**
- **M95T was the final retrospective carry-tail candidate. It improved pooled Brier/logloss/AUC and repaired 2023 directionally, but failed the frozen cross-season stability gates because 2025 calibration still worsened materially. Per the stopping rule, new retrospective RB carry-tail candidates stop here; M94C/M95F is the conservative workload foundation for M96 rushing-yard synthesis.**
- **M96A attributed 2025 rushing-yard error jointly to opportunity and efficiency: perfect carries recovered 7.68 MAE yards, perfect efficiency 6.73, and opportunity was the larger absolute component in 59.7% of games. The precommitted 1-yard dominance margin was missed by 0.048 yards, so M96B must model workload and efficiency as separate distributions rather than declaring either side solved.**
- **M96B formalized the modular/puzzle approach. Simple additive stacking did not produce a broad winner: M94C remained the point anchor; the transplanted M95C environment residual slightly worsened point MAE; M95F workload-tail fusion improved 75+/100+ metrics only directionally and below the retention gate; the isolated M95D upside residual was destructive when added to M94C; and M95I vacancy information remained a promising diagnostic only. The key lesson is that positive modules may be redundant, conditional, or interactive rather than directly additive.**
- **M96C trained efficiency residuals directly against M94C using strict 2025 expanding-week OOF evaluation. No global E/P/D block cleared the frozen gate. Opponent-defense efficiency D was best globally (MAE `21.5719 -> 21.3474`) and improved 0-14 carry games, but materially worsened true 15+/20+ workload games. E/P showed the same sign flip. This supports a pregame conditional efficiency router rather than a universal correction. Isolated explosive X again failed as a separable tail increment. M96D is next.**

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
18. **Modular capability rule:** do not judge every experiment only as a whole-model replacement. Record the exact capability it improved, the regime where it improved it, what it damaged, and whether that capability can coexist with other validated modules. Before inventing a new model, test whether retained capabilities can be combined through precommitted ablations and non-degradation gates. A module may own a narrow job (center, tail, vacancy, efficiency, explosive upside) without being allowed to alter other jobs. Positive signals are not automatically additive; they may be redundant, interacting, or conditional experts. Never force a combination merely because each component was individually promising.

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

## M95Q — expanded exact historical stable-workhorse reconstruction

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


## Latest completed migration: M95R — expanded-panel dynamic population-prior candidate

Full results: `docs/migrations/M95R_RB_DYNAMIC_POPULATION_PRIOR_RESULTS.md`.

Authoritative:

- workflow `M95R RB Dynamic Population Prior`
- run **`33453141931`**
- job **`99687269044`**
- tested SHA **`272e7537f9642a977e84ffeeda4b9d7d17832c71`**
- artifact **`9780473734`**
- artifact SHA256 **`0d114c1f7cc1d2132c7ebd2c7074bd840b36d669c4a31c33e151cbb85be76821`**
- execution success
- disposition **`M95R_RETAIN_DIAGNOSTIC_DO_NOT_PROMOTE`**
- feature search `0`; hyperparameter search `0`; sportsbook `0`; production change `0`

M95R froze M95F as the 20+ backbone and allowed only a bounded additive log-odds population/workload residual using five predeclared pregame workload-regime variables. Ridge lambda was `10.0`; delta was capped at `+/-0.75`; evaluation was strict expanding-season: 2023 trained on 2020-22, 2024 on 2020-23, 2025 on 2020-24.

Primary W13-18 results:

- 2023: Brier `.233221 -> .221407`, logloss `.673392 -> .638868`, AUC `.727041 -> .702381`; calibration underprediction improved by `3.90 pp`.
- 2024: essentially neutral, Brier `.206384 -> .206530`, AUC `.704444 -> .702963`.
- 2025: **failed materially**. M95F was already almost centered (actual `28.24%`, predicted `29.43%`); R raised mean probability to `43.12%`, worsening Brier `.194926 -> .216791` and absolute calibration gap by `13.69 pp`.
- pooled: R moved mean probability closer to pooled actual but worsened Brier `.210576 -> .214863`, logloss `.610548 -> .618125`, and AUC `.664417 -> .642000`.
- 2025 full secondary: actual `21.94%`, M95F mean `29.60%`, R mean `43.48%`; Brier `.186593 -> .231516`.

Scientific lesson: matching population-average tail mass is not sufficient. R assigned too much mass to the wrong player-games in 2025. The workload regime changed faster than the multi-season residual calibrator could adapt. This exact R architecture is frozen as failed evidence and must not be retuned against exposed 2023-25 outcomes.

### Durable player-source cache

The raw nflverse weekly player source sheets are now permanently cached so future historical research does not repeatedly redownload the full player universe.

- workflow `M95R Historical Player Source Cache`
- successful run **`33453001235`**
- job **`99686833509`**
- cache workflow SHA **`07e0ceb5b6f0e8eda704c4735ba7b2ff2efa32ee`**
- persisted cache commit **`402c4a28593e0c79438ea10c88d832a98f0a6f2e`**
- path `data/research_cache/nflverse_player_weekly/`
- seasons `2018-2025`; one parquet snapshot per season plus SHA256 manifest; 150 source columns per season.

# Current scientific interpretation

The system already individualizes player-week football inputs. The strongest current RB interpretation has four layers:

1. **Global opportunity backbone** — M94C central carries remains the reference.
2. **Stable-workhorse tail baseline** — M95F remains safer than the failed derivative candidates.
3. **Role regime** — vacancy/transition remains structurally separate (M95I diagnostic).
4. **Population/game regime** — real and useful, but its calibration moves faster than fixed or slow cross-season gates/priors can safely follow.

M95K showed historical feed/ceiling can improve player ordering in favorable regimes, especially 2025, but failed sealed 2023. M95N-P established regime dependence. M95O showed fixed thresholds fail. M95R now shows a slow rolling residual population calibrator also fails: it repaired 2023 underprediction but became stale and injected excessive tail mass in 2025.

The key next question is therefore not simply whether dynamic context matters. It is **how to separate population-level tail mass from player-level ranking and how quickly the population anchor must adapt**.

# Latest completed migration: M95S — population-mass vs player-ranking decomposition

Full results: `docs/migrations/M95S_RB_MASS_RANKING_DECOMPOSITION_RESULTS.md`.

Authoritative:

- workflow `M95S RB Mass Ranking Decomposition`
- run **`33454116869`**
- job **`99690286310`**
- tested SHA **`78f08a5fdb8e5ff34143e1a6dc72d6d901daa2f2`**
- artifact **`9780815195`**
- artifact SHA256 **`aa1c29ea8f665699278a484d5d0e41fc768ab337c0e06b9eabd58e7500f4eb88`**
- execution success
- disposition **`M95S_DECOMPOSITION_SUPPORTED_ADVANCE_TO_CONSTRAINED_M95T`**
- model fit `0`; feature search `0`; coefficient search `0`; sportsbook `0`; production change `0`

M95S found that early 2025 M95F was already overpredicting realized stable-workhorse 20+ rate by an average `15.3812 pp` across Weeks 2-9 while M95R added another `13.4238 pp` of mass; contemporaneous league prior-four lead20 averaged only `13.2826%`. The stale R correction was therefore visibly contradicted by current pregame workload state.

Frozen player ranking remained conditional: 2023 frozen M95K/M95L AUC gain was `-0.181973`, while authoritative 2025 M95K gain was `+0.059979`, with probability mass already preserved. Team recent workload was useful for raw player outcomes but not as a positive residual mass correction, indicating substantial double-count risk if reused on top of M95F.

Scientific synthesis: population tail mass should react to current league/population state; player ranking should be separately bounded/conditional and mass-preserving.

# Latest completed migration: M95T — constrained dual-layer stable-workhorse tail candidate

Full results: `docs/migrations/M95T_RB_CONSTRAINED_DUAL_LAYER_TAIL_RESULTS.md`.

Authoritative:

- workflow `M95T RB Constrained Dual-Layer Tail`
- run **`33455690862`**
- job **`99695055863`**
- tested SHA **`540edb67d9d5451764e997f19213b80285c15fab`**
- artifact **`9781352939`**
- artifact SHA256 **`bd54485d18de4f4df1f7613d9587281234bf07f8ce9de6df36b68bca26167c70`**
- execution success
- disposition **`M95T_FAIL_STOP_NEW_RB_TAIL_CANDIDATES_RETAIN_M94C_M95F_PROCEED_M96`**
- model fit `0`; feature search `0`; coefficient search `0`; hyperparameter search `0`; sportsbook `0`; production change `0`

M95T combined a fast relative league workload-mass anchor with a separately bounded within-week feed/carry-ceiling reranker. The rerank was exactly mass-preserving before the population layer and M94C central carries remained untouched.

Comparable W13-18 stable-workhorse panel (`n=412`):

- pooled Brier `.208196 -> .207208` (gain `+.000988`)
- pooled logloss `.607393 -> .603946` (gain `+.003447`)
- pooled AUC `.659138 -> .661866` (gain `+.002728`)
- pooled absolute calibration gap `.054464 -> .039068`
- 2023 improved materially: Brier `.233221 -> .228334`, logloss `.673392 -> .657658`, AUC `.727041 -> .741497`
- 2024 improved modestly
- 2022 improved modestly
- 2020 and 2025 regressed slightly on Brier/logloss
- critically, 2025 mean p20 rose `29.43% -> 33.54%` against actual `28.24%`, worsening absolute calibration gap by `4.103 pp`

Frozen cross-season gates therefore failed: only 3/6 seasons had non-negative Brier gain (required >=4), the max absolute-calibration-gap regression exceeded `2.5 pp`, and the predeclared 2023/2025 trouble-year guard failed. The gate is not waived and the candidate is not retuned.

The rushing-yard translation sanity guard confirmed why workload matters but also why carry research is not the end product. Across the exact late-season panel, carries vs rushing yards correlated `.789267` Pearson / `.788957` Spearman. M95T did not materially destroy 75+/100+ rushing-yard discrimination (pooled AUC changes `-.006312` and `-.003702`), but this is not a rushing-yard point model.

**Stopping-rule decision:** new retrospective RB carry-tail candidate development ends at M95T. M94C remains the central carry/opportunity foundation and M95F remains the stable-workhorse tail baseline. M95I vacancy/transition remains separate diagnostic evidence. The next task is not another tail formula; it is explicit rushing-yard synthesis.

# M96A — opportunity vs efficiency attribution

Full results: `docs/migrations/M96A_RB_OPPORTUNITY_EFFICIENCY_ATTRIBUTION_RESULTS.md`.

Authoritative:

- workflow `M96A RB Opportunity-Efficiency Attribution`
- run **`33459376333`**
- job **`99706110345`**
- tested SHA **`9e3b152e7e756b8a29798ef82cfa9a0730c51f89`**
- artifact **`9782611047`**
- artifact SHA256 **`a2d4f99b9e7b3f2b75c8694e79c6856dd09e4b5ee9921b2d53488e89cdab3d6e`**
- execution success
- disposition **`JOINT_ADVANCE_M96B_SEPARATE_WORKLOAD_AND_EFFICIENCY_DISTRIBUTIONS`**
- model fit `0`; feature search `0`; coefficient search `0`; sportsbook `0`; production change `0`

M96A used complete M94C 2025 rushing-yard truth (`1,393/1,393`) and exact frozen M94C/M95F/M95I workload outputs. The first green run was not accepted because M95F carried only 1,340 nonmissing yard-truth rows; Run #2 fixed only the truth-source mechanics and retained the frozen candidate/routing contract.

All-RB M94C rushing-yard MAE was **`21.0312`**. Perfect actual carries with frozen pregame efficiency reduced MAE to **`13.3535`** (recovery `7.6777` yards). Perfect game efficiency with frozen projected carries reduced MAE to **`14.3055`** (recovery `6.7256`). Opportunity was the larger absolute residual component in **`59.73%`** of games, but the oracle recovery advantage was only **`0.9520`** yards versus the frozen `1.0`-yard dominance requirement. The route is therefore JOINT.

The regime split is highly informative: 0-5 and 20+/25+ carry games are primarily opportunity-sensitive; 11-19 carry games are primarily efficiency-sensitive. Actual 20+ games recover `11.37` MAE yards with perfect opportunity versus `3.60` with perfect efficiency. Stable-workhorse yardage is more efficiency-sensitive overall (`8.48` opportunity recovery vs `12.54` efficiency recovery). Carries are therefore not solved, but neither are they the only remaining RB bottleneck.

Collapsing M95F's carry distribution to one expectation worsened all-RB yard MAE (`21.0312 -> 22.4455`) while improving true 20+ carry games (`40.0051 -> 36.3540`) and 25+ games (`49.3105 -> 44.8607`). This confirms that the post-M94C tail work belongs in a **distribution layer**, not as a universal point-mean boost. The frozen M95I deterministic vacancy point branch did not beat M94C globally; M95I's previously validated vacancy probability/ranking evidence remains separate research signal.

# Latest completed migration: M96B — modular joint workload × efficiency synthesis

Full results: `docs/migrations/M96B_RB_MODULAR_JOINT_SYNTHESIS_RESULTS.md`.

Authoritative:

- workflow `M96B RB Modular Joint Synthesis`
- run **`33461369073`**
- job **`99711988023`**
- tested SHA **`a3018bf828bf0c78b09a2e0b8a6cd1af60b25f40`**
- artifact **`9783267179`**
- artifact SHA256 **`81f25e134a34a6b2d8b28195bb7f804f6ce54b5823bead5a2e4b717ee544718b`**
- execution success
- disposition **`M96B_MODULAR_SYNTHESIS_COMPLETE`**
- feature search `0`; weight search `0`; sportsbook `0`; production change `0`
- only model fitting in M96B was the precommitted one-dimensional Platt calibration for tail probabilities

M96B froze a capability ledger before the result:

- **C = M94C central opportunity/yard point.** Owns the point center; cannot be globally inflated for tails.
- **W = M95F workload-tail distribution.** Allowed to inform upper workload/tail probability, not replace C with a universally higher mean.
- **V = M95I vacancy/transition evidence.** Vacancy-only; not allowed on stable incumbents and not production-promoted here.
- **E = M95C mean efficiency/environment information.** Allowed to modify efficiency/yards only, never carries.
- **X = M95D explosive/upside context.** Tail/ranking role only; not allowed to universally boost point YPC/yards.

### Source integrity

- 2025 M95D OOS rows `1290`; exact M94C+M95D+M95F common rows `1274`; **98.7597%** coverage vs the frozen `>=97%` gate — PASS.
- 2024 W13-18 common M95D/M95F temporal-calibration rows `449` of `479` M95F holdout rows (`93.7370%`).
- shared rushing-yard truth parity passed.

### C — point anchor RETAIN

On the exact 2025 M96B intersection (`n=1274`):

- M94C/C MAE **`21.8440`**
- RMSE `30.5811`
- bias `+1.0954`
- correlation `.5853`

C remains the point anchor.

### E — additive M95C environment residual REJECT

Frozen test:

`CE = M94C rush-yard point + (M95C-environment prediction - role-baseline prediction)`

2025:

- C MAE `21.8440`
- C+E MAE `21.9095`
- gain `-0.0654`
- bias `+1.0954 -> +1.3486`
- worst ordinary-slice MAE regression only `0.4274`, but all-RB MAE failed to improve.

Important nuance: E/environment still improved its **native M95D role baseline** slightly in both 2024 and 2025. M96B therefore did not prove environment information useless; it proved that a residual learned around a weaker/different baseline is not plug-compatible as a direct additive correction to M94C. The next efficiency model must be trained directly against the M94C residual.

### W — M95F workload-tail fusion directional positive, but formal RETENTION GATE FAILED

Full-2025 75+ rushing yards:

- B AUC `.799739`
- B+W `.802295` — `+.002556`
- Brier `.112127 -> .111325` — gain `+.000803`
- logloss `.356596 -> .354201`

Full-2025 100+:

- B AUC `.799035`
- B+W `.799324` — `+.000288`
- Brier `.063428 -> .063343` — gain `+.000085`

All four full-season metrics moved in the right direction and late-2025 did not materially reverse, but the improvements did not meet the frozen materiality requirement (`+.005` AUC or `+.001` Brier). W is therefore **not retained as a rushing-yard tail fusion module from M96B**. This does not erase M95F's role as a carry/workload distribution baseline.

### X — isolated M95D upside residual REJECT in additive form

Frozen residual:

`X_delta = full_environment_matchup - M95C_environment`

When rank-fused onto B it was strongly destructive:

- 75+ AUC `.799739 -> .726578`, Brier `.112127 -> .120215`
- 100+ AUC `.799035 -> .720655`, Brier `.063428 -> .065507`
- B+W+X also regressed and was not preferred.

Important interpretation: the M95D full matchup model had shown better 100+ discrimination than its own environment-only control in its native architecture in both 2024 and 2025. M96B shows that this value is **interactive/native-expert signal, not a separable additive residual over M94C**. Modular does not mean every positive signal can be added as a coefficient.

### V — M95I vacancy signal RETAIN DIAGNOSTIC ONLY

2025 vacancy rows `n=105`:

- 75+ events `17`: frozen comparison AUC `.63035 -> .67213`, gain `+.04178`
- 100+ events `9`: `.73264 -> .75752`, gain `+.02488`

Critical caveat: the frozen parent comparison was the predeclared `B+W+X` arm, which itself failed globally. Therefore M96B **does not establish that V beats the best M94C-only yard-tail baseline**. V remains a promising vacancy-specific diagnostic that needs a direct-baseline, precommitted/prospective test before promotion.

### M96B scientific synthesis

The user's modular/puzzle framing is now a permanent research principle. M96B demonstrates four possible module relationships:

1. **Compatible but redundant / non-portable:** E contains real information in its native family but does not improve the stronger M94C point when transplanted.
2. **Helpful but too small:** W directionally improves the 2025 rush-yard tail but does not clear the materiality gate.
3. **Interactive, not additive:** X has native full-model tail signal but its isolated residual destroys M94C ranking.
4. **Conditional regime expert:** V remains promising specifically for vacancy/transition rows but is not yet proven versus the best baseline.

Do not conclude that M95 work was wasted, and do not force a combined stack merely because components once looked positive. The correct workflow is: capability ledger -> precommitted ablations -> incremental/non-degradation gates -> retain only compatible responsibility-specific modules.

**Current surviving global point architecture after M96B remains M94C/C.** No new global rushing-yard tail fusion earned retention in M96B. Generic carry-tail tuning remains closed.

# Latest completed migration: M96C — M94C-anchored RB efficiency residual synthesis

Full results: `docs/migrations/M96C_RB_M94C_EFFICIENCY_RESIDUAL_RESULTS.md`.

Authoritative:

- workflow `M96C RB M94C Efficiency Residual`
- run **`33462888850`**
- job **`99716610968`**
- tested SHA **`708f9ff23b96cde8e023b6317fcaec30b76e76b0`**
- artifact **`9783799265`**
- artifact SHA256 **`6109a8b3afc6d2fdb963db9149bf3fb238cc476e291bf743cc4b496ad39abf72`**
- execution success
- disposition **`M96C_NO_GLOBAL_WINNER_CONDITIONAL_EFFICIENCY_SIGNAL_SUPPORTED`**
- model fit `1`; feature search `0`; weight search `0`; hyperparameter search `0`; sportsbook `0`; production change `0`

Source/protocol:

- frozen M94C player-level rush-yard point exists in the authoritative artifact for 2025 only, so M96C did **not** invent a synthetic 2024 M94C player point;
- strict expanding-week 2025 OOF: test Weeks 6-18, each week trained only on earlier 2025 weeks;
- M94C rush attempts and central rush-yard point frozen;
- residual model predicted YPC/efficiency error only; correction multiplied by frozen M94C carries;
- train residual winsorization/clipping used training-only 5th/95th percentiles;
- M95D->M94C 2025 join `1340/1357 = 98.7472%`; exact yard and carry truth parity max diff `0.0`.

Frozen blocks:

- E blocking/environment `14` features;
- P player-created efficiency `8` available features;
- D opponent run efficiency/resistance `16` features;
- X explosive/upside `16` features, tail-only primary role.

Weeks 6-18 OOF all-RB (`n=961`):

- C/M94C MAE **`21.5719`**, RMSE `30.4500`, bias `+0.3820`, corr `.6045`.
- E MAE `21.5063` (gain `+0.0656`), but RMSE worsened `0.2509`.
- P MAE `21.4261` (gain `+0.1458`), RMSE worsened `0.1654`.
- D was best: MAE **`21.3474`** (gain **`+0.2245`**), RMSE `30.4341` (gain `+0.0159`).
- E+P MAE `21.5880`; E+D `21.5676`; P+D `21.4684`; E+P+D `21.6526`. Additive block stacking did not create a winner.
- No arm reached the frozen `>=0.25` all-RB MAE gain and all arms failed the workload non-degradation gate.

The key sign flip was D by actual workload (postgame diagnostic only):

- 0-5 carries: MAE `13.4145 -> 12.7837`, gain `+0.6308`.
- 6-10: `21.8220 -> 21.3148`, gain `+0.5071`.
- 11-14: `25.7482 -> 25.0106`, gain `+0.7376`.
- 15-19: `29.5957 -> 30.3544`, **regression `0.7587`**.
- 20+: `39.7267 -> 41.8936`, **regression `2.1669`**.
- 25+ diagnostic: regression `1.3432`.

E and P showed the same broad pattern: low/mid-workload value, higher-workload damage. This means the efficiency information is not useless; it is **conditional**. M96A already showed 20+/25+ yard misses are opportunity-dominant, while middle workload games are more efficiency-sensitive. M96C independently fits that architecture.

Do **not** use actual carries as the future router. Actual workload is postgame truth and is used only to diagnose the sign flip. Any router must use pregame M94C/M95F/role-state signals and be frozen before outcome scoring.

X tail-only audit failed:

- 75+ AUC `.806478 -> .800355`, Brier `.114757 -> .115719`.
- 100+ AUC `.790822 -> .785170`, Brier `.067047 -> .067255`.
- X is rejected as an isolated separable increment in this form; prior native-model interaction evidence is not erased.

Capability ledger after M96C:

- **C/M94C:** RETAIN global center.
- **E:** CONDITIONAL_CLUE only.
- **P:** CONDITIONAL_CLUE only.
- **D:** CONDITIONAL_CLUE only; strongest simple efficiency block.
- **X:** REJECT isolated tail increment.
- **M95F:** still workload-distribution evidence, not a universal yard mean boost.
- **M95I/V:** remains vacancy/transition diagnostic evidence and must be compared directly against the best baseline in a separately frozen conditional test.

Scientific interpretation: M96C did not find a safe universal efficiency correction, but it found exactly the kind of module specialization the puzzle framework is designed to exploit. The next step is not another global coefficient blend; it is a pregame router that decides when an efficiency expert should be active without sacrificing high-workload games.

# Latest completed migration: M96D — Pregame Conditional Efficiency Routing Audit

Full results: `docs/migrations/M96D_RB_PREGAME_CONDITIONAL_EFFICIENCY_ROUTING_RESULTS.md`.

Authoritative:

- workflow `M96D RB Pregame Efficiency Router`
- run **`33467325153`**
- job **`99729782983`**
- tested SHA **`dc57217aaa8312edc6c97c43486330ba9894bbc4`**
- artifact **`9785311314`**
- artifact SHA256 **`e65ef0cc861e8f27863e5d5fda8ba91b34d921d37724cf431212a9cb4026bf30`**
- execution success
- disposition **`M96D_PRIMARY_ROUTER_FAILED`**
- model fit `0`; threshold search `0`; feature search `0`; sportsbook `0`; production change `0`

M96D tested one frozen deterministic pregame router: turn M96C opponent-defense efficiency D on only below 15 M94C projected carries and when the back is not an entrenched workhorse. It improved all-RB Weeks 6-18 MAE `21.5719 -> 21.4165` (`+0.1554`), preserved RMSE/bias/tail AUC gates and improved late-season MAE, but failed the high-workload safety gate: actual 20+ MAE regressed `0.7489` and 25+ `0.5373` yards. No threshold was retuned.

The controlled role-only diagnostic was stronger globally (MAE `21.3641`, gain `+0.2078`; RMSE/correlation also improved) but still leaked damage into rare unexpected high-workload games. Pregame strata showed why: non-entrenched backs had only `3.16%` actual 20+ incidence overall, yet those rare spikes matter disproportionately. This supports exactly one final router type using already-frozen M95F workload-tail distribution and M95I transition/vacancy evidence as a safety guard around the role-based D expert. It does **not** reopen carry-tail tuning.

# Latest completed migration: M96E — Role Router with Frozen Workload-Risk Guard

Full results: `docs/migrations/M96E_RB_ROLE_WORKLOAD_RISK_GUARD_RESULTS.md`.

Authoritative:

- workflow `M96E RB Role Workload Risk Guard`
- run **`33467630395`**
- job **`99730679349`**
- tested SHA **`db1a139a270b7c246d1b5b07dc1a3490cb8fa3a0`**
- artifact **`9785416331`**
- artifact SHA256 **`c73a728570516b77c04c4a68ec1541e4a94fb830e144f40f16df63dbcc36dfbe`**
- execution success
- disposition **`M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP`**
- model fit `0`; threshold search `0`; feature search `0`; sportsbook `0`; production change `0`

M96E was the final precommitted retrospective efficiency-router test. It started from M96D's stronger non-entrenched role-based D router and suppressed D whenever frozen M95F/M95I pregame workload-risk evidence indicated meaningful 20+ workload or vacancy-transition risk. M94C carries and center were unchanged; M95F/M95I were not refit.

Weeks 6-18 all-RB (`n=961`): C MAE `21.571881`; M96E `21.430091`, gain `+0.141791`. RMSE improved `30.449965 -> 30.431137`; correlation improved `.604528 -> .605692`; late W13-18 MAE improved `+0.097105`. 75+ AUC changed only `-.000407`; 100+ improved `+.001508`.

Crucially, the safety guard worked: actual 20+ MAE regression fell to only `+0.059047` and 25+ to `+0.159106`, both inside the frozen <=`.50` gate. The guard protected 69/75 actual 20+ and 20/21 actual 25+ games in evaluation-only accounting.

However, the frozen all-RB materiality gate required MAE gain >=`.150000`; observed was `.141791`, short by `.008209`. Eight of nine checks passed, but the materiality line is **not waived**. M96E is not retained/promoted.

Final retrospective RB architecture:
- **C/M94C** remains the conservative global rushing-yard point and central opportunity anchor.
- **M95F** remains workload-distribution/stable-workhorse tail evidence, not a universal point-mean boost.
- **M95I** remains vacancy/transition diagnostic evidence.
- **D/M96C** is validated as conditional scientific signal but did not earn a retained point-module role after the final safety/materiality test.
- **E/P** remain conditional clues only.
- **X** remains rejected as an isolated separable tail increment.
- no M91-M96E RB research is production-promoted by this closure.


# Latest downstream benchmark: RB Market Benchmark — M94C vs 2025 archived DK/FD rushing-yard lines

Full results: `docs/migrations/RB_MARKET_BENCHMARK_RESULTS.md`.

Authoritative:

- workflow `RB Market Benchmark`
- run **`33499129109`**
- job **`99828098063`**
- tested SHA **`a26ad1a9991c2f9303d30e4f5b4cff25c3e9d30c`**
- artifact **`9796956965`**
- artifact SHA256 **`6759e7d8157ade3d4f9237e21a30feacb2507f77f03904ca85740683b7f96475`**
- execution success
- sportsbook inputs into football model `0`; football model change `0`; feature/weight/threshold search `0`

This was a downstream benchmark only. The source is the public Action Network-derived 2025 archive previously audited in M60B. Only exact full-game `rushing_yards` straight props from DraftKings/FanDuel were eligible. The archive does not preserve a trustworthy fixed pre-kick timestamp, so these rows are **archived latest / closing-like**, not a 30-minute-before-kickoff snapshot.

The first benchmark run `33498879907` was mechanically green but scientifically unusable because a broad source filter admitted combo/milestone markets and the archive's abbreviated names did not match the first full-name join. Run #2 repaired only exact-market filtering and identity mechanics; no model or benchmark metric logic changed.

Exact common market-covered RB player-games: **`899`**.

- M94C MAE **`25.515051`**, RMSE `34.364907`, bias `-0.579911`, corr `.453546`.
- Vegas DK/FD consensus MAE **`23.701891`**, RMSE `32.493543`, bias `-4.327030`, corr `.529751`.
- Vegas consensus therefore beat M94C by **`1.813160` MAE yards** on the exact common sample.
- Head-to-head: M94C closer `403`, market closer `496`; model closer rate `44.83%`.

The market edge is not uniform. When M94C and market were within 5 yards (`n=277`), M94C had a tiny MAE edge: `24.5481` vs `24.6390`. As disagreement widened, the market advantage grew: at `15+` yards disagreement (`n=211`), M94C MAE `31.9875` vs market `26.4716`, a **`5.5159`-yard market advantage**.

The most damaging regime is M94C materially **above** market. At M94C >=15 yards above consensus (`n=144`), M94C was closer only `36.11%`; its MAE was `33.0303` vs market `25.3924`. Large model-high disagreement is therefore a strong forensic sign of stale/overconfident workload/role state, but the market line itself must not become an upstream football feature.

Postgame actual-carry diagnosis shows Vegas did **not** solve the high-workload tail:

- actual 0-5 carries (`n=188`): M94C MAE `19.8329`, market `15.6596` — market better by `4.1733`.
- actual 6-10 (`n=232`): M94C `22.1260`, market `19.9655` — market better by `2.1605`.
- actual 20+ (`n=94`): M94C **`38.9831`**, market `39.8032` — M94C better by `0.8201`.
- actual 25+ (`n=23`): M94C **`51.0291`**, market `54.7826` — M94C better by `3.7535`.

For actual 25+ games both systems were drastically low: actual mean `123.09`, M94C mean `73.28`, market mean `69.52`. The extreme workload tail is still unsolved by both.

The benchmark identifies a **new, separately justified football-data research path** rather than permission to retune exposed M96 thresholds:

1. false-high workload suppression / pregame workload-collapse detection;
2. Week-1, rookie, new-team and new-role initialization;
3. current depth chart / transaction / practice-injury / availability timing available before kickoff;
4. coaching/backfield usage priors and potentially offensive-line availability;
5. rookie/draft/college workload priors where leakage-safe.

The benchmark is external evidence about *where* the model is missing football information. Sportsbook lines remain downstream only and may not be used as a feature, training target, ensemble input, or pregame router in the independent football model.

# Latest completed diagnostic: RB-ND1 — Forensic Failure Atlas

Full results: `docs/migrations/RB_ND1_FORENSIC_FAILURE_ATLAS_RESULTS.md`.

Authoritative:

- workflow `RB-ND1 Forensic Failure Atlas`
- run **`33503240202`**
- job **`99841197836`**
- tested SHA **`1b689322b48e7de52530bca5a9e2d7039a7a1a9b`**
- artifact **`9798563163`**
- artifact SHA256 **`d92ccd3fdc36edf523ed6da3af3c76e227e2b5a5ae8c9a49680b6661a53c7d7c`**
- execution success; fit/search `0`; sportsbook input to football model `0`; production change `0`
- disposition **`ADVANCE_RB_ND2_PREGAME_ROLE_STATE_PLAYER_SHARE_RECONSTRUCTION`**

RB-ND1 reverse-engineered all 1,393 frozen M94C 2025 RB/FB player-games with exact two-factor Shapley decompositions.

Carry decomposition (`team rush attempts × player team-rush share`):

- team-volume absolute contribution share **`39.87%`**
- **player-share/backfield-allocation share `60.13%`**

Rushing-yard decomposition (`carries × YPC`):

- opportunity **`51.51%`**
- efficiency **`48.49%`**

This confirms the overall opportunity/efficiency problem is joint, while identifying **player allocation/share as the larger opportunity subproblem**.

Largest compound error classes by share of total absolute rushing-yard error:

- `OPPORTUNITY__PLAYER_SHARE`: **`31.63%`**
- `EFFICIENCY__TEAM_VOLUME`: `18.06%`
- `EFFICIENCY__PLAYER_SHARE`: `17.54%`
- `OPPORTUNITY__TEAM_VOLUME`: `10.37%`

The three player-share-primary classes together account for roughly **55.5% of total absolute rushing-yard error**.

Special overlapping forensic flags:

- explosive shock: n `157`, MAE `47.31`, involved `25.35%` of total abs error;
- non-RB/QB rushing competition: n `329`, involved `24.75%`;
- game-script miss: n `325`, involved `23.60%`;
- role collapse: n `59`, MAE `32.59`;
- new-role initialization: n `8`, MAE `41.81`.

On the exact market-covered M96E evaluation window (W6-18, n `633`):

- M94C MAE **`25.595447`**
- M96E MAE **`25.527995`**
- archived DK/FD consensus MAE **`24.139810`**

So M96E does move M94C toward the market, but only `0.06745` yards/game on this listed-RB universe. It is too small to solve the dominant role/allocation gap.

### Critical new information/data finding

The authoritative M94C 2025 RB trace has **zero populated `role` rows and zero populated `rules_role` rows across all 1,393 RB/FB games**. Historical inputs use nflverse weekly rosters, but depth information is merged only when its source is explicitly week-tagged; unsafe date/full-season depth snapshots are intentionally skipped to prevent leakage.

Therefore the backtest has been asking historical usage/context to infer current backfield role **without a true historical pregame depth-role state**. This is a new missing-information family and strongly matches the false-high/false-low market diagnosis, Week-1 rookie/new-team failures, role-collapse cases and the 60.13% player-share carry decomposition.

Do not retrospectively use today's Ourlads depth chart. Any role reconstruction must prove pregame/timestamp integrity.

# NEXT MIGRATION — RB-ND2

Name: **RB-ND2 — Pregame Role-State / Player-Share Reconstruction**

Primary question:

> Can leakage-safe current-role information materially improve player share/backfield allocation while keeping M94C's team rush-volume layer fixed, thereby reducing ordinary/listed-RB carry and rushing-yard MAE without damaging 20+/25+ workload outcomes?

Required source families to audit/build before fitting:

1. lagged nflverse participation/snap involvement and any stable role fields;
2. lagged carries/touches, team RB share, red-zone/short-yardage usage and backfield concentration;
3. current-week weekly roster membership/status;
4. same-week historical injury/practice/game-status fields with explicit week/timing semantics and no future fill;
5. competing-RB availability/vacancy/backfield member count;
6. team-change/new-team and no-history/rookie indicators;
7. Week-1 prior-role construction from prior-season continuity plus new-player priors;
8. timestamp-safe historical depth chart only if source provenance proves pregame availability;
9. QB/non-RB rushing competition as a separate allocation input.

Initial ND2 architecture must **freeze M94C team rush attempts** and predict player share/allocation, not raw rushing yards. Evaluate resulting carries first, then translate through the frozen yard layer and benchmark actual outcomes. Sportsbook remains downstream only.

Precommitted goals must include materially improving all/listed-RB carry and rush-yard MAE, reducing false-high 0-10 carry cases, and preserving 20+/25+ workload performance. Do not reopen old M96 threshold tuning.

# RETROSPECTIVE FAMILY STOP BOUNDARY

The old `AUTONOMOUS_RB_RESEARCH_STOP` applies specifically to further exposed-sample M96C/D/E router/threshold retuning. It **does not** close RB research and it does not prohibit RB-ND2, which is a separately justified missing-information/data family discovered through the downstream market benchmark and RB-ND1 forensic audit.

A specific family should still be stopped when further variants would amount to answer-key fitting. When that happens, move only to a genuinely different evidence-backed football mechanism/data family.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first and verify the latest authoritative GitHub Actions run/artifact. Continue from `NEXT MIGRATION — RB-ND2`. The old autonomous stop applies only to exposed M96 router retuning, not the new role-state/player-share data family. Preserve all modeling/validation rules and keep sportsbook data downstream only.

# LATEST CONTINUITY UPDATE — RB ENV1 / ND2B / PRODUCTION-STACK AUDIT

**This section is newer than the earlier `NEXT MIGRATION — RB-ND2` section and overrides its sequencing where they conflict. A fresh chat must read this section before continuing.**

## Research philosophy refinement

New RB evidence should be treated as a capability/add-on to the strongest current football system unless it explicitly proves the existing component should be replaced. Do **not** default to replacement contests. Ask: what part of the current system does this information improve, where does it help, what does it damage, and can it be safely integrated with existing validated capabilities?

Probability distributions quantify uncertainty around the football projection; they do not replace football state knowledge. Establish role, availability, team opportunity, backfield allocation, runner ability, offensive/defensive environment and game context first; then use workload/efficiency/explosive distributions around those assumptions.

## RB-ENV1 — environment × runner-quality atlas

Full results: `docs/migrations/RB_ENV1_ENVIRONMENT_QUALITY_ATLAS_RESULTS.md`.

Authoritative:
- workflow `RB-ENV1 Environment Quality Atlas`
- run `33508752571`
- job `99859029342`
- tested SHA `375e6a776d91132563937fec1d02d85ea56ae69d`
- artifact `9800707982`
- artifact SHA256 `2373625d06e58019d761acdd925dff7f1458835718d66cb1c3f206a3da143df6`
- success; no fit/search/sportsbook/production change

Pooled 2024-2025 (`n=2580`): BAD pregame spot averaged `30.8866` rush yards versus GOOD `44.8932`, a `+14.0065` yard difference. 75+ rate moved `10.25% -> 21.05%`; 100+ `4.66% -> 9.91%`. The relationship replicated in both seasons. Linear correlation with raw yards was only about `.125` and with 8+ carry YPC only about `.052`, so environment is meaningful but not deterministic and should not override role/opportunity.

Strong RB + good spot pooled averaged `52.96` yards with a `25.90%` 75+ rate versus weak RB + bad spot `23.86` yards / `5.36%`. Bad-spot monster games were often explained by large workload and/or explosive variance; many good-spot failures were workload/allocation collapses. Durable order: opportunity/allocation first, then environment/ability, then tail variance.

## RB-ND2B — timestamp-safe role-source audit

Full results: `docs/migrations/RB_ND2B_BACKFIELD_SOURCE_ASOF_AUDIT_RESULTS.md`.

Authoritative:
- workflow `RB-ND2B Backfield Source AsOf Audit`
- run `33509092341`
- job `99860131990`
- tested SHA `cafb52854b4476eb27cd12d5064d5e6f52247b73`
- artifact `9800848065`
- artifact SHA256 `e8f5d3dd60da7464a5000485061087a3b2ef7bdcb66ec9254025e829c1f4be77`
- success; no model/search/sportsbook/production change

The 2025 historical depth source has `554,215` records and begins `2025-08-03`. Using the latest snapshot strictly before kickoff gives `100%` coverage across all `544` regular-season team-games, median snapshot age `10.77h`, p90 `17.90h`. Exact M94C RB/FB rows receive pregame depth-rank coverage `94.9749%`; Week 1 `96.4706%`. Prior-week offensive-snap coverage is `83.7156%` for Weeks 2-18. Thus the missing current-role layer is genuinely reconstructable without target-game leakage.

Depth `pos_rank` is not a deterministic one-player RB1/RB2/RB3 share rule; combine it with lagged snap/carry share, competitor state, availability, continuity, rookie/new-team context, etc.

## Critical production-vs-research stack distinction

Full audit: `docs/migrations/RB_PRODUCTION_VS_RESEARCH_STACK_AUDIT_2026_09_01.md`.

Canonical production `main` remains `7532a2c29dde78a5c3758eb1427561cfed801d67`. `.github/workflows/full-slate.yml` is the only production orchestrator.

The real production slate is richer than M94C. It builds current Ourlads roles, TeamForm/context, injuries, weather, Coverage v2, PlayerForm, Team Context v3, empirical Bayesian baselines, supervised ML v2, Markov State v2, empirical football rules, Monte Carlo, and an evidence-weighted MC/ML/State ensemble before downstream sportsbook comparison. The canonical rule layer includes game-script/play-volume mechanics, success/pressure context, box/coverage effects, injury handling and RB rushing-efficiency multipliers.

Historical `walk_forward.py` / `component_predictions.py` also constructs MC (with Bayesian + canonical rules), ML and State components at each historical cutoff.

**However M94/M94B/M94C intentionally isolated team rushing opportunity and used frozen M91 `ml_proj` as their player/team baseline.** M94 baseline team carries are the sum of `ml_proj`; M94C strength features aggregate `ml_proj`; after M94C changes team rushing volume, individual carries use the player's inherited ML share; rushing yards use baseline ML implied YPC. Therefore **M94C is not equivalent to the final production MC + ML + State + Bayesian/rules ensemble.**

This does not invalidate M94C's scientific findings; it means the project has accumulated production machinery and M95/M96 capabilities that have not yet been assembled into one production-equivalent RB historical architecture.

## LATEST NEXT MIGRATION — RB-STACK1

Name: **RB-STACK1 — Production-Equivalent RB Historical Baseline + Integration Audit**

Primary question:

> On a leakage-safe historical panel, what does the complete canonical football stack (Bayesian/rules + MC + ML + State + calibrated ensemble where valid) actually achieve for RB rush attempts and rushing yards, and which retained RB capabilities add incremental value to that full system?

Required sequencing:

1. Reconstruct/verify a production-equivalent historical RB baseline using the same canonical components and semantic contracts as `full-slate.yml`.
2. Audit historical coverage of production roles, injuries, weather, context and rule inputs; no silent placeholder behavior.
3. Score RB carries and rush yards overall and by role/workload/committee/Week-1 slices.
4. Compare full-stack baseline to M91 ML-only and M94C so we know what each piece already contributes.
5. Then test **add-ons**, not presumed replacements: timestamp-safe backfield allocation; M94C team opportunity where incremental; M95F workload distribution; M95I vacancy/transition; retained M95C/M96 efficiency/environment capabilities.
6. Use precommitted ablations/non-degradation gates. Preserve a strong parent component when a new module helps only one regime.
7. Sportsbook remains downstream benchmark only.

The explicit backfield-allocation engine remains a high-priority add-on, but do not judge it only against M94C. Establish the production-equivalent parent first, then test `full stack + allocation` and compatible module combinations.

## Fresh-chat startup override

Tell a new chat:

> Continue my NFL stuff project from `research-current-state` in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` completely, especially the bottom section `LATEST CONTINUITY UPDATE — RB ENV1 / ND2B / PRODUCTION-STACK AUDIT`, plus `docs/migrations/RB_PRODUCTION_VS_RESEARCH_STACK_AUDIT_2026_09_01.md`. The current next migration is `RB-STACK1 — Production-Equivalent RB Historical Baseline + Integration Audit`. Treat new RB findings as additive capability modules unless replacement is explicitly proven. Keep sportsbook data downstream only and preserve all temporal/leakage/integrity rules.
