# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from `NEXT MIGRATION`.

## Repository / current state

- Repo: `dkaps6/imtiredofthis`
- Current research branch: `research-rb-m95o-agreement-gated-tail`
- Stable continuity ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No M91-M95O RB research has been promoted to production.
- Phase A production/data cleanup Waves 1-4 is complete.
- M94C remains the RB central carry/opportunity reference during tail research.
- M95F remains the safer stable-workhorse tail baseline after M95L/M95O.
- M95I vacancy/role-transition mechanism remains promising but not production-promoted.
- M95K was a strong 2024/2025 research result but failed sealed temporal confirmation in M95L.
- M95N established meaningful micro-regime dependence for 20+ workload risk.
- M95O tested a conservative fixed agreement gate and **did not pass**; its main new finding is that stable-workhorse population priors/calibration themselves shift materially across seasons/windows.

## Non-negotiable modeling rules

1. Predict real football first.
2. Sportsbook/player-prop lines are downstream benchmark/decision inputs only; never feed them upstream into football projections.
3. No fake/synthetic sportsbook lines.
4. Do not waive production or source-integrity gates because one slice improves.
5. Do not manually boost tail coefficients after seeing validation.
6. Mechanical/source-contract fixes may not change scientific candidate grids or validation gates after exposure.
7. Preserve ordinary-game performance while fixing tails.
8. Distinguish mechanical/data/source failures from scientific/model failures.
9. 2025 was inspected repeatedly during M91-M95K and is not a pristine final confirmation set.
10. 2023 W13-18 was opened in M95L and is no longer pristine for retuning or confirmation.
11. 2024 was used for M95K development/selection and is not an independent confirmation year for derivative candidates.
12. Failed experiments remain evidence. Do not erase or silently rewrite them.
13. Broad QB mean research stays frozen after M90 while RB work is active.
14. Do not build one model per player or hand-pick an expert for an individual game after seeing the matchup. Any conditional expert/gate must be defined from pregame variables and validated temporally.
15. Any derivative candidate after M95L-N-O requires a genuinely new untouched/prospective confirmation protocol before promotion.

## Sportsbook / 2026 operational state

- ordinary pushes use `FETCH_LIVE_ODDS=false`;
- explicit live-odds workflows only;
- sportsbook remains downstream;
- 2026 Week 1 live player-prop acceptance is not yet considered fully exercised because preseason no-credit rehearsals could not validate normal live prop inventory.

## QB state — frozen after M90

M90 headline:

- MAE ~`60.63 -> 56.56`
- RMSE `75.63 -> 69.63`
- corr `.173 -> .243`
- 100+ yard misses `81 -> 64`

Frozen PR #498 ensemble:

- MC `.208753`
- ML `.267121`
- State `.524126`

Future narrow QB residual-quality audit remains parked until RB work closes. A separate user hypothesis is parked for later QB/WR work: explosive receiver/TE/RB matchup probability may explain some very large QB passing-yard overs, while uniformly poor receiver matchups may suppress pass volume/yardage. Not yet validated.

## Research sequence

1. Production/data cleanup — complete.
2. RB refinement — current.
3. WR refinement — after RB closure.
4. Dedicated TE research.

# RB durable findings

## M91-M94C

- M91 2025 RB-only carry MAE `3.494731`, rush-yard MAE `21.018907`, rush+rec MAE `25.352140`.
- M92 oracle decomposition showed opportunity architecture is the primary RB failure.
- M93/M93B: backfield concentration is real; universal sharpening helps extreme tails but harms the middle.
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

Do not waive. M94D showed that sharpening lead-RB share of a carry pool that is itself too small cannot create realistic 25+ absolute carries.

## M95A-D — matchup / quality / environment

- M95A: established workhorses perform materially better against weak pregame run defenses than strong ones across 2023-25; giant correlated defensive soup failed.
- M95B: validated RB + offense × opponent-defense architecture and recovered advanced rushing data (YBC, YAC, broken tackles, expected rushing yards, RYOE, 8+ box, time to LOS).
- M95C: blocking/environment signals are more stable for the mean; runner-created ability is more useful for upside/tails.
- M95D: motion/RPO/formation/box, participation/personnel and missed-tackle data did not improve mean but improved 100+ rushing-yard discrimination in forward seasons. Retain as upside context.

## M95E/F — workload-state baseline

M95F 2025 calibrated tail scores:

- 20+ AUC ~`.846`
- 25+ AUC ~`.844`
- 25+ actual base `1.72%`
- raw score mean `22.51%`
- calibrated mean `3.06%`

Mixture distributions helped true high-workload games but harmed ordinary games. Stable workhorses remained overconfident.

## M95G/H/I — role and vacancy regime

M95G established **a vacancy is not a successor**. M95H validated only the >=70% RB-share entitlement target (AUC `.903118 -> .919599`, Brier `.096200 -> .090868`).

M95I authoritative run `33402566592`, job `99522191259`, artifact `9761827238`.

Vacancy 25+:

- AUC `.721739 -> .939130`
- Brier `.008840 -> .008445`
- logloss `.048953 -> .040330`

M95I remains diagnostic/not promoted.

## M95J — generic stable-week conversion failed

Run `33405821436`, job `99533036053`, artifact `9763096005`.

Generic week-specific script/matchup/competition variables selected on 2024 failed in 2025 stable workhorses, motivating persistent player/team feed tendency and workload ceiling.

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

Important 2024 role in M95K: fit W13-15, select W16-18. On 2024 W16-18 stable workhorses (`n=34`):

- 20+ M95F AUC `.619048` -> M95K `.725275`
- Brier `.247201 -> .225884`
- 25+ AUC `.553571 -> .595238`

2024 therefore supported the architecture, but because it was used for selection it is not independent confirmation.

2025 stable workhorses (`n=237`):

20+:

- M95F AUC `.581185`, Brier `.186593`
- M95K AUC `.641164`, Brier `.171528`
- AUC gain `+0.059979`
- mean probability exactly preserved at `29.60%`

25+:

- M95F AUC `.591714`, Brier `.053017`
- M95K AUC `.612631`, Brier `.051386`
- AUC gain `+0.020917`
- mean exactly preserved at `11.10%`

Strong 2025 player-current-season p95/p90 25+ AUCs were `.7170` / `.6840`.

## M95L — sealed temporal confirmation FAILED

Authoritative:

- run `33429747106`
- job `99611940386`
- SHA `caa9401eb50f6980e2a2c35ddd8e54467f57cbef`
- artifact `9772316395`
- disposition `M95K_SEALED_TEMPORAL_CONFIRMATION_FAILED`

Mechanical source repair moved player join from `95.585%` to **100.000%** via verified GSIS-backed aliases without lowering the frozen `97%` gate. A separate duplicate-tail merge was fixed before sealed metrics were exposed.

2023 W13-18 stable workhorses (`n=73`, 24 20+ events, 10 25+):

20+:
- M95F AUC `.727041`, Brier `.233221`
- frozen K/L AUC `.545068`, Brier `.244446`

25+:
- M95F AUC `.533333`, Brier `.123614`
- frozen K/L AUC `.442857`, Brier `.126356`

Stable probability mass and M94C central carries were preserved. Vacancy 25+ was inconclusive-small-N.

## M95M — cross-season postmortem

Authoritative run `33433593731`, job `99624596080`, SHA `52b537dcd52561a1545a8c87b381c1ea5fca63da`, artifact `9773558859`.

Key results:

- 2025 W13-18 M95K 20+ AUC gain `+0.086066` versus 2023 W13-18 `-0.181973` => genuine cross-season nonstationarity, not merely late-season window effect.
- 25+ unstable even within 2025 late (`-0.084388`) and 2023 (`-0.090476`).
- player current-season p95 25+ AUC: 2025 full `.717015`, 2025 late `.622363`, 2023 `.481746`.
- some broader feed/team ceiling signals remained useful in 2023.
- sample depth did not rescue K.

## M95N — conditional player-game micro-regime audit

Authoritative:

- run `33435092627`
- job `99629424342`
- SHA `13f86f95e548a4675d2030340b7a9e2caf6e5172`
- artifact `9774088423`
- no new model fit/search/change

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

Formal findings:

- aligned-high > aligned-low stable across both seasons;
- preferred side of disagreement flipped by season;
- `micro_regime_dependence_supported = 1`;
- interpretation `agreement_is_stable_signal_disagreement_requires_conditional_response`.

Secondary splits by volume, concentration, matchup and role momentum did not yield a clean sufficiently sampled resolver. 25+ remained too sparse.

## Latest completed migration: M95O — agreement-gated stable-workhorse 20+ candidate

Full results: `docs/migrations/M95O_RB_AGREEMENT_GATED_TAIL_RESULTS.md`.

Authoritative:

- workflow `M95O RB Agreement-Gated Tail Candidate`
- run **`33437157679`**
- job **`99636245739`**
- tested SHA **`d194a69a4d8939067b1c7d495de438c3062822eb`**
- artifact **`9774831420`**
- artifact SHA256 **`00bd31e6cb821edefc900a93d12b7e79a0f72056d8da49fa2b250138496e1cef`**
- execution success
- disposition **`RETAIN_M95O_AS_DIAGNOSTIC_DO_NOT_PROMOTE`**
- feature/coefficient search `0`; sportsbook `0`; production change `0`; M94C central changes `0`; stable 25+ changes `0`

Run #1 failed mechanically on a nested M95L artifact path. The workflow was changed only to recursive/path-agnostic artifact discovery; no science changed. Run #2 above is authoritative.

M95O precommitted a fixed agreement gate using 2024 W13-15 pregame distributions. Discordant stable workhorse games remained exact M95F; aligned rows used frozen M95K ranking, mean-anchored to M95F mass. Stable 25+ stayed M95F.

20+ metrics:

| Scope | M95F AUC | M95K AUC | M95O AUC | M95F Brier | M95O Brier |
|---|---:|---:|---:|---:|---:|
| 2024 W16-18 dev/selection | `.619048` | `.725275` | `.659341` | `.247201` | `.242236` |
| 2025 full research | `.581185` | `.641164` | `.589605` | `.186593` | `.184791` |
| 2025 W13-18 | `.646858` | `.732923` | `.635246` | `.194926` | `.197812` |
| 2023 W13-18 opened | `.727041` | `.545068` | `.568027` | `.233221` | `.243097` |

M95O therefore did not solve the instability. It partially reduced M95K damage in 2023, but remained far below M95F and became slightly worse than M95F in 2025 W13-18.

### Major new M95O finding — population prior / calibration shifts

Stable-workhorse workload prevalence differs substantially:

- 2024 W16-18: 20+ `38.24%`, 25+ `17.65%`, mean M95F p20 `26.43%`
- 2025 full: 20+ `21.94%`, 25+ `4.64%`, mean M95F p20 `29.60%`
- 2025 W13-18: 20+ `28.24%`, 25+ `7.06%`, mean M95F p20 `29.43%`
- 2023 W13-18: 20+ `32.88%`, 25+ `13.70%`, mean M95F p20 `16.19%`

2023 is therefore not simply the only "weird high-workload" year; the selected 2024 late window is even higher. More important, the frozen M95F probability level changes from strongly under-bullish in 2023/2024 to roughly calibrated in 2025 late and over-bullish in 2025 full.

The fixed 2024 gate classified 2023 very differently: only one 2023 row was `aligned_high` and 58 were `aligned_low`, yet the aligned-low group still had a 31.03% actual 20+ rate. This indicates a cross-season distribution-level shift in the meaning of absolute context/feed scores.

Formal M95O retrospective gates all failed. Do not retroactively change the gate/mass definition after seeing results.

# Current scientific interpretation

The system already individualizes player-game inputs. The current challenge is deeper than choosing the right expert when history and context disagree: the **population prior itself changes**. A player's context/feed score cannot safely be interpreted against one fixed historical scale if the league/team workload environment has shifted.

M95K is not promoted. M95O is not promoted. M95F remains the safer stable-workhorse 20+/25+ baseline; M94C remains central carries; M95I vacancy stays a separate promising diagnostic lineage.

The next question is whether the changing workload regime can itself be detected using only pregame information.

# NEXT MIGRATION — M95P

Name: **M95P — Dynamic Workload-Regime / Population-Prior Audit**

Primary question:

> Using only information available before each game, can we identify the current league/team stable-workhorse workload regime so that player history and current game context are normalized relative to the appropriate contemporaneous population prior rather than one fixed cross-season distribution?

Audit before fitting another candidate:

- prior-week / season-to-date league stable-workhorse 20+/25+ prevalence;
- prior-week/team lead-RB workload distribution;
- rolling team rush attempts and lead-RB concentration;
- QB rush siphoning, RB-count/committee environment;
- league/team offensive play volume and rush tendency;
- temporal trend / early-vs-late-season effects;
- whether leakage-safe rolling normalization makes 2023/2024/2025 player-game archetypes comparable;
- whether the pregame regime state explains the large M95F calibration/base-rate shifts.

Design constraints:

- audit first; no promoted model in M95P;
- no current/future-week outcomes in regime assignment;
- no sportsbook inputs;
- no player exceptions;
- M94C central carries unchanged;
- do not use 2023/2024/2025 as if any of them were pristine confirmation for a derivative candidate;
- any eventual M95Q-style candidate must be precommitted and receive a new untouched/prospective confirmation protocol.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from the `NEXT MIGRATION` section. Do not reconstruct the project from unrelated memories or restart old research. Preserve all modeling and validation rules in the handoff file.
