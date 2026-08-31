# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from `NEXT MIGRATION`.

## Repository / current state

- Repo: `dkaps6/imtiredofthis`
- Current research branch: `research-rb-m95n-micro-regime-audit`
- Stable continuity ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No M91-M95N RB research has been promoted to production.
- Phase A production/data cleanup Waves 1-4 is complete.
- M94C remains the RB central carry/opportunity reference during tail research.
- M95F remains the safer stable-workhorse tail baseline after M95L.
- M95I vacancy/role-transition mechanism remains promising but not production-promoted.
- M95K was a strong 2025 research result but failed sealed temporal confirmation in M95L.
- M95N now supports conditional player-game micro-regime dependence for 20+ workload risk, but no new model has yet been fit or promoted.

## Non-negotiable modeling rules

1. Predict real football first.
2. Sportsbook/player-prop lines are downstream benchmark/decision inputs only; never feed them upstream into football projections.
3. No fake/synthetic sportsbook lines.
4. Do not waive production or source-integrity gates because one slice improves.
5. Do not manually boost tail coefficients after seeing validation.
6. Mechanical/source-contract fixes may not change scientific candidate grids or validation gates after exposure.
7. Preserve ordinary-game performance while fixing tails.
8. Distinguish mechanical/data/source failures from scientific/model failures.
9. 2025 was inspected repeatedly during M91-M95K and is not a pristine final confirmation set for hypotheses developed in that sequence.
10. 2023 W13-18 was opened in M95L and is no longer sealed for retuning M95K or pristine validation of a derivative candidate.
11. Failed experiments remain evidence. Do not erase or silently rewrite them.
12. Broad QB mean research stays frozen after M90 while RB work is active.
13. Do not build one model per player or hand-pick an expert for an individual game after seeing the matchup. Any conditional expert/gate must be defined from pregame variables and validated temporally.

## Sportsbook / 2026 operational state

- ordinary pushes use `FETCH_LIVE_ODDS=false`;
- explicit live-odds workflows only;
- sportsbook remains downstream;
- 2026 Week 1 live player-prop acceptance is not yet considered fully exercised because preseason no-credit rehearsals could not validate normal live prop inventory.

## QB state — frozen

Broad QB mean research is frozen after M90.

M90 headline:

- MAE ~`60.63 -> 56.56`
- RMSE `75.63 -> 69.63`
- corr `.173 -> .243`
- 100+ yard misses `81 -> 64`

Frozen PR #498 ensemble:

- MC `.208753`
- ML `.267121`
- State `.524126`

Future narrow QB residual-quality audit remains parked until RB work closes. A separate user hypothesis is also parked for later QB/WR work: explosive receiver/TE/RB matchup probability may explain some very large QB passing-yard overs, while uniformly poor receiver matchups may suppress pass volume/yardage. Do not claim this is validated.

## Research sequence

1. Production/data cleanup — complete.
2. RB refinement — current.
3. WR refinement — after RB closure.
4. Dedicated TE research.

# RB durable findings

## M91 / M92

2025 RB-only M91:

- carry MAE `3.494731`
- rush-yard MAE `21.018907`
- rush+rec MAE `25.352140`

M92 oracle decomposition proved opportunity architecture is the primary RB failure. Correct team volume, player share and carries unlock large gains, especially on 20+ workloads.

## M93 / M93B

Backfield concentration is real. Universal sharpening helps the extreme tail but damages middle slices. Role-aware concentration is useful but insufficient alone.

## M94 / M94B / M94C / M94D

M94C is the current central carry/opportunity reference.

2025 M94C:

- team rush MAE `5.812091`
- RB carry MAE `3.411003`
- 20+ carry MAE `7.876590`
- 25+ carry MAE `11.954550`

Legacy all-player rush-yard guard:

- baseline `7.758864`
- M94C `7.762069`
- gain `-0.003205`

Do not waive. M94D showed that sharpening lead-RB share of a carry pool that is itself too small cannot generate realistic 25+ absolute carries.

## M95A-D — offense / defense / quality / environment

- M95A: established workhorses perform materially better against weak pregame run defenses than strong ones across 2023-25; giant correlated defensive feature soup failed.
- M95B: validated RB + offense × opponent-defense architecture and recovered advanced rushing data including YBC, YAC, broken tackles, expected rushing yards, RYOE, 8+ box rate and time to LOS.
- M95C: blocking/environment signals are more stable for the mean; runner-created ability is more useful for upside/tails.
- M95D: FTN motion/RPO/formation/box charting, nflverse participation/personnel and historical defensive missed tackles did not improve mean projection but did improve 100+ rushing-yard discrimination in both forward seasons. Retain as upside context.

## M95E / M95F — workload state

M95E showed workload-state ranking is strong even when deterministic carry mean remains compressed.

M95F 2025:

- 20+ AUC ~`.846`
- 25+ AUC ~`.844`
- 25+ actual base `1.72%`
- raw score mean `22.51%`
- calibrated mean `3.06%`

Mixture distributions improved true high-workload games but damaged ordinary games. Stable workhorses remained overconfident.

## M95G / M95H — role / entitlement

M95G proved roster/injury/depth/role information improves 20+ workload discrimination and established: **a vacancy is not a successor**.

M95H validated only the >=70% RB-share entitlement target:

- AUC `.903118 -> .919599`
- Brier `.096200 -> .090868`

Vacancy ranking improved, but probability was too high.

## M95I — vacancy regime

Authoritative run `33402566592`, job `99522191259`, artifact `9761827238`.

Vacancy 25+:

- AUC `.721739 -> .939130`
- Brier `.008840 -> .008445`
- logloss `.048953 -> .040330`

Disposition: diagnostic only, do not promote. This established a major regime split between vacancy/transition and stable incumbent workhorses.

## M95J — generic stable-week conversion failed

Authoritative run `33405821436`, job `99533036053`, artifact `9763096005`.

Generic week-specific script/matchup/competition variables selected on 2024 failed to generalize to 2025 stable workhorses. This motivated persistent player/team feed tendency and workload ceiling.

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

Strong 2025 individual signals included player current-season p95 carry ceiling (`25+ AUC .7170`) and p90 (`.6840`). Because 2025 had been repeatedly inspected, K correctly advanced to sealed confirmation instead of promotion.

## M95L — sealed temporal confirmation FAILED

Primary confirmation: 2023 W13-18, training cutoff W12.

Mechanical source issue was repaired without weakening the frozen 97% gate: verified GSIS-backed aliases moved M94C player join from `95.585%` to **100.000%**. A later duplicate calibrated-tail merge was also fixed mechanically before sealed metrics were exposed.

Authoritative:

- run `33429747106`
- job `99611940386`
- SHA `caa9401eb50f6980e2a2c35ddd8e54467f57cbef`
- artifact `9772316395`
- disposition `M95K_SEALED_TEMPORAL_CONFIRMATION_FAILED`

Stable workhorses (`n=73`, 24 20+ events, 10 25+ events):

20+:

- M95F AUC `.727041`, Brier `.233221`
- frozen K/L AUC `.545068`, Brier `.244446`
- AUC gain `-0.181973`

25+:

- M95F AUC `.533333`, Brier `.123614`
- frozen K/L AUC `.442857`, Brier `.126356`
- AUC gain `-0.090476`

Stable probability mass and M94C central carries were preserved exactly. Vacancy 25+ had zero positive events and was inconclusive-small-N.

## M95M — cross-season failure postmortem

Authoritative:

- run `33433593731`
- job `99624596080`
- SHA `52b537dcd52561a1545a8c87b381c1ea5fca63da`
- artifact `9773558859`
- no model fit/search/change

Key findings:

- 2025 W13-18 20+ M95K AUC gain was **`+0.086066`**, while 2023 W13-18 was **`-0.181973`**. The failure is genuinely cross-season, not merely a late-season-window artifact.
- 25+ was unstable even in 2025 W13-18 (`-0.084388` AUC gain) and remains especially unreliable.
- player current-season p95 25+ AUC moved from `.717015` in 2025 full to `.481746` in 2023 W13-18; p90 similarly fell from `.684031` to `.486508`.
- not all historical workload information died in 2023: `feed25_rate` and some team lead-RB ceiling measures retained signal.
- sample depth did not explain the failure; high-history groups were not rescued.

Interpretation: the exact global M95K mapping is the failure, not the broad idea that historical workload information can matter.

## Latest completed migration: M95N — conditional player-game micro-regime audit

Full results: `docs/migrations/M95N_RB_MICRO_REGIME_AUDIT_RESULTS.md`.

Authoritative:

- workflow `M95N RB Micro-Regime Audit`
- run **`33435092627`**
- job **`99629424342`**
- tested SHA **`13f86f95e548a4675d2030340b7a9e2caf6e5172`**
- artifact **`9774088423`**
- artifact SHA256 **`b5044b3b55f0a2ec2fc9090e2f0e9580c77ca1b097dcb4c715581b12b0aa74b4`**
- execution success
- feature search `0`; coefficient search `0`; new model fit `0`; sportsbook `0`; production change `0`

M95N compared the frozen M95F current-context rank against a fixed pregame historical feed/ceiling score and assigned four stable-workhorse micro-regimes without using outcomes:

- `aligned_high`: current context high + history high
- `context_only`: current context high + history low
- `history_only`: current context low + history high
- `aligned_low`: both low

### 20+ result

2023 W13-18:

- aligned-high: `12/21 = 57.14%`
- aligned-low: `3/18 = 16.67%`
- context-only: `7/16 = 43.75%`
- history-only: `2/18 = 11.11%`

2025 W13-18:

- aligned-high: `13/30 = 43.33%`
- aligned-low: `4/30 = 13.33%`
- context-only: `2/13 = 15.38%`
- history-only: `5/12 = 41.67%`

Therefore:

- aligned-high > aligned-low was stable in both seasons;
- the preferred side of disagreement **flipped by season**;
- `micro_regime_dependence_supported = 1`;
- interpretation: **`agreement_is_stable_signal_disagreement_requires_conditional_response`**.

This explains an important part of the M95L failure. In 2023, M95K reduced average 20+ probability in the `context_only` group by `-0.03187` despite a `43.75%` actual rate, while increasing the `history_only` group by `+0.03087` despite an `11.11%` actual rate. In 2025 W13-18 the history-preferring direction happened to be appropriate instead.

M95N therefore supports the user's micro-level game-environment concern in a disciplined form: individual player-game inputs already exist, but a single stable-workhorse response function is too coarse when evidence channels disagree.

Secondary precommitted splits by projected volume, backfield concentration, matchup weakness and role momentum did **not** reveal one clean sufficiently sampled universal rule for resolving disagreement. Do not hand-pick experts from opened 2023 labels.

25+ remains too sparse/unstable for a new specialized architecture from this audit alone; only 16 total 25+ events existed across the two same-window populations.

# Current scientific interpretation

The RB system is already individualized at the input level: player history, offense, opponent defense, expected game state, injuries, depth, role, competition, QB rush siphoning and team tendencies vary by player-week.

The emerging issue is the response function. M95I already proved vacancy and stable-incumbent situations need different mechanisms. M95M/M95N now show that even stable workhorses are heterogeneous: when current context and historical feed agree, the high-vs-low ordering is repeatable across seasons; when they disagree, the correct weighting changes by season/context.

The correct direction is therefore **not** one model per player. It is a global backbone plus precommitted conditional gating / mixture-of-experts behavior, with temporal validation and strong shrinkage toward the global model when evidence is weak.

# NEXT MIGRATION — M95O

Name: **M95O — Agreement-Gated Stable-Workhorse 20+ Tail Candidate**

Primary question:

> Can a conservative precommitted gate preserve M95F in disagreement cases while using historical feed information only when it agrees with the current pregame context, thereby avoiding M95K's cross-season reversal?

Required design:

- focus first on **20+ carries**;
- M95F remains the stable-workhorse backbone;
- M94C central carries unchanged;
- historical feed/ceiling information may modify stable 20+ only under a predeclared agreement condition;
- disagreement cases default to or strongly shrink toward M95F rather than force the M95K rerank;
- no player-specific exceptions;
- no sportsbook inputs;
- do not use opened 2023 W13-18 as pristine confirmation;
- separate development/model-selection data from a new genuinely untouched temporal confirmation protocol;
- keep 25+ diagnostic/frozen until a stronger rare-event architecture is justified;
- no production promotion unless temporal confirmation and existing production guards pass.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from the `NEXT MIGRATION` section. Do not reconstruct the project from unrelated memories or restart old research. Preserve all modeling and validation rules in the handoff file.
