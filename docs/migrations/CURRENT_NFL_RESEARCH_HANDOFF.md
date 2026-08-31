# CURRENT NFL RESEARCH HANDOFF

This is the authoritative continuity checkpoint for the `NFL stuff` project. A fresh ChatGPT conversation should read this file first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from `NEXT MIGRATION`.

## Repository / current state

- Repo: `dkaps6/imtiredofthis`
- Current research branch: `research-rb-m95m-cross-season-tail-postmortem`
- Stable continuity ref: `research-current-state`
- Last known production `main` SHA: `7532a2c29dde78a5c3758eb1427561cfed801d67`
- No M91-M95M RB research has been promoted to production.
- Phase A production/data cleanup Waves 1-4 is complete.
- M94C remains the RB central carry/opportunity reference during tail research.
- M95F remains the safer stable-workhorse tail baseline after M95L.
- M95I vacancy/role-transition mechanism remains promising but not production-promoted.
- M95K was a strong 2025 research result but **failed sealed temporal confirmation in M95L**.

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
10. 2023 W13-18 was opened in M95L and is **no longer sealed** for retuning M95K or validating a derivative candidate.
11. Failed experiments remain evidence. Do not erase or silently rewrite them.
12. Broad QB mean research stays frozen after M90 while RB work is active.

## Sportsbook / 2026 operational state

- ordinary pushes use `FETCH_LIVE_ODDS=false`;
- explicit live-odds workflows only;
- sportsbook remains downstream;
- 2026 Week 1 live player-prop acceptance is not yet considered fully exercised because preseason no-credit rehearsals could not validate normal live prop inventory.

## QB state — frozen, future audit flagged

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

Future narrow QB residual-quality audit is flagged, not active. After RB work, test explicit intrinsic QB quality after controlling for frozen M90: CPOE/accuracy over expectation, EPA/dropback, pressure-adjusted efficiency, sack avoidance, deep-ball/air-yard creation, YAC dependence, turnover tendency, red-zone efficiency, and separation of QB-created ability from protection/receivers/play calling/opponent environment.

A separate user hypothesis is also parked for later QB/WR work: explosive receiver/TE/RB matchup probability may explain some very large QB passing-yard overs, while uniformly poor receiver matchups may suppress pass volume/yardage. Do not claim this is validated yet.

## Research sequence

1. Production/data cleanup — complete.
2. RB refinement — current.
3. WR refinement — after RB closure.
4. Dedicated TE research.

# RB research — durable findings

## M91 / M92

2025 RB-only M91:

- carry MAE `3.494731`
- rush-yard MAE `21.018907`
- rush+rec MAE `25.352140`

M92 oracle decomposition proved opportunity architecture is the primary RB failure. Correct team volume, player share and carries unlock large gains, especially on 20+ workloads.

Never confuse the legacy all-player rushing scoreboard (~7.76-8 yards) with RB-only rush-yard MAE.

## M93 / M93B

Backfield concentration is real. Universal sharpening helps the extreme tail but damages middle slices. Role-aware concentration is useful but insufficient alone.

## M94 / M94B / M94C / M94D

Explicit football game-state decomposition improved team rushing opportunity. M94C is the current central carry/opportunity reference during tail research.

2025 M94C:

- team rush MAE `5.812091`
- RB carry MAE `3.411003`
- 20+ carry MAE `7.876590`
- 25+ carry MAE `11.954550`

Legacy all-player rush-yard guard:

- baseline `7.758864`
- M94C `7.762069`
- gain `-0.003205`

Tiny failure; do not waive.

M94D confirmed that sharpening lead-RB share of a carry pool that is itself too small cannot generate realistic 25+ absolute carries.

## M95A-D — offense / defense / quality / environment

M95A validated the core football hypothesis that established workhorses perform materially better against weak pregame run defenses than strong ones across 2023-25. A giant correlated defensive feature soup failed, so matchup data must remain compact/interpretable.

M95B validated RB + offense × opponent-defense architecture and recovered advanced rushing data including YBC, YAC, broken tackles, expected rushing yards, RYOE, 8+ box rate and time to LOS.

M95C found blocking/environment signals more stable for the mean while runner-created ability is more useful for upside/tails.

M95D recovered FTN motion/RPO/formation/box charting, nflverse participation/personnel and historical defensive missed tackles. Mean projection did not improve, but 100+ rushing-yard discrimination improved in both forward seasons. Retain as upside context.

## M95E / M95F — workload regime

M95E showed workload-state ranking is strong even when deterministic carry mean remains compressed.

M95F calibrated the class-balanced tail scores. 2025:

- 20+ AUC ~`.846`
- 25+ AUC ~`.844`
- 25+ actual base `1.72%`
- raw score mean `22.51%`
- calibrated mean `3.06%`

Mixture distributions improved true high-workload games but damaged ordinary games. Stable workhorses remained overconfident.

## M95G / M95H — current role and entitlement

M95G proved roster/injury/depth/role information improves 20+ workload discrimination but established that **a vacancy is not a successor**.

M95H tested exact lead identity, >=60% share and >=70% share. Exact lead and >=60% did not validate. >=70% share did:

- AUC `.903118 -> .919599`
- Brier `.096200 -> .090868`

Vacancy >=70% ranking jumped `.784 -> .865`, but absolute probability was too high.

## M95I — deep-concentration + tail integration

Authoritative:

- run `33402566592`
- job `99522191259`
- artifact `9761827238`
- disposition `RETAIN_M95I_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

Vacancy 25+:

- AUC `.721739 -> .939130`
- Brier `.008840 -> .008445`
- logloss `.048953 -> .040330`

M95I materially improved vacancy calibration and identified a major regime split. Stable workhorses remained too bullish. Their unresolved problem is not ownership; it is which specific week converts known lead status into extreme workload.

## M95J — generic stable-week conversion failed

Authoritative:

- run `33405821436`
- job `99533036053`
- artifact `9763096005`
- disposition `RETAIN_M95J_AS_DIAGNOSTIC_DO_NOT_PROMOTE`

Generic week-specific script/matchup/competition variables looked strong on 2024 selection but failed in 2025 stable workhorses. This suggested player/team/coaching feed tendency and workload ceiling differ persistently across workhorses.

25+ frequency in 2025:

- 24 / 1,393 RB player-games = `1.72%`
- 24 / 544 team-games = `4.41%`
- 24 / 272 NFL games = `8.82%`
- roughly one 25+ RB workload every `11.33` NFL games

## M95K — feed tendency / carry ceiling

Authoritative:

- run `33411719023`
- job `99552629521`
- SHA `daa39544bd895084223532073b5cb9aa2eb4e872`
- artifact ID `9765397828`
- disposition `ADVANCE_M95K_TAIL_ARCHITECTURE_TO_SEALED_CONFIRMATION`
- production change `0`
- sportsbook inputs `0`

Frozen architecture selected with 2024 only:

- shrink `k=4`
- spec `feed_compact_env`
- logistic `C=.03`
- mass-preserving stable 20+ rerank
- frozen conditional-ratio + mass-anchor 25+
- vacancy branch frozen M95I
- other RBs frozen M95F
- central carries M94C unchanged

2025 stable workhorses (`n=237`):

20+:

- M95F AUC `.581185`, Brier `.186593`
- M95K AUC `.641164`, Brier `.171528`
- AUC gain `+0.059979`
- aggregate mean probability exactly preserved at `29.60%`

25+:

- M95F AUC `.591714`, Brier `.053017`
- M95K AUC `.612631`, Brier `.051386`
- AUC gain `+0.020917`
- aggregate mean exactly preserved at `11.10%`

Strong individual 2025 signals included player current-season p95 carry ceiling (`25+ AUC .7170`), player p90 (`.6840`), team lead-RB p95 (`.6549`), team p90 (`.6335`).

Full 2025 population also improved 20+/25+ AUC and Brier. All M95K scientific tail gates passed, but the M94C legacy rush-yard guard remained negative and 2025 was not pristine, so K was correctly sent to sealed confirmation instead of promotion.

## M95L — sealed temporal confirmation

Primary confirmation: 2023 Weeks 13-18, training cutoff W12. Frozen M95K architecture; no new feature or coefficient search; no sportsbook; central carries unchanged.

Mechanical history before the authoritative run:

- initial M94C player join coverage was `95.585%`, below the frozen `97%` gate;
- diagnostic proved all 20 failed joins were verified player-identity aliases;
- a GSIS-stable identity bridge repaired coverage to **100.000%** without lowering the gate;
- one subsequent duplicate calibrated-tail merge bug was fixed mechanically;
- no sealed metrics were used to make either mechanical repair.

Authoritative completed sealed run:

- run `33429747106`
- job `99611940386`
- SHA `caa9401eb50f6980e2a2c35ddd8e54467f57cbef`
- artifact ID `9772316395`
- execution success
- source/player join `1.000000`
- disposition **`M95K_SEALED_TEMPORAL_CONFIRMATION_FAILED`**

2023 stable workhorses (`n=73`, 24 actual 20+ events, 10 actual 25+ events):

20+:

- M95F AUC `.727041`, Brier `.233221`
- frozen M95K/M95L AUC `.545068`, Brier `.244446`
- AUC gain `-0.181973`
- Brier gain `-0.011225`

25+:

- M95F AUC `.533333`, Brier `.123614`
- frozen M95K/M95L AUC `.442857`, Brier `.126356`
- AUC gain `-0.090476`
- Brier gain `-0.002742`

All-RB 20+/25+ also regressed modestly. Stable probability mass was preserved exactly. M94C central-carry reference was perfectly preserved. Vacancy 25+ had zero positive events and was correctly labeled inconclusive-small-N rather than passed.

Interpretation: M95K remained a legitimate strong 2025 research finding, but the exact architecture did not generalize to the independent 2023 late-season rotation and therefore cannot be promoted.

## Latest completed migration: M95M — cross-season failure postmortem

Authoritative:

- workflow `M95M RB Cross-Season Tail Postmortem`
- run **`33433593731`**
- job **`99624596080`**
- tested SHA **`52b537dcd52561a1545a8c87b381c1ea5fca63da`**
- artifact `migration-95m-rb-cross-season-tail-postmortem`
- artifact ID **`9773558859`**
- artifact SHA256 **`4b7277e815db7a5bf90a6abb9ef61efa23b7919fcfec5ef43c5c0ca6c206aead`**
- execution success
- role `postmortem_only_no_model_change`
- feature search `0`; coefficient search `0`; new model fit `0`; sportsbook `0`; production change `0`

Full results: `docs/migrations/M95M_RB_CROSS_SEASON_TAIL_POSTMORTEM_RESULTS.md`.

### Key M95M finding 1 — 20+ failure is genuinely cross-season, not merely a late-season-window artifact

2025 W13-18 stable workhorses (`n=85`) using the already-produced M95K trace:

- M95F 20+ AUC `.646858`
- M95K 20+ AUC `.732923`
- gain **`+0.086066`**
- Brier improvement `+0.008080`

2023 W13-18:

- AUC gain **`-0.181973`**
- Brier regression `-0.011225`

Thus the same W13-18 calendar slice still shows a strong 20+ M95K benefit in 2025 and a sharp reversal in 2023. Primary M95M pattern: **`cross_season_nonstationarity_same_window`**.

### Key M95M finding 2 — 25+ is even less robust

2025 full:

- AUC gain `+0.020917`

2025 W13-18:

- AUC gain **`-0.084388`**
- Brier regression `-0.004716`

2023 W13-18:

- AUC gain **`-0.090476`**

So the frozen 25+ conditional-ratio/mass-anchor result is not robust even inside late-2025. Treat it as especially unstable.

### Key M95M finding 3 — the strongest player-current-season ceiling signals are nonstationary

25+ univariate AUC:

- player current-season p95: `2025 full .717015`, `2025 W13-18 .622363`, `2023 W13-18 .481746`
- player current-season p90: `2025 full .684031`, `2025 W13-18 .544304`, `2023 W13-18 .486508`

20+:

- player current-season p95: `2025 full .631029`, `2025 W13-18 .732923`, `2023 W13-18 .534439`
- player current-season p90: `2025 full .624272`, `2025 W13-18 .708675`, `2023 W13-18 .538265`

But not all feed information died in 2023:

- `feed25_rate` AUC: 20+ `.640731`, 25+ `.586508`
- team current-season lead-RB p90: 25+ `.580952`
- composite carry-ceiling90: 25+ `.553968`

Therefore do **not** conclude that historical feed/workload information is useless. The exact global M95K combination is the failure.

### Key M95M finding 4 — sample depth does not explain the failure

2023 20+ AUC gain by sample-depth tercile:

- low `-0.152778`
- mid `-0.140625`
- high `-0.243697`

2023 25+:

- low `-0.090909`
- mid `-0.015873`
- high `-0.225000`

More history did not rescue K; the highest-depth 20+ group was the worst.

### Key M95M casebook

Representative harmful 2023 20+ reallocations:

- Kyren Williams W14 actual 25: `.24097 -> .14356`
- Chuba Hubbard W14 actual 23: `.20478 -> .11990`
- Rachaad White W14 actual 25: `.22113 -> .14362`
- Jonathan Taylor W17 actual 21: `.22770 -> .15378`
- Josh Jacobs W14 actual 13: `.16818 -> .24094`
- Derrick Henry W18 actual 19: `.09374 -> .18888`
- Derrick Henry W15 actual 16: `.10814 -> .17157`
- Derrick Henry W17 actual 12: `.09864 -> .15668`

This is consistent with a conditional-response / stale-ceiling issue: persistent historical workload reputation sometimes received too much weight relative to the actual current-week micro-environment.

# Current scientific interpretation

The RB architecture already contains meaningful individualization at the **input** level: player history, offense, opponent defense, game-state expectation, role, injuries, depth, backfield competition, QB rush siphoning and team workload tendencies vary player-by-player and week-by-week.

However, much of the mapping from those inputs to the output is still **globally shared within a regime**. M95I introduced one important regime split (vacancy/transition versus stable incumbent), but M95M now provides evidence that even the stable-workhorse regime may contain heterogeneous response types. The same feed/ceiling signal can be highly useful in one season/context and weak or inverse in another.

This motivates testing whether the correct architecture is a global backbone plus conditional/local response mechanisms rather than one stable-workhorse formula for every player-week.

Do not interpret this as permission to hand-pick a model for an individual game after seeing the matchup. Any expert/regime selection must be defined from pregame variables and validated temporally.

# NEXT MIGRATION — M95N

Name: **M95N — Conditional Game-Environment / Micro-Regime Audit**

Primary question:

> Are the useful RB feed/workload signals stable *within identifiable pregame player-game archetypes*, even though one global stable-workhorse mapping fails across seasons?

This is an audit first, not a production candidate.

Required design principles:

- no retuning M95K to make 2023 pass;
- no sportsbook inputs;
- no postgame features in regime assignment;
- no hand-picked player exceptions;
- keep M94C central carries unchanged;
- keep M95L failure authoritative;
- treat 2023 W13-18 as opened research data, not a pristine validation set;
- any later candidate derived from M95N must receive a new genuinely prospective/untouched confirmation protocol.

Audit pregame-only conditional response dimensions such as:

- incumbent role stability / recent share stability;
- player/team carry-ceiling recency versus stale career history;
- backfield competition and RB count;
- vacancy/transition status;
- team rush-opportunity environment;
- expected lead/trail game script;
- opponent run-defense strength / structural matchup;
- QB rushing siphon;
- team/play-caller rush tendency where available;
- player/team continuity and role tenure if a reliable leakage-safe source can be constructed.

Required outputs:

1. cross-season signal AUC/correlation within each pregame archetype;
2. whether signal direction is consistent across 2023 and 2025 within the same archetype;
3. minimum sample sizes and event counts for each archetype;
4. player-week casebook showing where a global model and a conditional archetype would disagree;
5. evidence for or against a mixture-of-experts / hierarchical random-slope architecture;
6. explicit fail-closed conclusion if no stable micro-regimes exist.

Do not fit a new promoted model in M95N. First establish whether the user's "individual game environment" hypothesis has repeatable statistical structure.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from the `NEXT MIGRATION` section. Do not reconstruct the project from unrelated memories or restart old research. Preserve all modeling and validation rules in the handoff file.
