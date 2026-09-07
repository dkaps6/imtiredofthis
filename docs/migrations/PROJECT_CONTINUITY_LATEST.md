# NFL Player Modeling — Current Continuity Ledger

**Last updated:** 2026-09-07

**Repository:** `dkaps6/imtiredofthis`

**Purpose:** This is the first-stop handoff file for a new ChatGPT thread/session. Read this file, then inspect the referenced branch plans/results and current GitHub Actions runs before doing new work.

---

## 1. North star

The sole modeling objective is the most accurate possible **individual pregame player projections/distributions** for QB, RB, WR and TE. The user cares directly about the football number (for example passing/rushing/receiving yards) being as close as possible to the actual outcome. Market superiority is the ultimate downstream scoreboard, but sportsbook information must never be used to alter upstream football projections.

Canonical architecture direction:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO OUTCOME`

Full architecture decision lives on branch `research-joint-opportunity-entitlement-v1`:
`docs/migrations/JOINT_OPPORTUNITY_ENTITLEMENT_V1_PLAN.md`
Architecture commit: `b69c828098bce9c7fb4a72ee9bc440a2c21d1db9`.

Key rule: stop treating final player yardage as an approximately independent positional output. Model the finite amount of football available in the game, allocate it to players, then model what each player does with that opportunity.

---

## 2. Permanent methodology rules

- Canonical production orchestration is `.github/workflows/full-slate.yml` only.
- Strict leakage-safe walk-forward historical evaluation.
- Sportsbook data downstream only; large model-vs-market gaps are **audit triggers, not correction triggers**.
- Freeze hypotheses, feature sets, coefficients, gates and stopping rules before results.
- Mechanical/data/integrity failure != scientific failure. Mechanical failures may receive minimal plumbing repair only; scientific failures are preserved as evidence and may not be rescued by threshold/alpha/window tuning.
- Preserve validated production work unless a separately frozen full-stack test earns replacement.
- Score individual-player quality, not only pooled MAE: MAE/RMSE/bias/correlation, median/p75/p90 AE, catastrophic misses, role/player tiers, season stability and mechanism coherence.
- Monte Carlo is the final distribution engine; rules modify inputs rather than vote on outputs.

---

## 3. Current production anchors

### QB
- Production mean: `QB_PASS_SYNTHESIS_V1` / M89-M90 synthesis.
- M89 run `33331073376`.
- M90 prospective confirmation run `33333730480`.
- M90 2023: MAE `60.632751 -> 56.559869`, RMSE `75.634635 -> 69.629921`, bias `-27.248932 -> -6.606555`, corr `.172628 -> .243467`, 100+ misses `81 -> 64`.
- Keep M89/M90 until a separately frozen full-stack candidate beats it.

### WR
- Production hierarchy: M38 target-share multipliers `1.40 / 1.14 / .91 / .78`.
- Exact M38 parent SHA `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- M38 confirmed 6/6 seasons 2020-2025; pooled n `12396`, receiving-yard MAE `24.215219 -> 23.247553`.

### RB
- Production central rushing anchor: P3 (`RB_P3_SYNTHESIS_V1` / Week-1 stack override where applicable).
- Production parent `754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`.
- Do not reopen M95 tail overlay family; M95T formally stopped that path.

### TE
- No promoted dedicated TE model yet. Dedicated TE research is now active.

---

## 4. Decisive cross-position evidence

### Shared pass-volume mechanism
QB attempt residual and aggregate WR opportunity residual are strongly coupled. 2025 team-game audit:
- Pearson `0.6973`
- Spearman `0.6706`
- same direction `73.2%`
- independent 2024-2025 WR reception-error confirmation Pearson `0.5239`

This is the main reason the architecture now uses one shared team pass state for QB and all receiving positions.

### Receiving conservation
Audit supported both position-group misallocation and material pass/receiving conservation gap.

Joint Pass/Receiving Conservation V1 canonical scientific run:
- run `34081764151`
- job `101618243530`
- SHA `4f620f1ea24a10cdf840d52d42a8930e5991f1ee`
- artifact `10004223287`
- digest `sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac`
- disposition: `CONSERVATION_ONLY_SUPPORTED`

C2 conservation result on exact 884 M89/M90 QB-games:
- QB mean MAE unchanged: `55.0601 -> 55.0601`
- CRPS `40.3866 -> 39.0938` (improvement ~`1.2928` yards)
- bootstrap probability improvement `1.000`
- 80% interval coverage `58.71% -> 71.27%`
- 90% interval coverage `69.34% -> 82.47%`
- conservation identity gap `0.0`
- macro WR/TE/RB receiving-yard MAE `16.9594 -> 16.9104`

C1 group target-mass calibration and C3 joint combination failed player-level protection gates. Do not reuse them.

---

## 5. Current full-stack C2 integration status — ACTIVE

Branch: `research-pass-receiving-conservation-integration-v1`
Frozen plan: `docs/migrations/PASS_RECEIVING_CONSERVATION_INTEGRATION_V1_PLAN.md`

Initial full-stack integration run:
- run `34131591002`
- job `101772755332`
- SHA `a4f1c3308f159fb462a5ae5e9767338780954fae`
- artifact `10022570140`
- artifact digest `sha256:804a49aa46aad7198bbef5d8c5c0a93381065e2d505b48476dd5b5cd205853dc`

This was **mechanical only**, not a science result. All six seasons built successfully; evaluator crashed because 2020-2023 QB distribution CSVs are intentionally empty outside the M89/M90 2024-2025 scoring cohort and `pd.read_csv` raised `EmptyDataError`.

Mechanical repair commit:
- `647514b7dbeb5cab3223fc603e0a74e721879ba5`
- message: `Repair integration evaluator for intentionally empty QB cohort files`
- repair changes evaluator I/O only; no frozen science/gates/model changes.

Canonical repair rerun:
- run `34139757238`
- head SHA `647514b7dbeb5cab3223fc603e0a74e721879ba5`
- status when this ledger was written: **queued**.

Immediate next action: inspect run `34139757238` when it completes. If mechanical integrity passes, use its exact frozen science disposition. Do not retune C2.

---

## 6. QB current status

### Production
M89/M90 remains authoritative mean.

### Week-1 concern
Large 2026 Week-1 model-vs-market discrepancies remain unresolved football-audit questions. Do not use the market to correct them.

### QB-PD3 internal disagreement diagnostic
- branch `research-qb-pd3-internal-disagreement-reliability`
- run `34122984048`
- job `101745071358`
- head SHA `6baa2ae4f4bb07277d01317da324ea8b8c42b35e`
- artifact `10018942911`
- digest `sha256:cde910939dba0631a2a68e9acfbcafcdb90af1d1b2128627442073854bf3f849`
- n `884` (2024-2025)
- disposition `NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE`

Meaning: large synthesis moves, +45 cap hits, component disagreement and multi-flag states do **not** provide a stable historical reason to distrust M89/M90. In several flagged states synthesis still beat the base. Therefore the Week-1 gaps must be investigated through real football inputs: team pass opportunity, attempts, YPA, current-team/player-history mapping, opponent, receiver ecosystem and game environment.

Authorized QB direction:
1. finish 2026 W1 football-only component/path integrity audit;
2. shared team pass state;
3. C2 receiving conservation distribution;
4. no new generic mean hunt unless architecture diagnostics identify a specific failed layer.

---

## 7. WR current status

### M38
Keep M38 as the baseline entitlement/hierarchy prior.

### WR-R10 NGS source eligibility
- run `34123975300`
- disposition `STRICT_PRIOR_NGS_FEATURES_ELIGIBLE`
- strict-prior NGS eligible rows `10427` (2021-2025)
- prior-1 coverage `86.33%`
- prior-3 coverage `74.65%`
- zero same/future observations used.

### WR-R11 NGS residual target model — SCIENTIFIC FAIL
- branch `research-wr-r11-strict-prior-ngs-target-model`
- run `34124533822`
- job `101749972264`
- head SHA `d2f251dc267db854e6747734d6cbe692d56f93ae`
- artifact `10019545653`
- digest `sha256:388973f0fc67c6a2f0e4868a7c874d310ff7711cd7b0a6fb38c74128dcd43370`
- eligible OOS rows `6383`
- disposition `WR_NGS_TARGET_MODEL_FAIL`

Important numbers:
- pooled target MAE `2.32344 -> 2.28789` (small improvement, below frozen >=0.05 gate)
- target RMSE `3.15783 -> 2.87132`
- target p90 `5.3334 -> 4.6509`
- but receiving-yard MAE **worsened** `24.84586 -> 26.00565`
- 2024-2025 receiving-yard MAE `24.28422 -> 25.26003`
- high-target-Q4 rec-yard MAE `31.86029 -> 32.23672`
- target correction was positive in `100%` of OOS games in all four test seasons.

Interpretation: like TE-R3, this model mostly learned a global underprojection correction rather than differentiated player entitlement. NGS is still an eligible source, but WR-R11's formulation is closed. Do not tune its alpha/window/blend to rescue it.

Authorized WR direction: model finite team/WR opportunity first and **relative player entitlement around M38** using role/participation/competition/injury/transition context; efficiency must remain separate from volume.

---

## 8. TE current status

### TE mechanism decomposition
Dedicated TE lane established. TE error decomposition (6371 TE player-games) showed targets are the largest receiving-yard error mechanism.

### TE-R2 pool vs individual allocation — ACTIONABLE DIAGNOSTIC
- run `34126026512`
- job `101754774701`
- artifact `10020131404`
- digest `sha256:54309eda0886cd0b327333ea89bde12a2fc506d141e2b72ddc3b1d1d778b874d`
- rows `6371`, team-games `3197`
- disposition `TE_TARGET_POOL_FIRST`

Pooled target-error absolute mass:
- TEAM_TE_POOL `60.405%`
- INDIVIDUAL_ALLOCATION `39.595%`

Highest-error quartile:
- TEAM_TE_POOL `72.073%`
- INDIVIDUAL_ALLOCATION `27.927%`

Thus TE architecture must be: team pass state -> finite TE pool -> individual TE entitlement -> catch/yard efficiency.

### TE-R3 target-pool context candidate — SCIENTIFIC FAIL
- run `34126813280`
- artifact `10020423842`
- digest `sha256:99699c7e4745ab8cf678b527fc57325c5f402a5c7be930ff9cb90dba96e63ff8`
- result commit `16747b8a0194dceaabd280d131e22d3281530d94`
- disposition `TE_TARGET_POOL_CONTEXT_MODEL_FAIL`

It improved team TE-pool MAE `2.604965 -> 2.293122` and player target MAE `1.666440 -> 1.605626`, but receiving-yard MAE worsened `16.183617 -> 16.339728`. Correction was positive in `99.48%` of games. It learned a generic underprojection correction, not sufficiently player-specific game behavior.

### TE-R4 strict-prior participation source — PASSED
- branch `research-te-r4-strict-prior-participation-source`
- run `34127474412`
- head SHA `ffe4e1101193b3502a6b069d2b77347049719b1d`
- artifact `10020700686`
- digest `sha256:56de7efe302cdf9c329c3a0386d798789d15e6a495c865535291173b6840c163`
- result documentation commit `a97ea9b6e39aee67c34bccc9668951828ae56af2`

Coverage:
- prior-1 any-team participation `97.91%`
- prior-1 same-team `96.59%`
- prior-3 any-team `95.92%`
- zero same/future observations used.

Authorized TE direction: build the next individual TE entitlement model from finite TE pool + strict-prior participation + same-team role + competition/transition context. Do not apply a universal TE boost.

---

## 9. RB current status

### Role/depth remap closed
Run `34063904515`, artifact `9998334300` proved simple depth-rank remapping is bad:
- carry MAE `3.483 -> 4.106`
- rushing-yard MAE `20.424 -> 22.837`

Depth chart is contextual evidence only, never direct workload authority.

### M95 tail family closed
M95T run `33455690862`, artifact `9781352939`, formal stop retained. Do not create another detached tail overlay candidate.

### PD3/PD4/PD5 residual family
Recent 2025-only residual calibration experiments repeatedly improved central MAE but hurt p90. PD5 example:
- carry MAE `3.4826 -> 3.4265`
- rushing-yard MAE `20.4242 -> 20.2385`
- rushing-yard p90 `44.3846 -> 44.6320` (fails frozen guard)

Critical integrity note: the recent PD5 work was only on the 2025 cohort (`1393` rows) even though some surrounding planning language suggested broader history. Treat it as exploratory 2025 evidence, not multiseason promotion evidence.

Authorized RB direction: team rushing opportunity -> finite RB room -> individual backfield entitlement based on multi-game carry share, participation, same-team continuity, teammate competition, injury-created vacancy, QB rushing competition and role transitions -> separate YPC/efficiency distribution. RB receiving must simultaneously participate in the shared receiving ecosystem, and rush+receiving total yards must be scored.

---

## 10. Receiving Attempt Semantics C4 — CLOSED

Canonical repaired run `34077637287`, job `101606772786`, SHA `d5f0ba8af56caa28f8665fbea15d74885e09a65c`, artifact `10002832205`.
Disposition: `ATTEMPT_SEMANTICS_CANDIDATE_FAIL`.

- team targets `6.3014 -> 6.1874`
- player target MAE `1.7747 -> 1.8015` (worse)
- macro receiving-yard MAE `16.9594 -> 16.9990` (worse)
- 0/6 seasons improved receiving-yard MAE.

Do not retry this semantics family.

---

## 11. The decisive current research thesis

The project has spent too much time adjusting final-stat outputs. The common causal failure is opportunity allocation.

The next architecture must predict, jointly and conservatively:

1. **Game environment**: plays, pace, pass/run tendency, score/game-script distribution, opponent environment.
2. **Team opportunity**: one finite pass state and one finite rush state.
3. **Position/room pool**: WR / TE / RB receiving target pools and RB rushing room.
4. **Player entitlement**: normalized shares using strict-prior role/participation/history, same-team continuity, teammate competition, injuries/vacancies, transitions/new-team/rookie context and eligible matchup data.
5. **Per-opportunity efficiency**: catch probability, YPR/YAC/air depth, YPC/explosive mixture, player + offensive environment + opponent interaction.
6. **Joint MC conservation**: QB passing yards emerge coherently with WR/TE/RB receiving yards; RB rushing and receiving share the same game script.

This does **not** discard M89/M90, M38 or P3. Those become strong priors/anchors inside the hierarchy until a frozen replacement beats them.

---

## 12. Immediate execution order

1. **Inspect C2 integration rerun `34139757238`** and formally disposition it.
2. If integration passes, stage a separate production/full-slate confirmation; if it scientifically fails, diagnose the exact failed layer without tuning the candidate.
3. Build next **TE individual entitlement** candidate using TE pool + TE-R4 participation.
4. Build next **WR relative entitlement** candidate around M38; WR-R11 formulation is closed, but strict-prior NGS remains an eligible supplemental feature source.
5. Build proper **RB shared room + entitlement + separate efficiency** historical candidate with multiseason evidence; do not reopen tail overlay/depth remap/residual retuning families.
6. Complete **QB 2026 W1 football-only component/path integrity audit** and connect QB attempts to the shared pass state.
7. After football model qualification, score downstream against sportsbook distributions/lines under frozen rules. Large market discrepancies never modify upstream football projections.

---

## 13. Continuity procedure for future sessions

A new session should:

1. read this file first;
2. read `AGENTS.md`;
3. inspect the current branch heads and GitHub Actions statuses for the exact branches above;
4. inspect result docs/artifacts before creating a new experiment;
5. continue from the `Immediate execution order` without reopening closed families.

Update this ledger whenever a meaningful run completes, a mechanical repair changes lineage, a candidate is formally passed/failed, or the authorized next step changes.
