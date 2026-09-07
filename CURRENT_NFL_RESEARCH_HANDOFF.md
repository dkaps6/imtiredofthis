# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the active NFL player-projection research program.  
**Last updated:** 2026-09-07 ~09:40 ET.  
**Main branch before this handoff commit:** `f1e5d59ad0db5800fdd730c7f4b4ebfbfb2a4daa`.

> **Future ChatGPT sessions / agents: read this file first, then `AGENTS.md`, then the exact active branch plan(s) named below before making any model or workflow change. Update this file on `main` after every substantive result, mechanical repair, frozen plan, promotion, failure, or change in authorized next step.**

---

## 1. User's north star

The user's simple objective is the controlling objective:

> If a QB, RB, WR, or TE is projected for **X yards**, make X as close as possible to the actual game outcome, player by player, pregame.

The ultimate market objective is to have the football-only player distributions and fair probabilities outperform sportsbook player-prop pricing. Sportsbook data is **downstream only**. Large model-vs-market discrepancies are audit triggers, never automatic corrections.

The user has a hard near-term deadline and does not want more circular diagnostics. Research must convert what has already been learned into real individual-player predictive gains.

---

## 2. Non-negotiable methodology

- Canonical production authority is **`.github/workflows/full-slate.yml`**. Read `AGENTS.md` before touching production.
- Historical tests are strict walk-forward / leakage-safe.
- Sportsbook inputs are prohibited upstream of football projections.
- Freeze hypothesis, candidate, cohort, thresholds, gates, and outcome definitions **before** results.
- Mechanical/data/integrity failures may receive the minimum plumbing repair and exact rerun; they are not scientific failures.
- Scientific failures are preserved. Do not lower gates, tune nearby thresholds, window-hunt, or rescue a near miss.
- Failed experiments remain evidence and must not be silently reopened.
- Individual-player evaluation is mandatory: MAE/RMSE/bias/correlation plus median/p75/p90 absolute error, catastrophic miss rates, player/role tiers, season stability, opportunity-vs-efficiency decomposition, and distribution metrics where available.
- A pooled MAE improvement that worsens important player tails or unstable tiers is not automatically promotable.
- Full-stack/production promotion always requires a separately frozen integration confirmation.

---

## 3. Decisive architecture decision — current highest-priority direction

Active branch: **`research-joint-opportunity-entitlement-v1`**  
Architecture-plan commit: **`b69c828098bce9c7fb4a72ee9bc440a2c21d1db9`**  
Plan: `docs/migrations/JOINT_OPPORTUNITY_ENTITLEMENT_V1_PLAN.md`

The architecture thesis is:

> **GAME / TEAM OPPORTUNITY → POSITION/ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER+MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

Stop treating a player's final yardage as the primary independent modeling object. The model should first determine how much football opportunity exists in the game, then conserve and allocate it to players, then model what each player does per opportunity.

This direction is supported by the strongest repeated evidence in the repo:

- QB-WR shared passing-volume residual coupling is very strong (`2025 Pearson ~0.6973`, Spearman `~0.6706`).
- WR errors are predominantly target/opportunity driven; M38 is the strongest durable WR win.
- TE-R2 shows team TE-pool error dominates TE target error.
- RB carries and rushing yards are strongly linked, while static depth remaps and retrospective tail overlays failed.
- C2 shared pass/receiving conservation materially improved QB distribution quality without moving the promoted QB mean.

The active architecture plan explicitly pauses broad generic feature hunts, fixed player-error corrections, static depth-rank remaps, and new RB tail overlays while this hierarchy is tested.

---

## 4. Shared pass / receiving ecosystem

### Receiving Ecosystem Conservation Audit

Parent diagnostic found both:

- `POSITION_GROUP_MISALLOCATION_SUPPORTED`
- `MATERIAL_PASS_RECEIVING_CONSERVATION_GAP`

Shared pass-volume mechanism is independently strong.

### Joint Pass/Receiving Conservation V1

Branch: `research-joint-pass-receiving-conservation-v1`  
Canonical scientific run: **`34081764151`**  
Job: **`101618243530`**  
SHA: **`4f620f1ea24a10cdf840d52d42a8930e5991f1ee`**  
Artifact: **`10004223287`**  
Digest: `sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac`

Disposition: **`CONSERVATION_ONLY_SUPPORTED`**.

Key C2 result:

- QB mean remained exactly the promoted M89/M90 mean.
- QB CRPS improved roughly `40.3866 -> 39.0938` (`~1.2928 yards`).
- Paired bootstrap probability of CRPS improvement = `1.0`.
- 80% interval coverage improved from about `58.71% -> 71.27%`.
- Macro WR/TE/RB receiving-yard MAE improved `16.959362 -> 16.910357`.
- Iteration-level QB passing yards = modeled receiver yards + residual receiver yards exactly within tolerance.

C1 position-group target calibration failed player-level protection gates. C3 joint combination also failed because it inherited C1's player-allocation harm. **Do not reintroduce C1/C3.**

### Full-stack C2 integration

Branch: **`research-pass-receiving-conservation-integration-v1`**  
Plan: `docs/migrations/PASS_RECEIVING_CONSERVATION_INTEGRATION_V1_PLAN.md`  
Current state: **frozen plan exists; no workflow run had been launched when this handoff was written.**

Authorized next step: implement/run exact B0 vs C2 full-stack integration with M89/M90 QB mean, M38 WR hierarchy, and RB P3 rushing unchanged. Passing only authorizes a separate production confirmation.

---

## 5. QB status

### Current production

**QB_PASS_SYNTHESIS_V1 / M89-M90** remains production authority for passing-yard mean.

Historical promotion evidence already established a real predictive improvement (prospective 2023 MAE about `60.63 -> 56.56`, RMSE `75.63 -> 69.63`, bias substantially repaired, 100+ misses reduced). Do not replace it casually.

### Current concern

2026 Week 1 has several very large model-vs-Vegas discrepancies. The market is **not** used to correct them, but the size of the gaps requires football explanation: attempts, YPA, current-team/player-history mapping, matchup/game environment, receiver ecosystem, component semantics, and any 2026 context gaps.

The no-odds production audit already proved sportsbook lines do not alter the football mean.

### QB-PD3 internal disagreement reliability

Branch: `research-qb-pd3-internal-disagreement-reliability`  
Run: **`34122984048`**  
Job: **`101745071358`**  
SHA: **`6baa2ae4f4bb07277d01317da324ea8b8c42b35e`**  
Artifact: **`10018942911`**  
Digest: `sha256:cde910939dba0631a2a68e9acfbcafcdb90af1d1b2128627442073854bf3f849`

Disposition: **`NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE`** on 884 QB-games (2024-2025).

Large component range, 30+ synthesis moves, +45 cap hits, and multiple simultaneous flags did **not** identify a stable state in which the synthesis should be distrusted versus the underlying base. In many such states synthesis remained better than base. Therefore do not shrink or override a QB merely because the synthesis correction is large.

### QB next authorized step

- Finish **football-only 2026 Week-1 component/path integrity audit** on the actual large-gap QBs.
- Audit exact `qb_pred_attempts`, `qb_pred_ypa`, current-team/player-history mapping, opponent/game environment, receiver ecosystem, cap/extrapolation, and missing/borrowed 2026 context.
- Then route QB attempts through the same shared team pass state as WR/TE/RB receiving and use C2 conservation for distribution shape.
- No broad new generic QB mean feature hunt.

---

## 6. WR status

### Current production

**M38 target-share hierarchy** remains the canonical WR winner. It improved receiving-yard MAE in **all 6 seasons, 2020-2025**, while preserving team WR target mass. M38 should be treated as the **baseline entitlement prior**, not thrown away.

### Strict-prior NGS source

WR-R10 established strict-prior tracking availability was sufficient for research (`STRICT_PRIOR_NGS_FEATURES_ELIGIBLE`).

### WR-R11 NGS target residual model

Branch: `research-wr-r11-strict-prior-ngs-target-model`  
Run: **`34124533822`**  
Job: **`101749972264`**  
SHA: **`d2f251dc267db854e6747734d6cbe692d56f93ae`**  
Artifact: **`10019545653`**  
Digest: `sha256:388973f0fc67c6a2f0e4868a7c874d310ff7711cd7b0a6fb38c74128dcd43370`

Disposition: **`WR_NGS_TARGET_MODEL_FAIL`**.

Important results on 6,383 eligible OOS rows (2022-2025):

- Target MAE improved only `2.32344 -> 2.28789` (not enough for frozen >=0.05 gate).
- Target p90 improved `5.3334 -> 4.6509`.
- Receiving-yard MAE **worsened badly** `24.8459 -> 26.0057`.
- 2024-2025 receiving-yard MAE worsened `24.2842 -> 25.2600`.
- 30+ yard miss rate worsened.
- High-target Q4 receiving-yard MAE worsened `31.8603 -> 32.2367`.
- Most revealing pathology: the learned target correction was **positive in 100% of OOS rows in all four folds**. It learned another broad underprojection correction, not true differentiated player entitlement.

Do not retry this candidate with nearby alpha/caps/windows.

### WR next authorized step

Build a **conserved WR entitlement-around-M38 model**, not another additive target bump. The model must allocate a finite team/WR opportunity pool using strict-prior target share, participation/routes/snaps where qualified, same-team continuity, teammate competition, injuries/vacancy, new-team/rookie transitions, shared pass state, and opponent coverage/matchup context. Normalize shares within team/WR pool. Keep catch conversion and YPR/yardage efficiency separate.

---

## 7. TE status

TE is now an independent research lane, not treated as generic WR.

### TE-R1

Initial mechanism decomposition on 6,371 TE player-games found receiving-yard error mass approximately:

- Targets `45.23%`
- Catch rate `33.66%`
- YPR `21.11%`

In the highest-error quarter, targets were overwhelmingly the dominant mechanism.

### TE-R2 — pool vs individual allocation

Branch: `research-te-r2-pool-vs-individual-allocation`  
Run: **`34126026512`**  
Job: **`101754774701`**  
SHA: **`234d58129a2d43126ed91715df9477c00c1d11f4`**  
Artifact: **`10020131404`**  
Digest: `sha256:54309eda0886cd0b327333ea89bde12a2fc506d141e2b72ddc3b1d1d778b874d`

Disposition: **`TE_TARGET_POOL_FIRST`**.

On 6,371 TE player-games:

- Team TE-pool component = `60.405%` of absolute target-error mass.
- Individual allocation component = `39.595%`.
- Highest-error quartile: team TE-pool component rises to `72.073%`.
- All six seasons independently support the pool-first route.

### TE-R3 target-pool context model

Branch: `research-te-r3-target-pool-context-model`  
Run: **`34126813280`**  
Job: **`101757309151`**  
SHA: **`c9d1cd1858e34900e33fabc07848fc3f9f86bd1e`**  
Artifact: **`10020423842`**  
Digest: `sha256:99699c7e4745ab8cf678b527fc57325c5f402a5c7be930ff9cb90dba96e63ff8`

Disposition: **`TE_TARGET_POOL_CONTEXT_MODEL_FAIL`**.

Useful partial signal:

- Team TE-pool MAE improved `2.60497 -> 2.29312` and won all 4 OOS seasons.
- Player target MAE improved `1.66644 -> 1.60563`.
- Player target p90 improved strongly.
- But receiving-yard MAE worsened `16.18362 -> 16.33973` and improved in only 1/4 seasons.
- Correction was positive in `99.48%` of games, revealing a broad underprojection fix rather than true game/player differentiation.
- High-volume Q4 TE receiving-yard MAE actually improved (`23.99385 -> 22.60513`) and tails improved, but the frozen pooled/temporal gates still failed. Preserve failure.

### TE-R4 strict-prior participation source

Branch: `research-te-r4-strict-prior-participation-source`  
Run: **`34127474412`**  
Job: **`101759439856`**  
SHA: **`ffe4e1101193b3502a6b069d2b77347049719b1d`**  
Artifact: **`10020700686`**  
Digest: `sha256:56de7efe302cdf9c329c3a0386d798789d15e6a495c865535291173b6840c163`

Disposition: **`STRICT_PRIOR_TE_PARTICIPATION_ELIGIBLE`**.

Source: `nflreadpy.load_snap_counts`, 150,909 source rows, 2020-2025.

On 5,371 target TE rows (2021-2025):

- prior-1 any-team coverage `97.9147%`
- prior-3 any-team `95.9225%`
- prior-1 same-team `96.5928%`
- prior-3 same-team `91.7892%`
- duplicate rate `0`
- same/future observations used `0`

### TE next authorized step

Build a **TE entitlement model** that first projects finite team TE target pool, then allocates it to individual TEs using strict-prior participation/snap state, same-team continuity, historical target share, room competition, roster/depth context, injuries/vacancy, QB/team pass environment, and opponent TE matchup. Keep catch-rate and YPR efficiency separate. Do not apply another universal TE target boost.

---

## 8. RB status

### Current production

**RB_P3_SYNTHESIS_V1**, 2026 Week-1 route `WEEK1_STACK_OVERRIDE`, remains production authority for qualified Week-1 rushing-yard means. Production parent `754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`.

### Role-order remap is CLOSED

Branch: `research-rb-role-order-remap-v1`  
Canonical run: **`34063904515`**  
Job: **`101569200535`**  
SHA: **`a13020f4fe098ef5f51df886b6ae4dd78f751b7e`**  
Artifact: **`9998334300`**  
Digest: `sha256:55ce439daef26637ec38b68144d4eb5b0dc48d4ec5d7114ef6fa01cb46c04594`

Disposition: **`ROLE_ORDER_REMAP_V1_NOT_ACTIONABLE`**.

Forcing current depth rank to own existing carry magnitudes worsened overall carry MAE `3.482576 -> 4.106428` and rushing-yard MAE `20.424163 -> 22.836638`. RB1/RB2/RB3 slices all worsened. Depth rank is contextual evidence, not direct opportunity authority.

### M95T is CLOSED and is the stopping rule for retrospective RB tail overlays

Branch: `research-rb-m95t-constrained-dual-layer-tail`  
Run: **`33455690862`**  
Job: **`99695055863`**  
SHA: **`540edb67d9d5451764e997f19213b80285c15fab`**  
Artifact: **`9781352939`**  
Digest: `sha256:bd54485d18de4f4df1f7613d9587281234bf07f8ce9de6df36b68bca26167c70`

Disposition: **`M95T_FAIL_STOP_NEW_RB_TAIL_CANDIDATES_RETAIN_M94C_M95F_PROCEED_M96`**.

Carries vs rushing yards on the comparable stable-workhorse panel: Pearson `0.789267`, Spearman `0.788957`. Opportunity is highly important, but point rushing-yard accuracy still requires efficiency, blocking/offensive environment, opponent run environment, and explosive variance.

### Recent PD residual calibration caution

Recent PD3/PD4/PD5 experiments on 2025 showed central-MAE hints but failed tail guards. Crucially, PD5 was **2025-only** despite initially being discussed as if it were multiseason. Treat it as exploratory 2025 evidence only, not multiseason proof. Do not tune neighboring alphas/caps/windows to rescue it.

### RB next authorized step

Proceed to the missing **M96-style opportunity-to-yardage translation inside the new joint opportunity/entitlement architecture**:

- shared team rushing opportunity/game script;
- finite RB-room carry pool;
- individual allocation via multi-game carry share, strict-prior snaps/participation, same-team continuity, teammate competition, injury/vacancy, QB rushing competition, rookie/new-team/role-transition state;
- separate per-carry efficiency model using player history + offensive environment + opponent run defense + explosive variance;
- add RB receiving targets/receptions/receiving yards and rush+receiving total yards into the same shared receiving ecosystem.

No new detached tail overlay and no static depth-rank remap.

---

## 9. Market-discrepancy rule

The user originally re-raised QB/WR concerns because some 2026 Week-1 football projections were extremely far from posted Vegas lines. This remains a required audit dimension.

Rules:

- Do **not** move a football projection toward Vegas just because the line differs.
- A large discrepancy triggers component/path inspection.
- After a football model is frozen, compare `P(over line)` / `P(under line)` to vig-removed market probability.
- Ultimate market scoreboard should include calibration, Brier/log loss, CLV, directional accuracy, and realized return under a frozen decision rule.

Known first 20 QB market lines from prior manual screenshots are preserved in conversation history, but do not invent missing lines. If needed, recover exact historical screenshots/notes before using them downstream.

---

## 10. Immediate execution order — do not circle back to broad hunts

1. **QB:** complete football-only 2026 W1 component/path integrity audit for the large-discrepancy players.
2. **Joint stack:** implement and run frozen C2 full-stack conservation integration (`research-pass-receiving-conservation-integration-v1`).
3. **WR:** freeze and run an M38-offset, finite-pool individual entitlement candidate; do not retry WR-R11 additive NGS bump.
4. **TE:** freeze and run pool-first individual TE entitlement using TE-R4 strict-prior participation.
5. **RB:** build M96-style team-opportunity → RB-entitlement → separate efficiency translation, including RB receiving/total yards.
6. Score all four positions with the same individual-player error framework, then run downstream sportsbook comparison only after football qualification.

Parallel work is allowed only if rigor and lineage are not sacrificed.

---

## 11. Permanent position-status format

Every future handoff/update should state for QB, RB, WR, TE:

**Current production model → current open concern → active experiment → frozen gates → last result → authorized next step.**

---

## 12. Continuity protocol for future chats / agents

Before doing new work:

1. Read this file.
2. Read `AGENTS.md`.
3. Inspect the exact branches/run IDs named above and fetch any newer runs created after this timestamp.
4. Read the exact active frozen plan on the branch before changing code.
5. Do not infer that an experiment is unresolved merely because an earlier run failed mechanically; inspect the latest branch runs/result docs.
6. Do not duplicate failed ideas under new names.
7. Update **this file on `main`** after each substantive milestone so the next chat can resume immediately without relying on conversation memory.

If this file conflicts with a newer branch result, the newer exact run/result artifact controls scientific status, but this file must then be updated immediately to restore continuity.
