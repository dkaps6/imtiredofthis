# NFL HANDOFF — 2026-09-23 — PR #626 MERGED + TE LIVE SCIENCE ACTIVE

GitHub is canonical. This handoff supersedes the prior PR-626-integrity checkpoint.

## Read first

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. newest Issue #535 comments after `5805281169`
5. active branch `research-te-live-entitlement-efficiency-v1`

## 1. Scoreboard/integrity closure is DONE

PR #626 is MERGED.

- final PR head: `42c66b690e5b318ae368945bacd6c7762dc2c391`
- merge commit / main at closure:
  `69e9c6211710c60de9aaf07d06aca6cafeff5ccf`
- Repo CI #1124: SUCCESS
- W1/W2 Full Board #100: SUCCESS
- preserved paid-artifact replay #457: SUCCESS
- unresolved correctness/integrity review threads at merge: 0
- paid OddsAPI pull: 0
- football-science mutation from #626: 0

Canonical 2026 Weeks 1-2 production record:

- selected settlement rows: **805**
- decided: **797**
- record: **395-402**
- units: **-42.86**
- QB pass yards: **36-16 / +15.99u**
- TE overall: **58-75 / -21.45u**
- TE receiving yards: **28-41 / -16.09u**
- TE receptions: **30-34 / -5.37u**

The exact scoreboard is now a measurement authority, not the active research
objective.

### Stopping rule

Do not reopen general scorecard hardening.

A future integrity item blocks model science only if it can materially:

1. change which wagers belong in the canonical production record;
2. change settlement/W-L/units materially; or
3. make the canonical replay fail/give the wrong answer.

Peripheral workflow polish is deferred.

## 2. User-directed model-learning priority

The user explicitly reset the project priority:

> Weekly backtesting is useful only insofar as it teaches the model what current
> NFL offenses, defenses, roles, usage and decisions are doing and improves the
> next slate.

The weekly learning architecture should therefore follow:

`pregame projection -> outcome -> diagnose opportunity/role vs efficiency vs
distribution -> compare against historically persistent state -> test one
leakage-safe candidate -> promote only if supported`.

Do not spend another multi-day cycle on scoreboard infrastructure.

## 3. Current-season learning authority remains valid

Current-Season State Persistence V1:

- run `35741758765`
- artifact `10699781744`
- digest
  `sha256:2d81bcb5222136ae812575b15a0d79952d27a03e68034cd312aa0f927503868a`
- rows: 41,745
- 2026 outcomes used: 0
- sportsbook inputs: 0
- production change from study: 0

Untouched 2025 replication improved 9/10 production-aligned metrics.

Key signals:

- RB rush share MAE: 0.1486 -> 0.1166 (**+21.53%**)
- TE target share: 0.05153 -> 0.04553 (**+11.66%**)
- WR target share: 0.06773 -> 0.06144 (**+9.30%**)
- QB YPA full-season: 1.7386 -> 1.6548 (**+4.82%**)
- RB YPC did not improve.

Standing interpretation:

> update opportunity/role relatively quickly; keep early efficiency much more
> heavily shrunk to history.

PR #627 is already merged and prospectively lets WR-R15 / TE-R5P consume
strict-prior 2026 snap participation beginning Week 3 after a schedule-aware
freshness gate. Do not duplicate it.

## 4. Historical TE mechanism authority

TE-R1:

- run `34123413402`
- artifact `10019112418`
- disposition `TE_MECHANISM_DECOMPOSITION_ACTIONABLE`
- scoreable rows: 6,371

Error-mass attribution:

- TARGETS: **45.2%**
- YPR: **29.5%**
- CATCH_RATE: **25.3%**

Targets are the largest individual mechanism, but downstream efficiency
(YPR + catch rate) is ~55% combined.

Do not rerun TE-R1.

## 5. TE Live Entitlement vs Efficiency V1 — COMPLETE

Active research branch:

`research-te-live-entitlement-efficiency-v1`

Frozen plan:

`docs/research/TE_LIVE_ENTITLEMENT_EFFICIENCY_V1_PLAN.md`

Result:

`docs/research/TE_LIVE_ENTITLEMENT_EFFICIENCY_V1_RESULT.md`

Canonical successful execution:

- branch scientific head: `8018f78bd305faf5592ac05c9c7a6838b5bff161`
- run `35939588747`
- job `107444309642`
- artifact `10784323345`
- artifact name `te-live-entitlement-efficiency-v1`
- digest
  `sha256:50fcbe73794dd6111480b9ac0b3e3286d57514515bbe189463571552c8975409`

Exact paid Week-2 football-origin authority:

- source run `35282021679`
- source SHA `c6ec55be70d6e05bbd1dbae83d7d5c86ac8aa00a`
- artifact `10523345092`
- digest
  `sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c`

Run #1 `35939373232` failed before scientific output because sportsbook event
hashes were not join-compatible with football schedule game IDs. That failure
is preserved in Issue #535 comment `5805376633`. The hypothesis/gates did not
change. The successful run uses team + deterministic suffix-normalized player
identity and fails closed on ambiguity.

### V1 scientific result

Disposition:

`CURRENT_SNAP_ENTITLEMENT_SIGNAL_SUPPORTED_EFFICIENCY_REMAINS_PRIMARY_LIVE_GAP`

Week-2 all-TE exact-origin cohort:

- matched rows: **72**
- production target-share MAE: **0.05395**
- W1-2026-snap counterfactual target-share MAE: **0.05270**
- relative improvement: **2.32%**
- production TE-room-share MAE: **0.23389**
- current-snap candidate: **0.22923**
- worst production target-share-error quartile:
  **0.12149 -> 0.11364**
- all frozen current-snap support gates PASS
- sportsbook inputs in football candidate: 0
- team TE-pool conservation exact

This directly supports the PR #627 Week-3+ strict-prior current-snap
continuation.

Canonical Week-2 selected TE receiving-yard cohort:

- rows / target matches: **34 / 34**
- record: **13-21**
- target-share MAE:
  **0.06950 -> 0.06753** with current snaps (**2.84% better**)
- final rec-yard MAE: **20.87 yd**
- current-snap-only entitlement counterfactual MAE: **21.00 yd**
- raw MC MAE: **20.63 yd**
- final-minus-MC MAE delta: **+0.24 yd** (final slightly worse)
- perfect-target-entitlement recoverable error: **+4.11 yd**
- perfect-realized-efficiency recoverable error on nonzero-target rows:
  **+5.68 yd**
- nonzero-target rows where larger recoverable component is:
  - efficiency: **19**
  - entitlement: **10**
- zero-actual-target rows: **5**

Interpretation:

1. current snaps contain real TE opportunity information;
2. opportunity improvement alone does not repair receiving yards;
3. efficiency/yardage translation is the larger remaining live mechanism;
4. the downstream ensemble did not rescue Week-2 TE receiving yards;
5. do not refit TE entitlement coefficients from two live weeks.

Issue #535 result checkpoint: `5805404034`.

## 6. Early-efficiency state interpretation

Current-only early efficiency is not supported.

On exact Week-2 TE outcome matches, current-only YPT/catch rate is materially
worse than historically anchored states.

Current-Season State Persistence two-completed-game evidence across 2022-2025:

- TE target share:
  prior MAE ~0.04353 -> blend4 **~0.03971**
- TE YPT:
  prior ~4.6436 -> blend4 **~4.5101**;
  current-only ~5.0195
- TE receptions/target:
  prior ~0.29376;
  blend4 ~0.29652;
  current-only ~0.32291

So do not solve the Week-2 efficiency miss by aggressively ingesting two-game
current-only efficiency.

## 7. Existing higher-fidelity TE distribution evidence

Do not duplicate the production-order historical replay.

PR #549 / fold-safe WR-R15 + TE-R5P production-order replay:

- run `34722725629`
- compact artifact `10307242156`
- digest
  `sha256:d5a991bd76df5b053e6411e9873b12bdaada1458592c2586e3c1416c6fe37044`
- specialist treatment is upstream of joint MC
- canonical reproduction/conservation/fold-authority gates passed

TE receiving-yard rows extracted from its exact paired result:

### 2024

- n = **682**
- TE-R5P specialist MAE = **19.819**
- mean specialist model SD = **14.781**
- realized residual SD = **25.545**
- implied residual/model width ratio = **1.728**

### 2025

- n = **671**
- specialist MAE = **19.219**
- mean specialist model SD = **14.957**
- realized residual SD = **25.250**
- implied ratio = **1.688**

Thus TE-R5P receiving distributions remain materially under-wide after the
production-order specialist, and the width ratio is unusually stable across
the two seasons.

This does NOT authorize copying PR #548's generic rec_yards factor or widening
2026 from live W1/W2 residuals.

## 8. TE-R5P Receiving-Yards Width V2 — ACTIVE

Frozen plan:

`docs/research/TE_R5P_REC_YARDS_WIDTH_V2_PLAN.md`

Implementation:

`scripts/research/te_r5p_rec_yards_width_v2.py`

Focused mechanics tests:

`tests/test_te_r5p_rec_yards_width_v2.py`

Workflow:

`.github/workflows/te-r5p-rec-yards-width-v2.yml`

Current branch head at handoff write:

`051d97b6eea263aa710aea42327f38c2c76b29d1`

Canonical first run:

`35940487155` — check live status before acting.

V2 contract:

- reconstruct exact fold-safe PR #549 production-order specialist arrays;
- TE `rec_yards` only;
- fit 2024 width k -> blind-test 2025;
- fit 2025 width k -> blind-test 2024;
- one formula only:
  residual SD / mean within-row MC SD;
- width only, exact mean invariant;
- primary gates: CRPS + 80/90 coverage in both directions;
- historical-line Brier/log loss secondary after k frozen;
- ROI descriptive only;
- no sportsbook input fits k;
- no 2026 outcome fits k;
- no post-result k search.

If both blind directions qualify, the predeclared future-only factor is computed
from pooled 2024+2025 fold-safe specialist rows for a **separate** Week-3
production-integration validation. V2 itself does not mutate production.

Issue #535 V2 freeze/run checkpoint: `5805487318`.

## 9. RB sanctioned parallel lane — next after TE V2 is in flight/settled

Do not reopen M96 historical router tuning.

Strongest legitimate new-information lane:

**backfield teammate availability / injury-created vacancy propagation into
rushing opportunity**.

Why:

- production removes unavailable RBs but does not explicitly transfer their
  missing carries to successors;
- current-season RB rush share is the strongest state-persistence signal found;
- prior-week snap share is available pregame;
- M96A already proves low/high workload misses are opportunity-dominant;
- existing receiving vacancy work shows the availability signal is real but
  role-dependent.

Required design:

- pregame availability only;
- current/prior snap + rush-share state allowed;
- no target-game outcomes in features;
- no retrospective M96 threshold/router reopening;
- use genuinely prospective 2026 evidence or separately justified untouched
  new data;
- separate opportunity/carry transfer from YPC efficiency.

## 10. Other standing science conclusions

- QB pass yards is the strongest early live market. Freeze it and score
  prospectively; do not retune from two weeks.
- Probability/distribution overconfidence is real across positions, but fixes
  must remain authority-specific.
- Do not globally multiply all SDs from W1/W2.
- `rush_rec_yards` remains a separate low-mean/construction concern and should
  not be folded into the TE width study.
- raw `edge_pct` is not supported as a trustworthy confidence/staking ranker
  from W1/W2.
- no sportsbook line as upstream football input.
- no paid OddsAPI pull without explicit user approval.

## 11. Immediate continuation instructions

1. Verify run `35940487155`.
2. If mechanical failure, repair mechanics only; do not alter V2 science/gates.
3. If success, preserve full result + artifact/digest in repo and Issue #535.
4. If V2 qualifies, build the separate Week-3 integration validation before
   production mutation.
5. Begin/freeze RB teammate-vacancy rushing-opportunity V1 in parallel once the
   TE result is settled enough not to create competing degrees of freedom.
6. Keep this handoff and Issue #535 current after every substantive result.

Do not return to scoreboard cleanup unless the stopping-rule conditions in
Section 1 are met.
