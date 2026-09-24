# NFL HANDOFF — 2026-09-24 — TE LIVE SCIENCE / WIDTH V2 PROVIDER-DRIFT CHECKPOINT

GitHub is canonical. This handoff is designed to let the next GPT-5.6 chat continue without re-reading the entire project history.

## Minimal read order

Read only these first:

1. `AGENTS.md`
2. the TOP active checkpoint of `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. Issue #535 comments starting at `5805281169`, especially `5805404034`, `5805487318`, and `5805942946`
5. live branch `research-te-live-entitlement-efficiency-v1`

Do NOT ingest every older handoff unless this file explicitly points you there. This is intentional to preserve chat/context memory.

---

## 1. User priority / operating philosophy

The user explicitly reset the project priority after ~48 hours of scoreboard hardening:

> Weekly backtesting matters because it should teach the model what current NFL offenses, defenses, player roles, usage, injuries, and decisions are doing so the NEXT slate improves.

The operating learning loop is:

`pregame projection -> outcome -> diagnose opportunity/role vs efficiency vs distribution -> compare with historically persistent state -> freeze one leakage-safe hypothesis -> test -> promote only if supported`.

Do not return to infrastructure/scoreboard work unless a defect can materially:
1. change which wagers belong in the canonical record;
2. change settlement/W-L/units; or
3. make the canonical replay fail/give a wrong answer.

Peripheral polish is deferred. Games are imminent; prioritize real model science.

Every substantive hypothesis, test, failure, result, rejection, and promotion decision must be left in GitHub and Issue #535.

---

## 2. PR #626 / scoreboard closure — DONE

PR #626 is MERGED/CLOSED.

- final PR head: `42c66b690e5b318ae368945bacd6c7762dc2c391`
- merge commit: `69e9c6211710c60de9aaf07d06aca6cafeff5ccf`
- exact-head Repo CI #1124: SUCCESS
- W1/W2 Full Board #100: SUCCESS
- preserved paid-artifact replay #457: SUCCESS
- unresolved correctness/integrity threads at merge: 0
- football-model science changed by #626: 0
- paid OddsAPI pull: 0

Canonical 2026 W1/W2 production record:

- selected settlement rows: **805**
- decided bets: **797**
- record: **395-402**
- units: **-42.86u**
- QB pass yards: **36-16 / +15.99u**
- TE overall: **58-75 / -21.45u**
- TE receiving yards: **28-41 / -16.09u**
- TE receptions: **30-34 / -5.37u**

This record is now a measurement authority, not the active research objective.

Issue #535 closure checkpoint: `5805281169`.

Do not reopen PR #626.

---

## 3. Current-season learning authority — already established

Current-Season State Persistence V1:

- run `35741758765`
- artifact `10699781744`
- digest `sha256:2d81bcb5222136ae812575b15a0d79952d27a03e68034cd312aa0f927503868a`
- 41,745 player-metric rows
- sportsbook inputs: 0
- 2026 outcomes used in fit/test: 0
- production mutation: 0

Untouched 2025 replication improved 9/10 production-aligned metrics.

Key signals:
- RB rush-share MAE: `0.1486 -> 0.1166` (**21.53% improvement**)
- TE target-share MAE: `0.05153 -> 0.04553` (**11.66%**)
- WR target-share MAE: `0.06773 -> 0.06144` (**9.30%**)
- QB YPA full-season: `1.7386 -> 1.6548` (**4.82%**)
- RB YPC did not improve.

Standing interpretation:

> update opportunity/role quickly; keep early efficiency much more heavily shrunk toward history.

PR #627 is already merged and activates strict-prior 2026 snap participation for WR-R15 / TE-R5P beginning Week 3 after a schedule-aware freshness gate. Do not duplicate/refit it.

---

## 4. Historical TE mechanism authority — do not rerun

TE-R1 recovered authority:

- run `34123413402`
- artifact `10019112418`
- rows: 6,371
- disposition: `TE_MECHANISM_DECOMPOSITION_ACTIONABLE`

Shapley error-mass attribution:
- TARGETS: **45.2%**
- YPR: **29.5%**
- CATCH_RATE: **25.3%**

Targets are the largest single mechanism, but combined efficiency (YPR + catch rate) is ~55%.

Do not rerun TE-R1.

---

## 5. TE Live Entitlement vs Efficiency V1 — COMPLETE / SUCCESS

Branch:
`research-te-live-entitlement-efficiency-v1`

Frozen plan:
`docs/research/TE_LIVE_ENTITLEMENT_EFFICIENCY_V1_PLAN.md`

Result:
`docs/research/TE_LIVE_ENTITLEMENT_EFFICIENCY_V1_RESULT.md`

Canonical successful execution:
- scientific head `8018f78bd305faf5592ac05c9c7a6838b5bff161`
- run `35939588747`
- job `107444309642`
- artifact `10784323345`
- digest `sha256:50fcbe73794dd6111480b9ac0b3e3286d57514515bbe189463571552c8975409`

Exact paid Week-2 football origin:
- run `35282021679`
- source SHA `c6ec55be70d6e05bbd1dbae83d7d5c86ac8aa00a`
- artifact `10523345092`
- digest `sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c`

Run #1 `35939373232` failed before science output because sportsbook event hashes were incorrectly treated as football game IDs. That was repaired without changing the hypothesis/gates: join uses team + deterministic suffix-normalized player identity and fails closed on ambiguity.

### V1 result

Disposition:
`CURRENT_SNAP_ENTITLEMENT_SIGNAL_SUPPORTED_EFFICIENCY_REMAINS_PRIMARY_LIVE_GAP`

All-TE Week-2 exact-origin cohort:
- matched rows: **72**
- production target-share MAE: **0.05395**
- W1-2026-snap counterfactual target-share MAE: **0.05270**
- relative improvement: **2.32%**
- TE-room-share MAE: **0.23389 -> 0.22923**
- worst production target-share-error quartile: **0.12149 -> 0.11364**
- all frozen support gates PASS
- TE-pool conservation exact
- sportsbook inputs in football candidate: 0

Canonical selected Week-2 TE receiving-yard cohort:
- rows: **34**
- record: **13-21**
- target-share MAE: **0.06950 -> 0.06753** with current snaps (**2.84% better**)
- production rec-yard MAE: **20.87 yd**
- current-snap-only entitlement counterfactual MAE: **21.00 yd**
- raw MC MAE: **20.63 yd**
- final blend minus MC MAE: **+0.24 yd** (final slightly worse)
- perfect target-entitlement recoverable error: **4.11 yd**
- perfect realized-efficiency recoverable error on nonzero-target rows: **5.68 yd**
- among 29 nonzero-target rows: efficiency larger component in **19**, entitlement in **10**
- zero-target rows: **5**

Interpretation:
1. current snaps genuinely improve TE opportunity/entitlement;
2. PR #627's Week-3 continuation is supported;
3. entitlement improvement alone does not fix TE rec yards;
4. **efficiency / yardage translation is the larger remaining live TE mean problem**;
5. downstream distribution/ensemble is also suspicious because raw MC slightly beat final blend;
6. do NOT refit entitlement from two live weeks.

Issue #535 result checkpoint: `5805404034`.

---

## 6. Early-season TE efficiency conclusion

Do not aggressively learn current-only TE efficiency from two games.

Historical exact-two-prior-game analogue shows:
- TE target share benefits from blended current state;
- TE YPT current-only is materially noisier than historical/blended state;
- TE catch rate current-only is also noisier.

Approximate authority values already documented:
- TE target share: prior MAE ~0.04353 -> blend4 ~**0.03971**
- TE YPT: prior ~4.6436 -> blend4 **~4.5101**, current-only ~5.0195
- TE receptions/target: prior ~0.29376, blend4 ~0.29652, current-only ~0.32291

Therefore: improve opportunity from current usage, but efficiency needs heavier historical shrinkage / better football context rather than raw two-game recency.

---

## 7. TE-R5P receiving-width evidence already observed

PR #549 fold-safe production-order replay authority:

- run `34722725629`
- compact artifact `10307242156`
- artifact name `wr-te-production-order-historical-replay-v1`
- digest `sha256:d5a991bd76df5b053e6411e9873b12bdaada1458592c2586e3c1416c6fe37044`
- source branch `research-wr-te-production-order-historical-replay-v1`
- source SHA `f04a8a775f4a56fe282cb292f6a52bd509bc8f24`

The original large raw-distribution artifact:
- artifact `10306649017`
- `wr-te-production-order-historical-distributions-v1`
- digest `sha256:987b8447c06e1f28b95ba4a19e80b2a6ef4c2af28e707ce4884adaedbe3e5361`
- is **expired**.

Exact compact artifact still contains:
- `component_predictions_2024_wrte.csv`
- `component_predictions_2025_wrte.csv`
- `full_stack_projection_trace_base.csv`
- `full_stack_projection_trace_wrte.csv`
- per-week distribution metadata
- `grade/distribution_alignment_audit.csv`
- `result/wrte_authorized_paired_rows.csv`
- other grade/result summaries

`result/wrte_authorized_paired_rows.csv` includes row-level:
`proj_specialist`, `model_sd_specialist`, `actual`, baseline equivalents, probabilities, and row identity.

Historical TE rec_yards specialist summary:
- 2024 n=**682**, MAE **19.819**, mean model SD **14.781**, residual SD **25.545**, implied k **1.728**
- 2025 n=**671**, MAE **19.219**, mean model SD **14.957**, residual SD **25.250**, implied k **1.688**

This stable, TE-specific under-width motivated Width V2. Do not apply these live as a global SD rescale.

---

## 8. TE-R5P Receiving-Yards Width V2 — ACTIVE BUT BLOCKED BEFORE SCIENCE

Frozen plan:
`docs/research/TE_R5P_REC_YARDS_WIDTH_V2_PLAN.md`

Implementation:
`scripts/research/te_r5p_rec_yards_width_v2.py`

Focused tests:
`tests/test_te_r5p_rec_yards_width_v2.py`

Workflow:
`.github/workflows/te-r5p-rec-yards-width-v2.yml`

Current research branch:
`research-te-live-entitlement-efficiency-v1`

**Physical branch head at this handoff:**
`839f7d206d47709ab8ed5576908dae205057be24`

### Frozen V2 scientific contract — do not change after seeing failures

Question:
If one TE rec-yard width multiplier is fit from one historical season using football outcomes only, does it improve the other season blindly?

Directions:
- fit 2024 k -> blind-test 2025
- fit 2025 k -> blind-test 2024

Construction:
- exact PR #549 TE-R5P specialist arrays;
- mean-align to frozen final football mean;
- row SD from exact 2,000 draws;
- `k = SD(actual - final_projection) / mean(row_MC_SD)`;
- widen around same mean;
- no factor search.

Primary gates both directions:
- point MAE invariant <=1e-10
- max mean shift <=1e-8
- CRPS strictly improves
- 80% coverage absolute gap improves
- 90% coverage absolute gap improves
- pooled downstream Brier/log-loss non-worse after football-only k is frozen
- no sportsbook input fits k
- all PR #549 replay/fold/conservation gates pass

No 2026 outcomes fit k. No global factor. No post-result k search.

### V2 run history

**Run #1 `35940487155`**
- FAILED CLOSED before science output.
- current 2024 historical provider rebuild had 7 additional non-QB player rows / 35 market rows compared with frozen PR #549 authority.
- 2025 row counts matched.
- This is provider-history drift, not a width-candidate result.

**Run #2 `35944049950`**
- triggered by an intermediate diagnostic patch;
- superseded;
- do not use scientifically.

**Run #3 `35944066618`**
- branch head `839f7d206d47709ab8ed5576908dae205057be24`
- job `107458022324`
- FAILED CLOSED before Width V2 science output.
- mechanics/tests PASS.
- frozen authority downloads PASS.
- independent historical input build PASS.
- baseline component build reaches reproduction gate.
- current rebuild rows: **51,232**
- frozen PR #549 baseline rows: **51,197**
- provider-history extras: **35** rows; these are disclosed/excluded from frozen authority.
- BUT on the frozen key intersection, `mc_proj` still drifts by up to:
  **14.0206492813 yards**.
- exact failure:
  `RuntimeError: mc_proj drift on frozen PR549 rows: 14.020649281259615`
- therefore the earlier theory "only extra provider-history rows explain the drift" is FALSE.
- specialist reconstruction, V2 grading, and result upload were skipped.
- **No V2 scientific conclusion exists yet.**

Issue #535 checkpoint that described run #3 as in-progress: `5805942946`.
This handoff supersedes that status: run #3 is now confirmed FAILED.

### What this failure means

The active problem is not Width V2 math. It is reproducing the exact frozen PR #549 football distribution authority after historical provider/source drift.

Do NOT:
- loosen parity tolerance;
- accept today's `mc_proj` as equivalent;
- refit k on a changed cohort;
- change V2 gates;
- treat the failure as evidence against widening;
- spend hours on unrelated scorekeeping.

---

## 9. Exact next action for the next chat

First verify branch head and run state live. If still as above:

### A. Diagnose frozen-row `mc_proj` drift, narrowly

Compare current rebuild vs frozen `full_stack_projection_trace_base.csv` on the exact frozen keys.

Rank the first upstream component fields that differ on the rows with largest `mc_proj` delta, especially:
- `rules_plays_est`
- `rules_pass_rate`
- `rules_tgt_share`
- `rules_ypt`
- `rules_catch_rate`
- relevant context availability flags
- team/game identity
- prior/player-form fields feeding MC
- any provider-source rows that changed historically

Goal: identify whether drift comes from:
1. provider historical data mutation;
2. code drift since PR #549;
3. changed row-universe affecting team allocation/conservation; or
4. another deterministic source seam.

Do not rerun V2 unchanged until this is understood.

### B. Prefer exact frozen authority over live-history reconstruction where scientifically valid

The compact PR #549 artifact is still alive until 2026-09-26 and must be treated as precious.

It already contains exact frozen projections, SDs, actuals, keys, and metadata.

However the V2 frozen plan requires CRPS/coverage from exact 2,000-draw arrays. The original raw-array artifact `10306649017` is expired.

Therefore:
1. search first for another surviving copy/recovery of the PR #549 exact distribution arrays before rebuilding them;
2. if exact draws can be recovered from another artifact/run, use them and verify digest/lineage;
3. otherwise the deterministic replay must reproduce frozen PR #549 before V2 proceeds.

Do not substitute a Normal approximation or summary-SD-only reconstruction; the frozen plan explicitly forbids that.

### C. Time-box artifact archaeology

The user has games imminent and does not want another multi-day plumbing loop.

If exact PR #549 draw recovery/reproduction becomes an open-ended infrastructure chase, document the blocker and pivot to the next football-science lane rather than burning the day.

The next sanctioned lane is RB teammate availability / injury-created vacancy propagation into rushing opportunity.

TE mean-efficiency/translation research is also legitimate, but do not create competing degrees of freedom with Width V2 unless Width V2 is explicitly parked/blocked.

---

## 10. RB sanctioned next lane

Do NOT reopen exposed M96 retrospective router threshold/feature variants.

Strongest sanctioned new-information hypothesis:

**backfield teammate availability / injury-created vacancy -> successor rushing opportunity / carry transfer**

Why:
- current-season RB rush share is the strongest state-persistence signal;
- production removes unavailable RBs but does not explicitly transfer missing carries;
- prior-week snap share is available pregame;
- M96A already proves low/high workload errors are opportunity-dominant;
- this uses genuinely current/prospective 2026 information rather than retuning exposed 2025 router outcomes.

Required design:
- pregame availability only;
- strict-prior snap/rush-share state;
- target-game outcomes only as labels;
- separate opportunity/carries from YPC efficiency;
- no sportsbook line upstream;
- no retrospective M96 router rescue.

Standing M96 prohibition remains in `CURRENT_NFL_RESEARCH_HANDOFF.md`.

---

## 11. Other standing conclusions / anti-retest rules

- QB pass yards: strongest early live market; freeze and score prospectively.
- Do not retune QB from two weeks.
- Probability/distribution overconfidence is real, but fixes must be authority-specific.
- No global two-week SD rescale.
- `rush_rec_yards` remains a separate low-mean/construction problem.
- raw `edge_pct` is not supported as a trustworthy staking/ranking variable.
- no sportsbook line as upstream football feature.
- no paid OddsAPI pull without explicit user approval.
- PR #625 is merged/finished; do not reopen.
- PR #627 is merged/finished; do not duplicate.
- PR #626 is merged/finished; do not reopen.
- do not rerun TE-R1.
- do not redo Claude's original W1/W2 paid-board acquisition/backtest.
- GitHub remote SHA is authoritative; never trust local-only claimed commits.

---

## 12. Historical trail / handoff discipline

After every substantive step:
- commit plan/result docs;
- post Issue #535 checkpoint with run/job/artifact/digest;
- distinguish mechanical failures from scientific failures;
- never overwrite failed runs;
- verify physical remote branch head after writes;
- update this handoff when the active scientific state materially changes.

The next chat should work from this checkpoint directly, not re-derive the prior 48 hours.
