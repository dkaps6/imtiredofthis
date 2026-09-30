# NFL HANDOFF — 2026-09-29 — RESEARCH LEAD / WR-CB SOURCE / CLV CURRENT

GitHub is canonical. This handoff supersedes earlier 2026-09-29 Week-3 postmortem checkpoints for immediate continuation.

## 1. Canonical production state

Current canonical `main`:

`deea0f68202c9ab6e85fa616f7444d0008ee10af`

This includes merged PR **#664 — Production: retire unsupported legacy WR coverage heuristic**.

Repo CI for PR #664:
- run `36648664504`
- **SUCCESS**

The separate preserved replay run `36648664345` failed before football execution because archived source artifact `run_35282021679` was expired/not found. Do **not** treat that as a scientific/model failure.

No OddsAPI spend occurred for this work.

## 2. Week 3 / Weeks 1-3 science is closed

Do not rerun completed settlement/postmortem work unless a concrete integrity defect is found.

Canonical cumulative result:
- **629-611**
- **50.7%**
- **-40.72u**

Frozen/closed dispositions:
- no simple position/market/side slice survived game-cluster-aware BH-FDR;
- projection authority = `MIXED_BY_MARKET_OR_POSITION`;
- RB Vacancy W3 = `WEEK3_OBSERVATIONAL_MIXED`;
- Public Intent W3 = `WEEK3_PUBLIC_INTENT_DIRECTIONALLY_INFORMATIVE`, observational only;
- Receiving Rule Semantics W3 = no cell promoted;
- Availability -> Opportunity = `NO_VALID_POSTGAME_GRADING_CONTRACT_DO_NOT_GRADE_POST_HOC`; structural rule-order gap remains confirmed;
- RB-PD2 W3 = Observation #1 only, **46/400 rows and 1/8 weeks**, `NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`;
- GSIS Week-3 snapshot remains private baseline-only prospective evidence.

Consolidated matrix:
`docs/research/WEEK3_PROSPECTIVE_DISPOSITION_MATRIX_V1.md`

Do not rerun:
- Weeks 1-3 settlement/postmortem;
- RB Vacancy W3 grade;
- Public Intent W3 grade;
- Receiving Rule Semantics W3 grade;
- Availability -> Opportunity W3;
- RB-PD2 W3 Observation #1.

## 3. Projection-authority science added after the postmortem

Frozen diagnostic:
`docs/research/WEEKS1_3_PROJECTION_AUTHORITY_LINE_CONFLICT_V1_PLAN.md`

Result:
`NO_CLEAR_CURRENT_SEASON_AUTHORITY_LINE_CONFLICT_SIGNAL`

Crossing the sportsbook line itself was not a supported failure state:
- crossed n=119 / 43 games;
- crossed win rate 50.42%;
- non-crossed 50.76%;
- clustered uncertainty crossed zero.

Do **not** create a crossed-line exclusion rule.

A secondary predeclared state pattern was discovery-only:
- `SAME_SIDE_STRENGTHENED`: n=250, 44.0%, -38.15u;
- `SAME_SIDE_WEAKENED`: n=471, 55.20%, +24.44u.

Because Weeks 1-3 outcomes were already known, this cannot promote anything.

Forward contract frozen for Week 4+:
`docs/research/WEEK4_PLUS_PROJECTION_AUTHORITY_MOVE_DIRECTION_FORWARD_V1_PLAN.md`

Support floor:
- >=8 eligible future weeks;
- >=400 strengthened rows;
- >=400 weakened rows.

Until then:
`FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT`

## 4. Specialist RNG repair remains open mechanical work

Original failed candidate run:
`36504441918`

Diagnosis:
- TE/WR protected arrays passed;
- intended specialist receiving arrays changed;
- QB-C2 scope/mean invariants passed;
- paid artifact/source authority passed;
- exact frozen paid-board fingerprint failed.

Concrete defect found:
- frozen research seed labels: `TE_CHANGED_ROOM` / `WR_CHANGED_ROOM`;
- production port had renamed them `TE_R5P_ROOM` / `WR_R15_ROOM`;
- room label participates in deterministic RNG key, so rename changed the finite sample.

Narrow patch:
- branch `repair-specialist-rng-isolation-v1`
- commit `30333ec2b853d5ba7bd420b74d63c53f097b5fff`

Do **not** weaken the frozen equivalence gate.
Do **not** call PASS until exact post-patch frozen fingerprint validation is observed.

## 5. RB route-volume mean-information lane

Source-readiness result:
`docs/research/RB_ROUTE_VOLUME_SOURCE_READINESS_V1.md`

Disposition:
`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`

Important source semantics:
- current/live 2026 route-volume sources exist;
- HeatRadar and StatRankings were identified as current candidates;
- nflverse `route` is **not** equivalent to total player routes run; it describes the route of the primary receiver on a play and must not be used as a fake historical total-route bridge.

Prospective capture plan:
`docs/research/RB_ROUTE_VOLUME_PROSPECTIVE_CAPTURE_V1_PLAN.md`

Implemented on `research-week3-postmortem-execution-v1`:
- immutable capture materializer;
- strict missing/identity/team-change fail-close;
- cumulative-delta derivation;
- cross-source parity tests.

This is research-only. No route-volume production feature is promoted.

## 6. WR-CB / coverage: authoritative current state

This section is critical because Issue #535 contains an interim contradictory comment. Trust **current main** and this handoff.

### Direct player-level WR↔CB assignment

Production already fails closed when direct matchup rows are absent:
- no fake `primary_cb` assignment;
- direct matchup availability is gated off when source is unavailable;
- Week-3 preserved production artifact had **0** frozen `primary_cb` assignments.

Disposition:
`PLAYER_LEVEL_WR_CB_ASSIGNMENT_PRODUCTION_UNAVAILABLE_KEEP_GATED_OFF`

Do not pay for a WR-CB source.
Do not reconstruct true assignment from nflverse/on-field defenders.
Do not silently promote editorial matchup grades.

### Legacy team-level `coverage_penalty()` heuristic

Outcome-free Week-3 mechanical audit found the grandfathered rule was materially active:
- 161 WR rows;
- 131 affected = 81.4%;
- 27/30 teams affected;
- zero primary-CB assignments;
- all active adjustments came from generic heavy-zone logic;
- affected WR entitlement×YPT kernel changed ~6.4% median in remove-only replay;
- conservation spillover moved TE/RB target pools ~2.9%.

The rule was not valid player-level matchup science and the closest older 2025 ablation was near-neutral.

User authorized removal rather than paying for unsupported WR-CB data.

PR **#664** merged:
- main `deea0f68202c9ab6e85fa616f7444d0008ee10af`
- `coverage_penalty()` removed from `rules_v2.py`;
- production call removed from `simulation_rules.apply_rules_to_metrics()`;
- static 0.92 / 0.94 / 1.06 / 1.04 multipliers are gone;
- no replacement coefficient;
- regression test verifies `primary_cb` no longer alters WR `rules_tgt_share` / `rules_ypt` through this path.

**Do not restore this heuristic.**

The older forward remove-only shadow plan is superseded by the merged production retirement and must not be executed as if the heuristic is still active.

### Free historical WR-CB source correction

After the initial source audit, public FantasyAlarm archived WR/CB report pages were found spanning at least 2021-2026.

This corrects the blanket statement "no external free historical WR-CB archive exists."

Research branch:
`research-wr-cb-free-archive-v1`

Draft PR:
**#665 — Research: recover free historical WR-CB assignment archive**

Purpose:
- acquire factual WR↔CB pairing + alignment from archived public pages;
- preserve publication timestamp / URL / raw label / identity;
- keep editorial Safe/Moderate/Risky grades **NOT_MODEL_ELIGIBLE**;
- fail closed on missing/ambiguous identity;
- 16 verified seed pages spanning 2021-2026.

This is **not yet production-grade or model-qualified**.

Exact next source question:
- does the archive have sufficient completeness, temporal integrity, alignment semantics, identity stability, and week coverage to support a strict-prior historical test?

Only if source audit clears may the already source-blocked M82/M83 `TOP_WEAPON_ESCAPE_HATCH` concept be tested:
- learn CB effect from strictly prior observed WR-CB assignments and our historical outcomes;
- confirm on untouched later season;
- no paid grade required;
- no editorial matchup score as a feature.

### Stale Issue comment warning

Issue #535 comment `5901927972` contains a stale statement saying the old team-level `coverage_penalty()` remains active.

That statement is superseded by:
- merged PR #664;
- current main `deea0f68...`;
- direct inspection of main, where the function/call is retired.

Do not use `5901927972` to restore or assume the heuristic is live.

## 7. CLV / market-snapshot architecture

The existing latest-row market ledger overwrites earlier captures, so by itself it cannot measure real closing-line movement.

Frozen architecture:
- `docs/research/MARKET_SNAPSHOT_HISTORY_CLV_CAPTURE_V1_PLAN.md`
- `docs/research/MARKET_SNAPSHOT_HISTORY_CLV_CAPTURE_V1_AMENDMENT_1.md`

Research branch:
`research-market-snapshot-history-v1`

Draft PR:
**#663 — Research: preserve immutable market snapshots for CLV**

Core rules:
- append-only snapshot history;
- exact source `props_raw.csv` `fetched_at` and `commence_time`;
- same-book comparison;
- T-30 or closer pre-kickoff only may be labeled `VALID_T30_CLOSE`;
- T30-T60 = late-market movement;
- >T60 = pregame movement only;
- no cross-book substitution;
- no paid odds pull solely to obtain a close.

Two existing archive provenance problems were also found:
- archive workflow must stamp triggering Full Slate SHA, not archive-workflow SHA;
- season/week must come from / agree with the source priced board, not current checkout/runtime week.

PR #663 is still **draft/open and not green yet**.
Last visible CI lineage had:
- Repo CI failure in immutable manifest idempotence handling;
- replay failure at old artifact download before football execution because `run_35282021679` expired.

Do not merge #663 until latest head is inspected, rebased onto current main if necessary, focused tests are green, and Repo CI is green.
Do not treat the expired-artifact replay failure as scientific evidence.

## 8. Claude collaboration state

Issue #535 is the shared lab notebook.

Research split previously assigned:
- GPT-5.6: RB route-volume / source architecture;
- Claude: opponent-injury propagation into matchup context;
- cross-audit each other's source conclusions before model tests.

At this handoff, there is **no authoritative Claude opponent-injury result posted in Issue #535 after the current GPT checkpoints**.

Next chat must read any newer Issue #535 comments before assuming Claude is still pending.

Claude cross-audit requests still outstanding:
- challenge route-volume historical/live parity;
- challenge FantasyAlarm archive completeness / pairing semantics;
- independently assess whether alignment pairings are stable enough for `TOP_WEAPON_ESCAPE_HATCH`;
- keep opponent-injury lane separate;
- include PR #663 market-snapshot architecture in review if relevant.

## 9. Active PR / branch map

Canonical main:
- `deea0f68202c9ab6e85fa616f7444d0008ee10af`

Open research:
- PR #663 / `research-market-snapshot-history-v1` — CLV snapshot history, draft, not green yet;
- PR #665 / `research-wr-cb-free-archive-v1` — free historical WR-CB archive source audit, draft/open;
- `research-week3-postmortem-execution-v1` — postmortem + route-volume + prospective research docs;
- `repair-specialist-rng-isolation-v1` — RNG exact-parity repair.

Merged:
- PR #664 — legacy coverage heuristic retirement.

PR #662 GSIS:
- still treat as research-only point-in-time/private snapshot work unless live GitHub says otherwise;
- raw GSIS remains private;
- temporal predictive use requires repeated immutable snapshots.

## 10. Exact next execution order

Do these in order unless newer GitHub state supersedes them:

1. Query live `main`, PRs #663/#665/#662, `repair-specialist-rng-isolation-v1`, and current Actions before editing anything.
2. Read Issue #535 from comment `5899500522` onward, prioritizing:
   - `5899645764`
   - `5899710345`
   - `5899966307`
   - `5900122307`
   - `5901422664`
   - `5901433362`
   - `5901608387`
   - any comments newer than this handoff.
3. Finish **PR #663** mechanically:
   - inspect actual latest head/logs;
   - fix only current test/integration defects;
   - rebase/current-main reconcile;
   - do not change CLV science contract;
   - do not spend OddsAPI credits;
   - merge only when green.
4. Continue **PR #665 source audit**:
   - validate the 16 seed pages live;
   - quantify season/week coverage and parser yield;
   - verify publication timing;
   - verify alignment/pairing semantics;
   - report unresolved identity rate;
   - no outcome model until source-readiness clears;
   - no editorial grades as features.
5. Check Issue #535 for Claude's opponent-injury result. If present, cross-audit it before any model test.
6. Verify the specialist RNG patch against the exact frozen fingerprint. Do not weaken the gate.
7. Preserve Week-4+ forward captures:
   - projection authority move-direction;
   - RB-PD2;
   - RB route-volume;
   - market snapshots/CLV only from already-acquired runs.
8. Only after source / forward gates clear, design new predictive tests. Do not recycle Weeks 1-3 outcomes.

## 11. Hard do-not-do list

- no paid OddsAPI run without explicit user approval;
- no paid WR-CB source;
- no restoring `coverage_penalty()`;
- no fake WR-CB assignment from nflverse;
- no editorial WR-CB rating as model input;
- no Week-3 post-hoc rescue;
- no rerun of closed Week-3 lanes;
- no scalar probability recalibration rescue;
- no threshold / position / market carveout after seeing the same outcomes;
- no weakening RNG exact-parity gate;
- no calling >T60 market movement "CLV";
- no raw GSIS publication.

## 12. Memory-efficient next-chat start

Read only:
1. `AGENTS.md`
2. newest top checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. Issue #535 from `5899500522` onward, especially the comments listed above
5. live main / PR #663 / PR #665 / PR #662 / RNG repair branch / Actions

Do **not** recursively read old handoffs.
