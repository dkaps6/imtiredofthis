# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack unchanged:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability + current-role Full Slate integration certification.  
**35-gate branch:** `ops-current-player-availability-35gate-cert-v1`.  
**Production has NOT been promoted.**

GitHub is canonical; chat memory is secondary. Preserve mechanical failures and first valid scientific/integration results. Never lower frozen gates, post-hoc route losing cohorts, allow sportsbook to define football, or mutate R26/R22/scientific model parameters without separately frozen authority.

## RB receiving-yard mean lane — CLOSED

Final R27D strict-prior YACOE residual result:
- branch `research-rb-r27d-yacoe-residual-v1`
- run/job `34436178615` / `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- 22/31 total gates; 4/13 scientific gates; vacancy RB1 B1 MAE `14.709395` -> C1 `14.710366`; 2023 `13.992139` -> `14.055947`; only 3/6 seasons improved.

No R27D integration/retuning/router. Reopen only for genuinely new pregame information.

## Availability authority

Frozen Full Slate integration plan:
- commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- blob `54eb4629c48062fcaef3153918b2069238584d0a`
- exactly 35 predeclared gates; all required before promotion.

Canonical T-75 first valid result: Run `34437715931`, Job `102746163583`:
- >75 min `NOT_YET_REQUIRED` and eligible;
- <=75 min with complete official pre-kickoff inactive sections `REQUIRED_AND_CERTIFIED`;
- <=75 min missing/incomplete official sections `REQUIRED_MISSING_FAIL_CLOSED` for that game;
- at/after kickoff `KICKED_OFF_LOCKED`.

Locked availability core blobs:
- Ourlads sidecar `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- resolver `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- NFL official inactives `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- T-75 validator `d67ae30ed6e837f62098671c499d05462fe9d837`
- reconciled active roles `b2d05882cada048337f1f5f8b8db8ec7f9eef001`.

## First mechanically valid no-odds availability candidate — SUCCESS

Branch/head `ops-current-player-availability-full-slate-v1` / `9800254f3ab208ab42501faac83a0d6e5fe3b93d`.

- run/job `34447900206` / `102776660124`
- artifact `10140425929`
- artifact name `current-player-availability-full-slate-candidate-v1`
- digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- 468 availability rows
- 1 definitive unavailable (TreVeyon Henderson, NE), 1 uncertain, 0 unknown
- 15 production-eligible games; NE-SEA withheld because already kicked off
- 437 production-eligible active-role rows across 30 teams
- sportsbook inputs to availability/opportunity `0`
- disposition `CURRENT_PLAYER_AVAILABILITY_NO_ODDS_FULL_SLATE_CANDIDATE_COMPLETED`
- production promoted `false`.

This satisfies gate 26 only; it is not the 35-gate result.

Preserved candidate plumbing failures remain canonical:
- Run1 `34439714153`, Job `102752015236`, Artifact `10137497418`, digest `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2` — Week1 TeamForm source selection.
- Run2 `34443710690`, Job `102763847787`, Artifact `10138897760`, digest `sha256:165c0e431e4771eba05472b17d6680457fedac37d5769bfc5c0b478aad830b1f` — PlayerForm history publication contract.

## Availability-aware current-team coverage seam

Initial frozen seam for full-universe and R26:
- plan commit `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- helper blob `77b591e431378ec984c51e8a032262e673d4c843`
- transformer blob `b64ec5ccd59728121a250433e40e77e3e1013a05`
- protected full-universe source `f8429ea5b6dd730f054460493facde4ab21b0998`
- protected R26 source `0c7528a3ca9e750d3b9ef2f08ef9721949b3e7fc`
- regression Run `34453027002`, Job `102792905910`, Artifact `10142304020`, digest `sha256:071f791c916d5c17c655b62d6858ca2adfbfd276009878281898608c2d9d3cc0`
- result commit `30558db704f81b6336514024e3ed0c5c75291fcc`
- disposition `ELIGIBLE_TEAM_COVERAGE_SEAM_REGRESSION_PASS`.

Semantics: legacy/no-explicit-availability mode still requires 32 teams; explicit `ACTIVE_ROLES_CSV` mode requires exactly the certified eligible team set. This changes coverage validation only, never model science.

## 35-gate certification implementation

Frozen implementation components on `ops-current-player-availability-35gate-cert-v1`:
- football-stack runner blob `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`
- fixture builder blob `dbe9f41e4d175b503bf0cd8caf51c1dc8d0da95d`
- gates 1-34 evaluator blob `3bc2bb390ab6762f1c20e775152812d3f8e9729e`
- gate35 finalizer blob `a5302186eadb748863c70b175421d064885dde60`
- workflow after Run1 repair blob `130ed21db8d23575a38e84f1cfe0cc865498f001`.

The baseline certification consumes immutable candidate Artifact `10140425929`; it does NOT regenerate the live snapshot. It creates synthetic football-only identity/market-key lookup rows with NO book/line/odds/market probability and exercises the actual production football wrapper: M38 -> TE-R5P -> WR-R15 -> QB C2 -> R22 -> R26 -> outer P3 rush+receiving conservation. Fixtures run in isolated worktrees: RB1 OUT; QB1 inactive; WR2+/TE1 unavailable.

## 35-gate Run1 — PRESERVED MECHANICAL FAILURE / NO DECISION

Original lock head `eb7f37fa6e2a9019ed0ec8f3bbe6fe73be202699`.

- run `34459655725`
- job `102814178762`
- failure before any gate 1-34 evaluation
- exact exception: `RuntimeError: QB C2 starter authority must cover 32 teams, got rows=30`
- disposition `MECHANICAL_FAILURE_NO_DECISION`
- record `docs/operations/CURRENT_PLAYER_AVAILABILITY_35GATE_RUN1_MECHANICAL_FAILURE_RECORD.md`.

Run1 had already passed frozen-blob/protected-model verification, immutable candidate digest verification, 30-team seam, all fixture construction, and M38/TE-R5P/WR-R15 baseline materialization/conservation. The failure was another legacy current-slate coverage guard in QB C2 starter-audit validation, not a QB model/starter-selection failure and not a frozen-gate result.

### Frozen QB C2 minimum repair

- repair plan commit `760fbcafec3df636b671b49c16a6d1d04c134162`
- repair lock head `3825235af0442e7868f474e905ea11ad1214432e`
- protected QB C2 source blob `7b677470b27b6776055c75c924a0ddf22d724a44`
- QB coverage transformer blob `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- dedicated regression Run `34460044387`, Job `102815456345` — SUCCESS.

Regression proved legacy mode still requires exactly 32, explicit availability mode accepts exactly certified 30, and both missing/extra teams fail. The transformer changes only the final current-team coverage validation in `annotate_primary_qbs()`. Starter ranking/selection, official-authority priority, Ourlads fallback, C2 parameters/distributions and the existing 32-team state-context SOURCE integrity assertion remain unchanged.

## CURRENT LIVE STATE — 35-gate Run2 in progress

Certification was relocked after the minimum mechanical repair:
- relock head `1d38953995842aa0236edda7124a79665d0b8628`
- Run2 `34460227422`
- Job `102816052394`
- current status at this handoff update: `in_progress`.

Run2 remains eligible to become the **first valid immutable 35-gate result** because Run1 evaluated zero gates. The 35 frozen gate definitions have not changed.

### Exact next action

1. Inspect Run `34460227422` / Job `102816052394` first.
2. If it fails before `Evaluate frozen gates 1 through 34`, preserve as `MECHANICAL_FAILURE_NO_DECISION`, diagnose exact exception, freeze only the minimum value-neutral plumbing repair, regression-test it, relock, retry.
3. If it reaches gate evaluation, the first valid gate disposition is immutable: any failed gate = `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_FAIL_NO_PROMOTION`; exact 35/35 = `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`.
4. A 35/35 PASS does NOT itself mutate production. Freeze a separate promotion implementation, wire availability/current-role seams into production, then execute post-promotion Full Slate verification before declaring the lane complete.
5. Update this handoff with exact run/job/artifact/digest/disposition at every material checkpoint.

## Remaining roadmap after availability

1. Finish current roster / late-week availability integration and production verification — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles, building on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution while preserving M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package and prospective grading.
