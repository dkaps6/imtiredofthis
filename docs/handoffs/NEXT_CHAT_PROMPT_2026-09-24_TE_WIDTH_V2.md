Continue my existing NFL Stuff project from the exact canonical GitHub checkpoint.

Repo: `dkaps6/imtiredofthis`

GitHub is canonical over chat memory. Do NOT make me re-explain prior work, restart completed research, or repeat already-documented tests.

IMPORTANT MEMORY RULE:
Do NOT ingest the entire project history up front. Read only:
1. `AGENTS.md`
2. ONLY the top active checkpoint of `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-09-24_TE_WIDTH_V2_PROVIDER_DRIFT_CURRENT.md`
4. Issue #535 comments from `5805281169` onward, especially `5805404034`, `5805487318`, `5805942946`, and the newest elite handoff checkpoint
5. the live state of branch `research-te-live-entitlement-efficiency-v1`

That is enough context. Do NOT open older handoffs unless the current handoff explicitly directs you to one.

Do not read older handoffs unless the current handoff explicitly sends you there. Do not recursively summarize the full project history. I am trying to preserve conversation memory and want this chat to last.

VERIFY LIVE STATE BEFORE MUTATION.

Expected checkpoint:
- main at or after canonical handoff refresh `b1c6ed318520088006640ce3af7c954fac84a6f8` (VERIFY LIVE)
- PR #626 MERGED/CLOSED; do not reopen
- canonical W1/W2 production record fixed at 805 selected / 797 decided / 395-402 / -42.86u
- TE Live Entitlement vs Efficiency V1 COMPLETE/SUCCESS
- successful V1 run `35939588747`, artifact `10784323345`
- V1 result: current W1 2026 snaps improve Week-2 TE target-share accuracy, but efficiency/translation remains the larger live receiving-yard gap
- PR #627 already merged and prospectively activates strict-prior 2026 snaps for Week 3+; do not duplicate/refit it

ACTIVE SCIENCE LANE:
TE-R5P Receiving-Yards Width V2 on branch `research-te-live-entitlement-efficiency-v1`.

Expected branch head:
`839f7d206d47709ab8ed5576908dae205057be24`

Frozen plan:
`docs/research/TE_R5P_REC_YARDS_WIDTH_V2_PLAN.md`

Do NOT change the frozen V2 hypothesis/gates after seeing failures.

Important run history:
- run #1 `35940487155`: failed closed before science due provider-history row-universe drift
- run #2 `35944049950`: superseded diagnostic run
- run #3 `35944066618`, job `107458022324`: failed closed before science
- current rebuild = 51,232 rows vs frozen PR #549 = 51,197
- 35 provider-history extra rows are already disclosed/excluded
- BUT frozen-key `mc_proj` still drifts by up to 14.0206492813 yards
- exact error: `RuntimeError: mc_proj drift on frozen PR549 rows: 14.020649281259615`
- therefore the problem is exact historical-authority reproduction, NOT a Width V2 scientific failure

PR #549 preserved authority:
- run `34722725629`
- compact artifact `10307242156`
- digest `sha256:d5a991bd76df5b053e6411e9873b12bdaada1458592c2586e3c1416c6fe37044`
- compact artifact is still alive until 2026-09-26 and is precious
- expired raw-draw artifact `10306649017` must NOT be replaced by an approximation

NEXT ACTION:
1. Verify branch/run state.
2. Narrowly diagnose the frozen-row `mc_proj` drift by comparing current rebuild to frozen `full_stack_projection_trace_base.csv` on exact frozen keys.
3. Identify the first upstream component(s) that changed: rules plays/pass rate/target share/YPT/catch rate/context/provider/player-form/etc.
4. Search for any surviving exact copy/recovery of the PR #549 2,000-draw arrays before doing expensive rebuild work.
5. Do NOT loosen parity tolerances, accept today's rebuild as equivalent, use a Normal approximation, search k, or alter V2 gates.
6. If exact reproduction/recovery becomes another open-ended infrastructure chase, document/park Width V2 and pivot to the sanctioned RB teammate-availability / injury-created-vacancy rushing-opportunity lane. Games are imminent and we are not spending another day on plumbing.

SCIENTIFIC FACTS TO PRESERVE:
- TE Live V1 all-TE W2 target-share MAE 0.05395 -> 0.05270 with current snaps (+2.32%)
- selected TE target-share MAE 0.06950 -> 0.06753 (+2.84%)
- selected TE rec-yard MAE does not improve from entitlement-only counterfactual
- perfect efficiency recoverable error 5.68 yd vs perfect entitlement 4.11 yd
- historical TE-R1: targets 45.2%, YPR 29.5%, catch rate 25.3%
- current-only early TE efficiency is noisy; keep efficiency heavily shrunk
- historical TE-R5P specialist appears under-wide: 2024 k~1.728, 2025 k~1.688
- no global SD rescale
- QB pass yards stays frozen/prospective
- raw edge_pct is not trusted for staking/ranking
- rush+receiving remains a separate low-mean construction concern

STANDING PROHIBITIONS:
- do not reopen PR #625
- do not reopen PR #626
- do not duplicate PR #627
- do not rerun TE-R1
- do not reopen exposed retrospective M96 RB router variants
- no sportsbook line as upstream football input
- no paid OddsAPI pull without explicit approval
- do not redo Claude's original W1/W2 paid-board acquisition/backtest

WORK STYLE:
Work autonomously. Do not merely narrate. Close loops. Verify the physical GitHub remote after every write. Leave a historical trail in GitHub and Issue #535 for every hypothesis/test/result/failure. Distinguish mechanical failures from scientific failures. Never overwrite failed runs.

Pick up exactly from this checkpoint and continue as if you were the prior chat. Do not spend the opening turn re-summarizing everything back to me; verify the live state and start the next action.
