# CURRENT NFL RESEARCH HANDOFF — READ FIRST

GitHub is canonical; chat memory is secondary.

## ACTIVE RESEARCH CHECKPOINT — 2026-09-15

The user's explicit current priority is **WR receiving yards**. RB is parked. Do not reopen paid Full Slate work.

Read in this order:

1. `AGENTS.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-15_WR_PHASE4C_STAGE1_CLOSED_CURRENT.md`
3. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_RESULT.md`
4. `docs/research/WR_PHASE4C_STAGE1_RESULT_INDEPENDENT_AUDIT.md`
5. `docs/handoffs/NFL_HANDOFF_2026-09-14_WR_RECEIVING_YARDS_EVIDENCE.md`
6. latest GPT-5.6 / Claude checkpoints in GitHub Issue #535, especially Claude Stage-1 audit comment `5688400426`, GPT post-Stage1 frontier-audit request `5688480341`, and any Claude response after it
7. `NFL_MASTER_CONTINUITY_RECORD.md` only if older lineage is needed.

### Stage-1 final state

Active branch: `research-wr-phase4c-stage1-pass-volume-predictor-v1`.

WR Phase4C Stage-1 blind outcomes were exposed **exactly once**.

Canonical outcome lineage:

- one-shot commit `8b49e35f194f8dfb6538be7148590370987f8534`
- run `35025966255` — SUCCESS
- job `104572956822` — SUCCESS
- artifact `10419147232`
- digest `sha256:09722be047456271ab2d73132cdd9c7a46dd0e771558314aba49992e096e407d`
- upstream Phase4B artifact `10404525877`
- upstream digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`

Frozen disposition:

`NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT`

Frozen gates:

- coverage_integrity PASS
- script_blind_skill PASS
- C_gt_B_material FAIL
- C_gt_A_material FAIL
- no_rmse_bias_tradeoff PASS
- wr_room_translation_positive FAIL
- tail_mechanism_alignment FAIL

Claude independently returned `STAGE1_RESULT_REVIEW_PASS` in Issue #535 comment `5688400426`. There is no mechanics/gate-application blocker.

Key nuance: the football-only pass-volume predictor has genuine blind skill, and C improved over B by `0.03463` targets/team-game with a positive paired CI, but the prospectively frozen materiality floor was `0.10`; WR-room translation was negligible and the key opportunity-dominant tails worsened. No Stage-2 integration is authorized.

### Hard boundaries

- No second `--run-outcomes` exposure.
- No Stage-1 rescue/tuning/model/alpha/feature/cohort/threshold/tail/materiality change.
- Preserve M38 and R15.
- Do not reopen R17-R20, M72/M75/M84, R3/R7/R9-R11, C1/C3, C2->WR1, ND3, PR #561 Vegas confirmation, or old closed WR feature hunts.
- No production change.
- No paid Full Slate.
- RB remains parked.

### Active next action

GPT-5.6 posted a **post-Stage1 WR frontier-audit request** to Claude in Issue #535 comment `5688480341`.

The audit asks whether the already-available leakage-safe historical data universe still contains a genuinely new WR receiving-yard efficiency/ceiling mechanism materially distinct from all closed families, or whether the existing-data frontier is exhausted and the next honest WR step requires a genuinely new information source/representation.

No new candidate outcome run is authorized while this frontier audit is unresolved.

If Claude identifies a surviving existing-data mechanism, independently verify novelty and source availability before freezing anything. If Claude returns `EXISTING_DATA_FRONTIER_EXHAUSTED`, that creates a genuinely consequential program decision and user input is required rather than inventing another feature transformation.
