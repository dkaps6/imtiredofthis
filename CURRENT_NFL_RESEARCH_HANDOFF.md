# CURRENT NFL RESEARCH HANDOFF — READ FIRST

GitHub is canonical; chat memory is secondary.

## ACTIVE RESEARCH CHECKPOINT — 2026-09-15

The user's explicit current priority is **WR receiving yards**. Do not skip ahead to RB unless the user changes priority.

Read in this order:

1. `AGENTS.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-15_WR_PHASE4C_STAGE1_RESULT_CURRENT.md`
3. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_RESULT.md`
4. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_PLAN.md`
5. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_SOURCE_AUDIT.md`
6. `docs/research/WR_PHASE4C_STAGE1_LAYER2_DOMAIN_AMENDMENT_V1.md`
7. latest GPT-5.6 / Claude checkpoints in GitHub Issue #535, especially GPT-5.6 result-review request comment `5688386255` and any Claude response after it
8. `NFL_MASTER_CONTINUITY_RECORD.md` only if older lineage is needed.

### Immediate checkpoint

Active branch: `research-wr-phase4c-stage1-pass-volume-predictor-v1`.

WR Phase4C Stage-1 blind outcomes were exposed **exactly once** after Claude returned `IMPLEMENTATION_REVIEW_PASS` in Issue #535 comment `5688276336`.

Canonical outcome lineage:

- one-shot orchestration commit `8b49e35f194f8dfb6538be7148590370987f8534`
- run `35025966255` — SUCCESS
- job `104572956822` — SUCCESS
- artifact `10419147232`
- digest `sha256:09722be047456271ab2d73132cdd9c7a46dd0e771558314aba49992e096e407d`
- upstream Phase4B artifact `10404525877`, digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- focused synthetics re-run 5/5 PASS
- 515/515 full Layer2 coverage
- 208/208 amended primary-tail coverage

Frozen disposition:

`NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT`

Do **not** rerun `--run-outcomes`, tune, rescue, alter gates/materiality, or change M38/R15.

GPT-5.6 posted the full result lineage to Claude in Issue #535 comment `5688386255` requesting an independent `STAGE1_RESULT_REVIEW_PASS`. Check for Claude's response first in a new chat. Claude review is an audit only and does not authorize a rescue.

### Load-bearing result

- Script predictor blind skill: PASS. Naive MAE `6.80946` -> Ridge `6.50376`; improvement `+0.30570`, paired 95% CI `[+0.14070, +0.47841]`.
- Decisive B -> C team-target MAE: `6.12207` -> `6.08743`; improvement `+0.03463`, paired 95% CI `[+0.00250, +0.06643]`.
- Frozen materiality floor was `0.10`, so `C_gt_B_material = FAIL` despite the positive CI.
- C vs A improvement `+0.05056`, CI `[-0.00671, +0.10787]`: FAIL.
- RMSE/bias tradeoff guard: PASS.
- WR-room translation: only `+0.00718`, CI `[-0.01030, +0.02451]`: FAIL.
- Primary opportunity-dominant underprojection tail: `-0.03199` B-minus-C, wrong direction: FAIL.
- Actual-100+ opportunity-dominant tail: `-0.08493`, CI entirely below zero: C worse.
- Full underprojection tail: tiny `+0.00941`, unstable; cannot rescue primary failure.

Frozen gates:

- coverage_integrity PASS
- script_blind_skill PASS
- C_gt_B_material FAIL
- C_gt_A_material FAIL
- no_rmse_bias_tradeoff PASS
- wr_room_translation_positive FAIL
- tail_mechanism_alignment FAIL

### Scientific state

- Phase4B remains closed with R15 healthy/improved; preserve M38 + R15.
- Phase4C Gate 0's descriptive scoring-environment bridge remains valid.
- Stage1 shows a legitimate football-only pass-volume predictor, but its downstream increment is too small and not WR-tail aligned enough to be actionable under the prospectively frozen contract.
- No Stage-2 integration is authorized.

### Hard boundaries

- Do not reopen R17-R20, R21-style feature fishing, PR #561 confirmation classifier, shared-tail/C2-WR1 rescue, or blind M38/R15 retuning.
- Do not rescue Stage1 with another alpha/model/PROE/spread/margin target/market threshold/interaction/subgroup/reverse rotation/materiality threshold.
- No second blind Stage1 exposure.
- No production/model/threshold change from this research.
- No paid Full Slate.
- RB is parked.
- Continue GPT-5.6 + Claude adversarial collaboration through Issue #535.

The detailed current handoff and Stage1 result document contain the full exact lineage, metrics, stopping rules, prior Phase4B/Gate0 evidence, and next-step constraints.