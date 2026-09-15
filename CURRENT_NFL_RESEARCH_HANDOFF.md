# CURRENT NFL RESEARCH HANDOFF — READ FIRST

GitHub is canonical; chat memory is secondary.

## ACTIVE RESEARCH CHECKPOINT — 2026-09-15

The user's explicit current priority is **WR receiving yards**. Do not skip ahead to RB unless the user changes priority.

Read in this order:

1. `AGENTS.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-15_WR_PHASE4C_STAGE1_PREFLIGHT_CURRENT.md`
3. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_PLAN.md`
4. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_SOURCE_AUDIT.md`
5. `docs/research/WR_PHASE4C_STAGE1_LAYER2_DOMAIN_AMENDMENT_V1.md`
6. latest GPT-5.6 / Claude checkpoints in GitHub Issue #535
7. `NFL_MASTER_CONTINUITY_RECORD.md` only if older lineage is needed.

### Immediate action

Active branch: `research-wr-phase4c-stage1-pass-volume-predictor-v1`.

Canonical sealed Stage-1 preflight is green:

- run `35024523296`
- job `104568212642`
- artifact `10418149149`
- digest `sha256:1752b8b434424f79091d9db9955c9ac91cb4216d314ef3e90af7b0b9dcb9cdea`
- upstream Phase4B artifact `10404525877`, digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- focused synthetics 5/5 PASS
- blind-output contract guard PASS
- blind 2024 Stage-1 outcomes remain SEALED; `--run-outcomes` has NOT been executed.

GPT-5.6 requested Claude's implementation review in Issue #535 comment `5688220080`. At the handoff update, no later Claude comment was visible through the GitHub API.

**First action in a new chat:** check Issue #535 for Claude's response to `5688220080`.

- If `IMPLEMENTATION_REVIEW_PASS`: run the single frozen `--run-outcomes` exposure and preserve the result regardless disposition. No retuning or rescue.
- If Claude raises a blocker: independently verify it, fix only the valid blocker, rerun sealed preflight, and keep outcomes sealed until implementation review passes.

### Scientific state

- Phase4B closed with R15 healthy/improved; preserve M38 + R15.
- Phase4C Gate 0 passed: scoring environment / `market_total` has a modest coherent bridge to team target-pool realization and opportunity-dominant WR yardage tails.
- Stage1 asks whether a strictly pregame **football-only predicted realized pass-volume state** adds beyond the market on the Phase4B team-pool residual.
- Decisive comparison is C > B OOS; all seven frozen gates must pass.
- A blind preflight-only Layer2-domain amendment repaired an impossible tail denominator before any Stage1 outcome scoring; addressable tails are 208 / 85 / 294 while raw Layer1 parent counts 222 / 87 / 312 remain disclosed.
- Historical `plays_est` / `dropback_rate` lineage was explicitly proven PBP-only with zero Vegas/market lineage.

### Hard boundaries

- Do not reopen R17-R20, R21-style feature fishing, PR #561 confirmation classifier, shared-tail/C2-WR1 rescue, or blind M38/R15 retuning.
- No production/model/threshold change from this research.
- No paid Full Slate.
- RB is parked.
- Continue GPT-5.6 + Claude adversarial collaboration through Issue #535.

The detailed current handoff contains all exact lineage, gates, anti-retest rules, Phase4B/Gate0 results, source audit, blind-domain amendment, and next-step instructions.
