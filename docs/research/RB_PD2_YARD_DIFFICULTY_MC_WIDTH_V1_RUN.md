# RB-PD2 Yard-Difficulty MC-Width V1 — Run Trigger

**STATUS: TRIGGERING THE FROZEN EVALUATOR. NO RESULT SEEN YET.**

This file's presence in this path, on this branch, is the sole trigger for
`.github/workflows/research-rb-pd2-yard-difficulty-mc-width-v1.yml`'s real
jobs (`rebuild-distributions`, `evaluate-frozen-candidate`). It deliberately
does not fire on pull_request events, so a PR can go through full review
without accidentally exposing a result before the design is settled.

Frozen plan head at trigger time: `2f77758926c1a0b0a27afd42fab909f8b492926d`
("research: freeze RB yard-width amendment 3").

Review completed before this trigger:
- GPT-5.6: authored the plan and evaluator, self-audited three pre-result
  fidelity amendments (2021-2024 history-universe fidelity, player-clustered
  paired bootstrap, exact strictly-prior percentile convention pinned to the
  existing WR-R3 precedent), plus a fourth strictness pass in amendment 3.
- Codex: automated PR review; findings addressed before this trigger.
- Claude: independent line-by-line adversarial review of the plan and the
  496-line evaluator (verified the empirical CRPS closed form by hand,
  the strict-prior two-pass percentile scoring, the mean-neutral width
  transform, and the player-cluster bootstrap) -- no blocking issue found.
  Full review posted on PR #562 and on Issue #535.

No candidate distribution result has been computed, run, or inspected by
either collaborator before this commit. Per the frozen plan
(`docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md`), a PASS here is
a research qualification only and does not by itself authorize any
production change -- that still requires a separately frozen
production-shadow/forward-confirmation design.
