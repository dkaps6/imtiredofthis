# RB-PD2 Yard-Difficulty MC-Width V1 — Run Trigger

**STATUS: RETRYING THE FROZEN EVALUATOR AFTER A MECHANICAL CALENDAR REPAIR. NO CANDIDATE RESULT HAS BEEN SEEN.**

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

Mechanical replay history:
- workflow-dispatch run `34763298746` reached historical input reconstruction
  but failed before any candidate distribution or evaluation because the
  generic historical player-log builder allowed nflreadpy's 2020 Week-18
  postseason-like rows into a season whose validated REG schedule ends at
  Week 17;
- the repair at commits `3162ddb110e91bb9de400d07cd410963468988e8`
  and `ac41ea6e8f6c674962befdae1a721d65b3741aa4` makes the validated historical
  schedule authoritative for valid regular-season week numbers and adds a
  regression proving 2020 W18 is excluded while valid-week team mismatches
  still fail closed;
- no hypothesis, coefficient, width mapping, cohort gate, CRPS/coverage/Brier
  rule, bootstrap rule, mean authority, or sportsbook boundary changed.

No candidate distribution result has been computed or inspected before this
retry. Per the frozen plan (`docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md`),
a PASS here is a research qualification only and does not by itself authorize
any production change -- that still requires a separately frozen
production-shadow/forward-confirmation design.
