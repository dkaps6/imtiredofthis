# RB STACK6B — Directional / Depth Postmortem Plan

## Why this audit is justified

The frozen STACK6B compact-role experiment completed with no passing arm. `COMPACT_ROLE` worsened both eligible carry and yard MAE. `AGG_PLUS_COMPACT` produced only a tiny eligible yard improvement, failed the carry gate, but showed a positive W13-18 yard improvement while leaving the protected M95F-risk and depth-rank-1 populations unchanged exactly.

This document freezes a **no-fit failure postmortem** before inspecting row-level STACK6B outputs. It is not a new retention experiment and cannot promote a model.

## Fixed input

Consume the frozen successful STACK6B artifact only. Do not refit Ridge, change features, alter alpha, alter clipping, search thresholds, or use sportsbook information.

## Predeclared diagnostic questions

For both frozen STACK6B arms (`COMPACT_ROLE`, `AGG_PLUS_COMPACT`):

1. **Direction** — separate rows where the fitted correction contracts carries (`delta < 0`), expands carries (`delta > 0`), or makes no numerical change.
2. **Depth hierarchy** — report the same diagnostics for depth-rank 2 and depth-rank 3+.
3. **Time stability** — report W6-12 and W13-18 separately.
4. **Residual alignment** — measure whether correction direction agrees with the realized P3 carry residual sign and whether it reduces absolute carry/yard error.
5. **Diagnostic directional counterfactuals** — score:
   - `CONTRACTION_ONLY`: preserve the frozen negative deltas and reset positive deltas to the P3 parent;
   - `EXPANSION_ONLY`: preserve the frozen positive deltas and reset negative deltas to the P3 parent.

The counterfactuals are **diagnostic only**. They are not eligible for retention or production and may only justify freezing a new prospective architecture.

## Interpretation contract

- If contraction is consistently useful while expansion is harmful, the next research architecture may be a precommitted one-sided contraction model; do not tune a delta threshold on 2025.
- If RB2 signal is materially cleaner than RB3+, treat the populations as different football problems rather than searching a depth threshold. RB3+ should then prioritize target-game active/inactive competitor state and rotation information.
- If neither direction/depth decomposition produces stable error recovery, stop reusing the compact-role fitted signal and move to a genuinely new information family.

Sportsbook data is excluded entirely from this postmortem.
