# Receiving Rule Semantics Integrity V1 — Stage-1 Result

Date: 2026-09-26

Status: **STRUCTURAL ABLATION COMPLETE — NO OUTCOMES / NO PRODUCTION CHANGE**

Frozen plan:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_PLAN.md`

## Authority

Successful Stage-1 run:
- run: `36276736046`
- source commit: `52238e3aa8e129db4a632c6e8fe1f80c6a109a52`
- artifact: `10917254062`
- digest: `sha256:c7ff51d17adfaf0f96e18808cec5948fa314f126f5c8b0392c8bed5568fa66fa`

Inputs:
- immutable Week-3 no-live-odds authority;
- accepted same-authority target-entitlement replay;
- target-game outcomes read: **0**
- sportsbook inputs: **0**
- fitted parameters: **0**
- production mutations: **0**

A0 baseline replay matched the accepted target-entitlement authority to max absolute gap:
`9.99e-17`.

## Deterministic semantic defects confirmed

### A — middle-open unit mismatch

Week-3 `team_context_v3.csv`:
- team rows: **32**
- `middle_open_rate > 1`: **32 / 32**
- invalid values outside the frozen 0-100 conversion contract: **0**

The production rule compares `middle_open_rate >= 0.50` as though it were a 0-1 fraction. The source field is in percentage points. Thus the gate is effectively always true in current production.

Frozen repair A1 only divides values in (1,100] by 100; the 0.50 threshold and multiplier stay unchanged.

Structural effect A1B0:
- `rules_tgt_share` rows changed: **59**
- all 59 are TE rows
- median absolute changed target-share delta: **0.012218**
- max: **0.028647**
- entitlement rows changed downstream: **286**
- max entitlement delta: **0.011131**
- non-target rule max delta: **0.0**

### B — slot alignment dropped before rule labeling

Preserved PlayerForm:
- WR rows: **161**
- SWR alignment rows: **56**

Current A0 production labels:
- WR1: **30**
- WR1_5: **30**
- SLOT: **0**
- unlabeled WR: **101**

Frozen B1 carries the already-produced `alignment_position == SWR` into the role-label step, without changing existing role ranking or multipliers.

Structural effect A0B1:
- SLOT labels restored: **56**
- `rules_tgt_share` rows changed: **56**
- all 56 are WR rows
- median absolute changed target-share delta: **0.014586**
- max: **0.029649**
- entitlement rows changed downstream: **424**
- max entitlement delta: **0.048108**
- non-target rule max delta: **0.0**

### Combined A1B1

- SLOT labels: **56**
- target-share rows changed: **77**
  - TE changed: **59**
  - WR changed: **18**
- median absolute changed target-share delta: **0.013320**
- max: **0.029649**
- entitlement rows changed: **424**
- max entitlement delta: **0.048108**
- non-target rule max delta: **0.0**

## Disposition

`STRUCTURAL_ABLATION_COMPLETE_NO_OUTCOMES`

Stage 1 proves both semantic defects are real and the frozen repairs are narrowly scoped. It does **not** authorize production.

Next required step is preregistered Stage-2 accuracy evidence using only completed leakage-safe pregame evidence from before Week 3. Week-3 outcomes remain forbidden.
