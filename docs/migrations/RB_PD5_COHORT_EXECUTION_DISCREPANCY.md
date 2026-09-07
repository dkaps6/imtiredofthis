# RB-PD5 Frozen-Cohort Execution Discrepancy

## Status
Mechanical/scientific-integrity block. Production unchanged. RB-PD6 must not launch until this is resolved with a valid out-of-sample/multi-season evidence path.

## Discovery
The frozen RB-PD5 plan at commit `3a0422809524de480d08237888dc6e4590ed12a0` states:

> Use the same canonical 2020-2025 RB evidence and same scoreable-row definitions as PD3/PD4.

However, the inherited PD3/PD4 lineage is explicitly 2025-only. The frozen RB-PD4 plan defines the population as the exact canonical 2025 STACK1 production-equivalent RB/HB/FB rows, and the PD5 evaluator/workflow also execute only 2025:

- evaluator `wide()` filters `season == 2025`;
- `EXPECTED_ROWS = 1393` is the 2025 cohort;
- workflow downloads `stack1_2025_rb_trace.csv` and the 2025 STACK2 casebook.

Therefore the RB-PD5 result from run `34109390989` is a valid execution of the inherited 2025 PD3/PD4 cohort, but it is **not** a valid execution of the literal 2020-2025 cohort text written into the PD5 frozen plan.

Because this discrepancy was discovered after results, the frozen plan must not be edited to rescue or reinterpret the experiment. The 2025 result remains useful exploratory evidence only and cannot be represented as a completed 2020-2025 confirmation.

## Required remediation
1. Preserve RB P3 production unchanged.
2. Do not relax or rewrite PD5 gates.
3. Do not launch RB-PD6 against the same 2025 cohort as a purported independent confirmation after seeing PD5 results.
4. Build or locate a canonical multi-season RB P3-equivalent evidence set using seasons outside the already-observed 2025 cohort.
5. Freeze a separate replication/confirmation plan before evaluating those unseen seasons.
6. Only after that replication is dispositioned may PD6 or another residual mechanism advance as a confirmatory candidate.

## Scientific interpretation
The 2025 PD5 finding still indicates that carry-only residual calibration improved central accuracy and repaired the late-season p90 issue while narrowly worsening pooled eligible rushing-yard p90. That observation may motivate future hypotheses, but it must not be treated as a frozen multi-season confirmation.

## Production
No production changes authorized.