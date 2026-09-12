# TE Gap Findings — Overnight Research Survey

**STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.**

## What's already in production

TE-R5P (`TE_R5P_PRODUCTION_MODEL_V1`) governs TE target entitlement, wired into live certification alongside WR-R15. Like WR, it does not cover receiving efficiency/distribution — that's a shared open lane named explicitly in `scripts/validate_certified_full_slate_stack_v3.py`.

## Landscape: R1 → R6, all Sept 7, 2026 (small, coherent, worth reading in full — and I did)

1. **R1** (`research-te-r1-individual-mechanism-decomposition`) — frozen plan → implemented → workflow launched, but **the branch stops there. No result doc was ever committed.** Unfinished/interrupted, exactly like QB-PD3 — not a rejection.
2. **R2** (`research-te-r2-pool-vs-individual-allocation`) → per-player residual correction is the wrong starting point. Disposition: model the **team TE pool** from pregame football info first, then a distinct individual-allocation layer beneath it.
3. **R3** (`research-te-r3-target-pool-context-model`) → scientific failure. Team-pool corrections can improve target MAE while making receiving-yard MAE **worse** when the correction isn't sufficiently player-differentiated. Explicit redirect: bring genuinely new **individual participation/role and matchup information**, not a retuned pool model or a post-hoc Q4-only exception.
4. **R4** (`research-te-r4-strict-prior-participation-source`) → participation belongs in a **hierarchical** opportunity model, not a blanket correction: `team pass state → team TE pool → individual TE entitlement → catch/yard efficiency`.
5. **R5** (`research-te-r5-participation-entitlement-v1`) → **`TE_PARTICIPATION_ENTITLEMENT_V1_PASS`**. The individual-entitlement layer described by R4 passed its frozen gates. Explicit note: "Do not directly promote TE-R5 from this result file" — i.e., passing the diagnostic isn't itself an authorization to ship; it needs the proper integration step.
6. **R6** (`research-te-r6-full-stack-entitlement-integration-v1`) — froze the integration plan; its later commits are QB-C2 full-stack conservation work, consistent with this being the branch that actually wired TE-R5 into the shared production C2 stack. This matches reality: TE-R5P is live in production today, so this thread completed successfully even without a separate "R6 RESULT.md" — the production adapter (`scripts/modeling/te_r5p_entitlement_adapter_v1.py`) is the completed artifact.

**Net: entitlement is solved and shipped (R2→R5→production). Efficiency/distribution is the acknowledged, unstarted-in-full open lane — R3/R4's call for "individual participation/role and matchup information" was answered for *entitlement* but never followed up for *efficiency*.**

## Ideas explored but not carried forward

- R1's mechanism-decomposition diagnostic never finished. It's the natural TE-side counterpart to R3's error-persistence idea on WR and to QB's PD3 — a frozen, half-built diagnostic sitting idle.

## Current open frontier

TE receiving-efficiency/distribution calibration, same acknowledged gap as WR, but TE's own R3/R4 literature already specifically calls for "matchup information" as the next legitimate signal family — and nothing in this 6-branch lineage ever tested one. Sample size is smaller than WR (fewer targets/game), so anything proposed here carries more overfitting risk and should be validated with that in mind.

## Proposed new research directions

1. **Finish R1 first.** It's frozen, scoped, already has a workflow — the same "shovel-ready, someone should just run it" situation as QB-PD3. Cheapest possible next step, and it may itself surface whether individual mechanism-level signal (independent of the R2-R5 pool/entitlement work) is worth pursuing before building anything new.
2. **Test Coverage v2 (`data/cb_coverage_player.csv`, `data/wr_cb_exposure.csv` — note: check whether a TE-specific matchup file exists or whether this is WR-only plumbing that would need a TE-side equivalent built) as the R3/R4 "matchup information" signal**, mirroring their exact frozen protocol. This is the same proposal I'm making for WR (see `WR_GAP_FINDINGS.md`) — if you run one, the infrastructure/lessons transfer directly to the other, so consider doing WR first (larger sample, lower risk) and only extending to TE if it clears WR's gates.
3. **Given the small TE sample, prioritize a route/personnel-based feature over anything requiring per-player history splits** (e.g., red-zone/contested-catch role, in-line vs. slot/flex alignment rate, blocking-snap share as a proxy for route-running opportunity) — these are lower-variance, team-and-role-level signals rather than individual-history signals, better suited to TE's smaller n.

## Open questions for you

- Is finishing R1 worth prioritizing, or should TE research explicitly wait on WR's Coverage v2 test (proposal #2) to see if it's worth the smaller-sample TE version at all?
- Does a TE-specific coverage/matchup dataset already exist anywhere, or would testing proposal #2 for TE require new plumbing beyond what WR would need?
