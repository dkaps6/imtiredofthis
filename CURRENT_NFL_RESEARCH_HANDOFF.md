# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ACTIVE PRODUCTION CHECKPOINT — 2026-09-11

Before doing anything else, read in this order:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_PARITY_PASS_WR_TE_AUDIT_CHECKPOINT.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_RB_PHASE_A_RECOVERY_CHECKPOINT.md`
3. `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_PRODUCTION_READINESS_CURRENT.md`
4. `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`
5. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

## Current highest priority

**Season-long 2026 production readiness is the active critical path.**

One stable week-aware production system must run every week. Weekly work refreshes pregame football state; it does not invent a new Week-N formula. Completed 2026 weeks are immutable grading/calibration evidence only.

## Latest proven checkpoint

STACK2 frozen allocation parity is now PASS:
- verification run `34659593197`
- artifact `10286921484`
- artifact digest `sha256:28b8fa73dbe249e55388d108c0e022ab781bb343c1b8952c2c08b131097ffa94`
- 2024 training rows `2102`
- archived 2025 parity rows `1393`
- max score/share difference `1.1102230246251565e-16`
- frozen training-matrix SHA256 `5e871fd673cdbfb0f03c82b454adf65ff726109195dd4dcd0c8b2e51969b492e`
- frozen serialized learner SHA256 `d33960cbfae8a61af0dfa92dc61a97b96027c0a1a8036bf4308734442c8bdd9b`
- no scientific search; no sportsbook inputs.

2026 weekly roster metadata source also passed source-contract discovery across all 32 teams.

## Season-long P3 rushing authority

- Week 1 initialization -> `WEEK1_STACK_OVERRIDE` / STACK1 full-stack rushing projection.
- Weeks 2-18 -> `WEEKS2_18_ENRICHED_OPP_STACK_EFF`: frozen STACK2 enriched RB allocation/opportunity × STACK1 implied efficiency.

Do not create separate weekly rushing models.

## Newly identified WR/TE runtime defect

WR-R15 and TE-R5P fitted science is generic across target weeks and uses strict-prior participation, but their shared snap loader is currently fixed to source seasons 2020-2025. If unchanged, 2026 Week 2+ would not consume earlier 2026 snap participation.

This is a runtime source-horizon defect, not a coefficient/science failure. Repair must include target-season snaps while preserving strict `ordinal < target_week` filtering and proving 2026 Week-1 numerical invariance.

## R26/R22 status

R26 receptions and R22 receiving-tail production adapters are genuinely hard-gated to 2026 Week 1. Do not silently carry the Week-1 vacancy classifier or simply delete the gates. Protected baseline future-week authorities remain the safe fallback unless season-long transport is prospectively qualified.

## Exact immediate work order

1. Store/freeze the exact passing STACK2 allocation authority in the production tree with hash validation.
2. Build one generic target-week STACK2 feature/state adapter for Weeks 2-18.
3. Feed `enriched_att` into existing P3 composition and replace Week-1-only downstream P3 guards with target-week routing invariants.
4. Repair WR/TE runtime snap-source horizon with Week-1 invariance + Week-2 prior-row tests, without changing fitted model parameters.
5. Audit QB runtime for source-horizon/week assumptions.
6. Close R26/R22 future-week routing, defaulting to protected baseline unless transport earns qualification.
7. Run generic target-week no-paid-odds certification.
8. Build immutable in-season postgame grading/calibration harness.

No paid OddsAPI acquisition is authorized for this work.

## Parked lanes

Park QB/WR public-intent V1B, new game ML/spread/total science, new anytime-TD science, and broad betting-card optimization unrelated to production sanity until season-long readiness is closed.

Important continuity correction: prior chat notes carried an `M108 = 26/26 PASS` label, but repository search did not recover authoritative M108 workflow/script/run/PR evidence. Do not invent or require M108 by name unless concrete GitHub lineage is recovered.