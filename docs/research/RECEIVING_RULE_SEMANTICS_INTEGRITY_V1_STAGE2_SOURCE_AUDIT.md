# Receiving Rule Semantics Integrity V1 — Stage 2 Source Audit

Date: 2026-09-26
Status: FROZEN SOURCE-AVAILABILITY CHECK ONLY

Purpose: determine whether the preregistered Stage-2 accuracy test has honest pre-Week-3 evidence for:
A1 middle-open unit repair;
B1 slot-alignment carry.

No candidate scoring is permitted in this source-audit step.

Checks:
1. Long-retention Week-1 artifact 10124274040: inventory only.
2. Expired canonical Week-2 artifact 10523345092: do not reconstruct from current state; inspect only surviving derivative artifacts.
3. Historical walk-forward builder:
   - verify whether team_weekly_history contains middle_open_rate / coverage rate history at the target cutoff;
   - verify whether pregame universe has honest week-tagged slot-alignment evidence from roster/depth source;
   - report source_name and counts of WR rows carrying SWR/LWR/RWR alignment-like labels.
4. No Week-3 outcomes.
5. No sportsbook inputs upstream.
6. No repair coefficients or thresholds may change.

Disposition options:
- STAGE2_A1_HISTORICAL_SOURCE_AVAILABLE
- STAGE2_B1_HISTORICAL_SOURCE_AVAILABLE
- STAGE2_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY
- mixed A1/B1 disposition allowed.
