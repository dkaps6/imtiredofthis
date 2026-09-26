# Availability -> Opportunity Rule-Order Audit V1 — Run Marker

Frozen plan:
`docs/research/AVAILABILITY_OPPORTUNITY_RULE_ORDER_AUDIT_V1_PLAN.md`

Immutable pregame authority:
- Full Slate run `36204768034`
- artifact `10892728623`
- source main `f7d2011b73950488ea209124ba895b92c401b2b1`
- sportsbook acquisition disabled
- target outcomes unavailable/unread

This marker triggers the diagnostic workflow only. It does not alter production or the frozen RB Vacancy Opportunity V1 experiment.

Mechanical retry 1: corrected actions/download-artifact's artifact-name directory nesting only. No methodology or data change.

Mechanical retry 2: carry preserved canonical game identity as production event_id before entitlement materialization. No methodology/data change.
