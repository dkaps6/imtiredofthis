# NFL HANDOFF — 2026-09-24 — TE WIDTH V2 FAILED CLOSED / NEXT SCIENCE

GitHub is canonical over chat memory.

This checkpoint supersedes the prior in-progress TE Width V2 handoff for immediate execution.

## Production state

Current production main before this docs-only checkpoint:
`cd6cc06459fc88f2bc6d7d2c23d544ca75e9e12f`

RB Rush+Receiving Conservation V2 remains production-active. Do not reopen it absent a concrete defect or genuinely new prospective evidence.

## TE-R5P Receiving-Yards Width V2 — CLOSED

Canonical frozen validation:
- run `36040911515` = **SUCCESS**
- job `107772560075`
- branch `research-te-live-entitlement-efficiency-v1`
- exact head `a9c7a93ae375f75e926139a0a29675aab89d2fdc`
- result artifact `10827222034`
- digest `sha256:f128cdf5b5d41c18ff1a08f1a9c23f346f911294d6e8e6afe678c8c64f526cc2`
- reconstructed distributions artifact `10826839080`
- digest `sha256:37fffe9d2b176ac4db98e18a96ebe4eb7bc418a1c67b2e501ffc5c69f45501be`

Disposition:
`TE_R5P_REC_YARDS_WIDTH_V2_FAILED_CLOSED`

Why:
- fit2024 -> test2025 CRPS worsened `11.5174 -> 11.6626` (-1.26%)
- that same direction's 80% coverage gap worsened `0.08495 -> 0.08780`
- fit2025 -> test2024 improved CRPS and 80/90 coverage
- 90% coverage improved in both directions
- pooled Brier/log loss improved
- point means were invariant
- but the frozen primary contract required CRPS and 80% coverage-gap improvement in **both** directions

No rescue is authorized:
- no alternate k
- no cap search
- no subgroup rescue
- no global TE SD multiplier
- no sportsbook-conditioned width
- no 2026 result fitting
- no production integration

Canonical result doc:
`docs/research/TE_R5P_REC_YARDS_WIDTH_V2_RESULT.md`

## Immediate next action

Move to genuinely new Week-3 model-improvement science.

The preferred first pass is a **read-only structural contradiction audit** of the latest preserved Full Slate / production artifacts:
- conservation identities;
- incompatible final component means;
- opportunity mass that disappears;
- post-specialist overrides that break joint football identities;
- player/team hierarchy states inconsistent with current usage;
- injury/personnel changes that do not propagate.

Do not restart old C1/C3 receiving formulations. Shared QB/receiver conservation is known architecture evidence, but any revisit must first distinguish a genuinely new bounded production contradiction from the old mechanically incomplete full-stack integration lane.

Keep prospective RB Vacancy V1 separate and unchanged. Week 2 still had zero qualifying preserved definitive-unavailable RB/FB events; next legitimate test remains pregame prospective capture.

## Closed/protected

Continue protecting all lanes listed in the previous checkpoint, including:
- TE Width V2 exact candidate now closed;
- TE target-quality global mean correction;
- generic QB pass-yard mean retuning;
- exposed retrospective M96 RB router variants;
- C1/C3 pass-receiving formulations;
- generic receiving attempt-semantics C4;
- global SD rescale;
- sportsbook lines upstream;
- paid OddsAPI pulls without explicit user approval.

## Continuity rule

Every new hypothesis/test/run/artifact/digest/commit/failure/qualification/promotion/stop decision goes to GitHub and Issue #535. Distinguish mechanical, provenance, scientific failure, scientific qualification, and production certification.
