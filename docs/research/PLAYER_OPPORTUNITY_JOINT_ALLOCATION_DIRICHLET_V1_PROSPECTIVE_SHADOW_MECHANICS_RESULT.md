# PLAYER OPPORTUNITY JOINT ALLOCATION — PROSPECTIVE SHADOW V1 MECHANICS RESULT

Date: 2026-10-13

Status: **PROSPECTIVE SHADOW MECHANICS CERTIFIED. RESEARCH-ONLY. WEEK-6 LOCK REMAINS SOURCE-GATED.**

Frozen contract:
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_FREEZE.md`

Historical PASS:
- run `38401069806`
- artifact `11815490905`
- digest `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`

Mechanics authority:
- workflow run: `38402018307`
- exact head: `fb4136b639ac4b2281a57a6d3a383c97b5c79fe6`
- workflow conclusion: `SUCCESS`

Certified:
- exact target-week player-state authority required;
- stale prior-week roster is rejected;
- only RB carry / WR target cells accepted;
- canonical mean q is preserved exactly;
- locked alpha/alpha0 equals canonical q within 1e-12;
- room probability mass conserved;
- strictly-prior source week required;
- zero-excess / unauthorized player state stays ineligible;
- no production simulator import/reference;
- strict repository audit passed.

Future runner:
- `.github/workflows/research-future-week-joint-opportunity-shadow-v1.yml`
- manual `workflow_dispatch`;
- defaults to 2026 Week 6;
- probes exact roster readiness before building the shadow;
- freezes player-state, volatility, and joint-shadow artifacts together;
- reads no target-week outcomes.

Current Week-6 source boundary:
- prior Week-5 stats/snaps: complete;
- exact Week-6 roster snapshot: not yet available;
- do not dispatch expecting a valid lock until that authority exists.
