# Post-Specialist Cross-Market Consistency Audit V1 — Result

Date: 2026-09-26

Disposition: **POST_SPECIALIST_CROSS_MARKET_CONSISTENCY_V1_CONTRADICTION_CONFIRMED**

This is a diagnostic result only. No repair is authorized by this document.

## Authority

- branch: `research-post-specialist-cross-market-consistency-v1`
- frozen plan commit: `97a33b46c3048af5b96de8205ad0c13d6a3ef2fd`
- implementation commit: `02d0937cf9a302977953904cd268d440432fdbdf`
- workflow head: `f85131bec8ca0c1e2c0b0c03515bf0f95c250324`
- run: `36252584883`
- job: `108433264136`
- artifact: `10909602741`
- artifact digest: `sha256:164efb48c0eacad0b84744cb9a074c61997c7aba8a5f44e2bd18c6916d17421b`

Immutable football source:

- no-live-odds Week-3 Full Slate run: `36204768034`
- artifact: `10892728623`
- source main: `f7d2011b73950488ea209124ba895b92c401b2b1`
- sportsbook inputs upstream: **0**

## Integrity

- 2026 Week 3 only
- 25,000 joint Monte Carlo iterations
- 99 RB/FB rows
- 30 teams represented in the frozen slate
- target-game outcomes read: **0**
- sportsbook inputs used: **0**
- new candidate variants constructed/scored: **0 / 0**
- parameters fit: **0**
- production mutations: **0**
- exact frozen ensemble weights reproduced with zero drift

## Contradiction 1 — independent market blending breaks opportunity/efficiency coupling

The joint simulator uses one finite rushing opportunity state and one player efficiency state.

On the frozen Week-3 RB/FB population:

MC implied YPC vs `rules_ypc`:
- median absolute gap: **0.008555**
- p90 absolute gap: **0.025528**
- max absolute gap: **0.113324**

So the joint MC path preserves the intended player efficiency extremely closely.

Final generic ensemble implied YPC vs MC implied YPC:
- median absolute gap: **0.489607**
- p90 absolute gap: **1.217239**
- max absolute gap: **1.973866**

Final generic ensemble implied YPC vs `rules_ypc`:
- median absolute gap: **0.502355**
- p90 absolute gap: **1.319792**
- max absolute gap: **2.135026**

The frozen production weights explain the seam:

`rush_att`
- MC 0.316492
- ML 0.652896
- State 0.030612

`rush_yards`
- MC 0.556954
- ML 0.443046
- State 0.000000

Because the two markets blend the shared football state differently, a pure change in joint-MC opportunity does not propagate proportionally through the final carry and yardage means.

Exact local diagnostic:

- MC implied-YPC sensitivity to a proportional MC opportunity scale at fixed MC efficiency: **0**
- final ensemble implied-YPC sensitivity:
  - median absolute derivative: **0.932114**
  - p90: **1.187337**
  - max: **1.964717**

This is structural, not a five-player vacancy anomaly.

## Contradiction 2 — final player carry means can exceed the finite team rushing environment

The canonical joint simulator is finite-volume and preserves a residual bucket.

But after independent player/market blending, the summed final RB/FB carry means alone exceed the simulator's entire team rushing-attempt mean on **7 of 30 teams** in the frozen Week-3 slate.

Largest examples:

- KC: final RB/FB carry sum **28.1378** vs team MC rush total **25.0258**; over by **3.1120**
- LAC: **27.0647** vs **24.6316**; over by **2.4331**
- MIA: **23.8794** vs **22.6935**; over by **1.1859**
- SEA: **25.7671** vs **24.7668**; over by **1.0003**

Across teams, p90 of final-RB/FB / team-MC-rush-total ratio is approximately **1.0416** and the maximum is approximately **1.1244**.

This cannot be explained by an unmodeled QB/WR rusher because the contradiction already occurs using RB/FB means alone against the entire team rushing total.

## Protected science remains intact

RB Rush+Receiving Conservation V2:

- eligible RB rows: **85**
- exact protected mean identity max gap: **0.0**
- underlying V2 pathwise identity max gap: **0.0**
- sportsbook inputs: **0**

Therefore this audit does **not** reopen or falsify RB Rush+Receiving Conservation V2.

Rush Pool Evidence Guard V1 also remains closed. That older lane changed top-five simulator membership based on evidence state. The present contradiction is downstream, after the finite simulator, in independently ensembled final means.

## Scientific interpretation

The production architecture contains a genuine cross-market reconciliation problem:

1. the joint simulator constructs a coherent finite opportunity/efficiency state;
2. independent market-specific blends improve marginal predictions;
3. those independent blends can destroy joint football identities after the simulator;
4. downstream opportunity specialists can therefore create implied-efficiency movement even when football efficiency is explicitly frozen;
5. final player carry means can become incompatible with the finite team opportunity environment.

This is exactly the kind of post-specialist contradiction the current research handoff authorized us to look for.

## Next action

Freeze one separate, parameter-free **post-ensemble rushing mean reconciliation** hypothesis before any historical scoring.

The diagnostic itself authorizes no production mutation and no threshold/router search.
