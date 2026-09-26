# Availability -> Opportunity Rule-Order Audit V1 — Result

Date: 2026-09-26

Disposition: **`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`**

Status: **DIAGNOSTIC ONLY — NO PRODUCTION CHANGE**

## Frozen authority

Plan:
`docs/research/AVAILABILITY_OPPORTUNITY_RULE_ORDER_AUDIT_V1_PLAN.md`

Immutable pregame Full Slate authority:
- source main: `f7d2011b73950488ea209124ba895b92c401b2b1`
- run: `36204768034`
- artifact: `10892728623`
- digest: `sha256:f52b36fb7a9c929fadca473bd8303823fb9f63bd42c884341cd6c3b52c26ed67`
- live sportsbook acquisition: disabled
- target Week-3 outcomes: unread / unavailable

Authoritative diagnostic:
- head: `ca9592d887338db984896d7d48419b754a89adc4`
- run: `36275905038`
- artifact: `10917451964`
- digest: `sha256:340b5992a58ff591baf7a4ccf92c36482c1e22a082fabdc14e2f4e7dbcb1f6fb`

Contract held:
- candidate variants scored = **0**
- parameters fit = **0**
- sportsbook inputs used = **0**
- target-game outcomes read = **0**
- production mutations = **0**

Earlier runs `36275685600`, `36275760074`, and `36275830839` stopped mechanically before an accepted result because of artifact nesting, event-id schema carry, and JSON scalar serialization respectively. None produced scientific evidence and no methodology changed.

## Primary finding

Production resolves definitive availability **before** player opportunity.

On the preserved Week-3 authority:
- current eligible teams: **30**
- definitive-unavailable RB/FB/WR/TE players on those teams: **11**
  - WR: **6**
  - TE: **3**
  - RB/FB: **2**
- unavailable players surviving eligible roles: **0**
- unavailable players surviving PlayerForm: **0**
- unavailable players surviving ModelContext: **0**

Therefore the existing `simulation_rules._injury_target_overrides()` rule cannot observe a definitive-unavailable WR once production reaches the rule layer.

This matters because that legacy rule is explicitly written to identify an injured alpha WR and redistribute part of his target share. Its OUT-player unit test constructs an OUT WR directly in PlayerContext, but current production availability-first routing removes that player before such a PlayerContext can exist.

The rule remains potentially reachable for a retained uncertain player such as DOUBTFUL. This result is specifically about **definitive unavailable** states.

## Exact Week-3 rule execution

Using the preserved no-odds Full Slate artifact and the actual production:
- Bayesian baseline,
- simulation rules,
- M38 target-share sharpening,
- explicit target-entitlement materialization,
- top-five rushing allocation semantics,

the number of production-reachable `rules_injury_redistribution` player rows was:

**0**

This is despite 6 definitive-unavailable WRs in the current eligible universe.

## Important correction to the initial hypothesis

The missing player's opportunity does **not** simply become a larger residual bucket in this Week-3 state.

For every affected current team, surviving rule-adjusted target and top-five rush share already exceeded the finite allocator's 0.95 explicit-player cap.

Therefore every affected team finished at:
- modeled target probability = **0.95**
- target residual = **0.05**
- modeled rush probability = **0.95**
- rush residual = **0.05**

The actual baseline behavior is more subtle:

> definitive-unavailable players disappear before vacancy-specific rules, then surviving player shares are globally normalized/capped by the generic allocator.

So production conserves the broad player-opportunity mass while lacking an explicit football-specific statement about **which successors inherit the vacated entitlement**.

## Current pregame examples

Strict-prior shares are listed only as evidence that removed players can carry material opportunity; no target-game results are used.

- HOU WR Nico Collins: most recent same-team strict-prior target share **0.270270**
- SF WR Demarcus Robinson: **0.142857**
- MIA WR Caleb Douglas: **0.130435**
- NYJ TE Mason Taylor: **0.130435**
- LAC WR Brenen Thompson: **0.115385**
- LAC TE Charlie Kolar: **0.115385**
- WAS TE Chig Okonkwo: **0.103448**
- DEN RB Jonah Coleman: strict-prior rush share **0.357143**
- PIT RB Rico Dowdle: strict-prior rush share **0.304348**

Every one of those players is absent from eligible roles, PlayerForm, and ModelContext in the frozen Week-3 production authority.

## RB relationship

This audit does **not** evaluate or alter the already-frozen RB Vacancy Opportunity V1 candidate.

Production baseline has no explicit definitive-RB vacancy transfer before Monte Carlo. The generic top-five allocator normalizes the surviving rush shares.

RB Vacancy Opportunity V1 remains the independent prospective test of whether a strict-prior successor transfer improves that baseline behavior for DEN/PIT.

## WR/TE relationship

WR/TE now have a confirmed structural question analogous to the RB vacancy question:

> Is generic survivor normalization enough, or does explicit pregame vacancy information improve the identity/concentration of successor receiving opportunity?

This result does **not** authorize the old 60/30/10 rule, a new transfer coefficient, a WR-only rescue, a TE rule, or a production mutation.

A separate prospectively frozen receiving-vacancy evaluation is required before any repair candidate is considered.

## Interpretation

This is a system-architecture result, not a claim that current projections are necessarily wrong for every vacancy.

What is confirmed:
1. definitive unavailability is correctly resolved and unavailable players are correctly removed;
2. the existing definitive-WR redistribution idea is ordered after removal and therefore cannot act on those players;
3. the allocator still fills its generic 95% modeled-player mass through surviving shares;
4. the model therefore handles definitive vacancy **implicitly through generic normalization**, not explicitly through vacancy-aware successor entitlement.

That is exactly the kind of transmission seam the systems-integrity audit was intended to find.

## Disposition

`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

No production change.

Next authorized action:
- preserve the current Week-3 receiving vacancies prospectively before kickoff;
- freeze current baseline survivor entitlement;
- after games are final, grade whether realized successor concentration/identity is systematically missed;
- do not alter RB Vacancy V1 or use Week-3 outcomes to design the pregame receiving audit.
