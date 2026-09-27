# Week-3 Specialist Non-Target Monte Carlo Invariance V1 — Frozen Result

Disposition:

`SPECIALIST_NONTARGET_MC_PATH_DRIFT_CONFIRMED`

Production changed: **false**

Repair authorized: **false**

## Authority

Frozen plan:
`docs/research/WEEK3_SPECIALIST_NONTARGET_MC_INVARIANCE_V1_PLAN.md`

Plan commit:
`b44d5fb49328ba8b480fa321d9b9257dfb494130`

Canonical source:
- paid Week-3 Full Slate run `36293274478`
- source artifact `10923570170`
- source digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`

Frozen audit run:
- run `36330399450` = **SUCCESS**
- exact audit head `3bc344713173c3cb434dcf180bac22462cedacd7`
- result artifact `10935587149`
- digest `sha256:f1bf35878758478a3871436b48531432c02a8102663c361942d7e7ff0669db39`

No OddsAPI acquisition occurred. No Week-3 outcome was used. Sportsbook data did not define the protection sets.

## Integrity

All frozen integrity gates passed.

In particular:
- TE-R5P still certified `non_te_entitlement_preserved=true`;
- TE room and team entitlement totals remained conserved;
- WR-R15 still certified `non_wr_entitlement_preserved=true`;
- M38 WR1 anchor remained exact;
- WR2+ room, WR room and team entitlement totals remained conserved;
- numeric trace reconstruction independently reproduced zero protected entitlement change to `1e-12`;
- TE and WR simulation delta files had the same simulation-key universe;
- no sportsbook inputs defined protected membership;
- no target/future outcomes entered the audit.

Therefore the output movement below occurs while the audited player's stage-specific target entitlement is unchanged.

## TE-R5P stage

Stage comparison:
`M38 explicit entitlement baseline -> TE-R5P`

Protected players: **335**

Protected simulation keys: **2,085**

Protected keys with nonzero mean drift:
**1,883 / 2,085 = 90.31%**

Protected keys with element drift:
**1,900 / 2,085 = 91.13%**

### Primary semantically-unrelated markets

Protected unrelated keys:
**745**

Keys with mean/element drift:
**560 / 745 = 75.17%**

Markets:

| Market | Protected keys | Mean-drift keys | Mean abs drift | P95 abs drift | Max abs drift |
|---|---:|---:|---:|---:|---:|
| pass_yards | 75 | 75 | 0.2149 yd | 0.6095 yd | 0.8089 yd |
| rush_att | 335 | 150 | 0.00544 att | 0.02476 att | 0.05596 att |
| rush_yards | 335 | 335 | 0.03608 yd | 0.13857 yd | 0.31124 yd |

Largest protected unrelated Stage-A mean movement:
`0.808893` yards.

### Protected receiving-linked markets

Protected receiving-linked keys:
**1,005**

Mean drift:
**1,002 / 1,005**

Mean absolute mean drift:
**0.04997**

These are descriptive in V1; they were not required to confirm the primary systems disposition.

## WR-R15 stage

Stage comparison:
`TE-R5P -> WR-R15`

Protected players: **290**

Protected simulation keys: **1,815**

Protected keys with nonzero mean drift:
**1,667 / 1,815 = 91.85%**

Protected keys with element drift:
**1,675 / 1,815 = 92.29%**

### Primary semantically-unrelated markets

Protected unrelated keys:
**655**

Keys with mean/element drift:
**515 / 655 = 78.63%**

Markets:

| Market | Protected keys | Mean-drift keys | Mean abs drift | P95 abs drift | Max abs drift |
|---|---:|---:|---:|---:|---:|
| pass_yards | 75 | 75 | 0.2325 yd | 0.6653 yd | 1.0445 yd |
| rush_att | 290 | 150 | 0.00639 att | 0.02472 att | 0.04932 att |
| rush_yards | 290 | 290 | 0.03871 yd | 0.14059 yd | 0.30043 yd |

Largest protected unrelated Stage-B mean movement:
`1.044543` yards.

### Protected receiving-linked markets

Protected receiving-linked keys:
**870**

Mean drift:
**868 / 870**

Mean absolute mean drift:
**0.04930**

Again, descriptive only for this V1 disposition.

## Interpretation

This audit confirms a **finite Monte Carlo path-dependence seam**.

A receiving-room specialist can preserve another player's football entitlement exactly while the full shared simulator re-run produces a different empirical draw sample for that protected player's output.

The finding is particularly clear because:
- all protected QB pass-yard means move at both specialist stages;
- protected rushing-yard means also move despite no rushing entitlement/efficiency input being changed by TE-R5P or WR-R15;
- protected receiving outputs move even where that player's own target probability is unchanged.

This result does **not** prove that the underlying theoretical marginal probability law is wrong. With a global RNG stream, changing another category's multinomial probabilities can alter the finite sampled path even when the protected player's marginal football probability is unchanged.

Therefore this result must not be described as a football-mean defect or predictive improvement opportunity yet.

It establishes the narrower systems fact:

> protected football inputs are not protected at the finite empirical Monte Carlo output surface.

## What is not authorized

Do not yet:
- split RNG streams;
- change seeds;
- increase iterations;
- introduce common-random-number routing;
- cache/splice protected arrays;
- change TE-R5P or WR-R15;
- alter M38;
- alter QB C2 / M89-M90;
- alter RB rushing / RB Rush+Receiving V2;
- change any betting threshold.

## Exact next action

Freeze a separate downstream **probability and board materiality** audit before evaluating it.

That audit must answer:

1. for protected sportsbook-priced rows, how much do fair probabilities move solely because of this path dependence?
2. how often does the preferred side change?
3. how much does raw EV move?
4. can `HAS EDGE` / `PASS`, publishability or board rank change?
5. are observed movements larger than ordinary finite-MC resampling noise under identical football inputs?

Only if downstream materiality is demonstrated may a separate mechanical RNG-isolation repair be designed.

No production change from V1.
