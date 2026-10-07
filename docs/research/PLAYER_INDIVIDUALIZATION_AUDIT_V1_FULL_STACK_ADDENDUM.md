# Player Individualization Audit V1 — Full-Stack Specialist Addendum

**STATUS: SCOPE CORRECTION / ADDENDUM — NO PRODUCTION CHANGE**

This addendum narrows the interpretation of
`PLAYER_INDIVIDUALIZATION_AUDIT_V1_RESULT.md`.

The original V1 audit correctly traced:
- PlayerForm v2;
- Bayesian v2;
- generic simulation rules;
- aggregate player-level error heterogeneity.

It did **not** fully credit every downstream promoted specialist already consumed by Full Slate.

Therefore the V1 result remains valid for the generic baseline/rules layer, but the full production interpretation must include the specialist lineage below.

## Full production player-individualization map

### QB pass yards — strong specialist coverage

Point mean:
- `QB_PASS_SYNTHESIS_V1 / M89-M90`.

Player-specific inputs include:
- strictly-prior eight-game QB attempts;
- strictly-prior eight-game QB YPA;
- stable Player Identity v3 lookup.

Environment inputs include:
- offense true PROE;
- pace;
- pass rate;
- plays;
- opponent pass EPA/success/YPA allowed;
- pass rate faced;
- offense/defense pressure;
- venue environment.

C2 can separately own distribution shape for selected QBs.

Interpretation:
QB pass-yard mean is already substantially individualized and should **not** be replaced by a generic Player State model.

The remaining limitation is that the M89/M90 ridge is additive: individual QB history and team/opponent context coexist in one model, but it does not explicitly learn a unique matchup-response function for every QB identity.

### WR receiving — individualized entitlement is already real

WR1:
- M38 hierarchy remains the immutable anchor.

WR2+:
- `WR_R15_PRODUCTION_MODEL_V1` is a promoted individual entitlement specialist.

WR-R15 uses strict-prior player participation information including:
- baseline individual secondary-room share;
- same-team prior offensive snap share;
- prior-any-team offensive snap share;
- prior-1 / prior-3 participation;
- same-team continuity counts;
- relative snap share within the current WR2+ room.

It redistributes only the conserved WR2+ room mass.

Interpretation:
WR opportunity is **not** merely generic position treatment.

Open gap:
- receiving efficiency (catch / YPT / YPR / YAC / ceiling) remains largely canonical after entitlement;
- opponent effects are not a player-specific WR matchup-response function.

### TE receiving — individualized entitlement is already real

`TE_R5P_PRODUCTION_MODEL_V1` is a promoted TE entitlement specialist.

Its frozen 16-feature model includes:
- baseline TE room share;
- finite TE pool state;
- room size;
- strict-prior same-team and any-team offensive participation;
- prior-1 / prior-3 snap information;
- relative snap share inside the TE room;
- continuity / availability indicators.

It may redistribute only the already-conserved TE room.

Interpretation:
TE target opportunity is already strongly player-centric.

Open gap:
- TE catch / yard efficiency and matchup response remain mostly canonical/shared after entitlement.

### RB — largest current player-state gap outside Week 1

Week-1-only promoted specialists include:
- `RB_P3_SYNTHESIS_V1` rushing;
- R26 receptions;
- R22 receiving-tail logic.

But those authorities are intentionally Week-1 scoped.

For non-Week-1 slates, current lineage states:
- RB rush yards: canonical calibrated rush-yards ensemble + joint MC;
- RB receiving: finite team target pool + canonical RB receiving entitlement;
- RB rush+receiving: V2 conservation of the final standalone authorities.

Thus the current Week-5-era RB stack does **not** have the same promoted multiseason individual-room entitlement specialist that WR and TE have.

The repo's own continuity record already identified this exact open lane:
`team rushing opportunity -> finite RB room -> individual backfield entitlement -> separate efficiency`.

### ATD

ATD remains generic football-only joint-MC/red-zone machinery without a dedicated individual entitlement probability specialist.

## Revised interpretation

The full current architecture is:

- QB: strong player + environment mean specialization;
- WR: strong individual target entitlement, generic/shared efficiency response;
- TE: strong individual target entitlement, generic/shared efficiency response;
- RB: partial player history baseline, but no promoted non-Week-1 multiseason room-allocation specialist;
- all positions: limited explicit player-specific response to the same opponent environment.

Therefore the next work should **not** be one generic cross-position Player State replacement.

It should fill the uncovered seams while retaining every promoted authority.

## Priority order authorized by this addendum

1. RB non-Week-1 individual room/allocation state.
2. WR/TE/RB player-specific efficiency × environment response, only after source/anti-retest review.
3. Distribution/tail individualization remains a separate lane and must respect already-frozen right-tail and PD2 authorities.
4. QB mean is not reopened absent materially new evidence.

No current production authority is invalidated.
No production change is authorized by this addendum.
