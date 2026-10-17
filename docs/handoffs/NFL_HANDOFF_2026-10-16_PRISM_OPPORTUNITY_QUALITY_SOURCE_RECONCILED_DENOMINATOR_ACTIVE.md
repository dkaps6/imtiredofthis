# NFL HANDOFF — 2026-10-16 — PRISM OPPORTUNITY QUALITY SOURCE RECONCILED / DENOMINATOR PERSISTENCE ACTIVE

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Active research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT

## 1. Non-negotiable objective

PRISM is an individual-player projection model.

Canonical order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is downstream QA.

No sportsbook input belongs upstream.
Exactly one paid Week-6 OddsAPI acquisition has already been used. Do not call OddsAPI again without explicit user authorization.

## 2. Historical opportunity quantity state

Stage 1:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`.

Stage 2:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Qualified Stage-2 cells:
- RB rushing;
- RB receiving;
- TE receiving.

Exact QB V1 is closed.
WR remains source-blocked/not scored.

Stage-2 scored authority:
- run `38957575645` SUCCESS;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

## 3. Week-6 prospective shadow — locked

Final guarded lock:
- run `38991032116` SUCCESS;
- exact head `3c0190cc8f1cd6271bfd6912cf84cff82008ff80`;
- artifact `12084848700`;
- digest `sha256:ec1376dfa90fff3b5fc31e199588f7fe5f570017946adfd0b85be847af901e6e`.

Disposition:
`PRISM_WEEK6_SUNMON_STAGE2_SHADOW_LOCKED`.

Population:
- 13 Sunday/Monday games;
- SEA-DEN excluded because it was already played before the Stage-2 freeze;
- 248 player-cell rows;
- RB rushing / RB receiving / TE receiving only.

Final source gates include:
- prior roster coverage 100%;
- official inactive team-game coverage 100%;
- prior snap team-game coverage 100%;
- inactive -> roster skill bridge 97.5535%;
- positive skill snap -> roster bridge **99.6386%**, using a pre-join denominator that retains unmatched skill rows;
- target identity >=99%;
- Stage-1 p_active attach 100%;
- canonical q finite 100%;
- max candidate mass gap 2.22e-16.

Do not grade until all included games are final.

## 4. Opportunity-quality source — reconciled ready

This is a distinct question from the terminal recent-player high-value-share persistence V1.

Closed question:
> Does recent named-player scoring-area share itself persist better than ordinary usage?

That remains closed.

Current question:
> Can PRISM separately represent how much high-value opportunity a team creates, then map that finite opportunity environment to the current named-player entitlement?

### Source authority A — Stage-2 player compatibility

Run:
- `38991429245` SUCCESS;
- head `d14ba07f7f0e79f2b3f16a0b018a50c30744be91`;
- artifact `12085262613`;
- digest `sha256:77739799f34237538d7ffcba9715939c59d6a5b50f811f06f2f5836ff9deb31d`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL`.

Source-ready cells:
- RB rush inside-10;
- RB receiving red-zone targets;
- TE receiving red-zone targets.

### Source authority B — raw full finite team pools

Canonical hardened run:
- `38991605252` SUCCESS;
- exact head `e590ce4754da2c7f7aa01beab2ee148aa5b613a0`;
- artifact `12085367606`;
- digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`.

2022-2025:
- scheduled team-game coverage 100%;
- rusher identity 100%;
- receiver identity 100%;
- strictly-prior offense state attach 100%;
- strictly-prior opponent state attach 100%;
- drive topology source-ready;
- optional end-zone input source-ready.

Optional readiness is not permission to use those fields predictively.

### Exact dual-source reconciliation

Frozen reconciliation:
`docs/research/PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILIATION_V1_FREEZE.md`.

Canonical run:
- `38991737241` SUCCESS;
- exact head `c1abb0bacd0c78dd0b38861214bd28c04cc9f376`;
- artifact `12085278127`;
- digest `sha256:0be3d571129876ce0a8df755762215e147faec2ff1ac1822e05cbd84508a12bb`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`.

Exact 2024/2025 Weeks 5-18 parity:
- 832 / 832 team-games overlap;
- 0 left-only keys;
- 0 right-only keys;
- 0 mismatch rows;
- team rush attempts max absolute drift 0;
- inside-10 rushes drift 0;
- team targets drift 0;
- red-zone targets drift 0.

Thus:
- raw PBP is the finite team-pool authority;
- the immutable Stage-2 compatibility audit remains the player-entitlement authority.

No predictive candidate has been scored.

## 5. Why a denominator-persistence step is active

The qualified Stage-2 source already reconstructs the numeric canonical pregame parent opportunity mean:

RB rushing:
`rules_plays_est * (1 - rules_pass_rate)`.

Receiving:
`rules_plays_est * rules_pass_rate`.

The published Stage-2 source artifact retained only a boolean `denominator_finite` field.

PRISM opportunity quality must remain physically attached to the existing pregame opportunity quantity rather than invent a detached scoring-area count model.

Therefore the following addendum was frozen before code change:

`docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_ADDENDUM.md`

Freeze commit:
`3a980c9ba821c5e0a3b41231314db5c21bb0adef`.

Allowed mechanical change:
persist `authority_denominator_mean`.

Not allowed:
- denominator formula change;
- entitlement change;
- source gate change;
- target outcome predictor;
- new share model;
- predictive opportunity-quality scoring.

Active exact-head run:
`38991915503`
on head:
`02e96ad9faabee62c1ecb31303b48746e4945a8c`.

The run must prove output-only parity against source artifact `12070806828`.

## 6. Exact next action

Wait only for the denominator-persistence parity result.

If and only if it passes:
1. publish the immutable denominator lineage authority;
2. freeze one opportunity-quality mechanism **before scoring**;
3. use 2024 development and 2025 independent confirmation;
4. keep the recent player high-value-share persistence formula closed;
5. use only the three historically qualified Stage-2 cells.

The first mechanism must isolate opportunity quality before adding opponent/environment transmission.

No predictive mechanism has been frozen or scored by this checkpoint.

## 7. Integrity

Current opportunity-quality source work:
- predictive candidates scored 0;
- player high-value-share rescue attempts 0;
- fitted predictive coefficients 0;
- sportsbook inputs 0;
- OddsAPI calls 0;
- 2026 outcomes 0;
- production changed false.

GitHub is canonical.
Fail closed rather than weaken a gate.
