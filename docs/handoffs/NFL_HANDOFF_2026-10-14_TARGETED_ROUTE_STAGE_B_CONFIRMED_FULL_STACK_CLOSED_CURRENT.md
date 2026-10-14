# NFL HANDOFF — 2026-10-14 — TARGETED-ROUTE STAGE B CONFIRMED / FULL-STACK MEAN INTEGRATION CLOSED

**Newest authoritative continuity for the active NFL research branch.**

Repository:
- `dkaps6/imtiredofthis`

Production main remains:
- `812ac2cde79fe3846174613a2f2777e57b34b739`

Active research:
- branch `research-individual-opportunity-roadmap-2026-10-08`
- draft PR #672
- continuity Issue #535

No production model code changed.
No paid OddsAPI call was used.

---

## 1. Read order

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. THIS FILE
4. `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_RESULT.md`
5. `docs/research/WR_TARGETED_ROUTE_MATCHUP_FULL_STACK_INTEGRATION_V1_RESULT.md`
6. `docs/research/WR_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md`
7. `docs/research/TE_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md`
8. newest Issue #535 comments
9. live PR #672 / branch / exact-head Actions

Do not recursively reread older handoffs.

---

## 2. Model standard remains player-first

This is an **individual-player projection system**.

Position-wide aggregate metrics are guardrails, not the hypothesis grain.

Canonical state remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`

Confirmed player-specific evidence remains:
- WR opportunity uncertainty confirmed;
- TE opportunity uncertainty confirmed;
- historical RB same-player yard-difficulty / MC-width qualification;
- WR targeted-route matchup Stage B confirmed.

Do not interpret a failed downstream implementation as erasing its upstream individual-player mechanism.

---

## 3. Canonical targeted-route Stage B — CONFIRMED

Corrected source authority:
- Stage-A run `38637901016` SUCCESS
- job `116105672352`
- exact source head `159b8213a734e3f4ee16fc86fbb51950099f3323`
- artifact `11928258474`
- digest `sha256:b957ff0b37f0b2d8f45232bcd44657928cecc75f5f5e6d1c292ee8e35729228c`

Fresh canonical Stage-B rerun:
- run `38641891257` SUCCESS
- job `116119186261`
- exact scored head `fe1d8ff52bd7cfa322e095a7a6f58752acb0b55f`
- artifact `11929999088`
- digest `sha256:0ff2c393e0daa0fa5163d34ce35e38eb25fb7c907cdf287019d609b804ca5128`

Disposition:
`PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED`

The corrected-source rerun reproduced the older Stage-B statistics exactly.

Confirmed:
- WR YPT route-matchup family
- WR catch-rate route-matchup family

Closed:
- WR direct receiving-yard residual family
- all TE route-matchup families

WR YPT pooled 2024-25:
- rho +0.03302
- player-cluster CI low +0.00404
- game-cluster CI low +0.00175
- depth-controlled rho +0.03213
- BH q 0.02849

WR catch pooled 2024-25:
- rho +0.05541
- player-cluster CI low +0.02285
- game-cluster CI low +0.02392
- depth-controlled rho +0.05580
- BH q 0.00300

This is genuine named-player matchup evidence:
`player targeted-route history × opponent route-family susceptibility`.

Critical semantics remain:
- participation `route` is only the primary/targeted receiver route label;
- not total routes run;
- not route participation;
- not TPRR/YPRR;
- no WR-CB assignment inference.

---

## 4. Full-stack WR mean integration — COMPLETE / CLOSED

Frozen result:
`docs/research/WR_TARGETED_ROUTE_MATCHUP_FULL_STACK_INTEGRATION_V1_RESULT.md`

Canonical full-stack run:
- `38642241032` SUCCESS
- job `116120366630`
- exact scored head `bdbc132e35939133e3fb9ee4fca1e85ff1857561`
- artifact `11930768340`
- digest `sha256:a9a0198efaca76ea8967813842468894b01c5d130781f2bb5b4c71881bfc36d3`

Mechanical blocker was repaired only by:
- pinning PyArrow compatible with repo NumPy 1.26.4;
- switching the casebook input to canonical Stage-B run `38641891257`.

No scientific construction changed.

Overall disposition:
`NO_ACTIONABLE_WR_ROUTE_MATCHUP_FULL_STACK_INTEGRATION`

YPT arm:
`NO_ACTIONABLE_WR_ROUTE_MATCHUP_YPT_INTEGRATION`
- direct MC MAE 23.18445 -> 23.19066 (worse)
- full-stack MAE 22.92198 -> 22.93119 (worse)
- full-stack RMSE worsened
- WR1 / W5-12 / CRPS / 80% coverage / 30+ / 40+ guards failed
- player median MAE delta was slightly favorable, but cannot rescue hard gates.

Catch arm:
`NO_ACTIONABLE_WR_ROUTE_MATCHUP_CATCH_INTEGRATION`
- direct MC MAE 1.56844 -> 1.56787 (+0.0364%)
- full-stack MAE 1.50869 -> 1.50771 (+0.0647%)
- below frozen >=0.50% direct-MC effect floor
- 80% coverage gap worsened
- >=2-reception miss rate worsened
- player median/fraction slightly favorable but insufficient.

Integrity:
- 1,916 player-games
- 187 named WRs
- 163 with >=4 games for player-level reporting
- all conservation / target-mass / non-WR mutation audits passed
- sportsbook 0
- OddsAPI 0
- 2026 outcomes 0
- production change false

Scientific interpretation:
- upstream WR route-matchup mechanism remains confirmed;
- exact frozen production-order **mean transmission** is closed;
- do NOT tune coefficient/window/support/route groups/subgroups/caps to rescue it.

---

## 5. Other closed lanes remain closed

Do not reopen unchanged:
- WR-R3 exact combined candidate;
- player × environment V1;
- broad 11/12/OTHER personnel-conditioned opportunity;
- generic matchup / defense-v-position;
- coverage / target-depth reuse;
- receiver conversion;
- efficiency-tail/Q90;
- NGS;
- WR-CB;
- RB retune;
- direct TE transport of the WR route result.

No paid OddsAPI without explicit authorization.

---

## 6. Clean next frontier

The targeted-route mean lane is now complete.

The strongest still-authorized player-first frontier is the separately confirmed **player-specific opportunity uncertainty / reliability** layer:

WR:
`WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

TE:
`TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

RB:
historical `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

Any next integration must:
- preserve promoted player means;
- preserve team target/carry mass;
- change only distribution/reliability mechanics;
- be frozen before scoring;
- keep WR/TE/RB lineages separate unless a later contract explicitly unifies them;
- use no sportsbook information upstream;
- make player-level probabilistic scoring primary, with aggregate metrics as guards.

Do not use the failed WR route mean integration as justification to return to position-wide averages.

