# Current Player Availability — Production Promotion V1 Frozen Plan

Status: `FROZEN_BEFORE_PRODUCTION_PROMOTION_IMPLEMENTATION`

## Promotion authority

The first complete immutable integration result is PASS:

- source certification branch `ops-current-player-availability-35gate-cert-v1`
- source head `76d01dd8e7b26ef8921cd70c18f27957da46c560`
- source run/job `34461561636` / `102820358570`
- source gates 1-34 evidence artifact `10145975346`
- source evidence digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- finalized 35/35 wrapper run `34463888613`
- finalized result artifact `10146675272`
- finalized result digest `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- disposition `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`
- passed gates `35/35`
- production promoted in certification result: `false`

Frozen parent integration plan remains `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_V1_FROZEN_PLAN.md` blob `54eb4629c48062fcaef3153918b2069238584d0a`.

## Production objective

Promote only the certified current-roster/current-availability input plumbing into the canonical Full Slate production path. Do not change scientific model parameters, trained artifacts, historical results, or sportsbook-to-football boundaries.

## Frozen production build order

For the requested slate:

1. authoritative schedule/kickoff timestamps;
2. timestamped Ourlads depth/status evidence;
3. weekly injury evidence;
4. official NFL inactive evidence;
5. T-75 game certification;
6. `current_player_availability.csv`;
7. `roles_current_production_eligible_v1.csv` containing only eligible current roles with provenance;
8. TeamForm/QB team context using the already-certified Week-1 strict-prior wrapper when resolved week is 1;
9. PlayerForm from the explicit current-role artifact;
10. canonical ML/state/rule/ensemble context;
11. RB P3 from current eligible roles;
12. complete QB C2 state-context source remains 32-team where required, while current-output coverage uses the exact eligible team set;
13. M38 -> TE-R5P -> WR-R15 -> QB C2 -> R22 -> R26 -> outer P3 conservation on the eligible football universe;
14. sportsbook matching/pricing only after football eligibility exists, and never able to resurrect unavailable/withheld players.

## Exact promotion implementation class

The production branch may add the previously certified availability/provider/build/wrapper/helper files and tests from the candidate/certification lineage. The canonical `full-slate.yml` may be reordered/wired only as needed to execute the frozen build order and pass `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` to current-role consumers.

For current-slate team-count guards, production must use the already-certified exact eligible-team semantics:

- legacy/no-explicit-availability mode retains the existing 32-team requirement;
- explicit `ACTIVE_ROLES_CSV` mode requires exactly the certified eligible team set;
- the separate complete 32-team QB state-context source-integrity guard remains unchanged.

The promotion may implement these guards directly or invoke the exact locked value-neutral seam transformers before the production pricing stack. Whichever form is chosen must be locked before post-promotion verification and must preserve the certified semantics exactly.

## Prohibited changes

No changes to:

- M89/M90 scientific parameters or artifacts;
- QB C2 selector/distribution science;
- M38, WR-R15, TE-R5P parameters/artifacts;
- RB P3 parameters/artifacts;
- R26 parameters/artifacts/vacancy science;
- R22 tail pools/parameters;
- historical research or backtest results;
- T-75 threshold or availability hierarchy;
- QUESTIONABLE/DOUBTFUL eligibility semantics;
- sportsbook inputs into roster, role, target, carry or pass opportunity.

## Required post-promotion verification

Before merging/promoting to `main`, run a dedicated no-odds Full Slate production verification from a clean checkout of this promotion branch. It must demonstrate:

- locked 35/35 certification authority is exact;
- availability core/provider blobs are exact;
- current-role artifact has provenance and no definitive unavailable rows;
- withheld games are excluded from the football universe;
- PlayerForm/P3/R26/R22/QB/WR/TE current outputs contain no definitive unavailable positive opportunity;
- M38/TE-R5P/WR-R15 conservation still passes;
- eligible-team coverage guards pass on the exact certified team set;
- complete QB state-context source integrity remains 32 teams;
- no sportsbook input is used to define football;
- strict repository and 2026 production-readiness audits pass;
- scientific model/artifact hashes remain unchanged relative to the certified production authority.

Only a complete verification PASS may be promoted to `main`. Any mechanical failure is preserved and repaired only under a separately frozen minimum plumbing repair. Any semantic/scientific failure means no promotion.

## Post-main verification

After main promotion, execute Full Slate again from `main` with live odds disabled. Production is not declared complete until that clean-main run passes the same availability/current-role and promoted-stack invariants.
