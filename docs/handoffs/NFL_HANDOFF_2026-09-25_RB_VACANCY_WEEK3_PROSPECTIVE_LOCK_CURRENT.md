# NFL HANDOFF — 2026-09-25 — RB VACANCY WEEK-3 PROSPECTIVE LOCK CURRENT

Repo: `dkaps6/imtiredofthis`

GitHub is canonical over chat memory.

## Current production state

Production behavior is unchanged by RB Vacancy Opportunity V1. This lane is research-only.

Canonical main at the moment the Week-3 pregame source was frozen:
`f7d2011b73950488ea209124ba895b92c401b2b1`

Protected production science remains unchanged.

## Latest completed receiver result

WR Anchor / Role-Transmission Audit V1:
**`WR_ANCHOR_ROLE_TRANSMISSION_NO_GAP_CLOSED`**

Authoritative run:
`36203881629`

Do not reopen that participation/hierarchy family.

## Active RB science state

RB Vacancy Opportunity V1 has its **first legitimate prospective qualifying cohort**.

### Frozen availability events

Canonical no-live-odds Full Slate:
- run `36204768034`
- source main SHA `f7d2011b73950488ea209124ba895b92c401b2b1`
- artifact `10892728623`
- digest `sha256:f52b36fb7a9c929fadca473bd8303823fb9f63bd42c884341cd6c3b52c26ed67`
- sportsbook upstream = 0

Qualifying definitive-unavailable RBs:
- DEN: Jonah Coleman
- PIT: Rico Dowdle

### Frozen no-outcome V1 transfer state

- run `36205201005`
- artifact `10893677385`
- digest `sha256:69b55421be779cb474eea0735bfe98490d8c7d5a8e0847e96f1f65d1c5e9e732`
- candidate rows = 4
- exclusions = 0
- outcomes attached = 0

Transfers:
- DEN Dobbins +0.142857143
- DEN Harvey +0.214285714
- PIT Nowakowski +0.012338425
- PIT Warren +0.292009401

### Exact-production pregame baseline/candidate lock

- run `36205758768`
- job `108301827908`
- artifact `10893588588`
- digest `sha256:54437c69f0a66c2c9bb97c520f3e8d9c933e0203dac39fe1fde5352dafa3eeea`
- exact production source `f7d2011b73950488ea209124ba895b92c401b2b1`
- 25,000 MC draws / seed 42
- active RB/FB cohort = 5
- projection rows = 10
- outcomes attached = 0
- sportsbook inputs = 0
- candidate changed only `rules_rush_share`
- YPC, ML, State, ensemble weights and every other football input unchanged

Locked projection movement:

DEN:
- JK Dobbins rush att 12.293331 -> 12.274076; rush yds 47.951505 -> 47.789661
- RJ Harvey rush att 7.672867 -> 8.260694; rush yds 35.579024 -> 39.787778

PIT:
- Jaylen Warren rush att 12.288710 -> 13.116147; rush yds 49.947595 -> 56.820052
- Riley Nowakowski rush att 2.524458 -> 2.233885; rush yds 16.466599 -> 14.208183
- Eli Heidenreich (no direct transfer) rush att 2.524197 -> 2.170025; rush yds 16.441968 -> 13.704573

## Important structural finding before outcomes

The canonical simulator already normalizes active rushing weights to a finite 95% modeled team allocation.

Baseline raw rushing-share sums are already above 0.95:
- DEN 1.192960
- PIT 1.387306

Therefore V1 does **not** restore vacated carries one-for-one. Under the real production architecture it primarily changes **who receives an already-finite room allocation**. This is why nonrecipient Heidenreich moves and why some direct recipients can fall after normalization.

Do not hide or repair this after outcomes. It is part of the locked production effect.

## Frozen Week-3 grading

Read:
- `docs/research/RB_VACANCY_OPPORTUNITY_V1_WEEK3_PROSPECTIVE_LOCK.md`
- `docs/research/RB_VACANCY_OPPORTUNITY_V1_WEEK3_EVALUATION_CONTRACT.md`

When DEN-LAR and PIT-CIN are final:
- attach canonical `rushes` and `rush_yards` once;
- grade all 5 locked active RB/FB rows;
- report direct-recipient subset separately;
- report each team separately;
- report M96A high-volume actual-carry slices 20+ and 25+;
- use no sportsbook fields in the football comparison;
- do not change V1 after seeing outcomes.

Week 3 is observational only and cannot promote V1 by itself.

## What to do while RB V1 waits for outcomes

Continue a **separate read-only structural audit** rather than retuning any closed family.

Preferred unresolved frontier:
**post-specialist cross-market football contradictions / lost opportunity consistency**.

Start from current production artifacts and answer whether independently promoted specialist layers leave final football means internally inconsistent across identities that should hold together, especially:
- rush attempts × frozen efficiency vs rushing yards;
- receptions/targets × efficiency vs receiving yards;
- RB rush + receiving component identities;
- room/team finite-volume conservation after specialist overrides.

This must begin diagnostic-only:
- candidate variants scored = 0;
- parameters fit = 0;
- sportsbook inputs upstream = 0;
- no production mutation;
- no retuning of M96, receiver-room, WR hierarchy, TE Width, or prior closed specialists.

If a concrete contradiction is found, freeze a separate repair hypothesis before scoring anything.

## Memory-efficient restart

Read only:
1. `AGENTS.md`
2. top checkpoint of `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff
4. latest Issue #535 comments, especially WR closure and RB lock
5. live `main` and `research-rb-vacancy-opportunity-v1`

Do not recursively load old handoffs unless explicitly pointed there.
