# PRISM Player Reliability Frontier V1 — Anti-Retest / Forward-State Audit

Date: 2026-10-16

Status: **AUDIT COMPLETE / PLAYER-SPECIFIC RELIABILITY EVIDENCE PRESERVED / NO NEW SCORE**

Model:
**PRISM — Player Role & Individual Situation Model**

## 1. Purpose

After two team-side opportunity-quality mechanisms closed and the
opponent/environment boundary audit found no clean new candidate that can be
scored without violating current instructions, this audit reconciles the
already-completed **named-player reliability** science.

This is intentionally player-first.

Reliability answers:

> How much confidence should PRISM place in this named player's existing
> football projection, given how difficult this same player has historically
> been to project under strictly-prior information?

Reliability does **not** answer:
- over vs under;
- mean direction;
- sportsbook value by itself;
- whether a player is "good" or "bad."

## 2. WR opportunity-difficulty trust — QUALIFIED

Disposition:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Authority:
- run `38823063793` — SUCCESS;
- exact head `35a10d09ba92fd1218d37e812768fda5d2f4023e`;
- artifact `12013153664`;
- digest `sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c`.

The named WR's strictly-prior opportunity-projection difficulty materially
separates reliable from fragile final WR-R15 receiving-yard projections.

2024 independent confirmation:
- primary rank association `+0.28453`;
- HIGH_TRUST absolute error `14.78 yd`;
- LOW_TRUST `30.45 yd`;
- player-cluster CI `[+0.23014,+0.33658]`;
- game-cluster CI `[+0.24337,+0.32665]`.

No mean or width change is authorized by this trust result.

## 3. WR efficiency-difficulty trust — QUALIFIED

Disposition:
`WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Authority:
- run `38826418315` — SUCCESS;
- exact head `c15f70f3bacf9d0d7f301d334419841ad6d9d492`;
- artifact `12015303274`;
- digest `sha256:cbab5ac5b424861d7dfd1fc8e965b6ad579936eab5e53820019392e25f2b6cdf`.

This is incremental to opportunity difficulty.

2024 confirmation:
- primary partial association `+0.16065`;
- HIGH_EFFICIENCY_TRUST final abs error `15.72 yd`;
- LOW_EFFICIENCY_TRUST `29.60 yd`;
- player-cluster CI `[+0.11340,+0.20499]`;
- team-week cluster CI `[+0.11521,+0.20595]`.

No directional betting rule is authorized.

## 4. WR dual reliability integration — HISTORICALLY SUPPORTED / PROSPECTIVE REQUIRED

Disposition:
`WR_DUAL_RELIABILITY_TRUST_HISTORICAL_INTEGRATION_SUPPORTED__PROSPECTIVE_REQUIRED`

Authority:
- run `38827426902` — SUCCESS;
- exact head `ca4da46fe07341900263dacc8602a7aef002ed03`;
- artifact `12015229072`;
- digest `sha256:2b27705844beaf9c35cf181e6fc5b7efff2c029df0255f035ba8c88bf24722c2`.

Frozen no-fit integration:

`dual_fragility_score = (opportunity_fragility_rank + efficiency_fragility_rank) / 2`.

2024 confirmation:
- opportunity-only association `+0.28453`;
- efficiency-only `+0.29458`;
- dual `+0.32334`;
- dual minus opportunity `+0.03881`;
- player-cluster gain CI `[+0.01875,+0.05956]`;
- team-week gain CI `[+0.01973,+0.05795]`.

Historical integration is not prospective promotion.

## 5. WR dual prospective confirmation — ALREADY FROZEN, STARTS WEEK 7

Frozen plan:
`docs/research/WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMATION_V1_PLAN.md`

Target weeks:
**2026 Weeks 7, 8, 9, 10**.

Important:
- Week 6 is explicitly **not** a composite target week;
- no Week-6 composite may be backfilled;
- fixed horizon / no optional stopping;
- final scientific disposition after Week 10 only, unless fixed support is
  insufficient;
- no sportsbook;
- no OddsAPI;
- no mean or width change.

The component Week-6 WR locks remain separate prospective evidence.

## 6. WR Week-6 component locks — FROZEN / DO NOT GRADE EARLY

Opportunity trust:
- run `38823520954`;
- artifact `12012884262`;
- supported WRs `141`;
- immutable prospective labels.

Efficiency trust:
- run `38826804491`;
- artifact `12015596169`;
- supported WRs `141`;
- immutable prospective labels.

Both read zero Week-6 outcomes at lock.

Do not combine them into a Week-6 dual score post hoc.

## 7. TE allocation-difficulty trust — QUALIFIED

Disposition:
`TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Authority:
- run `38821176218` — SUCCESS;
- exact head `270f01266bff125cfaccb53a57b378674d919901`;
- artifact `12013570473`;
- digest `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`.

2025 confirmation:
- primary rank association `+0.1153773`;
- player-cluster CI `[+0.0419128,+0.1853534]`;
- game-cluster CI `[+0.0467862,+0.1828009]`;
- HIGH_TRUST standardized CRPS `0.67012`;
- LOW_TRUST `1.09274`;
- raw absolute mean error `8.7035 yd` vs `23.5957 yd`.

This is successful same-player reliability evidence around the unchanged TE
projection.

## 8. TE Week-6 trust lock — FROZEN / DO NOT GRADE EARLY

Authority:
- run `38822346088` — SUCCESS;
- artifact `12013018536`;
- supported TEs `103`;
- HIGH_TRUST `25`;
- LOW_TRUST `26`;
- Week-6 outcomes read at lock: `0`.

The labels are not over/under signals.

## 9. TE efficiency-trust V1 — CLOSED

Disposition:
`NO_ACTIONABLE_TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY`.

The point direction was favorable but 2025 independent-confirmation clustered
intervals crossed zero.

Do not rescue with:
- pooled seasons;
- alternate windows;
- TE1 subsets;
- removal of opportunity control;
- alternate partial-correlation transforms.

No Week-6 TE efficiency lock exists by design.

## 10. Generic efficiency-volatility / tail-stability V1 — CLOSED

Disposition:
`PLAYER_EFFICIENCY_TAIL_STABILITY_NULL`.

Canonical:
- run `38408536257`;
- artifact `11819986612`;
- digest `sha256:4bb8bead9f3a40f6465909c2148111d0c2aec9ff0b21a6b85fdc6bb27a0bb1e5`.

The simple idea that recent per-opportunity efficiency SD predicts next-game
tail risk did not replicate after controls.

Do not reopen with alternate windows, buckets, role slices or matchup
interactions.

## 11. TE allocation-only difficulty — CONFIRMED upstream evidence

Disposition:
`TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`.

Canonical:
- run `38656442803`;
- artifact `11936407975`.

This confirms that TE player-specific allocation difficulty is not merely team
pass-volume noise.

Its downstream trust/selectivity layer is already qualified above.

WR's exact allocation-only reinterpretation closed and must not replace the
existing WR opportunity/Dirichlet authority.

## 12. RB reliability state

No analogous new PRISM named-RB trust/selectivity state is qualified by the
current individualization program.

Do not mislabel RB-PD2 mean-neutral width calibration as the same scientific
object as WR/TE named-player trust.

Any new RB reliability study requires a separate anti-retest audit because:
- M95/M96 families are protected;
- generic efficiency volatility V1 is closed;
- opportunity/role allocation research has extensive frozen lineage.

## 13. Ticket-construction interpretation

This is the exact separation PRISM must preserve:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> TICKET CONSTRUCTION`.

Trust may eventually help answer:
- which otherwise-valid player projection deserves more confidence;
- which edge is too fragile to anchor a ticket;
- where an alt line is preferable to a main line;
- how aggressively a player should be concentrated across tickets.

Trust does **not** by itself answer:
- over/under;
- bet/pass;
- what sportsbook line should be;
- whether an edge is real.

That prevents reliability from becoming another post-hoc selector.

## 14. Current forward boundary

Immediate prospective obligations:
1. leave all Week-6 WR/TE trust locks immutable;
2. do not grade them until the required games settle;
3. do not manufacture a Week-6 WR dual composite;
4. begin the frozen WR dual prospective confirmation with Week 7 exactly as
   planned;
5. Weeks 7-10 are fixed; no optional stopping.

The Week-7 dual lock itself cannot legitimately use Week-6 history until Week 6
is final. Lock mechanics may be prepared in advance, but the target-week row
artifact must satisfy the frozen chronology at execution.

## 15. Integrity

This audit:
- scores new outcomes: 0;
- reads Week-6 outcomes: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- fits models: 0;
- changes means: 0;
- changes distributions: 0;
- changes production: false.

Disposition:

`PRISM_PLAYER_RELIABILITY_FRONTIER_RECONCILED__PROSPECTIVE_LOCKS_GOVERN_NEXT`.
