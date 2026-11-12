# TNF 2026 Week 10 card, built without the pricing stage

**Event:** NYG @ WAS, kickoff 2026-11-13 01:15 UTC
**Board:** `data/market_track_record/alt_props/2026_wk10_thu6.csv`, paid pull at 2026-11-12T22:30:37Z
**Evaluated rungs:** `data/research/tnf_2026_wk10_alt_card.csv` (424)
**Builder:** `scripts/research/build_event_alt_card_v1.py`

## Why this card exists outside the pipeline

Full Slate run 43320022375 bought the odds and then failed at step 33, so no
`outputs/props_priced_clean.csv` exists for this slate. Rather than lose a slate
to a pipeline defect, the card was built from the archived paid board plus
nflverse weekly actuals.

**This card is missing gate G3 (cushion).** G3 is the only gate that consumes
`model_proj`, so eight of the nine alt gates are applied and one is not. Any
citation of this card must carry that caveat. It is not evidence about the
model's calibration, because the model did not price it.

No sportsbook variable constructs a projection here. Price is used to compute
breakeven for comparison, and line availability is used for selection. Actuals
come from nflverse. That is the AGENTS.md section 3 boundary.

## Selected plays

| play | price | book | record | floor | median | breakeven | cushion |
|---|---|---|---|---|---|---|---|
| Stefon Diggs (WAS) rec yds o29.5 | -182 | FanDuel | 7/7 | 33 | 44 | 64.5% | +3.5 |
| Malachi Fields (NYG) rec yds o14.5 | -260 | FanDuel | 8/8 | 19 | 27.5 | 72.2% | +4.5 |

Diggs' full log: 55, 47, 33, 35, 44, 38, 51 across weeks 1-4, 6, 8, 9. Range
33-55 on 5-9 targets every week. He has no game below 33.

Fields' full log: 25, 30, 19, 52, 20, 19, 83, 40 across weeks 1-7, 9.

## Binding criterion

`own_rate` on 7-8 games is a small-sample frequency, not a probability. This is
the same frequency-counting that produced the week-9 overstatement (picked side
claimed 63.8%, delivered 51.7%). The criterion that carries this card is
`margin`: the player's **worst observed game** minus the threshold. A positive
margin says every game on record clears the line, which does not depend on
treating 7/7 as 100%.

Week 9 also showed the shallow band (<0.85x line) returning -4.3% bet blind, so
shallow is not safe in itself. These rungs are selected on a player-specific
floor, not on being shallow.

## Rejected, and why

| candidate | reason |
|---|---|
| Antonio Williams rec yds o14.5 (-587 DK) | cushion +0.5 against a 15-yard game already on record, at 85.4% breakeven |
| Rachaad White rush yds o14.5 (-1400 FD) | 93.3% breakeven from a 7-carry back; outside the G9 price band |
| Treylon Burks | DNP on the week-10 report and a genuine 0-yard floor (week 2) |
| Malik Nabers | efficiently priced at every rung; best edge is +1.4 pts at o59.5 |
| Chig Okonkwo receptions o1.5 (-1450) | cushion +0.5 at 93.5% breakeven |

### Cam Skattebo: passed, not played

Skattebo's rushing log is 81, 36, 60, 58, 59, 124 -- **floor 36**, and his
rush+rec o49.5 at DK -456 clears the floor test (floor 55, cushion +5.5, 6/6).

He is nonetheless not on the card. He played weeks 1-5 and 7 and missed 6, 8
and 9. The week-10 club report has him at full practice participation with no
designation, and three books price a full ladder, so the state is `AVAILABLE`
rather than `UNRESOLVED_CONTEXT` -- but workload in a first game back after a
two-game absence is not established by any of that evidence. At -1020 on the
o29.5 the price asks for 91% and leaves nothing for a reduced-workload return.

An earlier statement in session
`session_01KGZg4tX2Es2h2BAF7HpHvk` that Skattebo had "a 0-yard floor game" was
wrong: that was his week-1 **receiving** line (0 catches), not rushing. The
reason to pass is the absence, not a zero.

## Market-exposed roster facts

Both quarterbacks in this event carry full 38-rung pass-yards ladders:
J.J. McCarthy (NYG) and Jayden Daniels (WAS). An earlier note in this lineage
about Marcus Mariota holding WAS QB1 in PlayerForm while the market priced
Daniels does **not** apply to this event -- Mariota is not in it. That
observation needs re-derivation from its original artifact before it is carried
into the handoff.

2026 roster state relevant here, read from nflverse week-9 rows rather than
asserted: Diggs, Rachaad White, Okonkwo and Ekeler are WAS; McCarthy, Likely,
Singletary, Fields, Nabers and Skattebo are NYG.
