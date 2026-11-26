# Testing the operator's MNF week-11 reads: RESULTS V1

**Preregistration:** `MNF_READS_SINGLE_GAME_TEST_PREREGISTRATION_V1.md`, committed at `0c8727e`
before any result was computed.

**Script:** `scripts/research/evaluate_mnf_reads_v1.py`. **Rows and results:** `data/research/mnf_reads/`.

**The question:** when a game looks like his CIN @ WAS read, does the market already price it,
or did reading it correctly pay? It is not a gate. Nothing here goes into the card scripts.

## Verdict

None of the three pieces is an edge **on its own, on average**. The market already prices the
average shootout and the average missing receiver. What won on Monday was reading that specific
game: Chase drawing double coverage, and Washington forced to throw. That is what rule 24
says.

## T1: shootout totals (≥ 49.5) and pass-catchers' main-line OVERS

**2024-25 (main test):**

| games | n | hit | breakeven | ROI | 95% CI |
|---|---|---|---|---|---|
| shootout | 1,462 | 51.1% | 52.7% | **-2.9%** | [-8.8%, +3.2%] |
| all other games | 9,447 | 48.0% | 52.9% | -9.6% | [-11.9%, -7.1%] |

- **Not a pass:** the CI includes 0.
- **What it does show:** receiving OVERS lose about 10% in normal games and only about 3% in
  shootouts. The books move those lines up, but not quite all the way.
- **By position in shootouts:**

  | position | 2024-25 | 2026 |
  |---|---|---|
  | TE | +4.5% [-7%, +16%] | -4.2% |
  | RB | -6.5% | +11.5% (n=107) |
  | WR | -4.6% | -14.4% |

  Nothing is consistent across seasons.

## T2: top target earner missing → the tight ends' and backs' receiving OVERS

**2024-25:**

| top target earner | n | ROI | 95% CI |
|---|---|---|---|
| missing | 448 | **-8.6%** | [-20.2%, +2.6%] |
| plays | 3,842 | -7.3% | |

- **Not a pass:** no difference. When a team's top receiver is out, the books already raise its
  tight ends' and backs' lines.
- **2026:** the missing case shows +6.0%, but n=46 (9 games) and the CI is [-37%, +51%].
- **Other WRs when the top one is missing:** -4.5% (2024-25).

## T1 + T2 together (his whole read)

| seasons | n | games | ROI |
|---|---|---|---|
| 2024-25 | 25 | 5 | -10% |
| 2026 | 4 | 1 | 4 of 4 |

Too small to say anything.

## T3: alt rungs near the projection (cushion 1.0-1.25, priced -150 to -700), 2026 weeks 10-11

| games | rungs | games | hit | breakeven | ROI |
|---|---|---|---|---|---|
| primetime single games | 86 | 6 | 70.9% | 72.1% | **-1.8%** |
| Sunday day slate | 254 | 20 | 62.6% | 72.0% | -13.1% |

- Descriptive only (6 games).
- Near-projection rungs lose badly on the Sunday slate and come out about even in primetime
  single games. That fits rule 21: single games are built differently. It does **not** show
  they are profitable.

## What goes in the rules file

**Context next to rule 24, not a rule:**
- A shootout total or a missing receiver is already in the line.
- When his read of a game goes further than that (who the defense keys on, who is forced to
  throw), the read is what has to be right; the situation alone is not enough.
- In single games, near-projection rungs have not been the money-loser they are on Sunday.
