# Testing the operator's MNF week-11 reads: PREREGISTRATION V1

**Date:** 2026-11-26. Written and committed before any result was computed.

**What this is.** After his CIN @ WAS ticket went 6/6 (about +511), the operator explained his
read of that game (rule 24 in `YOUR_RULES_PLAIN_ENGLISH.md`). The session promised to test the
three concrete pieces of it before the next single-game card.

**What this is NOT.** A new gate, filter or rule. Rule 24 says his football reading is fluid and
must not be turned into a checklist. The question here is narrower:

> When a game looks like that, **does the market already price it in**, or did reading it
> correctly pay?

The answer informs his judgment on a game. It never replaces it.

## Data

**Main lines:** `data/research/main_line_screens/main_line_screens_rows_v1.csv.gz`, 17,713 rows
with posted over and under prices and actual results:
- 2024-25: 15,157 rows. This is the main test.
- 2026: 2,556 rows. This is the check.

**Game context:** nflverse.
- Schedules supply the game total.
- Weekly stats and snap counts supply who played and the strictly-prior target leaders.

**Alt rungs:** `data/market_track_record/graded/w10_*` and `w11_*` alt boards (2026 weeks
10-11, with `model_proj`).

**Scoring:**
- Every bet is the OVER at its posted over price.
- A row is void if the player took no offensive snap.
- 95% CIs come from a bootstrap clustered by game (2,000 resamples, seed 20261126).

## T1: shootout totals and pass-catchers

**Rows:** WR, TE and RB `rec_yards` and `receptions` main-line OVERS, both teams.

**Split:** game total ≥ 49.5 ("shootout") vs < 49.5.

**PASS:** in 2024-25, shootout OVER ROI > 0 with the CI excluding 0, and 2026 has the same sign.

## T2: a missing top receiver and the tight ends and backs

**Missing:** the team's top target earner is missing when:
- he is the WR/TE with the most targets in the team's earlier games this season (strictly
  prior, at least 3 games played);
- he takes no offensive snap in this game.

Inactives are announced before kickoff, so this is pregame-knowable.

**Rows:** that team's TE and RB `rec_yards` and `receptions` main-line OVERS, compared with the
same rows when the top target earner plays. WRs are reported separately and do not gate.

**PASS:** in 2024-25, missing-receiver TE/RB OVER ROI > 0 with the CI excluding 0, and 2026 has
the same sign.

## T1 + T2 together (his whole read)

TE/RB receiving OVERS in shootout games where the team's top target earner is missing.
Reported, not gated, because it will be small.

## T3: rungs near the projection, single games vs full slates

**Rows:** alt OVER rungs with cushion (`model_proj` / rung) between 1.0 and 1.25, priced -150 to
-700 (his legs ranged from -174 to -670), 2026 weeks 10-11.

**Split:** primetime single games (kickoff after 7 pm ET on Thursday, Sunday or Monday) vs the
Sunday day slate.

**Descriptive only.** The primetime n will be small (about 4 games). It is reported with
counts, never as a pass or fail.

## What happens with the result

- **Pass:** the result is written into the rules file next to rule 24 as **context he can use
  when a game looks like that**. It is not a gate.
- **Null:** that is written too. Reading those games may still be right, but the market already
  prices the average case.

Either way, nothing is added to `build_alt_cards_v1.py`.
