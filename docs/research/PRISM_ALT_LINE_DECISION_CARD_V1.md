# PRISM — ALT-LINE DECISION CARD V1 (market-anchored)

**Date:** 2026-11-04 · **Source:** `data/research/alt_line_market_anchored_table_v1.csv`
· **Study:** `PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md` · **MARKET-ASSISTED**

**How to use it, per bet:**

1. Read the player's **main line** off the app. Find the **band** row for that market.
2. Take the **alt threshold** the book offers. Compute `threshold / main line`.
3. Read across to the nearest column. That is your **clearance %**.
4. The book's price must be **better (less negative) than the `need` figure**. Use the
   **`safe`** figure, which is the lower 95% bound clustered over players, when you want margin.
5. Skip `receptions` for now — it is an integer market and this parameterisation is wrong for it (study §6).

**Reading a sign flip.** Where a cell shows `need -126 / safe +110`, the point estimate is above
50% but its lower bound is below it. The proposition is a coin flip within sampling error, so the
margin-of-safety price is plus-money. Treat those cells as **no bet** unless the book is offering
plus-money.

**Legs are per-bet and they correlate.** Five 85% legs are 0.85^5 = 44% if independent,
and two legs in the same game are worse than independent.

---

## `rush_rec_yards`

| main line | ×0.40 | ×0.50 | ×0.60 | ×0.70 | ×0.80 | ×0.90 | ×1.00 | ×1.25 | ×1.50 |
|---|---|---|---|---|---|---|---|---|---|
| **<= 32.5**<br><sub>n=462, 140 players</sub> | 76%<br>need -320<br>safe -224 | 73%<br>need -264<br>safe -187 | 67%<br>need -206<br>safe -147 | 64%<br>need -178<br>safe -128 | 60%<br>need -148<br>safe -107 | 56%<br>need -126<br>safe +110 | 52%<br>need -107<br>safe +130 | 42%<br>need +138<br>safe +196 | 36%<br>need +180<br>safe +260 |
| **32.5-51.5**<br><sub>n=506, 134 players</sub> | 81%<br>need -438<br>safe -297 | 76%<br>need -322<br>safe -223 | 71%<br>need -240<br>safe -169 | 65%<br>need -184<br>safe -131 | 60%<br>need -148<br>safe -106 | 55%<br>need -120<br>safe +117 | 49%<br>need +102<br>safe +144 | 38%<br>need +166<br>safe +241 | 29%<br>need +249<br>safe +376 |
| **51.5-71.5**<br><sub>n=470, 105 players</sub> | 85%<br>need -553<br>safe -350 | 79%<br>need -380<br>safe -249 | 73%<br>need -264<br>safe -178 | 67%<br>need -207<br>safe -141 | 63%<br>need -167<br>safe -114 | 56%<br>need -125<br>safe +117 | 50%<br>need +100<br>safe +147 | 32%<br>need +211<br>safe +331 | 22%<br>need +356<br>safe +614 |
| **> 71.5**<br><sub>n=499, 72 players</sub> | 95%<br>need -1748<br>safe -840 | 90%<br>need -918<br>safe -499 | 83%<br>need -501<br>safe -296 | 75%<br>need -306<br>safe -189 | 69%<br>need -218<br>safe -137 | 60%<br>need -147<br>safe +108 | 52%<br>need -107<br>safe +149 | 30%<br>need +235<br>safe +418 | 15%<br>need +557<br>safe +1343 |

## `rush_yards`

| main line | ×0.40 | ×0.50 | ×0.60 | ×0.70 | ×0.80 | ×0.90 | ×1.00 | ×1.25 | ×1.50 |
|---|---|---|---|---|---|---|---|---|---|
| **<= 15.5**<br><sub>n=305, 72 players</sub> | 73%<br>need -267<br>safe -167 | 70%<br>need -235<br>safe -147 | 67%<br>need -205<br>safe -129 | 61%<br>need -158<br>safe -100 | 59%<br>need -144<br>safe +110 | 56%<br>need -126<br>safe +126 | 54%<br>need -116<br>safe +137 | 49%<br>need +105<br>safe +168 | 43%<br>need +135<br>safe +221 |
| **15.5-32.5**<br><sub>n=346, 76 players</sub> | 74%<br>need -280<br>safe -176 | 68%<br>need -215<br>safe -137 | 64%<br>need -179<br>safe -115 | 60%<br>need -151<br>safe +104 | 54%<br>need -119<br>safe +132 | 50%<br>need -101<br>safe +156 | 45%<br>need +122<br>safe +195 | 36%<br>need +175<br>safe +291 | 27%<br>need +264<br>safe +474 |
| **32.5-56.5**<br><sub>n=328, 66 players</sub> | 86%<br>need -598<br>safe -339 | 80%<br>need -413<br>safe -244 | 73%<br>need -264<br>safe -162 | 66%<br>need -193<br>safe -119 | 58%<br>need -138<br>safe +117 | 53%<br>need -112<br>safe +146 | 48%<br>need +108<br>safe +177 | 34%<br>need +193<br>safe +340 | 23%<br>need +337<br>safe +685 |
| **> 56.5**<br><sub>n=343, 47 players</sub> | 90%<br>need -853<br>safe -419 | 85%<br>need -586<br>safe -305 | 79%<br>need -376<br>safe -206 | 73%<br>need -269<br>safe -151 | 64%<br>need -179<br>safe -102 | 57%<br>need -132<br>safe +134 | 51%<br>need -105<br>safe +170 | 34%<br>need +193<br>safe +386 | 19%<br>need +428<br>safe +1191 |

## `rec_yards`

| main line | ×0.40 | ×0.50 | ×0.60 | ×0.70 | ×0.80 | ×0.90 | ×1.00 | ×1.25 | ×1.50 |
|---|---|---|---|---|---|---|---|---|---|
| **<= 15.5**<br><sub>n=672, 166 players</sub> | 76%<br>need -312<br>safe -225 | 73%<br>need -265<br>safe -193 | 69%<br>need -226<br>safe -165 | 65%<br>need -190<br>safe -139 | 62%<br>need -165<br>safe -121 | 59%<br>need -146<br>safe -108 | 56%<br>need -128<br>safe +106 | 49%<br>need +106<br>safe +144 | 42%<br>need +139<br>safe +191 |
| **15.5-27.5**<br><sub>n=788, 177 players</sub> | 79%<br>need -375<br>safe -269 | 75%<br>need -294<br>safe -215 | 69%<br>need -223<br>safe -165 | 66%<br>need -194<br>safe -144 | 60%<br>need -149<br>safe -111 | 56%<br>need -128<br>safe +104 | 52%<br>need -108<br>safe +124 | 44%<br>need +126<br>safe +170 | 37%<br>need +171<br>safe +235 |
| **27.5-44.5**<br><sub>n=733, 138 players</sub> | 83%<br>need -486<br>safe -329 | 77%<br>need -339<br>safe -236 | 71%<br>need -251<br>safe -177 | 66%<br>need -198<br>safe -141 | 61%<br>need -155<br>safe -111 | 56%<br>need -129<br>safe +108 | 51%<br>need -104<br>safe +134 | 39%<br>need +155<br>safe +222 | 29%<br>need +243<br>safe +363 |
| **> 44.5**<br><sub>n=737, 90 players</sub> | 86%<br>need -616<br>safe -373 | 81%<br>need -415<br>safe -263 | 75%<br>need -294<br>safe -191 | 69%<br>need -225<br>safe -148 | 63%<br>need -171<br>safe -113 | 56%<br>need -127<br>safe +118 | 50%<br>need +101<br>safe +154 | 33%<br>need +201<br>safe +325 | 24%<br>need +319<br>safe +563 |

---

## What this card does not tell you

It gives the price you **need**. It does not give the price the book **offers** — no
alt-line prices exist in this repository, only main lines. You supply the offered price.

## Known defects, disclosed

- The shipped stratification misses decile calibration by **0.38pp** (3.38 vs a 3pp bar),
  which is why the **`safe`** column exists and should be preferred.
- **`receptions` is omitted** from this card: integer thresholds collapse the fractions.
- Bands are fixed from the 2024 sample; `pass_yards` rests on 35 players and `rush_yards`/
  `rush_rec_yards` on 137/114, all under the 150-player power bar.
