# PRISM — W9 2026 PROSPECTIVE CAPTURE AND FROZEN GRADING RULE

**Captured:** 2026-11-04T00:41:10Z — **before every W9 game.** Earliest kickoff is
2026-11-05 20:15 ET (JAX @ BAL).
**Status:** FROZEN BOARD + PREREGISTERED GRADING RULE. No result here.
**Artifacts:** `data/market_track_record/w9_prospective_frozen_v1.csv` (478 players, 30 teams),
`..._v1.json`, `scripts/research/capture_w9_prospective_board_v1.py`.

---

## 0. Why this exists

**Every study in V22 failed to promote for one reason: no holdout.** W7+W8 only, and W3–W5 2026
are permanently lost (AGENTS.md §5a). **Week 9 is unplayed, so for the first time a genuine
out-of-sample test is available — and it expires at kickoff.** This is that test, frozen.

## 1. Research question

**Do the D3 and D1 fixes, validated in-sample on W7+W8, replicate out-of-sample on W9 — including
the rushing over-correction?**

## 2. Arms — frozen now, and what is deliberately pending

| arm | status |
|---|---|
| **baseline** (production-equivalent) | **FROZEN** at capture |
| **D3** (zero opportunity → zero yards) | **FROZEN** at capture |
| **D1** (inactives → 0, QUESTIONABLE ×0.8535 tgt / ×0.5718 rush) | **PENDING** — see §3 |

Frozen totals: `rec` baseline **7434.3** → D3 **7069.8** (−4.9%); `rush` baseline **3683.6** →
D3 **3228.5** (−12.4%).

**D2 is excluded** — the joint study showed it harms yards. **It must not be added later.**

## 3. The D1 overlay protocol — the one place lookahead could enter

**W9 injury designations are not published as of capture** (`nflreadpy.load_injuries` max week =
8, W9 rows = 0). D1 is a deterministic function of the pregame designation, so:

1. the **rule** is frozen now: `OUT`/`DOUBTFUL` → share 0; `QUESTIONABLE` → ×0.8535 targets,
   ×0.5718 rush; those constants were fitted on **W1–W6** and are not refittable;
2. the designations are fetched **once**, as published, and the fetch timestamp is recorded;
3. **the overlay must be committed before the kickoff of the games it covers.** Thursday's
   JAX @ BAL will therefore be **D3-only** unless designations publish before it;
4. **no designation may be read after a game it affects has started.** If that cannot be
   guaranteed for a game, that game is **excluded from the D1 comparison** and graded D3-only.

## 4. Frozen grading rule

Graded after W9 completes, against `nflreadpy.load_player_stats(2026, week=9)`.

- **Primary:** per-player bias on `rec_yards` and `rush_yards`, **each player counted once**,
  bootstrapped **over players**, 4,000 resamples, seed stated in the grading script.
- **Secondary:** tier table from the **baseline** projection, **both ends reported** (D3 acts at
  the bottom, D1 at the top).
- **Level:** projected/realized total yards per market.
- **Pooled MAE is a guardrail that may veto, never approve.**
- Players absent from W9 stats are graded as **0 yards** (they did not play), consistent with
  every prior study in this sequence.

## 5. Gates

- **G1** D3 beats baseline on `rec_yards` per-player bias, 95% CI excluding 0.
- **G2** D3's `rec_yards` projected/realized moves **toward 1.000** from the baseline's value.
- **G3** No baseline tier degrades by more than **2% relative**, **evaluated on an absolute floor
  of 1.0 yard** — tiers whose baseline mean is under 1.0 yard are reported but exempt, because a
  2% relative bar on a sub-yard mean is 0.02 yards. *(This fixes the mis-scaling that voided the
  rush-allocation study.)*
- **G4** The **rushing over-correction replicates**: D3's `rush_yards` projected/realized lands
  **below 0.95**, as it did in-sample (0.860). **This gate is confirmatory, not aspirational — a
  pass means the W7/W8 finding was real.**
- **G5** (if the D1 overlay lands) D1+D3 beats D3 alone on `rec_yards`, CI excluding 0.

## 6. Kill conditions

**K1** G1 fails → **D3 does not replicate out-of-sample and must not ship**, and the W7/W8 result
is sample-specific · **K2** G2 fails · **K3** G3 breached above the 1-yard floor — name the tier ·
**K4** **G4 fails, i.e. rushing does *not* over-correct** → the in-sample rushing finding was
sample-specific, and `D3` may be admissible for rushing after all; **this would be good news and
must be reported as such, not as a gate failure** · **K5** the D1 overlay cannot be committed
pregame for a game → that game is D1-excluded (§3), not quietly included · **K6** fewer than 300
players have gradeable W9 stats → underpowered, report as such.

## 7. Prediction

**G1 passes** — D3 removes phantom yards, which is a football identity, and W9 has no reason to
differ. **G4 passes**: `rush` falls 12.4% at capture, so the over-correction should replicate near
0.86–0.88. **The receiving level improves and the rushing level gets worse**, exactly as in-sample.

**Named risk:** this is a **reconstruction**, not a production run. `player_form_consensus.csv`
and the production universe are pipeline-generated and not committed, so the baseline is rebuilt
from nflverse. It validated at **r = +0.967** against shipped `model_proj` on W7+W8, but the share
*levels* differ from production (1.0886 against 1.3756). **So this tests the FIX, not the
production integration**, and a pass licenses promotion of the *mechanism* only — the production
path still needs its own validation run.

A second risk: **30 teams, 478 players, one week.** The in-sample studies had 799 player-weeks
across two weeks. A single week is thinner and the CIs will be wider.

My V22 record: **8 prediction hits, ~17 misses**, with the named risk landing in nearly every
study. **Weight the risks, not the prediction.**

## 8. Scope

`rec_yards` and `rush_yards` only. No sportsbook variable was read and **no odds were fetched** —
betting research remains CLOSED and this capture is football-only. The 0.95 cap is untouched
(authorization-gated per V20 §7). No production code modified.

**Integrity guard in the capture script:** an assertion fails the run if any week ≥ 9 is read —
`assert cur.wk.max() <= 8`.
