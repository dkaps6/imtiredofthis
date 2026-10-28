# PRISM — Generic Game Script Transmission Trace V1

**Status:** read-only, source-only. **No fitting, no scoring, no promotion, no
alternative proposed.**
**User-authorized** 2026-10-27 as the parallel lane to GPT's artifact-only
source work. Closes the last untraced item on the STILL OPEN list.
**Reproduce:** `python scripts/research/trace_game_script_transmission_v1.py`
**Artifacts:** `data/research/game_script/game_script_transmission_v1_status.json`

Artifacts read: **none**. Outcomes inspected: **none**. Sportsbook inputs: **0**.

## Finding

`rules_v2.project_game_script()` computes an opponent-aware
`diff = success_diff(offense, defense)` and turns it into lead/neutral/trail
probabilities via a softmax on `diff * 20`. Then, on the next line, it sets a
bare literal:

```python
diff = success_diff(offense, defense)
lead, neutral, trail = script_distribution(diff)
plays = estimate_plays(offense)
pass_share = 0.57                      # does not depend on diff
pass_attempts = plays * pass_share
rush_attempts = plays - pass_attempts
```

**Every opponent-aware output of this function reaches no projection:**

| `GameScriptProjection` field | built from | reaches a projection? |
|---|---|---|
| `projected_plays` | `estimate_plays(offense)` — **team-specific** | **yes** |
| `projected_pass_attempts` | `plays * 0.57` — **constant** | **yes** |
| `projected_rush_attempts` | `plays - pass_attempts` | diagnostics only |
| `lead_prob` | `success_diff` → softmax | **diagnostics only** |
| `trail_prob` | `success_diff` → softmax | **diagnostics only** |
| `neutral_prob` | `success_diff` → softmax | **nowhere** |
| `blowout_risk` | `abs(diff) >= 0.06` | **nowhere** |
| `shootout_risk` | `abs(diff) < 0.03 and plays >= 68` | **nowhere** |

`lead_prob` and `trail_prob` appear exactly once each, inside
`build_rule_diagnostics()`, which writes `model_rule_diagnostics.csv`. A
diagnostics writer is not a projection path.

So **team volume stays team-specific** through `estimate_plays` (which reads
`plays_est`, `neutral_pace`, `sec_per_play_last5`, `neutral_pace_last5`), while
**the pass/rush split is the same constant for every team whose context
resolves.**

## The constant preempts a richer signal

It does not merely sit beside one. `simulation_v2._team_inputs()`:

```python
pass_rate = _num(row, "rules_pass_rate")
if not np.isfinite(pass_rate):
    proe = _num(row, "proe", "pass_rate_over_expected", default=0.0)
    pass_rate = 0.58 + (proe if np.isfinite(proe) else 0.0)
    if np.isfinite(team_wp): pass_rate += -0.02 if team_wp >= 0.60 else 0.02 if ...
```

`rules_pass_rate` is read **first**. And `rules_pass_rate` is
`projected_pass_attempts / projected_plays` = `(plays * 0.57) / plays` =
**exactly 0.57 by construction**. `apply_rules_to_metrics()` writes it for every
row whose offense and defense context resolves (rows without context are
skipped entirely at `simulation_rules.py:179`).

So the **PROE-based, win-probability-nudged** fallback — the team-specific,
environment-aware path — runs **only when the rules layer fails to produce a
value.** Production's own `proe` field is preempted for pass rate in the normal
case.

Downstream, `simulation_v2` draws `pass_rate ~ Normal(mean, 0.035)` clipped to
`[0.25, 0.82]` and then `pass_att = Binomial(plays, pass_rate)`. With the mean
pinned at 0.57, every team's pass/rush split differs only by Monte Carlo noise.

## This resolves "read" versus "effective"

Audit `6449349551` blocker 3 correctly established that **`success_rate_def` IS
read by production** — via `rules_v2.success_diff()` inside
`project_game_script()` — and flagged whether it changes a named player's
projection as a separate routing question.

**This trace answers it.** `diff`'s only outputs are the script probabilities
and the two risk booleans. None of them reaches a projection. So
`success_rate_def` is **read and has no path to any projection**. Read and
effective come apart, exactly as the audit suspected.

## Why this is upstream of everything already traced

The pass/rush split is the **finite opportunity pool** from which every target
and every carry is allocated. A constant there sits upstream of every role
multiplier in the cleared role-routing trace: whatever `te_target_mult` or
`rb_rec_target_mult` do to a *share*, the *pool* they divide is the same shape
for every team.

That makes this more consequential than any downstream gate — and unlike the
gate work, it required no artifacts, no proxy season and no statistical
inference. It is a source fact.

## Not established

- **That a non-constant split would improve any projection.** Untested, the
  scoring window is closed, and **no alternative is proposed** — not a
  coefficient, not a formula, not a threshold.
- The numeric effect on any named player's mean.
- Whether 0.57 is a defensible league-average choice. It may well be a
  reasonable central value; this trace is about whether anything moves it, not
  about whether its level is wrong.
- Per-team finiteness of `rules_pass_rate` in a live run — that needs the
  Actions artifacts, which this session cannot read. It is in GPT's lane.

## Method validity

`assert_source_contract()` **raises** if `pass_share` stops being a bare
literal, if `success_diff` or `script_distribution(diff)` leaves the function,
if the split stops being `plays * pass_share`, if `rules_pass_rate` stops being
that ratio, if `simulation_v2` stops reading `rules_pass_rate` first, or if the
PROE fallback changes shape. So the finding cannot go stale silently.

A counting bug in the first pass of this trace reported `lead_prob` and
`trail_prob` as reaching a projection, because it compared a total that also
counted the quoted dict key against a diagnostics count of `script.<field>`
only. Both sides now count the same token.
