# PRISM — Rate-scale gate bug: path attribution and effect sizing (V1)

**Status:** the committed rate-scale guard (`ab29a096`) is **NOT VALIDATED as the gate
fix** and must not be merged as one. It is installed off the code path that produces
player target share.

**Trace:** `scripts/research/trace_rate_scale_path_attribution_v1.py`
**Machine record:** `data/research/rate_scale_path_attribution_v1.json`
**Tests:** `tests/test_trace_rate_scale_path_attribution_v1.py` (13)
**Authorization:** audit `6456181944` approved an isolated, source-pinned, read-only
shadow replay, merge explicitly unauthorized. Executing its requirement 2
("baseline parity first") produced the disposition below.

---

## 1. What the shadow replay was asked to certify, and what it found instead

The approved shadow was to certify the **player-mean consequence** of the rate-scale
guard before merge, on the reasoning that `te_target_mult` 1.265 → 1.15 would propagate
`rules_tgt_share` → M38 → entitlement and redistribute mass to teammates beyond the
directly affected TE rows.

Requirement 2 ("baseline parity first") is what broke the proposal. Establishing which
code path produces `rules_tgt_share` showed the guard is not on it.

### Finding 1 — path attribution: the guard cannot move target share

`scripts/modeling/simulation_rules.py:14` imports `load_model_contexts` from
**`context_bridge`** (v1), not `context_bridge_v3`. The v1 bridge reads
`team_form.csv` + `cb_coverage_team.csv` directly and **never reads
`team_context_v3.csv`**.

`apply_rules_to_metrics` then does `_, players = load_model_contexts()` — it
**discards the teams dict** and reads the `TeamContext` baked into each
`PlayerContext` by `build_player_contexts`.

The guard was installed at the `scripts/team_context_v3.py` boundary. The only
production consumer of `team_context_v3.csv` is `scripts/run_model_context_bridge.py`.
So the guard corrects that artifact and nothing that reaches `rules_tgt_share`.

This was confirmed empirically before it was believed: the first delta run, which
injected patched *teams* but stale *players*, moved **0 of 1382 rows**. That null was
a harness artifact — it is also exactly what the production shape of the bug looks
like, and it is why gate state must be injected through rebuilt `PlayerContext`s.

### Finding 2 — root cause: a four-entry omission from the repo's own allowlist

`scripts/make_team_form.py:331` defines `PERCENTY_COLS`, this repository's existing
percent→fraction mechanism, applied at Sharp ingest by `_prep_team_key_and_rates`
(`/100` iff `max > 1.0`).

| field | in `PERCENTY_COLS` | scale in W7 `team_form.csv` |
|---|---|---|
| `light_box_rate` | yes | 0.253 – 0.606 fraction |
| `heavy_box_rate` | yes | 0.149 – 0.342 fraction |
| `blitz_rate` | yes | 0.172 – 0.665 fraction |
| `coverage_man_rate` | **no** | 5.0 – 41.7 **percent** |
| `coverage_zone_rate` | **no** | 54.9 – 89.2 **percent** |
| `middle_open_rate` | **no** | 24.0 – 64.6 **percent** |
| `middle_closed_rate` | **no** | 27.6 – 73.0 **percent** |

Membership explains the scale exactly, in both directions. Sharp Football is not
supplying inconsistent data so much as the ingest allowlist is incomplete. The fix
belongs here — one layer upstream of where the guard was built — not in a new
normalization function at a downstream artifact boundary.

Two red herrings, excluded deliberately: `*_z` columns are z-scores and are
legitimately outside [0,1]; `*_sharp` columns are the raw Sharp values retained
beside already-normalized ones (`light_box_rate_sharp` 25.3–60.6 vs
`light_box_rate` 0.253–0.606). Neither is a scale bug.

### Finding 3 — effect size: one multiplier, one position

`rules_v2.matchup_multipliers` compares against **fraction** thresholds
(`zone >= 0.60`, `man >= 0.50`, `middle >= 0.50`) with no normalization anywhere in
`rules_v2` or the v1 bridge.

| gate | fires as-is | fires fixed | false fires removed |
|---|---|---|---|
| `middle_open_rate >= 0.50` | 32/32 | 13/32 | **19** |
| `coverage_man_rate >= 0.50` | 4/32 | 0/32 | 4 |
| `coverage_zone_rate >= 0.60` | 31/32 | 31/32 | 0 |

Because `slot_target_mult` is structurally unreachable dead code
(`PRISM_SLOT_ROLE_UNREACHABLE_V1`), the `middle_open` gate's only live effect is
`te_target_mult *= 1.10`. The measured consequence on `rules_tgt_share`:

- **206 of 1382 rows moved, all TE**;
- exactly **one** relative multiplier, `0.909091` = 1/1.10;
- **18 of 28** slate teams, **51** distinct players;
- mean delta among moved rows **−0.00917** target share;
- negative control: **10** unaffected slate teams, max |delta| **0.0**.

Count reconciliation: 19 false fires across all 32 teams; 18 of those teams are on
the 28-team W7 slate (one is on bye); all 18 have ≥1 TE, so 18/18 are affected.

The four false `man`-gate fires reached **zero** W7 rows: the percent-coverage
fallback teams are BUF, JAX, LAC, WAS, none of which play in W7. That is a latent
risk for any week in which a team plays while missing from `cb_coverage_team.csv`,
not an active W7 effect.

**Direction:** the bug *inflates* TE target share. Correcting it reduces TE targets
by 9.09% on affected rows.

---

## 2. Disclosed limitations — this is a counterfactual, not a production replay

Requirement 1 permitted "fail closed **or** disclose counterfactual". Disclosing:

**Baseline parity against the archive FAILS, and is not repairable from this fixture.**
`preserve_week7_opportunity_trace_v1` subsets columns (`usecols=keep`), which dropped
`neutral_pace_last5` and `plays_est` — both consumed by `context_bridge` v1 at lines
98 and 100. Consequences, measured:

- `rules_plays_est` differs from the archive by a **per-team constant** (within-team
  std exactly `0.0` across all 28 slate teams) — the signature of a differing
  team-context input, not a row-level bug;
- `rules_tgt_share` differs universally across every position, as a cascade from it;
- `rules_pass_rate` **does** reproduce exactly (max |diff| 1.11e-16).

Positional alignment against the archive is **valid** — all 7 identity columns
(`player_clean_key, team, opponent, season, week, position, role`) match row-for-row
across 1447 rows — so the mismatch is a value difference, not a join artifact.

**Absolute levels in this trace are therefore not production levels.** The delta
remains a clean differential because both arms consume the identical fixture and
differ in exactly one variable; the one-variable proof holds exactly
(`rules_plays_est`, `rules_pass_rate`, `rules_rush_share` max |diff| 0.0;
role/position/team/opponent unchanged).

**`rules_ypt` / `rules_ypc` invariance is UNMEASURED, not demonstrated** — both are
all-NaN in both arms because the fixture lacks the ypt/ypc inputs. The trace reports
`UNMEASURED_ALL_NAN` for these rather than `UNCHANGED`.

**Entitlement-stage teammate redistribution is still not measured.** At the rules
stage there is no room conservation, so the TE reduction is not yet offset onto
teammates. The original concern that motivated the shadow — that the 0.95 room
redistributes mass beyond the directly affected TE rows — remains open, and now
applies to the `PERCENTY_COLS` fix rather than to the guard.

---

## 3. Disposition

1. **Do not merge `ab29a096` as the gate fix.** It is off the projection path. It may
   still be defensible on its own narrower terms — `team_context_v3.csv` is a
   canonical artifact and should not carry percent-scaled rates — but that is an
   artifact-correctness argument, not the gate fix, and it must not be presented as
   one.
2. **The gate fix is four entries in `PERCENTY_COLS`.** Not yet written, not merged.
3. Its measured W7 effect is TE-only, −9.09% target share, 51 players, 18 teams,
   under the counterfactual caveats above.
4. Still open before any merge recommendation: entitlement-stage redistribution, and
   whether a repaired fixture (or a live no-odds dry run) can establish true baseline
   parity.
5. `PRISM_SLOT_ROLE_UNREACHABLE_V1` stays a **separate** finding. Nothing here fixes
   role routing, and the `middle_open` gate's `slot_target_mult` branch stays dead.

## 4. What this says about the earlier "19/32 false fires" figure

The earlier figure was right about the count and wrong in implication: the gate fires
for **all 32** teams, and is *false* for 19 of them. The differential effect is the 19,
but the gate is not discriminating at all at percent scale — it is a constant, which is
the stronger statement. 13 teams fire "correctly" only by coincidence of
`percent >= 50 ⟺ fraction >= 0.50`.
