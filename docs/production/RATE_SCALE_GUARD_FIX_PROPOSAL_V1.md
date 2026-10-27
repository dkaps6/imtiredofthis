# Surgical Fix Proposal — Rate-Scale Guard at the Canonical Team-Context Boundary

**Status:** **PROPOSAL. Implemented and tested on the research branch only.
NOT on `main`, so production is unchanged.** Requires explicit authorization to
merge.
**Requested by:** GPT artifact audit `6451272159` + addendum `6451282717`
("draft a surgical fix proposal … do not silently alter production").
**Confirmed independently** in the recovered Week-7 artifacts.

## The defect

Sharp publishes several rate fields as **percentage points**.
`scripts/team_context_v3.py` — the single canonical boundary where every source
converges — had **no rate normalization and no range validation at all**. So:

```python
# rules_v2.matchup_multipliers, expecting a fraction
if middle >= 0.50:
    values["slot_target_mult"] *= 1.10
    values["te_target_mult"]   *= 1.10
```

receives `middle_open_rate = 24.4`, and `24.4 >= 0.50` is **always true**.

### Confirmed in the data, two independent weeks

| | Week 7 (my recovered artifacts) | Week 8 (GPT's artifact read) |
|---|---|---|
| `middle_open_rate` scale | **32/32 values > 1**, min 24.0, med 48.15, max 64.6 | 32/32, range 24.4–62.8 |
| gate fires | **32/32** | 28/28 active |
| *should* fire after rescale | **13/32** | 13/28 active |
| **false fires** | **19/32** | **15/28** |
| `slot_target_mult` realized | **1.10 on 396/396 rows** | 1.10 on 404/404 rows |

**Week-7 player-row delta accounting:** 249 of 396 diagnostic rows (**62.9%**)
face a false-firing defense, across 18 distinct opposing defenses —
WR 91, TE 53, RB 51, QB 49, FB 5.

| multiplier | today | after fix |
|---|---|---|
| `slot_target_mult` | 1.10 × 249 | **1.00 × 249** |
| `te_target_mult` | 1.265 × 236, 1.10 × 13 | **1.15 × 236, 1.00 × 13** |

A multiplier change is a **potential** role-channel change. Actual named-player
movement still requires role routing and finite-room allocation, so this is
**not** a claim that 249 projections moved.

### A second, latent instance

The bye-team coverage fallback has the same defect. When Coverage-v2 has no row
(byes), `team_context_v3` falls back to legacy `team_form` **without rescaling**:
HOU 15.7/80.3, NO 15.5/74.7, NYG 26.1/68.6, SF 11.8/82.1 — **4/32** rows with
man/zone > 1. Those teams are not current opponents, so nothing is wrong today;
it is a live source-contract defect waiting for a different bye week.

**Also found, not flagged upstream:** `middle_closed_rate` has the identical
scale problem (W7 min 27.6, med 47.8, max 73.0, 32/32 > 1). No gate reads it
today, so it is latent — but it shows the bug is **family-level**, not a
one-field slip. Box rates, by contrast, arrive correctly as fractions
(`light_box_rate` 0.253–0.606).

## The proposed change

One helper at the canonical boundary, applied in both the coverage block and the
legacy-field loop:

```python
_RATE_FIELDS = frozenset({
    "coverage_man_rate", "coverage_zone_rate",
    "light_box_rate", "heavy_box_rate",
    "middle_open_rate", "middle_closed_rate",
    "rz_rate",
})

def _normalize_rate(feature, value) -> tuple[float, str]:
    if feature not in _RATE_FIELDS: return value, ""
    ...
    if 1.0 < v <= 100.0:  return v / 100.0, "_percent_rescaled"
    if v < 0.0 or v > 100.0: return nan, "_out_of_range_failed_closed"
    return v, ""
```

Four deliberate choices:

1. **Explicit allowlist, not "rescale anything > 1".** `neutral_pace_last5 =
   27.4` and `sec_per_play_last5 = 29.1` are legitimately > 1; a blanket rule
   would corrupt them. A test asserts these are never rescaled.
2. **Out of range fails closed to NaN, never clamps.** Clamping would hide a
   future provider change behind a plausible-looking number. The field is
   labelled unavailable instead, per the fail-closed rule for source/semantic
   mismatches.
3. **Exactly 1.0 is left alone** — a valid fraction, and the "100%" ambiguity
   cannot be resolved from the value.
4. **Provenance records what happened** via a `freshness_state` suffix
   (`_percent_rescaled` / `_out_of_range_failed_closed`), so the change is
   auditable rather than invisible.

**No threshold is changed.** A test asserts `middle >= 0.50`, `light >= 0.60`,
`heavy >= 0.60`, `zone >= 0.60` and `man >= 0.50` are all still exactly as they
are. This fix restores the **intended semantics** of existing gates; it does not
retune them.

## Verification

- 24 new tests in `tests/test_team_context_v3_rate_normalization.py`, covering
  both confirmed defects, the untouched-fraction cases, fail-closed behaviour,
  non-rate passthrough, the 1.0 edge, threshold immutability, and that every
  gated rate field is in the allowlist.
- Full suite **854 passed**, 1 known pre-existing failure, `audit_repo.py
  --strict` clean.

## What this proposal does NOT claim

- **That it improves accuracy.** Untested. The scoring window is closed. It is a
  correctness fix: the gate currently evaluates a percentage against a fraction
  threshold, which cannot be the intent.
- **That 0.50 is the right threshold** once the scale is correct. Separately:
  `light_box_rate >= 0.60` has a Week-7 max of 0.606 and a Week-8 max of 0.586,
  so on Sharp's scale that gate is barely or never reachable. **Whether 0.60 was
  chosen for a different source's scale is an open threshold-provenance
  question** — flagged, not fixed, because changing it needs scoring
  authorization.
- **That Sharp's definitions are now understood.** The denominator remains
  UNESTABLISHED; see below.

## Recommended before merge

A replay of the Week-7 or Week-8 artifact through the patched boundary, to
confirm the realized multipliers move exactly as the delta table predicts and
nothing else shifts. That needs a runner, since this session cannot read
artifacts.
