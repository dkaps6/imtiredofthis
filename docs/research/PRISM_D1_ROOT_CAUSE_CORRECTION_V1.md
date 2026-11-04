# PRISM — D1 ROOT-CAUSE CORRECTION: THE MECHANISM I PUBLISHED WAS WRONG

**Date:** 2026-11-04
**Corrects:** `PRISM_D1_INJURY_STATUS_PREREGISTRATION_V1.md` §0,
`PRISM_D1_INJURY_STATUS_RESULTS_V1.md` §6, `PRISM_RULES_SIM_BAYES_CODE_AUDIT_V1.md` D1, and the
V22 TOP checkpoint. **The effect stands. The cause does not.**

---

## 1. What I claimed

> *"Production `injuries.csv` ships with only `player, team, season, week, source`.
> `build_injuries_weekly.SCHEMA` declares eleven columns. Every status field is missing … the
> entire injury path in the rules layer is dead code."*

I traced that through five verified code links and stated it as established fact.

## 2. It is wrong, and here is the disproof

**The builder works.** Running `normalize_nflverse_injuries` on live 2026 data for week 8:

| | |
|---|---|
| rows | **320** — exactly the archived file's row count |
| columns | **all 11**, including `status` and `designation` |
| `status` non-null | **132 / 320** |
| values | **Out 50, Questionable 79, Doubtful 3** |

**The merge works.** Joining the builder's output to the W8 universe on `(team,
player_key_bridge)` — the exact keys `context_bridge` uses — delivers **84 rows, of which 13 are
OUT/DOUBTFUL skill players with `status` populated**, DeVonta Smith (`Out`) among them. In
production he held **0.2516** target share, unmodified.

**The rows were never missing.** All 16 nflverse OUT/DOUBTFUL skill players for W8 appear in the
production `injuries.csv`. **No `status` collision exists** either — `player_form.csv` has no
`status` column, so the merge's `suffixes=("", "_injury")` never triggers.

## 3. What actually misled me — and a caution that reaches further

**The archived trace copy of `injuries.csv` is lossy.** It carries 5 of 11 columns. So does the
companion `injuries_identity_audit.csv`: **3 columns where the code writes 7.** Two artifacts,
both short of what their writers emit, same run.

**I treated an archived artifact as a faithful record of production and it is not.**

**This caution generalises and is the more important half of this correction.** Several V22
conclusions were drawn from trace archives — the `rules_` vs `bayes_` coverage audit, the live
multiplier recovery, the D1/D2/D3 universes. Those used **columns that were present and
populated**, which is a different and safer claim than *"a column is absent, therefore the code
never wrote it."* **Absence in an archive is not evidence of absence in production.** No V22
finding that rests on observed values is affected; **only this one, which rested on an absence.**

## 4. What survives unchanged

**The defect is real and measured**, from the universe trace rather than the injuries file:

- **29 players nflverse ruled OUT or DOUBTFUL** appeared in the W7/W8 production universes holding
  a mean target share of **0.1392**;
- **not one recovered multiplier was near 0.50**, so `_injury_limited` demonstrably did not fire;
- several were **boosted** while inactive (Tyler Warren 0.1957 → 0.2476);
- **all 29 recorded zero targets.**

And the fix's value is unchanged: **D1 takes OUT players from 2.558 to 0.000 allocated targets**,
and `D1--D3` gains **+1.015** per-player on `rec_yards` in the joint study.

**Also unchanged:** OUT retention is exactly **0.0000** (n=124, 0.0% played), QUESTIONABLE is
**0.8535** target / **0.5718** rush (n=112, 64.3% played), and `_injury_limited` does not recognise
QUESTIONABLE at all.

## 5. The cause is now OPEN

Builder ✓, artifact rows ✓, merge keys ✓, no suffix collision ✓ — **and the rule still did not
fire.** I do not know why, and I am recording that rather than offering a fourth hypothesis.

Resolving it needs **runtime instrumentation of a live Full Slate run**, not more archaeology on
lossy archives. Candidates still open: the live-odds `refresh_current_injuries_nflcom_v1.py`
overwrite path (both W7 and W8 were live-odds runs), `player_form`'s own `player_key_bridge`
construction differing from `_player_key`, or a filter between the merge and `ctx.features`.

## 6. What I did instead of a blind repair

**I did not "repair" the builder, because I had just proved it works.** Patching a correct SCHEMA
would have been a no-op that masked the real cause.

Added instead: **`_injury_contract_findings()` in `scripts/audit_2026_production_readiness.py`**,
wired into `run()`. It makes this class of silent failure loud:

| condition | severity |
|---|---|
| `injuries.csv` has no `status` column | **P0** |
| `status` present but every row null | **P0** |
| no `status` value matches the tokens `_injury_limited` recognises | **P0** |
| `build_injuries_weekly.SCHEMA` lacks `status`/`designation` | P1 |
| `context_bridge` reads a field absent from the SCHEMA (contract drift) | P1 |
| provider emits QUESTIONABLE, which the rule ignores | **P1** |

**Each branch was verified to fire** against a synthetic artifact in each failing shape, and the
healthy shape correctly surfaces only the QUESTIONABLE P1. An audit gate never seen to fire is
worth nothing.

This is the AGENTS.md §4 requirement made executable: *"Never treat file existence alone as proof
that an artifact is valid."* The injury path ran all season with no assertion that the field it
depends on had arrived — **and when a live run next produces `injuries.csv`, this gate answers §5
directly.**

No production behaviour changed. The audit is a gate, not a projection path.
