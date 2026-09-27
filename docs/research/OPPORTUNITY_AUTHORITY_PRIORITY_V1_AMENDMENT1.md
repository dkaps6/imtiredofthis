# Opportunity Authority Priority V1 — Pre-Result Fidelity Amendment 1

**STATUS: FROZEN BEFORE ANY CANDIDATE FULL-STACK SCORE WAS COMPUTED OR INSPECTED.**

Parent frozen plan:
`docs/research/OPPORTUNITY_AUTHORITY_PRIORITY_V1_PLAN.md`

This amendment changes no candidate source-priority rule, gate threshold, metric,
target season, or stopping rule. It makes the historical replay obey existing
specialist OOS authority instead of fabricating unavailable 2025 WR-R15 folds.

## Exact specialist replay authority

The repository's canonical production-order historical replay proves:

- TE-R5P has fold-safe OOS authority for **2024 and 2025**;
- WR-R15 has fold-safe OOS authority for **2024 only**;
- WR-R15's own frozen science explicitly marks 2025 confirmation as forbidden.

Therefore the candidate replay must use:

- **2024:** M38 -> TE-R5P -> WR-R15 -> joint MC;
- **2025:** M38 -> TE-R5P -> joint MC, with no WR-R15 application.

The baseline and candidate use the **same** authorized specialist route within
each season. The experiment still changes only the frozen opportunity source
priority:

- RB rush share -> PlayerForm fast state;
- WR target share -> PlayerForm fast state;
- TE target share -> PlayerForm fast state.

All efficiency metrics remain Bayesian.

## Why this is required

Applying a nonexistent 2025 WR-R15 OOS fold would leak or invent authority and
would make the supposedly production-exact historical test less trustworthy.

The 2025 WR downstream gates remain binding. They now mean:

> candidate vs baseline under the exact leakage-safe 2025 historical authority
> available to this repository (M38 + TE-R5P; no WR-R15).

No 2025 WR gate is removed, weakened, or replaced.

## Monte Carlo opportunity trace

The candidate evaluator may record realized mean targets/carries from the
existing finite multinomial draws. This is read-only instrumentation:
- no additional RNG calls;
- no changed seed;
- no changed iteration count;
- no changed arrays.

A focused invariance test must prove simulation arrays are elementwise identical
with and without the opportunity trace enabled.

## Production consequence

None. This remains research-only. A scientific PASS still authorizes only a
separate production-integration proposal.
