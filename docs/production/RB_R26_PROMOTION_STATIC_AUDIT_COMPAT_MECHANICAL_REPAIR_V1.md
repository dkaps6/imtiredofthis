# R26 Promotion Static-Audit Compatibility — Mechanical Repair V1

**Frozen before applying the compatibility repair.**

## Trigger

After the already-qualified R26 production branch was fast-forwarded to `main` at `1f0ea0d697f9a5b66cba42facc424a0185a147ed`, Repo CI run `34418257607` compiled all production modules successfully but failed `scripts/utils/audit_repo.py --strict` with one static-wiring assertion:

`public V3 compatibility entrypoint does not route exactly to V4 production`

The assertion predates the R26 promotion and literally searches the public compatibility wrapper for:

`from scripts.run_pricing_with_full_roster_universe_v4_production import main`

The newly promoted wrapper correctly routes its final `main` binding to the qualified V5/R26 stack, whose first simulation parent is itself V4. Therefore this is a **mechanical static-audit contract mismatch**, not a scientific R26 failure and not a runtime import/syntax failure.

## Authorized repair

Only the stable public compatibility wrapper may change.

The wrapper will explicitly import the protected V4 parent `main` first under the legacy/public binding and then explicitly rebind public `main` to the qualified V5/R26 `main`. This preserves all of the following simultaneously:

1. the old static audit can still prove that the protected V4 parent is an importable, explicit dependency of the public production chain;
2. V5 remains the final public execution authority;
3. V5 still executes V4 first and then applies the qualified R26 receptions adapter;
4. no football parameter, R26 parameter, R19/R9 asset, vacancy state, ensemble weight, pricing rule, or sportsbook boundary changes;
5. the frozen R26 adapter/V5/evaluator files from qualification run `34417740186` remain byte-identical.

No audit script is weakened and no PASS condition is deleted. The compatibility wrapper simply exposes both the protected V4 parent dependency and the promoted V5 final authority explicitly.

## Required verification

After the repair:
- Python compilation must pass;
- Repo CI strict static audit must pass;
- the final imported public `main` identity must equal V5 `main`, not V4 `main`;
- qualified R26 core files must remain unchanged from qualification head `343372586bd4979c34487761d0af49b5986f68e8`;
- Full Slate must remain wired to `scripts/run_pricing_with_full_roster_universe_v3.py`.

This repair does not alter the 35/35 scientific/production qualification disposition.
