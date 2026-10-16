# PLAYER OPPORTUNITY JOINT ALLOCATION — DIRICHLET V1 MECHANICS RESULT

Date: 2026-10-13

Status: **MECHANICS CERTIFIED. RESEARCH-ONLY. NO PRODUCTION SIMULATOR CHANGE.**

Frozen plan:
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PLAN.md`

Canonical authority:
- workflow run: `38400116643`
- exact head: `ba6d5bb08663d3e944a9e4c90afd68b0812098d4`
- workflow conclusion: `SUCCESS`

Certified mechanics:
- finite team opportunity total conserved in every simulated draw;
- long-run player means remain at canonical baseline probabilities;
- no eligible player state uses the exact baseline multinomial path;
- invalid overdispersion state uses the exact baseline path;
- single eligible player reproduces the corresponding Beta-Binomial marginal variance;
- complement categories retain their baseline expected counts;
- room-level rho is the frozen probability-variance-weighted aggregation;
- only RB carry share / WR target share accepted;
- TE target / RB target not authorized;
- no fitted coefficient or threshold;
- research allocator is not referenced by production `simulation_v2.py`;
- strict repository audit passed.

The first run's red conclusion was a false research-boundary check: it compared
the entire long-lived research branch to main and detected older unrelated
simulator differences. All allocator mechanics tests had already passed. The
boundary check was repaired to test whether this new research allocator escapes
the research namespace; the canonical rerun then passed.

No scientific efficacy claim is made by this mechanics result.

Next legal move:
freeze and run a separate 2023-2025 historical joint-distribution scoring test
before any production or prospective simulator integration.
