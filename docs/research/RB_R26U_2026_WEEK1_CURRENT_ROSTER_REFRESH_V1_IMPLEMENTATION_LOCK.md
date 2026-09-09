# RB R26U — 2026 Week 1 Current-Roster Pregame Refresh V1 — IMPLEMENTATION LOCK

Status: **LOCKED BEFORE EXECUTION**

Frozen plan commit:
- `a823b8e52cb6de8ba01e8d6f41b091f0aaddb171`

R26U audit implementation commit:
- `6242f0d6a83fa6a6da7bbce49f2f3be18d3eedec`

R26U audit:
- `scripts/backtest/audit_rb_r26u_2026_week1_current_roster_refresh_v1.py`

R26U does not implement a new football model. The football candidate must be materialized by the original frozen R26N builder through its already-proven exact identity-key and dtype compatibility wrappers.

Pinned unchanged R26 implementation files:
- `scripts/backtest/build_rb_r26n_2026_week1_unmodified_r26_structural_candidate_v1.py` @ `5299ce54575ffcfe33ad203db0ee00285181291f`
- `scripts/backtest/stage_r26n_production_identity_key_repair_v1.py` @ `0b4c2df7c14b16bd0e945b425aac7f35e015fa3d`
- `scripts/backtest/run_rb_r26n_with_identity_dtype_compat_v1.py` @ `9abdaa9a6262aa87ac601d7bc665209b94036b20`

The workflow may refresh only pregame roster-driven PlayerForm/model-context/P3 artifacts inside an isolated copy of the protected Full Slate artifact, using protected production code. It must fail closed if observed 2026 Week-1 player-game rows appear.

No sportsbook fetch, R9 refit, R26 tuning, R22 change, production parameter change, production promotion, or live-shadow activation is authorized.