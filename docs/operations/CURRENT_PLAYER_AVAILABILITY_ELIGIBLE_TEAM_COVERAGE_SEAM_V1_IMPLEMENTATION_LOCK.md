# Current Player Availability — Eligible-Team Coverage Seam V1 Implementation Lock

Status: `IMPLEMENTATION_LOCKED_BEFORE_FIRST_SEAM_REGRESSION_RUN`

Frozen plan authority:
- plan commit `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- plan blob `d0a7f74cbc8a4fcb8aa46a0b8a0aaf4987c72006`

Exact implementation inputs:
- helper `scripts/utils/eligible_team_set_v1.py` blob `77b591e431378ec984c51e8a032262e673d4c843`
- transformer `scripts/operations/apply_current_availability_eligible_team_seam_v1.py` blob `b64ec5ccd59728121a250433e40e77e3e1013a05`
- regression `scripts/operations/test_current_availability_eligible_team_seam_v1.py` blob `778753400f9f1caa6831f2a71e121a13c268356c`
- protected full-universe source blob before transformation `f8429ea5b6dd730f054460493facde4ab21b0998`
- protected R26 adapter source blob before transformation `0c7528a3ca9e750d3b9ef2f08ef9721949b3e7fc`
- protected production authority `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

The transformer may change only the exact frozen source anchors documented in the plan: the full-universe 32-team coverage assertion plus canonical-game-count validation, and the R26 current RB/FB 32-team coverage assertion. It may not alter model assets, coefficients, R26 vacancy/control semantics, entitlement math, R22, QB C2, M38, WR-R15, TE-R5P, or sportsbook boundaries.

Regression requirements before 35-gate certification:
1. no-availability mode still requires 32 teams;
2. explicit availability mode accepts exactly the certified eligible team set;
3. explicit mode rejects a missing eligible team;
4. explicit mode rejects an extra withheld team;
5. transformed full-universe and R26 modules compile;
6. the source transformer refuses unexpected source-anchor drift.

No 35-gate integration disposition may be produced by this regression run. A failure is mechanical and must be preserved separately.
