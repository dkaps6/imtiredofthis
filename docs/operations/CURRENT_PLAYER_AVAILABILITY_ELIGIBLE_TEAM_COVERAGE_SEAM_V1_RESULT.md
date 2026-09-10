# Current Player Availability — Eligible-Team Coverage Seam V1 Result

Status: `ELIGIBLE_TEAM_COVERAGE_SEAM_REGRESSION_PASS`

This is a plumbing/integration prerequisite only. It is not the 35-gate Full Slate certification and authorizes no production promotion.

## Canonical lineage

- protected production authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- frozen Full Slate integration plan: commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`, blob `54eb4629c48062fcaef3153918b2069238584d0a`
- eligible-team seam frozen plan commit: `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- seam plan blob: `d0a7f74cbc8a4fcb8aa46a0b8a0aaf4987c72006`
- implementation lock commit: `7bccb4960ab0d2e538fd8c0e179ceedf4b1fb458`
- implementation lock blob: `ae77e8772ef30ae813b960196b6ebed219e3333b`
- helper blob: `77b591e431378ec984c51e8a032262e673d4c843`
- transformer blob: `b64ec5ccd59728121a250433e40e77e3e1013a05`
- regression blob: `778753400f9f1caa6831f2a71e121a13c268356c`
- protected full-universe source blob before transform: `f8429ea5b6dd730f054460493facde4ab21b0998`
- protected R26 adapter source blob before transform: `0c7528a3ca9e750d3b9ef2f08ef9721949b3e7fc`
- workflow/head: `c4529c1dcb21f807b0e539db5bad53f68bdf4dfb`
- Run `34453027002`
- Job `102792905910`
- Artifact `10142304020`
- Artifact name `current-player-availability-eligible-team-seam-v1`
- Artifact digest `sha256:071f791c916d5c17c655b62d6858ca2adfbfd276009878281898608c2d9d3cc0`

## Result

PASS on every frozen prerequisite:

1. exact frozen implementation/source blobs verified;
2. transformer source anchors verified before write;
3. legacy/no-availability mode accepted 32 teams and rejected the 30-team fixture;
4. explicit availability mode accepted exactly the certified 30-team fixture;
5. explicit mode rejected a missing eligible team;
6. explicit mode rejected an extra withheld team;
7. exact frozen transformation applied only to `scripts/run_pricing_with_full_roster_universe_v1.py` and `scripts/modeling/rb_r26_receptions_production_adapter_v1.py` in the CI workspace;
8. both transformed modules compiled;
9. no 35-gate integration result was produced.

The seam therefore removes the known legacy 32-team mechanical blocker without changing model parameters, R26 vacancy/control semantics, R22, M38, WR-R15, TE-R5P, QB C2, or sportsbook boundaries.

## Next action

Build the sportsbook-independent certification football universe from immutable candidate Run `34447900206` / Artifact `10140425929`, apply the locked seam transformation, execute the real M38 -> TE-R5P -> WR-R15 -> R26 -> R22 path on only the certified eligible teams, construct the three required unavailability fixtures, then lock evaluator/workflow blobs before the first 35-gate result.
