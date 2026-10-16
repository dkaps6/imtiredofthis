# QB TEAM PASS INTENT — COMMON CRAWL ARCHIVE TRANSPORT V1I-R1 RESULT

Date: 2026-10-11

Status: **STRICT ARCHIVE IDENTITY RECERTIFICATION PASSES THE FROZEN V1H TRANSPORT GATE. THIS RESULT SUPERSEDES THE ORIGINAL V1I 23/35 ARCHIVE-RESCUE COUNT. SOURCE-ONLY RESEARCH.**

Frozen plan:
- `docs/research/QB_TEAM_PASS_INTENT_COMMON_CRAWL_ARCHIVE_TRANSPORT_V1I_R1_PLAN.md`

Canonical authority:
- workflow run: `38186579290`
- exact head: `dd1dfba16ada02d3141881f65e00731d78c1b5e0`
- artifact: `11719552673`
- artifact digest: `sha256:c371cde47ac800fb9665dcdd511d33c6626e59a4f47ba3f45e10fe17f7f063bd`
- workflow conclusion: `SUCCESS`
- exact-head Repo CI: `38186581906` SUCCESS

## Why R1 was required

Inspection of the original V1I artifact showed geography-only false positives:
local-news URLs could match tokens such as `dallas` without containing an NFL
franchise identity or target-game relevance.

Therefore the original V1I archive-rescue count of 23/35 is invalid for
scientific advancement and is superseded.

R1 performs no new Common Crawl requests. It recertifies only the exact
preserved V1I captures with a stricter, frozen franchise-identity vocabulary.

## Frozen disposition

`STRICT_ARCHIVE_IDENTITY_RECERTIFICATION_PASSES_V1H_GATE`

## Corrected transport result

- unresolved universe: **158**
- repaired V1H direct-ready baseline: **123**
- archive-needed: **35**
- strict archive-rescued rows: **18 / 35**
- combined direct-or-strict-archive ready: **141 / 158**
- combined rate: **89.2405%**
- frozen transport gate: **>=127 / 158 (80%)**
- margin over gate: **+14 rows**

Strict archive rescue by franchise:
- DAL: **14**
- GB: **4**
- SEA: **0**

## Integrity

- no new network retrieval in R1;
- original frozen publisher map unchanged;
- geography-only tokens cannot satisfy archive rescue;
- football outcomes read = 0;
- model residuals read = 0;
- sportsbook/game-market inputs used = false;
- paid OddsAPI calls = 0;
- 2026 Week-5 outcomes read = 0;
- football predictive models fit = 0;
- production changed = false;
- final source qualification claimed = false;
- predictive candidate authorized = false.

## Interpretation

Historical local-fallback transport is qualified under the unchanged 80% gate
after strict franchise-identity recertification.

This result validates transport only. It does not imply that preserved archive
captures contain target-week eligible intent content. Publication-time,
target-game relevance, source attribution, and semantic intent still must pass
the original V1 rules during final source qualification.

The next active stage is the independent NLI semantic validator V1J on the
immutable gold set. V1G AUTO_REVIEW_READY rows may not be blanket-promoted until
that independent semantic promotion gate is resolved.

## Protected science

Unchanged:
- M89/M90;
- QB C2;
- generic QB mean/YPA CLOSED;
- same-data attempts CLOSED;
- schedule/rest D1 CLOSED;
- PBP D2 CLOSED;
- sportsbook downstream only;
- no Week-5 prospective grading;
- production unchanged.
