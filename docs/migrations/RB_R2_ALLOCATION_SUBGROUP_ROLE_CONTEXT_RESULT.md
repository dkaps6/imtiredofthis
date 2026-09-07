# RB-R2 Allocation-Subgroup Role Context — Result

## Canonical evidence
- Branch: `research-rb-r2-allocation-subgroup-role-context`
- Run: `34070930143`
- Job: `101588001813`
- Head SHA: `56b18687dda4d1decc6012a503a760e36f6911b4`
- Artifact: `10000420714`
- Artifact digest: `sha256:b31626b0b0cb8c67749aa54506fbe2dc70ff73e20a0ba20d16aac64210399e64`
- Frozen plan commit: `a85bcc3562118cddc5d63054022d3b22938a1e78`
- Evaluator commit: `12cfd34d68f9b018c0725d0a8813b861790e6a8d`

## Integrity
- Source RB-R1 rows: **1,393**
- Source qualifying profiles: **86**
- Allocation-dominant parent-CARRIES players: **17**
- Conditioned player-games: **212**
- Sportsbook inputs used: no
- Model fitting used: no
- Production changed: no

## Official disposition
**`NO_ACTIONABLE_RB_ALLOCATION_SUBGROUP_ROLE_CONTEXT_SIGNAL`**

The primary depth/carry-order state failed and zero of the four secondary transition states passed all frozen gates. No thresholds or state definitions are changed.

## Primary state: depth vs projected carry-order mismatch
- mismatch N **53**; matched N **159**
- mean absolute allocation component: **3.1821** vs **3.0098** attempts
- allocation-component ratio: **1.05726** — FAIL vs >=1.25
- absolute difference: **+0.17235** attempts — FAIL vs >=0.75
- carry MAE: **4.0886** vs **3.7251** attempts
- carry-MAE ratio: **1.09759** — FAIL vs >=1.15
- W2-18 allocation ratio: **1.0760**
- W13-18 allocation ratio: **1.4176**

The later-season ratio is descriptively elevated, but the full conditioned population misses the materiality gates by a large margin and therefore cannot be promoted.

## Secondary states
- Injury-created context: N1 **13**, allocation ratio **0.9068**, carry-MAE ratio **0.9164**.
- No prior same-team game: N1 **9**, allocation ratio **0.9181**, carry-MAE ratio **0.7428**; sample gate also fails.
- Rookie: N1 **76**, allocation ratio **1.0583**, carry-MAE ratio **0.9652**; W13-18 ratio **1.2897** but overall materiality fails.
- Limited prior history: N1 **18**, allocation ratio **0.9458**, carry-MAE ratio **0.7266**; sample/materiality gates fail.

## Scientific interpretation
RB-R1 established that individual carry errors are genuinely heterogeneous: some players miss because total RB-room volume is wrong, while others miss because the room volume is allocated to the wrong player. RB-R2 then shows that even inside the 17 allocation-dominant players, coarse static states—depth rank mismatch, rookie status, injury-created context, team continuity, or limited history—do not explain enough of the allocation error.

That further weakens a universal “make the depth chart more authoritative” fix. The production concern about current role remains legitimate, but the next allocation research needs richer dynamic workload information such as recent snap/carry rotation, coaching usage patterns, personnel competition, down/distance role, or similarly explicit workload-state mechanics rather than these broad flags.
