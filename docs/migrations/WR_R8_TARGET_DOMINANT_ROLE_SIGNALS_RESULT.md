# WR-R8 Target-Dominant Role Signals — Result

## Canonical evidence
- Branch: `research-wr-r8-target-dominant-role-signals`
- Run: `34072748077`
- Job: `101592996487`
- Tested SHA: `9b55bad23b9db2fb6e00b87df4103ab37457a242`
- Artifact: `10000997724` (`wr-r8-target-dominant-role-signals`)
- Artifact SHA256: `62ec2ac728b7e58cf355077e82d8b3bed0677b1f14d02af3cd3eafbd43a4028c`
- Sportsbook inputs used: **false**
- Model fitting used: **false**
- Production changed: **false**

## Cohort
- ND5 source rows: **2130**
- WR-R5 qualifying profiles: **133**
- Exact WR-R5 `TARGETS`-dominant players: **72**
- Conditioned 2025 rows: **1000**

This was not a retry of ND5's global question. The new hypothesis, frozen before the run, was whether the exact already-recovered snap/depth signals become predictive specifically inside the later-discovered TARGETS-dominant player class.

## Results

| Signal | Valid N | Coverage | Spearman | Allocation gap | Raw target gap | Tail enrich | W2-18 gap | W13-18 gap | Within-player positive rate | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SNAP_LEVEL_PRIOR1 | 975 | .975 | .036138 | .010242 | **1.148714** | **1.723722x** | .008208 | -.001955 | **.614286** | No |
| SNAP_ACCEL_1V4 | 890 | .890 | .064957 | .015058 | .138476 | .760093x | .016116 | .029003 | .500000 | No |
| DEPTH_TOP2_STATE | 996 | .996 | -.110245 | -.036317 | .367052 | 1.425171x | -.039147 | -.081823 | .529412 | No |
| DEPTH_RANK_PROMOTION | 935 | .935 | .027449 | .023713 | -.057514 | .243807x | .023713 | .044281 | .461538 | No |

## Frozen disposition
**`NO_ACTIONABLE_WR_TARGET_DOMINANT_ROLE_SIGNAL`**

Zero of four exact signals passed all preregistered gates.

### Most informative partial pattern: prior-game snap level
`SNAP_LEVEL_PRIOR1` did identify a subset with larger raw target misses and strong tail enrichment:
- high-minus-low raw target error: **+1.148714 targets**;
- entitlement-miss tail enrichment: **1.723722x**;
- positive within-player association among 70 computable players: **61.43%**.

But it did not explain the actual within-WR allocation residual strongly enough:
- Spearman only **.036138** vs frozen .10;
- allocation gap only **.010242** vs frozen .030;
- late-season W13-18 gap **-.001955**.

Therefore it is not a correction and receives no threshold/window rescue.

### Static depth does not solve the target-dominant problem
`DEPTH_TOP2_STATE` was actually negatively associated with allocation residual (Spearman **-.110245**) and both W2-18 and W13-18 gaps were negative. `DEPTH_RANK_PROMOTION` also failed magnitude, tail, and player-consistency gates.

## Interpretation
The TARGETS-dominant player class is real, but the already-available static participation/depth variables do not tell us precisely enough which of those receivers will be under- or over-allocated in a given game. Snap level can flag some high-error/tail cases, but it is not a reliable continuous allocation mechanism.

## Authorized next step
Do not run another ND5 variant. The next WR source family must add genuinely richer football information, preferably route participation / route type / alignment / air-yard role / coverage matchup / separation or other tracking-quality context, with source coverage and timestamp safety audited before testing. M38 remains unchanged until a candidate passes the established full-stack integration protocol.
