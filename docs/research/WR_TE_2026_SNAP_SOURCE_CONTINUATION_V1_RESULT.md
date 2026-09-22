# WR/TE 2026 Snap Source Continuation V1 — Result

**Disposition:** `WR_TE_2026_SNAP_SOURCE_CONTINUATION_READY`  
**Canonical run:** `35742765095`  
**Head:** `741ae291aa43f7b7f42aa8047d36e18a1ffa064d`  
**Artifact:** `10699873027` / `wr-te-2026-snap-source-continuation-v1`  
**Digest:** `sha256:dbd15f7f9d2eb34d2c28aec07fb71dc0dccb80e8f6d40f0165b92c91eee62ddd`  
**Model refit:** 0  
**Sportsbook inputs:** 0  
**2026 outcomes used:** 0  
**Production changed:** 0

## Gates

All frozen gates passed.

### Historical parity
Dynamic source through 2025 reproduces the frozen production source exactly:

- normalized rows: 150,909;
- max offense_pct gap: 0.0;
- max offense_snaps gap: 0.0;
- canonical parity digest:
  `8e9c2563c5f585f955b8cd82bb13ee6f549ff464a550415a1a6380204923685d`.

### Week-1 invariance
Adding 2026 to the candidate source cannot affect a 2026 Week-1 target because
the strict-prior filter excludes all 2026 rows.

- target rows: 255;
- WR: 156;
- TE: 99;
- feature parity: exact PASS.

### Week-2 strict-prior
Using completed Week-1 identities as the target frame:

- target rows: 279;
- WR: 169;
- TE: 110;
- rows with at least one feature change: **279 / 279**;
- rows gaining prior-1 same-team participation: 91;
- rows gaining prior-1 any-team participation: 43;
- latest 2026 week used: 1;
- target/future usage: 0.

### Week-3 readiness
Using identities established in completed Weeks 1-2:

- target rows: 298;
- WR: 183;
- TE: 115;
- rows with at least one feature change: **298 / 298**;
- rows gaining prior-1 same-team participation: 100;
- rows gaining prior-1 any-team participation: 50;
- latest 2026 week used: 2;
- target/future usage: 0.

### 2026 source quality

- weeks available: [1, 2];
- total rows: 2,994;
- teams: 32;
- WR rows: 335;
- TE rows: 220;
- offense_pct non-null: 2,994 / 2,994;
- offense_snaps non-null: 2,994 / 2,994;
- duplicate rate under production keys: 0.0.

Frozen model JSONs were untouched:

- TE-R5P SHA256:
  `a47730312a6d7ea8de2c4034ddd2be5ebbac72c67647dfbfa7b1985305e1249b`
- WR-R15 SHA256:
  `ac1058f534c7923e8ad41e52a92e7001c1de309bb04874d76ae60a83e3a1b2ff`

## Interpretation

WR-R15 and TE-R5P already learned strict-prior participation features and the
historical source contract already includes same-season prior games. The live
2026 omission is therefore a source-continuation gap rather than a request to
invent a new feature or refit a model.

The safe production candidate must activate **prospectively at 2026 Week 3**.
Weeks 1-2 must retain legacy source behavior so preserved Week-2 replay remains
historically exact.

No production change was made by this research result.

## Next authorized step

Build a production-integration candidate with:

- legacy `2020..2025` source for all seasons <=2025 and 2026 Weeks 1-2;
- `2020..2026` source for 2026 Week 3+ only;
- unchanged R15/R5P coefficients and feature construction;
- exact Week-2 preserved replay;
- new unit tests for activation boundary and no same/future rows;
- Repo CI and preserved Full Slate replay before any merge.

