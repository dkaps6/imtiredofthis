# RB STACK6C / ND4 — Rotation Source Audit Results

## Authoritative run

- Branch: `research-rb-stack6c-rotation-source-audit`
- Run: `33571118247`
- Job: `100065065070`
- Tested SHA: `c3d5d892f0cfa7d5323f9a0611c040ce9b6c9126`
- Artifact: `9824961408`
- Artifact SHA256: `a5b21df7f72a705983e4130dee5449321d406872cd9f11d03e169575ee6338d1`
- Frozen plan: `docs/migrations/RB_STACK6C_ROTATION_SOURCE_AUDIT_PLAN.md`

No rushing outcome model was fit. Sportsbook data was not loaded.

## Source integrity

2024–2025 regular-season source counts:

- participation rows: `91,103`
- PBP rows: `98,263`
- joined rows: `86,915`
- participation/PBP play-key join rate: `1.000`
- player/position array alignment: `1.000`
- drive-id coverage on joined offensive plays: `1.000`
- parsed offensive plays: `86,915`
- RB player-play rows: `91,609`
- RB player-games: `3,643`
- RB PBP rush/target events: `29,714`
- strict-prior leakage pass rate: `1.000`

All infrastructure gates passed.

## Live-proxy evidence

The audit compared a live-capable PBP rush+target drive proxy with delayed historical on-field participation truth.

Predeclared evidence gates:

1. player touch-opportunity share vs true RB on-field presence share:
   - correlation `0.867401`
   - gate >= `.60`: **PASS**
2. player touch-drive share vs true drive-presence share:
   - correlation `0.460976`
   - gate >= `.60`: **FAIL**
3. team-game top-RB identity agreement, touch share vs true presence share:
   - `0.847426`
   - gate >= `.70`: **PASS**
4. minimum 2025 prior-3 coverage across the six predeclared core proxy features:
   - `0.963848`
   - gate >= `.75`: **PASS**

Three of four proxy evidence gates passed, satisfying the frozen GO rule.

Frozen disposition:

`GO_STACK6C_ROTATION_PROXY_BUILD`

## Additional descriptive structure

The source audit also shows that hierarchy-specific touch measures are materially closer to true on-field hierarchy than the crude binary drive-touch measure:

- true RB on-field presence share vs PBP touch-opportunity share: `0.867401`
- true RB on-field presence share vs PBP touch-drive share: `0.814870`
- true lead-drive share vs PBP touch-lead-drive share: `0.905120`
- PBP touch-opportunity share vs touch-lead-drive share: `0.962542`

The failed predeclared comparison was specifically **drive-presence share vs touch-drive share** (`0.460976`): merely touching the ball on a drive is not a faithful proxy for being present on that drive. Do not treat that variable as an on-field participation replacement.

## Coverage

Among 1,715 2025 player-games with at least one earlier team game, every core prior-3 PBP proxy field had the same non-null coverage:

`0.963848`

Fields audited:

- prior3 touch-opportunity share
- prior3 touch-drive share
- prior3 touch-lead-drive share
- prior3 opening-drive touch share
- prior3 team touch-leader switch rate
- prior3 team touch HHI

## Capability boundary

Historical on-field participation remains delayed/postseason-release data and is **not** live-2026 capable.

The PBP rush+target rotation proxy is live-capable after a completed prior game, subject to using the canonical production identity/position bridge rather than target-game participation.

Exact target-game active/inactive competitor state remains unqualified and separate. Target-game participation remains forbidden as a substitute.

## Interpretation

The source audit supports a new architecture, not a new point projection by itself.

The strongest live-capable evidence is about **backfield hierarchy/concentration and which player actually owns the meaningful touches/lead drives**. That is directly relevant to STACK6B's failure mechanism: false-high secondary-back projections need selective contraction, while broad expansion is harmful.

Because the simple touch-drive-presence fidelity gate failed, the next model must not claim PBP touch drives are a full replacement for on-field drive presence. The model should use the PBP variables as their own pregame rotation/usage signals.

## Next

Freeze a one-sided false-high/contraction architecture before fitting:

- same P3 parent;
- same protected M95F-risk and depth-rank-1 populations;
- Week 6+ secondary backs only;
- live-capable PBP rotation history plus existing aggregate pregame state;
- separate classification of whether P3 is overallocated from magnitude of contraction;
- no positive carry correction permitted;
- no sportsbook upstream;
- original STACK6B football retention gates remain in force.
