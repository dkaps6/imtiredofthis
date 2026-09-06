# QB-WR Shared Pass-Volume Mechanism Result

## Canonical run

- Run: `34066549394`
- Job: `101576202002`
- Tested SHA: `ba8fce70caaf147ae8ac49003473a3aa6142fbae`
- Artifact: `qb-wr-shared-pass-volume-mechanism`
- Artifact ID: `9999119623`
- Artifact SHA256: `4e148f982d2f8db8a8e19cd2bbe2dbeb775dee296efc01aaf37193d98db2ddc6`
- Conclusion: `success`

## Frozen disposition

`STRONG_SHARED_PASS_VOLUME_MECHANISM`

All preregistered primary gates passed.

## Primary 2025 target-mass test

QB attempt residual was compared with aggregate WR target-mass residual across aligned team-games.

- n: `440`
- Pearson: `0.6973048915041206`
- Spearman: `0.670648420892482`
- same-sign rate: `0.7318181818181818`
- top-vs-bottom WR target-residual quartile QB-attempt-residual gap: `13.749357718414185` attempts

Stability slices:

- W2-18: n `417`, Pearson `0.700610159167479`, Spearman `0.6756443575356865`, gap `13.788492177121832` attempts
- W13-18: n `145`, Pearson `0.663422194687807`, Spearman `0.6610612501968193`, gap `12.64668715474548` attempts

## Secondary 2024-2025 WR-reception-mass replication

Aggregate WR reception residual versus QB attempt residual:

- all 884 team-games: Pearson `0.5238926874684945`, Spearman `0.49940402459076466`, same-sign `0.7092760180995475`, gap `10.7375990863037` attempts
- 2024: Pearson `0.502344821228796`
- 2025: Pearson `0.5483085094177219`

## Scientific interpretation

The prior QB-WR receiving-yard residual coupling could partly reflect the accounting identity that QB passing yards become receiver yards. This result goes materially deeper: the independent models' **QB attempt error and WR target-opportunity error are strongly coupled**.

That supports a shared latent team-game pass-volume mechanism as a legitimate future cross-position integration target: one pregame team plays/pass-tendency/game-script state can feed QB attempts and team target mass in the same Monte Carlo iteration, while individual WR target allocation remains a separate player-level process.

This does not authorize realized WR targets as QB inputs, realized QB attempts as WR inputs, sportsbook information upstream, or collapsing independent QB/WR research into one model.

M72 remains a failed aggregate explosive-weapon x defense screen. This is a different mechanism.

- Sportsbook inputs used: `false`
- Model fitting used: `false`
- Production changed: `false`