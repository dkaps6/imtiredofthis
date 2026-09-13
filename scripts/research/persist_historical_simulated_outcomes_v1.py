#!/usr/bin/env python3
"""Rebuild leakage-safe historical Monte Carlo outcome arrays for fair-probability research.

This script intentionally reruns only the historical football simulation path.
Sportsbook lines/odds are not inputs. The saved arrays are keyed by historical
pregame identity and are later consumed by a separate downstream grader.

The component file is used only as an integrity checksum: every observed-row
mc_proj emitted by the canonical walk-forward must match the mean of the
reconstructed array under the identical seed/iteration contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.simulation_v2 import lookup, simulate
from scripts.utils.canonical_names import canon_team

# Include opponent even though authoritative team/week implies it. This makes
# wrong-opponent/stale sidecars fail closed at the persistence seam itself.
KEYS = ["season", "week", "team", "opponent", "player_clean_key", "market"]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _read_optional(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _exact_week(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if not {"season", "week"}.issubset(x.columns):
        raise RuntimeError("historical enrichment requires season/week")
    s = pd.to_numeric(x["season"], errors="coerce")
    w = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[s.eq(int(season)) & w.eq(int(week))].copy()
    return x.drop(columns=["season", "week"], errors="ignore")


def _parse_weeks(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out or range(1, 19)))


def _historical_outcomes(sims, row: pd.Series) -> np.ndarray | None:
    outcomes = lookup(sims, row, str(row["market"]))
    if outcomes is None or len(outcomes) == 0:
        return None
    out = np.asarray(outcomes, dtype=float)
    if str(row["market"]).lower() == "pass_yards":
        attempt_rate = pd.to_numeric(
            pd.Series([row.get("mc_pass_attempts_per_dropback")]), errors="coerce"
        ).iloc[0]
        share = pd.to_numeric(
            pd.Series([row.get("qb_pass_att_share")]), errors="coerce"
        ).iloc[0]
        if pd.notna(attempt_rate):
            out = out * float(np.clip(attempt_rate, 0.50, 1.00))
        if pd.notna(share):
            out = out * float(np.clip(share, 0.0, 1.0))
    return out


def _canon_keys(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = sorted(set(KEYS) - set(x.columns))
    if missing:
        raise RuntimeError(f"historical distribution identity missing columns: {missing}")
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    x["market"] = x["market"].astype(str).str.lower()
    if x["team"].eq("").any() or x["opponent"].eq("").any():
        raise RuntimeError("historical distribution identity contains invalid team/opponent")
    return x


def persist_season(
    *,
    player_logs_path: Path,
    team_weekly_path: Path,
    schedule_path: Path,
    universe_dir: Path,
    component_file: Path,
    season: int,
    prior_season: int,
    weeks: list[int],
    out_dir: Path,
    injuries_path: Path | None,
    weather_path: Path | None,
    iterations: int,
    seed_offset: int = 42,
) -> pd.DataFrame:
    player_logs = _read(player_logs_path, "player logs")
    team_weekly = _read(team_weekly_path, "historical team-week features")
    schedule = _read(schedule_path, "historical schedule")
    component = _canon_keys(_read(component_file, "component predictions"))
    injuries_history = _read_optional(injuries_path)
    weather_history = _read_optional(weather_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict] = []
    for week in weeks:
        universe_path = universe_dir / f"{season}_week_{week:02d}.csv"
        universe = _read(universe_path, f"pregame universe for {season} week {week}")
        injuries = _exact_week(injuries_history, season, week)
        weather = _exact_week(weather_history, season, week)
        seed = int(seed_offset) + int(week)

        bundle = build_historical_context_bundle(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=int(season),
            week=int(week),
            prior_season=int(prior_season),
            injuries=injuries,
            weather=weather,
        )
        metrics = build_mc_predictions(bundle, iterations=int(iterations), seed=seed)

        # build_mc_predictions already simulated once to emit mc_proj. Running the
        # same deterministic simulator again with the same prepared metrics/seed
        # reproduces that exact football distribution without introducing market data.
        # The exact mean check below hard-fails if deterministic equivalence breaks.
        sims = simulate(metrics, iterations=int(iterations), seed=seed)

        arrays: dict[str, np.ndarray] = {}
        rows: list[dict] = []
        for i, (_, row) in enumerate(metrics.iterrows()):
            arr = _historical_outcomes(sims, row)
            if arr is None:
                continue
            if len(arr) != int(iterations):
                raise RuntimeError(
                    f"{season} W{week:02d}: simulation draw-count mismatch "
                    f"{row.get('player_clean_key')} {row.get('market')}: "
                    f"{len(arr)} != {int(iterations)}"
                )
            array_key = f"a{i:06d}"
            arrays[array_key] = arr
            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "team": canon_team(row.get("team")),
                    "opponent": canon_team(row.get("opponent")),
                    "player": row.get("player"),
                    "player_clean_key": str(row.get("player_clean_key")),
                    "market": str(row.get("market")).lower(),
                    "event_id": row.get("event_id"),
                    "array_key": array_key,
                    "draws": int(len(arr)),
                    "mc_mean": float(np.mean(arr)),
                    # Match production's recorded model SD convention.
                    "mc_sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                }
            )

        meta = pd.DataFrame(rows)
        if meta.empty:
            raise RuntimeError(f"{season} W{week:02d}: no simulated arrays")
        meta = _canon_keys(meta)
        if meta.duplicated(KEYS).any():
            bad = meta.loc[meta.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")
            raise RuntimeError(f"{season} W{week:02d}: duplicate distribution identities: {bad}")

        shard = f"{season}_week_{week:02d}.npz"
        meta_file = f"{season}_week_{week:02d}_metadata.csv"
        meta["npz_file"] = shard

        comp = component.loc[
            component["season"].eq(int(season)) & component["week"].eq(int(week))
        ].copy()
        chk = comp[KEYS + ["mc_proj"]].merge(
            meta[KEYS + ["mc_mean"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        missing = int(chk["mc_mean"].isna().sum())
        if missing:
            sample = chk.loc[chk["mc_mean"].isna(), KEYS].head(10).to_dict("records")
            raise RuntimeError(
                f"{season} W{week:02d}: {missing} component rows missing distribution lineage: {sample}"
            )
        delta = (
            pd.to_numeric(chk["mc_proj"], errors="coerce")
            - pd.to_numeric(chk["mc_mean"], errors="coerce")
        ).abs()
        max_delta = float(delta.max()) if len(delta) else 0.0
        if not np.isfinite(max_delta) or max_delta > 1e-8:
            raise RuntimeError(
                f"{season} W{week:02d}: reconstructed MC mean mismatch max_abs={max_delta:.12g}"
            )

        np.savez_compressed(out_dir / shard, **arrays)
        meta.to_csv(out_dir / meta_file, index=False)
        audit_rows.append(
            {
                "season": int(season),
                "week": int(week),
                "seed": seed,
                "iterations": int(iterations),
                "distribution_rows": int(len(meta)),
                "observed_component_rows_checked": int(len(chk)),
                "missing_component_rows": missing,
                "max_abs_mc_mean_delta": max_delta,
                "status": "PASS",
            }
        )
        print(
            f"[empirical-dist] {season} W{week:02d} arrays={len(meta)} "
            f"observed_checked={len(chk)} max_mc_delta={max_delta:.3g}"
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / f"{season}_distribution_rebuild_audit.csv", index=False)
    return audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--prior-season", type=int, required=True)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir", type=Path, required=True)
    ap.add_argument("--component-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    ap.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument(
        "--seed-offset", type=int, default=42,
        help="week-offset added to derive the per-week simulation seed (seed = offset + week). "
             "Default 42 matches walk_forward.py's standard component-prediction seed policy, "
             "used by every existing caller of this script. Callers whose component file was "
             "built with a different seed policy (e.g. build_qb_synthesis_confirmation_inputs_v1.py's "
             "53 + week) must pass the matching offset here or the mean-reproduction checksum "
             "below will correctly fail closed.",
    )
    a = ap.parse_args()

    persist_season(
        player_logs_path=a.player_logs,
        team_weekly_path=a.team_weekly,
        schedule_path=a.schedule,
        universe_dir=a.universe_dir,
        component_file=a.component_file,
        season=a.season,
        prior_season=a.prior_season,
        weeks=_parse_weeks(a.weeks),
        out_dir=a.out_dir,
        injuries_path=a.injuries,
        weather_path=a.weather,
        iterations=a.iterations,
        seed_offset=a.seed_offset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
