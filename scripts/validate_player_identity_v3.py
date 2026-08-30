#!/usr/bin/env python3
"""Validate Player Identity v3 artifacts before downstream modeling.

This is intentionally diagnostic plus fail-closed on structural identity errors.
Temporary identities are allowed for genuine rookies/new players, but are counted
and surfaced so a provider regression cannot quietly turn the entire slate into
name-only fallbacks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path("data")
FORM = DATA / "player_form.csv"
REGISTRY = DATA / "player_identity_registry.csv"
SLATE = DATA / "player_identity_slate.csv"
OUT = DATA / "player_identity_validation.csv"


def _read(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise RuntimeError(f"Required identity artifact missing/empty: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"Required identity artifact has zero rows: {path}")
    return df


def validate(form: pd.DataFrame, registry: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    required = {
        "player", "team", "player_identity_key", "identity_resolution", "identity_confidence"
    }
    missing = required - set(form.columns)
    if missing:
        raise RuntimeError(f"PlayerForm missing Player Identity v3 columns: {sorted(missing)}")

    keys = form["player_identity_key"].astype("string").fillna("").str.strip()
    if keys.eq("").any():
        sample = form.loc[keys.eq(""), [c for c in ("player", "team") if c in form.columns]].head(20)
        raise RuntimeError(f"PlayerForm has unresolved player identities: {sample.to_dict('records')}")

    teams = form["team"].astype("string").fillna("").str.strip()
    if teams.eq("").any():
        raise RuntimeError("PlayerForm has missing team identity")

    # One stable person cannot appear on two current teams in the same slate.
    current_team_count = (
        form.assign(_identity=keys, _team=teams)
        .groupby("_identity", dropna=False)["_team"]
        .nunique()
    )
    cross_team = current_team_count.loc[current_team_count.gt(1)]
    if not cross_team.empty:
        raise RuntimeError(
            f"Same resolved player identity appears on multiple current teams: {cross_team.to_dict()}"
        )

    # PlayerForm should have one row per team/player identity.  Duplicate market
    # offers belong downstream in metrics, not in the player baseline artifact.
    duplicate = form.assign(_identity=keys, _team=teams).duplicated(
        ["_team", "_identity"], keep=False
    )
    if duplicate.any():
        sample = form.loc[duplicate, [c for c in ("player", "team", "player_identity_key") if c in form.columns]].head(20)
        raise RuntimeError(f"Duplicate PlayerForm identities: {sample.to_dict('records')}")

    resolution = form["identity_resolution"].astype("string").fillna("").str.strip()
    if resolution.eq("").any():
        raise RuntimeError("PlayerForm contains identities without resolution provenance")

    confidence = pd.to_numeric(form["identity_confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.between(0.0, 1.0, inclusive="both").all():
        raise RuntimeError("PlayerForm identity confidence is missing/out of range")

    temp = keys.str.startswith("temp:")
    stable = keys.str.startswith("gsis:")
    fallback_hist = keys.str.startswith("hist-name:")
    unknown_prefix = ~(temp | stable | fallback_hist)
    if unknown_prefix.any():
        bad = sorted(keys.loc[unknown_prefix].unique().tolist())[:20]
        raise RuntimeError(f"Unknown player identity key scheme: {bad}")

    # Stable identities should carry their source player ID.
    if stable.any():
        if "player_id" not in form.columns:
            raise RuntimeError("Stable Player Identity v3 rows missing player_id column")
        pid = form["player_id"].astype("string").fillna("").str.strip()
        if pid.loc[stable].eq("").any():
            raise RuntimeError("Stable GSIS identity row is missing player_id")

    rows = []
    for method, count in resolution.value_counts(dropna=False).sort_index().items():
        rows.append({"metric": f"resolution:{method}", "value": int(count)})
    rows += [
        {"metric": "slate_players", "value": int(len(form))},
        {"metric": "stable_gsis", "value": int(stable.sum())},
        {"metric": "historical_name_fallback", "value": int(fallback_hist.sum())},
        {"metric": "temporary_new_or_unmapped", "value": int(temp.sum())},
        {"metric": "registry_rows", "value": int(len(registry))},
        {"metric": "identity_slate_rows", "value": int(len(slate))},
    ]
    return pd.DataFrame(rows)


def main() -> int:
    form = _read(FORM)
    registry = _read(REGISTRY, required=False)
    slate = _read(SLATE)
    summary = validate(form, registry, slate)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)

    values = dict(zip(summary["metric"], summary["value"]))
    print(
        "[player_identity_v3] "
        f"players={values.get('slate_players', 0)} "
        f"stable={values.get('stable_gsis', 0)} "
        f"temporary={values.get('temporary_new_or_unmapped', 0)} "
        f"hist_fallback={values.get('historical_name_fallback', 0)}"
    )
    resolution_rows = summary.loc[summary["metric"].str.startswith("resolution:")]
    print(
        "[player_identity_v3] resolution methods:",
        {r.metric.removeprefix("resolution:"): int(r.value) for r in resolution_rows.itertuples()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
