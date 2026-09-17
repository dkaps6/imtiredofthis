from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_VERSION = "BDB_2024_CONTACT_GEOMETRY_V1"
FIELD_LENGTH_YARDS = 120.0
CONTACT_DISTANCE_YARDS = 1.0
CONTACT_MIN_CONSECUTIVE_FRAMES = 2

TRACKING_REQUIRED = {"gameId", "playId", "frameId", "playDirection", "x", "y", "club"}
PLAY_REQUIRED = {"gameId", "playId", "ballCarrierId"}
TACKLE_REQUIRED = {"gameId", "playId", "nflId", "tackle", "assist", "forcedFumble", "pff_missedTackle"}


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def offense_x(x: pd.Series, play_direction: pd.Series) -> pd.Series:
    x_num = _num(x)
    d = play_direction.astype("string").str.lower().str.strip()
    invalid = ~d.isin(["left", "right"])
    if invalid.any():
        bad = sorted(d.loc[invalid].dropna().unique().tolist())
        raise ValueError(f"Unresolvable playDirection values: {bad[:10]}")
    return pd.Series(
        np.where(d.eq("left"), FIELD_LENGTH_YARDS - x_num, x_num),
        index=x.index,
        dtype="float64",
    )


def normalize_tracking(raw: pd.DataFrame) -> pd.DataFrame:
    _require_columns(raw, TRACKING_REQUIRED, "tracking")
    x = raw.copy()
    for c in ["gameId", "playId", "frameId", "nflId"]:
        if c in x.columns:
            x[c] = _num(x[c])
    x["x"] = _num(x["x"])
    x["y"] = _num(x["y"])
    for c in ["s", "a", "dis", "o", "dir"]:
        if c in x.columns:
            x[c] = _num(x[c])
    if x[["gameId", "playId", "frameId", "x", "y"]].isna().any(axis=None):
        raise ValueError("tracking contains null required numeric key/coordinate values")
    x["gameId"] = x["gameId"].astype("int64")
    x["playId"] = x["playId"].astype("int64")
    x["frameId"] = x["frameId"].astype("int64")
    x["club"] = x["club"].astype("string").str.strip()
    x["playDirection"] = x["playDirection"].astype("string").str.lower().str.strip()
    x["x_offense"] = offense_x(x["x"], x["playDirection"])
    x["source_dataset"] = "BDB_2024"
    football_mask = x["club"].str.lower().eq("football")
    if "displayName" in x.columns:
        football_mask |= x["displayName"].astype("string").str.lower().eq("football")
    x["is_football"] = football_mask
    players = x.loc[~football_mask].copy()
    if players["nflId"].isna().any():
        raise ValueError("player tracking contains null nflId rows")
    players["nflId"] = players["nflId"].astype("int64")
    dup = players.duplicated(["gameId", "playId", "frameId", "nflId"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate player-frame keys: {int(dup.sum())}")
    return x.sort_values(
        ["gameId", "playId", "frameId", "club", "nflId"], na_position="last"
    ).reset_index(drop=True)


def normalize_plays(raw: pd.DataFrame) -> pd.DataFrame:
    _require_columns(raw, PLAY_REQUIRED, "plays")
    x = raw.copy()
    for c in ["gameId", "playId", "ballCarrierId"]:
        x[c] = _num(x[c])
    if x[["gameId", "playId"]].isna().any(axis=None):
        raise ValueError("plays contain null game/play keys")
    x["gameId"] = x["gameId"].astype("int64")
    x["playId"] = x["playId"].astype("int64")
    if x.duplicated(["gameId", "playId"]).any():
        raise ValueError("plays contain duplicate gameId/playId keys")
    return x.sort_values(["gameId", "playId"]).reset_index(drop=True)


def normalize_tackles(raw: pd.DataFrame) -> pd.DataFrame:
    _require_columns(raw, TACKLE_REQUIRED, "tackles")
    x = raw.copy()
    for c in [
        "gameId",
        "playId",
        "nflId",
        "tackle",
        "assist",
        "forcedFumble",
        "pff_missedTackle",
    ]:
        x[c] = _num(x[c])
    x = x.dropna(subset=["gameId", "playId", "nflId"]).copy()
    x[["gameId", "playId", "nflId"]] = x[["gameId", "playId", "nflId"]].astype("int64")
    for c in ["tackle", "assist", "forcedFumble", "pff_missedTackle"]:
        x[c] = x[c].fillna(0).astype("int8")
    x["source_dataset"] = "BDB_2024"
    return x.sort_values(["gameId", "playId", "nflId"]).reset_index(drop=True)


def _first_event_frame(play_track: pd.DataFrame, labels: Iterable[str]) -> int | None:
    if "event" not in play_track.columns:
        return None
    allowed = {str(v).lower() for v in labels}
    e = play_track["event"].astype("string").str.lower().str.strip()
    z = play_track.loc[e.isin(allowed), "frameId"]
    return None if z.empty else int(z.min())


def _carrier_track(play_track: pd.DataFrame, carrier_id: int) -> pd.DataFrame:
    return play_track.loc[
        (~play_track["is_football"]) & play_track["nflId"].eq(carrier_id)
    ].sort_values("frameId").copy()


def _defender_rows(play_track: pd.DataFrame, play: pd.Series, carrier_id: int) -> pd.DataFrame:
    dclub = str(play.get("defensiveTeam", "")).strip()
    if dclub:
        return play_track.loc[
            (~play_track["is_football"]) & play_track["club"].eq(dclub)
        ].copy()
    pclub = str(play.get("possessionTeam", "")).strip()
    if pclub:
        return play_track.loc[
            (~play_track["is_football"])
            & (~play_track["club"].eq(pclub))
            & (~play_track["nflId"].eq(carrier_id))
        ].copy()
    return play_track.loc[
        (~play_track["is_football"]) & (~play_track["nflId"].eq(carrier_id))
    ].copy()


def _distance_frame_table(carrier: pd.DataFrame, defenders: pd.DataFrame) -> pd.DataFrame:
    ccols = ["frameId", "x_offense", "y"]
    extra = [c for c in ["s", "a", "dir", "o"] if c in carrier.columns]
    c = carrier[ccols + extra].rename(
        columns={
            "x_offense": "carrier_x",
            "y": "carrier_y",
            **{k: f"carrier_{k}" for k in extra},
        }
    )
    dextra = [c for c in ["s", "a", "dir", "o"] if c in defenders.columns]
    d = defenders[["frameId", "nflId", "x_offense", "y"] + dextra].rename(
        columns={
            "x_offense": "defender_x",
            "y": "defender_y",
            **{k: f"defender_{k}" for k in dextra},
        }
    )
    m = d.merge(c, on="frameId", how="inner", validate="many_to_one")
    m["distance"] = np.hypot(
        m["defender_x"] - m["carrier_x"], m["defender_y"] - m["carrier_y"]
    )
    return m.sort_values(["frameId", "distance", "nflId"]).reset_index(drop=True)


def _first_contact_candidates(
    dist: pd.DataFrame, start_frame: int, end_frame: int
) -> tuple[int | None, list[int]]:
    q = dist.loc[dist["frameId"].between(start_frame, end_frame)].copy()
    qualifying: list[tuple[int, int]] = []
    for nfl_id, g in q.groupby("nflId", sort=False):
        frames = (
            g.loc[g["distance"].le(CONTACT_DISTANCE_YARDS), "frameId"]
            .astype(int)
            .sort_values()
            .tolist()
        )
        if not frames:
            continue
        run_start = prev = frames[0]
        run_len = 1
        found = run_start if CONTACT_MIN_CONSECUTIVE_FRAMES == 1 else None
        for fr in frames[1:]:
            if fr == prev + 1:
                run_len += 1
            else:
                run_start, run_len = fr, 1
            prev = fr
            if run_len >= CONTACT_MIN_CONSECUTIVE_FRAMES:
                found = run_start
                break
        if found is not None:
            qualifying.append((int(found), int(nfl_id)))
    if not qualifying:
        return None, []
    first = min(fr for fr, _ in qualifying)
    return first, sorted({pid for fr, pid in qualifying if fr == first})


def _nearest_summary(dist: pd.DataFrame, frame_id: int, prefix: str) -> dict:
    z = dist.loc[dist["frameId"].eq(frame_id)].sort_values("distance")
    if z.empty:
        return {
            f"{prefix}_nearest_defender_distance": np.nan,
            f"{prefix}_second_nearest_defender_distance": np.nan,
            f"{prefix}_defenders_within_1": 0,
            f"{prefix}_defenders_within_2": 0,
            f"{prefix}_defenders_within_3": 0,
            f"{prefix}_defenders_within_5": 0,
        }
    d = z["distance"].to_numpy(float)
    return {
        f"{prefix}_nearest_defender_distance": float(d[0]),
        f"{prefix}_second_nearest_defender_distance": float(d[1]) if len(d) > 1 else np.nan,
        f"{prefix}_defenders_within_1": int((d <= 1).sum()),
        f"{prefix}_defenders_within_2": int((d <= 2).sum()),
        f"{prefix}_defenders_within_3": int((d <= 3).sum()),
        f"{prefix}_defenders_within_5": int((d <= 5).sum()),
    }


def derive_contact_features(
    tracking: pd.DataFrame, plays: pd.DataFrame, tackles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, disposition_rows = [], []
    track_groups = {
        (g, p): x for (g, p), x in tracking.groupby(["gameId", "playId"], sort=False)
    }
    tackle_groups = {
        (g, p): x for (g, p), x in tackles.groupby(["gameId", "playId"], sort=False)
    }
    for _, play in plays.iterrows():
        gid, pid = int(play["gameId"]), int(play["playId"])
        key = (gid, pid)
        base = {"gameId": gid, "playId": pid}
        if pd.isna(play["ballCarrierId"]):
            disposition_rows.append({**base, "benchmark_disposition": "NO_BALL_CARRIER"})
            continue
        carrier_id = int(play["ballCarrierId"])
        pt = track_groups.get(key)
        if pt is None or pt.empty:
            disposition_rows.append({**base, "benchmark_disposition": "MISSING_TRACKING"})
            continue
        carrier = _carrier_track(pt, carrier_id)
        defenders = _defender_rows(pt, play, carrier_id)
        if carrier.empty:
            disposition_rows.append({**base, "benchmark_disposition": "NO_BALL_CARRIER"})
            continue
        if defenders.empty:
            disposition_rows.append({**base, "benchmark_disposition": "MISSING_TRACKING"})
            continue
        start = _first_event_frame(pt, ["handoff"])
        handoff_method = "source_handoff_event"
        if start is None:
            start = _first_event_frame(pt, ["ball_snap"])
            handoff_method = "source_ball_snap_fallback"
        if start is None:
            disposition_rows.append(
                {**base, "benchmark_disposition": "NO_CONTACT_EVENT_RESOLUTION"}
            )
            continue
        end = _first_event_frame(
            pt, ["tackle", "out_of_bounds", "touchdown", "fumble", "qb_slide"]
        )
        if end is None:
            end = int(pt["frameId"].max())
        dist = _distance_frame_table(carrier, defenders)
        fc_frame, fc_ids = _first_contact_candidates(dist, start, end)
        if fc_frame is None:
            disposition_rows.append(
                {**base, "benchmark_disposition": "NO_CONTACT_EVENT_RESOLUTION"}
            )
            continue
        c_start = carrier.loc[carrier["frameId"].eq(start)]
        c_fc = carrier.loc[carrier["frameId"].eq(fc_frame)]
        c_end = carrier.loc[carrier["frameId"].le(end)].tail(1)
        if c_start.empty or c_fc.empty or c_end.empty:
            disposition_rows.append(
                {**base, "benchmark_disposition": "NO_CONTACT_EVENT_RESOLUTION"}
            )
            continue
        handoff_x = float(c_start.iloc[0]["x_offense"])
        fc_x = float(c_fc.iloc[0]["x_offense"])
        end_x = float(c_end.iloc[0]["x_offense"])
        source_t = tackle_groups.get(key, pd.DataFrame())
        source_ids = (
            set(source_t["nflId"].astype(int).tolist()) if not source_t.empty else set()
        )

        def vals(col: str) -> list[int]:
            if source_t.empty:
                return []
            return sorted(source_t.loc[source_t[col].eq(1), "nflId"].astype(int).tolist())

        fc_row = c_fc.iloc[0]
        row = {
            "gameId": gid,
            "playId": pid,
            "ballCarrierId": carrier_id,
            "handoffFrameId": int(start),
            "handoffMethod": handoff_method,
            "firstContactFrameId": int(fc_frame),
            "playEndFrameId": int(c_end.iloc[0]["frameId"]),
            "firstContactDefenderIds": "|".join(map(str, fc_ids)),
            "sourcePrimaryTacklerIds": "|".join(map(str, vals("tackle"))),
            "sourceAssistIds": "|".join(map(str, vals("assist"))),
            "sourceMissedTacklerIds": "|".join(map(str, vals("pff_missedTackle"))),
            "firstContactSourceLabelOverlap": int(bool(set(fc_ids) & source_ids)),
            "firstContactXOffense": fc_x,
            "firstContactY": float(fc_row["y"]),
            "yardsBeforeContactGeom": fc_x - handoff_x,
            "yardsAfterFirstContactGeom": end_x - fc_x,
            "carrierSpeedAtFirstContact": (
                float(fc_row["s"]) if "s" in fc_row and pd.notna(fc_row["s"]) else np.nan
            ),
            "carrierAccelAtFirstContact": (
                float(fc_row["a"]) if "a" in fc_row and pd.notna(fc_row["a"]) else np.nan
            ),
            "simultaneousFirstContactDefenders": len(fc_ids),
            "featureVersion": FEATURE_VERSION,
            "sourceDataset": "BDB_2024",
        }
        row.update(_nearest_summary(dist, start, "handoff"))
        row.update(_nearest_summary(dist, fc_frame, "first_contact"))
        rows.append(row)
        disposition_rows.append({**base, "benchmark_disposition": "SCOREABLE"})
    features = pd.DataFrame(rows)
    dispositions = pd.DataFrame(disposition_rows).sort_values(
        ["gameId", "playId"]
    ).reset_index(drop=True)
    if not features.empty:
        features = features.sort_values(["gameId", "playId"]).reset_index(drop=True)
    return features, dispositions


def build_qa(features: pd.DataFrame, dispositions: pd.DataFrame, manifest: dict) -> dict:
    counts = (
        dispositions["benchmark_disposition"].value_counts(dropna=False).to_dict()
        if len(dispositions)
        else {}
    )
    overlap = float(features["firstContactSourceLabelOverlap"].mean()) if len(features) else None
    return {
        "feature_version": FEATURE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempted_plays": int(len(dispositions)),
        "scoreable_plays": int(counts.get("SCOREABLE", 0)),
        "disposition_counts": {str(k): int(v) for k, v in counts.items()},
        "first_contact_source_label_overlap_rate": overlap,
        "source_manifest": manifest,
        "predictive_metrics_computed": False,
        "sportsbook_inputs_used": False,
        "production_changed": False,
    }


def load_inputs(input_dir: Path):
    plays_path = input_dir / "plays.csv"
    tackles_path = input_dir / "tackles.csv"
    if not plays_path.exists() or not tackles_path.exists():
        raise FileNotFoundError("Expected plays.csv and tackles.csv in input directory")
    tracking_paths = sorted(input_dir.glob("tracking_week_*.csv"))
    if not tracking_paths:
        raise FileNotFoundError("No tracking_week_*.csv files found")
    manifest = {
        "files": [
            {"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in [plays_path, tackles_path, *tracking_paths]
        ]
    }
    plays = normalize_plays(pd.read_csv(plays_path))
    tackles = normalize_tackles(pd.read_csv(tackles_path))
    tracking = normalize_tracking(
        pd.concat((pd.read_csv(p) for p in tracking_paths), ignore_index=True, sort=False)
    )
    return tracking, plays, tackles, manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build BDB 2024 contact-geometry benchmark artifacts."
    )
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    tracking, plays, tackles, manifest = load_inputs(args.input_dir)
    features, dispositions = derive_contact_features(tracking, plays, tackles)
    qa = build_qa(features, dispositions, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_dir / "rb_contact_features.csv", index=False)
    dispositions.to_csv(args.out_dir / "benchmark_dispositions.csv", index=False)
    (args.out_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "qa_summary.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "attempted_plays": len(dispositions),
                "scoreable_plays": int(
                    (dispositions["benchmark_disposition"] == "SCOREABLE").sum()
                ),
                "out_dir": str(args.out_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
