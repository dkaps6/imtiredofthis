#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

EXPECTED = {
    ("CHI", "D'Andre Swift", "d'andreswift"): ("dandreswift", "RB1"),
    ("MIA", "De'Von Achane", "de'vonachane"): ("devonachane", "RB1"),
    ("WAS", "Jacory Croskey-Merritt", "jacorycroskey-merritt"): ("jacorycroskeymerritt", "RB1"),
}


def one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-r26r-root", type=Path, required=True)
    ap.add_argument("--staged-r26r-root", type=Path, required=True)
    ap.add_argument("--full-slate-root", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, required=True)
    args = ap.parse_args()

    src = args.source_r26r_root.resolve()
    dst = args.staged_r26r_root.resolve()
    full = args.full_slate_root.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    src_roles = one(src, "source_current_roles_identity_only.csv")
    dst_roles = one(dst, "source_current_roles_identity_only.csv")
    rush_path = one(full, "rb_rush_synthesis_context.csv")

    before = pd.read_csv(src_roles, low_memory=False)
    after = before.copy()
    rush = pd.read_csv(rush_path, low_memory=False)
    production_keys = set(zip(rush["team"].astype(str), rush["player_clean_key"].astype(str)))

    changes: list[dict] = []
    for (team, player, old_key), (new_key, expected_role) in EXPECTED.items():
        mask = (
            after["team"].astype(str).eq(team)
            & after["player"].astype(str).eq(player)
            & after["player_clean_key"].astype(str).eq(old_key)
        )
        if int(mask.sum()) != 1:
            raise RuntimeError(f"expected exactly one frozen role row for {(team, player, old_key)}, found {int(mask.sum())}")
        idx = after.index[mask][0]
        if str(after.at[idx, "model_role"]) != expected_role:
            raise RuntimeError(f"unexpected frozen model_role for {player}: {after.at[idx, 'model_role']}")
        if (team, new_key) not in production_keys:
            raise RuntimeError(f"authorized replacement key {(team, new_key)} is absent from protected production universe")
        after.at[idx, "player_clean_key"] = new_key
        changes.append(
            {
                "row_index": int(idx),
                "team": team,
                "player": player,
                "column": "player_clean_key",
                "source_value": old_key,
                "staged_value": new_key,
                "model_role": expected_role,
            }
        )

    if len(changes) != 3:
        raise RuntimeError(f"expected exactly 3 authorized cell changes, got {len(changes)}")

    # Verify every cell other than the three authorized player_clean_key cells is identical.
    changed_cells: list[tuple[int, str]] = []
    for idx in before.index:
        for col in before.columns:
            left = before.at[idx, col]
            right = after.at[idx, col]
            equal = (pd.isna(left) and pd.isna(right)) or left == right
            if not equal:
                changed_cells.append((int(idx), str(col)))
    expected_cells = sorted((c["row_index"], "player_clean_key") for c in changes)
    if sorted(changed_cells) != expected_cells:
        raise RuntimeError(f"unauthorized staged role-cell changes: {changed_cells}")

    after.to_csv(dst_roles, index=False)

    src_files = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())
    if src_files != dst_files:
        raise RuntimeError("staged R26R file set differs from source")
    role_rel = src_roles.relative_to(src)
    mismatches = []
    for rel in src_files:
        if rel == role_rel:
            continue
        if sha256_file(src / rel) != sha256_file(dst / rel):
            mismatches.append(str(rel))
    if mismatches:
        raise RuntimeError(f"non-role R26R files changed during compatibility staging: {mismatches}")

    audit = {
        "repair": "RB_WEEK1_2026_PREGAME_READINESS_RUN1_ROLE_KEY_MECHANICAL_REPAIR_V1",
        "source_roles_sha256": sha256_file(src_roles),
        "staged_roles_sha256": sha256_file(dst_roles),
        "authorized_changes": changes,
        "authorized_change_count": len(changes),
        "non_role_files_byte_exact": True,
        "non_role_file_count": len(src_files) - 1,
        "football_values_changed": False,
        "roles_changed": False,
        "depth_values_changed": False,
        "r26_arrays_changed": False,
        "fuzzy_matching_used": False,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
