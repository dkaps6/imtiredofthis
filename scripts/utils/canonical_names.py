#!/usr/bin/env python3
"""
Canonicalize player names in player_form.csv

Priority:
  1) roles_ourlads: player_key -> player (ignore middle initials; keep suffixes)
  2) manual overrides: player_source_name -> full First Last (unconditional if --force-manual)

Also updates canonical name fields when present:
  canonical_player_name, player_name_canonical, player_canonical, player_display, display_name
"""

import argparse
import re
from pathlib import Path
import pandas as pd

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

CANON_COLS = [
    "canonical_player_name",
    "player_name_canonical",
    "player_canonical",
    "player_display",
    "display_name",
]


def strip_middle_initial(full_name: str) -> str:
    """Return 'First Last' (preserve suffix like 'Jr', 'III')."""
    if not isinstance(full_name, str):
        return ""
    parts = re.split(r"\s+", full_name.replace(",", " ").strip())
    parts = [p.replace(".", "") for p in parts if p.strip()]
    if not parts:
        return ""
    first = parts[0]
    last = parts[-1]
    if last.lower() in SUFFIXES and len(parts) >= 3:
        last = parts[-2] + " " + parts[-1]
    return f"{first} {last}".strip()

def norm_key(s: str) -> str:
    """Lowercase; remove spaces, apostrophes, hyphens, periods."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for ch in (" ", "'", "-", "."):
        s = s.replace(ch, "")
    return s


def load_roles_map(path: Path) -> dict:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "player_key" not in cols or "player" not in cols:
        raise SystemExit("roles_ourlads.csv must contain 'player_key' and 'player' columns.")
    key_col = cols["player_key"]
    name_col = cols["player"]
    df["_key"] = df[key_col].astype(str).map(norm_key)
    df["_full"] = df[name_col].astype(str).map(strip_middle_initial)
    return dict(zip(df["_key"], df["_full"]))


def load_manual_map(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    need = {"player_source_name", "full_name"}
    if not need.issubset(df.columns):
        raise SystemExit("manual_name_overrides.csv must have columns: player_source_name, full_name")
    df["_key"] = df["player_source_name"].astype(str).map(norm_key)
    return dict(zip(df["_key"], df["full_name"]))


def canonicalize(
    player_form: Path,
    roles_ourlads: Path,
    manual_overrides: Path,
    out: Path,
    force_manual: bool,
) -> None:
    pf = pd.read_csv(player_form)

    if "player_source_name" not in pf.columns:
        raise SystemExit("player_form.csv must include 'player_source_name'.")

    if "player" not in pf.columns:
        pf["player"] = pf["player_source_name"]

    roles_map = load_roles_map(roles_ourlads)
    manual_map = load_manual_map(manual_overrides)

    keys = pf["player_source_name"].astype(str).map(norm_key)

    # Pass 1: roles mapping
    from_roles = keys.map(roles_map)

    # Pass 2: manual mapping
    from_manual = keys.map(manual_map)

    if force_manual:
        # roles as base; manual overrides wherever present
        full_names = from_roles.where(~keys.isin(manual_map.keys()), from_manual)
    else:
        # only fill where roles didn't resolve
        full_names = from_roles.fillna(from_manual)

    # Write into player + canonical columns
    pf["player"] = full_names.fillna(pf["player"])
    for c in CANON_COLS:
        if c in pf.columns:
            pf[c] = full_names.fillna(pf[c])

    out.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(out, index=False)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-form", type=Path, required=True)
    ap.add_argument("--roles-ourlads", type=Path, required=True)
    ap.add_argument("--manual-overrides", type=Path, default=Path("data/manual_name_overrides.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force-manual", action="store_true", help="Always apply manual overrides where keys exist.")
    ns = ap.parse_args(argv)

    canonicalize(
        player_form=ns.player_form,
        roles_ourlads=ns.roles_ourlads,
        manual_overrides=ns.manual_overrides,
        out=ns.out,
        force_manual=ns.force_manual,
    )

if __name__ == "__main__":
    main()
