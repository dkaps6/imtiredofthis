#!/usr/bin/env python3
"""
Backwards-compatible name canonicalization utilities.

Exports:
  - canonicalize_player_name(source_key: str) -> str
      (legacy API expected by make_player_form.py)
  - build_roles_map(), build_manual_map() helpers
  - norm_key(), strip_middle_initial()

The function canonicalize_player_name uses:
  1) roles_ourlads.csv (player_key -> full name, ignoring middle initials)
  2) data/manual_name_overrides.csv (player_source_name -> full name), overrides roles
"""

import re
from pathlib import Path
import pandas as pd
from functools import lru_cache

# ---------- helpers ----------

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def strip_middle_initial(full_name: str) -> str:
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
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for ch in (" ", "'", "-", "."):
        s = s.replace(ch, "")
    return s

# ---------- map builders (cached) ----------

@lru_cache(maxsize=1)
def build_roles_map(roles_path: str = "data/roles_ourlads.csv") -> dict:
    p = Path(roles_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    if "player_key" not in cols or "player" not in cols:
        return {}
    key_col = cols["player_key"]
    name_col = cols["player"]
    df["_key"] = df[key_col].astype(str).map(norm_key)
    df["_full"] = df[name_col].astype(str).map(strip_middle_initial)
    return dict(zip(df["_key"], df["_full"]))

@lru_cache(maxsize=1)
def build_manual_map(overrides_path: str = "data/manual_name_overrides.csv") -> dict:
    p = Path(overrides_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    need = {"player_source_name", "full_name"}
    if not need.issubset(df.columns):
        return {}
    df["_key"] = df["player_source_name"].astype(str).map(norm_key)
    return dict(zip(df["_key"], df["full_name"]))

# ---------- legacy API (kept for make_player_form.py) ----------

def canonicalize_player_name(source_key: str) -> str:
    """
    Legacy entrypoint used throughout the repo.
    Accepts 'player_source_name' or 'player_key' like 'Zflowers'/'Jsmith-Njigba'
    and returns the full 'First Last' (with suffix if present).
    """
    k = norm_key(source_key)
    manual = build_manual_map()
    if k in manual:
        return manual[k]
    roles = build_roles_map()
    return roles.get(k, source_key or "")

# Optional: explicit alias used in some places
canonicalize_name = canonicalize_player_name
