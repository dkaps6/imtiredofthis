"""Season-aware nflverse play-by-play loader with maintained-library fallbacks."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _load_with_nflreadpy(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nflv
    return _to_pandas(nflv.load_pbp(seasons=seasons))


def _load_with_nfl_data_py(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    if hasattr(nfl, "import_pbp_data"):
        return _to_pandas(nfl.import_pbp_data(seasons, downcast=True))
    if hasattr(nfl, "import_pbp"):
        return _to_pandas(nfl.import_pbp(seasons))
    raise RuntimeError("nfl_data_py exposes neither import_pbp_data nor import_pbp")


def get_pbp(season: int | Iterable[int], *, min_rows: int = 0) -> pd.DataFrame:
    seasons = [int(season)] if isinstance(season, (int, str)) else [int(v) for v in season]
    if not seasons:
        raise RuntimeError("At least one PBP season is required")

    errors: list[str] = []
    for label, loader in (("nflreadpy", _load_with_nflreadpy), ("nfl_data_py", _load_with_nfl_data_py)):
        try:
            df = loader(seasons)
            if df is None or df.empty:
                raise RuntimeError("returned 0 rows")
            df.columns = [str(c).lower() for c in df.columns]
            if "season" in df.columns:
                season_num = pd.to_numeric(df["season"], errors="coerce")
                df = df.loc[season_num.isin(seasons)].copy()
            if df.empty:
                raise RuntimeError(f"no rows after season filter {seasons}")
            if min_rows and len(df) < int(min_rows):
                raise RuntimeError(f"rows too small: {len(df)} < {min_rows}")
            return df.reset_index(drop=True)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise RuntimeError(
        "Unable to load nflverse PBP for season(s) "
        f"{seasons}. Errors: {' | '.join(errors)}"
    )
