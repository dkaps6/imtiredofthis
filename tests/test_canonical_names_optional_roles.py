import pandas as pd

from scripts.utils import canonical_names as cn


def _reset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cn, "_ROLES_CACHE", None)
    monkeypatch.setattr(cn, "_ROLES_LOOKUP_CACHE", None)
    cn.build_manual_map.cache_clear()


def test_safe_canonicalization_without_roles_returns_normalized_key(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    name, key = cn.canonicalize_player_name_safe("Patrick Mahomes")
    assert name == "Patrick Mahomes"
    assert key == "patrickmahomes"


def test_manual_override_still_works_without_roles(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(
        [{"player_source_name": "Hollywood Brown", "full_name": "Marquise Brown"}]
    ).to_csv(data / "manual_name_overrides.csv", index=False)
    cn.build_manual_map.cache_clear()

    name, key = cn.canonicalize_player_name_safe("Hollywood Brown")
    assert name == "Marquise Brown"
    assert key == "marquisebrown"
