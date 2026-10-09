"""Immutable provenance guard for recovered WR-R15 OOS fold coefficients."""
import csv
import hashlib
from pathlib import Path

ORIGINAL_ARTIFACT_ID = 10061328722
ORIGINAL_ARTIFACT_ZIP_SHA256 = "8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce"
FOLD_COEFFICIENT_SHA256 = "e0c95ed6a302495e4b2b06eecc4d9e2e5a66882d49393d853cfa70b337d4dab6"
ARCHIVED = Path("docs/research/authority_snapshots/wr_r15_fold_coefficients_original_10061328722.csv")


def test_archived_wr_r15_exact_original_bytes_and_folds():
    raw = ARCHIVED.read_bytes()
    assert len(raw) == 3600
    assert hashlib.sha256(raw).hexdigest() == FOLD_COEFFICIENT_SHA256
    lines = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    assert len(lines) == 30
    assert sorted(set((int(x["train_season"]), int(x["test_season"])) for x in lines)) == [
        (2022, 2023),
        (2023, 2024),
    ]
    assert all(len([x for x in lines if int(x["test_season"]) == y]) == 15 for y in (2023, 2024))
