from pathlib import Path

from scripts.data_frontier.bdb_2024_corpus_preflight import inspect_corpus


def _touch(path: Path, text: str = "x\n") -> None:
    path.write_text(text, encoding="utf-8")


def _complete(tmp_path: Path) -> None:
    _touch(tmp_path / "plays.csv")
    _touch(tmp_path / "tackles.csv")
    for week in range(1, 10):
        _touch(tmp_path / f"tracking_week_{week}.csv")


def test_complete_official_fileset_passes(tmp_path):
    _complete(tmp_path)
    report = inspect_corpus(tmp_path)
    assert report["passed"] is True
    assert report["observed_tracking_weeks"] == list(range(1, 10))
    assert report["contact_detector_changed"] is False


def test_partial_tracking_corpus_fails_closed(tmp_path):
    _complete(tmp_path)
    (tmp_path / "tracking_week_9.csv").unlink()
    report = inspect_corpus(tmp_path)
    assert report["passed"] is False
    assert report["missing_tracking_weeks"] == [9]


def test_missing_static_file_fails_closed(tmp_path):
    _complete(tmp_path)
    (tmp_path / "tackles.csv").unlink()
    report = inspect_corpus(tmp_path)
    assert report["passed"] is False
    assert report["missing_static_files"] == ["tackles.csv"]


def test_zero_byte_required_file_fails_closed(tmp_path):
    _complete(tmp_path)
    (tmp_path / "plays.csv").write_bytes(b"")
    report = inspect_corpus(tmp_path)
    assert report["passed"] is False
    assert report["zero_byte_files"] == ["plays.csv"]
