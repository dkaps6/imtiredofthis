from pathlib import PurePosixPath
import subprocess


def _tracked_paths() -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def test_runtime_outputs_and_retired_external_paths_are_not_tracked():
    tracked = _tracked_paths()

    forbidden_prefixes = (
        "outputs/",
        "logs/",
        "external/",
    )
    offenders = sorted(
        path for path in tracked if path.startswith(forbidden_prefixes)
    )
    assert not offenders, f"runtime/retired paths must not be tracked: {offenders}"


def test_root_data_csvs_are_runtime_except_explicit_static_artifacts():
    tracked = _tracked_paths()
    allowed = {
        "data/model_ensemble_weights.csv",
        "data/stadiums.csv",
    }

    root_data_csvs = {
        path
        for path in tracked
        if PurePosixPath(path).parent == PurePosixPath("data")
        and path.lower().endswith(".csv")
    }
    offenders = sorted(root_data_csvs - allowed)
    assert not offenders, (
        "root data/*.csv files are runtime artifacts unless explicitly allowlisted; "
        f"tracked offenders={offenders}"
    )


def test_promoted_static_production_artifacts_remain_tracked():
    tracked = _tracked_paths()
    required = {
        "data/model_ensemble_weights.csv",
        "data/stadiums.csv",
        "model/qb_pass_synthesis_v1.json",
    }
    missing = sorted(required - tracked)
    assert not missing, f"required static production artifacts were untracked: {missing}"
