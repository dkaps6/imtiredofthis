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


def test_obsolete_standalone_model_pipeline_does_not_return():
    tracked = _tracked_paths()
    forbidden = {
        "config.yaml",
        "model/cli.py",
        "model/features/__init__.py",
        "model/features/build.py",
        "model/ingest/__init__.py",
        "model/ingest/loaders.py",
        "model/pricing/__init__.py",
        "model/pricing/price.py",
    }
    offenders = sorted(forbidden & tracked)
    assert not offenders, f"retired standalone model pipeline returned: {offenders}"


def test_retired_engine_and_legacy_provider_paths_do_not_return():
    tracked = _tracked_paths()
    forbidden_prefixes = ("engine/",)
    forbidden_files = {
        "scripts/providers/apisports_pull.py",
        "scripts/providers/gsis_pull.py",
        "scripts/providers/injuries.py",
        "scripts/providers/msf_pull.py",
        ".gitmore",
    }
    offenders = sorted(
        {path for path in tracked if path.startswith(forbidden_prefixes)}
        | (tracked & forbidden_files)
    )
    assert not offenders, f"retired engine/provider paths returned: {offenders}"


def test_historical_schedule_helper_is_research_only():
    tracked = _tracked_paths()
    helper = "scripts/providers/build_schedule.py"
    assert helper in tracked, "historical backtest schedule helper was removed"

    full_slate = PurePosixPath(".github/workflows/full-slate.yml")
    assert str(full_slate) in tracked
    text = open(str(full_slate), encoding="utf-8").read()
    assert helper not in text, (
        "historical providers/build_schedule.py must never become the live Full Slate schedule authority"
    )


def test_frozen_qb_research_is_not_an_active_actions_surface():
    tracked = _tracked_paths()
    qb_workflows = sorted(
        path
        for path in tracked
        if path.startswith(".github/workflows/backtest-qb-")
    )
    assert not qb_workflows, (
        "broad QB research is frozen after M90; historical scripts/docs may remain, "
        f"but active QB workflow YAMLs must not return: {qb_workflows}"
    )
    assert ".github/workflows/freeze-qb-frontier-canonical-v1.yml" not in tracked
    assert ".github/workflows/2026-full-slate-smoke.yml" not in tracked


def test_canonical_and_next_position_workflows_remain_available():
    tracked = _tracked_paths()
    required = {
        ".github/workflows/full-slate.yml",
        ".github/workflows/repo-ci.yml",
        ".github/workflows/audit-only.yml",
        ".github/workflows/backtest-canonical-rushing-trace.yml",
        ".github/workflows/backtest-keyed-rushing-trace.yml",
        ".github/workflows/backtest-keyed-receiving-trace.yml",
    }
    missing = sorted(required - tracked)
    assert not missing, f"required production/RB-WR research workflows missing: {missing}"
