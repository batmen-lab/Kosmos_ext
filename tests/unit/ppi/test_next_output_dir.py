"""next_output_dir, and the two entry points that use it."""
from __future__ import annotations

from pathlib import Path

from kosmos.ppi.flow import next_output_dir


def test_a_free_path_is_used_as_is(tmp_path):
    assert next_output_dir(tmp_path / "run") == tmp_path / "run"


def test_an_empty_directory_is_reused(tmp_path):
    (tmp_path / "run").mkdir()
    assert next_output_dir(tmp_path / "run") == tmp_path / "run"


def test_a_finished_run_is_never_written_over(tmp_path):
    """The guard that stops a re-run from overwriting what it meant to compare."""
    first = tmp_path / "run"
    first.mkdir()
    (first / "summary.md").write_text("a previous run")

    assert next_output_dir(first) == tmp_path / "run-2"
    assert (first / "summary.md").read_text() == "a previous run"


def test_the_next_free_suffix_is_taken(tmp_path):
    for name in ("run", "run-2", "run-3"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.md").write_text("x")
    assert next_output_dir(tmp_path / "run") == tmp_path / "run-4"


def test_run_py_points_the_training_at_a_free_directory(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_entry", Path(__file__).resolve().parents[3] / "run.py")
    run_entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_entry)
    args = run_entry.parse_args(["a question", "--out", str(tmp_path)])
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "summary.md").write_text("the first attempt")

    env = run_entry.environment(args, tmp_path)

    assert Path(env["PPI_OUTPUT_DIR"]) == tmp_path / "run-2"
