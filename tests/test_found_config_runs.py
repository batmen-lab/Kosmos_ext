"""A config that cannot run is not a discovery.

`kosmos find-data --emit` wrote a config whose first `kosmos run` refused:

    found_cis_pqtl would occupy about 1.7 GiB in memory (3,791,847 rows x 14
    columns), over the budget of 807.9 MiB. This is a capacity limit, not a
    disclosure one: the data is public.

The gateway defaults to a quarter of available memory, which is the right
posture for a server with many callers and the wrong one for a single run told
to fetch one dataset. The emitted command now carries a ceiling this machine
can honour.
"""

from __future__ import annotations

from unittest.mock import patch

from kosmos.datasearch.emit import (
    _FALLBACK_BUDGET_BYTES,
    _open_data_budget,
    _serve_command,
)


def _cmd(**kw):
    defaults = dict(
        serve_cmd="autoevidence-serve",
        reference="hf://owner/name#table.parquet",
        dataset_id="found_table",
        signing_key=None,
        staging_dir="/tmp/staging",
    )
    defaults.update(kw)
    return _serve_command(**defaults)


def test_the_emitted_command_carries_a_budget():
    assert "--open-data-budget-bytes" in _cmd()


def test_the_budget_is_large_enough_for_the_table_that_failed():
    """3,791,847 rows x 14 columns needed 1.7 GiB and got 807.9 MiB."""
    needed = int(1.7 * 1024**3)
    assert _open_data_budget() > needed


def test_the_budget_falls_back_when_memory_cannot_be_read():
    with patch("psutil.virtual_memory", side_effect=RuntimeError("no /proc")):
        assert _open_data_budget() == _FALLBACK_BUDGET_BYTES


def test_a_tiny_machine_still_gets_a_usable_floor():
    """A run on a loaded machine must not emit a budget smaller than the data."""

    class _VM:
        available = 256 * 1024**2  # 256 MiB free

    with patch("psutil.virtual_memory", return_value=_VM()):
        assert _open_data_budget() == _FALLBACK_BUDGET_BYTES


def test_a_large_machine_gets_a_proportional_budget():
    class _VM:
        available = 64 * 1024**3

    with patch("psutil.virtual_memory", return_value=_VM()):
        budget = _open_data_budget()
    assert budget > _FALLBACK_BUDGET_BYTES
    assert budget < 64 * 1024**3, "must not claim all of memory"


def test_the_rest_of_the_command_is_unchanged():
    """Checked after splitting: the arguments are now quoted, so a substring
    match would be testing the quoting style rather than the arguments."""
    import shlex

    parts = shlex.split(_cmd())

    assert parts[parts.index("--source") + 1] == "hf://owner/name#table.parquet"
    assert parts[parts.index("--dataset") + 1] == "found_table"
    assert parts[parts.index("--staging-dir") + 1] == "/tmp/staging"
    assert "--key" not in parts, "a key is emitted only when one is known"


def test_a_known_key_is_still_emitted():
    assert "--key /keys/k.key" in _cmd(signing_key="/keys/k.key")


# --- the emitted line has to survive shlex.split ----------------------------

def test_a_staging_path_with_spaces_survives_splitting():
    """`data/myocardial fibrosis exp` is a real path in this project.

    Unquoted it split into `--staging-dir .../data/myocardial` plus a stray
    `fibrosis`, so the gateway staged somewhere else or refused outright.
    """
    import shlex

    staging = "/Users/x/data/myocardial fibrosis exp/found/staging"
    parts = shlex.split(_cmd(staging_dir=staging))

    assert parts[parts.index("--staging-dir") + 1] == staging


def test_a_reference_with_a_file_selector_survives_splitting():
    """`#cis_pqtl.parquet` is a comment character to a shell."""
    import shlex

    ref = "hf://MarkA040999/KosmosMyocardialFibrosisTest#cis_pqtl.parquet"
    parts = shlex.split(_cmd(reference=ref))

    assert parts[parts.index("--source") + 1] == ref


def test_a_dataset_id_with_a_space_survives_splitting():
    import shlex

    parts = shlex.split(_cmd(dataset_id="found table"))
    assert parts[parts.index("--dataset") + 1] == "found table"


def test_a_key_path_with_spaces_survives_splitting():
    import shlex

    key = "/Users/x/my keys/capsule.key"
    parts = shlex.split(_cmd(signing_key=key))
    assert parts[parts.index("--key") + 1] == key


# --- the emitted command must be executable ---------------------------------

class TestServeCommandResolves:
    """`[Errno 2] No such file or directory: 'autoevidence-serve'`.

    The emitted config named the console script by bare name. That only works
    when the venv holding it is on the spawned process's PATH -- and it usually
    is not, because Kosmos is invoked by its absolute venv path, so the child
    inherits a PATH with no venv in it. Search worked, emit worked, and the
    first run died on a missing binary.
    """

    def test_the_default_is_an_existing_executable(self):
        import os

        from kosmos.cli.commands.find_data import DEFAULT_SERVE_CMD

        assert os.path.exists(DEFAULT_SERVE_CMD), (
            f"emitted serve command {DEFAULT_SERVE_CMD!r} does not exist; a "
            f"config naming it cannot run"
        )

    def test_path_wins_over_the_interpreter_sibling(self):
        """An operator's own install should not be overridden by ours."""
        from unittest.mock import patch

        from kosmos.cli.commands.find_data import _default_serve_cmd

        with patch("shutil.which", return_value="/usr/local/bin/autoevidence-serve"):
            assert _default_serve_cmd() == "/usr/local/bin/autoevidence-serve"

    def test_the_interpreter_sibling_is_used_when_path_has_none(self, tmp_path):
        from unittest.mock import patch

        from kosmos.cli.commands.find_data import _default_serve_cmd

        binary = tmp_path / "autoevidence-serve"
        binary.write_text("#!/bin/sh\n")
        fake_python = tmp_path / "python"
        fake_python.write_text("")

        with patch("shutil.which", return_value=None), \
             patch("sys.executable", str(fake_python)):
            assert _default_serve_cmd() == str(binary)

    def test_a_missing_binary_falls_back_to_the_bare_name(self, tmp_path):
        """Better a failure naming the binary than a path this machine never had."""
        from unittest.mock import patch

        from kosmos.cli.commands.find_data import _default_serve_cmd

        fake_python = tmp_path / "python"
        fake_python.write_text("")

        with patch("shutil.which", return_value=None), \
             patch("sys.executable", str(fake_python)):
            assert _default_serve_cmd() == "autoevidence-serve"
