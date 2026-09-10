"""
Integration tests for parallel experiment execution.

Tests ParallelExperimentExecutor and concurrent experiment workflows.

**Rewritten against the module's actual API.** The previous version was written
for a different design and could not pass any of its main cases: it called a
`shutdown()` and read an `executor.executor` on a class that owns no pool between
calls, patched a `_execute_experiment_task` method that does not exist,
constructed `ParallelExecutionResult(protocol_id=...)` when the field is
`experiment_id`, and passed bare id strings to `execute_batch`, which requires
`ExperimentTask` objects and sorts them on `.priority`.

The real shape, which these tests now cover:

  * `ExperimentTask(experiment_id, code, data_path=None, config=None, priority=0)`
  * `ParallelExecutionResult(experiment_id, success, result, execution_time,
    error=None, started_at=None, completed_at=None)`
  * `execute_batch(tasks, use_sandbox=False, timeout_per_task=None)` opens a
    ProcessPoolExecutor inside a `with` block, so the pool is per call and there
    is nothing to shut down -- which is why no `shutdown()` exists.

Tasks run in real worker processes, so their code must be self-contained and
picklable. These tests therefore submit small real snippets rather than mocking
the executor's internals, which also makes them a stronger test of the module.

NOTE, deliberately not papered over: `ResearchDirectorAgent.execute_experiments_batch`
(research_director.py) calls `execute_batch(protocol_ids)` with a list of STRINGS
and then reads each result with `result.get("success")` as though it were a dict.
Both are wrong against this API, so that production path raises on its first task.
It is reachable only when `enable_concurrent` is set, which is off by default.
Fixing it means deciding how a stored protocol becomes an ExperimentTask (it needs
generated code), which is a design change rather than a test repair -- so it is
reported rather than silently rewritten here.
"""

import pytest

from kosmos.execution.parallel import (
    ExperimentTask,
    ParallelExecutionResult,
    ParallelExperimentExecutor,
)


class TestParallelExperimentExecutor:
    """Test ParallelExperimentExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor with 2 workers.

        No teardown: `execute_batch` context-manages its own pool, so the
        executor object holds no resource between calls.
        """
        return ParallelExperimentExecutor(max_workers=2, enable_progress_logging=False)

    def test_initialization(self, executor):
        """Worker counts are configuration; the pool itself is per call."""
        assert executor.max_workers == 2
        assert executor.max_workers_io == 4

    def test_the_pool_is_per_call_not_per_executor(self):
        """The executor holds no pool between calls, so none can leak."""
        executor = ParallelExperimentExecutor(max_workers=2)
        assert not hasattr(executor, "executor")
        assert not hasattr(executor, "shutdown")

    def test_an_empty_batch_short_circuits(self, executor):
        """No tasks means no pool is opened at all."""
        assert executor.execute_batch([]) == []

    @pytest.mark.integration
    def test_execute_single_experiment(self, executor):
        """One task in, one result out, carrying its experiment_id."""
        task = ExperimentTask(experiment_id="exp1", code="result = 6 * 7")
        results = executor.execute_batch([task])

        assert len(results) == 1
        assert isinstance(results[0], ParallelExecutionResult)
        assert results[0].experiment_id == "exp1"

    @pytest.mark.integration
    def test_execute_batch(self, executor):
        """Every submitted task produces exactly one result, addressable by id."""
        tasks = [
            ExperimentTask(experiment_id=f"exp{i}", code=f"result = {i} * 2")
            for i in range(3)
        ]
        results = executor.execute_batch(tasks)

        assert len(results) == 3
        assert {r.experiment_id for r in results} == {"exp0", "exp1", "exp2"}

    @pytest.mark.integration
    def test_a_failing_task_does_not_take_the_batch_down(self, executor):
        """One task's failure is contained: the batch still returns every result.

        This is the property that actually matters for a batch, and it holds.
        Whether the failure is *reported* as a failure is a separate question --
        see the xfail below.
        """
        tasks = [
            ExperimentTask(experiment_id="ok", code="result = 1"),
            ExperimentTask(experiment_id="bad", code="raise ValueError('boom')"),
        ]
        results = {r.experiment_id: r for r in executor.execute_batch(tasks)}

        assert set(results) == {"ok", "bad"}
        assert all(isinstance(r, ParallelExecutionResult) for r in results.values())

    @pytest.mark.integration
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "PRE-EXISTING BUG, not in this module: code that RAISES is reported "
            "success=True with the traceback buried in result['return_value']. "
            "_execute_single_experiment faithfully copies result.get('success') "
            "from execute_protocol_code -> CodeExecutor.execute -> "
            "ExecutionResult.to_dict(), and the miscall is upstream of all three. "
            "Left xfail rather than fixed because changing what `success` means "
            "in CodeExecutor changes the contract every experiment path reads. "
            "An experiment that crashed is currently recorded as a successful "
            "result, which matters for the science, so this is kept visible: if "
            "someone fixes the executor this XPASSes and should be un-xfailed."
        ),
    )
    def test_raising_code_should_be_reported_as_a_failure(self, executor):
        task = ExperimentTask(experiment_id="bad", code="raise ValueError('boom')")
        result = executor.execute_batch([task])[0]

        assert result.success is False
        assert result.error

    @pytest.mark.integration
    def test_every_result_carries_a_timing(self, executor):
        tasks = [
            ExperimentTask(experiment_id="a", code="result = sum(range(10000))"),
            ExperimentTask(experiment_id="b", code="result = 1"),
        ]
        results = executor.execute_batch(tasks)
        assert {r.experiment_id for r in results} == {"a", "b"}
        assert all(r.execution_time >= 0 for r in results)

    @pytest.mark.integration
    def test_a_batch_larger_than_the_worker_pool_still_completes(self, executor):
        """More tasks than workers queue rather than being dropped."""
        tasks = [
            ExperimentTask(experiment_id=f"e{i}", code=f"result = {i}")
            for i in range(6)
        ]
        results = executor.execute_batch(tasks)
        assert {r.experiment_id for r in results} == {f"e{i}" for i in range(6)}

    def test_max_workers_configuration(self):
        """Worker counts are honoured, and I/O workers default to twice them."""
        executor1 = ParallelExperimentExecutor(max_workers=2)
        assert executor1.max_workers == 2
        assert executor1.max_workers_io == 4

        executor2 = ParallelExperimentExecutor(max_workers=8)
        assert executor2.max_workers == 8
        assert executor2.max_workers_io == 16

    def test_explicit_io_workers_override_the_default(self):
        executor = ParallelExperimentExecutor(max_workers=4, max_workers_io=3)
        assert executor.max_workers_io == 3


class TestParallelExecutionResult:
    """The result record's real field names."""

    def test_success_result(self):
        result = ParallelExecutionResult(
            experiment_id="exp1",
            success=True,
            result={"value": 42},
            execution_time=1.5,
        )
        assert result.experiment_id == "exp1"
        assert result.success is True
        assert result.result == {"value": 42}
        assert result.execution_time == 1.5
        assert result.error is None

    def test_failure_result(self):
        result = ParallelExecutionResult(
            experiment_id="exp2",
            success=False,
            result=None,
            execution_time=0.2,
            error="Execution failed",
        )
        assert result.experiment_id == "exp2"
        assert result.success is False
        assert result.error == "Execution failed"


class TestExperimentTask:
    """The task record's defaults, which decide scheduling and data access."""

    def test_defaults(self):
        task = ExperimentTask(experiment_id="e", code="result = 1")
        assert task.data_path is None
        assert task.config is None
        # Priority defaults to 0, so an unprioritised batch keeps a stable order.
        assert task.priority == 0

    def test_a_task_may_name_a_data_file_and_a_priority(self, tmp_path):
        csv = tmp_path / "d.csv"
        csv.write_text("a,b\n1,2\n")
        task = ExperimentTask(
            experiment_id="e", code="result = 1", data_path=str(csv), priority=5
        )
        assert task.data_path == str(csv)
        assert task.priority == 5

    def test_tasks_sort_by_priority_highest_first(self):
        """The ordering `execute_batch` applies before submitting."""
        tasks = [
            ExperimentTask(experiment_id="low", code="", priority=1),
            ExperimentTask(experiment_id="high", code="", priority=10),
            ExperimentTask(experiment_id="mid", code="", priority=5),
        ]
        ordered = sorted(tasks, key=lambda t: t.priority, reverse=True)
        assert [t.experiment_id for t in ordered] == ["high", "mid", "low"]
