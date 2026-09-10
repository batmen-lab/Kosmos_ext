"""
Code execution engine.

Executes generated Python code safely with output capture, error handling, and retry logic.
Supports both direct execution and Docker-based sandboxed execution.
"""

import sys
import os
from kosmos.utils.compat import model_to_dict
import io
import traceback
import signal
import platform
import concurrent.futures
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional sandbox import
try:
    from kosmos.execution.sandbox import DockerSandbox, SandboxExecutionResult
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    logger.warning("Docker sandbox not available. Install docker package for sandboxed execution.")

# Optional R executor import
try:
    from kosmos.execution.r_executor import RExecutor, RExecutionResult, is_r_code
    R_EXECUTOR_AVAILABLE = True
except ImportError:
    R_EXECUTOR_AVAILABLE = False
    logger.debug("R executor not available")

# Execution timeout (seconds) for unsandboxed exec (F-19)
DEFAULT_EXECUTION_TIMEOUT = 300

# Restricted builtins for unsandboxed execution (F-16)
SAFE_BUILTINS = {
    # Core types
    'None': None, 'True': True, 'False': False,
    'int': int, 'float': float, 'str': str, 'bool': bool,
    'bytes': bytes, 'bytearray': bytearray, 'complex': complex,
    # Collections
    'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'frozenset': frozenset,
    # Iteration
    'range': range, 'enumerate': enumerate, 'zip': zip,
    'map': map, 'filter': filter, 'reversed': reversed, 'sorted': sorted,
    'iter': iter, 'next': next,
    # Math
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
    'pow': pow, 'divmod': divmod,
    # String/repr
    'repr': repr, 'format': format, 'chr': chr, 'ord': ord,
    'ascii': ascii, 'bin': bin, 'hex': hex, 'oct': oct, 'hash': hash,
    # Type introspection (safe subset)
    'isinstance': isinstance, 'issubclass': issubclass,
    'type': type, 'id': id, 'callable': callable,
    'hasattr': hasattr, 'len': len,
    # IO
    'print': print,
    # Object creation helpers
    'property': property, 'staticmethod': staticmethod, 'classmethod': classmethod,
    'super': super, 'object': object,
    # Exceptions (needed for try/except)
    'Exception': Exception, 'BaseException': BaseException,
    'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
    'IndexError': IndexError, 'AttributeError': AttributeError,
    'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
    'ZeroDivisionError': ZeroDivisionError, 'NameError': NameError,
    'ImportError': ImportError, 'FileNotFoundError': FileNotFoundError,
    'OSError': OSError, 'IOError': IOError, 'NotImplementedError': NotImplementedError,
    'OverflowError': OverflowError, 'ArithmeticError': ArithmeticError,
    'MemoryError': MemoryError, 'RecursionError': RecursionError,
    'UnicodeError': UnicodeError, 'UnicodeDecodeError': UnicodeDecodeError,
    'UnicodeEncodeError': UnicodeEncodeError,
    # Misc safe builtins
    'any': any, 'all': all, 'slice': slice, 'memoryview': memoryview,
}

# Modules allowed for restricted import (F-16)
_ALLOWED_MODULES = {
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
    'statsmodels', 'math', 'statistics', 'collections', 'itertools',
    'functools', 'operator', 'decimal', 'fractions', 'random',
    'datetime', 'time', 'json', 'csv', 'io', 'pathlib', 'copy',
    're', 'string', 'textwrap', 'hashlib', 'warnings', 'logging',
    'typing', 'dataclasses', 'enum', 'abc',
    'xarray', 'netCDF4', 'h5py', 'zarr', 'dask',
}


def _make_restricted_import(allowed: set = _ALLOWED_MODULES):
    """Create a restricted __import__ that only allows scientific modules."""
    _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split('.')[0]
        if top_level not in allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed in sandboxed execution. "
                f"Allowed top-level modules: {sorted(allowed)}"
            )
        return _real_import(name, globals, locals, fromlist, level)

    return _restricted_import


class ExecutionResult:
    """Result of code execution."""

    def __init__(
        self,
        success: bool,
        return_value: Any = None,
        stdout: str = "",
        stderr: str = "",
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        execution_time: float = 0.0,
        profile_result: Optional[Any] = None,  # ProfileResult from kosmos.core.profiling
        data_source: Optional[str] = None  # 'file' or 'synthetic'
    ):
        self.success = success
        self.return_value = return_value
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.error_type = error_type
        self.execution_time = execution_time
        self.profile_result = profile_result
        self.data_source = data_source

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'success': self.success,
            'return_value': self.return_value,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error': self.error,
            'error_type': self.error_type,
            'execution_time': self.execution_time,
            'data_source': self.data_source,
        }

        # Include profile data if available
        if self.profile_result:
            try:
                result['profile_data'] = model_to_dict(self.profile_result)
            except Exception as e:
                logger.debug("Failed to serialize profile data: %s", e)
                result['profile_data'] = None

        return result


# Static part of the sandbox shim: a universal never-crash stub + a fallback
# finder that resolves any kosmos.* import the container can't satisfy. The
# `kosmos` package is NOT installed in the isolated sandbox, so without this the
# generator's `from kosmos... import ...` lines die with ModuleNotFoundError.
_SANDBOX_SHIM_STATIC = '''# --- injected kosmos sandbox shim ---
import sys as _sys, types as _types
import importlib.util as _ilu
class _Stub:
    # Universal never-crash mock for un-bundled kosmos.* helpers: constructable,
    # callable/chainable, subscriptable, formattable, iterable, arithmetic-safe.
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Stub()
    def __getattr__(self, _n): return _Stub()
    def __getitem__(self, _k): return _Stub()
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __contains__(self, _x): return False
    def __bool__(self): return False
    def __int__(self): return 0
    def __float__(self): return 0.0
    def __str__(self): return "0"
    def __repr__(self): return "stub"
    def __format__(self, _spec): return "0"
    def __add__(self, o): return _Stub()
    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = __add__
def _mkstub(name):
    m = _types.ModuleType(name); m.__path__ = []
    m.__getattr__ = lambda n: _Stub()
    _sys.modules[name] = m
    return m
for _pn in ("kosmos", "kosmos.execution", "kosmos.analysis"):
    if _pn not in _sys.modules: _mkstub(_pn)
class _KosmosStubFinder:
    def find_spec(self, name, path=None, target=None):
        if (name == "kosmos" or name.startswith("kosmos.")) and name not in _sys.modules:
            return _ilu.spec_from_loader(name, self)
        return None
    def create_module(self, spec): return _mkstub(spec.name)
    def exec_module(self, module): pass
_sys.meta_path.insert(0, _KosmosStubFinder())
'''


def _build_sandbox_shim() -> str:
    """Build the shim prepended to sandboxed experiment code.

    Injects the REAL, self-contained kosmos analysis helpers by embedding their
    source (kosmos/execution/data_analysis.py -> DataAnalyzer/DataCleaner,
    kosmos/execution/ml_experiments.py -> MLAnalyzer) so experiments produce
    genuine results. Any other kosmos.* import (e.g. PublicationVisualizer, which
    has kosmos deps we don't bundle) falls back to the never-crash stub. The
    kosmos package isn't installed in the isolated sandbox, hence the injection.
    """
    _dir = os.path.dirname(os.path.abspath(__file__))
    real = {}
    for modname, fname in (
        ("kosmos.execution.data_analysis", "data_analysis.py"),
        ("kosmos.execution.ml_experiments", "ml_experiments.py"),
    ):
        try:
            with open(os.path.join(_dir, fname), "r") as fh:
                real[modname] = fh.read()
        except Exception as e:  # pragma: no cover - best effort
            logger.warning("Could not read sandbox helper %s: %s", fname, e)

    parts = [_SANDBOX_SHIM_STATIC, "_REAL_KOSMOS_SRC = {"]
    for modname, src in real.items():
        parts.append("    %r: %r," % (modname, src))
    parts.append("}")
    parts.append(
        "for _rn, _rsrc in _REAL_KOSMOS_SRC.items():\n"
        "    try:\n"
        "        _rm = _types.ModuleType(_rn); _rm.__name__ = _rn\n"
        "        _sys.modules[_rn] = _rm\n"
        "        exec(compile(_rsrc, _rn.replace('.', '/') + '.py', 'exec'), _rm.__dict__)\n"
        "        _pp, _, _leaf = _rn.rpartition('.')\n"
        "        if _pp in _sys.modules: setattr(_sys.modules[_pp], _leaf, _rm)\n"
        "    except Exception:\n"
        "        _mkstub(_rn)\n"
        "# --- end shim ---"
    )
    return "\n".join(parts)


# Appended to sandboxed experiment code. The generated code computes a `results`/
# `result` dict but only prints human-readable lines; the sandbox's return-value
# extractor looks for a stdout line starting with "RESULT:" + JSON. Without this
# the captured return_value was always empty (-> empty Result rows, 0 "successful"
# experiments). This trailer serializes the result dict to that marker.
# The variables the capture below will look for, in order. Named here so the
# code generator can gate on the same contract instead of guessing at it: a
# script that binds none of these at module scope produces nothing, however
# correct its analysis. `test_the_gate_uses_the_executor's_own_capture_names`
# fails if this drifts from the literal inside the string.
RESULT_CAPTURE_NAMES = ("results", "result", "output", "summary", "analysis")

_SANDBOX_RESULT_CAPTURE = '''
# --- injected result-capture: emit RESULT: <json> for the sandbox extractor ---
try:
    import json as _rc_json
    _rc_payload = None
    _rc_g = globals()
    for _rc_vn in ("results", "result", "output", "summary", "analysis"):
        _rc_cand = _rc_g.get(_rc_vn)
        if isinstance(_rc_cand, dict) and _rc_cand:
            _rc_payload = _rc_cand
            break
    if _rc_payload is not None:
        def _rc_safe(o):
            if o is None or isinstance(o, (str, int, float, bool)):
                return o
            if isinstance(o, dict):
                return {str(k): _rc_safe(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_rc_safe(v) for v in o]
            try:
                import numpy as _rc_np
                if isinstance(o, _rc_np.integer): return int(o)
                if isinstance(o, _rc_np.floating): return float(o)
                if isinstance(o, _rc_np.ndarray): return o.tolist()
            except Exception:
                pass
            return str(o)
        print("RESULT: " + _rc_json.dumps(_rc_safe(_rc_payload)))
except Exception as _rc_e:
    print("RESULT_CAPTURE_ERROR:", _rc_e)
# --- end result-capture ---
'''


class CodeExecutor:
    """
    Executes Python code with safety measures and output capture.

    Provides:
    - Stdout/stderr capture
    - Return value extraction
    - Error handling
    - Execution retry logic
    - Optional Docker sandbox isolation
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        allowed_globals: Optional[Dict[str, Any]] = None,
        use_sandbox: bool = True,
        sandbox_config: Optional[Dict[str, Any]] = None,
        enable_profiling: bool = False,
        profiling_mode: str = "light",
        test_determinism: bool = False,
        execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT
    ):
        """
        Initialize code executor.

        Args:
            max_retries: Maximum number of retry attempts on error
            retry_delay: Delay between retries in seconds
            allowed_globals: Optional dictionary of allowed global variables
            use_sandbox: If True, use Docker sandbox for execution (default: True for F-17)
            sandbox_config: Optional sandbox configuration (cpu_limit, memory_limit, timeout)
            enable_profiling: If True, profile code execution (default: False)
            profiling_mode: Profiling mode: light, standard, full (default: light)
            test_determinism: If True, run determinism check after successful execution (default: False)
            execution_timeout: Timeout in seconds for unsandboxed execution (default: 300, F-19)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.allowed_globals = allowed_globals or {}
        self.use_sandbox = use_sandbox
        self.sandbox_config = sandbox_config or {}
        self.enable_profiling = enable_profiling
        self.profiling_mode = profiling_mode
        self.test_determinism = test_determinism
        self.execution_timeout = execution_timeout

        # Initialize retry strategy for self-correcting execution (Issue #54)
        self.retry_strategy = RetryStrategy(max_retries=max_retries, base_delay=retry_delay)

        # Initialize sandbox if requested (F-17: graceful fallback)
        self.sandbox = None
        if self.use_sandbox:
            if not SANDBOX_AVAILABLE:
                logger.warning(
                    "Docker sandbox requested but not available. "
                    "Falling back to restricted builtins execution."
                )
                self.use_sandbox = False
            else:
                # Filter to what DockerSandbox actually accepts rather than
                # splatting blindly. `sandbox_config` reaches here from callers
                # that build it themselves (parallel.py passes an arbitrary
                # task-supplied dict), so an unexpected key would otherwise
                # raise TypeError during executor construction and take down the
                # whole run -- which is exactly how the memory_limit_mb key name
                # mismatch stayed invisible. Unknown keys are dropped with a
                # warning: losing one setting is recoverable, losing the
                # executor is not.
                import inspect

                accepted = set(
                    inspect.signature(DockerSandbox.__init__).parameters
                ) - {"self"}
                unknown = set(self.sandbox_config) - accepted
                if unknown:
                    logger.warning(
                        "Ignoring sandbox_config key(s) %s that DockerSandbox "
                        "does not accept; valid keys are %s. The corresponding "
                        "limits are NOT being applied.",
                        sorted(unknown),
                        sorted(accepted),
                    )
                safe_config = {
                    k: v for k, v in self.sandbox_config.items() if k in accepted
                }
                self.sandbox = DockerSandbox(**safe_config)
                logger.info(
                    "Docker sandbox initialized for code execution "
                    "(memory_limit=%s, timeout=%s)",
                    self.sandbox.memory_limit,
                    self.sandbox.timeout,
                )

        # Initialize R executor for R language support (Issue #69)
        self.r_executor = None
        if R_EXECUTOR_AVAILABLE:
            r_timeout = self.sandbox_config.get('timeout', 300) if self.sandbox_config else 300
            self.r_executor = RExecutor(
                timeout=r_timeout,
                use_docker=self.use_sandbox,
                docker_image="kosmos-sandbox-r:latest"
            )
            logger.info("R executor initialized for R language support")

    def execute(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None,
        retry_on_error: bool = False,
        llm_client: Optional[Any] = None,
        language: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code and capture results with self-correcting retry (Issue #54).

        Supports both Python and R code. Language is auto-detected if not specified.

        Args:
            code: Code to execute (Python or R)
            local_vars: Optional local variables to make available (Python only)
            retry_on_error: If True, retry on execution errors with code fixes
            llm_client: Optional LLM client for intelligent code repair
            language: Language to use ('python', 'r'). Auto-detected if None.

        Returns:
            ExecutionResult with output and results
        """
        # Auto-detect language if not specified (Issue #69)
        if language is None:
            if R_EXECUTOR_AVAILABLE and self.r_executor:
                detected_lang = self.r_executor.detect_language(code)
                language = detected_lang
            else:
                language = 'python'

        # Route to R executor for R code
        if language == 'r':
            return self._execute_r(code)

        # Python execution continues below
        attempt = 0
        last_error = None
        current_code = code  # Track the current version of code

        while attempt < (self.max_retries if retry_on_error else 1):
            attempt += 1

            try:
                logger.info(f"Executing code (attempt {attempt})")
                result = self._execute_once(current_code, local_vars)

                if result.success:
                    # Track successful repair if code was modified
                    if current_code != code and attempt > 1:
                        self.retry_strategy.record_repair_attempt(
                            result.error_type or "Unknown", True
                        )
                    logger.info(f"Code executed successfully in {result.execution_time:.2f}s")

                    # Optional determinism check
                    if self.test_determinism:
                        try:
                            from kosmos.safety.reproducibility import ReproducibilityManager
                            mgr = ReproducibilityManager()
                            is_deterministic = mgr.test_determinism(
                                experiment_function=lambda: self._execute_once(current_code, local_vars),
                                seed=42, n_runs=2
                            )
                            if not is_deterministic:
                                logger.warning("Non-deterministic results detected")
                                if result.return_value and isinstance(result.return_value, dict):
                                    result.return_value.setdefault('warnings', []).append(
                                        "Non-deterministic results detected"
                                    )
                        except Exception as det_err:
                            logger.debug(f"Determinism check failed (non-fatal): {det_err}")

                    return result
                else:
                    last_error = result.error
                    error_type = result.error_type or "Unknown"

                    # Surface the actual sandbox failure (previously swallowed),
                    # so experiment-execution errors are debuggable from the log.
                    logger.error(
                        "[SANDBOX-EXEC-FAIL] attempt=%s type=%s error=%s\n"
                        "--- stderr (first 2000) ---\n%s\n"
                        "--- code (first 1500) ---\n%s",
                        attempt, error_type, (result.error or "")[:1000],
                        (result.stderr or "")[:2000], (current_code or "")[:1500],
                    )

                    # Terminal: the code refused to invent a missing dataset.
                    # Both fields are checked because the refusal surfaces as
                    # the error on some sandbox backends and only inside the
                    # traceback on others.
                    #
                    # Only this one condition is consulted here, rather than
                    # `should_retry` in full. Wiring the whole predicate in
                    # would also make SyntaxError and FileNotFoundError terminal
                    # in this loop for the first time -- a larger behaviour
                    # change, on paths that today do sometimes repair, and not
                    # the one this change is for. `should_retry` remains correct
                    # for callers that ask it.
                    if is_fabrication_refusal(result.error) or \
                            is_fabrication_refusal(result.stderr):
                        logger.error(
                            "[NO-DATA] experiment refused to fabricate data; not "
                            "retrying. There is no code fix for an absent "
                            "dataset -- give the run a data source "
                            "(--data-path / --evidence-config), or find one "
                            "with `kosmos find-data`."
                        )
                        return result

                    if retry_on_error and attempt < self.max_retries:
                        # Try to fix the code using RetryStrategy (Issue #54)
                        fixed_code = self.retry_strategy.modify_code_for_retry(
                            original_code=current_code,
                            error=result.error or "",
                            error_type=error_type,
                            traceback_str=result.stderr or "",
                            attempt=attempt,
                            llm_client=llm_client
                        )

                        if fixed_code and fixed_code != current_code:
                            logger.info(f"Applying code fix for {error_type}, attempt {attempt + 1}")
                            current_code = fixed_code
                        else:
                            logger.warning(f"No code fix available for {error_type}")

                        # Wait before retry
                        delay = self.retry_strategy.get_delay(attempt)
                        logger.warning(f"Execution failed, retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        # Record failed repair attempt
                        if current_code != code:
                            self.retry_strategy.record_repair_attempt(error_type, False)
                        return result

            except Exception as e:
                logger.error(f"Unexpected error during execution: {e}")
                last_error = str(e)
                error_type = type(e).__name__

                # Same terminal case as above, reached when the refusal
                # propagates as an exception rather than as a failed result.
                if is_fabrication_refusal(str(e)):
                    logger.error(
                        "[NO-DATA] experiment refused to fabricate data; not "
                        "retrying. Give the run a data source, or find one with "
                        "`kosmos find-data`."
                    )
                    return ExecutionResult(
                        success=False, error=str(e), error_type=error_type
                    )

                if retry_on_error and attempt < self.max_retries:
                    # Try to fix the code
                    fixed_code = self.retry_strategy.modify_code_for_retry(
                        original_code=current_code,
                        error=str(e),
                        error_type=error_type,
                        traceback_str=traceback.format_exc(),
                        attempt=attempt,
                        llm_client=llm_client
                    )

                    if fixed_code and fixed_code != current_code:
                        logger.info(f"Applying code fix for {error_type}")
                        current_code = fixed_code

                    delay = self.retry_strategy.get_delay(attempt)
                    time.sleep(delay)
                else:
                    return ExecutionResult(
                        success=False,
                        error=str(e),
                        error_type=error_type
                    )

        # All retries failed
        return ExecutionResult(
            success=False,
            error=f"Failed after {self.max_retries} attempts. Last error: {last_error}",
            error_type="MaxRetriesExceeded"
        )

    def _execute_r(self, code: str) -> ExecutionResult:
        """
        Execute R code using the R executor (Issue #69).

        Args:
            code: R code to execute

        Returns:
            ExecutionResult converted from RExecutionResult
        """
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return ExecutionResult(
                success=False,
                error="R execution not available. Install R and the r_executor module.",
                error_type="RNotAvailable"
            )

        logger.info("Executing R code")
        r_result = self.r_executor.execute(code, capture_results=True)

        # Convert RExecutionResult to ExecutionResult
        return ExecutionResult(
            success=r_result.success,
            return_value=r_result.parsed_results or r_result.return_value,
            stdout=r_result.stdout,
            stderr=r_result.stderr,
            error=r_result.error,
            error_type=r_result.error_type,
            execution_time=r_result.execution_time
        )

    def execute_r(
        self,
        code: str,
        capture_results: bool = True,
        output_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Explicitly execute R code (Issue #69).

        Use this method when you know the code is R and want to skip
        language detection.

        Args:
            code: R code to execute
            capture_results: If True, capture results as JSON
            output_dir: Directory for output files

        Returns:
            ExecutionResult with R execution results
        """
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return ExecutionResult(
                success=False,
                error="R execution not available. Install R and the r_executor module.",
                error_type="RNotAvailable"
            )

        logger.info("Explicitly executing R code")
        r_result = self.r_executor.execute(
            code,
            capture_results=capture_results,
            output_dir=output_dir
        )

        # Convert RExecutionResult to ExecutionResult
        return ExecutionResult(
            success=r_result.success,
            return_value=r_result.parsed_results or r_result.return_value,
            stdout=r_result.stdout,
            stderr=r_result.stderr,
            error=r_result.error,
            error_type=r_result.error_type,
            execution_time=r_result.execution_time
        )

    def is_r_available(self) -> bool:
        """Check if R execution is available."""
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return False
        return self.r_executor.is_r_available()

    def get_r_version(self) -> Optional[str]:
        """Get R version if available."""
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return None
        return self.r_executor.get_r_version()

    def _execute_once(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code once with output capture and optional profiling."""

        # Route to sandbox if enabled
        if self.use_sandbox:
            return self._execute_in_sandbox(code, local_vars)

        # Initialize profiler if enabled
        profiler = None
        profile_result = None
        if self.enable_profiling:
            try:
                from kosmos.core.profiling import ExecutionProfiler, ProfilingMode
                mode = ProfilingMode(self.profiling_mode)
                profiler = ExecutionProfiler(mode=mode)
            except Exception as e:
                logger.warning(f"Failed to initialize profiler: {e}")

        # Otherwise execute directly
        start_time = time.time()

        # Prepare execution environment
        exec_globals = self._prepare_globals()
        exec_locals = local_vars.copy() if local_vars else {}

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Start profiling if enabled
            if profiler:
                profiler._start_profiling()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code with timeout (F-19)
                self._exec_with_timeout(code, exec_globals, exec_locals)

            # Stop profiling if enabled
            if profiler:
                profiler._stop_profiling()
                profile_result = profiler.get_result()

            execution_time = time.time() - start_time

            # Extract return value (look for 'results' variable)
            return_value = exec_locals.get('results', exec_locals.get('result'))
            data_source = exec_locals.get('_data_source')

            return ExecutionResult(
                success=True,
                return_value=return_value,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=execution_time,
                profile_result=profile_result,
                data_source=data_source,
            )

        except Exception as e:
            # Stop profiling even on error
            if profiler:
                try:
                    profiler._stop_profiling()
                    profile_result = profiler.get_result()
                except Exception as e:
                    logger.warning(f"Profiler stop/result retrieval failed: {e}")

            execution_time = time.time() - start_time

            # Capture full traceback
            error_traceback = traceback.format_exc()

            logger.error(f"Code execution failed: {e}\n{error_traceback}")

            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + error_traceback,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time,
                profile_result=profile_result
            )

    def _execute_in_sandbox(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code in Docker sandbox."""
        logger.info("Executing code in Docker sandbox")

        # Prepare data files if data_path provided
        import os

        data_files = {}
        # Every host path this execution may reference, primary first. A
        # multi-dataset run supplies the rest under `__data_files__`; a
        # single-dataset run leaves that absent, so this list holds exactly one
        # entry and the behaviour below is the pre-existing one line for line.
        host_paths: list[str] = []
        if local_vars and local_vars.get('data_path'):
            host_paths.append(local_vars['data_path'])
        for extra in (local_vars or {}).get('__data_files__', {}).values():
            if extra not in host_paths:
                host_paths.append(extra)

        for host_path in host_paths:
            filename = os.path.basename(host_path)
            clash = data_files.get(filename)
            if clash is not None and clash != host_path:
                # Two datasets whose files share a basename would land on one
                # container path and silently overwrite each other (sandbox.py
                # copies into /workspace/data/<basename>). An experiment reading
                # the wrong dataset and reporting a confident result is the worst
                # outcome available here, so refuse rather than pick.
                raise ValueError(
                    f"two datasets share the filename {filename!r} "
                    f"({clash} and {host_path}); they would collide at "
                    f"/workspace/data/{filename}. Stage them under distinct names."
                )
            data_files[filename] = host_path

        if host_paths:
            # The host path is meaningless inside the container — each file is
            # mounted at /workspace/data/<filename>. Rewrite every reference to
            # a host path (the assignments injected by execute_with_data, or a
            # path the LLM hardcoded into the code) to the mounted container
            # path, then guarantee data_path is defined up-front. Rewriting (not
            # just prepending) is essential: a prepended assignment would be
            # shadowed by execute_with_data's later `data_path = '<host>'` line.
            #
            # LONGEST PATH FIRST. With several datasets one host path can be a
            # prefix of another (…/run/a.csv and …/run/a.csv.bak, or a directory
            # and a file beneath it). Replacing the shorter one first corrupts
            # the longer into a half-rewritten hybrid naming no real file.
            for host_path in sorted(host_paths, key=len, reverse=True):
                container_path = f"/workspace/data/{os.path.basename(host_path)}"
                code = code.replace(host_path, container_path)
            primary = f"/workspace/data/{os.path.basename(host_paths[0])}"
            code = f"data_path = {primary!r}\n{code}"

        # Prepend the kosmos-stub shim so generated `from kosmos.* import ...`
        # lines resolve inside the sandbox (kosmos pkg isn't installed there).
        code = _build_sandbox_shim() + "\n" + code

        # Append the result-capture trailer so the code's results dict is emitted
        # as a "RESULT: <json>" stdout line the sandbox extractor can parse.
        code = code + "\n" + _SANDBOX_RESULT_CAPTURE

        # Execute in sandbox
        sandbox_result = self.sandbox.execute(code, data_files=data_files if data_files else None)

        # Convert SandboxExecutionResult to ExecutionResult
        return ExecutionResult(
            success=sandbox_result.success,
            return_value=sandbox_result.return_value,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            error=sandbox_result.error,
            error_type=sandbox_result.error_type,
            execution_time=sandbox_result.execution_time
        )

    def _prepare_globals(self) -> Dict[str, Any]:
        """Prepare global namespace with restricted builtins (F-16)."""
        exec_globals = self.allowed_globals.copy()

        # Use restricted builtins instead of full Python builtins
        restricted = dict(SAFE_BUILTINS)
        restricted['__import__'] = _make_restricted_import()
        exec_globals['__builtins__'] = restricted

        return exec_globals

    def _exec_with_timeout(
        self,
        code: str,
        exec_globals: Dict[str, Any],
        exec_locals: Dict[str, Any]
    ) -> None:
        """Execute code with timeout protection (F-19)."""
        if platform.system() != 'Windows' and hasattr(signal, 'SIGALRM'):
            # Unix: use signal.alarm for same-thread timeout
            old_handler = signal.getsignal(signal.SIGALRM)
            def _timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Code execution timed out after {self.execution_timeout}s"
                )
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.execution_timeout)
            try:
                exec(code, exec_globals, exec_locals)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows/other: use ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(exec, code, exec_globals, exec_locals)
                try:
                    future.result(timeout=self.execution_timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(
                        f"Code execution timed out after {self.execution_timeout}s"
                    )

    def execute_with_data(
        self,
        code: str,
        data_path: str,
        retry_on_error: bool = False,
        *,
        data_files: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute code with data file path(s) provided.

        Args:
            code: Python code to execute (expects `data_path` variable)
            data_path: Path to the PRIMARY data file (made available as variable)
            retry_on_error: If True, retry on errors
            data_files: Optional {dataset_name: host_path} for a multi-dataset
                experiment. Exposed to the code as a `datasets` dict. Omit it and
                nothing about this call changes.

        Returns:
            ExecutionResult

        Note:
            The data_path variable is prepended to code and also provided
            in local_vars for templates that use `pd.read_csv(data_path)`.

            `datasets` is injected as a LITERAL in the code string rather than
            through local_vars, and that is not a style choice. The sandbox path
            rewrites host paths to container paths by string-replacing them in
            the code; a mapping passed through local_vars never passes through
            that rewrite, so inside the container every one of its values would
            still name a host path that does not exist there. The literal is
            rewritten with everything else. It also has to work on the
            non-sandbox path, where the same literal simply holds host paths --
            which is what that path wants.
        """
        # Prepend data_path assignment so templates can use it (Issue #51)
        # This ensures data_path is defined even if templates use it directly
        preamble = f"# Data path injected by executor\ndata_path = {repr(data_path)}\n"
        if data_files:
            preamble += (
                "# Datasets available to this experiment, injected by executor.\n"
                f"datasets = {repr(dict(data_files))}\n"
            )
        augmented_code = f"{preamble}\n{code}"

        # Also inject as local variables for safety. `__data_files__` is read by
        # `_execute_in_sandbox` to decide what to mount; it is dunder-named
        # because it is plumbing for the executor, not a name generated code
        # should ever reference (`datasets`, above, is the public one).
        local_vars: Dict[str, Any] = {'data_path': data_path}
        if data_files:
            local_vars['__data_files__'] = dict(data_files)

        return self.execute(augmented_code, local_vars, retry_on_error)


# Re-export CodeValidator from canonical safety module (F-22: removed duplicate)
from kosmos.safety.code_validator import CodeValidator  # noqa: F811,F401


# The sentences `execution/code_generator.py` writes into every generated
# experiment's data-loading preamble when there is no dataset to load. They are
# a REFUSAL, not a defect: the generated code decided, correctly, that it would
# rather fail than invent numbers.
#
# Matched on the message rather than on the exception type because the type is
# a bare RuntimeError, which covers far too much to make terminal. Kept as a
# substring match on the exact phrases the generator emits, so a generator that
# reworded them would fall back to today's behaviour (retry) rather than to
# something new -- the wrong direction to fail in, but a visible one.
FABRICATION_REFUSAL_MARKERS = (
    "refusing to fabricate synthetic data",
    "Real data only; no synthetic fallback",
)


def is_fabrication_refusal(message: Optional[str]) -> bool:
    """True when a failure is the generated code refusing to invent its data.

    There is no repair for this. The code is not broken; the dataset is absent.
    Retrying hands the same error to `_repair_with_llm`, whose prompt asks for
    working code and says nothing about where data may come from -- so the only
    "fix" available to a model is to write the data itself, three times over, at
    the end of which the experiment reports a result. That is the one failure
    mode this system must never convert into a success.
    """
    if not message:
        return False
    return any(marker in message for marker in FABRICATION_REFUSAL_MARKERS)


class RetryStrategy:
    """
    Strategy for retrying failed code execution (Issue #54).

    Provides different retry approaches:
    - Simple retry (same code)
    - Modified retry (with error feedback) - handles 10+ error types
    - LLM-assisted retry (if LLM available)

    Paper Claim: "If error occurs → reads traceback → fixes code → re-executes"
    """

    # Common missing imports for auto-fix
    COMMON_IMPORTS = {
        'pd': 'import pandas as pd',
        'np': 'import numpy as np',
        'plt': 'import matplotlib.pyplot as plt',
        'sns': 'import seaborn as sns',
        'os': 'import os',
        'sys': 'import sys',
        'json': 'import json',
        're': 'import re',
        'math': 'import math',
        'datetime': 'from datetime import datetime',
        'Path': 'from pathlib import Path',
        'defaultdict': 'from collections import defaultdict',
        'Counter': 'from collections import Counter',
        'scipy': 'import scipy',
        'stats': 'from scipy import stats',
        'sklearn': 'import sklearn',
    }

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries (exponential backoff)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Repair statistics tracking (Issue #54)
        self.repair_stats = {
            "attempted": 0,
            "successful": 0,
            "by_error_type": {}
        }

    def should_retry(
        self, attempt: int, error_type: str, error_message: str = ""
    ) -> bool:
        """Determine if execution should be retried.

        `error_message` is optional and defaults to empty, so every existing
        two-argument call keeps its exact answer. It exists for the one case the
        error TYPE cannot decide: the generated code's refusal to fabricate a
        missing dataset arrives as a bare RuntimeError, and making every
        RuntimeError terminal would stop repairing the large class of errors
        this strategy was written for. See `is_fabrication_refusal`.
        """
        if attempt >= self.max_retries:
            return False

        # No dataset is not a code defect, so there is no code fix. Checked
        # before the type list because the type here is RuntimeError, which is
        # and must remain retryable in general.
        if is_fabrication_refusal(error_message):
            return False

        # Don't retry on certain errors that can't be fixed
        # Issue #51 fix: FileNotFoundError is terminal - use synthetic data instead
        non_retryable_errors = [
            'SyntaxError',           # Requires code rewrite
            'FileNotFoundError',     # Missing data is terminal - use synthetic data
            'DataUnavailableError',  # Custom error for missing data
        ]

        return error_type not in non_retryable_errors

    def get_delay(self, attempt: int) -> float:
        """Get delay for retry attempt (exponential backoff)."""
        return self.base_delay * (2 ** (attempt - 1))

    def record_repair_attempt(self, error_type: str, success: bool):
        """Track repair attempt statistics."""
        self.repair_stats["attempted"] += 1
        if success:
            self.repair_stats["successful"] += 1

        if error_type not in self.repair_stats["by_error_type"]:
            self.repair_stats["by_error_type"][error_type] = {
                "attempted": 0,
                "successful": 0
            }
        self.repair_stats["by_error_type"][error_type]["attempted"] += 1
        if success:
            self.repair_stats["by_error_type"][error_type]["successful"] += 1

    def modify_code_for_retry(
        self,
        original_code: str,
        error: str,
        error_type: str,
        traceback_str: str = "",
        attempt: int = 1,
        llm_client: Optional[Any] = None
    ) -> Optional[str]:
        """
        Modify code based on error for retry (Issue #54 - Enhanced).

        Handles 10+ common error types with intelligent fixes.

        Args:
            original_code: Original code that failed
            error: Error message
            error_type: Type of error (e.g., 'KeyError', 'NameError')
            traceback_str: Full traceback string for context
            attempt: Retry attempt number
            llm_client: Optional LLM client for intelligent repair

        Returns:
            Modified code or None if no modification strategy
        """
        import re as regex_module

        # Try LLM-based repair first if available (only first 2 attempts)
        if llm_client and attempt <= 2:
            try:
                fixed = self._repair_with_llm(
                    original_code, error, traceback_str, llm_client
                )
                if fixed and fixed != original_code:
                    logger.info(f"LLM repair applied for {error_type}")
                    return fixed
            except Exception as e:
                logger.warning(f"LLM repair failed: {e}")

        # Pattern-based fixes for common error types
        if 'KeyError' in error_type:
            return self._fix_key_error(original_code, error)

        elif 'FileNotFoundError' in error_type:
            return self._fix_file_not_found(original_code, error)

        elif 'NameError' in error_type:
            return self._fix_name_error(original_code, error)

        elif 'TypeError' in error_type:
            return self._fix_type_error(original_code, error)

        elif 'IndexError' in error_type:
            return self._fix_index_error(original_code, error)

        elif 'AttributeError' in error_type:
            return self._fix_attribute_error(original_code, error)

        elif 'ValueError' in error_type:
            return self._fix_value_error(original_code, error)

        elif 'ZeroDivisionError' in error_type:
            return self._fix_zero_division(original_code, error)

        elif 'ImportError' in error_type or 'ModuleNotFoundError' in error_type:
            return self._fix_import_error(original_code, error)

        elif 'PermissionError' in error_type:
            return self._fix_permission_error(original_code, error)

        elif 'MemoryError' in error_type:
            return self._fix_memory_error(original_code, error)

        # No specific fix available
        return None

    def _repair_with_llm(
        self,
        code: str,
        error: str,
        traceback_str: str,
        llm_client: Any
    ) -> Optional[str]:
        """Use LLM to analyze error and fix code.

        The CONSTRAINT below is not decoration. This prompt is reached with
        whatever error the sandbox produced, and one of those errors is the
        generated code saying it will not invent a dataset it cannot find. Asked
        only to "fix this code", a model's cheapest fix for a missing file is to
        write the file -- and the experiment then reports a result, from numbers
        nothing measured. `is_fabrication_refusal` keeps that error from
        reaching here at all; this is the second line, for every other phrasing
        of "the data is not there" that the marker list does not catch.
        """
        prompt = f"""Fix the following Python code that produced an error.

ORIGINAL CODE:
```python
{code}
```

ERROR:
{error}

TRACEBACK:
{traceback_str}

CONSTRAINT: do not invent, synthesise, simulate, mock or hard-code data, and do
not replace a missing dataset with a generated one. If the error is that no
dataset is available, or that a required file or column does not exist, there is
no fix -- return the code unchanged.

Return ONLY the fixed Python code, no explanations. Wrap the code in ```python``` markers."""

        try:
            response = llm_client.generate(prompt, max_tokens=2000)

            # Extract code from response
            if "```python" in response:
                start = response.find("```python") + 9
                end = response.find("```", start)
                if end > start:
                    return response[start:end].strip()

            # If no markers, return as-is if it looks like code
            if "import" in response or "def " in response or "=" in response:
                return response.strip()

        except Exception as e:
            logger.warning(f"LLM code repair error: {e}")

        return None

    def _fix_key_error(self, code: str, error: str) -> str:
        """Fix KeyError by adding safe dict access."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except KeyError as e:
    print(f"KeyError: {{e}}. Using default value.")
    results = {{'error': 'KeyError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_file_not_found(self, code: str, error: str) -> Optional[str]:
        """
        Handle FileNotFoundError - return None to indicate terminal failure.

        Issue #51 fix: Instead of wrapping in a try-except that produces a failed
        result (which causes infinite loops), we return None to signal that this
        error cannot be fixed by retry. The caller should use synthetic data or
        fail gracefully.

        Args:
            code: Original code
            error: Error message

        Returns:
            None to indicate no fix possible - caller should handle differently
        """
        import re as regex_module
        # Try to extract the file path from error
        match = regex_module.search(r"'([^']+)'", error)
        file_path = match.group(1) if match else "unknown"

        logger.error(
            f"FileNotFoundError is terminal - data file missing: {file_path}. "
            "Code templates should use synthetic data generation (Issue #51)."
        )

        # Return None to indicate no fix possible - this is a terminal error
        return None

    def _fix_name_error(self, code: str, error: str) -> str:
        """Fix NameError by adding missing imports or definitions."""
        import re as regex_module
        match = regex_module.search(r"name '(\w+)' is not defined", error)
        if match:
            name = match.group(1)
            if name in self.COMMON_IMPORTS:
                return self.COMMON_IMPORTS[name] + '\n' + code

        # Generic fix: wrap in try-except
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except NameError as e:
    print(f"NameError: {{e}}. Variable not defined.")
    results = {{'error': 'NameError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_type_error(self, code: str, error: str) -> str:
        """Fix TypeError by adding type checks."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except TypeError as e:
    print(f"TypeError: {{e}}. Type conversion issue.")
    results = {{'error': 'TypeError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_index_error(self, code: str, error: str) -> str:
        """Fix IndexError by adding bounds checking."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except IndexError as e:
    print(f"IndexError: {{e}}. Index out of bounds.")
    results = {{'error': 'IndexError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_attribute_error(self, code: str, error: str) -> str:
        """Fix AttributeError by adding attribute check."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except AttributeError as e:
    print(f"AttributeError: {{e}}. Attribute not found.")
    results = {{'error': 'AttributeError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_value_error(self, code: str, error: str) -> str:
        """Fix ValueError by adding value validation."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except ValueError as e:
    print(f"ValueError: {{e}}. Invalid value.")
    results = {{'error': 'ValueError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_zero_division(self, code: str, error: str) -> str:
        """Fix ZeroDivisionError by adding zero checks."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {{e}}. Division by zero.")
    results = {{'error': 'ZeroDivisionError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_import_error(self, code: str, error: str) -> str:
        """Fix ImportError by providing fallback."""
        import re as regex_module
        match = regex_module.search(r"No module named '(\w+)'", error)
        module = match.group(1) if match else "unknown"

        indented = self._indent(code, 4)
        return f"""try:
{indented}
except (ImportError, ModuleNotFoundError) as e:
    print(f"ImportError: Module '{module}' not available. {{e}}")
    results = {{'error': 'ImportError', 'module': '{module}', 'status': 'failed'}}
"""

    def _fix_permission_error(self, code: str, error: str) -> str:
        """Fix PermissionError by using alternative path."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except PermissionError as e:
    print(f"PermissionError: {{e}}. Access denied.")
    results = {{'error': 'PermissionError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_memory_error(self, code: str, error: str) -> str:
        """Fix MemoryError by suggesting batch processing."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except MemoryError as e:
    print(f"MemoryError: {{e}}. Consider processing in smaller batches.")
    results = {{'error': 'MemoryError', 'details': 'Out of memory', 'status': 'failed'}}
"""

    def _indent(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)


def execute_protocol_code(
    code: str,
    data_path: Optional[str] = None,
    max_retries: int = 2,
    use_sandbox: bool = True,
    sandbox_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute protocol code with full pipeline.

    Always validates code safety before execution (F-21: removed validate_safety bypass).

    Args:
        code: Generated code to execute
        data_path: Optional path to data file
        max_retries: Maximum retry attempts
        use_sandbox: If True, execute in Docker sandbox (default: True)
        sandbox_config: Optional sandbox configuration

    Returns:
        Dictionary with execution results
    """
    # Check SafetyGuardrails emergency stop before execution
    try:
        from kosmos.safety.guardrails import SafetyGuardrails
        guardrails = SafetyGuardrails()
        guardrails.sync_from_flag_file()
        if guardrails.is_emergency_stop_active():
            return {
                'success': False,
                'error': 'Emergency stop is active',
                'validation_errors': [f"Emergency stop: {guardrails.emergency_stop.reason}"],
                'validation_warnings': []
            }
        # Use guardrails-enforced resource limits for sandbox config.
        #
        # These keys must match DockerSandbox.__init__ EXACTLY, because they are
        # splatted into it (`DockerSandbox(**self.sandbox_config)`). They
        # previously read `memory_limit_mb` / `timeout_seconds`, which that
        # constructor does not accept -- it takes `memory_limit` (a Docker size
        # STRING like "2g", not an int of MB) and `timeout` (seconds). The
        # mismatch raised TypeError on the one path that would have applied the
        # operator's configured limit, so MAX_MEMORY_MB was silently inert and
        # every sandbox ran on the hardcoded 2g default no matter what the
        # config said.
        if use_sandbox and sandbox_config is None:
            enforced_limits = guardrails.enforce_resource_limits()
            sandbox_config = {
                'memory_limit': f"{int(enforced_limits.max_memory_mb)}m",
                'timeout': int(enforced_limits.max_execution_time_seconds),
            }
    except ImportError:
        logger.debug("SafetyGuardrails not available, proceeding without guardrails")

    # Always validate code safety (F-21)
    validator = CodeValidator(allow_file_read=True)
    report = validator.validate(code)
    if not report.passed:
        return {
            'success': False,
            'error': 'Code validation failed',
            'validation_errors': [v.message for v in report.violations],
            'validation_warnings': [str(w) for w in report.warnings]
        }

    # Execute code
    executor = CodeExecutor(
        max_retries=max_retries,
        use_sandbox=use_sandbox,
        sandbox_config=sandbox_config or {}
    )

    if data_path:
        result = executor.execute_with_data(code, data_path, retry_on_error=True)
    else:
        result = executor.execute(code, retry_on_error=True)

    # Convert to dict and add validation info
    result_dict = result.to_dict()
    result_dict['validation_warnings'] = [str(w) for w in report.warnings]

    return result_dict
