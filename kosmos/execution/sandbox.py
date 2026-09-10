"""
Sandboxed code execution environment.

Provides Docker-based isolated execution with resource limits, security constraints,
and monitoring capabilities for running generated experiment code safely.
"""

import docker
import os
import tempfile
import shutil
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SandboxExecutionResult:
    """Result from sandboxed code execution."""

    def __init__(
        self,
        success: bool,
        return_value: Any = None,
        stdout: str = "",
        stderr: str = "",
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        execution_time: float = 0.0,
        exit_code: Optional[int] = None,
        timeout_occurred: bool = False,
        resource_stats: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.return_value = return_value
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.error_type = error_type
        self.execution_time = execution_time
        self.exit_code = exit_code
        self.timeout_occurred = timeout_occurred
        self.resource_stats = resource_stats or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'return_value': self.return_value,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error': self.error,
            'error_type': self.error_type,
            'execution_time': self.execution_time,
            'exit_code': self.exit_code,
            'timeout_occurred': self.timeout_occurred,
            'resource_stats': self.resource_stats
        }


class DockerSandbox:
    """
    Docker-based sandboxed execution environment.

    Provides:
    - Isolated code execution in Docker containers
    - Resource limits (CPU, memory, timeout)
    - Security constraints (network isolation, read-only filesystem)
    - Execution monitoring (CPU/memory usage tracking)
    - Graceful timeout handling
    """

    # Default Docker image name
    DEFAULT_IMAGE = "kosmos-sandbox:latest"

    # Default resource limits
    DEFAULT_CPU_LIMIT = 2.0  # CPU cores
    # Memory ceiling for the analysis container. `None` means "no explicit cap":
    # the container may use whatever the Docker VM has, which IS the upper bound
    # on this machine -- there is nothing above it to grant.
    #
    # It defaults to None because the old 2g default was wrong for the data this
    # runs on. A 548 MB staged CSV needs roughly 3 GB to read into pandas, so a
    # run over full summary statistics was killed before it computed anything:
    # exit 137, no stderr, nothing to debug from, and a correct analysis lost to
    # a number that had nothing to do with it.
    #
    # What this gives up is real and worth stating: the cap also bounded runaway
    # generated code, and without it a pathological script can pressure the
    # whole Docker VM rather than dying alone. Set KOSMOS_SANDBOX_MEMORY (e.g.
    # "6g") to put a bound back.
    DEFAULT_MEMORY_LIMIT = None  # set below, once the reader is defined
    DEFAULT_TIMEOUT = 300  # seconds (5 minutes)

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        cpu_limit: float = DEFAULT_CPU_LIMIT,
        # Sentinel, not DEFAULT_MEMORY_LIMIT: a default argument binds at
        # `def` time to the class-body value, so reassigning the class attribute
        # afterwards never reached this signature and KOSMOS_SANDBOX_MEMORY was
        # silently ignored -- the setting existed and did nothing.
        memory_limit: Optional[str] = "__default__",
        timeout: int = DEFAULT_TIMEOUT,
        network_disabled: bool = True,
        read_only: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize Docker sandbox.

        Args:
            image: Docker image to use
            cpu_limit: CPU cores limit (e.g., 2.0 = 2 cores)
            memory_limit: Memory limit (e.g., "2g", "512m")
            timeout: Execution timeout in seconds
            network_disabled: If True, disable network access
            read_only: If True, use read-only filesystem
            enable_monitoring: If True, monitor resource usage
        """
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = (
            _default_memory_limit() if memory_limit == "__default__" else memory_limit
        )
        self.timeout = timeout
        self.network_disabled = network_disabled
        self.read_only = read_only
        self.enable_monitoring = enable_monitoring

        # Initialize Docker client
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except docker.errors.DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker not available: {e}")

        # Verify image exists
        self._verify_image()

    def _verify_image(self):
        """Verify Docker image exists, build if needed."""
        try:
            self.client.images.get(self.image)
            logger.info(f"Docker image '{self.image}' found")
        except docker.errors.ImageNotFound:
            logger.warning(f"Docker image '{self.image}' not found, attempting to build...")
            self._build_image()

    def _build_image(self):
        """Build Docker image from Dockerfile."""
        dockerfile_path = Path(__file__).parent.parent.parent / "docker" / "sandbox"

        if not (dockerfile_path / "Dockerfile").exists():
            raise RuntimeError(f"Dockerfile not found at {dockerfile_path}")

        logger.info(f"Building Docker image from {dockerfile_path}...")

        try:
            image, build_logs = self.client.images.build(
                path=str(dockerfile_path),
                tag=self.image,
                rm=True,
                pull=True
            )

            for log in build_logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())

            logger.info(f"Successfully built image '{self.image}'")

        except docker.errors.BuildError as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise RuntimeError(f"Docker build failed: {e}")

    def execute(
        self,
        code: str,
        data_files: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> SandboxExecutionResult:
        """
        Execute code in sandboxed Docker container.

        Args:
            code: Python code to execute
            data_files: Dictionary of filename -> file_path to mount
            environment: Environment variables to set

        Returns:
            SandboxExecutionResult with execution details
        """
        # Create temporary directory for code and data
        temp_dir = tempfile.mkdtemp(prefix="kosmos_sandbox_")

        try:
            # Write code to temporary file
            code_file = Path(temp_dir) / "code" / "experiment.py"
            code_file.parent.mkdir(exist_ok=True)
            code_file.write_text(code)

            # Copy data files if provided
            if data_files:
                data_dir = Path(temp_dir) / "data"
                data_dir.mkdir(exist_ok=True)

                for filename, source_path in data_files.items():
                    dest_path = data_dir / filename
                    shutil.copy2(source_path, dest_path)

            # Create output directory
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir(exist_ok=True)

            # Execute in container
            result = self._run_container(
                temp_dir=temp_dir,
                environment=environment or {}
            )

            return result

        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

    def _run_container(
        self,
        temp_dir: str,
        environment: Dict[str, str]
    ) -> SandboxExecutionResult:
        """Run code in Docker container with resource limits and monitoring."""

        # Prepare volume mounts
        # Convert paths for Windows Docker compatibility
        import platform

        def docker_path(path):
            """Convert path for Docker volume mounting on Windows."""
            path_str = str(path)
            if platform.system() == 'Windows':
                # Convert Windows paths like C:\path to /c/path for Docker
                import re
                path_str = re.sub(r'^([A-Za-z]):', r'/\1', path_str.replace('\\', '/'))
            return path_str

        code_dir = Path(temp_dir) / "code"
        output_dir = Path(temp_dir) / "output"

        volumes = {
            docker_path(code_dir): {'bind': '/workspace/code', 'mode': 'ro'},
            docker_path(output_dir): {'bind': '/workspace/output', 'mode': 'rw'}
        }

        # Add data volume if exists
        data_dir = Path(temp_dir) / "data"
        if data_dir.exists():
            volumes[docker_path(data_dir)] = {'bind': '/workspace/data', 'mode': 'ro'}

        # Prepare environment variables
        env = {
            'PYTHONUNBUFFERED': '1',
            'MPLBACKEND': 'Agg',
            **environment
        }

        # Container configuration
        # Note: 'remove' is not valid for containers.create(), only for containers.run()
        container_config = {
            'image': self.image,
            'command': ['python3', '/workspace/code/experiment.py'],
            'volumes': volumes,
            'environment': env,
            'detach': True,
            # Omitted entirely when unset: passing mem_limit=None is not the
            # same as not passing it, and Docker rejects the former.
            **({'mem_limit': self.memory_limit} if self.memory_limit else {}),
            'nano_cpus': int(self.cpu_limit * 1e9),  # Convert to nano CPUs
            'network_disabled': self.network_disabled,
            'read_only': self.read_only,
            'tmpfs': {
                '/tmp': 'rw,noexec,nosuid,size=100m',
                '/home/sandbox/.local': 'rw,noexec,nosuid,size=50m',
            },
            'security_opt': ['no-new-privileges'],
            'cap_drop': ['ALL'],
            'labels': {'kosmos.sandbox': 'true'},
            'working_dir': '/workspace'
        }

        container = None
        start_time = time.time()
        resource_stats = {}

        try:
            # Create and start container
            container = self.client.containers.create(**container_config)
            logger.info(f"Created container {container.short_id}")

            # Start monitoring thread if enabled
            monitor_thread = None
            if self.enable_monitoring:
                monitor_thread = threading.Thread(
                    target=self._monitor_container,
                    args=(container, resource_stats),
                    daemon=True
                )
                monitor_thread.start()

            # Start container
            container.start()
            logger.info(f"Started container {container.short_id}")

            # Wait for container with timeout
            try:
                exit_status = container.wait(timeout=self.timeout)
                timeout_occurred = False
            except docker.errors.APIError as e:
                # Docker API errors (including timeout)
                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    logger.warning(f"Container timeout after {self.timeout}s: {e}")
                    timeout_occurred = True

                    # Try graceful shutdown
                    try:
                        container.stop(timeout=5)
                    except Exception as e:
                        logger.warning("Graceful container stop failed, forcing kill: %s", e)
                        container.kill()

                    exit_status = {'StatusCode': -1}
                else:
                    logger.error(f"Docker API error: {e}")
                    raise
            except Exception as e:
                # Other unexpected errors
                logger.error(f"Unexpected error waiting for container: {e}")
                raise

            execution_time = time.time() - start_time

            # Get logs
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')

            # Parse exit code
            exit_code = exit_status.get('StatusCode', -1)
            success = (exit_code == 0) and not timeout_occurred

            # Extract return value from output if execution was successful
            return_value = None
            error = None
            error_type = None

            if not success:
                if timeout_occurred:
                    error = f"Execution timeout after {self.timeout} seconds"
                    error_type = "TimeoutError"
                else:
                    # Carry the container's own stderr into the error. Without
                    # it the failure reads "Container exited with code 1" all
                    # the way up into the experiments table and the database --
                    # which says only that something went wrong, while the
                    # traceback saying WHAT is sitting right here in `stderr`
                    # and was being discarded. Tail, not head: the exception
                    # line is at the end of a traceback.
                    detail = (stderr or "").strip()
                    # 137 is SIGKILL, and in a container it is the memory cap
                    # essentially every time. The diagnosis used to sit only in
                    # the no-stderr branch, on the assumption that a killed
                    # process writes nothing -- but a pandas DtypeWarning on a
                    # 548 MB CSV lands on stderr before the kill, so the branch
                    # that explains the failure was skipped in exactly the case
                    # that produces it.
                    oom = (
                        (
                            f" -- exit 137 is the kernel killing the container, "
                            f"almost always the memory limit of "
                            f"{self.memory_limit}, set via KOSMOS_SANDBOX_MEMORY; "
                            f"raise or unset it, or read fewer columns/rows."
                            if self.memory_limit else
                            " -- exit 137 is the kernel killing the container, "
                            "almost always memory. No container limit is set, so "
                            "this is the Docker VM's own allocation: raise it in "
                            "Docker Desktop (Settings > Resources), or read fewer "
                            "columns/rows."
                        )
                        if exit_code == 137 else ""
                    )
                    if detail:
                        tail = detail[-1500:]
                        error = (
                            f"Container exited with code {exit_code}: "
                            f"{tail.splitlines()[-1]}{oom}\n"
                            f"--- container stderr ---\n{tail}"
                        )
                    else:
                        error = (
                            f"Container exited with code {exit_code} "
                            f"(no stderr produced)"
                            + (
                                # 137 is SIGKILL, and in a memory-capped
                                # container it is the cap essentially every
                                # time. The process is killed outright, so it
                                # never gets to write a traceback -- which is
                                # why this message has to carry the diagnosis
                                # the missing stderr would have.
                                f": killed by the kernel, almost always "
                                + (
                                    f"the container memory limit of "
                                    f"{self.memory_limit}, set via "
                                    f"KOSMOS_SANDBOX_MEMORY. Raise or unset it, "
                                    f"or read fewer columns/rows."
                                    if self.memory_limit else
                                    "memory. No container limit is set, so this "
                                    "is the Docker VM's own allocation -- raise "
                                    "it in Docker Desktop (Settings > "
                                    "Resources), or read fewer columns/rows."
                                )
                                if exit_code == 137 else ""
                            )
                        )
                    error_type = "ExecutionError"
            else:
                # Try to parse return value from stdout
                return_value = self._extract_return_value(stdout)

            return SandboxExecutionResult(
                success=success,
                return_value=return_value,
                stdout=stdout,
                stderr=stderr,
                error=error,
                error_type=error_type,
                execution_time=execution_time,
                exit_code=exit_code,
                timeout_occurred=timeout_occurred,
                resource_stats=resource_stats
            )

        except docker.errors.ContainerError as e:
            execution_time = time.time() - start_time
            logger.error(f"Container error: {e}")

            return SandboxExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                error=str(e),
                error_type="ContainerError",
                execution_time=execution_time,
                resource_stats=resource_stats
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error during container execution: {e}")

            return SandboxExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time,
                resource_stats=resource_stats
            )

        finally:
            # Cleanup container
            if container:
                try:
                    container.remove(force=True)
                    logger.info(f"Removed container {container.short_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

    def _monitor_container(self, container, stats_dict: Dict[str, Any]):
        """Monitor container resource usage."""
        try:
            for stats in container.stats(stream=True, decode=True):
                # Extract CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']

                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                else:
                    cpu_percent = 0.0

                # Extract memory usage
                memory_usage = stats['memory_stats'].get('usage', 0)
                memory_limit = stats['memory_stats'].get('limit', 1)
                memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0

                # Update stats (keep max values)
                if 'cpu_percent_max' not in stats_dict or cpu_percent > stats_dict['cpu_percent_max']:
                    stats_dict['cpu_percent_max'] = cpu_percent

                if 'memory_mb_max' not in stats_dict or memory_usage > stats_dict.get('memory_bytes_max', 0):
                    stats_dict['memory_bytes_max'] = memory_usage
                    stats_dict['memory_mb_max'] = memory_usage / (1024 * 1024)
                    stats_dict['memory_percent_max'] = memory_percent

                # Check if container stopped
                if container.status != 'running':
                    break

        except Exception as e:
            logger.warning(f"Monitoring error: {e}")

    @staticmethod
    def _extract_return_value(stdout: str) -> Any:
        """Extract return value from stdout (look for JSON result marker)."""
        # Look for lines containing "RESULT:" marker
        for line in stdout.split('\n'):
            if line.startswith('RESULT:'):
                try:
                    result_str = line[7:].strip()  # Remove "RESULT:" prefix
                    return json.loads(result_str)
                except json.JSONDecodeError:
                    return result_str

        return None

    def cleanup(self):
        """Cleanup Docker resources."""
        try:
            self.client.close()
            logger.info("Docker client closed")
        except Exception as e:
            logger.warning(f"Error closing Docker client: {e}")


def _default_memory_limit() -> Optional[str]:
    """The configured ceiling, read at call time.

    A function rather than an inline `os.environ.get` so a test can exercise it
    by setting the variable, instead of reloading this module -- a reload
    replaces `DockerSandbox` itself and quietly breaks any test already holding
    a reference to the old class. Observed exactly that: a reload-based test
    passed alone and broke `test_execute_simple_code` when run beside it.
    """
    return os.environ.get("KOSMOS_SANDBOX_MEMORY") or None


DockerSandbox.DEFAULT_MEMORY_LIMIT = _default_memory_limit()


def execute_in_sandbox(
    code: str,
    data_files: Optional[Dict[str, str]] = None,
    cpu_limit: float = DockerSandbox.DEFAULT_CPU_LIMIT,
    memory_limit: Optional[str] = DockerSandbox.DEFAULT_MEMORY_LIMIT,
    timeout: int = DockerSandbox.DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """
    Convenience function to execute code in sandbox.

    Args:
        code: Python code to execute
        data_files: Dictionary of filename -> file_path
        cpu_limit: CPU cores limit
        memory_limit: Memory limit string
        timeout: Timeout in seconds

    Returns:
        Dictionary with execution results
    """
    sandbox = DockerSandbox(
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        timeout=timeout
    )

    try:
        result = sandbox.execute(code, data_files=data_files)
        return result.to_dict()
    finally:
        sandbox.cleanup()
