"""
Kosmos Execution Module.

Provides sandboxed code execution capabilities for running generated
scientific code safely with resource limits and security isolation.

Usage:
    from kosmos.execution import DockerSandbox, CodeExecutor

    # Direct sandbox usage
    sandbox = DockerSandbox()
    result = sandbox.execute("print('Hello!')")

    # Code executor with optional sandbox
    executor = CodeExecutor(use_sandbox=True)
    result = executor.execute("x = 1 + 1")
"""

from .sandbox import (
    DockerSandbox,
    SandboxExecutionResult,
    execute_in_sandbox,
)

from .executor import (
    CodeExecutor,
    ExecutionResult,
    CodeValidator,
    RetryStrategy,
    execute_protocol_code,
)

__all__ = [
    "DockerSandbox",
    "SandboxExecutionResult",
    "execute_in_sandbox",
    "CodeExecutor",
    "ExecutionResult",
    "CodeValidator",
    "RetryStrategy",
    "execute_protocol_code",
]
