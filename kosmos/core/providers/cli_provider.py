"""
CLI-backed LLM providers: Claude Code (`claude -p`) and Codex (`codex exec`).

Both run under the operator's own subscription login rather than an API key.
API keys and base URLs are stripped from the subprocess environment, so a
stray ANTHROPIC_API_KEY / OPENAI_API_KEY can never redirect a call to a
metered API.

Neither CLI exposes temperature or max_tokens flags. Claude's output length is
bounded per call through CLAUDE_CODE_MAX_OUTPUT_TOKENS, derived from the
max_tokens the calling agent requested.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from kosmos.core.providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ProviderAPIError,
    UsageStats,
)
from kosmos.core.utils.json_parser import JSONParseError, parse_json_response

logger = logging.getLogger(__name__)

_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


class _CLIProvider(LLMProvider):
    """Shared plumbing: subprocess call, env scrubbing, usage, run token limit."""

    binary = ""
    provider_label = ""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model") or None
        self.timeout = int(config.get("timeout") or 900)
        self.run_output_limit = int(config.get("run_output_token_limit") or 0)
        self.cli = config.get("binary") or shutil.which(self.binary)
        if not self.cli:
            raise ProviderAPIError(
                self.provider_label,
                f"`{self.binary}` CLI not found on PATH",
                recoverable=False,
            )

    def _env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        if extra:
            env.update(extra)
        return env

    def _check_run_limit(self) -> None:
        if self.run_output_limit and self.total_output_tokens >= self.run_output_limit:
            raise ProviderAPIError(
                self.provider_label,
                f"run output-token limit reached ({self.total_output_tokens} >= "
                f"{self.run_output_limit}); raise or unset the *_RUN_OUTPUT_TOKEN_LIMIT setting",
                recoverable=False,
            )

    def _run(self, prompt: str, system: Optional[str], max_tokens: Optional[int]) -> LLMResponse:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        self._check_run_limit()
        started = time.time()
        try:
            response = self._run(prompt, system, max_tokens)
        except subprocess.TimeoutExpired as e:
            raise ProviderAPIError(
                self.provider_label, f"timed out after {self.timeout}s", raw_error=e
            )
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or e.stdout or str(e)).strip()[:1000]
            raise ProviderAPIError(
                self.provider_label, f"CLI exited {e.returncode}: {detail}", raw_error=e
            )
        self._update_usage_stats(response.usage)
        logger.info(
            "[%s] model=%s in=%d out=%d run_out_total=%d %.1fs",
            self.provider_label,
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            self.total_output_tokens,
            time.time() - started,
        )
        return response

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        return await asyncio.to_thread(self.generate, prompt, system, max_tokens)

    def generate_with_messages(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        convo = "\n\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages if m.role != "system"
        )
        return self.generate(convo, system=system, max_tokens=max_tokens)

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        json_system = (
            (system or "")
            + "\n\nRespond with ONLY valid JSON matching this schema, no prose and no code fences:\n"
            + json.dumps(schema, indent=2)
        )
        text = self.generate(prompt, system=json_system, max_tokens=max_tokens).content
        try:
            return parse_json_response(text, schema=schema)
        except JSONParseError:
            retry = self.generate(
                prompt + "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object.",
                system=json_system,
                max_tokens=max_tokens,
            ).content
            try:
                return parse_json_response(retry, schema=schema)
            except JSONParseError as e:
                raise ProviderAPIError(
                    self.provider_label, f"invalid JSON after retry: {e}", raw_error=e
                )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_label,
            "model": self.model,
            "transport": "cli",
            "binary": self.cli,
        }


class ClaudeCLIProvider(_CLIProvider):
    """Claude via `claude -p` on the user's Claude subscription."""

    binary = "claude"
    provider_label = "claude_cli"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.effort = config.get("effort") or None
        self.max_output_cap = int(config.get("max_output_tokens") or 32000)
        self.token_multiplier = float(config.get("token_multiplier") or 1.25)
        self.min_output_tokens = int(config.get("min_output_tokens") or 1024)
        self.default_max_tokens = int(config.get("default_max_tokens") or 4096)

    def _output_budget(self, requested: Optional[int]) -> int:
        base = int(requested or self.default_max_tokens)
        return max(self.min_output_tokens, min(self.max_output_cap, int(base * self.token_multiplier)))

    def _run(self, prompt: str, system: Optional[str], max_tokens: Optional[int]) -> LLMResponse:
        budget = self._output_budget(max_tokens)
        cmd = [self.cli, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        if system:
            cmd += ["--system-prompt", system]
        out = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            env=self._env({"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(budget)}),
            timeout=self.timeout,
            check=True,
        )
        try:
            data = json.loads(out.stdout)
        except json.JSONDecodeError as e:
            raise ProviderAPIError(
                self.provider_label, f"unparseable CLI output: {out.stdout[:500]}", raw_error=e
            )
        if data.get("is_error"):
            raise ProviderAPIError(self.provider_label, str(data.get("result"))[:1000])
        usage = data.get("usage") or {}
        input_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        )
        output_tokens = int(usage.get("output_tokens") or 0)
        return LLMResponse(
            content=data.get("result") or "",
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=0.0,
                model=self.model,
                provider=self.provider_label,
            ),
            model=self.model or "claude-cli-default",
            finish_reason=data.get("subtype"),
            raw_response=data,
            metadata={
                "output_budget": budget,
                "effort": self.effort,
                "notional_cost_usd": data.get("total_cost_usd"),
            },
        )


class CodexCLIProvider(_CLIProvider):
    """Codex via `codex exec` on the user's ChatGPT subscription.

    Isolated from the operator's personal Codex setup: --ignore-user-config
    skips $CODEX_HOME/config.toml (personal model/effort, memories, plugins,
    MCP servers, notify hook), --ignore-rules skips personal command
    approvals, and --ephemeral keeps these calls out of the session history.
    Auth still comes from CODEX_HOME. Features are disabled with
    `-c features.X=false`, which tolerates names a future codex drops.
    """

    binary = "codex"
    provider_label = "codex_cli"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.effort = config.get("effort") or None

    def _run(self, prompt: str, system: Optional[str], max_tokens: Optional[int]) -> LLMResponse:
        full = f"{system}\n\n{prompt}" if system else prompt
        with tempfile.TemporaryDirectory() as workdir:
            last = os.path.join(workdir, "last_message.txt")
            cmd = [
                self.cli, "exec",
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "-c", "features.memories=false",
                "-c", "features.plugins=false",
                "-c", "features.apps=false",
                "-c", "features.multi_agent=false",
                "--output-last-message", last,
            ]
            if self.model:
                cmd += ["--model", self.model]
            if self.effort:
                cmd += ["-c", f'model_reasoning_effort="{self.effort}"']
            cmd.append("-")
            subprocess.run(
                cmd,
                input=full,
                capture_output=True,
                text=True,
                env=self._env(),
                cwd=workdir,
                timeout=self.timeout,
                check=True,
            )
            with open(last) as fh:
                text = fh.read()
        return LLMResponse(
            content=text,
            usage=UsageStats(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                model=self.model,
                provider=self.provider_label,
            ),
            model=self.model or "codex-cli-default",
            finish_reason="stop",
        )
