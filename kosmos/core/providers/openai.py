"""
OpenAI provider implementation.

Supports OpenAI API and OpenAI-compatible endpoints (Ollama, OpenRouter, etc.).
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = None

try:
    import httpx  # a hard dependency of the openai SDK, so present whenever it is
except ImportError:  # pragma: no cover
    httpx = None

from kosmos.core.providers.base import (
    LLMProvider,
    Message,
    UsageStats,
    LLMResponse,
    ProviderAPIError
)
from kosmos.core.utils.json_parser import parse_json_response, JSONParseError

logger = logging.getLogger(__name__)

# Distinguishes "caller passed nothing" from "caller explicitly passed None to
# mean no reasoning". A plain None default could not tell those apart.
_UNSET = object()

# Ceiling for the "retry with a bigger budget" self-heal below. When reasoning is
# already OFF, the reasoning-off retries cannot help an empty/truncated/unparseable
# structured response -- the only lever left is a larger output budget. max_tokens
# is a CEILING (a healthy call stops at `stop` well under it), so raising it on a
# failed structured call costs nothing on the calls that were already fine. Capped
# so a pathological prompt cannot request an unbounded generation.
_STRUCT_MAX_TOKENS_CAP = int(os.environ.get("OPENAI_STRUCT_MAX_TOKENS_CAP", "49152"))


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.

    Supports:
    - OpenAI official API (GPT-4, GPT-3.5, etc.)
    - OpenAI-compatible APIs (OpenRouter, Together AI, etc.)
    - Local models (Ollama, LM Studio, LocalAI, etc.)

    Features:
    - Unified interface matching Anthropic provider
    - Response caching (via generic LLM cache)
    - Usage tracking and cost estimation
    - Custom base URLs for compatible providers

    Example:
        ```python
        # OpenAI official
        config = {
            'api_key': 'sk-...',
            'model': 'gpt-4-turbo',
            'max_tokens': 4096,
            'temperature': 0.7,
        }

        # Ollama local
        config = {
            'api_key': 'ollama',  # Dummy key
            'base_url': 'http://localhost:11434/v1',
            'model': 'llama3.1:70b',
        }

        # OpenRouter
        config = {
            'api_key': 'sk-or-...',
            'base_url': 'https://openrouter.ai/api/v1',
            'model': 'anthropic/claude-3.5-sonnet',
        }

        provider = OpenAIProvider(config)
        response = provider.generate("Explain quantum computing")
        print(response.content)
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.

        Args:
            config: Configuration dict with keys:
                - api_key: OpenAI API key (or dummy for local)
                - model: Model name (e.g., gpt-4-turbo, llama3.1:70b)
                - max_tokens: Max tokens (default: 4096)
                - temperature: Sampling temperature (default: 0.7)
                - base_url: Custom base URL for OpenAI-compatible APIs
                - organization: OpenAI organization ID (optional)
                - timeout: Request timeout in seconds (default: 120)
        """
        super().__init__(config)

        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        # Extract configuration (handle both dict and Pydantic model)
        def get_config_value(key, default=None):
            """Get value from config (dict or Pydantic model)."""
            if isinstance(config, dict):
                return config.get(key, default)
            else:
                return getattr(config, key, default)

        self.api_key = get_config_value('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not provided in config or environment."
            )

        self.model = get_config_value('model') or 'gpt-4-turbo'
        self.max_tokens = get_config_value('max_tokens') or 4096
        temperature = get_config_value('temperature')
        self.temperature = temperature if temperature is not None else 0.7
        self.base_url = get_config_value('base_url') or os.environ.get('OPENAI_BASE_URL')
        self.organization = get_config_value('organization') or os.environ.get('OPENAI_ORGANIZATION')
        def _env_float(name):
            raw = os.environ.get(name)
            try:
                return float(raw) if raw not in (None, "") else None
            except ValueError:
                return None

        self.timeout = get_config_value('timeout') or _env_float('OPENAI_TIMEOUT') or 120
        self.reasoning_effort = get_config_value('reasoning_effort') or os.environ.get('OPENAI_REASONING_EFFORT')

        # Hard per-request budget so a stalled OpenRouter socket cannot wedge the
        # run. The bare float timeout the SDK took before set connect == read ==
        # self.timeout and the client kept the SDK default of ~2 retries, so a
        # half-open socket (connection ESTABLISHED, no bytes) burned the full
        # read window on EACH of connect, read, and every retry -- observed as
        # 16-minute silent hangs on a dead connection the 120s "timeout" never
        # broke. An explicit httpx.Timeout gives connect a SHORT fuse (a dead
        # peer fails in seconds, not minutes) while leaving read long enough for
        # a genuine generation, and OPENAI_MAX_RETRIES caps the multiplier.
        # Together the worst case is bounded to ~ (retries+1) * read, not open-
        # ended. Tunable: OPENAI_TIMEOUT (read seconds), OPENAI_CONNECT_TIMEOUT,
        # OPENAI_MAX_RETRIES.
        try:
            self.max_retries = int(os.environ.get('OPENAI_MAX_RETRIES', '1'))
        except ValueError:
            self.max_retries = 1
        _connect = _env_float('OPENAI_CONNECT_TIMEOUT') or 10.0
        self._http_timeout = (
            httpx.Timeout(float(self.timeout), connect=_connect, write=30.0, pool=_connect)
            if httpx is not None else self.timeout
        )

        # Reasoning tokens are drawn from the SAME max_tokens budget as the
        # final answer on OpenRouter's unified reasoning API (this is exactly
        # what produced the pre-existing "response likely truncated mid-
        # reasoning" warning below, empirically, before this field existed). A
        # low max_tokens with reasoning enabled can consume the whole budget on
        # the reasoning trace and leave nothing for content. Warned, not
        # silently raised -- an operator's explicit OPENAI_MAX_TOKENS is theirs
        # to keep, but they should know why output might come back empty.
        if self._reasoning_enabled() and self.max_tokens < 8192:
            logger.warning(
                f"reasoning_effort={self.reasoning_effort!r} is set with "
                f"max_tokens={self.max_tokens}. Reasoning tokens count against "
                f"the same budget as the final answer, so a low max_tokens "
                f"risks a reasoning-only, content-empty response. Consider "
                f"OPENAI_MAX_TOKENS >= 8192 for effort='high'/'xhigh'."
            )

        # Detect provider type from base_url
        if self.base_url:
            if 'ollama' in self.base_url or 'localhost' in self.base_url or '127.0.0.1' in self.base_url:
                self.provider_type = 'local'
            elif 'openrouter' in self.base_url:
                self.provider_type = 'openrouter'
            elif 'together' in self.base_url:
                self.provider_type = 'together'
            else:
                self.provider_type = 'compatible'
        else:
            self.provider_type = 'openai'

        # Initialize OpenAI client
        try:
            client_args = {
                'api_key': self.api_key,
                # Bound every request (and the SDK's own retries) at the client
                # level, so a path that forgets the per-call timeout is still
                # covered. See the self._http_timeout note above.
                'timeout': self._http_timeout,
                'max_retries': self.max_retries,
            }
            if self.base_url:
                client_args['base_url'] = self.base_url
            if self.organization:
                client_args['organization'] = self.organization

            self.client = OpenAI(**client_args)

            # Lazy-initialized async client (same config as sync client)
            self._async_client: Optional[AsyncOpenAI] = None
            self._async_client_args = client_args.copy()

            logger.info(f"OpenAI provider initialized (type: {self.provider_type}, model: {self.model})")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ProviderAPIError("openai", f"Failed to initialize: {e}", raw_error=e)

    # Values (case-insensitive) that mean "reasoning OFF" when they appear as the
    # reasoning_effort config/override. Anything else truthy is treated as an
    # effort level passed straight through to the provider.
    _REASONING_OFF_TOKENS = frozenset({"", "off", "none", "false", "no", "disabled", "0"})

    def _is_openrouter(self) -> bool:
        """Whether this provider is talking to OpenRouter (vs plain OpenAI / local).

        The `reasoning` request field is an OpenRouter convention. The disable
        form in particular (`{"enabled": False}`) must be sent ONLY to OpenRouter:
        a plain OpenAI endpoint rejects unknown top-level fields.
        """
        return bool(self.base_url and "openrouter" in self.base_url.lower())

    def _reasoning_is_off(self, override: Any = _UNSET) -> bool:
        """Is reasoning meant to be OFF for this call?

        True when the effort is unset or one of the explicit "off" tokens. This is
        the single source of truth the self-heal paths use, so an absent effort is
        never mistaken for "reasoning is already handled" when, on OpenRouter, the
        model would reason by default.
        """
        effort = self.reasoning_effort if override is _UNSET else override
        if effort is None:
            return True
        return str(effort).strip().lower() in self._REASONING_OFF_TOKENS

    def _reasoning_enabled(self, override: Any = _UNSET) -> bool:
        """Inverse of :meth:`_reasoning_is_off` -- whether reasoning is ON."""
        return not self._reasoning_is_off(override)

    def _reasoning_extra_body(self, override: Any = _UNSET) -> Dict[str, Any]:
        """OpenRouter's unified `reasoning` request field for this call.

        `override` lets one call change reasoning without touching the provider's
        configuration -- pass None (or an "off" token) to force reasoning off for
        a retry. `generate_structured` uses this when a call came back empty or
        unparseable, because reasoning and a long JSON schema compete for the same
        token budget.

        CRUCIAL on OpenRouter: an ABSENT `reasoning` field is NOT "reasoning off" --
        it is "use the model's default", and a reasoning model (e.g.
        deepseek-v4.1) then reasons anyway, draws the reasoning trace from the SAME
        max_tokens as the answer, and routinely returns content-empty responses at
        finish_reason=length. So when reasoning is meant to be off we send an
        EXPLICIT `{"reasoning": {"enabled": False}}` to OpenRouter, which frees the
        whole budget for the answer. Off OpenRouter we send nothing, because a
        provider that never implemented the field would reject the unknown key.

        `{}` rather than `None` so it can always be splatted into `extra_body`.
        """
        # The `reasoning` field is an OpenRouter convention; a plain OpenAI (or
        # local) endpoint rejects the unknown top-level key. So NEITHER the enable
        # nor the disable form may be sent off OpenRouter.
        if not self._is_openrouter():
            return {}
        if self._reasoning_is_off(override):
            return {"reasoning": {"enabled": False}}  # explicit disable
        effort = self.reasoning_effort if override is _UNSET else override
        return {"reasoning": {"effort": str(effort).strip().lower()}}

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from OpenAI.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate. When omitted, the provider's
                configured budget (`OPENAI_MAX_TOKENS` / config `max_tokens`) is
                used rather than a small hard-coded default -- so a deployment that
                raised the budget for a reasoning model actually gets it on every
                call, not only the few that pass max_tokens explicitly.
            temperature: Sampling temperature (0.0-1.0)
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional args

        Returns:
            LLMResponse: Unified response object

        Raises:
            ProviderAPIError: If the API call fails
        """
        import time as time_module
        if max_tokens is None:
            max_tokens = self.max_tokens
        try:
            # Check if LLM call logging is enabled
            log_llm = False
            try:
                from kosmos.config import get_config
                config = get_config()
                log_llm = config.logging.log_llm_calls
            except Exception as e:
                logger.debug("Failed to load LLM call logging config: %s", e)

            # Build messages (OpenAI format: system is first message)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Prepare API call arguments
            api_args = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if stop_sequences:
                api_args["stop"] = stop_sequences

            # Native JSON mode: when the caller (e.g. generate_structured) asks
            # for structured output, force the model to emit a raw JSON object so
            # it cannot prepend prose like "Here is the protocol:" that defeats
            # JSON parsing. deepseek-v3-0324 / OpenRouter honor this.
            response_format = kwargs.get("response_format")
            if response_format:
                api_args["response_format"] = response_format

            # Pre-call logging
            if log_llm:
                logger.debug(
                    "[LLM] Request: model=%s, prompt_len=%d, system_len=%d, "
                    "max_tokens=%d, temp=%.2f",
                    self.model,
                    len(prompt),
                    len(system or ""),
                    max_tokens,
                    temperature
                )

            start_time = time_module.time()

            # Call OpenAI API
            reasoning_override = kwargs.get("reasoning_effort", _UNSET)
            response = self.client.chat.completions.create(
                **api_args,
                timeout=self._http_timeout,
                extra_body=self._reasoning_extra_body(reasoning_override),
            )

            # Extract text and usage. Reasoning models (e.g. deepseek-v4-flash)
            # return content=None when the response is truncated mid-reasoning
            # (finish_reason='length') — guard so len(text) doesn't crash.
            _msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Reasoning ate the budget. Callers pass small per-call budgets
            # (the data analyst asks for 2000 tokens for a JSON verdict; the
            # experiment designer 1000 for a yes/no), sized for the answer, not
            # for a reasoning trace that shares the same max_tokens on
            # OpenRouter's unified API. When the trace consumes it all, content
            # comes back empty with finish_reason='length' and the caller gets
            # nothing usable -- the result rows with NULL verdict and NULL
            # interpretation. The answer alone fits the budget; ask for it
            # without reasoning, once. JSON-mode requests are handled below by
            # `generate_structured`, which already retries the same way.
            reasoning_on = self._reasoning_enabled(reasoning_override)
            if (
                not (_msg.content or "").strip()
                and finish_reason == "length"
                and reasoning_on
                and not api_args.get("response_format")
            ):
                logger.warning(
                    "Reasoning trace consumed max_tokens=%d with no answer "
                    "(finish_reason=length); retrying once with reasoning off.",
                    max_tokens,
                )
                response = self.client.chat.completions.create(
                    **api_args,
                    timeout=self._http_timeout,
                    extra_body=self._reasoning_extra_body(None),
                )
                _msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

            # A reasoning trace is the model's scratchpad, NOT its answer. It is
            # an acceptable last resort for free-text (better to surface
            # something than an empty string), but it can never be valid JSON --
            # so when the caller asked for a JSON object, substituting it just
            # converts "the model ran out of tokens while reasoning" into a
            # baffling "could not parse JSON after 6 strategies" several frames
            # away. Fail here instead, naming the actual cause.
            # `.strip()` because a truncated reasoning response can come back as
            # whitespace rather than None -- which is falsy-looking to a human
            # but truthy to Python, so a bare `not _msg.content` would sail past
            # it and hand " " to the JSON parser.
            if not (_msg.content or "").strip() and api_args.get("response_format"):
                err = ProviderAPIError(
                    "openai",
                    f"Model returned no content for a JSON-mode request "
                    f"(finish_reason={finish_reason}). With reasoning enabled the "
                    f"reasoning trace and the answer share max_tokens "
                    f"({max_tokens}), so a long schema can leave nothing for the "
                    f"JSON itself. Raise max_tokens or lower reasoning effort.",
                )
                # Tagged so `generate_structured` can tell this budget
                # interaction apart from a real API failure and retry with
                # reasoning off -- the same self-healing it already applies when
                # a reasoning-truncated answer is present but unparseable.
                err.empty_content = True
                raise err

            text = _msg.content or getattr(_msg, "reasoning", None) or ""
            if not _msg.content and finish_reason == "length":
                logger.warning(
                    "OpenRouter/model returned no content (finish_reason=length); "
                    "response likely truncated mid-reasoning — increase max_tokens."
                )

            # Handle usage stats (may not be present for local models)
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Estimate for local models without usage stats
                input_tokens = self._estimate_tokens(prompt + (system or ""))
                output_tokens = self._estimate_tokens(text)
                total_tokens = input_tokens + output_tokens

            # Post-call logging
            latency_ms = int((time_module.time() - start_time) * 1000)
            if log_llm:
                logger.debug(
                    "[LLM] Response: model=%s, in_tokens=%d, out_tokens=%d, "
                    "latency=%dms, finish=%s",
                    self.model,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    finish_reason or "unknown"
                )

            # Calculate cost (only for OpenAI official)
            cost = self._calculate_cost(input_tokens, output_tokens) if self.provider_type == 'openai' else None

            usage_stats = UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                model=self.model,
                provider="openai",
                timestamp=datetime.now()
            )

            # Update stats
            self._update_usage_stats(usage_stats)

            logger.debug(f"Generated {len(text)} characters from OpenAI")

            return LLMResponse(
                content=text,
                usage=usage_stats,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=response,
                metadata={'provider_type': self.provider_type}
            )

        except ProviderAPIError:
            # Already ours -- in particular the empty-content error raised
            # above for JSON-mode requests carries the `empty_content` tag that
            # `generate_structured` keys its reasoning-off retry on. Wrapping it
            # in a fresh ProviderAPIError (as the generic branch below does)
            # dropped that tag, so the retry never fired. Let it through.
            raise
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise ProviderAPIError("openai", f"Generation failed: {e}", raw_error=e)

    @property
    def async_client(self) -> 'AsyncOpenAI':
        """
        Lazy-initialize async client with same config as sync client.

        Returns:
            AsyncOpenAI: Async OpenAI client instance
        """
        if self._async_client is None:
            if AsyncOpenAI is None:
                raise ImportError("AsyncOpenAI not available. Upgrade openai package.")
            self._async_client = AsyncOpenAI(**self._async_client_args)
        return self._async_client

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text asynchronously using true async OpenAI client.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional stop sequences
            **kwargs: Additional arguments

        Returns:
            LLMResponse: Unified response object
        """
        import time as time_module
        start_time = time_module.time()

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use async client for true async execution
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                extra_body=self._reasoning_extra_body(),
            )

            # Parse response. Same reasoning-model guard as the sync path
            # (generate()) -- content is None while the response is reasoning
            # tokens only, so without this fallback a reasoning model would
            # silently return "" here instead of surfacing its answer.
            _msg = response.choices[0].message
            content = _msg.content or getattr(_msg, "reasoning", None) or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            duration = time_module.time() - start_time

            # Log if enabled
            try:
                from kosmos.config import get_config
                if get_config().logging.log_llm_calls:
                    logger.debug(
                        "[LLM] OpenAI async: model=%s, in=%d, out=%d, duration=%.2fs",
                        self.model, input_tokens, output_tokens, duration
                    )
            except Exception as e:
                logger.debug(f"OpenAI call logging failed: {e}")

            return LLMResponse(
                content=content,
                model=self.model,
                usage=UsageStats(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
            )

        except Exception as e:
            logger.error(f"Async OpenAI API error: {e}")
            raise ProviderAPIError("openai", f"Async generation failed: {e}", raw_error=e)

    def generate_with_messages(
        self,
        messages: List[Message],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from conversation history.

        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            LLMResponse: Unified response object
        """
        try:
            # Convert Message objects to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Call API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self._http_timeout,
                extra_body=self._reasoning_extra_body(),
            )

            # Extract and convert (guard content=None for reasoning models)
            _msg2 = response.choices[0].message
            text = _msg2.content or getattr(_msg2, "reasoning", None) or ""
            finish_reason = response.choices[0].finish_reason

            # Handle usage stats
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Estimate for local models
                all_text = " ".join([msg.content for msg in messages])
                input_tokens = self._estimate_tokens(all_text)
                output_tokens = self._estimate_tokens(text)
                total_tokens = input_tokens + output_tokens

            cost = self._calculate_cost(input_tokens, output_tokens) if self.provider_type == 'openai' else None

            usage_stats = UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                model=self.model,
                provider="openai",
                timestamp=datetime.now()
            )

            self._update_usage_stats(usage_stats)

            return LLMResponse(
                content=text,
                usage=usage_stats,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=response
            )

        except Exception as e:
            logger.error(f"OpenAI multi-turn generation failed: {e}")
            raise ProviderAPIError("openai", f"Multi-turn generation failed: {e}", raw_error=e)

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: The user prompt
            schema: JSON schema or example structure
            system: Optional system prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Parsed JSON object

        Raises:
            ProviderAPIError: If generation or parsing fails
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        try:
            # Add JSON instruction to system prompt
            json_system = (system or "") + "\n\nYou must respond with valid JSON matching this schema:\n" + json.dumps(schema, indent=2)
            json_system += "\n\nIMPORTANT: Return ONLY valid JSON, no additional text or explanations."

            # Prefer native JSON mode so the model can't wrap the object in prose.
            # Fall back to a plain call if the model/provider rejects the param.
            gen_kwargs = dict(kwargs)
            gen_kwargs.setdefault("response_format", {"type": "json_object"})
            try:
                response = self.generate(
                    prompt=prompt,
                    system=json_system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **gen_kwargs
                )
            except ProviderAPIError as e:
                if (
                    getattr(e, "empty_content", False)
                    and self._reasoning_enabled()
                    and gen_kwargs.get("reasoning_effort", _UNSET) is not None
                ):
                    # The reasoning trace consumed the whole budget and no JSON
                    # came back at all. This is the empty-answer twin of the
                    # unparseable-answer case handled below, and it used to
                    # escape as a hard failure: the hypothesis generator
                    # (max_tokens=4000) hit it on most calls under
                    # reasoning_effort=high, burning three attempts per
                    # iteration before one happened to fit. Retry once with
                    # reasoning off; the schema dictates the shape anyway.
                    logger.warning(
                        "Model returned no content for structured output with "
                        "reasoning_effort=%r (finish_reason=length); retrying once "
                        "with reasoning disabled.",
                        self.reasoning_effort,
                    )
                    retry_kwargs = dict(gen_kwargs)
                    retry_kwargs["reasoning_effort"] = None
                    response = self.generate(
                        prompt=prompt,
                        system=json_system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **retry_kwargs,
                    )
                    gen_kwargs = retry_kwargs
                elif getattr(e, "empty_content", False):
                    # Reasoning is already OFF (so the retry above did not apply),
                    # yet the model still returned no JSON at finish_reason=length:
                    # the prompt + schema did not leave room for the answer in this
                    # budget. Turning reasoning off is not available, so raise the
                    # ceiling once. Without this the run silently degrades (e.g.
                    # 1 hypothesis instead of N) because nothing else recovers an
                    # empty structured response when reasoning was never on.
                    bumped = min(max_tokens * 2, _STRUCT_MAX_TOKENS_CAP)
                    if bumped <= max_tokens:
                        raise
                    logger.warning(
                        "Structured output empty (finish_reason=length) with "
                        "reasoning off; retrying once at max_tokens=%d.", bumped,
                    )
                    response = self.generate(
                        prompt=prompt,
                        system=json_system,
                        max_tokens=bumped,
                        temperature=temperature,
                        **gen_kwargs,
                    )
                    max_tokens = bumped
                elif "response_format" not in str(getattr(e, "raw_error", "")) and \
                   "response_format" not in str(e):
                    raise
                else:
                    logger.warning("Model rejected response_format=json_object; retrying without it")
                    gen_kwargs.pop("response_format", None)
                    response = self.generate(
                        prompt=prompt,
                        system=json_system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **gen_kwargs
                    )

            # A THIRD budget-interaction case, between "empty" and "unparseable":
            # the reasoning trace left just enough budget for a PREFIX of the
            # object, which parses (the salvage strategies close a truncated
            # array) but is SHORT -- a 5-hypothesis request comes back with the
            # one hypothesis that fit before the cut. It raises no error, so
            # neither retry above fires, and the run silently proceeds on a
            # single hypothesis. Catch it by the same signal the others use --
            # finish_reason=length with reasoning on -- and refetch with
            # reasoning off, which frees the whole budget for the complete array.
            if (
                getattr(response, "finish_reason", None) == "length"
                and self._reasoning_enabled()
                and gen_kwargs.get("reasoning_effort", _UNSET) is not None
            ):
                logger.warning(
                    "Structured output was truncated (finish_reason=length) with "
                    "reasoning_effort=%r; refetching once with reasoning disabled "
                    "so the full object fits.",
                    self.reasoning_effort,
                )
                retry_kwargs = dict(gen_kwargs)
                retry_kwargs["reasoning_effort"] = None
                try:
                    retry = self.generate(
                        prompt=prompt,
                        system=json_system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **retry_kwargs,
                    )
                    # Prefer the refetch only if it is itself not truncated; a
                    # still-truncated retry is no better than what we have.
                    if (retry.content or "").strip() and getattr(
                        retry, "finish_reason", None
                    ) != "length":
                        response = retry
                        gen_kwargs = retry_kwargs
                except ProviderAPIError:
                    # The refetch is a best-effort improvement; if it fails, fall
                    # back to parsing the (short but valid) original.
                    pass
            elif getattr(response, "finish_reason", None) == "length":
                # Same truncation, but reasoning is already OFF: the object itself
                # did not fit the budget. Refetch once at a higher ceiling so the
                # full array comes back (the parseable-but-short case that leaves a
                # run on a single hypothesis).
                bumped = min(max_tokens * 2, _STRUCT_MAX_TOKENS_CAP)
                if bumped > max_tokens:
                    logger.warning(
                        "Structured output truncated (finish_reason=length) with "
                        "reasoning off; refetching once at max_tokens=%d.", bumped,
                    )
                    try:
                        retry = self.generate(
                            prompt=prompt,
                            system=json_system,
                            max_tokens=bumped,
                            temperature=temperature,
                            **gen_kwargs,
                        )
                        if (retry.content or "").strip():
                            response = retry
                            max_tokens = bumped
                    except ProviderAPIError:
                        pass

            response_text = response.content

            # Parse JSON with robust fallback strategies
            try:
                return parse_json_response(response_text, schema=schema)

            except JSONParseError:
                # Reasoning and the answer compete for one token budget, so a
                # long schema (the experiment designer's is deeply nested) can
                # come back truncated and unparseable. Structured extraction
                # gains little from reasoning anyway -- the schema already
                # dictates the shape -- so retry once with it off before giving
                # up. Self-healing beats failing a whole research run on a
                # budget interaction the caller cannot see.
                # `_UNSET` as the default matters: a plain .get() returns None
                # when the key is ABSENT, which is also the value meaning
                # "reasoning already disabled" -- so the two cases would be
                # indistinguishable and the retry would never fire on the first
                # attempt, which is the only attempt that matters.
                if (
                    self._reasoning_enabled()
                    and gen_kwargs.get("reasoning_effort", _UNSET) is not None
                ):
                    logger.warning(
                        "Structured output was unparseable with "
                        "reasoning_effort=%r; retrying once with reasoning "
                        "disabled.",
                        self.reasoning_effort,
                    )
                    retry_kwargs = dict(gen_kwargs)
                    retry_kwargs["reasoning_effort"] = None
                    retry = self.generate(
                        prompt=prompt,
                        system=json_system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **retry_kwargs,
                    )
                    return parse_json_response(retry.content, schema=schema)
                # Reasoning already OFF: the unparseable text is almost always a
                # truncation the budget caused. Retry once at a higher ceiling.
                bumped = min(max_tokens * 2, _STRUCT_MAX_TOKENS_CAP)
                if bumped > max_tokens:
                    logger.warning(
                        "Structured output unparseable with reasoning off; "
                        "retrying once at max_tokens=%d.", bumped,
                    )
                    retry = self.generate(
                        prompt=prompt,
                        system=json_system,
                        max_tokens=bumped,
                        temperature=temperature,
                        **gen_kwargs,
                    )
                    return parse_json_response(retry.content, schema=schema)
                raise

        except JSONParseError as e:
            logger.error(f"Failed to parse JSON after {e.attempts} attempts")
            logger.error(f"Response text: {response_text[:500]}")

            # Provide helpful guidance for local model issues
            if self.provider_type == 'local':
                logger.error(
                    f"\n{'='*60}\n"
                    f"JSON parsing failed with local model ({self.model}).\n"
                    f"Local models may not reliably produce structured JSON output.\n\n"
                    f"Suggestions:\n"
                    f"  1. Try a larger model (e.g., llama3.1:70b instead of :8b)\n"
                    f"  2. Set LOCAL_MODEL_STRICT_JSON=false for lenient parsing\n"
                    f"  3. Use a cloud provider for complex structured outputs\n"
                    f"  4. Simplify the JSON schema if possible\n"
                    f"{'='*60}"
                )

            # JSON parse errors are NOT recoverable - retrying won't help
            raise ProviderAPIError(
                "openai",
                f"Invalid JSON response: {e.message}",
                raw_error=e,
                recoverable=False
            )

        except Exception as e:
            if isinstance(e, ProviderAPIError):
                raise

            # Provide helpful guidance for timeout errors with local models
            error_str = str(e).lower()
            if self.provider_type == 'local' and 'timeout' in error_str:
                logger.error(
                    f"Request to local model ({self.model}) timed out.\n"
                    f"Consider increasing LOCAL_MODEL_REQUEST_TIMEOUT or using a smaller model."
                )

            logger.error(f"Structured generation failed: {e}")
            raise ProviderAPIError("openai", f"Structured generation failed: {e}", raw_error=e)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict with model details
        """
        model_info = {
            "name": self.model,
            "provider": "openai",
            "provider_type": self.provider_type,
            "base_url": self.base_url or "https://api.openai.com/v1",
        }

        # Add pricing and context for known OpenAI models
        if self.provider_type == 'openai':
            if "gpt-4-turbo" in self.model.lower() or "gpt-4-1106" in self.model.lower():
                model_info["max_tokens"] = 128000
                model_info["cost_per_million_input_tokens"] = 10.00
                model_info["cost_per_million_output_tokens"] = 30.00
            elif "gpt-4" in self.model.lower():
                model_info["max_tokens"] = 8192
                model_info["cost_per_million_input_tokens"] = 30.00
                model_info["cost_per_million_output_tokens"] = 60.00
            elif "gpt-3.5-turbo" in self.model.lower():
                model_info["max_tokens"] = 16385
                model_info["cost_per_million_input_tokens"] = 0.50
                model_info["cost_per_million_output_tokens"] = 1.50
            elif "o1-preview" in self.model.lower():
                model_info["max_tokens"] = 128000
                model_info["cost_per_million_input_tokens"] = 15.00
                model_info["cost_per_million_output_tokens"] = 60.00
            elif "o1-mini" in self.model.lower():
                model_info["max_tokens"] = 128000
                model_info["cost_per_million_input_tokens"] = 3.00
                model_info["cost_per_million_output_tokens"] = 12.00

        return model_info

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for OpenAI official API.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            float: Cost in USD
        """
        if self.provider_type != 'openai':
            return 0.0  # No cost tracking for non-OpenAI providers

        # Pricing per million tokens (as of Nov 2024)
        if "gpt-4-turbo" in self.model.lower() or "gpt-4-1106" in self.model.lower():
            input_cost_per_m = 10.00
            output_cost_per_m = 30.00
        elif "gpt-4" in self.model.lower():
            input_cost_per_m = 30.00
            output_cost_per_m = 60.00
        elif "gpt-3.5-turbo" in self.model.lower():
            input_cost_per_m = 0.50
            output_cost_per_m = 1.50
        elif "o1-preview" in self.model.lower():
            input_cost_per_m = 15.00
            output_cost_per_m = 60.00
        elif "o1-mini" in self.model.lower():
            input_cost_per_m = 3.00
            output_cost_per_m = 12.00
        else:
            # Default to GPT-4 pricing
            input_cost_per_m = 30.00
            output_cost_per_m = 60.00

        input_cost = (input_tokens / 1_000_000) * input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_m

        return input_cost + output_cost

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimate for local models without usage stats.

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics.

        Returns:
            Dict with usage metrics
        """
        stats = super().get_usage_stats()

        # Add OpenAI-specific stats
        stats.update({
            "provider_type": self.provider_type,
            "base_url": self.base_url or "https://api.openai.com/v1",
        })

        return stats
