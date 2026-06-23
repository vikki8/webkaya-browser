"""Claude code generation and the agent loop for the Python client.

Mirrors the TypeScript ``src/llm`` layer: a provider-agnostic interface, a
Claude implementation over the official ``anthropic`` SDK (an optional
dependency — ``pip install webkaya[claude]``), and a ``CodeAgent`` that drives
the generate -> run -> repair loop against a Python ``Sandbox``.

The Python sandbox runs Python guest code, so Claude here writes Python that
operates on ``ctx`` (state/args/log) — governed by the same policy, probes, and
memory tiers as any other run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .policy import DISALLOWED_GUEST_TOKENS
from .sandbox import RunResult, Sandbox


@dataclass
class CodeGenResult:
    code: str
    explanation: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class LlmProvider:
    """Structural interface: any object with ``name`` and ``generate_code``."""

    name: str = "provider"

    def generate_code(self, system: str, prompt: str) -> CodeGenResult:  # pragma: no cover
        raise NotImplementedError


_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "The generated Python, ready to execute."},
        "explanation": {"type": "string", "description": "One or two sentences on what the code does."},
    },
    "required": ["code", "explanation"],
    "additionalProperties": False,
}


class ClaudeProvider(LlmProvider):
    """Claude-backed code generation via the official ``anthropic`` SDK.

    Defaults to ``claude-opus-4-8`` with adaptive thinking and structured JSON
    output. The SDK is imported lazily so the core package needs no extra deps
    unless you use this provider.
    """

    name = "claude"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-8",
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client: Any = None
        self._anthropic: Any = None

    def _load(self) -> Any:
        if self._anthropic is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    'ClaudeProvider needs the "anthropic" package. Install it with: '
                    "pip install webkaya[claude]"
                ) from exc
            self._anthropic = anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic

    def generate_code(self, system: str, prompt: str) -> CodeGenResult:
        anthropic = self._load()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=system,
                output_config={"format": {"type": "json_schema", "schema": _CODE_SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError("Claude API: invalid or missing API key.") from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError("Claude API: rate limited — wait and retry.") from exc
        except anthropic.APIError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Claude API error: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("Claude declined to generate code for this request.")
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise RuntimeError("Claude hit the output token limit — raise max_tokens and retry.")

        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if not text:
            raise RuntimeError("Claude returned no text content.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Claude returned output that was not valid JSON.") from exc
        if not isinstance(parsed.get("code"), str) or not parsed["code"].strip():
            raise RuntimeError("Claude returned no code.")

        return CodeGenResult(
            code=parsed["code"],
            explanation=parsed.get("explanation", "") if isinstance(parsed.get("explanation"), str) else "",
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )


_GUEST_SYSTEM_PROMPT = f"""You write Python that runs inside the WebKaya sandbox.

Environment contract:
- Your code is the body of a function receiving one argument `ctx`.
- `ctx.state` is a dict holding the sandbox's persistent state. Mutations are committed only if the code finishes without raising.
- `ctx.args` carries this run's input (may be None).
- `ctx.log(message)` records a line of output.
- End with `return <value>` to produce the run's result.

Hard constraints (the sandbox rejects code containing these substrings, so never use them): {", ".join(repr(t.strip()) for t in DISALLOWED_GUEST_TOKENS)}.
That means NO imports — only Python builtins and `ctx`. There is no network, filesystem, or module system. Keep the code short and self-contained."""


@dataclass
class AgentAttempt:
    code: str
    explanation: str
    result: RunResult


@dataclass
class AgentOutcome:
    ok: bool
    result: RunResult
    code: str
    explanation: str
    attempts: List[AgentAttempt] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class CodeAgent:
    """Generate -> run -> repair loop driven against a Python ``Sandbox``.

    The model writes guest code, the sandbox runs it under full policy (token
    scan, memory budget, probes, memory tiers), and any failure — a raised
    exception or a probe veto — is fed back for another attempt. Every attempt
    is recorded in the sandbox's run log.
    """

    def __init__(
        self,
        provider: LlmProvider,
        sandbox: Sandbox,
        max_attempts: int = 3,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._provider = provider
        self._sandbox = sandbox
        self._max_attempts = max(1, max_attempts)
        self._on_log = on_log or (lambda _m: None)

    def run(self, task: str, args: Any = None) -> AgentOutcome:
        attempts: List[AgentAttempt] = []
        input_tokens = 0
        output_tokens = 0
        prompt = f"Task: {task}\n\nWrite the sandbox function body that accomplishes it."

        for attempt in range(1, self._max_attempts + 1):
            self._on_log(f"[agent] generating code (attempt {attempt}/{self._max_attempts})")
            generated = self._provider.generate_code(_GUEST_SYSTEM_PROMPT, prompt)
            input_tokens += generated.input_tokens
            output_tokens += generated.output_tokens

            result = self._sandbox.run(generated.code, name=f"agent-attempt-{attempt}", args=args)
            attempts.append(AgentAttempt(generated.code, generated.explanation, result))

            if result.ok:
                self._on_log(f"[agent] succeeded on attempt {attempt}")
                return AgentOutcome(
                    ok=True,
                    result=result,
                    code=generated.code,
                    explanation=generated.explanation,
                    attempts=attempts,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            self._on_log(f"[agent] attempt {attempt} failed: {result.error}")
            prompt = (
                f"Task: {task}\n\nThe previous attempt failed. Fix it.\n\n"
                f"Previous code:\n{generated.code}\n\nError:\n{result.error}"
            )

        last = attempts[-1]
        return AgentOutcome(
            ok=False,
            result=last.result,
            code=last.code,
            explanation=last.explanation,
            attempts=attempts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
