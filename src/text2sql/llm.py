"""Provider-agnostic chat client for any OpenAI-compatible endpoint.

Only a tiny slice of the chat-completions API is used: send a list of messages,
get back one assistant message and a token count. Keeping the surface this small
means the same code path serves local Ollama, hosted free-tier gateways, and the
deterministic fake used by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .config import LLMConfig

Message = dict[str, str]


@dataclass
class ChatResponse:
    """An assistant reply and the token usage reported by the provider."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatClient(Protocol):
    """Anything that can turn a message list into a :class:`ChatResponse`."""

    def chat(self, messages: Sequence[Message]) -> ChatResponse: ...


class OpenAIChatClient:
    """Chat client backed by the ``openai`` SDK pointed at any base URL."""

    def __init__(self, config: LLMConfig | None = None):
        # Imported lazily so tests and the SQL sandbox never require the SDK.
        from openai import OpenAI

        self.config = config or LLMConfig.from_env()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.request_timeout,
        )

    def chat(self, messages: Sequence[Message]) -> ChatResponse:
        completion = self._client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            temperature=self.config.temperature,
        )
        usage = completion.usage
        return ChatResponse(
            content=completion.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


@dataclass
class FakeChatClient:
    """Deterministic client for tests and offline demos.

    Supply either a fixed list of replies (returned in order) or a callable that
    maps the message list to a reply string. Token counts are estimated from text
    length so usage-reporting code paths still exercise real numbers.
    """

    replies: list[str] = field(default_factory=list)
    responder: Callable[[Sequence[Message]], str] | None = None
    calls: list[list[Message]] = field(default_factory=list)
    _index: int = 0

    def chat(self, messages: Sequence[Message]) -> ChatResponse:
        self.calls.append(list(messages))
        if self.responder is not None:
            content = self.responder(messages)
        else:
            if self._index >= len(self.replies):
                raise AssertionError("FakeChatClient ran out of scripted replies.")
            content = self.replies[self._index]
            self._index += 1
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        return ChatResponse(content, prompt_tokens, len(content.split()))
