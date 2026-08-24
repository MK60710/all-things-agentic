"""Small Vertex AI chat adapter used when no research paper is attached."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    text: str


class GeneralChatAgent:
    """A deliberately thin wrapper around Gemini on Vertex AI."""

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str = "global",
        model: str = "gemini-3.5-flash-lite",
        client: Any | None = None,
        timeout_ms: int = 20_000,
        max_output_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._client = client
        if self._client is None and project is not None:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        self._timeout_ms = timeout_ms
        self._max_output_tokens = max_output_tokens

    def answer(self, message: str, history: list[ChatTurn] | None = None) -> str:
        cleaned = message.strip()
        if not cleaned:
            return "Please enter a message."
        if self._client is None:
            raise RuntimeError("Vertex AI chat is not configured")

        contents = [
            types.Content(
                role="model" if turn.role == "assistant" else "user",
                parts=[types.Part(text=turn.text)],
            )
            for turn in (history or [])[-20:]
            if turn.text.strip()
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=cleaned)]))
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are Atlas, a clear and helpful research assistant. Answer "
                    "normally when no paper is attached. Be honest about uncertainty, "
                    "and do not claim to have read a paper unless its contents were supplied."
                ),
                # Every other Gemini call in this codebase (extraction,
                # QueryAgent's answer synthesis) already uses temperature=0.
                # This was the one inconsistent spot - it's also the exact
                # path that produced two different answers (one hallucinated,
                # one a decline) to the identical MIA question before the
                # routing fix sent that query to QueryAgent instead.
                temperature=0,
                max_output_tokens=self._max_output_tokens,
                http_options=types.HttpOptions(timeout=self._timeout_ms),
            ),
        )
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty response")
        return answer
