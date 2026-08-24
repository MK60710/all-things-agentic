"""Shared machinery behind every structured-output Gemini judge in this
codebase - GapFinder's GeminiExplainer, ContradictionFinder's
GeminiContradictionJudge, and FeynmanChecker's GeminiFeynmanJudge each
independently reimplemented the same ~80-100 line "lazy Vertex AI client,
call generate_content with a structured response_schema, check for
truncation/an empty response, parse it, log and return None on any
failure" machinery, including a previously-found production bug fix
(thinking_config=ThinkingConfig(thinking_budget=0) - "thinking" tokens
silently eating the whole max_output_tokens budget on a short judgment
task) that each copy had to independently remember and correctly
re-apply. A future fix to the shared client/timeout/auth logic - or a
fourth judge class - now only needs to touch this one place.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import TypeVar

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=BaseModel)


class LazyVertexClient:
    """Lazy, thread-safe Vertex AI client construction - shared base for
    every judge class below. Constructed lazily (not in __init__) so a
    bad ADC/project config surfaces inside a judge's own try/except at
    call time, not at construction time - every judge's contract is
    that an auth/config issue never crashes the caller, only ever
    yields a None verdict."""

    def __init__(
        self,
        client: genai.Client | None = None,
        project: str | None = None,
        location: str | None = None,
    ):
        self._project = project
        self._location = location
        self._client_lock = threading.Lock()
        self._client = client

    def _get_client(self) -> genai.Client:
        with self._client_lock:
            if self._client is None:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    location=self._location
                    or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                )
            return self._client


def call_structured_judge(
    get_client: Callable[[], genai.Client],
    *,
    model: str,
    contents: str,
    system_instruction: str,
    response_schema: type[_T],
    max_output_tokens: int,
    timeout_ms: int,
    caller_name: str,
) -> _T | None:
    """One real, structured Gemini call - temperature=0 (deterministic,
    matching every other Gemini call in this codebase) and thinking
    disabled (see module docstring). Returns the parsed response, or
    None on any failure - a truncated/empty/invalid response, or an
    outage/quota/auth exception - never raises.

    Takes a client *getter* (e.g. a judge's own bound self._get_client,
    not a pre-resolved client) and calls it inside this function's own
    try/except, not before - a bad ADC/project config only ever
    surfaces on the first real call, and if the caller resolved the
    client before calling in, that failure would raise outside this
    function's try/except and propagate uncaught, breaking every
    judge's "auth/config issues never crash the caller" contract.
    """
    try:
        client = get_client()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
                max_output_tokens=max_output_tokens,
                http_options=genai_types.HttpOptions(timeout=timeout_ms),
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        raw = (response.text or "").strip()
        candidates = getattr(response, "candidates", None) or []
        finish_reason = candidates[0].finish_reason if candidates else None
        truncated = finish_reason == genai_types.FinishReason.MAX_TOKENS
        if raw and not truncated:
            try:
                return response_schema.model_validate_json(raw)
            except ValueError:
                logger.warning(
                    "%s response failed schema validation", caller_name, exc_info=True
                )
        elif truncated:
            logger.warning(
                "%s response was truncated by max_output_tokens", caller_name
            )
        else:
            logger.warning("%s got an empty response", caller_name)
    except Exception:
        # Never let a Gemini outage/quota/config issue raise out of the
        # caller - every judge's fallback (a template, or just None)
        # must always be reachable.
        logger.warning("%s call failed", caller_name, exc_info=True)
    return None
