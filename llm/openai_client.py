from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from settings import settings


def _client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}

    base_url = getattr(settings, "openai_base_url", None)
    if base_url:
        kwargs["base_url"] = base_url

    return kwargs


def create_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(**_client_kwargs())
