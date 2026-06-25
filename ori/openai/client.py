from openai import AsyncOpenAI

from settings import settings


def create_client() -> AsyncOpenAI:
    _validate_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _validate_settings() -> None:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required to create the OpenAI client")

    if not settings.openai_base_url:
        raise ValueError("OPENAI_BASE_URL is required to create the OpenAI client")
