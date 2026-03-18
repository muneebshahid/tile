from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    model_config = {"env_file": ".env"}


settings = Settings()  # type: ignore[call-arg]
