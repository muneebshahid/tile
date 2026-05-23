from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.4"
    chatgpt_backend: str = "https://chatgpt.com/backend-api/"

    model_config = {"env_file": ".env"}


settings = Settings()
