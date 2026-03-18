import unittest
from unittest.mock import patch

from llm.openai_client import create_openai_client


class CreateOpenAIClientTests(unittest.TestCase):
    def test_raises_when_api_key_is_missing(self) -> None:
        with (
            patch("llm.openai_client.settings.openai_api_key", None),
            patch(
                "llm.openai_client.settings.openai_base_url",
                "https://api.openai.com/v1",
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "OPENAI_API_KEY is required to create the OpenAI client",
            ):
                create_openai_client()

    def test_raises_when_base_url_is_missing(self) -> None:
        with (
            patch("llm.openai_client.settings.openai_api_key", "test-key"),
            patch("llm.openai_client.settings.openai_base_url", None),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "OPENAI_BASE_URL is required to create the OpenAI client",
            ):
                create_openai_client()

    def test_returns_async_client_when_config_is_present(self) -> None:
        with (
            patch("llm.openai_client.settings.openai_api_key", "test-key"),
            patch(
                "llm.openai_client.settings.openai_base_url",
                "https://api.openai.com/v1",
            ),
        ):
            client = create_openai_client()

        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(str(client.base_url), "https://api.openai.com/v1/")


if __name__ == "__main__":
    unittest.main()
