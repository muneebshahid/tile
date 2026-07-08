"""System prompt construction for the default coding agent."""

from pathlib import Path

PROMPT = """
You are an expert coding assistant operating inside Tile, a headless Python agent runtime.
You help users by reading files, executing commands, editing code, and writing new files.

Current working directory: {cwd}
"""


def build_system_prompt(prompt: str, cwd: Path) -> str:
    """Return the system prompt with runtime prompt variables applied."""

    return prompt.replace("{cwd}", str(cwd))
