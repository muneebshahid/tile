"""Path resolution helpers for built-in filesystem tools."""

import re
from pathlib import Path

UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")


def resolve_to_cwd(path: str, cwd: Path) -> Path:
    """Resolve a user path relative to a tool working directory."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return cwd / candidate


def normalize_cwd(cwd: Path | str) -> Path:
    """Return a normalized tool working directory."""

    return Path(cwd).expanduser().resolve()


def normalize_at_prefix(path: str) -> str:
    """Strip a leading at sign used when users paste referenced paths."""

    if path.startswith("@"):
        return path[1:]
    return path


def normalize_unicode_spaces(path: str) -> str:
    """Normalize uncommon Unicode spaces to ordinary spaces."""

    return UNICODE_SPACES.sub(" ", path)
