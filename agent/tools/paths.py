"""Path resolution helpers for built-in filesystem tools."""

from pathlib import Path


def resolve_to_cwd(path: str, cwd: Path) -> Path:
    """Resolve a user path relative to a tool working directory."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return cwd / candidate


def normalize_cwd(cwd: Path | str) -> Path:
    """Return a normalized tool working directory."""

    return Path(cwd).expanduser().resolve()
