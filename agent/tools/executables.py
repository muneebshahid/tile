"""External executable availability helpers for built-in tools."""

import asyncio
import shutil
from collections.abc import Sequence
from pathlib import Path


def require_executable(command: str, display_name: str) -> str:
    """Return an executable path or raise a clear tool availability error."""

    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(f"{display_name} is not available.")
    return executable


async def execute(
    executable: str,
    args: Sequence[str],
    allowed_exit_codes: tuple[int, ...] = (0,),
    *,
    cwd: Path,
) -> str:
    """Run an executable asynchronously and return captured stdout."""

    process = await asyncio.create_subprocess_exec(
        executable,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    exit_code = process.returncode or 0
    if exit_code not in allowed_exit_codes:
        error = (
            stderr_bytes.decode().strip()
            or f"{executable} exited with code {exit_code}"
        )
        raise RuntimeError(error)
    return stdout_bytes.decode()
