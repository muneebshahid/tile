"""Shell command tool for the default agent."""

import asyncio
import os
import signal
import sys
from pathlib import Path

from pydantic import BaseModel

from ai.types.tools import ToolDefinition, ToolResult


class ExecutionResult(BaseModel):
    """Captured shell command execution result."""

    output: str
    exit_code: int | None
    timed_out: bool
    timeout: float | None


async def fn(command: str, timeout: float | None = None, *, cwd: Path) -> ToolResult:
    """Execute a shell command from the agent working directory."""

    resolved_cwd = _resolve_cwd(cwd)
    result = await _execute(command, timeout, resolved_cwd)
    return ToolResult.text(_format_results(result))


def _resolve_cwd(cwd: Path) -> Path:
    """Resolve and validate the shell working directory."""

    resolved_cwd = cwd.expanduser().resolve(strict=True)
    if not resolved_cwd.is_dir():
        raise NotADirectoryError(resolved_cwd)
    return resolved_cwd


async def _execute(
    command: str,
    timeout: float | None,
    cwd: Path,
) -> ExecutionResult:
    """Execute a shell command and capture combined stdout and stderr."""

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        start_new_session=_supports_process_groups(),
    )
    output, timed_out = await _wait_for_process(process, timeout)
    return ExecutionResult(
        output=output,
        exit_code=process.returncode,
        timed_out=timed_out,
        timeout=timeout,
    )


async def _wait_for_process(
    process: asyncio.subprocess.Process,
    timeout: float | None,
) -> tuple[str, bool]:
    """Wait for a shell process while enforcing the optional timeout."""

    output_chunks: list[bytes] = []
    output_task = asyncio.create_task(_read_output(process, output_chunks))
    wait_task = asyncio.create_task(process.wait())
    timed_out = False

    try:
        await asyncio.wait_for(
            asyncio.shield(wait_task),
            timeout=_effective_timeout(timeout),
        )
    except TimeoutError:
        timed_out = True
        await _stop_timed_out_process(process, wait_task)
    finally:
        await output_task

    return b"".join(output_chunks).decode(errors="replace"), timed_out


async def _read_output(
    process: asyncio.subprocess.Process,
    output_chunks: list[bytes],
) -> None:
    """Read merged process output until the pipe closes."""

    if process.stdout is None:
        return

    while chunk := await process.stdout.read(4096):
        output_chunks.append(chunk)


async def _stop_timed_out_process(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
) -> None:
    """Terminate a timed-out process and escalate if it does not exit."""

    _terminate_process(process)
    try:
        await asyncio.wait_for(wait_task, timeout=1)
    except TimeoutError:
        _kill_process(process)
        await wait_task


def _format_results(result: ExecutionResult) -> str:
    """Format shell output or raise for command failure states."""

    if result.timed_out:
        raise RuntimeError(
            _append_status(result.output, _timeout_status(result.timeout))
        )
    if result.exit_code not in (0, None):
        raise RuntimeError(
            _append_status(
                result.output, f"Command exited with code {result.exit_code}"
            )
        )
    return result.output or "(no output)"


def _append_status(output: str, status: str) -> str:
    """Append a command status line after captured output."""

    if output:
        return f"{output}\n\n{status}"
    return status


def _timeout_status(timeout: float | None) -> str:
    """Return a timeout failure status message."""

    if timeout is None:
        return "Command timed out"
    return f"Command timed out after {timeout:g} seconds"


def _effective_timeout(timeout: float | None) -> float | None:
    """Return a positive timeout value or no timeout."""

    if timeout is None or timeout <= 0:
        return None
    return timeout


def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Send a graceful termination signal to a process or process group."""

    if process.returncode is not None:
        return
    if _supports_process_groups() and process.pid is not None:
        _signal_process_group(process.pid, signal.SIGTERM)
        return
    process.terminate()


def _kill_process(process: asyncio.subprocess.Process) -> None:
    """Force-kill a process or process group."""

    if process.returncode is not None:
        return
    if _supports_process_groups() and process.pid is not None:
        _signal_process_group(process.pid, signal.SIGKILL)
        return
    process.kill()


def _signal_process_group(pid: int, signal_number: signal.Signals) -> None:
    """Send a signal to a POSIX process group if it still exists."""

    try:
        os.killpg(pid, signal_number)
    except ProcessLookupError:
        return


def _supports_process_groups() -> bool:
    """Return whether subprocesses can be isolated in POSIX process groups."""

    return sys.platform != "win32"


tool = ToolDefinition(
    name="bash",
    description="Execute a bash command.",
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. Optional, with no default timeout.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    fn=fn,
)
