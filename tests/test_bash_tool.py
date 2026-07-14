"""Tests for the default shell command tool scaffold."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import tile.tools.bash as bash
from tile.tools.bash import BashDetails
from tile.types.tools import ToolResult
from tile.tools.support.output_accumulator import OutputAccumulator, OutputSnapshot
from tile.tools.support.truncation import OUTPUT_LINE_LIMIT
from tests.support.tool_results import tool_text


def test_bash_schema_requires_only_command() -> None:
    """Require only the command so callers can omit optional timeout."""

    assert bash.tool.input_schema["required"] == ["command"]


def test_bash_schema_exposes_command_controls() -> None:
    """Expose shell command inputs without execution-injected fields."""

    properties = bash.tool.input_schema["properties"]

    assert bash.tool.name == "bash"
    assert isinstance(properties, dict)
    assert set(properties) == {"command", "timeout"}


def test_bash_input_accepts_integer_timeout() -> None:
    """Accept whole-second JSON numbers for a floating-point timeout."""

    params = bash.BashInput.model_validate({"command": "true", "timeout": 30})

    assert params.timeout == 30.0


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch shell execution with an async mock for fn-level tests."""

    execution_mock = AsyncMock()
    monkeypatch.setattr(bash, "_execute", execution_mock)
    return execution_mock


@pytest.mark.asyncio
async def test_fn_runs_command_from_supplied_cwd(tmp_path: Path) -> None:
    """Run shell commands from the injected working directory."""

    (tmp_path / "sample.txt").write_text("content", encoding="utf-8")

    tool_result = await bash.fn(bash.BashInput(command="pwd && ls"), cwd=tmp_path)
    result = tool_text(tool_result)

    assert result.splitlines() == [str(tmp_path.resolve()), "sample.txt"]
    assert tool_result.details is None


@pytest.mark.asyncio
async def test_fn_combines_stderr_with_stdout(tmp_path: Path) -> None:
    """Return stdout and stderr in one text result."""

    result = tool_text(
        await bash.fn(
            bash.BashInput(command="printf 'out'; printf 'err' >&2"),
            cwd=tmp_path,
        )
    )

    assert result == "outerr"


@pytest.mark.asyncio
async def test_fn_returns_no_output_marker(tmp_path: Path) -> None:
    """Return an explicit marker when a successful command emits nothing."""

    tool_result = await bash.fn(bash.BashInput(command="true"), cwd=tmp_path)
    result = tool_text(tool_result)

    assert result == "(no output)"
    assert tool_result.details is None


@pytest.mark.asyncio
async def test_fn_truncates_to_tail_output(
    execution: AsyncMock,
    tmp_path: Path,
) -> None:
    """Keep the end of large bash output instead of returning all content."""

    output = "\n".join(f"line {index}" for index in range(2002))
    execution.return_value = _snapshot(output)

    tool_result = await bash.fn(
        bash.BashInput(command="generate-output"),
        cwd=tmp_path,
    )
    result = tool_text(tool_result)

    assert result.startswith("line 2\n")
    assert result.endswith("\n\n[Showing lines 3-2002 of 2002]")
    details = _bash_details(tool_result)
    assert details.output.truncated is True
    assert details.output.truncated_by == "lines"
    assert details.output.keep == "tail"
    assert details.output.output_lines == OUTPUT_LINE_LIMIT
    assert details.output.total_lines == OUTPUT_LINE_LIMIT + 2
    execution.assert_awaited_once_with(
        "generate-output",
        bash.DEFAULT_TIMEOUT_SECONDS,
        tmp_path.resolve(),
    )


@pytest.mark.asyncio
async def test_fn_raises_with_output_on_nonzero_exit(tmp_path: Path) -> None:
    """Raise command output and exit code when the command fails."""

    with pytest.raises(RuntimeError) as error:
        await bash.fn(bash.BashInput(command="printf 'boom'; exit 7"), cwd=tmp_path)

    assert str(error.value) == "boom\n\nCommand exited with code 7"


@pytest.mark.asyncio
async def test_fn_raises_status_only_when_failed_command_has_no_output(
    tmp_path: Path,
) -> None:
    """Avoid adding a no-output marker to failed commands."""

    with pytest.raises(RuntimeError) as error:
        await bash.fn(bash.BashInput(command="exit 7"), cwd=tmp_path)

    assert str(error.value) == "Command exited with code 7"


@pytest.mark.asyncio
async def test_fn_raises_with_timeout_status(tmp_path: Path) -> None:
    """Terminate commands that exceed the supplied timeout."""

    with pytest.raises(RuntimeError) as error:
        await bash.fn(
            bash.BashInput(command="printf 'start'; sleep 2", timeout=0.1),
            cwd=tmp_path,
        )

    assert str(error.value) == "start\n\nCommand timed out after 0.1 seconds"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 0, -5], ids=["omitted", "zero", "negative"])
async def test_fn_applies_default_timeout(
    execution: AsyncMock,
    tmp_path: Path,
    timeout: float | None,
) -> None:
    """Fall back to the default timeout for omitted or non-positive values."""

    execution.return_value = _snapshot("ok")

    await bash.fn(bash.BashInput(command="true", timeout=timeout), cwd=tmp_path)

    execution.assert_awaited_once_with(
        "true",
        bash.DEFAULT_TIMEOUT_SECONDS,
        tmp_path.resolve(),
    )


@pytest.mark.asyncio
async def test_stop_timed_out_process_escalates_to_kill_when_sigterm_ignored() -> None:
    """No CancelledError escapes when SIGTERM is ignored and SIGKILL is required."""

    process = await asyncio.create_subprocess_shell(
        "trap '' SIGTERM; sleep 10",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=bash._supports_process_groups(),
    )
    wait_task = asyncio.create_task(process.wait())

    await bash._stop_timed_out_process(process, wait_task)

    assert process.returncode is not None


def _bash_details(result: ToolResult) -> BashDetails:
    """Return bash details from a tool result."""

    assert isinstance(result.details, BashDetails)
    return result.details


def _snapshot(output: str) -> OutputSnapshot:
    """Build an output snapshot from text."""

    accumulator = OutputAccumulator()
    accumulator.accumulate(output.encode("utf-8"))
    return accumulator.finish()
