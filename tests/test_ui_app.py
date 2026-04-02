import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

from agent.agent import Agent
from ai.types.conversation import UserMessage
from ai.types.stream import (
    AssistantMessage,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)
from main import main
from textual.content import Content
from textual.widgets import Static
from ui import PiyApp
from ui.widgets import (
    AgentMessageWidget,
    InputSection,
    OutputSection,
    UserMessageWidget,
)


def test_piy_app_can_be_constructed() -> None:
    app = PiyApp()

    assert app.title == "piy"
    assert isinstance(app._agent, Agent)
    composed_widgets = list(app.compose())
    output_widget = composed_widgets[0]

    assert len(composed_widgets) == 2
    assert isinstance(output_widget, OutputSection)
    assert output_widget.id == "output"
    assert output_widget._messages == []
    assert output_widget._empty_state_widget is not None
    assert output_widget._empty_state_widget.text == "Welcome to piy."
    assert isinstance(composed_widgets[1], InputSection)
    assert composed_widgets[1].id == "input"
    assert composed_widgets[1].read_only is False


def test_piy_app_uses_injected_agent() -> None:
    agent = Agent(stream_fn=AsyncMock(), model="gpt-5.4")
    app = PiyApp(agent=agent)

    assert app._agent is agent


def test_create_agent_uses_default_openai_dependencies() -> None:
    app = PiyApp()

    assert app._agent._stream_fn is not None
    assert app._agent._model == "gpt-5.4"


def test_main_uses_piy_app() -> None:
    with patch.object(PiyApp, "run") as run_mock:
        main()

    run_mock.assert_called_once_with()


def test_piy_app_focuses_input_on_mount() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.focused is app.query_one(InputSection)

    asyncio.run(_run())


def test_clicking_output_does_not_move_focus_from_input() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#output")
            await pilot.pause()

            assert app.focused is app.query_one(InputSection)

    asyncio.run(_run())


def test_ctrl_c_quits_the_app_without_help_prompt() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.is_running is True

            await pilot.press("ctrl+c")
            await pilot.pause()

            assert app.is_running is False

    asyncio.run(_run())


def test_pressing_enter_moves_input_text_into_output_history() -> None:
    async def _run() -> None:
        agent = _build_agent([])
        app = PiyApp(agent=agent)

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)
            output_area = app.query_one(OutputSection)

            input_area.load_text("Hello, piy!")
            await pilot.press("enter")
            await pilot.pause()

            assert len(output_area.children) == 1
            message = output_area.children[-1]
            assert isinstance(message, UserMessageWidget)
            assert message.text == "Hello, piy!"
            assert input_area.text == ""

    asyncio.run(_run())


def test_pressing_enter_appends_user_message_to_agent_history() -> None:
    async def _run() -> None:
        agent = Agent(stream_fn=AsyncMock(), model="gpt-5.4")
        app = PiyApp(agent=agent)

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)

            input_area.load_text("Hello from agent history")
            await pilot.press("enter")
            await pilot.pause()

            assert len(agent.history) == 1
            user_message = agent.history[0]
            assert isinstance(user_message, UserMessage)
            assert user_message.content == "Hello from agent history"

    asyncio.run(_run())


def test_pressing_enter_streams_agent_text_into_output() -> None:
    async def _run() -> None:
        partial_message = AssistantMessage(
            response_id="resp_123",
            content=[TextBlock(text="Hello from agent")],
        )
        final_message = AssistantMessage(
            response_id="resp_123",
            content=[TextBlock(text="Hello from agent")],
        )
        agent = _build_agent(
            [
                StreamStartEvent(
                    type="start",
                    partial=AssistantMessage(response_id="resp_123"),
                ),
                TextStartEvent(type="text_start", partial=partial_message),
                TextDeltaEvent(
                    type="text_delta",
                    delta="Hello from agent",
                    partial=partial_message,
                ),
                TextEndEvent(type="text_end", partial=partial_message),
                StreamDoneEvent(type="done", message=final_message),
            ]
        )
        app = PiyApp(agent=agent)

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)
            output_area = app.query_one(OutputSection)

            input_area.load_text("Hello, piy!")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            assert len(output_area.children) == 2
            user_message = output_area.children[0]
            agent_message = output_area.children[1]
            assert isinstance(user_message, UserMessageWidget)
            assert user_message.text == "Hello, piy!"
            assert isinstance(agent_message, AgentMessageWidget)
            assert agent_message.text == "Hello from agent"
            assert _read_rendered_message_text(agent_message) == "Hello from agent"

    asyncio.run(_run())


def test_message_end_renders_final_text_without_text_deltas() -> None:
    async def _run() -> None:
        final_message = AssistantMessage(
            response_id="resp_456",
            content=[TextBlock(text="Final text only")],
        )
        agent = _build_agent(
            [
                StreamStartEvent(
                    type="start",
                    partial=AssistantMessage(response_id="resp_456"),
                ),
                StreamDoneEvent(type="done", message=final_message),
            ]
        )
        app = PiyApp(agent=agent)

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)
            output_area = app.query_one(OutputSection)

            input_area.load_text("Hello, piy!")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            assert len(output_area.children) == 2
            agent_message = output_area.children[1]
            assert isinstance(agent_message, AgentMessageWidget)
            assert agent_message.text == "Final text only"
            assert _read_rendered_message_text(agent_message) == "Final text only"

    asyncio.run(_run())


def test_message_end_renders_stream_error_without_text_deltas() -> None:
    async def _run() -> None:
        error_message = AssistantMessage(
            response_id="resp_error",
            stop_reason="error",
            error_message="Socket closed",
        )
        agent = _build_agent(
            [
                StreamStartEvent(
                    type="start",
                    partial=AssistantMessage(response_id="resp_error"),
                ),
                StreamErrorEvent(type="error", error=error_message),
            ]
        )
        app = PiyApp(agent=agent)

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)
            output_area = app.query_one(OutputSection)

            input_area.load_text("Hello, piy!")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            assert len(output_area.children) == 2
            agent_message = output_area.children[1]
            assert isinstance(agent_message, AgentMessageWidget)
            assert agent_message.text == "Socket closed"
            assert _read_rendered_message_text(agent_message) == "Socket closed"

    asyncio.run(_run())


def test_input_area_expands_for_wrapped_content() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test(size=(40, 20)) as pilot:
            input_area = app.query_one(InputSection)

            await pilot.pause()
            initial_height = input_area.size.height

            input_area.load_text(
                "This is a long prompt that should wrap and expand the input area "
                "beyond a single line."
            )
            await pilot.pause()

            assert input_area.size.height > initial_height

    asyncio.run(_run())


def _build_agent(stream_events: Sequence[StreamEvent]) -> Agent:
    async def _stream_fn(*_: object, **__: object) -> AsyncIterator[StreamEvent]:
        return _iter_stream_events(stream_events)

    return Agent(stream_fn=_stream_fn, model="gpt-5.4")


def _iter_stream_events(
    stream_events: Sequence[StreamEvent],
) -> AsyncIterator[StreamEvent]:
    async def _iterate() -> AsyncIterator[StreamEvent]:
        for event in stream_events:
            yield event

    return _iterate()


def _read_rendered_message_text(widget: AgentMessageWidget) -> str:
    """Return the text currently rendered inside a transcript widget."""

    content = cast(Content, widget.query_one(Static).render())
    return content.plain
