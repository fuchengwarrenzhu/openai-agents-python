from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import socket
import sys
from typing import Any, TextIO

from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from .agent import Agent
from .items import TResponseInputItem
from .result import RunResultBase
from .run import DEFAULT_MAX_TURNS, Runner
from .run_context import TContext
from .stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent, RunItemStreamEvent

_INPUT_PROCESS_SCRIPT = r"""
import json
import os
import signal
import sys


def read_exactly(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


terminal = sys.argv[3] == "terminal"
sys.stdin.reconfigure(encoding=sys.argv[1], errors=sys.argv[2], newline=None)

signal.signal(signal.SIGINT, signal.SIG_IGN)
readline_module = None
if len(sys.argv) > 4:
    control = os.fdopen(int(sys.argv[4]), "rb", buffering=0)
    config_size = int.from_bytes(read_exactly(control, 8), "big")
    config = json.loads(read_exactly(control, config_size))
    if config["readline"]:
        import readline as readline_module

        for item in config["history"]:
            readline_module.add_history(item)
else:
    control = None
output = sys.stderr.buffer if terminal else sys.stdout.buffer
prompt = sys.argv[5] if terminal else ""

while True:
    if control is not None and control.read(1) != b"R":
        break
    history_length = (
        readline_module.get_current_history_length() if readline_module is not None else 0
    )
    try:
        value = input(prompt)
    except EOFError:
        result = ["eof", None, []]
    except UnicodeDecodeError as exc:
        result = [
            "decode_error",
            [exc.encoding, bytes(exc.object).hex(), exc.start, exc.end, exc.reason],
            [],
        ]
    except BaseException as exc:
        result = ["error", type(exc).__name__, []]
    else:
        history = (
            [
                readline_module.get_history_item(index)
                for index in range(
                    history_length + 1, readline_module.get_current_history_length() + 1
                )
            ]
            if readline_module is not None
            else []
        )
        result = ["line", value, history]

    payload = json.dumps(result).encode()
    output.write(len(payload).to_bytes(8, "big"))
    output.write(payload)
    output.flush()
    if result[0] != "line" or control is None:
        break
"""


class _StdinReader:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._terminal = sys.platform != "win32" and stream.isatty()
        self._process: asyncio.subprocess.Process | None = None
        self._spawn_task: asyncio.Task[asyncio.subprocess.Process] | None = None
        self._control_socket: socket.socket | None = None
        self._protocol_reader: asyncio.StreamReader | None = None
        self._terminal_fd: int | None = None
        self._terminal_attributes: list[Any] | None = None
        self._prompt_task: asyncio.Task[Any] | None = None
        self._prompt_cancelled_by_sigint = False
        self._previous_sigint_handler: Any = None
        self._sigint_handler: Any = None
        self._install_sigint_handler()

    @property
    def prompt_cancelled_by_sigint(self) -> bool:
        return self._prompt_cancelled_by_sigint

    async def readline(self, prompt: str) -> str:
        self._prompt_task = asyncio.current_task()
        self._prompt_cancelled_by_sigint = False
        try:
            return await self._readline(prompt)
        finally:
            self._prompt_task = None

    async def _readline(self, prompt: str) -> str:
        if not self._terminal:
            sys.stdout.write(prompt)
            sys.stdout.flush()

        process = await self._get_process(prompt)
        if self._control_socket is not None:
            await asyncio.get_running_loop().sock_sendall(self._control_socket, b"R")
        protocol_reader = self._protocol_reader
        if protocol_reader is None:
            raise RuntimeError("The stdin helper process has no output pipe.")

        try:
            header = await protocol_reader.readexactly(8)
            payload_size = int.from_bytes(header, "big")
            if payload_size == 0:
                raise RuntimeError("The stdin helper process returned an invalid response.")
            payload = await protocol_reader.readexactly(payload_size)
        except asyncio.IncompleteReadError:
            await process.wait()
            raise RuntimeError(
                f"The stdin helper process exited unexpectedly with code {process.returncode}."
            ) from None

        if sys.platform == "win32":
            await process.wait()
            self._process = None
            self._protocol_reader = None

        kind, value, history = json.loads(payload)
        if kind == "eof":
            raise EOFError
        if kind == "decode_error":
            try:
                encoding, object_hex, start, end, reason = value
                error = UnicodeDecodeError(encoding, bytes.fromhex(object_hex), start, end, reason)
            except (TypeError, ValueError):
                raise RuntimeError(
                    "The stdin helper process returned an invalid response."
                ) from None
            raise error
        if kind == "error":
            raise RuntimeError(f"The stdin helper process failed with {value}.")
        if kind != "line" or not isinstance(value, str):
            raise RuntimeError("The stdin helper process returned an invalid response.")
        if self._terminal:
            if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
                raise RuntimeError("The stdin helper process returned an invalid response.")
            for item in history:
                _record_readline_history(item)
        return value

    async def aclose(self) -> None:
        cancellation: asyncio.CancelledError | None = None
        process = self._process
        if process is None and self._spawn_task is not None:
            process, cancellation = await _finish_spawn(self._spawn_task)
            self._spawn_task = None
            self._process = process

        if process is None:
            self._restore_terminal()
            self._restore_sigint_handler()
            if cancellation is not None:
                raise cancellation
            return

        try:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            wait_cancellation = await _wait_for_process(process)
            if cancellation is None:
                cancellation = wait_cancellation
        finally:
            self._restore_terminal()
            self._restore_sigint_handler()
            self._process = None
            self._protocol_reader = None
            if self._control_socket is not None:
                self._control_socket.close()
                self._control_socket = None
        if cancellation is not None:
            raise cancellation

    async def _get_process(self, prompt: str) -> asyncio.subprocess.Process:
        if self._process is not None:
            return self._process

        if self._spawn_task is None:
            encoding = self._stream.encoding
            if not encoding:
                raise RuntimeError("The stdin stream has no text encoding.")
            self._spawn_task = asyncio.create_task(
                self._spawn_process(prompt, encoding, self._stream.errors or "strict")
            )

        process = await asyncio.shield(self._spawn_task)
        self._spawn_task = None
        self._process = process
        return process

    async def _spawn_process(
        self, prompt: str, encoding: str, errors: str
    ) -> asyncio.subprocess.Process:
        if sys.platform == "win32":
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                "-c",
                _INPUT_PROCESS_SCRIPT,
                encoding,
                errors,
                "stream",
                stdin=self._stream,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
            )
            self._protocol_reader = process.stdout
            return process

        parent_socket, child_socket = socket.socketpair()
        parent_socket.setblocking(False)
        if self._terminal:
            import termios

            self._terminal_fd = self._stream.fileno()
            self._terminal_attributes = termios.tcgetattr(self._terminal_fd)
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                "-c",
                _INPUT_PROCESS_SCRIPT,
                encoding,
                errors,
                "terminal" if self._terminal else "stream",
                str(child_socket.fileno()),
                prompt,
                stdin=self._stream,
                stdout=None if self._terminal else asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if self._terminal else None,
                pass_fds=(child_socket.fileno(),),
            )
        except BaseException:
            parent_socket.close()
            raise
        finally:
            child_socket.close()

        self._control_socket = parent_socket
        self._protocol_reader = process.stderr if self._terminal else process.stdout
        config = json.dumps(
            _readline_config() if self._terminal else {"readline": False, "history": []}
        ).encode()
        try:
            await asyncio.get_running_loop().sock_sendall(
                parent_socket, len(config).to_bytes(8, "big") + config
            )
        except BaseException:
            parent_socket.close()
            self._control_socket = None
            self._protocol_reader = None
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await _wait_for_process(process)
            raise
        return process

    def _restore_terminal(self) -> None:
        terminal_fd = self._terminal_fd
        terminal_attributes = self._terminal_attributes
        self._terminal_fd = None
        self._terminal_attributes = None
        if terminal_fd is None or terminal_attributes is None:
            return

        import termios

        with contextlib.suppress(OSError):
            termios.tcsetattr(terminal_fd, termios.TCSANOW, terminal_attributes)

    def _install_sigint_handler(self) -> None:
        previous_handler = signal.getsignal(signal.SIGINT)
        if not callable(previous_handler):
            return

        def handle_sigint(signum: int, frame: Any) -> None:
            prompt_task = self._prompt_task
            cancelling_before = prompt_task.cancelling() if prompt_task is not None else 0
            try:
                previous_handler(signum, frame)
            finally:
                if (
                    prompt_task is not None
                    and cancelling_before == 0
                    and prompt_task.cancelling() > 0
                ):
                    self._prompt_cancelled_by_sigint = True

        try:
            signal.signal(signal.SIGINT, handle_sigint)
        except ValueError:
            return
        self._previous_sigint_handler = previous_handler
        self._sigint_handler = handle_sigint

    def _restore_sigint_handler(self) -> None:
        handler = self._sigint_handler
        previous_handler = self._previous_sigint_handler
        self._sigint_handler = None
        self._previous_sigint_handler = None
        if handler is None or signal.getsignal(signal.SIGINT) is not handler:
            return
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, previous_handler)


def _readline_config() -> dict[str, bool | list[str]]:
    readline: Any = sys.modules.get("readline")
    if readline is None:
        return {"readline": False, "history": []}

    history = [
        item
        for index in range(1, readline.get_current_history_length() + 1)
        if (item := readline.get_history_item(index)) is not None
    ]
    return {"readline": True, "history": history}


def _record_readline_history(value: str) -> None:
    readline: Any = sys.modules.get("readline")
    if readline is not None:
        readline.add_history(value)


async def _finish_spawn(
    task: asyncio.Task[asyncio.subprocess.Process],
) -> tuple[asyncio.subprocess.Process | None, asyncio.CancelledError | None]:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    try:
        return task.result(), cancellation
    except BaseException:
        return None, cancellation


async def _wait_for_process(
    process: asyncio.subprocess.Process,
) -> asyncio.CancelledError | None:
    cancellation: asyncio.CancelledError | None = None
    wait_task = asyncio.create_task(process.wait())
    while not wait_task.done():
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    wait_task.result()
    return cancellation


def _create_stdin_reader() -> _StdinReader:
    return _StdinReader(sys.stdin)


async def run_demo_loop(
    agent: Agent[Any],
    *,
    stream: bool = True,
    context: TContext | None = None,
    max_turns: int | None = DEFAULT_MAX_TURNS,
) -> None:
    """Run a simple REPL loop with the given agent.

    This utility allows quick manual testing and debugging of an agent from the
    command line. Conversation state is preserved across turns. Enter ``exit``
    or ``quit`` to stop the loop.

    Args:
        agent: The starting agent to run.
        stream: Whether to stream the agent output.
        context: Additional context information to pass to the runner.
        max_turns: Maximum number of turns for the runner to iterate. Pass ``None`` to disable
            the turn limit.
    """

    stdin_reader = _create_stdin_reader()
    try:
        current_agent = agent
        input_items: list[TResponseInputItem] = []
        while True:
            try:
                user_input = await stdin_reader.readline(" > ")
            except asyncio.CancelledError:
                if not stdin_reader.prompt_cancelled_by_sigint:
                    raise
                print()
                break
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            if not user_input.strip():
                continue

            input_items.append({"role": "user", "content": user_input})

            result: RunResultBase
            if stream:
                result = Runner.run_streamed(
                    current_agent, input=input_items, context=context, max_turns=max_turns
                )
                async for event in result.stream_events():
                    if isinstance(event, RawResponsesStreamEvent):
                        if isinstance(event.data, ResponseTextDeltaEvent):
                            print(event.data.delta, end="", flush=True)
                    elif isinstance(event, RunItemStreamEvent):
                        if event.item.type == "tool_call_item":
                            print("\n[tool called]", flush=True)
                        elif event.item.type == "tool_call_output_item":
                            print(f"\n[tool output: {event.item.output}]", flush=True)
                    elif isinstance(event, AgentUpdatedStreamEvent):
                        print(f"\n[Agent updated: {event.new_agent.name}]", flush=True)
                print()
            else:
                result = await Runner.run(
                    current_agent, input_items, context=context, max_turns=max_turns
                )
                if result.final_output is not None:
                    print(result.final_output)

            current_agent = result.last_agent
            input_items = result.to_input_list()
    finally:
        await stdin_reader.aclose()
