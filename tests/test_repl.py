import asyncio
import contextlib
import os
import shutil
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

import agents.repl as repl_module
from agents import Agent, run_demo_loop
from agents.testing import ScriptedModel

from .test_responses import (
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_input_item,
    get_text_message,
)


class StubStdinReader:
    def __init__(self, inputs: Iterator[str | BaseException]) -> None:
        self.inputs = inputs
        self.closed = False

    async def readline(self, _prompt: str) -> str:
        value = next(self.inputs)
        if isinstance(value, BaseException):
            raise value
        return value

    async def aclose(self) -> None:
        self.closed = True


def patch_stdin_reader(
    monkeypatch: pytest.MonkeyPatch, inputs: list[str | BaseException]
) -> StubStdinReader:
    reader = StubStdinReader(iter(inputs))
    monkeypatch.setattr(repl_module, "_create_stdin_reader", lambda: reader)
    return reader


@pytest.mark.asyncio
async def test_run_demo_loop_conversation(monkeypatch, capsys):
    model = ScriptedModel()
    model.extend([[get_text_message("hello")], [get_text_message("good")]])

    agent = Agent(name="test", model=model)

    reader = patch_stdin_reader(monkeypatch, ["Hi", "How are you?", "quit"])

    await run_demo_loop(agent, stream=False)

    output = capsys.readouterr().out
    assert "hello" in output
    assert "good" in output
    assert model.calls[-1].input == [
        get_text_input_item("Hi"),
        get_text_message("hello").model_dump(exclude_unset=True),
        get_text_input_item("How are you?"),
    ]
    assert reader.closed


@pytest.mark.asyncio
async def test_run_demo_loop_streaming(monkeypatch, capsys):
    model = ScriptedModel()
    target_agent = Agent(name="target", model=model)
    agent = Agent(
        name="test",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
        handoffs=[target_agent],
    )

    # A single user turn that exercises every streamed event branch:
    # a tool call, the tool output, a handoff (agent update), then a text answer.
    model.extend(
        [
            [get_function_tool_call("foo", "{}")],
            [get_handoff_tool_call(target_agent)],
            [get_text_message("all done")],
        ]
    )

    patch_stdin_reader(monkeypatch, ["Hello", "exit"])

    await run_demo_loop(agent, stream=True)

    output = capsys.readouterr().out
    assert "all done" in output
    assert "[tool called]" in output
    assert "[tool output: tool_result]" in output
    assert "[Agent updated: target]" in output


@pytest.mark.asyncio
async def test_run_demo_loop_exits_on_eof(monkeypatch, capsys):
    model = ScriptedModel()
    agent = Agent(name="test", model=model)

    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with os.fdopen(read_fd, encoding="utf-8") as stdin:
        monkeypatch.setattr(sys, "stdin", stdin)
        await run_demo_loop(agent, stream=False)

    # The loop should terminate cleanly without ever invoking the model.
    assert not model.calls


@pytest.mark.asyncio
async def test_run_demo_loop_skips_empty_input(monkeypatch, capsys):
    model = ScriptedModel()
    model.extend([[get_text_message("hello")]])
    agent = Agent(name="test", model=model)

    # Empty lines are ignored; only the non-empty input reaches the runner.
    patch_stdin_reader(monkeypatch, ["", "Hi", "quit"])

    await run_demo_loop(agent, stream=False)

    output = capsys.readouterr().out
    assert "hello" in output
    assert model.calls[-1].input == [get_text_input_item("Hi")]


@pytest.mark.asyncio
async def test_run_demo_loop_skips_whitespace_only_input(monkeypatch, capsys):
    model = ScriptedModel()
    agent = Agent(name="test", model=model)
    patch_stdin_reader(monkeypatch, ["   ", "quit"])

    await run_demo_loop(agent, stream=False)

    assert not model.calls


@pytest.mark.asyncio
async def test_run_demo_loop_does_not_block_the_event_loop(monkeypatch):
    model = ScriptedModel()
    agent = Agent(name="test", model=model)

    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, encoding="utf-8") as stdin:
        monkeypatch.setattr(sys, "stdin", stdin)
        os.write(write_fd, b"qu")

        async def release_prompt() -> None:
            await asyncio.sleep(0)
            os.write(write_fd, b"it\n")

        try:
            await asyncio.gather(run_demo_loop(agent, stream=False), release_prompt())
        finally:
            os.close(write_fd)

    assert not model.calls


@pytest.mark.asyncio
async def test_run_demo_loop_handles_crlf_input(monkeypatch, capsys):
    model = ScriptedModel()
    model.extend([[get_text_message("ok")]])
    agent = Agent(name="test", model=model)

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"hello\r\nquit\r\n")
    os.close(write_fd)
    with os.fdopen(read_fd, encoding="utf-8", newline=None) as stdin:
        monkeypatch.setattr(sys, "stdin", stdin)
        await run_demo_loop(agent, stream=False)

    assert model.calls[-1].input == [get_text_input_item("hello")]
    assert capsys.readouterr().out.count(" > ") == 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGINT behavior is under test")
@pytest.mark.asyncio
async def test_run_demo_loop_ctrl_c_does_not_wait_for_stdin_worker():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = str(repo_root / "src")
    if existing_path := env.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, existing_path])
    env["PYTHONPATH"] = python_path

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import asyncio\n"
            "from agents import Agent, run_demo_loop\n"
            "asyncio.run(run_demo_loop(Agent(name='test'), stream=False))\n"
            "print('RETURNED', flush=True)\n"
        ),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    assert process.stdin is not None

    try:
        prompt = await asyncio.wait_for(process.stdout.readexactly(3), timeout=5)
        assert prompt == b" > "

        process.send_signal(signal.SIGINT)
        await asyncio.wait_for(process.wait(), timeout=5)
        stdout = await process.stdout.read()
        stderr = await process.stderr.read()
    finally:
        process.stdin.close()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.wait_closed()
        if process.returncode is None:
            process.kill()
            await process.wait()

    assert process.returncode == 0
    assert stdout == b"\nRETURNED\n"
    assert stderr == b""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminal behavior is under test")
@pytest.mark.asyncio
async def test_run_demo_loop_preserves_readline_history():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = str(repo_root / "src")
    if existing_path := env.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, existing_path])
    env["PYTHONPATH"] = python_path
    expect = shutil.which("expect")
    if expect is None:
        pytest.skip("expect is required for the POSIX readline regression")
    child_code = (
        "import asyncio, readline\n"
        "from agents import Agent, run_demo_loop\n"
        "from agents.testing import ScriptedModel\n"
        "from tests.test_responses import get_text_message\n"
        "model = ScriptedModel()\n"
        "model.extend([[get_text_message('ok')]])\n"
        "readline.add_history('quit')\n"
        "asyncio.run(run_demo_loop(Agent(name='test', model=model), stream=False))\n"
        "print('CALLS=' + str(len(model.calls)), flush=True)\n"
    )
    expect_script = (
        "log_user 1\n"
        "set timeout 5\n"
        f"spawn {{{sys.executable}}} -c {{{child_code}}}\n"
        'expect " > "\n'
        'send "\\033\\[A"\n'
        'expect "quit"\n'
        'send "\\r"\n'
        "expect eof\n"
    )
    process = await asyncio.create_subprocess_exec(
        expect,
        "-c",
        expect_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

    assert process.returncode == 0, stderr.decode()
    assert b"CALLS=0" in stdout
    assert b"^[[A" not in stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX readline history is under test")
@pytest.mark.asyncio
async def test_run_demo_loop_updates_parent_readline_history():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = str(repo_root / "src")
    if existing_path := env.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, existing_path])
    env["PYTHONPATH"] = python_path
    expect = shutil.which("expect")
    if expect is None:
        pytest.skip("expect is required for the POSIX readline regression")
    child_code = (
        "import asyncio, readline\n"
        "from agents import Agent, run_demo_loop\n"
        "readline.clear_history()\n"
        "asyncio.run(run_demo_loop(Agent(name='test'), stream=False))\n"
        "history = [readline.get_history_item(index) for index in "
        "range(1, readline.get_current_history_length() + 1)]\n"
        "print('HISTORY=' + repr(history), flush=True)\n"
    )
    expect_script = (
        "log_user 1\n"
        "set timeout 5\n"
        f"spawn {{{sys.executable}}} -c {{{child_code}}}\n"
        'expect " > "\n'
        'send "\\r"\n'
        'expect " > "\n'
        'send "   \\r"\n'
        'expect " > "\n'
        'send "   \\r"\n'
        'expect " > "\n'
        'send "quit\\r"\n'
        "expect eof\n"
    )
    process = await asyncio.create_subprocess_exec(
        expect,
        "-c",
        expect_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

    assert process.returncode == 0, stderr.decode()
    assert b"HISTORY=['   ', 'quit']" in stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminal SIGINT is under test")
@pytest.mark.asyncio
async def test_run_demo_loop_ctrl_c_reaps_terminal_input_helper():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = str(repo_root / "src")
    if existing_path := env.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, existing_path])
    env["PYTHONPATH"] = python_path
    expect = shutil.which("expect")
    if expect is None:
        pytest.skip("expect is required for the POSIX terminal regression")
    child_code = (
        "import asyncio\n"
        "from agents import Agent, run_demo_loop\n"
        "asyncio.run(run_demo_loop(Agent(name='test'), stream=False))\n"
    )
    expect_script = (
        "log_user 1\n"
        "set timeout 5\n"
        f"spawn {{{sys.executable}}} -c {{{child_code}}}\n"
        'expect " > "\n'
        'send "\\003"\n'
        "expect eof\n"
    )
    process = await asyncio.create_subprocess_exec(
        expect,
        "-c",
        expect_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    _, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

    assert process.returncode == 0, stderr.decode()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminal cleanup is under test")
@pytest.mark.asyncio
async def test_run_demo_loop_cancellation_restores_terminal_mode():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = str(repo_root / "src")
    if existing_path := env.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, existing_path])
    env["PYTHONPATH"] = python_path
    expect = shutil.which("expect")
    if expect is None:
        pytest.skip("expect is required for the POSIX terminal regression")
    child_code = (
        "import asyncio, readline, sys, termios\n"
        "from agents import Agent, run_demo_loop\n"
        "async def main():\n"
        "    before = termios.tcgetattr(sys.stdin.fileno())\n"
        "    task = asyncio.create_task(run_demo_loop(Agent(name='test'), stream=False))\n"
        "    while termios.tcgetattr(sys.stdin.fileno()) == before:\n"
        "        await asyncio.sleep(0)\n"
        "    task.cancel()\n"
        "    try:\n"
        "        await task\n"
        "    except asyncio.CancelledError:\n"
        "        pass\n"
        "    assert termios.tcgetattr(sys.stdin.fileno()) == before\n"
        "    print('TERMINAL_RESTORED', flush=True)\n"
        "asyncio.run(main())\n"
    )
    expect_script = (
        "log_user 1\n"
        "set timeout 5\n"
        f"spawn {{{sys.executable}}} -c {{{child_code}}}\n"
        'expect "TERMINAL_RESTORED"\n'
        "expect eof\n"
    )
    process = await asyncio.create_subprocess_exec(
        expect,
        "-c",
        expect_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    _, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

    assert process.returncode == 0, stderr.decode()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX cancellation is under test")
@pytest.mark.asyncio
async def test_run_demo_loop_ctrl_c_interrupts_continuously_readable_stdin():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = str(repo_root / "src")
    if existing_path := env.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, existing_path])
    env["PYTHONPATH"] = python_path

    with open("/dev/zero", "rb") as stdin:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            (
                "import asyncio\n"
                "from agents import Agent, run_demo_loop\n"
                "asyncio.run(run_demo_loop(Agent(name='test'), stream=False))\n"
            ),
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        try:
            prompt = await asyncio.wait_for(process.stdout.readexactly(3), timeout=5)
            assert prompt == b" > "

            process.send_signal(signal.SIGINT)
            await asyncio.wait_for(process.wait(), timeout=5)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    assert process.returncode == 0


@pytest.mark.asyncio
async def test_stdin_helper_is_reaped_on_cancellation(monkeypatch, capsys):
    read_fd, write_fd = os.pipe()
    process_created = asyncio.Event()
    created_process: asyncio.subprocess.Process | None = None
    real_create_subprocess_exec = asyncio.create_subprocess_exec

    async def create_subprocess_exec(*args, **kwargs):
        nonlocal created_process
        created_process = await real_create_subprocess_exec(*args, **kwargs)
        process_created.set()
        return created_process

    monkeypatch.setattr(repl_module.asyncio, "create_subprocess_exec", create_subprocess_exec)

    with os.fdopen(read_fd, encoding="utf-8") as stdin:
        monkeypatch.setattr(sys, "stdin", stdin)
        agent = Agent(name="test", model=ScriptedModel())
        read_task = asyncio.create_task(run_demo_loop(agent, stream=False))
        try:
            await asyncio.wait_for(process_created.wait(), timeout=5)
            read_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(read_task, timeout=5)
        finally:
            os.close(write_fd)

    assert created_process is not None
    assert created_process.returncode is not None


@pytest.mark.asyncio
async def test_handled_sigint_does_not_mask_later_prompt_cancellation(monkeypatch):
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    prompt_pending = asyncio.Event()
    signal_seen = False

    class PromptReader(repl_module._StdinReader):
        def __init__(self) -> None:
            super().__init__(sys.stdin)
            self.calls = 0

        async def _readline(self, _prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "hello"
            prompt_pending.set()
            await asyncio.Future()
            raise AssertionError("The pending prompt should be cancelled.")

        async def aclose(self) -> None:
            self._restore_sigint_handler()

    class FakeRunner:
        @staticmethod
        async def run(agent, input_items, context=None, max_turns=None):
            model_started.set()
            await release_model.wait()
            return type(
                "Result",
                (),
                {
                    "final_output": None,
                    "last_agent": agent,
                    "to_input_list": lambda self: input_items,
                },
            )()

    def handle_sigint(_signum, _frame) -> None:
        nonlocal signal_seen
        signal_seen = True

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    reader = PromptReader()
    monkeypatch.setattr(repl_module, "_create_stdin_reader", lambda: reader)
    monkeypatch.setattr(repl_module, "Runner", FakeRunner)

    try:
        task = asyncio.create_task(
            run_demo_loop(Agent(name="test", model=ScriptedModel()), stream=False)
        )
        await asyncio.wait_for(model_started.wait(), timeout=5)
        current_handler = signal.getsignal(signal.SIGINT)
        assert callable(current_handler)
        current_handler(signal.SIGINT, None)
        release_model.set()
        await asyncio.wait_for(prompt_pending.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        reader._restore_sigint_handler()
        signal.signal(signal.SIGINT, previous_handler)

    assert signal_seen


@pytest.mark.asyncio
async def test_programmatic_cancellation_wins_over_overlapping_sigint(monkeypatch):
    prompt_pending = asyncio.Event()
    task_box: list[asyncio.Task[None]] = []

    class PromptReader(repl_module._StdinReader):
        async def _readline(self, _prompt: str) -> str:
            prompt_pending.set()
            await asyncio.Future()
            raise AssertionError("The pending prompt should be cancelled.")

        async def aclose(self) -> None:
            self._restore_sigint_handler()

    def handle_sigint(_signum, _frame) -> None:
        task_box[0].cancel("sigint")

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    reader = PromptReader(sys.stdin)
    monkeypatch.setattr(repl_module, "_create_stdin_reader", lambda: reader)

    try:
        task = asyncio.create_task(
            run_demo_loop(Agent(name="test", model=ScriptedModel()), stream=False)
        )
        task_box.append(task)
        await asyncio.wait_for(prompt_pending.wait(), timeout=5)
        task.cancel("programmatic")
        current_handler = signal.getsignal(signal.SIGINT)
        assert callable(current_handler)
        current_handler(signal.SIGINT, None)
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        reader._restore_sigint_handler()
        signal.signal(signal.SIGINT, previous_handler)


@pytest.mark.asyncio
async def test_cancellation_during_stdin_cleanup_is_propagated(monkeypatch):
    class CleanupProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.waiting = asyncio.Event()
            self.release = asyncio.Event()
            self.killed = False

        def kill(self) -> None:
            self.killed = True
            self.returncode = -signal.SIGKILL

        async def wait(self) -> int:
            self.waiting.set()
            await self.release.wait()
            assert self.returncode is not None
            return self.returncode

    class QuittingReader(repl_module._StdinReader):
        async def readline(self, _prompt: str) -> str:
            return "quit"

    process = CleanupProcess()
    reader = QuittingReader(sys.stdin)
    reader._process = cast(asyncio.subprocess.Process, process)
    monkeypatch.setattr(repl_module, "_create_stdin_reader", lambda: reader)

    task = asyncio.create_task(
        run_demo_loop(Agent(name="test", model=ScriptedModel()), stream=False)
    )
    await asyncio.wait_for(process.waiting.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    process.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed
    assert process.returncode == -signal.SIGKILL
