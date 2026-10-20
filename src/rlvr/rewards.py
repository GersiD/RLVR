'''Verifiable rewards for code generation: run the model's program against unit tests.'''
import contextlib
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# A base model keeps writing after the function; cut it at the first top-level statement (HumanEval's stop sequences).
STOP_SEQUENCES = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


@dataclass
class CodeProblem:
    '''
    A coding problem for the model to complete.

    Attributes:
        prompt: Code the model continues, typically a signature and docstring,
            e.g. 'def reverse(s):\\n    """Return s reversed."""\\n'.
        tests: Python statements (usually asserts) run after the program. Any exception counts as a failure.
        task_id: Identifier from the source dataset, for tracking per-problem results.
    '''
    prompt: str
    tests: str
    task_id: str = ""


RewardFn = Callable[[CodeProblem, str], float]


def truncate(completion: str) -> str:
    '''Cut completion at the first stop sequence.'''
    cut = min((i for s in STOP_SEQUENCES if (i := completion.find(s)) != -1), default=len(completion))
    return completion[:cut]


def run_tests(problem: CodeProblem, completion: str, timeout: float = 5.0) -> float:
    '''
    Return 1.0 if problem.prompt + completion passes problem.tests, else 0.0.

    The program runs in a separate Python process with a timeout, an empty environment and a temporary working
    directory, so crashes and infinite loops can't take down training. On timeout the program's whole process
    group is killed, so subprocesses it started don't outlive it. This is not a security sandbox: the program
    can still read and write your files. Run training in a container if that matters.

    Passing requires a random nonce, printed after the tests, to appear in stdout. Otherwise a program that calls
    sys.exit(0) before the asserts run would exit cleanly and pass. The program is fed through stdin, so there is
    no source file for it to read the nonce from.
    '''
    nonce = secrets.token_hex(8)
    program = problem.prompt + truncate(completion) + "\n\n" + problem.tests + f"\nprint({nonce!r})\n"
    with tempfile.TemporaryDirectory() as tmp, subprocess.Popen(
        [sys.executable, "-I", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp,
        env={},
        start_new_session=True,
    ) as proc:
        try:
            stdout, _ = proc.communicate(program.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError): # the group may have exited since the timeout
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 0.0
    return float(proc.returncode == 0 and nonce.encode() in stdout)


def score(
    problems: Sequence[CodeProblem],
    completions: Sequence[str],
    reward_fn: RewardFn = run_tests,
    max_workers: int | None = None,
) -> list[float]:
    '''
    Score completions[i] against problems[i] for every i, running the reward_fn calls concurrently.

    Uses threads, not processes: run_tests already runs each program in its own OS process, so a thread only
    waits on its subprocess. A process pool would fork the training process, which is unsafe on macOS with torch
    and an MPS context loaded, or spawn workers that each re-import torch.

    Args:
        problems: One problem per completion; repeat a problem for each completion in its group.
        completions: Model outputs, paired with problems element by element.
        reward_fn: Called as reward_fn(problem, completion). Must be thread-safe.
        max_workers: Maximum concurrent calls. Defaults to os.cpu_count(), since each test program is CPU-bound.

    Returns:
        Rewards in input order.

    Raises:
        ValueError: If problems and completions have different lengths.
    '''
    if len(problems) != len(completions):
        raise ValueError(f"{len(problems)} problems but {len(completions)} completions")
    if not problems:
        return []
    with ThreadPoolExecutor(max_workers or os.cpu_count()) as pool:
        return list(pool.map(reward_fn, problems, completions))
