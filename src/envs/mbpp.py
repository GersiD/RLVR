'''
MBPP (Mostly Basic Python Problems; Austin et al., 2021) as CodeProblems, loaded from the local mbpp.jsonl.

Each prompt gives the code that precedes the tested function in the reference solution (imports, constants,
helper functions, classes the tests construct), the task description and the first test as comments, then the
tested function's def line; the model writes the body. The reward runs all three tests, so hard-coding the
example that was shown doesn't pass.
'''
import ast
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Literal

from rlvr.rewards import CodeProblem

DATA = Path(__file__).with_name("mbpp.jsonl")

Split = Literal["train", "test"]


def load_rows(split: Split, train_fraction: float = 0.8, seed: int = 0) -> list[dict]:
    '''
    Raw MBPP rows in the given split.

    Rows are ordered by a seeded hash of their task_id and the first train_fraction of them go to train. The split
    is the same on every run, and removing a row from mbpp.jsonl moves at most one other problem across it.
    '''
    with DATA.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['task_id']}".encode()).digest())
    cut = round(train_fraction * len(rows))
    return rows[:cut] if split == "train" else rows[cut:]


def split_reference(row: dict) -> tuple[str, str, str]:
    '''
    Split a row's reference solution around the function its tests call.

    Returns:
        The code before that function, its def line (several lines if the signature wraps), and everything
        after the def line.
    '''
    code = row["code"].replace("\r\n", "\n")
    lines = [line.rstrip() for line in code.split("\n")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # some solutions use regexes like "\w" in plain strings
        tree = ast.parse(code)
    func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and re.search(rf"\b{n.name}\s*\(", row["test_list"][0])
    )
    preamble = "\n".join(lines[:func.lineno - 1]).strip()
    first = func.body[0]
    if first.lineno == func.lineno:
        # One-liner like `def f(x): return x`: move the body onto its own indented line.
        line = lines[func.lineno - 1]
        body = "    " + line[first.col_offset:] + "\n" + "\n".join(lines[func.lineno:])
        return preamble, line[:first.col_offset].rstrip(), body
    signature = "\n".join(lines[func.lineno - 1:first.lineno - 1])
    return preamble, signature, "\n".join(lines[first.lineno - 1:])


def to_problem(row: dict) -> CodeProblem:
    '''Turn an MBPP row into a CodeProblem.'''
    preamble, signature, _ = split_reference(row)
    header = f"{preamble}\n\n" if preamble else ""
    description = "".join(f"# {line}\n" for line in row["text"].splitlines())
    prompt = f"{header}{description}# Example: {row['test_list'][0]}\n{signature}\n"
    tests = "\n".join([row["test_setup_code"], *row["test_list"]]).strip()
    return CodeProblem(prompt=prompt, tests=tests, task_id=f"mbpp/{row['task_id']}")


def load_problems(split: Split, train_fraction: float = 0.8, seed: int = 0) -> list[CodeProblem]:
    '''MBPP problems in the given split. See load_rows for how the split is made.'''
    return [to_problem(row) for row in load_rows(split, train_fraction, seed)]
