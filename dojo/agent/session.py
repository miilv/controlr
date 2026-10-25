"""Persistent Python session for the agent's code cells (notebook semantics, bounded, restricted).

* One globals dict per episode: names a cell defines are there for the next cell.
* A trailing expression is evaluated and echoed (``repr``), like a notebook.
* ``print`` writes to the cell's buffer (not to the process stdout: the eval driver prints from another
  thread); output is clipped to ``max_output`` characters, head and tail kept.
* Restricted: imports only from ``ALLOWED_IMPORTS``, no ``open`` / ``exec`` / ``eval`` / ``compile`` /
  ``__import__`` builtins, no dunder attribute access (AST check). The code is the model's; the restriction keeps
  it on the robot API and off the host.
* Bounded: the ``robo`` API raises ``CellCheckpoint`` (a BaseException, so a bare ``except Exception`` in the
  model's code cannot swallow it) when a cell's env-step or contact budget is spent; ``EpisodeFinished`` ends the
  episode. Both end the cell at a primitive boundary; the robot is never left mid-motion.
"""

from __future__ import annotations

import ast
import builtins
import math
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

ALLOWED_IMPORTS = {"math", "numpy", "json", "itertools", "collections", "re", "random", "statistics", "functools",
                   "dataclasses", "typing"}
BLOCKED_BUILTINS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "exit", "quit", "help",
                    "globals", "locals", "vars", "memoryview", "getattr", "setattr", "delattr"}


class CellCheckpoint(BaseException):
    """The cell's step / contact budget is spent: control returns to the model."""


class EpisodeFinished(BaseException):
    """finish() was called (or the episode ended): no further cells."""


class CellRefused(Exception):
    pass


@dataclass
class CellResult:
    stdout: str = ""
    value: str | None = None
    error: str | None = None
    checkpoint: str | None = None
    finished: bool = False
    elapsed_s: float = 0.0
    extra: dict = field(default_factory=dict)


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CellRefused(f"attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            raise CellRefused(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise CellRefused(f"import {a.name!r} is not allowed (allowed: {sorted(ALLOWED_IMPORTS)})")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise CellRefused(f"import from {node.module!r} is not allowed")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    head = s[: n * 2 // 3]
    tail = s[-(n // 3):]
    return f"{head}\n... [{len(s) - len(head) - len(tail)} characters cut] ...\n{tail}"


class Session:
    def __init__(self, namespace: dict | None = None, max_output: int = 4000):
        self.max_output = max_output
        self._buf: list[str] = []
        safe = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS and not k.startswith("_")}
        safe["__import__"] = _safe_import
        safe["print"] = self._print
        safe["__build_class__"] = builtins.__build_class__
        self.g: dict = {"__builtins__": safe, "__name__": "cell", "np": np, "math": math}
        self.g.update(namespace or {})
        self.n_cells = 0

    def _print(self, *args, sep=" ", end="\n", **_kw):
        self._buf.append(sep.join(str(a) for a in args) + end)

    def run(self, code: str) -> CellResult:
        self.n_cells += 1
        self._buf = []
        res = CellResult()
        t0 = time.perf_counter()
        try:
            tree = ast.parse(code, filename=f"<cell {self.n_cells}>", mode="exec")
            _check_ast(tree)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, f"<cell {self.n_cells}>", "exec"), self.g)  # noqa: S102
            if last is not None:
                v = eval(compile(last, f"<cell {self.n_cells}>", "eval"), self.g)  # noqa: S307
                if v is not None:
                    res.value = _clip(repr(v), self.max_output // 2)
        except CellCheckpoint as e:
            res.checkpoint = str(e) or "checkpoint"
        except EpisodeFinished as e:
            res.finished = True
            res.extra["finish"] = str(e)
        except CellRefused as e:
            res.error = f"refused: {e}"
        except SyntaxError as e:
            res.error = f"SyntaxError: {e.msg} (line {e.lineno})"
        except Exception as e:  # noqa: BLE001 (the model's error, reported back)
            tb = traceback.extract_tb(e.__traceback__)
            frames = [f for f in tb if f.filename.startswith("<cell")]
            where = f" (line {frames[-1].lineno})" if frames else ""
            res.error = f"{type(e).__name__}: {e}{where}"
        res.stdout = _clip("".join(self._buf), self.max_output)
        res.elapsed_s = time.perf_counter() - t0
        return res


__all__ = ["Session", "CellResult", "CellCheckpoint", "EpisodeFinished", "ALLOWED_IMPORTS"]
