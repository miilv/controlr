"""Test-only runtime tripwires; never imported by the deployed controller."""
from contextlib import contextmanager
import builtins
import io
from pathlib import Path
import sys
import threading

from scripts.legality.scan import SCHEMA_NAMES


class ForbiddenRead(AssertionError):
    pass


class TaskIdentity:
    """May cross a metadata boundary, but cannot be inspected or used as a key."""
    def _read(self, *args, **kwargs):
        raise ForbiddenRead('task identity was inspected')
    __str__ = __repr__ = __bool__ = __hash__ = __eq__ = __getitem__ = _read
    __getattr__ = __iter__ = __len__ = __contains__ = __format__ = _read


@contextmanager
def instruction_only(monkeypatch):
    """Catch function calls even through pre-imported aliases, and file opens.

    The sticky hits list also catches errors swallowed by worker/session code.
    Does not import optional perception dependencies or call network services.
    """
    hits = []
    def refuse(what):
        hits.append(what)
        raise ForbiddenRead(what)
    def profile(frame, event, arg):
        if event == 'call':
            code = frame.f_code
            filename = code.co_filename.replace('\\', '/')
            if '/dojo/' in filename and (code.co_name in SCHEMA_NAMES or
                    '/dojo/perception/tasks.py' in filename or
                    code.co_name == '_template' and '/dojo/runtime/estimate.py' in filename):
                refuse(filename + ':' + code.co_name)
    def wrap(opener):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                path = str(file).replace('\\', '/')
                if path.endswith('tasks.json') or '/task/RoboDojo/config/' in path or '/task/RoboDojo/tasks/' in path:
                    refuse('task data file: ' + path)
            return opener(file, *args, **kwargs)
        return guarded
    monkeypatch.setattr(builtins, 'open', wrap(builtins.open))
    monkeypatch.setattr(io, 'open', wrap(io.open))
    # Empty cached loader results before profiling; otherwise an old cached schema
    # could cross the boundary without entering its Python function body.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith('dojo.'):
            for name, value in list(vars(module).items()):
                if name in SCHEMA_NAMES and callable(getattr(value, 'cache_clear', None)):
                    value.cache_clear()
    old, old_thread = sys.getprofile(), threading.getprofile()
    sys.setprofile(profile)
    threading.setprofile(profile)
    try:
        yield hits
    finally:
        sys.setprofile(old)
        threading.setprofile(old_thread)
