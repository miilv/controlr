"""Per-task expert programs. Each module exposes ``run(sk: Skills) -> dict`` and ``TASK``.

A program reads only the WorldState (``sk.world`` / ``sk.refresh()``), the instruction and the
proprio; under G3 the world comes from dev ground truth, later from perception.
"""

from importlib import import_module

PROGRAMS = {
    "stack_blocks": "dojo.skills.programs.stack_blocks",
    "build_tower": "dojo.skills.programs.build_tower",
    "stack_bowls": "dojo.skills.programs.stack_bowls",
    "put_bottles_into_dustbin": "dojo.skills.programs.put_bottles",
    "swap_blocks": "dojo.skills.programs.swap_blocks",
    "press_by_number": "dojo.skills.programs.press_by_number",
}


def get(task: str):
    return import_module(PROGRAMS[task])
