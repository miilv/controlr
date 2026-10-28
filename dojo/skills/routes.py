"""Planner routes that are off by default (stream N): each is asked by ``planner.plan`` before the built-in rules, only
when its name is listed in the provider's ``notes["routes"]`` (``RuntimeConfig.new_routes``, G3 ``--routes``). With no
name listed the planner is unchanged. A route reads only the instruction text and the WorldState, like every rule.

| name | skill | module |
|---|---|---|
| equation | complete_equation | ``dojo.skills.tiles`` |
| largest_number | arrange_digits | ``dojo.skills.tiles`` |
| align | align_in_row | ``dojo.skills.align`` |
"""

from __future__ import annotations

NEW_ROUTES = ("equation", "largest_number", "align")


def route(text: str, world, routes) -> tuple | None:
    """(skill, args, why) from the first listed route that handles the instruction, else None."""
    routes = set(routes or ()) & set(NEW_ROUTES)
    if not routes:
        return None
    from dojo.skills import align, tiles
    for mod in (tiles, align):
        r = mod.route(text, world, routes)
        if r is not None:
            return r
    return None
