"""Named, versioned RuntimeConfig presets. An eval or a bridge episode names one (``--preset submit_v1``) instead of
repeating its fields; ``--config`` fields override it, and the name is in every report (``RuntimeConfig.preset``).

A preset never changes once it is used: a new configuration is a new name (``submit_v2``), with its evidence in the
journal. The code defaults of ``RuntimeConfig`` stay as they are (rule 1: no silent default changes).

``submit_v1`` (2026-10-26; evidence: journal 2026-10-25-dojo-integ2 §10):
- closed2: refine (``refine_max_shift_m`` 0.07, ``max_rehovers`` 2), post_place, search;
- place_confirm, watch;
- the episode-start scan when perception's trigger asks (``scan: trigger``);
- agentic perceiver on budget (at most 3 estimates), held_look ``b1_then_agent``, tracker ``auto``;
- ``search_lost`` off, ``held_pose`` off;
- perception side (the service's environment, not a RuntimeConfig field): the agentic estimate snapped to B1
  (``AGENTIC_SNAP`` unset or 1, perception's default).

``submit_v2`` (2026-10-27; evidence: journal 2026-10-27-dojo-pegs §9, the rendered A/B): ``submit_v1`` with
``held_pose`` "request". Only a place that opts in (``held_read``) reads the held object; today just insert_key's, so
every other task runs as under ``submit_v1``. insert_key genuine 5.1, 100000-09: 1/6 vs 0/6 (rescued 1, lost 0).

``submit_v3`` (2026-10-27, candidate; the lead's call, evidence to come from the paired breadth3 A/B against round 3):
``submit_v2`` with the pick / place refine searching a 7 cm grid when the prior's own fit is not ok
(``refine_search_radius`` 0.07, #183 / #184) and pick refines telling perception the hands are empty
(``refine_empty_hand``). D0 5.1 wrist looks with priors +-50 mm: correct fits 187 -> 315 of 334, no new false ok.
"""

from __future__ import annotations

import copy
import os

PRESETS: dict[str, dict] = {
    "submit_v1": {
        "mode": "perception", "verify": "bundle",
        "refine": True, "refine_max_shift_m": 0.07, "max_rehovers": 2, "post_place": True, "search": True,
        "search_lost": False, "place_confirm": True, "watch": True,
        "scan": "trigger",
        "agentic": "budget", "agentic_max": 3, "held_look": "b1_then_agent", "tracker": "auto",
        "held_pose": "off",
    },
}
PRESETS["submit_v2"] = {**PRESETS["submit_v1"], "held_pose": "request"}
PRESETS["submit_v3"] = {**PRESETS["submit_v2"], "refine_search_radius": 0.07, "refine_empty_hand": True}

# perception-side settings a preset assumes (environment of the perception service / agentic perceiver)
PERCEPTION_ENV: dict[str, dict[str, str]] = {
    "submit_v1": {"AGENTIC_SNAP": "1"},
    "submit_v2": {"AGENTIC_SNAP": "1"},
    "submit_v3": {"AGENTIC_SNAP": "1"},
}


def preset(name: str) -> dict:
    """A copy of preset ``name``'s RuntimeConfig fields."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


def check_env(name: str, env=None) -> list[str]:
    """Perception-side settings of ``name`` that this environment contradicts (an unset variable takes perception's
    default, which the preset assumes)."""
    env = os.environ if env is None else env
    out = []
    for k, v in PERCEPTION_ENV.get(name, {}).items():
        if k in env and str(env[k]).strip().lower() != str(v).lower() and not (
                v == "1" and str(env[k]).strip().lower() in ("1", "true", "yes")):
            out.append(f"{k}={env[k]!r} (preset {name} assumes {v})")
    return out


def resolve(name: str | None, overrides: dict | None = None, mode: str | None = None, verify: str | None = None) -> dict:
    """RuntimeConfig fields: the preset (if any), then ``overrides`` (``--config``), then explicit ``mode`` / ``verify``
    (command-line flags given explicitly). With a preset, ``preset`` names it."""
    out = preset(name) if name else {}
    out.update(overrides or {})
    if mode is not None:
        out["mode"] = mode
    if verify is not None:
        out["verify"] = verify
    if name:
        out["preset"] = name
    return out


__all__ = ["PRESETS", "PERCEPTION_ENV", "preset", "resolve", "check_env"]
