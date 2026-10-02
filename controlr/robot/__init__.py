"""Robot backends. ``make_robot(cfg)`` is the only constructor the loop uses.

The Isaac backend is imported lazily: its client is light, but importing it
must never be a side effect of using the mock / replay backends in tests.
"""

from __future__ import annotations

from controlr.robot.base import Robot


def make_robot(cfg) -> Robot:
    """Build the backend named by ``cfg.robot.backend`` with ``cfg.robot.params``
    (backends that run their own fallback envelope get ``cfg.safety``)."""
    backend = cfg.robot.backend
    params = dict(cfg.robot.params or {})
    if backend == "mock":
        from controlr.robot.mock import MockRobot
        return MockRobot(params, safety=cfg.safety)
    if backend == "replay":
        from controlr.robot.replay import ReplayRobot
        return ReplayRobot(params)
    if backend == "isaac":
        from controlr.robot.isaac.client import IsaacRobot
        return IsaacRobot.from_config(cfg)
    raise ValueError(f"unknown robot backend {backend!r} (mock | replay | isaac)")


__all__ = ["Robot", "make_robot"]
