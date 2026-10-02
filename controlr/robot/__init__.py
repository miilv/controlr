"""Robot backends. ``make_robot(cfg)`` is the only constructor the loop uses.

The Isaac backend is imported lazily: its client is light, but importing it
must never be a side effect of using the mock / replay backends in tests.
"""

from __future__ import annotations

from controlr.robot.base import Robot


def make_robot(cfg) -> Robot:
    """Build the backend named by ``cfg.robot.backend`` with ``cfg.robot.params``."""
    backend = cfg.robot.backend
    params = dict(cfg.robot.params or {})
    if backend == "mock":
        from controlr.robot.mock import MockRobot
        return MockRobot(params)
    if backend == "replay":
        from controlr.robot.replay import ReplayRobot
        return ReplayRobot(params)
    if backend == "isaac":
        # ARCHITECTURE names ``controlr.robot.isaac.IsaacRobot``; fall back to the
        # client submodule in case the package __init__ does not re-export it.
        import importlib
        pkg = importlib.import_module("controlr.robot.isaac")
        robot_cls = getattr(pkg, "IsaacRobot", None)
        if robot_cls is None:
            robot_cls = importlib.import_module("controlr.robot.isaac.client").IsaacRobot
        if hasattr(robot_cls, "from_config"):
            return robot_cls.from_config(cfg)
        return robot_cls(params)
    raise ValueError(f"unknown robot backend {backend!r} (mock | replay | isaac)")


__all__ = ["Robot", "make_robot"]
