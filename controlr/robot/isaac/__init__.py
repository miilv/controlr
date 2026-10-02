"""Isaac Sim 6.0 backend (PHANTOM's calibrated UR3 CB3 + Robotiq + D435 rig).

``server.py`` runs inside Isaac's python; everything else here is plain
python + numpy. ``IsaacRobot`` is exported lazily so importing this package
(e.g. for ``tasks`` in the server process) never pulls in the client side.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "IsaacRobot":
        from controlr.robot.isaac.client import IsaacRobot
        return IsaacRobot
    raise AttributeError(name)


__all__ = ["IsaacRobot"]
