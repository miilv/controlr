"""Robot backend interface.

A backend owns the physical/simulated robot, its cameras and the task scene.
The turn loop is strictly: observe -> (model thinks; robot holds) -> execute
-> settle -> observe. Backends never move while the model is thinking.

Safety is enforced in two layers:
  * ``controlr.robot.safety.SafetyEnvelope`` (backend-independent) filters
    every Action before it reaches ``execute`` and reports clamps as events;
  * the backend itself reports what physically happened (collisions, IK
    failure, contact forces) as SafetyEvents in the ExecReport.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from controlr.types import Action, ExecReport, GoalReport, Observation, RobotSpec, RobotState


class Robot(ABC):
    spec: RobotSpec

    @abstractmethod
    def reset(self, task_cfg: dict, seed: int | None = None) -> Observation:
        """Reset robot + scene for a task (sim: rebuild/randomise; real: move home)."""

    @abstractmethod
    def observe(self) -> Observation:
        """Capture all configured cameras + state. Must not move the robot."""

    @abstractmethod
    def state(self) -> RobotState:
        """Current proprioceptive state (cheap, no rendering)."""

    @abstractmethod
    def execute(self, actions: list[Action]) -> ExecReport:
        """Execute already safety-filtered actions in order, block until the
        robot has settled (or a STOP event), and report what happened.
        Actions are in SI units, base frame (see controlr.types)."""

    @abstractmethod
    def check_goal(self) -> GoalReport:
        """Task success / progress from privileged state (sim) or a checker (real)."""

    def reference_state(self) -> RobotState | None:
        """The commanded (nominal) reset state, if the backend knows it — e.g. FK of the
        commanded start joints, free of settle jitter/gravity sag. Used for the fixed tool
        orientation (rotation=none) and the manual's tool description, so they are
        identical across episodes. Default None: the loop uses the measured reset state."""
        return None

    def hold(self) -> None:
        """Keep the current pose (default: nothing to do for position-controlled robots)."""

    def close(self) -> None:
        pass
