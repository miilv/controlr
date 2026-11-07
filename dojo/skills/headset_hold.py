"""Optional loaded proof for a headset that settled during its initial lift."""
from __future__ import annotations

import numpy as np

from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import angle_between_R, inv_T
from dojo.skills.kinematics import TABLE_Z


def loaded_hold(sk, pp, result, before):
    """Prove a loose hold by following a checked lift, without changing slip limits.

    Reads must reflect physical object motion. A cached object pose cannot pass
    the following test. Preserve the original nominal-drift verdict in the log.
    """
    from dojo.skills.skills import CLOSED, SkillResult

    if not result.ok or result.status != "loose" or sk.world.obj(pp.label).category != "headset":
        return result
    if sk.ex.steps_left - sk.ex.reserve < 100 or sk.held.get(pp.arm) is None:
        return result
    samples = []
    phase = "stationary"
    observer = sk.ex.on_step

    def read():
        w = sk.refresh()
        o = w.obj(pp.label)
        if o.asset.mesh is None:
            return None
        try:
            vertices = np.asarray(o.asset.mesh[0], float)
        except (TypeError, ValueError):
            return None
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or not np.isfinite(o.T).all()):
            return None
        tcp = sk.kin.tcp(pp.arm, w.arms[pp.arm].q)
        return inv_T(tcp) @ o.T, float((vertices @ o.R.T + o.pos)[:, 2].min()), float(o.pos[2])

    def stable(value, reference):
        return (np.linalg.norm(value[0][:3, 3] - reference[0][:3, 3]) <= .003
                and angle_between_R(value[0][:3, :3], reference[0][:3, :3]) <= np.deg2rad(3))

    def capture(ex):
        if observer is not None:
            observer(ex)
        samples.append((phase, read()))

    initial = read()
    if initial is None or initial[1] < TABLE_Z + .003:
        return result
    try:
        sk.ex.on_step = capture
        sk.ex.hold(30, name="headset_loaded_settle")
        reference = read()
        if (reference is None or reference[1] < TABLE_Z + .003
                or any(v is None or not stable(v, reference) for _, v in samples)):
            return result
        checked = sk.verify_pick(pp, before)  # same original rise/30 mm hard-slip gates
        if not checked.ok:
            return checked
        held = sk.held[pp.arm]
        q = sk.world.arms[pp.arm].q
        tcp = sk.kin.tcp(pp.arm, q)
        raised = tcp.copy()
        raised[2, 3] += .02
        up = sk.motion.line(pp.arm, q, raised, .02, "headset_load_up", gripper=CLOSED,
                            held=held.box, ignore=(pp.label,), check=True)
        if not up.ok:
            return result
        down = sk.motion.line(pp.arm, up.q_end, tcp, .02, "headset_load_down", gripper=CLOSED,
                              held=held.box, ignore=(pp.label,), check=True)
        if not down.ok:
            return result
        phase = "challenge"
        plan = ArmPlan(pp.arm, q, CLOSED).move(up, "up").hold(8, "loaded").move(down, "down").hold(8, "returned")
        sk.ex.run({pp.arm: plan}, name="headset_loaded_proof")
        challenged = [v for ph, v in samples if ph == "challenge"]
        if not challenged or any(v is None or not stable(v, reference) for v in challenged):
            return result
        rise = max(v[2] for v in challenged) - reference[2]
        clearance = max(v[1] for v in challenged) - TABLE_Z
        if clearance < .02 or rise < .017:
            return result
        final = sk.verify_pick(pp, before)
        if not final.ok:
            return final
        drift = max(float(np.linalg.norm(v[0][:3, 3] - reference[0][:3, 3])) for v in challenged)
        turn = max(float(np.rad2deg(angle_between_R(v[0][:3, :3], reference[0][:3, :3]))) for v in challenged)
        return SkillResult(True, "pick", "ok", result.steps + 30 + plan.steps, pp.arm, final.expected, final.measured,
                           result.note + f"; loaded proof: rise {rise*1000:.1f} mm, clearance {clearance*1000:.1f} mm, "
                           f"relative {drift*1000:.2f} mm/{turn:.2f} deg; final nominal: {final.note}")
    except (BudgetExceeded, EpisodeEnded) as exc:
        return SkillResult(False, "pick", "step_budget" if isinstance(exc, BudgetExceeded) else "ended",
                           0, pp.arm, note=str(exc))
    finally:
        sk.ex.on_step = observer
