"""Dev probe (``g3 snippet``): run fold_garment with the garment's named points logged after every phase."""
# names in scope: sk, env, run, np, lay, rebuild
from dojo.skills import garment as GM

read0 = sk.provider.garment


def read(label=None):
    g = read0(label)
    kp = g["keypoints"]
    print("[fold] read step %d: " % sk.ex.steps_used + " ".join(
        f"{k} {np.round(kp[k], 3).tolist()}" for k in ("left_sleeve", "right_sleeve", "left_chest", "right_chest",
                                                     "left_hem", "right_hem", "left_shoulder", "right_shoulder")),
          flush=True)
    return g


sk.provider.garment = read
orig_lay = GM.plan_lay


def plan_lay(sk_, arm, q0, dst, yaw, carry_z=GM.CARRY_Z):
    r = orig_lay(sk_, arm, q0, dst, yaw, carry_z)
    print(f"[fold] plan_lay {arm} dst {np.round(dst, 3).tolist()} -> {'plan' if not isinstance(r, str) else r}",
          flush=True)
    return r


GM.plan_lay = plan_lay
w = sk.refresh()
res = GM.fold_garment(sk, [o.label for o in w.objects.values() if o.obj_type == "Garment"][0])
print("[fold] result", res, flush=True)
for h in sk.history:
    print("[fold] hist", h.skill, h.status, h.steps, h.arm, h.note[:200], flush=True)
f = sk.finish()
for _ in range(10):
    run.step(run.hold_action())
print("[fold] reward", run.reward(), "score", run.score(), flush=True)
