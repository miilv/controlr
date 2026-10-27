"""Dev probe (``g3 snippet``, sweep_blocks): which broom grasps lift it? Tries handle grasps and the default candidates
with the nearer arm, one per fresh load; prints the rise and the in-hand drift."""
# names in scope: sk, env, run, np, lay, rebuild
from dojo.skills import grasps as G
from dojo.skills import sweep as SW
from dojo.skills.skills import SkillResult

w = sk.refresh()
o = w.obj("broom")
arm = min(("left", "right"), key=lambda a: float(np.linalg.norm(sk.kin.base(a)[:2, 3] - o.pos[:2])))
print(f"[broom] pos {np.round(o.pos, 3).tolist()} arm {arm}", flush=True)
cands = [("handle", g) for g in SW.handle_grasps(sk, o, arm)[:6]]
cands += [("default", g) for g in (G.box_grasps(o, sk.kin, arm) + G.candidates(o, sk.kin, arm)
                                   + G.slice_grasps(o, sk.kin, arm))[:40:5]]
for kind, g in cands:
    sk2 = rebuild(lay)
    o2 = sk2.refresh().obj("broom")
    p_l = (np.linalg.inv(o2.T) @ np.r_[g.T[:3, 3], 1.0])[:3]
    pp = sk2.plan_pick("broom", arm, grasps=[g], keep=1, lift=0.06)
    if isinstance(pp, SkillResult):
        print(f"[broom] {kind} {g.method} {g.note[:40]} local {np.round(p_l, 3).tolist()}: no plan ({pp.note[:60]})", flush=True)
        continue
    r = sk2.run_pick(pp)
    print(f"[broom] {kind} {g.method} {g.note[:40]} w {g.width * 1000:.0f} local {np.round(p_l, 3).tolist()}: "
          f"{r.status} {r.note[:90]}", flush=True)
