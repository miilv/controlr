import json
from pathlib import Path
import numpy as np
from dojo.skills.env import KinematicEnv
from dojo.skills.skills import Skills, GTProvider
from dojo.skills.world import AssetDB
from dojo.skills import holders as H
from dojo.skills.egg_reach import practical_limits,grasp_candidates
from dojo.simlab import paths
root=Path.home()/'dojo-data/egg2'; result=[]; assets=AssetDB(paths.OBJECTS)
(root/'geometry-endpoints.jsonl').write_text('')
for lp in sorted((root/'dev10').glob('*.settled.json')):
 lay=json.loads(lp.read_text());rows=[]
 for typ in ['Geometry','Rigid','Articulation']:
  for group in (lay.get(typ) or {}).values():
   for r in group:rows.append(dict(label=r['label'],type=typ,category=r['category'],category_idx=r['category_idx'],pos=r['default_pos'],quat=r['default_ori'],clutter=False))
 env=KinematicEnv(rows);logs=[];sk=Skills(env,GTProvider(env,assets),log=logs.append);sk.egg_reach='search'
 with practical_limits(sk.kin):
  for lab in ['target0','target1','target2','target3']:
   c=H.container_of(sk,lab);prof=H.wall_profile(c) if c is not None else None
   lift_z=float((c.R@np.r_[prof['center'],prof['rim']]+c.pos)[2])+H.EGG_LIFT_OVER if prof else None
   for arm in ['left','right']:
    raw=H.round_grasps(sk.world.obj(lab),sk.kin,arm,rolls=(0,12,-12,25,-25,40,-40,55,-55),tilts=(0,20,-20,40,-40,55,-55),ends=False)
    raw_ok=grasp_candidates(sk,lab,arm,raw,lift_z)
    wall=H.egg_grasps(sk,lab,arm);wall_ok=grasp_candidates(sk,lab,arm,wall,lift_z)
    result.append(dict(seed=int(lp.name.split('.')[0]),label=lab,arm=arm,raw=len(raw),raw_ik=len(raw_ok),wall=len(wall),wall_ik=len(wall_ok)))
 (root/'geometry-summary.json').write_text(json.dumps(result,indent=2))
 print(lp.name,result[-8:],flush=True)
 with (root/'geometry-endpoints.jsonl').open('a') as f:
  for x in logs:
   if x.startswith('[egg-reach] '):f.write(x[12:]+'\n')
