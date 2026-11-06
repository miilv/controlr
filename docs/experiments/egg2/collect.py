import json,re,statistics,hashlib
from pathlib import Path
root=Path.home()/'dojo-data/egg2'
out=[]
for p in sorted(root.glob('*/fill_egg_holder_episodes.jsonl')):
 mode='on' if p.parent.name=='pilot-on' or p.parent.name.endswith('-on') else 'off'
 logs=(root/'logs'/(p.parent.name+'.log')).read_text(errors='replace')
 for r in map(json.loads,p.read_text().splitlines()):
  rows=[]
  for line in r.get('log',[]):
   if line.startswith('[egg-reach] '):rows.append(json.loads(line[12:]))
  screen_queries=screen_failed=0
  for phase in rows:
   for v in phase['rows']:
    screen_queries+=len(v['distance_m'])-(v.get('reason')=='reach')
    screen_failed+=v.get('reason')=='ik'
  matches=re.findall(r'\[egg2-ik\] (\{[^\n]+\})',logs)
  counts=json.loads(matches[-1]) if matches else None
  item={k:r[k] for k in ['seed','success','score','stable','steps','step_lim','wall_s','plan_s','category','result','skills','code']}
  item.update(mode=mode,run=p.parent.name,all_ik=counts,screen_ik=dict(queries=screen_queries,failed=screen_failed),screen_phases=[{k:v for k,v in phase.items() if k!='rows'} for phase in rows],holes_log=[x for x in r.get('log',[]) if x.startswith('[holes]')])
  if counts:item['planner_ik']={k:counts[k]-dict(queries=screen_queries,failed=screen_failed)[k] for k in counts}
  out.append(item)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'dev10').glob('*.settled.json'))}
result=dict(controller='8be287c3',layout_sha256=manifest,episodes=out)
(root/'results.json').write_text(json.dumps(result,indent=2))
for mode in ['off','on']:
 a=[r for r in out if r['mode']==mode];print(mode,len(a),'score',statistics.mean(r['score'] for r in a) if a else None)
for r in out:print(r['seed'],r['mode'],r['score'],r['steps'],r['planner_ik'])
