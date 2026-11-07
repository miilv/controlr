"""MATCH2 screen: stage design.json, the 20 queue rows and manifest.json (inputs + hashes). No queue write here."""
import hashlib,json,sys
from pathlib import Path
H=Path.home();R=H/'dojo-data/match2-cam';SHA=sys.argv[1];REPO=H/f'dojo-integ-c6-{SHA[:8]}'
assert (REPO/'.deployed_commit').read_text().strip()==SHA
sys.path.insert(0,str(REPO))
from dojo.simlab.paths import is_our_measurement_id
LD=R/'screen1/simlab/layouts/match_and_pick_from_conveyor';IDS=list(range(100000,100010))
assert all(is_our_measurement_id(i) for i in IDS)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cand=[l for l in (R/'candidate.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]
assert len(cand)==3 and cand[2].startswith('--config ')
base=json.loads(cand[2][len('--config '):])
assert base.get('belt_wrist_veto')=='on' and base.get('save_look_frames') is True
MODEL='claude/claude-sonnet-5-5'
A_over={'save_belt_look_frames':6}
B_over={**A_over,'belt_ref':'tracked','belt_veto':'evidence','belt_speed':'lock','belt_high_lift':'on','belt_retry':'window','belt_wrist_id':'on'}
for k in B_over:assert k not in base,k
rows=[];sched=[]
for k,lid in enumerate(IDS):
 for arm in (('A','B') if k%2==0 else ('B','A')):
  cfg={**base,**(A_over if arm=='A' else B_over)}
  name=f'c6b_matchpicksg_m2scr{arm}r1_{lid}';lay=str(LD/f'{lid}.json')
  row=' '.join([name,'--repo',str(REPO),cand[0],cand[1],'--config',json.dumps(cfg,separators=(',',':')),
   '--setenv','AGENTIC_MODEL='+MODEL,'--setenv','DOJO_DATA='+str(R/'screen1'),'--task','match_and_pick_from_conveyor',
   '--renderer','isaac51','--episode-max-minutes','25','--max-minutes','25','--layouts',lay])
  rows.append(row+'\n');sched.append({'name':name,'arm':arm,'id':lid,'pair_ordinal':k,'layout':lay})
q=''.join(rows).encode();(R/'queue-match2-screen.txt').write_bytes(q)
design={'stream':'MATCH2-SCREEN','role':'diagnostic screen, not an adoption test','code':SHA,'deployment':str(REPO),
 'baseline_candidate':'scripts/dojo/queues/2026-11-06-candidate.txt','baseline_candidate_sha256':sha(REPO/'scripts/dojo/queues/2026-11-06-candidate.txt'),
 'candidate_robotics_sha256':sha(R/'candidate.txt'),'model':MODEL,'task':'match_and_pick_from_conveyor','ids':IDS,
 'salt':'match2-screen-20261107-r1','salt_sha256':hashlib.sha256(b'match2-screen-20261107-r1').hexdigest(),
 'A_overrides':A_over,'B_overrides':B_over,'order':'AB for even pair ordinals, BA for odd','repeats_per_arm':1}
(R/'design.json').write_text(json.dumps(design,indent=2)+'\n')
files=[{'path':str(LD/f'{i}.json'),'sha256':sha(LD/f'{i}.json')} for i in IDS]
man={'design_sha256':sha(R/'design.json'),'queue':{'path':str(R/'queue-match2-screen.txt'),'sha256':hashlib.sha256(q).hexdigest(),'rows':len(rows)},
 'schedule':sched,'files':files,'raw_index_compute3_sha256':sha(R/'screen1-c3-index/raw_index.compute3.json'),
 'raw_index_compute6_sha256':sha(LD/'raw_index.json'),'parity':json.loads((R/'parity.json').read_text()).get('diff'),
 'deployed_json_sha256':sha(REPO/'.deployed.json')}
(R/'manifest.json').write_text(json.dumps(man,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'queue_sha256':man['queue']['sha256'],'manifest_sha256':sha(R/'manifest.json')}))
