import collections,datetime,fcntl,hashlib,json,os,re,statistics,sys
from pathlib import Path
H=Path.home();R=H/'dojo-data/match2-cam';S=H/'dojo-integ-c6-state';Q=S/'queue2.txt';N=20
def pair(name):
 m=re.match(r'^(c6b_[^_]+_tie)([AB])([^_]+)_([0-9]{6})(?:(?:[.]|__retry).*)?$',name)
 if '_tie' in name:assert m,'Unknown tie identity '+name
 return (m[1],m[3],int(m[4])) if m else None
def position(lines,started):
 protected=[i for i,l in enumerate(lines) if pair(l.split()[0]) in started and pair(l.split()[0]) is not None]
 floor=max(protected,default=-1)+1
 return next((i for i in range(floor,len(lines)) if pair(lines[i].split()[0]) is not None),len(lines)),protected
if '--self-test' in sys.argv:
 t=lambda a,r,i:f'c6b_imitatesg_tie{a}{r}_{i} args\n'
 assert position([t('A','r1pb1',100001),t('B','r1pb1',100001)],set())[0]==0
 st={pair(t('A','r1pb1',100000).split()[0])}
 assert position([t('B','r1pb1',100000),t('A','r1pb1',100001)],st)[0]==1
 assert position([t('A','r2pb1',100000),t('B','r2pb1',100000)],st)[0]==0
 assert pair('c6b_imitatesg_tieBr1pb1_100000__retry')==pair(t('A','r1pb1',100000).split()[0])
 assert pair('c6b_matchpicksg_m2scrAr1_100000') is None
 print('Insertion tests pass');raise SystemExit
D=json.loads((R/'design.json').read_text());M=json.loads((R/'manifest.json').read_text());new=(R/'queue-match2-screen.txt').read_bytes();newrows=new.decode().splitlines(keepends=True)
assert not (R/'dispatch.json').exists() and len(newrows)==N
assert hashlib.sha256(new).hexdigest()==M['queue']['sha256']
assert (R/'prereg-commit.txt').exists(),'preregistration must be on main first'
for line,m in zip(newrows,M['schedule']):
 t=line.split();assert t[0]==m['name'] and t[0].startswith('c6b_') and pair(t[0]) is None
 assert t[t.index('--repo')+1]==D['deployment'] and t[t.index('--layouts')+1]==m['layout']
 assert [t[i+1] for i,z in enumerate(t[:-1]) if z=='--setenv' and t[i+1].startswith('AGENTIC_MODEL=')]==['AGENTIC_MODEL='+D['model']]
 cfg=json.loads(t[t.index('--config')+1]);over=D['A_overrides'] if m['arm']=='A' else D['B_overrides']
 assert all(cfg.get(k)==v for k,v in over.items())
 if m['arm']=='A':assert all(k not in cfg for k in D['B_overrides'] if k not in D['A_overrides'])
 assert hashlib.sha256(Path(m['layout']).read_bytes()).hexdigest()==next(f['sha256'] for f in M['files'] if f['path']==m['layout'])
with (S/'wave-start.lock').open('a') as admission:
 fcntl.flock(admission,fcntl.LOCK_EX)
 with (S/'queue2.txt.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  starts={};dones={};samples=collections.defaultdict(list)
  for p in S.glob('lane2_heavy*.log'):
   for l in p.read_text().splitlines():
    m=re.match(r'^(\S+) \[([^]]+)\] start (\S+)',l)
    if m:starts[m[3]]={'time':m[1],'lane':m[2]}
    m=re.match(r'^(\S+) \[([^]]+)\] done (\S+) rc=(\d+) lock (\d+) s',l)
    if m:
     dones[m[3]]=m[1]
     if int(m[4])==0:samples[m[3].split('_')[1]].append(int(m[5]))
  names={m['name'] for m in M['schedule']};assert not names.intersection(starts)
  before=Q.read_bytes();assert not before or before.endswith(b'\n');existing=before.decode().splitlines(keepends=True)
  assert not names.intersection(l.split()[0] for l in existing)
  started={pair(n) for n in starts if pair(n) is not None};at,protected=position(existing,started)
  assert all(i<at for i in protected)
  if at<len(existing):assert pair(existing[at].split()[0]) is not None and pair(existing[at].split()[0]) not in started
  after=b''.join(l.encode() for l in existing[:at])+new+b''.join(l.encode() for l in existing[at:])
  assert [l for l in after.decode().splitlines(keepends=True) if l.split()[0] not in names]==existing
  (R/'queue-before.txt').write_bytes(before)
  tmp=S/'queue2.match2.tmp';tmp.write_bytes(after)
  with tmp.open('rb') as f:os.fsync(f.fileno())
  os.replace(tmp,Q);(R/'queue-after.txt').write_bytes(after)
  now=datetime.datetime.now(datetime.timezone.utc);means={k:statistics.mean(v[-20:]) for k,v in samples.items() if v}
  active=[];wait=sum(means.get(l.split()[0].split('_')[1],600) for l in existing[:at])
  for p in S.glob('job.heavy*'):
   n=p.read_text().strip()
   if n in starts and (n not in dones or starts[n]['time']>dones[n]):
    el=(now-datetime.datetime.fromisoformat(starts[n]['time'].replace('Z','+00:00'))).total_seconds();rem=max(0,means.get(n.split('_')[1],900)-el)
    wait+=rem;active.append({'name':n,'elapsed_s':round(el),'estimated_remaining_s':round(rem)})
  block=N*means.get('matchpicksg',420)
  receipt={'inserted_utc':now.isoformat(),'prereg_commit':(R/'prereg-commit.txt').read_text().strip(),'positions_1_based':[at+1,at+N],'new_rows':N,
   'existing_rows':len(existing),'queue_rows_after':len(existing)+N,'queue_sha256_before':hashlib.sha256(before).hexdigest(),
   'queue_sha256_after':hashlib.sha256(after).hexdigest(),'manifest_sha256':hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest(),
   'new_rows_sha256':hashlib.sha256(new).hexdigest(),'preserved_old_bytes_and_relative_order':True,
   'protected_started_pair_rows':[existing[i].split()[0] for i in protected],'preserved_prefix_rows':[l.split()[0] for l in existing[:at]],
   'next_unstarted_tie':existing[at].split()[0] if at<len(existing) else None,'active_jobs':active,'route_checked_rows':N,'model':D['model'],
   'deployment':D['code'],'estimated_start_utc':(now+datetime.timedelta(seconds=wait/2)).isoformat(),
   'estimated_finish_utc':(now+datetime.timedelta(seconds=(wait+block)/2)).isoformat(),'estimated_block_lane_s':round(block),
   'timing_basis':{'recent_mean_seconds':means,'fallback_matchpicksg_s':420,'lanes':2,'note':'rough; admission/startup/time caps/retries may delay'}}
  (R/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
  with (R/'notes.log').open('a') as f:f.write(now.isoformat()+' DISPATCH '+json.dumps(receipt)+'\n')
  print(json.dumps(receipt,indent=2))
