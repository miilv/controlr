"""MATCH2 screen: re-run the unchanged raw-check of the copied screen block on compute6 (admission + one free light
lock + scoped). Physics-only, zero model calls. The conveyor runs its RAW record (settled: false): raw-check is its
presettle stability test; no settle pass exists for it."""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
H=Path.home();R=H/'dojo-data/match2-cam';P=Path(sys.argv[1]).resolve();sys.path.insert(0,str(P))
from controlr.keyguard import child_env
from dojo.simlab.paths import is_our_measurement_id
B=R/'screen1';LD=B/'simlab/layouts/match_and_pick_from_conveyor';IDS=list(range(100000,100010))
assert all(is_our_measurement_id(i) for i in IDS)
def note(s):
 with (R/'notes.log').open('a') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+s+'\n')
want={l.split()[1]:l.split()[0] for l in (R/'screen1.sha256').read_text().splitlines()}
for n,h in want.items():
 if n!='raw_index.json':assert hashlib.sha256((LD/n).read_bytes()).hexdigest()==h,n
lock=None
while lock is None:
 used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 if used<=20480 and os.getloadavg()[0]<=26:
  for name in ['dojo-gpu-light.lock']+[f'dojo-gpu-light{i}.lock' for i in range(1,9)]:
   p=H/name
   if not p.exists() or p.is_symlink():continue
   f=p.open('r')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:f.close();continue
   lock=f;break
 if lock is None:time.sleep(20)
assert int(subprocess.check_output(['df','--output=avail','-B1',str(R)],text=True).splitlines()[-1])>=50*1024**3
note(f'RAWCHECK START lock={lock.name} used_MiB={used} load={os.getloadavg()[0]:.2f}')
env=child_env({'DOJO_DATA':str(B),'SIMLAB_GPU':'0','TMPDIR':str(R/'tmp'),'PYTHONDONTWRITEBYTECODE':'1'})
cmd=['systemd-run','--user','--scope','-q','-p','MemoryMax=20G','-p','CPUQuota=600%','timeout','-k','60','3600','bash',str(P/'scripts/simlab/rdpy.sh'),
     '-u','-m','dojo.simlab.presettle','--task','match_and_pick_from_conveyor','--ids',*map(str,IDS),'--raw-check']
with (R/'rawcheck.log').open('a') as log:
 p=subprocess.Popen(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT);(R/'rawcheck.pid').write_text(str(p.pid)+'\n');rc=p.wait()
lock.close()
raw=json.loads((LD/'raw_index.json').read_text())
st={str(i):(raw.get(str(i)) or {}).get('status') for i in IDS}
same=all(hashlib.sha256((LD/n).read_bytes()).hexdigest()==h for n,h in want.items() if n!='raw_index.json')
note('RAWCHECK END rc=%d status=%s records_unchanged=%s'%(rc,json.dumps(st),same))
print(rc,st,same)
