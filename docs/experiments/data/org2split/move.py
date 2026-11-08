"""ORG2SPLIT: two held queue locks; fail closed; restore source on append failure."""
from pathlib import Path
import subprocess,json,shlex,hashlib,datetime
M=Path(__file__).resolve().parent
source=(M/'rows-source.txt').read_text().splitlines()
target=(M/'rows-target.txt').read_text().splitlines()
REMOTE=r"""
import sys,json,fcntl,hashlib,datetime,re
from pathlib import Path
side=sys.argv[1];rows=json.loads(sys.stdin.readline());s=Path.home()/('dojo-integ-'+side+'-state');q=s/'queue2.txt'
f=(s/'queue2.txt.lock').open('r+');fcntl.flock(f,fcntl.LOCK_EX)
before=q.read_bytes();lines=before.decode().splitlines()
if side=='c6':
 assert all(lines.count(r)==1 for r in rows),'source rows absent/duplicated'
 logs='\n'.join((s/f'lane2_heavy{i}.log').read_text() for i in (1,2))
 for r in rows:
  assert not re.search(r'(?:start|done) '+re.escape(r.split()[0])+r'(?:\s|$)',logs),'row already started'
 rest=[r for r in lines if r not in set(rows)];after=('\n'.join(rest)+'\n').encode() if rest else b''
else:
 assert not any(r.split()[0] in before.decode() for r in rows),'target duplicated'
 assert all(r.strip() for r in lines) or not any(r.strip() for r in lines),'interior blank queue line'
 prefix=before if any(r.strip() for r in lines) else b''
 if prefix and not prefix.endswith(b'\n'):prefix+=b'\n'
 after=prefix+('\n'.join(rows)+'\n').encode()
print(json.dumps({'ready':True,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'before_rows':len([r for r in lines if r.strip()]),'after_rows':len([r for r in after.decode().splitlines() if r.strip()]),'before':before.decode(),'after':after.decode()}),flush=True)
for command in sys.stdin:
 cmd=command.strip()
 if cmd=='commit':
  assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,8,7,40,tzinfo=datetime.timezone.utc)
  assert q.read_bytes()==before,'queue changed despite held lock'
  q.write_bytes(after);print(json.dumps({'committed':True,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}),flush=True)
 elif cmd=='rollback':
  q.write_bytes(before);print(json.dumps({'rolled_back':True}),flush=True)
 elif cmd=='finish':break
 else:raise ValueError(cmd)
"""
procs=[];states={};committed=[]
try:
 for side,rows in [('c6',source),('c4',target)]:
  cmd='python3 -u -c '+shlex.quote(REMOTE)+' '+side
  p=subprocess.Popen(['ssh','compute'+side[1],cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True);procs.append((side,p))
  p.stdin.write(json.dumps(rows)+'\n');p.stdin.flush()
  line=p.stdout.readline();assert line,(side,'lock/preflight failed')
  state=json.loads(line);assert state['ready'];states[side]=state
 # Both locks remain held from here through both writes and receipt capture.
 for side,p in procs:
  p.stdin.write('commit\n');p.stdin.flush();resp=json.loads(p.stdout.readline());assert resp['committed'];committed.append((side,p))
 for side,state in states.items():
  for kind in ['before','after']:(M/f'{side}-queue-{kind}.txt').write_text(state.pop(kind))
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'moved_rows':80,'moved_pairs':40,'queues':states,'both_locks_held':True,'route':'claude/claude-sonnet-5-5','prereg_commits':['a43e1154','002f9426']}
 (M/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
except BaseException:
 for side,p in reversed(committed):
  p.stdin.write('rollback\n');p.stdin.flush();print(side,p.stdout.readline())
 raise
finally:
 for side,p in procs:
  if p.poll() is None:
   p.stdin.write('finish\n');p.stdin.flush();p.stdin.close();p.wait(timeout=20)
