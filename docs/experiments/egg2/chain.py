import os,sys,time,subprocess
from pathlib import Path
root=Path.home()/'dojo-data/egg2'
worker=int(sys.argv[1]); pid=int((root/('pilot-off.pid' if worker==0 else 'pilot-on.pid')).read_text())
while Path(f'/proc/{pid}').exists():time.sleep(10)
ids=([100000,100010,100015,100019,100054] if worker==0 else [100001,100012,100016,100053,100055])
for i,seed in enumerate(ids):
 modes=['on','off'] if (i+worker)%2 else ['off','on']
 for mode in modes:
  if (seed,mode) in [(100000,'off'),(100001,'on')]:continue
  name=f'paired-{seed}-{mode}'
  with (root/'logs'/f'{name}.log').open('w') as log:
   result=subprocess.run(['bash',str(root/'job.sh'),name,str(seed),mode],stdout=log,stderr=subprocess.STDOUT)
  print(name,result.returncode,flush=True)
  if result.returncode:sys.exit(result.returncode)
  time.sleep(90)
