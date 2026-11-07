import json
from dojo.simlab.gen_layouts import generate
from dojo.simlab import paths
p=paths.LAYOUTS/'pick_from_conveyor_by_image'
p.mkdir(parents=True,exist_ok=True)
for i in [100020,100021,*range(100030,100040)]:
 d=generate('pick_from_conveyor_by_image',i,variant=7,seed_salt='byimg-20261107-fresh')
 (p/f'{i}.json').write_text(json.dumps(d,indent=1))
print('generated v15 variant7 fresh')
