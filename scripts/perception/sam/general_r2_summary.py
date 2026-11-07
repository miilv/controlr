"""All 32 predeclared ablations, unchanged harm gates, and strict latency gates."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

from dojo.perception.general_filters import enabled
from scripts.perception.sam.general_summary import aggregate, harms, paired_ci


def rank(row):
    return (row['failed_gate_families'], row['harm_groups'],
            -min(row['micro_delta'],row['macro_delta']), row['worst_latency_ratio'],
            len(enabled(row['bits'])), row['bits'])


def main():
    ap=argparse.ArgumentParser()
    for name in ('input','meta','reference','out'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(s) for s in Path(a.input).read_text().splitlines()]
    assert len(records)==5940 and len({(r['run'],r['frame'],r['cam'],r['arm']) for r in records})==5940
    assert len({r['config_sha256'] for r in records})==1
    assert len({r['task'] for r in records})==11
    meta=json.loads(Path(a.meta).read_text());groups=defaultdict(list);arms=defaultdict(list)
    for r in records:
        groups[(r['task'],r['frame'],r['cam'],r['arm'])].append(r);arms[r['arm']].append(r)
    reference={(r['task'],r['frame'],r['cam']):r for r in csv.DictReader(open(a.reference)) if r['arm']=='v2b'}
    diffs=[]
    for (task,phase,cam,arm),rows in groups.items():
        if arm!='v2b':continue
        b=aggregate(rows);old=reference[(task,phase,cam)]
        for k in ('n','recall','mean_iou','phantoms'):
            if b[k] is not None and abs(b[k]-float(old[k]))>1e-6:diffs.append([task,phase,cam,k,b[k],float(old[k])])
    tasks=sorted({r['task'] for r in records});base=aggregate(arms['v2b'])
    macro=lambda rows:statistics.mean(aggregate([r for r in rows if r['task']==t])['recall'] for t in tasks)
    bmacro=macro(arms['v2b']);nmax=max(x['n_prompts'] for x in meta['scaling'])
    concurrent={cam:max(s for x in meta['scaling'] if x['cam']==cam and x['n_prompts']==nmax for s in x['two_client_s']) for cam in {x['cam'] for x in meta['scaling']}}
    ablations=[];risks=[];cells=[];byprompt=defaultdict(list)
    for r in arms['f00']:byprompt[(r['cam'],r['n_prompts'])].append(r['latency_s'])
    for bits in range(32):
        arm=f'f{bits:02}';rows=arms[arm];pooled=aggregate(rows);m=macro(rows)
        local=[]
        for (task,phase,cam,k),part in groups.items():
            if k!=arm:continue
            b=aggregate(groups[(task,phase,cam,'v2b')]);g=aggregate(part);flags=harms(b,g)
            rr={'bits':bits,'task':task,'phase':phase,'cam':cam,'harms':flags,
                'recall_delta':None if b['recall'] is None else g['recall']-b['recall'],
                'IoU_delta':None if b['mean_iou'] is None else g['mean_iou']-b['mean_iou'],
                'phantoms_added':g['phantoms']-b['phantoms'],'phantoms_per_image_delta':g['phantoms_per_image']-b['phantoms_per_image']}
            risks.append(rr);local.append(rr)
        limits={'cam_head':2.,'cam_left_wrist':1.,'cam_right_wrist':1.}
        maxima={cam:max(r['latency_s'] for r in rows if r['cam']==cam) for cam in limits}
        latency_ok=all(max(maxima[c],concurrent[c]) <= limits[c] for c in limits)
        gain_ok=pooled['recall']-base['recall'] >= .05 and m-bmacro >= .05
        nharm=sum(bool(r['harms']) for r in local)
        row={'bits':bits,'filters':'+'.join(enabled(bits)) or 'none','recall':pooled['recall'],'macro_recall':m,
            'mean_iou':pooled['mean_iou'],'phantoms':pooled['phantoms'],
            'micro_delta':pooled['recall']-base['recall'],'macro_delta':m-bmacro,
            'recall_cost_vs_unfiltered':aggregate(arms['f00'])['recall']-pooled['recall'],
            'harm_groups':nharm,'head_max_s':maxima['cam_head'],'wrist_max_s':max(maxima[c] for c in limits if c!='cam_head'),
            'head_p50_s':statistics.median(r['latency_s'] for r in rows if r['cam']=='cam_head'),
            'wrist_p50_s':statistics.median(r['latency_s'] for r in rows if r['cam']!='cam_head'),
            'worst_latency_ratio':max(max(maxima[c],concurrent[c])/limits[c] for c in limits),
            'qualifies':gain_ok and not nharm and latency_ok and not diffs,
            'failed_gate_families':int(not gain_ok)+int(bool(nharm))+int(not latency_ok)+int(bool(diffs))}
        ablations.append(row)
        for task in tasks:
            b=aggregate([r for r in arms['v2b'] if r['task']==task]);g=aggregate([r for r in rows if r['task']==task]);u=aggregate([r for r in arms['f00'] if r['task']==task])
            cells.append({'bits':bits,'filters':row['filters'],'task':task,'images':g['images'],'n':g['n'],
                'v2b_recall':b['recall'],'unfiltered_recall':u['recall'],'recall':g['recall'],
                'recall_cost_vs_unfiltered':u['recall']-g['recall'],'v2b_IoU':b['mean_iou'],'mean_iou':g['mean_iou'],
                'v2b_phantoms':b['phantoms'],'unfiltered_phantoms':u['phantoms'],'phantoms':g['phantoms']})
    closest=min(ablations,key=rank)
    verdict={'qualifies':any(r['qualifies'] for r in ablations),'qualified_bits':[r['bits'] for r in ablations if r['qualifies']],
        'closest':closest,'ranking':'failed gate families, harm-group count, largest minimum micro/macro gain, lower latency ratio, fewer filters, bits',
        'baseline':base,'baseline_macro':bmacro,'baseline_reference_differences':diffs,
        'shared_server_two_client_full_bank_max_s':concurrent,'ablations':ablations,
        'closest_ci':paired_ci(arms['v2b'],arms[f"f{closest['bits']:02}"]),'images':180,'LLM_calls':0}
    for name,rs in [('ablations.csv',ablations),('cells.csv',cells),('risk.csv',risks),
                    ('latency-by-count.csv',[{'cam':c,'prompts':n,'images':len(v),'p50_s':statistics.median(v),'max_s':max(v)} for (c,n),v in sorted(byprompt.items())])]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps({'qualifies':verdict['qualifies'],'closest':closest,'baseline_diffs':diffs}))


if __name__=='__main__':main()
