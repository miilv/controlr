"""Complete-panel general-vs-v2b/per-task-reference tables and predeclared harm gates."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

import numpy as np


def primary(row):
    return [o for o in row['rows'] if o['px'] >= 30 and not o['clutter'] and o['label'] != 'camera_stand']


def aggregate(rows):
    obs=[o for row in rows for o in primary(row)]
    n=len(obs);tp=sum(o['iou']>=.5 for o in obs)
    return {'images':len(rows),'n':n,'tp':tp,'recall':tp/n if n else None,
            'mean_iou':sum(o['iou'] for o in obs)/n if n else None,
            'phantoms':sum(r['phantoms'] for r in rows),
            'phantoms_per_image':sum(r['phantoms'] for r in rows)/len(rows) if rows else None}


def harms(base, union):
    flags=[]
    if base['recall'] is not None and union['recall'] <= base['recall']-.05+1e-12:flags.append('recall_drop>=5pp')
    if base['mean_iou'] is not None and union['mean_iou'] <= base['mean_iou']-.02+1e-12:flags.append('IoU_drop>=0.02')
    if union['phantoms']-base['phantoms']>=2 and union['phantoms_per_image']>=base['phantoms_per_image']+.5-1e-12:
        flags.append('phantoms_increase>=0.5/image_and_2_total')
    return flags


def paired_ci(base, union):
    """Pair and cluster by immutable input identity, not individual objects/views."""
    groups=defaultdict(lambda:[[],[]])
    for arm,rows in enumerate([base,union]):
        for row in rows:groups[(row['task'],row['layout_sha256'])][arm].append(row)
    clusters=list(groups.values());rng=np.random.default_rng(0);draws=rng.integers(len(clusters),size=(10000,len(clusters)))
    vals=[]
    for a,b in clusters:
        x,y=aggregate(a),aggregate(b)
        assert x['n']==y['n'] and x['images']==y['images']
        vals.append([x['n'],x['tp'],y['tp'],x['phantoms'],y['phantoms'],x['images']])
    v=np.array(vals,float)[draws].sum(axis=1)
    recall=np.divide(v[:,2]-v[:,1],v[:,0],out=np.zeros(len(v)),where=v[:,0]>0)
    phantom=(v[:,4]-v[:,3])/v[:,5]
    return {'clusters':len(clusters),'recall_delta_95ci':np.quantile(recall,[.025,.975]).tolist(),
            'phantoms_per_image_delta_95ci':np.quantile(phantom,[.025,.975]).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();records=[json.loads(s) for s in Path(a.input).read_text().splitlines() if s.strip()]
    assert len(records)==540,'incomplete or duplicate panel: require all 180 views x 3 arms'
    uid=lambda r:(r['run'],r['frame'],r['cam'],r['arm'])
    assert len({uid(r) for r in records})==540
    assert len({r['config_sha256'] for r in records})==1
    assert len({r['task'] for r in records})==11
    groups=defaultdict(list)
    for r in records:groups[(r['task'],r['frame'],r['cam'],r['arm'])].append(r)
    reference={(r['task'],r['frame'],r['cam'],r['arm']):r for r in csv.DictReader(Path(a.reference).open())}
    summary=[];risk=[];baseline_diffs=[];category_groups=defaultdict(list)
    for r in records:
        for obj in primary(r):category_groups[(r['task'],r['frame'],r['cam'],obj['key'].split('/')[1],r['arm'])].append(obj)
    for task,phase,cam,arm in sorted(groups):
        rows=groups[(task,phase,cam,arm)];s={'task':task,'frame':phase,'cam':cam,'arm':arm,**aggregate(rows)}
        # The evaluator timed the whole joint call. Only union rows are a latency sample.
        s['joint_call_median_s']=statistics.median(r['latency_s'] for r in rows) if arm=='general_union' else None
        summary.append(s)
        if arm!='general_union':continue
        base=aggregate(groups[(task,phase,cam,'v2b')]);current=aggregate(rows)
        flags=harms(base,current)
        risk.append({'task':task,'frame':phase,'cam':cam,'material_harm':flags,
                     'recall_delta':current['recall']-base['recall'] if base['recall'] is not None else None,
                     'IoU_delta':current['mean_iou']-base['mean_iou'] if base['mean_iou'] is not None else None,
                     'phantoms_delta':current['phantoms']-base['phantoms'],
                     'phantoms_per_image_delta':current['phantoms_per_image']-base['phantoms_per_image'],
                     **paired_ci(groups[(task,phase,cam,'v2b')],rows)})
        old=reference.get((task,phase,cam,'v2b'))
        if old:
            for key,col in [('n','n'),('recall','recall'),('mean_iou','mean_iou'),('phantoms','phantoms')]:
                val=base[key]
                if val is not None and abs(val-float(old[col]))>1e-6:baseline_diffs.append({'task':task,'frame':phase,'cam':cam,'field':key,'new':val,'published':float(old[col])})
    pooled={arm:aggregate([r for r in records if r['arm']==arm]) for arm in ['v2b','general_sam','general_union']}
    tasks=sorted({r['task'] for r in records});macro={}
    for arm in pooled:
        macro[arm]=statistics.mean(aggregate([r for r in records if r['task']==t and r['arm']==arm])['recall'] for t in tasks)
    micro_delta=pooled['general_union']['recall']-pooled['v2b']['recall'];macro_delta=macro['general_union']-macro['v2b']
    qualifies=micro_delta>=.05 and macro_delta>=.05 and not any(r['material_harm'] for r in risk) and not baseline_diffs
    table=[]
    for task in tasks:
        get=lambda arm:aggregate(groups[(task,'frame0','cam_head',arm)])
        b,g=get('v2b'),get('general_union');pt=reference[(task,'frame0','cam_head','union_sam3')]
        pt31=reference[(task,'frame0','cam_head','union_sam3.1')]
        s=get('general_sam')
        table.append({'task':task,'layouts':b['images'],'n':b['n'],'v2b_recall':b['recall'],'task_sam3_recall':float(reference[(task,'frame0','cam_head','sam3')]['recall']),
          'task_union_sam3_recall':float(pt['recall']),'task_union_sam31_recall':float(pt31['recall']),
          'general_sam_recall':s['recall'],'general_union_recall':g['recall'],'recall_delta':g['recall']-b['recall'],
          'v2b_IoU':b['mean_iou'],'task_union_IoU':float(pt['mean_iou']),'general_union_IoU':g['mean_iou'],
          'v2b_phantoms':b['phantoms'],'task_union_phantoms':int(pt['phantoms']),'general_union_phantoms':g['phantoms'],
          'material_harm':'; '.join(harms(b,g))})
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    categories=[{'task':t,'frame':ph,'cam':c,'category':cat,'arm':arm,'n':len(obs),'recall':sum(o['iou']>=.5 for o in obs)/len(obs),'mean_iou':statistics.mean(o['iou'] for o in obs)} for (t,ph,c,cat,arm),obs in sorted(category_groups.items())]
    for name,rows in [('summary.csv',summary),('head-table.csv',table),('risk.csv',risk),('category-table.csv',categories)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verdict={'qualifies_offline':qualifies,'micro_recall_delta':micro_delta,'macro_task_recall_delta':macro_delta,
        'macro_task_recall':macro,'pooled':pooled,'material_harm_groups':[r for r in risk if r['material_harm']],
        'v2b_reference_differences':baseline_diffs,'head_table':table,'images':180,'tasks':11,'LLM_calls':0,
        'camera_dispatch':'PROHIBITED: requires offline qualification and end-to-end general path; no automatic queue operation'}
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps({'qualifies_offline':qualifies,'micro_delta':micro_delta,'macro_delta':macro_delta,'material_harm_groups':len(verdict['material_harm_groups']),'baseline_differences':len(baseline_diffs)}))


if __name__=='__main__':main()
