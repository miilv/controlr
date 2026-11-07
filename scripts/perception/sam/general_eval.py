"""DEV ONLY: score the frozen general proposer on SAMBAKE's exact image manifest.

Prediction receives RGB, logged instruction and the shared catalogue only. Ground
truth and task names are used solely after prediction by the historical metric.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dojo.perception.general_masks import GeneralMaskConfig, GeneralProposal, propose
from dojo.perception.sam_masks import LocalSam, MODELS_DIR
from dojo.perception.seg_twin import Det, asset_key, category_of, instances, iou, match_greedy


def score(dets, codes, objs):
    """Identical >=30px/greedy/phantom metric to SAMBAKE; GT never enters propose()."""
    gt = instances(codes, 30)
    matches = match_greedy(gt, dets, thr=0.0)
    tiny = {gi: gm for gi, gm in instances(codes, 1).items() if gi not in gt}
    unused = [i for i in range(len(dets)) if i not in {pi for pi, _ in matches.values()}]
    tm = match_greedy(tiny, [dets[i] for i in unused], thr=0.0)
    matches.update({gi: (unused[pi], v) for gi, (pi, v) in tm.items()})
    rows = []
    for gi, gm in {**gt, **tiny}.items():
        obj = objs[gi]; key = asset_key(obj); pred = None; overlap = 0.
        if gi in matches:
            pi, overlap = matches[gi]; pred = dets[pi]
        cats = pred.key.split('|') if pred else []
        cat_ok = (category_of(pred.key) == category_of(key)) if pred and '/' in pred.key else (len(cats) == 1 and category_of(key) in cats)
        rows.append({'gi':gi,'label':obj.get('label'),'key':key,'px':int(gm.sum()),'clutter':bool(obj.get('clutter')),
                     'iou':overlap,'pred_key':pred.key if pred else None,'class_ok':bool(pred and pred.key == key),'cat_ok':bool(cat_ok)})
    robot = codes == 65534
    phantom = sum(max([iou(p.mask, g) for g in gt.values()] + [0.0]) < .3
                  and (p.mask & robot).sum() < .5 * p.mask.sum() for p in dets)
    return rows, int(phantom)


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()


class BaseAdapter:
    def __init__(self, model):self.model=model;self.last=[]
    def detect(self, image, conf):
        self.last=[Det(m.astype(np.uint8),key,float(sc),tuple(box)) for m,key,sc,box in self.model.detect(image,conf=conf)]
        return self.last


class RecordingSam:
    def __init__(self, sam):self.sam=sam;self.last=[]
    def text(self,*args,**kw):self.last=self.sam.text(*args,**kw);return self.last


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--instructions',required=True);ap.add_argument('--catalogue',required=True)
    ap.add_argument('--weights',required=True);ap.add_argument('--classes',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args()
    # Validate all immutable image identities before loading any GPU model.
    expected=json.loads(Path(a.manifest).read_text());by_source={}
    for p in sorted(Path(a.root).glob('*/*/source.json')):
        src=json.loads(p.read_text());by_source[(src['run'],src['frame'])]=p.parent
    work=[]
    for src in expected:
        lid=src['layout_id'];assert 100000<=lid<=100009 or 100020<=lid<=100024
        folder=by_source[(src['run'],src['frame'])]
        assert digest(folder/'settled.json')==src['layout_sha256']
        for cam,want in src['images'].items():assert digest(folder/(cam+'.png'))==want
        work.append((src,folder))
    instructions={(r['run'],r['frame']):r for r in json.loads(Path(a.instructions).read_text())}
    assert all(instructions[(s['run'],s['frame'])]['source'] for s,_ in work)
    catalogue=json.loads(Path(a.catalogue).read_text());cfg=GeneralMaskConfig()
    sam_checkpoint=Path(MODELS_DIR)/'sam3/sam3.pt'
    assert digest(sam_checkpoint)=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
    assert digest(a.weights)=='d7b2f167a69584697628ba967c0f82f4ecfacd3152267f12b23644cc46e8ca88'
    import torch
    from dojo.perception.seg_twin import _SegModel
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.30)
    base=BaseAdapter(_SegModel(a.weights,json.loads(Path(a.classes).read_text())['names'],conf=cfg.v2b_confidence))
    sam=RecordingSam(LocalSam([cfg.model]));start=time.monotonic()
    config_hash=hashlib.sha256(json.dumps(cfg.__dict__,sort_keys=True).encode()).hexdigest()
    with Path(a.out).open('w') as out,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for src,folder in work:
            instruction=instructions[(src['run'],src['frame'])]['instruction']
            for cam in src['images']:
                image=np.array(Image.open(folder/(cam+'.png')).convert('RGB'))
                t=time.monotonic();union,audit=propose(image,instruction,catalogue,base,sam,cfg)
                elapsed=time.monotonic()-t
                # Scoring starts only here. No GT/cell information reaches the proposer.
                objects=json.loads((folder/'gt.json').read_text())['objects']
                codes=np.array(Image.open(folder/('gt_'+cam+'.png'))).astype(np.int64)
                converted=[Det(d.mask,'|'.join(d.categories),d.score) if isinstance(d,GeneralProposal) else d for d in union]
                smasks=[Det(m.mask,'',m.score) for m in sam.last if m.score>=cfg.sam_threshold]
                from dojo.perception.general_masks import deduplicate
                for arm,dets in [('v2b',base.last),('general_sam',deduplicate([],smasks,cfg.dedup_iou)),('general_union',converted)]:
                    rows,phantoms=score(dets,codes,objects)
                    out.write(json.dumps({**src,'cam':cam,'image_sha256':src['images'][cam],'arm':arm,'rows':rows,
                        'n_pred':len(dets),'phantoms':phantoms,'latency_s':elapsed,'config_sha256':config_hash,
                        'n_prompts':len(audit['prompts']),'prompts':audit['prompts'] if arm=='general_union' else None})+'\n')
                out.flush();print(src['task'],src['layout_id'],src['frame'],cam,len(audit['prompts']),len(converted),round(elapsed,2),flush=True)
                gc.collect()
    meta={'config':cfg.__dict__,'config_sha256':config_hash,'wall_s':time.monotonic()-start,'images':len(work)*3,
          'catalogue_sha256':digest(a.catalogue),'frames_sha256':digest(a.manifest),'instructions_sha256':digest(a.instructions),
          'SAM_checkpoint_sha256':digest(sam_checkpoint),'weights_sha256':digest(a.weights),'classes_sha256':digest(a.classes),'torch':torch.__version__,
          'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'peak_reserved_gb':torch.cuda.max_memory_reserved()/1e9,
          'LLM_calls':0,'task_config_inputs':0,'per_task_bank_inputs':0}
    Path(a.out+'.meta.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
