"""Frozen instruction-only SAM probe; prediction/geometry precede any GT read.

32 uniform filter combinations share one set of raw proposals. Only the scoring
layer receives source task/GT metadata. The owned shared server serves two clients.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from dojo.perception.cameras import Camera
from dojo.perception.general_filters import FilterConfig, enabled, evidence, select
from dojo.perception.general_masks import deduplicate
from dojo.perception.general_prompts import PromptConfig, instruction_prompts
from dojo.perception.sam_masks import SamClient, MODELS_DIR
from dojo.perception.seg_twin import Det, _SegModel
from scripts.perception.sam.general_eval import BaseAdapter, digest, score


def service_request(sock, image, bank):
    client = SamClient(sock, loud=True)
    try:
        t = time.monotonic()
        masks = client.text(image, bank, thr=.2, model='sam3')
        return time.monotonic()-t, masks
    finally:
        client.conn.close()


def benchmark(sock, folder, bank):
    rows = []
    for view in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        image = np.array(Image.open(folder/(view+'.png')).convert('RGB'))
        for n in sorted({1, min(3, len(bank)), min(6, len(bank)), min(12, len(bank)), len(bank)}):
            times = [service_request(sock, image, bank[:n])[0] for _ in range(3)]
            with ThreadPoolExecutor(2) as pool:
                both = list(pool.map(lambda _: service_request(sock, image, bank[:n])[0], range(2)))
            rows.append({'cam': view, 'n_prompts': n, 'single_s': times, 'two_client_s': both})
    return rows


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'manifest', 'instructions', 'catalogue', 'weights', 'classes', 'out', 'sock', 'geometry_manifest'):
        ap.add_argument('--'+name.replace('_', '-'), required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(a.manifest).read_text())
    instructions = {(x['run'], x['frame']): x['instruction'] for x in json.loads(Path(a.instructions).read_text())}
    folders = {}
    for p in Path(a.root).glob('*/*/source.json'):
        s = json.loads(p.read_text()); folders[(s['run'], s['frame'])] = p.parent
    geometry_manifest = json.loads(Path(a.geometry_manifest).read_text())
    geometry_hashes = {(x['run'], x['frame']): x['sha256'] for x in geometry_manifest}
    assert len(manifest) == 60 and len(instructions) == 60
    for s in manifest:
        assert 100000 <= s['layout_id'] <= 100009 or 100020 <= s['layout_id'] <= 100024
        folder = folders[(s['run'], s['frame'])]
        assert digest(folder/'settled.json') == s['layout_sha256']
        assert digest(folder/'calib.json') == geometry_hashes[(s['run'], s['frame'])]
        for cam, expected in s['images'].items():
            assert digest(folder/(cam+'.png')) == expected
    assert digest(Path(MODELS_DIR)/'sam3/sam3.pt') == '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
    assert digest(a.weights) == 'd7b2f167a69584697628ba967c0f82f4ecfacd3152267f12b23644cc46e8ca88'
    assert digest(a.catalogue) == '0c49bf1d05d435f24c37ab3981fb1a7bb4de9dc663b785958768f937c2489cb1'
    catalogue = json.loads(Path(a.catalogue).read_text())
    cfg, pcfg = FilterConfig(), PromptConfig()
    config = {'filters': asdict(cfg), 'prompts': asdict(pcfg), 'sam_threshold': .2,
              'v2b_confidence': .25, 'sam_self_dedup_iou': .5, 'union_dedup_iou': .5}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    # No source/task/GT argument is passed to any prompt, detector or filter function.
    banks = {key: instruction_prompts(text, pcfg) for key, text in instructions.items()}
    import torch
    torch.set_num_threads(6); torch.cuda.set_per_process_memory_fraction(.05)
    base_model = BaseAdapter(_SegModel(a.weights, json.loads(Path(a.classes).read_text())['names'], conf=.25))
    t0 = time.monotonic()
    server_log = (out/'server.log').open('w')
    if Path(a.sock).exists():
        raise RuntimeError('refusing an existing socket path')
    server = subprocess.Popen([sys.executable, '-u', 'scripts/perception/sam/general_r2_server.py', a.sock], stdout=server_log, stderr=subprocess.STDOUT)
    try:
        while not Path(a.sock).exists():
            if server.poll() is not None or time.monotonic()-t0 > 180:
                raise RuntimeError('owned shared SAM server failed to start')
            time.sleep(.2)
        ready_s = time.monotonic()-t0
        # Fixed warmup on one already-registered image; no scoring or parameter tuning.
        first = manifest[0]; folder = folders[(first['run'], first['frame'])]
        image = np.array(Image.open(folder/'cam_head.png').convert('RGB'))
        warm_s, _ = service_request(a.sock, image, banks[(first['run'], first['frame'])])
        started = time.monotonic()
        with (out/'records.jsonl').open('w') as results, (out/'proposals.jsonl').open('w') as audits, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for bi, src in enumerate(manifest):
                folder = folders[(src['run'], src['frame'])]; bank = banks[(src['run'], src['frame'])]
                bases, proposals, cams, timing = {}, {}, {}, {}
                calib = json.loads((folder/'calib.json').read_text())['calib']
                for view in src['images']:
                    image = np.array(Image.open(folder/(view+'.png')).convert('RGB'))
                    # Only camera intrinsics and extrinsics; eef/config fields ignored.
                    c = calib[view]
                    cams[view] = Camera(view, np.array(c['K']), np.array(c['T_world_cam']), image.shape[1], image.shape[0])
                    t = time.monotonic(); bases[view] = base_model.detect(image, conf=.25)
                    torch.cuda.synchronize(); base_s = time.monotonic()-t
                    sam_s, raw = service_request(a.sock, image, bank)
                    t = time.monotonic(); proposals[view] = deduplicate([], raw, .5); prep_s = time.monotonic()-t
                    timing[view] = {'base_s': base_s, 'sam_rpc_s': sam_s, 'sam_self_dedup_s': prep_s}
                flags, geom, cpu = evidence(proposals, cams, bases, catalogue, cfg)
                bundles = {}
                for view in src['images']:
                    bundles[view] = {}
                    for bits in range(32):
                        t = time.monotonic()
                        ds = select(proposals[view], flags[view], bits)
                        converted = [Det(m.mask, '', m.score) for m in ds]
                        union = deduplicate(bases[view], converted, .5)
                        select_s = time.monotonic()-t
                        active = enabled(bits)
                        extra_cpu = sum(cpu[view][k] for k in active)
                        dependency_s = 0.
                        if bits & 7:
                            extra_cpu += cpu['geometry_s']
                            dependency_s = sum(sum(timing[v].values()) for v in timing if v != view)
                        if bits & 4:
                            extra_cpu += cpu['ranges_s']
                        latency = sum(timing[view].values())+select_s+extra_cpu+dependency_s
                        bundles[view][bits] = (union, latency, select_s, dependency_s)
                # All prediction/filter decisions complete before GT is opened.
                expected_gt = next(x['scoring_sha256'] for x in geometry_manifest
                                   if (x['run'], x['frame']) == (src['run'], src['frame']))
                for name, want in expected_gt.items():
                    assert digest(folder/name) == want
                objects = json.loads((folder/'gt.json').read_text())['objects']
                for view in src['images']:
                    audit = {'bundle': bi, 'cam': view, 'prompts': bank, 'timing': timing[view], 'cpu': cpu,
                             'n_raw_kept_sam': len(proposals[view]), 'flags': flags[view],
                             'v2b': [{'key':m.key,'score':m.score} for m in bases[view]],
                             'sam': [{'prompt':m.prompt, 'score':m.score, 'pixels':int(m.mask.sum())} for m in proposals[view]],
                             'geometry': [None if g is None else {'low':g.low.tolist(),'high':g.high.tolist(),'partner':[g.partner[0],int(g.partner[1])],'coverage':g.coverage,'ray_gap_m':g.ray_gap_m} for g in geom[view]]}
                    audits.write(json.dumps(audit)+'\n')
                    # Preserve masks for reproducibility; no additional model calls for ablations.
                    shape = proposals[view][0].mask.shape if proposals[view] else (480, 640)
                    masks = np.array([np.packbits(m.mask) for m in proposals[view]], dtype=np.uint8)
                    base_masks = np.array([np.packbits(m.mask) for m in bases[view]], dtype=np.uint8)
                    np.savez_compressed(out/f'masks-{bi:02}-{view}.npz', sam=masks, v2b=base_masks, shape=shape)
                    codes = np.array(Image.open(folder/('gt_'+view+'.png'))).astype(np.int64)
                    for bits in [-1, *range(32)]:
                        dets = bases[view] if bits == -1 else bundles[view][bits][0]
                        rows, ph = score(dets, codes, objects)
                        results.write(json.dumps({**src,'cam':view,'arm':'v2b' if bits == -1 else f'f{bits:02}',
                            'filters':[] if bits == -1 else enabled(bits),'rows':rows,'phantoms':ph,'n_pred':len(dets),
                            'n_prompts':len(bank),'latency_s':timing[view]['base_s'] if bits == -1 else bundles[view][bits][1],
                            'config_sha256':config_hash})+'\n')
                results.flush(); audits.flush()
                print(bi+1, '/60 bundles', src['task'], 'prompts', len(bank), 'SAM RPC s', [round(timing[v]['sam_rpc_s'],3) for v in timing], 'geometry s', round(cpu['geometry_s'],3), flush=True)
        # Scaling uses instruction-only prefixes on the maximum-count instruction.
        # Selection depends on prompt count alone, never a task/GT outcome.
        chosen = max(manifest, key=lambda s: len(banks[(s['run'], s['frame'])]))
        bench = benchmark(a.sock, folders[(chosen['run'], chosen['frame'])], banks[(chosen['run'], chosen['frame'])])
        c = SamClient(a.sock, loud=True); info = c.info(); c.conn.close()
        meta = {'config':config,'config_sha256':config_hash,'images':180,'combinations':32,'wall_s':time.monotonic()-started,
                'server_ready_s':ready_s,'warmup_s':warm_s,'server':info,'server_pid':server.pid,'scaling':bench,
                'input_sha256':{k:digest(getattr(a,k)) for k in ['manifest','instructions','catalogue','geometry_manifest','weights','classes']},
                'LLM_calls':0,'task_config_inputs':0,'checker_inputs':0,'depth_inputs':0,
                'geometry_note':'centroid triangulation and planar silhouette bounds: proxy, not measured depth/contact'}
        (out/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    finally:
        # Only the Popen child owned by this evaluator is terminated.
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        server_log.close()
        Path(a.sock).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
