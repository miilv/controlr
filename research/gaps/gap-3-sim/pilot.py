"""Gap-3 CPU pilot: which conditioning makes a light head follow an LLM-issued target?

Kinematic + contact toy (NOT MuJoCo, NOT images): 2 mm-clearance peg-in-hole with K=3
identical fixtures at random heights. A privileged scripted expert with DART-style
clean-label noise generates demos; a small object-centric head is trained by BC with
L1 on 8-step action chunks under several conditionings / training regimes, then
evaluated closed-loop under keypoint noise, staleness (target moved after the spec),
condition erasure and condition shift. Analytic baselines are run on identical scenes.

Units: metres, 20 Hz control. Run with the cached CPU torch:
  PYTHONPATH=<uv-archive-with-torch> python3 pilot.py
"""
import json, sys, time, itertools
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(4)
K = 3; MAXSTEP = 0.006; CLR = 0.002; FOOT = 0.025; DEPTH = 0.015; SUCC = 0.012
T = 120; H = 8; EXEC = 4; V_OBJ = 0.005  # object speed for staleness (m/s)


# ---------------------------------------------------------------- scenes
def sample_scenes(rng, N, near_start=False):
    pos = np.zeros((N, K, 3))
    for i in range(N):
        while True:
            xy = rng.uniform(-0.10, 0.10, (K, 2))
            d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(K)
            if d.min() > 0.06:
                break
        pos[i, :, :2] = xy
        pos[i, :, 2] = rng.uniform(0.02, 0.06, K)  # fixture top heights (hole depth 15 mm fits)
    tgt = rng.integers(0, K, N)
    perm = np.stack([rng.permutation(K) for _ in range(N)])  # slot s shows fixture perm[s]
    if near_start:  # for the staged primitive: start 2-4 cm above target, <=1.5 cm lateral
        h = pos[np.arange(N), tgt]
        r = 0.015 * np.sqrt(rng.uniform(0, 1, N)); th = rng.uniform(0, 2 * np.pi, N)
        peg = np.c_[h[:, 0] + r * np.cos(th), h[:, 1] + r * np.sin(th), h[:, 2] + rng.uniform(0.02, 0.04, N)]
    else:
        peg = np.c_[rng.uniform(-0.12, 0.12, (N, 2)), rng.uniform(0.12, 0.18, N)]
    u = rng.normal(size=(N, 2)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    return dict(pos=pos, tgt=tgt, perm=perm, peg=peg, udir=u)


def clipn(a, m):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a * np.minimum(1.0, m / np.maximum(n, 1e-12))


class Sim:
    def __init__(self, sc):
        self.pos = sc['pos']; self.tgt = sc['tgt']; self.peg = sc['peg'].copy()
        self.N = len(self.peg); self.ins = -np.ones(self.N, int)
        self.contact = np.zeros(self.N); self.force = np.zeros(self.N)
        self.done = np.zeros(self.N, bool); self.result = -np.ones(self.N, int)  # fixture inserted

    def step(self, a):
        a = clipn(a, MAXSTEP); a[self.done] = 0
        old = self.peg; new = old + a
        contact = np.zeros(self.N); force = np.zeros(self.N)
        idx = np.arange(self.N)
        ins = self.ins >= 0
        if ins.any():
            hp = self.pos[idx[ins], self.ins[ins]]
            d = new[ins, :2] - hp[:, :2]; dn = np.linalg.norm(d, axis=1, keepdims=True)
            new[ins, :2] = np.where(dn > CLR, hp[:, :2] + d / np.maximum(dn, 1e-12) * CLR, new[ins, :2])
            new[ins, 2] = np.maximum(new[ins, 2], hp[:, 2] - DEPTH)
            out = new[ins, 2] > hp[:, 2]
            tmp = self.ins[ins]; tmp[out] = -1; self.ins[ins] = tmp
        for k in range(K):
            hk = self.pos[:, k]
            free = self.ins < 0
            inside = np.linalg.norm(new[:, :2] - hk[:, :2], axis=1) < FOOT
            below = new[:, 2] < hk[:, 2]
            above = old[:, 2] >= hk[:, 2] - 1e-9
            lat = np.linalg.norm(new[:, :2] - hk[:, :2], axis=1)
            c = free & inside & below
            enter = c & above & (lat < CLR)
            self.ins[enter] = k
            blocked = c & above & ~enter
            force[blocked] = hk[blocked, 2] - new[blocked, 2]
            new[blocked, 2] = hk[blocked, 2]
            new[blocked, :2] = old[blocked, :2] + 0.5 * a[blocked, :2]  # friction while pressed
            contact[blocked] = 1
            side = c & ~above
            new[side, :2] = old[side, :2]; contact[side] = 1
        new[:, 2] = np.where(self.ins < 0, np.maximum(new[:, 2], 0.0), new[:, 2])
        self.peg = np.where(self.done[:, None], old, new)
        self.contact, self.force = contact, force
        deep = (self.ins >= 0)
        if deep.any():
            top = self.pos[idx, np.maximum(self.ins, 0), 2]
            fin = deep & (self.peg[:, 2] <= top - SUCC) & ~self.done
            self.result[fin] = self.ins[fin]; self.done |= fin


def expert_action(peg, h, ins_ok, contact):
    lat = h[:, :2] - peg[:, :2]; e = np.linalg.norm(lat, axis=1); dz = peg[:, 2] - h[:, 2]
    a = np.zeros_like(peg)
    goal = h + np.array([0, 0, 0.02])
    appr = clipn(0.35 * (goal - peg), MAXSTEP)
    low = (e > 0.0008) & (dz < 0.015)
    a[:] = appr
    a[low] = np.c_[0.5 * lat[low], np.full(low.sum(), 0.002)]
    desc = (e <= 0.0008)
    a[desc] = np.c_[0.5 * lat[desc], np.full(desc.sum(), -0.004)]
    c = contact > 0
    a[c] = np.c_[0.6 * lat[c], np.full(c.sum(), 0.002)]
    a[ins_ok] = np.c_[0.5 * lat[ins_ok], np.full(ins_ok.sum(), -0.004)]
    return a


# ---------------------------------------------------------------- perception / conditions
def detect(rng, sim, perm):
    d = sim.pos[np.arange(sim.N)[:, None], perm]  # (N,K,3) in slot order
    dist = np.linalg.norm(d - sim.peg[:, None], axis=-1)
    sig = 0.0005 + 0.03 * dist
    return d + rng.normal(size=d.shape) * sig[..., None]


def make_condition(rng, sc, sigma, stale_s, shift=False, det0=None):
    """LLM keypoint: noisy, computed on the scene *before* the target moved by V_OBJ*stale_s."""
    N = len(sc['tgt']); idx = np.arange(N)
    cond_fix = sc['tgt'].copy()
    if shift:  # point at a distractor instead
        cond_fix = (sc['tgt'] + 1 + rng.integers(0, K - 1, N)) % K
    h = sc['pos'][idx, cond_fix].copy()
    if not shift:
        h[:, :2] -= sc['udir'] * (V_OBJ * stale_s)[..., None] if np.ndim(stale_s) else sc['udir'] * V_OBJ * stale_s
    kp = h + rng.normal(size=h.shape) * np.asarray(sigma).reshape(-1, 1)
    # mask: seeded on the current frame from the (stale, noisy) point, then tracked
    dd = np.linalg.norm(det0[:, :, :2] - kp[:, None, :2], axis=-1)
    mslot = dd.argmin(1)
    mask = np.zeros((N, K)); mask[idx, mslot] = 1
    return kp, mask, cond_fix


COND = ['phase', 'kp3d', 'kp2d', 'mask', 'kp3d+mask']


def build_obs(det, peg, contact, force, color, cond_type, kp, mask, valid, use_color):
    N = len(peg)
    rel = (det - peg[:, None]) * 100.0  # cm
    col = color if use_color else np.zeros((N, K))
    m = mask * valid[:, None] if 'mask' in cond_type else np.zeros((N, K))
    slot = np.concatenate([rel, col[..., None], m[..., None]], -1)  # (N,K,5)
    kpr = np.zeros((N, 3))
    if cond_type in ('kp3d', 'kp3d+mask'):
        kpr = (kp - peg) * 100.0
    elif cond_type == 'kp2d':
        kpr[:, :2] = (kp[:, :2] - peg[:, :2]) * 100.0
    kv = valid.astype(float) if cond_type != 'phase' else np.zeros(N)
    kpr = kpr * kv[:, None]
    g = np.c_[peg * 10.0, contact, force * 1000.0, kpr, kv]  # (N,9)
    return slot.astype(np.float32), g.astype(np.float32)


class Head(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(5 + 9, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.score = nn.Linear(hid, 1)
        self.glob = nn.Sequential(nn.Linear(9, hid), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * hid, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, H * 3))

    def forward(self, slot, g):
        z = self.slot(torch.cat([slot, g[:, None].expand(-1, slot.shape[1], -1)], -1))
        w = torch.softmax(self.score(z), 1)
        return self.out(torch.cat([(w * z).sum(1), self.glob(g)], -1)).view(-1, H, 3)


# ---------------------------------------------------------------- demos
def gen_demos(seed, n_demo, scenario, near_start=False):
    rng = np.random.default_rng(seed)
    sc = sample_scenes(rng, n_demo, near_start)
    sim = Sim(sc); idx = np.arange(n_demo)
    if near_start:  # primitive target = nearest fixture to start
        dn = np.linalg.norm(sc['pos'][:, :, :2] - sc['peg'][:, None, :2], axis=-1)
        sc['tgt'] = dn.argmin(1); sim.tgt = sc['tgt']
    h = sc['pos'][idx, sc['tgt']]
    color = (sc['perm'] == sc['tgt'][:, None]).astype(float)
    rec = []
    for t in range(T):
        det = detect(rng, sim, sc['perm'])
        lab = expert_action(sim.peg, h, sim.ins == sc['tgt'], sim.contact)
        rec.append(dict(det=det, peg=sim.peg.copy(), contact=sim.contact.copy(), force=sim.force.copy(),
                        lab=lab, alive=~sim.done.copy()))
        e = np.linalg.norm(h[:, :2] - sim.peg[:, :2], axis=1)
        sig = np.where(e > 0.02, 0.002, 0.0003)  # DART-style executed-action noise, clean labels
        sim.step(lab + rng.normal(size=lab.shape) * sig[:, None])
        if sim.done.all():
            break
    ok = sim.result == sc['tgt']
    return sc, color, rec, ok


def demo_dataset(sc, color, rec, ok, cond_type, regime, use_color, rng):
    N = len(ok); idx = np.arange(N); TT = len(rec)
    det0 = rec[0]['det']
    if regime in ('R2', 'R3'):
        sig = rng.uniform(0, 0.005, N); stale = rng.uniform(0, 5.0, N)  # <=2.5 cm displacement
    else:
        sig = np.zeros(N); stale = np.zeros(N)
    kp, mask, _ = make_condition(rng, sc, sig, stale, det0=det0)
    S, G, Y = [], [], []
    for t in range(TT):
        r = rec[t]; keep = r['alive'] & ok
        if not keep.any():
            continue
        chunk = np.stack([rec[min(t + j, TT - 1)]['lab'] for j in range(H)], 1)  # (N,H,3)
        for j in range(H):  # pad after episode end with zeros
            if t + j >= TT:
                chunk[:, j] = 0
        valid = np.ones(N, bool)
        if regime in ('R1', 'R2', 'R3'):
            valid = rng.uniform(size=N) > 0.2
        col = color.copy()
        if regime == 'R3' and use_color:
            col[rng.uniform(size=N) < 0.5] = 0  # redundant-cue dropout
        s, g = build_obs(r['det'], r['peg'], r['contact'], r['force'], col, cond_type, kp, mask, valid, use_color)
        S.append(s[keep]); G.append(g[keep]); Y.append(chunk[keep] / MAXSTEP)
    return np.concatenate(S), np.concatenate(G), np.concatenate(Y).astype(np.float32)


def train(S, G, Y, seed, steps=3000, bs=512):
    torch.manual_seed(seed)
    m = Head(); ps = list(m.parameters())  # manual Adam (torch.optim needs sympy, absent here)
    mu = [torch.zeros_like(p) for p in ps]; nu = [torch.zeros_like(p) for p in ps]
    S, G, Y = map(torch.from_numpy, (S, G, Y))
    n = len(S); g = torch.Generator().manual_seed(seed); lr = 1e-3; b1, b2 = 0.9, 0.999
    for it in range(steps):
        b = torch.randint(0, n, (bs,), generator=g)
        loss = (m(S[b], G[b]) - Y[b]).abs().mean()
        for p in ps: p.grad = None
        loss.backward()
        if it == steps * 2 // 3: lr = 3e-4
        with torch.no_grad():
            for p, a, v in zip(ps, mu, nu):
                a.mul_(b1).add_(p.grad, alpha=1 - b1); v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                ah = a / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
                p.sub_(lr * ah / (vh.sqrt() + 1e-8))
    return m, float(loss.detach())


# ---------------------------------------------------------------- evaluation
def rollout(policy_fn, sc, seed, kp, mask, valid, color):
    rng = np.random.default_rng(seed + 999)
    sim = Sim(sc); chunk = None
    for t in range(T):
        det = detect(rng, sim, sc['perm'])
        if t % EXEC == 0:
            chunk = policy_fn(sim, det, kp, mask, valid, color, t)
        sim.step(chunk[:, t % EXEC])
        if sim.done.all():
            break
    return sim.result


def head_policy(model, cond_type, use_color):
    def f(sim, det, kp, mask, valid, color, t):
        s, g = build_obs(det, sim.peg, sim.contact, sim.force, color, cond_type, kp, mask, valid, use_color)
        with torch.no_grad():
            return model(torch.from_numpy(s), torch.from_numpy(g)).numpy() * MAXSTEP
    return f


def ik_kp_policy():  # B1: open-loop servo to the LLM keypoint, no vision
    def f(sim, det, kp, mask, valid, color, t):
        a = expert_action(sim.peg, kp, sim.ins >= 0, sim.contact)
        return np.repeat(a[:, None], H, 1)
    return f


def servo_policy():  # B2: associate keypoint->mask slot at t=0, EMA-filtered wrist visual servo + contact retry
    st = {}
    def f(sim, det, kp, mask, valid, color, t):
        sl = mask.argmax(1); d = det[np.arange(sim.N), sl]
        st['h'] = d if t == 0 else 0.7 * st['h'] + 0.3 * d
        a = expert_action(sim.peg, st['h'], sim.ins >= 0, sim.contact)
        return np.repeat(a[:, None], H, 1)
    return f


def staged_policy(prim):  # B3: analytic transport to kp+3cm, then frozen no-condition primitive
    st = {}
    def f(sim, det, kp, mask, valid, color, t):
        if t == 0:
            st['stage'] = np.zeros(sim.N, bool)
        goal = kp + np.array([0, 0, 0.03])
        st['stage'] |= np.linalg.norm(goal - sim.peg, axis=1) < 0.003
        a = np.repeat(clipn(0.35 * (goal - sim.peg), MAXSTEP)[:, None], H, 1)
        if st['stage'].any():
            s, g = build_obs(det, sim.peg, sim.contact, sim.force, color, 'phase', kp, mask, np.zeros(sim.N, bool), False)
            with torch.no_grad():
                p = prim(torch.from_numpy(s), torch.from_numpy(g)).numpy() * MAXSTEP
            a[st['stage']] = p[st['stage']]
        return a
    return f


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * p, 1), round(100 * (c - w), 1), round(100 * (c + w), 1)


TESTS = [('sig0', 0.0, 0, False, False), ('sig2', 0.002, 0, False, False), ('sig5', 0.005, 0, False, False),
         ('sig10', 0.010, 0, False, False), ('stale2', 0.002, 2, False, False), ('stale5', 0.002, 5, False, False),
         ('stale10', 0.002, 10, False, False), ('erase', 0.0, 0, False, True), ('shift', 0.0, 0, True, False)]


def evaluate(policy_fn, scenario, n_eval, seed, cond_type='kp3d'):
    out = {}
    for name, sig, stale, shift, erase in TESTS:
        rng = np.random.default_rng(seed)
        sc = sample_scenes(rng, n_eval)
        color = (sc['perm'] == sc['tgt'][:, None]).astype(float)
        sim0 = Sim(sc); det0 = detect(np.random.default_rng(seed + 7), sim0, sc['perm'])
        crng = np.random.default_rng(seed + 11)
        # move target: scene positions are "now"; condition built from pre-move position
        kp, mask, cfix = make_condition(crng, sc, sig, stale, shift=shift, det0=det0)
        valid = np.zeros(n_eval, bool) if erase else np.ones(n_eval, bool)
        if erase:
            kp = sc['peg'].copy(); mask = np.zeros((n_eval, K))  # Fast-Plans-style: endpoint := start
        res = rollout(policy_fn, sc, seed, kp, mask, valid, color)
        succ = int((res == sc['tgt']).sum()); follow = int((res == cfix).sum())
        wrong = int(((res >= 0) & (res != sc['tgt'])).sum())
        out[name] = dict(n=n_eval, succ=succ, succ_pct=wilson(succ, n_eval), follow=follow,
                         follow_pct=wilson(follow, n_eval), wrong_hole=wrong)
    return out


if __name__ == '__main__':
    n_demo = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seeds = [0, 1]
    results = {}
    t0 = time.time()
    # analytic baselines (scenario-independent; color unused)
    for nm, fn in [('B1_ik_to_kp', ik_kp_policy), ('B2_servo_mask', servo_policy)]:
        results[nm] = evaluate(fn(), 'S1', n_eval, 12345)
        print(nm, {k: v['succ_pct'] for k, v in results[nm].items()}, flush=True)
    for scen in ['S1', 'S2']:
        use_color = scen == 'S2'
        regimes = ['R0', 'R1', 'R2'] + (['R3'] if use_color else [])
        for seed in seeds:
            sc, color, rec, ok = gen_demos(1000 + seed, n_demo, scen)
            print(f'{scen} seed{seed}: expert demo success {ok.mean():.3f}', flush=True)
            for ct, rg in itertools.product(COND, regimes):
                if ct == 'phase' and rg != 'R0':
                    continue
                drng = np.random.default_rng(seed * 100 + 5)
                S, G, Y = demo_dataset(sc, color, rec, ok, ct, rg, use_color, drng)
                m, loss = train(S, G, Y, seed)
                r = evaluate(head_policy(m, ct, use_color), scen, n_eval, 12345)
                key = f'{scen}|{ct}|{rg}|seed{seed}'
                results[key] = dict(loss=loss, n_train=len(S), **r)
                print(key, f'loss={loss:.3f} n={len(S)}', {k: v['succ_pct'][0] for k, v in r.items()},
                      'follow_shift', r['shift']['follow_pct'][0], f'{time.time()-t0:.0f}s', flush=True)
        # staged primitive baseline (B3)
        if scen == 'S1':
            for seed in seeds:
                scp, colp, recp, okp = gen_demos(2000 + seed, n_demo, scen, near_start=True)
                S, G, Y = demo_dataset(scp, colp, recp, okp, 'phase', 'R0', False, np.random.default_rng(seed))
                prim, loss = train(S, G, Y, seed)
                results[f'B3_staged_primitive|seed{seed}'] = evaluate(staged_policy(prim), 'S1', n_eval, 12345)
                print('B3 seed', seed, {k: v['succ_pct'][0] for k, v in results[f'B3_staged_primitive|seed{seed}'].items()}, flush=True)
    json.dump(results, open('results.json', 'w'), indent=1)
    print('total', time.time() - t0)
