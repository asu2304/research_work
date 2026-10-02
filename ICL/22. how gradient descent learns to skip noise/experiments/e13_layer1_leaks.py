"""
E13 -- Where does layer 1 go wrong, and does GD balance its two kinds of error?

At a scored query i, layer 1 should put all its attention on the last signal token before i
(distance g back, the 'gap'). Its missing mass 1 - r_i[pv(i)] goes to
    leak_noise  : noise keys        (closer noise is favoured by recency; the veto a1 fights it)
    leak_older  : older signal keys (recency p not sharp enough)
    leak_same   : older signal keys that happen to carry the SAME token as the last signal
                  (these are harmless for r: they still write the right one-hot)
Measured on GD's own layer 1 (runs/main) along training, overall and by gap length g.
Score at distance d: -p*d (+ a1 if the key is noise). No layer 2 involved.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
STEPS = (10, 30, 100, 200, 500, 1000)
out = {}
for a in (0.2, 0.4, 0.6, 0.8, 0.9, 0.99):
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    tr = M.load(a, 'train'); x = tr['x'][:5000].to(dev)
    noise = (x >= M.VS)
    idx = torch.arange(M.T, device=dev)
    last = torch.where(~noise, idx, torch.full_like(idx, -1)).cummax(1).values      # last signal <= t
    lastb = torch.full_like(last, -1); lastb[:, 1:] = last[:, :-1]                   # strictly before t
    rows = []
    for s in STEPS:
        th = torch.tensor(TH[s], dtype=torch.float32, device=dev)
        with torch.no_grad():
            _, P = M.forward(th, x, parts=True)
            A1 = P['A1'][:, M.TS:]                                                   # (B, 50, T)
            q_last = lastb[:, M.TS:]
            ok = q_last >= 0
            tgt_tok = x.gather(1, q_last.clamp_min(0))                               # token of last signal
            keyidx = idx.view(1, 1, -1)
            is_last = keyidx == q_last.unsqueeze(-1)
            older_sig = (~noise).unsqueeze(1) & (keyidx < q_last.unsqueeze(-1))
            same_tok = x.unsqueeze(1) == tgt_tok.unsqueeze(-1)
            m_last = (A1 * is_last).sum(-1)
            l_noise = (A1 * noise.unsqueeze(1).float()).sum(-1)
            l_older = (A1 * (older_sig & ~same_tok)).sum(-1)
            l_same = (A1 * (older_sig & same_tok)).sum(-1)
            gap = (idx[M.TS:].view(1, -1) - q_last)
            byg = {}
            for g in range(1, 9):
                m_ = ok & (gap == g)
                if m_.sum() > 100:
                    byg[g] = dict(n=int(m_.sum()), leak_noise=float(l_noise[m_].mean()),
                                  leak_older=float(l_older[m_].mean()), last=float(m_last[m_].mean()))
        rows.append(dict(step=s, p=float(TH[s, 0]), a1=float(TH[s, 1]),
                         last=float(m_last[ok].mean()), leak_noise=float(l_noise[ok].mean()),
                         leak_older=float(l_older[ok].mean()), leak_same=float(l_same[ok].mean()),
                         by_gap=byg))
    out[str(a)] = rows
    print(f"\n=== alpha={a}")
    print(f"{'step':>5} {'p':>5} {'a1':>7} | {'on last':>7} {'->noise':>8} {'->older':>8} {'->same tok':>10} | leak_noise / leak_older by gap g=1..6")
    for r in rows:
        bg = '  '.join(f"{g}:{v['leak_noise']:.3f}/{v['leak_older']:.3f}" for g, v in r['by_gap'].items() if int(g) <= 6)
        print(f"{r['step']:>5} {r['p']:>5.2f} {r['a1']:>7.2f} | {r['last']:>7.3f} {r['leak_noise']:>8.4f} {r['leak_older']:>8.4f} {r['leak_same']:>10.4f} | {bg}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e13_layer1_leaks.json'), 'w'), indent=1)
