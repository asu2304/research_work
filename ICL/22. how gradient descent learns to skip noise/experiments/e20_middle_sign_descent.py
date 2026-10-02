"""
E20 -- The middle stage as SIGN DESCENT along a straight line.

Under Adam the early updates are ~sign steps of size eta = 1/sqrt(108) logits (raw lr 1). From
zero, the start signs are proved (gating + skip seed + pull sign): layer 2 moves first with
s = (a2 -, b2 +, c2 +) [a2 = 0 at alpha = 1], layer 1 one step later with (p +, a1 -).
So the predicted path is the straight line
    theta_SD(k) = eta * ( +(k-1), -(k-1), -k, +k, +k )
until some gradient changes sign. This script walks that line (no GD) and reports, per parameter,
the first k where its descent direction disagrees with s, plus where c2's post-signal pull turns
negative and layer 1's state mass there. Then the same events read off GD's OWN saved gradients
(runs/main/traj_*.npz). If they agree, the middle is predicted from the start signs alone.
"""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
eta = 1 / M.SQ
K = 60
out = []
for a in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
    tr = M.load(a, 'train'); x, y = tr['x'][:6000].to(dev), tr['y'][:6000].to(dev)
    xs = x[:, M.TS:]; wn, ws = (xs >= M.VS).float(), (xs < M.VS).float(); N = float(xs.numel())
    ps = M.prev_signal(x)[:, M.TS:]
    s = np.array([1, -1, -1, 1, 1], float)
    if a >= 1:
        s[1] = s[2] = 0
    first = {n: None for n in M.NAMES}; c2sig_flip = None; mass_at = None
    rows = []
    for k in range(2, K + 1):
        th0 = eta * np.array([s[0] * (k - 1), s[1] * (k - 1), s[2] * k, s[3] * k, s[4] * k])
        th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        g, = torch.autograd.grad(M.ce(th, x, y, a), th)
        g = g.cpu().numpy()
        Ls = M.ce(th, x, y, a, weights=ws) * (float(ws.sum()) / N)
        gs, = torch.autograd.grad(Ls, th)
        with torch.no_grad():
            _, P = M.forward(th.detach(), x, parts=True)
            r = P['r'][:, M.TS:]; ok = ps >= 0
            mass = float(r.gather(-1, ps.clamp_min(0).unsqueeze(-1)).squeeze(-1)[ok].mean())
        for n_, idx in zip(M.NAMES, range(5)):
            if s[idx] != 0 and first[n_] is None and np.sign(-g[idx]) != s[idx]:
                first[n_] = k
        if c2sig_flip is None and -float(gs[4]) < 0:
            c2sig_flip, mass_at = k, mass
        rows.append(dict(k=k, theta=th0.tolist(), neg_grad=(-g).tolist(), c2_signal_pull=-float(gs[4]), state_mass=mass))
    # GD's own gradients
    G = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['grad']
    gd_first = {}
    for idx, n_ in enumerate(M.NAMES):
        if s[idx] == 0:
            gd_first[n_] = None; continue
        bad = [k for k in range(2, K + 1) if np.sign(-G[k, idx]) != s[idx]]
        gd_first[n_] = bad[0] if bad else None
    an = json.load(open(os.path.join(M.F22, 'runs', 'main', f'anatomy_x_{M.afmt(a)}.json')))['rows']
    out.append(dict(alpha=a, predicted_first_sign_change=first, gd_first_sign_change=gd_first,
                    predicted_c2_signal_flip=c2sig_flip, predicted_state_mass_at_flip=mass_at, rows=rows))
    fmt = lambda d: '  '.join(f"{n_}:{d[n_] if d[n_] is not None else '-'}" for n_ in M.NAMES)
    print(f"alpha={a}: first step each gradient changes sign (within {K} steps)\n"
          f"     straight-line prediction: {fmt(first)}   | c2 after-signal pull flips at k={c2sig_flip} (state mass {mass_at if mass_at is None else round(mass_at, 3)})\n"
          f"     GD's own gradients      : {fmt(gd_first)}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e20_middle_sign_descent.json'), 'w'), indent=1)
