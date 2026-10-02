"""
E8 -- The state weight c2 in an IDEALISED model, as a step towards a theorem.

Idealisation: layer 1 is PERFECT (r_t = one-hot of the last signal token before t, via
five.perfect_r), noise keys are vetoed (a2 very negative), and the induction weight b2 is held
fixed. Then the loss is a function of c2 alone, and it splits by query type:

    L(c2) = (fraction after noise) * L_noise(c2) + (fraction after signal) * L_signal(c2)

where L_noise / L_signal are the MEAN loss over post-noise / post-signal positions.
Conjecture to be proved:
    L_noise  decreases in c2   (after noise, the state block points at the right keys)
    L_signal increases in c2   (after signal, it points at the previous state's successors)
=> the best c2 is positive for alpha < 1 and moves to <= 0 as alpha -> 1.
This script only CHECKS the shapes on real data; nothing is trained.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
C2 = np.round(np.arange(-2, 14.01, 0.5), 2)
out = {}
for b2 in (4.0, 8.0, 12.0):
    for a in (0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0):
        tr = M.load(a, 'train')
        x, y = tr['x'][:6000].to(dev), tr['y'][:6000].to(dev)
        xs = x[:, M.TS:]
        wn, ws = (xs >= M.VS).float(), (xs < M.VS).float()
        Ln, Ls = [], []
        with torch.no_grad():
            for c in C2:
                th = torch.tensor([0., 0., -30., b2, float(c)], device=dev)
                ln = float(M.ce(th, x, y, a, weights=wn, perfect_layer1=True)) if wn.sum() > 0 else float('nan')
                ls = float(M.ce(th, x, y, a, weights=ws, perfect_layer1=True))
                Ln.append(ln); Ls.append(ls)
        Ln, Ls = np.array(Ln), np.array(Ls)
        fn = float(wn.mean())
        tot = (np.nan_to_num(Ln) * fn + Ls * (1 - fn))
        k = int(np.argmin(tot))
        r = dict(b2=b2, alpha=a, frac_noise=fn, c2_grid=C2.tolist(), L_noise=Ln.tolist(),
                 L_signal=Ls.tolist(), L_total=tot.tolist(), best_c2=float(C2[k]),
                 noise_monotone_down=bool(np.all(np.diff(Ln) <= 1e-6)) if fn > 0 else None,
                 signal_monotone_up_from0=bool(np.all(np.diff(Ls[C2 >= 0]) >= -1e-6)),
                 signal_argmin=float(C2[int(np.argmin(Ls))]))
        out[f'b2={b2},alpha={a}'] = r
        print(f"b2={b2:>4} alpha={a:<5} best c2 {r['best_c2']:>5}  | L_noise falls with c2? "
              f"{r['noise_monotone_down']}  ({Ln[0]:.4f} -> {Ln[-1]:.4f}) | L_signal rises for c2>=0? "
              f"{r['signal_monotone_up_from0']}  (min at c2={r['signal_argmin']})", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e8_ideal_c2_curves.json'), 'w'), indent=1)
