"""
E15 -- What does the layer-2 noise veto a2 do?

At GD's step-1000 weights (runs/main), on test data:
  * loss with a2 as learned, with a2 = 0 (veto removed), with a2 = -30 (perfect veto)
  * mass that layer 2 puts on NOISE tokens (should be ~0: the readout bias already supplies the
    noise probability, so any noise mass in f is wasted)
  * a2 over training (does it settle or drift?)
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
out = []
print(f"{'alpha':>6} {'a2':>6} | {'loss':>8} {'a2=0':>9} {'a2=-30':>9} | {'noise mass in f':>15} {'(a2=0)':>8} | a2 @ 50/200/500/1000")
for a in (0.2, 0.4, 0.6, 0.8, 0.9, 0.99):
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    th = TH[1000].copy()
    te = M.load(a, 'test'); x, y = te['x'].to(dev), te['y'].to(dev)
    def run(a2):
        t = th.copy(); t[2] = a2; t = torch.tensor(t, dtype=torch.float32, device=dev)
        with torch.no_grad():
            L = float(M.ce(t, x, y, a))
            nm = []
            for i in range(0, x.shape[0], 2500):
                f = M.forward(t, x[i:i + 2500])[:, M.TS:]
                nm.append(f[..., M.VS:].sum(-1).flatten())
        return L, float(torch.cat(nm).mean())
    L, nm = run(th[2]); L0, nm0 = run(0.0); Lv, _ = run(-30.0)
    r = dict(alpha=a, a2=float(th[2]), loss=L, loss_a2_0=L0, loss_a2_m30=Lv, noise_mass=nm, noise_mass_a2_0=nm0,
             a2_traj={s: float(TH[s, 2]) for s in (50, 200, 500, 1000)})
    out.append(r)
    print(f"{a:>6} {th[2]:>6.2f} | {L:>8.5f} {L0 - L:>+9.4f} {Lv - L:>+9.5f} | {nm:>15.4f} {nm0:>8.3f} | "
          + ' / '.join(f'{TH[s, 2]:.2f}' for s in (50, 200, 500, 1000)), flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e15_a2_role.json'), 'w'), indent=1)
