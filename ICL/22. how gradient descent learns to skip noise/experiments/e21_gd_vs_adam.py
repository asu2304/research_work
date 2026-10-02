"""
E21 -- Which findings survive without Adam? Plain full-batch gradient descent (runs/gd_lr30,
lr = 30 in logit units, 3000 steps) against Adam (runs/main, 1000 steps).

  1. final loss vs Adam and vs ideal
  2. the c2 formula  c2* = b2/2 + 1/2 log((1-a)/a) + kappa   (kappa = 0.514 from theory)
  3. b2: settles (dense noise) or drifts (sparse noise)?  growth per e-fold over steps 1000..3000
  4. the skip-horizon law: is alpha * H roughly the same across alpha at a fixed step?
  5. the first step: which parameters move, and by how much
  6. middle stage (from anatomy): does c2's post-signal pull flip, and at what layer-1 state mass?
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

R = os.path.join(M.F22, 'runs')
kappa = json.load(open(os.path.join(R, 'e9_kappa_theory.json')))['kappa']
alphas = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0)
out = []
print(f"{'alpha':>6} | {'GD test':>8} {'Adam test':>9} {'ideal':>7} | {'GD c2':>6} {'formula':>7} | "
      f"{'b2':>6} {'b2/e-fold':>9} | alpha*H @100 / 300 / 1000 / 3000")
for a in alphas:
    f = os.path.join(R, 'gd_lr30', f'traj_x_{M.afmt(a)}.npz')
    if not os.path.exists(f):
        print(f'{a:>6} | missing'); continue
    d = np.load(f); TH = d['theta']; te = d['test']; S = len(TH) - 1
    adam = np.load(os.path.join(R, 'main', f'traj_x_{M.afmt(a)}.npz'))['test'][-1]
    p, a1, a2, b2, c2 = TH[-1]
    pred = b2 / 2 + 0.5 * math.log((1 - a) / a) + kappa if a < 1 else float('nan')
    t = np.arange(1000, S + 1); k = float(np.polyfit(np.log(t), TH[1000:S + 1, 3], 1)[0])
    aH = {s: (a * (1 + abs(TH[s, 1]) / TH[s, 0]) if a < 1 and TH[s, 0] > 0 else float('nan')) for s in (100, 300, 1000, S)}
    row = dict(alpha=a, gd_test=float(te[-1]), adam_test=float(adam), ideal=M.ideal_ce(a),
               theta_final=TH[-1].tolist(), c2_formula=pred, b2_growth_per_efold=k,
               alphaH=aH, theta_step1=TH[1].tolist())
    an = os.path.join(R, 'gd_lr30', f'anatomy_x_{M.afmt(a)}.json')
    if os.path.exists(an) and a < 1:
        rows = json.load(open(an))['rows']
        flip = next((r for r in rows if r['step'] > 0 and -r['grad_signal'][4] < 0), None)
        row['c2_signal_flip_step'] = flip['step'] if flip else None
        row['state_mass_at_flip'] = flip['layer1_state_mass'] if flip else None
        last = rows[-1]
        row['final_pull_ratio'] = -last['grad_noise'][4] / last['grad_signal'][4] if last['grad_signal'][4] else None
        row['final_state_mass'] = last['layer1_state_mass']
    out.append(row)
    print(f"{a:>6} | {te[-1]:>8.5f} {adam:>9.5f} {M.ideal_ce(a):>7.4f} | {c2:>6.2f} {pred:>7.2f} | "
          f"{b2:>6.2f} {k:>9.2f} | " + ' / '.join(f'{v:.2f}' for v in aH.values()), flush=True)
print('\nfirst GD step (logits): ' + '  '.join(f"a={r['alpha']}: {np.round(r['theta_step1'], 3).tolist()}" for r in out[:3]))
for r in out:
    if 'c2_signal_flip_step' in r:
        print(f"alpha={r['alpha']}: c2 post-signal pull flips at step {r['c2_signal_flip_step']} "
              f"(layer-1 state mass {r['state_mass_at_flip']}) | final noise/signal pull ratio "
              f"{r['final_pull_ratio']} | final state mass {r['final_state_mass']:.3f}")
json.dump(out, open(os.path.join(R, 'e21_gd_vs_adam.json'), 'w'), indent=1)
