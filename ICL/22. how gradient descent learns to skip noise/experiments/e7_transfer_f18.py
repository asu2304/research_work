"""
E7 -- TRANSFER TEST: does the five-number model's state-weight story also happen in folder 18's
10,060-entry model, where every entry moves on its own (no tying, free position block, full 4x4
state block with off-diagonals)?

Folder 18 saved its weights at every step, so nothing is retrained. At log-spaced steps:
  0. check: my full-batch gradient at w[k] equals folder 18's stored g[k];
  1. split the gradient of the 4x4 STATE block W2[108:112, 108:112] by query type
     (post-noise vs post-signal), and read it along two directions, in logit units:
        diag      = the five-number c2 direction (raise all 4 diagonal entries)
        contrast  = raise diagonal, lower off-diagonal (gauge-robust version)
  2. the state block's own value, as the same two summaries;
  3. layer-1 state mass (is layer 1 skipping noise yet?).
The five-number prediction: early both types push the state block up; once layer 1 localises,
post-signal flips to pushing down; then the two balance.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M
import f18

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
STEPS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 499]
C = slice(f18.D1, f18.D1 + M.VS)
I4 = torch.eye(M.VS, device=dev)
U_diag = I4
U_con = I4 - (1 - I4) / (M.VS - 1)


def along(G, U):                       # derivative along direction U, logit units
    return float((G * U).sum()) * M.SQ


out = {}
for a in (0.6, 0.8, 1.0):
    S = f18.snapshots(a)
    tr = M.load(a, 'train')
    x, y = tr['x'].to(dev), tr['y'].to(dev)
    xs = x[:, M.TS:]
    wn, ws = (xs >= M.VS).float(), (xs < M.VS).float()
    N = float(xs.numel())
    ps = M.prev_signal(x)[:, M.TS:]
    rows = []
    t0 = time.time()
    for k in STEPS:
        W1 = torch.tensor(f18.full(S['w1'][k], S['m1']), device=dev, requires_grad=True)
        W2 = torch.tensor(f18.full(S['w2'][k], S['m2']), device=dev, requires_grad=True)
        row = dict(step=k)
        # 0. total gradient vs stored
        L = f18.ce(W1, W2, x, y, a)
        g1, g2 = torch.autograd.grad(L, (W1, W2))
        s1 = torch.tensor(f18.full(S['g1'][k], S['m1']), device=dev)
        s2 = torch.tensor(f18.full(S['g2'][k], S['m2']), device=dev)
        # folder 18 stores only the TRAINABLE entries' gradient, so compare inside the masks
        m1 = torch.tensor(S['m1'], device=dev); m2 = torch.tensor(S['m2'], device=dev)
        row['grad_check_rel_err'] = float(max((g1 * m1 - s1).abs().max() / (s1.abs().max() + 1e-30),
                                              (g2 * m2 - s2).abs().max() / (s2.abs().max() + 1e-30)))
        # 1. split by query type
        for name, w in (('noise', wn), ('signal', ws)):
            if w.sum() == 0:
                row[f'pull_diag_{name}'] = row[f'pull_contrast_{name}'] = 0.0
                continue
            Lt = f18.ce(W1, W2, x, y, a, weights=w) * (float(w.sum()) / N)
            _, gt = torch.autograd.grad(Lt, (W1, W2))
            Gc = gt[C, C]
            row[f'pull_diag_{name}'] = -along(Gc, U_diag)            # minus: descent direction
            row[f'pull_contrast_{name}'] = -along(Gc, U_con)
        # 2. the block's value (logits)
        Wc = W2.detach()[C, C] / M.SQ
        off = ~torch.eye(M.VS, dtype=torch.bool, device=dev)
        row['state_diag_mean'] = float(torch.diag(Wc).mean())
        row['state_contrast'] = float(torch.diag(Wc).mean() - Wc[off].mean())
        # 3. layer-1 state mass
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for i in range(0, x.shape[0], 2500):
                _, P = f18.forward(W1.detach(), W2.detach(), x[i:i + 2500], parts=True)
                r = P['r'][:, M.TS:]; p_ = ps[i:i + 2500]; ok = p_ >= 0
                tot += float(r.gather(-1, p_.clamp_min(0).unsqueeze(-1)).squeeze(-1)[ok].sum()); cnt += int(ok.sum())
        row['layer1_state_mass'] = tot / cnt
        with torch.no_grad():
            row['test_ce'] = float(f18.ce(W1.detach(), W2.detach(), M.load(a, 'test')['x'].to(dev),
                                          M.load(a, 'test')['y'].to(dev), a))
        rows.append(row)
        print(f"alpha={a} step {k:>3}: state contrast {row['state_contrast']:+6.2f} | pull(noise) "
              f"{row['pull_contrast_noise']:+.2e}  pull(signal) {row['pull_contrast_signal']:+.2e} | "
              f"L1 state mass {row['layer1_state_mass']:.3f} | grad check {row['grad_check_rel_err']:.1e}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    out[str(a)] = rows
    json.dump(out, open(os.path.join(M.F22, 'runs', 'e7_transfer_f18.json'), 'w'), indent=1)
