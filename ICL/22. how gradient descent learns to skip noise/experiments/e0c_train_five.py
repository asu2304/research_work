"""
Train the five-number model (src/five.py) exactly as folder 19 does: zero init, full batch,
Adam lr 1.0 on the RAW scalars (raw = logit * sqrt(108)), loss on tokens [50:100).
E0 verified the forward and gradient match folder 19 to ~1e-7.

Saves, per alpha, runs/<tag>/traj_x_<alpha>.npz with
    theta  (S+1, 5)   logits before step k+1 (theta[0] = init = 0)
    grad   (S, 5)     dL/d(logit) at theta[k]   (full train set)
    train  (S+1,)     train CE at theta[k]
    test   (S+1,)     test  CE at theta[k]
usage: python e0c_train_five.py --alphas 0.6 0.8 --steps 1000 --device cuda:2 --tag main
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

ap = argparse.ArgumentParser()
ap.add_argument('--alphas', type=float, nargs='+', required=True)
ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--lr', type=float, default=1.0,
                help='adam: lr on RAW values (folder-19 convention). sgd: lr in LOGIT units '
                     '(a step moves each logit by lr * dL/d(logit))')
ap.add_argument('--opt', choices=('adam', 'sgd'), default='adam',
                help='sgd = plain full-batch gradient descent, no momentum')
ap.add_argument('--log-every', type=int, default=100)
ap.add_argument('--device', default='cuda:2')
ap.add_argument('--tag', default='main')
ap.add_argument('--init', type=float, nargs=5, default=None,
                help='start from these five LOGITS instead of zero')
ap.add_argument('--freeze', type=str, default='',
                help='comma list of names among p,a1,a2,b2,c2 to hold fixed at their init')
ap.add_argument('--chunk', type=int, default=5000)
a = ap.parse_args()
dev = torch.device(a.device)
out_dir = os.path.join(M.F22, 'runs', a.tag); os.makedirs(out_dir, exist_ok=True)
frozen = [M.NAMES.index(n) for n in a.freeze.split(',') if n]

for alpha in a.alphas:
    f = os.path.join(out_dir, f'traj_x_{M.afmt(alpha)}.npz')
    if os.path.exists(f):
        print('[skip]', f); continue
    tr, te = M.load(alpha, 'train'), M.load(alpha, 'test')
    xtr, ytr = tr['x'].to(dev), tr['y'].to(dev)
    xte, yte = te['x'].to(dev), te['y'].to(dev)
    model = M.FiveParam(a.init).to(dev)
    if a.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    else:
        # raw = logit * SQ  =>  d/d(raw) = d/d(logit) / SQ, and a raw step moves the logit by 1/SQ of it,
        # so lr_raw = lr_logit * SQ^2 gives  delta(logit) = -lr_logit * dL/d(logit)
        opt = torch.optim.SGD(model.parameters(), lr=a.lr * M.SQ ** 2, momentum=0.0)
    S = a.steps
    TH = np.zeros((S + 1, 5)); GR = np.zeros((S, 5)); TR = np.zeros(S + 1); TE = np.zeros(S + 1)
    t0 = time.time()
    for k in range(S + 1):
        th = model.theta()
        TH[k] = th.detach().cpu().double().numpy()
        with torch.no_grad():
            TE[k] = float(M.ce(th.detach(), xte, yte, alpha, chunk=a.chunk))
        if k == S:
            with torch.no_grad():
                TR[k] = float(M.ce(th.detach(), xtr, ytr, alpha, chunk=a.chunk))
            break
        opt.zero_grad()
        L = M.ce(model.theta(), xtr, ytr, alpha, chunk=a.chunk)
        L.backward()
        TR[k] = float(L)
        GR[k] = (model.raw.grad * M.SQ).detach().cpu().double().numpy()   # dL/d(logit)
        if frozen:
            model.raw.grad[frozen] = 0.0
        opt.step()
        if not np.isfinite(TR[k]):
            print(f'  [diverged] alpha={alpha} step {k}', flush=True); break
        if k in (0, 1, 10) or k % a.log_every == 0:
            print(f'  alpha={alpha} step {k:>5} train {TR[k]:.5f} test {TE[k]:.5f} '
                  f'theta {np.round(TH[k], 3)}  ({time.time()-t0:.0f}s)', flush=True)
    np.savez(f, theta=TH, grad=GR, train=TR, test=TE, alpha=alpha, lr=a.lr, steps=S,
             frozen=np.array(frozen, dtype=int), opt=a.opt)
    print(f'[done] alpha={alpha} final theta {np.round(TH[-1], 4)} test {TE[-1]:.5f} '
          f'ideal {M.ideal_ce(alpha):.5f}  ({(time.time()-t0)/60:.1f} min) -> {f}', flush=True)
