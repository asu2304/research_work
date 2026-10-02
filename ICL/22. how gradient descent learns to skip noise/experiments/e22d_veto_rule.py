"""
E22d -- Find the rule for when the noise veto strengthens, from data.

At random points of E22's box record the sign of dL/da1 together with
    p, a1, a2, b2, c2, c2/b2,
    split = mean over scored positions of rtilde_j[pv(j)]: how much of layer 1's attention AMONG
            SIGNAL KEYS lands on the last signal token (depends on p only, by the gain lemma),
    edge  = mean m_j (1 - m_j): how many positions sit on the skip cliff (where the veto acts).
Fit a depth-3 decision tree predicting 'veto weakens' and print its rules, plus the same features
along the training runs for comparison.
"""
import glob, json, os, sys
import numpy as np, torch
from sklearn.tree import DecisionTreeClassifier, export_text
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
rng = np.random.default_rng(2)
FEAT = ['p', 'a1', 'a2', 'b2', 'c2', 'c2_over_b2', 'split', 'edge']


def feats(th0, x, ps):
    th = torch.tensor(th0, dtype=torch.float32, device=dev)
    with torch.no_grad():
        _, P = M.forward(th, x, parts=True)
        A1 = P['A1'][:, M.TS:]
        sig = (x < M.VS).unsqueeze(1).float()
        As = A1 * sig; ms = As.sum(-1)
        last = torch.zeros_like(A1)
        idx = torch.arange(M.T, device=dev)
        lastb = torch.where(x < M.VS, idx, torch.full_like(idx, -1)).cummax(1).values
        lb = torch.full_like(lastb, -1); lb[:, 1:] = lastb[:, :-1]
        q = lb[:, M.TS:]; ok = (q >= 0) & (ms > 1e-6)
        on_last = As.gather(-1, q.clamp_min(0).unsqueeze(-1)).squeeze(-1) / ms.clamp_min(1e-30)
        m = 1 - ms
    return [th0[0], th0[1], th0[2], th0[3], th0[4], th0[4] / max(th0[3], 1e-3),
            float(on_last[ok].mean()), float((m * (1 - m))[ok].mean())]


out = {}
for a in (0.4, 0.6, 0.8):
    tr = M.load(a, 'train'); x, y = tr['x'][:2000].to(dev), tr['y'][:2000].to(dev)
    X, Y = [], []
    for _ in range(1200):
        th0 = np.array([rng.uniform(0, 8), rng.uniform(-30, 0), rng.uniform(-10, 0),
                        rng.uniform(0, 15), rng.uniform(0, 10)])
        th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        g, = torch.autograd.grad(M.ce(th, x, y, a), th)
        if abs(float(g[1])) < 1e-9:
            continue
        X.append(feats(th0, x, None)); Y.append(int(float(g[1]) <= 0))      # 1 = veto weakens
    X, Y = np.array(X), np.array(Y)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X, Y)
    acc = float(tree.score(X, Y))
    rules = export_text(tree, feature_names=FEAT, decimals=3)
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    traj = {k: dict(zip(FEAT, np.round(feats(TH[k], x, None), 3).tolist())) for k in (5, 20, 100, 1000)}
    out[str(a)] = dict(n=len(Y), frac_weaken=float(Y.mean()), tree_accuracy=acc, rules=rules, trajectory=traj)
    print(f"\n=== alpha={a}: {len(Y)} points, veto weakens at {100*Y.mean():.1f}% | depth-3 tree accuracy {100*acc:.1f}%")
    print(rules)
    print('   along Adam training: ' + ' | '.join(f"k={k}: split {v['split']}, c2/b2 {v['c2_over_b2']}, edge {v['edge']}" for k, v in traj.items()), flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e22d_veto_rule.json'), 'w'), indent=1)
