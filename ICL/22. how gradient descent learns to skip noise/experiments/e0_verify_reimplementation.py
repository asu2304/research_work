"""
E0 -- does src/five.py reproduce folder 19 exactly?

Checks, at every alpha folder 19 trained:
  1. test CE of the folder-19 final checkpoint, through folder 19's OWN model class
     vs through src/five.py's direct five-scalar forward;
  2. the gradient stored in folder 19's snapshots (train set, full batch) at steps 0, 1, 50, 199
     vs autograd through src/five.py.
Imports folder 19's script read-only (nothing there is modified or written).
"""
import importlib.util, os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
spec = importlib.util.spec_from_file_location(
    'f19', os.path.join(M.ICL, '19. reducing even more', 'train_five_params.py'))
f19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f19)

out = []
for a in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    te = M.load(a, 'test'); tr = M.load(a, 'train')
    xte, yte = te['x'].to(dev), te['y'].to(dev)
    sd = torch.load(os.path.join(M.RUN19, f'model_final_seed0_x_{M.afmt(a)}.pth'), map_location=dev)
    ref = f19.ReducedTransformer(4, 4, 100).to(dev); ref.load_state_dict(sd); ref.eval()
    with torch.no_grad():
        ce_ref = f19.eval_ce(ref, xte, yte, dev, f19.readout_bias(a, dev))
        th = M.f19_final(a).to(dev)
        ce_new = float(M.ce(th, xte, yte, a))
    W, G = M.f19_snapshots(a)
    xtr, ytr = tr['x'].to(dev), tr['y'].to(dev)
    gerr = []
    for k in (0, 1, 50, len(G) - 1):
        t = torch.tensor(W[k], dtype=torch.float32, device=dev, requires_grad=True)
        L = M.ce(t, xtr, ytr, a); g, = torch.autograd.grad(L, t)
        g = g.double().cpu().numpy(); s = G[k]
        rel = float(abs(g - s).max() / (abs(s).max() + 1e-30))
        gerr.append((k, rel))
    r = dict(alpha=a, ce_f19_class=ce_ref, ce_direct=ce_new, abs_diff=abs(ce_ref - ce_new),
             theta_final=[round(float(v), 4) for v in th.cpu()], grad_rel_err=gerr)
    out.append(r)
    print(f"alpha={a}  CE f19={ce_ref:.6f}  direct={ce_new:.6f}  |diff|={abs(ce_ref-ce_new):.1e}  "
          f"grad rel.err {['%d:%.1e' % e for e in gerr]}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e0_verify.json'), 'w'), indent=2)
