"""ANCHOR metric: Hessian spectrum of the suffix-100 CE loss in the COMBINED vs
SEPARATE parametrization, at the same function point.
  - lam_max, lam_min (power iteration via HVP)
  - negative-curvature census over random directions (Rayleigh quotients)
  - Hutchinson trace
Evaluated at the GD solution (c=1.0) and an off-solution point (params*c, c=0.7)
where the gradient is nonzero so the factorization's bilinear cross-term is active.
"""
import sys; sys.path.append('/home/hariguru/ashutosh/ICL/_landscape_scratch')
from landscape_lib import *
import torch

dev=pick_device(); print('device',dev, flush=True)

def scaled_combined(thr, c):
    m=load_gd(thr,dev)
    if c!=1.0:
        with torch.no_grad():
            for p in m.parameters(): p.mul_(c)
    return m

def report(tag, model, x, y, V):
    g=grad_vec(model,x,y,V); gn=_norm(g)
    lt=top_eig(model,x,y,V,dev,iters=30)
    lm=min_eig(model,x,y,V,dev,lt,iters=30)
    rs=rayleigh_stats(model,x,y,V,K=24)
    print(f"  [{tag:9s}] CE={ce(model,x,y,V):.4f}  |grad|={gn:.3e}  "
          f"lam_max={lt:+.3e}  lam_min={lm:+.3e}  "
          f"frac_neg={rs['frac_neg']:.2f}  ray_mean={rs['mean']:+.2e}  "
          f"ray_p10={rs['p10']:+.2e}  trace={rs['trace_hutch']:+.3e}", flush=True)
    return dict(gn=gn,lmax=lt,lmin=lm,**rs)

for thr in [0.9, 0.5]:
    x,y,V=load_eval(thr,128); x=x.to(dev); y=y.to(dev)
    print(f"\n===== alpha={thr} =====", flush=True)
    for c in [1.0, 0.7]:
        dtc=scaled_combined(thr,c); dts=factorize(dtc,dev)
        print(f" point c={c} (c=1.0 is the GD solution):", flush=True)
        report('COMBINED', dtc, x, y, V)
        report('SEPARATE', dts, x, y, V)
        del dtc, dts
print("\nDONE", flush=True)
