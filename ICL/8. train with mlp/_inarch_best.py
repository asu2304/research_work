"""STRICTLY in-architecture hand construction (no IPR, no equalize, no architecture change):
attention W* -> GELU-FFN log readout -> linear -> softmax.
The GELU-FFN computes, per signal coord, log(f_i+d); a linear all-ones unit gives the signal sum
and another GELU computes log(sum); fc_out subtracts -> z_sig = log(f_i+d) - log(sum_j(f_j+d)) + log a,
z_noise = log((1-a)/n)  (a constant bias). softmax(z) is then the divide-by-sum freq distribution.
Everything is Linear/GELU/softmax -- exactly the trained model's ops. Search the floor d.
Compare to ctx-count (the floor) and to the GD-trained minimal-MLP (same architecture)."""
import json, os, glob, math
import numpy as np, torch

dev='cpu'
if torch.cuda.is_available():
    fr=[(torch.cuda.mem_get_info(i)[0],i) for i in range(torch.cuda.device_count())]
    if fr: dev=f'cuda:{max(fr)[1]}'
nb=json.load(open('analyse.ipynb')); src=''.join(nb['cells'][0]['source'])
defs=src[:src.index('# ════════════════════════════════════════════════════════════════════════════════\n# LAZY')]
defs=defs.replace("DEVICE           = 'cuda:1'", f"DEVICE           = '{dev}'")
g={}; exec(defs,g); print('device:',dev,flush=True)
load_data,_wlogits,alpha_ce,mlp_ce=g['load_data'],g['_wlogits'],g['alpha_ce'],g['mlp_ce']
TS,TE,DATA_DIR=g['TOKEN_START'],g['TOKEN_END'],g['DATA_DIR']

def ce_psig(psig,pn,yw,V,Vs):
    p=torch.cat([psig,torch.full(psig.shape[:-1]+(V-Vs,),pn)],-1).reshape(-1,V).clamp_min(1e-12)
    yy=yw.reshape(-1); return (-torch.log(p[torch.arange(len(yy)),yy])).mean().item()

DELTA=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]
thrs=sorted(float(os.path.basename(f)[7:-3]) for f in glob.glob(os.path.join(DATA_DIR,'test','data_x_*.pt')))
print(f"{'a':>4} {'in-arch':>8} {'best_d':>7} {'ctx':>8} {'GD-mlp':>8} {'inarch-ctx':>10} {'mlp-ctx':>8}",flush=True)
for thr in thrs:
    x,y,sl,V,blob=load_data('test',thr); Vs=int(blob['vocab_set1']); yw=y[:,TS:TE]
    f=_wlogits(blob)[...,:Vs].clamp_min(0.0)                 # W* signal successor freq
    pn=(1-thr)/(V-Vs)
    # in-architecture readout: softmax(log(f+d) normalised) + exact noise floor = divide-by-sum freq
    best=(1e9,None)
    for d in DELTA:
        psig=thr*(f+d)/(f+d).sum(-1,keepdim=True)            # = softmax of [log(f+d) - log sum] + log a
        ce=ce_psig(psig,pn,yw,V,Vs)
        if ce<best[0]: best=(ce,d)
    ce_in,d=best
    ce_ctx=alpha_ce(blob,thr); ce_mlp=mlp_ce(thr,sl,V,x,y)
    print(f"{thr:>4} {ce_in:>8.4f} {d:>7} {ce_ctx:>8.4f} {(ce_mlp or float('nan')):>8.4f} {ce_in-ce_ctx:>+10.4f} {((ce_mlp or float('nan'))-ce_ctx):>+8.4f}",flush=True)
print('DONE',flush=True)
