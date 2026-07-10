"""
Shared infrastructure for comparing the loss-landscape geometry of two
parametrizations of the SAME disentangled transformer:

  COMBINED  parameters = (W_qk, W_ov) per layer  + fc_out          [class DTC]
  SEPARATE  parameters = (Wq, Wk, Wv, Wo) per layer + fc_out        [class DTS]

with  W_qk = Wqᵀ·Wk   and   W_ov = Wo·Wv   so the two compute the IDENTICAL
function.  `factorize(dtc) -> dts` produces a balanced (SVD) factorization of a
loaded combined checkpoint so both nets sit at the SAME point in function space.

Everything is on the suffix-100 softmax-CE objective the model was trained on.

Usage from an agent:
    import sys; sys.path.append('/home/hariguru/ashutosh/ICL/_landscape_scratch')
    from landscape_lib import *
    dev = pick_device()
    dtc = load_gd(0.9, dev)                 # combined, at the GD solution
    dts = factorize(dtc, dev)               # separate, same function
    x, y, V = load_eval(0.9, n=256); x=x.to(dev); y=y.to(dev)
    lam = top_eig(dts, x, y, V, dev); ...
"""
import os, math, copy, torch, torch.nn as nn, torch.nn.functional as F
import subprocess

F7     = '/home/hariguru/ashutosh/ICL/7. unique_Wqk_Wok_test.ipynb'
DATA   = '/home/hariguru/ashutosh/ICL/6. understanding what gradient descent is firuring out/dataset'
MODELS = os.path.join(F7, 'models')
SUF    = 100          # loss/eval window = last 100 tokens
ATTN   = 220

def pick_device():
    try:
        out = subprocess.run(['nvidia-smi','--query-gpu=memory.free',
                              '--format=csv,noheader,nounits'],
                             capture_output=True,text=True).stdout.split()
        return f'cuda:{max(range(len(out)), key=lambda i:int(out[i]))}'
    except Exception:
        return 'cpu'

# ───────────────────────── COMBINED (W_qk, W_ov) ─────────────────────────
class CAB(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.input_dim=i; s.output_dim=o
        s.W_qk=nn.Parameter(torch.empty(i,i)); s.W_ov=nn.Parameter(torch.empty(o,i))
        nn.init.xavier_uniform_(s.W_qk); nn.init.xavier_uniform_(s.W_ov)
    def forward(s,x,mask=None):
        sc=torch.bmm(x@s.W_qk, x.transpose(1,2))/(s.output_dim**0.5)
        if mask is not None: sc=sc+mask
        return torch.cat([x, torch.bmm(F.softmax(sc,-1), x@s.W_ov.t())],-1)

class DTC(nn.Module):
    """Combined parametrization (matches train_combined.py checkpoints)."""
    def __init__(s,V,T,ad=ATTN,bias=True):
        super().__init__(); s.total_vocab=V; s.seq_len=T; s.attn_dim=ad; s.initial_dim=V+T
        s.layer1=CAB(s.initial_dim,ad); s.layer2=CAB(s.initial_dim+ad,ad)
        s.fc_out=nn.Linear(s.initial_dim+2*ad,V,bias=bias)
    def forward(s,x):
        B,T=x.shape
        tok=F.one_hot(x,s.total_vocab).float()
        pos=F.one_hot(torch.arange(T,device=x.device),s.seq_len).float().unsqueeze(0).expand(B,T,-1)
        st=torch.cat([tok,pos],-1)
        mask=torch.triu(torch.ones(T,T,device=x.device)*float('-inf'),1)
        st=s.layer1(st,mask); st=s.layer2(st,mask); return s.fc_out(st)

# ───────────────────────── SEPARATE (Wq,Wk,Wv,Wo) ────────────────────────
# Faithful factored mirror of CAB:
#   scores = (x·(WqᵀWk))·xᵀ / √attn_dim      (FIXED √out scaling, as in CAB)
#   attn   = softmax(scores)·(x·(WoWv)ᵀ)
# r_qk / r_ov are the inner (head) ranks; set to the circuit's full rank so the
# factorization is EXACT for any trained combined checkpoint:
#   layer1: W_qk 220×220 (r_qk=220),  W_ov 220×220 (r_ov=220)
#   layer2: W_qk 440×440 (r_qk=440!), W_ov 220×440 (r_ov=220)
class _CA(nn.Module):
    def __init__(s, in_dim, out_dim, r_qk, r_ov):
        super().__init__(); s.scale=out_dim**0.5
        s.W_q=nn.Parameter(torch.empty(r_qk,in_dim)); s.W_k=nn.Parameter(torch.empty(r_qk,in_dim))
        s.W_v=nn.Parameter(torch.empty(r_ov,in_dim)); s.W_o=nn.Parameter(torch.empty(out_dim,r_ov))
        for p in (s.W_q,s.W_k,s.W_v,s.W_o): nn.init.xavier_uniform_(p)
    def forward(s,x,mask=None):
        Q=x@s.W_q.t(); K=x@s.W_k.t(); Vv=x@s.W_v.t()
        sc=torch.bmm(Q,K.transpose(1,2))/s.scale
        if mask is not None: sc=sc+mask
        o=torch.bmm(F.softmax(sc,-1),Vv)@s.W_o.t()
        return torch.cat([x,o],-1)

class DTS(nn.Module):
    """Separate parametrization: W_qk=W_qᵀW_k, W_ov=W_o·W_v (same function as DTC)."""
    def __init__(s,V,T,ad=ATTN,bias=True):
        super().__init__(); s.total_vocab=V; s.seq_len=T; s.attn_dim=ad; s.initial_dim=V+T
        d0=s.initial_dim                       # 220
        s.layer1=_CA(d0,     ad, r_qk=d0,   r_ov=ad)   # 220,220,220,220
        s.layer2=_CA(d0+ad,  ad, r_qk=d0+ad,r_ov=ad)   # in=440,out=220,r_qk=440,r_ov=220
        s.fc_out=nn.Linear(s.initial_dim+2*ad,V,bias=bias)
    def forward(s,x):
        B,T=x.shape
        tok=F.one_hot(x,s.total_vocab).float()
        pos=F.one_hot(torch.arange(T,device=x.device),s.seq_len).float().unsqueeze(0).expand(B,T,-1)
        st=torch.cat([tok,pos],-1)
        mask=torch.triu(torch.ones(T,T,device=x.device)*float('-inf'),1)
        st=s.layer1(st,mask); st=s.layer2(st,mask); return s.fc_out(st)

# ───────────────────────── loaders / factorization ───────────────────────
def load_eval(thr, n=256, split='test'):
    b=torch.load(os.path.join(DATA,split,f'data_x_{thr}.pt'),weights_only=False)
    return b['x'][:n], b['y'][:n], int(b['total_vocab'])

def seqT(thr):
    b=torch.load(os.path.join(DATA,'test',f'data_x_{thr}.pt'),weights_only=False)
    return int(b['seq_len']), int(b['total_vocab'])

def load_gd(thr, dev):
    T,V=seqT(thr)
    m=DTC(V,T,ATTN,bias=True).to(dev)
    m.load_state_dict(torch.load(os.path.join(MODELS,f'model_x_{thr}.pth'),map_location=dev,weights_only=True))
    return m.eval()

def _bal(M, k):
    """Balanced SVD factor (float64): return (L,R) with L@R = M (rank k)."""
    dt=M.dtype
    U,S,Vh=torch.linalg.svd(M.double(), full_matrices=False)
    U,S,Vh=U[:,:k],S[:k],Vh[:k,:]
    s=torch.sqrt(torch.clamp(S,min=0.0))
    return (U*s).to(dt), (s[:,None]*Vh).to(dt)   # L=U·diag(s), R=diag(s)·Vh

def factorize(dtc, dev):
    """Balanced (SVD) factorization of a combined model into the separate one,
    at FULL rank per circuit so the function is preserved exactly."""
    V,T=dtc.total_vocab, dtc.seq_len
    dts=DTS(V,T,ATTN,bias=True).to(dev)
    for lc, ls in [(dtc.layer1, dts.layer1), (dtc.layer2, dts.layer2)]:
        Wqk=lc.W_qk.data.to(dev); Wov=lc.W_ov.data.to(dev)
        # W_qk (in×in), full rank = in :  Wqᵀ·Wk = W_qk
        L,R=_bal(Wqk, Wqk.shape[0])                # L=[in,in], R=[in,in]
        ls.W_q.data = L.t().contiguous()           # [r_qk=in, in]  (Wqᵀ=L)
        ls.W_k.data = R.contiguous()               # [r_qk=in, in]
        # W_ov (out×in), full rank = out :  Wo·Wv = W_ov
        Lo,Ro=_bal(Wov, Wov.shape[0])              # Lo=[out,out], Ro=[out,in]
        ls.W_o.data = Lo.contiguous()              # [out, r_ov=out]
        ls.W_v.data = Ro.contiguous()              # [r_ov=out, in]
    dts.fc_out.weight.data = dtc.fc_out.weight.data.clone().to(dev)
    dts.fc_out.bias.data   = dtc.fc_out.bias.data.clone().to(dev)
    return dts.eval()

# ───────────────────────── loss / gradient / HVP ─────────────────────────
def loss_fn(model, x, y, V):
    lo=model(x)[:,-SUF:,:]; yw=y[:,-SUF:]
    return F.cross_entropy(lo.reshape(-1,V), yw.reshape(-1))

@torch.no_grad()
def ce(model, x, y, V):
    return loss_fn(model,x,y,V).item()

def params_of(model):
    return [p for p in model.parameters() if p.requires_grad]

def grad_vec(model, x, y, V):
    ps=params_of(model)
    g=torch.autograd.grad(loss_fn(model,x,y,V), ps, create_graph=False)
    return [gi.detach() for gi in g]

def hvp(model, x, y, V, vec):
    ps=params_of(model)
    loss=loss_fn(model,x,y,V)
    g=torch.autograd.grad(loss, ps, create_graph=True)
    gv=sum((gi*vi).sum() for gi,vi in zip(g,vec))
    Hv=torch.autograd.grad(gv, ps, retain_graph=False)
    return [h.detach() for h in Hv]

def _rand_like(ps): return [torch.randn_like(p) for p in ps]
def _dot(a,b): return float(sum((ai*bi).sum() for ai,bi in zip(a,b)))
def _norm(a):  return math.sqrt(sum((ai*ai).sum().item() for ai in a))
def _axpy(a,alpha,b): return [ai+alpha*bi for ai,bi in zip(a,b)]
def _scale(a,c): return [ai*c for ai in a]

def top_eig(model, x, y, V, dev, iters=40, tol=1e-4):
    ps=params_of(model); v=_rand_like(ps); v=_scale(v,1.0/_norm(v)); lam=0.0
    for _ in range(iters):
        Hv=hvp(model,x,y,V,v); nv=_norm(Hv)
        if nv==0: return 0.0
        new=_dot(Hv,v)
        v=_scale(Hv,1.0/nv)
        if abs(new-lam)<tol*max(1.0,abs(new)): lam=new; break
        lam=new
    return lam

def min_eig(model, x, y, V, dev, lam_top, iters=40, tol=1e-4):
    """Power-iterate on (shift·I - H); its top eigenvalue = shift - lam_min."""
    ps=params_of(model); v=_rand_like(ps); v=_scale(v,1.0/_norm(v)); mu=0.0
    shift=abs(lam_top)+1.0
    for _ in range(iters):
        Hv=hvp(model,x,y,V,v)
        w=[shift*vi - hi for vi,hi in zip(v,Hv)]   # (shift·I - H) v
        nv=_norm(w)
        if nv==0: break
        new=_dot(w,v); v=_scale(w,1.0/nv)
        if abs(new-mu)<tol*max(1.0,abs(new)): mu=new; break
        mu=new
    return shift-mu    # lam_min

def rayleigh_stats(model, x, y, V, K=64):
    """Distribution of vᵀHv/vᵀv over random unit directions -> curvature-sign census.
       Returns dict(frac_neg, mean, p10, p50, p90, trace_hutch)."""
    import numpy as np
    ps=params_of(model); vals=[]; tr=0.0
    for _ in range(K):
        v=_rand_like(ps); n2=sum((vi*vi).sum().item() for vi in v)
        Hv=hvp(model,x,y,V,v); r=_dot(Hv,v)/n2
        vals.append(r)
        z=[torch.randint(0,2,p.shape,device=p.device,dtype=p.dtype)*2-1 for p in ps]
        Hz=hvp(model,x,y,V,z); tr+=_dot(Hz,z)
    a=np.array(sorted(vals))
    return dict(frac_neg=float((a<0).mean()), mean=float(a.mean()),
                p10=float(np.percentile(a,10)), p50=float(np.percentile(a,50)),
                p90=float(np.percentile(a,90)), trace_hutch=tr/K, K=K)

def nparams(model): return sum(p.numel() for p in params_of(model))

# export underscore helpers too, so `from landscape_lib import *` brings them in
__all__=['pick_device','DTC','DTS','CAB','_CA','load_eval','seqT','load_gd',
         'factorize','loss_fn','ce','params_of','grad_vec','hvp','top_eig',
         'min_eig','rayleigh_stats','nparams','_norm','_dot','_scale','_axpy',
         '_rand_like','_bal','SUF','ATTN','F7','DATA','MODELS']

if __name__=='__main__':
    dev=pick_device(); print('device',dev)
    for thr in [0.9]:
        dtc=load_gd(thr,dev); dts=factorize(dtc,dev)
        x,y,V=load_eval(thr,128); x=x.to(dev); y=y.to(dev)
        with torch.no_grad():
            d=(dtc(x)-dts(x)).abs().max().item()
        print(f'thr={thr}  factorization max|logit diff|={d:.2e}  '
              f'CE_comb={ce(dtc,x,y,V):.5f}  CE_sep={ce(dts,x,y,V):.5f}')
        print(f'  n_params  combined={nparams(dtc):,}  separate={nparams(dts):,}')
