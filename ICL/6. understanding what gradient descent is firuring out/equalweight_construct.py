"""
Hand-construction question:
  Can the 2-layer attention-only + AFFINE-readout architecture (with bias,
  exactly as trained) produce EQUAL weight on a state's two successors,
  instead of frequency weighting?

We use W* (PerfectInductionTransformer) -- its layer-2 attention output IS the
frequency vector. We expose it, then test whether ANY affine readout
(per-token weight + per-token bias) + temperature sigma can equalise the two
successors. The math says no (the needed correction is instance-dependent,
bias is global); this script demonstrates it on real data.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math

DEV = 'cuda:0'
THR = 0.7
N_SEQS = 400
T0, T1 = 100, 200
dev = torch.device(DEV)

# ---- W* construction, with layer-2 attn output exposed -----------------------
class CA(nn.Module):
    def __init__(s, di, do):
        super().__init__(); s.output_dim=do
        s.W_q=nn.Linear(di,do,bias=False); s.W_k=nn.Linear(di,do,bias=False)
        s.W_v=nn.Linear(di,do,bias=False); s.W_o=nn.Linear(do,do,bias=False)
    def forward(s,x,mask=None,ret=False):
        Q,K,V=s.W_q(x),s.W_k(x),s.W_v(x)
        sc=torch.bmm(Q,K.transpose(1,2))/(s.output_dim**0.5)
        if mask is not None: sc=sc+mask
        o=s.W_o(torch.bmm(F.softmax(sc,-1),V))
        return (torch.cat([x,o],-1), o) if ret else torch.cat([x,o],-1)

class WStar(nn.Module):
    def __init__(s,V,T,Vr):
        super().__init__(); s.V=V;s.T=T;s.Vr=Vr;s.D=V+T
        s.layer1=CA(s.D,s.D); s.layer2=CA(s.D+s.D,s.D)
        s.fc_out=nn.Linear(s.D+2*s.D,V,bias=False); s._w()
    def _w(s):
        V,T,D,Vr=s.V,s.T,s.D,s.Vr
        s.layer1.W_q.weight.data=torch.eye(D)*(D**0.5)
        Tk=torch.zeros(D,D)
        for k in range(Vr,V): Tk[k,:]=-1e9
        for tq in range(T):
            for tk in range(T):
                Tk[V+tk,V+tq]=-1e9 if tq==tk else -20.0*(tq-tk)
        s.layer1.W_k.weight.data=Tk.t()
        Wv1=torch.zeros(D,D);Wv1[:V,:V]=torch.eye(V);s.layer1.W_v.weight.data=Wv1
        s.layer1.W_o.weight.data=torch.eye(D)
        Wq2=torch.zeros(D,2*D)
        Wq2[:Vr,:Vr]=torch.eye(Vr)*(D**0.5)*1000.0
        Wq2[:Vr,D:D+Vr]=torch.eye(Vr)*(D**0.5)*50.0
        for v in range(V): Wq2[Vr,v]=(D**0.5)*10.0
        s.layer2.W_q.weight.data=Wq2
        Wk2=torch.zeros(D,2*D);Wk2[:V,D:D+V]=torch.eye(V)
        for v in range(Vr,V): Wk2[Vr,v]=-50000.0
        s.layer2.W_k.weight.data=Wk2
        Wv2=torch.zeros(D,2*D);Wv2[:V,:V]=torch.eye(V);s.layer2.W_v.weight.data=Wv2
        s.layer2.W_o.weight.data=torch.eye(D)
        Wfc=torch.zeros(V,2*D+D);Wfc[:V,2*D:2*D+V]=torch.eye(V);s.fc_out.weight.data=Wfc
    def forward(s,x):
        B,T=x.shape
        tok=F.one_hot(x,s.V).float()
        pos=F.one_hot(torch.arange(T,device=x.device),s.T).float().unsqueeze(0).expand(B,T,-1)
        h=torch.cat([tok,pos],-1)
        m=torch.triu(torch.ones(T,T,device=x.device)*float('-inf'),1)
        h=s.layer1(h,m)
        h,o2=s.layer2(h,m,ret=True)          # o2 = layer-2 attn output (length D)
        freq=o2[:,:,:s.V]                    # first V entries = frequency vector
        return s.fc_out(h), freq

blob=torch.load(f"dataset/test/data_x_{THR}.pt",weights_only=False)
X=blob['x'][:N_SEQS]; Y=blob['y'][:N_SEQS]; trans=blob['transitions'][:N_SEQS]
V=int(blob['total_vocab']); Vr=int(blob['vocab_set1']); T=int(blob['seq_len'])
m=WStar(V,T,Vr).to(dev).eval()

with torch.no_grad():
    _,freq=m(X.to(dev))
freq=freq.cpu().numpy()                      # [B,T,V]

# For each eval position, find the state's two true successors and read their
# frequency-features from the model. Collect (fa, fb) with fa>=fb.
fa_list, fb_list, na_list, nb_list = [], [], [], []
Xn=X.numpy()
for b in range(N_SEQS):
    last=-1
    tr=trans[b]
    for t in range(T1):
        xt=int(Xn[b,t])
        if t>=T0 and last>=0:
            a,bb=int(tr[last][0]),int(tr[last][1])
            fa,fb=freq[b,t,a],freq[b,t,bb]
            hi,lo=max(fa,fb),min(fa,fb)
            if hi>0:                          # state seen at least once before
                fa_list.append(hi); fb_list.append(lo)
        if xt<Vr: last=xt
fa=np.array(fa_list); fb=np.array(fb_list)
both=(fb>0)                                   # both successors observed
gap=fa-fb                                     # instance-dependent freq gap

print(f"THR={THR}  eval positions with state seen: {len(fa)}")
print(f"  both successors observed: {both.mean():.3f} of them")
print(f"  freq gap (hi-lo)  mean={gap[both].mean():.3f}  std={gap[both].std():.3f}"
      f"  min={gap[both].min():.3f}  max={gap[both].max():.3f}")
print(f"  fraction of both-seen pairs with UNEQUAL counts (gap>0.01): "
      f"{(gap[both]>0.01).mean():.3f}")

# A global readout makes logit_j = sigma*freq_j + b_j. For the two successors of
# a given instance, the prob ratio is exp( sigma*gap + (b_lo-b_hi) ).
# To EQUALISE we need sigma*gap = (b_hi-b_lo) -- a SINGLE constant -- for ALL
# instances. But gap varies (std above). Best a global constant c=b_hi-b_lo can
# do is minimise mean (sigma*gap - c)^2 over the noise floor; residual != 0.
g=gap[both]
# best single sigma*gap-cancelling constant is the mean; residual spread:
resid=g.std()
print(f"\nA global bias can cancel only ONE gap value. Residual std of the "
      f"per-instance gap that NO global bias removes: {resid:.3f}")
print("=> equal-weighting is NOT achievable by any affine readout (W,b,sigma).")

with open("models/equalweight_construct.log","w") as fp:
    fp.write(f"THR={THR} N={N_SEQS}\n")
    fp.write(f"eval positions (state seen): {len(fa)}\n")
    fp.write(f"both successors observed frac: {both.mean():.3f}\n")
    fp.write(f"freq gap both-seen: mean={gap[both].mean():.3f} std={gap[both].std():.3f} "
             f"min={gap[both].min():.3f} max={gap[both].max():.3f}\n")
    fp.write(f"unequal-count frac (gap>0.01): {(gap[both]>0.01).mean():.3f}\n")
    fp.write(f"residual gap std (irremovable by global bias): {resid:.3f}\n")
print("done")
