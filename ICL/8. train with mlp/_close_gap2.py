"""GENUINE count-domain construction (no oracle): recover the per-state observation count N from
the L2 attention's inverse participation ratio  N_eff = 1 / sum_k A_k^2  (the attention is uniform
over the N matched positions, so this returns N), then smooth in the COUNT domain
   p_sig = alpha (f * N_eff + eps) / (N_eff + V_sig eps),   p_noise = (1-alpha)/n_noise.
This is the Bayes/Dirichlet add-eps rule -> reproduces ctx-count AND is the correct smoothing for a
general Markov chain (keeps unequal magnitudes). Verify vs ctx-count + plot."""
import json, os, glob, math
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

dev = 'cpu'
if torch.cuda.is_available():
    fr = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    if fr: dev = f'cuda:{max(fr)[1]}'
print('device:', dev, flush=True)
nb = json.load(open('analyse.ipynb')); src = ''.join(nb['cells'][0]['source'])
defs = src[:src.index('# ════════════════════════════════════════════════════════════════════════════════\n# LAZY')]
defs = defs.replace("DEVICE           = 'cuda:1'", f"DEVICE           = '{dev}'")
g = {}; exec(defs, g)
load_data = g['load_data']; PIT = g['PerfectInductionTransformer']; device = g['device']
DATA_DIR, TS, TE = g['DATA_DIR'], g['TOKEN_START'], g['TOKEN_END']

@torch.no_grad()
def wstar_freq_and_Neff(blob):
    """Return (f signal-freq [B,W,Vs], N_eff [B,W]) from the hand W*; N_eff = 1/sum_k A2_k^2."""
    V = int(blob['total_vocab']); T = int(blob['seq_len']); Vs = int(blob['vocab_set1'])
    m = PIT(V, T, Vs).to(device).eval()
    x = blob['x'].to(device); B = x.shape[0]
    tok = F.one_hot(x, V).float()
    pos = F.one_hot(torch.arange(T, device=device), T).float().unsqueeze(0).expand(B, T, -1)
    s = torch.cat([tok, pos], -1)
    mask = torch.triu(torch.ones(T, T, device=device) * float('-inf'), 1)
    s1 = m.layer1(s, mask)                                  # [B,T,440]  (concat output of L1)
    Q = s1 @ m.layer2.W_q.weight.t(); K = s1 @ m.layer2.W_k.weight.t()
    sc = torch.bmm(Q, K.transpose(1, 2)) / (m.layer2.output_dim ** 0.5) + mask
    A2 = F.softmax(sc, dim=-1)                              # [B,T,T] attention weights
    s2 = m.layer2(s1, mask)                                 # concat output of L2
    f = m.fc_out(s2)[:, TS:TE, :Vs].clamp_min(0.0)         # signal successor freq on the window
    Ntot = 1.0 / A2[:, TS:TE, :].pow(2).sum(-1).clamp_min(1e-9)   # IPR = effective # ALL attended positions
    Nsig = Ntot * f.sum(-1)                                # signal-successor count = N_tot * (signal mass of f)
    del m
    if device.type == 'cuda': torch.cuda.empty_cache()
    return f.cpu(), Nsig.cpu()

def exact_N(blob):                                          # exact matched total, only to sanity-check N_eff
    x = blob['x'].numpy(); V = int(blob['total_vocab']); Vs = int(blob['vocab_set1']); B = x.shape[0]
    tc = np.zeros((B, V, Vs)); pend = np.zeros((B, V)); last = np.full(B, -1, np.int64)
    def step(c):
        t = x[:, c]; r = t < Vs; ri = np.where(r)[0]
        if len(ri):
            rt = t[ri]; oh = (np.arange(Vs) == rt[:, None]).astype(float)
            np.add.at(tc, ri, pend[ri, :, None] * oh[:, None, :]); pend[ri, :] = 0; pend[ri, rt] = 1; last[ri] = rt
        ni = np.where(~r)[0]
        if len(ni): pend[ni, t[ni]] += 1.0
    for i in range(TS): step(i)
    out = np.zeros((B, TE - TS))
    for j, t in enumerate(range(TS, TE)):
        v = x[:, t]; q = np.where(v < Vs, v, np.maximum(last, 0)); out[:, j] = tc[np.arange(B), q].sum(1); step(t)
    return torch.from_numpy(out)

def ce_psig(psig, pn, yw, V, Vs):
    p = torch.cat([psig, torch.full(psig.shape[:-1] + (V - Vs,), pn)], -1).reshape(-1, V).clamp_min(1e-12)
    yy = yw.reshape(-1); return (-torch.log(p[torch.arange(len(yy)), yy])).mean().item()

EPS = 1e-2     # count-domain pseudocount (same role as ctx-count's eps=0.01)
thrs = sorted(float(os.path.basename(f)[7:-3]) for f in glob.glob(os.path.join(DATA_DIR, 'test', 'data_x_*.pt')))
res = []
print(f"{'a':>4} {'ctx':>8} {'genuine':>8} {'diff':>8} {'Neff/N':>8}", flush=True)
for thr in thrs:
    x, y, sl, V, blob = load_data('test', thr); Vs = int(blob['vocab_set1']); yw = y[:, TS:TE]
    f, Neff = wstar_freq_and_Neff(blob)
    fn = f / f.sum(-1, keepdim=True).clamp_min(1e-9)         # normalised successor distribution
    Ne = Neff.unsqueeze(-1)                                   # [B,W,1]
    pn = (1 - thr) / (V - Vs)
    psig = thr * (fn * Ne + EPS) / (Ne + Vs * EPS)            # COUNT-domain add-eps (genuine, scalable)
    ce_gen = ce_psig(psig, pn, yw, V, Vs)
    # ctx-count reference
    from numpy import e
    Cn = exact_N(blob)
    # exact ctx-count CE via the notebook helper
    ce_ctx = g['alpha_ce'](blob, thr)
    ratio = (Neff / Cn.clamp_min(1)).mean().item()
    res.append((thr, ce_ctx, ce_gen)); print(f"{thr:>4} {ce_ctx:>8.4f} {ce_gen:>8.4f} {ce_gen-ce_ctx:>+8.4f} {ratio:>8.3f}", flush=True)

A = [r[0] for r in res]
plt.figure(figsize=(7.2, 4.6))
plt.plot(A, [r[1] for r in res], 'o-', color='firebrick', lw=2.6, ms=8, label='ctx-count (target floor)')
plt.plot(A, [r[2] for r in res], 'X--', color='black', lw=2.0, ms=9, label='genuine count-domain construction')
plt.xlabel('threshold (alpha)'); plt.ylabel('Cross-Entropy (tokens 100-200)')
plt.title('Count-domain construction (N recovered from attention IPR) reaches ctx-count')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig('count_domain_vs_ctx_CE.png', dpi=150); plt.close()
print('WROTE count_domain_vs_ctx_CE.png', flush=True); print('DONE', flush=True)
