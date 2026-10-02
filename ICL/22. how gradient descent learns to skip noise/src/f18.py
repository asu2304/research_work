"""
Folder 18's 10,060-entry model, written as a function of its two W_qk matrices.

Same architecture as folder 18 `master_train.py` (blocks 'relevant', strict causal mask, W_ov =
identity on the 8x8 token corner and frozen, readout bias), and the same as five.py except that
every free entry moves on its own: no tying, a free 100x100 position block, 4x4 induction and
state blocks with off-diagonals, an 8x8 token block in each layer.

Weights are RAW entries (scores divide by sqrt(108)), exactly as stored in folder 18's snapshots.
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
import five as M

V, VS, T, D1, D2 = M.V, M.VS, M.T, M.V + M.T, 2 * (M.V + M.T)


def snapshots(alpha):
    """(W1 rows, W2 rows, G1 rows, G2 rows, masks): free entries per step, row-major in mask."""
    a = M.afmt(alpha)
    ld = lambda n: np.load(os.path.join(M.RUN18, n), mmap_mode='r')
    return dict(w1=ld(f'w_layer1.W_qk_x_{a}.npy'), w2=ld(f'w_layer2.W_qk_x_{a}.npy'),
                g1=ld(f'g_layer1.W_qk_x_{a}.npy'), g2=ld(f'g_layer2.W_qk_x_{a}.npy'),
                m1=np.load(os.path.join(M.RUN18, f'mask_layer1.W_qk_x_{a}.npy')),
                m2=np.load(os.path.join(M.RUN18, f'mask_layer2.W_qk_x_{a}.npy')))


def full(row, mask):
    W = np.zeros(mask.shape, dtype=np.float32); W[mask] = np.asarray(row); return W


def forward(W1, W2, x, parts=False):
    """W1 (108,108), W2 (216,216) raw torch tensors. Returns f (B,T,V) [and r]."""
    B, Tn = x.shape
    dev = x.device
    oh = F.one_hot(x, V).float()
    pos = torch.eye(T, device=dev)[:Tn].unsqueeze(0).expand(B, Tn, T)
    s = torch.cat([oh, pos], -1)                                    # (B, T, 108)
    mask = M.causal_mask(Tn, dev)
    A1 = torch.softmax((s @ W1) @ s.transpose(1, 2) / M.SQ + mask, -1)
    r = A1 @ oh                                                     # W_ov1 = I on 8x8
    s2 = torch.cat([s, F.pad(r, (0, D1 - V))], -1)                  # (B, T, 216)
    A2 = torch.softmax((s2 @ W2) @ s2.transpose(1, 2) / M.SQ + mask, -1)
    f = A2 @ oh                                                     # W_ov2 = I on 8x8
    return (f, dict(r=r, A1=A1, A2=A2)) if parts else f


def ce(W1, W2, x, y, alpha, chunk=2500, weights=None):
    b = M.readout_bias(alpha, x.device)
    n = x.shape[0] * (T - M.TS) if weights is None else float(weights.sum())
    tot = 0.0
    for i in range(0, x.shape[0], chunk):
        nll = M.token_nll(forward(W1, W2, x[i:i + chunk])[:, M.TS:], y[i:i + chunk, M.TS:], b)
        if weights is not None:
            nll = nll * weights[i:i + chunk]
        tot = tot + nll.sum()
    return tot / n
