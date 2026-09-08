"""Layer-1 QK only: layer 2, both OV circuits and the readout are pinned to the hand construction.

The mirror image of train_fixing_qk_layer1_and_ov.py. There, layer 1 was handed to the model and
only the layer-2 induction match was learned. Here layer 2 is handed to the model and only the
layer-1 previous-signal-token head is learned.

Frozen in both architectures (identical circuits, verified at startup):
    layer 2  W_qk  = Wq2^T Wk2               48 non-zeros: 1000*sqrt(D) induction diagonal,
                                             50*sqrt(D) fallback, 10*sqrt(D)*gamma = -5.2e6
                                             noise kill
    layer 1  W_ov  = I_8 at [0:8, 0:8]       keep the token one-hot, drop position
    layer 2  W_ov  = I_8 at [0:8, 0:8]       copy the attended position's own token
    unembedding    = identity selector on the count slots, no bias
    readout        = z = log(f + b), bias BEFORE the log, no FFN, no residual

Trainable:
    separate : layer1.W_q, layer1.W_k       2 x (108 x 108)      = 23,328
    combined : layer1.W_qk                  108 x 108            = 11,664
                                                    
Note how this differs from the layer-2 experiment. For layer 2 the two parametrizations had EQUAL
parameter counts (46,656) but UNEQUAL expressivity (separate capped at rank 108 of 216). For layer
1, input_dim = output_dim = 108, so it is the other way round: separate carries 2x the parameters
but the two are EXACTLY equally expressive -- W_q^T W_k with inner dimension 108 spans every
108x108 matrix. This run therefore isolates a pure optimization-path question: given identical
reachable sets, do the factorized and the direct parametrization land in the same place?

Everything else -- data, hyperparameters, readout, logging, resume-safety -- is unchanged from
train_fixing_qk_layer1_and_ov.py.

Run in tmux:
    python -u train_freezing_Wqk2_and_Wov.py
    python -u train_freezing_Wqk2_and_Wov.py --check     # build check only, trains nothing
"""

import glob
import json
import math
import os
import sys
import tempfile
import time

_TMP = os.path.join(os.path.expanduser('~'), '.cache', 'ashu_tmp')
os.makedirs(_TMP, exist_ok=True)
for _v in ('TMPDIR', 'TMP', 'TEMP'):
    os.environ[_v] = _TMP
tempfile.tempdir = _TMP
os.environ.setdefault('MPLCONFIGDIR', _TMP)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    'data_dir': os.path.join(_SCRIPT_DIR, 'dataset'),
    'runs_dir': os.path.join(_SCRIPT_DIR, 'runs'),
    'device':   'cuda:1',

    'v_sig':       4,
    'n_noise':     4,
    'seq_len':     100,
    'token_start': 50,

    # ── readout: bias added BEFORE the log ────────────────────────────────────
    'b_sig':       0.01,        # pseudo-count for signal tokens
    'b_noise_rule': 'over_alpha',   # 'over_alpha' -> (1-a)/(4a)   [matches the construction]
                                    # 'plain'      -> (1-a)/4      [valid only if sum f_sig = a]
    'noise_floor': 1e-9,        # b_noise at alpha=1, where (1-a)/(4a) = 0
    'log_floor':   1e-30,       # hard guard so log() can never see 0

    'optimizer':   'adam',
    'lr':          0.005,
    'batch_size':  256,
    'init':        'normal',
    'weight_decay': 0.0,
    'eval_batch':  256,
    'epochs':      1000,
    'early_stop':  False,
    'patience':    50,
}

ARCHS      = ['separate', 'combined']
SEEDS      = [0]
THRESHOLDS = [0.4, 0.6, 0.8, 1.0]

CELL_PREFIX   = 'qk1only'
SKIP_DONE     = True
USE_WANDB     = True
WANDB_PROJECT = 'ICL-qk-layer1-only'
WANDB_ENTITY  = None

# construction constants
P1, P2, NF, GAMMA, BIG, ALPHA_REC = 1000.0, 50.0, 10.0, -50000.0, -1e9, 20.0


def cell_tag():
    return (f"{CELL_PREFIX}_opt-{CFG['optimizer']}_lr-{CFG['lr']}_b-{CFG['batch_size']}"
            f"_init-{CFG['init']}_ep-{CFG['epochs']}")


def noise_bias(alpha):
    if alpha >= 1.0:
        return CFG['noise_floor']
    if CFG['b_noise_rule'] == 'plain':
        return (1.0 - alpha) / CFG['n_noise']
    return (1.0 - alpha) * (1.0 + CFG['v_sig'] * CFG['b_sig']) / (CFG['n_noise'] * alpha)


def layer1_qk_construction(V, VS, T, dev):
    """The hand construction's layer-1 query/key factors, in nn.Linear [out, in] form.

    Not installed by this script -- layer 1 is the thing being trained here. Kept so the build
    check can report how far the random start is from the construction.
    """
    D = V + T
    Wq1 = torch.eye(D, device=dev) * math.sqrt(D)
    T1 = torch.zeros(D, D, device=dev)
    for k in range(VS, V):
        T1[k, :] = BIG
    for tq in range(T):
        for tk in range(T):
            T1[V + tk, V + tq] = BIG if tq == tk else -ALPHA_REC * (tq - tk)
    return Wq1, T1.t()


def layer2_qk_construction(V, VS, T, dev):
    """The hand construction's layer-2 query/key factors, in nn.Linear [out, in] form.

    Wq2 rows 0:4  -- 1000*sqrt(D) on the query's own signal token (priority 1) and 50*sqrt(D) on
                     the query's layer-1 predecessor (priority 2, used when the current token is
                     noise).
    Wq2 row 4     -- 10*sqrt(D) against every own-token coordinate: the trigger that pairs with
    Wk2 row 4     -- gamma = -5e4 on the key's own NOISE tokens, so the product carries
                     10*sqrt(D)*gamma = -5,196,152 there: never attend to a noise key.
    Wk2 rows 0:8  -- read the key's layer-1 predecessor identity.
    The sqrt(D) factors cancel the 1/sqrt(d) the attention applies to the scores.
    """
    D, IN2 = V + T, 2 * (V + T)
    SD = math.sqrt(D)
    Wq2 = torch.zeros(D, IN2, device=dev)
    Wq2[:VS, :VS] = torch.eye(VS, device=dev) * SD * P1
    Wq2[:VS, D:D + VS] = torch.eye(VS, device=dev) * SD * P2
    for v in range(V):
        Wq2[VS, v] = SD * NF
    Wk2 = torch.zeros(D, IN2, device=dev)
    Wk2[:V, D:D + V] = torch.eye(V, device=dev)
    for v in range(VS, V):
        Wk2[VS, v] = GAMMA
    return Wq2, Wk2


class SeparateAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x, mask=None):
        scores = torch.bmm(self.W_q(x), self.W_k(x).transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = self.W_o(torch.bmm(F.softmax(scores, dim=-1), self.W_v(x)))
        return torch.cat([x, attn_out], dim=-1)


class CombinedAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.empty(input_dim, output_dim))
        s = input_dim ** -0.5
        nn.init.uniform_(self.W_qk, -s, s)
        nn.init.uniform_(self.W_ov, -s, s)

    def forward(self, x, mask=None):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = torch.bmm(F.softmax(scores, dim=-1), x @ self.W_ov)
        return torch.cat([x, attn_out], dim=-1)


_BLOCKS = {'combined': CombinedAttentionBlock, 'separate': SeparateAttentionBlock}


class QKOnlyTransformer(nn.Module):
    def __init__(self, total_vocab, seq_len, attn_dim, v_sig, alpha, attn_param):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len     = seq_len
        self.v_sig       = v_sig
        self.attn_param  = attn_param
        self.initial_dim = total_vocab + seq_len
        Block = _BLOCKS[attn_param]
        self.layer1 = Block(self.initial_dim, attn_dim)
        self.layer2 = Block(self.initial_dim + attn_dim, attn_dim)
        self.D   = self.initial_dim + 2 * attn_dim
        self.in2 = self.initial_dim + attn_dim
        self.fc_out = nn.Linear(self.D, total_vocab, bias=False)
        b = torch.full((total_vocab,), CFG['b_sig'])
        b[v_sig:] = noise_bias(alpha)
        self.register_buffer('readout_bias', b)

    def stream(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        return self.layer2(self.layer1(s, mask), mask)

    def forward(self, x):
        s = self.stream(x)
        sl = slice(self.in2, self.in2 + self.total_vocab)
        z = s.clone()
        z[..., sl] = torch.log((s[..., sl] + self.readout_bias).clamp_min(CFG['log_floor']))
        return self.fc_out(z)


def qk_params(model):
    """The LAYER-1 QK circuit only -- layer 2 is frozen to the construction here."""
    layer = model.layer1
    if hasattr(layer, 'W_qk'):
        return [layer.W_qk]
    return [layer.W_q.weight, layer.W_k.weight]


def install_fixed(model):
    """Layer-2 QK, both OV circuits and fc_out <- hand construction. Everything frozen except
    the layer-1 QK circuit.

    Combined stores W_qk / W_ov directly; separate stores the factors. Since
    W_qk = Wq^T Wk and W_ov = Wv^T Wo^T, installing the factors in one and their product in the
    other gives byte-identical operators.
    """
    V, VS, T_ = model.total_vocab, model.v_sig, model.seq_len
    dev = next(model.parameters()).device
    Wq2, Wk2 = layer2_qk_construction(V, VS, T_, dev)
    with torch.no_grad():
        if hasattr(model.layer2, 'W_qk'):
            model.layer2.W_qk.copy_(Wq2.t() @ Wk2)
        else:
            model.layer2.W_q.weight.copy_(Wq2)
            model.layer2.W_k.weight.copy_(Wk2)
        for layer in (model.layer1, model.layer2):
            if hasattr(layer, 'W_ov'):
                layer.W_ov.zero_()
                layer.W_ov[:V, :V] = torch.eye(V, device=dev)
            else:
                layer.W_v.weight.zero_()
                layer.W_v.weight[:V, :V] = torch.eye(V, device=dev)
                layer.W_o.weight.copy_(torch.eye(layer.output_dim, device=dev))
        model.fc_out.weight.zero_()
        for v in range(V):
            model.fc_out.weight[v, model.in2 + v] = 1.0
    for p in model.parameters():
        p.requires_grad_(False)
    ps = qk_params(model)
    for p in ps:
        p.requires_grad_(True)
    return ps


def apply_init(model, seed):
    """N(0, 1/sqrt(fan_in)) on the LAYER-1 QK matrices only -- layer 2 is the construction and
    must not be randomised. fan_in is shape[0] for combined (applied as x @ W) and shape[1] for
    separate (nn.Linear weight is [out, in])."""
    torch.manual_seed(seed)
    layer = model.layer1
    with torch.no_grad():
        if hasattr(layer, 'W_qk'):
            layer.W_qk.normal_(0.0, 1.0 / math.sqrt(layer.W_qk.shape[0]))
        else:
            for lin in (layer.W_q, layer.W_k):
                lin.weight.normal_(0.0, 1.0 / math.sqrt(lin.weight.shape[1]))


_DATA = {}

def load_data(split, thr):
    key = (split, thr)
    if key not in _DATA:
        if any(k[1] != thr for k in _DATA):
            _DATA.clear()
        _DATA[key] = torch.load(os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt'),
                                weights_only=False)
    return _DATA[key]


@torch.no_grad()
def eval_ce(model, x, y, V, device):
    model.eval()
    ts, te = CFG['token_start'], CFG['seq_len']
    total, n = 0.0, 0
    for i in range(0, x.shape[0], CFG['eval_batch']):
        xb = x[i:i + CFG['eval_batch']].to(device)
        yb = y[i:i + CFG['eval_batch'], ts:te].to(device)
        total += F.cross_entropy(model(xb)[:, ts:te].reshape(-1, V), yb.reshape(-1),
                                 reduction='sum').item()
        n += yb.numel()
    model.train()
    return total / n


@torch.no_grad()
def slot_stats(model, x, device, batch=256, n_max=2048):
    """How much attention mass reaches the signal slots, and how many are exactly zero.

    signal_mass -> 1 and frac_zero -> the construction's 0.27 is the signature of the whole
    circuit working. Here the only thing that can move it is layer 1: if the previous-token head
    does not point at the last SIGNAL token, layer 2 matches against the wrong predecessor.
    """
    was = model.training
    model.eval()
    V, VS = model.total_vocab, model.v_sig
    ts, te = CFG['token_start'], CFG['seq_len']
    sig = noi = zer = 0.0
    n = 0
    for i in range(0, min(x.shape[0], n_max), batch):
        f = model.stream(x[i:i + batch].to(device))[:, ts:te,
                                                    model.in2:model.in2 + V]
        sig += float(f[..., :VS].sum(-1).sum())
        noi += float(f[..., VS:].sum(-1).sum())
        zer += float((f[..., :VS] == 0).float().sum())
        n += f.shape[0] * f.shape[1]
    if was:
        model.train()
    return {'signal_mass': sig / n, 'noise_mass': noi / n, 'frac_zero': zer / (n * VS)}


_WANDB_OK = None

def wandb_start(run_cfg, name, group, tags):
    global _WANDB_OK
    if not USE_WANDB or _WANDB_OK is False:
        return None
    try:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group,
                         tags=tags, config=run_cfg, reinit='finish_previous',
                         dir=os.path.join(_SCRIPT_DIR, 'wandb_local'))
        _WANDB_OK = True
        return run
    except Exception as e:
        if _WANDB_OK is None:
            print(f'  [wandb] DISABLED -- {type(e).__name__}: {e}')
            try:
                import wandb as _wb
                api = _wb.Api()
                print(f'  [wandb] key belongs to : {api.viewer.username}')
                print(f'  [wandb] default entity : {api.default_entity}')
                print(f'  [wandb] teams          : {getattr(api.viewer, "teams", None)}')
                print(f'  [wandb] tried project  : {WANDB_PROJECT}  entity: {WANDB_ENTITY}')
                print('  [wandb] if the default entity is a team you cannot write to, set '
                      'WANDB_ENTITY to your own username and retry.')
            except Exception as e2:
                print(f'  [wandb] could not query the account ({type(e2).__name__}: {e2})')
            print('  [wandb] training continues; local json/checkpoints are unaffected.')
        _WANDB_OK = False
        return None


def metrics_path(arch, seed, thr):
    return os.path.join(CFG['runs_dir'], cell_tag(), f'metrics_{arch}_seed{seed}_x_{thr}.json')


def train_one(arch, seed, thr, device):
    V = CFG['v_sig'] + CFG['n_noise']
    ts, tend = CFG['token_start'], CFG['seq_len']
    tag = cell_tag()
    out_dir = os.path.join(CFG['runs_dir'], tag)
    os.makedirs(out_dir, exist_ok=True)
    stem = f'{arch}_seed{seed}_x_{thr}'
    mpath = os.path.join(out_dir, f'metrics_{stem}.json')
    cpath = os.path.join(out_dir, f'model_{stem}.pth')
    if SKIP_DONE and os.path.exists(mpath):
        print(f'  [skip done] {stem}')
        return False

    b_n = noise_bias(thr)
    run_cfg = {**{k: v for k, v in CFG.items() if k not in ('runs_dir', 'data_dir')},
               'attn_param': arch, 'seed': seed, 'threshold': thr, 'cell': tag,
               'b_noise': b_n, 'trainable': 'layer-1 QK circuit only',
               'succ_split': 'equiprobable'}

    torch.manual_seed(seed); np.random.seed(seed)
    tr_blob, te_blob = load_data('train', thr), load_data('test', thr)
    tr_x, tr_y = tr_blob['x'], tr_blob['y']
    te_x, te_y = te_blob['x'], te_blob['y']

    model = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'],
                              thr, arch).to(device)
    apply_init(model, seed)
    params = install_fixed(model)
    opt = torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=CFG['batch_size'], shuffle=True)

    wb = wandb_start(run_cfg, name=f'{tag}__{stem}', group=f'thr{thr}',
                     tags=[arch, f'thr{thr}', f'seed{seed}', 'qk1-only', 'direct-log'])

    tr0 = eval_ce(model, tr_x, tr_y, V, device)
    te0 = eval_ce(model, te_x, te_y, V, device)
    st0 = slot_stats(model, te_x, device)
    history = {'train_loss': [tr0], 'test_loss': [te0],
               'signal_mass': [st0['signal_mass']], 'frac_zero': [st0['frac_zero']]}
    if wb:
        for m in ('train_ce', 'test_ce'):
            wb.define_metric(m, summary='min')
        for m in ('best_train', 'best_test'):
            wb.define_metric(m, summary='last')
        wb.log({'train_ce': tr0, 'test_ce': te0, 'best_train': tr0, 'best_test': te0,
                **st0}, step=0)

    print(f'  {stem}  trainable={sum(p.numel() for p in params):,}  '
          f'b_sig={CFG["b_sig"]}  b_noise={b_n:.6f}')
    print(f'    init: tr={tr0:.4f} te={te0:.4f}  signal_mass={st0["signal_mass"]:.4f} '
          f'(target 1.0)  frac_zero={st0["frac_zero"]:.3f}', flush=True)

    best_train = float('inf'); test_at_best = None
    best_test = te0
    best_state = None; stale = 0
    t0 = time.time(); ep = 0
    interrupted_twice = False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb[:, ts:tend].to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(xb)[:, ts:tend].reshape(-1, V), yb.reshape(-1))
                loss.backward()
                opt.step()
                ep_loss += loss.item() * yb.numel(); ep_n += yb.numel()
            tr_loss = ep_loss / ep_n
            te_loss = eval_ce(model, te_x, te_y, V, device)
            st = slot_stats(model, te_x, device)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['signal_mass'].append(st['signal_mass'])
            history['frac_zero'].append(st['frac_zero'])

            if tr_loss < best_train - 1e-6:
                best_train = tr_loss; test_at_best = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            best_test = min(best_test, te_loss)

            if wb:
                wb.log({'train_ce': tr_loss, 'test_ce': te_loss,
                        'best_train': best_train, 'best_test': best_test, **st}, step=ep)
            if ep % 100 == 0:
                print(f'    ep {ep:5d}/{CFG["epochs"]}  tr={tr_loss:.4f}  te={te_loss:.4f}  '
                      f'best_tr={best_train:.4f}  best_te={best_test:.4f}  '
                      f'sig_mass={st["signal_mass"]:.4f}', flush=True)
            if CFG['early_stop'] and stale >= CFG['patience']:
                print(f'    [early stop] ep={ep}')
                break
    except KeyboardInterrupt:
        print('    [INTERRUPTED] saving best-so-far (Ctrl+C again within ~1s exits)', flush=True)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            interrupted_twice = True
            print('    [INTERRUPTED x2] finishing this run cleanly, then exiting', flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), cpath)
    st_end = slot_stats(model, te_x, device)
    result = {'cell': tag, 'arch': arch, 'seed': seed, 'threshold': float(thr),
              'history': history, 'best_train': best_train, 'test_at_best': test_at_best,
              'best_test': best_test,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'epochs_run': ep, 'time_s': round(time.time() - t0, 1), 'config': run_cfg,
              'slot_stats_best': st_end}
    with open(mpath, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'test_at_best': test_at_best,
                           'best_test': best_test, 'final_train': result['final_train'],
                           'final_test': result['final_test'], 'epochs_run': ep,
                           **{f'final_{k}': v for k, v in st_end.items()}})
        try:
            wb.finish()
        except Exception as e:
            print(f'    [wandb] finish failed ({type(e).__name__}: {e})', flush=True)
    print(f'    done in {result["time_s"]/60:.1f}m  best_train={best_train:.4f}  '
          f'best_test={best_test:.4f}  signal_mass={st_end["signal_mass"]:.4f}  -> {mpath}',
          flush=True)
    if interrupted_twice:
        raise KeyboardInterrupt
    return True


def rebuild_summary():
    rows = []
    for mp in sorted(glob.glob(os.path.join(CFG['runs_dir'], f'{CELL_PREFIX}_*',
                                            'metrics_*.json'))):
        m = json.load(open(mp))
        rows.append({k: m.get(k) for k in
                     ('cell', 'arch', 'seed', 'threshold', 'best_train', 'test_at_best',
                      'best_test', 'final_train', 'final_test', 'epochs_run')})
    out = os.path.join(CFG['runs_dir'], f'{CELL_PREFIX}_summary.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nSummary ({len(rows)} runs) -> {out}')


def check_dataset():
    for thr in THRESHOLDS:
        for split in ('train', 'test'):
            p = os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt')
            if not os.path.exists(p):
                sys.exit(f'missing {p}\nrun: python data_generation.py')
        if load_data('test', thr).get('succ_split') != 'equiprobable':
            sys.exit(f'{CFG["data_dir"]}/test/data_x_{thr}.pt is not the equiprobable dataset')
    _DATA.clear()


def build_check(device):
    """Assert the frozen circuits are correct and that a gradient reaches layer 1.

    The layer-2 construction is deliberately extreme (entries of +-5e6), so its softmax is nearly
    one-hot and a saturated softmax has near-zero derivative. Layer 1's only path to the loss runs
    through it. If the gradient norm is 0 the sweep would sit still for 500 epochs.
    """
    V, VS = CFG['v_sig'] + CFG['n_noise'], CFG['v_sig']
    Wq2c, Wk2c = layer2_qk_construction(V, VS, CFG['seq_len'], device)
    C2 = Wq2c.t() @ Wk2c
    Wq1c, Wk1c = layer1_qk_construction(V, VS, CFG['seq_len'], device)
    C1 = Wq1c.t() @ Wk1c

    thr = THRESHOLDS[1]
    blob = load_data('test', thr)
    xb = blob['x'][:8].to(device)
    yb = blob['y'][:8, CFG['token_start']:CFG['seq_len']].to(device)

    print('build check (nothing is trained here)')
    ok = True
    for a in ARCHS:
        m = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], VS, thr, a).to(device)
        apply_init(m, 0)
        ps = install_fixed(m)

        l1, l2 = m.layer1, m.layer2
        qk2 = (l2.W_qk if hasattr(l2, 'W_qk') else l2.W_q.weight.t() @ l2.W_k.weight).detach()
        ov2 = (l2.W_ov if hasattr(l2, 'W_ov')
               else l2.W_v.weight.t() @ l2.W_o.weight.t()).detach()
        ov1 = (l1.W_ov if hasattr(l1, 'W_ov')
               else l1.W_v.weight.t() @ l1.W_o.weight.t()).detach()
        qk1 = (l1.W_qk if hasattr(l1, 'W_qk') else l1.W_q.weight.t() @ l1.W_k.weight).detach()

        ov1_t = torch.zeros_like(ov1); ov1_t[:V, :V] = torch.eye(V, device=device)
        ov2_t = torch.zeros_like(ov2); ov2_t[:V, :V] = torch.eye(V, device=device)
        sel = torch.zeros_like(m.fc_out.weight)
        for v in range(V):
            sel[v, m.in2 + v] = 1.0

        loss = F.cross_entropy(m(xb)[:, CFG['token_start']:CFG['seq_len']].reshape(-1, V),
                               yb.reshape(-1))
        loss.backward()
        gn = math.sqrt(sum(float((p.grad ** 2).sum()) for p in ps))
        ok = ok and gn > 0

        print(f'  {a}')
        print(f'    trainable        : {[n for n, p in m.named_parameters() if p.requires_grad]}'
              f'  ({sum(p.numel() for p in ps):,})')
        print(f'    layer2 W_qk      : max|installed - constr| = '
              f'{float((qk2 - C2).abs().max()):.3e}')
        print(f'    layer1/2 W_ov    : max|installed - I8|     = '
              f'{max(float((ov1 - ov1_t).abs().max()), float((ov2 - ov2_t).abs().max())):.3e}')
        print(f'    fc_out           : max|installed - sel|    = '
              f'{float((m.fc_out.weight.detach() - sel).abs().max()):.3e}')
        print(f'    layer1 W_qk      : random start, ||.||_F = {float(qk1.norm()):.4g} '
              f'(construction {float(C1.norm()):.4g})')
        print(f'    loss @ init      : {float(loss):.4f}')
        print(f'    grad norm        : {gn:.4e}   '
              f'{"ALIVE" if gn > 0 else "*** DEAD -- do not launch ***"}')
        del m
    return ok


def main():
    device = torch.device(CFG['device'] if torch.cuda.is_available() else 'cpu')
    check_dataset()

    if '--check' in sys.argv:
        build_check(device)
        return

    runs = [(arch, seed, thr) for thr in THRESHOLDS for arch in ARCHS for seed in SEEDS]
    n_done = sum(1 for r in runs if os.path.exists(metrics_path(*r)))

    V = CFG['v_sig'] + CFG['n_noise']
    counts, ref = {}, None
    for a in ARCHS:
        probe = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'], 0.6, a)
        apply_init(probe, 0)
        counts[a] = (sum(p.numel() for p in install_fixed(probe)),
                     sum(p.numel() for p in probe.parameters()))
        l2 = probe.layer2
        eff = (l2.W_qk if hasattr(l2, 'W_qk')
               else l2.W_q.weight.t() @ l2.W_k.weight).detach()
        if ref is None:
            ref = eff
        else:
            assert torch.equal(eff, ref), 'layer-2 QK differs between architectures'
        del probe

    print(f'device     : {device}')
    print(f'data       : {CFG["data_dir"]}  [equiprobable]')
    print(f'cell       : {cell_tag()}')
    print(f'layer 2 QK : frozen to the construction, identical in both archs '
          f'(nnz {int((ref != 0).sum())}, min {float(ref.min()):.3g}, '
          f'max {float(ref.max()):.3g})')
    print(f'model      : LAYER-1 QK trainable, layer 2 + OV + unembedding frozen')
    for a in ARCHS:
        print(f'               {a:9s} {counts[a][0]:,} of {counts[a][1]:,} params')
    print(f'             counts differ by design: layer 1 is 108->108, so separate holds the two')
    print(f'             factors W_q, W_k while combined holds one 108x108 product -- but the')
    print(f'             reachable sets are identical (inner dim 108 spans every 108x108)')
    print(f'readout    : z = log(f + b),  bias BEFORE the log, no FFN, no residual')
    print(f'             b_sig = {CFG["b_sig"]}   b_noise rule = {CFG["b_noise_rule"]}')
    for t in THRESHOLDS:
        print(f'               alpha={t}: b_noise = {noise_bias(t):.6f}')
    print(f'optimizer  : {CFG["optimizer"]} lr={CFG["lr"]} b={CFG["batch_size"]} '
          f'init={CFG["init"]} epochs={CFG["epochs"]} early_stop={CFG["early_stop"]}')
    print(f'archs      : {ARCHS}   thresholds: {THRESHOLDS}   seeds: {SEEDS}')
    print(f'runs       : {len(runs)} ({n_done} done, will skip)')
    print(f'wandb      : {WANDB_PROJECT if USE_WANDB else "OFF"}\n')

    os.makedirs(CFG['runs_dir'], exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'wandb_local'), exist_ok=True)
    t_start = time.time(); n_run = 0
    try:
        for i, (arch, seed, thr) in enumerate(runs):
            if train_one(arch, seed, thr, device):
                n_run += 1
                avg = (time.time() - t_start) / n_run
                left = sum(1 for r in runs[i + 1:] if not os.path.exists(metrics_path(*r)))
                print(f'  [{i + 1}/{len(runs)}]  avg={avg/60:.1f}m/run  '
                      f'ETA={left * avg / 3600:.1f}h\n', flush=True)
    except KeyboardInterrupt:
        print('\n[sweep stopped by Ctrl+C]', flush=True)
    rebuild_summary()


if __name__ == '__main__':
    main()
