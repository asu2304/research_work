"""
Test the user's hypothesis: a structure-AWARE 'distinct-set, equal-weight'
predictor vs the frequency-counting ctx-count baseline.

Both know alpha (=threshold). The ONLY difference:
  - ctx-count : P(j|s) ~ count[s][j] + eps        (frequency-weighted)
  - set-equal : alpha/|succ_set| on each distinct observed signal successor,
                (1-alpha)/10 on each noise token   (uniform over the 2)
  - ideal     : alpha/2 on the TRUE 2 successors  (oracle lower bound)

Reports CE on:
  (a) all target positions
  (b) positions where BOTH true successors already appeared in context
      (the regime where set-equal can actually be right)
"""
import numpy as np, torch, math

VS, V = 10, 20
EPS_GRID = [0.01,0.03,0.05,0.1,0.2,0.3,0.5,1.0]   # tune ctx-count's smoothing
THRS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
N_SEQS = 1000
CLIP = 1e-12

def ce_for(thr):
    d = torch.load(f"dataset/test/data_x_{thr}.pt", weights_only=False)
    X = d['x'][:N_SEQS].numpy(); Y = d['y'][:N_SEQS].numpy()
    trans = d['transitions'][:N_SEQS]
    n, L = X.shape
    noise_floor = (1.0 - thr) / (V - VS)

    # accumulators: (sum nll, count) for each predictor, each subset
    keys = ['set','ideal'] + [f'ctx{e}' for e in EPS_GRID]
    acc = {k: {'all':[0.0,0], 'both':[0.0,0]} for k in keys}

    for b in range(n):
        x = X[b]; y = Y[b]; tr = trans[b]
        cnt = np.zeros((VS, V), dtype=np.float64)   # count[s][j]
        state = None                                # last_ruled
        for i in range(L):
            tgt = int(y[i])
            # ---- state after x[0..i] is `state` updated by x[i] ----
            xi = int(x[i])
            if state is not None and i >= 0:
                pass
            # update count/state with the transition that PRODUCED x[i]
            # (transition from prev-state -> x[i]); we update AFTER using,
            # but counts must reflect context 0..i, which includes x[i].
            # prev-state stored before this loop body updated state below.
            # We update state with x[i] first (x[i] is part of context for y[i]).
            if xi < VS:
                # x[i] is a signal: the transition (state_prev -> xi) is observed
                if state is not None:
                    cnt[state, xi] += 1
                state = xi
            else:
                # noise: transition (state_prev -> xi) observed, state unchanged
                if state is not None:
                    cnt[state, xi] += 1
                # state stays

            s = state
            # ---- build the three predictive distributions for state s ----
            preds = {}
            # noise floor lives ONLY on the 10 noise tokens; impossible
            # signals get 0 (true p* puts 0 on non-successor signals).
            def floor_vec():
                v = np.zeros(V); v[VS:] = noise_floor; return v
            if s is None:
                base = floor_vec(); base[:VS] = thr/VS
                for e in EPS_GRID: preds[f'ctx{e}'] = base
                preds['set'] = base; preds['ideal'] = base
                both = False
            else:
                row = cnt[s]
                # ctx-count: frequency over all tokens, Laplace eps (tuned).
                # (it CANNOT zero impossible signals -- that's its weakness)
                for e in EPS_GRID:
                    pc = row + e; preds[f'ctx{e}'] = pc / pc.sum()

                # set-equal: distinct observed SIGNAL successors, equal weight
                succ = np.where(row[:VS] > 0)[0]
                p_set = floor_vec()
                if len(succ) == 0:
                    p_set[:VS] = thr/VS
                else:
                    p_set[succ] = thr/len(succ)
                preds['set'] = p_set

                # ideal-alpha: true 2 successors, alpha/2 each
                a, bb = int(tr[s][0]), int(tr[s][1])
                p_ide = floor_vec()
                p_ide[a] += thr/2; p_ide[bb] += thr/2
                preds['ideal'] = p_ide

                both = (row[a] > 0 and row[bb] > 0)

            for key, p in preds.items():
                nll = -math.log(max(p[tgt], CLIP))
                acc[key]['all'][0] += nll; acc[key]['all'][1] += 1
                if both:
                    acc[key]['both'][0] += nll; acc[key]['both'][1] += 1

    out = {}
    for key in acc:
        out[key] = {sub:(acc[key][sub][0]/max(acc[key][sub][1],1))
                    for sub in ('all','both')}
    # pick best-eps ctx-count per subset
    ctx_all  = min(out[f'ctx{e}']['all']  for e in EPS_GRID)
    ctx_both = min(out[f'ctx{e}']['both'] for e in EPS_GRID)
    best_eps = min(EPS_GRID, key=lambda e: out[f'ctx{e}']['all'])
    frac_both = acc['set']['both'][1] / max(acc['set']['all'][1],1)
    return out, ctx_all, ctx_both, best_eps, frac_both

hdr = (f"{'a':>4} | {'ctx*_all':>8} {'set_all':>8} {'ide_all':>8} "
       f"| {'ctx*both':>8} {'set_both':>8} {'ide_both':>8} | eps* bothFr")
print(hdr); lines=[]
for thr in THRS:
    o, ca, cb, be, fb = ce_for(thr)
    line = (f"{thr:>4} | {ca:8.4f} {o['set']['all']:8.4f} {o['ideal']['all']:8.4f} "
            f"| {cb:8.4f} {o['set']['both']:8.4f} {o['ideal']['both']:8.4f} "
            f"| {be:4} {fb:5.3f}")
    print(line, flush=True); lines.append(line)

with open("models/setequal_test.log","w") as f:
    f.write("set-equal vs BEST-eps ctx-count vs ideal-alpha (CE, test)\n")
    f.write("ctx* = ctx-count with best Laplace eps per threshold\n")
    f.write("both = subset where BOTH true successors already seen in context\n")
    f.write(hdr+"\n"); f.write("\n".join(lines)+"\n")
print("done")
