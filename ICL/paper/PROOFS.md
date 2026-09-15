# Gradient dynamics at zero initialisation — notation, proofs, and what remains

Working document for the ICLR/COLT paper. Written 2026-09-15.
Everything here is either **PROVED**, **VERIFIED NUMERICALLY** (with the measurement), or
**TO DO**. Each claim is labelled. Nothing is asserted without one of those three labels.

---

## 1. The task

Vocabulary `V = V_s + V_d`, split into

- **signal** symbols `S = {0, …, V_s−1}` — these carry the Markov state,
- **distractor** symbols `D = {V_s, …, V−1}` — these carry nothing.

For **each sequence independently** we draw a transition table `π`. For every state
`s ∈ S` it assigns a successor set `succ_π(s) ⊂ S` of size 2.

**How the successors are drawn** (this matters for the proof — checked in
`14. elimination final convergence confusion/data_generation.py`, line 70):

```python
order     = rng.random((B, v_sig, v_sig)).argsort(axis=2)
trans_arr = order[:, :, :2]          # first two of a random permutation of ALL of S
```

So `succ_π(v)` is a **uniform random 2-subset of all of `S`, including `v` itself**. Hence

> **Fact 1.** `P(w ∈ succ_π(v)) = 2/V_s` for **every** `w ∈ S`, including `w = v`.

*Verified numerically:* 3000 tables at `V_s = 4`, fraction of state-entries where the state is
its own successor = **49.95%**, against `2/4 = 50%`.

> **Fact 2.** The law of `π` is invariant under any relabelling (permutation) of `S`.

Generation, with `st(t)` the state after position `t`:

```
with probability α      : emit a uniform element of succ_π(st(t−1)),  state advances
with probability 1−α    : emit a uniform element of D,                state unchanged
```

Target `y_t = v_{t+1}`. Loss averaged over the scored window `T = [T/2, T)`.

**Why this task is "in-context".** Because `π` is redrawn each sequence, no function of the
weights alone beats the marginal. Everything achievable must be read off the current context.

---

## 2. Architecture and notation

No learned embedding. Position `t` is embedded as two concatenated one-hots:

```
x_t = [ e_{v_t} ; e_t ]  ∈ R^d ,        d = V + T
      └ symbol ┘ └ position ┘
```

Two attention layers, **strictly causal** mask (`u < t`, never `u = t`). Combined query–key
matrices `W⁽¹⁾ ∈ R^{d×d}` and `W⁽²⁾ ∈ R^{2d×2d}`. The value circuits are **fixed** to the
selector that passes the symbol one-hot.

```
Layer 1:   S⁽¹⁾_{t,u} = (1/√d) · x_t W⁽¹⁾ x_uᵀ  + M_{t,u}
           a⁽¹⁾_{t,·} = softmax( S⁽¹⁾_{t,·} )
           h_t        = Σ_u a⁽¹⁾_{t,u} · e_{v_u}            ∈ R^d , lives in the simplex

           x⁽²⁾_t = [ x_t ; h_t ] = [ e_{v_t} ; e_t ; h_t ]  ∈ R^{2d}

Layer 2:   S⁽²⁾_{t,u} = (1/√d) · x⁽²⁾_t W⁽²⁾ x⁽²⁾_uᵀ + M_{t,u}
           a⁽²⁾_{t,·} = softmax( S⁽²⁾_{t,·} )
           f_t        = Σ_u a⁽²⁾_{t,u} · e_{v_u}            ∈ R^V , in the simplex
```

**Important structural point.** `f_t` is built from `e_{v_u}` — the key's **own symbol** — not
from `h_u`. So `h` influences the loss **only through the attention scores `S⁽²⁾`**, never
through the value path. This is what makes Lemma 1(i) work.

Readout: a plain renormalisation against a fixed bias (no unembedding, no log, no softmax):

```
p_t = (f_t + b) / Σ_w (f_t[w] + b_w) ,     b_w = b_s  for w ∈ S
                                            b_w = b_d  for w ∈ D
```

Because `f_t` is a convex combination of one-hots, `Σ_w f_t[w] = 1` **identically**, so

```
Dnm := Σ_w (f_t[w] + b_w) = 1 + V_s·b_s + V_d·b_d          — a CONSTANT, context-independent
```

with `b_d = (1−α)(1 + V_s b_s) / (V_d α)`, derived in §8.

---

## 3. The state at zero initialisation

Set `W⁽¹⁾ = W⁽²⁾ = 0`. Then every score is `0`, so both softmaxes are uniform over the
admissible keys `{u < t}`:

```
a⁽¹⁾_{t,u} = a⁽²⁾_{t,u} = 1/t
```

and therefore

```
h_t = f_t = m_t := (1/t) Σ_{u<t} e_{v_u}
```

`m_t` is the **running symbol frequency** of the context. So **the model at zero init is the
in-context unigram predictor.**

*Verified numerically:* cross-entropy at init `= 2.039121` at `α = 0.6`, matching the closed form
for the unigram predictor.

> Note: this is exactly the "unigram phase" that Edelman et al. (2024) observe mid-training.
> We do not pass through it — we start there.

---

## 4. LEMMA 1 — the gradient at initialisation **[PROVED]**

> **Lemma 1.** At `W⁽¹⁾ = W⁽²⁾ = 0`, writing `ω_t := 1 / (m_t[y_t] + b_{y_t}) > 0`:
>
> **(i) Gating.**  `∂L/∂W⁽¹⁾ = 0` identically.
>
> **(ii) Closed form.**
> ```
> −∂L/∂W⁽²⁾ = (1 / (|T|·√d)) · Σ_{t∈T} (ω_t / t) · Σ_{u<t} ( 1[v_u = y_t] − m_t[y_t] ) · x⁽²⁾_t ⊗ x⁽²⁾_u
> ```

### 4.1 Proof of (ii)

**Step 1 — derivative of the loss w.r.t. `f`.**
Write `L_t = −log p_t[y_t]` with `p_t[w] = (f_t[w] + b_w)/Dnm`. Treating `f` as free,

```
g_t[w] := ∂L_t / ∂f_t[w] = − δ_{w, y_t} · ω_t  +  1/Dnm
```

The first term is from the numerator (only the target's coordinate appears), the second from the
denominator (every coordinate appears once).

**Step 2 — push through the attention.**
Since `f_t = Σ_u a⁽²⁾_{t,u} e_{v_u}`, we get `∂L_t / ∂a⁽²⁾_{t,u} = g_t[v_u]`. The softmax
Jacobian is `∂a_{t,u'} / ∂S_{t,u} = a_{t,u}(δ_{uu'} − a_{t,u'})`, so

```
∂L_t / ∂S⁽²⁾_{t,u} = a⁽²⁾_{t,u} · ( g_t[v_u] − Σ_{u'} a⁽²⁾_{t,u'} g_t[v_{u'}] )
                    = a⁽²⁾_{t,u} · ( g_t[v_u] − g_t · f_t )
```

**Step 3 — the bracket simplifies.**
Using `Σ_w f_t[w] = 1`:

```
g_t · f_t = Σ_w f_t[w] g_t[w] = − m_t[y_t]·ω_t + 1/Dnm
```

so the two `1/Dnm` terms cancel and

```
g_t[v_u] − g_t·f_t = − ω_t · ( 1[v_u = y_t] − m_t[y_t] )
```

**Step 4 — chain to the weights.**
`∂S⁽²⁾_{t,u} / ∂W⁽²⁾ = x⁽²⁾_t ⊗ x⁽²⁾_u / √d`, and at init `a⁽²⁾_{t,u} = 1/t`. Assembling and
flipping sign for the descent direction gives (ii). ∎

**Plain reading of (ii).** The coefficient on key `u` is positive exactly when the key's symbol
*is* the target. The first step says: **"attend more to positions showing the answer."**

### 4.2 Proof of (i)

`h` enters the loss only through `S⁽²⁾` (§2). Differentiating,

```
∂S⁽²⁾_{t,u} / ∂h_u = (1/√d) · ( x⁽²⁾_t W⁽²⁾ )_{d:2d}
∂S⁽²⁾_{t,u} / ∂h_t = (1/√d) · ( W⁽²⁾ x⁽²⁾_uᵀ )_{d:2d}
```

Both carry `W⁽²⁾` as a **multiplicative factor**, so both vanish at `W⁽²⁾ = 0`. Every path from
`W⁽¹⁾` to `L` runs through `h`, so `∂L/∂W⁽¹⁾ = 0`. ∎

*Verified numerically:* `‖∂L/∂W⁽¹⁾‖ = 0.000e+00` exactly (not small — zero), `α = 0.6`.

**Consequence.** The two layers **cannot form simultaneously**. Layer 2 must move first.

---

## 5. The block decomposition

Split `W⁽²⁾` into four `d × d` blocks matching `x⁽²⁾ = [x ; h]`:

```
W⁽²⁾ = [ P   Q ]          √d · S⁽²⁾_{t,u} =  x_t P x_uᵀ   (symbol × symbol)
       [ R   C ]                           + x_t Q h_uᵀ   (symbol × key's h)   ← INDUCTION
                                           + h_t R x_uᵀ   (cross)
                                           + h_t C h_uᵀ   (state × state)
```

Restrict to the signal sub-blocks and read off from Lemma 1(ii). For `v, w ∈ S`:

```
G_P[v,w] ∝ Σ_t 1[v_t=v] (ω_t/t) Σ_{u<t} ( 1[v_u=y_t] − m_t[y_t] ) · 1[v_u = w]

G_Q[v,w] ∝ Σ_t 1[v_t=v] (ω_t/t) Σ_{u<t} ( 1[v_u=y_t] − m_t[y_t] ) · m_u[w]
```

(The `x⁽²⁾_t ⊗ x⁽²⁾_u` outer product supplies `1[v_t=v]` on the query side; on the key side it
supplies `1[v_u=w]` for the `P` block and `h_u[w] = m_u[w]` for the `Q` block.)

**What each block would mean if it grew a positive diagonal:**

| block | rule it encodes | needs what to be true? |
|---|---|---|
| `G_P` | "when I am `v`, prefer keys whose **symbol** is `w`" | a **fixed** bigram table — stored in weights |
| `G_Q` | "when I am `v`, prefer keys whose **preceding context** looks like `w`" | nothing fixed — read from the context |

---

## 6. STATEMENT (a) — the bigram block **[TO PROVE]**

> **(a)**   `E[ G_P[v,v] ] − E[ G_P[v,w] ] ≤ 0`   for all `w ≠ v`.

Expectation over the table `π` and the sequence.

### 6.1 What is already clear

By **Fact 2** (relabelling symmetry), a permutation of `S` that fixes `v` and swaps `w ↔ w′` maps
the ensemble to itself. Therefore

```
E[ G_P[v,w] ] = E[ G_P[v,w′] ]      for all w, w′ ≠ v
```

so all off-diagonal entries in a row are equal. **This does not cover the diagonal**: no
permutation fixes `v` while moving `v`, so `E[G_P[v,v]]` is a genuinely separate quantity.

### 6.2 The two terms

```
G_P[v,w] = Σ_t 1[v_t=v] (ω_t/t) · [  1[w = y_t] · N_t(w)   −   m_t[y_t] · N_t(w)  ]
                                     └── term A ──┘           └──── term B ────┘
```
where `N_t(w) := #{u < t : v_u = w}` is the prefix count of `w`.

- **Term A.** By **Fact 1**, `E[ 1[w = y_t] | st(t)=v ] = α/V_s` for *every* `w` including
  `w = v`. So this term contributes the **same** to diagonal and off-diagonal.
- **Term B** is where the asymmetry lives. Conditioned on `v_t = v`, the prefix count `N_t(v)` is
  **elevated** relative to `N_t(w)` — the chain has just visited `v`, and states recur. The
  baseline subtraction `−m_t[y_t]·N_t(w)` is therefore **more negative** at `w = v`.

**So the expected conclusion is not "the diagonal is zero" but "the diagonal is strictly
negative".** The in-weights block is actively pushed *away* from the identity.

*Verified numerically* (`α = 0.6`, diag minus off-diag of the signal `4×4`):
```
G_P  (bigram)      −1.475e-04      ← negative
G_Q  (induction)   +9.621e-05      ← positive
G_C  (state×state) +9.804e-06      ← positive
```

### 6.3 What the proof needs

1. Show term A cancels exactly between diagonal and off-diagonal (immediate from Fact 1).
2. Bound `E[ m_t[y_t] · N_t(v) ] − E[ m_t[y_t] · N_t(w) ] > 0`, i.e. that the prefix count of the
   current state exceeds that of any other symbol in expectation. This is a statement about the
   stationary behaviour of the chain and should not need mixing — a direct computation of
   `P(v_s = v | v_t = v)` versus `P(v_s = w | v_t = v)` summed over `s < t`.

**Start here.** It is self-contained and will tell you within a few hours whether the algebra is
tractable.

---

## 7. STATEMENT (b) — the induction block **[TO PROVE — the hard one]**

> **(b)**   `E[ G_Q[v,v] ] − E[ G_Q[v,w] ] > 0`   for all `w ≠ v`.

### 7.1 The mechanism, in words

`G_Q[v,w]` weights each key `u` by `m_u[w]`, the frequency of `w` **in the key's own prefix**,
and rewards keys whose symbol is the target. Concretely, with `V_s = 4`:

```
position:  0   1   2   3   4   5   6
symbol:    0   5   2   6   0   4   2
state:     0   0   2   2   0   0   2
                ↑                   ↑
        positions where a successor of 0 appears
```

Suppose at time `t` the state is `v = 0` and the target is `y_t = 2`. The gradient rewards keys
with symbol `2` — positions 2 and 6. **Both were produced by a transition out of `0`.** That is
not coincidence, it is how the data is generated. So `m_u` at those keys carries extra mass on
`0`, which is `v` itself. **That is the diagonal signal.**

### 7.2 Reduction to a three-point correlation

Expanding `m_u[w] = (1/u) Σ_{s<u} 1[v_s = w]`, the quantity to bound is

```
E[ 1[v_u = y_t] · m_u[w] ]  =  (1/u) · Σ_{s<u} E[ 1[v_u = y_t] · 1[v_s = w] ]
```

with the ordering `s < u < t`. So (b) is a **three-point correlation** in the chain, averaged
over the random table.

### 7.3 Proposed proof structure

Split the inner sum at some lag `L`:

- **Distant terms (`s ≤ u − L`).** Show
  ```
  | P(v_s = w | v_u, v_t, y_t) − P(v_s = w) |  ≤  C · ρ^{u−s}
  ```
  so these contribute **equally to every `w`** and cancel in the diagonal-minus-off-diagonal
  contrast, leaving `O(ρ^L)`.
- **Recent terms (`s > u − L`).** Dominated by `s = ` the last signal position before `u`. A
  one-step Bayes computation gives a strictly positive constant on the diagonal, since
  occurrences of a successor of `v` are generated by transitions out of `v`.

### 7.4 The two technical obstacles

**(i) Mixing over a random 2-out graph.** The bound `ρ` is a mixing rate for the state chain. For
a *fixed* table the chain need not mix well — with `V_s = 4` you can draw tables with transient
states, so the stationary distribution is not uniform and `ρ` is not uniformly bounded. You need a
**high-probability statement over tables**, plus control of the exceptional set.

**(ii) The `1/u` dilution.** `m_u` averages the *whole* prefix, so the per-position diagonal boost
is `O(1/u)` while `m_u[w]` itself is `O(1/V_s)`. The contrast survives the sum over `t` and `u`,
but the constants should be explicit, or a referee will ask whether the effect is real at the
sequence lengths actually trained on (`T = 100`, scored on `[50,100)`).

---

## 8. The readout bias — derivation **[PROVED]**

Write `Dnm = Σ_w (f_t[w] + b_w)`. With a perfect distractor filter, `f_t` is supported on `S` and
sums to 1, so `Dnm = (1 + V_s b_s) + V_d b_d`, a **constant**. Demand each distractor receive
exactly `(1−α)/V_d`:

```
b_d / Dnm = (1−α)/V_d
  ⟹  V_d b_d = (1−α)·Dnm
  ⟹  Dnm = (1 + V_s b_s) + (1−α)·Dnm
  ⟹  Dnm = (1 + V_s b_s) / α
  ⟹  b_d = (1−α)(1 + V_s b_s) / (V_d α)
```

and the signal side then gets `p_t[w] = α(f_t[w] + b_s)/(1 + V_s b_s)` — exactly proportional to
the smoothed count. The `/α` is the part people drop: the renormalisation shrinks whatever you
put in.

**Relation to a log + softmax readout.** If the nonlinearity before an unembedding is
`z_w = log(f_t[w] + b_s)` on signal coordinates and a constant `β` on distractor coordinates, then
`softmax(z)_w = (f_t[w] + b_s)/Σ e^{z}`, which is the same renormalisation with `b_d = e^β`. The
log and the exponential cancel. *Verified numerically:* difference `5.6e-17`.

**Why the log is not optional.** Without it the softmax returns `e^{f_t[w]}` — exponential in the
count rather than linear — and no rescaling repairs it, because `softmax(κf)` is a different
function of `f` than `f/Σf` for every `κ`.

---

## 9. LEMMA 2 — layer 1 unlocks **[PROVED, modulo the Adam caveat]**

> **Lemma 2.** Let `W⁽²⁾_1 = η·G⁽²⁾` be one gradient step from Lemma 1, and `W⁽¹⁾_1 = 0` by
> Lemma 1(i). Then
>
> (i) layer 1 is unchanged, so `h_t = m_t` still;
> (ii) `a⁽²⁾_{t,u} = (1/t)(1 + (η/√d)(σ_{t,u} − σ̄_t)) + O(η²)` with
>      `σ_{t,u} = x⁽²⁾_t G⁽²⁾ x⁽²⁾_uᵀ`;
> (iii) `−∂L/∂W⁽¹⁾ |₁ = η·Φ(G⁽²⁾) + O(η²)`, where `Φ` is **linear in the blocks of `G⁽²⁾` that
>       contract with `h`** — namely `Q`, `R`, `C`.

*Proof.* (i) is Lemma 1(i). (ii) is a first-order softmax expansion about uniform. For (iii), the
derivatives in §4.2 are now evaluated at `W⁽²⁾ = η G⁽²⁾` instead of `0`, so they are `η` times the
corresponding blocks; composing with `∂h/∂W⁽¹⁾`, which is `O(1)` and non-zero, gives the claim. ∎

**Meaning.** The second layer **recruits** the first. Layer 1 receives gradient only in proportion
to how much layer 2 has already learned to read `h`, and in the direction dictated by the blocks
that consume `h` — predominantly `G_Q`, whose diagonal structure is statement (b).

**CAVEAT — Adam.** The experiments use Adam, whose first update has magnitude set by the learning
rate almost independently of the gradient scale. The `O(η)` bookkeeping above is the
gradient-flow / plain-GD statement. Either restate the scope, or repeat the measurement under
plain GD.

---

## 10. LEMMA 3 — what layer 1 is pushed to compute **[STATEMENT ONLY — NOT PROVED]**

> **Lemma 3 (intended).** To leading order the update to `W⁽¹⁾` increases `S⁽¹⁾_{u,s}` for pairs
> `s < u` with `v_s ∈ S`, decreases it for `v_s ∈ D`, and among admissible `s` the increase is
> monotone in `s`. Hence `h_u` moves from the uniform prefix average `m_u` toward the indicator of
> the most recent signal symbol before `u` — i.e. **state recovery**.

Intended argument:

1. By Lemma 2(iii) the layer-1 gradient is `Φ(G⁽²⁾)`, dominated by the `Q` block.
2. `G_Q[v,·]` rewards keys `u` whose `h_u` places mass on `v`, when `v` is the query symbol.
3. `h_u` places mass on `v` in proportion to how often `v` occurs in `u`'s prefix; the update
   therefore sharpens `h_u` onto whichever symbol best predicts `v_u`, which by the generative
   process is the state — the most recent signal symbol.
4. Distractor suppression and the recency ordering come from the two terms of the coefficient
   (indicator minus baseline).

**This is the second piece of remaining work.**

---

## 11. Everything verified numerically so far

All at `α = 0.6` unless stated, single seed, exact full-batch gradients (not stochastic).

| quantity | measured | where |
|---|---|---|
| CE at zero init | `2.039121` | matches unigram closed form |
| `‖∂L/∂W⁽¹⁾‖` at zero init | `0.000e+00` exactly | Lemma 1(i) |
| `G_P` diag − off-diag | `−1.475e-04` | statement (a), sign |
| `G_Q` diag − off-diag | `+9.621e-05` | statement (b), sign |
| `G_C` diag − off-diag | `+9.804e-06` | |
| `cos(G_Q, construction)` | `+0.8670` | random baseline `0.0099` |
| `cos(G_C, construction)` | `+0.8737` | |
| self-successor rate | `49.95%` vs `2/V_s = 50%` | Fact 1 |
| log+softmax ≡ renormalisation | `5.6e-17` | §8 |
| phase transition, post-distractor error | `0.27–1.16%` for `α ≤ 0.7`; `51.59%` at `α = 0.8`; `49.94%` at `α = 0.9` | V=20 models, 1500 held-out sequences |

---

## 12. What remains, in order

1. **Statement (a)** — §6.3. Self-contained, no mixing needed. Start here.
2. **Statement (b)** — §7. Needs the mixing bound over a random 2-out graph (§7.4).
3. **Lemma 3** — §10.
4. **Scope.** The phase transition is measured on the `V=20` models with a trained unembedding and
   a log+softmax readout; the dynamics above are for `V=8` with the fixed readout. These must be
   unified — either replicate the transition at `V=8`, or re-derive the dynamics for `V=20`.
5. **Seeds.** Every number in §11 is a single seed.
6. **Post-signal control.** To claim "state recovery *specifically* fails", the post-signal error
   must stay flat while the post-distractor error explodes. Not yet measured correctly — the
   earlier attempt had an off-by-one in the state update (it used the previous signal symbol as
   the state rather than `x_t` itself at post-signal positions).
7. **α → 1 reduction.** Stated, not proved or measured.
