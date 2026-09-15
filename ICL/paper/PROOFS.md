# Gradient dynamics at zero initialisation — notation, proofs, and what remains

Working document for the ICLR/COLT paper. Written 2026-09-15.

> **Rendering.** Math below uses `$…$` and `$$…$$`. This renders on GitHub, in VS Code's
> built-in Markdown preview (`Ctrl+Shift+V`), in Obsidian and in JupyterLab. It will **not**
> render in a plain terminal — use one of those.

Every claim is labelled **PROVED**, **VERIFIED NUMERICALLY** (with the measurement), or
**TO DO**. Nothing is asserted without one of those three.

---

## 1. The task

Vocabulary $\mathcal V = \mathcal S \cup \mathcal D$, of sizes $V_s$ and $V_d$:

- **signal** symbols $\mathcal S = \lbrace 0,\dots,V_s-1\rbrace $ — these carry the Markov state;
- **distractor** symbols $\mathcal D = \lbrace V_s,\dots,V-1\rbrace $ — these carry nothing.

For **each sequence independently** we draw a transition table $\pi$, assigning to every state
$s \in \mathcal S$ a successor set $\mathrm{succ}_\pi(s) \subset \mathcal S$ of size 2.

**How the successors are drawn** — this matters for the proof. From
`14. elimination final convergence confusion/data_generation.py`, line 70:

```python
order     = rng.random((B, v_sig, v_sig)).argsort(axis=2)
trans_arr = order[:, :, :2]      # first two of a random permutation of ALL of S
```

So $\mathrm{succ}_\pi(v)$ is a uniform random 2-subset of **all** of $\mathcal S$, **including
$v$ itself**.

**FACT 1.** $\mathbb P\bigl(w \in \mathrm{succ}_\pi(v)\bigr) = 2/V_s$ for **every**
$w \in \mathcal S$, including $w = v$.

*Verified numerically:* 3000 tables at $V_s=4$; fraction of state-entries where the state is its
own successor $= 49.95\%$, against $2/4 = 50\%$.

**FACT 2.** The law of $\pi$ is invariant under any permutation (relabelling) of $\mathcal S$.

Generation, writing $\mathrm{st}(t)$ for the state after position $t$:

$$
v_t \;=\;
\begin{cases}
\text{uniform from } \mathrm{succ}_\pi\bigl(\mathrm{st}(t-1)\bigr), & \text{w.p. } \alpha
\quad\text{(state advances)},\\
\text{uniform from } \mathcal D, & \text{w.p. } 1-\alpha
\quad\text{(state unchanged)} .
\end{cases}
$$

Target $y_t = v_{t+1}$; loss averaged over the scored window $\mathcal T = [T/2,\,T)$.

**Why the task is "in-context".** Because $\pi$ is redrawn every sequence, no function of the
weights alone beats the marginal. All achievable performance must be read off the current context.

---

## 2. Architecture and notation

No learned embedding. Position $t$ is two concatenated one-hots:

$$x_t \;=\; \bigl[\;\underbrace{e_{v_t}}_{\text{symbol}}\;;\;\underbrace{e_t}_{\text{position}}\;\bigr] \;\in\; \mathbb R^{d}, \qquad d = V + T .$$

Two attention layers, **strictly causal** mask ($u<t$, never $u=t$), combined query–key matrices
$W^{(1)} \in \mathbb R^{d\times d}$ and $W^{(2)} \in \mathbb R^{2d\times 2d}$. Value circuits are
**fixed** to the selector that passes the symbol one-hot.

**Layer 1.**

$$S^{(1)}_{t,u} = \tfrac{1}{\sqrt d}\, x_t W^{(1)} x_u^\top + M_{t,u},
\qquad a^{(1)}_{t,\cdot} = \mathrm{softmax}\bigl(S^{(1)}_{t,\cdot}\bigr),
\qquad h_t = \sum_{u<t} a^{(1)}_{t,u}\, e_{v_u}$$

**Layer 2**, reading the concatenation $x^{(2)}_t = [\,x_t \,;\, h_t\,] \in \mathbb R^{2d}$:

$$S^{(2)}_{t,u} = \tfrac{1}{\sqrt d}\, x^{(2)}_t W^{(2)} x^{(2)\top}_u + M_{t,u},
\qquad a^{(2)}_{t,\cdot} = \mathrm{softmax}\bigl(S^{(2)}_{t,\cdot}\bigr),
\qquad f_t = \sum_{u<t} a^{(2)}_{t,u}\, e_{v_u}$$

> **Structural point that drives everything.** $f_t$ is built from $e_{v_u}$, the key's **own
> symbol** — not from $h_u$. So $h$ influences the loss **only through the scores** $S^{(2)}$,
> never through the value path. This is what makes Lemma 1(i) work.

**Readout** — a plain renormalisation against a fixed bias; no unembedding, no log, no softmax:

$$p_t \;=\; \frac{f_t + b}{\sum_w \bigl(f_t[w] + b_w\bigr)},
\qquad b_w = \begin{cases} b_s, & w \in \mathcal S,\\ b_d, & w \in \mathcal D.\end{cases}$$

Since $f_t$ is a convex combination of one-hots, $\sum_w f_t[w] = 1$ **identically**, so

$$\mathcal Z \;:=\; \sum_w \bigl(f_t[w] + b_w\bigr) \;=\; 1 + V_s b_s + V_d b_d$$

is a **constant**, independent of the context. The value of $b_d$ is derived in §8.

---

## 3. The state at zero initialisation

Set $W^{(1)} = W^{(2)} = 0$. Every score is $0$, so both softmaxes are uniform over $\lbrace u<t\rbrace $,
giving $a^{(1)}_{t,u} = a^{(2)}_{t,u} = 1/t$ and therefore

$$h_t \;=\; f_t \;=\; m_t \;:=\; \frac1t \sum_{u<t} e_{v_u} .$$

$m_t$ is the **running symbol frequency** of the context: at zero init the model *is* the
in-context unigram predictor.

*Verified numerically:* cross-entropy at init $=2.039121$ at $\alpha=0.6$, matching the closed form.

> This is exactly the "unigram phase" Edelman et al. (2024) observe mid-training. We do not pass
> through it — we start there.

---

## 4. LEMMA 1 — the gradient at initialisation &nbsp; **[PROVED]**

**LEMMA 1.** At $W^{(1)}=W^{(2)}=0$, with $\omega_t := \bigl(m_t[y_t]+b_{y_t}\bigr)^{-1} > 0$:

**(i) Gating.** $\partial L/\partial W^{(1)} = 0$ identically.

**(ii) Closed form.**

$$-\frac{\partial L}{\partial W^{(2)}}
\;=\; \frac{1}{|\mathcal T|\sqrt d}\sum_{t\in\mathcal T}\frac{\omega_t}{t}
\sum_{u<t}\Bigl(\mathbf{1}[v_u=y_t]-m_t[y_t]\Bigr)\; x^{(2)}_t \otimes x^{(2)}_u$$

### 4.1 Proof of (ii)

**Step 1 — derivative w.r.t. $f$.** With $L_t = -\log p_t[y_t]$ and
$p_t[w] = (f_t[w]+b_w)/\mathcal Z$, treating $f$ as free:

$$g_t[w] \;:=\; \frac{\partial L_t}{\partial f_t[w]}
\;=\; -\,\delta_{w,y_t}\,\omega_t \;+\; \frac{1}{\mathcal Z} .$$

The first term comes from the numerator (only the target's coordinate appears), the second from
the denominator (every coordinate appears once).

**Step 2 — push through the attention.** Since $f_t=\sum_u a^{(2)}_{t,u} e_{v_u}$, we have
$\partial L_t/\partial a^{(2)}_{t,u} = g_t[v_u]$. The softmax Jacobian is
$\partial a_{t,u'}/\partial S_{t,u} = a_{t,u}(\delta_{uu'}-a_{t,u'})$, hence

$$\frac{\partial L_t}{\partial S^{(2)}_{t,u}}
= a^{(2)}_{t,u}\Bigl(g_t[v_u] - \sum_{u'} a^{(2)}_{t,u'} g_t[v_{u'}]\Bigr)
= a^{(2)}_{t,u}\bigl(g_t[v_u] - g_t \cdot f_t\bigr).$$

**Step 3 — the bracket collapses.** Using $\sum_w f_t[w]=1$,

$$g_t \cdot f_t \;=\; \sum_w f_t[w]\,g_t[w] \;=\; -\,m_t[y_t]\,\omega_t + \frac{1}{\mathcal Z},$$

so the two $1/\mathcal Z$ terms cancel and

$$g_t[v_u] - g_t \cdot f_t \;=\; -\,\omega_t\bigl(\mathbf{1}[v_u=y_t]-m_t[y_t]\bigr).$$

**Step 4 — chain to the weights.** With
$\partial S^{(2)}_{t,u}/\partial W^{(2)} = x^{(2)}_t \otimes x^{(2)}_u/\sqrt d$ and
$a^{(2)}_{t,u}=1/t$ at init, assembling and flipping sign gives (ii). $\blacksquare$

> **Plain reading.** The coefficient on key $u$ is positive exactly when that key's symbol *is*
> the target. The first step says: **attend more to positions showing the answer.**

### 4.2 Proof of (i)

$h$ enters the loss only through $S^{(2)}$ (§2). Differentiating,

$$\frac{\partial S^{(2)}_{t,u}}{\partial h_u}
= \tfrac{1}{\sqrt d}\bigl(x^{(2)}_t W^{(2)}\bigr)_{d:2d},
\qquad
\frac{\partial S^{(2)}_{t,u}}{\partial h_t}
= \tfrac{1}{\sqrt d}\bigl(W^{(2)} x^{(2)\top}_u\bigr)_{d:2d}.$$

Both carry $W^{(2)}$ as a **multiplicative factor** and so vanish at $W^{(2)}=0$. Every path from
$W^{(1)}$ to $L$ runs through $h$, hence $\partial L/\partial W^{(1)} = 0$. $\blacksquare$

*Verified numerically:* $\|\partial L/\partial W^{(1)}\| = 0.000\mathrm{e}{+00}$ exactly — zero,
not small.

> **Consequence.** The two layers **cannot form simultaneously**. Layer 2 must move first.

---

## 5. The block decomposition

Split $W^{(2)}$ into four $d\times d$ blocks matching $x^{(2)}=[x;h]$:

$$W^{(2)}=\begin{bmatrix} P & Q \\ R & C \end{bmatrix},
\qquad
\sqrt d\, S^{(2)}_{t,u}
= \underbrace{x_t P x_u^\top}_{\text{symbol}\times\text{symbol}}
+ \underbrace{x_t Q h_u^\top}_{\textbf{induction}}
+ \underbrace{h_t R x_u^\top}_{\text{cross}}
+ \underbrace{h_t C h_u^\top}_{\text{state}\times\text{state}} .$$

Restricting to the signal sub-blocks and reading off Lemma 1(ii), for $v,w\in\mathcal S$:

$$G_P[v,w] \;\propto\; \sum_{t} \mathbf{1}[v_t{=}v]\,\frac{\omega_t}{t}
\sum_{u<t}\bigl(\mathbf{1}[v_u{=}y_t]-m_t[y_t]\bigr)\,\mathbf{1}[v_u{=}w]$$

$$G_Q[v,w] \;\propto\; \sum_{t} \mathbf{1}[v_t{=}v]\,\frac{\omega_t}{t}
\sum_{u<t}\bigl(\mathbf{1}[v_u{=}y_t]-m_t[y_t]\bigr)\, m_u[w]$$

The outer product $x^{(2)}_t\otimes x^{(2)}_u$ supplies $\mathbf{1}[v_t{=}v]$ on the query side;
on the key side it supplies $\mathbf{1}[v_u{=}w]$ for $P$ and $h_u[w]=m_u[w]$ for $Q$.

| block | the rule a positive diagonal would encode | what must be true for it to help |
|---|---|---|
| $G_P$ | "when I am $v$, prefer keys whose **symbol** is $w$" | a **fixed** bigram table, stored in weights |
| $G_Q$ | "when I am $v$, prefer keys whose **preceding context** looks like $w$" | nothing fixed — read from context |

---

## 6. STATEMENT (a) — the bigram block &nbsp; **[TO PROVE]**

$$\textbf{(a)}\qquad
\mathbb E\bigl[G_P[v,v]\bigr] - \mathbb E\bigl[G_P[v,w]\bigr] \;\le\; 0
\qquad \forall\, w \neq v .$$

Expectation over the table $\pi$ and the sequence.

### 6.1 What symmetry already gives

By **Fact 2**, a permutation of $\mathcal S$ fixing $v$ and swapping $w \leftrightarrow w'$ maps
the ensemble to itself, so

$$\mathbb E\bigl[G_P[v,w]\bigr] = \mathbb E\bigl[G_P[v,w']\bigr] \qquad \forall\, w,w' \neq v .$$

All off-diagonal entries in a row are equal. **This says nothing about the diagonal**: no
permutation fixes $v$ while moving $v$, so $\mathbb E[G_P[v,v]]$ is a genuinely separate quantity
and nothing forces it to equal the others.

### 6.2 The two terms

With $N_t(w) := \lvert \lbrace u<t : v_u = w \rbrace \rvert$ the prefix count of $w$,

$$G_P[v,w] \;=\; \sum_t \mathbf{1}[v_t{=}v]\frac{\omega_t}{t}
\Bigl[\;\underbrace{\mathbf{1}[w{=}y_t]\,N_t(w)}_{\text{term A}}
\;-\; \underbrace{m_t[y_t]\,N_t(w)}_{\text{term B}}\;\Bigr]$$

- **Term A.** By **Fact 1**, $\mathbb E\bigl[\mathbf{1}[w{=}y_t]\mid \mathrm{st}(t)=v\bigr] = \alpha/V_s$
  for *every* $w$, including $w=v$. So term A contributes **equally** to diagonal and off-diagonal.
- **Term B** carries the asymmetry. Conditioned on $v_t=v$, the prefix count $N_t(v)$ is
  **elevated** relative to $N_t(w)$ — the chain has just visited $v$, and states recur. So the
  baseline subtraction bites **harder** at $w=v$.

> **Therefore the expected conclusion is not "the diagonal is zero" but "the diagonal is strictly
> negative".** The in-weights block is pushed *away* from the identity.

*Verified numerically* ($\alpha=0.6$, diagonal minus off-diagonal of the signal $4\times4$):

| block | value | sign |
|---|---|---|
| $G_P$ (bigram) | $-1.475\times10^{-4}$ | negative |
| $G_Q$ (induction) | $+9.621\times10^{-5}$ | positive |
| $G_C$ (state×state) | $+9.804\times10^{-6}$ | positive |

### 6.3 What the proof needs

1. Show term A cancels exactly between diagonal and off-diagonal. Immediate from Fact 1.
2. Show $\;\mathbb E\bigl[m_t[y_t]\,N_t(v)\bigr] - \mathbb E\bigl[m_t[y_t]\,N_t(w)\bigr] > 0$ —
   that the prefix count of the *current* state exceeds that of any other symbol in expectation.
   This is a statement about the stationary behaviour of the chain and should **not** need a
   mixing bound: it reduces to comparing $\mathbb P(v_s{=}v \mid v_t{=}v)$ against
   $\mathbb P(v_s{=}w \mid v_t{=}v)$, summed over $s<t$.

> **Start here.** Self-contained, no mixing. A few hours will tell you whether the algebra opens up.

---

## 7. STATEMENT (b) — the induction block &nbsp; **[TO PROVE — the hard one]**

$$\textbf{(b)}\qquad
\mathbb E\bigl[G_Q[v,v]\bigr] - \mathbb E\bigl[G_Q[v,w]\bigr] \;>\; 0
\qquad \forall\, w \neq v .$$

### 7.1 The mechanism, concretely

$G_Q[v,w]$ weights each key $u$ by $m_u[w]$ — the frequency of $w$ in **the key's own prefix** —
and rewards keys whose symbol is the target. With $V_s=4$:

```
position:  0   1   2   3   4   5   6
symbol:    0   5   2   6   0   4   2
state:     0   0   2   2   0   0   2
               ↑                   ↑
        positions where a successor of 0 appears
```

Suppose at time $t$ the state is $v=0$ and the target is $y_t=2$. The gradient rewards keys with
symbol $2$ — positions 2 and 6. **Both were produced by a transition out of $0$.** That is not
coincidence; it is how the data is generated. So $m_u$ at those keys carries extra mass on $0$,
which is $v$ itself.

> **That is the diagonal signal.** The bigram block asks a question the resampled table has
> erased; the induction block asks one the sequence answers locally, and that answer survives
> averaging over $\pi$.

### 7.2 Reduction to a three-point correlation

Expanding $m_u[w] = \frac1u\sum_{s<u}\mathbf{1}[v_s{=}w]$, the quantity to bound is

$$\mathbb E\bigl[\mathbf{1}[v_u{=}y_t]\; m_u[w]\bigr]
\;=\; \frac1u \sum_{s<u} \mathbb E\bigl[\mathbf{1}[v_u{=}y_t]\,\mathbf{1}[v_s{=}w]\bigr],
\qquad s < u < t .$$

So (b) is a **three-point correlation** in the chain, averaged over the random table.

### 7.3 Proposed proof structure

Split the inner sum at a lag $L$:

- **Distant terms ($s \le u-L$).** Show
  $$\bigl|\;\mathbb P(v_s{=}w \mid v_u, v_t, y_t) - \mathbb P(v_s{=}w)\;\bigr| \;\le\; C\rho^{\,u-s}$$
  so they contribute **equally to every $w$** and cancel in the diagonal-minus-off-diagonal
  contrast, leaving $O(\rho^L)$.
- **Recent terms ($s > u-L$).** Dominated by $s =$ the last signal position before $u$. A one-step
  Bayes computation gives a strictly positive constant on the diagonal, since occurrences of a
  successor of $v$ are generated by transitions out of $v$.

### 7.4 The two technical obstacles

**(i) Mixing over a random 2-out graph.** $\rho$ is a mixing rate for the state chain. For a
*fixed* table the chain need not mix well — at $V_s=4$ one can draw tables with transient states,
so the stationary distribution is not uniform and $\rho$ is not uniformly bounded. A
**high-probability statement over tables** is needed, plus control of the exceptional set.

**(ii) The $1/u$ dilution.** $m_u$ averages the *whole* prefix, so the per-position diagonal boost
is $O(1/u)$ while $m_u[w]$ itself is $O(1/V_s)$. The contrast survives the sums over $t$ and $u$,
but the constants should be explicit — otherwise a referee will ask whether the effect is real at
the lengths actually trained on ($T=100$, scored on $[50,100)$).

---

## 8. The readout bias &nbsp; **[PROVED]**

With a perfect distractor filter, $f_t$ is supported on $\mathcal S$ and sums to 1, so
$\mathcal Z = (1+V_sb_s) + V_db_d$ is constant. Demanding each distractor receive exactly
$(1-\alpha)/V_d$:

$$\frac{b_d}{\mathcal Z}=\frac{1-\alpha}{V_d}
\;\Longrightarrow\; V_d b_d = (1-\alpha)\mathcal Z
\;\Longrightarrow\; \mathcal Z = (1+V_sb_s) + (1-\alpha)\mathcal Z
\;\Longrightarrow\; \mathcal Z = \frac{1+V_sb_s}{\alpha},$$

$$\boxed{\; b_d = \frac{(1-\alpha)(1+V_s b_s)}{V_d\,\alpha} \;}$$

and the signal side then gets $p_t[w] = \alpha\,(f_t[w]+b_s)/(1+V_sb_s)$ — exactly proportional to
the smoothed count. The $/\alpha$ is the part people drop: **the renormalisation shrinks whatever
you put in**.

**Relation to a log + softmax readout.** If the nonlinearity before an unembedding is
$z_w = \log(f_t[w]+b_s)$ on signal coordinates and a constant $\beta$ on distractor coordinates,
then $\mathrm{softmax}(z)_w = (f_t[w]+b_s)/\sum_{w'} e^{z_{w'}}$ — the same renormalisation with
$b_d = e^{\beta}$. The log and the exponential cancel. *Verified numerically:* difference
$5.6\times10^{-17}$.

**Why the log is not optional.** Without it the softmax returns $e^{f_t[w]}$ — exponential in the
count rather than linear — and no rescaling repairs this, because $\mathrm{softmax}(\kappa f)$ is
a different function of $f$ than $f/\sum f$ for every $\kappa$.

---

## 9. LEMMA 2 — layer 1 unlocks &nbsp; **[PROVED, modulo the Adam caveat]**

**LEMMA 2.** Let $W^{(2)}_1 = \eta G^{(2)}$ be one gradient step from Lemma 1, and
$W^{(1)}_1 = 0$ by Lemma 1(i). Then:

**(i)** layer 1 is unchanged, so $h_t = m_t$ still.

**(ii)** the second layer's attention is perturbed to first order, with
$\sigma_{t,u} = x^{(2)}_t G^{(2)} x^{(2)\top}_u$ and $\bar\sigma_t$ its mean over $u<t$:

$$a^{(2)}_{t,u} = \frac1t\Bigl(1+\frac{\eta}{\sqrt d}\bigl(\sigma_{t,u}-\bar\sigma_t\bigr)\Bigr)+O(\eta^2)$$

**(iii)** the first layer's gradient becomes non-zero, of order $\eta$:

$$-\frac{\partial L}{\partial W^{(1)}}\Big|_{1} \;=\; \eta\,\Phi\bigl(G^{(2)}\bigr) + O(\eta^2)$$

where $\Phi$ is **linear in exactly those blocks of $G^{(2)}$ that contract with $h$** — namely
$Q$, $R$ and $C$.

*Proof.* (i) is Lemma 1(i). (ii) is a first-order softmax expansion about uniform. For (iii), the
derivatives of §4.2 are now evaluated at $W^{(2)}=\eta G^{(2)}$ rather than $0$, so they equal
$\eta$ times the corresponding blocks; composing with $\partial h/\partial W^{(1)}$, which is
$O(1)$ and non-zero, gives the claim. $\blacksquare$

> **Meaning.** The second layer **recruits** the first. Layer 1 receives gradient only in
> proportion to how much layer 2 has already learned to read $h$, and in the direction dictated by
> the blocks that consume $h$ — predominantly $G_Q$, whose diagonal structure is statement (b).

**CAVEAT — Adam.** The experiments use Adam, whose first update has magnitude set by the learning
rate almost independently of the gradient scale. The $O(\eta)$ bookkeeping above is the
gradient-flow / plain-GD statement. Either restate the scope, or repeat the measurement under
plain GD.

---

## 10. LEMMA 3 — what layer 1 is pushed to compute &nbsp; **[STATEMENT ONLY — NOT PROVED]**

**LEMMA 3 (intended).** To leading order the update to $W^{(1)}$ increases $S^{(1)}_{u,s}$ for
pairs $s<u$ with $v_s\in\mathcal S$, decreases it for $v_s\in\mathcal D$, and among admissible
$s$ the increase is monotone in $s$. Hence $h_u$ moves from the uniform prefix average $m_u$
toward the indicator of the most recent signal symbol before $u$ — i.e. **state recovery**.

Intended argument:

1. By Lemma 2(iii) the layer-1 gradient is $\Phi(G^{(2)})$, dominated by the $Q$ block.
2. $G_Q[v,\cdot]$ rewards keys $u$ whose $h_u$ places mass on $v$, when $v$ is the query symbol.
3. $h_u$ places mass on $v$ in proportion to how often $v$ occurs in $u$'s prefix; the update
   therefore sharpens $h_u$ onto whichever symbol best predicts $v_u$, which by the generative
   process is the state — the most recent signal symbol.
4. Distractor suppression and the recency ordering come from the two terms of the coefficient
   (indicator minus baseline).

**This is the second piece of remaining work.**

---

## 11. Everything verified numerically so far

All at $\alpha=0.6$ unless stated; single seed; exact full-batch gradients, not stochastic.

| quantity | measured | relates to |
|---|---|---|
| CE at zero init | $2.039121$ | §3, matches unigram closed form |
| gradient norm $\lVert \partial L/\partial W^{(1)} \rVert$ at zero init | $0.000\mathrm{e}{+}00$ exactly | Lemma 1(i) |
| $G_P$ diag $-$ off-diag | $-1.475\times10^{-4}$ | statement (a), sign |
| $G_Q$ diag $-$ off-diag | $+9.621\times10^{-5}$ | statement (b), sign |
| $G_C$ diag $-$ off-diag | $+9.804\times10^{-6}$ | |
| $\cos(G_Q,\ \text{construction})$ | $+0.8670$ | random baseline $0.0099$ |
| $\cos(G_C,\ \text{construction})$ | $+0.8737$ | |
| self-successor rate | $49.95\%$ vs $2/V_s = 50\%$ | Fact 1 |
| log+softmax $\equiv$ renormalisation | $5.6\times10^{-17}$ | §8 |
| post-distractor error, $\alpha\le0.7$ | $0.27\%$–$1.16\%$ | phase transition |
| post-distractor error, $\alpha=0.8$ / $0.9$ | $51.59\%$ / $49.94\%$ | phase transition |

The phase-transition rows are measured on the $V=20$ models, 1500 held-out sequences each.

---

## 12. What remains, in order

1. **Statement (a)** — §6.3. Self-contained, no mixing needed. **Start here.**
2. **Statement (b)** — §7. Needs the mixing bound over a random 2-out graph, §7.4.
3. **Lemma 3** — §10.
4. **Scope.** The phase transition is measured on $V=20$ models with a trained unembedding and a
   log+softmax readout; the dynamics above are for $V=8$ with the fixed readout. These must be
   unified — replicate the transition at $V=8$, or re-derive the dynamics for $V=20$.
5. **Seeds.** Every number in §11 is a single seed.
6. **Post-signal control.** To claim state recovery *specifically* fails, the post-signal error
   must stay flat while the post-distractor error explodes. Not yet measured correctly — the
   earlier attempt had an off-by-one in the state update (it used the previous signal symbol as
   the state at post-signal positions, rather than $x_t$ itself).
7. **$\alpha\to1$ reduction.** Stated, neither proved nor measured.
