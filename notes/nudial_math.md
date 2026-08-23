# The ℓ2 max-margin theory of Fourier committee size — every math element

Self-contained derivation of everything in the nu-dial line, in the order the
logic runs. Each section ends with what is proved, what is numerically
verified, and what is assumed. Sources: `nudial/kstar_pscale.py`,
`notes/kstar_pscale.json`, scratch `kstar.py`/`ktable.txt`, Morwani et al.
2311.07568, Lyu & Li 2019 (ICLR 2020), Hoff 2017 / Kolb 2025.

---

## 0. Setup and notation

- Task: $f:\mathbb Z_p\times\mathbb Z_p\to\mathbb R^p$, target $c=a+b \bmod p$, $p$ prime.
- Frequencies: $k\in\{1,\dots,\tfrac{p-1}{2}\}$ (pairs $\pm k$ identified; $\tfrac{p-1}{2}$ of them).
- Offset: $\Delta = a+b-c \bmod p$. The correct logit has $\Delta=0$; a wrong logit has $\Delta\neq0$.
- A *committee* is a set $S$ of frequencies, $|S|=K$.
- Network: parameters $\theta$, logits $f_c(a,b;\theta)$.

**Margin** (functional):
$$g(\theta)=\min_{(a,b)}\Big[f_{a+b}(a,b)-\max_{c\neq a+b}f_c(a,b)\Big].$$

**Homogeneity degree** $\nu$: $f(a,b;\,t\theta)=t^{\nu}f(a,b;\theta)$ for all $t>0$.

**Normalised margin**:
$$\gamma(\theta)=\frac{g(\theta)}{\|\theta\|_2^{\nu}} .$$
Scale-free by construction: $\gamma(t\theta)=\gamma(\theta)$.

---

## 1. Why the normalised margin is the object

**Lyu–Li (2019).** For a $\nu$-homogeneous network trained by gradient flow on
an exponential-tail loss (CE), once training loss is small the parameter
direction $\theta/\|\theta\|$ converges to a KKT point of

$$\max_{\theta}\ \gamma(\theta)\quad\Longleftrightarrow\quad \min \|\theta\|^2\ \text{s.t.}\ g(\theta)\ge 1 .$$

Weight decay does not change the stationary points of this problem; it fixes
the norm scale at which they are reached and makes them reachable in finite
time. Grokking is the late-time regime where this bias acts.

Consequence: *the trained solution is characterised by the max of $\gamma$,
not by anything about the data split or the optimiser*, to leading order.
Everything below is about where $\max\gamma$ sits.

- Status: theorem (Lyu–Li) for gradient flow, homogeneous nets, no biases.
  AdamW is not gradient flow; we use it as the heuristic target and check the
  result empirically (§7).

---

## 2. The single-frequency-circuit ansatz

A *frequency-$k$ circuit* is a sub-network whose logit contribution is

$$f^{(k)}_c(a,b)=A_k\cos\!\Big(\frac{2\pi k(a+b-c)}{p}\Big)=A_k\cos\!\Big(\frac{2\pi k\Delta}{p}\Big),$$

with amplitude $A_k\ge0$. The ansatz: the network is a sum of such circuits
over a committee $S$, and nothing else,

$$f_c(a,b)=\sum_{k\in S}A_k\cos\!\Big(\frac{2\pi k\Delta}{p}\Big)+\text{const}.$$

Then the correct logit is $\sum_k A_k$ and the wrong logit at offset $\Delta$ is
$\sum_k A_k\cos(2\pi k\Delta/p)$, so

$$g=\min_{\Delta\neq0}\sum_{k\in S}A_k\Big(1-\cos\frac{2\pi k\Delta}{p}\Big).\tag{2.1}$$

- Assumed: (i) every neuron is a pure single-frequency circuit (no harmonics,
  no cross-terms); (ii) the only dependence on $(a,b)$ is through $\Delta$.
  Both are violated by real nonlinearities; §8 quantifies the violation.

---

## 3. What ℓ2 charges for amplitude — the $\ell_{2/\nu}$ quasi-ball

A frequency-$k$ circuit in a degree-$\nu$ net is a chain of $\nu$ weight
blocks $W_1,\dots,W_\nu$ with norms $r_1,\dots,r_\nu$. Its amplitude is the
product, its ℓ2 cost the sum of squares:

$$A_k\propto\prod_{i=1}^{\nu}r_i,\qquad \text{cost}_k=\sum_{i=1}^{\nu}r_i^2 .$$

**Balancing.** By AM–GM, for fixed product the sum of squares is minimised at
$r_1=\dots=r_\nu=A_k^{1/\nu}$. (Gradient flow also conserves $r_i^2-r_j^2$, so
balanced inits stay balanced.) Hence

$$\text{cost}_k=\nu A_k^{2/\nu}.\tag{3.1}$$

**Total budget.** With $\|\theta\|^2=1$ (normalised margin is scale-free, so
WLOG),

$$\sum_{k\in S}A_k^{2/\nu}\le\frac1\nu .\tag{3.2}$$

This is an $\ell_{2/\nu}$ ball on the amplitude vector. For $\nu=2$ it is the
$\ell_1$ ball (linear in $A$: no preference for concentration). For $\nu\ge3$,
$2/\nu<1$ and the "ball" is a non-convex star with spikes on the axes:
concentrating mass on few coordinates buys more total amplitude. This is the
*only* source of sparsity in the theory.

**Contrast with Morwani et al.** Their Theorem 7 maximises margin under the
$L_{2,\nu}$ norm $\big(\sum_j\|\theta_j\|_2^{\nu}\big)^{1/\nu}$ over neurons
$j$, for which the per-frequency budget is $\sum_k A_k\le$ const — linear —
and the optimum is dense. Same task, different norm, opposite answer. Plain
weight decay is ℓ2, not $L_{2,\nu}$.

Credit for the primitive "ℓ2 on a product of $D$ factors $=\ell_{2/D}$ on the
product": Hoff 2017 (1611.00040), Kolb et al. 2025 (2502.02496).

- Status: algebra.

---

## 4. The trade-off and $K^*(\nu)$

**Equal amplitudes.** Take $A_k=A$ for all $k\in S$. From (3.2),
$A=(\nu K)^{-\nu/2}$; dropping the $\nu$-only constant,

$$\gamma(S)=K^{-\nu/2}\min_{\Delta\neq0}\sum_{k\in S}\Big(1-\cos\frac{2\pi k\Delta}{p}\Big).\tag{4.1}$$

Define the **best worst-case gap** of a $K$-committee,

$$M_K(p)=\max_{|S|=K}\ \min_{\Delta\neq0}\ \sum_{k\in S}\Big(1-\cos\frac{2\pi k\Delta}{p}\Big),\tag{4.2}$$

and the **leak** $c_K(p)=K-M_K(p)=\min_S\max_\Delta\sum_{k\in S}\cos(2\pi k\Delta/p)$.

Then within the ansatz

$$K^*(\nu,p)=\arg\max_K\ K^{-\nu/2}M_K(p).\tag{4.3}$$

Monotonicity: $M_K$ is non-decreasing in $K$ (each summand is $\ge0$ for every
$\Delta$, so adding a frequency cannot lower the min). $K^{-\nu/2}$ is
decreasing. The product peaks.

**Closed form in the linear window.** Where $M_K\approx K-c$ with $c$ slowly
varying (true for $K\le12$, $p\sim10^2$; see §5 for why it is only a window),

$$\frac{d}{dK}\Big[-\tfrac\nu2\log K+\log(K-c)\Big]=0\ \Rightarrow\ K^*=\frac{c\,\nu}{\nu-2}.\tag{4.4}$$

At $p=113$, $c\approx2.1$: $K^*=$ dense ($\nu=2$), $6.3$ ($\nu=3$), $4.2$
($\nu=4$), $3.5$ ($\nu=5$), $3.2$ ($\nu=6$). Exact integer argmax from the
table: $6,4,4,3$. Sparse/dense margin ratio at $p=113$: $2\times,\ 6.5\times,\
25\times,\ 97\times$ for $\nu=3,4,5,6$.

**Is equal amplitude OK?** Re-optimising $A_k$ freely on the optimal support
(scratch `ktable.txt`, "u" column): gain $\le2.5\%$ for $K\le K^*$ at every
$\nu$; the integer $K^*$ never changes. Past $K^*$ the free-amplitude solution
gains up to 45% by putting most mass on $\approx K^*$ members — i.e. it *is* a
$K^*$ solution with vestigial members, which is why $K$ must be measured by
participation ratio $k_{\rm eff}=1/\sum_k s_k^2$ ($s_k$ = energy share) and
not by a count.

**Why equal amplitude is natural.** The objective is a min over $\Delta$; a
min is raised by strengthening its weakest contributor, and optimal sets are
collision-free (no $k_i\pm k_j=\pm k_l$), so no member is distinguished.

- Status: (4.1)–(4.4) are algebra given the ansatz; $M_K$ is numerical;
  equal amplitude checked numerically.

---

## 5. $M_K(p)$: the leak is not a constant

### 5.1 Theorem (Dirichlet pigeonhole): $M_K(p)\to0$ for every fixed $K$

Map $\Delta\in\mathbb Z_p$ to $P(\Delta)=(k_1\Delta/p,\dots,k_K\Delta/p)\bmod1$
on the torus $\mathbb T^K$. This is $p$ points. Partition $\mathbb T^K$ into
$N^K$ cubes of side $1/N$ with $N^K<p$. Two points $P(\Delta_1),P(\Delta_2)$
share a cube; $\Delta'=\Delta_1-\Delta_2\neq0$ satisfies $\|k_i\Delta'/p\|<1/N$
for all $i$ (here $\|\cdot\|$ = distance to nearest integer). Therefore, for
every $S$,

$$\min_{\Delta\neq0}\sum_{k\in S}\Big(1-\cos\frac{2\pi k\Delta}{p}\Big)<K\big(1-\cos(2\pi/N)\big),\qquad N=\lfloor (p-1)^{1/K}\rfloor,$$

$$\boxed{M_K(p)<K\big(1-\cos(2\pi p^{-1/K})\big)\approx2\pi^2K\,p^{-2/K}\xrightarrow[p\to\infty]{}0 .}\tag{5.1}$$

So the "$M_K\approx K-2.1$" form cannot hold for large $p$ at fixed $K$. It is
a description of a finite-$p$ window where $p^{-1/K}=O(1)$ (at $p=113$, $K=4$:
$p^{-1/4}=0.31$, and (5.1) is vacuous).

### 5.2 Lattice regime ($K\ll\log p$): the constant from Hermite

The set $\{(k_1\Delta,\dots,k_K\Delta)\bmod p\}$ lifts to a lattice
$L_S\subset\mathbb Z^K$ containing $p\mathbb Z^K$ (the $\Delta\equiv0$ coset),
with $\det L_S=p^{K-1}$. In the small-angle regime
$1-\cos(2\pi x/p)\approx2\pi^2x^2/p^2$, the adversary's best $\Delta$ is the
shortest vector $\lambda_1$ of $L_S\setminus p\mathbb Z^K$, and the best $S$
is the densest realisable such lattice. Minkowski–Hermite:
$\lambda_1^2\le\gamma_K(\det L_S)^{2/K}=\gamma_K\,p^{2(K-1)/K}$, hence

$$M_K(p)\lesssim\frac{2\pi^2}{p^2}\lambda_1^2=2\pi^2\gamma_K\,p^{-2/K},\tag{5.2}$$

with Hermite constants $\gamma_2=2/\sqrt3,\ \gamma_3=2^{1/3},\ \gamma_4=\sqrt2,\
\gamma_5=8^{1/5},\ \gamma_6=(64/3)^{1/6},\ \gamma_8=2$, and
$\gamma_K\sim K/(2\pi e)$ for large $K$. Equality requires the cyclic lattice
mod a prime to realise the densest packing, which it approaches but need not
reach.

Numerical check ($M_K\,p^{2/K}$, `notes/kstar_pscale.json`):

| $K$ | $p=113$ | 211 | 401 | 1009 | 2003 | 4001 | bound $2\pi^2\gamma_K$ |
|---|---|---|---|---|---|---|---|
| 2 | 21.2 | 21.1 | 20.8 | 22.5 | 22.2 | 22.3 | 22.8 |
| 3 | 19.7 | 21.3 | 22.6 | 23.1 | 23.0 | 23.5 | 24.9 |
| 4 | 20.0 | 20.8 | 21.6 | 22.6 | 23.1 | 23.7 | 27.9 |

Saturating under the bound as required. $K=2$ is essentially exact
($c_2=K-M_2\to2$: 1.81, 1.90, 1.95, 1.98, 1.99, 1.99).

### 5.3 Equidistribution regime ($K\gtrsim\log p$): the unproven "2"

When $p^{-1/K}$ is not small, the $p$ points fill $\mathbb T^K$; the adversary
can align only $O(1)$ coordinates and the rest average out. This is the
plateau where $c_K\approx2$ at $p\sim10^2$, $K\le12$.

Bounds: for collision-free $S$ and $f(\Delta)=\sum_{k\in S}\cos(2\pi k\Delta/p)$,
$\sum_{\Delta\neq0}f=-K$ and $\sum_{\Delta\neq0}f^2=\tfrac{pK}{2}-K$. With
$f\ge-K$ these force $\max_\Delta f\ \ge\ \tfrac12-\tfrac1{4K}+o(1)$. A random
$K$-set has $\max_\Delta f\sim\sqrt{K\log p}$. The optimised value ($\approx2$)
sits between; deriving it is an extremal problem of Chowla-cosine type and is
**open**.

### 5.4 Computed $c_K(p)=K-M_K(p)$ (local search, 40 restarts; $K\ge7$ values are lower bounds on $M_K$)

| $K$ | 113 | 211 | 401 | 1009 | 2003 | 4001 |
|---|---|---|---|---|---|---|
| 2 | 1.81 | 1.90 | 1.95 | 1.98 | 1.99 | 1.99 |
| 3 | 2.16 | 2.40 | 2.58 | 2.77 | 2.86 | 2.91 |
| 4 | 2.12 | 2.57 | 2.92 | 3.29 | 3.48 | 3.63 |
| 6 | 2.21 | 2.70 | 3.19 | 3.78 | 4.17 | 4.48 |
| 8 | 2.35 | 2.92 | 3.49 | 4.19 | 4.64 | 5.11 |
| 12 | 2.54 | 3.25 | 3.87 | 4.72 | 5.40 | 5.96 |

- Status: (5.1) theorem; (5.2) derived, verified $K\le4$; §5.3 open.

---

## 6. Consequence: $K^*$ grows with $\log p$

Insert (5.2) into (4.3):

$$\log\gamma(K)=-\frac\nu2\log K+\log(2\pi^2\gamma_K)-\frac2K\log p .$$

With $\log\gamma_K\approx\log K+$const at large $K$,

$$\frac{d}{dK}:\ -\frac{\nu}{2K}+\frac1K+\frac{2\log p}{K^2}=0\ \Rightarrow\
\boxed{K^*(\nu,p)\approx\frac{4\log p}{\nu-2}}\quad(K\ll\log p\ \text{regime}).\tag{6.1}$$

Same $\nu-2$ denominator as (4.4); the "constant" $c$ is $\log p$ in disguise.
At experimental $p$ the truth is a crossover between (4.4) and (6.1).

Computed integer argmax of (4.3), $K\le12$:

| $p$ | $\log p$ | $\nu=3$ | $\nu=4$ | $\nu=5$ | $\nu=6$ |
|---|---|---|---|---|---|
| 113 | 4.73 | 6 | 4 | 4 | 3 |
| 211 | 5.35 | 6 | 5 | 4 | 4 |
| 401 | 5.99 | 9 | 5 | 5 | 4 |
| 1009 | 6.92 | 11 | 7 | 5 | 5 |
| 2003 | 7.60 | 11 | 7 | 6 | 5 |
| 4001 | 8.29 | 12+ | 9 | 6 | 6 |

Slope in $\log p$: $\approx1.4$ ($\nu=4$), $\approx0.8$ ($\nu=6$); (6.1)
predicts $2$ and $1$ asymptotically. Runner-up ratios at $\nu=6$: $p=113$,
$K^*=3$ with $K=4$ at $0.94$; $p=211$, $K^*=4$ with $K=3$ at $0.996$ — the
observed transformer 3:74 / 4:80 split at $p=113$–$157$ is this crossover.

**Correction to earlier claims.** "$K^*$ is $p$-independent" is false; it is
flat over $p\in[97,157]$ ($\log p\in[4.6,5.1]$) and grows logarithmically.
This reconciles McCracken 2505.18266's $O(\log p)$ count.

- Status: (6.1) derived from (5.2); table numerical.

---

## 7. Empirical anchor: the normalised margin of trained nets

Same scalar $\gamma=g/\|\theta\|^{\nu}$ over the full $p^2$ grid, relu-4,
$m=512$, $p=113$:

| source | $\gamma$ | $K$ |
|---|---|---|
| direct margin ascent, 32 inits (cohort D) | $6.4\times10^{-7}$ | 4 |
| plain GD + ℓ2 | $2.4\times10^{-6}$ | 3 |
| AdamW, init scale 0.1 | $2.5\times10^{-6}$ | 2 |
| AdamW, standard | $4.7$–$4.9\times10^{-6}$ | 5 |

relu-6: D $2.4\times10^{-9}$ ($K=3$), GD $2.8\times10^{-8}$, AdamW
$4.5\times10^{-8}$ ($K=4$).

Trained nets have the highest $\gamma$ of anything produced. Hence the
observed $K=K^*+1$ is not a failure to reach the optimum; the ansatz optimum
undercounts. Margin-ascent runs are lower bounds, never "the optimum."

---

## 8. The architecture term $\delta_{\rm arch}$ (the +1; not derived)

Write $K_{\rm true}=K^*(\nu,p)+\delta_{\rm arch}$. The ansatz assumed a
frequency-$k$ circuit emits only frequency $k$. Real nonlinearities leak.

**ReLU harmonics.** $\mathrm{relu}(\cos\phi)=\frac1\pi+\frac12\cos\phi+\frac{2}{3\pi}\cos2\phi-\frac{2}{15\pi}\cos4\phi+\dots$
A neuron with preactivation $\cos(ka+\phi_i)+\cos(kb+\psi_i)$ has a Fourier
expansion in $(a,b)$ with terms $\cos(mka+nkb+m\phi_i+n\psi_i)$. Readout
$w_{ic}\propto\cos(kc-\phi_i-\psi_i)$ and averaging over neuron phases keeps
only terms whose phase dependence cancels:

- $(m,n)=(1,1)$: $\cos(k(a+b)+\phi+\psi)\cos(kc-\phi-\psi)\to\tfrac12\cos(k\Delta)$. The signal.
- $(m,n)=(2,0)$: $\cos(2ka+2\phi)\cos(kc-\phi-\psi)$, phases $3\phi+\psi$ and $\phi-\psi$ — averages to 0 **if $\phi_i,\psi_i$ are independent across neurons**.
- If $\psi_i=\phi_i$ (tied phases), the $(2,0)$ term survives as $\tfrac12\cos(2ka-kc)$: logit energy at $(2a-c)$, not a function of $\Delta$. This is the one-hot quadratic cross-talk $2u_av_bw_c$ → dense optimum.

Surviving off-target energy lowers $\min_{(a,b)}$ in a way not captured by
(2.1). Adding a member at equal amplitude spreads and partially cancels it (a
numerator gain the ansatz omits, not paid for by $K^{-\nu/2}$), shifting the
argmax up by $\approx1$ for ReLU/shared-embedding nets and to "dense" for
one-hot quadratic nets. Democratic amplitudes in trained nets are the
fingerprint.

- Status: mechanism identified; the magnitude of $\delta_{\rm arch}$ is
  inferred from §7, not computed. The computable version: evaluate the
  surviving leakage amplitudes for relu with $K$ equal members and redo the
  argmax.

---

## 9. $\nu$ for an arbitrary architecture

**Definition.** A path $\pi$ from input to logit has degree $\nu_\pi$ =
number of weight tensors multiplied along it, with activation rules:

| component | effect on degree |
|---|---|
| weight matrix | $+1$ |
| ReLU, leaky, abs (positively 1-homogeneous) | $\times1$ |
| $x^d$ / degree-$d$ polynomial | $\times d$ |
| bilinear / gated unit (GLU, ignoring sigmoid) | degrees add |
| softmax | $0$ if saturated; fractional $\epsilon=\partial\log\mathrm{softmax}/\partial\log t>0$ otherwise |
| LayerNorm / RMSNorm | **reset to 0** (upstream weights invisible to scaling) |
| bias | degree-0 additive term; breaks homogeneity, $\nu_{\rm eff}$ becomes non-integer |
| residual sum | mixture of path degrees |

**Effective degree.** The scaling test measures
$$\nu_{\rm eff}=\frac{d\log f}{d\log t}\Big|_{t=1}=\sum_\pi\nu_\pi\frac{g_\pi}{g},$$
the margin-weighted mean of path degrees.

**Which path wins under ℓ2.** Amplitude $A$ on a degree-$\nu$ path costs
$\nu A^{2/\nu}$ (3.1); larger $\nu$ is more concave, so given enough mass the
max-degree path carries the margin and $\nu_{\rm eff}\to\max_\pi\nu_\pi$.

**1-layer transformer, no LN, no biases.** Paths:
$W_UW_OW_VW_E$ (4), $W_UW_{\rm out}\,\mathrm{relu}(W_{\rm in}W_E)$ (4),
$W_UW_{\rm out}\,\mathrm{relu}(W_{\rm in}W_OW_VW_E)$ (**6**). Max = 6,
measured $\nu_{\rm eff}=6.6$ (logit) / $6.1$ (margin) on 148 grokked
checkpoints: $6+\epsilon_{\rm softmax}$.

**MLP families used in nudial** (layout = #linear layers before/after the
nonlinearity): relu $\nu=2+\text{pre}+\text{post}$; quad $\nu=2(1+\text{pre})+1+\text{post}$.

**Predictions (unrun).** 2-layer transformer: max path 8–10 → $K^*\approx2$–3.
Final-LN transformer: $\nu=3$ → broad plateau $K\sim5$–11. Shared-embedding
quadratic: $\nu=5$ → 3–4 (observed). Biases → sub-integer $\nu_{\rm eff}$.

- Status: path rule derived from homogeneity + (3.1); transformer value
  explained; predictions untested.

---

## 10. Probabilistic form $P(K\mid\nu)$

Boltzmann over the ansatz margin curve with an architecture tilt:

$$P(K\mid\nu,p)\ \propto\ \exp\!\big(\beta\,\tilde\gamma(K,\nu,p)+\gamma_{\rm tilt}K\big),\qquad
\tilde\gamma=\frac{K^{-\nu/2}M_K(p)}{\max_{K'}K'^{-\nu/2}M_{K'}(p)} .$$

Fit on relu $m=512$, $p=113$: $\beta=9.1$, $\gamma_{\rm tilt}=0.30$; mean
$|K-\hat K|\approx0.5$. $\gamma_{\rm tilt}$ is $\delta_{\rm arch}$ of §8 in
log-probability form; it tracks init scale ($-3.14$ at scale 0.1 → $+0.12$
at 3), and small-init nets have *lower* $\gamma$ (§7), i.e. the tilt's
init-dependence is a seed barrier, not better regularisation.

Calibration requires replacing $\tilde\gamma$ by the *measured*
$\gamma(K)$ curve (AdamW at fixed frequency support). Unrun.

---

## 11. Ledger

| claim | status |
|---|---|
| trained direction maximises $g/\|\theta\|^\nu$ | theorem (Lyu–Li), for GF/homogeneous |
| ℓ2 on degree-$\nu$ chain $\Rightarrow$ $\ell_{2/\nu}$ budget on amplitudes | algebra |
| sparse for $\nu\ge3$, dense for $\nu=2$ (within ansatz) | algebra + numerics |
| $K^*=c\nu/(\nu-2)$, $c\approx2.1$ at $p=113$ | numerics in the window |
| $M_K(p)\to0$, fixed $K$ | theorem (pigeonhole) |
| $M_K\approx2\pi^2\gamma_K'p^{-2/K}$ | derived; verified $K\le4$ |
| $K^*\approx4\log p/(\nu-2)$ asymptotically; grows with $\log p$ | derived; numerics 113–4001 |
| $c\approx2$ in the $K\gtrsim\log p$ plateau | open |
| equal amplitudes at $K\le K^*$ | numerically harmless ($\le2.5\%$) |
| trained nets have the highest measured $\gamma$ | measured |
| $\delta_{\rm arch}\approx+1$ (ReLU), dense (one-hot quad) | mechanism identified, magnitude not derived |
| $\nu=\max$ path degree; transformer $=6+\epsilon$ | derived; matches 6.1–6.6 |
| $K$ vs $\log p$ in trained nets; depth; LN | unrun |
