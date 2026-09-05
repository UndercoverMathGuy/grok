# Adversarial review pack — committee size as an ℓ2 max-margin optimum

Prepared 2026-08-24 for hostile review. Everything the claims rest on is in
this one file: statements, derivations, exact numbers, the scripts that
regenerate them, prereg history including failures, prior art, and the
weaknesses we already know about. Repo: `grok/` (private). Data:
`runs_nudial{,.tar.gz,2,3}` (4320 runs), `runs_dial` (transformer cohorts),
`notes/*.json`.

**Reviewer instructions.** Every claim C1–C12 has a status tag, a
verification route V1–V6, and a "how to kill" line. We ask you to attack the
starred claims first — they carry the paper.

---

## 0. Task, notation, instruments

Modular addition $c=a+b \bmod p$, $p$ prime, one-hot inputs. Frequencies
$k\in\{1..\frac{p-1}{2}\}$ ($\pm k$ identified). Offset $\Delta=a+b-c$.

- Margin $g(\theta)=\min_{(a,b)}[f_{a+b}-\max_{c\ne a+b}f_c]$, full $p^2$ grid.
- Homogeneity degree $\nu$: $f(t\theta)=t^\nu f(\theta)$.
- Normalised margin $\gamma=g/\|\theta\|_2^\nu$ (scale-free).
- Fourier committee: project final logits onto $\cos(2\pi k(a+b-c)/p)$;
  energy share $s_k$.
- Instruments: `K_gap` = size at the largest log-gap in the top-12
  amplitudes, 2% floor (**invalid on dense spectra** — returns 1–2 when ~56
  amplitudes tie; this bit us once, see P-N1); `K_eps` = #{$s_k\ge0.005$};
  `k_eff` $=1/\sum_k s_k^2$ (participation ratio; the only instrument that is
  well-defined when amplitudes are unequal).
- "Grokked" = final test acc ≥ 0.99; cells scored on grokked runs only.

MLP families (nudial): one-hot → [pre linear layers] → {ReLU | $x^2$} →
[post linear layers] → logits. $\nu_{\rm relu}=2+\text{pre}+\text{post}$,
$\nu_{\rm quad}=2(1+\text{pre})+1+\text{post}$. Widths $m\in\{128,512,2048\}$,
$d=r=128$, $p=113$, frac_train 0.5 unless stated, MLX-semantics AdamW
(lr 1e-3, β 0.9/0.98, eps 1e-8, decoupled wd, no bias correction), 10k epochs,
init $N(0,1/\text{fan\_in})\times$`init_scale`. Trainer:
`nudial/train_nudial.py` (idempotent, seeds logged in every `config.json`;
META_SEED 20260822, init seeds from [200000,300000), data mask seed = init
seed mod 1000). Analyzer: `nudial/analyze_nudial.py` →
`notes/nudial_summary.json` (per-run rows + cell medians + prereg tests).

Transformer cohorts (earlier program): 1-layer, no LayerNorm, attention +
ReLU MLP, `runs_dial`, 148 grokked checkpoints across widths, $p\in\{113,127,157\}$.

---

## 1. The claims

Tags: **[T]** theorem/algebra · **[D]** derived under stated assumptions ·
**[N]** numerical/exact computation · **[E]** experimental · **[O]** open.
★ = load-bearing.

| # | claim | tag | verify |
|---|---|---|---|
| C1 | Late-time CE+wd training of a $\nu$-homogeneous net targets $\max\gamma$ | [T] cited (Lyu–Li), [E] for AdamW | §2.1, V3 |
| C2 ★ | Plain ℓ2 on a degree-$\nu$ circuit chain ⇒ amplitude budget $\sum_k A_k^{2/\nu}\le$ const ⇒ sparse committees for $\nu\ge3$, dense for $\nu=2$ | [T] algebra | §2.2 |
| C3 ★ | Within the equal-amplitude single-frequency ansatz, $K^*(\nu,p)=\arg\max_K K^{-\nu/2}M_K(p)\approx c(p)\,\nu/(\nu-2)$; at $p=113$: dense/6/4/4/3 for $\nu=2..6$ | [N] | V1 |
| C4 ★ | $M_K(p)\to0$ for fixed $K$ (pigeonhole); $M_K\approx2\pi^2\gamma'_Kp^{-2/K}$ in the lattice regime; hence $K^*$ grows $\sim4\log p/(\nu-2)$ — **not** $p$-independent | [T]+[D]+[N] | §2.4, V2 |
| C5 ★ | Trained relu MLPs realise the dial: k_eff falls monotonically with $\nu$ (Spearman −1.0 within family), modal K width-invariant 16× for $\nu\ge4$, dense at $\nu=2$; trained K = optimum+1 | [E] 4320 runs | V3 |
| C6 | The transformer's $\nu$ is structural: max-degree path E→OV→MLP→U = 6; measured $\nu_{\rm eff}$ = 6.6 logit / 6.1 margin; its 3:74/4:80 committee split is the $\nu=6$, $p\approx10^2$ crossover ($K^*=3$ with 4 at 94%) | [D]+[E] | §2.6, V5 |
| C7 | Morwani reconciliation: their dense max-margin theorem is an $L_{2,\nu}$-norm result; under the norm actually trained with (ℓ2) the optimum is sparse for $\nu\ge3$. At $\nu=3$ specifically, quad one-hot nets are dense under BOTH norms (their cell; we do not contest it) | [T]+[E] | §2.2, §3 P-N5 |
| C8 ★ | The optimum count has an architecture term: one-hot circuits emit cross-talk ⇒ dense is the genuine ℓ2 optimum for one-hot quad (hand constructions + brute force agree); a shared embedding before the nonlinearity removes it (2.3–2.5× k_eff drop at fixed $\nu$); for relu/embedding nets the residual leak ⇒ +1 | [E]+[D mechanism, magnitude not derived] | V3, §2.7 |
| C9 ★ | AdamW-trained $K^*{+}1$ nets have the **highest measured** $\gamma$ of anything produced — 7–20× above direct margin-ascent, 1.6–2× above plain GD+ℓ2; so margin-ascent outputs are lower bounds, and the +1 is the ansatz undercounting, not sub-optimality | [E] | V3 |
| C10 | Dynamics residue: init scale 0.1 → sparser nets with *lower* $\gamma$ (seed barrier); memorisation phase sets *which* extra frequency (88–90% pre-grok top-K) but not the count (frac 1.0 gives same K); the training-subset margin landscape is not the cause (subset optimum = full optimum) | [E] | V3 |
| C11 | $P(K\mid\nu)\propto\exp(\beta\tilde\gamma(K,\nu)+\gamma_t K)$ fits with $\beta=9.1$, $\gamma_t=0.30$, mean $|K-\hat K|\approx0.5$; $\gamma_t$ is C8's term and tracks init scale (−3.14 at 0.1 → +0.12 at 3) | [E fit, not calibrated] | V4 |
| C12 | Predictions made and not yet run (falsifiers): relu-4 at $p=1009$ → $K\approx8$; relu-6 → $\approx6$; 2-layer transformer → $K$ 2–3; final-LN transformer → $\nu=3$, broad $K\sim5$–11; measured $\gamma(K)$ at fixed support peaks at $K^*{+}1$ | [O] | §6 |

We do **not** claim: a theorem outside the ansatz (C3 is "optimum within the
family"); a derivation of the +1's magnitude; a derivation of the leak
plateau $c\approx2$; novelty of the $\ell_2$-vs-$L_{2,3}$ *observation* (Tian
2410.01779 flags it first); $p$-independence (we claimed it before 08-23; it
is false and retracted, see C4).

---

## 2. Reasoning and math, in full

### 2.1 Why $\gamma$ (C1)

Lyu & Li 2019: for $\nu$-homogeneous nets under gradient flow on
exponential-tail losses, after separation the direction converges to a KKT
point of $\max g/\|\theta\|^\nu$. Weight decay changes the norm scale and the
timescale, not the stationary directions. Grokking is this late phase.
**Gap we acknowledge:** AdamW is not gradient flow and the theorem does not
cover it. Our use is: (i) motivation, (ii) an empirical check — C9 shows the
AdamW endpoint has higher $\gamma$ than everything else we can construct,
which is what the target predicts. An adversary can attack the yardstick
itself: see W1 in §5.

### 2.2 The $\ell_{2/\nu}$ budget (C2, C7)

A frequency-$k$ circuit of degree $\nu$ is a chain of weight blocks with
norms $r_1..r_\nu$; amplitude $A_k\propto\prod r_i$, ℓ2 cost $\sum r_i^2$.
AM–GM: for fixed product the sum of squares is minimised at
$r_i=A_k^{1/\nu}$, so $\text{cost}(A_k)=\nu A_k^{2/\nu}$; gradient flow
conserves $r_i^2-r_j^2$, so balance is preserved, not assumed. Total budget
$\|\theta\|^2=1$ (WLOG, $\gamma$ scale-free):

$$\sum_{k\in S}A_k^{2/\nu}\le\tfrac1\nu.$$

$\nu=2$: $\ell_1$ ball — linear, no concentration preference, dense wins
(margin grows with K, §2.3). $\nu\ge3$: $\ell_{2/\nu}$ quasi-ball, concave
cost, concentration is cheaper per unit amplitude — the only sparsity source
in the theory. Credit for the primitive (ℓ2 over $D$ factors = $\ell_{2/D}$
on the product): Hoff 1611.00040; Kolb 2502.02496.

Morwani et al. 2311.07568 Thm 7 maximises margin under
$\big(\sum_j\|\theta_j\|_2^\nu\big)^{1/\nu}$ (their $L_{2,\nu}$; p.9 "we will
primarily focus on maximum margin with respect to the $L_{2,\nu}$ norm";
their experiments train with $L_{2,3}$ regularisation, not weight decay —
verified by direct read of the PDF, 2026-08-22). Under $L_{2,\nu}$ the
per-frequency budget is linear ⇒ dense. Same task, different norm, opposite
optimum. There is no contradiction with their theorem; there is a
contradiction with citing their theorem as a statement about weight-decay
training.

### 2.3 The trade-off (C3)

Ansatz: net = sum of single-frequency circuits over committee $S$, logits
$\sum_{k\in S}A_k\cos(2\pi k\Delta/p)$. Then
$g=\min_{\Delta\ne0}\sum_kA_k(1-\cos\frac{2\pi k\Delta}p)$. Equal amplitudes
($A=(\nu K)^{-\nu/2}$) give, up to a $\nu$-only constant,

$$\gamma(K)=K^{-\nu/2}M_K(p),\qquad
M_K(p)=\max_{|S|=K}\min_{\Delta\ne0}\sum_{k\in S}\Big(1-\cos\frac{2\pi k\Delta}p\Big).$$

$M_K$ is non-decreasing in $K$ (each summand ≥0 for every $\Delta$). Where
$M_K\approx K-c$ locally, $\frac{d}{dK}[-\frac\nu2\log K+\log(K-c)]=0$ gives
$K^*=c\nu/(\nu-2)$. At $p=113$ ($c\approx2.1$): integer argmax 6, 4, 4, 3
for $\nu=3..6$; sparse/dense margin ratios 2×, 6.5×, 25×, 97×.

Equal-amplitude check (V1 `unequal_mass`): free amplitudes on the optimal
support gain ≤2.5% for $K\le K^*$ at every $\nu$, and never change the
integer argmax; gains up to 45% past $K^*$ are solutions that park most mass
on $\approx K^*$ members — i.e. the assumption fails only where "K" itself
stops being well-defined, which is why k_eff is the primary instrument.
Optimal sets are additive-collision-free (no $k_i\pm k_j\equiv\pm k_l$),
matching the empirical veto law from the transformer program (0/70
distinct-trio finals with collisions).

### 2.4 $M_K(p)$ exactly: the leak is not a constant (C4)

**Theorem (pigeonhole).** For any $S$, $|S|=K$: map $\Delta\mapsto
(k_1\Delta/p,..,k_K\Delta/p)\bmod1\in\mathbb T^K$; $p$ points; cut
$\mathbb T^K$ into $N^K<p$ cubes of side $1/N$, $N=\lfloor(p-1)^{1/K}\rfloor$;
two points share a cube; their difference $\Delta'\ne0$ has
$\|k_i\Delta'/p\|<1/N\ \forall i$. Hence
$$M_K(p)<K\big(1-\cos(2\pi p^{-1/K})\big)\approx2\pi^2Kp^{-2/K}\xrightarrow{p\to\infty}0.$$
So "$M_K\approx K-2.1$" is a finite-$p$ window, not a law.

**Lattice regime** ($K\ll\log p$): $\{(k_1\Delta,..,k_K\Delta)\bmod p\}$
lifts to a lattice $L_S\supset p\mathbb Z^K$, $\det=p^{K-1}$; small-angle,
the adversary's best $\Delta$ is the shortest vector outside $p\mathbb Z^K$;
Minkowski–Hermite $\lambda_1^2\le\gamma_K p^{2(K-1)/K}$ ⇒
$M_K\lesssim2\pi^2\gamma_Kp^{-2/K}$ with $\gamma_2=2/\sqrt3$,
$\gamma_3=2^{1/3}$, $\gamma_4=\sqrt2$. Verified (V2): $M_Kp^{2/K}$ over
$p=113..4001$ → 22.3 ($K{=}2$, bound 22.8), 23.5 ($K{=}3$, 24.9), 23.7
($K{=}4$, 27.9), monotone, under the bound. $c_2\to2$ exactly (1.81, 1.90,
1.95, 1.98, 1.99, 1.99).

**Equidistribution regime** ($K\gtrsim\log p$): rigorous bounds only.
Collision-free second moments ($\sum_\Delta f=-K$,
$\sum_\Delta f^2=\frac{pK}2-K$, $f\ge-K$) force
$\max_\Delta f\ge\frac12-\frac1{4K}+o(1)$; random sets give
$\sim\sqrt{K\log p}$; the observed optimised plateau $\approx2$ is between
and **underived** (Chowla-cosine-type extremal problem).

**Consequence.** $\log\gamma(K)=-\frac\nu2\log K+\log(2\pi^2\gamma_K)-\frac2K\log p$
with $\gamma_K\sim K/2\pi e$ gives $K^*\approx4\log p/(\nu-2)$
asymptotically. Computed (V2, local search $K\le12$, 40/15 restarts —
$K\ge7$ rows are lower bounds on $M_K$):

| $p$ | 113 | 211 | 401 | 1009 | 2003 | 4001 |
|---|---|---|---|---|---|---|
| $\nu=3$ | 6 | 6 | 9 | 11 | 11 | 12+ |
| $\nu=4$ | 4 | 5 | 5 | 7 | 7 | 9 |
| $\nu=5$ | 4 | 4 | 5 | 5 | 6 | 6 |
| $\nu=6$ | 3 | 4 | 4 | 5 | 5 | 6 |

$c_4=K-M_4$: 2.12, 2.57, 2.92, 3.29, 3.48, 3.63 — not a constant.
**Retraction:** we previously wrote "$K^*$ essentially $p$-independent"; that
was an artifact of testing $p\in[59,211]$. Correct statement: $K^*$ depends
on $p$ through $\log p$, flat over the range any of our training runs cover
($\log p\in[4.6,5.1]$). This *reconciles* McCracken 2505.18266's $O(\log p)$
count with our $O(1)$-at-fixed-$p$ observation. $\nu=6$ runner-up ratios:
$p=113$ $K^*{=}3$ (4 at 94%), $p=211$ $K^*{=}4$ (3 at 99.6%) — the observed
transformer 3/4 near-tie sits on this crossover.

### 2.5 What training actually does (C5, C9, C10)

Headline dial (A cohort, 1920 runs, grokked only, medians; V3):

| family-$\nu$ | k_eff m=128 / 512 / 2048 | modal K_gap (512) | theory $K^*(p{=}113)$ |
|---|---|---|---|
| relu-2 | 27.2 / 45.1 / 54.0 | dense | dense |
| relu-3 | 5.96 / 7.5 / 7.96 | 12 | 6 (plateau 5–11) |
| relu-4 | 3.78 / 4.25 / 4.48 | 5 | 4 (5 within 2%) |
| relu-5 | 2.56 / 3.32 / 3.18 | 4 | 4 |
| relu-6 | 2.16 / 2.99 / 2.89 | 4 | 3 (4 within 6%) |
| quad-3 | 30.6 / 44.0 / 51.0 | dense | 6 — **wrong; see C8** |
| quad-4 | 12.6 / 15.0 / 16.0 | – | 4–5 — **wrong; see C8** |
| quad-5 | 9.57 / 9.7 / 10.1 | 11 | 4 — **wrong; see C8** |

(V3 prints modal K_gap = 1 on the dense cells relu-2/quad-3/quad-4 — that is
the documented instrument failure on flat spectra, W5; the table writes
"dense" from K_eps/k_eff instead.)

Spearman(k_eff, $\nu$): −1.0 within relu, −1.0 within quad, −0.76 pooled
(the pooled prereg bar was −0.8: **narrow fail**, caused by the cross-family
offset that became C8). Width: modal K identical across 16× width for every
$\nu\ge4$; k_eff ratios (2048/128) 1.06–1.34. relu-6 histogram (m=512)
3:15, 4:47, 5:29 vs transformer dial cohort 3:20, 4:92, 5:33.

The margin ledger (relu-4, $p{=}113$, $m{=}512$; identical $\gamma$ formula
over the full grid; V3):

| source | n | median $\gamma$ | K mode | k_eff | amplitude profile |
|---|---|---|---|---|---|
| D: margin ascent, random init | 32 | 6.4e-7 | 4 | 2.4 | [1,.61,.39,.13] |
| H: margin ascent, frac-0.5 subset | 32 | 2.7e-7 | 3 | 1.9 | |
| I: plain GD+ℓ2 (wd 3e-4), grokked | 18 | 2.4e-6 | 3 | 1.6 | [1,.53,.18,.03] |
| F: AdamW, init scale 0.1, grokked | 24 | 2.5e-6 | 3 | 1.3 | |
| J: margin ascent warm-started from AdamW | 32 | 2.1e-6 (**regressed**) | 5 | 3.9 | |
| A: AdamW standard, m=512, grokked | 96 | 4.4e-6 | 5 | 4.3 | [1,.85,.69,.60,.46] |
| G: AdamW, frac 1.0 (full table) | 96 | **4.7e-6** | 5 | 3.9 | |
| G: AdamW, frac 0.5 wd 1 | 32 | **5.0e-6** | 5 | 4.0 | |

relu-6: D 2.2e-9 (K=3), H 2.1e-9 (K=3), GD 2.8e-8 (n=8, K=2), AdamW A
4.0e-8 / G frac-1.0 4.5e-8 / G frac-0.5 4.7e-8 (all K=4). All rows are the
unfiltered per-cohort medians V3 prints; `notes/nudial_results.md` quotes
slightly different H/F values from a converged-runs subfilter (n=25/28) —
the ordering and every conclusion are identical either way.

Killed candidate explanations of the +1 (each with its test): finite-temp
objective (D's CE phase already at optimum-K), instrument ghost (weakest
member at 0.33–0.44 of max), wd level (flat across 0.3/1/3), AdamW $\ell_\infty$
box (max|θ|·wd 0.13–0.63, never near 1/wd), mask-specific circuit (lesion
cuts train 0.40× and test 0.39× equally — generalising), unfinished pruning
(transformer 25k→50k frozen), memorisation remnant *for the count*
(frac 1.0: K = 5.6 / 3.8, same as frac 0.5), subset landscape (H optimum
K=3 both families, test acc 1.0), Adam-vs-GD geometry as *distortion*
(GD is sparser and **worse** by $\gamma$).

What memorisation does do: the extra frequency is in the pre-grok top-K in
88–90% of runs (random non-member: 1%), and full data thins it
(relu-6 k_eff 2.9→2.0).

### 2.6 $\nu$ for arbitrary architectures (C6)

Path rule: $\nu_\pi$ = number of weight tensors on an input→logit path;
ReLU ×1; $x^d$ multiplies incoming degree by $d$; gated units add branch
degrees; softmax contributes 0 saturated / fractional $\epsilon>0$ else;
LayerNorm resets to 0 (upstream invisible to scaling); biases break
homogeneity. $\nu_{\rm eff}=\sum_\pi\nu_\pi g_\pi/g$ = what the scaling test
$d\log f/d\log t|_{t=1}$ measures. Under ℓ2 the max-degree path wins (cost
$\nu A^{2/\nu}$ most concave), so $\nu_{\rm eff}\to\max_\pi\nu_\pi$ given
mass.

1-layer no-LN transformer paths: $W_UW_OW_VW_E$ (4);
$W_UW_{\rm out}\mathrm{relu}(W_{\rm in}W_E)$ (4);
$W_UW_{\rm out}\mathrm{relu}(W_{\rm in}W_OW_VW_E)$ (**6**). Measured on 148
grokked checkpoints (V5, `notes/nu_eff_dial.json`, committed **before** any
nudial result was read): $\nu_{\rm eff}$ = 6.6 logit / 6.1 margin, IQR
6.5–6.7, width-flat. Reading: $6+\epsilon_{\rm softmax}$, the Nanda circuit
path. The MLP cells have exact structural $\nu$ and reproduce the
transformer's committee statistics at $\nu=6$ with no attention (§2.5
histogram) — the cross-architecture unification claim.

### 2.7 The architecture term (C8)

Mechanism. ReLU of a cosine has harmonics
($\mathrm{relu}(\cos\phi)=\frac1\pi+\frac12\cos\phi+\frac2{3\pi}\cos2\phi-\dots$).
Expand a neuron with preactivation $\cos(ka{+}\phi_i)+\cos(kb{+}\psi_i)$ in
$(a,b)$: terms $\cos(mka+nkb+m\phi_i+n\psi_i)$. With readout
$\propto\cos(kc-\phi_i-\psi_i)$ and phase averaging across neurons, the
$(1,1)$ term survives as the signal $\frac12\cos(k\Delta)$; the $(2,0)$ term
survives **iff phases are tied** ($\psi_i=\phi_i$), landing at $(2a-c)$ —
not a function of $\Delta$. One-hot quadratic circuits $(u_a{+}v_b)^2w_c$
have exactly this tied-phase cross-talk ($2u_av_bw_c$); it adds incoherently
($\sqrt K$) while signal adds coherently ($K$), making density nearly free.
Evidence that dense is the *genuine* one-hot-quad optimum, not a dynamics
trap: hand-built Morwani constructions at $p=113$ rise monotonically in
$\gamma$ with K (K=4 2.7e-7, K=6 1.8e-6, K=12 5.4e-6, dense 5.4e-6) and the
brute-force D search lands just above dense (6.1e-6) — the prereg P-N7
quad kill condition was HIT and we accepted it.

Placement test (E cohort, fixed $\nu$): one-hot→ReLU→2-readout ($\nu$=4)
k_eff 8.84 vs embed→ReLU→1-readout 4.19 vs 2-embed→ReLU 3.89; quad $\nu$=5
one-hot 9.70 vs embedded 3.85. Shared embedding before the nonlinearity =
the sparsity switch (2.3–2.5×), degree modulates within each class.

For relu/embedding nets the residual (untied-phase) leak is small; the +1
with democratic amplitudes cancels it better than $K^*$ concentrated ones —
consistent with the amplitude profiles and with C9's margin ordering.
**The magnitude (+1 exactly) is not derived**; it is measured (C9) and
representable as the tilt in C11.

---

## 3. Pre-registration history, including failures

All preregs written before the corresponding runs (`nudial/README.md`, three
dated rounds; runs 2026-08-22/23 on rented 5090s, pods deleted after pull).

| test | bar | outcome |
|---|---|---|
| P-N1 monotone dial | Spearman ≤ −0.8 both instruments | **narrow FAIL** pooled (−0.76; −1.0 per family); K_gap arm invalid on dense cells (instrument error, acknowledged) |
| P-N2 K=4 at $\nu$=4 | modal K_gap=4, ≥60% in {3,4,5} | **FAIL** (modal 5; 32%) — resolved by C8/C9: optimum is 4, trained is 4+1; "sharp 4" was an overclaim (4/5 within 2% in the exact table) |
| P-N3 dense at $\nu$=2 | ≥80% K_eps ≥ 45 | PASS (83%) |
| P-N4 width | relu-2 grows ≥2×; $\nu\ge4$ ≤1.3× | mostly PASS (relu-2 1.99; relu-6 1.34 misses 1.3) |
| P-N5 norm swap | $L_{2,3}$ dense, plain-wd control sparse | **UNINFORMATIVE** — control also dense (quad-3 is dense under both norms; Tian/He were right about this cell) |
| P-N6 brute-force optimum | – | relu-2 dense ✓; quad-3 sparser than trained; relu-4 **optimizer failure** (negative margins) — led to the two-phase sphere optimiser of round 2 |
| P-N7 optimum vs trained | trained within ±2 of optimum; quad-3 optimum sparse else kill | relu part PASS (optima 4/3/3/dense = $K^*$ exactly); **quad kill condition HIT** (optimum dense) — accepted, became C8 |
| P-N8 init scale | quad falls with small init; relu flat within 1.5× | **both halves failed as registered**: quad dense at every scale (consistent with C8); relu NOT flat (k_eff 1.34 at scale 0.1) — became the seed-barrier result (C10) |
| P-N9 placement | ≥2× between placements, shared-E smaller | PASS (2.3×/2.5×, direction right) |
| P-N10 frac sweep | K(frac 1.0) = optimum | **FAIL as registered** (K stays 5.6/3.8) — this is what killed the memorisation-remnant reading and forced C9 |
| P-N11 subset optimum | subset mode = full mode | PASS (K=3 both) |

Round-1 composite pass condition was **NOT MET**. The program continued
because the failures were informative (each failure became a claim: P-N2 →
C9, P-N5 → C7 scope, P-N7-kill → C8, P-N8 → C10). An adversary should check
we did not silently re-score any bar; the original bars are verbatim in
`nudial/README.md` and the scoring in `notes/nudial_results.md`.

---

## 4. Prior art and what is/isn't novel

Uniqueness audit 2026-08-22 (`notes/uniqueness_sparse_margin.md`; 33 Sonnet
agents, ~200 queries, 0 kills, 6 near-misses):

- **Morwani et al. 2311.07568**: dense max-margin under $L_{2,\nu}$;
  experiments train $L_{2,3}$. We build directly on their construction; our
  point is the norm, not an error in their theorem.
- **Tian 2410.01779**: already flags the $L_{2,3}$-vs-ℓ2 gap and proves ℓ2
  prefers lower-order solutions (their Thm 6). **We do not claim the
  gap-spotting.** Our claim: the quantitative $K^*(\nu,p)$, the $M_K$
  analysis, and the cross-architecture $\nu$ unification.
- **He et al. 2602.16849**: proves network-level dense under wd for
  quadratic ($\nu$=3) — consistent with our quad-3 cell; $\nu$=3 is the
  contested/flat case in general and we make no sharp claim there.
- **McCracken 2505.18266**: $O(\log p)$ frequency count. Was listed as a
  tension under our old $p$-independence claim; C4 dissolves it — they are
  right asymptotically, both agree at fixed $p$.
- **Hoff 1611.00040, Kolb 2502.02496**: the $\ell_{2/D}$ primitive; credited.
- **Liu et al. 2022**: verified by direct read to contain no
  wd-product law relevant here.
- **Lyu & Li 2019** (and Wei et al.): the implicit-bias target.

Claimed novel: (i) $K^*(\nu,p)=\arg\max K^{-\nu/2}M_K$ with the $M_K$
lattice analysis and the $\log p$ law; (ii) the $\nu$-dial experiment
(degree varied by inserting linear layers only, 4320 runs) and its match;
(iii) the measured-margin resolution of the +1 (trained > all constructed
"optima"); (iv) the shared-embedding cross-talk switch at fixed $\nu$;
(v) the transformer $\nu_{\rm eff}=6$ path identification tying (i)–(iv) to
the original grokking setting.

---

## 5. Known weaknesses (steelman before you start)

- **W1 (yardstick circularity).** C9 compares optimisers by $\gamma$, the
  functional our theory privileges. If $\gamma$ is the wrong late-time
  target for AdamW, "trained nets are optimal" is unsupported — though the
  $K$-dial data (C5) stands independently. Attack: exhibit a solution with
  higher $\gamma$ at $K=K^*$, or show AdamW's endpoint is not a KKT point of
  the $\gamma$ program.
- **W2 (post-hoc +1).** We predicted $K^*$, observed $K^*{+}1$, then
  explained it. The explanation is falsifiable (C12: fixed-support
  $\gamma(K)$ curve must peak at $K^*{+}1$) but that experiment is unrun.
- **W3 (margin-ascent weakness).** "Highest of anything we produced" — our
  direct maximisers are demonstrably weak (J regressed from its warm start).
  A stronger maximiser could in principle find a $K^*$ solution above
  4.9e-6, killing C9's reading.
- **W4 (ansatz).** $K^*$ is the optimum within single-frequency circuits.
  No theorem excludes better non-Fourier ℓ2 solutions; the empirical spectra
  are Fourier-committee-like, which is evidence, not proof.
- **W5 (instruments).** K_gap saturates at ~13 and is invalid on dense
  spectra (bit us in P-N1). k_eff conflates count and profile. We report
  three instruments and histograms; an adversary should check instrument
  agreement per cell in `notes/nudial_summary.json`.
- **W6 (one prime for training).** All 4320 MLP runs are $p=113$;
  cross-prime evidence is transformer-only (113/127/157) and spans too
  little $\log p$ to test C4's slope. C12's $p=1009$ run is the direct test.
- **W7 (local search).** $M_K$ for $K\ge7$ (and all $p\ge1009$ rows at 15
  restarts) are lower bounds; if under-optimised at large $p$, the $K^*$
  growth could be *underestimated* (the direction that would strengthen,
  not weaken, C4 — but the exact integers in the table could shift by 1).
- **W8 ($\nu$=3 plateau).** The objective is flat over K 5–11 at $\nu$=3 and
  two papers see dense there; we deliberately make no sharp $\nu$=3 claim.
  relu-3's modal K_gap 12 vs theory 6 is consistent with the plateau + the
  +1 term but is the weakest dial cell.
- **W9 (AdamW variant).** MLX-semantics AdamW (no bias correction). We
  believe this is immaterial (β2=0.98, 10k epochs); not ablated.
- **W10 (softmax term).** $\epsilon_{\rm softmax}=0.1$–$0.6$ is measured,
  not modelled; the path rule's "LN resets degree" prediction is untested.

---

## 6. Falsifiable predictions on the table (none run)

1. relu-4 MLP, $p=1009$, otherwise identical recipe: median K ≈ 7–8
   (vs 5 at $p=113$). Kill C4 if K stays ≤5.
2. relu-6 at $p=1009$: K ≈ 5–6 (vs 4). Same kill.
3. Fixed-support AdamW training (frequency-masked), relu-4 $p=113$:
   measured $\gamma(K)$ peaks at K=5, not 4. Kill C8/C9 if it peaks at 4.
4. 2-layer no-LN transformer, $p=113$: committees of 2–3. Kill C6 path rule
   if K stays 4–5.
5. Final-LN transformer: $\nu_{\rm eff}\approx3$ and broad K (5–11) with
   high seed variance. Kill "LN resets" if $\nu_{\rm eff}$ stays ≈6.
6. Any stronger margin maximiser at $K=K^*$: must stay below the AdamW
   $\gamma$ (4.7–4.9e-6 relu-4). Beating it kills C9's interpretation.

---

## 7. Verification scripts

All runnable from repo root with `uv run python`. V1/V2 are pure
numerics (no GPU, no run data); V3–V5 read committed JSON; V6 regenerates
raw runs (GPU).

### V1 — $M_K$ table, $K^*(\nu)$, equal-vs-free amplitudes

`nudial/kstar_table.py` (also inlined here in full):

```python
"""M_K(p) = max_{|S|=K} min_{Delta!=0} sum_{k in S} (1 - cos 2pi k Delta/p);
exhaustive over subsets (WLOG 1 in S) while C(nf-1,K-1) <= 4e6, else local
search; unequal_mass = free-amplitude refinement on the fixed support."""
import numpy as np, itertools, sys
rng = np.random.default_rng(0)

def G(p):
    k = np.arange(1, (p-1)//2 + 1); d = np.arange(1, p)
    return 1 - np.cos(2*np.pi*np.outer(k, d)/p)

def best_subset(p, K, exhaustive_limit=4_000_000, restarts=200):
    g = G(p); nf = g.shape[0]; others = np.arange(1, nf)
    if K == 1: return float(g[0].min()), (1,), 'exact'
    from math import comb
    if comb(nf-1, K-1) <= exhaustive_limit:
        best, bestS = -1, None
        it = itertools.combinations(others, K-1)
        while True:
            block = list(itertools.islice(it, 200_000))
            if not block: break
            idx = np.array(block)
            m = (g[0][None, :] + g[idx].sum(1)).min(1)
            j = m.argmax()
            if m[j] > best: best, bestS = m[j], (1,)+tuple(int(x)+1 for x in idx[j])
        return float(best), bestS, 'exhaustive'
    best, bestS = -1, None
    for r in range(restarts):
        S = set([0]) | set(rng.choice(others, K-1, replace=False).tolist())
        cur = g[list(S)].sum(0).min(); improved = True
        while improved:
            improved = False
            for out in list(S):
                if out == 0: continue
                base = g[[x for x in S if x != out]].sum(0)
                cand = (base[None, :] + g).min(1); cand[list(S)] = -1
                j = int(cand.argmax())
                if cand[j] > cur + 1e-12:
                    S.remove(out); S.add(j); cur = cand[j]; improved = True; break
        if cur > best: best, bestS = float(cur), tuple(sorted(x+1 for x in S))
    return best, bestS, 'local'

def unequal_mass(p, S, L, steps=3000):
    g = G(p)[[s-1 for s in S]]; K = len(S)
    n = np.ones(K)/np.sqrt(K); lr, beta = 0.02, 60.0
    best = ((n**L) @ g).min()
    for t in range(steps):
        f = (n**L) @ g
        w = np.exp(-beta*(f - f.min())); w /= w.sum()
        grad = L * n**(L-1) * (g @ w); grad -= n * (grad @ n)
        n = np.abs(n + lr*grad); n /= np.linalg.norm(n)
        best = max(best, ((n**L) @ g).min())
    return float(best)

if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:] or ['113']]:
        rows = {}
        for K in range(1, 13):
            M, S, how = best_subset(p, K)
            rows[K] = (M, S, how)
            print(f"p={p} K={K:2d} M_K={M:.4f} c={K-M:.3f} {how} S={S}", flush=True)
        for nu in (2, 3, 4, 5, 6):
            mg = {K: K**(-nu/2)*rows[K][0] for K in rows}
            mu = {K: unequal_mass(p, rows[K][1], nu) for K in rows}
            Kd = (p-1)//2; dense = Kd**(-nu/2)*(p/2)
            ks = max(mg, key=mg.get)
            print(f"p={p} nu={nu}: K*={ks} margin_eq={mg[ks]:.4g} "
                  f"(free-amp argmax {max(mu,key=mu.get)}) dense={dense:.3g} "
                  f"sparse/dense={mg[ks]/dense:.1f}x")
```

Verified output (2026-08-24), $p=113$: $M_K$ = 0.0015, 0.1875, 0.8446,
1.8842, 2.8943, 3.9060 (exhaustive through $K=6$), …, 9.6177 ($K=12$,
local); $K^*$ = 6/4/4/3 for $\nu$=3/4/5/6, free-amp argmax identical;
sparse/dense = 2.0×, 6.5×, 24.5×, 97.2×; at $\nu$=2 dense (1.01) beats the
best $K\le12$ set (0.80), as predicted.

### V2 — the $\log p$ law

`nudial/kstar_pscale.py` (same $M_K$ machinery, $p$ up to 4001, plus the
Hermite comparison). Output committed: `notes/kstar_pscale.json`,
`notes/kstar_pscale.log`. Reproduce:
`uv run python nudial/kstar_pscale.py 113 211 401 1009 2003 4001`
(~15 min CPU). Check: the $K^*(\nu,p)$ table of §2.4;
`M_K * p**(2/K)` → 22.3/23.5/23.7 for K=2/3/4; $c_2\to2$.

### V3 — every experimental number in §2.5 / §3 from committed JSON

```python
"""Regenerate the dial table, margin ledger, and prereg stats from
notes/nudial_summary.json (no GPU). Numbers must match §2.5."""
import json, numpy as np
from collections import Counter
d = json.load(open('notes/nudial_summary.json'))
rows = d['rows']
def med(rr, f): return float(np.median([r[f] for r in rr])) if rr else float('nan')
def sel(**kw):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()): out.append(r)
    return out
grok = [r for r in rows if r['mode'] != 'margin' and r['test_acc'] >= 0.99]
print("== dial (cohort A, grokked, k_eff median / modal K_gap) ==")
for fam in ('relu', 'quad'):
    for nu in (2, 3, 4, 5, 6):
        for m in (128, 512, 2048):
            rr = [r for r in grok if r['cohort'] == 'A' and r['family'] == fam
                  and r['nu'] == nu and r['m'] == m]
            if rr:
                print(f"{fam}{nu} m={m}: k_eff={med(rr,'k_eff'):.2f} "
                      f"K_gap mode={Counter(r['K_gap'] for r in rr).most_common(1)[0][0]} n={len(rr)}")
print("== Spearman(k_eff, nu) within family, m=512 ==")
from scipy.stats import spearmanr
for fam in ('relu', 'quad'):
    cells = [(nu, med([r for r in grok if r['cohort']=='A' and r['family']==fam
              and r['nu']==nu and r['m']==512], 'k_eff'))
             for nu in (2,3,4,5,6)]
    cells = [c for c in cells if not np.isnan(c[1])]
    print(fam, spearmanr([c[0] for c in cells], [c[1] for c in cells]).correlation)
print("== margin ledger (relu-4 then relu-6; median gamma / K_gap mode / k_eff) ==")
def ledger(nu):
    groups = {
      'D margin-ascent': [r for r in rows if r['cohort']=='D' and r['family']=='relu' and r['nu']==nu],
      'H subset-ascent': [r for r in rows if r['cohort']=='H' and r['nu']==nu],
      'I GD+l2 (grokked)': [r for r in rows if r['cohort']=='I' and r['nu']==nu and r['test_acc']>=0.99],
      'F small-init 0.1': [r for r in rows if r['cohort']=='F' and r['family']=='relu'
                           and r.get('init_scale')==0.1 and r['test_acc']>=0.99] if nu==4 else [],
      'J warm-refine': [r for r in rows if r['cohort']=='J' and r['nu']==nu],
      'AdamW std (A, m=512, grokked)': [r for r in grok if r['cohort']=='A' and r['family']=='relu'
                           and r['nu']==nu and r['m']==512],
      'AdamW frac1.0 (G)': [r for r in rows if r['cohort']=='G' and r['nu']==nu
                           and r['frac']==1.0 and r['train_acc']>=0.99],
    }
    for name, rr in groups.items():
        if rr:
            print(f"  relu-{nu} {name:28s} n={len(rr):3d} gamma={med(rr,'margin_norm'):.2e} "
                  f"K mode={Counter(r['K_gap'] for r in rr).most_common(1)[0][0]} k_eff={med(rr,'k_eff'):.1f}")
for nu in (4, 6): ledger(nu)
print("== E placement (k_eff by layout, fixed nu) ==")
for r in sorted({(x['family'], x['nu'], x['layout']) for x in rows if x['cohort']=='E'}):
    rr = [x for x in rows if x['cohort']=='E' and (x['family'],x['nu'],x['layout'])==r
          and x['test_acc']>=0.99]
    print(f"  {r}: k_eff={med(rr,'k_eff'):.2f} grokked={len(rr)}")
print("== F init scale (relu-4 / quad-3 k_eff by scale) ==")
for fam in ('relu','quad'):
    for sc in (0.1, 0.3, 1.0, 3.0):
        rr = [x for x in rows if x['cohort']=='F' and x['family']==fam
              and x.get('init_scale')==sc and x['test_acc']>=0.99]
        if rr: print(f"  {fam} scale {sc}: k_eff={med(rr,'k_eff'):.2f}")
```

Expected: the tables of §2.5 (dial medians ±0.1, ledger $\gamma$ medians as
printed, Spearman −1.0/−1.0, E 8.84/4.19/3.89 and 9.70/3.85, F relu
1.34/3.74/4.13/4.30).

### V4 — refit $P(K\mid\nu)$

```python
"""Boltzmann fit P(K) ∝ exp(beta*margin_curve_norm + tilt*K) on cohort-A
relu m=512 grokked K_gap counts; margin curve = K^{-nu/2} M_K(113)."""
import json, numpy as np
from collections import Counter
M = {1:0.0015,2:0.187,3:0.845,4:1.884,5:2.894,6:3.906,7:4.745,8:5.831,
     9:6.658,10:7.658,11:8.727,12:9.646}          # V1 output, p=113
rows = json.load(open('notes/nudial_summary.json'))['rows']
data = {nu: Counter(r['K_gap'] for r in rows
        if r['cohort']=='A' and r['family']=='relu' and r['nu']==nu
        and r['m']==512 and r['test_acc']>=0.99 and 1<=r['K_gap']<=12)
        for nu in (3,4,5,6)}
Ks = np.arange(1,13)
def nll(beta, tilt):
    tot = 0.0
    for nu, cnt in data.items():
        gam = Ks**(-nu/2)*np.array([M[k] for k in Ks]); gam /= gam.max()
        logp = beta*gam + tilt*Ks; logp -= np.log(np.exp(logp).sum())
        tot -= sum(n*logp[k-1] for k, n in cnt.items())
    return tot
grid = [(b, t, nll(b, t)) for b in np.arange(2,16,.25) for t in np.arange(-1,1,.05)]
b, t, v = min(grid, key=lambda x: x[2])
print(f"beta={b} tilt={t:.2f} nll={v:.1f}")
for nu, cnt in data.items():
    gam = Ks**(-nu/2)*np.array([M[k] for k in Ks]); gam /= gam.max()
    p = np.exp(b*gam + t*Ks); p /= p.sum()
    mean_pred = (Ks*p).sum(); obs = sum(k*n for k,n in cnt.items())/sum(cnt.values())
    print(f"nu={nu}: E[K] pred {mean_pred:.2f} obs {obs:.2f}")
```

Verified output (2026-08-24): `beta=9.0 tilt=0.30`; E[K] pred/obs =
9.11/9.18 ($\nu$3), 6.46/6.00 ($\nu$4), 4.75/4.58 ($\nu$5), 3.77/4.26
($\nu$6). (Original fit β=9.1 used a finer grid. Fit and data share the
same runs — this is a fit check, not calibration; see C11.)

### V5 — transformer $\nu_{\rm eff}$

```python
import json, numpy as np
d = json.load(open('notes/nu_eff_dial.json'))     # 148 grokked runs_dial ckpts
for f in ('all_nu_logit_med', 'all_nu_margin_med'):
    print(f, float(np.median([r[f] for r in d])))
```

Expected: 6.6 / 6.1 (IQR 6.5–6.7 logit). File committed before any nudial
result existed (git history is the timestamp).

### V6 — full retrain (GPU, ~4 h on one RTX 5090)

```
uv run python nudial/train_nudial.py --smoke      # CPU sanity, ~1 min
python nudial/train_nudial.py --cohort A          # then B..J per README
python nudial/analyze_nudial.py                   # -> notes/nudial_summary.json
```

Deterministic given seeds (seeds in every `config.json`; TF32 off).
Raw spectra for all 4320 runs: `runs_nudial.tar.gz` + `runs_nudial2.tar.gz`
+ `runs_nudial3.tar.gz`. The analyzer's prereg-test block prints P-N1…P-N11
directly.

---

## 8. File manifest

| file | contents |
|---|---|
| `nudial/README.md` | prereg rounds 1–3, verbatim bars |
| `nudial/train_nudial.py` | trainer (all cohorts A–J) |
| `nudial/analyze_nudial.py` | instruments + prereg scoring |
| `nudial/kstar_table.py` | V1 |
| `nudial/kstar_pscale.py` | V2 |
| `notes/nudial_results.md` | scored rounds 1–3, dated |
| `notes/nudial_summary.json` | 4320 per-run rows + cell medians + tests |
| `notes/kstar_pscale.json` | $M_K(p)$, $p\le4001$ |
| `notes/nu_eff_dial.json` | transformer scaling test, 148 ckpts |
| `notes/nudial_math.md` | long-form derivations (superset of §2) |
| `notes/uniqueness_sparse_margin.md` | prior-art audit report |
| `runs_nudial*.tar.gz` | raw spectra/metrics/configs |
