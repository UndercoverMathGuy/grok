# "No deep holes in a well-separated cyclic orbit": status of the lemma (two-sided version)

Directory: `/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/unifiedB/` (all scripts and outputs).

**Target (corrected by the coordinator).** For $S\subseteq F=\{1,\dots,\frac{p-1}2\}$, weights $u_l\ge0$ on $S$, $U=\sum u_l$, phases $\theta_l$, and $Q(w)=\sum_lu_l\cos(2\pi lw/p+\theta_l)$:
$$\sigma_2(S,u):=\inf_\theta\ \tfrac12\operatorname{osc}_wQ/U\ \ \ge\ \sigma(t,\kappa)\quad\text{for all }u\ge0\text{ on }S,$$
where coverage is $\min_{\delta\ne0}\sum u_l\kappa_l(\delta)\ge tU$. The one-sided quantity $\sigma_1(S,u)=\inf_\theta(-\min_wQ)/U$ is used only as a tool.

Two facts used throughout:
* $\sigma_2\ge\sigma_1$. Indeed $\max_wQ_\theta=-\min_wQ_{\theta+\pi}$, so $\frac12\operatorname{osc}Q_\theta=\frac12(-\min Q_{\theta+\pi})+\frac12(-\min Q_\theta)\ge\sigma_1U$.
* Parseval/Bhatia–Davis: $Q$ has mean 0 and variance $\frac12\sum u_l^2$, so $\sigma_2\ge1/\sqrt{2K_{\rm eff}}$, where $K_{\rm eff}=U^2/\sum u_l^2$.

Notation: $\Lambda=\{k\in\mathbb Z^S:\sum k_ll\equiv0\}$ (relations), $|k|_1=\sum|k_l|$, $\Lambda_D=\{k\in\Lambda:0<|k|_1\le D\}/\pm$. $Y=\frac1U\sum_lu_l\cos2\pi\mathsf U_l$ with $\mathsf U_l$ i.i.d. uniform (the "independent model").

## 0. Summary

**Proved.**
1. **With $K_{\rm eff}\le\kappa\ln p$ the lemma is false, also in the two-sided form, for every $\kappa>0$ and every $t<1$, and $K_{\rm eff}$ gives exactly Parseval.** Thm 1 constructs data with coverage $\ge t$ and
$$\frac1{\sqrt{2K_{\rm eff}}}\ \le\ \sigma_2\ \le\ \frac{\sqrt2}{\sqrt{K_{\rm eff}}}+O\Big(\frac{\ln p}{\sqrt p}\Big).$$
The construction is a Gauss-sum "thin layer" on all of $F$, which supplies coverage $\approx f$ at no cost in $K_{\rm eff}$ and no oscillation, plus a Rudin–Shapiro progression of length $m\approx K_{\rm eff}(1-f)^2$.
So under a $K_{\rm eff}$ hypothesis the best $p$-free statement is Parseval itself, $\sigma\asymp1/\sqrt{\kappa\ln p}\to0$. The same holds for $e^{\text{Shannon entropy}}$ (Thm 1', sketch).
The size hypothesis must count frequencies with Rényi order $<1$, e.g. $|S|\le\kappa\ln p$; the application has this ($S\subseteq\operatorname{spt}a$). With it the lemma is the "no unicorn" statement.
Quantitatively, $\sigma_2<0.18$ in this family needs $K_{\rm eff}\gtrsim60$ in absolute terms, whatever the coverage. That is $\kappa\gtrsim60/\ln p$ ($\approx3.7$ at $p=10^7$, $2.6$ at $p=10^{10}$).
Without a thin layer (support size), glued flat blocks need $\kappa\approx50$–$60$ at $t=0.5$ (first-moment estimate, §3, not a theorem); this confirms the coordinator's estimate.
2. **Theorem A (only short relations matter).** For any polynomial $q$ of degree $D$ and all phases,
$\frac1p\sum_wq(Q(w)/U)=\sum_{k\in\Lambda,|k|_1\le D}\hat\psi_q(k)e^{ik\cdot\theta}$ **exactly**, with explicit Bessel-series coefficients.
Phase-uniform lower bounds on $\sigma_1$ and directly on $\sigma_2$ (by excluding every interval of half-width $\sigma_0$) therefore need only the finite set $\Lambda_D$. No hypothesis on long ("random") relations is needed, and no (RM).
*Cor. A2:* if $\Lambda_{2n+1}=\varnothing$ then $\sigma_2\ge\sigma_1\ge|\xi_{n+1}(u)|$, the smallest Gauss node of $Y$, with **no coverage hypothesis**.
If $\Lambda_{2n}=\varnothing$ then $\sigma_2\ge w_n(Y)$, the least half-width of an interval carrying a measure with the first $2n$ moments of $Y$.
*Cor. A3:* a uniformly random $S$ with $|S|\le\kappa\ln p$ is w.h.p. relation-free up to length $(1-\varepsilon)\alpha_0(\kappa)|S|$, so Lemma$'$ holds for it.
3. **Obstruction (Prop. C).** The $k=0$ term of the Laplace/Bessel identity gives, for every code with coverage $t$, the packing bound $\prod I_0(\lambda u_l)\le e^{\lambda U}/p+e^{\lambda(1-t)U}$.
The minimum-coverage tilt (either side) needs the right side below $2\prod I_0$. It therefore works only within a factor 2 of this bound, and as $p\to\infty$ it certifies nothing at any $\kappa$.
Minimum coverage gives no dual-distance information (adjoin $2a$), and only a weak $\sqrt{2\pi d}(1-t)^d$ bound on short-relation density.
The missing input is **stability**: near-maximal coverage $\Rightarrow$ few short relations. Given that, Theorem A finishes.

**Numerical** ($u=y$, the programme weights; exact over $\mathbb Z_p$):
* Theorem A certificates (phase-uniform, rigorous up to floating point):
  * $\sigma_1\ge0.23$–$0.43$ on 20 programme supports, $p=101$ to $100003$, $K=7$–$22$. The value decreases with $K/\ln p$: $\approx0.40$–$0.43$ at $K/\ln p\approx1.3$–$1.4$, $\approx0.25$–$0.32$ at $K/\ln p\approx1.9$–$2.4$;
  * direct two-sided certificates $\sigma_2\ge0.30$–$0.44$ on 9 supports, often above the one-sided ones (e.g. $0.395$ vs $0.317$ at $p=101$; $0.439$ at $p=5003$, $K=11$);
  * on the sparsest supports $\sigma_1\ge0.42$–$0.43$ already ($p=5003$, $K=11$; $p=10007$, $K=12$; $p=100003$, $K=16$; and $0.417$ at $p=30011$, $K=14$).
* Adversarial upper bounds: $\sigma_2\le0.52$–$0.61$ (9 supports) and $\sigma_1\le0.42$–$0.60$. The needed $0.18$ is certified on every support tested; $0.42$ only on the sparsest ones.
* The min-coverage tilt certifies 0 on all 40 supports tested.
* **Programme supports repel short relations**: 83 of 85 have no 3-term relation (2 in total, against 77.8 expected for random sets of the same sizes), and they have 235 four-term relations against 468 expected. From length 5 on the counts are random-like.
* Progression + greedy-cover sets ($p=1009$, $K=11$): coverage falls to $0.20$–$0.50$ (optimum $0.55$–$0.59$), and adversarial $\sigma_2\ge0.45$ in every case. No unicorn found.
* Gauss nodes at $n=\alpha K$ converge as $K\to\infty$, apparently to $\sqrt{2\alpha(1-\alpha)}$; equal weights are the worst case in all trials.

**Conjectural.** Lemma$'$ (two-sided, $|S|\le\kappa\ln p$) holds.
The sharper and more usable conjecture is **(Stab)**: near-optimal covers have small leakage on $\Lambda_D$ for $D\le(1-\varepsilon)\alpha_0(\kappa)K$. With (Stab), Theorem A gives $\sigma_2\ge\sigma_1\gtrsim\sqrt{\alpha_0(1-\alpha_0/2)}$, i.e. $\approx0.43,0.36,0.31$ at $\kappa=1.5,2,2.5$ in the limit, which is $\ge0.18$.

## 1. Geometric reformulation

$E(\theta)_l=e^{2\pi i\theta_l}$ and $\langle a,b\rangle_u=\sum u_la_l\bar b_l$ give $d_u(\theta,\theta')=\sum u_l\kappa(\theta_l-\theta'_l)=\frac12\|E\theta-E\theta'\|_u^2$, and $Q(w)=U-d_u(h_w,y)$ with $h_w=(lw/p)_l\in H$, $y=-\theta/2\pi$.
So coverage $t$ means that $H$ (cyclic, order $p$) is a code of minimum distance $tU$ in $\mathbb T^K$; the relation lattice $\Lambda$ is its dual.

* $\sigma_1\ge\sigma_0$ for all phases means that the covering radius of $H$ is $\le(1-\sigma_0)U$.
* $\sigma_2\ge\sigma_0$ means that no point $y$ has all of $H$ inside a shell $\{d_u(\cdot,y)\in[a,a+2\sigma_0U]\}$, i.e. the code has no thin "equatorial" shell around any point of the torus.

For random group codes, min distance and covering radius both sit at the GV radius; for general codes packing does not imply covering (in $\mathbb F_2^n$, $C'\otimes\mathrm{Rep}_2$ has a hole at exactly $n/2$, via dual words of weight 2).
Length-2 relations cannot occur here ($S\subseteq F$), and Parseval keeps $\sigma_2\ge1/\sqrt{2K_{\rm eff}}$. The covering behaviour is governed by the **dual** distance (Tietäväinen; Cor. A2), not by $t$.

## 2. The lemma with $K_{\rm eff}$ is false (also two-sided), and $K_{\rm eff}$ gives exactly Parseval

**Theorem 1.** Let $f\in(0,1)$, $m$ a power of 2 with $2m<\frac{p-1}2$, $A=\{1,\dots,m\}$, $T=F\setminus A$, $U=1$.
Set $c_j=\frac{1-f}m\varepsilon_{j-1}$ for $j\in A$ ($\varepsilon$ the Rudin–Shapiro signs) and $c_l=\frac f{|T|}e^{2\pi i l^2/p}$ for $l\in T$, and $Q(w)=\Re\sum_{l\in F}c_le^{2\pi ilw/p}$ (so $S=F$, $u_l=|c_l|$, $\theta_l=\arg c_l$). Then:
1. $K_{\rm eff}\le m/(1-f)^2$;
2. coverage $t\ge f\big(1-\frac{2m-1}{p-1-2m}\big)$;
3. $\frac12\operatorname{osc}Q\le\|Q\|_\infty\le(1-f)\sqrt{2/m}+\frac{2f(\sqrt p(2+\ln p)+m)}{p-1-2m}$, and $(1-f)\sqrt{2/m}\le\sqrt{2/K_{\rm eff}}$.

With $m=2^{\lfloor\log_2(\kappa(1-f)^2\ln p)\rfloor}$ this gives $K_{\rm eff}\le\kappa\ln p$, $t\to f$ and $\sigma_2,\sigma_1\le2/\sqrt{\kappa\ln p}+o(1)\to0$, for every $\kappa>0$ and $f<1$.

*Proof.*
(1) $\sum u_l^2\ge m(\frac{1-f}m)^2$.
(2) For $\delta\ne0$, $\sum_{l\in F}\cos(2\pi l\delta/p)=-\frac12$, so $\sum_{l\in T}\cos\le m-\frac12$ and $\sum u_l\kappa_l(\delta)\ge\frac f{|T|}(|T|-m+\frac12)$.
(3) Rudin–Shapiro pairs satisfy $|P_m|^2+|Q_m|^2=2m$ on the circle, so $\|Q_A\|_\infty\le(1-f)\sqrt{2/m}$.
For the thin layer, complete the incomplete Gauss sum: $\sum_{l=1}^Ne(\frac{l^2+lw}p)=\sum_j\hat1_{[1,N]}(j)G(w+j)$ with $|G|=\sqrt p$, and $\sum_j|\hat1_{[1,N]}(j)|\le\frac Np+\frac1p\sum_{j=1}^{p-1}\frac1{\sin(\pi j/p)}\le2+\ln p$ (using $\sin\pi x\ge2x$ on $[0,\frac12]$). Remove the $m$ terms of $A$. $\square$

Numerics (`cex_keff.py`, `cex_twosided.py`; exact FFT, $f=\frac12$, $t=0.500$):

| $p$ | $m$ | $K_{\rm eff}/\ln p$ | $\frac12\operatorname{osc}/U$ | Parseval floor $1/\sqrt{2K_{\rm eff}}$ | ratio | $\sigma_1$ (one-sided, these phases) |
|---|---|---|---|---|---|---|
| 100003 | 4 | 1.39 | 0.261 | 0.177 | 1.48 | 0.264 |
| 100003 | 16 | 5.56 | 0.150 | 0.088 | 1.70 | 0.151 |
| 100003 | 64 | 22.2 | 0.088 | 0.044 | 1.98 | 0.089 |
| 1000003 | 8 / 32 / 64 | 2.32 / 9.26 / 18.5 | 0.234 / 0.124 / 0.086 | 0.125 / 0.063 / 0.044 | 1.88 / 1.99 / 1.94 | – |
| 10000019 | 4 / 8 / 16 / 32 / 64 | 0.99 / 1.99 / 3.97 / 7.94 / 15.9 | 0.259 / 0.234 / 0.148 / 0.124 / 0.086 | 0.177 / 0.125 / 0.088 / 0.063 / 0.044 | 1.46 / 1.87 / 1.67 / 1.98 / 1.93 | 0.262 / – / 0.149 / – / 0.087 |

The ratio to the Parseval floor stays $\le2$, as the theorem says. Optimised (ultraflat-type) phases on $A$ would push it toward $\sqrt2$.
For $\sigma_2<0.18$ one needs $K_{\rm eff}\gtrsim(2/0.18)^2/2\approx62$ with Rudin–Shapiro blocks, e.g. $m=16$, $f=\frac12$, giving $0.150$ at $K_{\rm eff}=64$. In $\kappa$ units that is $\kappa\approx62/\ln p$.

**Theorem 1' (Shannon version; sketch).** Replace $T$ by $N=(\ln p)^c$ ($c>1$) random elements with random phases and weight $f<1/c$, and take $m\asymp(\ln p)^{(1-fc)/(1-f)}$.
Hoeffding (sampling without replacement) and a union bound give coverage $\ge f(1-\sqrt{2\ln(2p)/N})$ and $\|Q_T\|_\infty\le f\sqrt{2\ln(4p)/N}$, while $e^{H(u/U)}=O(\ln p)$. So $\sigma_2\to0$.
For Rényi order $\alpha<1$, $N_\alpha(u)\ge f^{\alpha/(1-\alpha)}N$: the thin layer must itself have $O(\ln p)$ elements and must itself be a unicorn.

## 3. The corrected lemma and the glued-block estimate

> **Lemma$'$.** If $|S|\le\kappa\ln p$ and coverage is $\ge t$ (for $u$), then $\sigma_2(S,u)\ge\sigma(t,\kappa)>0$, independent of $p$.

Parseval gives only $1/\sqrt{2\kappa\ln p}$.

**Glued flat blocks (no thin layer).** Take $r$ dilates $b_i\{1,\dots,m\}$, each with Rudin–Shapiro phases, which are two-sided flat with $\frac12\operatorname{osc}\le c_{RS}/\sqrt m$ relative ($c_{RS}\approx1.2$ measured, $\le\sqrt2$ proved); the triangle inequality keeps the union flat.
Coverage $t$ requires that no $\delta\ne0$ lie in the hole sets ($\|b_i\delta/p\|\lesssim1/m$) of more than $(1-t)r$ dilates. The first-moment count gives $\kappa\gtrsim m/((1-t)\ln(m/2)-H(t))$.
$\sigma_2<0.18$ needs $m\ge(c_{RS}/0.18)^2\approx44$–$62$, hence $\kappa\gtrsim44/((1-t)\ln22-H(t))\approx52$ at $t=0.5$ (and $\approx60$ with $c_{RS}=\sqrt2$). This confirms the coordinator's "$\kappa\sim60$".
**Numerical check of structure at programme sizes** (`structured.py`, `structured2.py`, $p=1009$, $K=11$; progression $a\{1..m\}$ of weight $1-f$ plus greedy cover):

| $m$ | $f$ | $t$ | adversarial $\sigma_2$ | adversarial $\sigma_1$ | Parseval floor |
|---|---|---|---|---|---|
| 0 (greedy cover only) | 1 | 0.551 | 0.507 | 0.464 | 0.213 |
| 4 | 0.3 / 0.5 / 0.7 | 0.255 / 0.376 / 0.498 | 0.476 / 0.486 / 0.503 | 0.364 / 0.400 / 0.443 | 0.26 / 0.22 / 0.22 |
| 6 | 0.3 / 0.5 / 0.7 | 0.240 / 0.366 / 0.443 | 0.454 / 0.500 / 0.562 | 0.343 / 0.415 / 0.525 | 0.22 / 0.21 / 0.24 |
| 8 | 0.3 / 0.5 / 0.7 | 0.197 / 0.286 / 0.336 | 0.457 / 0.573 / 0.682 | 0.374 / 0.495 / 0.655 | 0.21 / 0.24 / 0.30 |

Structure costs coverage (programme optimum $0.55$–$0.59$), and the two-sided value stays $\ge0.45$.

## 4. Theorem A: only short relations matter

For $k\in\mathbb Z^S$ and $q(x)=\sum_{j\le D}q_jx^j$ put $\hat\psi_q(k)=\sum_{j=0}^Dq_jU^{-j}j!\,[z^j]\prod_{l\in S}I_{|k_l|}(u_lz)$; this is $0$ if $|k|_1>D$.

**Theorem A.** For every phase vector $\theta$: $\displaystyle\frac1p\sum_{w\in\mathbb Z_p}q\Big(\frac{Q(w)}U\Big)=\sum_{k\in\Lambda,\ |k|_1\le D}\hat\psi_q(k)\,e^{ik\cdot\theta}$.
Consequently, if $q\le\eta$ on an interval $I$ and $\mathbb E\,q(Y)-2\sum_{k\in\Lambda_D}|\hat\psi_q(k)|>\eta$, then for **no** phase vector is the range of $Q/U$ contained in $I$.

* One-sided: take $I=[-\sigma_0,1]$; this gives $\sigma_1>\sigma_0$.
* Two-sided: if $\frac12\operatorname{osc}Q<\sigma_0U$, the range lies in $[c-\sigma_0,c+\sigma_0]$ for some $|c|\le\sigma_0$. Since $Q\mapsto-Q$ ($\theta\mapsto\theta+\pi$) preserves $Y$ and $|\hat\psi|$, one may take $c\ge0$. Excluding the finitely many intervals $I_j=[c_j-\sigma_0,c_j+\varepsilon+\sigma_0]$, $c_j=j\varepsilon$, then gives $\sigma_2\ge\sigma_0$.

*Proof.* Write $X(\vartheta)=\sum_lu_l\cos\vartheta_l=\sum_l\frac{u_l}2(e^{i\vartheta_l}+e^{-i\vartheta_l})$. The coefficient of $e^{ik\cdot\vartheta}$ in $X^j$ is $\sum j!\prod_l\frac{(u_l/2)^{a_l+b_l}}{a_l!b_l!}$ over $a_l-b_l=k_l$, $\sum(a_l+b_l)=j$, which equals $j![z^j]\prod_lI_{|k_l|}(u_lz)$.
Substitute $\vartheta_l=2\pi lw/p+\theta_l$ and use $\frac1p\sum_we^{2\pi iw(k\cdot S)/p}=[k\in\Lambda]$; note $\hat\psi_q(-k)=\hat\psi_q(k)$. $\square$

The identity was checked to $10^{-15}$ (`check_identity.py`). $\Lambda_D$ is enumerated exactly by meet-in-the-middle (`relations.py`, checked against brute force).
The tilt $e^{\pm\lambda Q}$ has full support in $\mathbb Z^S$, so it sees the $\approx e^{\lambda U}/p$ counting relations; a degree-$D$ test sees only $\Lambda_D$.

**Corollary A2 (dual distance $\Rightarrow$ no flat polynomials).** If $\Lambda_{2n+1}=\varnothing$ then $\sigma_2\ge\sigma_1\ge|\xi_{n+1}|$, where $\xi_{n+1}<0$ is the smallest zero of the orthogonal polynomial $\pi_{n+1}$ of the law of $Y$.
If $\Lambda_{2n}=\varnothing$ then $\sigma_2\ge w_n(Y):=\min\{\frac{b-a}2:$ some measure on $[a,b]$ has the moments $\mathbb EY^j$, $j\le2n\}$; for example $w_2\ge\sqrt3\,\mathrm{sd}(Y)(1-o(1))$.
No coverage hypothesis is needed.
*Proof.* For $\sigma_1$: if $Q\ge-\sigma_0U$ with $-\sigma_0>\xi_{n+1}$, take $q=(-\sigma_0-x)r^2$ with $r=\pi_{n+1}/(x-\xi_{n+1})$. Gauss–Christoffel gives $\mathbb Eq(Y)=(-\sigma_0-\xi_{n+1})\mathbb Er^2>0$, while Theorem A with $\Lambda_{2n+1}=\varnothing$ gives $\frac1p\sum q(Q/U)=\mathbb Eq(Y)$, although $q(Q/U)\le0$ pointwise.
For $w_n$: with $\Lambda_{2n}=\varnothing$ the empirical law of $Q(w)/U$ has the first $2n$ moments of $Y$. $\square$

**Corollary A3 (random supports).** Let $E(\alpha)=\min_{0<z<1}[\ln\frac{1+z}{1-z}-\alpha\ln z]$, so that $\#\{|k|_1\le\alpha K\}=e^{KE(\alpha)+o(K)}$, and $E(\alpha_0)=1/\kappa$; then $\alpha_0=0.202,0.135,0.100$ for $\kappa=1.5,2,2.5$.
A uniformly random $K$-subset of $F$ with $K\le\kappa\ln p$ has $\Lambda_D=\varnothing$ for $D=(1-\varepsilon)\alpha_0K$ with probability $1-p^{-c}$ (first moment, $\Pr(k\cdot S\equiv0)\le1/(|F|-K)$).
Hence $\sigma_2\ge|\xi_{n+1}(u)|$ for every $u$ on $S$. Pigeonhole on half-vectors caps the relation-free length of any $S$ at $2\alpha_0K$.

**Gauss nodes** (equal weights; exact rational moments, 300-digit Hankel/Jacobi; `gq_asym.py`):

| $n/K$ | 0.025 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|
| $K=40$ | 0.112 | 0.192 | 0.313 | 0.404 | 0.478 | 0.540 |
| $K=80$ | 0.137 | 0.224 | 0.347 | 0.439 | 0.512 | 0.572 |
| $K=160$ | 0.159 | 0.249 | 0.374 | 0.464 | 0.536 | 0.595 |
| $\sqrt{2\alpha(1-\alpha)}$ | 0.221 | 0.308 | 0.424 | 0.505 | 0.566 | 0.612 |

The values increase with $K$ toward $\sqrt{2\alpha(1-\alpha)}$. This is the cosine analogue of the Krawtchouk asymptotics behind Tietäväinen/MRRW; the limit is numerical, not proved.
Equal weights were the worst case in all trials (`gq_weights.py`, $K=30$, $n=2,4,6$: $0.222/0.359/0.461$ for equal weights, $\ge0.242/0.395/0.491$ for random weights).
Random $S$ ($n\approx\alpha_0K/2$): limit values $0.43/0.36/0.31$ at $\kappa=1.5/2/2.5$.

## 5. Proposition C: the minimum-coverage tilt is self-defeating

**Proposition C.** If $S$ has coverage $t$ then, for every $\lambda>0$,
$$\textstyle\prod_lI_0(\lambda u_l)\le Z(\lambda)=\sum_{k\in\Lambda}\prod_lI_{|k_l|}(\lambda u_l)=\frac1p\sum_\delta e^{\lambda\sum u_l\cos(2\pi l\delta/p)}\le e^{\lambda U}/p+e^{\lambda(1-t)U}=:B(\lambda).$$
The min-coverage tilt bounds $\pm\min Q$ (hence $\frac12\operatorname{osc}$) below by $\frac1{\lambda U}\log(2\prod I_0-B)$, which is positive only if $B\in[\prod I_0,2\prod I_0-1)$.
In rate form ($|S|=N\le\kappa\ln p$, $\beta=\lambda U/N$, $g_1=\frac1\kappa-\beta+\log I_0\beta$, $g_2=\log I_0\beta-\beta(1-t)$), Jensen gives $e^{-Ng_1}+e^{-Ng_2}\ge1$, so $\min(g_1,g_2)\le\ln2/N$.
The region "$\kappa<\kappa_{\rm tilt}(t)$" where the tilt would work is exactly what the same identity forbids, via the packing bound $t\le t_L(\kappa)=1/(\kappa\beta_\kappa)$ with $\beta_\kappa-\log I_0\beta_\kappa=1/\kappa$.
For reference, $t_L=0.5,0.6,0.7,0.8,0.9$ at $\kappa=0.74,0.87,1.08,1.50,2.75$.
Numerically the min-coverage tilt gives 0 on all 40 programme supports (`laplace_window.py`). The Note's tilt works only with the actual $Z(\lambda)$, i.e. under (RM).

## 6. What minimum coverage cannot give; the exact missing input

* **No dual-distance information.** Adjoining $2a$ (small weight $w$) to a good cover keeps coverage $\ge(1-w)t$ and creates a 3-term relation.
* **Weak weighted control.** With $W_d(\Lambda)=\sum_{k\in\Lambda,|k|_1=d}\prod\frac{(u_l/2)^{|k_l|}}{|k_l|!}$, Prop. C gives $W_d(\Lambda)\le\inf_\lambda\lambda^{-d}B(\lambda)$. Relative to $W_d(\mathbb Z^S)\approx U^d/d!$ this is a density bound of only $\approx\sqrt{2\pi d}(1-t)^d$ for $d\ll\ln p$.
* Coverage enters relation sums only through $Z(\lambda)$, which by Prop. C is useless at fixed margin.

**Sufficient extra hypothesis, checkable per support:** a degree-$D$ test excluding all intervals $I_j$ (§4). This is exactly what the certificates in §7 verify.
**Asymptotic form (Stab):** for near-maximal coverage at size $\kappa\ln p$, the leakage on $\Lambda_D$, $D=(1-\varepsilon)\alpha_0K$, is $o(1)$ of the quadrature margin. The data below (no 3-term relations, half the 4-term ones) is the first visible layer of (Stab).

## 7. Numerics

**Theorem A certificates, one-sided $\sigma_1$** (lower bounds for $\sigma_2$; `cert_poly.py`, `run_cert.py`). Each is an LP over degree-$D$ $q$ (Chebyshev basis) with the exact $\Lambda_D$. The margin is re-verified outside the LP (exact critical-point maximisation of $q$, exact leakage), with bisection on $\sigma_0$. $u=y$. "GQ" ignores relations. The last column is the adversarial one-sided value (an upper bound for $\sigma_1$). Small non-monotonicities in $D$ (e.g. $p=10007$, $K=14$) are LP/bisection tolerance; take the maximum over $D$.

| $p$ | $K$ | $K/\ln p$ | $t$ | $\Lambda_D$ by length | $D=5$ cert (GQ) | $D=7$ cert (GQ) | $D=9$ cert (GQ) | adversarial $\sigma$ |
|---|---|---|---|---|---|---|---|---|
| 101 | 7 | 1.52 | 0.731 | {3: 1, 4: 4, 5: 34, 6: 73, 7: 118} | 0.314 (0.455) | 0.317 (0.583) | – | 0.449 |
| 211 | 8 | 1.49 | 0.657 | {4: 3, 5: 28, 6: 72, 7: 153} | 0.337 (0.426) | 0.338 (0.551) | – | 0.485 |
| 401 | 8 | 1.33 | 0.592 | {5: 12, 6: 45, 7: 87} | 0.397 (0.426) | 0.397 (0.550) | – | 0.539 |
| 601 | 12 | 1.88 | 0.674 | {4: 4, 5: 62, 6: 267, 7: 854} | 0.273 (0.350) | 0.273 (0.460) | – | – |
| 1009 | 11 | 1.59 | 0.592 | {4: 3, 5: 15, 6: 87, 7: 318} | 0.347 (0.372) | 0.347 (0.485) | – | 0.495 |
| 2003 | 10 | 1.32 | 0.509 | {5: 1, 6: 25, 7: 98} | 0.377 (0.379) | 0.405 (0.496) | – | 0.575 |
| 2003 | 12 | 1.58 | 0.561 | {5: 12, 6: 74, 7: 276} | 0.337 (0.352) | 0.343 (0.461) | – | 0.502 |
| 2003 | 14 | 1.84 | 0.580 | {4: 10, 5: 15, 6: 171, 7: 782} | 0.297 (0.331) | 0.297 (0.435) | – | – |
| 2003 | 16 | 2.10 | 0.627 | {4: 5, 5: 67, 6: 404, 7: 1841} | 0.263 (0.309) | 0.263 (0.407) | – | – |
| 2003 | 18 | 2.37 | 0.655 | {4: 11, 5: 116, 6: 787, 7: 4225} | 0.232 (0.292) | 0.232 (0.386) | – | – |
| 5003 | 11 | 1.29 | 0.467 | {5: 2, 6: 15, 7: 58, 8: 180, 9: 474} | 0.362 (0.364) | 0.427 (0.477) | 0.429 (0.566) | 0.584 |
| 5003 | 19 | 2.23 | 0.622 | {4: 2, 5: 58, 6: 437, 7: 2464} | 0.268 (0.288) | 0.268 (0.381) | – | 0.424 |
| 10007 | 12 | 1.30 | 0.466 | {5: 1, 6: 8, 7: 46, 8: 186, 9: 500} | 0.348 (0.348) | 0.430 (0.457) | 0.430 (0.544) | 0.591 |
| 10007 | 14 | 1.52 | 0.509 | {5: 2, 6: 34, 7: 147, 8: 545, 9: 1858} | 0.323 (0.324) | 0.371 (0.427) | 0.362 (0.510) | 0.532 |
| 10007 | 16 | 1.74 | 0.540 | {5: 10, 6: 61, 7: 386} | 0.298 (0.304) | 0.323 (0.402) | – | 0.497 |
| 10007 | 20 | 2.17 | 0.586 | {4: 2, 5: 36, 6: 273, 7: 1727} | 0.257 (0.275) | 0.254 (0.364) | – | 0.424 |
| 30011 | 14 | 1.36 | 0.445 | {4: 1, 6: 5, 7: 38, 8: 188, 9: 636} | 0.324 (0.324) | 0.412 (0.427) | 0.417 (0.510) | 0.586 |
| 30011 | 20 | 1.94 | 0.542 | {5: 5, 6: 79, 7: 613} | 0.275 (0.277) | 0.315 (0.367) | – | 0.481 |
| 100003 | 16 | 1.39 | 0.423 | {5: 1, 6: 5, 7: 32, 8: 142, 9: 581} | 0.308 (0.308) | 0.400 (0.406) | 0.431 (0.486) | 0.600 |
| 100003 | 22 | 1.91 | 0.524 | {5: 3, 6: 47, 7: 275} | 0.260 (0.261) | 0.321 (0.346) | – | – |

**Direct two-sided certificates and adversarial two-sided values** (`twosided.py`; 8 sub-intervals, $u=y$):

| $p$ | $K$ | two-sided cert $D=5$ | two-sided cert $D=7$ | best one-sided cert | adversarial $\sigma_2$ |
|---|---|---|---|---|---|
| 101 | 7 | 0.395 | 0.395 | 0.317 | 0.525 |
| 211 | 8 | 0.395 | 0.395 | 0.338 | 0.537 |
| 401 | 8 | 0.403 | 0.403 | 0.397 | 0.575 |
| 1009 | 11 | 0.347 | 0.347 | 0.347 | 0.536 |
| 2003 | 10 | 0.376 | 0.432 | 0.405 | 0.589 |
| 2003 | 12 | 0.336 | 0.358 | 0.343 | 0.534 |
| 5003 | 11 | 0.362 | 0.439 | 0.429 | 0.608 |
| 10007 | 12 | 0.348 | 0.430 | 0.430 | 0.614 |
| 10007 | 16 | 0.298 | 0.353 | 0.323 | 0.518 |

**Short relations: programme supports vs random sets of the same size** (`rel_stats.py`, 20 random sets each; counts up to sign at lengths 3/4/5/6; $t_{\rm unif}$ = coverage with equal weights):

| $p$ | $K$ | support | random mean | $t_{\rm unif}$ support / random |
|---|---|---|---|---|
| 211 | 8 | 0 / 3 / 28 / 72 | 1.5 / 7.0 / 21.4 / 62.2 | 0.58 / 0.41 |
| 601 | 12 | 0 / 4 / 62 / 267 | 2.2 / 11.0 / 55.2 / 240.6 | 0.62 / 0.44 |
| 1009 | 11 | 0 / 3 / 15 / 87 | 1.2 / 5.0 / 22.0 / 81.6 | 0.53 / 0.38 |
| 2003 | 12 | 0 / 0 / 12 / 74 | 0.6 / 4.3 / 17.1 / 68.2 | 0.54 / 0.36 |
| 2003 | 20 | 0 / 23 / 182 / 1521 | 2.5 / 27.4 / 212.1 / 1465.3 | 0.66 / 0.51 |
| 5003 | 13 | 0 / 0 / 4 / 46 | 0.3 / 1.6 / 10.4 / 45.2 | 0.50 / 0.35 |
| 10007 | 16 | 0 / 0 / 10 / 61 | 0.4 / 3.0 / 13.8 / 80.8 | 0.52 / 0.36 |
| 10007 | 26 | 0 / 4 / 147 / 1400 | 1.2 / 15.4 / 156.8 / 1402.9 | 0.64 / 0.50 |

Over all 85 supports with $p\ge29$ (every rep, `rel34_all.py`): 2 three-term relations against 77.8 expected, and 235 four-term against 468.

## 8. Conjectural picture and next steps

1. In unified.tex (proof plan, step 1), replace $K_{\rm eff}$ by $|S|$ (or a Rényi order $<1$) and use the two-sided quantity. With $K_{\rm eff}$ the statement is false even two-sided (Thm 1); only Parseval survives.
2. Recast step 1 as **Theorem A + (Stab)**. The analytic half is proved and involves only short relations. The open half is combinatorial: efficient covers have no relations of length $\le(1-\varepsilon)\alpha_0K$.
   Heuristic mechanism: a length-$d$ relation inflates the coverage-ball volume seen by the code by $\approx1+2(1-t)^d$, which costs coverage rate at fixed $p$. Making this rigorous needs upper bounds on optimal coverage (GV-tightness type).
3. Constants and uniformity.
   * Theorem A's certificates on programme supports, with $u=y$, already exceed the relations-case need ($0.18$) at every $p$ tested, and reach $\approx0.4$ on sparse supports.
   * The application needs uniformity over $u\le\tau a$ on $L'$. Theorem A suits this: it needs no coverage of $u$, so the blind-spot step is unnecessary on this route; and $\Lambda_D(L')\subseteq\Lambda_D(S)$, so one enumeration per support serves all $u$, with only continuous $u$-dependence in moments and leakage. This uniform certificate is not done.
