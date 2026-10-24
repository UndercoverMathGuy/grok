# Reductions to the clean lemma: blind spots, plane to line, compensation ($n=2$)

Scratch work for steps 2–4 of the proof plan in `heracles/conjecture20/unified.tex`. "Note, X" refers to `mechanism_note.tex`.
All scripts are in this directory, and every numerical claim names the output file it comes from.
**Proved** means proved here or in the Note. **Numerical** means a computer search, i.e. evidence and not proof.
**Conditional** means proved assuming the clean lemma (CL) or a hypothesis that is stated explicitly.

## 0. Setting and constants

We work at $n=2$, so $\nu=3$ and $\tau=3^{s-1}$. The numbers are quoted at $b=2$, i.e. $s=3/2$.
$c_{12}=1-(2/3)^{1/2}=0.1835$, $c_1=1-3^{-1/2}=0.4226$, $c_A=1/(2\sqrt3)=0.2887$, $\tau=\sqrt3$, $c_{12}\tau=\sqrt3-\sqrt2=0.318$.
We also use $s_l(\delta)=|\sin(\pi l\delta/p)|$, so $\kappa_l=2s_l^2$ and $|c_l|=2s_l$.

For an allocation $a\ge0$ write $Q=\sum a_l^r$, $\|a\|=Q^s$, $m(a)=\min_\delta(Ka)_\delta$, and $e=m(a)/(\Gamma\|a\|)$ with $\Gamma=\Gamma_s(p)$.
The participation number is $K_r(x)=(\sum x^r)^{s/(s-1)}/(\sum x)^{1/(s-1)}$ (Note, Thm. thm:theta), and $K_{\rm eff}(x)=(\sum x)^2/\sum x^2$.

**Relations case** ($T=\{1,2\}$): take $0\le v\le\tau a$ and phases $\theta$. Put $\varphi(w)=\sum_l v_l\cos(2\pi lw/p+\theta_l)$ and
$D_\delta(w)=\varphi(w)-\varphi(w-\delta)$. Proposition classC together with Lemma osc gives $\tfrac12(-\min F_{\delta_0}-\min F_{-\delta_0})=\tfrac12\operatorname{osc}D_{\delta_0}$.
With $\Pi=c_{12}|\beta_{12}|$ (P$_{12}$), Conjecture "no unicorns" reads, for $\delta_0\in\Delta(a)$ and $v\ne0$,

$$(\star)\qquad \tfrac12\operatorname{osc}D_{\delta_0}+(1/e-1)\,m(a)\;>\;c_{12}\,(Kv)_{\delta_0}.$$

The (O1) form is $Q(a,v,\theta):=\min_\delta[(Ka)_\delta+c_{12}(Kv)_\delta-\tfrac12\operatorname{osc}D_\delta]\le\Gamma\|a\|$. It lets the prover use any $\delta$, and it is legitimate because Lemma osc holds at every $\delta$.
In the language of (G), a non-binding $\mu$ costs $X(\mu)=\sum_\delta\mu_\delta((Ka)_\delta-m(a))$. By the master-inequality proof, $\mathrm{Det}_\mu+U+G>\mathrm{Pay}_\mu+X(\mu)$ for any one symmetric $\mu$ suffices.

**(CL), two-sided form** (as corrected by the coordinator). Take weights $w\ge0$ with coverage $t(w)=\min_{\delta\ne0}(Kw)_\delta/\sum w$ and $K_{\rm eff}(w)\le\kappa\ln p$. Then, for all phases,
$$\tfrac12\operatorname{osc}_x\textstyle\sum_lw_l\cos(2\pi lx/p+\psi_l)\ \ge\ \sigma(t,\kappa)\sum w .$$
The one-sided form $-\min\ge\sigma\sum w$ is false (one-sided-flat spikes; another agent's counterexample). It is used nowhere below.
Two consequences matter. First, $\max(\max P,-\min P)\ge\frac12\operatorname{osc}P$, so any one-sided *maximum of the two sides* is covered too. Second, the two-sided constant of a set is at most that of each subset, so a flat sub-block anywhere in the support matters.
Heuristic: $\sigma\approx C\sqrt{t/\kappa}$, with $C\approx0.8$ calibrated on the Note's measurement ($0.54K$ at $p=10007$, $K=13$).

---------------------------------------------------------------------------------------------------------------------------

## 1. Task 1: blind spots and arbitrary interference supports (relations case)

### 1.1 Which polynomial CL must be applied to (proved)

**Lemma 1.1 (two exact forms of the detection).**

(a) *Per offset.* $D_\delta(w)=\sum_l b_l(\delta)\cos(2\pi lw/p+\theta_l')$ with $b_l(\delta)=2v_ls_l(\delta)$ and shifted phases $\theta'$. Also $(Kv)_\delta=\sum_lb_l(\delta)s_l(\delta)$.

(b) *Averaged.* Let $\mu$ be any symmetric probability on $\mathbb Z_p^\times$ and put $u=v\odot K(\mu)$, i.e. $u_l=v_lK_l(\mu)$, and $\psi_\mu(w)=\sum_lu_l\cos(2\pi lw/p+\theta_l)$. Then
$\sum_\delta\mu_\delta\tfrac12\operatorname{osc}D_\delta=\sum_\delta\mu_\delta(-\min D_\delta)\ge\max(\max\psi_\mu,-\min\psi_\mu)$ and $\sum_\delta\mu_\delta(Kv)_\delta=\sum_lu_l$.

*Proof.*
(a) $\cos A-\cos(A-2\pi l\delta/p)=2\sin(\pi l\delta/p)\cos(A-\pi l\delta/p+\pi/2)$.
(b) First, $D_{-\delta}(w)=-D_\delta(w+\delta)$, so $\max D_\delta=-\min D_{-\delta}$, and $\mu$ is symmetric. For every $w$, $-\min D_\delta\ge\varphi(w-\delta)-\varphi(w)$. Averaging over $\mu$ gives $\mu*\varphi-\varphi=-\psi_\mu$, because symmetric $\mu$ multiplies the $l$-th coefficient by $1-K_l(\mu)$. In the same way $\sum\mu_\delta\max D_\delta\ge\max_w\sum\mu_\delta D_\delta(w)=\max\psi_\mu$. $\square$

Both forms are two-sided. In (a), $\mathrm{Det}=\frac12\operatorname{osc}D_{\delta_0}$ exactly. In (b), $\max(\max\psi_\mu,-\min\psi_\mu)\ge\frac12\operatorname{osc}\psi_\mu$. So, at a binding $\mu$:
* ($\star$) follows from the two-sided CL applied to **$b=b(\delta_0)$** if $\sigma(t(b),\kappa_b)\ge c_{12}\rho_b-(1/e-1)m(a)/\sum b$, where $\rho_b=\sum bs/\sum b\le1$;
* it also follows from CL applied to **$u=v\odot K(\mu)$** if $\sigma(t(u),\kappa_u)\ge c_{12}-(1/e-1)m(a)/\sum u$.
The weights entering CL are therefore never $v$ itself. Coverage and $K_{\rm eff}$ must be those of $b$ or of $u$.

### 1.2 Blindness at the evaluation offset costs nothing (proved)

**Lemma 1.2 (payment–coverage).** The payment is $c_{12}(Kv)_{\delta_0}=c_{12}\sum_l b_ls_l(\delta_0)\le c_{12}\sum b\,\sqrt{t_{\delta_0}(b)/2}$, where $t_{\delta_0}(b)=(Kb)_{\delta_0}/\sum b$ (Cauchy–Schwarz). In the $u$-form with $\mu=\frac12(1_{\delta_0}+1_{-\delta_0})$, a frequency with $\kappa_l(\delta_0)\le\eta$ has $u_l\le\eta v_l$.

This is the exact content of "at blind offsets the cheap features neither cover nor earn their discount". **Corollary.** Suppose $\sigma(t,\kappa)\ge C\sqrt{t/\kappa}$. Then CL closes $(\star)$ unless
$$t_{\min}(b)\ <\ \frac{c_{12}^2\,\kappa_b}{2C^2}\;t_{\delta_0}(b)\ \approx\ 0.026\,\kappa_b\,t_{\delta_0}(b)\qquad(C=0.8).$$
The $\sqrt t$ of the payment cancels the $\sqrt t$ of the detection. Only the ratio of the minimal coverage to the coverage *at $\delta_0$* matters.
A block that is blind at $\delta_0$ is harmless. The dangerous configurations are those that are **visible at $\delta_0$ and blind elsewhere**.

*Numerical* (`ratio_t.out`, all 27 programme vertices for $23\le p\le1009$ and all 59 near-optimal allocations for $2003\le p\le100003$, full profile $v=\tau a$).
$t_{\min}(b)/t_{\delta_0}(b)\in[0.11,0.61]$, while the requirement is at most $0.06$. So with full interference, CL in its heuristic form closes every computed allocation, with a factor $\ge6.6$ to spare row by row.

### 1.3 The min-coverage form of CL cannot be applied to sub-blocks (proved and numerical)

**Lemma 1.3 (Dirichlet).** Every set $B$ of $k$ frequencies has some $\delta\ne0$ with $\|l\delta/p\|<1/\lfloor p^{1/k}\rfloor$ for all $l\in B$. Hence every weight vector on $B$ has coverage $\le2\pi^2/\lfloor p^{1/k}\rfloor^2$ (pigeonhole on $\delta\mapsto(l\delta/p)_{l\in B}\in\mathbb T^k$).

For generic blocks a $t$-blind direction appears much earlier, once $\ln p\gtrsim kI(t)$. Here $I(t)=\sup_\lambda[\lambda(1-t)-\log I_0(\lambda)]$ is the Cramér rate of $1-\cos U$ (`cramer.out`): $I(0.1)=1.20$, $I(0.3)=0.58$. For example, a generic 30-block is $0.3$-blind somewhere once $p\gtrsim10^{7.5}$.

*Numerical* (`coverage.out`). Even the full profile of a programme vertex has $t(u)\in[0.05,0.48]$ and $t(b)\in[0.13,0.58]$. Every vertex and allocation computed has a **half-block**, i.e. the $\lfloor K/2\rfloor$ frequencies least visible at some $\delta_1$, that carries $12$–$73\%$ of $\sum u$ and has coverage $0.001$–$0.08$.

**Consequence.** Step 2 of the plan cannot be done literally. Reducing an arbitrary interference support to the min-coverage CL fails on sub-blocks, which have such blind spots at every $p$ when structured and for $p\ge e^{kI(t)}$ when generic. Two kinds of blindness must be separated:

* **accidental**: few blind offsets, the generic large-$p$ phenomenon. It is harmless for an *averaged-coverage* CL, in which each blind offset costs $1/p$ in the Laplace transform $\frac1p\sum_\delta e^{-\lambda(Kw)_\delta}$. This is the tilt / relation-mass (RM) route of `relations3/result.tex`, Prop. cond. **Proposal:** state CL with the hypothesis "relation mass $\frac1p\sum_{\delta\ne0}e^{-\lambda(Kw)_\delta}\le\eta e^{-E(\lambda w)}$" instead of min coverage.
* **structural**: Bohr sets of positive density. These are exactly the relation-rich blocks on which flat polynomials live (§1.4), and they have to be paid by the efficiency term (§1.5).

### 1.4 Structure of poorly covering blocks (proved)

**Lemma 1.4 (poor coverage ⇒ dilated short interval).** If $t(w)<t$, there is $\delta_1\ne0$ such that for every $\eta\in(t,2]$ the weight of $\{l:\kappa_l(\delta_1)>\eta\}$ is $<(t/\eta)\sum w$ (Markov at $\delta_1$).
Put $l'=\delta_1l$. This relabelling is an automorphism of the whole problem: it preserves $(Ka)_\delta$ after $\delta\mapsto\delta_1^{-1}\delta$, together with $\|a\|$, $\Gamma$, $e$, $\operatorname{osc}$ and the payments. After it, a fraction $\ge1-t/\eta$ of $w$ sits on $\{|l'|\le\varepsilon_\eta p\}$, where $1-\cos2\pi\varepsilon_\eta=\eta$.

**Lemma 1.5 (low-frequency blocks are a circle problem).** Let $w$ be supported on $|l'|\le\varepsilon p$, and let $\Psi(x)=\sum w_l\cos(2\pi l'x+\theta_l)$ on $\mathbb R/\mathbb Z$. Then
$|\min_{\mathbb Z_p}\psi-\min_{\mathbb T}\Psi|\le\pi\varepsilon\sum w$, because $|\Psi'|\le2\pi\varepsilon p\sum w$ and the grid spacing is $1/p$.

So for structurally blind blocks the detection is a **$p$-independent** problem about trigonometric polynomials on the circle. There flat polynomials exist. With Rudin–Shapiro signs on $\{1,\dots,K\}$ and uniform weights, $-\min\Psi\le\sqrt{2K}\,w=\sqrt{2/K}\sum w$. No CL-type constant holds for such blocks, so they have to be paid by efficiency.
(*Numerical*, `progblock_*.out`: adversarial phases on progression blocks reach $\tfrac12\operatorname{osc}\approx1.4\times$ the Parseval floor.)

### 1.5 The efficiency side (proved, with one explicit hypothesis)

**Lemma 1.6 (blind-set efficiency bound).** Split $a=a^R+a^B$, let $f=Q_B/Q$ be the budget share of $B$, and let $\Delta_B=\{\delta:(Ka^B)_\delta\le\eta\sum_Ba\}$. Then
$$e(a)\ \le\ \frac{\Gamma(\Delta_B)}{\Gamma_s(p)}(1-f)^s+\frac{\eta\sum_Ba}{\Gamma\|a\|},\qquad \Gamma(\Delta):=\sup_y\min_{\delta\in\Delta}(Ky)_\delta/\|y\|.$$
*Proof.* $m(a)\le\min_{\Delta_B}[(Ka^R)+(Ka^B)]\le\Gamma(\Delta_B)\|a^R\|+\eta\sum_Ba$, and $\|a^R\|=(1-f)^s\|a\|$. $\square$

The heuristic "the rest must over-cover the blind set, costing efficiency" is the statement
(H$_\Delta$): $\Gamma(\Delta_B)\le(1+\eta_\Delta)\Gamma_s(p)$ with small $\eta_\Delta$. Under it, ($\star$) holds whenever $(1+\eta_\Delta)(1-f)^s(1+c_{12}\tau\pi_B)\le1$, where $\pi_B=(Ka^B)_{\delta_0}/m(a)$ (ignore detection; payment $\le c_{12}\tau\pi_Bm(a)$).

*Numerical* (`blindcost_1009.out`): H$_\Delta$ is **far from true at $p=1009$**. Up to dilation, the Bohr set of a low-frequency block is $\{1..J\}$. Relative to $\Gamma(\{1..F\})$ computed by the same solver:

| $J$ | $F/2$ | $F/4$ | $F/10$ | $F/20$ | $F/50$ |
|---|---|---|---|---|---|
| $\Gamma(\{1..J\})/\Gamma(\{1..F\})$ | $1.15$ | $1.30$ | $1.62$ | $2.02$ | $2.89$ |

So covering a blind set is substantially cheaper than covering everything. The heuristic factor $(\ln p/\ln J)^{s-1}$ predicts $1.05$–$1.33$, which is too optimistic. The blind-set bound alone does not close anything at accessible $p$.

**Lemma 1.7 (thin blocks pay for themselves, with participation; proved).** Let $a=a^R+a^B$ and $v=v^R+v^B$. Suppose $Q(a^R,v^R)\le\Gamma\|a^R\|$, which is automatic when $v^R=0$. Then $Q(a,v)\le\Gamma\|a\|$ whenever
$$\Gamma\,(1-(1-f)^s)\ >\ C\,f^s\,K_r(a^B)^{1-s},\qquad C=\begin{cases}2(1+c_{12}\tau)&v^R=0\\2(1+c_{12}\tau+\tau)&\text{general}\end{cases}$$
At $s=3/2$ it suffices that $f<\Gamma^2K_r(a^B)/6.95$, resp. $f<\Gamma^2K_r(a^B)/37.2$.
*Proof.* Evaluate at the offset $\delta_R$ that closes $(a^R,v^R)$. We have $(Ka^B)+c(Kv^B)+\frac12\operatorname{osc}D^B\le(2+2c\tau+2\tau)\sum_Ba$, using $\operatorname{osc}(D^R+D^B)\ge\operatorname{osc}D^R-\operatorname{osc}D^B$; the term $\frac12\operatorname{osc}D^B$ is dropped when $v^R=0$. Also $\sum_Ba=Q_B^s/K_r(a^B)^{s-1}=f^s\|a\|K_r^{1-s}$, and $\Gamma\|a\|-\Gamma\|a^R\|=\Gamma\|a\|(1-(1-f)^s)$. $\square$

This is the quantitative form of "a thin block gains $O(f^s)$ but costs $\Omega(f)$". The extra factor $K_r(a^B)^{s-1}$ beats the Note's thin-frequency lemma by $\sqrt{|B|}$ for spread blocks. Since $\Gamma^2\approx\gamma_*^2/K^*$, the threshold is $f\lesssim0.06\,K_r(a^B)/K^*$ when $v^R=0$.

**Lemma 1.7b (tied minimum: lopsided help cannot lift a tied minimum; proved).** Let $a=a^R+a^B$, let $v$ be supported on $B$, and write $h_B(\delta)=(Ka^B)_\delta$ for the block's help profile. Let $\varepsilon\ge0$ and $\Delta'\subseteq\{\delta:(Ka^R)_\delta\le m(a^R)+\varepsilon\}$, the near-tied offsets of the rest. Then $Q(a,v)\le\Gamma\|a\|$ whenever
$$(1+c_{12}\tau)\,\big(\varepsilon+\min_{\Delta'}h_B\big)\ \le\ \Gamma\|a\|\,(1-(1-f)^s).$$
Write $\lambda=\min_{\Delta'}h_B/\sum_Ba$ for the block's relative help at its weakest tied offset. At $\varepsilon=0$, $s=3/2$, the condition holds for every share $f\le\Gamma^2K_r(a^B)/((1+c_{12}\tau)^2\lambda^2)=\Gamma^2K_r(a^B)/(1.74\lambda^2)$.
*Proof.* Evaluate the (O1) functional at $\delta'=\arg\min_{\Delta'}h_B$. There $(Ka)_{\delta'}\le m(a^R)+\varepsilon+h_B(\delta')$, $c_{12}(Kv)_{\delta'}\le c_{12}\tau h_B(\delta')$, and $-\frac12\operatorname{osc}\le0$. Also $m(a^R)\le\Gamma\|a^R\|=\Gamma(1-f)^s\|a\|$. $\square$

This formalizes the coordinator's observation that "flat blocks have lopsided help profiles and cannot lift a minimum tied across many spread-out offsets". A relation-rich block is nearly blind on its Bohr set. If one tied offset of the rest falls there, $\lambda$ is small, and the block is closed for every share once $\Gamma^2K_r\ge1.74\lambda^2$.
For example, at the coordinator's measured strength $\lambda=0.18$, a 32-block is closed for all $f$ as long as $\Gamma\ge0.042$, i.e. up to $\ln p\approx80$.
*Numerical* (`tiedmin.out`; 300 random dilations, exact binding sets of the best allocations at $p=1009,10007,30011,100003$):
* for progression blocks of length $16$–$64$, $\lambda<0.2$ in $5$–$37\%$ of dilations, more often for shorter blocks and larger $p$;
* random blocks never have $\lambda<0.5$ at these sizes;
* at the median $\lambda$ ($0.4$–$0.98$) the lemma still closes shares $f\le0.19$–$1.6$.

**Lemma 1.8 ($K_{\rm eff}$ of the interference; proved).**
(i) $K_{\rm eff}(x)\le K_r(x)$ for every $x\ge0$, by monotonicity of Rényi entropies with $r=1/s<2$.
(ii) Let $\mu\in\mathcal M(a)$, $v\le\tau a$ with $\operatorname{spt}v\subseteq B$, $u=v\odot K(\mu)$, and let $\pi=\sum u/(\tau m(a))\in[0,1]$ be the payment share. Then
$$K_{\rm eff}(u)\le K_r(u)\le\Big(\frac{2f^s}{\pi\,e\,\Gamma}\Big)^{1/(s-1)},\qquad\text{full interference }(f=\pi=1):\ \ K_{\rm eff}(u)\le\Big(\frac2{e\Gamma}\Big)^{\frac1{s-1}}\le\Big(\frac4e\Big)^{\frac1{s-1}}\lceil8\ln p\rceil .$$
*Proof.* $\sum u^r\le(2\tau)^rQ_B$ and $\sum u=\pi\tau e\Gamma Q^s$. Insert these into $K_r$ and use $\Gamma\ge\frac12\lceil8\ln p\rceil^{1-s}$ (paper, Thm 10). $\square$

So $K_{\rm eff}=O(\ln p)$ with an explicit constant ($16\lceil8\ln p\rceil/e^2$ at $b=2$) for any interference that pays a fixed fraction of its budget share. A large $K_{\rm eff}$ forces $\pi\lesssim f^s/\sqrt{K_{\rm eff}}$, a block that pays little, and Lemma 1.7 handles that case when it is thin.
*Numerical* (`coverage.out`): $K_{\rm eff}(u)/\ln p\in[0.38,1.46]$ on every computed allocation, against the bound $4/(e\Gamma)^2/\ln p\in[2.1,7.6]$.

### 1.6 Theorem 1 (conditional reduction) and the escaping class

**Theorem 1.** Assume (CL). In the relations case ($n=2$, $b=2$, any $p$) ($\star$) holds in each of the following cases:
(A) $e(a)\le1/(1+c_{12}\tau)=0.759$ (Note, effsidon);
(B) $\sum_{\operatorname{spt}v}\kappa_l(\delta_0)\le(c_{12}-(1/e-1)/\tau)_+^{-2}$, which is $29.7$ at $e=1$ (Parseval);
(C) the near-IF condition of Note, Thm. nearif(a);
(D) CL applied to $b(\delta_0)$ or to $u$ with the constants of §1.1. Under $\sigma\ge C\sqrt{t/\kappa}$ this means $t_{\min}(b)\ge0.026\,\kappa_b\,t_{\delta_0}(b)$;
(E) a decomposition $v=v^R+v^B$ with $(a^R,v^R)$ in (A)–(D) and $B$ thin in the sense of Lemma 1.7;
(F) $v$ supported on a block $B$ that is weak at some near-tied offset of the rest (Lemma 1.7b).

**Escapes.** A configuration that escapes satisfies all of the following:
* $e>0.759$;
* $\ge15$ interfering frequencies with $\sum\kappa(\delta_0)>29.7$;
* a part of $b$ that is visible at $\delta_0$ but $t$-blind at some $\delta_1$, with $t<0.026\kappa\,t_{\delta_0}$; by Lemma 1.4 it is concentrated on a $\delta_1$-dilated interval of length $2\varepsilon p$;
* that part is not thin and is strong ($\lambda$ large) at every near-tied offset of the rest.

If its blindness is accidental, it is closed by an averaged-coverage CL (§1.3). If it is structural, it is the circle problem of Lemma 1.5, and only efficiency can close it. Efficiency through H$_\Delta$ fails at $p=1009$ with useful constants. Lemma 1.7b, which needs a weak tied offset, and the direct test below close it anyway.

**How large $p$ must be.**
* *Proved:* nothing escapes for $p\le59$, by Cor. parseval with $\sum_F\kappa=p/2$.
* *Numerical:* every computed near-optimal allocation up to $p=100003$ has $\sum_S\kappa(\delta_0)\le19.8<29.7$ (`coverage.out`), so (B) closes it.
* *Numerical:* the Note's water-filling extrapolation (with P$_{12}$) puts the first allocations not closed by (A)+(B) at $p\approx3\cdot10^9$. Below that, CL is not needed at all.
* *Numerical:* full-profile escapes are absent at all computed $p$ (§1.2).
* *Accidental escapes:* the criterion needs $t_{\min}\lesssim0.03\,t_{\delta_0}$. A random $k$-block reaches this once $\ln p\gtrsim kI(0.03)=1.82k$, i.e. $p\gtrsim10^{12}$ for $k=15$ and $10^{24}$ for $k=30$ (`cramer.out`). However, the network chooses *which* features interfere. Among $\binom Kk$ sub-blocks the union bound loses $\ln\binom Kk$, which is $\approx39$ for $K=60$, $k=30$, and the measured adversarial half-blocks already have $t\le0.08$ at every $p$. So accidental escapes should be expected as soon as Parseval fails, i.e. from $p\approx3\cdot10^9$. They are the reason CL has to be stated with averaged coverage.
* *Structural escapes* (relation-rich blocks) are possible at every $p$, but they pay through efficiency. The direct test below gives a ratio $\ge3.96$ at $p=1009$ and $10007$. A scaling estimate (below) puts a progression block with $e\ge0.759$ at $\Gamma^{-2/3}\gtrsim5L^{1/3}$, i.e. $\ln p$ in the hundreds.

*Numerical: the most dangerous structural configuration* (`progblock.py`). The block is $B=\{1,\dots,L\}$, which up to dilation is every progression; it is relation-rich and flat polynomials exist on it. It has uniform weight $\beta$. The rest is optimised by reweighted LP over all of $F$ to cover everything. Interference $v=\tau a$ on $B$ only, with adversarial phases (chirp starts and local search), at the binding offset where $B$ pays most. Ratio $=(\mathrm{Det}+\text{slack})/\text{payment}$:

| $p$ | $L$ | $\beta L$ | $e$ | $f$ | $\pi_B$ | Det | Parseval | pay | slack | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1009 | 10 | 0.10 | 0.895 | 0.137 | 0.128 | 0.0904 | 0.0619 | 0.0406 | 0.1171 | 5.108 |
| 1009 | 10 | 0.30 | 0.829 | 0.271 | 0.302 | 0.2277 | 0.1649 | 0.0960 | 0.2069 | 4.528 |
| 1009 | 10 | 0.60 | 0.723 | 0.393 | 0.606 | 0.4827 | 0.3302 | 0.1925 | 0.3831 | 4.498 |
| 1009 | 10 | 1.00 | 0.645 | 0.512 | 0.326 | 0.4792 | 0.3127 | 0.1036 | 0.5516 | 9.952 |
| 1009 | 30 | 0.10 | 0.819 | 0.187 | 0.117 | 0.0482 | 0.0343 | 0.0373 | 0.2211 | 7.217 |
| 1009 | 30 | 0.30 | 0.724 | 0.358 | 0.332 | 0.1400 | 0.0998 | 0.1054 | 0.3813 | 4.944 |
| 1009 | 30 | 0.60 | 0.648 | 0.527 | 0.620 | 0.2717 | 0.1928 | 0.1969 | 0.5423 | 4.134 |
| 1009 | 30 | 1.00 | 0.543 | 0.659 | 0.510 | 0.3364 | 0.2258 | 0.1621 | 0.8407 | 7.263 |
| 1009 | 60 | 0.10 | 0.763 | 0.224 | 0.109 | 0.0331 | 0.0234 | 0.0348 | 0.3110 | 9.889 |
| 1009 | 60 | 0.30 | 0.642 | 0.416 | 0.314 | 0.0956 | 0.0686 | 0.0998 | 0.5580 | 6.547 |
| 1009 | 60 | 0.60 | 0.554 | 0.598 | 0.610 | 0.1866 | 0.1353 | 0.1938 | 0.8048 | 5.115 |
| 1009 | 60 | 1.00 | 0.482 | 0.766 | 0.499 | 0.2329 | 0.1580 | 0.1586 | 1.0757 | 8.250 |
| 1009 | 100 | 0.10 | 0.717 | 0.255 | 0.106 | 0.0249 | 0.0178 | 0.0337 | 0.3944 | 12.437 |
| 1009 | 100 | 0.30 | 0.574 | 0.458 | 0.311 | 0.0737 | 0.0529 | 0.0990 | 0.7408 | 8.230 |
| 1009 | 100 | 0.60 | 0.479 | 0.644 | 0.607 | 0.1450 | 0.1045 | 0.1930 | 1.0864 | 6.381 |
| 1009 | 100 | 1.00 | 0.413 | 0.819 | 0.495 | 0.1784 | 0.1218 | 0.1572 | 1.4239 | 10.191 |
| 1009 | 150 | 0.10 | 0.578 | 0.253 | 0.103 | 0.0199 | 0.0144 | 0.0329 | 0.7293 | 22.796 |
| 1009 | 150 | 0.30 | 0.464 | 0.455 | 0.310 | 0.0592 | 0.0431 | 0.0986 | 1.1542 | 12.306 |
| 1009 | 150 | 0.60 | 0.404 | 0.657 | 0.620 | 0.1192 | 0.0863 | 0.1972 | 1.4781 | 8.100 |
| 1009 | 150 | 1.00 | 0.370 | 0.872 | 0.960 | 0.1893 | 0.1385 | 0.3050 | 1.7051 | 6.212 |
| 10007 | 30 | 0.10 | 0.830 | 0.159 | 0.105 | 0.0454 | 0.0324 | 0.0334 | 0.2046 | 7.480 |
| 10007 | 30 | 0.30 | 0.757 | 0.312 | 0.311 | 0.1362 | 0.0966 | 0.0989 | 0.3211 | 4.625 |
| 10007 | 30 | 0.60 | 0.670 | 0.456 | 0.600 | 0.2617 | 0.1897 | 0.1907 | 0.4929 | 3.958 |
| 10007 | 30 | 1.00 | 0.551 | 0.563 | 0.458 | 0.3175 | 0.2141 | 0.1457 | 0.8147 | 7.772 |
| 10007 | 60 | 0.10 | 0.780 | 0.193 | 0.105 | 0.0322 | 0.0229 | 0.0334 | 0.2817 | 9.389 |
| 10007 | 60 | 0.30 | 0.674 | 0.364 | 0.316 | 0.0966 | 0.0688 | 0.1003 | 0.4833 | 5.782 |

Every ratio is $\ge3.96$, and $\ge4.5$ among rows with $e>0.759$. Even when the block is flat enough that Det $<$ payment ($L\ge60$), the efficiency loss from carrying a progression is $4$–$22\times$ the payment.
The rows at $p=10007$ (pool: 800 random frequencies plus the best support) match those at $p=1009$; there is no trend towards $1$.
Scaling estimate (heuristic): a progression block that supplies coverage on its visible set costs $L^{1/3}$ in $Q$, while the saving from the smaller blind set is $\approx\Gamma^{-2/3}\ln L/(3\ln p)\to0$. Keeping $e\ge0.759$ needs $\Gamma^{-2/3}\gtrsim5L^{1/3}$, i.e. $\ln p$ in the hundreds for $L\ge60$.

---------------------------------------------------------------------------------------------------------------------------

## 2. Task 2: plane to line (class $\mathcal C$, slot-sets $\{1\},\{2\},\{1,2\}$)

### 2.1 The right payment

**Lemma 2.1 (slot symmetry; proved).** Write $\beta_i=r_ie^{i\alpha_i}$, $\xi_i=e^{i\theta_i}$, $\eta_i=e^{i\phi_i}$ and set
$(\omega_0,\omega_1,\omega_2)=(\alpha+\frac{\theta+\phi}2,-\frac\theta2,-\frac\phi2)$. Then
$A=\sum re^{i(\omega_0+\omega_1+\omega_2)}$, $B=\sum re^{i(\omega_0-\omega_1+\omega_2)}$ (flip slot 1), $C$ (flip slot 2), and
$\bar D=\sum re^{i(-\omega_0+\omega_1+\omega_2)}$ (flip slot 0). The quantities $Q$ and $|A|$ do not depend on the labels.
Permuting $(\omega_0,\omega_1,\omega_2)$ gives a valid atom system ($\alpha'=\sum\omega'$, $\theta'=-2\omega_1'$, $\phi'=-2\omega_2'$). It permutes $(|D|,|B|,|C|)$ and fixes $|A|$ and $Q$.
**Hence every admissible payment may be taken symmetric in $(|\beta_1|,|\beta_2|,|\beta_{12}|)$, and P$_{12}$ gives $|A|-c_{12}|\beta_T|\le3^{1-s}Q^s$ for *each* single slot-set $T$ (P$_1$, P$_2$).** This settles the "numerically sharp single-slot constant" of the measurement protocol: it is $c_{12}$, and it is proved.

**Lemma 2.2 (the $\ell^1$ payment).** $\Pi_\Sigma(\beta)=c_{12}(|\beta_1|+|\beta_2|+|\beta_{12}|)$ is admissible
* **numerically:** the supremum of bonus$/\|\beta\|_1$ over $\le6$ atoms is $0.18350=c_{12}$ at $s=1.5$, $0.09640=c_{12}$ at $s=1.25$ and $0.03974=c_{12}$ at $s=1.1$. It is attained by two aligned atoms on one slot-set (`l1pay_*.out`);
* **proved on the axes** (Lemma 2.1);
* **proved where $\|\beta\|_1\ge2.303\|\beta\|_\infty$** ($c_1\max\le c_{12}\|\cdot\|_1$; Note, Thm c1sharp) **or $\|\beta\|_1\ge1.573\|\beta\|_2$** (Theorem A);
* **proved where $\|\beta\|_\infty\le c_{12}L/K_2$**, with $L=\sum_i|\beta_i|$ the atom mass and $K_2=1$ at $b=2$. Note, quadpay(i) gives bonus $\le K_2\|\beta\|_2^2/L\le K_2\|\beta\|_\infty\|\beta\|_1/L$;
* **proved everywhere with the constant $c_A=0.2887$** in place of $c_{12}$.

The open region is "one dominant $T$ with $|\beta_T|>0.18L$, plus small others". It needs a robust P$_{12}$: $|A|-c_{12}|D|\le3^{1-s}Q^s+c_{12}(|B|+|C|)$.
(`slotsym.py` checks Lemma 2.1 to $2\cdot10^{-15}$ on $2000$ random systems and all $6$ permutations.)

### 2.2 Lines

**Lemma 2.3 (line restriction; proved).** For $m\notin\{0,-1\}$ and any base point $x^*$,
$F_\delta(x^*+(u,mu))=\sum_{l,T}W_{l,T}(\delta)\cos(2\pi\cdot2l\,m_Tu/p+\psi'_{l,T})$, where $m_{\{1\}}=1$, $m_{\{2\}}=m$, $m_{\{1,2\}}=1+m$ and $W_{l,T}=|\beta_{l,T}||c_l(\delta)|$.
Call $m$ *coincidence-free* if the $3K$ residues $\pm l\,m_T$ are distinct. The bad $m$ are those with $l\,m_T\equiv\pm l'm_{T'}$; there are at most $6K^2+2$ of them.
For coincidence-free $m$, both restrictions of $F_{\pm\delta_0}$ are real polynomials with weights $W$ on $3K$ frequencies, and $\mathrm{Det}\ge\frac12(-\min_{\rm line}F_{\delta_0}-\min_{\rm line}F_{-\delta_0})$.
With $\Pi_\Sigma$ the payment is $\sum_l\kappa_l(\delta_0)c_{12}\|\beta_l\|_1=c_{12}\sum_{l,T}W_{l,T}s_l(\delta_0)\le c_{12}\rho\sum W$, the same constant as in the relations case.
Note that $m=1$ ($s_1=s_2$) or $m=-2$ ($s_{12}=-s_1$) makes slot-sets coincide, and the adversary can cancel them exactly; numerically $R_{\rm line}(m=1)=0$ at $p=31,53,101$. Degenerate lines must be avoided.

**The twist (why lines are not two-sided).** Write $F_\delta(x)=\sum_TD_T(2s_T(x))$ with $D_T(w)=\varphi_T(w)-\varphi_T(w-\delta)$. Then $-F_{-\delta}(x)=\sum_TD_T(2s_T(x)+\delta)$. No translation $x\mapsto x'$ achieves $2s_T(x')=2s_T(x)+\delta$ for all three $T$, because $s_{12}=s_1+s_2$; any translation matches exactly two of them.
Hence $\mathrm{Det}\ge\frac12(\max\tilde P-\min P)$ with $P=F_{\delta_0}$ and $\tilde P=P$ with the $T^*$-components shifted by $\delta_0$ in $w_{T^*}$, for any one $T^*$. Since $|\tilde P-P|\le\sum_l2W_{l,T^*}s_l(\delta_0)$,
$$\mathrm{Det}\ \ge\ \tfrac12\operatorname{osc}_{\rm line}P\;-\;\min_{T^*}\textstyle\sum_lW_{l,T^*}s_l(\delta_0)\ \ge\ \sigma_{\rm line}\sum W-\min_{T^*}\sum_lW_{l,T^*}s_l(\delta_0)\qquad\text{(two-sided CL)}.$$
So the two-sided CL on a line closes the plane if $\sigma_{\rm line}\ge(c_{12}+\theta_{\min})\rho$, where $\theta_{\min}\le\frac13$ is the smallest slot-set share of $\sum Ws$. In the even split this is $\sigma_{\rm line}\ge0.517\rho$; with one slot-set empty it is $c_{12}\rho$.

**Lemma 2.5 (slot projections are exactly two-sided; proved).** In class $\mathcal C$, $\mathbb E[F_{\pm\delta}\mid s_T(x)=z]$ is $D_T(2z)$, resp. $-D_T(2z+\delta)$ (Note, proof of Lemma potential). Hence
$$\mathrm{Det}=\tfrac12(-\min F_{\delta_0}-\min F_{-\delta_0})\ \ge\ \max_T\tfrac12\operatorname{osc}D_T\ \ge\ \sigma\max_T\textstyle\sum_lW_{l,T}\qquad\text{(two-sided CL per }T).$$
Each $D_T$ is a relations-type polynomial on the $K$ frequencies $2S$, so its CL hypotheses are those of the relations case. The plane closes if $\sigma\,\max_T\sum_lW_{l,T}\ge c_{12}\sum_{l,T}W_{l,T}s_l-(\text{slack})$. A sufficient condition is $\sigma\ge3c_{12}\rho=0.55\rho$ in the even split, and $c_{12}\rho$ when one $T$ carries everything.
In the large-deviation heuristic the per-$T$ constant is $\approx\sqrt3$ times the line constant ($K$ instead of $3K$ frequencies). So the projection is the better of the two two-sided routes, and it needs no choice of $m$.

**Lemma 2.4 (choice of $m$ for coverage; proved).** Let $P_T(\delta)=\sum_lW_{l,T}\kappa_l(\delta)$ and $N_T(t)=\#\{\delta\ne0:P_T(\delta)<t\sum W\}$. Then
$$\#\{m:\ t(\text{line }m)<t\}\ \le\ N_{\{1\}}(t)\cdot\min(N_{\{2\}}(t),N_{\{1,2\}}(t)),$$
and by Lemma 2.1 any slot-set may play the undilated role.
*Proof.* Coverage $<t$ at $\delta'$ forces each of $P_1(\delta')$, $P_2(m\delta')$, $P_{12}((1+m)\delta')$ to be $<t\sum W$, and $m\mapsto m\delta'$ is a bijection. $\square$
So if two slot-sets have blind sets with $N_TN_{T'}<p-2$, some line has coverage $\ge t$. Averaging gives $\mathbb E_m(\text{coverage at }\delta')=P_1(\delta')+\sum W_2+\sum W_{12}+O(1/p)$. In addition, $K_{\rm eff}(W)\le3\max_TK_{\rm eff}(W_T)$.

**Theorem 2 (conditional).** Assume the two-sided CL, and that $\Pi_\Sigma$ is admissible (otherwise replace $c_{12}$ by $0.2887$, which is proved). Class $\mathcal C$ at $n=2$ in the plane then reduces to the relations-case CL (Lemma 2.5), with the payment constant of the relations case. The price is a factor of at most $3$ in the requirement on $\sigma$, reached for an even split over the slot-sets.
A second route uses a generic line, which carries all three slot-sets on $3K$ frequencies and whose coverage is controlled by Lemma 2.4. It needs $\sigma_{\rm line}\ge(c_{12}+\theta_{\min})\rho$ because of the twist.
*Answer to "the right $m$ and the right $\Pi$".* The right $\Pi$ is the $\ell^1$ payment, because both routes detect in $\ell^1$ of the weights. The right "$m$" is no line at all but the three slot projections. A line is useful only when one slot-set is nearly empty, since then the twist defect $\theta_{\min}$ vanishes. It must then be coincidence-free (Lemma 2.3) and should be chosen by Lemma 2.4. The averaging-over-$m$ heuristic gives coverage $P_1+\sum W_2+\sum W_{12}$.

*Numerical* (`pl_*.out`, `pl2_101.out`, `lines_only.out`). $R=\inf\mathrm{Det}/\sum_{l,T}W_{l,T}\kappa_l(\delta_0)$ is in $\ell^1$ units, so closing needs $R\ge c_{12}=0.1835$, or $\ge0.2887$ with the proved constant. Values from adversarial search are upper bounds for the infima.

| $p$ | $K$ | $R_{\rm rel}$ (one $T$) | $R_{\rm plane}$ | projection $=R_{\rm rel}/3$ (even split; two-sided) | $\max_mR_{\rm line}$, coincidence-free $m$ | lines with coincidences | degenerate $m=1$ |
|---|---|---|---|---|---|---|---|
| 31 | 4 | 0.891 | 0.564 | 0.297 | none exist | 0.233–0.275 | 0.000 |
| 53 | 5 | 0.757 | 0.473 | 0.252 | none exist | 0.089–0.266 | 0.000 |
| 101 | 7 | 0.873–0.877 | 0.557 | 0.291 | 0.295–0.311 (all 6) | – | 0.000 |
| 151 | 6 | 1.056 | – | 0.352 | 0.403 | – | – |
| 211 | 8 | 0.899 | – | 0.300 | 0.329 | – | – |
| 307 | 9 | 0.880 | – | 0.293 | 0.335 | – | – |

Files: `pl_31.out`, `pl_53.out`, `pl2_101.out`, `lines_only.out`. All values are at one binding $\delta_0$; the plane values come from $3$–$4$ restarts on the full $p^2$ grid.
The projection value is exact given $R_{\rm rel}$: the infimum is attained by the even split with relation-worst phases in each $T$.

*Reading.*
* Restricting to the projection or to a line loses a factor $\approx1.8$–$1.9$ against the plane ($0.29$/$0.31$ vs $0.557$ at $p=101$).
* Both stay above $c_{12}=0.1835$ by a factor $1.4$–$2.2$, so the $\ell^1$ payment closes them. Against the proved constant $0.2887$ they are marginal ($0.25$–$0.40$).
* Lines with coincidences can lose much more ($0.089$ at $p=53$, $m=26$, where $T$-components cancel), and the degenerate line $m=1$ loses everything.
* Lines are about $1.1\times$ better than projections numerically, but only projections are covered by the two-sided CL without a twist defect.

---------------------------------------------------------------------------------------------------------------------------

## 3. Task 3: compensation (outside class $\mathcal C$, $n=2$, any network)

### 3.1 Exact tools (proved)

**Lemma 3.1 (slot projection).** For every network, $T\neq\emptyset$ and $\delta$,
$G_{\delta,T}(z):=\mathbb E[g(\cdot,\delta)\mid s_T(x)=z]-\mathbb Eg=\sum_{k\ne0}\hat g_\delta(k\mathbf 1_T)\rho^{kz}$ and $\mathbb Eg-\min_xg\ge-\min_zG_{\delta,T}$.
This is a conditional expectation, so it is positive and its minimum is $\ge$ the minimum of $g$. A character $\chi$ survives iff $\chi\in\mathbb Z_p\mathbf 1_T$.
The same holds for a line restriction $x=x^*+(u,mu)$, under which characters fold as $\chi\mapsto\chi_1+m\chi_2$.

**Lemma 3.1' (two-sidedness and the twist defect; proved).** A mode of $\Phi_j$ at $k\mathbf 1_T$ contributes $\frac12\phi c_j(\delta)$ to $\hat g_\delta(k\mathbf 1_T)$; a conjugate-type mode contributes $\frac12\overline{\phi c_j(\delta)}$.
Under $\delta\mapsto-\delta$ these acquire the factor $-\rho^{\pm j\delta}$, because $c_j(-\delta)=-\rho^{j\delta}c_j(\delta)$. So $-G_{-\delta,T}(z-\delta/2)$ equals $G_{\delta,T}(z)$ with every mode multiplied by $\rho^{\theta_m\delta}$, where $\theta_m=\pm j_m-k_m/2$.
For feature $l$'s own sign-flipped mode $\theta=0$. Hence in class $\mathcal C$, $\mathrm{Det}_\mu=\frac12\operatorname{osc}$ exactly (this is Lemma osc). In general
$$\mathrm{Det}_\mu\ \ge\ \tfrac12\operatorname{osc}_zG_{\delta_0,T}\;-\;\sum_m|Z_m|\,|\sin(\pi\theta_m\delta_0/p)|,$$
with $|Z_m|$ the real amplitude of mode $m$ and $\mu=\frac12(1_{\delta_0}+1_{-\delta_0})$.
For the own partner ($j=l$, conjugate type, $k=2l$), $\theta=-2l$ and the defect is $|Z||\sin(2\pi l\delta_0/p)|$. This is the polynomial-level form of the both-edge factor of Note, Thm q1n2.
*Proof.* $-\min G_{-\delta_0}=\max G'$ with $G'$ the rotated polynomial, and $\frac12(\max G'-\min G)\ge\frac12\operatorname{osc}G-\frac12\sup|G'-G|$, where $|\rho^{\theta\delta}-1|=2|\sin(\pi\theta\delta/p)|$. $\square$
So under the two-sided CL, compensation acts **twice**: it hides ($|O+Z|<|O|$), and it destroys the reflection between the two edges. Both effects are linear in the mode size.

**Lemma 3.2 (what lands on an own frequency).** By Note, Lemma channels, a mode of $\Phi_j$ at $\pm k\mathbf 1_T$ needs input content $q=\pm k-j$ on $T$. The coefficient of $G_{\delta,T}$ at $k=2l$ is
$\tfrac12[\beta_{l,T}c_l(\delta)+Z^{\rm free}_{l,T}(\delta)+Z^{\rm paid}_{l,T}(\delta)]$, where:
* $Z^{\rm free}$ comes from neurons in which both $j$ and $q$ are clean output frequencies (shared neurons, which cost nothing at $b=2$). It requires **(F2)**: $2l\equiv\pm j\pm q$ with $j\ne q$ outputs of one neuron. This includes the own partner when $3l$ is an output of $l$'s own neuron.
* $Z^{\rm paid}$ is everything else, including the own partner when $3l$ is *not* an output of that neuron. By Note, Lemma lorentz and Aczél, the payers' clean-amplitude loss weighted by $K_j(\mu)$ is $\ge\sum_jK_j(\mu)\sum_{m\to j}|M_m|^2/(2a_j)$, and it is collected in $U$.
* Modes at other $k$ are **foreign**. Under CL they only *add* to $\sum$weights: CL gives $-\min G\ge\sigma\sum_k|\text{coef}_k|\ge\sigma\sum_l|\text{own}+Z|$. They are harmless as long as the *total* polynomial meets the CL hypotheses.

Folding remark: on a line, off-line paid modes can also reach a target. Placing a hiding mode at the own character costs the same as spreading it over the $p$ characters that fold onto the target for each $m$; the minimum-norm solutions coincide. So lines do not change the paid game.

### 3.2 Theorem 3 (cleanest reduction) and why it cannot be shortened

**Theorem 3 (conditional).** Let $n=2$, $b=2$, and $\theta$ any network. Fix $\delta_0\in\Delta$ and $\mu=\frac12(1_{\delta_0}+1_{-\delta_0})$. Assume the two-sided (CL) for the projected polynomials $G_{\delta_0,T}$, *including* their foreign modes. By Lemma 3.1' and CL, $\mathrm{Det}_\mu\ge\sigma\sum_k|\mathrm{coef}_k|-\sum_m|Z_m||\sin(\pi\theta_m\delta_0/p)|$. Then the strict master inequality (MI) at $\mu$ (Note, Prop. master) follows from the two statements below. This is (G) with the bonus $\Re\beta_l-a_l$ itself in place of $\Pi$, so that the quadratic payment can be used.

**(NoFree / RUP).** Either no cheap $l$ is in an (F2) relation, or the free channels satisfy
$$\sigma\cdot H_{\rm common}\ \ge\ c_{12},\qquad H_{\rm common}=\max_{\mu}\min_{l}\ \min_{z}\frac{\sum_\delta\mu_\delta|c_l(\delta)+\sum_{j\in M_l^+}z_jc_j(\delta)+\sum_{j\in M_l^-}z'_j\overline{c_j(\delta)}|}{K_l(\mu)},$$
with $M_l^\pm=\{j\in S:2l\mp j\in\pm(S\setminus\{j\})\}$ (a robust uncertainty principle for free channels).

**(QG) the paid quadratic game, one feature at a time.** For every cheap $l$, every $\beta_{l,T}$ and all paid modes $M$ landing on $\chi_{l,T}$:
$$\mathrm{bonus}_l(\beta)\,K_l(\mu)\le\sigma\big|\beta_{l,T}c_l(\delta_0)+\textstyle\sum_mM_mc_{j_m}(\delta_0)\big|-\sum_m|M_mc_{j_m}(\delta_0)||\sin(\pi\theta_m\delta_0/p)|+\sum_mK_{j_m}(\mu)\frac{|M_m|^2}{2a_{j_m}},$$
Here $\mathrm{bonus}_l=\Re\beta_l-a_l\le\min(c_{12}\|\beta\|_1,\ K_2\|\beta\|_2^2/L_l)$: the first term is Lemma 2.2 (numerical; proved with $c_A\|\beta\|_2$), the second is Note, quadpay(i) (proved, $K_2=1$).
This is the projection version, with one $T$ per feature; a feature with interference on several $T$ is charged in $\ell^1$ and detected through $\max_T$ (Lemma 2.5).

*Proof.* Lemmas 3.1, 3.1' and CL bound Det from below by $\sigma\sum$ of the own coefficients after compensation, minus the twist defect (Lemma 3.2). Aczél makes the paid costs separable per mode, and they sit in $U$. The payment of a cheap feature is additive over $T$ for $\Pi_\Sigma$. Adding up the per-feature inequalities gives (MI). $\square$

**Why it cannot be reduced to (CL) plus a statement about coincidences only.**
1. *Paid compensation is a constants question, not an arithmetic one.* With stray outputs, exact first-order hiding of $\chi_{l,\{1,2\}}$ from every binding direction is possible at every sparse prime (Note, Prop. dich; $|M_{[2]}|\ge|B|$ computed for $p=11..41$).
   After first-order hiding, the CL detection of the own coefficient is $O(|\beta|^2)$, which is the cost term. The bonus near the IF block is also $O(|\beta|^2)$ (Note, quadpay; compensation note, Prop. local).
   So (QG) at small interference compares two quadratic constants: the protection constant $\kappa_2(l,T)$ (cheapest exact first-order hiding) and the bonus constant ($\frac{1-r}{8+r}$ near the IF block). The first depends on $(p,\text{allocation},\text{binding set})$ through $c_j(\delta)$ and $K_j(\mu)$. No coincidence condition and no value of $\sigma$ implies the comparison. (Measured room: $\ge21\times$ near IF and $\ge1.05$ / $\ge1.69$ globally for one cheap feature at $p\le41$; `mechanism_scripts/compensation/`.)
2. *Free channels are a robust uncertainty principle.* (F2) relations are common at small $p$ (`f2count.out`: in an F2 triple for most features at $p\le101$; none on the computed supports for $p\ge2003$, where $K^3/p<1$). By Chebotarev the free slope is positive whenever $|M_l|<|B|$. A $p$-uniform lower bound is not a coincidence statement either.

*Numerical (RUP; a twist-free proxy)* (`freehide_common.out`, all 31 programme vertices with $11\le p\le1009$, every feature treated as cheap at once, one common $\mu$, free channel coefficients unbounded).
This measures the $\pm\delta$-averaged per-character detection, which is what Riesz tests see. Under the two-sided CL the twist defect of the free modes also has to be subtracted, so these values are upper bounds for the relevant slope; it is a necessary condition, not a sufficient one.
$H_{\rm common}\ge0.719$ ($p=17$), and $\ge1.08$ for every other prime. Without channels it is $1.35$–$1.60$. So free compensation alone would need $\sigma<c_{12}/H$, i.e. $\sigma<0.255$ at $p=17$ and $\sigma<0.17$ elsewhere, to defeat CL. CL's heuristic $\sigma\approx0.5$ is far above this.
Per-feature values (`freehide_rup.out`) have minimum $0.72$, also at $p=17$, $l=8$.
*Numerical (lines)* (`f2count.out`): coincidence-free line directions exist for $p=61,67,79,89,97,101$ and for every computed $p\ge151$; there are none for $p\le59$ and for $p=71,73,83$. The bad $m$ are $2.7\%$ of $\mathbb Z_p$ at $p=100003$ ($K=22$).

---------------------------------------------------------------------------------------------------------------------------

## 4. Status (two-sided CL throughout)

| task | proved | reduced to | open / numerical |
|---|---|---|---|
| 1 blind spots | detection forms, both two-sided (1.1); payment–coverage, blind at $\delta_0$ ⇒ irrelevant (1.2); Dirichlet (1.3); dilated-interval structure and circle reduction (1.4–1.5); blind-set efficiency bound (1.6); thin blocks with participation (1.7); tied-minimum lemma, i.e. lopsided help (1.7b); $K_{\rm eff}(u)\le(2f^s/(\pi e\Gamma))^{1/(s-1)}=O(\ln p)$ (1.8) | CL with $\sigma\ge c_{12}\rho$ closes everything except non-thin blocks that are visible at $\delta_0$, blind elsewhere, and strong at every tied offset of the rest, with $\ge15$ frequencies and $e>0.759$ | Min-coverage CL is the wrong hypothesis for sub-blocks; accidental blindness needs an averaged-coverage (RM/tilt) CL. Structural blocks are closed numerically: ratio $\ge3.96$ at $p=1009$, $10007$, and $\ge4.5$ among $e>0.759$. H$_\Delta$ is false at $p=1009$. Nothing escapes for $p\le59$ (proved), nor for any computed allocation up to $10^5$ |
| 2 plane | slot symmetry ⇒ P$_T$ with $c_{12}$ for every single $T$ (closes the single-slot constant); slot projections are exactly two-sided (2.5); line restriction plus twist defect; coincidence count; coverage counting lemma | the relations-case CL per $T$ with the $\ell^1$ payment; requirement $\le3\times$ (even split) | $\ell^1$ payment $c_{12}\|\beta\|_1$: numerically sharp at $s=1.1,1.25,1.5$; proved on the axes, away from the axes, and for $\|\beta\|_\infty\le0.18L$; fallback $0.2887$. Measured: projection and line both lose $\approx1.9\times$ vs the plane and stay $\ge1.4\times c_{12}$ |
| 3 compensation | projection; twist defect (3.1', the polynomial form of both-edge averaging); free/paid classification; foreign modes harmless under CL | CL + (NoFree or RUP for F2 channels) + (QG) per feature, with hiding and twist both linear in the mode | (QG) compares quadratic constants (forced by Prop. dich) and cannot follow from CL plus coincidences. RUP proxy measured $H\ge0.72$ |
