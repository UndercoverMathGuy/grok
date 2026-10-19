# Stability: do efficient allocations avoid short relations? (follow-up to result.md)

Directory: `/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/unifiedB/`.

**Notation.** Allocation $a\ge0$ on $F$, with $r=2/3$ and $s=3/2$.
* $\|a\|:=(\sum a_l^r)^{1/r}$, $A=\sum a_l$.
* $\operatorname{cov}_a(\delta)=\sum a_l\kappa_l(\delta)$, and $\Gamma_s=\max_a\min_\delta\operatorname{cov}_a/\|a\|$.
* Efficiency $e(a)=\min_\delta\operatorname{cov}_a(\delta)/(\Gamma_s\|a\|)$.
* $K_r(a)=(\sum a^r)^{3}/(\sum a)^2$, the $r$-participation number.
* Interference $v\le\tau a$ with $\tau=\sqrt3$; payment constant $c=c_{12}=1-\sqrt{2/3}=0.1835$.

The relations-case inequality used (Note, proof of Prop. effsidon) is
$$h\le\operatorname{cov}_a(\delta)+c\,(Kv)_\delta-\tfrac12\operatorname{osc}_wF_\delta,\qquad F_\delta=\sum_l2v_ls_l(\delta)\cos(2\pi lw/p+\theta_l).$$
$\Gamma_s$ is taken to be the best programme value on file (`relations3/*.json`).

## 0. Summary

**Proved.**
1. *Lemma′ with $K_r(a)\le\kappa\ln p$, in the uniform form "$\tfrac12\operatorname{osc}\ge\sigma\sum u$ for every $u\le\tau a$", is false for every $e_0<1$.* (Prop. 1)
   - Adjoining to an optimal $a^\*$ a dilated Rudin–Shapiro progression $B$ of $N$ frequencies with total $r$-budget $\beta$ gives $e(a)\ge(1+\beta)^{-3/2}$ and $K_r(a)\le(1+\beta)^3K_r(a^\*)$.
   - $B$ contains $\ge N^2/4$ three-term relations, and $u=b$ has $\tfrac12\operatorname{osc}/\sum u\le\sqrt{2/N}$.
   - So efficient allocations "full of short relations" exist, and $K_r$ does not see them.
2. *Dilution lemma (Prop. 2): thin blocks never pay.*
   - Statement: if $(a',v')$ is closed at some $\delta$, then so is $(a'+b,\,v'+v_B)$ for any block with $\sum_Bb_l\varphi_l(\delta)\le\frac32\Gamma_s\beta$, where $\varphi_l=(1+c\tau)\kappa_l+2\tau s_l\le C_\tau=6.10$.
   - In particular this holds whenever $\beta\le(1.5\Gamma_s/C_\tau)^2|B|$, at any $\delta$.
   - At $\delta$ in the block's blind set the block term vanishes, and only $\beta>0$ is needed.
   - Together with Prop. effsidon, every single Rudin–Shapiro thin block is closed up to $p\approx10^8$ (`gap.py`; this uses $\sigma_{\delta_0}(\operatorname{spt}a)\approx\sigma_2(B)$). A gap opens near $p\sim10^{12}$, for $N\approx76$–$312$ and $\beta\approx0.06$–$0.08$; there the blind-spot step is needed.
3. *What minimum coverage gives about relations, exactly:*
   - (a) Co-blindness: $\sqrt{\kappa_{l_j}(\delta)}\le\sum_{l\ne j}|k_l|\sqrt{\kappa_l(\delta)}$ along any relation with $k_j=\pm1$.
   - (b) Third moment: $\min_{\delta\ne0}\operatorname{cov}\le A-\frac{pM_3-A^3}{pS_2/2-A^2}$, where $M_3=\frac32W_{\rm Schur}+\frac34W_{\rm dbl}$ and $S_2=\sum a_l^2$. For equal weights this gives $\#\{\text{Schur triples}\}\le(1-t)K^2/3+O(K^3/p)$.
   - (c) Laplace: the relative density of length-$d$ relations is $\lesssim\sqrt{2\pi d}(1-t)^d$.
   - All three are true but weak, and by Prop. C of result.md nothing much stronger follows from minimum coverage alone. Strong stability needs near-optimality, i.e. an exchange argument.

**Numerical.**
* Forced relations cost efficiency but are not forbidden (constrained local search, weights re-optimised; `forced_ls.py`). At $K=11$–$16$, $e\approx0.98$–$1.0$ with none, $0.96$–$0.99$ with one, $0.93$–$0.95$ with two, $0.88$–$0.94$ with three, and $0.91$ with four three-term relations in $\operatorname{spt}a$: about 2.5–3.5% per relation.
  * So $e_0=0.76$ tolerates several three-term relations. The strong stability ($\varepsilon(0.76,3)\ll1$ in count or in absolute $a$-weighted mass) is **false**.
* Near-optimal weights avoid relations.
  * When a relation-creating frequency is added to an optimal support, the re-optimised programme gives it non-zero weight in 11–14% of cases, against 25–40% for random frequencies; its mean weight is 0.02–0.035 against 0.05–0.105 (×mean weight).
  * At $p=10007$ the optimiser often zeroes a member of a forced relation at no loss in $e$.
* **Relations do not make the polynomial flat at these densities.** Allocations with 3–4 forced relations still have certified $\sigma_2\ge0.32$–$0.36$ for $u=a$ (Theorem A, $D=5$). Their relation sub-blocks have certified $\sigma_2\ge0.37$ and adversarial $\ge0.57$.
* Flatness needs dense structure (progressions of $\gtrsim50$ terms). Such structure costs far more coverage ($t$ falls from 0.55 to 0.20–0.50, result.md §3), or it lives in a thin block (Prop. 1), which Prop. 2 plus effsidon closes.

**Conjectural (the right stability statement, Stab′).** For $e(a)\ge e_0$ and $K_r(a)\le\kappa\ln p$, every sub-block $B\subseteq\operatorname{spt}a$ with $\sigma_2(B)<c$ either
* (i) is thin, $\beta_B\le(1.5\Gamma_s/C_\tau)^2|B|$ (closed by Prop. 2); or
* (ii) costs efficiency $(1/e-1)/\tau\ge c-\sigma_2(B)$ (closed by effsidon); or
* (iii) has a blind set on which $a-b_B$ cannot over-cover (proof-plan step 2).

The "few short relations" form is false, and is not needed: $O((1-e)K)$ three-term relations leak only $O(\mathrm{poly}(D)/K^2)$ into Theorem A at fixed $D$.

## 1. Lemma′ with $K_r$, and why the uniform form fails

> **Lemma′$_r$ (uniform form).** Suppose $K_r(a)\le\kappa\ln p$ and $e(a)\ge e_0$. Then for every $u\le\tau a$ and all phases, $\tfrac12\operatorname{osc}_w\sum u_l\cos(2\pi lw/p+\theta_l)\ge\sigma\sum u_l$.

**Proposition 1 (thin flat block).** Let $a^\*$ be a programme maximiser, normalised so that $\sum a^{\*r}=1$. Let $B=\{mj\bmod p:1\le j\le N\}$ (folded into $F$, disjoint from $\operatorname{spt}a^\*$), and $b_l=(\beta/N)^{3/2}$ on $B$. Then for $a=a^\*+b$:
* $e(a)\ge(1+\beta)^{-3/2}$, because $\min\operatorname{cov}_a\ge\Gamma_s$ and $\|a\|=(1+\beta)^{3/2}$;
* $K_r(a)=(1+\beta)^3/(A^\*+\beta^{3/2}N^{-1/2})^2\le(1+\beta)^3K_r(a^\*)$;
* $u=b$ with Rudin–Shapiro signs has $\tfrac12\operatorname{osc}\le\|Q\|_\infty\le\sqrt{2/N}\sum u$;
* $B$ inherits the $\lfloor N^2/4\rfloor$ Schur triples of $\{1,\dots,N\}$.

Hence the uniform form fails at every $\kappa\ge(1+\beta)^3K_r(a^\*)/\ln p$ and every $e_0<1$; $N$ is arbitrary.

Numbers (`thinblock.py`, exact):

| $p$ | $N$ | $\beta$ | $e(a)$ | $K_r/\ln p$ ($a^\*$ alone) | three-term relations | $\tfrac12\operatorname{osc}/\sum u$ on $B$ |
|---|---|---|---|---|---|---|
| 10007 | 64 | 0.01 / 0.05 / 0.20 | 0.986 / 0.939 / 0.820 | 1.77 / 1.97 / 2.73 (1.72) | $\ge1024$ | **0.170** |
| 10007 | 256 | 0.01 / 0.05 / 0.20 | 0.986 / 0.934 / 0.790 | 1.77 / 1.98 / 2.85 | $\ge16384$ | **0.088** |
| 100003 | 64 | 0.01 / 0.05 / 0.20 | 0.985 / 0.930 / 0.763 | 1.95 / 2.17 / 2.96 (1.90) | $\ge1024$ | **0.170** |

So the statement cannot be "for every $u\le\tau a$". In the application the interference bonus is weighted by $u\le\tau a$ and has to be paid for in budget. The correct target is the weighted inequality, and thin blocks are handled by budget accounting.

**Proposition 2 (dilution: thin blocks never pay).** Write $a=a'+b$ with disjoint supports, $B=\operatorname{spt}b$, $\beta=\sum_Bb^r$, normalised $\sum a'^r=1$, and $v=v'+v_B$ with $v'\le\tau a'$, $v_B\le\tau b$, phases arbitrary. Put
$$H(a,v;\delta)=\operatorname{cov}_a(\delta)+c(Kv)_\delta-\tfrac12\operatorname{osc}F_\delta .$$
If $H(a',v';\delta)\le\Gamma_s\|a'\|$ and $\sum_{l\in B}b_l\varphi_l(\delta)\le\frac32\Gamma_s\beta$ with $\varphi_l=(1+c\tau)\kappa_l+2\tau s_l$, then $H(a,v;\delta)\le\Gamma_s\|a\|$.
Since $\varphi_l\le C_\tau:=2(1+c\tau)+2\tau=6.10$ and $\sum_Bb\le\beta^{3/2}|B|^{-1/2}$ (Hölder), the hypothesis on $B$ holds at every $\delta$ once $\beta\le(1.5\Gamma_s/C_\tau)^2|B|$; that is $\approx1.1\cdot10^{-3}|B|$ at $p=10^4$.

*Proof.*
* Osc of the full polynomial: $\tfrac12\operatorname{osc}F_\delta\ge\tfrac12\operatorname{osc}F'_\delta-\|F_{B,\delta}\|_\infty\ge\tfrac12\operatorname{osc}F'_\delta-\sum_B2v_ls_l$.
* The remaining block terms: $\operatorname{cov}_b(\delta)+c(Kv_B)_\delta\le\sum_B(b_l+c\tau b_l)\kappa_l$.
* So $H(a,v)\le H(a',v')+\sum_Bb_l\varphi_l\le\Gamma_s+\frac32\Gamma_s\beta\le\Gamma_s(1+\beta)^{3/2}=\Gamma_s\|a\|$. $\square$

**Two-regime closure of thin blocks** (`gap.py`). Model a block by Rudin–Shapiro flatness $\sigma_2(B)\approx1.38/\sqrt N$ and $e=(1+\beta)^{-3/2}$.
* Prop. effsidon closes it if $\beta\ge(1+\tau(c-\sigma_2(B)))^{2/3}-1$.
* Prop. 2 closes it if $\beta\le(1.5\Gamma_s/C_\tau)^2N$.
* Using $\Gamma_s\approx0.137\sqrt{\ln10007/\ln p}$, the two ranges overlap for every $N$ up to $p\approx10^8$.
* From $p\sim10^{12}$ a window opens ($N\approx76$–$312$, $\beta\approx0.06$–$0.08$); this is where the blind-spot argument is genuinely needed.
* Caveat: the model uses $\sigma_{\delta_0}(\operatorname{spt}a)\approx\sigma_2(B)$, which is optimistic, since mixing $B$ with $a^\*$ frequencies could be flatter.

## 2. What can be proved about relations from coverage

**Lemma (co-blindness, exact).** If $\sum_lk_ll\equiv0$ with $k_j=\pm1$, then for every $\delta$: $|e(l_j\delta/p)-1|\le\sum_{l\ne j}|k_l||e(l\delta/p)-1|$, i.e. $\sqrt{\kappa_{l_j}(\delta)}\le\sum_{l\ne j}|k_l|\sqrt{\kappa_l(\delta)}$.
*Proof:* $|\prod z_i-1|\le\sum|z_i-1|$ and $|z^k-1|\le|k||z-1|$ on the circle. $\square$
On the set where the other members of $T=\operatorname{supp}k$ are blind, $l_j$ is blind too. So $T$ covers like $|T|-1$ frequencies.

**Proposition (third moment, exact).** Let $C(\delta)=\sum a_l\cos(2\pi l\delta/p)$ and $M_3=\frac1p\sum_\delta C^3=\frac32W_{\rm Schur}+\frac34W_{\rm dbl}$, where $W_{\rm Schur}=\sum_{\{x,y,z\}:x\pm y\equiv\pm z}a_xa_ya_z$ and $W_{\rm dbl}=\sum_{2x\equiv\pm y}a_x^2a_y$. Then
$$\min_{\delta\ne0}\operatorname{cov}_a\ \le\ A-\frac{pM_3-A^3}{pS_2/2-A^2}.$$
*Proof:* $\sum_{\delta\ne0}C^3\le\max_{\delta\ne0}C\cdot\sum_{\delta\ne0}C^2$, with $\sum_{\delta\ne0}C^2=pS_2/2-A^2$ exactly (frequencies in $F$). $\square$

Hence $e(a)\ge e_0$ forces $\frac32W_{\rm Schur}+\frac34W_{\rm dbl}\le(A-e_0\Gamma_s\|a\|)S_2/2+A^3/p$. For equal weights this gives at most $\approx(1-t)K^2/3$ Schur triples: true, and checked to $10^{-17}$ (`moment3.py`), but weak.
Higher odd moments and the Laplace transform give the $\sqrt{2\pi d}(1-t)^d$ density bound of result.md §6.
By Prop. C of result.md (the coverage bound $B(\lambda)$ is never within $1+o(1)$ of $\prod I_0$ at relevant $\lambda$), minimum coverage alone cannot bound short-relation mass by $\varepsilon\ll1$. A strong statement must use near-optimality, i.e. compare with an improved allocation.

## 3. Numerics: the efficiency-versus-relations curve

**ADD** (`stability.py add`). Add one frequency to the optimal support and re-optimise (warm start; weights may vanish):

| $p$ | additions | non-zero weight | mean weight / mean programme weight |
|---|---|---|---|
| 1009 | relation-creating (132) / random (132) | 11% / 25% | 0.024 / 0.053 |
| 2003 | 156 / 156 | 13% / 35% | 0.032 / 0.089 |
| 10007 | 272 / 272 | 14% / 40% | 0.035 / 0.105 |

**FORCE** (`forced_ls.py`). Local search maximising coverage subject to $\ge k$ three-term relations in $S$; then weights are re-optimised. The table reports $e=\Gamma_{S}/\Gamma_{\rm best}$ against the true number of three-term relations inside $\operatorname{spt}a$ (`erel_table.py`); $k=0$ is the search-quality control.

| #3-term relations in spt $a$ | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $p=1009$, $K=11$ | 0.985 (3 runs) | 0.957 (9) | 0.928 (6) | 0.894 (2) | 0.906 (1) |
| $p=2003$, $K=12$ | 0.975 (4) | 0.968 (10) | 0.949 (5) | 0.878 (2) | – |
| $p=10007$, $K=16$ | 0.997 (4) | 0.986 (1) | 0.938 (1) | 0.943 (2) | 0.912 (2) |

This is about 2.5–3.5% of efficiency per forced three-term relation (programme supports have none). Four-term counts stay at 0–2.
Extrapolating linearly, $e_0=0.76$ tolerates roughly 7 three-term relations at these sizes.

The optimiser frequently drops one member of a forced relation; a forced relation then often survives in $S$ but not in $\operatorname{spt}a$ (e.g. $p=10007$: $e=1.007$ with the relation's member at weight 0).

**Theorem A on forced allocations** (`cert_forced.py`, $u=a$, $D=5$; "sub-block" = the frequencies involved in three-term relations):

| $p$ | 3-term relations | $e$ | cert $\sigma_1$ / cert $\sigma_2$ ($u=a$) | adversarial $\sigma_2$ | relation sub-block: cert $\sigma_2$ / adversarial / Parseval |
|---|---|---|---|---|---|
| 1009 | 0 (control, 3 runs) | 0.98–0.99 | 0.32–0.33 / 0.34 | 0.51–0.52 | – |
| 1009 | 2 | 0.931 / 0.934 | 0.33 / 0.35, 0.32 / 0.34 | 0.54 | 0.47–0.51 / 0.70–0.71 / 0.32–0.36 |
| 1009 | 3–4 | 0.867–0.922 | 0.28–0.35 / 0.34–0.36 | 0.52–0.56 | 0.37–0.44 / 0.57–0.68 / 0.24–0.30 |
| 2003 | 0 (control) | 0.95–1.01 | 0.33–0.34 / 0.33–0.34 | 0.54 | – |
| 2003 | 2 | 0.935–0.961 | 0.31–0.32 / 0.33 | 0.53–0.55 | 0.49–0.50 / 0.70–0.77 / 0.32–0.38 |
| 2003 | 3 | 0.869 / 0.888 | 0.28 / 0.32–0.33 | 0.52 | 0.37–0.38 / 0.60–0.62 / 0.24 |

Forcing relations lowers efficiency, not flatness. The certified two-sided value moves by $\le0.02$, and the relation sub-blocks are less flat than the whole.

## 4. Assessment

* The strong conjecture (few short relations whenever $e\ge e_0$) is false at $e_0=0.76$, by forced relations (about 3% per relation) and by thin blocks.
* It is also unnecessary. A handful of relations (or $O((1-e)K)$ of them) does not make anything flat; Theorem A certificates stay at 0.32–0.36 against the 0.18 needed. Asymptotically the leakage of $N_3$ three-term relations into a degree-$D$ test is $O(N_3\,\mathrm{poly}(D)/K^3)$.
* What must be excluded is dense structure: flat sub-blocks with $\sigma_2(B)<c$, which need $|B|\gtrsim50$.
  * Thin ones are closed rigorously by Prop. 2; with effsidon this covers every single Rudin–Shapiro block up to $p\approx10^8$.
  * Thick ones cost coverage heavily (result.md §3; the parallel finding that an embedded 32-block's help profile has min 0).
  * The remaining proof obligation is the blind-spot statement (iii): the rest of an efficient allocation cannot over-cover the blind set of a flat block by more than a factor $1+O(\ln N/\ln p)$.
* This is a covering bound for Bohr-type sets, a cleaner target than "few relations".
