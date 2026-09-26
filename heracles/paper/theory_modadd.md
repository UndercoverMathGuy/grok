# Theory: modular addition

Draft of the theoretical section, 2026-09-24 (rev 2: model-agnostic statement). Sources:
`THEORY_nsum.md` §1–§3, §8; `ELEGANT.md` §1–§3; `LEMMA_ATTEMPT.md` §2; `PROOF_AUDIT.md` F1, F2, F6;
`LADDER.md` §3–§4. Everything stated as a lemma, proposition or theorem is proved in full here.
Everything not proved is in §9 and labelled as such.

**How the section is organised.** The argument is about the *table of logits* a network produces,
not about the network. §1–§3 define the table, its clean part and its spectrum, and prove three
lemmas that hold for any table whatsoever. The architecture enters in exactly two places, stated as
assumptions in §4: a *budget* (how much clean amplitude one unit of cost buys) and a *sign rule*
(what a neuron's junk looks like). Every theorem is then stated for any model satisfying the
assumption it needs, and §8 checks the assumptions for the concrete models we train.

---

## 1. The objects

**Task.** $p$ is an odd prime and $n \ge 1$ is the number of operands. An *input* is an $n$-tuple
$a = (a_1, \dots, a_n)$ of residues mod $p$; its *answer* is the sum $s(a) = a_1 + \dots + a_n \bmod p$.
For $n = 2$ the input is a pair $(a, b)$ and the answer is $a + b \bmod p$. The network is trained on
the *full table*: all $p^n$ inputs, every one of them. (This matters: Lemma 1 averages over every input
with a given answer.)

**The logit table.** Whatever the model is, on input $a$ it outputs one score per candidate answer
$c \in \mathbb{Z}_p$. Write $z(a, c)$ for that score. So $z$ is a table with $p^n$ rows (one per input)
and $p$ columns (one per candidate answer). Everything below is a statement about this table.

**Margin.** The margin is the worst-case gap between the score of the right answer and the best
score of a wrong one:

$$\operatorname{margin}(z) \;=\; \min_{a}\ \min_{c \ne s(a)}\ \big[\,z(a, s(a)) - z(a, c)\,\big].$$

**Neuron-additive homogeneous models.** We consider models whose logit table is a sum of
per-neuron tables, each neuron paying its own cost:

$$z \;=\; \sum_{i=1}^{m} z_i, \qquad z_i = z_i(\theta_i), \qquad P_q(\theta) = \sum_{i=1}^m \|\theta_i\|^q ,$$

where $\theta_i$ is neuron $i$'s weight vector, $\|\cdot\|$ is the Euclidean norm, and each $z_i$ is
homogeneous of degree $\nu$ in $\theta_i$: scaling $\theta_i$ by $t$ scales $z_i$ by $t^\nu$. The
exponent $q \in [2, \nu]$ is the penalty exponent: $q = 2$ is plain weight decay,
$\lambda\sum_i\|\theta_i\|^2 = \lambda\|\theta\|^2$; $q = \nu$ is the $L_{2,\nu}$ penalty of Morwani et al.
Two numbers derived from $(q, \nu)$ run the whole section:

$$r := q/\nu \in (0, 1], \qquad s := \nu/q = 1/r \ge 1 .$$

*Reading: $r$ is how much cheaper a feature gets per unit of amplitude as it grows; $s$ is the
return on concentrating budget into few features. $r = 1$ (Morwani's penalty) means no return.*

**Normalised margin.** $\gamma_q(\theta) = \operatorname{margin}(z)\,/\,P_q(\theta)^{\nu/q}$. Scaling all
weights by $t$ multiplies the numerator by $t^\nu$ and the denominator by $(t^q)^{\nu/q} = t^\nu$, so
$\gamma_q$ depends only on the *direction* of the weights. By Wei et al. (2019, Thm 4.1) the global
minimiser of $\mathrm{CE} + \lambda P_q$ has normalised margin converging to $\sup_\theta\gamma_q$ as
$\lambda \to 0$. The section characterises that supremum and its maximisers, and says nothing about
whether gradient descent finds them.

**Concrete instances** (checked in §8): the *product model*
$z_i(a, c) = w_i(c)\prod_j e_{ij}(a_j)$, one lookup vector per operand plus a readout, degree
$\nu = n+1$; Morwani's *power model* $z_i(a, c) = w_i(c)\big(\sum_j e_{ij}(a_j)\big)^n$, same degree;
and the one-hidden-layer *ReLU model* $z_i(a,c) = w_i(c)\,\mathrm{ReLU}\big(\sum_j e_{ij}(a_j)\big)$,
degree $2$.

---

## 2. Clean part, spectrum, and the programme

**Clean part.** The score of the answer sitting $D$ steps below the right one, averaged over all
inputs, is

$$g(D) \;=\; \frac{1}{p^n}\sum_{a} z\big(a,\ s(a) - D\big), \qquad D \in \mathbb{Z}_p .$$

$g$ is the part of the table that depends only on the difference (answer $-$ candidate). Everything
in $z$ that is not of this form we call *junk*: it is content the table carries that is not a function
of whether $c$ is right.

**Spectrum.** Let $\omega = e^{2\pi i/p}$ and $F = \{1, \dots, (p-1)/2\}$ be the positive
frequencies ($k$ and $-k$ are the same cosine). Expand the clean part in cosines:

$$g(D) \;=\; \frac{1}{p^{\nu}}\Big[\,G_0 + 2\sum_{k \in F}\operatorname{Re}\big(G_k\,\omega^{kD}\big)\Big],
\qquad G_k := p^{\nu - 1}\sum_D g(D)\,\omega^{-kD}.$$

(The normalising power of $p$ is chosen so that $G_k$ is a polynomial in the weights with integer
coefficients in the concrete models; nothing depends on it.) $G_k$ is the network's *clean
amplitude at frequency $k$*. A *feature* is a frequency $k \in F$ with $G_k \ne 0$, and the *feature
count* of a table is the number of its features. Since $z = \sum_i z_i$ and $g$ is linear in $z$,

$$G_k = \sum_{i=1}^m c_{ik}, \qquad c_{ik} := \text{the clean amplitude of neuron } i\text{'s own table } z_i \text{ at } k,$$

and we call $c_{ik}$ the *atom* of neuron $i$ at frequency $k$. Atoms are defined for any
neuron-additive model.

**Coverage.** Frequency $k$ separates the right answer from the wrong answer $D$ steps away by

$$g_k(D) \;=\; 1 - \cos(2\pi kD/p) \;\ge 0, \qquad \sum_{D \ne 0} g_k(D) = p\quad (k \ne 0).$$

*Reading: a single cosine feature barely tells apart answers whose offset $D$ satisfies
$kD \approx 0 \bmod p$. Several frequencies are needed so that every offset is covered by someone.*

**The programme.** For $s \ge 1$,

$$\Gamma_s(p) \;=\; \max\Big\{\ \min_{D \ne 0}\ \sum_{k \in F} y_k\, g_k(D)\ :\ y \ge 0,\ \sum_{k \in F} y_k^{1/s} \le 1\ \Big\}.$$

*Reading: split one unit of budget over frequencies, where a budget share of $y_k^{1/s}$ buys clean
amplitude $y_k$; collect the worst-case coverage over offsets; maximise.* The constraint set is
compact and the objective continuous, so the maximum exists. Let

$$K^*(s, p) = \big|\operatorname{supp}(y^\star)\big| \quad\text{for a maximiser } y^\star,$$

the *feature count of the programme* (the set of such sizes if maximisers are not unique; it is a
single integer at every $(s, p)$ we report). The *equal-share ladder* restricts to equal shares on a
$K$-set: with $M_K(p) = \max_{|S| = K}\min_{D \ne 0}\sum_{k \in S} g_k(D)$,

$$\Gamma_s(p) \;\ge\; \max_K\ K^{-s} M_K(p),$$

by taking $y_k = K^{-s}$ on the best $K$-set. The ladder's argmax approximates $K^*$; §7 reports where
they differ.

---

## 3. The ansatz, derived lightly

This subsection is heuristic; every step is made rigorous below.

A feature of a degree-$\nu$ model is carried by a chain of $\nu$ weight vectors along the path from
input to logit. If those vectors have norms $\rho_1, \dots, \rho_\nu$, the feature's amplitude is
proportional to the product $\prod_l \rho_l$ and its penalty is $\sum_l \rho_l^q$. At fixed product the
sum is smallest when all norms are equal (AM–GM), so

$$\text{cost of amplitude } A \;\propto\; A^{q/\nu} = A^{1/s}.$$

Because $q \le \nu$ the exponent is at most $1$: *amplitude is cheaper per unit the more of it one
feature carries*. The margin, on the other hand, is worst-case coverage: with amplitudes $y_k$ the
clean gap at offset $D$ is $\sum_k y_k g_k(D)$ and the margin is its minimum over $D \ne 0$.
Maximising margin per unit cost is exactly the programme $\Gamma_s$, and with equal amplitudes on a
$K$-set it reduces to $M_K/K^s$: coverage $M_K$ grows with $K$, the concentration price $K^{-s}$
falls with $K$, and the feature count is where the product peaks. Two limits are visible already:

- **$q = \nu$ ($s = 1$).** The budget $\sum_k y_k \le 1$ is linear; coverage averaged over offsets is
  $\frac{p}{p-1}\sum_k y_k$ for *every* $y$, so the maximiser is whatever makes coverage flat in $D$:
  every frequency, equal shares. This is Morwani et al.'s dense solution (Corollary 5.2).
- **$q < \nu$ ($s > 1$).** Concentration pays. Since $M_K \le K$, $K^{-s}M_K \le K^{1-s} \to 0$: the
  dense solution loses and the argmax is a finite $K^*$ that falls as $s$ grows, that is, as the
  degree rises or the penalty exponent falls.

Two things the heuristic hides, both supplied below: (i) the margin bound must be shown to depend
*only* on the clean amplitudes $G_k$, although a trained table carries junk (Lemmas 1–3);
(ii) a single neuron does not produce a clean feature but a clean term plus sign-flipped copies of it
(junk), and cancelling those copies requires splitting the feature over several neurons, which at
$q < \nu$ costs a factor. Theorem 6.1 shows the minimal split is $\nu$ neurons and prices it exactly.

---

## 4. Three lemmas for any table, and one bound for any neuron-additive model

> **Lemma 1 (junk never helps).** For every table $z$ on the full table and every $D \ne 0$,
> $$\operatorname{margin}(z) \;\le\; g(0) - g(D).$$

*Proof.* For every input $a$, the candidate $s(a) - D$ is wrong, so
$z(a, s(a)) - z(a, s(a) - D) \ge \operatorname{margin}(z)$. Average over all $p^n$ inputs. $\square$

*Reading: the worst input is no better than the average input at any fixed offset, and only the clean
part survives the average. Junk can only lower the margin.* The proof uses every input, which is why
all statements are full-table statements.

> **Lemma 2 (phase lemma).** For every table $z$ and every $D \ne 0$,
> $$\operatorname{margin}(z) \;\le\; \frac{2}{p^{\nu}}\sum_{k \in F}|G_k|\,g_k(D),
> \qquad\text{hence}\qquad
> \operatorname{margin}(z) \;\le\; \frac{2}{p^{\nu}}\min_{D \ne 0}\sum_{k \in F}|G_k|\,g_k(D).$$

*Proof.* Apply Lemma 1 at $D$ and at $-D$ and average. From the spectrum formula in §2, the
$k = 0$ term cancels in $g(0) - g(\pm D)$, and
$\tfrac12\big[(g(0) - g(D)) + (g(0) - g(-D))\big] = \frac{2}{p^\nu}\sum_{k \in F}\operatorname{Re}\big[G_k\big(1 - \tfrac12(\omega^{kD} + \omega^{-kD})\big)\big] = \frac{2}{p^\nu}\sum_{k \in F}\operatorname{Re}(G_k)\,g_k(D)$,
and $\operatorname{Re}(G_k) \le |G_k|$ with $g_k(D) \ge 0$. $\square$

The averaging over $D$ and $-D$ is needed: without it the terms $\operatorname{Im}(G_k)\sin(2\pi kD/p)$
make the pointwise inequality false. Equality holds iff every $G_k$ is real and non-negative, i.e.
all features are phase-aligned at offset zero.

> **Proposition 4.1 (coverage bound).** With $\|G\|_r := \big(\sum_{k \in F}|G_k|^r\big)^{1/r}$ and $s = 1/r$,
> $$\operatorname{margin}(z) \;\le\; \frac{2}{p^{\nu}}\ \Gamma_s(p)\ \|G\|_r .$$

*Proof.* If $\|G\|_r = 0$ the margin is $\le 0$ by Lemma 2. Otherwise $y_k = |G_k|/\|G\|_r$ is feasible
for the programme ($y \ge 0$, $\sum_k y_k^{1/s} = \sum_k y_k^r = 1$), so
$\min_D\sum_k y_k g_k(D) \le \Gamma_s(p)$; multiply by $\|G\|_r$ and use Lemma 2. $\square$

So far nothing about the model has been used. The one thing a model must supply is how much clean
amplitude a neuron can buy:

> **Assumption (B) — budget.** There is a constant $C_q > 0$ such that for every neuron $i$,
> $$\sum_{k \in F}|c_{ik}|^{q/\nu} \;\le\; C_q\,\|\theta_i\|^q .$$
> *Reading: one neuron of weight norm $\|\theta_i\|$ can create only so much clean amplitude, priced
> with exponent $q/\nu$, however it spreads it over frequencies.*

> **Theorem 4.2 (universal margin bound).** For every neuron-additive $\nu$-homogeneous model
> satisfying (B), every width, and every $q \in [2, \nu]$,
> $$\gamma_q(\theta) \;\le\; \frac{2}{p^{\nu}}\,C_q^{\,\nu/q}\ \Gamma_{\nu/q}(p).$$

*Proof.* Let $r = q/\nu \le 1$. The function $t \mapsto t^r$ is subadditive on $[0, \infty)$ and
$|G_k| \le \sum_i|c_{ik}|$, so
$$\sum_{k \in F}|G_k|^r \;\le\; \sum_i\sum_{k \in F}|c_{ik}|^r \;\le\; C_q\sum_i\|\theta_i\|^q = C_q\,P_q(\theta)$$
by (B). Raising to the power $1/r = \nu/q$ gives $\|G\|_r \le C_q^{\nu/q}P_q^{\nu/q}$; insert into
Proposition 4.1 and divide by $P_q^{\nu/q}$. $\square$

No hypothesis beyond (B) is used: neurons may mix frequencies, carry junk, or be inactive. For the
product and power models $C_q = (p/2\nu)^{q/2}$ up to a model constant (§8), which makes the
prefactor $B := \frac{2}{p^\nu}C_q^{\nu/q} = 2(2p\nu)^{-\nu/2}$.

---

## 5. The dense endpoint

> **Corollary 5.1.** $\Gamma_1(p) = p/(p-1)$, and its *unique* maximiser is $y_k = 2/(p-1)$ for every
> $k \in F$: all frequencies, equal shares.

*Proof.* For any feasible $y$, the minimum over $D$ is at most the average over $D \ne 0$:
$\min_{D \ne 0}\sum_k y_k g_k(D) \le \frac{1}{p-1}\sum_{D \ne 0}\sum_k y_k g_k(D) = \frac{p}{p-1}\sum_k y_k \le \frac{p}{p-1}$,
using $\sum_{D \ne 0}g_k(D) = p$. Equality forces $\sum_k y_k = 1$ and
$h(D) := \sum_{k \in F}y_k\cos(2\pi kD/p)$ constant on $D \ne 0$. The function $h$ is even with
$\hat h(\pm k) = y_k/2$ for $k \in F$ and $\hat h(0) = 0$. A function equal to $\kappa$ off zero and $1$ at
zero has $\hat h(k) = (1 - \kappa)/p$ for $k \ne 0$ and $\hat h(0) = \kappa + (1-\kappa)/p$; so
$\kappa = -1/(p-1)$ and $y_k = 2(1-\kappa)/p = 2/(p-1)$. $\square$

> **Corollary 5.2 (dense at $q = \nu$).** For every model satisfying (B), at $q = \nu$ the bound of
> Theorem 4.2 is $\frac{2}{p^\nu}C_\nu\,\frac{p}{p-1}$; any table attaining it has $|G_k|$ equal for all
> $k \in F$ with all $G_k$ of one phase. For the product model the bound is attained (Proposition 7.1
> with $r = 1$), so the $L_{2,\nu}$ maximum-margin product network uses every frequency.

*Proof.* Equality in Proposition 4.1 forces $|G|/\|G\|_1$ to be the unique maximiser of $\Gamma_1$
(flat), and equality in the subadditivity step at $r = 1$ forces the atoms at each $k$ to share a
phase. Attainment is Proposition 7.1. $\square$

This recovers Morwani et al.'s dense theorem as the $s = 1$ endpoint of the programme. For $q < \nu$
Theorem 4.2 still holds but is *not* attained: a single neuron per frequency carries junk (§6), so its
atoms cannot be clean and phase-aligned at once. Theorem 7.2 gives the exact value on the class where
junk is absent.

---

## 6. The symmetric class and the feature-cost theorem

### 6.1 What junk looks like, and the sign rule

Fix a frequency $k$. A neuron that carries the clean mode of $k$ (the cosine of
$k(a_1 + \dots + a_n - c)$) will in general also carry *sign-flipped* copies: for a non-empty set
$T \subseteq \{1, \dots, n\}$ of operands, the cosine of
$k\big(\sum_{j \notin T}a_j - \sum_{j \in T}a_j - c\big)$. That mode depends on the inputs and the
candidate but not on whether the candidate is right; it is junk in the sense of Lemma 1, and it sits at
the same frequency as the feature. Write $J_{k,T}$ for the table's Fourier coefficient at the mode
$(k \text{ on } j \notin T,\ -k \text{ on } j \in T;\ -k \text{ on } c)$, normalised like $G_k$, and
$J_{ik,T}$ for neuron $i$'s. Then $J_{k,\emptyset} = G_k$ and $J_{k,T} = \sum_i J_{ik,T}$.

> **Assumption (S) — sign rule.** For every neuron $i$ and frequency $k$ there are unimodular
> numbers $v_{ijk}$ ($j = 1..n$) such that
> $$J_{ik,T} \;=\; c_{ik}\prod_{j \in T} v_{ijk} \qquad\text{for every } T \subseteq \{1, \dots, n\}.$$
> *Reading: a neuron's sign-flipped copy of its feature has the same size as the feature and a phase
> obtained by flipping the phases of the operands in $T$.*

(S) holds whenever the neuron's contribution to the $\pm k$ modes comes from a product of one
function per operand and one of the candidate; §8 verifies it for the product and power models, where
$v_{ijk} = \overline{\hat e_{ij}(k)}/\hat e_{ij}(k)$.

> **Definition (symmetric class).** A table is *symmetric* (junk-free at every feature) if
> $$(\mathrm{JF})\qquad J_{k,T} = 0\quad\text{for every } k \in F \text{ and every non-empty } T .$$

Equivalently: at every frequency, the table's content at the $2^\nu$ modes with entries $\pm k$ is a
function of (answer $-$ candidate) alone, i.e. it is invariant under the task's symmetry
$(a, c) \mapsto (a + t,\ c + \sum_j t_j)$. The class is a property of the *table*, not of the neurons:
a symmetric network may have neurons that mix frequencies. Junk at other modes (two different
frequencies mixed, or a zero entry) is not restricted by (JF); Lemma 1 already discounts it and (B)
already prices it.

### 6.2 The feature-cost theorem

Under (S) and (JF), the atoms at a fixed frequency satisfy a finite system that involves neither $p$
nor the task. Relabel the neurons with $c_{ik} \ne 0$ as $l = 1..N$, write $c_l = c_{ik}$,
$v_{lj} = v_{ijk}$, $A = G_k$:

$$(\mathrm{JF}_k)\qquad \sum_{l=1}^N c_l\prod_{j \in T}v_{lj} \;=\; A\,[T = \emptyset]
\qquad\text{for every } T \subseteq \{1, \dots, n\}.$$

*Reading: $N$ complex numbers $c_l$ with phases $v_{lj}$ attached, whose plain sum is $A$ and whose
sum twisted by any non-empty pattern of phase flips is $0$.*

> **Theorem 6.1 (feature cost).** Let $v_{lj} \in \mathbb{C}$ ($l = 1..N$, $j = 1..n$) have modulus $1$
> and let $c_1, \dots, c_N$ satisfy $(\mathrm{JF}_k)$ with total $A$. Then for every $r \in (0, 1]$,
> $$\sum_{l=1}^N|c_l|^r \;\ge\; \nu^{1-r}\,|A|^r, \qquad \nu = n + 1 .$$
> For $r \in (0, 1)$ equality holds if and only if exactly $\nu$ of the $c_l$ are non-zero, they all
> have modulus $|A|/\nu$, and their chain vectors (Lemma 6.2) are pairwise orthogonal.

*Reading: to carry clean amplitude $A$ while cancelling every sign-flipped copy, a feature must be
spread over at least $\nu$ neurons, evenly, and extra neurons never make it cheaper.* At $r = 1$ the
inequality is the triangle inequality and every phase-aligned configuration attains it; this is why
the number of neurons per feature is pinned only for $q < \nu$.

The proof is a change of coordinates followed by three classical facts. If $A = 0$ there is nothing to
prove; assume $A \ne 0$ and normalise $A = 1$ by dividing every $c_l$ by $A$.

> **Lemma 6.2 (chain lemma).** The *chain vector* of atom $l$ is
> $$m_l := \big(1,\ v_{l1},\ v_{l1}v_{l2},\ \dots,\ v_{l1}\cdots v_{ln}\big) \in \mathbb{C}^{\nu},$$
> the running products of its phases. Then $\|m_l\|^2 = \nu$, and $M := \sum_l c_l\,m_l m_l^*$ is upper
> triangular with every diagonal entry equal to $A$.

*Proof.* Every entry of $m_l$ has modulus $1$, so $\|m_l\|^2 = \nu$ and the diagonal of $m_lm_l^*$ is
$1$; the diagonal of $M$ is $\sum_l c_l = A$ (the $T = \emptyset$ equation). For $k > k'$, since
$\bar v = 1/v$ for unimodular $v$,
$(m_lm_l^*)_{k,k'} = \prod_{j \le k}v_{lj}\,\overline{\prod_{j \le k'}v_{lj}} = \prod_{j=k'+1}^{k}v_{lj}$,
the monomial of the non-empty *interval* pattern $T = \{k'+1, \dots, k\}$; summing against $c_l$
gives $0$ by $(\mathrm{JF}_k)$. $\square$

Only the $\binom{\nu}{2}$ interval patterns are used. The entries *above* the diagonal are not
controlled and are generically non-zero, so $M \ne AI$ in general; triangularity is all that is needed.

> **Lemma 6.3 (minor expansion).** For $u_1, \dots, u_N \in \mathbb{C}^\nu$, $d \in \mathbb{C}^N$ and
> $M' = \sum_l d_l\,u_lu_l^*$,
> $$\det(I + tM') \;=\; \sum_{L \subseteq \{1..N\}} t^{|L|}\Big(\prod_{l \in L}d_l\Big)\det\operatorname{Gram}(u_l : l \in L).$$

*Proof.* With $U = [u_1 \cdots u_N]$ and $\Delta = \operatorname{diag}(d)$, $M' = U\Delta U^*$, and
Sylvester's identity gives $\det(I_\nu + tU\Delta U^*) = \det(I_N + t\Delta U^*U)$. Expand the right
side by principal minors: $\det(I_N + tX) = \sum_L t^{|L|}\det X_{L,L}$, and
$\det\big((\Delta U^*U)_{L,L}\big) = \prod_{l \in L}d_l\cdot\det(U^*U)_{L,L}$, where $(U^*U)_{L,L}$ is
the Gram matrix of $(u_l)_{l \in L}$. $\square$

> **Lemma 6.4 (Hadamard).** If the $u_l$ are unit vectors, $0 \le \det\operatorname{Gram}(u_l : l \in L) \le 1$,
> with value $1$ iff $(u_l)_{l \in L}$ is orthonormal.

*Proof.* A Gram matrix is positive semidefinite, so its determinant is $\ge 0$; Hadamard's inequality
bounds it by the product of the diagonal entries, all $1$, with equality iff it is diagonal. $\square$

> **Proposition 6.5 (symmetric-function bounds).** Under $(\mathrm{JF}_k)$ with $A = 1$, put
> $d_l = \nu c_l$ and $x_l = |d_l|$. Then for every $k = 0, 1, \dots, \nu$,
> $$e_k(x) \;\ge\; \binom{\nu}{k},$$
> where $e_k$ is the $k$-th elementary symmetric polynomial. In particular $e_\nu(x) \ge 1$ forces
> at least $\nu$ non-zero atoms.

*Proof.* With $u_l = m_l/\sqrt\nu$ (unit vectors), $c_l\,m_lm_l^* = d_l\,u_lu_l^*$, so
$M = \sum_l d_l\,u_lu_l^*$. By Lemma 6.2, $M$ is upper triangular with diagonal $1$, so all its
eigenvalues are $1$ and $\det(I + tM) = (1 + t)^\nu$. Compare the coefficient of $t^k$ with Lemma 6.3
and take moduli:
$$\binom{\nu}{k} = \Big|\sum_{|L|=k}\Big(\prod_{l \in L}d_l\Big)G_L\Big| \le \sum_{|L|=k}\Big(\prod_{l \in L}x_l\Big)G_L \le \sum_{|L|=k}\prod_{l \in L}x_l = e_k(x),$$
using $0 \le G_L \le 1$ (Lemma 6.4). $\square$

> **Proposition 6.6 (generating inequality).** $\prod_{l=1}^N(1 + x_lt) \;\ge\; (1 + t)^\nu$ for every $t \ge 0$.

*Proof.* $\prod_l(1 + x_lt) = \sum_{k=0}^N e_k(x)t^k$ with every $e_k(x) \ge 0$, and
$e_k(x) \ge \binom{\nu}{k}$ for $k \le \nu$ by Proposition 6.5. So the sum is at least
$\sum_{k=0}^\nu\binom{\nu}{k}t^k = (1 + t)^\nu$. $\square$

> **Lemma 6.7 (Bernstein integral).** For $0 < r < 1$ and $x \ge 0$,
> $$x^r \;=\; \frac{r\sin(\pi r)}{\pi}\int_0^\infty\log(1 + xt)\,t^{-1-r}\,dt,$$
> the integral converging absolutely.

*Proof.* Near $t = 0$ the integrand is $O(t^{-r})$, near $\infty$ it is $O(t^{-1-r}\log t)$; both are
integrable for $0 < r < 1$. Integrate by parts (boundary terms vanish) and substitute $u = xt$:
$$\int_0^\infty\log(1 + xt)\,t^{-1-r}dt = \frac1r\int_0^\infty\frac{x\,t^{-r}}{1 + xt}dt = \frac{x^r}{r}\int_0^\infty\frac{u^{-r}}{1 + u}du = \frac{x^r}{r}\cdot\frac{\pi}{\sin(\pi r)},$$
by the Beta integral $\int_0^\infty u^{\alpha - 1}/(1 + u)\,du = \pi/\sin(\pi\alpha)$ at $\alpha = 1 - r$. $\square$

*Proof of Theorem 6.1.* For $r = 1$, Proposition 6.5 at $k = 1$ gives $\sum_l x_l \ge \nu$, i.e.
$\sum_l|c_l| \ge |A|$. For $0 < r < 1$, apply Lemma 6.7 to each $x_l$ and integrate the logarithm of
Proposition 6.6 against the positive measure $t^{-1-r}dt$:
$$\sum_l x_l^r = \frac{r\sin\pi r}{\pi}\int_0^\infty\Big[\sum_l\log(1 + x_lt)\Big]t^{-1-r}dt
\;\ge\; \frac{r\sin\pi r}{\pi}\int_0^\infty\nu\log(1 + t)\,t^{-1-r}dt = \nu\cdot 1^r = \nu.$$
Unwinding $x_l = \nu|c_l|/|A|$ gives $\sum_l|c_l|^r \ge \nu^{1-r}|A|^r$.

*Equality, $0 < r < 1$.* The integrand $\sum_l\log(1 + x_lt) - \nu\log(1 + t)$ is continuous and
non-negative (Proposition 6.6) and integrates to zero against a strictly positive density, so
$\prod_l(1 + x_lt) = (1 + t)^\nu$ for all $t > 0$, hence as polynomials. By unique factorisation the
non-zero $x_l$ are exactly $\nu$ copies of $1$: exactly $\nu$ atoms, each of modulus $|A|/\nu$. Then
$e_\nu(x) = 1 = \binom{\nu}{\nu}$ and Proposition 6.5 is tight at $k = \nu$; the only $\nu$-subset with
all $d_l \ne 0$ is the support $L_0$, so $|\prod_{l \in L_0}d_l|\,G_{L_0} = 1$ with $|\prod d_l| = 1$,
giving $G_{L_0} = 1$: the $\nu$ chain vectors are orthonormal (Lemma 6.4). Conversely any $\nu$ atoms of
modulus $|A|/\nu$ satisfying $(\mathrm{JF}_k)$ give $\sum_l|c_l|^r = \nu(|A|/\nu)^r = \nu^{1-r}|A|^r$. $\square$

> **Corollary 6.8 (atom count).** Under (S), a symmetric table needs at least $\nu$ neurons with a
> non-zero atom at every feature; with $n$ or fewer the feature's clean amplitude is zero.

> **Corollary 6.9 (orthonormal chain frame).** Two atoms' chain vectors are orthogonal iff
> $\sum_{k=0}^n\prod_{j \le k}\big(v_{l'j}\overline{v_{lj}}\big) = 0$. If the slot ratios
> $v_{l'j}\overline{v_{lj}} = \zeta$ agree across slots this reads $1 + \zeta + \dots + \zeta^n = 0$:
> $\zeta$ is a non-trivial $\nu$-th root of unity. The block of Proposition 7.1 is such a frame.

**Remark (aligned case).** If all $c_l$ share a phase, rotate so $c_l \ge 0$; then $M$ is Hermitian, and
a Hermitian upper-triangular matrix is diagonal: $M = AI$. Evaluating at $m_{l_0}$,
$A\nu = \sum_l c_l|\langle m_{l_0}, m_l\rangle|^2 \ge c_{l_0}\nu^2$, so no atom of an aligned junk-free
feature carries more than a $\nu$-th of it. Without alignment $M$ is not Hermitian and only
triangularity survives, which is why the determinantal route is needed.

**Remark (other penalties).** $x \mapsto x^r$ is a positive average of $x \mapsto \log(1 + xt)$, so the
same proof gives $\sum_l f(\nu|c_l|/|A|) \ge \nu f(1)$ for every
$f(x) = bx + \int_0^\infty\log(1 + xt)\,\mu(dt)$ with $b, \mu \ge 0$. The result is not special to
power penalties.

---

## 7. The margin ceiling in the symmetric class

> **Theorem 7.1 (margin ceiling, symmetric class).** For every neuron-additive $\nu$-homogeneous
> model satisfying (B) and (S), every width, every $q \in [2, \nu]$, and every symmetric table,
> $$\gamma_q(\theta) \;\le\; \nu^{-(\nu - q)/q}\ \frac{2}{p^\nu}\,C_q^{\,\nu/q}\ \Gamma_{\nu/q}(p).$$

*Proof.* Let $r = q/\nu$. Fix $k \in F$. By (S) and (JF) the atoms $(c_{ik}, v_{i\cdot k})_i$ satisfy
$(\mathrm{JF}_k)$ with total $G_k$ (neurons with $c_{ik} = 0$ contribute nothing to either side), so
Theorem 6.1 gives $\sum_i|c_{ik}|^r \ge \nu^{1-r}|G_k|^r$. Sum over $k \in F$ and apply (B) neuron by
neuron:
$$\nu^{1-r}\sum_{k \in F}|G_k|^r \;\le\; \sum_i\sum_{k \in F}|c_{ik}|^r \;\le\; C_q\,P_q(\theta).$$
Hence $\|G\|_r \le \nu^{(r-1)/r}C_q^{\nu/q}P_q^{\nu/q}$; insert into Proposition 4.1 and note
$(r - 1)/r = (q - \nu)/q$. $\square$

The class bound differs from the universal bound by the factor $\nu^{(\nu-q)/q}$: $\sqrt3$ at $\nu = 3$,
$4$ at $\nu = 4$, $5^{3/2}$ at $\nu = 5$ under plain weight decay, and $1$ at $q = \nu$. This is exactly
the price of splitting each feature over $\nu$ neurons to cancel its junk.

> **Corollary 7.2 (feature count).** Let $q < \nu$. Every symmetric table attaining the bound of
> Theorem 7.1 has clean amplitudes $|G_k| \propto y^\star_k$ for a maximiser $y^\star$ of
> $\Gamma_{\nu/q}(p)$; in particular it has exactly $K^*(\nu/q, p)$ features. At each feature exactly
> $\nu$ neurons carry a non-zero atom, all of modulus $|G_k|/\nu$, with orthonormal chain vectors.

*Proof.* Equality must hold in every step. Equality in Proposition 4.1 forces $y = |G|/\|G\|_r$ to be a
maximiser of $\Gamma_s$; equality in Theorem 6.1 at each used $k$ (with $r < 1$) forces the
$\nu$-atom structure. $\square$

Whether the bound is *attained* is a per-model question, and it is where the product model earns its
place. Write $\theta_j = 2\pi ka_j/p$ and $\theta_c = 2\pi kc/p$.

> **Proposition 7.3 (the $\nu$-block attains it for the product model).** Fix $k \in F$, $\rho > 0$,
> and for $l = 0, \dots, n$ let $\varphi_l = \pi l/\nu$. The $\nu$ product neurons
> $$e_{lj}(a) = \rho\cos(\theta_j + \varphi_l)\ (j = 1..n), \qquad w_l(c) = \rho\cos(\theta_c + n\varphi_l)$$
> have total table
> $$\sum_{l=0}^n w_l(c)\prod_{j=1}^n e_{lj}(a_j) \;=\; \nu\rho^\nu 2^{-n}\cos\big(\theta_1 + \dots + \theta_n - \theta_c\big),$$
> a function of (answer $-$ candidate) alone, so the block is exactly junk-free; each neuron costs
> $\|\theta_l\|^2 = \nu\rho^2p/2$. Consequently, if $y^\star$ maximises $\Gamma_{\nu/q}(p)$, the network
> that places at each $k \in \operatorname{supp}(y^\star)$ a block with $\rho_k^\nu = 2^ny^\star_k/\nu$ is
> symmetric and attains the bound of Theorem 7.1 with $C_q = (p/2\nu)^{q/2}$, i.e.
> $\gamma_q = \nu^{-(\nu-q)/q}\,B\,\Gamma_{\nu/q}(p)$ with $B = 2(2p\nu)^{-\nu/2}$.

*Proof of the identity.* Expand each cosine into exponentials. Neuron $l$ contributes
$2^{-\nu}\rho^\nu\sum_{\varepsilon \in \{\pm1\}^n,\,\delta = \pm1}\exp\big(i(\varepsilon\cdot\theta + \delta\theta_c)\big)\exp\big(i\varphi_l(\sum_j\varepsilon_j + \delta n)\big)$.
With $t = |\{j : \varepsilon_j = -1\}|$, $\sum_j\varepsilon_j = n - 2t$, so the $l$-dependent phase is
$\exp(2\pi i\,lu/\nu)$ with $u = n - t$ if $\delta = +1$ and $u = -t$ if $\delta = -1$. Summing over
$l = 0..n$ gives $\nu[u \equiv 0 \bmod \nu]$; as $0 \le t \le n$ this happens only for $(t = n, \delta = +1)$
and $(t = 0, \delta = -1)$, i.e. $e^{\mp i(\sum_j\theta_j - \theta_c)}$, whose sum is
$2\cos(\sum_j\theta_j - \theta_c)$. For the cost, $\sum_a\cos^2(2\pi ka/p + \varphi) = p/2$ for
$k \not\equiv 0$.

*Proof of attainment.* Scaling $y^\star$ up increases the objective, so $\sum_k(y^\star_k)^r = 1$. With
$x_k := \nu\rho_k^\nu 2^{-n} = y^\star_k$ the table is exactly
$z(a, c) = \sum_k y^\star_k\cos(2\pi k(s(a) - c)/p)$, so (JF) holds and, on every input,
$z(a, s(a)) - z(a, s(a) - D) = \sum_k y^\star_kg_k(D)$; hence
$\operatorname{margin} = \min_{D \ne 0}\sum_ky^\star_kg_k(D) = \Gamma_s(p)$. The cost is
$P_q = \sum_k\nu(\nu\rho_k^2p/2)^{q/2} = \nu(\nu p/2)^{q/2}\sum_k\rho_k^q$, and
$\rho_k^q = (\rho_k^\nu)^r = (2^n/\nu)^r(y^\star_k)^r$, so $P_q = \nu(\nu p/2)^{q/2}(2^n/\nu)^r$ and
$P_q^{\nu/q} = \nu^{\nu/q}(\nu p/2)^{\nu/2}\,2^n/\nu$. Therefore
$$\gamma_q = \frac{\Gamma_s}{\nu^{\nu/q - 1}(\nu p/2)^{\nu/2}2^n} = \nu^{-(\nu - q)/q}\cdot 2^{1 - \nu/2}(p\nu)^{-\nu/2}\,\Gamma_s = \nu^{-(\nu-q)/q}\,B\,\Gamma_s,$$
using $-n + \nu/2 = 1 - \nu/2$. $\square$

The *share weighting* $\rho_k^\nu \propto y^\star_k$ is essential: the optimal shares are not equal (at
$p = 59$, $s = 3/2$ they range from $0.148$ to $0.248$), and an equal-amplitude block on the optimal
support does not attain the bound.

**What the theorems say together.** The programme $\Gamma_s$ and its support size $K^*$ are the same
objects in Theorem 4.2 and Theorem 7.1; only the constant differs. Under plain weight decay the count
of the class optimum falls with the degree and grows only slowly with the modulus:

| $p$ | $\nu = 3$ ($s = 3/2$) | $\nu = 4$ ($s = 2$) | $\nu = 5$ ($s = 5/2$) |
|---|---|---|---|
| 59 | 5 | 4 | 3 |
| 71 | 5 | 4 | 3 |
| 113 | 7 | 4 | 4 |

($K^*$ from the free-share programme, `LADDER.md` §4.1; the equal-share ladder argmax agrees except at
$(59, 3)$: 6 and $(113, 3)$: 5. Support enumeration is exhaustive for $K \le 8, 7, 5$ at
$p = 59, 71, 113$ and heuristic above, so a missed support can only raise $K^*$.) The dense solution of
Corollary 5.1, evaluated by the same machinery, reaches $0.52$–$0.63$ of the class optimum at
$\nu = 3$, $0.15$–$0.25$ at $\nu = 4$ and $0.04$–$0.08$ at $\nu = 5$: the preference for sparsity
sharpens with every rung of the degree.

---

## 8. Checking the assumptions for the concrete models

Write $\hat f(k) = \sum_a f(a)\omega^{-ka}$ for a vector $f \in \mathbb{R}^p$, so
$\sum_k|\hat f(k)|^2 = p\|f\|^2$ (Parseval) and $\hat f(-k) = \overline{\hat f(k)}$. For a real vector,
$\sum_{k \in F}|\hat f(k)|^2 = \tfrac12\sum_{k \ne 0}|\hat f(k)|^2 \le \tfrac p2\|f\|^2$.

**Product model** $z_i(a, c) = w_i(c)\prod_je_{ij}(a_j)$, $\nu = n + 1$,
$\|\theta_i\|^2 = \sum_j\|e_{ij}\|^2 + \|w_i\|^2$.

*Atoms.* Inserting the inverse transforms into $g$ and averaging over inputs kills every Fourier term
unless all operand frequencies equal $k$ and the candidate frequency is $-k$; so
$c_{ik} = \prod_j\hat e_{ij}(k)\,\overline{\hat w_i(k)}$ (with the normalisation of §2). *(S)* holds with
$v_{ijk} = \overline{\hat e_{ij}(k)}/\hat e_{ij}(k)$ ($:= 1$ if $\hat e_{ij}(k) = 0$): the coefficient at
the mode flipping $T$ is $\prod_{j \notin T}\hat e_{ij}(k)\prod_{j \in T}\hat e_{ij}(-k)\,\hat w_i(-k)$ and
$\hat e_{ij}(-k) = v_{ijk}\hat e_{ij}(k)$.

*(B)* with $C_q = (p/2\nu)^{q/2}$. Let $m_{ik} = \sum_j|\hat e_{ij}(k)|^2 + |\hat w_i(k)|^2$. AM–GM over the
$\nu$ factors: $|c_{ik}| \le (m_{ik}/\nu)^{\nu/2}$, so $|c_{ik}|^{q/\nu} \le (m_{ik}/\nu)^{q/2}$. Since
$q/2 \ge 1$, $t \mapsto t^{q/2}$ is superadditive: $\sum_km_{ik}^{q/2} \le (\sum_km_{ik})^{q/2}$. Parseval:
$\sum_{k \in F}m_{ik} \le \tfrac p2\|\theta_i\|^2$. Combine. Equality iff the neuron is balanced (all
$\nu$ Fourier magnitudes equal at each frequency it uses) and, for $q > 2$, single-frequency; at
$q = 2$ the superadditivity step is an identity, so a neuron may carry several frequencies at no extra
cost.

**Power model** $z_i(a, c) = w_i(c)\big(\sum_je_{ij}(a_j)\big)^n$, $\nu = n + 1$. Expand the power into
monomials $\prod_je_{ij}(a_j)^{m_j}$, $\sum_jm_j = n$. A monomial with some $m_j = 0$ has no Fourier
content in slot $j$, and one with some $m_j \ge 2$ has some other $m_{j'} = 0$; so only the multilinear
monomial, with coefficient $n!$, reaches any mode with all $\nu$ entries non-zero. Hence
$c_{ik} = n!\prod_j\hat e_{ij}(k)\overline{\hat w_i(k)}$, (S) holds with the same $v_{ijk}$, and (B) holds
with $C_q = (n!)^{q/\nu}(p/2\nu)^{q/2}$. Theorems 4.2 and 7.1 therefore apply; Proposition 7.3 does
*not* (the block's squared terms are junk at modes with a zero slot, so the block is not junk-free;
see §9).

**ReLU model** $z_i(a, c) = w_i(c)\,\sigma(\sum_je_{ij}(a_j))$, $\sigma = \mathrm{ReLU}$, $\nu = 2$. (B) at
$q = \nu = 2$: with $h_i = \sigma(\sum_je_{ij}(a_j))$, the atom is $c_{ik} = \hat h_i(k, \dots, k)\overline{\hat w_i(k)}$
(diagonal multi-index). Cauchy–Schwarz over $k$, then $\sum_{k \in F}|\hat h_i(k..k)|^2 \le \tfrac12\operatorname{Var}$-mass of
$h_i$, which is at most the input variance because $\sigma$ is $1$-Lipschitz, gives
$\sum_{k \in F}|c_{ik}| \le C_2\|\theta_i\|^2$ with an explicit $C_2$. So Theorem 4.2 and Corollary 5.2's
flat maximiser apply: plain weight decay on this degree-2 model *is* the $q = \nu$ endpoint, and the
bound's unique maximiser is the flat spectrum. (S) is not claimed for ReLU, and the bound is not
known to be attained, so density of the max-margin ReLU network is a statement about the bound's
maximiser, not a theorem about the network.

**Where the assumptions fail.** A model with a shared linear layer in front of the neurons (the
"ReLU with linear layers" family, transformers) does not decompose into neurons paying their own
cost: the pre-layer is priced by a nuclear norm that rewards carrying many frequencies in one
neuron, and (B) does not hold per neuron. For such models the only available statement is a
symmetrisation bound with an explicit, measured coupling factor (`THEORY_relu.md` §3); it is not part
of this section.

---

## 9. Scope: what is proved, what is not

**Proved for every table.** Lemmas 1–2, Proposition 4.1.

**Proved for every neuron-additive $\nu$-homogeneous model with (B).** Theorem 4.2 (universal
bound), Corollary 5.1 (flat is the unique maximiser at $s = 1$), Corollary 5.2.

**Proved with (B) and (S), in the symmetric class.** Theorem 6.1 (feature cost, every $N$, every
$r \in (0, 1]$, with equality case), Corollaries 6.8–6.9, Theorem 7.1 (class ceiling), Corollary 7.2
(feature count). Attained for the product model (Proposition 7.3).

**Hypotheses that are load-bearing and must appear in the statements.**
1. *Full table.* Lemma 1 averages over every input with a given answer. With a training fraction
   below 1 the average includes inputs whose margins are unconstrained, and the training-set margin
   need not obey the bound.
2. *(JF) is a hypothesis on the table, not a fact about trained networks.* Trained networks under
   plain weight decay carry junk at the sign-flipped modes; the empirical section reports how much
   and in which direction it moves with $q$.
3. *$K^*$ is the support size of the free-share programme.* The equal-share ladder is a lower bound
   on $\Gamma_s$ and its argmax is an approximation to $K^*$. Adjacent rungs are often within a few
   percent, so at some $(p, s)$ the honest statement is a band of width one.

**Not proved.**
- *Global optimality outside the class (Conjecture).* For odd prime $p$, $n \ge 1$, $q \in [2, \nu]$:
  the supremum of $\gamma_q$ over *all* product networks equals the symmetric-class value of
  Theorem 7.1, i.e. no junk-carrying network beats it. Known: true at $p = 3$ for every $n$; true in
  the window $\nu - 4/(3(p-1)) \le q \le \nu$ for every $p$, and for $q \ge 5/2$ at $p = 5$; true on
  the class of single-frequency neurons at $n = 2$, $q = 2$ for every support whose certified aiming
  depth is at least $0.3476$, which includes the $K^*$ supports at $p \in \{59, 71, 97, 113, 127\}$; the
  symmetric optimum is a strict local maximum; no ascent from any start at $p \le 23$ has exceeded it.
  Also known: a junk-carrying *local* maximum exists at $p = 5$, $n = 2$, $q = 2$ at $0.837$ of the
  class value, so no local-to-global argument can work; and the group analogue is *false* on
  $\mathbb{Z}_{12}$, $D_3$ and $S_4$, where structured junk beats the symmetric ceiling by $1.06$–$1.62$.
  Any proof must be genuinely global and specific to prime modulus.
- *Attainment for the power model.* Theorems 4.2 and 7.1 hold; no junk-free population of power
  neurons is known to attain the class bound. The smallest known junk-free population at $n = 2$
  uses 6 neurons and reaches $0.408$ of the universal bound, against the product model's
  $3^{-1/2} = 0.577$. With tied embeddings ($e_{i1} = e_{i2}$) no junk-free population exists at all.
- *Dynamics.* Nothing here says gradient descent reaches the maximiser. Wei et al. (2019, Thm 4.1)
  connects the $\lambda \to 0$ global minimiser of $\mathrm{CE} + \lambda P_q$ to it ($P_q^{1/q}$ is a
  norm for $q \ge 1$ and the model is $\nu$-homogeneous). Lyu and Li's KKT statement applies only at
  $q = 2$.
- *$q$ outside $[2, \nu]$.* For $q > \nu$ the ratio is unbounded at unbounded width. For $q < 2$ the
  superadditivity step in (B) fails (one neuron carrying every frequency violates it by a factor
  $|F|^{1 - q/2}$); the available bound is the $q = 2$ programme with a rescaled budget, not attained,
  so nothing about the count is claimed there.
