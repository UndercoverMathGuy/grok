# The chain-frame proof: P1 closed for every `N`, every `n`, every `r`

Sixth pass on the global-optimality programme. Continues `GOPT_STATE.md` / `GOPT_STATE2.md` and the
write-ups they point to (`LEMMA_ATTEMPT.md`, `GLOBAL_OPT.md`, `GLOBAL_OPT3.md`, `GLOBAL_OPT4.md`,
`ADVERSARY.md`, `P1A.md`, `P1B.md`, `P1C.md`). Brief: turn the correct-but-ugly proofs into elegant
ones for the ICLR 2027 paper.

**Headline.** Problem **P1** — the sharp constant `n+1` for *every* number of atoms `N` — is now
**PROVED outright**: every `n >= 1`, every `N`, every `r in (0,1]`, in about a page, with no peeling
constants, no tail price, no two-variable programme and no interval-arithmetic certificate. The
mechanism is a change of coordinates. Map each atom to the vector of its *running phase products*;
then a junk-free feature becomes a **triangular resolution of the identity of `C^{n+1}` by rank-one
projections**, and the whole inequality follows from

1. the minor (Cauchy–Binet) expansion of the elementary symmetric functions of that resolution,
2. Hadamard's determinant inequality for Gram matrices of unit vectors,
3. the integral representation `x^r = (r sin(pi r)/pi) int_0^inf log(1 + x t) t^{-1-r} dt`.

Immediate consequences:

* the open exponent window `r in (r*, 1)` closes (`ADVERSARY` §6's caveat, `P1A` §4, `P1B` §7);
* `nu >= 6` closes (`P1B`'s failing cell), as does every `n`;
* every `q` in `[2, nu]` closes, not just `q <= 2.5`;
* the slack factor `F(n,q)` of `GLOBAL_OPT3` Theorem D becomes `1` at every `nu` and every `q`, so
  the paper's main theorem is **exact on the junk-free class**;
* the Key Lemma (`N = n+1`) becomes a three-line corollary and its equality case becomes
  "the frame is orthonormal"; agent 2's Theorem A (`N >= n+1`) and the trivial `r = 1` triangle
  inequality become the two extreme degrees `k = n+1` and `k = 1` of one family of identities;
* the theorem holds under a **weaker hypothesis** than `(KL)`: only the `binom(n+1,2)` *interval*
  sign patterns `T = {s, s+1, ..., t}` are used, not all `2^n - 1`;
* `P1B`'s Proposition A1 conjecture — *a probability multi-affine cubature rule has no weight above
  `1/(n+1)`*, proved there only for `n <= 2` — falls out in two lines for every `n` (Corollary 1.4,
  §2.7), and with it `P1A`'s entropy statement `(E0)`;
* three of the four repairs the hostile referee asked for (`VERIFY.md` R1–R4) disappear rather than
  being repaired, and the fourth is superseded (§3.1).

**What stayed ugly / open.** Junk (P2, P3) is untouched: §6 explains precisely why the chain
identity does not extend to the junk case as it stands (the "mirror" entries of the matrix are
uncontrolled), and what the smallest missing ingredient is.

Local CPU, fp64, no pods, $0. New files only: `ELEGANT.md`, `eleg_chain.py`, `eleg_bigN.py`,
`eleg_aligned.py`, `eleg_routes.py`, outputs in `FINAL_RUNS/ELEG/`. No background jobs remain.

---

## 0. Symbols, with a plain reading of each

| symbol | meaning | plain reading |
|---|---|---|
| `n` | number of operand slots | how many numbers are added |
| `nu = n+1` | homogeneity degree of the gated chain | scaling every weight by `s` scales every logit by `s^nu` |
| `q in [2, nu]`, `r = q/nu in (0,1]` | penalty exponent and its ratio | `q = 2` is plain weight decay, `q = nu` is Morwani's `L_{2,nu}`; `r = 1` means splitting a feature over several neurons is free |
| `N` | number of atoms at one frequency | one atom per neuron that touches this frequency (no purity needed — `GLOBAL_OPT3` §2) |
| `v_lj in C`, `\|v_lj\| = 1` | atom `l`'s phase in slot `j` | the phase of that neuron's `j`-th embedding at this frequency |
| `c_l in C` | atom `l`'s complex amplitude | that neuron's signed clean contribution |
| `A in C` | the clean total `sum_l c_l` | the feature's Fourier coefficient `G_k` |
| `T` subset of `{1..n}` | a sign pattern | the set of operands whose Fourier sign is flipped |
| `(KL)` | `sum_l c_l prod_{j in T} v_lj = A [T empty]` for every `T` | "the neurons carry clean amplitude `A` and cancel all `2^n - 1` junk sign patterns" |
| `D := n+1` | **dimension of the chain space** | the new name for `n+1`; everything below happens in `C^D` |
| `m_l in C^D` | `(m_l)_k = v_l1 v_l2 ... v_lk` for `k = 0..n` (empty product `= 1`) | the **chain vector**: atom `l`'s running phase products |
| `u_l = m_l / sqrt(D)` | unit vector in `C^D` | the same, normalised; `M_l := u_l u_l^*` is a rank-one projection |
| `d_l = D c_l / A` | rescaled amplitude | measures an atom in units of "one `D`-th of the feature"; `\|d_l\| = 1` at the optimum |
| `x_l = \|d_l\|` | modulus | the atom's size in those units |
| `e_k(x)` | `sum over k-subsets L of prod_{l in L} x_l` | the `k`-th elementary symmetric function |
| `E_k(M)` | coefficient of `z^k` in `det(I + zM)` | the `k`-th elementary symmetric function of `M`'s eigenvalues |
| `G_L` | `det Gram(u_l : l in L)` | how independent the atoms of `L` are: `1` if orthonormal, `0` if dependent |
| `[P]` | `1` if `P` holds, else `0` | Iverson bracket |

If `A = 0` there is nothing to prove, so `A != 0` throughout, and we normalise `A = 1` by dividing
every `c_l` by `A`.

---

## 1. Paper-ready: main theorem, proof sketch, appendix lemma graph

### 1.1 The two theorems

> **THEOREM 1 (Feature Cost Theorem).** *Let `v_lj` (`l = 1..N`, `j = 1..n`) be complex numbers of
> modulus `1` and let `c_1, ..., c_N` be complex numbers with*
>
>     sum_{l=1}^N c_l prod_{j in T} v_lj  =  A [T empty]      for every T subset of {1..n}.
>
> *Then for every `r in (0, 1]`*
>
>     sum_{l=1}^N |c_l|^r  >=  (n+1)^{1-r} |A|^r ,
>
> *and for every `r in (0, 1)` equality holds if and only if exactly `n+1` of the `c_l` are
> non-zero, they all have modulus `|A|/(n+1)`, and their chain vectors are pairwise orthogonal.*
> *(At `r = 1` the inequality is the triangle inequality and its equality class is much larger —
> every phase-aligned configuration, any number of atoms — so the rigidity statement is for
> `r < 1` only.)*
>
> *Plain reading: to cancel all `2^n - 1` junk sign patterns while keeping clean amplitude `A`, a
> feature must be spread over at least `n+1` neurons, it must spread evenly, and no number of extra
> neurons ever makes it cheaper.*

> **COROLLARY 1.4 (Tight Frame; the aligned case).** *If the `c_l` all share one phase —
> equivalently `sum_l |c_l| = |A|`, i.e. `c/A` is a genuine **probability** cubature rule — then the
> chain vectors form a **tight frame**,*
>
>     sum_l c_l m_l m_l^*  =  A * I_{n+1}      (exactly, not merely triangular),
>
> *and consequently `|c_l| <= |A|/(n+1)` for every `l`, with equality attained at every `n`.*
>
> This is `P1B`'s Proposition A1 conjecture ("`max_l x_l <= 1/(n+1)` for probability cubature
> rules"), which `P1B` proved only for `n <= 2` by a kernel certificate that fails at `n = 3`
> (`A_3 = 3/10 > 1/4`) and listed as its ranked open item 2. It is proved here for every `n`, in two
> lines (§2.7), and it implies `P1A`'s entropy statement `(E0)` — *a probability measure on `T^n`
> killing every non-trivial multi-affine character has Shannon entropy at least `log(n+1)`* — since
> `H >= H_inf = -log max_l p_l >= log(n+1)`.

> **THEOREM 2 (Margin Ceiling; the paper's main theorem).** *Let `theta` be any gated-chain network
> on the full `n`-operand table mod a prime `p` — any width, neurons may mix frequencies, no purity
> assumption — and suppose its same-frequency junk vanishes:*
>
>     (H)   J_{k,T} = 0    for every frequency k and every non-empty T subset of {1..n}.
>
> *Then for every `q in [2, nu]`,*
>
>     gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p) ,     B = 2 (2 p (n+1))^{-nu/2},
>
> *and the share-weighted `(n+1)`-neuron common-phase block placed on a maximiser of the programme
> `Gamma_{nu/q}(p)` attains it. In particular the maximiser carries exactly `K*(nu/q)` frequencies,
> each on exactly `n+1` neurons.*

Theorem 2 is Theorem 1 plus the two bookkeeping steps already in `GLOBAL_OPT3` §2; it is written out
in §3.

### 1.2 The proof of Theorem 1 in one paragraph

Send atom `l` to its **chain vector** `m_l = (1, v_l1, v_l1 v_l2, ..., v_l1...v_ln)` in `C^{n+1}`.
Because the phases are unimodular, `(m_l m_l^*)_{k,k'} = prod_{j=k'+1..k} v_lj` whenever `k > k'` —
exactly the junk monomial of the *interval* sign pattern `{k'+1, ..., k}` — and `= 1` on the
diagonal. So the matrix `M := sum_l c_l m_l m_l^*` is **upper triangular with every diagonal entry
equal to `A`**, hence all its eigenvalues are `A` and `E_k(M) = binom(n+1, k) A^k`. On the other
hand `M` is a combination of rank-one projections: with `d_l := (n+1) c_l / A` and
`u_l := m_l/sqrt(n+1)` (a unit vector, since `||m_l||^2 = n+1`) we have `M/A = sum_l d_l u_l u_l^*`,
and Sylvester's determinant identity expands

    E_k(M/A) = sum over k-subsets L of ( prod_{l in L} d_l ) * G_L ,   G_L = det Gram(u_l : l in L).

Gram matrices of unit vectors are positive semi-definite with unit diagonal, so `0 <= G_L <= 1`
(Hadamard), and comparing moduli gives `e_k(x) >= binom(n+1, k)` for every `k`, where `x_l = |d_l|`.
Multiplying by `t^k` and summing over `k` turns the whole family into a single inequality between
polynomials,

    prod_{l=1}^N (1 + x_l t)  >=  (1 + t)^{n+1}       for every t >= 0 ,

whose logarithm, integrated against the positive measure `t^{-1-r} dt` and read through
`x^r = (r sin(pi r)/pi) int_0^inf log(1+xt) t^{-1-r} dt`, gives
`sum_l x_l^r >= (n+1) * 1^r = n+1`, which is the theorem. ∎

### 1.3 The three ideas, named

1. **The chain map** `v |-> (1, v_1, v_1v_2, ...)`. It converts the *multiplicative* junk conditions
   into a *linear-algebraic* one. This is the ingredient every earlier pass lacked:
   `GLOBAL_OPT4`'s Theorem O forbids proving P1 with one averaged test polynomial, and `P1B` §8.2
   shows the slot-product structure is load-bearing — the chain map is precisely how that structure
   enters.
2. **Minor expansion + Hadamard.** One identity per degree `k = 1..n+1`. The two extreme degrees are
   the two facts that were already known — `k = 1` is `sum_l |c_l| >= |A|` (the case `r = 1`),
   `k = n+1` is "at least `n+1` non-zero atoms" (`LEMMA_ATTEMPT` Theorem 1b / agent 2's Theorem A,
   the case `r -> 0`) — and the `n-1` middle degrees are exactly what interpolates between them.
3. **The Bernstein integral.** `x |-> x^r` is a *complete Bernstein function*: a positive average of
   the functions `x |-> log(1 + x t)`. So the polynomial inequality of step 2 upgrades for free:

   > **COROLLARY 1.3.** For every function `f(x) = b x + int_0^inf log(1 + x t) mu(dt)` with
   > `b >= 0` and `mu` a positive measure (the complete Bernstein functions vanishing at `0`),
   > `sum_l f(|d_l|) >= (n+1) f(1)`. Examples: `x^r` (`0 < r < 1`), `log(1+x)`, `x/(1+x)`,
   > `x^r log(1+x)`. So the result is not special to a power penalty.

   This is the interpolation between `r -> 0` and `r = 1` the brief asked for — but it interpolates
   the **constraints**, not the optimal value. (Log-convexity of the optimal value in `r` is a red
   herring; see §5.1.)

### 1.4 Appendix lemma graph (proposed for the paper)

```
   (JF)   junk-free feature at one frequency:  N atoms (c_l, v_l),  clean total A
     |
     |  Lemma 1  (Chain Lemma)          -- only the binom(n+1,2) interval patterns are used
     v
   M = sum_l c_l m_l m_l^*  is upper triangular with diagonal A        [ m_l in C^{nu} ]
     |
     |  Lemma 2  (Minor Expansion, Sylvester/Cauchy-Binet)
     |  Lemma 3  (Hadamard for Gram matrices of unit vectors)
     v
   Prop 4  e_k(x) >= binom(nu, k),   k = 0..nu ,   x_l = nu |c_l| / |A|
     |                                          \
     |  (sum against t^k)                        --> Corollary 1.1  N >= nu
     |                                                [was Theorem 1b / "Theorem A"]
     v
   Prop 5  prod_l (1 + x_l t) >= (1 + t)^nu   for all t >= 0
     |
     |  Lemma 4  (Bernstein integral for x^r)
     v
   THEOREM 1  sum_l |c_l|^r >= nu^{1-r} |A|^r
     |                                          \
     |                                           --> Corollary 1.2  equality = orthonormal
     |                                                chain frame with equal moduli
     |                                                [was "Key Lemma", Fekete/roots of unity]
     |                                          \
     |                                           --> Corollary 1.4  aligned case: M = A I is a
     |                                                TIGHT FRAME, so |c_l| <= |A|/nu
     |                                                [was P1B Prop A1, open for n >= 3]
     |  Prop 6  (Level-set Bound: orbit averaging + phase lemma)   [GLOBAL_OPT3 Step 3]
     |  Prop 7  (Neuron Budget: per-neuron AM-GM + Parseval)       [GLOBAL_OPT3 Step 4]
     v
   THEOREM 2  (H) ==> gamma_q <= nu^{-(nu-q)/q} B Gamma_{nu/q}(p)   [FULL TABLE]
     |
     |  Prop 8  (Attainment: the share-weighted (n+1)-block)
     v
   COROLLARY 2.1  (q < nu) any attaining network has K*(nu/q) features,
                  each carried by exactly n+1 neurons on an orthonormal chain frame
```

Four lemmas, five propositions, two theorems, five corollaries — and **not one continuum
minimisation, case split, certificate or numerically evaluated constant anywhere in it.**
**Everything it replaces:**
Theorem A, Theorem B, Theorem E, Theorem H, Lemma T (both of them), Lemma C, Lemmas R / R' / R\* /
R'', Propositions A1 / A2 / B1 / B2, the constants `kappa_1, kappa_2, kappa_3`, `D_h(n)`, `L_r(n)`,
`F(n,q)`, `eps0`, `eps1`, `r*(n)`, `a*(n)`, the entropy criterion `(E_rho)` and `(E0)`, the
two-variable programmes `G(beta, tau)` / `Val(R, s)`, and the 101-box interval-arithmetic
certificate. They remain in the record as independent confirmation (§7).

---

## 2. Theorem 1, proved

Fix `n >= 1`, put `D := n+1`, normalise `A = 1`.

### 2.1 Lemma 1 (Chain Lemma)

> **LEMMA 1.** Define the **chain vector** of atom `l` by
>
>     m_l := ( 1, v_l1, v_l1 v_l2, ..., v_l1 v_l2 ... v_ln )  in  C^D ,
>     (m_l)_k := prod_{j=1}^{k} v_lj    (k = 0..n; the empty product is 1).
>
> Then `||m_l||^2 = D`, and the matrix `M := sum_{l=1}^N c_l m_l m_l^*` satisfies
>
>     M_{k,k}   = sum_l c_l = A                                   for every k,
>     M_{k,k'}  = sum_l c_l prod_{j = k'+1}^{k} v_lj  =  0        for every k > k',
>
> i.e. **`M` is upper triangular with constant diagonal `A`**. Only the `binom(D,2) = n(n+1)/2`
> equations of `(KL)` whose sign pattern `T` is a non-empty *interval* `{k'+1, ..., k}` are used.
>
> *Plain reading: write down an atom's running phase products; the outer product of that list with
> itself has the feature's interval junk coefficients below the diagonal and the clean amplitude on
> it. Junk-freeness makes the matrix triangular.*

*Proof.* Each `|(m_l)_k| = 1`, so `||m_l||^2 = D` and the diagonal entry of `m_l m_l^*` is `1`; the
diagonal of `M` is therefore `sum_l c_l = A`, which is the `T = empty` equation. For `k > k'`,
unimodularity gives `conj(v_lj) = 1/v_lj`, so

    (m_l m_l^*)_{k,k'} = (prod_{j<=k} v_lj) * conj(prod_{j<=k'} v_lj) = prod_{j = k'+1}^{k} v_lj ,

the junk monomial of the non-empty interval `T = {k'+1, ..., k}`; summing against `c_l` gives
`C_T = 0`. ∎

**Warning (this is where a natural stronger claim is false).** The entries *above* the diagonal are
`M_{k,k'} = sum_l c_l conj(prod_{j=k+1}^{k'} v_lj)` for `k < k'` — "mirror" moments in which the
phases, but not the amplitudes, are conjugated. These are **not** controlled by `(KL)` and are
generically non-zero, so `M != A I`. An explicit witness at `n = 1`, `N = 2`: `v_1 = 1`, `v_2 = i`
gives `c = ((1-i)/2, (1+i)/2)` and `M = [[1, 1-i], [0, 1]]` (`eleg_routes.py` block `mirror`).
Triangularity is all that is needed, and all that is true. (I made exactly this slip in a first
draft; it is recorded in §5.4 because it is the one trap on this route.)

### 2.2 Lemma 2 (Minor Expansion) and Lemma 3 (Hadamard)

> **LEMMA 2.** Let `u_1, ..., u_N in C^D` and `d_1, ..., d_N in C`, and put
> `M' := sum_l d_l u_l u_l^*`. Then for every `k = 0, ..., D`
>
>     E_k(M')  =  sum over k-subsets L of {1..N}  ( prod_{l in L} d_l ) * det( Gram(u_l : l in L) ) ,
>
> where `Gram(u_l : l in L)_{l,m} = u_l^* u_m` and `E_k` is the `k`-th elementary symmetric function
> of the eigenvalues.

*Proof.* Let `U = [u_1 ... u_N]` (a `D x N` matrix) and `Delta = diag(d)`, so `M' = U Delta U^*`.
Sylvester's determinant identity gives
`det(I_D + z U Delta U^*) = det(I_N + z Delta U^* U)`. Expanding the right-hand determinant by
principal minors, `det(I_N + zX) = sum_L z^{|L|} det(X_{L,L})`, and
`(Delta U^* U)_{L,L} = diag(d_L) (U^* U)_{L,L}`, so
`det((Delta U^*U)_{L,L}) = (prod_{l in L} d_l) det((U^*U)_{L,L})`, and `(U^*U)_{L,L}` is the Gram
matrix of `(u_l)_{l in L}`. Matching powers of `z` gives the claim. ∎

> **LEMMA 3.** If the `u_l` are unit vectors then `0 <= G_L := det Gram(u_l : l in L) <= 1` for
> every `L`, with `G_L = 1` iff `(u_l)_{l in L}` is orthonormal and `G_L = 0` iff it is linearly
> dependent.

*Proof.* A Gram matrix is positive semi-definite, hence has non-negative determinant, and
Hadamard's inequality for positive semi-definite matrices bounds the determinant by the product of
the diagonal entries, here all `1`. ∎

### 2.3 The elementary-symmetric inequalities

> **PROPOSITION 4.** Under `(KL)` (indeed under its interval sub-system alone), with
> `d_l := D c_l / A` and `x_l := |d_l|`,
>
>     e_k(x)  >=  binom(D, k)        for every k = 0, 1, ..., D .
>
> In particular `e_D(x) >= 1` forces at least `D = n+1` non-zero atoms (Corollary 1.1), and
> `e_1(x) >= D` is `sum_l |c_l| >= |A|`.

*Proof.* By Lemma 1, `M/A = sum_l d_l u_l u_l^*` with `u_l = m_l/sqrt D` and `M/A` upper triangular
with every diagonal entry `1`. A triangular matrix's eigenvalues are its diagonal entries, so all
`D` eigenvalues of `M/A` equal `1` and `E_k(M/A) = binom(D,k)`. By Lemma 2 and Lemma 3,

    binom(D,k) = | sum_{|L|=k} (prod_{l in L} d_l) G_L |
               <= sum_{|L|=k} (prod_{l in L} |d_l|) G_L
               <= sum_{|L|=k} prod_{l in L} |d_l|  =  e_k(x) .  ∎

### 2.4 The generating inequality

> **PROPOSITION 5.** `prod_{l=1}^N (1 + x_l t) >= (1 + t)^D` for every `t >= 0`.

*Proof.* `prod_l (1 + x_l t) = sum_{k=0}^N e_k(x) t^k`. All `e_k(x) >= 0`, and `e_k(x) >= binom(D,k)`
for `k <= D` by Proposition 4 (`e_0 = 1 = binom(D,0)`); note `N >= D` automatically, since otherwise
`e_D(x) = 0 < 1`. Hence
`sum_{k=0}^N e_k(x) t^k >= sum_{k=0}^D binom(D,k) t^k = (1+t)^D`. ∎

### 2.5 Lemma 4 (Bernstein integral) and the conclusion

> **LEMMA 4.** For `0 < r < 1` and `x >= 0`,
>
>     x^r  =  (r sin(pi r) / pi) * int_0^inf log(1 + x t) t^{-1-r} dt ,
>
> the integral converging absolutely at both ends.

*Proof.* Near `t = 0` the integrand is `O(t^{-r})` and near `t = inf` it is `O(t^{-1-r} log t)`, both
integrable for `0 < r < 1`. Integration by parts, then `u = xt`:

    int_0^inf log(1+xt) t^{-1-r} dt = (1/r) int_0^inf x t^{-r} / (1 + x t) dt
                                    = (x^r / r) int_0^inf u^{-r} / (1 + u) du
                                    = (x^r / r) * pi / sin(pi r) ,

using the standard Beta-function value `int_0^inf u^{a-1}/(1+u) du = pi / sin(pi a)` with
`a = 1 - r`. ∎

> **PROOF OF THEOREM 1.** For `r = 1` take `k = 1` in Proposition 4. For `0 < r < 1`, apply Lemma 4
> to each `x_l` and integrate Proposition 5's logarithm against the positive measure
> `t^{-1-r} dt`:
>
>     sum_l x_l^r = (r sin(pi r)/pi) int_0^inf [ sum_l log(1 + x_l t) ] t^{-1-r} dt
>                >= (r sin(pi r)/pi) int_0^inf D log(1 + t) t^{-1-r} dt
>                =  D * 1^r  =  D = n+1 .
>
> Unwinding `x_l = (n+1)|c_l|/|A|` turns `sum_l x_l^r >= n+1` into
> `sum_l |c_l|^r >= (n+1)^{1-r}|A|^r`. ∎

**Equality (for `0 < r < 1`; at `r = 1` the equality class is every aligned configuration).** The
integrand `sum_l log(1+x_l t) - D log(1+t)` is continuous and non-negative by Proposition 5, and its
integral against the strictly positive density `t^{-1-r}` vanishes, so equality forces
`sum_l log(1+x_l t) = D log(1+t)` for every `t > 0`, hence
(both sides being continuous, and the exponentials being polynomials in `t`)
`prod_l (1 + x_l t) = (1+t)^D` identically. By unique factorisation the non-zero `x_l` are exactly
`D` copies of `1`, i.e. exactly `n+1` atoms with `|c_l| = |A|/(n+1)`. Then `e_D(x) = 1` and
Proposition 4 is tight at `k = D`, which forces `G_L = 1` for the support `L`: **the `n+1` chain
vectors are orthogonal**. Conversely any such configuration gives equality. Adding back the
non-interval equations of `(KL)` pins the orthogonal frame down further, to the common-phase
`(n+1)`-block up to gauge (`LEMMA_ATTEMPT` §4 rigidity, unchanged).

*Two atoms' chain vectors are orthogonal iff `sum_{k=0}^n prod_{j<=k} (v_l'j conj(v_lj)) = 0` — and
when the ratio `v_l'j conj(v_lj) = zeta` is the same in every slot this says `1 + zeta + ... +
zeta^n = 0`, i.e. `zeta` is a non-trivial `(n+1)`-st root of unity. That is the old "Fekete points /
roots of unity" statement, now read as orthogonality rather than as a Vandermonde maximum.*

### 2.6 What the theorem actually needs

Only the interval patterns. Stated for the record:

> **THEOREM 1'.** The conclusion of Theorem 1 holds if merely
> `sum_l c_l prod_{j=s}^{t} v_lj = 0` for every `1 <= s <= t <= n` (`binom(n+1,2)` equations)
> and `sum_l c_l = A`.

Verified directly (`eleg_chain.py`, sampler `interval`; `eleg_bigN.py`, hypothesis `interval`):
systems satisfying only those constraints, `n = 1..6`, `N` up to `44`, minimum ratio to the target
`1.0342` (`n=2, N=4`) and never below `1`.

### 2.7 Corollary 1.4: the aligned case is a tight frame

*Proof of Corollary 1.4.* Rotate globally so that every `c_l >= 0` and `A = sum_l c_l > 0`. Then
`M = sum_l c_l m_l m_l^*` is a non-negative combination of Hermitian positive semi-definite
rank-ones, hence Hermitian. By Lemma 1 its strictly lower triangle vanishes; a Hermitian matrix with
vanishing lower triangle is diagonal, so `M = A I`. Evaluating at `m_{l_0}`,

    A * (n+1) = A ||m_{l_0}||^2 = m_{l_0}^* M m_{l_0} = sum_l c_l |<m_{l_0}, m_l>|^2
              >= c_{l_0} * (n+1)^2 ,

because every term is non-negative and the `l = l_0` term is `c_{l_0} (n+1)^2`. Hence
`c_{l_0} <= A/(n+1)`. ∎

*The same bound, in dual form (the test polynomial one would otherwise have to guess).* The
inequality above is the pairing of the rule against the **Fejér kernel along the chain**: for
`w in T^n`,

    Lambda_w(z) := 1 + (2/(n+1)) * sum over intervals 1 <= s <= t <= n of
                                    prod_{j=s}^{t} z_j conj(w_j)
    satisfies    Re Lambda_w(z) = (1/(n+1)) | sum_{k=0}^{n} prod_{j<=k} z_j conj(w_j) |^2  >= 0 ,

with Haar mean `1` (its constant coefficient) and `Re Lambda_w(w) = n+1`. It is multi-affine
because every monomial is a product of *distinct* variables — the intervals of a chain, which is
exactly why the chain map is the right one. Pairing with the rule gives
`1 = sum_l p_l Re Lambda_w(v_l) >= p_{l_0} (n+1)`.

At `n = 2` this is `Lambda = 1 + (2/3)(z_1 + z_2 + z_1 z_2)`, whose non-negativity is the classical
`cos a + cos b + cos(a+b) >= -3/2`; `P1B` §7 obtained the same constant numerically as
`m_2 = -1/2, A_2 = 1/3` but its kernel `prod_j (1+z_j)` gives only `A_3 = 3/10` at `n = 3`, whereas
the chain Fejér kernel gives `1/(n+1)` at every `n`.

**Checked** (`eleg_aligned.py`): the identity `Re Lambda_w = |chain sum|^2/(n+1)` to `6.2e-15` and
`min Re Lambda >= 0` over `2 x 10^5` torus points at `n = 1..6`; `M = A I` to `6.3e-13` over 800
LP-extremal probability cubature rules (`n = 1..5`, up to `30` nodes); `(n+1) max_l p_l = 1.000000000`
— attained, never exceeded — and a direct LP with one node pinned at `(1,...,1)` returns exactly
`1/2, 1/3, 1/4, 1/5` at `n = 1, 2, 3, 4`.

*Why this does not extend to the general case:* without the common phase `M` is not Hermitian and
its strictly upper part is genuinely non-zero (§5.2), so only triangularity survives — which is why
Theorem 1 needs the determinantal route rather than this one.

### 2.8 Why Theorem 1' (the interval sub-system) is worth stating

The interval sub-system is what a proof has to control, so a junk
bound at the network level only needs the `n(n+1)/2` interval junk coefficients, not all `2^n - 1`
of them — a fourfold reduction already at `n = 4` (`10` instead of `15`) and an exponential one
after that. It also shows the theorem is not about the Boolean cube of sign patterns but about a
chain (flag) of nested products. Any of the `n!` slot orderings gives another valid chain, hence
another triangular certificate; the proof uses only one.

---

## 3. Theorem 2: the assembly, with Theorem 1 as a black box

This is deliverable (b): the cleanest statement and proof of "no same-`|k|` junk ⇒ (T) exact". The
two propositions below are `GLOBAL_OPT3` §2 Steps 3 and 4, restated; nothing in them is new, and
both are already checked against brute-force logit tables to `2.7e-15` (`gopt3_chain.py` V2, V3).

**Setting.** `p` an odd prime; the task is `a_1 + ... + a_n = c (mod p)` on all `p^n` rows. A
gated-chain network of width `W` has, for each neuron `i`, embeddings `e_ij in R^p` (`j = 1..n`) and
a readout `w_i in R^p`, and computes `z(a,c) = sum_i w_i(c) prod_j e_ij(a_j)`. Write
`ehat_ij(k) = sum_a e_ij(a) omega^{-ka}` (`omega = e^{2 pi i/p}`) and likewise `what_i(k)`.

**Atoms (no purity needed).** For each frequency `k != 0` and neuron `i` put

    c_ik := ( prod_j ehat_ij(k) ) * conj( what_i(k) ) ,
    v_ijk := conj(ehat_ij(k)) / ehat_ij(k)      (unit modulus; if ehat_ij(k) = 0 then c_ik = 0)

— so each neuron contributes **one atom per frequency**, whatever else it is doing. The clean
Fourier coefficient of the feature at `k` is `G_k = sum_i c_ik`, and the logit tensor's coefficient
at the multi-index that flips the operands in `T` and takes class frequency `-k` is exactly
`J_{k,T} = sum_i c_ik prod_{j in T} v_ijk`. So hypothesis `(H)` — *the same-`|k|` junk vanishes* —
says precisely that, at each frequency, the atoms form a `(KL)` system with `A = G_k`.

> **PROP 6 (Level-set Bound; needs the FULL table).** For every gated-chain network on the full
> `p^n`-row table and every `D != 0`,
> `margin(theta) <= mean over rows a of [ z(a, s(a)) - z(a, s(a) - D) ]`, and averaging that bound
> over `D` and `-D` gives
>
>     margin(theta)  <=  2 p^{-nu} * min_{D != 0} sum_{k > 0} |G_k| ( 1 - cos(2 pi k D / p) ) .
>
> *Plain reading: the worst row is no better than the average row on any level set, and only the
> clean spectrum survives the average.* (`GLOBAL_OPT3` §2 Step 3; equality at the block.) The first
> step is an orbit average over the translations
> `(a, c) -> (a + t, c + sum_j t_j)`, which act transitively on each level set **and stay inside the
> table only if the table is complete** — so Theorem 2 is a full-table statement and must say so
> (`PROOF_AUDIT` F6, `VERIFY.md` link 1).

> **PROP 7 (Neuron Budget).** With `m_ik := sum_j |ehat_ij(k)|^2 + |what_i(k)|^2`, for every neuron
> and every `q in [2, nu]`,
>
>     sum_{k > 0} |c_ik|^{q/nu}  <=  ( p / (2 nu) )^{q/2} * ||theta_i||^q .
>
> *Plain reading: one neuron of a given weight norm can only create so much clean amplitude, spread
> over however many frequencies it likes.* (Per-neuron AM-GM over the `nu` factors, then `q/nu <= 1
> <= q/2`, then Parseval. `GLOBAL_OPT3` §2 Step 4; equality for a balanced single-frequency neuron.)

> **PROOF OF THEOREM 2.** Put `r = q/nu` and `s = 1/r = nu/q`, and write
> `||(G_k)||_r := (sum_{k>0} |G_k|^r)^{1/r}`.
> By `(H)` and Theorem 1, at each frequency `sum_i |c_ik|^r >= nu^{1-r} |G_k|^r`. Summing over `k`
> and using Prop 7 neuron by neuron,
>
>     nu^{1-r} sum_{k>0} |G_k|^r  <=  sum_i sum_{k>0} |c_ik|^r  <=  (p/(2 nu))^{q/2} P_q(theta) ,
>
> i.e. `|| (G_k) ||_r^r <= nu^{r-1} (p/(2nu))^{q/2} P_q`. Prop 6 and the definition
> `Gamma_s(p) = max { min_{D != 0} sum_k y_k (1 - cos(2 pi k D/p)) : y >= 0, ||y||_{1/s} <= 1 }`
> give `margin <= 2 p^{-nu} Gamma_s * || (G_k) ||_r`. Combining, and using the identity
> `2 p^{-nu} (p/(2 nu))^{nu/2} = 2 (2 p nu)^{-nu/2} = B`,
>
>     gamma_q = margin / P_q^{nu/q}  <=  nu^{(r-1)/r} B Gamma_s  =  nu^{-(nu-q)/q} B Gamma_{nu/q}(p) . ∎

> **PROP 8 (Attainment).** Let `(b_k)` be a maximiser of `Gamma_{nu/q}(p)` and build, for each `k`
> with `b_k > 0`, the `(n+1)`-neuron common-phase block at frequency `k` (neuron `l = 0..n`:
> operand phases `pi l/(n+1)`, readout phase `n pi l/(n+1)`), scaled by `b_k^{1/q}`. The resulting
> network satisfies `(H)` exactly and attains the bound of Theorem 2.
> *(`ADVERSARY` §5 correction: the **share-weighted** scaling `b_k^{1/q}` is essential — an
> equal-amplitude block attains the bound only when the optimal shares are equal.)*

> **COROLLARY 2.1.** Let `q < nu` (so `r < 1`). Any network in the `(H)` class that **attains** the
> bound of Theorem 2 carries, at each frequency it uses, exactly `n+1` non-zero atoms of equal
> modulus with orthogonal chain vectors (Theorem 1's equality case), and its set of frequencies is a
> maximising support of `Gamma_{nu/q}(p)`; in particular it has exactly `K*(nu/q)` features, where
> `K*` is the number of non-zero shares at that optimum.
> *(Stated for `q < nu` on purpose: at `q = nu` the cost is `r = 1`-linear, every phase-aligned
> configuration ties, and the neuron count per feature is not pinned.)*

### 3.1 What this does to the hostile referee's repair list (`VERIFY.md` R1–R4)

The verifier audited the *old* chain and found it sound but in need of four repairs. Three of them
disappear rather than being repaired, because the objects they concern are no longer used; the
fourth is superseded. Stated one by one so the paper can drop the corresponding paragraphs.

| ref | the issue with the old chain | status under Theorem 1 |
|---|---|---|
| **R1** (provenance/scope) | Theorem B bounds `R = min over the CONTINUOUS torus of Re h`, whereas the margin is a min over the grid `mu_p^n`; so `ADVERSARY` §6's P1 inherits a continuum quantity and needs Theorem C's `1 + O(p^{-2})` bracket (`1.42` at `n=2,p=11`, `19.6` at `n=4,p=11`) | **gone.** The chain proof never uses Theorem B, Theorem C, or any minimisation over the torus. Every object in §2 is a finite sum over the `N` atoms or a determinant of an `(n+1) x (n+1)` matrix. The only continuum object anywhere in the proof is the one-dimensional integral of Lemma 4, which is a *representation of `x^r`* and has nothing to do with `p` or with the network. The verifier's prescription ("route through Theorem D + a finite-`p` P1 only") is exactly what §3 does. |
| **R2** (`L_r` is not `N`-independent; the repair is `N = n+2` via `ceil(delta/x_{n+1})` and `(PEEL_n)`) | `P1B` §5 asserted `L_r(n, delta)` independent of `N` for `N >= n+2`; false, `L_r(2,0.3)` drops from `1.79256` at `N=4` to `1.68046` from `N=5` | **moot.** `L_r` is retired: Theorem 1 gives the constant `(n+1)^{1-r}` directly, uniformly in `N`, with no vertex programme to be `N`-stable about. The verifier's repair is correct and should still be recorded in the "prior attempts" note, not in the proof. |
| **R3** (equality case for `N > n+1` nowhere stated) | the `K*` conclusion needs it | **supplied, and for every `N` at once.** §2.5: equality in the Bernstein step forces `prod_l (1 + x_l t) = (1+t)^{n+1}` identically, so by unique factorisation exactly `n+1` atoms survive and all have modulus `|A|/(n+1)`; then `E_{n+1}` is tight and the chain frame is orthonormal. No branch-1/branch-2 strictness argument, no `eps0`, and no non-degeneracy hypothesis. This also removes the verifier's cosmetic complaint that the Key Lemma's equality case was proved only at `n = 2` and had an unstated non-degeneracy step. |
| **R4** (no interval certificate at `nu = 5`; `verify_p1_rigorous.py` now certifies `nu = 3,4,5` at `q = 2` by closed-form branch 1 + exact-rational vertex enumeration) | the certificate was the load-bearing numerical ingredient | **superseded.** There is no certificate, no branch-and-bound and no case split: Theorem 1 holds at every `nu` and every `q in [2, nu]`. `verify_p1_rigorous.py` remains the best *independent* confirmation of the old route and should be cited as such, but the paper no longer needs it. Likewise the unadvertised dependency on Theorem E's top rung `h = n`, and the constants `kappa_1 = 1/2`, `kappa_2 = 3 sqrt 3/8`, `kappa_3 = 49/72`, leave the proof entirely. |

Two of the verifier's other points survive untouched and must stay in the paper, because they are
about the *network* half of the argument and not about P1:

* **the full table is load-bearing** (link 1): the orbit-averaging step of Prop 6 needs every row of
  the `p^n`-row table, so Theorem 2 is a full-table statement and must say so;
* **R7** (the paper's `K*` and the theorem's `K*` must be the same object): unchanged — Corollary
  2.1 defines `K*` as the number of non-zero shares at the optimum of `Gamma_{nu/q}(p)`, and that is
  the object the empirical sections must also mean.

**The factor `F` is gone.** `GLOBAL_OPT3`'s Theorem D had an extra factor
`F(n,q) = ((n+1)^{1-r}/L_r(n))^{1/r}` (`1.000905 / 1.030841 / 1.101809 / 1.201816` at
`nu = 3/4/5/6`, `q = 2`), because the peeling ladder only reached `L_r(n) < (n+1)^{1-r}`. Theorem 1
gives `L_r(n) = (n+1)^{1-r}` exactly, so `F = 1` in every cell and at every `q`, and Theorem 2 is
an equality statement rather than a bracket.

---

## 4. Naming and consistency (deliverable (c))

### 4.1 Clashes and misattributions in the inherited material

| issue | where | fix |
|---|---|---|
| **two different "Lemma T"** | `P1A` §3.3 (the staircase tail price `k beta^r + (tau - k beta)^r`) vs `P1C` §1/§2.1 (the Key Lemma with a tail, `sum_{l in S}\|c_l\|^r >= (n+1)^{1-r}(\|A\| - 2^n R)_+^r`) | both retired by Theorem 1; if either is quoted, call them **Staircase Bound** (P1A) and **Robust Key Lemma** (P1C) |
| **"`2^n` is attained" misattributed** | `P1B` §11.3 credits the statement to `P1C`; it is `P1A` §6.1's Observation (free probe `u_j = conj(v_j)` on a single tail atom). `P1C` §9 item 2 says the opposite for the *bijection* probes, and `P1B` §11.3's own reconciliation is correct | credit `P1A` §6.1; the two statements are about different probes and both are right |
| **`D_h(n)` defined with `sum d_i = n`** | `GLOBAL_OPT3` §3 | corrected in `P1B` §5 to `sum d_i <= n`; retired with Theorem E |
| **"`build_chain` attains the target"** | `fvf_obj.py` / early notes | false for unequal optimal shares; the equality case is the **share-weighted** block (`ADVERSARY` §5) — carried into Prop 8 above |
| **`Gamma_{3/2}(23)`** | numerical constant | `0.402933388` (`ADVERSARY`) |
| **`L_r(n, delta)` claimed `N`-independent** | `P1B` §5 | false (`VERIFY.md` R2); retired with Theorem E, so the repair only needs recording in a prior-attempts note |
| **scope: the full table** | `THEORY_nsum` L1 / `GLOBAL_OPT3` Step 3 / `PROOF_AUDIT` F6 | the orbit-averaging step needs all `p^n` rows; **Theorem 2 must say "full table" in its statement** (`VERIFY.md` link 1) |
| **two meanings of `K*`** | theorem vs empirical sections | `VERIFY.md` R7: fix `K*` once, as the number of non-zero shares at the optimum of `Gamma_{nu/q}(p)`, and use only that |

### 4.2 One coherent scheme for the paper

* **Objects.** `nu = n+1` (never write `n+1` in a theorem statement — write `nu`, and say once that
  `nu` is both the homogeneity degree and the dimension of the chain space; the whole point is that
  they are the same number). `(JF)` for the junk-free system (retire `(KL)`, `(dagger)`, `(S1)`,
  `(P1)`, `(T)` as separate names). `Gamma_s(p)`, `K*(s)`, `B` unchanged.
* **Results.** Lemma 1 Chain, Lemma 2 Minor Expansion, Lemma 3 Hadamard, Lemma 4 Bernstein,
  Prop 4 Symmetric Bounds, Prop 5 Generating Inequality, **Theorem 1 Feature Cost**,
  Cor 1.1 Atom Count, Cor 1.2 Orthonormal Frame, Cor 1.3 Bernstein penalties, Cor 1.4 Tight Frame;
  Prop 6 Level-set Bound, Prop 7 Neuron Budget, **Theorem 2 Margin Ceiling**, Prop 8 Attainment,
  Cor 2.1 Feature Count. (Theorem 1' — the interval sub-system — as a remark under Theorem 1.)
* **Retire entirely** (move to a "prior attempts" paragraph or drop): Theorems A, B, C, D, E, H, J,
  O, P, Q, R, S; Lemmas C3, C4, P, T(×2), C, R, R', R\*, R''; Propositions A, A1, A2, B1, B2, D1, L,
  S; constants `kappa_d`, `D_h`, `L_r`, `F`, `eps0`, `eps1`, `r*`, `a*`, `rho`, `C(n,r)`,
  `tau_hi`, `eps_cert`; programmes `G(beta,tau)`, `Val(R,s)`, `Phi(E)`, `Psi(m)`, `(E_rho)`, `(E0)`.
  None of them is needed once Theorem 1 is available. (`(E0)` and `P1B` Prop A1 are not merely
  retired — they are now *theorems*, Corollary 1.4; they leave the proof because nothing depends on
  them, not because they are unproved.)
* **Keep, unchanged, for the junk gap** (still open): `Lemma Q` (rename **Ripple Bound**),
  `Theorem J` / the junk-cost curve `Cl2_r(eta)` (rename **Junk Relief**), and the chord criterion
  of `P3A` (rename **Chord Criterion**, and record that it is refuted at `nu = 4`).

---

## 5. Routes that failed, with exact breaking points

### 5.1 Log-convexity in `r` of the optimal value — a red herring, now settled

The brief asked to check numerically whether the optimal value `V(r) := min { sum_l |c_l|^r }` is
log-convex in `r`, with a view to interpolating between the proven endpoints `V(0^+) = n+1` and
`V(1) = 1`. Two things are now clear and should stop anyone spending time on it.

* **Convexity is the wrong direction.** For a *fixed* configuration, `r |-> log sum_l x_l^r` is
  convex (Hölder), so it lies *below* the chord through its endpoints — an upper bound, whereas P1
  needs a lower bound. What would have sufficed is log-**concavity** of `V`, which combined with
  `V(0^+) >= n+1` and `V(1) >= 1` gives `V(r) >= (n+1)^{1-r}`.
* **It is exactly affine, so the route was never going to be easier than the theorem.** Theorem 1
  plus the block shows `V(r) = (n+1)^{1-r}` identically, i.e. `log V(r) = (1-r) log(n+1)` is a
  straight line. So "log-concavity of `V`" is *equivalent* to P1, not a lever on it. Measured as a
  cross-check (`eleg_routes.py` block `logconv`, six exponents): at `n = 1` the numerical `V(r)`
  matches `(n+1)^{1-r}` to `1e-10` at every `r` and the second differences of `log V` are `0` to
  `1.9e-14` — affine, as the theorem says; at `n = 2` the multistart returns `1.0000 .. 1.0004` of
  the target (an optimiser's upper bound on a minimum).

What *did* work is an interpolation of a completely different kind: not of the value in `r`, but of
the **constraints** in the degree `k` of the symmetric function (Prop 4), with the two known
endpoints appearing as `k = 1` and `k = nu`. The `r`-dependence is then bought in one step by the
Bernstein integral, which is why the proof is uniform in `r`.

### 5.2 `M = A I` — the natural stronger claim, and why it is false

The elegant statement one wants is "a junk-free feature is a **tight frame**":
`sum_l c_l m_l m_l^* = A I`. That would give the max-weight bound `|c_l| <= |A|/(n+1)` in one line
(apply the identity to `m_l` and use positivity), and hence P1 for all `r` on the aligned class.
**It is false.** The strictly-upper entries of `M` are `sum_l c_l prod_{j in T} conj(v_lj)` —
the phases are conjugated but the amplitudes are not, so they are not `(KL)` equations and not
conjugates of any. Witness at `n = 1`, `N = 2`: `v = (1, i)` forces `c = ((1-i)/2, (1+i)/2)` and
`M = [[1, 1-i], [0, 1]]` — upper entry `1 - i`, not `0`. Measured over 540 random systems
(`eleg_routes.py` block `mirror`, `n = 1,2,3`): the strictly-lower part of `M/A` is `0` to
`4.40e-14`, while the strictly-upper part reaches `0.9998` of `sum_l |c_l| / |A|` — i.e. as large as
it could possibly be.

*Why this does not matter, and where it becomes true.* Triangularity already pins the whole
spectrum, which is all Proposition 4 needs — so nothing is lost. And the tight-frame statement *is*
true exactly when the amplitudes are aligned, because then `M` is Hermitian and a Hermitian matrix
with a vanishing lower triangle is diagonal (Corollary 1.4, §2.7). So the "false" route is not dead,
it is the aligned case, and it is what proves `max_l |c_l| <= |A|/(n+1)` there. What the asymmetry
does block is the junk extension (§6).

### 5.3 The `e_k` relaxation alone: is it enough?

Proposition 4 discards the phases entirely and keeps only `e_k(x) >= binom(D,k)`. One should ask
whether that relaxation is already too lossy. It is not — and that is the content of Propositions 5
and Lemma 4, which prove

    min { sum_l x_l^r : x >= 0, e_k(x) >= binom(D,k) for k = 1..D }  =  D ,

attained at `x = (1,...,1)` (`D` ones). Confirmed adversarially by SLSQP with 60 restarts per cell
over `D = 2..6`, `r in {1/3, 1/2, 2/3, 0.9}`, `N in {D, D+1, D+3, 2D}` (`eleg_chain.py`, 80 cells):
the ratio to `D` lies in `[0.9999999999999871, 1.0000034890925538]` — never below the target — and
the argmin is always `D` atoms of size `1` with the rest exactly `0`.

**But the family cannot be trimmed at either end.** Keeping only the top constraint `e_D(x) >= 1`
(which is what a "product of the `n+1` largest atoms" argument would use — `P1A` F5-1, `P1B` §8.1)
allows `N` equal atoms `x_l = binom(N,D)^{-1/D}`, value `N binom(N,D)^{-r/D}`; at `D = 3`, `r = 2/3`,
`N = 4` that is `4^{7/9} = 2.9395 < 3`. Keeping only the trace constraint `e_1(x) >= D` allows one
atom of size `D`, value `D^r`. Measured (`eleg_routes.py` block `partial`, SLSQP, 90 restarts,
`N = 2D`):

| `D` | `r` | all `k = 1..D` | only `e_D >= 1` | only `e_1 >= D` |
|---|---|---|---|---|
| 3 | 1/2 | `3.000000005` | `3.000000005` | `1.732050823` |
| 3 | 2/3 | `3.000000000` | **`2.939468985`** | `2.080083823` |
| 4 | 1/2 | `4.000000024` | `4.000000023` | `2.000000017` |
| 4 | 2/3 | `4.000000000` | **`3.820639317`** | `2.519842100` |
| 5 | 1/2 | `5.000000013` | `5.015752829` | `2.236068012` |
| 5 | 2/3 | `5.000000000` | **`4.664468172`** | `2.924017738` |

**The middle degrees are load-bearing**, and so is the exponent: the top constraint alone suffices at
`r = 1/2` but not at `r = 2/3`. That is the precise sense in which P1 is neither an `r = 1` statement
nor an `r -> 0` statement — it is the whole ladder of degrees, read through one measure in `r`.

### 5.4 The trap on this route, recorded

The one thing that can go wrong when reproducing this proof is asserting `M = A I` (§5.2) — it is
tempting because the diagonal and the lower triangle are both forced, and because `M` is a sum of
Hermitian rank-ones (with complex coefficients, so `M` itself need not be Hermitian). The first
draft of this note asserted it. Everything downstream is unaffected, because Lemma 2's expansion
never uses Hermiticity and triangularity already fixes the spectrum.

### 5.5 Previously recorded dead routes

Unchanged and not re-attempted: Morwani-style dual certificates; the domination lemma; the
capped-mass/majorisation conjecture; gauge symmetrisation and grid rotation; Mahler measure;
Gram / Carathéodory–Fejér; induction on `n` through the last slot; Carathéodory support counting;
one averaged test measure (`GLOBAL_OPT4` Theorem O); product test measures; the diagonal relaxation;
mixtures of `(n+1)`-atom solutions; equal-moduli rigidity; the max-weight route into the `r -> 1`
corner (`P1B` §11.2, the `sqrt(delta)` is real). Note that **Theorem O does not apply** to the chain
proof: Theorem O forbids certifying P1 by a single averaged multi-affine test polynomial on the
torus, whereas Lemma 2 uses `binom(N,k)` determinantal functionals per degree and `D` degrees, tight
at different subsets of atoms — the non-uniform shape Theorem O explicitly leaves open.

---

## 6. What the chain identity does *not* give: the junk gap (P2/P3)

It is worth being precise, because the machinery looks as if it should extend and it does not, yet.

With junk present, Lemma 1's computation gives `M = A I + L + U` where `L` (strictly lower) has
entries `J_{k,T}` for the interval patterns — the junk we want to charge for — and `U` (strictly
upper) has the **mirror** entries of §5.2, which `(KL)` never controlled and which are therefore
*equally uncontrolled in the junk case*. In the exact case `L = 0` makes `M` triangular and `U`
irrelevant; with `L != 0` the spectrum of `A I + L + U` is not a small perturbation of `A`, because
the matrix is strongly non-normal: a tiny `L` against a large `U` moves eigenvalues a lot.

There is one concrete opening, and the shape of the argument is already clear. Write
`M/A = I + L' + U'` with `L' = L/A`, `U' = U/A`. Two facts combine:

* `E_k(I + U') = binom(D,k)` **exactly**, because `I + U'` is triangular with unit diagonal. So in
  the multilinear expansion of `E_k(I + L' + U')` every term that does *not* contain a factor from
  `L'` sums to `binom(D,k)`, and the remainder is a finite sum of terms each carrying at least one
  `L'`. Hence `|E_k(M/A) - binom(D,k)| <= C_k(||U'||) ||L'||` with `C_k` a polynomial of degree
  `k-1`, explicit and computable.
* `||U'||` is controlled in the only regime that matters: every entry of `U` is bounded by
  `sum_l |c_l|`, and we may assume `sum_l |c_l| <= Sigma` for an explicit `Sigma`, because otherwise
  the trivial branch `sum_l |c_l|^r >= (sum_l |c_l|)^r` already finishes.

That gives `e_k(x) >= binom(D,k) - eps_k` with `eps_k` explicit in the junk size, and then
Propositions 4 and 5 and Lemma 4 go through unchanged (Proposition 5 becomes
`prod_l (1 + x_l t) >= sum_{k<=D} (binom(D,k) - eps_k)_+ t^k`, whose logarithm is still integrable
against `t^{-1-r} dt`), producing a **proved** minorant for the junk-cost curve `Cl2_r(eta)` of
**P2** — which at present has only the conjectured quadratic law QPL.

What is *not* settled: whether the resulting constant is good enough. The `Sigma`-dependence is the
worry, since `C_k(||U'||)` grows with the total modulus while the relief being bought is only linear
in the junk. And even a sharp `Cl2_r` does not close **P3**: the chord criterion that would convert
it into a junk threshold is refuted at `nu = 4` (`GOPT_STATE2`, `P3A`). **This is the natural next
piece of work and it is not done here.**

---

## 7. Numerical record

fp64, local CPU, no pods. Scripts `heracles/eleg_*.py`, outputs `heracles/FINAL_RUNS/ELEG/`.

Reproduce:

    uv run --with scipy python eleg_chain.py      # the five steps + the e_k relaxation
    uv run --with scipy python eleg_bigN.py       # large n and large N
    uv run --with scipy python eleg_aligned.py    # Corollary 1.4: tight frame, Fejer kernel, LP
    uv run --with scipy python eleg_routes.py     # the settled routes of §5

### 7.1 `eleg_chain.py` — the five steps

Four samplers of `(KL)` systems (`full`: all `2^n` constraints; `interval`: only the interval
sub-system of Theorem 1'; `pertblock`: perturbed block phases, feasible below `N = 2^n`;
`nearcancel`: a large null-space component added, the family that refutes the domination lemma),
`n = 1..4`, `N = n+1 .. 12`, 40 systems per cell (15 at `n = 4`), null-space jitter on every solve.

| quantity checked | worst value over all cells |
|---|---|
| `max` modulus of the strictly lower triangle of `M` (Lemma 1: must be `0`) | `2.4e-13` |
| `max \|diag(M) - A\|` (Lemma 1) | `8.7e-14` |
| `\|E_k(M/A) - binom(D,k)\|` via the **brute-force Cauchy–Binet sum** over all `binom(N,k)` subsets (Lemma 2 + Lemma 1) | `1.0e-06` (worst on `nearcancel`, where `\|c\|` reaches `10^2`; `< 2.1e-12` elsewhere) |
| `min_k e_k(x)/binom(D,k)` (Prop 4: must be `>= 1`) | `1.000059` |
| `min_t [ sum_l log(1+x_l t) - D log(1+t) ]` on a 600-point log grid (Prop 5: must be `>= 0`) | `+1.2e-10` |
| `sum_l \|c_l\|^r / ((n+1)^{1-r}\|A\|^r)` over `r in {0.05, 0.2, 1/3, 1/2, 2/3, 0.75, 0.9, 0.95, 0.99, 1}` (Theorem 1) | `1.000002932` |
| Lemma 4's integral representation, `r in {0.15, 1/3, 1/2, 2/3, 0.8, 0.95}` x `x in {0.01, 0.3, 1, 2.5, 17, 10^3}` | max relative error `6.1e-14` |

(The reported eigenvalue residual `max \|eig(M) - A\|` reaches `4e-2` on `nearcancel` cells; that is
pure conditioning — `M` is defective and strongly non-normal, so `numpy.linalg.eigvals` loses digits.
The mathematically equivalent statement, triangularity, holds to `2.4e-13` in the same cells, and
the `E_k` check done through the Cauchy–Binet sum agrees to `1e-6` there and to `1e-12` elsewhere.)

### 7.2 `eleg_bigN.py` — large `n`, large `N`

`n = 2..6`, both hypotheses (`full`: all `2^n` constraints; `interval`: only the `binom(n+1,2)`
interval ones), `N` from the feasibility threshold up to `44`, 40 systems per cell, 43 cells;
`E_k(M)` read from the characteristic polynomial instead of the `binom(N,k)`-term sum.

| quantity | worst over the 43 cells |
|---|---|
| strictly-lower triangle of `M`, relative to `\|A\|` (Lemma 1) | `1.5e-13`, and `3.3e-12` in the single square cell `n = 5, N = 32` |
| `\|diag(M) - A\| / \|A\|` (Lemma 1) | `7.9e-13` |
| `\|E_k(M) - binom(D,k) A^k\| / \|A\|^k` (Lemma 1 + triangularity) | `8.5e-10` over the 41 cells with a clean triangularity residual |
| `min_k e_k(x)/binom(D,k)` (Prop 4, must be `>= 1`) | `1.026655` (`n = 2`, `N = 4`, full) |
| `min_t` slack in Prop 5 (must be `>= 0`) | `+8.0e-08` |
| `sum_l \|c_l\|^r` ratio to target, min over the ten exponents (Theorem 1) | `1.026654768` |

The `E_k` check degrades at `D = 6` on the square cell `n = 5, N = 32` (relative error `5.7e4`) —
`numpy.poly` on a `6 x 6` defective matrix with entries of size `10^3`. This is a conditioning
artefact of the *check*, not of the proof: `E_k` is determined by triangularity alone, and
triangularity holds there to `3.3e-12`.

### 7.3 `eleg_chain.py` — the `e_k` relaxation, adversarially

`min { sum_l x_l^r : e_k(x) >= binom(D,k), k = 1..D }`, SLSQP, 60 restarts, `D = 2..6`,
`r in {1/3, 1/2, 2/3, 0.9}`, `N in {D, D+1, D+3, 2D}` — **80 cells**. Ratio of the minimum to `D`:
lowest `0.9999999999999871`, highest `1.0000034890925538`, **no cell below the target**; the argmin
is always `D` atoms of size `1` with the rest exactly `0` (e.g. `D = 6, r = 1/3, N = 12`:
`(1.0017, 1.0005, 0.9997, 0.9995, 0.9994, 0.9992, 0, 0, 0, 0, 0, 0)`). Trimming the constraint
family at either end breaks it by an explicit margin (§5.3).

### 7.4 `eleg_aligned.py` — Corollary 1.4 (the tight frame)

| block | what it measures | result |
|---|---|---|
| `kernel` | the chain Fejér kernel `Lambda_w`: the identity `Re Lambda_w(z) = \|sum_k prod_{j<=k} z_j conj(w_j)\|^2/(n+1)` and `min Re Lambda_w` over `2 x 10^5` torus points, `n = 1..6` | identity to `6.2e-15`; `min Re Lambda_w` `>= +1.0e-06` (it touches `0`, as it must: the block's nodes are its zeros); `Lambda_w(w) = n+1` exactly |
| `frame` | `max \|M - A I\|` and `(n+1) max_l p_l` on 800 LP-extremal probability cubature rules built from unions of `1..5` rotated blocks, `n = 1..5`, `N` up to `30` | `\|M - A I\| <= 6.3e-13`; `(n+1) max_l p_l = 1.000000000` in every cell — the bound is attained and never exceeded |
| `lp` | LP maximisation of the weight at a node pinned to `(1,...,1)` | `0.500000000 / 0.333333333 / 0.250000000 / 0.200000000` at `n = 1/2/3/4`, i.e. exactly `1/(n+1)` — reproducing `P1B`'s measured `n <= 3` values and extending them |

### 7.5 `eleg_routes.py` — §5's settled routes

| block | what it measures | result |
|---|---|---|
| `mirror` | the strictly-upper entries of `M` on 540 `(KL)` systems (`n = 1,2,3`), and the `n = 1` witness | strictly-lower triangle `<= 4.40e-14`; strictly-upper triangle up to `93.5` in units of `\|A\|`, and up to `0.9998` of `sum_l\|c_l\|/\|A\|` — so it is as large as it can possibly be; the `n=1, N=2` witness `M = [[1, 1-i], [0, 1]]` exact, `\|M_01\| = sqrt 2` |
| `logconv` | `V(r) = min sum_l\|c_l\|^r` by direct multistart minimisation over phases, `n = 1, 2`, `r in {0.2, 1/3, 1/2, 2/3, 0.8, 0.95}`; second differences of `log V` | `n = 1`: `V(r)/(n+1)^{1-r} = 1.0000000000` at all six exponents, second differences of `log V` `<= 1.9e-14` — **affine**, so `log V` is both log-convex and log-concave and the interpolation route is equivalent to the theorem. `n = 2`: `>= 1` at every exponent (worst `1.00037`, Nelder–Mead returns an upper bound on the min) |
| `partial` | the relaxation with the constraint family trimmed | table in §5.3: `top only` drops to `0.9798 / 0.9552 / 0.9329` of the target at `D = 3/4/5`, `r = 2/3`; `trace only` to `0.577 / 0.500 / 0.447` at `r = 1/2` |
