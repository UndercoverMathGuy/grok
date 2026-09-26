# LADDER — exact feature-count ladders, and what free training does near them

Empirical leg of *Explaining Feature Sparsity Through the Margin–Regularization Trade-off*.
Part 1 is **exact closed-form evaluation of constructed networks, with no training at all**; Part 2 is
**trends from free training**, reported as trends and nothing more.

Scripts `heracles/ladder_*.py` (+ `ladder_pod.sh`), outputs `heracles/FINAL_RUNS/LADDER/`.
Pod: one RTX 4090 (secure, $0.74/h), 7.0 h, **$5.69 total**, deleted.
Nothing here is committed and no existing file was edited.

---

## 0. Symbols — every one, with its plain reading

| symbol | definition | plain-language reading |
|---|---|---|
| `p` | an odd prime | the modulus of the task |
| `n` | number of operands | the task is the **full table** of `a_1 + … + a_n = c (mod p)`: all `p^n` rows |
| `nu = n + 1` | homogeneity degree | multiply every weight by `s` and the logits multiply by `s^nu` |
| `nf = (p-1)/2` | number of positive frequencies | `k = 1 … nf`; `k` and `p-k` are the same real frequency |
| `E_j[i, a]`, `W[c, i]` | operand-`j` embedding and readout of neuron `i` | the only weights; no biases, no extra linear layers |
| **gated chain** | `z(c; a) = Σ_i W[c,i] Π_{j=1..n} E_j[i, a_j]` | each layer multiplies in one more operand's embedding (a rank-`m` CP factorisation of the logit tensor) |
| **x^n family** | `z(c; a) = Σ_i W[c,i] (Σ_j E_j[i, a_j])^n` | one hidden layer with activation `x^n` (Morwani's `x²` at `n = 2`) |
| `q` | penalty exponent, `2 ≤ q ≤ nu` | `P_q(θ) = Σ_i ‖θ_i‖^q`, the sum over **neurons** of the `q`-th power of the neuron's Euclidean weight norm. `q = 2` is plain weight decay; `q = nu` is Morwani's `L_{2,nu}` |
| `margin` | `min` over all `p^n` rows and all wrong classes of `z_correct − z_wrong` | how far the worst row is from being misclassified |
| `gamma_q` | `margin / P_q^{nu/q}` | the **normalized margin**: scale-invariant, because the numerator is degree `nu` in the weights and the denominator is degree `q · nu/q = nu` as well |
| `g_k(D)` | `1 − cos(2 π k D / p)` | how much frequency `k` separates the correct class from the wrong class that sits `D` away from it |
| `s = nu/q` | **programme exponent** | how superlinearly a frequency's budget share buys amplitude |
| `Gamma_s(p)` | `max_{b ≥ 0, Σ b_k = 1} min_{D ≠ 0} Σ_k b_k^s g_k(D)` | the **programme**: spend one unit of `l_q` budget over the frequencies to make the worst wrong class as far away as possible |
| `b_k` | share of the `l_q` budget on frequency `k` | `Σ_k b_k = 1` |
| `K*` | `#{k : b_k > 0}` at the programme's optimum | **the predicted feature count** |
| `val(S, b)` | `min_{D≠0} Σ_{k∈S} b_k^s g_k(D)` | the programme's value at one support and one share vector |
| `M_K(p)` | `max_{|S| = K} min_{D≠0} Σ_{k∈S} g_k(D)` | "coverage" of the best `K`-set; `val` with **equal** shares is `M_K · K^{−s}` |
| `B` | `2 (2 p (n+1))^{−nu/2}` | the constant of the unrestricted upper bound `gamma_q ≤ B · Gamma_{nu/q}(p)` (chain); the `x^n` bound carries an extra factor `n!` |
| `N` | neurons per frequency in the junk-free **block** | chain: `N = n+1`. `x^n`: `N = 6, 18, 54` for `n = 2, 3, 4` (found here, §1) |
| `mu` | clean-coefficient multiplier of the family | `1` (chain), `n!` (`x^n`: only the multilinear monomial of `(Σ_j e_j)^n` reaches the diagonal, and it carries the multinomial `n!`) |
| `junk` | any part of the logit table that is **not** a function of `D = (Σ_j a_j) − c` | it cannot help the margin (junk lemma, THEORY_nsum L1) but it costs norm |
| `lam` | weight-decay coefficient | the training objective is `CE + lam · P_q` |
| `obj*(K)` | `min over the global weight scale of CE + lam·P_q` for the constructed `K`-frequency net | the best training objective that direction can reach; zero-padding to any larger width costs nothing, so it is directly comparable to a trained net of any width |

---

## 1. The two junk-free blocks (the objects everything is evaluated on)

A **block** at frequency `k` is a small population of neurons whose *sum* produces a logit table that is
exactly a clean cosine `A_k cos(2πk((Σ_j a_j) − c)/p)` — every other Fourier term cancels identically.

**Chain (inherited, THEORY_nsum §8).** `N = n+1` neurons `l = 0..n`; operand phase `π l /(n+1)` on every
operand, readout phase `n π l /(n+1)`.

**x^n (new here, `ladder_core.find_xn_block`).** Expand `(Σ_j cos(θ_j + φ_j))^n · cos(θ_c + ψ)`. Each term is
indexed by a monomial `m` (with `Σ_j m_j = n`), a harmonic `c_j ∈ {−m_j, −m_j+2, …, m_j}` per operand, and a
readout sign `δ = ±1`; its dependence on the neuron index `l` is `exp(2πi l u / N)` with

    u = Σ_j c_j α_j + δ ψ,     φ_{l,j} = 2π l α_j / N,  ψ_l = 2π l (Σ_j α_j) / N.

Summing over `l = 0..N−1` kills every term with `u ≢ 0 (mod N)`. The clean term (`m = (1,…,1)`, `c = (1,…,1)`,
`δ = −1`) has `u = 0` by the choice of `ψ`. A brute-force search over `(N, α)` gives the **smallest** cyclic
junk-free block:

| `n` (`nu = n+1`) | chain `N` | `x^n` `N` | `x^n` `α` | `mu` (chain / `x^n`) |
|---|---|---|---|---|
| 2 (`nu` 3) | 3 | **6** | `(1, 4)` | 1 / 2 |
| 3 (`nu` 4) | 4 | **18** | `(1, 3, 12)` | 1 / 6 |
| 4 (`nu` 5) | 5 | **54** | `(1, 3, 9, 36)` | 1 / 24 |

`N = 2·3^{n−1}` for `x^n`; the `n = 2` row reproduces the inherited 6-neuron `build_x2`. (A fallback product
population `l ∈ Z_{n+2}^n` is also junk-free but larger: 9 / 64 / 625.) **This closes THEORY_nsum §8's "not
covered: x^n"**: exactly junk-free `x^n` blocks do exist at `nu = 4, 5`, they are just expensive.

---

## 2. The identity the whole of Part 1 rests on

Put frequency `k`'s block at radius `r_k` (every weight of those `N` neurons is `r_k·cos(…)`). Then every
neuron of frequency `k` has `‖θ_i‖² = (n+1) r_k² p/2`, so that frequency's slice of the penalty is
`N ((n+1)p/2)^{q/2} r_k^q` — **proportional to `r_k^q`**, which is why writing `b_k ∝ r_k^q` makes `b` exactly
the `l_q` budget share. The block's clean amplitude is `A_k = N · mu · 2^{−n} r_k^{nu}`. Doing the algebra:

    gamma_q(block on support S with shares b)  =  mu · N^{−(nu−q)/q} · B · val(S, b),      B = 2(2p(n+1))^{−nu/2}.

Plain reading: *the normalized margin of a constructed net is the programme's value times two constants — one
for the family (`mu`), one for the price of splitting a frequency over `N` neurons to cancel its junk.*
Both constants are **independent of `K`**, so the shape of the `gamma_con(K)` ladder is a property of the
programme alone; the family only sets its height.

At `q = nu` the splitting factor is 1 (splitting is free); at `q = 2` it is `N^{−(nu−2)/2}`:

| | `nu` 3 | `nu` 4 | `nu` 5 |
|---|---|---|---|
| chain `N^{−(nu−2)/2}` | 0.5774 | 0.2500 | 0.08944 |
| `x^n`, `mu·N^{−(nu−2)/2}` | **0.8165** | **0.3333** | 0.06050 |

So at `q = 2` the `x^n` family reaches a **higher** absolute normalized margin than the chain at `nu = 3, 4`
(by `√2` and `4/3`) and a lower one at `nu = 5` — the multinomial `n!` outruns the cost of the bigger block
until `n = 4`. (As a *fraction of each family's own upper bound* the chain is always ahead: `N^{−(nu−q)/q}` is
0.577/0.250/0.089 for the chain vs 0.408/0.056/0.0025 for `x^n`.)

**A second consequence, used throughout:** a junk-free net's whole `p^n × p` logit table depends only on
`D = (Σ_j a_j) − c`. So both the margin and the **cross-entropy** are functions of the `p−1` numbers
`gap(D) = −Σ_k A_k g_k(D)`, and `obj*` can be computed exactly for any `p` and any `n` without ever touching
the `p^n`-row table.

### 2.1 Verification (`ladder_verify.py`, output `FINAL_RUNS/LADDER/verify.json`)

21 cases (both families, `n = 2,3,4`, `p = 11..23`, `q = 2 … nu`), each built as explicit tensors and evaluated
by brute force over the **entire** `p^n × p` table in fp64:

| check | worst relative error over the 21 cases |
|---|---|
| (J) junk: table equals `Σ_k A_k cos(2πkD/p)` | `1.2e-13` |
| (I) identity `gamma_q = mu N^{−(nu−q)/q} B val(S,b)` | `6.7e-16` |
| (M) margin from the `p−1` gap vector vs brute-force margin over the full table | `1.9e-13` |
| (P) `P_q` from the construction vs `P_q` from the tensors | `4.4e-16` |

and, separately, at the **paper primes themselves** (`verify_paperprimes.json`: chain and `x^n` at
`p = 59, 71, 113` for `n = 2`, chain at `p = 59` for `n = 3`, on the actual `K*` supports — the full
`59³ × 59` table is 12 M logits and is enumerated):

| check | worst relative error at `p = 59, 71, 113` |
|---|---|
| (J) junk | `7.3e-14` |
| (I) identity | `7.9e-15` |
| (M) margin | `5.3e-14` |
| brute-force `gamma_q` vs `mu N^{−(nu−q)/q} B Gamma`-prediction | `5.4e-14` |

**ADVERSARY's code correction reproduced.** Equal amplitudes miss the target whenever the programme's optimal
shares are unequal; the share-weighted block hits it exactly:

| cell | `gamma(equal)/target` | `gamma(share-weighted)/target` |
|---|---|---|
| `p 11, n 2, q 2, K 3` | 0.8615 | 1.000000000000 |
| `p 11, n 2, q 2, K 4` | 0.8296 | 1.000000000000 |
| `p 59, n 2, q 2, K 5` | 0.9232 | 1.000000000000 |
| `p 71, n 3, q 2, K 4` | 0.9224 | 1.000000000000 |

(Every number below uses share-weighted blocks.)

---

## 3. Two different `K*`s, and how the programme was actually solved

A referee-level distinction that the verifier (`VERIFY.md` R7) is right to insist on, and that this
write-up keeps separate everywhere:

* **`K*_eq`** — the argmax of the **equal-amplitude ladder** `M_K(p) · K^{−s}`. This is what
  `kstar.py` computes and what the THEORY_nsum tables quote. It answers: *if the `K` frequencies must
  share the budget equally, which `K` is best?*
* **`K*_free`** — the size of the support of the **free-amplitude** programme `Gamma_s(p)`'s
  maximiser. This is the quantity the theorem actually pins, because the upper bound
  `gamma_q ≤ B · Gamma_{nu/q}(p)` is stated with free shares.

They are different objects. Equal shares are a *restriction*, so `M_K K^{−s} ≤ Gamma_s` always; the
two argmaxes can differ, and (per the verifier) do differ at small `p`.

**Definitions used here, with their thresholds stated.**
A share is **active** if `b_k > t_a = 1e-3` (0.1% of the `l_q` budget). A rung **reaches the
optimum** if its value is `≥ (1 − t_v)·Gamma_s` with `t_v = 1e-3`. Then

    K*_free  =  the smallest number of active shares among all rungs that reach the optimum.

This definition is needed because **the free ladder is non-decreasing in `K`**: a `K`-support
contains every smaller support, so rung `K` can always copy rung `K−1`'s solution and set the extra
shares to zero. Its plain `argmax` over `K` is therefore the *last* rung up to search noise and says
nothing. Table T5 reports `K*_free` for `t_a ∈ {1e-2, 1e-3, 1e-4}` and `t_v ∈ {1e-2, 1e-3, 1e-4}` so
the threshold dependence is visible rather than buried.

**How the support was searched (and what is provable).**
1. *Frequencies.* WLOG `1 ∈ S`: multiplying every frequency by `u^{-1} (mod p)` permutes the columns
   `D`, leaving the programme value invariant, and every orbit contains a set containing 1. This is
   exact, not heuristic.
2. *Shortlist.* All `K`-sets containing 1 are enumerated **exhaustively** whenever
   `C(nf−1, K−1) ≤ 3·10^6` — **`K ≤ 8` at `p = 59`, `K ≤ 7` at `p = 71`, `K ≤ 5` at `p = 113`** —
   ranked by the equal-share coverage `M(S)`, and the top 150 are kept. Beyond that, swap local
   search with 200 restarts. So `M_K` is **provably** the best `K`-set in those ranges (which cover
   `K*` in every `q = 2` cell except `p = 113, nu = 3`, where `K*_free = 7 > 5`), and a lower bound
   above them.
3. *Shares.* SLSQP on the epigraph form (`nu4_program.solve`) from 9 starts per support (uniform +
   8 Dirichlet), then the leaders re-solved from 9 more.
4. *Polish and repair* (`ladder_polish.py`): swap local search **on the free-amplitude value** from
   each rung's winner, plus the "active-support repair" — a solution with `k < K` active shares is
   re-scored as a rung-`k` net with its shares renormalised — and the monotone closure in `K`.
5. *Cross-check and certificate* (`ladder_gamma.py`): an algorithmically unrelated solver
   (projected supergradient ascent in `x = b^s`, where the objective is concave and the constraint
   set is a non-convex "sparse ball"), and a **branch-and-bound upper bound** on each reported
   support. The bound comes from the fact that `t ↦ t^{1/s}` is concave, so on a box `l ≤ x ≤ u` it
   lies above its chord and `Σ_k chord_k(x_k) ≤ 1` is a valid *linear* relaxation of the budget
   `Σ_k x_k^{1/s} ≤ 1`; maximising `min_D Σ_k x_k g_k(D)` under it is a linear program. Table T7.

**What is still not certified.** The branch-and-bound bound certifies the *share* problem on a given
support. The *choice* of support is certified only where the enumeration is exhaustive (`K ≤ 5` at
the paper primes); above that, and everywhere on the large-prime axis, `Gamma_s` is a lower bound
from a heuristic search. Since `Gamma_s` is a maximum, a missed support can only *raise* it — which
moves `K*_free` up, never down — so every "`K*` is small" statement is the conservative direction and
every "`K*` is at least this large" statement is safe.

---

## 4. The exact ladders

Full tables: `FINAL_RUNS/LADDER/tables.md` (T1–T7), raw numbers `FINAL_RUNS/LADDER/json/cells.json`,
figures `FINAL_RUNS/LADDER/figs/ladder_{chain,xn}_p{59,71,113}.{html,png}`.
78 cells: 2 families × `nu ∈ {3,4,5}` × `p ∈ {59,71,113}` × the `q` ladder, `K = 1..12` each.

### 4.1 The paper's main setting, plain weight decay `q = 2`

| family | p | `nu` | `s` | **`K*_free`** | `K*_eq` | `Gamma_s(p)` | `gamma_con(K*)` | argmin `obj*` (at the archived `lam`) |
|---|---|---|---|---|---|---|---|---|
| chain | 59 | 3 | 1.500 | **5** | 6 | 0.313049 | 5.427e-05 | 4 (1e-4) |
| chain | 71 | 3 | 1.500 | **5** | 5 | 0.308038 | 4.045e-05 | 5 (1e-4) |
| chain | 113 | 3 | 1.500 | **7** | 5 | 0.275936 | 1.805e-05 | 5 (1e-4) |
| chain | 59 | 4 | 2.000 | **4** | 4 | 0.152806 | 3.430e-07 | 3 (1e-5) |
| chain | 71 | 4 | 2.000 | **4** | 4 | 0.142982 | 2.216e-07 | 4 (1e-5) |
| chain | 113 | 4 | 2.000 | **4** | 4 | 0.123191 | 7.537e-08 | 4 (1e-5) |
| chain | 59 | 5 | 2.500 | **3** | 3 | 0.083463 | 1.766e-09 | 3 (1e-5) |
| chain | 71 | 5 | 2.500 | **3** | 3 | 0.076957 | 1.025e-09 | 3 (1e-5) |
| chain | 113 | 5 | 2.500 | **4** | 4 | 0.061628 | 2.568e-10 | 3 (1e-5) |
| x^n | 59 | 3 | 1.500 | **5** | 6 | 0.313049 | 7.675e-05 | 4 (1e-4) |
| x^n | 71 | 3 | 1.500 | **5** | 5 | 0.308038 | 5.721e-05 | 5 (1e-4) |
| x^n | 113 | 3 | 1.500 | **7** | 5 | 0.275936 | 2.552e-05 | 5 (1e-4) |
| x^n | 59 | 4 | 2.000 | **4** | 4 | 0.152806 | 4.573e-07 | 3 (1e-5) |
| x^n | 71 | 4 | 2.000 | **4** | 4 | 0.142982 | 2.955e-07 | 4 (1e-5) |
| x^n | 113 | 4 | 2.000 | **4** | 4 | 0.123191 | 1.005e-07 | 4 (1e-5) |
| x^n | 59 | 5 | 2.500 | **3** | 3 | 0.083463 | 1.194e-09 | 3 (1e-5) |
| x^n | 71 | 5 | 2.500 | **3** | 3 | 0.076957 | 6.930e-10 | 3 (1e-5) |
| x^n | 113 | 5 | 2.500 | **4** | 4 | 0.061628 | 1.737e-10 | 1 (1e-5) |

Readings.

* **The two families have the same ladder and differ only in height.** `Gamma_s` and `K*` depend on
  `(p, s)` only; `gamma_con(K) = mu N^{−(nu−q)/q} B · val(K)` puts the family in the prefactor. So
  the `x^n` and chain rows agree on every `K*` and differ by the fixed ratios of §2 (×1.414 at
  `nu = 3`, ×1.333 at `nu = 4`, ×0.677 at `nu = 5`). This is a *prediction of the algebra*, not a
  coincidence, and it is the cleanest statement the ladder makes: **the feature count is a property
  of the norm and the degree, not of the activation.**
* **`K*_free` and `K*_eq` agree in 7 of the 9 primes×degrees at `q = 2`**, and disagree by 1–2 in the
  two cells `(p 59, nu 3)` (5 vs 6) and `(p 113, nu 3)` (7 vs 5). Across all 78 cells they disagree
  in 20. The disagreement is not noise: at `p = 113, s = 3/2` the free optimum genuinely uses 7
  frequencies with unequal shares and beats the best equal-share 5-set by 6.6% in `gamma`.
* **The count falls with the degree at fixed `q`**: `nu` 3 → 4 → 5 gives `K*` 5 → 4 → 3 at `p = 59`
  and `p = 71`, and 7 → 4 → 4 at `p = 113`. That is the "sparser by degree alone" statement.
* **`argmin obj*` sits within 1 of `K*_eq` in 8 of 9 `q = 2` cells** — the training objective of the
  constructed nets is minimised by essentially the same `K` that maximises the margin, which is the
  finite-`lam` version of the `lam → 0` theorem.

### 4.1b The matched-`s` triple: the two dials really are one dial

`(nu, q) = (3, 2)`, `(4, 8/3)`, `(5, 10/3)` all have `s = nu/q = 3/2`. The programme sees only `s`,
so the exact prediction is that these three cells — three different **degrees**, three different
**penalty exponents** — have the *same support and the same shares*:

| `p` | `K*` | `Gamma_{3/2}` | support `S` | shares `b` |
|---|---|---|---|---|
| 59 | 5 | 0.31304888 | `{1, 6, 14, 16, 19}` | `.190 .248 .148 .240 .175` |
| 71 | 5 | 0.30803827 | `{1, 4, 11, 17, 31}` | `.188 .194 .212 .190 .215` |
| 113 | 7 | 0.27593576 | `{1, 6, 16, 31, 35, 44, 55}` | `.149 .172 .131 .133 .135 .110 .171` |

identical to every printed digit across the three `(nu, q)` cells at each prime. Note the shares are
**not equal** (`.148` to `.248` at `p = 59`), which is exactly why the share-weighted block of §2 is
needed and the inherited equal-amplitude `build_chain` is not. This triple is what Part 2 tests in
training (§7).

### 4.2 How sharply is `K*` resolved (T2)

`gamma_eq(K)/gamma_eq(K*)` around the peak, equal-share ladder:

| cell | `K*−2` | `K*−1` | **`K*`** | `K*+1` | `K*+2` |
|---|---|---|---|---|---|
| chain `p 113 nu 3 q 2` (`K*_eq` 5) | 0.628 | 0.910 | 1.000 | 0.997 | 0.988 |
| chain `p 71 nu 3 q 2` (5) | 0.722 | 0.905 | 1.000 | 0.961 | 0.953 |
| chain `p 59 nu 3 q 2` (6) | 0.954 | 0.983 | 1.000 | 0.975 | 0.939 |
| chain `p 113 nu 4 q 2` (4) | 0.398 | 0.797 | 1.000 | 0.983 | 0.895 |
| chain `p 113 nu 4 q 3` (9) | 0.992 | 0.998 | 1.000 | 1.000 | 0.995 |

Plain reading: **`nu = 4, q = 2` is the best-resolved cell** (the neighbours are 20% and 2% down) and
`nu = 3, q = 2` is resolved on the left but not the right (`K*+1` within 0.3% at `p = 113`). At
`q` close to `nu` the ladder is flat to within 1% over five rungs — there is no peak to find, which
is the same statement as "dense wins" (§4.3). Against the theorem's slack `F` (1.0009 / 1.031 /
1.102 at `nu = 3/4/5, q = 2`), only `nu = 4, q = 2` has its `K*±1` neighbours outside `F` on both
sides; everywhere else the honest claim is a **band**.

### 4.3 The dense flat competitor (T4)

The all-frequency equal-share net, evaluated by exactly the same machinery:

| `nu` | `q = 2` | `q = 2.5` | `q = 3` | `q = nu` |
|---|---|---|---|---|
| 3 (p 59 / 71 / 113) | 0.634 / 0.571 / 0.521 | 0.975 / 0.964 / 0.919 | **1.171 / 1.195 / 1.251** | same as `q = 3` |
| 4 (p 59 / 71 / 113) | 0.247 / 0.213 / 0.153 | 0.541 / 0.470 / 0.409 | 0.824 / 0.783 / 0.740 | **1.171 / 1.195 / 1.251** |
| 5 (p 59 / 71 / 113) | 0.084 / 0.068 / 0.041 | 0.247 / 0.213 / 0.153 | 0.478 / 0.413 / 0.348 | **1.171 / 1.195 / 1.251** |

(`gamma_dense / gamma_best-sparse-rung`, identical for both families.) **Dense loses at every `q < nu`
and wins at `q = nu`**, by 17–25% — it wins because the `K ≤ 12` ladder is truncated; at `q = nu`
the programme is linear and the true optimum is the full spectrum, which is exactly Morwani's
`L_{2,nu}` answer. The loss at `q = 2` is by a factor 1.6 at `nu = 3`, 4–7 at `nu = 4` and 12–24 at
`nu = 5`: **the preference for sparsity sharpens with every rung of the degree ladder.**

### 4.4 Is `Gamma_s` a local-optimiser artefact? (T7)

24 supports (the `K*−1, K*, K*+1` rungs at `p = 59, 71, 113` and `s = 3/2, 8/5, 2, 5/2`):

* the SLSQP epigraph solver and the independent projected-supergradient ascent in `x = b^s` agree to
  **1.7e-6 – 2.7e-2 relative**, with SLSQP always the larger (i.e. SLSQP is the better optimiser and
  the independent method never finds anything higher);
* branch and bound closes every one of the 24 to a **0.10% gap**, in 53–417 LP nodes and under 3 s,
  with the lower bound attained by SLSQP's own share vector to all printed digits.

So on the supports the ladder reports, the share problem is solved to 0.1% and `Gamma_s` is not an
artefact. The support search is the part that is not certified (§3).

### 4.5 Third-party cross-check, and where the two `K*` definitions really part company

The verifier agent's `FINAL_RUNS/VERIFY/gamma_small.json` solves the same programme with its own
code. Against this write-up's prime axis:

| `p` | `s` | `Gamma_s` (here) | `Gamma_s` (verifier) | ratio | `K*_free` here / verifier | `K*_eq` |
|---|---|---|---|---|---|---|
| 11 | 3/2 | 0.53572593 | 0.53572593 | 1.00000000 | 2 / 2 | 5 |
| 11 | 2 | 0.38019599 | 0.38019599 | 1.00000000 | 2 / 2 | 2 |
| 11 | 5/2 | 0.26942838 | 0.26942838 | 1.00000000 | 2 / 2 | 2 |
| 23 | 3/2 | 0.40293339 | 0.40293339 | 1.00000000 | 4 / 5 | 4 |
| 23 | 2 | 0.22100942 | 0.22100942 | 1.00000000 | 3 / 4 | 3 |
| 23 | 5/2 | 0.15064010 | 0.15064010 | 1.00000000 | 2 / 4 | 2 |
| 31 | 3/2 | 0.37009369 | 0.37009369 | 1.00000000 | 4 / 5 | 4 |
| 31 | 2 | 0.19740220 | 0.19740220 | 1.00000000 | 3 / 4 | 3 |
| 31 | 5/2 | 0.11414640 | 0.11414640 | 1.00000000 | 4 / 4 | 3 |

**The value agrees to every printed digit in all nine cells** — two independent implementations,
including `Gamma_{3/2}(23) = 0.402933388`, which is ADVERSARY's corrected value. The *count* differs
by 0–2 because the verifier reads the support size off the programme's argmax rung while this
write-up takes the **smallest** support within `t_v` of `Gamma_s` (§3). At `p = 23, s = 5/2` the same
value `0.15064010` is attained by a 2-frequency net and by 4-frequency nets with two near-zero
shares; the honest count is 2, and the 4 is the threshold artefact that motivated the rule. **This
is the concrete evidence that `K*` must be reported with its thresholds, which T5 does.** It also
confirms the verifier's headline observation independently: at `p = 11, s = 3/2` the free-amplitude
support is 2 while the equal-amplitude ladder peaks at 5.

---

## 5. `obj*(K)`: the training objective of the constructed nets

For a junk-free block the whole `p^n × p` logit table is the function `D ↦ Σ_k A_k cos(2πkD/p)` of a
single variable, so the cross-entropy of the full table is

    CE(scale) = log( 1 + Σ_{D ≠ 0} exp( scale^nu · gap(D) ) ),    gap(D) = −Σ_k A_k g_k(D) ≤ 0,

one `p−1`-vector, exactly, for every `n` and every `p`. Minimising `CE + lam·P_q` over the **global
weight scale** (the net is `nu`-homogeneous, the penalty `q`-homogeneous, so the direction is fixed
and only the scale is free) gives `obj*(K)`. Because the constructed net has `(n+1)K` or `NK`
neurons and any wider net can hold it by zero-padding — and zero neurons cost nothing — `obj*(K)` is
directly comparable to the training objective of a width-500 trained net.

### 5.1 `argmin_K obj*` and its `lam`-stability

At the `lam` the archived runs used, `argmin_K obj*` agrees with `K*_eq` within 1 in 8 of the 9
`q = 2` cells (§4.1) and moves by at most one rung over the ladder `lam ∈ {1e-3 … 1e-6}` in the
non-collapsed cells (T3). **Collapsed cells** are flagged: when `lam` is large enough that the best
global scale is essentially zero, `CE` sits at `log p` for every `K` and the argmin is decided by the
penalty alone, so the smallest net wins and `argmin obj* = 1`. This happens exactly where
RESULTS_PROD saw it in training (`q` near `nu` at `lam = 1e-5`), and it is the same phenomenon: at
`nu = 5, q = 10/3` and `q = 5` the `lam = 1e-6` column is already collapsed at `p = 59, 71, 113`.

### 5.2 Are the archived trained endpoints local, non-global minima? (`ladder_archived.py`)

Each archived endpoint's objective `min_scale CE + lam P_q` (exact, full `p^n` table, fp64) against
the best constructed net **including the dense flat one**, zero-padded into the same width:

| cell | `nu` | `q` | `lam` | best construction | constructed `obj*` | best free `obj*` (seeds) | free − constructed |
|---|---|---|---|---|---|---|---|
| chain `p 71` | 3 | 2 | 1e-4 | `K = 5` | 0.329463 | 0.353648 (16) | **+0.0242** |
| chain `p 71` | 3 | 2 | 1e-5 | `K = 5` | 0.039263 | 0.043814 (16) | **+0.0046** |
| chain `p 71` | 3 | 3 | 1e-5 | **dense (35)** | 0.362573 | 0.362574 (16) | **+0.000001** |
| chain `p 71` | 4 | 2 | 1e-5 | `K = 4` | 0.064838 | 0.092710 (8) | **+0.0279** |
| chain `p 71` | 4 | 4 | 1e-6 | **dense (35)** | 1.113755 | 1.157023 (2) | **+0.0433** |
| `x²` `p 71` | 3 | 2 | 1e-4 | `K = 5` | 0.266766 | 0.263239 (32) | **−0.0035** ← mismatch |
| `x²` `p 113` | 3 | 2 | 1e-4 | `K = 5` | 0.444882 | 0.453029 (16) | **+0.0081** |

Three things, in order of importance.

1. **In 6 of 7 cells the construction beats every trained seed**, so those endpoints are provably
   *local, non-global* minima of the training objective — the construction sits inside their own
   parameter space. This is the honest form of "training does not find the optimum".
2. **At `q = nu` the trained nets essentially reach the exact optimum.** At `nu = 3, q = 3` the
   dense flat construction gives `0.3625731` and the best of 16 seeds gives `0.3625742` — a relative
   gap of `3e-6`. At the Morwani endpoint the trained net *is* the global minimiser to six digits.
   This is a much stronger statement than anything in the archived write-ups, and it is only visible
   because the dense competitor was added to the candidate set (the `K ≤ 12` ladder truncates it).
3. **One mismatch, reported as such.** For `x²` at `p = 71, lam = 1e-4` the best of 32 trained seeds
   is *lower* than the best construction by `0.0035` (1.3% of the objective). The junk-free `x²`
   block costs 6 neurons per frequency; at finite `lam` a cheaper impure/junky direction can win on
   the **objective** even though the theorem is about the **normalized margin** as `lam → 0`. The
   same cell at `p = 113` goes the other way. So: at finite `lam` the constructed junk-free net is
   not always the objective's global minimiser in the `x^n` family, and we say so.

---

## 6. Prime axis

`ladder_prime.py`, no training. Two runs: **69 primes `11 … 1069`** at `s = 3/2, 2, 5/2` (i.e.
`nu = 3, 4, 5` at plain weight decay) with `K ≤ 10`, and a **denser pass over every prime
`11 … 199`** at all six `s` values with `K ≤ 12`. Figure `figs/prime_axis.{html,png}`, data
`json/prime_axis_{big,small}.json`. `K*_free` uses the threshold rule of §3 with `t_a = 1e-4` here
(the prime-axis pass stores activity at that threshold).

`K*_eq / K*_free` at plain weight decay:

| p | 11 | 23 | 47 | 71 | 113 | 199 | 313 | 499 | 733 | 1069 |
|---|---|---|---|---|---|---|---|---|---|---|
| `nu` 3 (`s` 3/2) | 2/5 | 4/4 | 5/5 | 5/5 | 7/5 | 7/8 | 8/7 | 10/8 | 10/10 | 9/9 |
| `nu` 4 (`s` 2) | 2/2 | 3/3 | 4/3 | 4/4 | 4/4 | 5/5 | 5/5 | 6/6 | 7/7 | 7/6 |
| `nu` 5 (`s` 5/2) | 2/2 | 2/2 | 3/3 | 3/3 | 4/4 | 4/4 | 5/4 | 5/5 | 5/5 | 5/5 |

**`K*` grows logarithmically in `p`**, with a slope that falls as the degree rises:

| | fit (least squares over the 69 primes) | `R²` | median abs. residual |
|---|---|---|---|
| `nu` 3, `q` 2 | `K*_free ≈ −1.56 + 1.68 ln p` | 0.927 | 0.29 |
| `nu` 4, `q` 2 | `K*_free ≈ −0.13 + 0.97 ln p` | 0.910 | 0.28 |
| `nu` 5, `q` 2 | `K*_free ≈ +0.01 + 0.76 ln p` | 0.904 | 0.22 |

(the equal-share fits are within 0.1 of these slopes). Plain reading: *the number of features the
`l_2` optimum uses grows like the logarithm of the modulus, and each extra degree of homogeneity
divides that growth rate by roughly √2.* This is the quantitative form of the memoed "`c` grows
`~log p`" and it now has a slope per degree.

**Where the argmax is fragile.** The `resolution` column in `prime_axis_*.json` is the runner-up
rung's `gamma_hat` divided by the winner's:

| `s` | median resolution over the 69 primes | range | `K*_free = K*_eq` |
|---|---|---|---|
| 3/2 (`nu` 3, `q` 2) | **0.990** | 0.916 – 1.000 | 37/69 |
| 2 (`nu` 4, `q` 2) | 0.982 | 0.796 – 0.999 | 55/69 |
| 5/2 (`nu` 5, `q` 2) | 0.952 | 0.650 – 0.998 | 59/69 |

**At `nu = 3, q = 2` the equal-share argmax is a near-tie (within 1%) at half the primes**, and the
two `K*` definitions agree at only 37 of 69. At `nu = 4` and `nu = 5` the peak is sharper and they
agree at 55 and 59. So the `nu = 3` cell — the one Morwani's `x²` lives in — is precisely the worst
place to quote a single integer, and `nu = 4, q = 2` is the best. Fragility was handled by (i)
reporting both definitions, (ii) the value tolerance `t_v` of §3, and (iii) never quoting a `K*`
without its neighbours' ratios (T2).

---

## 7. Part 2 — trends from free training

**Trends only.** No pass/fail and no pre-registration: the theorem is about the *global minimiser* of
the regularised loss as `lam → 0` (Wei et al. 2019, Thm 4.1), so where gradient descent lands is
outside its scope. These are "we trained the nets and this is roughly what we saw" tables.

Setting: gated chain (`--act prod`), `p ∈ {59, 71}` at `nu = 3` and `p = 59` at `nu = 4, 5`, four
seeds per cell, recipe from PRIME_AXIS §5.2 (penalty off until train CE `< 0.8 ln p`, constant lr,
grad-norm clip 1, init 4 / 14). Full tables in `FINAL_RUNS/LADDER/trends.md`, raw rows in
`FINAL_RUNS/LADDER/train/scored*.json`.

### 7.1 The count instruments, and what they are worth

The **primary causal instrument is loss ablation in logit space** (Nanda et al. 2301.05217 §4.4 /
Fig. 6), implemented in `ladder_loss.py`. Margin ablation is kept as an appendix column, because a
thin margin makes it meaningless (§7.3).

| instrument | definition | on a constructed `K`-net |
|---|---|---|
| **`K_loss25`** | project `cos/sin(2πkc/p)` out of every row's logit vector along the **class axis**, recompute the full-table CE, and count the frequencies whose loss increase `ΔCE_k` is `≥ 25%` of the largest | **exact, 4/4** (and it recovers the exact support) |
| `K_loss10` | the same at a `10%` threshold | exact, 4/4 |
| `K_loss_abs` | `ΔCE_k ≥ 0.01 ln p` (an absolute threshold, 1% of the uniform-guess loss) | exact, 4/4 at the normalised scale |
| **`K_5pct`** | frequencies with `≥ 5%` of the clean spectral energy | — |
| Morwani `n5pct` | per neuron, the `argmax` frequency of its token-indexed vectors' Fourier power; count the frequencies holding `≥ 5%` of the neuron-norm-weighted mass (Morwani et al. 2311.07568 Fig. 4/7) | exact, 4/4 |
| Morwani `eff` | `1 / Σ_k share_k²`, the effective number of frequencies in that histogram | `= K`, 4/4 |
| `N_.05` | frequencies with `≥ 5%` of the top amplitude | a **tail statistic**, never "the count" |
| `K_abl`, `K_d25` | margin-ablation counts (appendix) | exact, 4/4, but see §7.3 |

**Scale.** A trained net's logits are large, so its unablated CE is `2·10⁻³ … 0.13` nats and every
`ΔCE_k` is dominated by the scale rather than the structure. Every loss quantity is therefore
computed twice: at the **trained** scale and at a **normalised** scale, where the logits are
multiplied by one scalar `alpha` chosen so the unablated CE is `0.1` nats. **All `K_loss` numbers
quoted below are at the normalised scale**; both are stored in `train/loss_*.json`, together with
Nanda's RESTRICTED loss (keep only the counted frequencies) and EXCLUDED loss (remove them).

**Validation** (`train/loss_validate.json`): on constructed share-weighted junk-free chain nets at
`p = 59` (`n = 2, 3`; `K = 3, 4, 5`), `K_loss25` returns **exactly `K`** and exactly the right
frequencies in 4/4 cases, `ΔCE_k` is identically 0 off the support, and the Morwani histogram gives
`n5pct = K` with `eff = K.0`.

Two implementation notes on the *margin* instrument, for the appendix. (i)
`nudial/ablation_count.py` hardwires "the top 32 by interface energy" for the null cut, which leaves
an **empty null pool** whenever `nf = (p−1)/2 < 48` — at `p = 59, nf = 29` — and then the floor
degenerates to `1e-6` and every candidate is counted. The cut is made to scale with `p` here.
(ii) On a **dense** net no single frequency carries much of the margin, so `K_d25`/`K_d50` collapse
to 0.

### 7.1b All ten cells, at a glance

`K_loss25` and `K_5pct` lead; the Morwani histogram is next; margin ablation is the appendix.
Full table in `FINAL_RUNS/LADDER/trends.md`. Four seeds everywhere; gated chain; `p = 59` unless
stated. **`K*` is the exact free-amplitude count of §4; "dense" means the programme's optimum is the
whole spectrum.**

| `nu` | `q` | `s` | `p` | `lam` | **`K_loss25`** | **`K_5pct`** | Morwani `n5pct` | Morwani `eff` | `N_.05` | `K_abl` (appx) | **`K*`** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 3/2 | 59 | 1e-5 | **6 [4–7]** | 6 [6–7] | 8 [7–10] | 11.1 | 7 | 8 [8–13] | **5** |
| 3 | 2 | 3/2 | 59 | 1e-4 | **6 [3–7]** | 6 [3–7] | 7 [6–8] | 8.5 | 7 | 10 [8–12] | **5** |
| 3 | 2 | 3/2 | 71 | 1e-5 | **6 [5–8]** | 5 [3–5] | 7 [7–10] | 10.8 | 6 | 11 [8–15] | **5** |
| 3 | 2 | 3/2 | 71 | 1e-4 | **6 [5–9]** | 5 [4–7] | 7 [5–9] | 8.5 | 7 | 10 [8–14] | **5** |
| 4 | 8/3 | 3/2 | 59 | 1e-6 | **5 [3–10]** | 4 [4–5] | 2 [2–4] | 26.5 | 8 | 10 [4–13] | **5** |
| 5 | 10/3 | 3/2 | 59 | 1e-6 | **4 [4–6]** | 4 [4–6] | 6 [6–8] | 19.5 | 5 | — | **5** |
| 4 | 3 | 4/3 | 59 | 1e-6 | **7 [5–14]** | 5 [4–6] | 4 [2–5] | 25.7 | 10 | 12 [10–14] | **7** |
| 5 | 3 | 5/3 | 59 | 1e-6 | **5 [3–6]** | 5 [3–6] | 6 [5–7] | 22.1 | 5 | — | **4** |
| 3 | 3 | 1 | 59 | 1e-6 | **27 [24–29]** | 7 [1–11] | **0** | 28.0 | 24 | 3 [0–8] | **29 dense** |
| 3 | 3 | 1 | 71 | 1e-6 | **31 [28–32]** | 6 [5–9] | **0** | 33.0 | 26 | 10 [4–14] | **35 dense** |
| 4 | 4 | 1 | 59 | 1e-6 | **19 [18–21]** | 8 [7–12] | 1 [1–4] | 26.6 | 18 | 7 [6–10] | **29 dense** |

Four readings.

1. **The causal count tracks `K*` across the whole dial.** `K_loss25` medians: 6, 6, 6, 6, 5, 4, 7, 5
   in the eight sparse cells against `K*` = 5, 5, 5, 5, 5, 5, 7, 4 — **within 1 of the exact value in
   8/8**, and 19–31 in the three `q = nu` cells against a dense optimum of 29/35/29. Nothing else in
   this project separates the sparse and dense endpoints by a factor of 4 on a *causal* measurement.
2. **The matched-`s` collapse is visible.** `(nu, q) = (3, 2), (4, 8/3), (5, 10/3)` all have
   `s = 3/2` and the same exact `K* = 5`; the trained `K_loss25` medians are **6, 5, 4** and the
   `K_5pct` medians **6, 4, 4**. Three degrees, three penalty exponents, one count — as trends.
3. **The Morwani neuron histogram behaves as expected at `nu = 3` and does not at `nu ≥ 4`.** At
   `q = 2` the neuron-norm-weighted mass concentrates on 7–8 frequencies with an effective count of
   8.5–11.1; at `q = nu` **no frequency holds 5% of the mass** and the effective count is 28–33 of
   29/35 — the uniform endpoint, checked rather than assumed. At `nu ≥ 4` the instrument stops
   concentrating (median max normalized power 0.13–0.16, effective count 20–27) because the trained
   neurons are not single-frequency: **at higher degree the neuron basis is not the feature basis**,
   which is a fact about the solution, not about the instrument.
4. **Margin ablation is not usable here and is demoted.** At `nu = 4` the trained min-margin is
   0.8–1.9 against logits of order `10²`, so cutting *any* frequency (null-pool ones included) drives
   it negative — the measured null floors are 0.81–4.37, i.e. drops of 81–437%. At `nu = 5` the
   margin is negative outright and the instrument is undefined (reported as `—`, not as 0). Loss
   ablation has no such failure mode, which is exactly why Nanda et al. count with it.

### 7.1c The archived endpoints under the same instrument (48 seeds, free)

Re-scoring the archived `q = 2` cells with loss ablation costs nothing and widens the seed count:

| archived cell | seeds | `K_loss25` | `K_loss10` | Morwani `n5pct` | Morwani `eff` | `K_5pct` | `K*` exact |
|---|---|---|---|---|---|---|---|
| chain `n 2, p 71, lam 1e-4` | 16 | 7 [6–14] | 11 [7–20] | 7 [6–8] | 11.5 | 6 [4–8] | **5** |
| `x²`, `p 71, lam 1e-4` | 16 | 10 [5–13] | 12 [8–16] | 11 [8–13] | 11.6 | 7 [5–9] | **5** |
| `x²`, `p 113, lam 1e-4` | 16 | 9 [6–15] | 25 [13–30] | 8 [6–11] | 21.1 | 7 [5–9] | **7** |

Consistent with §7.1b, and the `x²` family runs 2–4 counts higher than the chain at the same `p` and
`lam` — its trained solutions carry more junk (clean fraction 0.41 vs 0.58–0.87, RESULTS_PROD), which
is what the extra counted frequencies are.

### 7.2 `nu = 3` (`n = 2`), both primes

| `p` | `q` | `s` | `lam` | acc | margin | **quality** | junk | `K_abl` | `K_d25` | `K_5pct` | `N_.05` | `K*` exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 59 | 2 | 1.5 | 1e-4 | 1.000 | 2.98 [2.39–3.24] | 0.386 [0.30–0.44] | 0.21 | 10 [8–12] | 9 [6–13] | 6 [3–7] | 7 [4–8] | 5 |
| 59 | 2 | 1.5 | 1e-5 | 1.000 | 5.59 [5.08–5.93] | 0.511 [0.45–0.58] | 0.16 | 8 [8–13] | 9 [7–14] | 6 [6–7] | 7 [6–9] | 5 |
| 59 | 3 | 1.0 | 1e-6 | 1.000 | 8.38 [8.25–8.52] | **0.941 [0.92–0.96]** | 0.05 | 3 [0–8] | 0 | 7 [1–11] | 24 [22–25] | 12+ (dense) |
| 71 | 2 | 1.5 | 1e-4 | 1.000 | 3.04 [2.14–3.17] | 0.400 [0.27–0.42] | 0.21 | 10 [8–14] | 9 [8–15] | 5 [4–7] | 7 [6–9] | 5 |
| 71 | 2 | 1.5 | 1e-5 | 1.000 | 5.77 [4.58–6.29] | 0.533 [0.40–0.60] | 0.13 | 11 [8–15] | 9 [6–15] | 5 [3–5] | 6 [6–8] | 5 |
| 71 | 3 | 1.0 | 1e-6 | 1.000 | 8.15 [7.99–8.24] | **0.944 [0.93–0.96]** | 0.10 | 10 [4–14] | 0 | 6 [5–9] | 26 [24–27] | 12+ (dense) |

Three trends, stated as trends.

1. **The `q = nu` (Morwani) endpoint is where training gets close to the exact optimum**, and the
   `q = 2` endpoint is where it does not: quality `0.94` vs `0.39–0.53`, junk `0.05–0.10` vs
   `0.13–0.21`. This is the same fact as §5.2's objective match to `3e-6` at `q = 3`, seen through
   the margin instead of the objective.
2. **At `q = 2` the spectral head is about the right size and the ablation tail is not.** `K_5pct`
   has median 5–6 against `K* = 5`; `K_abl` and `K_d25` sit at 8–11 because the trained net's
   min-margin leans on a tail of small frequencies that the construction does not have. Two honest
   readings are available and we give both: *the head matches `K*`*, and *the trained net is not a
   `K*`-frequency net.*
3. **At `q = nu` the spectrum is dense** (`N_.05` 24–27 out of 29 and 35 frequencies) — the ladder's
   `q = nu` prediction, seen in training.

### 7.3 `nu = 4` and `nu = 5`: the matched-`s` triple

| `nu` | `q` | `s` | `p` | acc | margin | quality | junk | `K_abl` | `K_d25` | `K_5pct` | `N_.05` | `K*` exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 3/2 | 59 | 1.000 | 5.59 | 0.511 | 0.16 | 8 [8–13] | 9 [7–14] | **6 [6–7]** | 7 | **5** |
| 3 | 2 | 3/2 | 71 | 1.000 | 5.77 | 0.533 | 0.13 | 11 [8–15] | 9 [6–15] | **5 [3–5]** | 6 | **5** |
| 4 | 8/3 | 3/2 | 59 | 1.000 | 1.49 [0.76–1.88] | 0.097 | 0.34 | 10 [4–13] | 13 [11–14] | **4 [4–5]** | 8 [6–11] | **5** |
| 5 | 10/3 | 3/2 | 59 | 0.998 [0.996–1.000] | **−12.2 [−21.4, −3.1]** | −0.51 | 0.51 | — | — | **4 [4–6]** | 5 [4–9] | **5** |

**`nu = 5` did not reach a positive margin inside the budget, and we say so plainly.** Four seeds,
60 k steps, width 250, tied embeddings, `p = 59`: accuracy 0.996–0.9998 but the *minimum* margin over
all `59⁴` rows stays negative (−3.1 to −21.4). This is the same picture the archived `nu = 5` runs
showed at 120 k steps (−6 to −8, RESULTS_PROD), so it is the degree, not the shortened budget. With
a negative margin the normalized margin and every ablation instrument are meaningless — `quality` is
reported as negative and the ablation columns as undefined rather than as zero — and only the
spectral instruments carry information.

**The matched-`s` prediction is that these rows share a count.** On the spectral-head instrument they
do: `K_5pct` medians **6, 5, 4, 4** at `(nu,q) = (3,2)p59, (3,2)p71, (4,8/3), (5,10/3)` against
`K* = 5` — within ±1 of the exact value and of each other, across three degrees and three penalty
exponents that share `s = 3/2`. That is the collapse the exact theory predicts, seen as a trend.

On the ablation instruments they do not agree, for a reason worth stating: **at `nu = 4` the trained
min-margin is only 0.76–1.88 while the logits are of order 10², so cutting *any* frequency —
including the null-pool ones — drives the margin negative** (measured null floors 0.81–4.37, i.e.
drops of 81–437%). The ablation count is only meaningful when the margin has room. Quality falls
hard with degree (0.51 → 0.10 → negative) and junk rises (0.15 → 0.34 → 0.51): **the higher the
degree, the further gradient descent ends up from the exact optimum**, which is the honest
counterpart to the exact result that the higher the degree, the more the optimum prefers sparsity.

---

## 8. What the paper can say

Ordered by how much of the weight each claim can carry.

**A. The count is a property of the norm and the degree, not of the activation.** Two structurally
different exact families — the gated chain and `x^n` — have *the same* `gamma_con(K)` ladder up to a
`K`-independent constant, because `gamma_con(K) = mu · N^{−(nu−q)/q} · B · val(S_K, b_K)` and only
`val` depends on `K` (§2, §4.1). This is proved algebra plus an exact numerical check to `6.7e-16`,
and it is the cleanest thing in this write-up.

**B. Exact, training-free evaluation exhibits the predicted feature count.** For every paper cell we
build the share-weighted junk-free block, verify it is junk-free to `1.2e-13` against the full
`p^n × p` table, and read off `gamma_con(K)` and `obj*(K)` in closed form. At plain weight decay the
count falls with the degree — `K* = 5, 4, 3` at `p = 59, 71` for `nu = 3, 4, 5` — and the margin
advantage of sparse over dense sharpens with the degree (`1.6×`, `4–7×`, `12–24×`).

**C. The dense (`L_{2,nu}`) solution and the sparse (`l_2`) solution are the two endpoints of one
dial, and the dial is exactly `s = nu/q`.** Dense loses at every `q < nu` and wins at `q = nu`, in
both families and at all three primes (§4.3). Two different `(nu, q)` pairs with the same `s` give
the same ladder, so the degree dial and the penalty dial are the same dial.

**D. `K*(p)` grows like `log p`, with a slope that falls with the degree**: `1.68`, `0.97`, `0.76`
per `ln p` at `nu = 3, 4, 5` under plain weight decay, `R² ≈ 0.90–0.93` over 69 primes to `p = 1069`
(§6). This is the first quantitative form of "the count grows slowly with the modulus".

**E. Trained endpoints are local, not global, minima of the regularised loss — except at
`q = nu`, where they are essentially global.** In 6 of 7 archived cells a constructed net, zero-padded
into the trained net's own parameter space, has a strictly lower training objective; at the
`L_{2,nu}` endpoint the trained net matches the exact dense optimum to `3e-6` relative (§5.2). The
one counter-case (`x²`, `p = 71`, `lam = 1e-4`) is reported.

**F. Free training roughly follows the ladder, measured causally.** With the Nanda-protocol
loss-ablation count, the trained count is **within 1 of the exact `K*` in all eight sparse cells**
(medians 6, 6, 6, 6, 5, 4, 7, 5 against `K*` 5, 5, 5, 5, 5, 5, 7, 4) and jumps to 19–31 in the three
`q = nu` cells against a dense optimum of 29/35/29. The matched-`s` triple `(3,2), (4,8/3), (5,10/3)`
gives 6, 5, 4. Quality against the exact ceiling is 0.94 at `q = nu`, 0.39–0.53 at `nu = 3, q = 2`,
0.10 at `nu = 4` and negative at `nu = 5`: **the higher the degree, the further gradient descent
finishes from the optimum whose sparsity the theory explains.** Trends only.

**What the paper should NOT say.** That any of this proves no network beats the construction (that is
target (T), still open off the no-same-frequency-junk class); that `K*` is pinned to a single integer
(T2 shows it is a band in most cells); that `N_.05` is a count; that training finds the optimum at
`q = 2`.

## 9. Caveats a referee would raise

1. **`Gamma_s` has no global optimality certificate for the support.** The programme is a non-convex
   max–min. The *share* problem on a given support is certified here by branch and bound (T7), and
   the *support* search is exhaustive only for `K ≤ 5` at `p = 59, 71, 113`. Everything above that,
   and the whole large-prime axis, rests on a heuristic search. Direction of the error: `Gamma_s` is
   a maximum, so a missed support can only raise it and raise `K*_free`.
2. **Two different `K*`s.** `kstar.py`'s equal-amplitude argmax and the free-amplitude support size
   are different quantities that the earlier write-ups did not distinguish. §3 and T1/T5/T6 report
   both and flag disagreements. The paper must say which one its theorem pins (the free one).
3. **`K*` is often resolved only to ±1.** T2 gives the relative gaps between adjacent rungs; the
   runner-up rung is typically within 1–8% of the peak, i.e. inside or near the theorem's slack
   `F` (1.0009 / 1.031 / 1.102 at `nu = 3/4/5, q = 2` on the no-same-frequency-junk class, and
   `F = 1` at `q = 2` if the P1 fan-out's claimed closure holds). Where the gap is smaller than `F`
   the honest statement is a **band**, not a point.
4. **The constructions are a lower bound on the family's margin.** `gamma_con` is the normalized
   margin of *one* explicit net per rung (the share-weighted junk-free block). The claim "no network
   beats it" is target (T) of the global-optimality programme, which is proved only on the class
   with no same-frequency junk (GOPT_STATE2). This write-up does not prove (T); it measures the
   construction.
5. **The `x^n` blocks are large.** `N = 6, 18, 54` neurons per frequency at `n = 2, 3, 4`. They are
   exactly junk-free (verified to `1.2e-13`), and they are the smallest *cyclic* such populations,
   but there may be smaller non-cyclic ones, which would raise the `x^n` ladder's height (not its
   shape — the `K`-dependence is entirely in `val(S, b)`).
6. **`obj*` compares a construction to trained endpoints at one width and one `lam`.** It is a
   statement about the training objective's landscape, not about optimisation; and the "collapsed"
   cells (flagged in T1/T3) are ones where the best scale is essentially zero and the argmin over
   `K` is decided by the penalty alone, which is a real regime (`lam 1e-5` at `q = 4`,
   RESULTS_PROD) but not an informative one.
7. **Part 2 is trends.** Four seeds per cell, one recipe, two primes at `nu = 3` and one at
   `nu = 4, 5`; counts from two instruments that do not always agree. Nothing here is a test of the
   theorem, which is about the global minimiser as `lam → 0` (Wei et al. 2019, Thm 4.1).
8. **`N_.05` is not a count.** It counts frequencies above 5% of the top amplitude and therefore
   grows with the size of the tail. It is reported as a tail statistic only.
9. **The instruments disagree, and the disagreement is structured.** All four count instruments are
   exact on constructed nets (4/4 or 8/8); on trained nets the causal loss-ablation count sits within
   1 of `K*`, the spectral head `K_5pct` within 1–2, and the *margin*-ablation count 3–7 above — the
   last because the trained min-margin is thin and leans on a tail the construction does not have.
   The paper should lead with loss ablation (Nanda protocol) and the spectral head, quote the
   Morwani neuron histogram as the third view, and report margin ablation only with the thin-margin
   caveat. Any single-number claim about "the count" is a choice of instrument and must say so.
10. **`K_loss` depends on a threshold and on a scale.** Both are stated (25% of the max `ΔCE`, at a
   logit scale normalised to CE = 0.1 nats); the 10% threshold gives counts 1–3 higher, and at the
   trained scale the absolute threshold misfires because the unablated CE is `10⁻³`. The full sorted
   `ΔCE_k` curves are in `train/loss_*.json` so a referee can pick another threshold.
9. **No TF32** anywhere; margins and spectra are fp64 on the full table.
