# (T) at p = 3: unrestricted global optimality of the ansatz

**Theorem.** Take n = 2 operands, q = 2 (plain weight decay), modular addition mod 3, and the gated
chain `z_c(x) = sum_i u_i(x1) v_i(x2) w_i(c)`. For every width, every real weight array (mixed
frequencies/DC allowed, junk allowed), the normalized margin satisfies

    gamma(theta) = margin / P^{3/2}  <=  sqrt(3) / 18^{3/2}  =  3^{-1/2} * 2 * (6p)^{-3/2} * Gamma_{3/2}(3),

where `P = sum_i ||u_i||^2 + ||v_i||^2 + ||w_i||^2` and Gamma_{3/2}(3) = 3/2. The three-neuron
common-phase block attains this value. So the ansatz with K* = 1 is the global margin maximizer: no
width, impurity or junk pattern beats it.

Status: **PROVED** (computer-assisted). The last step (§5) is a two-variable inequality. Away from its
single tight point it is certified by interval arithmetic: `v3_certify.py` certified **108 cells with 0
failures** (`v3_certify.log`). The tight point is handled analytically.

## 0. Symbols

| symbol | meaning |
|---|---|
| `omega = e^{2 pi i/3}`, `mu_3 = {1, omega, omega^2}` | cube roots of unity |
| per-neuron atom `(c_i, v_i, w_i)` | the frequency-1 content of neuron `i`: if `u_i`'s frequency-1 part is `alpha cos(2 pi x/3 + phi)`, `v_i`'s is `beta cos(2 pi x/3 + psi)` and `w_i = a cos(2 pi c/3 + chi)` (the readout is centered, so this is all of it), then `c_i = alpha beta a e^{i(chi - phi - psi)}/8`, `v_i = e^{2 i phi}`, `w_i = e^{2 i psi}` |
| `G = sum c_i` | clean coefficient (row-average signal) |
| `J_1 = sum c_i v_i`, `J_2 = sum c_i w_i`, `J_12 = sum c_i v_i w_i` | the three same-frequency junk coefficients |
| `r = 2/3` | `q / nu` |
| `x_i = |c_i|`, `e_k` | the atom amplitudes and their elementary symmetric functions |

## 1. Cost

For each neuron, AM-GM followed by dropping the DC parts gives
`||u||^2 + ||v||^2 + ||w||^2 >= 3 (||u|| ||v|| ||w||)^{2/3} >= 3 ((3/2)^{3/2} 8 |c_i|)^{2/3} = 18 |c_i|^{2/3}`.
Hence `P >= 18 sum_i |c_i|^r`, with equality for balanced DC-free neurons.

## 2. Margin: only frequency-1 content is seen (IMPURE.md, Theorem 1)

Every distribution `pi` on (row, wrong offset) pairs gives `margin <= E_pi gap`. Use:

* (clean) rows uniform, one offset `D`: `E gap = 2 Re[G (1 - omega^D)]`;
* (type T) `D` uniform on {1,2}, and the row placed on the line `x1 + D/2 = s` (T={1}), `x2 + D/2 = s`
  (T={2}) or `x1 + x2 + D/2 = s` (T={1,2}), uniform along it. Then `E gap = 3 Re(G - J_T zeta)` with
  `zeta = e^{2 pi i 2 s/3}` ranging over `mu_3`.

These distributions are blind to every cross-frequency/DC component of the logit tensor (IMPURE.md
Theorem 1, checked to 1e-16 on arbitrary tensors). So the formulas hold for **every** network, with
(G, J) summed over the per-neuron atoms of §0. `p3_convention.py` checks the constant 3 exactly.

Hence `margin <= 3 b`, with `b := min{ Re G - |Im G|/sqrt3 , min_T min_{zeta in mu_3} Re(G - J_T zeta) }`.
Since `max_{zeta in mu_3} Re(J zeta) >= |J|/2` (some cube root lies within 60 degrees of `arg J`),
`b <= |G| - max_T |J_T| / 2`.

## 3. Reduction

If **(V3)** `sum_i |c_i|^r >= 3^{1-r} (|G| - max_T |J_T|/2)^r` holds for every population, then

    gamma <= 3 b / (18 sum|c|^r)^{3/2} <= 3 (sum|c|^r / 3^{1/3})^{3/2} / (18^{3/2} (sum|c|^r)^{3/2}) = sqrt3 / 18^{3/2}.

## 4. Proof of (V3): two lower bounds on the cost

Normalize `G = 1` by rotating and scaling every `c_i`. Put `delta = max_T |J_T|`, `e1 = sum x_i >= 1` and
`c_i = x_i e^{i alpha_i}`; then `sum x_i cos alpha_i = 1`. If `e1 >= sqrt3` then
`sum x^r >= e1^r >= 3^{1-r}` and we are done, so `e1 in [1, sqrt3]`. If `delta >= 2`, the right side is 0.

**(C) Complex chain matrix (ELEGANT).** Let `m_i = (1, v_i, v_i w_i)` and `M = sum c_i m_i m_i^*`. Its
diagonal is `G = 1`, and its lower triangle holds exactly `J_1, J_2, J_12`. Cauchy-Binet with
Hadamard gives `|E_k(M)| <= 3^k e_k(x)`. The upper triangle holds the mirrors `U_T = sum c_i conj(v_T,i)`,
and `U_T - conj(J_T) = 2i sum x_i sin(alpha_i) conj(v_T,i)`. By Jensen (`sqrt(1-c^2)` is concave),
`sum x_i |sin alpha_i| <= sqrt(e1^2 - 1)`. So `|U_T| <= mu := min(e1, delta + 2 sqrt(e1^2 - 1))`.
Expanding `E_2 = 3 - sum_T J_T U_T` and `det M` then gives

    e2 >= (1 - delta mu)/3,     e3 >= (1 - 3 delta mu - delta mu^2 - delta^2 mu)/27 .

The Bernstein identity `x^r = C_r int log(1 + x t) t^{-1-r} dt` gives
`sum x^r >= C_r int log(1 + e1 t + e2 t^2 + e3 t^3) t^{-1-r} dt`.
For any `(1+at)(1+bt)(1+ct)` or `(1+at)(1 + 2R cos(th) t + R^2 t^2)` that is coefficientwise dominated
by that cubic, the right side is at least `a^r + b^r + c^r` or `a^r + 2 R^r cos(r th)`, respectively.
The complex formula was checked numerically in `bern_check.py`.

**(H) Hermitian chain matrix with absorbed phases.** Let `u_i = (1, e^{i alpha_i} v_i, e^{i alpha_i} v_i w_i)/sqrt3`
and `M_h = sum 3 x_i u_i u_i^*`, which is positive semidefinite. Rotfel'd's trace inequality
(`tr f(sum A_i) <= sum tr f(A_i)` for concave `f` with `f(0) = 0`) gives `sum (3x_i)^r >= sum_j lam_j(M_h)^r`.
The diagonal of `M_h` is `e1`. Its lower entries are `J_1`, `J_12` exactly, and `K_2 = sum x_i w_i`, with
`|K_2 - J_2| <= sum x_i |1 - e^{i alpha_i}| <= sqrt(2 e1 (e1 - 1))` (Jensen). Hence `lam = e1 + m` with
`sum m = 0`, `|m|^2 <= rho^2 := 2(2 delta^2 + (delta + sqrt(2 e1(e1-1)))^2)` and `lam >= 0`. The minimum of
`sum lam^r` over this convex set is attained at an extreme point: either a sphere point with two equal
coordinates (Lagrange: `r lam^{r-1} = a + b lam` has at most two roots), or a face point, or the vertex
`(3 e1, 0, 0)`. `v3_certify.py` enumerates all of these.

**§5. The two-variable inequality.** For all `delta in [0,2]` and `e1 in [1, sqrt3]`:

    max( (C), (H)/3^r, e1^r )  >=  3^{1-r} (1 - delta/2)^r .

* *Tight corner* `delta <= 0.05`, `e1 = 1 + eps <= 1.01`. Here `rho^2 <= 8 delta^2 + 8 e1 eps <= 0.101`, so
  every `lam >= 0.74`. A Taylor bound with remainder gives
  `sum lam^r >= 3 + 0.672 eps - 1.661 eps^2 - 1.328 delta^2 >= 3 - 1.328 delta^2`. Also
  `3 (1 - delta/2)^r <= 3 - delta`, and `3 - 1.328 delta^2 >= 3 - delta` for `delta <= 0.75`.
  Equality holds only at `delta = eps = 0`, i.e. the block.
* *Everywhere else*: an adaptive grid with interval arithmetic (mpmath `iv`, 30 digits). Each cell
  `[da,db] x [ea,eb]` is certified from its worst corner. This is valid because (C) is monotone (its
  deficits increase in `delta` and `e1`, and the `e1 t` coefficient uses `ea`), and (H) is monotone
  (scaling `lam` by `ea/e1` maps the feasible set into the one centered at `ea`). Result: **108/108 cells certified, 0 failures**
  (`v3_certify.log`). Each cell's certificate is one of: the trivial bound `e1^r`, (H), or (C) via an
  explicit dominated factorization (the exact reciprocal roots of the cubic scaled by `1 - 1e-9`, or a
  searched real triple or complex pair), all re-checked in interval arithmetic. When the interval feasibility of a two-equal-coordinate candidate is
  inconclusive, it is still included, with its possibly-zero term clipped to 0. That can only lower the
  bound, so no case is lost.

Pointwise margins away from the corner are at least 4.9% (`margin_map.py`), and the whole chain was
validated on 3000 random or adversarial populations (`v3_chain_check.py`, `flag_check.py`).

## 6. Why p = 3 closes and p >= 5 does not (yet)

At p = 3 there is one frequency. The blind family's forced uniform offset weights cost nothing there,
because with G real both offsets see the same clean value. The blind relaxation is therefore sharp
(`xblind_p3.log`: best 0.990 <= 1), so impurity drops out exactly. At p = 5 and 7 it is not sharp
(`xblind_p5.log` 1.115, `xblind_p7.log` 1.265): with several frequencies sharing one aiming density,
impure networks are not excluded by blindness alone, and the charging lemma of IMPURE.md §4 is still
needed.
