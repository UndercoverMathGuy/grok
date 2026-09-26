# S5: a three-dimensional projector beats the degree-four symmetric ceiling

2026-09-24. Companion to `S5_COUNTEREXAMPLE.md`. This stronger construction
has 193 neurons at degree four and exceeds the exact symmetric ceiling by
24.694%. At degree three it has 49 neurons and exceeds the schoolbook
reference by 2.979%; the latter comparison is not yet a refutation of the
true degree-three symmetric ceiling.

## 1. One formula, both degrees

Let V be the sum-zero subspace of R^5, and let R be the standard
representation of S5 on V. Set

\[
 u=(-3,-3,2,2,2)/\sqrt{30},\qquad A=I_V-uu^T.
\]

Thus A is the orthogonal projection onto the three-dimensional subspace
whose first two coordinates sum to zero and whose last three coordinates
sum to zero. Given n=nu-1 operands with product P and a proposed label c,
define the logit

\[
 z(a_1,\ldots,a_n;c)=\operatorname{tr}(A R_{Pc^{-1}})
             +\tfrac12\operatorname{sgn}(Pc^{-1}).             \tag{1}
\]

This is a real product network: the explicit expansion and its exact
penalty are below. Its correct score is 7/2.

## 2. Exact margin, by cycle type

For D in S5, let f(D) be its number of fixed letters and let k(D) be the
number of letters in {1,2} that D sends into {3,4,5}. Equally many letters
cross in the other direction. Hence

\[
 1-u^T R_Du=5k(D)/6,
 \qquad\operatorname{tr}(A(I-R_D))=5-f(D)-5k(D)/6.
\]

The classification gap of (1) is therefore the rational number

\[
 g(D)=5-f(D)-5k(D)/6+\tfrac12(1-\operatorname{sgn}(D)).       \tag{2}
\]

All nonidentity permutations have one of the following types:

| Type | Fixed letters | Maximum k | Parity patch | Gap lower bound |
|---|---:|---:|---:|---:|
| (12) | 3 | 1 | 1 | 13/6 |
| (123) | 2 | 1 | 0 | 13/6 |
| (12)(34) | 1 | 2 | 0 | 7/3 |
| (1234) | 1 | 2 | 1 | 10/3 |
| (123)(45) | 0 | 2 | 1 | 13/3 |
| (12345) | 0 | 2 | 0 | 10/3 |

A transposition exchanging a letter between the two blocks attains 13/6,
as does a three-cycle meeting both blocks. Thus the exact full-table
margin is **13/6**, for either number of operands.

## 3. Product decomposition and cost

An explicit orthonormal basis of the range of A is

\[
 f_1=(1,-1,0,0,0)/\sqrt2,\quad
 f_2=(0,0,1,-1,0)/\sqrt2,\quad
 f_3=(0,0,1,1,-2)/\sqrt6.
\]

For each f_alpha, expand

\[
 f_\alpha^T R_{a_1}\cdots R_{a_n}R_c^T f_\alpha
\]

over its nu-1 internal indices in any orthonormal basis of V. There are
4^(nu-1) neurons per alpha. Every factor is a matrix coefficient x^T R_g y
with unit x,y; its squared norm over the group is 120/4=30. The factors
are balanced, and every neuron costs 30 nu in P_2.

One more neuron, with each factor 2^(-1/nu) sgn(g), realizes the sign term.
Consequently

\[
 m=3\cdot4^{\nu-1}+1,
 \qquad P_2=90\nu\,4^{\nu-1}+120\nu\,2^{-2/\nu}.          \tag{3}
\]

The standard-sign schoolbook reference has margin 3 and
P_2=30 nu 4^nu. At nu=4 it is the exact entire symmetric-class optimum.

## 4. Degree four: unconditional counterexample

Here m=193 and

\[
 P_2=23040+240\sqrt2,\qquad
 \gamma=\frac{13/6}{(23040+240\sqrt2)^2}
        =3.9639207304\ldots\,10^{-9}.
\]

Against gamma_H=3/30720^2, the exact ratio is

\[
 R_4=\frac{13}{18}\left(\frac{128}{96+\sqrt2}\right)^2
       =1.2469416431\ldots.
\]

An entirely rational proof of a strict win uses sqrt(2)<3/2:

\[
 R_4>\frac{13}{18}\left(\frac{256}{195}\right)^2>\frac{31}{25}>1.
\]

The last strict inequality is an integer comparison. Thus the gain is
rigorously more than 24%, without relying on any decimal arithmetic.

## 5. Degree three: schoolbook beaten, true ceiling unresolved

Here m=49, P_2=4320+360*2^(-2/3), and

\[
 \gamma_3=\frac{13/6}{(4320+360\,2^{-2/3})^{3/2}}
        =7.0670043888\ldots\,10^{-6}.
\]

Relative to the schoolbook standard-sign value 3/5760^(3/2),

\[
 R_3=\frac{13}{18}\left(\frac{16}{12+2^{-2/3}}\right)^{3/2}
       =1.0297880111\ldots>1.
\]

For an exact comparison, 2^(2/3)>3/2 (cube both positive sides), so

\[
 R_3^2>\frac{169}{324}\left(\frac{24}{19}\right)^3>1.
\]

But at nu=3 the currently proven bound is only 4^(2/3)<=kappa(4)<=4.
Even the best pure standard-sign feature could exceed the schoolbook
reference if kappa(4)<4. Its comparison with this witness changes sign at

\[
 \kappa(4)=4/R_3^{2/3}=3.92248\ldots.
\]

The other irreps must also be accounted for to compare against the full
symmetric ceiling. If the schoolbook-price conjecture holds, this is a
degree-three counterexample as well. Without that conjecture, the result
proved here is specifically a win over the schoolbook reference.

## 6. Why this works

The character trace treats all four representation directions equally.
Removing the direction u reduces the product expansion from 4^nu to
3*4^(nu-1) standard neurons. Some wrong-label gaps shrink, but the amount
can be counted exactly by how many letters cross the 2+3 partition.
A single sign neuron repairs the smaller odd gaps. The resulting margin
loss is smaller than the gain in normalized representation cost.

This feature is not invariant under arbitrary conjugation: the partition
of the five letters is fixed. It carries same-irrep interference in the
standard block, so G2 does not apply to it. All neuron factors still lie
in standard or sign. The result refutes the unrestricted ceiling; it
does not show that any of these constructions is globally optimal.

Exact integer checks and an independent expansion of all network rows are
recorded in `s5_projector_verification.json`; the weights are saved in
`s5_projector_nu3.npz` and `s5_projector_nu4.npz`.

The degree-four comparison ceiling was also independently certified by
`s5_ceiling_verify.py`: it reconstructs the characters from permutation
fixed-point counts, enumerates all 43 programme vertices using Fractions,
and compares their costs using outward rational square-root intervals.
The unique winner is standard-sign alone, with exact ceiling 3/30720^2.
The certificate is in `s5_ceiling_verification.json`.

Reproduce from the project root:

```sh
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/s5_ceiling_verify.py
OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/s5_projector_verify.py
OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/s5_counterexample_verify.py
```
