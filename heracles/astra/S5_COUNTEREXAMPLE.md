# S5: an exact unrestricted counterexample at degree four

2026-09-24. This resolves the degree-four question in
`../s5theory/S5_THEORY.md`, §12.1. Degree three remains open.

**Stronger companion result:** `S5_PROJECTOR.md` gives a 193-neuron witness
with a 24.694% gain at degree four, and a 49-neuron degree-three witness
beating the schoolbook reference by 2.979%. The smaller 65-neuron witness
below remains useful as a separate, simple construction.

## Statement

For S5, three operands, degree nu = 4 and weight decay q = 2, there is
an explicit **65-neuron real product network** whose normalized margin is

\[
 \gamma=\frac{45/214}{(7680+1440/\sqrt{107})^2}
       =3.4393231281\ldots\,10^{-9}.
\]

The exact interference-free ceiling from G2 and K2 is

\[
 \gamma_H=\frac{3}{30720^2}=3.1789143880\ldots\,10^{-9}.
\]

Thus gamma/gamma_H = 1.0819175065... > 1.08. The unrestricted S5 analogue
of (T) is false at degree four. This is a construction and exact proof,
not a conclusion from numerical ascent. It does not identify the unrestricted
global optimizer or settle its feature count.

## 1. Replace a trace by a matrix coefficient

Let P_g be the natural permutation matrix on R^5, with P_g e_j = e_{g(j)}.
Let V = {x : sum x_j = 0}, and let R_g denote its four-dimensional standard
representation in any orthonormal basis. Set

\[
 v=(-10,-4,1,4,9),\qquad u=v/\sqrt{214}\in V,\qquad
 \beta=9/107.
\]

For three inputs a,b,d and label c define

\[
 z(a,b,d;c)=u^T P_aP_bP_dP_c^T u
       +\beta\,\operatorname{sgn}(a)\operatorname{sgn}(b)
                       \operatorname{sgn}(d)\operatorname{sgn}(c).
\]

Write D=abd c^{-1}. Then z = u^T P_D u + beta sgn(D). At the correct
label D=e the score is 1+beta. At an incorrect label its gap is

\[
 g(D)=\frac{214-v^TP_Dv}{214}+\beta(1-\operatorname{sgn}(D)).
\]

This depends on the offset D, but is not a conjugacy-class function.
That distinction is exactly where the symmetric-class hypothesis fails.

## 2. Exact margin

For every permutation D,

\[
 214-v^TP_Dv=\frac12\sum_j(v_j-v_{D(j)})^2.
\]

The smallest nonzero coordinate separation is 3. Every nonidentity
permutation moves at least two coordinates, so this quantity is at least 9;
the transposition exchanging the values 1 and 4 attains 9 and is odd.

Every nonidentity even permutation of five letters is a three-cycle,
a product of two disjoint transpositions, or a five-cycle:

* A three-cycle on ordered values x<y<z has gap
  (y-x)^2+(y-x)(z-y)+(z-y)^2. A minimizing triple can be chosen consecutive
  in the sorted list v. The three values are 91, 49, 49, so its gap is >=49.
* For two disjoint transpositions, fix their four ordered values. Pairing
  consecutive values minimizes the sum of the two squared distances;
  the other two pairings have larger distances term by term. Deleting
  each of the five possible unused values gives costs 50,146,61,100,45.
  The minimum is 45, attained by exchanging -10 with -4 and 1 with 4.
* A five-cycle has two edges incident to the value -10 and distinct
  neighbors. Those two edges alone contribute at least (6^2+11^2)/2
  =157/2>45 to the gap.

Consequently the minimum unpatched even gap is 45/214 and the minimum
odd gap is 9/214. Our sign coefficient adds 2 beta = 36/214 on the odd
offsets, giving the **exact full-table margin 45/214**.

## 3. Explicit product network and its penalty

Choose an orthonormal basis e_1,...,e_4 of V. Expanding the three internal
matrix indices gives 64 product neurons:

\[
 u^TR_aR_bR_dR_c^Tu
 =\sum_{i,j,k=1}^4
    (u^TR_a e_i)(e_i^TR_b e_j)(e_j^TR_d e_k)(u^TR_c e_k).
\]

Schur orthogonality for the standard representation gives, for any unit
x,y in V,

\[
 \sum_{g\in S_5}(x^TR_gy)^2=120/4=30.
\]

All four factors of each of the 64 neurons therefore have squared norm 30.
Each neuron is already balanced and costs 4*30=120 in P_2; their total
cost is 7680. A final neuron has all four factors beta^(1/4) sgn(g),
so its cost is 4*120*sqrt(beta)=1440/sqrt(107). This proves the stated P_2.

For comparison, the standard-sign character uses 4^4=256 schoolbook
neurons with the same factor norms. Its penalty is 30720 and its minimum
gap is 3. G2 and K2 certify that this is the entire interference-free
class's optimum, not just the score of one reference network.

## 4. Strict comparison without floating point

The ratio is

\[
 R=\frac{15}{214}\left(\frac{64}{16+3/\sqrt{107}}\right)^2.
\]

Since sqrt(107)>10,

\[
 R>\frac{6,144,000}{5,685,766}>\frac{27}{25}>1.
\]

The second comparison follows from
25*6,144,000 - 27*5,685,766 = 84,318 > 0.

## 5. What changes, and what does not

This network uses **standard + sign**, with 64+1 neurons. Its standard
block has same-irrep interference. No mixing with standard-sign inside
individual neurons is needed. The sign component handles the cheap
separation between the two parity cosets; the matrix coefficient distinguishes
the even offsets. Restricting standard to A5 gives precisely the suggested
four-dimensional A5 representation, but replacing a trace by a coefficient
is the useful reduction in representation cost.

The construction saves a factor four in the number of standard neurons.
At degree four this enters the normalized penalty as a factor sixteen,
which outweighs the smaller raw gap and the sign cost.

At degree three the same construction has 16+1 neurons and penalty
1440+360*(9/107)^(2/3). Against the standard-sign schoolbook reference its
ratio is only about 0.52. It therefore makes no claim at degree three,
where the true interference-free price also remains incompletely known.

Verification is in `s5_counterexample_verify.py`. Its exact integer checks
are supplementary to the proof above. Its separate floating-point pass
expands the actual neuron weights and checks every input row and wrong
label, without using the offset formula to calculate logits.
