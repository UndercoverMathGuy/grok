# T9-positive-fourier: literature check

Item: appendix app:scaling, eq:posfourier:
(1/p) sum_delta exp(-t F_y(delta)) >= prod_j e^{-t y_j} I_0(t y_j).
Reason given: e^{z cos} has non-negative Bessel Fourier coefficients, so resonances only add non-negative terms.

## Verdict: PARTIALLY_KNOWN (the technique is standard, the weighted/budgeted use is ours)

### What the inequality is
Take unit weights y_j = 1 on K frequencies. Then F_y(delta) = K - (1/2) lambda_chi, where lambda_chi is the adjacency eigenvalue of the Cayley graph Cay(Z_p, {+-j}) at character delta.
- The LHS is e^{-tK} times the exponential generating function of closed walks, i.e. the continuous-time heat-kernel return probability on the abelian Cayley graph.
- The RHS, prod e^{-t}I_0(t), is the same return probability on the free abelian cover Z^K.
- So the inequality says that return probability on the quotient is at least return probability on the cover Z^K. Closed walks on Z^K project to closed walks mod p, and extra relations ("resonances") only add walks.

### Prior art found (all fetched this session)
1. S. M. Cioaba, "Closed walks and eigenvalues of Abelian Cayley graphs", C. R. Acad. Sci. Paris Ser. I 342 (2006) 635-638, doi:10.1016/j.crma.2006.03.005 (numdam PDF fetched).
   - Sec. 2: closed 2r-walks correspond to 2r-tuples with product 1. Counting only the "trivial" tuples (each x_i paired with x_i^{-1}, each involution an even number of times) gives Phi_{2r}(X) >= n W_{2r}(X) (eq. 2, Lemma 2.1).
   - This is exactly our combinatorial/moment argument: trivial pairings = constant term, and other product-1 tuples only add.
   - Our Bessel form is its exponential generating function; sum over r of t^{2r}/(2r)! times the trivial count gives prod I_0.
   - Cioaba uses it for a Serre-type theorem: many eigenvalues >= k - eps.
2. J. Friedman, R. Murty, J.-P. Tillich, "Spectral estimates for Abelian Cayley graphs", JCTB 96 (2006) 111-121, doi:10.1016/j.jctb.2005.06.012 (author PDF fetched).
   - Abstract: a d-regular Abelian Cayley graph on n vertices has lambda_2 >= d - O(d n^{-4/d}), and this is tight for prime n.
   - Sec. 2: sphere-packing/pigeonhole proof.
   - Sec. 3: covering argument, which pushes a Dirichlet eigenfunction of the universal cover (Z^r) down to G.
   - Sec. 1 names the "large trace" approach as another route, citing Cioaba.
   - This is the unweighted, fixed-degree analogue of our upper bound on m(y). Our m(y) is the normalised weighted spectral gap: m(y) small means the weighted abelian Cayley graph is not an expander.
3. Alon & Roichman, "Random Cayley graphs and expanders" (RSA 1994; author PDF fetched).
   - Sec. 3: the log n generator bound is "essentially tight" for abelian groups. Prop. 2 is for Z_2^m only; the paper calls tightness trivial there.
   - FMT cite AR94 as proving the folklore result that abelian Cayley graphs cannot be good expanders, i.e. lambda_2 -> d.
   - The paper already cites alon1994random for the lower-bound half.

### Checked, not the same thing
- Chowla cosine problem (Bourgain 1986, Ruzsa 2004, Sanders, Bedert 2025 arXiv:2509.05260): these bound how NEGATIVE min_x sum cos(a x) must be, with unit coefficients. Our inequality controls the other side: some delta != 0 makes sum y_j cos(j delta) close to sum y_j (simultaneous Diophantine approximation / Bohr-set non-emptiness). Different direction and different technique (Bourgain and Ruzsa use L^1-norm / Riesz-product arguments). No overlap.
- Wang & Ye, arXiv:2503.06908: lower bound for the second-largest value of positive definite functions on finite abelian groups. Adjacent, since the Cayley eigenvalue list is a positive definite function. I read only the abstract and did not verify the content. It may be worth a look but is not needed.
- The Bessel generating function e^{z cos theta} = sum I_n(z) e^{in theta} is textbook (DLMF 10.35.1). I did not fetch DLMF this session.

### What is ours
- The weighted version, with an l_{1/s} (concave) budget on the weights, together with the scalar bound psi(z) <= a_s ln(1+z^r). This turns the heat-kernel comparison into m(y) <= C (ln p)^{1-s} for every feasible y, plus the K-dependent bound eq:Kbound. The published results are all for fixed degree d with unit weights, and give d - O(d n^{-4/d}). They have no budget-normalised, log p-scale statement and no bound on the number of frequencies.
- The "number of frequencies" consequence (K* = Theta(log p)) is the weighted analogue of the folklore statement that abelian Cayley graphs need Omega(log n) generators to expand. We should say this explicitly.

### Recommended change
After "No independence of the phases j delta is used: resonances only add non-negative terms." add:
"This is the analytic form of the closed-walk comparison for abelian Cayley graphs: extra relations can only add closed walks relative to the free cover Z^K \citep{cioaba2006closed}; cf.\ the covering argument of \citet{friedman2006spectral}, who show that unweighted $d$-regular abelian Cayley graphs on $n$ vertices have $\lambda_2\ge d-O(dn^{-4/d})$. Our contribution is the weighted, budget-normalised version and its $\log p$ consequence."
Optionally, in the main text next to Thm scaling, one sentence: "the upper bound is a weighted form of the folklore fact that abelian Cayley graphs need $\Omega(\log n)$ generators to expand \citep{alon1994random,friedman2006spectral}."

## Bib
@article{cioaba2006closed,
  author={Cioab\u{a}, Sebastian M.},
  title={Closed walks and eigenvalues of {Abelian} {Cayley} graphs},
  journal={Comptes Rendus Math\'ematique. Acad\'emie des Sciences. Paris. S\'erie I},
  volume={342}, number={9}, pages={635--638}, year={2006},
  doi={10.1016/j.crma.2006.03.005}}
(The issue number 9 is from memory and not confirmed. Volume, pages and year are confirmed from the PDF header.)

@article{friedman2006spectral,
  author={Friedman, Joel and Murty, Ram and Tillich, Jean-Pierre},
  title={Spectral estimates for {Abelian} {Cayley} graphs},
  journal={Journal of Combinatorial Theory, Series B},
  volume={96}, number={1}, pages={111--121}, year={2006},
  doi={10.1016/j.jctb.2005.06.012}}

## Searches run
- abelian Cayley graph spectral gap log n generators lower bound
- modified Bessel nonnegative Fourier coefficients exp(z cos) minimum cosine sum
- Alon Roichman abelian tight Klawe
- Friedman Murty Tillich spectral estimates abelian Cayley graphs
- continuous-time random walk abelian Cayley return probability I_0 covering Z^k
- Chowla cosine problem Bourgain Ruzsa Sanders
