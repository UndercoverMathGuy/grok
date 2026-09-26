# T9-positive-fourier: verification, novelty_math lens (task 58)

## Result
- The checker's label PARTIALLY_KNOWN is correct.
- Both of the checker's confirmed citations hold up.
- Its account of what is new is WRONG. The checker says "published results are for fixed degree with unit weights" and "what is new is the weighted version". A 2025 paper already proves a weighted version (Wang and Ye; details below).
- The recommended sentence has to be narrowed before it goes into the paper.

## Citations checked (both PDFs fetched and converted to text this session)
1. **Cioaba, C. R. Acad. Sci. Paris Ser. I 342 (2006) 635-638, doi:10.1016/j.crma.2006.03.005.** Confirmed.
   - Sec. 2: closed r-walks correspond one-to-one with r-tuples whose product is 1.
   - W_{2r} counts the tuples in which each x_i appears as often as x_i^{-1}.
   - Eq. (2) gives Phi_{2r}(X) >= n W_{2r}(X), and Lemma 2.1 gives the formula for W_{2r}. Thm 1.2 is Serre-type.
   - The exponential generating function of the balanced-word count is prod I_0(2t), which is our right-hand side with unit weights. So the "resonances only add" sentence is correctly attributed.
   - The issue number is not on the PDF header, so leave `number` out.
2. **Friedman-Murty-Tillich, JCTB 96 (2006) 111-121, doi:10.1016/j.jctb.2005.06.012.** Confirmed.
   - Abstract: lambda_2 >= d - O(d n^{-4/d}), and this cannot be improved for odd prime n.
   - Sec. 3 is titled "A covering argument". It pushes a Dirichlet eigenfunction of the universal cover forward to G.
   - Sec. 1 calls "Abelian Cayley graphs cannot be good expanders" folklore, citing [AR94, **Proposition 3**, Section 3]. CHECK.md says "Prop. 2"; FMT's own pointer is Prop. 3. This is minor.
   - Sec. 1 also names the large-trace approach [Ci05].

## Missed prior art (closer than anything the checker cited)
**Wang & Ye, "An estimate for positive definite functions on finite abelian groups and its applications", arXiv:2503.06908 (math.FA, 10 Mar 2025).** Full PDF fetched and read. The checker read only the abstract and called it "adjacent".
- It is not just adjacent. Thm 4.6 and Prop 4.5 bound the second-largest value nu_2(f) for ANY real positive definite f, i.e. any function with nonnegative Fourier coefficients.
  - Take f = zeta = sum_j y_j cos(2 pi j delta/p), whose coefficients are y_j/2 >= 0, on G = Z_p. Then t = 0 and r = K.
  - Since kappa_j = 1 - cos, we get m(y) = nu_1(f) - nu_2(f).
- Thm 4.6, eq. (11), then reads: for any integer m with (m-1)^K < p,
  m(y) <= [2p sin^2(pi/2m) / (p - (m-1)^K)] * ||y||_1.
  This gives m(y) = O(||y||_1 p^{-2/K}) for arbitrary nonnegative weights. That is a weighted, support-size-dependent version of the bound the checker attributes only to unit-weight FMT.
- Prop 4.5 lets the radius m_i vary from frequency to frequency.
- Sec. 4.1 is titled "A generalized covering argument", described as "grounded in Fourier analysis".
- Prop 4.11 applies this to weighted Cayley graphs Cay(G, w). Prop 5.5 gives relaxation time Omega(|G|^{4/s}) for any probability measure with support size <= s.
- Remark 5.6 cites earlier weighted and probabilistic versions that we did NOT fetch: Greenhalgh (1989 thesis), Hough (EJP 2017) giving T_rel = Omega(p^{2/s}) on Z/pZ, and Hermon and Olesker-Taylor (AAP 2023, Thm 6.1).

What remains ours (my own derivation, not a citation):
- Wang-Ye bound through ||y||_1 and the support size. Under our concave budget sum y^r <= 1 with r < 1, we have ||y||_1 <= 1. Thm 4.6 then gives only K >~ ln p / ln ln p for a maximizer, and no bound that holds for every support size.
- Our eq:posfourier plus eq:psisum gives a support-free bound, m(y) <= C (ln p)^{1-s} for every feasible y, and K >= c_s ln p. Wang-Ye does not state these.
- Prop 4.5 with variable radii might reproduce something like eq:Kbound up to constants. I did not work this through.
- The Bessel exponential-moment route itself is different from their test-function pushforward.

## Other checks
- The Chowla / Bourgain / Ruzsa / Sanders line bounds how negative a cosine sum must get. The direction is opposite to ours, so I agree there is no overlap.
- The unit-weight log p statement also follows from Dirichlet's pigeonhole (Bohr-set size >= rho^K p; Tao-Vu, Additive Combinatorics, Lemma 4.20). I could not fetch a source, so this is not a citation.

## Recommended wording (replaces the checker's last sentence)
"...\citep{cioaba2006closed}; cf.\ the covering arguments of \citet{friedman2006spectral} (unit weights) and \citet{wang2025positive}, who bound $\min_{\delta\ne0}\sum_jy_j\kappa_j(\delta)=O(\|y\|_1p^{-2/K})$ for arbitrary non-negative weights on $K$ frequencies. What is new here is a bound that does not depend on the support size, under the concave budget $\sum_jy_j^{1/s}\le1$, and its $\log p$ consequence."

## Harness note (why this workflow is slow)
- WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)". The budget is shared across the whole session, so every remaining verifier has to work without search.
- The arXiv export API returned HTTP 429, and later calls came back empty. Parallel verifiers are being rate-limited by arXiv.
- All subagents share one scratchpad directory, which already holds generic names such as a.txt and c.txt, so their files can overwrite each other.
