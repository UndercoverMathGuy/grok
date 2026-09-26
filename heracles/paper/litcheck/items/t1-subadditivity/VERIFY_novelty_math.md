# VERIFY (novelty_math) — T1-subadditivity (task 42)

Claim (approach.tex, proof of thm:universal, l.64-67): with r=b/nu<=1, A_j <= sum_i |beta_ij|
and subadditivity of t^r give sum_j A_j^r <= sum_{i,j} |beta_ij|^r; "splitting a feature over
several neurons never makes it cheaper".

## Verdict: KNOWN (upheld)
The step is the elementary inequality (sum_i a_i)^r <= sum_i a_i^r for a_i>=0, 0<r<=1,
equivalently ||a||_1 <= ||a||_r (the l_r quasi-norm dominates l_1 for r<=1; the r-th power of
the l_r quasi-norm is subadditive). This is textbook (Jensen's inequality for sums /
p-normed spaces). The proof is correct as written (monotonicity of t^r + triangle inequality
+ subadditivity). Nothing to claim as new; the contribution is the use (no neuron hypothesis).

## Citations checked (fetched this session)
- ongie2026representation, arXiv 2606.14954 — EXISTS. Title/authors (Greg Ongie, Rahul Parhi),
  submitted 12 Jun 2026, correct. Downloaded PDF and grepped: Def of p-seminorm
  Q(f+g)^p <= Q(f)^p+Q(g)^p; Prop 3.6 (abs. p-homogeneous + subadditive K => K^{1/p} p-seminorm);
  Def 3.11/Prop 3.13 (parameter q-subadditivity => subadditive representation cost);
  Sec. 8 (shallow ReLU, q=p=1): "the subadditivity follows by concatenating neurons";
  Lemma 9.2 / Thm 9.3: depth-L ReLU R_L is a (2/L)-seminorm. All as the checker said.
  CAVEAT: their subadditivity is function-level and comes from concatenation
  (cost(f+g) <= cost(f)+cost(g), an UPPER bound on merged cost); the p exponent comes from
  homogeneity, not from scalar subadditivity of t^r. Our step is the LOWER bound
  (parameter cost >= l_r-quasinorm of amplitudes). Same p-normed geometry, different direction
  and mechanism. So the proposed sentence "This is the subadditivity that makes weight-decay
  representation costs ... p-norms" overstates the identity; say "the same p-normed
  (quasi-Banach) geometry appears in weight-decay representation costs of deep/product
  parametrisations" instead.
- gunasekar2018implicit — EXISTS. arXiv 1806.00468 abstract and NeurIPS proceedings page
  (Advances in NeurIPS 31, 2018) fetched: depth-L linear conv nets -> "l_{2/L} bridge penalty in
  the frequency domain". Bib correct (optionally add volume={31}). Good analog: our r=b/nu equals
  2/L with L=nu when b=2.
- Kalton, Peck, Roberts, An F-space Sampler (textbook suggestion) — bibliographic data now
  CONFIRMED on the Cambridge Core page (via DOI 10.1017/CBO9780511662447): N. J. Kalton,
  N. T. Peck, James W. Roberts; LMS Lecture Note Series 89; CUP 1984; ch. 8 "Operators between
  Lp spaces, 0<p<1". Specific page/lemma for p-normed inequality not read. Safe as a generic
  "p-normed spaces" reference; for the scalar inequality alone no citation is really needed.

## Missed prior art
None that changes the verdict. Searches were not possible for alternates (see below), so a more
elementary canonical source (Hardy-Littlewood-Polya, Inequalities, the monotonicity of
(sum a^r)^{1/r} in r) could not be fetched and is NOT recommended here.

## Workflow note (relevant to the slowness complaint)
WebSearch returned: "this session has used its web search budget (200 of 200 WebSearch calls)".
The verifier fan-out appears to share one per-session WebSearch cap, now exhausted; every
later verifier gets no search and must guess URLs for WebFetch (dblp is behind an Anubis wall).
Raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION or cap searches per verifier.
