# VERIFY novelty_math: R21-nu-block (task 18)

## Verdict
PARTIALLY_KNOWN. The checker's verdict stands, and all five citations it marked confirmed check out. The block is more classical on the mathematics side than the checker says, though:
- the minimal number of terms is a theorem, not only a conjecture, in the tied/symmetric case;
- at nu = 3 the block's norm-optimality is already in print.

What is ours: the embedding in product networks with the L_{2,b} accounting, the weights lambda_j^nu = 2^n y*_j / nu, and the fact that one block per support frequency attains Gamma_s.

## What the block is, in tensor language
One frequency of a product network's table is a real tensor in (R^2)^{tensor nu}, evaluated at the unit vectors (cos theta_t, sin theta_t). The target is cos(sum theta_t - theta_c), which is the tensor Re(z^{tensor n} tensor zbar) with z = e1 + i e2. The sign-flipped copies are the other components of the same tensor.

Using theta_nu := -theta_c, the output factor becomes cos(theta_c + n alpha_l) = (-1)^l cos(theta_nu + alpha_l). So prop:block is exactly

  sum_{l=0}^{nu-1} (-1)^l prod_{k=1}^{nu} cos(theta_k + pi l/nu) = nu 2^{1-nu} cos(sum theta_k).

This is the polarised (multilinear) form of the classical real Waring decomposition Re((x+iy)^nu) = (2^{nu-1}/nu) sum_l (-1)^l (cos(l pi/nu) x - sin(l pi/nu) y)^nu. I checked it numerically to 1e-15 for nu = 2..8. prop:block as written also checks to 1e-15.

## Prior art found (math lens), all read this session
1. **Mohlenkamp and Monzon 2005**, Math. Intelligencer 27(2):65-69, DOI 10.1007/BF02985795. Crossref confirmed; preprint PDF read.
   - Thm 2 holds for arbitrary distinct alpha_j, and there is a shifted version with sum a_j = 0.
   - They write that sin(sum x_j) "is exactly separated with separation rank n", against 2^{n-1} terms from the usual expansion.
   - Fig. 1 (right) uses alpha = 0, pi/3, 2pi/3.
   - Minimality: "We believe that our Theorems 2 and 5 are an example of such minimal representations" (a conjecture, citing Rassias-Simsa Problem 4).
   - I checked numerically that a_k = -alpha_k and alpha_k = pi k/nu, with prod_{k=1}^{nu-1} sin(pi k/nu) = nu/2^{nu-1}, give the product form above for nu = 2..8.
   - Cor. 6, the special case with all x_j equal, is credited to Calogero 1999.
2. **Comon and Ottaviani 2012**, Linear Multilinear Algebra 60(6):657-667, DOI 10.1080/03081087.2011.624097 (arXiv 0909.4865, read).
   - Prop 3.1: "a real binary form of degree d with d real distinct roots has real rank d."
   - Re((x+iy)^nu) has nu distinct real roots, at angles (2k+1)pi/(2nu). So in the tied/symmetric case, nu real powers are **necessary**, and the equally spaced decomposition has minimal length.
   - This is a theorem. The checker's statement that "the minimal-rank question is only conjectured" is true of Mohlenkamp-Monzon's non-symmetric separated representation, not of the symmetric one.
3. **Reznick 2013**, "On the length of binary forms", in Quadratic and Higher Degree Forms, Dev. Math., Springer, pp. 207-232, DOI 10.1007/978-1-4614-7488-3_8 (arXiv 1007.5485, read).
   - Thm 3.2 is a homogenised Sylvester 1864: if f has tau real linear factors, then any real representation sum lambda_j (cos theta_j x + sin theta_j y)^d has at least tau sign changes in (lambda_1, ..., lambda_r, (-1)^d lambda_1). So with nu terms the signs must alternate. This is the classical reason for the (-1)^l, which prop:block absorbs into the output phase n alpha_l.
   - Cor 5.6 / eq. (5.8) applies the same roots-of-unity filter with k+1 equally spaced directions to (x^2+y^2)^k. The technique is standard in real algebraic geometry.
4. **Friedland and Lim 2018**, "Nuclear norm of higher-order tensors", Math. Comp. 87:1255-1281, DOI 10.1090/mcom/3239 (arXiv 1410.6072, read).
   - Sec. 6: A = x(x)x(x)x - x(x)y(x)y + y(x)x(x)y + y(x)y(x)x = (1/2)(z(x)zbar(x)zbar + zbar(x)z(x)z) has rank_C 2 < 3 = rank_R, which they call "well-known". This A is the single-frequency modular-addition table cos(theta_1 - theta_2 - theta_3). So "3 real neurons per frequency are needed" is classical at nu = 3.
   - Lemma 6.1: B = (1/2) Im(z^{(x)3}) has ||B||_{*,R} = 2. Its symmetric real nuclear decomposition (31) uses 3 directions at 30, 150 and 270 degrees, which are equally spaced lines, each with weight 2/3. That is the nu = 3 block, together with its norm-optimality at b = nu.
   - For general nu, the same optimality follows in two lines. ||T||_* >= ||T||_F^2 / ||T||_sigma, with ||Re z^{(x)nu}||_F^2 = 2^{nu-1} and ||.||_sigma = 1. The block's coefficient l1 is nu * 2^{nu-1}/nu = 2^{nu-1}, which I checked numerically for nu = 2..6.
   - So at b = nu (r = 1), the per-frequency equality case of lem:cost is standard nuclear-norm duality. Only r < 1 needs the paper's own argument.
5. **Derksen 2016**, FoCM 16(3):779-811, DOI 10.1007/s10208-015-9264-x (arXiv 1308.3860, read). This is optional and belongs with cor:dense rather than prop:block.
   - Thm 1.13: the DFT decomposition, with one rank-1 term per frequency, is the nuclear decomposition of the Z_n structure tensor (the modular-addition table) over C, and ||T_{C_n}||_* = n sqrt n.
   - This is the complex analogue of "one block per frequency attains the b = nu optimum". Over R, each conjugate pair has to be split into a real block, as in items 2 and 4.

## Checker's confirmed citations, re-verified from PDFs fetched this session
- Morwani 2311.07568 App. F.2: "4(p-1) neurons: 8 neurons for each of the frequencies", with neuron pairs at (0,0,0)/(0,pi,pi) and so on. Confirmed.
- Li et al. 2402.09469 Lemma 4.3/E.3: "2^{2k-1} (p-1)/2 neurons", with 2^k trig terms times 2^{k-1} neurons each. Confirmed. It uses Lemma E.1 polarisation, not roots of unity.
- Gromov 2301.02679: the phases "are random, sampled from a uniform distribution". Confirmed. Fig. 4 caption: "The optimizer can tune phases better than random uniform distribution in order to ensure better cancellations."
- He et al. 2602.16849 Def 4.1(iii): for iota in {2,4}, the phase-moment sum vanishes. Confirmed, as is Prop 4.2 (the 2x/2y peaks survive).
- Mohlenkamp-Monzon: see above. The bibtex fields match Crossref.

## Issues with the checker
- It says "None of the ML constructions uses a nu-neuron roots-of-unity block". That is false at nu = 3: Tian 2410.01779 was found by the novelty_ml verifier (task 17), and I did not re-read it here.
- It says "minimal-rank question is only conjectured". That is incomplete. In the symmetric case, minimality of nu is a theorem (Comon-Ottaviani Prop 3.1, via Sylvester). The nu = 3 non-symmetric real rank 3 is classical (Friedland-Lim Sec. 6).
- It does not note that norm-optimality at b = nu is nuclear-norm duality, which Friedland-Lim state explicitly at nu = 3.
- The paper should therefore not present "nu neurons suffice", "the equally spaced phases cancel every sign-flipped copy", or b = nu cost-optimality per frequency as new. Novel: the L_{2,b} network accounting, lambda_j^nu proportional to y*_j, attainment of Gamma_s, and the r < 1 cost lemma.

## Recommended text (extends checker's)
After "The phases are a rotated copy of the nu-th roots of unity...", add:
"The table identity is the polarised form of the classical nu-term real Waring decomposition of Re((x+iy)^nu), the equally spaced case of \citet[Thm 2]{mohlenkamp2005trig}. Since Re((x+iy)^nu) has nu real roots, no tied decomposition uses fewer terms \citep[Prop 3.1]{comon2012typical}. For nu=3 the decomposition is a real nuclear decomposition \citep[Lemma 6.1]{friedland2018nuclear}."
Keep the checker's contrasts with Morwani (8 neurons), Li et al. (2^{2k-1}) and Gromov (random phases), and add Tian for nu = 3 (task 17).

## Search notes
WebSearch was refused because the session budget was exhausted (200/200). I used the arXiv API, Crossref, the StackExchange API (no hits) and curl+pdftotext on primary PDFs. DuckDuckGo returned a CAPTCHA, and Bing gave irrelevant results.
