# VERIFY (novelty_math): T4 roots-of-unity phases (task 48)

Claim checked: in Proposition prop:block (cyclic.tex l.52-58) and the appendix remark (appendix.tex l.79-81), nu product neurons with phases alpha_l = pi l/nu cancel every sign-flipped product-to-sum copy.

Checker verdict: PARTIALLY_KNOWN. **Upheld.** The checker missed two pieces of mathematical prior art, and the recommended text needs two fixes.

Note: the WebSearch budget for this session was exhausted. Searches went through the arXiv API, OpenAlex (including works citing Mohlenkamp-Monzon), and Crossref. Every source below was downloaded and read.

## Numerical checks (my own, nu = 2..7)
- The block identity holds: sum_l prod_t cos(th_t + a_l) cos(th_c + n a_l) = nu 2^{-n} cos(sum th - th_c).
- The block is a *symmetric* real Waring decomposition of the harmonic binary form:
  Re((x+iy)^nu) = (2^{nu-1}/nu) sum_{l=0}^{nu-1} (-1)^l (x cos(pi l/nu) - y sin(pi l/nu))^nu.
  Why: cos(th_c + n a_l) = (-1)^l cos(-th_c + a_l), so every factor of neuron l is the same linear form v_l.
- Mohlenkamp-Monzon Thm 2 (sin) with alpha_j = pi j/nu holds, and all nu coefficients have equal modulus (4/3, 2, 16/5, ...). This is their "amplitude equidistributed". Choosing their free shifts a_k = mean(alpha) - alpha_k makes all phases within a term equal, which gives exactly our block.

## Confirmed citations (re-fetched and read)
1. **mohlenkamp2005trig**
   - Crossref: Math. Intelligencer 27(2):65-69, 2005, doi 10.1007/BF02985795. OK.
   - Preprint read: Thm 2 is the rank-n identity for sin of a sum. Fig. 1 (right) uses alpha = 0, pi/3, 2pi/3, "amplitude equidistributed". The Remarks say it extends to cos. They only *believe* rank n is minimal (Rassias-Simsa, Problem 4).
   - CONFIRMED.
2. **gromov2023grokking** (arXiv 2301.02679)
   - Eq. (12) is phi1+phi2=phi3. Eqs. (14)-(15) use i.i.d. uniform phases, and the sum is << N (approximate).
   - Quote: "each frequency appears several times with different random phases ultimately leading to a better wave interference". Fig. 4 caption: "The optimizer can tune phases better than random uniform distribution".
   - CONFIRMED.
3. **morwani2024feature** (arXiv 2311.07568), App. F.2
   - 8 neurons per frequency for the *quadratic* network. Each of the 4 terms of cos(a+b-c) comes from a sign pair ((u+v)^2 - (u-v)^2), with phase triples (0,0,0)/(0,pi,pi), etc.
   - App. C parity: "every term other than Pi x_{S_i} cancels out in the sum".
   - CONFIRMED.
4. **li2025nsum** (arXiv 2402.09469), Lemma E.3
   - 2^{2k-1} neurons per frequency for the *power* network (sum u)^k w with k inputs, i.e. 2^{2nu-3}.
   - Construction: 2^k terms of Eq. (25), each built from 2^{k-1} neurons via 2^k k! prod d_i = sum_{c in {+-1}^k} (-1)^{(k-sum c)/2} (sum c_j d_j)^k.
   - CONFIRMED.

## Prior art the checker missed (all read this session)
A. **Friedland & Lim, "Nuclear norm of higher-order tensors", Math. Comp. 87(311):1255-1281, 2018, doi 10.1090/mcom/3239 (arXiv 1410.6072v3, Lemma 6.1).**
   - B = (1/2)(e1e1e2 + e1e2e1 + e2e1e1 - e2e2e2) = (1/2) Im((e1+ie2)^{(x)3}). This is the nu=3 target tensor up to rotation.
   - Eq. (31) gives a *symmetric nuclear decomposition over R* with three equal weights 2/3. Its vectors are at 30, 150 and 270 degrees, i.e. lines equally spaced by pi/3. This is exactly the nu=3 roots-of-unity block.
   - ||B||_{*,R} = 2, and ||B||_{*,C} = sqrt 2.
   - So at nu=3 the block's **norm optimality** is already known in the r=1 case (nuclear norm = the paper's cost at b=nu). The paper's cost Sum|beta_l|^r at pure-frequency neurons is the l_r quasi-nuclear norm of Re(w^{(x)nu}).
   - What stays new: general nu, the r<1 case, and the uniqueness/equality characterisation (exactly nu terms, orthogonal chain vectors). At r=1 the 4-term expansion (their Eq. 30) is also a nuclear decomposition, which matches the paper's remark that at b=nu only a same-phase condition is needed.
B. **Real Waring rank of binary forms with all real roots.**
   - Comon & Ottaviani, Linear Multilinear Algebra 60(6):657-667, 2012, Prop. 3.1: "a real binary form of degree d with d real distinct roots [has] real rank ... equal to d". Their Section 2 example: 2x^3 - 6xy^2 = 2 Re((x+iy)^3) has complex rank 2 and real rank 3. Causa & Re, Ann. Mat. Pura Appl. 190(1):55-59, 2011, complete their argument.
   - Reznick, "On the length of binary forms" (Quadratic and Higher Degree Forms, Springer 2013, pp. 207-232, doi 10.1007/978-1-4614-7488-3_8; arXiv 1007.5485), Cor. 4.11: the real length of a product of d real linear forms equals d. His Thm 5.5/Cor 5.6 are roots-of-unity decompositions, eq. (5.8) with angles j pi/(k+1).
   - Re((x+iy)^nu) has nu distinct real roots. So the fact that nu terms are needed and suffice is classical *for symmetric decompositions*. The paper's neurons are general (CP) rank-one terms, so the paper's count statement is stronger. The equally-spaced block, however, is one of the classical symmetric minimal decompositions.
C. **Roots-of-unity filter / series multisection.** Textbook. The paper's proof (appendix l.109) is exactly this.

## Residual novelty
- The identity is known: Mohlenkamp-Monzon 2005, and equivalently a real Waring decomposition of Re((x+iy)^nu).
- nu=3 norm-optimality in the nuclear norm is known: Friedland-Lim 2018.
- The minimal count nu is known in the symmetric setting.
- New as far as I can find:
  - the l_r (r<1) cost lemma for general nu, with its equality case (exactly nu equal-modulus terms, pairwise orthogonal chain vectors, equivalently roots of unity);
  - its use to certify the interference-free margin optimum.
- I found no ML paper that uses a nu-neuron roots-of-unity block. The ML constructions use 8 or 2^{2nu-3} neurons, or random phases.

## Issues with the recommended change / paper text
1. **Not like-for-like.** The recommended sentence compares "8 neurons (Morwani) / 2^{2nu-3} (Li)" against nu. Those counts are for quadratic and power networks, where the sign pairs also cancel squared / zero-slot monomials. The paper's own appendix (l.257) says the product block does not directly give an attaining power network.
   Suggested phrasing: "the +-1 constructions of \citet{morwani2024feature,li2025nsum} for quadratic and power networks use 8 and 2^{2nu-3} neurons per frequency; in a product network the textbook product-to-sum expansion alone uses 2^{nu-1}".
2. **Wording in cyclic.tex l.58.** "The phases are a rotated copy of the nu-th roots of unity" is imprecise. e^{i alpha_l} = e^{i pi l/nu} are 2nu-th roots of unity (half of them). It is the doubled phases e^{2i alpha_l}, which give the ratios xi in the remark, that are the nu-th roots of unity.
   Suggest: "The phases are equally spaced by pi/nu, so the doubled phases 2 alpha_l are the nu-th roots of unity (the configuration of the remark in App. ...)".
3. **Add citations.** Cite Friedland-Lim (Lemma 6.1) next to the norm-optimality sentence, e.g. "at nu=3 and b=nu this is the real nuclear decomposition of Friedland & Lim (2018, Lemma 6.1)". Optionally cite Reznick (2013, Cor. 4.11) or Comon-Ottaviani (2012, Prop. 3.1), for the fact that a symmetric decomposition of Re((x+iy)^nu) needs nu terms.
