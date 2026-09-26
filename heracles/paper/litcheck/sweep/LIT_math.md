# LIT_math: mathematics literature sweep, 2026-09-25

Scope: the maths behind Gamma_s(p) and the proofs in sections/approach.tex, sections/cyclic.tex and sections/appendix.tex. Every reference below comes from a page fetched this session; its URL is given. "Confirmed" means the title, authors, venue and year were read on a fetched page. "Via X" means the bibliographic data was read from X's reference list, not from the primary source. "Unconfirmed" means no bib entry should be made from these notes.

## 0. Headline: what already proves, or nearly proves, parts of the paper

1. **Unweighted Corollary `cor:few` and the Omega(log p) lower bound on K*, up to constants, are classical.** This is the equal-weight case, "K = o(log p) frequencies cannot cover every offset".
   - The weights y_j/2 on generators ±j define a Cayley graph on Z_p. Its Laplacian eigenvalue at character delta is exactly F_y(delta) = sum_j y_j cov_j(delta). So Gamma_s(p) is the *largest achievable spectral gap* of a weighted Cayley graph on Z_p, with the weights normalised in l_{1/s}. With equal weights, M_K(p) is the spectral gap of Cay(Z_p, ±S).
   - **Friedman, Murty and Tillich (JCTB 2006)**: a d-regular Cayley graph on an abelian group of order n has second eigenvalue >= d - O(d n^{-4/d}), with an absolute constant. With d = 2K this says M_K(p) = O(K p^{-2/K}). That is the unweighted analogue of the paper's (x/(e^{x/(2a_s)}-1))^s factor, with a *better* exponent: p^{-2/K} against the paper's roughly p^{-1/(2 a_s K)}.
   - **Alon and Roichman (1994), Proposition 3**: "For every fixed delta>0 there is c(delta)>0 such that ... if A is an Abelian group of order n and X(A,S) is a delta-expander, then |S| >= c log_2 n." This is proved by a diameter/volume count.
   - **Hermon and Olesker-Taylor (AAP 2023)**: for random abelian Cayley graphs the spectral gap has order |G|^{-2/k}.
   - Consequence: the paper must not present "K* = Omega(log p)" as new *in the unweighted/equal-share case*. What is new, as far as this sweep found, is the **weighted version with an l_{1/s} (s>1) budget**, the **matching upper bound K* = O(log p) with a share floor**, and the **transfer to the network margin gamma*_b = Theta(p^{-nu/2}(log p)^{1-nu/b})**.
2. **The Gamma_s lower bound (random allocation, Hoeffding plus a union bound) is Alon–Roichman Theorem 2 specialised to Z_p.** The paper already says so ("cyclic case of the random Cayley graph argument"). It is also Maurey's empirical method (Pisier 1981): the allocation y_j = n_j L^{-s} is an L-sample empirical measure. Consider citing Pisier as well.
3. **The positive-Fourier (Bessel) inequality eq:posfourier is a return-probability / closed-walk comparison.**
   - (1/p) sum_delta e^{-tF_y(delta)} is the return probability at time t of the continuous-time random walk on Z_p that jumps by ±j at rate y_j/2.
   - prod_j e^{-t y_j} I_0(t y_j) is the return probability of the lifted walk on Z^K, one independent coordinate per frequency.
   - The quotient map Z^K -> Z_p can only add returns ("resonances only add non-negative terms"). The same mechanism is **Cioabă (C. R. Acad. Sci. Paris 2006), Section 2**: closed walks on an abelian Cayley graph are at least the "balanced" tuples, i.e. the closed walks of the free abelian lift.
   - The Bessel generating function is DLMF 10.35.2 and the integral for I_0 is DLMF 10.32.1.
   - The weighted, exponential-moment form used to bound an l_r-constrained max-min seems to be new in this packaging. The method is not.
4. **Proposition `prop:shapes` (i), minimisers at vertices, is the standard concave-minimisation fact.**
   - For l_p (p<1) objectives it is **Ge, Jiang and Ye (Math. Program. 2011), Theorem 3**: "The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers." Their problem is equality-form and bounded. The paper's polyhedron is inequality-form, recession cone R_{>=0}, and the paper's own proof handles that. Cite GJY as precedent. GJY's strong NP-hardness of global l_p minimisation over polytopes also justifies the exhaustive and heuristic support search in Table tab:kstar. That is a useful sentence for reviewers.
   - Bauer's maximum principle (1958) covers the compact case.
5. **Proposition `prop:shapes` (ii), finitely many switches, is Descartes' rule of signs for real exponents after x = e^u.**
   - Koiran, Portier and Tavenas, Lemma 2 ("weak rule of signs"): sum_{i=1}^t a_i x^{alpha_i} with real alpha_i has at most t-1 positive roots. Their Lemma 1 gives the strong form: at most the number of sign changes. They credit Pólya–Szegő vol. II.
   - Curtiss (Ann. Math. 1918) proved the real-exponent case (via Feliu–Telek).
   - Suggestion: cite the rule instead of re-proving it by Rolle. The *strong* form gives a sharper count: switches between two vertices are at most the number of sign changes of c_k ordered by lambda_k.

Nothing found proves the nu-block optimality (Theorem thm:cyclic), the cost-of-a-feature Lemma lem:cost, the ablation bound (prop:ablation), the bottleneck-offset proposition, or the matching upper bound K* = O(log p). All searches for the weighted l_r-budget cosine-coverage programme itself came back empty (queries are listed in §9).

## 1. Random Cayley graphs and expanders (Theorem thm:scaling, lower bound)

**[AR94] N. Alon and Y. Roichman, "Random Cayley graphs and expanders", Random Structures & Algorithms 5(2):271–284, 1994.** DOI 10.1002/rsa.3240050203.
- Author PDF, fetched and read: https://web.math.princeton.edu/~nalon/PDFS/exp1.pdf. Journal data confirmed on Cioabă's reference list [1] and in the web-search index. The Wiley page returned 403.
- Theorem 1: c(delta) log_2 n random elements give E|mu_1^*| < 1-delta, for any group.
- Theorem 2: for abelian A, (1+eps) log_2 n random elements give |mu_1^*| <= 1-delta almost surely.
- Remark after Theorem 2: for cyclic Z_n, eps log_2 n random elements already give some fixed gap delta(eps).
- Proposition 3: every abelian delta-expander needs |S| >= c(delta) log_2 n.
- **Relevance:** this is the paper's `alon1994random`, already cited. Proposition 3 is the unweighted lower bound on K. Consider citing it at Corollary cor:few, not only at the Hoeffding step.

**[FMT06] J. Friedman, R. Murty and J.-P. Tillich, "Spectral estimates for Abelian Cayley graphs", J. Combin. Theory Ser. B 96(1):111–121, 2006.** DOI 10.1016/j.jctb.2005.06.012 (inferred from the pii; **the DOI string is unconfirmed**).
- URL: https://www.sciencedirect.com/science/article/pii/S0095895605000948. Returned 403.
- Data via Cioabă's reference [6], which reads "Spectral estimates of Abelian Cayley graphs", and via the search-index abstract: "for fixed d, a d-regular Cayley graph on an Abelian group of order n has second eigenvalue bounded below by d - O(dn^{-4/d}), where the implied constant is absolute."
- Full text not read.
- **Relevance:** this is the closest prior result to Corollary cor:few in the equal-share case. **Must cite.**

**[Cio06] S. M. Cioabă, "Closed walks and eigenvalues of Abelian Cayley graphs", C. R. Acad. Sci. Paris, Ser. I 342:635–638, 2006.** doi:10.1016/j.crma.2006.03.005.
- URL: https://www.numdam.org/item/10.1016/j.crma.2006.03.005.pdf. Read in full.
- Theorem 1.2: at least C(eps,k)|G| eigenvalues are >= k - eps.
- Section 2 lower-bounds closed walks by counting tuples with balanced ±x_i, which is the lift-to-Z^s argument.
- **Relevance:** method precedent for eq:posfourier.

**[HOT23] J. Hermon and S. Olesker-Taylor, "Geometry of random Cayley graphs of Abelian groups", Ann. Appl. Probab. 33(5):3520–3562, 2023.**
- URL: https://arxiv.org/abs/2102.02801, arXiv 2102.02801.
- Abstract as rendered: "Under certain density conditions on |G|, the spectral gap order is |G|^(-2/k)."
- **Relevance:** the sharp order of the gap for k random generators, matching FMT. It is optional context for how loose the constants of Theorem thm:scaling are.

**[HOT-cut] J. Hermon and S. Olesker-Taylor, "Cutoff for almost all random walks on Abelian groups", arXiv 2102.02809.**
- URL: https://arxiv.org/abs/2102.02809. The abs page says accepted at JEMS in 2025.
- **Relevance:** low. It compares walks on G with walks on Z^k, the same lift idea.

**[BKMT26] A. Basu, P. K. Kothari, R. Meka and S. Tudose, "Optimal sparsifiers for Abelian Cayley graphs", arXiv 2607.08261, July 2026.**
- URL: https://arxiv.org/abs/2607.08261, with HTML read.
- Theorem 1.5: every weighted Cayley graph on an abelian G has an eps-Cayley sparsifier with O(eps^{-2} log|G|) generators.
- Theorem 1.7: log|G| is necessary.
- **Relevance:** medium. Sparsifying the complete Cayley graph on Z_p, which is the s=1 optimum y uniform, to O(log p) weighted generators gives Gamma_s lower bounds of order (log p)^{1-s} with constants depending on eps. It is recent and weighted, and it shows that "O(log p) weighted generators suffice and are needed" is live in TCS. Cite as related. It is preprint-only.

## 2. Bohr sets, Dirichlet and rectification (the "why log p" intuition)

**[TV06] T. Tao and V. Vu, Additive Combinatorics, Cambridge Studies in Advanced Mathematics 105, Cambridge Univ. Press, 2006.**
- Data via Sanders, APDE 5(3) 2012, reference list: https://msp.org/apde/2012/5-3/apde-v5-n3-p05-p.pdf. Sanders cites "[Tao and Vu 2006, Lemma 4.20]" for a lower bound on Bohr-set size by averaging.
- The Bohr set {delta : ||j delta/p|| <= rho for all j in S} has size >= rho^{|S|} p, up to constants.
- **Relevance:** with rho ≈ p^{-1/K} this gives a nonzero delta with coverage O(K p^{-2/K}), the one-line proof of the unweighted Corollary cor:few. Suggest a remark: "for equal shares this is the Bohr-set/Dirichlet bound [TV06, Lemma 4.20]; eq:posfourier is its weighted analogue".
- Lemma number via Sanders; the book itself was not fetched.

**Dirichlet's simultaneous approximation theorem.** Page: https://en.wikipedia.org/wiki/Dirichlet%27s_approximation_theorem. The same pigeonhole idea. Cite TV06 rather than Wikipedia.

**[BLR98] Y. F. Bilu, V. F. Lev and I. Z. Ruzsa, "Rectification principles in additive number theory", Discrete Comput. Geom. 19(3):343–353, 1998.**
- Data via Bedert, Combinatorica 2023, reference [1]: https://d-nb.info/1316427056/34 (PDF read).
- Theorem 3.1 as quoted there: Z ⊂ Z/pZ with |Z| <= log_2 p is Freiman-isomorphic to a set of integers.
- **Relevance:** low to medium. It is another manifestation of the log_2 p threshold for Z_p. Optional.

**[Bed23] B. Bedert, "On unique sums in Abelian groups", Combinatorica, 2023,** doi:10.1007/s00493-023-00069-w. PDF: https://d-nb.info/1316427056/34. Low relevance. Only the rectification context above.

## 3. Small-bias sets and thin sets with small Fourier coefficients (coding side)

**[AIKPS90] M. Ajtai, H. Iwaniec, J. Komlós, J. Pintz and E. Szemerédi, "Construction of a thin set with small Fourier coefficients", Bull. London Math. Soc. 22(6):583–590, 1990.** doi:10.1112/blms/22.6.583.
- URL: https://academic.oup.com/blms/article-abstract/22/6/583/305433. Metadata confirmed; paywalled, content not read.
- **Relevance:** explicit sets of about log p residues with small |Fourier| are two-sided eps-biased sets. That is stronger than the one-sided condition needed for coverage. They give explicit candidates for near-optimal supports. Cite only as "explicit constructions exist". Do not claim constants.

**[RSW93] A. Razborov, E. Szemerédi and A. Wigderson, "Constructing small sets that are uniform in arithmetic progressions", Combin. Probab. Comput. 2(4):513–518, 1993.**
- URL: https://www.cambridge.org/core/journals/combinatorics-probability-and-computing/article/abs/constructing-small-sets-that-are-uniform-in-arithmetic-progressions/FB1C17433FC3F9CC6C0F831787669A41
- The rendered abstract says size "(log N)^o(1)"; the exponent was garbled in rendering. Low relevance.

**Small-bias lower bound Omega(n/(eps^2 log(1/eps))), Alon.** Only surfaced through search snippets (https://web.math.princeton.edu/~nalon/PDFS/identity1.pdf, not fetched). **Unconfirmed**; no bib entry. It is the two-sided analogue of the K lower bound as eps -> 0.

## 4. Chowla cosine problem (asked for; adjacent, not overlapping)

Chowla asks for the *minimum* of sum_{a in A} cos(a x), i.e. how negative it gets. The paper needs the *near-peak* behaviour: some delta≠0 where the positive-coefficient cosine sum is close to its value at 0. These are different extremal problems, and no Chowla result implies ours or the reverse.
- **[Bed25] B. Bedert, "Polynomial bounds for the Chowla cosine problem", arXiv 2509.05260** (v1 2025-09-05, v3 2026-07-24). https://arxiv.org/abs/2509.05260. Shows min_x f_A(x) <= -n^{1/5-o(1)}.
- **[JMTZ25] Z. Jin, A. Milojević, I. Tomon and S. Zhang, "From small eigenvalues to large cuts, and Chowla's cosine problem", arXiv 2509.03490.** https://arxiv.org/abs/2509.03490. This one connects Cayley-graph eigenvalues to Chowla.
- **[Sha26] A. Shankar, "A log-free n^{1/5} bound for Chowla's cosine problem", arXiv 2609.05338** (2026-09-04). https://arxiv.org/abs/2609.05338.
- Recommendation: do not cite in the main text. At most one clause in related work: "unlike Chowla's problem, which asks how negative a cosine sum must become, ...".

## 5. Bessel functions and the positive-Fourier step

**[DLMF] NIST Digital Library of Mathematical Functions.** https://dlmf.nist.gov/10.35 and https://dlmf.nist.gov/10.32.
- Eq. 10.35.2 (read): e^{z cos θ} = I_0(z) + 2 sum_{k>=1} I_k(z) cos(kθ).
- Eq. 10.32.1 (read): I_0(z) = (1/π) ∫_0^π e^{±z cos θ} dθ.
- **Use:** cite DLMF 10.35.2 for "the Fourier coefficients of e^{z cos θ} are I_k(z)". I_k(z) >= 0 for z >= 0 follows from the power series (DLMF 10.25.2, not fetched; standard).

## 6. Concave minimisation, sparse l_r (r<1) optimisation, vertex optimality

**[GJY11] D. Ge, X. Jiang and Y. Ye, "A note on the complexity of L_p minimization", Math. Program. 129:285–299, 2011.**
- Author PDF read: https://web.stanford.edu/~yyye/lpmin_v14.pdf. Volume and pages from the search index; the PDF is the preprint.
- Theorem 3: "The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers."
- Section 2: global L_p minimisation over {Ax=b, x>=0} is strongly NP-hard.
- **Relevance:** high as a citation for prop:shapes (i), and for justifying support enumeration in tab:kstar.

**[Bau58] H. Bauer, "Minimalstellen von Funktionen und Extremalpunkte", Arch. Math. 9(4):389–393, 1958.** doi:10.1007/BF01898615.
- Data via https://en.wikipedia.org/wiki/Bauer_maximum_principle.
- Statement: a convex, continuous function on a compact convex set attains its maximum at an extreme point. The compact case only.

**Rockafellar, Convex Analysis (Princeton, 1970), §32.** Section 32 treats the maximum of convex functions and handles closed convex sets containing no lines. **The theorem number is unconfirmed** (the JSTOR page failed to load); cite as "§32" at most.

**Horst and Tuy, Global Optimization: Deterministic Approaches (Springer).** Metadata from search only; **unconfirmed**. It would be the textbook reference for concave minimisation.

## 7. Descartes' rule for exponential sums (prop:shapes (ii))

**[KPT] P. Koiran, N. Portier and S. Tavenas, "A Wronskian approach to the real τ-conjecture", arXiv 1205.1015** (v3 2014).
- URLs: https://arxiv.org/abs/1205.1015 (abs) and https://arxiv.org/html/1205.1015v3 (HTML read).
- Lemma 1 (strong rule of signs) and Lemma 2 (weak rule, "also true in the case where the exponents α_i are real").
- Journal version: J. Symbolic Comput., ScienceDirect pii S0747717114001047. **Volume and pages unconfirmed.**

**[Cur18] D. R. Curtiss, "Recent extentions of Descartes' rule of signs", Ann. of Math. 19(4):251–278, 1918.**
- Data via Feliu–Telek, arXiv 2107.10002, reference [17]: https://arxiv.org/pdf/2107.10002 (read). Its introduction reads: "In 1918, Curtiss gave a proof that works for real exponents."
- Recommended classical citation.

**Pólya–Szegő, Problems and Theorems in Analysis, Vol. II, Springer 1976.** Via KPT reference [23]. The exact problem number is **unconfirmed**.

**[Jam06] G. J. O. Jameson, "Counting zeros of generalised polynomials: Descartes' rule of signs and Laguerre's extensions", Math. Gazette 90(518):223–234, 2006.**
- URL: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/CCA0F4742F4040DE65C7378D337DE475/S0025557200179628a.pdf/counting-zeros-of-generalised-polynomials-descartes-rule-of-signs-and-laguerres-extensions.pdf
- Metadata confirmed. The body was paywalled; the excerpt says the rule follows "by deduction from Rolle's theorem". This is the same proof as the paper's.

## 8. Maurey's empirical method, the Hoeffding sampling step

**[Pis81] G. Pisier, "Remarques sur un résultat non publié de B. Maurey", Séminaire d'Analyse fonctionnelle (Maurey–Schwartz) 1980–1981, exposé no. 5, 12 pp.**
- URL: https://www.numdam.org/item/SAF_1980-1981____A5_0/. Metadata confirmed.
- **Relevance:** sampling L atoms i.i.d. and taking the empirical measure is Maurey's lemma. The paper's y_j = n_j L^{-s} is exactly this. The concavity of t^{1/s} makes repeated draws free, which is the step "sum n_j^r <= sum n_j". Optional citation.

## 9. Cost-of-a-feature lemma (lem:cost): AM–GM, Hadamard, and the t^{-1-r} integral

- **Hadamard's inequality** (PSD matrix: det <= product of the diagonal; equality iff diagonal for PD). Page: https://en.wikipedia.org/wiki/Hadamard%27s_inequality. It points to Różański, Wituła and Hetmaniok, LAA 532:500–511, 2017 (via Wikipedia; unconfirmed beyond that). Textbook citation: Horn & Johnson, Matrix Analysis (theorem number unconfirmed).
  - **Check for the authors:** "equality iff diagonal" needs the Gram matrix to be nonsingular, or else det=0<1. The appendix states it for Gram matrices with unit diagonal, which is fine.
- **The integral representation of x^r.** ∫_0^∞ log(1+xt) t^{-1-r} dt = (π/(r sin πr)) x^r, and x^r = (r/Γ(1-r)) ∫(1-e^{-tx}) t^{-1-r} dt. These are the standard Bernstein-function representations of the fractional power. Reference: R. L. Schilling, R. Song and Z. Vondraček, *Bernstein Functions: Theory and Applications*, De Gruyter Studies in Mathematics 37 (2nd ed. 2012). Book page: https://www.degruyterbrill.com/document/doi/10.1515/9783110269338/html. Metadata from search only; **the exact formula number is unconfirmed**.
  - The step "prod_l (1+x_l t) >= (1+t)^ν for all t >= 0 implies sum_l x_l^r >= ν" did not turn up in any searched form, whether majorization, Newton/Maclaurin or elementary-symmetric. It looks like a clean, self-contained argument. Treat it as the paper's own.
- **AM–GM pricing of a product of ν factors,** the reason the cost is A^{b/ν}: classical, no citation needed. The dense-case constant is already in Morwani et al.

## 10. Searches that found nothing (evidence for novelty claims)

- "nonnegative coefficients cosine polynomial f(0)=1 minimize maximum over nonzero points of Z_p sparse number of terms log p": nothing relevant.
- "one-sided small bias set cyclic group Z_p ... lower bound size log p": only two-sided eps-bias results and Bedert's balanced-set log_2 p.
- "weighted abelian Cayley graph spectral gap ... sum of powers of weights ... logarithmic": only BKMT26 (sparsification, l_1-type weights), not an l_r (r<1) budget.
- "nonnegative weights cosine sum minimize peak sidelobe sparse ... log N ... Hoeffding": only window and sequence-design papers, no l_r budget or log-count theorem.
- "prod(1+x_i t) >= (1+t)^n implies sum x_i^r >= n": nothing.
- Descartes/exponential-sum and concave-vertex facts: classical, see §§6–7.

## 11. Recommended changes to the paper (math side)

1. At Corollary cor:few and after Theorem thm:scaling, add the following, or similar: "For equal shares, the statement that o(log p) frequencies cannot cover all offsets is the classical fact that abelian Cayley graphs with few generators are not expanders [AR94, Prop. 3; FMT06]; our bound extends it to weighted allocations under the l_{1/s} budget and transfers it to the margin." Without this, a reviewer who knows expanders will call Theorem thm:scaling "known".
2. In the proof of eq:posfourier, add one sentence: the left side is the return probability of a continuous-time walk on Z_p, and the right side is that of its lift to Z^K [Cio06; DLMF 10.35.2].
3. At prop:shapes (i), cite GJY11 Theorem 3, and optionally Bauer 1958 or Rockafellar §32. At (ii), cite the real-exponent Descartes rule (Curtiss 1918, or Pólya–Szegő vol. II, or KPT Lemma 2). Consider the strong, sign-change form.
4. At the lower bound, cite Pisier 1981 (Maurey) alongside Alon–Roichman.
5. Optionally, in related work, cite BKMT26 for weighted O(log|G|) generators, and AIKPS90 for explicit log-size supports.

## Bib entries (only for confirmed items)

```bibtex
@article{friedman2006spectral,
  title={Spectral estimates for {A}belian {C}ayley graphs},
  author={Friedman, Joel and Murty, Ram and Tillich, Jean-Pierre},
  journal={Journal of Combinatorial Theory, Series B},
  volume={96}, number={1}, pages={111--121}, year={2006}
}
@article{cioaba2006closed,
  title={Closed walks and eigenvalues of {A}belian {C}ayley graphs},
  author={Cioab{\u{a}}, Sebastian M.},
  journal={Comptes Rendus Math{\'e}matique. Acad{\'e}mie des Sciences. Paris},
  volume={342}, pages={635--638}, year={2006}, doi={10.1016/j.crma.2006.03.005}
}
@article{hermon2023geometry,
  title={Geometry of random {C}ayley graphs of {A}belian groups},
  author={Hermon, Jonathan and Olesker-Taylor, Sam},
  journal={Annals of Applied Probability},
  volume={33}, number={5}, pages={3520--3562}, year={2023}
}
@article{basu2026sparsifiers,
  title={Optimal sparsifiers for {A}belian {C}ayley graphs},
  author={Basu, Arpon and Kothari, Pravesh K. and Meka, Raghu and Tudose, Stefan},
  journal={arXiv preprint arXiv:2607.08261}, year={2026}
}
@book{tao2006additive,
  title={Additive Combinatorics},
  author={Tao, Terence and Vu, Van},
  series={Cambridge Studies in Advanced Mathematics}, volume={105},
  publisher={Cambridge University Press}, year={2006}
}
@article{ge2011lp,
  title={A note on the complexity of {$L_p$} minimization},
  author={Ge, Dongdong and Jiang, Xiaoye and Ye, Yinyu},
  journal={Mathematical Programming}, volume={129}, pages={285--299}, year={2011}
}
@article{bauer1958,
  title={Minimalstellen von {F}unktionen und {E}xtremalpunkte},
  author={Bauer, Heinz},
  journal={Archiv der Mathematik}, volume={9}, number={4}, pages={389--393}, year={1958},
  doi={10.1007/BF01898615}
}
@article{curtiss1918,
  title={Recent extentions of {D}escartes' rule of signs},
  author={Curtiss, D. R.},
  journal={Annals of Mathematics}, volume={19}, number={4}, pages={251--278}, year={1918}
}
@article{koiran2012wronskian,
  title={A {W}ronskian approach to the real $\tau$-conjecture},
  author={Koiran, Pascal and Portier, Natacha and Tavenas, S{\'e}bastien},
  journal={arXiv preprint arXiv:1205.1015}, year={2012}
}
@incollection{pisier1981maurey,
  title={Remarques sur un r{\'e}sultat non publi{\'e} de {B}. {M}aurey},
  author={Pisier, Gilles},
  booktitle={S{\'e}minaire d'Analyse Fonctionnelle (Maurey--Schwartz) 1980--1981},
  note={Expos{\'e} no. 5}, year={1981}
}
@misc{dlmf,
  title={{NIST} Digital Library of Mathematical Functions},
  note={\url{https://dlmf.nist.gov/}, Eqs. 10.32.1 and 10.35.2}
}
@article{ajtai1990thin,
  title={Construction of a thin set with small {F}ourier coefficients},
  author={Ajtai, Mikl{\'o}s and Iwaniec, Henryk and Koml{\'o}s, J{\'a}nos and Pintz, J{\'a}nos and Szemer{\'e}di, Endre},
  journal={Bulletin of the London Mathematical Society}, volume={22}, number={6}, pages={583--590}, year={1990},
  doi={10.1112/blms/22.6.583}
}
@article{bilu1998rectification,
  title={Rectification principles in additive number theory},
  author={Bilu, Yuri F. and Lev, Vsevolod F. and Ruzsa, Imre Z.},
  journal={Discrete \& Computational Geometry}, volume={19}, number={3}, pages={343--353}, year={1998}
}
```
Alon–Roichman and Hoeffding are already in refs.bib. The existing alon1994random entry should carry volume 5(2), pages 271–284, 1994; this was checked against Cioabă's reference list.
