# VERIFY (novelty_ml): T12-dense-uniqueness, task 61

Claim (app:dense, Cor. cor:dense): Gamma_1(p) = p/(p-1). The flat vector y_j = 2/(p-1) is the unique maximizer. The proof averages the coverage over delta != 0, then applies Fourier inversion to zeta = c0 + (1-c0)[delta=0].

## Checker's verdict: PARTIALLY_KNOWN. Upheld, with a missed precedent and one overstatement.

### Morwani et al., arXiv 2311.07568 (ICLR 2024 spotlight). Confirmed.
- I fetched the abs page myself. Title and authors match refs.bib. v2 is dated 19 Feb 2024, and the comments field reads "Accepted as Spotlight at ICLR 2024".
- I read the full text. The Sec. 4 proof outline says: "We will use the uniform class weighting: tau(a,b)[c'] := 1/(p-1) for all c' != a+b". Lemma 6 is on text line ~453.
- App. F.3, "Proof that all frequencies are used": by Lemma 6, f = F(a,b) + lambda*1_{a+b=c}. So f2^(j,k,l) = lambda p^2 when j=k=-l, and f1^ is zero unless l=0. From this they conclude "at least one neuron for each frequency".

### MISSED: Li, Liang, Shi, Song, Zhou, arXiv 2402.09469 (AISTATS 2025)
- I fetched the abs page myself. v4 is dated 9 Mar 2025, with the comment "AIStats 2025". This is `li2025nsum` in refs.bib.
- Lemma F.2, "All Frequencies are Used for General k Version", applies the same argument to the k-operand x^k network at L_{2,k+1}. That is b = nu with n = k operands, which is this paper's setting. It uses the uniform class weighting 1/(p-1) and the decomposition f = f1 + lambda*1_{a1+...+ak=c}, which gives f2^ = lambda p^k at every diagonal (j,...,j,-j). The construction (App. E.2/E.3) then attains the maximum using every frequency with the same cosine weight.
- **The current source cites li2025nsum nowhere.** It appears only in refs.bib, the stale paper.bbl and archive_v0/theory.tex. The n-operand, b = nu case of this corollary is Li et al.'s setting, so they must be cited here.

### Overstatement: "the new part is equal amplitudes"
The decomposition used by Morwani (F.3) and by Li et al. (F.2) gives the same value at every diagonal frequency. For j != 0, f^(j,...,j,-j) = f2^ = lambda p^n, because f1^ vanishes off l = 0. In this paper's notation that is G_j := f^_theta(j,...,j;-j), and beta_j = 2 p^{-(n+1)} G_j. So equal aggregate amplitude in every max-margin network already follows in one line from their argument. They do not state it, but it is not "stronger" than their argument. What is actually new:
1. The statement is about the reduced LP Gamma_1, and gives a unique maximizer of that programme.
2. It is network-agnostic: it holds for any homogeneity-nu network through Thm universal, including product networks, which neither paper treats.
3. The one-variable inversion replaces the (n+1)-dimensional DFT.

### Other ML candidates checked (none closer)
- Mohamadi, Li, Wu, Sutherland, arXiv 2407.12332 (ICML 2024). I fetched the abstract. It covers the kernel regime and l_inf-bounded quadratic nets, with no flat-spectrum or uniqueness claim.
- Tian, arXiv 2509.21519 (Li^2 scaling laws). Its "unique maximizer spreads" result (A.4) is about a memorization KKT problem, not about Fourier flatness.
- He et al. 2602.16849: its "fully diversified" definition assumes equal per-neuron magnitudes across frequencies. That is a construction, not uniqueness. Its text is in the scratchpad; I did not cite it.

### Recommended sentence (replaces the checker's)
"The averaging step is the uniform class-weighting certificate of \citet{morwani2024feature} (Lemmas 5--6). The Fourier-inversion step is their App. F.3, extended to $n$ operands in \citet[Lemma F.2]{li2025nsum}. Their decomposition $f=F+\lambda\mathbf 1_{a_1+\dots+a_n=c}$ already gives every diagonal Fourier coefficient the value $\lambda p^n$. What is new here is the statement for the reduced programme $\Gamma_1$, which has a unique flat maximizer, and the fact that it holds for every network of homogeneity $\nu$ through Theorem~\ref{thm:universal}."

### Process note, for the slow-workflow question
This session has used all of its WebSearch budget: "used its web search budget (200 of 200 WebSearch calls)", seen on this task's first search. From then on, every verifier in the run gets no searches. It can only use WebFetch, or text already downloaded into the shared scratchpad. That produces slower, lower-recall runs and retries. To lift the cap, raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or shard the verify stage across sessions.
