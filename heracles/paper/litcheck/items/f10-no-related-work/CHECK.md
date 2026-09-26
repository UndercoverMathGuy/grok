# F10: related work missing

## What the source contains (checked with grep, 2026-09-25)
- intro.tex:22 has the placeholder `% TODO(user): related work (Gromov 2023; Li et al. 2025; McCracken et al. 2025; Lyu & Li 2020)`. There is no related-work section anywhere.
- These bib entries are never cited in paper.tex or sections/: lyu2020gradient, gromov2023grokking, li2025nsum, power2022grokking. li2025nsum is cited only in archive_v0.
- The mccracken2025 entry is WRONG: it gives the title "Emergence of sparse and dense circuits..." and the author "Andrew McCracken". The paper at 2505.18266 has a different title and authors (see below).
- Verdict: SUPPORTED. The gap is real, and the papers below must be positioned against.

## Verified sources (all fetched this session)
1. Li, Liang, Shi, Song, Zhou. "Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs". AISTATS 2025, per the arXiv comment on 2402.09469v4.
   - Thm 4.1: one hidden layer, neurons of the form (u_1^T x_1 + ... + u_k^T x_k)^k w, with ||theta||_{2,k+1} <= 1.
   - Max margin gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}).
   - "For each frequency zeta in {1..(p-1)/2}, there exists one neuron using this frequency only", i.e. it is DENSE. It needs m >= 2^{2k-1}(p-1)/2 neurons.
   - Position: this is the b = nu endpoint for power neurons, which our universal bound recovers. They own the x^n / general-nu dense result. We own b < nu, where the solution is sparse.
2. McCracken, Moisescu-Pareja, Letourneau, Precup, Love. "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks". arXiv 2505.18266, 23 May 2025.
   - Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)". This margin is unnormalised, and the result is sufficiency only.
   - One-hidden-layer one-hot MLPs learn floor(n/2) neuron types. With depth or trainable embeddings the count transitions to "much fewer". They say "we don't know why it occurs".
   - Position: precedent for the random-log-p lower bound. Our results answer their open question through the norm exponent.
3. Jacobs, Montufar. "Implicit Bias of Mirror Flow in Homogeneous Neural Networks: Sparse and Dense Feature Learning". arXiv 2605.19458, 19 May 2026.
   - Thm 4.9 reformulates max-margin as min sum |a_j|^{alpha/2}. alpha = 1 gives sparse and alpha >= 2 gives dense.
   - No modular arithmetic or Fourier content.
   - Position: they own the generic claim "a homogeneity exponent decides sparse vs dense". Ours is specific to the task: exact counts, K* = Theta(log p), and the answer is set by the norm and not by the optimiser.
4. Tian. "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets". arXiv 2410.01779 (v1 Oct 2024), NeurIPS 2025 camera ready.
   - Two-layer quadratic nets on abelian groups. The solution space has semi-ring structure, and global solutions are composed from partial ones.
   - "weight decay favors simpler solutions". No count law.
5. Zhong, Liu, Tegmark, Andreas. "The Clock and the Pizza". NeurIPS 2023.
   - Hyperparameters and initialisation change which algorithm is found (Clock vs Pizza).
   - Position: mechanisms differ across architectures. Ours is a count claim at the max-margin endpoint.
6. Doshi, He, Das, Gromov. "Grokking Modular Polynomials". arXiv 2406.03495, June 2024. Venue unconfirmed: Semantic Scholar returned 429, and the arXiv page gives no venue.
   - Analytic solution for S-term sums: 2-layer MLP with phi(x) = x^S, cosine weights, and a frequency sigma(k) per neuron. Cross-terms vanish "due to random phase approximation" as width N grows.
   - Constructive, with no norm and no margin. This is the analytic precedent for the n-operand (nu = n+1) family.
7. Gromov. "Grokking modular arithmetic". arXiv 2301.02679, 2023. Analytic weights and feature maps, found by GD and AdamW. The analytic precedent for phases.
8. Lyu, Li. "Gradient Descent Maximizes the Margin of Homogeneous Neural Networks". ICLR 2020. GD/GF on logistic/CE loss increases a smoothed normalised margin, and converges to KKT points of L2 max-margin.

## Recommended related-work paragraph (appendix if space is tight)
The paper should state, in order:
- Max-margin feature theory: Morwani et al. give the quadratic dense result at L_{2,3}. Li et al. 2025 extend it to power neurons at L_{2,k+1}, where every frequency is used. Our b = nu case recovers both.
- Implicit bias toward margin: Wei et al. 2019 and Lyu-Li 2020. Only KKT points are guaranteed, so we do not claim that GD finds the optimum.
- Analytic constructions: Gromov 2023; Doshi et al. 2024 for the many-term sum.
- Counts observed in trained nets: Nanda; Chughtai; Zhong et al. 2023 for algorithm diversity; McCracken et al. 2025 (O(log n) frequencies suffice; the dense-to-sparse transition is unexplained).
- The exponent deciding sparse vs dense, in general: Jacobs-Montufar 2026 (mirror potentials). Algebraic solution structure: Tian 2025.
- Also: power2022grokking is uncited. Cite it in the intro sentence about grokking, or drop it.
