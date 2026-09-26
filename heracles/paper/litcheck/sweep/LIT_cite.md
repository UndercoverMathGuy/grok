# LIT_cite: citation-chasing sweep (2026-09-25)

Scope: forward and backward citations of Morwani et al. 2024, Nanda et al. 2023, Chughtai et al. 2023, McCracken et al. 2025, Gromov 2023, Wei et al. 2019, Lyu & Li 2020, Li et al. 2025, plus 2024-2026 papers on feature counts and sparsity in modular arithmetic and group tasks.
Method: Semantic Scholar citation lists (full lists for Nanda (991), Wei (277), Lyu-Li (434), Chughtai, Gromov, McCracken and Li; Morwani's list came back with only ~40 rows, so treat it as a sample, and the API rate-limited later queries), filtered by keyword. Then WebSearch, then WebFetch of every paper recommended below (arXiv abs/html/pdf, PMLR, NeurIPS/ICLR virtual pages, Crossref, GitHub).
Every entry marked VERIFIED had its title, authors, year and venue confirmed from a page fetched in this session.

---

## 0. Bib errors to fix in refs.bib (act on these first)

1. **`mccracken2025` has the wrong title and authors.** refs.bib says "Emergence of sparse and dense circuits in modular arithmetic networks", McCracken and others. No paper with that title exists: an exact-phrase search returns nothing. arXiv:2505.18266 is:
   *Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks*, by Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Létourneau, Doina Precup and Jonathan Love. **NeurIPS 2025** (poster). Sources: https://arxiv.org/abs/2505.18266, https://neurips.cc/virtual/2025/poster/115005, https://openreview.net/forum?id=zuHs6RHQwT
   The intro also says the count "moves from dense to sparse as the *degree* of the network changes \citep{mccracken2025}". McCracken et al. attribute the change to **depth or trainable embeddings**, not degree. Their Sec. 4.4 says one-hot MLPs go from ⌊n/2⌋ frequency types to "much fewer types" with depth or trainable embeddings. Their Sec. 5.1 says: "While we are the first to point out how to cause a phase transition from O(n) to O(log n) learned frequencies..., we don't know why it occurs." Reword to something like "as depth or trainable embeddings are added \citep{mccracken2025}". Our reading that this raises ν is our own interpretation and should be labelled as such.
2. **`wei2019regularization`, Theorem number.** Wei et al. Theorem 4.1 (arXiv v4, Sec. 4) is stated for **binary** logistic loss with regularizer λ‖Θ‖^r, for "some general norm ‖·‖", a-positive-homogeneous f and the global optimum Θ_λ. The **multi-class cross-entropy** version, which is what we use, is **Theorem C.1** ("Assume γ* > 0 in the multi-class setting with cross entropy loss. Then as λ → 0, γ_λ → γ*."). The text notes: "Though the result in this section is stated for binary classification, it extends to the multi-class setting ... We provide formal definitions and results in Section C." Suggested wording: "\citet[Thm. 4.1; multi-class form Thm. C.1]{wei2019regularization}". L_{2,b} is a norm for b ≥ 1, so the "general norm" hypothesis covers it. Source: https://arxiv.org/abs/1810.05369 (PDF v4 read in full).
3. **`li2025nsum`: credit Li et al. in Corollary `cor:dense`.** Li et al. Theorem 4.1 (arXiv v4) treats exactly our power neuron, (u_1ᵀx_1+…+u_kᵀx_k)^k w, under L_{2,k+1} with width m ≥ 2^{2k-1}(p-1)/2. They give γ* = 2(k!) / [(2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}] and prove that every frequency is used. With ν = k+1, our universal bound at b = ν for the power neuron is n!·B_p·p/(p-1) = 2 n! /((2ν)^{ν/2}(p-1)p^{(ν-2)/2}), **which is exactly Li et al.'s value**. The corollary currently says only that it "recovers the maximum margin of Morwani". It should also say that for power neurons with any n it recovers Li et al. Thm 4.1, provided the C_b ≤ n!B_p bound is tight at b = ν. Owning this credit up front also protects us: Li et al. already own the general-ν dense result.
4. **`morwani2024feature`.** Title and authors are confirmed. It was an ICLR 2024 **spotlight**. The modular-addition result is **Theorem 7**, with width m ≥ 4(p-1) and γ* = √(2/27)/(p^{1/2}(p-1)), matching Corollary `cor:dense`. The class-weighted margin is **Sec. 3.2, Lemma 5**. We cite "Section 3", which is fine; "Section 3.2" would be more precise. I did **not** verify that the phase condition is in "Section 4" (`sec:approach` cites \citet[Section~4]); the HTML I got was ambiguous (the section numbering there comes from the arXiv HTML render of v2). Sparse parity is Thm 8 and finite groups Thm 9.
5. The other entries are confirmed as written: nanda2023progress (ICLR 2023, oral); chughtai2023toy (ICML 2023, PMLR 202:6243–6267); lyu2020gradient (arXiv comment "Published in ICLR 2020"); gromov2023grokking (arXiv only, no venue listed); power2022grokking (arXiv); alon1994random (RSA 5(2):271–284, 1994, via Crossref); hoeffding1963probability (JASA 58(301):13–30, via Crossref). Optional: add `pages={6243--6267}, volume={202}, series={PMLR}` to chughtai.

---

## 1. Closest prior work on each of our results (adversarial reading)

| Our result | Closest prior | Overlap verdict |
|---|---|---|
| Thm universal / Cor dense (b = ν ⇒ all frequencies) | Morwani Thm 7 (ν=3); **Li et al. Thm 4.1 (all ν, power neuron)** | Prior at b = ν. Our novelty is b < ν. Must credit Li et al. (item 0.3). |
| Budget programme Γ_s, K* = Θ(log p), γ* = Θ(p^{-ν/2}(log p)^{1-ν/b}) | **McCracken et al. Thm 4.7 / Cor 4.8**: m > 2 ln n − … frequencies suffice; "Learning O(log n) distinct frequencies gives a logit margin Ω(log n)" (Sec. 4.3) | They prove *sufficiency* of O(log n) random frequencies for a logit margin, with no norm pricing, no optimality and no lower bound, and they say "we don't know why" the drop happens. Our Θ(log p) *optimal* count under L_{2,b}, with matching lower bound, is not in their paper. We must cite them as the first observation plus the sufficiency bound. |
| Same | **MAIS-O52** (Levine registry): "How many Fourier frequencies do modular-addition networks use?" Status "Open". Asks for lim (95%-mass count)/p; notes max-margin theory predicts all frequencies while training gives 3–15 | Our Thm scaling answers it *for max-margin optima in our model class*: K*/p → 0 at rate log p/p. Cite as the posed problem. The O52 page cites MAIS-A5 Problem 5.2. |
| Lemma cost (ν neurons per clean feature) + Prop ν-block | **Gromov 2023 Sec. 3, Eqs. (11)–(16)**: phases φ¹+φ²=φ³ give constructive interference; the other terms cancel *approximately* because the random phases give Σ cos(…) ≪ N; "larger width … introduces redundancy: each frequency appears several times with different random phases" | Same mechanism (phase cancellation of sign-flipped copies), but approximate and random. Ours is exact with ν deterministic roots-of-unity phases and a matching lower bound. Cite at Prop block. |
| Same | **Morwani Thm 7 construction**: u,v,w = λcos(θ+2πζ·/p) with θ_u+θ_v=θ_w, 4(p−1) neurons total (8 per frequency) | At b = ν splitting is free (our Lemma cost, r=1), so their 8/frequency is also optimal there. For b < ν our count ν = 3 is the unique minimum. Worth one sentence. |
| Same | **Tian, CoGS (NeurIPS 2025)**: quadratic activation, **MSE** loss, algebraic composition of partial solutions; order-4 and order-6 per-frequency global solutions (Cors 2 and 4). Thm 5 ("Occam's razor"): "lower-order solutions are preferred if trained with L2 regularization." Order-3 *partial* solutions appear in Table 1. No lower bound on nodes per frequency ("sufficient conditions (Lemma 1)", constructions show feasibility not necessity) | Closest to Lemma cost. Different loss (MSE, not margin), no minimality proof, no count of frequencies. Our "exactly ν, and more never cheaper" is a genuine lower bound. Cite and contrast; reviewers will know CoGS. |
| Lemma cost (the ℓ_{2/ν}-type price A^{b/ν} via AM–GM) | **Gunasekar et al. NeurIPS 2018**: depth-L linear convnets converge to a predictor "related to the ℓ_{2/L} bridge penalty in the frequency domain". **Kolb et al. ICLR 2025**: L2 on D weight factors ≡ non-convex ℓ_{2/D} sparse regularization | The mechanism (a product of ν factors under ℓ2 gives a concave ℓ_{b/ν} price on the product, hence sparsity in frequency) is known in the implicit-bias literature. We must not claim it. The novelty is the coverage functional and its argmax on Z_p. Cite both where the paper explains the AM–GM exponent. |
| "exponent decides sparse vs dense" (framing) | **Jacobs & Montúfar 2605.19458**: mirror flow in homogeneous nets, max-margin, representations "spanning sparse to dense neuron activations"; **Tsilivis, Vardi, Kempe ICLR 2025**: algorithm-dependent geometric margin | Generic framing is prior. Cite in related work. |
| Prop ablation (removal costs ≥ 1−(1−w)^s) | **Rangamani ICASSP 2025**: in RNNs "individual frequencies can be removed with modest impact" but "performance degrades drastically as more frequencies are ablated"; Nanda et al. ablations | Empirical only; no quantitative bound. Good citation for motivation. |
| Groups section (tensor-rank price of an irrep, one irrep on S5) | **Shutman, Louidor, Tessler 2509.06931**: learning group word operations reframed as learning a 3-tensor of low rank; width tied to tensor rank via fusion structure. **Chughtai et al.**; **Stander et al.** (cosets, S5/S6); **Wu, Jaburi, Drori, Gross ICLR 2025**; **He et al. 2606.02993**: Abelian groups give "uniform diversification across nontrivial representations"; **Marchetti et al. ICML 2026**: learns "one irreducible representation … at a time" | Shutman et al. is the closest to "schoolbook d^ν neurons per irrep". Must cite in `sec:groups`. He et al. gives a dense/uniform picture, consistent with our b = ν endpoint. Marchetti et al. is about order, not count. |
| Counterpoint on "fixed neurons per irrep" | **Kam, Bessafi, Cadet 2609.17926** (NeurReps 2026, under review): weight decay changes the width-scaling constant 47×, group order only 1.10×; "contradicts capacity-counting theory that assigns fixed neurons per irreducible representation" | Possible reviewer ammunition against a naive capacity reading. Our claim concerns optimal-norm structure, not critical width. One line would pre-empt it. |

No paper found proves any of the following: a Θ(log p) optimal feature count under weight decay; finitely many optimal shapes in s; bottleneck offsets; the (1−w)^s ablation bound; or γ* = Θ(p^{-ν/2}(log p)^{1-ν/b}). Caveat: Morwani's forward-citation list was sampled, not exhausted, and concurrent ICLR 2027 submissions are invisible.

---

## 2. Forward citations: what the chase turned up (2024–2026, relevant subset)

Grouped by relevance. "Low" means the paper is safe to ignore or needs only a related-work mention.

**High (should be cited)**
- McCracken et al. NeurIPS 2025 (above).
- Li et al. AISTATS 2025 (above).
- Tian, *Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets*, NeurIPS 2025. https://arxiv.org/abs/2410.01779
- He, Wang, Chen, Yang, *On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking*, arXiv 2602.16849 (Feb 2026). Obs. 3 (§3.1): "every frequency k is represented among the neurons". Def 4.1(i): exactly N neurons per frequency, so the population is dense. §3.3: weight decay "refines the learned sparse features by pruning the remaining components", which is *within-neuron* sparsity. Remark 5.1 links the early loss to the average margin of Morwani. The OpenReview forum id=Qc4RLl7t3S exists, but the venue is **unconfirmed** (Cloudflare check). This is the dense-population theory we should contrast with. https://arxiv.org/abs/2602.16849
- He, Wang, Zhang, Chen, Yang, *Neural Networks Provably Learn Spectral Representations for Group Composition*, arXiv 2606.02993 (v2 Jul 2026). https://arxiv.org/abs/2606.02993
- Shutman, Louidor, Tessler, *Learning words in groups: fusion algebras, tensor ranks and grokking*, arXiv 2509.06931. https://arxiv.org/abs/2509.06931
- Levine et al., MAIS-O52 open problem. https://github.com/lionellevine/MAIS/blob/main/open-problems/MAIS-O52.md . The page attribution reads "Authored by: Claude Fable 5 directed by Lionel Levine · Audited by: GPT 5.6 Sol". Cite as `@misc`.

**Medium (related work)**
- Kunin et al., *Alternating Gradient Flows*, NeurIPS 2025. Gives a "complete characterization of the training dynamics" for quadratic nets on modular addition: Fourier features learned in order of coefficient magnitude. This covers order, not count. https://arxiv.org/abs/2506.06489
- Marchetti, Kunin, Myers, Acosta, Miolane, *Sequential Group Composition*, ICML 2026. https://arxiv.org/abs/2602.03655
- Wu, Jaburi, Drori, Gross, *Towards a Unified and Verified Understanding of Group-Operation Networks*, ICLR 2025. https://arxiv.org/abs/2410.07476
- Ding, Guo, Michaud, Liu, Tegmark, *Survival of the Fittest Representation*, arXiv 2405.17420. Reports that "higher embedding dimensions allow more frequencies to persist". This is a known validity attack on width-invariance of the count (see memory scoop-check). https://arxiv.org/abs/2405.17420
- Mohamadi, Li, Wu, Sutherland, *Why Do You Grok?*, ICML 2024. Its ∞-norm-bounded quadratic networks are an alternative-norm margin analysis. https://arxiv.org/abs/2407.12332
- Lyu, Jin, Li, Du, Lee, Hu, *Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking*, ICLR 2024. Shows the late phase goes to min-norm/max-margin; this justifies studying max-margin. https://arxiv.org/abs/2311.18817
- Tsilivis, Vardi, Kempe, *Flavors of Margin*, ICLR 2025. https://arxiv.org/abs/2410.22069
- Jacobs, Montúfar, *Implicit Bias of Mirror Flow in Homogeneous Neural Networks: Sparse and Dense Feature Learning*, arXiv 2605.19458. https://arxiv.org/abs/2605.19458
- Gunasekar, Lee, Soudry, Srebro, NeurIPS 2018. https://proceedings.neurips.cc/paper/2018/hash/0e98aeeb54acf612b9eb4e48a269814c-Abstract.html
- Kolb, Weber, Bischl, Rügamer, *Deep Weight Factorization*, ICLR 2025. https://arxiv.org/abs/2502.02496
- Doshi, He, Das, Gromov, *Grokking Modular Polynomials*, arXiv 2406.03495. Gives analytic MLP solutions for "modular addition with many terms", i.e. the n-operand task we use. Cite next to Li et al. for the n-operand setting. Venue unconfirmed (arXiv only). https://arxiv.org/abs/2406.03495
- Rangamani, *Low Rank and Sparse Fourier Structure in Recurrent Networks Trained on Modular Addition*, ICASSP 2025. https://arxiv.org/abs/2503.22059
- Varma, Shah, Kenton, Kramár, Kumar, *Explaining grokking through circuit efficiency*, arXiv 2309.02390. Parameter-norm efficiency of generalizing circuits; related to our "price" framing. https://arxiv.org/abs/2309.02390
- Stander, Yu, Fan, Biderman, *Grokking Group Multiplication with Cosets*, arXiv 2312.06581. Venue not confirmed on the fetched page (S2 lists ICML). https://arxiv.org/abs/2312.06581
- Wen, Jacot, *Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning*, arXiv 2402.08010. Networks keep "only a few frequencies"; a cost-based analogue in CNNs. https://arxiv.org/abs/2402.08010

**Low (checked; no overlap)**
- Memana, *A Counterexample to Fourier Alignment in Single-Neuron Modular Addition*, arXiv 2608.04451 (Aug 2026). A single ReLU neuron with CE and no weight decay can die with energy spread over all frequencies. It answers MAIS-O60 negatively. A dynamics result, not a max-margin one.
- Pracher et al., *A Spectral Theory of Grokking: Weight Decay induces Feature Learning*, arXiv 2609.26679 (22 Sep 2026). Squared loss, NTK growth versus weight decay, timescales. No feature-count result. Very recent; worth one line.
- Musat, *The Geometry of Grokking: Norm Minimization on the Zero-Loss Manifold*, 2511.01938. No frequency-count claim.
- Tian, *Provable Scaling Laws of Feature Emergence* (Li₂), 2509.21519. The v5 note removes the dependence on weight decay; the abstract makes no count claim.
- Huang, Li, *Provable Benefits of Sinusoidal Activation for Modular Addition*, 2511.23443.
- Xu 2605.08119; Swaroop 2603.23784; Kam et al. 2609.15047; Kim 2607.23967; Defilippis et al. 2509.24882 (ICLR 2026; quadratic-net scaling via LASSO/compressed sensing, a different question).

---

## 3. Suggested placement

- Intro, first paragraph: Nanda, Chughtai, Morwani, Wei; add Lyu et al. 2024 (the late phase is max-margin) and Tsilivis et al. 2025 as the justification for studying max-margin.
- Intro, "Trained networks do not look like this": Nanda, Chughtai, **McCracken (fixed title, depth/embeddings wording)**, **MAIS-O52**, He et al. 2602.16849 (the dense theory), Ding et al. (the count varies with embedding dimension).
- `sec:approach` AM–GM paragraph: Gunasekar 2018, Kolb 2025 ("the ℓ_{2/ν}-type price of a product is standard; what is new is the coverage functional").
- Cor dense: Morwani Thm 7 **and Li et al. Thm 4.1**.
- Prop ν-block / Lemma cost: Gromov Sec. 3 (random-phase approximate cancellation), Morwani Thm 7 construction (8 neurons per frequency), Tian CoGS (order-4/6, MSE, Thm 5).
- Thm scaling: McCracken Thm 4.7 / Cor 4.8 (sufficiency of O(log n)), and say explicitly that ours is the optimal count with a matching lower bound. Alon–Roichman is already cited for the random construction.
- Prop ablation: Nanda ablations and Rangamani 2025.
- `sec:groups`: Chughtai, Morwani Thm 9, Stander, Wu et al., Shutman et al. (tensor rank), He et al. 2606.02993, Marchetti et al.
- Related-work appendix: Kunin AGF, Mohamadi et al., Jacobs & Montúfar, Varma et al., Doshi et al., Wen & Jacot, Power et al.

---

## 4. BibTeX (all VERIFIED from fetched pages unless noted)

```bibtex
@inproceedings{mccracken2025,
  title={Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks},
  author={McCracken, Gavin and Moisescu-Pareja, Gabriela and L{\'e}tourneau, Vincent and Precup, Doina and Love, Jonathan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2505.18266}
}
@misc{mais_o52,
  title={{MAIS-O52}: How many {F}ourier frequencies do modular-addition networks use?},
  author={Levine, Lionel},
  howpublished={MAIS open problem registry, \url{https://github.com/lionellevine/MAIS/blob/main/open-problems/MAIS-O52.md}},
  note={Posed in MAIS-A5 as Problem 5.2; page credits drafting to Claude Fable 5, directed by L. Levine},
  year={2026}
}
@inproceedings{tian2025cogs,
  title={Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets},
  author={Tian, Yuandong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2410.01779}
}
@article{he2026mechanism,
  title={On the Mechanism and Dynamics of Modular Addition: {F}ourier Features, Lottery Ticket, and Grokking},
  author={He, Jianliang and Wang, Leda and Chen, Siyu and Yang, Zhuoran},
  journal={arXiv preprint arXiv:2602.16849},
  year={2026}
}
@article{he2026spectral,
  title={Neural Networks Provably Learn Spectral Representations for Group Composition},
  author={He, Jianliang and Wang, Leda and Zhang, Fengzhuo and Chen, Siyu and Yang, Zhuoran},
  journal={arXiv preprint arXiv:2606.02993},
  year={2026}
}
@article{shutman2025words,
  title={Learning words in groups: fusion algebras, tensor ranks and grokking},
  author={Shutman, Maor and Louidor, Oren and Tessler, Ran},
  journal={arXiv preprint arXiv:2509.06931},
  year={2025}
}
@inproceedings{kunin2025agf,
  title={Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks},
  author={Kunin, Daniel and Marchetti, Giovanni Luca and Chen, Feng and Karkada, Dhruva and Simon, James B. and DeWeese, Michael R. and Ganguli, Surya and Miolane, Nina},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2506.06489}
}
@inproceedings{marchetti2026sequential,
  title={Sequential Group Composition: A Window into the Mechanics of Deep Learning},
  author={Marchetti, Giovanni Luca and Kunin, Daniel and Myers, Adele and Acosta, Francisco and Miolane, Nina},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  note={arXiv:2602.03655}
}
@inproceedings{wu2025unified,
  title={Towards a Unified and Verified Understanding of Group-Operation Networks},
  author={Wu, Wilson and Jaburi, Louis and Drori, Jacob and Gross, Jason},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  note={arXiv:2410.07476}
}
@article{ding2024survival,
  title={Survival of the Fittest Representation: A Case Study with Modular Addition},
  author={Ding, Xiaoman Delores and Guo, Zifan Carl and Michaud, Eric J. and Liu, Ziming and Tegmark, Max},
  journal={arXiv preprint arXiv:2405.17420},
  year={2024}
}
@inproceedings{mohamadi2024grok,
  title={Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition},
  author={Mohamadi, Mohamad Amin and Li, Zhiyuan and Wu, Lei and Sutherland, Danica J.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024},
  note={arXiv:2407.12332}
}
@inproceedings{lyu2024dichotomy,
  title={Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking},
  author={Lyu, Kaifeng and Jin, Jikai and Li, Zhiyuan and Du, Simon S. and Lee, Jason D. and Hu, Wei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  note={arXiv:2311.18817}
}
@inproceedings{tsilivis2025flavors,
  title={Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks},
  author={Tsilivis, Nikolaos and Vardi, Gal and Kempe, Julia},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  note={arXiv:2410.22069}
}
@article{jacobs2026mirror,
  title={Implicit Bias of Mirror Flow in Homogeneous Neural Networks: Sparse and Dense Feature Learning},
  author={Jacobs, Tom and Mont{\'u}far, Guido},
  journal={arXiv preprint arXiv:2605.19458},
  year={2026}
}
@inproceedings{gunasekar2018conv,
  title={Implicit Bias of Gradient Descent on Linear Convolutional Networks},
  author={Gunasekar, Suriya and Lee, Jason D. and Soudry, Daniel and Srebro, Nathan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={31},
  year={2018}
}
@inproceedings{kolb2025dwf,
  title={Deep Weight Factorization: Sparse Learning Through the Lens of Artificial Symmetries},
  author={Kolb, Chris and Weber, Tobias and Bischl, Bernd and R{\"u}gamer, David},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  note={arXiv:2502.02496}
}
@article{doshi2024polynomials,
  title={Grokking Modular Polynomials},
  author={Doshi, Darshil and He, Tianyu and Das, Aritra and Gromov, Andrey},
  journal={arXiv preprint arXiv:2406.03495},
  year={2024}
}
@inproceedings{rangamani2025rnn,
  title={Low Rank and Sparse {F}ourier Structure in Recurrent Networks Trained on Modular Addition},
  author={Rangamani, Akshay},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  note={arXiv:2503.22059}
}
@article{varma2023efficiency,
  title={Explaining grokking through circuit efficiency},
  author={Varma, Vikrant and Shah, Rohin and Kenton, Zachary and Kram{\'a}r, J{\'a}nos and Kumar, Ramana},
  journal={arXiv preprint arXiv:2309.02390},
  year={2023}
}
@article{stander2023cosets,
  title={Grokking Group Multiplication with Cosets},
  author={Stander, Dashiell and Yu, Qinan and Fan, Honglu and Biderman, Stella},
  journal={arXiv preprint arXiv:2312.06581},
  year={2023}
}
@article{wen2024cnn,
  title={Which Frequencies do {CNN}s Need? Emergent Bottleneck Structure in Feature Learning},
  author={Wen, Yuxiao and Jacot, Arthur},
  journal={arXiv preprint arXiv:2402.08010},
  year={2024}
}
@article{kam2026orbits,
  title={Symmetry without a manifold: intrinsic dimension on orbits},
  author={Kam, Chon-Fai and Bessafi, Miloud and Cadet, Fr{\'e}d{\'e}ric},
  journal={arXiv preprint arXiv:2609.17926},
  year={2026}
}
```

Unconfirmed, so no bib entry: the venue of He et al. 2602.16849 (OpenReview blocked); the venue of Stander et al. (S2 says ICML 2024, not seen on a fetched page); the venue of Doshi et al.

## 5. URLs fetched this session
arxiv.org/abs/{2505.18266, 2311.07568, 2402.09469, 1906.05890, 2301.02679, 2302.03025, 1810.05369 (pdf v4, full text), 2609.26679, 2608.04451, 2509.21519, 2605.19458, 2511.23443, 2407.12332, 2405.17420, 2410.07476, 2410.01779, 2506.06489, 2602.03655, 2509.06931, 2410.22069, 2602.16849, 2606.02993, 2503.22059, 2605.08119, 2609.15047, 2511.01938, 2603.23784, 1806.00468, 2402.08010, 2404.10824, 2502.02496, 2311.18817, 2312.06581, 2309.02390, 2406.03495, 2509.24882, 2609.17926, 2607.23967, 2201.02177}; arxiv.org/html/{2505.18266, 2311.07568, 2402.09469v4, 2602.16849v1, 2410.01779, 2608.04451}; arxiv.org/pdf/2301.02679 (full text); neurips.cc/virtual/2025/poster/115005; iclr.cc/virtual/2023/oral/12572; proceedings.mlr.press/v202/chughtai23a.html; mlanthology.org/iclr/2025/tsilivis2025iclr-flavors/; github.com/lionellevine/MAIS/.../MAIS-O52.md; api.crossref.org (Alon–Roichman, Hoeffding); api.semanticscholar.org citation lists.
