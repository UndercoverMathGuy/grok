# ICLR 2026 mech-interp landscape survey (recovered)

Recovered 2026-08-13 from the 08-12 research session (OpenReview API sweep + Chrome-session
forum scrapes). Companion file: `iclr2026_reviews_digest.md` (per-paper reviews, rebuttals,
meta-reviews for the 8 grokking/toy posters).

## Methodology & caveats

- OpenReview `/notes/search` API swept across 14 mech-interp terms, each capped at 50 results;
  165 unique **accepted** papers, ~101 genuinely relevant after keyword/area filtering.
  Representative sample and lower bound, not an exhaustive census.
- Per-term accepted counts: mechanistic interpretability 16, grokking 10, sparse autoencoder 10,
  circuit analysis 14, superposition 11, induction heads 16, training dynamics/phase transition 14,
  interpretability 13, feature representation LM 13, emergence capabilities 14, activation
  steering 16, probing representations 9, modular arithmetic 12, lottery ticket init 11.
- Review scores scraped from OpenReview forums via the user's logged-in Chrome session
  (Cloudflare blocks anonymous API/WebFetch access to forums).

## Headline findings

1. **Mech interp is large at ICLR** — its own primary area, 100+ accepted papers. Center of
   mass is LLM-applied work: SAE variants/evals, attention-head taxonomy in LLMs, and ~12+
   activation-steering papers.
2. **Interp orals were method/theory-flavored, not toy-phenomenon.** No grokking paper made
   oral/spotlight in the sweep.
3. **But at least 7-8 grokking/toy posters got in — all marginally** (avg 4.7–6.0, several
   carrying an outright 2 into acceptance). The decision was made by the AC in every case.

## The three hooks that bought toy-paper acceptance

Every accepted grokking/toy paper had exactly one of:
- **(a) a theory claiming generality** — Li₂, neural collapse, spectral bias/preconditioning
- **(b) a correction of something the field believed** — Clock ≡ Pizza (Manifolds)
- **(c) a bridge to real LLM training** — grokking-in-pretraining, Evolution of Concepts
  (crosscoders), Hidden Breakthroughs (hidden phase transitions)

"Careful new empirical result inside the toy" alone did **not** appear in the accepted set.

## The grokking/toy cohort: scores and survival

ICLR 2026 scale: 2/4/6/8 = reject/weak-reject/weak-accept/accept.

| Paper | Forum | Ratings | Avg | How it survived |
|---|---|---|---|---|
| Manifolds of Modular Addition | 2olkCiSELH | 2, 4, 8 | 4.7 | AC override — championed despite two negative reviews |
| Neural Collapse ↔ grokking+IB | sLX5P7FTfT | 6, 4, 4 | 4.7 | Rebuttal + AC judgment |
| Li₂ (Yuandong Tian, solo) | ceIBRhJpUr | 4, 4, 4, 8 | 5.0 | Rebuttal fixed math inconsistency; AC bought "first-principles insight" |
| Preconditioned GD → rich regime | CXlsqTAf1E | 6, 6, 6, 2 | 5.0 | AC dismissed the 2 ("reviewer provided no evidence") |
| In-Context Algebra (Bau lab) | J2peqXPQbB | 4, 6, 8, 2 | 5.0 | Huge rebuttal (16 new models); one reviewer signaled a raise |
| Grokking in LLM Pretraining | blfwRondjY | 8, 4, 6, 4 | 5.5 | Trained a whole new model (nanoMoE) mid-rebuttal |
| Egalitarian Gradient Descent | wCnHeql3ow | 6, 6, 8, 4 | 6.0 | Added MNIST/CIFAR + Grokfast comparisons in rebuttal |
| Decoupling Dynamical Richness | 7Mbz5uSf2J | 8, 4, 4, 8 | 6.0 | Split panel, AC sided with the 8s |

Empirical bar for a grokking poster: **~5 average, one enthusiastic reviewer, a heavyweight
rebuttal, and an AC willing to carry you.** "It's contrived and toy" appeared explicitly in the
In-Context Algebra and Manifolds reviews — both still accepted because the causal/mechanistic
work was sound.

Consistent reviewer kill-shots across the cohort: single setting; correlation-not-causation;
overclaiming; poor engagement with adjacent work. Every acceptance was won in rebuttal with
**new experiments** — budget a held-in-reserve cohort to fire during discussion.

## What each toy paper says + attack surface for us

- **Li₂ (Tian)** — 3-stage framework (lazy → independent feature learning → interactive) for
  2-layer nets; features emerge as local maxima of an energy function; reviewers hit patchwork
  assumptions, weight-decay dependence, rushed writing. **Attack: dialogue partner, not scoop.**
  Li₂ quantifies the *menu* of features; T_k/compiler is the missing *selection rule* (basin
  assignment readable at init, steerable by surgery). Empirically testing his energy-ascent
  picture in our cohort = "we complete an accepted framework."
- **Manifolds of Modular Addition** — Clock and Pizza are topologically/geometrically the same
  algorithm (universality). In at 2/4/8 on pure AC advocacy; reviewers wanted an LLM connection.
  **Attack: complementary theorem** — geometry universal, *identity* quenched-random and
  init-readable/writable. One figure from existing runs.
- **Egalitarian GD + Preconditioned GD** — grokking delay = ill-conditioned optimization;
  equalize per-direction speeds and the plateau dies. Neither asks about *solution identity*.
  **Attack: optimizer-invariance arm** — accelerate grokking, show T_k still predicts the
  committee (σ0.3 result already hints the lottery survives plateau removal).
- **Grokking in LLM Pretraining** — memorization→generalization transition in OLMoE via MoE
  routing metrics, no test set. Criticisms: single model, correlation-not-causation.
  **Attack: rhetorical** — we predict circuit identity *before* training with causal surgery;
  strictly stronger on both axes they were praised for. Cite and frame.
- **Neural Collapse / Decoupling Richness / In-Context Algebra** — unifying-lens theory, a
  metric/tool paper, and a new-toy-task mechanisms paper, all cleared at ~5. In-Context
  Algebra's reviews preview exactly the objections a toy submission gets and how a Bau-lab
  rebuttal answers them.

## AC-story strategy

Target isn't three random reviewers; it's **(a) give one expert reviewer a reason to be the 8,
(b) give the AC a one-paragraph story they can defend in the meta-review.** The story assembled
from the accepted papers' own argument:

> "Universality tells you what solution forms (Manifolds); Li₂ tells you how features climb
> out; we show what neither can — which features a given seed gets is a lottery drawn at init,
> readable in closed form and steerable by surgery. Selection, not formation."

## Background explainers (Clock / Pizza / Li₂)

- **Clock** (Nanda et al. 2023): embed tokens as (cos ωa, sin ωa); attention+MLP implement the
  angle-addition trig identities; logit for c = Σ_ω cos ω(a+b−c), constructive interference on
  the right answer. The committee is the set of ω's. "Which ω's and why" was left open — our hole.
- **Pizza** (Zhong et al. 2023): with uniform/weak attention the net instead averages embedding
  points and uses |cos ω(a−b)/2|-flavored quantities; needs auxiliary correction circuits.
  Thesis: algorithmic non-universality — architecture nudges change the learned algorithm.
- **Li₂** (Tian, ICLR 2026): stage I lazy/memorization; stage II gradient G_F carries label
  structure so each hidden neuron independently ascends an energy E whose local maxima are the
  generalizing (Fourier) features; stage III interaction/pruning → grok. Cannot say which
  features a specific seed gets — the theory quantifies over the menu, not the draw.

## Related follow-ups already identified (end of that session)

1. **Separation theorem** (cheap, local): Zhong et al. algorithm metrics (gradient symmetricity,
   distance-irrelevance) on 102 compiled arms vs natural — prediction: compilation writes
   *identity*, never perturbs *implementation*.
2. **Compile the algorithm**: flatten attention at init + dictate targets → pizza on chosen
   frequencies? Headline: frequency set AND algorithm chosen at init, weights-only.
3. **Cross-architecture compilation**: port compiler to He et al.'s 2-layer MLP / Gromov's
   quadratic MLP; also the falsification test for the variational capacity law K*(architecture).

## Accepted-paper list (101 relevant of 165 unique)

[ICLR 2026 Oral] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability
   area: interpretability and explainable AI | kw: Semantic associations, Interpretability, LLM
[ICLR 2026 Oral] Temporal superposition and feature geometry of RNNs under memory demands
   area: interpretability and explainable AI | kw: RNNs, superposition, representational geometry, features, capacity, memory demands
[ICLR 2026 Oral] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability
   area: interpretability and explainable AI | kw: Interpretability, Dictionary Learning, Machine Learning, Large Language Models
[ICLR 2026 Poster] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees
   area: interpretability and explainable AI | kw: interpretability, mechanistic interpretability, circuit discovery
[ICLR 2026 Poster] Small Transformers Don’t Need LayerNorm at Inference Time: Scaling LayerNorm Removal to GPT-2 XL and Implications for Mechanistic Interpretability
   area: interpretability and explainable AI | kw: mechanistic interpretability, language models
[ICLR 2026 Poster] Circuit Insights: Towards Interpretability Beyond Activations
   area: interpretability and explainable AI | kw: mechanistic interpretability, automated interpretability, explainable AI, transcoders, large language models, circuits
[ICLR 2026 Poster] Tracking Equivalent Mechanistic Interpretations Across Neural Networks
   area: interpretability and explainable AI | kw: mechanistic interpretability
[ICLR 2026 Poster] Hedonic Neurons: A Mechanistic Mapping of Latent Coalitions in Transformer MLPs
   area: interpretability and explainable AI | kw: mechanistic interpretability, feature discovery, MLPs
[ICLR 2026 Poster] Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers
   area: interpretability and explainable AI | kw: Sparse Autoencoders, SAEs, LLMs, interpretability
[ICLR 2026 Poster] Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insights
   area: interpretability and explainable AI | kw: Mechanistic Interpretability, Large Language Model, Task Vector, In-context Learning
[ICLR 2026 Poster] Mechanistic Detection and Mitigation of Hallucination in Large Reasoning Models
   area: alignment, fairness, safety, privacy, and societal considerations | kw: Reasoning, Hallucination, Mechanistic Interpretability
[ICLR 2026 Poster] Learning Concept Bottleneck Models from Mechanistic Explanations
   area: interpretability and explainable AI | kw: interpretability, concept bottleneck models, computer vision, explainable ai
[ICLR 2026 Poster] Attributing Response to Context: A Jensen–Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation
   area: interpretability and explainable AI | kw: context attribution, mechanistic interpretability, RAG
[ICLR 2026 Poster] MICLIP: Learning to Interpret Representation in Vision Models
   area: interpretability and explainable AI | kw: mechanistic interpretability, contrastive learning, sparse autoencoder
[ICLR 2026 Poster] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning
   area: interpretability and explainable AI | kw: Mechanistic Interpretability, Unsupervised Learning, Representation Space Geometry
[ICLR 2026 Poster] Evaluating SAE interpretability without generating explanations
   area: interpretability and explainable AI | kw: interpretability, explanation, sae, transcoder
[ICLR 2026 Poster] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences
   area: interpretability and explainable AI | kw: Mechanistic Interpretability, Steering, Automated interpretability, Benchmarking interpretability
[ICLR 2026 Poster] Latent Concept Disentanglement in Transformer-based Language Models
   area: interpretability and explainable AI | kw: Mechanistic interpretability, in-context learning, transformers, large language models, disentanglement
[ICLR 2026 Poster] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test
   area: interpretability and explainable AI | kw: Generalization, Large Language Models
[ICLR 2026 Poster] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking
   area: optimization | kw: grokking, optimization, generalization, acceleration
[ICLR 2026 Poster] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence
   area: optimization | kw: deep learning, grokking, information bottleneck, neural collapse, training dynamics
[ICLR 2026 Poster] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime
   area: learning theory | kw: spectral bias, preconditioned gradient descent, grokking, optimization dynamics, neural tangent kernel, higher-order methods
[ICLR 2026 Poster] $\mathbf{Li_2}$: A Framework on Dynamics of Feature Emergence and Delayed Generalization
   area: interpretability and explainable AI | kw: grokking, gradient dynamics, generalization, memorization, modular addition, scaling laws
[ICLR 2026 Poster] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement
   area: unsupervised, self-supervised, semi-supervised, and supervised representation learning | kw: training dynamics, representation learning, lazy/rich regime, neural collapse, grokking, kernel methods
[ICLR 2026 Poster] RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?
   area: foundation or frontier models, including LLMs | kw: Large Language Models, Reinforcement Learning, Generalization, Learnability
[ICLR 2026 Poster] FACT: a first-principles alternative to the Neural Feature Ansatz for how networks learn representations
   area: learning theory | kw: feature learning, deep learning, neural feature ansatz, convergence, theory
[ICLR 2026 Poster] In-Context Algebra
   area: interpretability and explainable AI | kw: Interpretability, In-Context Learning, ICL, Algebra, Grokking, Symbolic Reasoning
[ICLR 2026 Poster] AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features
   area: interpretability and explainable AI | kw: Sparse Autoencoder, Mechanistic Interpretability
[ICLR 2026 Poster] On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy
   area: interpretability and explainable AI | kw: sparse autoencoder, SAE, theoretical understanding
[ICLR 2026 Poster] Cross-Modal Redundancy and the Geometry of Vision–Language Embeddings
   area: interpretability and explainable AI | kw: multimodal, concepts, sparse autoencoder, modality gap, applications of interpretability
[ICLR 2026 Poster] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders
   area: alignment, fairness, safety, privacy, and societal considerations | kw: Sparse Autoencoder, Model Interpretability, Retreival-augmented Generation, LLM Hallucination, RAG Faithfulness
[ICLR 2026 Poster] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs
   area: interpretability and explainable AI | kw: LLMs, interpretability, multilingualism
[ICLR 2026 Poster] Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders
   area: datasets and benchmarks | kw: data filtering, submodular, sparse autoencoders
[ICLR 2026 Poster] Si-GT: Fast Interconnect Signal Integrity Analysis for Integrated Circuit Design via Graph Transformers
   area: applications to physical sciences (physics, chemistry, biology, etc.) | kw: Graph Transformer, Integrated Circuit, Signal Integrity
[ICLR 2026 Poster] Evolving Graph Structured Programs for Circuit Generation with Large Language Models
   area: applications to computer vision, audio, language, and other modalities | kw: Electronic Design Automation; Logic Synthesis; Large Language Models;
[ICLR 2026 Poster] A Hierarchical Circuit Symbolic Discovery Framework for Efficient Logic Optimization
   area: neurosymbolic & hybrid AI systems (physics-informed, logic & formal reasoning, etc.) | kw: Electronic Design Automation; Logic Synthesis; Large Language Models;
[ICLR 2026 Poster] Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition
   area: interpretability and explainable AI | kw: Mechanistic Interpretability, Attention Superposition, Sparse Dictionary Learning, Circuit Analysis
[ICLR 2026 Poster] OSIRIS: Bridging Analog Circuit Design and Machine Learning with Scalable Dataset Generation
   area: datasets and benchmarks | kw: electronic design automation, analog circuits, reinforcement learning, layout design, parasitic-aware, dataset generator
[ICLR 2026 Poster] CircuitNet 3.0: A Multi-Modal Dataset with Task-Oriented Augmentation for AI-Driven Circuit Design
   area: datasets and benchmarks | kw: Dataset, Benchmark, Machine learning, Electric design automatic
[ICLR 2026 Poster] Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping
   area: interpretability and explainable AI | kw: Mechanistic Interpretability, Model Editing, Circuit Reshaping
[ICLR 2026 Poster] Topology Matters in RTL Circuit Representation Learning
   area: other topics in machine learning (i.e., none of the above) | kw: RTL repressentation, EDA
[ICLR 2026 Poster] Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP
   area: interpretability and explainable AI | kw: Multimodality, Circuit analysis, Probing, AI Safety, Vision transformers
[ICLR 2026 Poster] PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing
   area: datasets and benchmarks | kw: LLMs, Printed Circuit Board, Placement and Routing, Multimodal Benchmark
[ICLR 2026 Poster] Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning
   area: neurosymbolic & hybrid AI systems (physics-informed, logic & formal reasoning, etc.) | kw: Information-Theoretic Bounds, Compositional Circuits, Reasoning Generalization, CoT Training
[ICLR 2026 Poster] From Data Statistics to Feature Geometry: How Correlations Shape Superposition
   area: interpretability and explainable AI | kw: Mechanistic Interpretability, Superposition, Linear Representation Hypothesis, Feature Geometry, Feature Manifold
[ICLR 2026 Poster] Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought
   area: learning theory | kw: chain of continuous thought, training dynamics, reasoning, superposition
[ICLR 2026 Poster] Flow Straight and Fast in Hilbert Space: Functional Rectified Flow
   area: generative models | kw: Hilbert space, superposition principle
[ICLR 2026 Poster] Features Emerge as Discrete States: The First Application of SAEs to 3D Representations
   area: interpretability and explainable AI | kw: sparse autoencoders, mechanistic interpretability, computer vision
[ICLR 2026 Poster] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition
   area: interpretability and explainable AI | kw: interpretability, language models, task generalization, induction heads
[ICLR 2026 Poster] Identifying and Evaluating Inactive Heads in Pretrained LLMs
   area: interpretability and explainable AI | kw: dormant attention, multi-head attention, attention heads, attention sinks
[ICLR 2026 Poster] Expert Heads: Robust Evidence Identification for Large Language Models
   area: interpretability and explainable AI | kw: Large language model, Knowledge Integration, Attention Mechanisms
[ICLR 2026 Poster] Token Alignment Heads: Unveiling Attention's Role in LLM Multilingual Translation
   area: foundation or frontier models, including LLMs | kw: LLM, Multilinguistic, Interpretability
[ICLR 2026 Poster] LLMs Process Lists With General Filter Heads
   area: interpretability and explainable AI | kw: interpretability, language models, map-filter-reduce, functional programming, symbolic systems
[ICLR 2026 Poster] Localizing Task Recognition and Task Learning in In-Context Learning via Attention Head Analysis
   area: interpretability and explainable AI | kw: Mechanistic Interpretability, In-context Learning, Large Language Model
[ICLR 2026 Poster] Structural Inference: Interpreting Small Language Models with Susceptibilities
   area: interpretability and explainable AI | kw: Interpretability, Statistical Physics, Singular Learning Theory
[ICLR 2026 Poster] Training Dynamics Impact Post-Training Quantization Robustness
   area: optimization | kw: Efficiency, quantization, optimization
[ICLR 2026 Poster] Predictive Differential Training Guided by Training Dynamics
   area: optimization | kw: Training Dynamics, Koopman Operator Theory, Predictive Training, Deep Neural Networks
[ICLR 2026 Poster] Evolution of Concepts in Language Model Pre-Training
   area: interpretability and explainable AI | kw: Large Language Model; Pre-Training; Mechanistic Interpretability; Training Dynamics; Crosscoder
[ICLR 2026 Poster] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection
   area: learning theory | kw: Reinforcement Learning, Language Models, Reasoning Patterns, Training Dynamics
[ICLR 2026 Poster] Hidden Breakthroughs in Language Model Training
   area: interpretability and explainable AI | kw: interpretability techniques, loss disaggregation, phase transitions
[ICLR 2026 Poster] Sparse CLIP: Co-Optimizing Interpretability and Performance in Contrastive Learning
   area: interpretability and explainable AI | kw: contrastive learning, multimodal learning, interpretability
[ICLR 2026 Poster] Medical Interpretability and Knowledge Maps of Large Language Models
   area: interpretability and explainable AI | kw: Large Language Models, Interpretability, Explainability, Medicine, Healthcare, Knowledge Maps
[ICLR 2026 Poster] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts
   area: interpretability and explainable AI | kw: prompt tuning, explainable AI, knowledge discovery, prototype learning
[ICLR 2026 Poster] Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders
   area: interpretability and explainable AI | kw: Sparse Autoencoders; Interpretability; Utility
[ICLR 2026 Poster] The Tutor-Pupil Augmentation: Enhancing Learning and Interpretability via Input Corrections
   area: applications to physical sciences (physics, chemistry, biology, etc.) | kw: Model Augmentation, Machine learning for physical sciences
[ICLR 2026 Poster] Priors in time: Missing inductive biases for language model interpretability
   area: interpretability and explainable AI | kw: Top-Down Interpretability, Sparse Autoencoders, Temporal Structure, Stationarity
[ICLR 2026 Poster] Multi-Feature Quantized Self-Attention for Fair Large Language Models
   area: alignment, fairness, safety, privacy, and societal considerations | kw: Large language models, multi-attribute social bias, quantized adversarial autoencoder
[ICLR 2026 Poster] The Lattice Representation Hypothesis of Large Language Models
   area: interpretability and explainable AI | kw: Interpretability, formal concept analysis, language models, ontology
[ICLR 2026 Poster] Bilinear representation mitigates reversal curse and enables consistent model editing
   area: interpretability and explainable AI | kw: model editing, reversal curse, language model, relational knowledge, knowledge editing
[ICLR 2026 Poster] Learning Dynamics Feature Representation via Policy Attention for Dynamic Path Planning in Urban Road Networks
   area: reinforcement learning | kw: Dynamic Path Planning; Reinforcement Learning; State Representation; Dynamics Feature Representation; Policy Attention Mechanism
[ICLR 2026 Poster] Reverse Distillation: Consistently Scaling Protein Language Model Representations
   area: applications to physical sciences (physics, chemistry, biology, etc.) | kw: Protein language models, model scaling, Representation learning, Subspace decomposition, interpretability, Model distillation
[ICLR 2026 Poster] Readout Representation: Redefining Neural Codes by Input Recovery
   area: applications to neuroscience & cognitive science | kw: neural representation, readout representation, representation size, misrepresentation, neural variability, information recovery
[ICLR 2026 Poster] On the Predictive Power of Representation Dispersion in Language Models
   area: interpretability and explainable AI | kw: Embedding geometry, Unsupervised evaluation, Mechanistic interpretability, Large Language Models, Label-free metrics
[ICLR 2026 Poster] Optimizer Choice Matters For The Emergence of Neural Collapse
   area: unsupervised, self-supervised, semi-supervised, and supervised representation learning | kw: neural collapse, implicit bias, deep learning theory, classification, adaptive optimizers, training dynamics
[ICLR 2026 Poster] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors
   area: interpretability and explainable AI | kw: next-token prediction, transformers, interpretability
[ICLR 2026 Poster] Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations
   area: foundation or frontier models, including LLMs | kw: transformer, in-context learning, task vector
[ICLR 2026 Poster] Context and Diversity Matter: The Emergence of In-Context Learning in World Models
   area: transfer learning, meta learning, and lifelong learning | kw: In-Context Learning; World Models
[ICLR 2026 Poster] Evolution and compression in LLMs: on the emergence of human-aligned categorization
   area: interpretability and explainable AI | kw: LLMs, information theory, semantics
[ICLR 2026 Poster] Fine-Grained Activation Steering: Steering Less, Achieving More
   area: foundation or frontier models, including LLMs | kw: Activation Steering, Large Language Models, Fine-Grained Intervention
[ICLR 2026 Poster] Activation Steering with a Feedback Controller
   area: foundation or frontier models, including LLMs | kw: activation steering, behaviour control, alignment, PID control, mechanistic interpretability, language models
[ICLR 2026 Poster] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering
   area: optimization | kw: activation steering, generation diversity, manifold opimization
[ICLR 2026 Poster] AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint
   area: alignment, fairness, safety, privacy, and societal considerations | kw: Large Language Models, Safety, Activation Steering
[ICLR 2026 Poster] ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment
   area: foundation or frontier models, including LLMs | kw: LLM alignment, Representation Engineering, Activation Steering, ODE-based Framework, Barrier Functions
[ICLR 2026 Poster] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection
   area: foundation or frontier models, including LLMs | kw: Large Language Models, LLM Steering, Instruction following, Activation engineering
[ICLR 2026 Poster] Hallucination Reduction with CASAL:  Contrastive Activation Steering for Amortized Learning
   area: interpretability and explainable AI | kw: hallucination, representation learning, interpretability, finetuning, steering
[ICLR 2026 Poster] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models
   area: alignment, fairness, safety, privacy, and societal considerations | kw: Large Vision-Language Models, Hallucination
[ICLR 2026 Poster] Steering MoE LLMs via Expert (De)Activation
   area: foundation or frontier models, including LLMs | kw: Steering, MoE, Mixture-of-Experts, LLM, Safety
[ICLR 2026 Poster] Steering Language Models with Weight Arithmetic
   area: alignment, fairness, safety, privacy, and societal considerations | kw: steering, alignment, safety, model editing, 
