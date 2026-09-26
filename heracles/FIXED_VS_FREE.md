# FIXED vs FREE — is the fixed-support ladder a real experiment, and can free training be driven to the optimum?
(2026-09-19, hostile methodology pass. Local CPU (~75 min) + one RTX 5090 secure pod (50 min, **$0.83**, terminated
and confirmed 404). New code: `fvf_obj.py`,
`fvf_fixed.py`, `fvf_free.py`, `fvf_table.py`, `fvf_jobA/B/C.sh`, `fvf_pod.sh`; output under `FINAL_RUNS/FVF/`.
Nothing existing was edited. Companion to RESULTS_NU4, PROOF_AUDIT, INIT_AUDIT.)

---

## 0. Symbols, in plain language

| symbol | formula | what it means in words |
|---|---|---|
| p | — | the prime modulus; the task is a_1 + … + a_n ≡ c (mod p) on the FULL table of p^n rows |
| n | — | number of operands |
| ν | ν = n + 1 | homogeneity degree: multiply every weight by s and the logits multiply by s^ν |
| chain | z(c; a) = Σ_i w_i(c) Π_j e_{ij}(a_j) | the gated chain (`--act prod`): each neuron multiplies one embedding row per operand, then reads out |
| x² net | z(c; a,b) = Σ_i w_i(c)(e_i(a) + f_i(b))² | Morwani's quadratic net (`--act pow`, n = 2) |
| λ | — | weight-decay coefficient; the trained objective is **obj = CE + λ‖θ‖²** |
| ‖θ‖² | Σ over every weight of its square | the quantity plain weight decay prices |
| margin | min over rows of (correct logit − best wrong logit) | the worst-case separation on the whole table |
| γ | γ = margin / ‖θ‖^ν | the **normalised margin**: scale-free, a property of the weight *direction* only |
| K | — | number of Fourier frequencies the solution uses |
| S_K | argmax over K-sets of M_K | the coverage-optimal K-set (`kstar.best_subset`) |
| M_K | M_K = min over D ≠ 0 of Σ_{k∈S}(1 − cos 2πkD/p) | "coverage": how badly the worst wrong answer at offset D is beaten by the frequencies in S |
| Γ_S | Γ_S = max over shares b ≥ 0, Σb = 1 of min_{D≠0} Σ_k b_k^{ν/2}(1 − cos 2πkD/p) | the **free-amplitude programme**: split one unit of ℓ₂ budget over the frequencies of S, frequency k then reaches amplitude b_k^{ν/2}; the value is the worst-case coverage it buys (`nu4_program.py`) |
| K* | support size of the maximiser of Γ | the predicted optimal feature count |
| clean fraction | share of the centred-logit energy that is a function of (Σa_j − c) alone | 1 − junk; junk = logit content that is not a function of the task's invariant |
| N_.05 | — | number of frequencies holding the top 95 % of the *clean* logit energy (the count instrument) |
| top_share | energy-weighted mean over neurons of (largest single-frequency share of that neuron's embedding energy) | 1.000 means every neuron carries exactly one frequency |

"Fixed-support run" (RESULTS_NU4 method T1) = build the analytic junk-free K-frequency net on S_K, then fine-tune on
CE + λ‖θ‖² with **every embedding row and readout column projected onto the Fourier span of S_K after each step**.

---

## 1. VERDICT ON Q1 — **keep it, rename it, add the three controls, and stop calling it a measurement of trained networks.**

Decisive form:

> **Keep it, with controls, and rename it.** The fixed-support ladder is **not circular in the way a reviewer will
> first suspect**: at ν 3 the analytic start is not doing the work — random initialisation inside the span
> reproduces the whole ladder to five significant figures at p 71 and independently finds the argmax K* = 6 at
> p 113, for both the gated chain and Morwani's x² net. But the experiment **is load-bearing in a second way the
> current write-up hides**: the Fourier *projection*, not the construction, is what produces the agreement.
> Removing the projection (random init, free support, identical optimiser) drops γ to 0.47–0.58 of the ladder's own
> peak at ν 3 and to *zero clean content* at ν 4. And at **ν 4 the random-init control itself fails at K = 4**
> (0.77 of the constructed value) — precisely the rung RESULTS_NU4's headline rests on.
> So the ladder measures **the normalised margin attainable by a K-frequency solution**, which is exactly the
> theory's functional evaluated in network space — a legitimate and useful measurement — and it measures **nothing
> about which K gradient descent picks**. Present it under that name; put the three controls (§1b, §1c) in the
> appendix; finish the ν 4 control before the paper (§6); and scope the abstract's
> "the normalized margin attained by these networks peaks at the predicted optimum" to *attainable* (§6).

### 1a. Anatomy of the tautology risk — what is and is not empirical content

Inside the junk-free, single-frequency-per-neuron class with support S and budget shares b, the relation

    γ = C(population) · val(S, b),     val(S, b) = min_{D ≠ 0} Σ_{k∈S} b_k^{ν/2} (1 − cos 2πkD/p)

is an **identity**, not a measurement: C(population) is the analytic constant N^{−(ν−2)/2} for an N-neuron
cancelling population (PROOF_AUDIT F1) and val is the programme's own objective. So if a run starts in that class
and stays in it, "measured = predicted" is true by algebra, and gradient descent is only re-optimising the shares b.

How much freedom does the projection leave? For support size K, width m and n operands, the projected parameter
space has dimension **m (n+1) 2K** (each of the n+1 tables per neuron is a real function on Z_p restricted to K
frequencies: K cosine and K sine coefficients). At ν 3, K = 5, m = 15 that is 450 real parameters. The junk-free
single-frequency class has ~m(2n+2) = 90 (one frequency label per neuron, plus n+1 amplitudes and n+1 phases), and
junk-freeness is a *further* condition coupling the phases across neurons. GD is therefore free, inside the
projection, to (i) mix frequencies within a neuron, (ii) build junk — the mixed-sign term cos(k(a₁ − a₂ + …)) lives
at frequencies of S and is fully reachable, (iii) redistribute neurons over frequencies, (iv) switch frequencies off.

What the archived runs actually did with that freedom (`FINAL_RUNS/NU4/fixed_prod_n{2,3}_p71*`, read today):

- **Junk: never added, and existing junk removed.** The 5 % weight noise puts the start at clean 0.980–0.999; every
  run ends at clean **1.0000** (chain), so GD moved *towards* the junk-free class rather than away. In the x² family
  it is weaker: clean ends at 0.93–0.99 for K ≥ 12 and 0.61–0.79 at K = 35 (`fixed_pow_n2_p71`), i.e. the quadratic
  net does use the junk freedom when the support is large.
- **Frequency mixing: never.** top_share = 1.0000 at every rung of the chain ladder.
- **Amplitudes: yes, and they move to the programme's optimum.** K = 5 ends at shares 0.222/0.209/0.201/0.188/0.179
  rather than the equal 0.2 it started from, and γ rises 3.61e-5 → 4.00e-5.
- **Frequencies switched off above K*: yes.** K = 8 → N_.05 6, K = 10 → 5, K = 24 → 6, K = 35 → 7 (common-phase
  population, p 71, ν 3). This is the free-amplitude programme's own behaviour (Γ_S puts exact zeros on some
  frequencies inside large supports) and it is a genuine, non-trivial agreement.
- **Seed variance is essentially zero**: at ν 3 both seeds land on the same γ to 5 significant figures at every rung.
  That is a warning label in itself — this is the numerical solution of an analytic problem, not a sample of runs.

**Honest summary of the empirical content of the archived ladder:** (1) the junk-free single-frequency class is an
attractor of CE + λ‖θ‖² inside a fixed span, not a knife-edge; (2) inside it GD finds the free-amplitude optimum,
including killing frequencies above K*; (3) the analytic population constant is flat in K (0.244–0.251 at ν 4
common-phase), which is what lets the measured curve have the *shape* of Γ_S at all. That is worth reporting. It is
not "networks attain their margin peak at K*".

### 1b. CONTROL 1 (new, measured): random initialisation inside the span — **the ladder survives**

`fvf_fixed.py --start rand`: Gaussian weights projected onto span(S_K), no analytic construction anywhere, same
optimiser, same λ schedule. Chain n = 2, p = 71, ν = 3; 2 seeds per rung; width 4K (≈ twice the 3K neurons the
construction uses).

| K | Γ_S (free-amplitude), rel | archived **constructed** start, γ (rel) | **random-init-in-span**, γ (rel) | N_.05 (rand, per seed) | clean |
|---|---|---|---|---|---|
| 2 | 0.321 | 1.2704e-05 (0.318) | **1.2704e-05 (0.318)** | 2, 2 | 1.000 |
| 3 | 0.749 | 2.9796e-05 (0.745) | **2.9796e-05 (0.745)** | 3, 3 | 1.000 |
| 4 | 0.891 | 3.5549e-05 (0.889) | **3.5549e-05 (0.889)** | 4, 4 | 1.000 |
| **5** | **1.000** | **3.9981e-05 (1.000)** | **3.9981e-05 (1.000)** | 5, 5 | 1.000 |
| 6 | 0.952 | 3.8076e-05 (0.952) | 3.4991e-05 (0.875) [3.19–3.81e-5] | 6, 4 | 1.000 |
| 8 | 0.940 | 3.6262e-05 (0.907) | 3.2864e-05 (0.822) [3.14–3.43e-5] | 7, 5 | 0.974 |
| 12 | 0.903 | 3.7136e-05 (0.929) | 3.5590e-05 (0.890) [3.50–3.62e-5] | 5, 6 | 1.000 |

Replicated on the pod at **p = 113** (the paper's prime), where K*(113, ν 3) = 6 and the random-init ladder peaks at
exactly 6 with an interior peak — see §5A — and for **Morwani's x² net** at p 113, §5B. At **ν 4 it only partly
survives** (exact at K 2, 3; 0.77 of the constructed value at K 4) — §5C.

Readings. (i) **Through the argmax the two starts agree to 5 significant figures** — the analytic construction is
*not* doing the work, and the measured argmax is K = 5 = K* from random init as well. (ii) Above K* the random start
sometimes fails to find the class optimum (12 % low at K = 6 in one seed of two, 13 % low at K = 8 with 5 % junk):
above the peak the random-init ladder is a **lower bound only**, and the assembly problem gets harder as K grows.
(iii) Width matters and is a real confound. At K = 5 with width exactly matched to the construction (m = 3K = 15) the
random start is width-limited: γ 1.72e-5, 1.91e-5, 3.43e-5 over three seeds (mean 0.59 of the ceiling), N_.05 3, 3, 4.
At m = 30 and m = 60 **all six seeds hit 3.9981e-05, i.e. the constructed value exactly**. So the control passes only
with ≈ 2× the minimal width; report the width sweep, do not hide it.

### 1c. CONTROL 2 (new, measured): release the projection — **the construction is a stable point, and the projection is what carries the ladder**

Two separate questions. First, is a K-rung held in place only by the projection? `fvf_fixed.py --start con --no_proj`
(constructed start, 5 % noise, **no projection at all**), chain ν 3 p 71, 2 seeds:

| K | with projection, γ | **released** (no projection), γ | N_.05 released |
|---|---|---|---|
| 3 | 2.9796e-05 | 2.9793e-05 | 3, 3 |
| 5 | 3.9981e-05 | 3.9977e-05 | 5, 5 |
| 8 | 3.6262e-05 | 3.6262e-05 | **8, 8** |
| 12 | 3.7136e-05 | 3.7136e-05 | 6 |

So every rung — including rungs **above** the argmax — is a stable point of unprojected training. K = 8 released does
not fall to 5. **The number of features is not selected by descent; many K are locally stable.** This is the single
most important thing the ladder shows and the paper should say it out loud: the theory locates the best K, it does
not predict the K a given run ends at. (The archived K = 35 rung is already an almost-unprojected run — the mask
removes only the DC term — and it collapses to N_.05 5–7 at 0.86–0.97 of the K5 rung, which is consistent.)

Second, and less comfortable: does the projection matter when you *also* remove the construction?
`fvf_fixed.py --start randfull` (random init, no projection, same optimiser, same λ schedule), chain ν 3 p 71:

| width | γ per seed | fraction of the K5 ceiling 3.998e-05 | clean | N_.05 |
|---|---|---|---|---|
| 30 | 1.88, 1.91, 1.97 e-05 | 0.47 – 0.49 | 0.45 – 0.62 | 9, 17, 10 |
| 60 | 2.32, 2.27, 2.16 e-05 | 0.54 – 0.58 | 0.73 – 0.83 | 6, 8, 6 |
| 150 | 2.15, 2.25, 2.16 e-05 | 0.54 – 0.56 | 0.65 – 0.76 | 10, 8, 8 |

(Width saturates: going from 60 to 150 neurons does not help, so this is not a capacity limit.)
At ν 4 (chain n 3, p 71, pod) the same arm does not even separate the data: γ = −4.3e-07 (width 64, 3 seeds) and
−3.5e-08 (width 250, 3 seeds), clean fraction **0.000**, N_.05 27–30 — the net never acquires clean content at all
under normalised GD from a random start at degree 4 (the s^{ν−2} escape problem of THEORY_nsum §6).

**This is the load-bearing admission.** With the support constraint removed, random init + the same strong optimiser
lands in the junk basin at about half the ceiling with an inflated count. Therefore the fixed-support ladder's
agreement with Γ_S is produced by the *projection*, and the honest description is: "we measure, per support size,
the normalised margin attainable by a network restricted to that support, and it peaks at K*."

### 1d. Q1(c) — what the projection licenses about the unrestricted optimum

Formally, fixed support at K gives a number L(K) ≤ sup{γ : net with clean support ⊆ S_K}, i.e. a **lower bound per
K**. The theorem gives an upper bound per support, γ ≤ B·Γ_S with B = 2c_n(2p)^{−ν/2} (PROOF_AUDIT notation), and the
best known attainment constant for the chain is (n+1)^{−(ν−2)/2} (0.577 at ν 3, 0.25 at ν 4, 0.089 at ν 5). Two-sided
exclusion therefore works only where

    Γ_{S_K} · B  <  L(K*)  =  (n+1)^{−(ν−2)/2} · B · Γ_{S_{K*}}   ⟺   Γ_{S_K}/Γ_{S_{K*}} < 0.577  (ν 3).

At p 71 ν 3 the only K with Γ ratio below 0.577 is K = 2 (0.321). **The theorem rigorously excludes K = 2 and
nothing else**; the certified band is K ≥ 3 and open above. This is PROOF_AUDIT F1's statement and it must be in the
paper next to the ladder figure, or a referee will find it. What the ladder adds on top of the certified band is: a
matched *lower* bound curve whose peak is at K*, and the observation that the shortfall constant C(population) is
flat in K (measured 0.244–0.251 at ν 4, 0.122–0.127 for the 2^n cube), so the curve's *shape* is the programme's
shape and not an artefact of the populations varying with K. That flatness is a measurement and is worth its own row
in a table.

Not licensed: "the unrestricted maximum-margin net uses K* frequencies". Not licensed: "sparse beats dense" as a
theorem (it is measured, in this family, with this attainment constant).

### 1e. Q1(d) — is the optimiser a fair stand-in for "trained with weight decay"?

Partly, and the gap is nameable.

- **What it shares with training:** the objective is exactly CE + λ‖θ‖² on the full table, full batch, no
  momentum-flavoured implicit bias, λ annealed towards 0 as Wei et al. Thm 4.1 requires.
- **What it does not share:** (i) *normalised* GD (step = η‖θ‖·g/‖g‖) removes the small-gradient-at-small-norm
  pathology that INIT_AUDIT H6 identified as the actual mechanism of the init-scale effect (weight decay eating 61 %
  of the norm before escape); (ii) the exact golden-section line search on the global scale solves the slow radial
  mode *exactly* every 100 steps, which ordinary GD does over tens of thousands of steps and often never; (iii) it
  starts near a good point in the fixed-support arms.
- **Verdict:** as an *optimiser for the constrained problem* it is fair and, if anything, should be sold as such
  ("we solve the constrained problem well"). As a *model of training* it is not, and the paper should not blur the
  two. The honest sentence: "the ladder is an optimisation study of the constrained problem; §(free training) is the
  study of what ordinary weight decay does."

### 1f. How to present it, and the strongest alternative design

**Name.** "Support-constrained attainable margin Γ̂(K)" (or "restricted-support ladder"). Not "measured margin of
trained networks".

**Claim it supports.** "For each feature count K, the largest normalised margin attainable by a network of this
family restricted to K frequencies matches the programme value Γ_S to 1–3 % through the optimum, and peaks at K*.
Dense supports are strictly worse and, when optimised, collapse to 4–7 active frequencies."

**Claims it does not support** (and that the current drafts imply): that trained networks attain that peak; that the
count a run exhibits is the optimum; that the theorem pins K*.

**Strongest alternative designs, in order:**
1. **Two-sided per-K bracket.** For every K report both the rigorous upper bound B·Γ_{S_K} and the attained lower
   bound Γ̂(K), and shade the K's that are rigorously excluded. This turns a curve-match into an exclusion argument
   and is honest about how weak the exclusion currently is. Cheap (the numbers exist).
2. **Objective-space version of the same figure** (§2 below): minimise the *training objective* CE + λ‖θ‖² exactly
   over the global scale for each constructed K-net and show it is minimised at K = K* at the λ the runs actually
   used. This has no λ → 0 limit, no attainment constant, and no projection; it is the cleanest statement in the
   whole package and it is free.
3. **Support-free ascent with a measured count**: run direct normalised-margin ascent from random init with no
   support constraint and *measure* K at the optimum. If it lands at K*, the claim becomes "the optimum found by
   optimisation uses K* features" with no projection at all. Status: at ν 3 it does not get there yet (§4), so this
   is a target, not a result.
4. **Width-and-seed distribution for the fixed-support arms** so the ladder has error bars that are not 1e-5 wide.

---

## 2. THE CHEAP DECISIVE RESULT: the junk basin is **not** a global minimum of the training objective

`fvf_obj.py`, fp64, **no training at all**. For any weight direction the objective can be minimised exactly over the
global scale s (the net is ν-homogeneous, so logits scale as s^ν and the penalty as s²):

    obj*(direction) = min over s of [ CE(s^ν z) + λ s² ‖θ‖² ]      (golden section on log s, 60 iterations)

Because every free run has width m = 500 and every constructed net has m ≤ 72, the constructed direction **lives
inside the free run's own parameter space** (pad with zero neurons — they contribute nothing to the logits and
nothing to the norm). So obj*(constructed) < obj*(free endpoint) is a proof that the free endpoint is not the global
minimiser of the objective it was trained on.

| cell (all p = 71, full table) | λ | best constructed K: obj* | free endpoints obj* | gap | free γ / constructed γ |
|---|---|---|---|---|---|
| x², ν 3 (Morwani's net), archived `REP` init 4, 4 seeds | 1e-4 | **K = 5: 0.266717** | 0.277917, 0.278087, 0.280436, 0.280444 | **+4.2 %** | 2.27–2.65e-5 vs 5.58e-5 (0.41–0.48) |
| x², ν 3, archived `REP` init 1, 4 seeds | 1e-4 | K = 5: 0.266717 | 0.324543, 0.330597, 0.330658, 0.358546 | **+21.7 %** | 5.5–11.1e-6 vs 5.58e-5 (0.10–0.20) |
| chain n 2, ν 3, archived λ1e-5 cell, 4 seeds | 1e-5 | **K = 5: 0.039274** | 0.045443, 0.047829, 0.047950, 0.048604 | **+15.7 %** | 1.54–1.79e-5 vs 3.95e-5 (0.39–0.45) |
| chain n 3, ν 4, archived 120k baseline, 4 seeds | 1e-5 | **K = 4: 0.065151** | 0.092710, 0.095009, 0.106027, 0.112145 | **+42.3 %** | γ < 0 vs 2.11e-7 |

Three things fall out, all measured:

1. **The junk basin is a strictly suboptimal point of the training objective**, by 4 % (best free seed, x², λ1e-4) to
   42 % (ν 4). Every free endpoint sits at its own scale optimum (the line search returns s = 1.0000), so this is not
   an under-training artefact in the radial direction. Combined with Wei–Lee–Liu–Ma Thm 4.1 (the global minimiser of
   CE + λ‖θ‖^r converges to the maximum normalised margin as λ → 0), **the counts free GD exhibits are not the
   theory's object**, and RESULTS_NU4's "the excess count is a junk basin, not the optimum" is now proved rather than
   argued.
2. **The objective itself, at the λ the runs used, is minimised at K = K***: K = 5 at ν 3 (both families), K = 4 at
   ν 4 (K = 3 within 0.3 %). This is a *statics* statement with no λ → 0 limit and no attainment constant — the
   single most defensible empirical claim available, and it costs nothing.
3. The free x² endpoint's objective (0.2779–0.2804) is the objective of a constructed net with **K ≈ 7** (0.2801),
   while its γ is 0.41–0.48 of the K = 5 net's. The junk basin buys its (adequate) objective with junk and norm, not
   with features.

Caveat to state in the paper: obj* is compared at one λ per cell, the λ each cohort trained at; the constructed
family is the junk-free one, so this is a *lower* bound on how far the global optimum is below the free endpoints.

---

## 3. Q2 — routes out of the junk basin: ranking, mechanism, cost, and what a referee would accept

Ranked by (expected payoff × probability) / cost, with the status after today's runs.

| # | route | mechanism | why it might fail | cost | counts as "free training"? | status today |
|---|---|---|---|---|---|---|
| 1 | **Show the basin is not global** (`fvf_obj`) | exact scale minimisation of the same objective over a strictly better direction | nothing — it is arithmetic | ~0 | n/a (it is an argument, not a route) | **DONE, positive, §2** |
| 2 | **q = ν pre-training → plain ℓ₂** | at q = ν splitting a frequency over many neurons is free, so the junk-free representation is free; the net arrives clean, then plain ℓ₂ prunes | starts dense, may stop above K* (archived counts 4, 5, 6, 6, 6) | 1 training run + a resume | yes, but a two-phase recipe | archived 0.77–0.94 of ceiling, n = 5; re-run at ν 4 on the pod (§5) |
| 3 | **q-homotopy, q annealed ν → 2** | follow a path of near-global optima instead of jumping; the support collapses continuously | the path may be discontinuous at the point where the optimum becomes sparse | 1 run | yes | tested (§4, §5) |
| 4 | **Support-constrained solve + release** | the projection is what works (§1c); solve at K, then release and verify stability | it is not free training; it is a two-stage construction | 1 run per K | **no** — a referee will call it a construction | measured: released rungs are stable (§1c) |
| 5 | **Junk penalty then release** | the junk lemma says junk only costs margin; penalise the junk *fraction* of the logit table, then remove the penalty | the penalty is an explicit inductive bias towards the theory's own answer | 1 run | weakly — it is a regulariser, not the standard one | tested (§4, §5) |
| 6 | **Direct normalised-margin ascent from random init** | optimises γ itself, which is the object the theorem is about | at p ≥ 23 the assembly problem is combinatorial; at ν ≥ 4 escape from zero is the s^{ν−2} problem | 1 run | not "training with weight decay", but it *is* the max-margin object | tested (§4, §5); at p 11 it reached the constant exactly (PROOF_AUDIT) |
| 7 | **Better optimiser, free support** (normalised GD + exact scale line search + λ → 0) | removes the radial pathology INIT_AUDIT H6 blames for the init-scale effect | does not address the phase-assembly problem | 1 run | **yes, fully** | **tested: FAILS.** ν 3 → 0.47–0.58 of ceiling at widths 30/60/150; ν 4 → clean 0.000 (§1c, §5D) |
| 8 | **Decay-free warm-up / larger init / width** | keeps the net from being eaten before escape (INIT_AUDIT H6) | does not address junk | cheap | yes | already the standard recipe in the archived cells; not sufficient |
| 9 | **λ-homotopy with convergence at each stage** | Wei et al. is about global minimisers; continuation is the standard way to track a global branch | the branch that leaves θ = 0 is the one GD already follows; RESULTS_Q shows λ → 0 moves no count | cheap | yes | **prior null** (RESULTS_Q reading 2), not re-run |
| 10 | **Prune-and-regrow / phase re-randomisation of junky neurons** | the barrier looks like a *phase-assembly* problem: a frequency needs ≥ n+1 neurons with matched relative phases, and greedy GD builds partial, non-cancelling sets | fiddly, many knobs | medium | yes | **not tested — the most promising untried idea** (see §6 plan) |

**Why q = ν works and plain ℓ₂ does not (mechanism, consistent with everything measured).** Under plain weight decay
a frequency must be carried by a *coordinated* population — at least n+1 neurons whose phases cancel the mixed-sign
junk — and each extra neuron costs norm. GD builds neurons one at a time and greedily, so it reaches partial
cancellation and then has no downhill path to the coordinated set: the measured clean fractions are 0.4–0.7 and the
margin is ≤ 0. At q = ν the cost of splitting one frequency over many neurons is exactly zero, so the coordinated
population is free and the net arrives junk-free (clean 0.99 at q = 3, RESULTS_Q reading 3); switching to ℓ₂ then
only has to prune. This predicts the failure mode of the q-recipe: it starts dense and stops wherever pruning stalls,
which is why its counts (4, 5, 6, 6, 6) sit at or above K* rather than on it.

---

## 4. Local tests run today (chain n = 2, p = 71, ν = 3, width 100 unless stated; CPU, fp64 analysis)

`fvf_free.py`, one seed per route, every route finished with the *same* polish optimiser (normalised GD + exact
scale line search, λ 1e-5 → 1e-7, **no projection**) so that γ is comparable. Ceiling for this cell = 3.9981e-05
(the K = 5 fixed-support value, reproduced from random init in §1b).

| route | end of training: γ (frac of ceiling) / clean / N_.05 | after polish: γ (frac) / clean / N_.05 |
|---|---|---|
| `l2` (plain weight decay, 20k steps, the baseline) | 1.4478e-05 (0.36) / 0.673 / 9 | 2.0621e-05 (**0.52**) / 0.691 / 7 |
| `qnu` (q = ν = 3 for 12k, λ_q 1e-6, then plain ℓ₂ 8k) | 1.6626e-05 (0.42) / 0.751 / 9 | 2.2696e-05 (**0.57**) / 0.806 / 7 |
| `qanneal` (q annealed 3 → 2 over 7.2k, then ℓ₂) | 1.3623e-05 (0.34) / 0.649 / 11 | 2.0544e-05 (**0.51**) / 0.692 / 7 |
| `junk` (ℓ₂ + 3.0 × junk *fraction* for 6k, then pure ℓ₂) | — | 2.2970e-05 (**0.57**) / 0.838 / 6 |
| `ascent` (direct normalised-margin ascent, Adam, 12k, no CE, no wd) | 1.6963e-05 (**0.42**) / 0.677 / 4 | — |

Readings, and they are mostly nulls:

- **Every route stalls at 0.5–0.6 of the ceiling with 30–35 % junk.** None reaches the 0.77–0.94 the archived ν 4
  recipe reached, and none is junk-free. The counts (N_.05 6–7) sit just above K* = 5 exactly as the free runs do.
- **`qanneal` is not better than `qnu`, and here it is no better than plain ℓ₂.** The continuous homotopy does not
  beat the hard switch; if anything it spends its budget in the wrong place. *Caveat:* this local `qnu` had only 12k
  steps at q = ν against the archived 120k, and it ends at clean 0.81 rather than the archived 1.000 — so this is
  an under-trained version of the recipe, not a refutation of it. The archived ν 4 numbers stand.
- **The junk penalty is the best of the local routes** (clean 0.838, N_.05 6, 0.57 of ceiling) but it is an explicit
  inductive bias towards the theory's own answer and would not be accepted as "free training" without heavy caveats.
- **Direct margin ascent reaches 0.42 of the ceiling with 32 % junk and lands on 4 frequencies.** At p 11 it found
  the exact attainment constant (PROOF_AUDIT); at p 71 it does not. The assembly problem scales with p.
- Note the polish step alone adds 40 % to γ for every route (0.36 → 0.52 for the baseline): a large part of what
  looks like "basin" is unconverged radial/soft directions that a good optimiser fixes — but only a part.

---

## 5. Pod runs (RTX 5090 secure, $0.99/h, pod `fvf-nu4`, deleted after the pull)

Phases: `R4` random-init-in-span ladder at ν 4 (chain n 3, p 71, K 2…16, width 12K, 3 seeds); `R3` the same at ν 3
on p 113 (chain) and for Morwani's x² net at p 113; `RF4` free-support random init at ν 4 and at ν 3 p 113;
`FR4` the five free-training routes at ν 4 with the archived recipe (init 14, decay-free warm-up, λ 1e-5, m 500).

The pod ran ~50 min (**$0.83 spent**, terminated and confirmed 404). The phases shared one GPU, so the ν 4 ladder
reached only K = 2, 3, 4 and `FR4` only its baseline arm; everything below is what completed. Raw metrics and logs
are in `FINAL_RUNS/FVF/pod/`.

**A. Random-init-in-span ladder at p = 113 (the paper's prime), gated chain, ν 3, 3 seeds, width 6K.**
K*(113, ν 3) = 6 from `kstar.py`. **No construction is used anywhere in these runs.**

| K | 2 | 3 | 4 | 5 | **6** | 7 | 8 |
|---|---|---|---|---|---|---|---|
| γ (mean, ×1e-5) | 0.451 | 1.069 | 1.580 | 1.691 | **1.735** | 1.635 | 1.594 |
| relative | 0.260 | 0.616 | 0.911 | 0.975 | **1.000** | 0.942 | 0.919 |
| clean | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| N_.05 (per seed) | 2,2,2 | 3,3,3 | 4,4,4 | 5,5,5 | 6,6,6 | 7,7,5 | 7,6 |

**The argmax is K = 6 = K\*, with an interior peak (K5 0.975 < K6 1.000 > K7 0.942), from random initialisation.**
Seed spread is again ~0 through the peak; it opens only above it (K7 has one seed at 1.540e-5 that fell to 5 active
frequencies).

**B. The same for Morwani's x² net at p = 113** (cube population width 12K, 3 seeds; rungs K ≤ 5 did not finish):

| K | **6** | 7 | 8 | 10 |
|---|---|---|---|---|
| γ (mean, ×1e-5) | **2.269** | 2.218 | 2.175 | 2.015 |
| relative | **1.000** | 0.977 | 0.958 | 0.888 |
| clean | 0.998 | 0.991 | 0.985 | 0.976 |

Monotone decline above K* = 6, and the x² family shows the junk freedom being used (clean drifts 0.998 → 0.976).

**C. Random-init-in-span at ν 4 (chain n 3, p 71, width 12K, 3 seeds) — partial, and it does NOT fully survive:**

| K | random-init γ | archived constructed-start γ | ratio | N_.05 (rand) |
|---|---|---|---|---|
| 2 | 1.0588e-07 | 1.0584e-07 | **1.000** | 2,2,2 |
| 3 | 2.0323e-07 | 2.0322e-07 | **1.000** | 3,3,3 |
| 4 | 1.6133e-07 | 2.0970e-07 | **0.769** | 3, 2 |

At ν 4 the control reproduces the constructed ladder *exactly* at K = 2 and K = 3 and then **fails at K = 4**, where
random starts collapse to 2–3 active frequencies at 77 % of the attainable value — with width 48 = 3× the minimal
16, so it is not capacity. **The de-circularising control passes at ν 3 (two primes, two families) and is unproven
at ν 4.** That must be stated; it is the weakest point in the whole package and it sits exactly where RESULTS_NU4's
headline (ν 4 argmax 4 ≈ 5) lives.

**D. Free-support random init at ν 4** (`randfull`, widths 64 and 250, 3 seeds each): γ = −4.3e-07 and −3.5e-08,
clean **0.000**, N_.05 27–30 — see §1c.

**E. Free-training baseline at ν 4** (`l2`, m 500, archived recipe, 60k steps): at 20k steps γ = −1.11e-08,
clean 0.444, N_.05 7 — reproduces the archived junk basin exactly. The other four routes did not finish.

---

## 6. Recommended experiment plan for the paper (and what it costs)

**Immediately, for free (CPU, no pod):**

- **P0. Objective-space ladder** (`fvf_obj.py`) for every cell that appears in the paper: p ∈ {59, 71, 97, 113},
  ν ∈ {3, 4, 5}, both families, plus S5/A5 if the group section stays. Report obj*(K) and obj*(free endpoints).
  This is the cheapest and most defensible figure in the package. **Cost: ~0.**
- **P1. Two-sided bracket** per K: rigorous upper bound B·Γ_{S_K} and attained Γ̂(K); shade the excluded K's.
  **Cost: ~0** (numbers already computed).

**On one 5090 (~$0.9/h), ≈ 6 GPU-hours ≈ $6 total, comfortably before 09-25:**

| cell | arms | runs | est. time | est. $ |
|---|---|---|---|---|
| **chain ν 4, p 71** — **highest priority**, the control that failed today | random-init-in-span ladder K ∈ {2..8, 10, 16}, widths {3×, 6×, 12× minimal}, 6 seeds, 12k steps, plus a restart-best-of-4 variant | 150 | 2.5 h | 2.3 |
| chain ν 3, p 113 | finish the ladder K ∈ {9..16} and the x² rungs K ∈ {2..5}, 4 seeds | 40 | 0.5 h | 0.5 |
| chain ν 5, p 59 (tied) | same, K ∈ {2..6, 8}, 4 seeds | 24 | 0.5 h | 0.5 |
| x² ν 3, p 113 | same (Morwani's own net), K ∈ {2..8, 10}, 4 seeds | 32 | 0.5 h | 0.5 |
| all four | **released** control (`--no_proj`) at K ∈ {K*−2, K*, K*+3, K*+7}, 2 seeds | 32 | 0.5 h | 0.5 |
| all four | **free-support** control (`--start randfull`), widths {minimal, 4×, 500}, 4 seeds | 48 | 1.0 h | 0.9 |
| chain ν 4, p 71 | free-training routes: `l2`, `qnu`, `qanneal`, `junk`, `ascent`, 8 seeds each, with polish | 40 | 1.0 h | 0.9 |
| chain ν 4, p 71 | **prune-and-regrow** (route 10): every 5k steps, re-randomise the phases of the neurons in the worst-cancelling frequency; 8 seeds × 3 settings | 24 | 0.5 h | 0.5 |

Run 4–5 processes per GPU (RESULTS_NU4 measured ~2× wall-clock scaling), one pod, delete on the verified pull.
Pre-register before launching: (a) the random-init ladder's argmax must equal the constructed ladder's argmax in
≥ 3 of 4 cells, else the ladder is reported as an upper-envelope estimate only; (b) the free-support control is
expected to stay below 0.7 of the ceiling — if it does not, promote it to the headline; (c) the q-recipe's count
distribution over 8 seeds must have its mode in {K*, K*+1}, else report it as "clean but not optimal".

**Abstract consequence (needs a decision before 09-25).** The submitted sentence is
"we empirically show that the normalized margin attained by these networks peaks at the predicted optimum".
As things stand that is supported only with *attainable* in place of *attained*. Two honest repairs:
(i) "the normalized margin **attainable** by these networks peaks at the predicted optimum" — supported today; or
(ii) keep "attained" and add the objective-space result as the supporting evidence ("networks trained with weight
decay do not reach it; the solution that does is the predicted one, and it has strictly lower training objective
than the trained networks") — also supported today, and arguably a more interesting paper.

---

## 7. Nulls and limitations, stated plainly

- The fixed-support ladder's seed variance at ν 3 is ~0 (5 significant figures). It is a numerical optimisation of an
  analytic problem; error bars must not be presented as if they came from a sample of networks.
- Above K* the random-init ladder is 5–13 % below the constructed ladder in some seeds; that part of the curve is a
  lower bound, not a measurement.
- **At ν 4 the random-init control fails at K = 4** (0.769 of the constructed value, collapsing to 2–3 active
  frequencies, width 3× minimal, 2 seeds). Until that is fixed, the ν 4 ladder is de-circularised only at K ≤ 3, and
  the ν 4 argmax claim rests on the constructed start alone. This is the single most exposed claim in the paper.
- The pod's ν 4 ladder (K ≥ 5), the ν 3 p 113 rungs above K 8, the x² p 113 rungs below K 6, and four of the five
  ν 4 free-training routes did not finish inside the 50-minute pod window; they are queued in §6, not reported here.
- The local free-training routes (§4) used only 20k steps and one seed; `qnu` there is an under-trained version of
  the archived recipe (clean 0.81 vs the archived 1.000) and is **not** evidence against it.
- The random-init control needs ≈ 2× the minimal width; at exactly minimal width it fails (0.59 of the ceiling).
- The free-support control (random init, no projection, same optimiser) does **not** reach the ceiling at ν 3 and
  fails outright at ν 4. There is currently **no** route from random init and plain weight decay to the optimum that
  a referee would accept without qualification; the best is the two-phase q = ν → ℓ₂ recipe (0.77–0.94 of ceiling,
  n = 5 usable runs, archived).
- The objective comparison uses one λ per cell and the junk-free constructed family; it is a lower bound on the gap.
- Everything here is full-table training. PROOF_AUDIT F6 stands: the junk lemma is a full-table statement, and the
  frac < 1 cohorts are outside the theorem.
