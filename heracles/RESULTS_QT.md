# QT — polynomial targets on the quadratic full-data recipe (2026-09-16, pod yhzassx562govv 4090, 16 runs, ~$0.40)   [runs/QT, quad_task.py]

Question: does the ansatz survive a target that is not a group law? Two targets on Z_p, recipe as RESULTS_QM.md
(plain l2 1e-4, lr cap 12.8, init x1 at p 71 / x4 at p 113, 4 seeds, full dataset, 40k steps, m 500, x^2 activation).
  sq   : c = a^2 + b^2            separable, f(a) + g(b): a free embedding reduces it to addition on residues
  norm : c = a^2 + a b + b^2      not separable (norm form of Z[omega]); symmetries: swap, (a,b)->(b,-a-b), (a,b,c)->(ta,tb,t^2 c)
Instruments: clean count = frequencies of the level-set projection g(D), D = f(a,b) - c, at 2% / 5% energy;
per-neuron a^2-frequency share = max_k ||P_k u_i||^2 / ||u_i||^2 with P_k the projector onto span{cos k a^2, sin k a^2}
(the least-squares split over all k in quad_task.py is NOT valid: {cos k a^2} is overcomplete on the (p+1)/2 residues).

| target | p | acc | K at 5% (4 seeds) | K at 2% | neuron a^2-freq share | clean amps sorted (seed 0) | gamma med | addition (QM) |
|---|---|---|---|---|---|---|---|---|
| sq | 71 | 1.000 | 6, 11, 6, 5 | 14, 17, 13, 14 | 0.94, 0.94, 0.91, 0.90 | 1, .60, .57, .56, .50, .40, .32 | 2.1e-5 | K 5% = 5,5,3,2; gamma 9e-6; amps 1, .64, .60, .54, .52, .16 |
| sq | 113 | 1.000 | 7, 6, 4, 5 | 16, 15, 21, 17 | 0.98, 0.98, 0.98, 0.97 | 1, .89, .87, .83, .80, .79, .71, .64, .54 | 9.5e-6 | K 5% = 8,6,7,6; gamma 6.4e-6; amps 1, .99, .59, .59, .58, .56, .55, .52, .43, .17 |
| norm | 71 | 1.000 | 0 | 32-35 | (no basis fits: R^2 0.52-0.57) | flat | 8.7e-6 | |
| norm | 113 | 1.000 | 0 | 5-9 | (R^2 0.40-0.44) | flat | 2.6e-6 | |

Symmetry test on norm (corr of the trained logit table with its image): swap 1.000 (U = V learned), rotation
(a,b)->(b,-a-b) 0.34-0.36, scaling t = 2,3,5 0.39-0.49; variance explained by (f(a,b), c) only 0.16-0.21.

Reading.
1. sq: the net builds the reduction (every neuron a single cos(k a^2) feature, R^2 = 1.000 in the a^2 function
   space) but the committee is WIDER and FLATTER than addition's at the same prime, and its normalised margin is
   1.5-2x higher. Reason (exact): a -> a^2 is 2-to-1, the data realise only residue pairs, translation is no longer
   a symmetry of the level sets, so the junk lemma's premise fails: non-clean content can raise the margin and the
   coverage functional is not M_K. The closed form K* = argmax M_K/K^{nu/2} does not apply; the general form
   (sup margin/cost, |supp| <= K) does, as a program on the restricted domain, not yet solved.
2. norm: memorised. Accuracy 1 on the full table with no Fourier structure in either basis and the table not
   a function of the task invariant. Weight decay alone did not find a symmetric solution here; the theory has
   nothing to say and the net builds no circuit.
3. Scope for the paper: the ansatz is about targets with a symmetry group transitive on the margin's level sets,
   i.e. f(a) + g(b) - c on a group with a free embedding. The first target outside that class changed the count
   for the stated reason. "Algebraic tasks" is not earned by these two; the earned expansion is the n-operand
   sum with x^n activation (AM-GM-exact nu = n + 1), not yet run.
