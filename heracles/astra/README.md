# ASTRA — unrestricted ansatz optimality

Target: for full-table modular addition and the gated-chain architecture, prove that the
share-weighted ansatz attains the unrestricted maximum of normalized margin. If achieved,
connect this value theorem to weakly regularized global minimizers and distinguish existence
of an optimal K-feature representation from uniqueness/convergence of feature counts.

Everything in this directory is new work. No parent-directory research files are modified.
Local CPU only; no commits. Provisional arguments and numerical evidence are not proofs.

**Logarithmic scaling (2026-09-25):** `LOGARITHMIC_SCALING.md` proves, for every
fixed s>1, Gamma_s(p)=Theta_s((log p)^(1-s)) and K*=Theta_s(log p) for EVERY
programme optimizer. The existing two-sided bounds transfer the value order
to unrestricted product networks: at fixed 2<=b<nu, gamma*_b(p) has order
p^(-nu/2)*(log p)^(1-nu/b), and minimum unit-margin P_b has order
p^(b/2)*(log p)^(1-b/nu). This does not require the unresolved sharp
interference-free ceiling. Any fixed-factor optimal unrestricted network
needs Omega(log p) nonzero clean frequencies; an O(log p) upper count is
proved for programme optimizers, not all unrestricted optimizers. These are
order bounds, not leading constants or training convergence claims. The
proof uses a random-frequency lower construction, a positive-Fourier
exponential upper bound, and a delete-and-rescale support bound. Supplementary
checks are in `logarithmic_scaling_verification.json`.

**S5 result (2026-09-24):** `S5_PROJECTOR.md` gives an exact 193-neuron product
network at q=2,nu=4 that beats the entire interference-free ceiling by **24.694%**.
It uses a rank-three projection of the standard representation plus one sign neuron.
The same construction at nu=3 uses 49 neurons and beats the schoolbook reference
by **2.979%**; the true nu=3 symmetric price is still unknown, so the full
degree-three comparison remains open. `S5_COUNTEREXAMPLE.md` gives a separate
65-neuron nu=4 witness with an 8.192% gain. Both are proved analytically and
independently verified on all 1,728,000 input rows. Weights and verification
reports are saved as `s5_projector_nu*.npz`, `s5_counterexample_nu4.npz`, and
`s5_*_verification.json`. `s5_ceiling_verify.py` independently certifies the
degree-four symmetric ceiling by exact enumeration of all 43 vertices and
rational bounds on square roots. These group counterexamples do not change
the odd-prime modular-addition status below.

**Latest unrestricted extension:** `PHASE_FRAME_EXTENSION.md` proves global dense
optimality for every odd prime p and nu>=3 on nu-4/[3(p-1)]<=q<=nu, with a sharper
explicit formula in section 5. At p=5,nu=3 it reaches EVERY 5/2<=q<=3. Below the
endpoint, all optimizers in these ranges are junk-free with nu neurons per frequency.
The proof retains all frequency contributions, rotates their phase frames, and
separates phase loss from excess margin. Exact checks are in `phase_frame_verification.json`.
Section 7 gives an exact p=5,q=2 counterexample to the retained-matrix relaxation,
NOT to the full network theorem: a cheap network matching the matrix violates a
margin constraint. The full q=2 comparison must still price the missing components.

**Earlier explicit unrestricted result:** `EXPLICIT_ENDPOINT.md` proves global dense
optimality for every odd prime p and nu>=3 throughout the depth-uniform strip
nu-6/(9p-7)<=q<=nu. The sharper bound is q>=nu*c*h/(1+c*h),
where h=(p-1)/2 and c=nu+sqrt(nu^2+8nu).
For q<nu in this interval, every optimizer is junk-free, with exactly nu neurons
at each frequency. At nu=3,p=5, the simpler rational bound is q>=54/19.
The direct proof uses a matrix projection estimate with constant 2, valid for every
competitor, followed by a spectral moment bound. It has no purity or width restriction
and no compactness step. Exact checks are in `explicit_endpoint_verification.json`.
The full q=2 and sparse-feature questions remain open.

`CONTINUATION.md` additionally extends dense unrestricted LOCAL optimality at nu=3
to every 2<=q<3. If global dense optimality is lost below the explicit interval,
a competitor outside a fixed neighborhood must tie it at the transition; local
stability alone cannot continue the global theorem through that transition.

The earlier `ENDPOINT_NEIGHBORHOOD.md` gives a separate nonquantitative proof via
endpoint equality, entropy selection, and a uniform local cost comparison.
`endpoint_verify.py` and `endpoint_verification.json` retain its exact checks.

The single proof question is whether the worst-case margin discount and the coverage discount
always pay for any representation-cost discount from same-frequency or cross-frequency junk.
We retain the whole network until a reduction is actually justified.

The intended theorem is not assumed true in untested edge cases. Every claimed obstruction
must be checked on the full row/class table, not only on a surrogate objective.

`PROJECTION_OBSTRUCTION.md` answers the direct junk-deletion question analytically.
Clean projection improves raw margin, but an explicit positive-margin cubic-phase family
has strictly better normalized margin than the cheapest representation of its projection.
A different, explicitly constructed clean spectrum beats that family at large p.
Thus a general replacement must allow the clean spectrum to change; merely deleting
junk cannot be a universal proof, even on positive-margin pure-neuron networks.

Earlier extension: `GENERAL_DEPTH.md` proves the unrestricted GLOBAL theorem at p=3
for EVERY depth nu>=3 and nu-1<=q<=nu, with a slightly wider explicit range in section 5.
It also proves the pure one-frequency theorem at every odd prime in that range. All
proofs are analytic. The full lower-q range remains open: section 6 gives an exact
counterexample to extending the largest-junk-coefficient lemma to q=2 at any nu>=4;
this is not a counterexample to the margin theorem itself.

`P3_ANALYTIC.md` proves unrestricted GLOBAL ansatz optimality at p=3,
nu=3, for the entire interval 2<=q<=3, by a fully analytic argument. At q<3, equality
forces zero junk and exactly three nonzero neurons. The same population inequality proves
the exact pure one-frequency theorem at every odd prime, with arbitrary width and junk.
This replaces the interval-certified last step of opus55's p=3 proof and extends its
stated exponent range. The unrestricted multifeature theorem at general p remains open.

`OPTIMIZER_ATTACK.md` starts from an attained hypothetical global
counterexample. It proves a finite-replacement criterion, a finite-deletion bound excluding
vanishing nonzero atoms, and precise multiplier conditions. It also proves two analytic
counterexamples to tempting shortcuts. It uses no numerical evidence. The missing
addition-specific replacement, and therefore the unrestricted ceiling, remain open.

`LOCAL_THEOREM.md` proves unrestricted local optimality of every optimal
ansatz at n=2,q=2, for every odd prime, including extra zero neurons. It also proves dense
suboptimal local maxima at p=11 and suboptimal regularized CE local minima for all
sufficiently small lambda. The global ceiling is still open. The proof combines prime
Fourier uncertainty with an exact, prime-independent block Hessian certificate.

Read `ARGUMENT.md` for the unified derivation and exact current status. The unrestricted
sharp bound remains open. New proved results here are the finite ansatz vertex formulation,
generic feature-count uniqueness, unrestricted width compression/attainment, and the direct
multiclass weak-regularization limit including its conditional ansatz-achievability consequence.

`vertices.py` certifies the ansatz optimum at r=2/3 for p=3,5,7,11,13,17 using exact algebraic
linear algebra and outward intervals; results are in `vertices_p*.json`. `verify.py` and
`check_reductions.py` supply numerical consistency checks, recorded in `verification.json`
and `reduction_checks.json`. The `raw_margin.py` checks are restricted to p=3 so their
one-frequency reference value cannot be mistaken for a certified multifeature optimum.

Reproduce from the project root:

```sh
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/vertices.py --primes 3 5 7 11 13 17
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/verify.py
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/check_reductions.py
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/local_block.py
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/local_exact.py
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/local_duals.py
UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/local_support_check.py
OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/astra/local_multifeature.py
```
