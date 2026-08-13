# Uniqueness + needed audit: init-time mode-spectrum theory (2026-08-13)

Proposed core: derive the committee selector as the linearized init-time growth spectrum λ_k
(exact, numerical, per-seed, weights-only), with degeneracy → chaos as the determinism
boundary, mode-interaction coefficients → veto grammar, competition → capacity K*, and the
existing compiler as the causal arm.

## Uniqueness verdict per component

| Component | Nearest prior work | Verdict |
|---|---|---|
| Per-seed frequency selector from init | **He et al. 2602.16849** — 2-layer MLP, *small-init* gradient flow decouples per (neuron, frequency); winner = initial spectral magnitude + phase alignment; "final predictor entirely determined by the random initialization" | **HOTTER THREAT THAN RECORDED.** They own the per-neuron selector in the MLP small-init regime. Bare "λ_k in a transformer" = He-flavored increment. Novelty must come from: transformer/OV (their decoupling breaks under attention), standard init scale (their asymptotics don't apply), network-level committee vs per-neuron winner, exact numerics vs asymptotic regime |
| Menu/dynamics theory | Li₂ (ICLR 2026) + Tian's scaling-laws follow-up (arXiv 2509.21519) | Safe — both 2-layer, population-level, no per-seed selection, no attention, no degeneracies, no causal interventions. Dialogue partners |
| Early kernel/feature alignment | Silent alignment (Atanasov et al. 2111.00034); early-alignment two-edged-sword (2401.10791) | Safe — regime-bound (small init, whitened data), continuous kernel eigenstructure, not discrete per-seed circuit identity. Cite |
| Degeneracy → determinism boundary | Frankle et al. linear mode connectivity / instability to SGD noise (1912.05671) | **OPEN — our most unique claim.** LMC is basin-level, measured *during* training, no from-init spectral criterion, no circuit identity. Nobody has "gapped spectrum ⇒ deterministic circuit, degenerate ⇒ provably coin-flip" |
| Triadic resonance → veto grammar | Nonlinear-mode-coupling machinery exists in physics (quadratic mixing → i±j combination frequencies); zero application to modular-arithmetic nets found | **Unique if it closes.** Even the measured-coupling-coefficient version (no derivation) is new |
| Capacity from mode competition | Li₂ scaling laws (sample size/representation, not committee size) | Semi-unique; no direct competitor found |
| Causal init compiler | (prior audit) He et al. do no interventions | Still unique |

## Needed — the gap list, in order

1. **Probe ceiling at n=96** (learned readout on init weights): bounds what ANY selector can
   achieve; tells us if 0.76 is functional error or physics. Local, hours.
2. **λ machinery that survives the transformer at standard init**: exact linearized operator via
   HVP/Lanczos on frequency subspaces. Must verify the linear window actually governs selection
   (lock-in by ~e500-1000 per catch-the-wave data).
3. **Beat He et al.'s regime, don't repeat it**: show their small-init/decoupled selector
   transfers poorly to the standard-init transformer while exact λ works. Converts the scoop
   threat into the field-belief-correction hook.
4. **Gap-conditional determinism** on twins/swap data — the headline-unique claim's evidence.
5. **λ-collapse of the Phase B dose×K sweep** (96 arms on disk, free): adoption probability as a
   single curve in λ-margin; ×1.10 coin-flip boundary should sit at λ-degeneracy.
6. **Disagreement arms** (equalize T_k, split λ; and vice versa) — the causal discriminator,
   only new GPU spend (~$1-2).
7. **Triad coupling coefficients** measured numerically; derivation is the stretch goal.
8. **Related-work spine**: He et al., Li₂ + 2509.21519, Manifolds, silent alignment, Frankle
   LMC (position: basin-level vs our circuit-level determinism).

Kill-switches: if (1) probe ceiling ≈ 0.76 AND (4) gaps don't sort the chaos → physics limit,
fall back to read/write/limits + attack-surface framing. If λ ≈ T_k in AUC but gaps DO sort
chaos → the determinism-boundary paper survives with T_k as the practical readout.
