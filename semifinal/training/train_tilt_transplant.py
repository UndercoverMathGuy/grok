"""Tilt transplant: causal carrier test for the W_E init energy spectrum.

Design: three same-mask natural runs runs/seed0/seed{37099,55327,93077}
(p=113, data_seed 0, epoch-0 checkpoints). For each ordered pair (R <- D),
build a surgical init from R's epoch-0 weights in which every frequency's
(d x 2) W_E Fourier block is rescaled to DONOR D's per-frequency energy —
R keeps its own directions, DC, '=' column, positional embedding,
attention, MLP, unembedding, and optimizer stream. 12k epochs.

Why this design: "flatten the tilt -> committees change" is a weak
necessity proof because committee identity is chaotic to sub-threshold init
perturbations — ANY edit moves it. The transplant has two named attractors
(D's committee vs R's committee), both same-mask, and asks which twin the
hybrid resembles.

PRE-REGISTERED PREDICTIONS:
  P-TR1 (primary): pooled over the 6 transplants, the final committee
        contains MORE members unique to D than members unique to R.
        [Outcome in SEMIFINAL.md: FAILED 2-vs-8 — energy alone, without the
        alignment factor G_k, barely moves the product T_k. The failure is
        part of the two-knob story, so the script is kept as-is.]
  P-TR2: menu closure at post-transplant init-energy rank.
  P-TR3: exact reproduction of D's committee is NOT expected (chaos guard).

Runs land in runs/transplant/<R>_from_<D>. Idempotent.
"""
import numpy as np

from _shared import ROOT, freq_energy, report, save_surgical_ckpt

from grok.config import Config                          # noqa: E402
from grok.model import Transformer                      # noqa: E402
from grok.fourier import Fourier                        # noqa: E402
from grok.train import train                            # noqa: E402

P = 113
EPOCHS = 12000
RUNS = {name: ROOT / "runs" / "seed0" / name
        for name in ("seed37099", "seed55327", "seed93077")}

fourier = Fourier(P)
nf = P // 2

# --- gather epoch-0 embeddings, energies, final committees -----------------
from _shared import committee_from_coeffs               # noqa: E402
info = {}
for name, d in RUNS.items():
    cfg = Config.load(d / "config.json")
    assert cfg.p == P and cfg.data_seed == 0
    model = Transformer(cfg)
    model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
    W_E = np.array(model.W_E, dtype=np.float64)
    F = W_E[:, :P] @ fourier.basis.T
    z = np.load(d / "spectra.npz")
    info[name] = dict(cfg=cfg, W_E=W_E, F=F, E=freq_energy(F, nf),
                      comm=set(committee_from_coeffs(z["coeffs"][-1])))

print(__doc__, flush=True)
for name, i in info.items():
    print(f"{name}: natural committee {sorted(i['comm'])}", flush=True)
pairs = [(r, dn) for r in RUNS for dn in RUNS if r != dn]

model = Transformer(info["seed37099"]["cfg"])            # reused shell


def transplant_ckpt(r, dn):
    R, D = info[r], info[dn]
    Fm = R["F"].copy()
    for k in range(nf):
        s = np.sqrt(D["E"][k] / R["E"][k])
        Fm[:, 2 * k + 1] *= s
        Fm[:, 2 * k + 2] *= s
    W = R["W_E"].copy()
    W[:, :P] = Fm @ fourier.basis
    model.load_weights(str(RUNS[r] / "checkpoints" / "epoch_00000.safetensors"))
    ck = save_surgical_ckpt(model, W, f"transplant_{r}_from_{dn}")
    # sanity: post-surgery energies == donor energies
    F2 = np.array(model.W_E, dtype=np.float64)[:, :P] @ fourier.basis.T
    err = np.abs(freq_energy(F2, nf) - D["E"]).max() / D["E"].mean()
    print(f"  surgical ckpt {ck.name}: max energy err {err:.1e}", flush=True)
    return ck


tally_D = tally_R = 0
for r, dn in pairs:
    run_dir = ROOT / "runs" / "transplant" / f"{r}_from_{dn}"
    if (run_dir / "spectra.npz").exists():
        print(f"skip {run_dir.name} (exists)", flush=True)
    else:
        ck = transplant_ckpt(r, dn)
        c2 = Config.load(RUNS[r] / "config.json")
        c2.num_epochs = EPOCHS
        c2.save_every = 4000
        print(f"\n=== transplant {r} <- tilt({dn}) ===", flush=True)
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=2000)
    final = report(run_dir, info[r]["comm"] | info[dn]["comm"])
    uR = info[r]["comm"] - info[dn]["comm"]
    uD = info[dn]["comm"] - info[r]["comm"]
    tally_D += len(final & uD)
    tally_R += len(final & uR)
    print(f"    adopted D-unique {sorted(final & uD)}  "
          f"kept R-unique {sorted(final & uR)}", flush=True)

print(f"\nP-TR1 tally: D-unique adopted {tally_D} vs R-unique kept {tally_R} "
      f"(prediction was D > R; observed outcome refuted it)", flush=True)
