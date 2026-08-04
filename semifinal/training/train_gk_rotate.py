"""Causal control of the ALIGNMENT factor G_k (the second knob).

T_k = E_k (W_E energy) x G_k (alignment of frequency k's W_E subspace with
the attention OV pathway). The energy knob is causally proven (dose farm).
This suite turns the OTHER knob: rotate frequency 7's W_E 2-plane toward
the OV map's high-gain input directions AT EXACTLY FIXED ENERGY, choosing
angles so the transmitted energy T_7 is multiplied by the same factors as
the energy doses {1.05, 1.20, 2.25}. Base: og_seed0/seed27058 (same base
as the energy curve). Runs land in runs/gkrotate/gain_<g>. Idempotent.

PRE-REGISTERED PREDICTIONS:
  P-C1 (primary): outcomes match the energy-dose curve AT MATCHED T_7 GAIN
        — i.e. T_k, not E_k, is the causal variable.
  P-C2: per-frequency W_E ENERGY is unchanged by the surgery (verified
        numerically below) — so any effect cannot be an energy effect.
  P-C3: menu closure and bystander chaos as usual.
"""
import numpy as np

from _shared import (ROOT, committee_from_coeffs, freq_energy, grok_epoch,
                     save_surgical_ckpt)

from grok.config import Config                          # noqa: E402
from grok.model import Transformer                      # noqa: E402
from grok.fourier import Fourier                        # noqa: E402
from grok.train import train                            # noqa: E402

P, TARGET = 113, 7
nf = P // 2
fourier = Fourier(P)
GAINS = [1.05, 1.20, 2.25]
EPOCHS = 10000

BASE = ROOT / "runs" / "og_seed0" / "seed27058"
cfg = Config.load(BASE / "config.json")
model = Transformer(cfg)
E0 = BASE / "checkpoints" / "epoch_00000.safetensors"
model.load_weights(str(E0))
W_E = np.array(model.W_E, dtype=np.float64)
F = W_E[:, :P] @ fourier.basis.T
at = model.blocks[0].attn
W_V = np.array(at.W_V, dtype=np.float64)
W_O = np.array(at.W_O, dtype=np.float64)
h, dh, _ = W_V.shape
OVs = [W_O[:, i * dh:(i + 1) * dh] @ W_V[i] for i in range(h)]

B0 = F[:, [2 * TARGET - 1, 2 * TARGET]].copy()          # (d, 2) freq-7 block


def T7(B):
    return sum(((OV @ B) ** 2).sum() for OV in OVs)


# high-gain input directions of the summed transmission operator
M2 = sum(OV.T @ OV for OV in OVs)                       # symmetric PSD (d, d)
w_eig, V = np.linalg.eigh(M2)
top = V[:, np.argsort(w_eig)[::-1][:2]]                 # (d, 2)


def rotated(alpha):
    """Mix the block toward the top-gain plane, preserving each column's
    norm exactly (so per-frequency W_E energy is untouched)."""
    B = (1 - alpha) * B0 + alpha * top * np.linalg.norm(B0, axis=0)
    B *= np.linalg.norm(B0, axis=0) / np.linalg.norm(B, axis=0)
    return B


t0 = T7(B0)


def gain(alpha):
    return T7(rotated(alpha)) / t0


print(__doc__, flush=True)
print(f"base T_7 {t0:.4f}; max achievable gain {gain(1.0):.2f}", flush=True)

for g_target in GAINS:
    if gain(1.0) < g_target:
        print(f"gain {g_target} unreachable (max {gain(1.0):.2f}) — skipped",
              flush=True)
        continue
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if gain(mid) < g_target:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    B = rotated(alpha)
    Fm = F.copy()
    Fm[:, 2 * TARGET - 1] = B[:, 0]
    Fm[:, 2 * TARGET] = B[:, 1]
    # P-C2 verification: energies untouched
    err = (np.abs(freq_energy(Fm, nf) - freq_energy(F, nf)).max()
           / freq_energy(F, nf).mean())
    name = f"gain_{int(round(g_target * 100)):03d}"
    print(f"\n=== {name}: alpha {alpha:.4f}, achieved T_7 gain "
          f"{gain(alpha):.4f}, max energy err {err:.1e} ===", flush=True)
    run_dir = ROOT / "runs" / "gkrotate" / name
    if (run_dir / "spectra.npz").exists():
        print(f"skip {name} (exists)", flush=True)
    else:
        W = W_E.copy()
        W[:, :P] = Fm @ fourier.basis
        model.load_weights(str(E0))
        ck = save_surgical_ckpt(model, W, f"gkrotate_{name}")
        c2 = Config.load(BASE / "config.json")
        c2.num_epochs = EPOCHS
        c2.save_every = 4000
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=4000)
    z = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(z["coeffs"][-1])
    tr = np.abs(z["coeffs"][:, TARGET - 1])
    print(f"RESULT {name}: grok@{grok_epoch(z)} committee {final} f{TARGET} "
          f"peak {tr.max():.0f} final {tr[-1]:.0f} "
          f"acc {z['test_acc'][-1]:.4f}", flush=True)
print("\nGK ROTATE DONE", flush=True)
