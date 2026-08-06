"""Suite C — causal control of the ALIGNMENT factor G_k (the second knob).

The ticket variable factorizes: T_k = E_k (W_E energy) x G_k (alignment of
frequency k's W_E subspace with the attention OV pathway). The energy knob
is causally proven (dose farm). This suite turns the OTHER knob: rotate
frequency 7's W_E 2-plane toward the OV map's high-gain input directions
AT EXACTLY FIXED ENERGY, choosing rotation angles so the transmitted
energy T_7 is multiplied by the same factors as the energy doses
{1.05, 1.20, 2.25}. Base: og_seed0/seed27058 (same as the energy curve).

PRE-REGISTERED PREDICTIONS:
  P-C1 (primary): outcomes match the energy-dose curve AT MATCHED T_7 GAIN
        — gain 1.05: not adopted; gain 1.20: adopted; gain 2.25: dominant
        or strongly adopted. I.e. T_k, not E_k, is the causal variable.
  P-C2: W_E per-frequency ENERGY is unchanged by the surgery (verified
        numerically below) — so any effect cannot be an energy effect.
  P-C3: menu closure and bystander chaos as usual.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
import mlx.core as mx
from grok.config import Config
from grok.model import Transformer
from grok.fourier import Fourier
from grok.train import train
from mask_lottery import committee_from_coeffs

SCRATCH = Path(__file__).parent
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

B0 = F[:, [2 * TARGET - 1, 2 * TARGET]].copy()      # (d, 2) freq-7 block

def T7_clean(B):
    return sum(((OV @ B) ** 2).sum() for OV in OVs)

# high-gain input directions of the summed transmission operator
M2 = sum(OV.T @ OV for OV in OVs)                    # symmetric PSD (d, d)
w_eig, V = np.linalg.eigh(M2)
top = V[:, np.argsort(w_eig)[::-1][:2]]              # (d, 2) top directions

def rotated(alpha):
    """Mix the block toward the top-gain plane, preserving each column's
    norm exactly (so per-frequency W_E energy is untouched)."""
    B = (1 - alpha) * B0 + alpha * top * np.linalg.norm(B0, axis=0)
    B *= np.linalg.norm(B0, axis=0) / np.linalg.norm(B, axis=0)
    return B

t0 = T7_clean(B0)
def gain(alpha):
    return T7_clean(rotated(alpha)) / t0

print(f"base T_7 {t0:.4f}; max achievable gain {gain(1.0):.2f}", flush=True)
print(__doc__, flush=True)

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
    W = W_E.copy()
    W[:, :P] = Fm @ fourier.basis
    # P-C2 verification: energies untouched
    E_before = (F ** 2).sum(0)[1::2][:nf] + (F ** 2).sum(0)[2::2][:nf]
    E_after = (Fm ** 2).sum(0)[1::2][:nf] + (Fm ** 2).sum(0)[2::2][:nf]
    err = np.abs(E_after - E_before).max() / E_before.mean()
    name = f"gain_{int(round(g_target * 100)):03d}"
    print(f"\n=== {name}: alpha {alpha:.4f}, achieved T_7 gain "
          f"{gain(alpha):.4f}, max energy err {err:.1e} ===", flush=True)
    run_dir = ROOT / "runs" / "gkrotate" / name
    if (run_dir / "spectra.npz").exists():
        print(f"skip {name} (exists)", flush=True)
    else:
        model.load_weights(str(E0))
        model.embed["W_E"] = mx.array(W.astype(np.float32))
        ck = SCRATCH / f"gkrotate_{name}.safetensors"
        model.save_weights(str(ck))
        c2 = Config.load(BASE / "config.json")
        c2.num_epochs = EPOCHS
        c2.save_every = 4000
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=4000)
    z = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(z["coeffs"][-1])
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    ge = int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1
    tr = np.abs(z["coeffs"][:, TARGET - 1])
    print(f"RESULT {name}: grok@{ge} committee {final} f{TARGET} peak "
          f"{tr.max():.0f} final {tr[-1]:.0f} acc {z['test_acc'][-1]:.4f}",
          flush=True)
print("\nGK ROTATE DONE", flush=True)
