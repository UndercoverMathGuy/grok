"""Shared helpers for the SEMIFINAL training scripts.

These scripts TRAIN runs into runs/ (unlike ../analysis/*, which only read).
Everything here is self-contained — no imports from findings/ or scripts/.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CKPT_DIR = Path(__file__).resolve().parent / "_ckpt"   # surgical init ckpts


def fold(x, p):
    x %= p
    return min(x, p - x)


def freq_energy(F, nf):
    """Per-frequency energy from a Fourier coefficient matrix F (d, p)."""
    E = (F ** 2).sum(0)
    return E[1::2][:nf] + E[2::2][:nf]


def committee_from_coeffs(coeffs, floor=0.02):
    """Unified committee detector (same as analysis/common.py): largest
    log-gap, then drop stragglers below floor x the run's max |coeff|."""
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    gaps = logs[:-1] - logs[1:]
    cut = int(np.argmax(gaps[:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())


def grok_epoch(z):
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    return int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1


def report(run_dir, watch=()):
    """Print the standard post-run summary; return the final committee set."""
    z = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(z["coeffs"][-1])
    print(f"RESULT {run_dir.name}: grok@{grok_epoch(z)}  committee {final}  "
          f"acc {z['test_acc'][-1]:.4f}", flush=True)
    for k in sorted(watch):
        traj = np.abs(z["coeffs"][:, k - 1])
        pk = int(z["epochs"][int(np.argmax(traj))])
        print(f"    f{k}: peak {traj.max():.0f} @ep{pk}  final {traj[-1]:.0f}",
              flush=True)
    return set(final)


def save_surgical_ckpt(model, W_E_new, name):
    """Swap W_E_new into model and save an init checkpoint under _ckpt/."""
    import mlx.core as mx
    CKPT_DIR.mkdir(exist_ok=True)
    model.embed["W_E"] = mx.array(W_E_new.astype(np.float32))
    out = CKPT_DIR / f"{name}.safetensors"
    model.save_weights(str(out))
    return out


def scale_freq_energy(F, scales):
    """Return a copy of Fourier coeffs F with freq k's ENERGY scaled by s
    for each k: s in scales.items() (amplitude scales by sqrt(s))."""
    Fs = F.copy()
    for k, s in scales.items():
        amp = np.sqrt(s)
        Fs[:, 2 * k - 1] *= amp
        Fs[:, 2 * k] *= amp
    return Fs
