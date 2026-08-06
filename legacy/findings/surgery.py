"""Init surgery on og_seed0/seed27058 (committee {14,49,52}, grok ~3500).
Arms (in order): control, collision, boost-strong, suppress, boost-subtle.
Surgery = scale chosen frequencies' Fourier components of W_E ONLY at epoch 0.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx
from grok.config import Config
from grok.model import Transformer
from grok.fourier import Fourier
from grok.train import train
from mask_lottery import committee_from_coeffs

BASE = ROOT / "runs/og_seed0/seed27058"
SCRATCH = Path(__file__).parent
EPOCHS = 12000

cfg = Config.load(BASE / "config.json")
p = cfg.p; nf = p // 2
fourier = Fourier(p)
model = Transformer(cfg)
E0 = BASE / "checkpoints" / "epoch_00000.safetensors"
model.load_weights(str(E0))
W_E = np.array(model.W_E, dtype=np.float64)          # (d, p+1)
F = W_E[:, :p] @ fourier.basis.T                     # coeffs (d, p)
energy = (F ** 2).sum(0)[1::2][:nf] + (F ** 2).sum(0)[2::2][:nf]
z = (energy - energy.mean()) / energy.std()

comm = [14, 49, 52]
menu = [49, 52, 29, 14, 5, 26, 53, 37]               # e3000 top-8
# suppress: committee member with highest init z
k_s = comm[int(np.argmax([z[k - 1] for k in comm]))]
# collision: fold(49+52)=12 -> {12,49,52} carries the additive relation
k_c = 12
# boost: mid-energy freq, not in menu, not harmonically tied to committee
def fold(x): x %= p; return min(x, p - x)
excl = set(menu) | {k_c}
for k in comm:
    excl |= {fold(2 * k), fold(3 * k)}
    for j in comm:
        if j != k: excl |= {fold(k + j), fold(k - j)}
cands = [k for k in range(1, nf + 1) if k not in excl]
k_b = cands[int(np.argmin([abs(z[k - 1]) for k in cands]))]

print(f"targets: suppress k_s={k_s} (z={z[k_s-1]:+.2f}), boost k_b={k_b} "
      f"(z={z[k_b-1]:+.2f}), collision k_c={k_c} (z={z[k_c-1]:+.2f})", flush=True)
sys.path.insert(0, str(ROOT / "scripts"))
from margin_analysis import relM_equal  # noqa: E402
print(f"relM {{14,49,52}} = {relM_equal(comm, p):.3f}; "
      f"with k_c: {relM_equal(sorted(comm + [k_c]), p):.3f}; "
      f"with k_b: {relM_equal(sorted(comm + [k_b]), p):.3f}", flush=True)
print("""PREDICTIONS (pre-registered):
  control      -> {14,49,52}, grok ~3500 (pipeline determinism)
  collision    (k_c x2.25 energy) -> k_c leads audition but is NOT in the
                final committee (floor veto), or adopted-then-evicted
  boost-strong (k_b x2.25) -> k_b joins the final committee
  suppress     (k_s x0.5)  -> k_s drops out, replaced from menu
  boost-subtle (k_b x1.2)  -> k_b audition rank improves; adoption uncertain
""", flush=True)

def surgical_ckpt(scales: dict, name: str) -> Path:
    Fs = F.copy()
    for k, s_energy in scales.items():
        amp = np.sqrt(s_energy)
        Fs[:, 2 * k - 1] *= amp
        Fs[:, 2 * k] *= amp
    W = W_E.copy()
    W[:, :p] = Fs @ fourier.basis
    model.load_weights(str(E0))
    model.embed["W_E"] = mx.array(W.astype(np.float32))
    out = SCRATCH / f"surg_{name}.safetensors"
    model.save_weights(str(out))
    return out

ARMS = [
    ("control", {}),
    ("collision", {k_c: 2.25}),
    ("boost_strong", {k_b: 2.25}),
    ("suppress", {k_s: 0.5}),
    ("boost_subtle", {k_b: 1.2}),
]
for name, scales in ARMS:
    run_dir = ROOT / "runs" / "surgery" / name
    ck = surgical_ckpt(scales, name)
    c2 = Config.load(BASE / "config.json")
    c2.num_epochs = EPOCHS
    c2.save_every = 2000
    print(f"\n=== arm {name} (scales {scales}) ===", flush=True)
    train(c2, run_dir, init_from=ck, spectra_every=100, log_every=2000)
    zz = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(zz["coeffs"][-1])
    gi = np.argmax(zz["test_acc"] >= 0.99)
    ge = zz["epochs"][gi] if zz["test_acc"][gi] >= 0.99 else -1
    print(f"ARM {name}: grok@{ge}  committee {final}  "
          f"test_acc {zz['test_acc'][-1]:.4f}", flush=True)
    for k, lab in [(k_c, "k_c"), (k_b, "k_b"), (k_s, "k_s")]:
        traj = np.abs(zz["coeffs"][:, k - 1])
        pk = int(zz["epochs"][int(np.argmax(traj))])
        print(f"  {lab}={k}: peak |coeff| {traj.max():.0f} @ep{pk}, "
              f"final {traj[-1]:.0f}", flush=True)
