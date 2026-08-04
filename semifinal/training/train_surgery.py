"""Original five-arm init surgery on base og_seed0/seed27058
(natural committee {14,49,52}, grok ~3500).

Surgery = scale chosen frequencies' Fourier components of W_E ONLY, at
epoch 0; everything else (attention, MLP, unembed, data, optimizer stream)
is untouched.

Arms: control, collision (fold(49+52)=12 x2.25), boost_strong (x2.25),
suppress (strongest member x0.5), boost_subtle (x1.2). Targets are chosen
by pre-registered rules printed before training. Runs land in
runs/surgery/<arm>. 12k epochs. Idempotent.

PRE-REGISTERED PREDICTIONS:
  control      -> {14,49,52}, grok ~3500 (pipeline determinism)
  collision    -> k_c leads audition but is NOT in the final committee
                  (floor veto), or adopted-then-evicted
  boost_strong -> k_b joins the final committee
  suppress     -> k_s drops out, replaced from the menu
  boost_subtle -> k_b audition rank improves; adoption uncertain
"""
import numpy as np

from _shared import (ROOT, committee_from_coeffs, fold, freq_energy, report,
                     save_surgical_ckpt, scale_freq_energy)

from grok.config import Config                          # noqa: E402
from grok.model import Transformer                      # noqa: E402
from grok.fourier import Fourier                        # noqa: E402
from grok.train import train                            # noqa: E402

BASE = ROOT / "runs" / "og_seed0" / "seed27058"
EPOCHS = 12000

cfg = Config.load(BASE / "config.json")
p, nf = cfg.p, cfg.p // 2
fourier = Fourier(p)
model = Transformer(cfg)
E0 = BASE / "checkpoints" / "epoch_00000.safetensors"
model.load_weights(str(E0))
W_E = np.array(model.W_E, dtype=np.float64)             # (d, p+1)
F = W_E[:, :p] @ fourier.basis.T                        # (d, p) coeffs
energy = freq_energy(F, nf)
z = (energy - energy.mean()) / energy.std()

comm = [14, 49, 52]
menu = [49, 52, 29, 14, 5, 26, 53, 37]                  # e3000 top-8
k_s = comm[int(np.argmax([z[k - 1] for k in comm]))]    # suppress target
k_c = fold(49 + 52, p)                                  # collision: 12
excl = set(menu) | {k_c}
for k in comm:
    excl |= {fold(2 * k, p), fold(3 * k, p)}
    for j in comm:
        if j != k:
            excl |= {fold(k + j, p), fold(k - j, p)}
cands = [k for k in range(1, nf + 1) if k not in excl]
k_b = cands[int(np.argmin([abs(z[k - 1]) for k in cands]))]  # boost target

print(__doc__, flush=True)
print(f"targets: suppress k_s={k_s} (z={z[k_s-1]:+.2f}), boost k_b={k_b} "
      f"(z={z[k_b-1]:+.2f}), collision k_c={k_c} (z={z[k_c-1]:+.2f})",
      flush=True)

ARMS = [
    ("control", {}),
    ("collision", {k_c: 2.25}),
    ("boost_strong", {k_b: 2.25}),
    ("suppress", {k_s: 0.5}),
    ("boost_subtle", {k_b: 1.2}),
]
for name, scales in ARMS:
    run_dir = ROOT / "runs" / "surgery" / name
    if (run_dir / "spectra.npz").exists():
        print(f"skip {name} (exists)", flush=True)
    else:
        W = W_E.copy()
        W[:, :p] = scale_freq_energy(F, scales) @ fourier.basis
        model.load_weights(str(E0))
        ck = save_surgical_ckpt(model, W, f"surg_{name}")
        c2 = Config.load(BASE / "config.json")
        c2.num_epochs = EPOCHS
        c2.save_every = 2000
        print(f"\n=== arm {name} (energy scales {scales}) ===", flush=True)
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=2000)
    report(run_dir, watch={k_c, k_b, k_s})
print("\nSURGERY DONE", flush=True)
