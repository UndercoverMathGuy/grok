"""Double-flat init: orthogonal W_E + isometric attention heads.

The ticket variable T_k = sum_h ||W_O^h W_V^h W_E|_k||^2 factorizes as
E_k (W_E spectrum) x G_k (frame-to-OV alignment). The orth-flat cohort
freezes E_k and the lottery runs on G_k. This experiment freezes BOTH:
with embed_init='orthogonal' and attn_init='isometric', T_k is identically
flat by construction — verified numerically per seed below, before any
training.

PRE-REGISTERED PREDICTIONS (printed before training):
  P-DF1 (primary): every epoch-0 committee readout is at chance (~0.5) on
        these runs. Tested by analysis/claim1_readout.py (double-flat row).
  P-DF2: the runs still grok (acc 1.0) and form normal-looking committees —
        erasing the readable lottery hides the selector, not selection.
  P-DF3 (falsification arm): any readout significantly above 0.5 means T_k
        is NOT the complete readable ticket.

16 init seeds, cell (p=113, data_seed=2034), 20k epochs, spectra every 100.
Runs land in runs/doubleflat/p-113/seed2034/seed<iseed>. Idempotent.
"""
import numpy as np

from _shared import ROOT, committee_from_coeffs, freq_energy, grok_epoch

import mlx.core as mx                                   # noqa: E402
from grok.config import Config                          # noqa: E402
from grok.model import Transformer                      # noqa: E402
from grok.fourier import Fourier                        # noqa: E402
from grok.train import train                            # noqa: E402

P, DSEED = 113, 2034
SEEDS = [11285, 33428, 4242, 777, 1001, 1002, 1003, 1004,
         2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
EPOCHS = 20000
nf = P // 2
fourier = Fourier(P)


def T_k(model):
    W_E = np.array(model.W_E, dtype=np.float64)[:, :P]
    at = model.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)
    W_O = np.array(at.W_O, dtype=np.float64)
    h, dh, _ = W_V.shape
    t = np.zeros(nf)
    for i in range(h):
        OV = W_O[:, i * dh:(i + 1) * dh] @ W_V[i]
        t += freq_energy((OV @ W_E) @ fourier.basis.T, nf)
    return t


def make_cfg(iseed):
    return Config(p=P, data_seed=DSEED, init_seed=iseed,
                  embed_init="orthogonal", attn_init="isometric",
                  num_epochs=EPOCHS, save_every=2000)


print(__doc__, flush=True)

# ---- init-time flatness verification (before any training) ----------------
print("=== init flatness check (rel sd; Gaussian init ~0.09) ===", flush=True)
for iseed in SEEDS:
    mx.random.seed(iseed)
    m = Transformer(make_cfg(iseed))
    e = freq_energy(np.array(m.W_E, dtype=np.float64)[:, :P]
                    @ fourier.basis.T, nf)
    t = T_k(m)
    print(f"  seed {iseed}: W_E energy rel-sd {e.std()/e.mean():.2e}, "
          f"T_k rel-sd {t.std()/t.mean():.2e}", flush=True)

# ---- train ----------------------------------------------------------------
for iseed in SEEDS:
    run_dir = ROOT / "runs" / "doubleflat" / f"p-{P}" / f"seed{DSEED}" / f"seed{iseed}"
    if (run_dir / "spectra.npz").exists():
        print(f"skip seed{iseed} (exists)", flush=True)
        continue
    print(f"\n=== training doubleflat seed {iseed} ===", flush=True)
    train(make_cfg(iseed), run_dir, spectra_every=100, log_every=4000)

# ---- P-DF2 spot check (full readout battery: analysis/claim1_readout.py) --
print("\n=== P-DF2: grok status and committees ===", flush=True)
for iseed in SEEDS:
    run_dir = ROOT / "runs" / "doubleflat" / f"p-{P}" / f"seed{DSEED}" / f"seed{iseed}"
    z = np.load(run_dir / "spectra.npz")
    comm = committee_from_coeffs(z["coeffs"][-1])
    print(f"seed {iseed}: acc {float(z['test_acc'][-1]):.3f} "
          f"grok@{grok_epoch(z)} committee {comm}", flush=True)
