"""Triple-implant (C2) + fine dose (C5) surgery on base og_seed0/seed27058.

C2: two arms, both boost menu outsiders 5 and 37 by x1.5 energy; the third
boosted freq differs:
  triple_deg    adds 42 = fold(5+37)  (additive-degenerate trio)
  triple_clean  adds an arithmetic-clean, init-energy-matched third.
C5: two arms boosting freq 7 at x1.05 and x1.10 energy, bracketing the
known x1.2 adoption threshold on this base.

Runs land in runs/surgery2/<arm>. 10k epochs. Idempotent.

PRE-REGISTERED PREDICTIONS:
  P-C2a (primary): #survivors of the boosted trio in the final committee is
        STRICTLY GREATER in triple_clean than in triple_deg.
  P-C2b: triple_deg does NOT end with all of {5,37,42} in the committee.
  P-C2d: all final members come from MENU u boosted set (closure).
  P-C5:  adoption iff post-boost init rank <= 8 (menu size); known
        calibration on this base: x1.0 no, x1.2 yes, x2.25 dominant.
"""
import numpy as np

from _shared import (ROOT, fold, freq_energy, report, save_surgical_ckpt,
                     scale_freq_energy)

from grok.config import Config                          # noqa: E402
from grok.model import Transformer                      # noqa: E402
from grok.fourier import Fourier                        # noqa: E402
from grok.train import train                            # noqa: E402

BASE = ROOT / "runs" / "og_seed0" / "seed27058"
EPOCHS = 10000

cfg = Config.load(BASE / "config.json")
p, nf = cfg.p, cfg.p // 2
fourier = Fourier(p)
model = Transformer(cfg)
E0 = BASE / "checkpoints" / "epoch_00000.safetensors"
model.load_weights(str(E0))
W_E = np.array(model.W_E, dtype=np.float64)
F = W_E[:, :p] @ fourier.basis.T
energy = freq_energy(F, nf)
z = (energy - energy.mean()) / energy.std()

COMM = {14, 49, 52}
MENU = [49, 52, 29, 14, 5, 26, 53, 37]
j, k = 5, 37
m_deg = fold(j + k, p)                                  # 42


def is_clean(x):
    """No sum/diff/harmonic relation inside {j,k,x} u COMM."""
    S = [j, k, x] + sorted(COMM)
    if len(set(S)) < 6:
        return False
    for a in S:
        for b in S:
            if a < b and (fold(a + b, p) in S or fold(a - b, p) in S):
                return False
        if fold(2 * a, p) in S and fold(2 * a, p) != a:
            return False
    return True


cands = [x for x in range(1, nf + 1)
         if x not in MENU and x != m_deg and is_clean(x)]
m_clean = cands[int(np.argmin([abs(energy[x - 1] - energy[m_deg - 1])
                               for x in cands]))]
k7 = 7

print(__doc__, flush=True)
print(f"boosted pair {j},{k}; degenerate third {m_deg} "
      f"(z {z[m_deg-1]:+.2f}), clean third {m_clean} (z {z[m_clean-1]:+.2f}, "
      f"energy-matched)", flush=True)
for beta in (1.05, 1.10, 1.20):
    e2 = energy.copy()
    e2[k7 - 1] *= beta
    rank = int((e2 > e2[k7 - 1]).sum()) + 1
    print(f"C5 x{beta}: post-boost emb rank {rank} -> rank-rule predicts "
          f"{'ADOPT' if rank <= 8 else 'NO'}", flush=True)

ARMS = [
    ("triple_deg", {j: 1.5, k: 1.5, m_deg: 1.5}),
    ("triple_clean", {j: 1.5, k: 1.5, m_clean: 1.5}),
    ("dose_105", {k7: 1.05}),
    ("dose_110", {k7: 1.10}),
]
watch = sorted({j, k, m_deg, m_clean, k7} | COMM)
for name, scales in ARMS:
    run_dir = ROOT / "runs" / "surgery2" / name
    if (run_dir / "spectra.npz").exists():
        print(f"skip {name} (exists)", flush=True)
    else:
        W = W_E.copy()
        W[:, :p] = scale_freq_energy(F, scales) @ fourier.basis
        model.load_weights(str(E0))
        ck = save_surgical_ckpt(model, W, f"surg2_{name}")
        c2 = Config.load(BASE / "config.json")
        c2.num_epochs = EPOCHS
        c2.save_every = 2000
        print(f"\n=== arm {name} (energy scales {scales}) ===", flush=True)
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=2000)
    report(run_dir, watch=watch)
print("\nC2/C5 SURGERY DONE", flush=True)
