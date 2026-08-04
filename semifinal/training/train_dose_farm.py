"""Cross-mask dose-response farm (claim 3's causal breadth).

Replicates the energy dose-response on three bases with DIFFERENT data
masks, plus the missing x1.50 point on the original base seed27058.

Bases (all p=113, natural, spectra-logged, epoch-0 checkpoints):
  runs/seed1/seed21245           mask ds=1
  runs/seed2/seed51376           mask ds=2
  runs/p-113/seed2034/seed33428  mask ds=2034
Target per base: pre-registered rule — the most median-energy frequency
(|z| minimal) that is NOT in the e3000 top-8 menu, NOT in the committee,
and NOT harmonically (2k,3k) or additively (i+-j) related to any member.

Doses per base: x1.00 (control) 1.05 1.10 1.20 1.50 2.25 on the target's
W_E Fourier block at epoch 0. 12k epochs. Idempotent.

PRE-REGISTERED PREDICTIONS (from the seed27058 curve):
  P-B1: per base, final |coeff| of the target is monotone in dose.
  P-B2: adoption threshold in (1.05, 1.20] on every mask.
        [Outcome: FAILED as universal — thresholds are context-dependent
        (2.25 on two bases, never on one); kept as-is, see SEMIFINAL.md.]
  P-B3: the x1.00 control reproduces the base's natural committee.
  P-B4: menu closure; bystander recomposition allowed (chaos).
Primary outcome: final target |coeff| (detector-free);
adoption = final |coeff| > 10x background median.
"""
import numpy as np

from _shared import (ROOT, committee_from_coeffs, fold, freq_energy,
                     grok_epoch, save_surgical_ckpt, scale_freq_energy)

from grok.config import Config                          # noqa: E402
from grok.model import Transformer                      # noqa: E402
from grok.fourier import Fourier                        # noqa: E402
from grok.train import train                            # noqa: E402

P = 113
nf = P // 2
fourier = Fourier(P)
EPOCHS = 12000
BASES = ["seed1/seed21245", "seed2/seed51376", "p-113/seed2034/seed33428"]
DOSES = [1.00, 1.05, 1.10, 1.20, 1.50, 2.25]


def pick_target(comm, menu, z):
    excl = set(menu) | set(comm)
    for k in comm:
        excl |= {fold(2 * k, P), fold(3 * k, P)}
        for j in comm:
            if j != k:
                excl |= {fold(k + j, P), fold(k - j, P)}
    cands = [k for k in range(1, nf + 1) if k not in excl]
    return cands[int(np.argmin([abs(z[k - 1]) for k in cands]))]


def run_arm(dose, model, e0_path, W_E, F, target, cfg_path, tag):
    name = f"dose_{int(round(dose * 100)):03d}"
    run_dir = ROOT / "runs" / "dosefarm" / tag / name
    if (run_dir / "spectra.npz").exists():
        print(f"skip {tag}/{name} (exists)", flush=True)
    else:
        W = W_E.copy()
        W[:, :P] = scale_freq_energy(F, {target: dose}) @ fourier.basis
        model.load_weights(str(e0_path))
        ck = save_surgical_ckpt(model, W, f"dosefarm_{tag}_{name}")
        c2 = Config.load(cfg_path)
        c2.num_epochs = EPOCHS
        c2.save_every = 4000
        print(f"\n=== {tag} {name} (target {target} x{dose}) ===", flush=True)
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=4000)
    z = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(z["coeffs"][-1])
    tr = np.abs(z["coeffs"][:, target - 1])
    bg = float(np.median(np.abs(z["coeffs"][-1])))
    print(f"RESULT {tag} {name}: grok@{grok_epoch(z)} committee {final} "
          f"target f{target} peak {tr.max():.0f} final {tr[-1]:.0f} "
          f"(bg median {bg:.1f}) acc {z['test_acc'][-1]:.4f}", flush=True)


for base_rel in BASES:
    d = ROOT / "runs" / base_rel
    cfg = Config.load(d / "config.json")
    assert cfg.p == P
    z = np.load(d / "spectra.npz")
    comm = committee_from_coeffs(z["coeffs"][-1])
    i3k = int(np.argmin(np.abs(z["epochs"] - 3000)))
    menu = (np.argsort(np.abs(z["coeffs"][i3k]))[::-1][:8] + 1).tolist()
    model = Transformer(cfg)
    e0 = d / "checkpoints" / "epoch_00000.safetensors"
    model.load_weights(str(e0))
    W_E = np.array(model.W_E, dtype=np.float64)
    F = W_E[:, :P] @ fourier.basis.T
    energy = freq_energy(F, nf)
    zsc = (energy - energy.mean()) / energy.std()
    target = pick_target(comm, menu, zsc)
    print(f"\n##### base {base_rel}: committee {comm}, menu {menu}, "
          f"target {target} (z {zsc[target-1]:+.2f}) #####", flush=True)
    for dose in DOSES:
        run_arm(dose, model, e0, W_E, F, target, d / "config.json", d.name)

# missing x1.50 point on the original curve (base seed27058, target 7)
d = ROOT / "runs" / "og_seed0" / "seed27058"
cfg = Config.load(d / "config.json")
model = Transformer(cfg)
e0 = d / "checkpoints" / "epoch_00000.safetensors"
model.load_weights(str(e0))
W_E = np.array(model.W_E, dtype=np.float64)
F = W_E[:, :P] @ fourier.basis.T
print("\n##### base og_seed0/seed27058 gap-fill: target 7 x1.50 #####",
      flush=True)
run_arm(1.50, model, e0, W_E, F, 7, d / "config.json", "seed27058")
print("\nDOSE FARM DONE", flush=True)
