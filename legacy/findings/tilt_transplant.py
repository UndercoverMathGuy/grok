"""Tilt transplant: causal carrier test for the W_E init energy spectrum,
plus the two cancelled C5 dose arms (x1.05, x1.10 on freq 7, seed27058).

Design (transplant): the three same-mask runs runs/seed0/seed{37099,55327,
93077} (p=113, data_seed 0, 30k-epoch naturals with epoch-0 checkpoints).
For each ordered pair (R <- D), build a surgical init from R's epoch-0
weights in which every frequency's (d x 2) W_E Fourier block is rescaled to
DONOR D's per-frequency energy — R keeps its own directions, DC, '=' column,
positional embedding, attention, MLP, unembedding, and optimizer stream.
Train 12k epochs (committees settled by 8k in every prior surgery arm).

Why this design: the orthWE test (flatten tilt -> committees change) is a
weak necessity proof because committee identity is chaotic to sub-threshold
init perturbations (T45) — ANY edit moves the committee somewhere. The
transplant is the discriminating version: the outcome has two named
attractors (D's committee vs R's committee), both same-mask so mask-level
popularity is shared, and the question is which twin the hybrid resembles.

PRE-REGISTERED PREDICTIONS (printed before training):
  P-TR1 (primary, carrier): pooled over the 6 transplants, the final
        committee contains MORE members unique to D's natural committee
        than members unique to R's natural committee (shared members
        excluded from the count on both sides).
  P-TR2 (rank rule): every adopted frequency sits in the post-transplant
        init-energy top-8 of the surgical W_E (menu closure at init rank).
  P-TR3 (chaos guard): exact reproduction of D's committee is NOT expected
        (the non-W_E init structure differs); prediction is distributional
        pull, not identity.
  P-D1  (dose, pre-registered in c2c5_surgery.py): x1.05 on freq 7 -> NOT
        adopted (post-boost emb rank > 8); x1.10 -> boundary (rank 9),
        rule predicts NOT adopted; known calibration x1.2 -> adopted.
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

SCRATCH = Path(__file__).parent
EPOCHS = 12000
RUNS = {name: ROOT / "runs" / "seed0" / name
        for name in ("seed37099", "seed55327", "seed93077")}

fourier = Fourier(113)
nf = 113 // 2

def freq_energy(F):
    E = (F ** 2).sum(0)
    return E[1::2][:nf] + E[2::2][:nf]

# --- gather epoch-0 embeddings, energies, and final committees -------------
info = {}
for name, d in RUNS.items():
    cfg = Config.load(d / "config.json")
    assert cfg.p == 113 and cfg.data_seed == 0
    model = Transformer(cfg)
    model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
    W_E = np.array(model.W_E, dtype=np.float64)
    F = W_E[:, :113] @ fourier.basis.T
    z = np.load(d / "spectra.npz")
    comm = committee_from_coeffs(z["coeffs"][-1])
    info[name] = dict(cfg=cfg, W_E=W_E, F=F, E=freq_energy(F), comm=set(comm))

print("=" * 72)
print("PRE-REGISTERED SETUP (printed before any training)")
print("=" * 72)
for name, i in info.items():
    print(f"{name}: natural committee {sorted(i['comm'])}")
pairs = [(r, dn) for r in RUNS for dn in RUNS if r != dn]
for r, dn in pairs:
    uR = sorted(info[r]["comm"] - info[dn]["comm"])
    uD = sorted(info[dn]["comm"] - info[r]["comm"])
    print(f"  {r} <- tilt({dn}):  R-unique {uR}  D-unique {uD}")
print(__doc__[__doc__.index("PRE-REGISTERED"):])

model = Transformer(info["seed37099"]["cfg"])  # reused shell for surgery

def transplant_ckpt(r, dn):
    R, D = info[r], info[dn]
    Fm = R["F"].copy()
    for k in range(nf):
        s = np.sqrt(D["E"][k] / R["E"][k])
        Fm[:, 2 * k + 1] *= s
        Fm[:, 2 * k + 2] *= s
    W = R["W_E"].copy()
    W[:, :113] = Fm @ fourier.basis
    model.load_weights(str(RUNS[r] / "checkpoints" / "epoch_00000.safetensors"))
    model.embed["W_E"] = mx.array(W.astype(np.float32))
    out = SCRATCH / f"transplant_{r}_from_{dn}.safetensors"
    model.save_weights(str(out))
    # sanity: post-surgery energies == donor energies
    F2 = np.array(model.W_E, dtype=np.float64)[:, :113] @ fourier.basis.T
    err = np.abs(freq_energy(F2) - D["E"]).max() / D["E"].mean()
    print(f"  surgical ckpt {out.name}: max energy err {err:.1e}", flush=True)
    return out

def report(run_dir, watch):
    z = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(z["coeffs"][-1])
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    ge = int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1
    print(f"RESULT {run_dir.name}: grok@{ge}  committee {final}  "
          f"acc {z['test_acc'][-1]:.4f}", flush=True)
    for kk in sorted(watch):
        traj = np.abs(z["coeffs"][:, kk - 1])
        print(f"    f{kk}: peak {traj.max():.0f} final {traj[-1]:.0f}", flush=True)
    return set(final)

# --- transplant arms --------------------------------------------------------
tally_D, tally_R = 0, 0
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
    nD, nR = len(final & uD), len(final & uR)
    tally_D += nD; tally_R += nR
    print(f"    adopted D-unique {sorted(final & uD)} ({nD})  "
          f"kept R-unique {sorted(final & uR)} ({nR})", flush=True)

print(f"\nP-TR1 tally: D-unique members adopted {tally_D} vs "
      f"R-unique members kept {tally_R} (prediction: D > R)", flush=True)

# --- C5 dose arms (base seed27058, freq 7) ---------------------------------
BASE = ROOT / "runs" / "og_seed0" / "seed27058"
bcfg = Config.load(BASE / "config.json")
bmodel = Transformer(bcfg)
E0 = BASE / "checkpoints" / "epoch_00000.safetensors"
bmodel.load_weights(str(E0))
W_Eb = np.array(bmodel.W_E, dtype=np.float64)
Fb = W_Eb[:, :113] @ fourier.basis.T
Eb = freq_energy(Fb)
for beta, name in ((1.05, "dose_105"), (1.10, "dose_110")):
    run_dir = ROOT / "runs" / "surgery2" / name
    if (run_dir / "spectra.npz").exists():
        print(f"skip {name} (exists)", flush=True)
    else:
        Fm = Fb.copy()
        Fm[:, 2 * 7 - 1] *= np.sqrt(beta)
        Fm[:, 2 * 7] *= np.sqrt(beta)
        W = W_Eb.copy()
        W[:, :113] = Fm @ fourier.basis
        bmodel.load_weights(str(E0))
        bmodel.embed["W_E"] = mx.array(W.astype(np.float32))
        ck = SCRATCH / f"surg2_{name}.safetensors"
        bmodel.save_weights(str(ck))
        e2 = Eb.copy(); e2[7 - 1] *= beta
        rank = int((e2 > e2[7 - 1]).sum()) + 1
        print(f"\n=== dose arm {name} (freq 7 x{beta}, post-boost emb rank "
              f"{rank}) ===", flush=True)
        c2 = Config.load(BASE / "config.json")
        c2.num_epochs = 10000
        c2.save_every = 2000
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=2000)
    report(run_dir, {7, 14, 49, 52})

print("\nALL ARMS DONE", flush=True)
