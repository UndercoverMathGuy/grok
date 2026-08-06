"""C2 (triple-implant) + C5 (dose-response) surgery on og_seed0/seed27058.

C2: two arms, both boost menu outsiders 5 and 37 by x1.5 energy; the third
boosted freq differs:
  triple_deg   adds 42 = fold(5+37)  (additive-degenerate trio)
  triple_clean adds x, arithmetic-clean w.r.t. {5,37,14,49,52} and
               init-energy-matched to 42.
Pre-registered predictions printed below BEFORE training.

C5: two arms boosting freq 7 (the T28/T31 target) at x1.05 and x1.10 energy,
bracketing the known x1.2 adoption. Adoption rule pre-registered from
post-boost init rank (emb and OV variants).

All arms: 10k epochs (committees provably settled by 8k in every prior arm),
spectra every 100.
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

BASE = ROOT / "runs/og_seed0/seed27058"
SCRATCH = Path(__file__).parent
EPOCHS = 10000

cfg = Config.load(BASE / "config.json")
p = cfg.p; nf = p // 2
fourier = Fourier(p)
model = Transformer(cfg)
E0 = BASE / "checkpoints" / "epoch_00000.safetensors"
model.load_weights(str(E0))
W_E = np.array(model.W_E, dtype=np.float64)          # (d, p+1)
F = W_E[:, :p] @ fourier.basis.T                     # (d, p) fourier coeffs

def freq_energy_of(Fmat):
    E = (Fmat ** 2).sum(0)
    return E[1::2][:nf] + E[2::2][:nf]

energy = freq_energy_of(F)
z = (energy - energy.mean()) / energy.std()

# OV energies at init (for the C5 OV-rank rule)
at = model.blocks[0].attn
W_V = np.array(at.W_V, dtype=np.float64)
W_O = np.array(at.W_O, dtype=np.float64)
h, dh, dm = W_V.shape
ov_energy = np.zeros(nf)
for i in range(h):
    OV = W_O[:, i*dh:(i+1)*dh] @ W_V[i]
    ov_energy += freq_energy_of((OV @ W_E[:, :p]) @ fourier.basis.T)

def fold(x):
    x = x % p
    return min(x, p - x)

COMM = {14, 49, 52}
MENU = [49, 52, 29, 14, 5, 26, 53, 37]
j, k = 5, 37
m_deg = fold(j + k)      # 42

# clean third member: no sum/diff/harmonic relation inside {5,37,x} u COMM,
# not in menu, energy-matched to m_deg
def is_clean(x):
    S = [j, k, x] + sorted(COMM)
    if len(set(S)) < 6:
        return False
    for a_ in S:
        for b_ in S:
            if a_ < b_:
                if fold(a_ + b_) in S or fold(a_ - b_) in S:
                    return False
        if fold(2 * a_) in S and fold(2 * a_) != a_:
            return False
    return True

cands = [x for x in range(1, nf + 1)
         if x not in MENU and x not in (m_deg,) and is_clean(x)]
m_clean = cands[int(np.argmin([abs(energy[x-1] - energy[m_deg-1]) for x in cands]))]

print("=" * 70)
print("PRE-REGISTERED SETUP AND PREDICTIONS (printed before any training)")
print("=" * 70)
print(f"base committee {sorted(COMM)}, menu {MENU}")
print(f"C2 boosted pair: {j},{k} (menu outsiders) x1.5 energy each")
print(f"  degenerate third: {m_deg} = fold({j}+{k}), init z {z[m_deg-1]:+.2f}, "
      f"energy {energy[m_deg-1]:.3f}")
print(f"  clean third:      {m_clean}, init z {z[m_clean-1]:+.2f}, "
      f"energy {energy[m_clean-1]:.3f}  (energy-matched, arithmetic-clean)")
print(f"  sanity: fold({j}+{k})={fold(j+k)}, fold({k}-{j})={fold(k-j)}; "
      f"clean-third relations checked: {is_clean(m_clean)}")
print("""C2 predictions:
  P-C2a (primary): #survivors of boosted trio in final committee is
        STRICTLY GREATER in triple_clean than in triple_deg.
  P-C2b: triple_deg does NOT end with all of {5,37,42} in the committee.
  P-C2c: if exactly one trio member is evicted in triple_deg, it is the one
        with the lowest amplitude at memorization end (e150-e300 window).
  P-C2d: all final-committee members come from MENU u boosted set (closure).
""")

k7 = 7
r_emb_now = int((energy > energy[k7-1]).sum()) + 1
r_ov_now = int((ov_energy > ov_energy[k7-1]).sum()) + 1
print(f"C5 target freq {k7}: init emb rank {r_emb_now}, OV rank {r_ov_now} "
      f"(of {nf}); z {z[k7-1]:+.2f}")
for beta in (1.05, 1.10, 1.20):
    e2 = energy.copy(); e2[k7-1] *= beta
    o2 = ov_energy.copy(); o2[k7-1] *= beta   # OV energy scales ~linearly in emb energy boost
    re_ = int((e2 > e2[k7-1]).sum()) + 1
    ro_ = int((o2 > o2[k7-1]).sum()) + 1
    print(f"  x{beta}: post-boost emb rank {re_}, OV rank {ro_} -> "
          f"emb-rule predicts {'ADOPT' if re_ <= 8 else 'NO'}, "
          f"OV-rule predicts {'ADOPT' if ro_ <= 8 else 'NO'}")
print("""C5 pre-registered rule: adoption iff post-boost init rank <= 8
  (menu size), evaluated separately for emb-energy and OV-energy ranks.
  Known calibration: x1.0 -> not adopted, x1.2 -> adopted, x2.25 -> dominant.
""", flush=True)

def surgical_ckpt(scales: dict, name: str) -> Path:
    Fs = F.copy()
    for kk, s_energy in scales.items():
        amp = np.sqrt(s_energy)
        Fs[:, 2 * kk - 1] *= amp
        Fs[:, 2 * kk] *= amp
    W = W_E.copy()
    W[:, :p] = Fs @ fourier.basis
    model.load_weights(str(E0))
    model.embed["W_E"] = mx.array(W.astype(np.float32))
    out = SCRATCH / f"surg2_{name}.safetensors"
    model.save_weights(str(out))
    return out

ARMS = [
    ("triple_deg", {j: 1.5, k: 1.5, m_deg: 1.5}),
    ("triple_clean", {j: 1.5, k: 1.5, m_clean: 1.5}),
    ("dose_105", {k7: 1.05}),
    ("dose_110", {k7: 1.10}),
]
for name, scales in ARMS:
    run_dir = ROOT / "runs" / "surgery2" / name
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
    watch = sorted(set([j, k, m_deg, m_clean, k7]) | COMM)
    for kk in watch:
        traj = np.abs(zz["coeffs"][:, kk - 1])
        print(f"  f{kk}: peak {traj.max():.0f} "
              f"@ep{int(zz['epochs'][int(np.argmax(traj))])}, final {traj[-1]:.0f}",
              flush=True)
print("\nALL ARMS DONE", flush=True)
