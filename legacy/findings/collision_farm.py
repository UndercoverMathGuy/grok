"""Suite D — engineered repair events across masks (claim 4's n=2 problem).

The 'repair strips additive degeneracy' claim has exactly two on-demand
demonstrations, both on base seed27058 (T26 collision, T28 boost). This
farm manufactures four more repair events on four bases spanning three
masks: boost t = fold(i+j) of the two strongest committee members by x2.25
energy, inviting an additive trio {i, j, t}.

Bases: runs/seed1/seed21245 (ds1), runs/seed2/seed51376 (ds2),
runs/p-113/seed2034/seed33428 (ds2034), runs/og_seed0/seed63523 (ds0).

PRE-REGISTERED PREDICTIONS (from T26/T28 and the composition census):
  P-D1 (primary): no final committee contains the full additive trio
        {i, j, t} — the degenerate triple is always broken.
  P-D2: t is adopted (x2.25 was dominant-adoption dose in both prior arms).
  P-D3: the break is a LOCAL single-member repair: exactly one of {i, j}
        evicted (plus possibly unrelated churn), recruits from the menu.
  P-D4: final committees are additive-clean (no sum/diff pairs).
Each evicted-member identity is logged for the swap-choice census.
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
P = 113
nf = P // 2
fourier = Fourier(P)
EPOCHS = 10000
BASES = ["seed1/seed21245", "seed2/seed51376", "p-113/seed2034/seed33428",
         "og_seed0/seed63523"]

def fold(x):
    x %= P
    return min(x, P - x)

def viol_pairs(S):
    S = sorted(S); ss = set(S); out = []
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            i, j = S[a], S[b]
            if fold(i + j) in ss or fold(i - j) in ss:
                out.append((i, j))
    return out

print(__doc__, flush=True)
for base_rel in BASES:
    d = ROOT / "runs" / base_rel
    cfg = Config.load(d / "config.json")
    z = np.load(d / "spectra.npz")
    coeffs = z["coeffs"][-1]
    comm = committee_from_coeffs(coeffs)
    # two strongest committee members whose fold-sum is a valid outsider
    order = sorted(comm, key=lambda k: -abs(coeffs[k - 1]))
    t = None
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            for cand in (fold(i + j), fold(i - j)):
                if cand not in (0, i, j) and cand not in comm:
                    t, pi, pj = cand, i, j
                    break
            if t:
                break
        if t:
            break
    if t is None:
        print(f"{base_rel}: no valid collision target — skipped", flush=True)
        continue
    tag = d.name
    print(f"\n##### base {base_rel}: committee {comm}, boost t={t} = "
          f"fold({pi}(+/-){pj}) x2.25 #####", flush=True)
    run_dir = ROOT / "runs" / "collisionfarm" / f"{tag}_t{t}"
    if (run_dir / "spectra.npz").exists():
        print(f"skip {tag} (exists)", flush=True)
    else:
        model = Transformer(cfg)
        model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
        W_E = np.array(model.W_E, dtype=np.float64)
        F = W_E[:, :P] @ fourier.basis.T
        s = np.sqrt(2.25)
        F[:, 2 * t - 1] *= s
        F[:, 2 * t] *= s
        W = W_E.copy()
        W[:, :P] = F @ fourier.basis
        model.embed["W_E"] = mx.array(W.astype(np.float32))
        ck = SCRATCH / f"collision_{tag}_t{t}.safetensors"
        model.save_weights(str(ck))
        c2 = Config.load(d / "config.json")
        c2.num_epochs = EPOCHS
        c2.save_every = 4000
        train(c2, run_dir, init_from=ck, spectra_every=100, log_every=4000)
    zz = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(zz["coeffs"][-1])
    gi = int(np.argmax(zz["test_acc"] >= 0.99))
    ge = int(zz["epochs"][gi]) if zz["test_acc"][gi] >= 0.99 else -1
    trio_in = [k for k in (pi, pj, t) if k in final]
    evicted = sorted(set(comm) - set(final))
    print(f"RESULT {tag}: grok@{ge} committee {final} acc "
          f"{zz['test_acc'][-1]:.4f}", flush=True)
    print(f"  trio {{{pi},{pj},{t}}} members in final: {trio_in} "
          f"(P-D1 broken trio: {len(trio_in) < 3})", flush=True)
    print(f"  evicted from original committee: {evicted}; final additive "
          f"violations: {viol_pairs(final)}", flush=True)
print("\nCOLLISION FARM DONE", flush=True)
