"""Was the collision-arm repair (evict 52, recruit 29) margin-rational?"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from margin_analysis import relM_equal, lp_relM, homeostat

run = ROOT / "runs/surgery/collision"
p = 113
z = np.load(run / "spectra.npz")

# menu: top-8 at mid-audition (epoch 2000; grok was ~3000)
i2000 = int(np.argmin(np.abs(z["epochs"] - 2000)))
menu = sorted((np.argsort(np.abs(z["coeffs"][i2000]))[::-1][:8] + 1).tolist())
print(f"mid-audition menu (e2000 top-8): {menu}")

chosen = [12, 14, 29, 49]
named = {
    "chosen {12,14,29,49}": chosen,
    "keep-all {12,14,49,52}": [12, 14, 49, 52],
    "swap-49 {12,14,29,52}": [12, 14, 29, 52],
    "original {14,49,52}": [14, 49, 52],
    "orig+12 {12,14,49,52,29}": [12, 14, 29, 49, 52],
}
print("\nmargins (equal-amp | LP-optimal):")
for name, S in named.items():
    print(f"  {name:<28} relM {relM_equal(S, p):.3f} | {lp_relM(S, p)[0]:.3f}")

# rank of chosen among ALL 4-subsets of the menu containing 12
subs = [list(s) for s in combinations(menu, 4) if 12 in s]
scored = sorted(((lp_relM(list(s), p)[0], sorted(s)) for s in subs), reverse=True)
rank = [s for _, s in scored].index(sorted(chosen)) + 1
print(f"\nchosen set rank by LP margin among {len(scored)} 4-subsets "
      f"of menu containing 12: #{rank}")
for m, s in scored[:5]:
    tag = " <== chosen" if s == sorted(chosen) else ""
    print(f"    {m:.3f}  {s}{tag}")

# actual allocation vs LP-optimal for the chosen set; homeostat
A_tot, relM_act, minGap = homeostat(run, p)
print(f"\nhomeostat: A_tot {A_tot:.1f}  relM(actual alloc) {relM_act:.3f}  "
      f"minGap {minGap:.2f} nats  (control run for comparison below)")
A2, r2, g2 = homeostat(ROOT / "runs/surgery/control", p)
print(f"control:   A_tot {A2:.1f}  relM {r2:.3f}  minGap {g2:.2f} nats")

lpm, alloc = lp_relM(chosen, p)
# actual member amplitudes off final logits
import mlx.core as mx
from grok.config import Config
from grok.model import Transformer
from grok.metrics import all_logits
from grok.data import make_dataset
cfg = Config.load(run / "config.json")
model = Transformer(cfg)
ck = sorted((run / "checkpoints").glob("epoch_*.safetensors"))[-1]
model.load_weights(str(ck))
tokens, _ = make_dataset(cfg)
logits = all_logits(model, tokens)
a, b = np.divmod(np.arange(p * p), p)
x = (a[:, None] + b[:, None] - np.arange(p)[None, :]) % p
Lx = np.zeros(p); np.add.at(Lx, x.ravel(), logits.ravel()); Lx /= p * p
ks = np.arange(1, p // 2 + 1)
amp = (2.0 / p) * (Lx[None, :] * np.cos(2 * np.pi * ks[:, None]
                                        * np.arange(p)[None, :] / p)).sum(1)
act = np.array([amp[k - 1] for k in chosen]); act /= act.sum()
print(f"\nallocation over {chosen}:")
print(f"  actual      {np.round(act, 3)}")
print(f"  LP-optimal  {np.round(alloc, 3)}")
frac = relM_act / lpm if lpm > 0 else float('nan')
print(f"  achieved margin = {100*frac:.1f}% of the set's LP optimum")
