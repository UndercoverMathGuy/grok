"""Margin ledger for the boost-strong arm (evict 14, recruit 29)."""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from margin_analysis import relM_equal, lp_relM, homeostat

run = ROOT / "runs/surgery/boost_strong"
p = 113
z = np.load(run / "spectra.npz")
i2000 = int(np.argmin(np.abs(z["epochs"] - 2000)))
menu = sorted((np.argsort(np.abs(z["coeffs"][i2000]))[::-1][:8] + 1).tolist())
print(f"mid-audition menu (e2000 top-8): {menu}")

chosen = [7, 29, 49, 52]
named = {
    "chosen {7,29,49,52}": chosen,
    "keep-all {7,14,49,52}": [7, 14, 49, 52],
    "evict-49 {7,14,29,52}": [7, 14, 29, 52],
    "evict-52 {7,14,29,49}": [7, 14, 29, 49],
    "original {14,49,52}": [14, 49, 52],
    "all-five {7,14,29,49,52}": [7, 14, 29, 49, 52],
}
print("\nmargins (equal-amp | LP-optimal):")
for name, S in named.items():
    print(f"  {name:<26} relM {relM_equal(S, p):.3f} | {lp_relM(S, p)[0]:.3f}")

subs = [list(s) for s in combinations(menu, 4) if 7 in s]
scored = sorted(((lp_relM(list(s), p)[0], sorted(s)) for s in subs), reverse=True)
rank = [s for _, s in scored].index(sorted(chosen)) + 1
print(f"\nchosen rank by LP margin among {len(scored)} menu 4-subsets "
      f"containing 7: #{rank}")
for m, s in scored[:5]:
    tag = " <== chosen" if s == sorted(chosen) else ""
    print(f"    {m:.3f}  {s}{tag}")

A, r, g = homeostat(run, p)
print(f"\nhomeostat: A_tot {A:.1f}  relM {r:.3f}  minGap {g:.2f} nats")

lpm, alloc = lp_relM(chosen, p)
import mlx.core as mx
from grok.config import Config
from grok.model import Transformer
from grok.metrics import all_logits
from grok.data import make_dataset
cfg = Config.load(run / "config.json")
model = Transformer(cfg)
model.load_weights(str(sorted((run / "checkpoints").glob("epoch_*.safetensors"))[-1]))
tokens, _ = make_dataset(cfg)
logits = all_logits(model, tokens)
a, b = np.divmod(np.arange(p * p), p)
x = (a[:, None] + b[:, None] - np.arange(p)[None, :]) % p
Lx = np.zeros(p); np.add.at(Lx, x.ravel(), logits.ravel()); Lx /= p * p
ks = np.arange(1, p // 2 + 1)
amp = (2.0 / p) * (Lx[None, :] * np.cos(2 * np.pi * ks[:, None]
                                        * np.arange(p)[None, :] / p)).sum(1)
act = np.array([amp[k - 1] for k in chosen]); act /= act.sum()
print(f"allocation over {chosen}: actual {np.round(act,3)}  "
      f"LP {np.round(alloc,3)}  -> {100*r/lpm:.1f}% of LP optimum")
