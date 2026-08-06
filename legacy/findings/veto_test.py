"""Veto-only assembly test: does a run reconfigure (final != blind
amplitude-topK draw at e3000) exactly when the blind draw is low-margin?
Plus: init W_E tilt statistics (the chi-square whisper numbers)."""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from grok.model import Transformer
from grok.fourier import Fourier
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs
from margin_analysis import relM_equal

rng = np.random.default_rng(0)
rows, tilt_sd, tilt_z = [], [], []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    K = len(final)
    i3000 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    blind = sorted((np.argsort(np.abs(z["coeffs"][i3000]))[::-1][:K] + 1).tolist())
    reconf = set(blind) != set(final)
    # percentile of the BLIND draw's equal-amp margin among random K-subsets
    obs = relM_equal(blind, p)
    null = np.array([relM_equal(rng.choice(np.arange(1, nf + 1), K,
                     replace=False).tolist(), p) for _ in range(3000)])
    pct = 100.0 * (null < obs).mean()
    rows.append((str(d.relative_to(ROOT/'runs')), reconf, pct,
                 relM_equal(final, p), obs))
    # init W_E tilt stats
    model = Transformer(cfg)
    model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
    fourier = Fourier(p)
    F = fourier.fft1d(np.array(model.W_E, dtype=np.float64)[:, :p]) ** 2
    e = F.sum(0)[1::2][:nf] + F.sum(0)[2::2][:nf]
    tilt_sd.append(e.std() / e.mean())
    zsc = (e - e.mean()) / e.std()
    tilt_z.append(np.mean([zsc[k - 1] for k in final]))

print("=== blind-draw margin percentile vs reconfiguration ===")
for name, rc, pct, mf, mb in sorted(rows, key=lambda r: r[2]):
    print(f"  {'RECONF' if rc else '      '}  blind-pct {pct:5.1f}  "
          f"blind relM {mb:.3f}  final relM {mf:.3f}  {name}")
rc_p = [r[2] for r in rows if r[1]]
ok_p = [r[2] for r in rows if not r[1]]
print(f"\nreconfigurers n={len(rc_p)}, mean blind pct {np.mean(rc_p):.1f}")
print(f"loyal         n={len(ok_p)}, mean blind pct {np.mean(ok_p):.1f}")
u = mannwhitneyu(rc_p, ok_p, alternative="less")
print(f"Mann-Whitney (reconf lower): one-sided p = {u.pvalue:.4f}")
lo_rc = sum(p_ < 25 for p_ in rc_p); lo_ok = sum(p_ < 25 for p_ in ok_p)
tab = [[lo_rc, len(rc_p) - lo_rc], [lo_ok, len(ok_p) - lo_ok]]
print(f"bottom-quartile blind draws: reconf {lo_rc}/{len(rc_p)}, "
      f"loyal {lo_ok}/{len(ok_p)}; Fisher p = {fisher_exact(tab, 'greater')[1]:.4f}")

print(f"\n=== init W_E tilt ===")
print(f"relative sd of per-freq init energy: mean {np.mean(tilt_sd):.3f} "
      f"(chi2(2d) prediction {np.sqrt(1/128):.3f})")
print(f"committee members' mean init-energy z-score: {np.mean(tilt_z):+.3f} "
      f"(sd across runs {np.std(tilt_z):.3f})")
