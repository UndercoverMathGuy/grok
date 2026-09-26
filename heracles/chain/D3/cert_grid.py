# Certified graded curve: for each depth a, kappa_PSD(a) from a dense float grid, then certify kappa = 0.999 kappa_PSD(a)
# with interval arithmetic (cert_curve). Two variants: fable's single chain order (cross = 4) and two chain orders (2 sqrt2).
import json, sys, numpy as np
import cert_curve as cc
E = np.linspace(1, np.sqrt(3), 2001)[:, None]; H = np.linspace(0, 1, 3001)[None, :] * np.sqrt(3) * E
def kap_psd(a, cross):
    u = np.sqrt(2 * E * (E - 1)); B = 3 * E ** 2 + 2 * H ** 2 + cross * H * u + 2 * u ** 2
    Lv = np.maximum(E ** (2 / 3), 3 ** (1 / 3) * E ** (4 / 3) * (3 / B) ** (1 / 3))
    rhs = np.clip(1 - a * H, 0, None) ** (2 / 3)
    with np.errstate(divide='ignore'):
        return min(np.where(rhs > 0, Lv / rhs, np.inf).min(), 3 ** (1 / 3))
variant = sys.argv[1]
cc.CROSS = 4.0 if variant == 'one' else 2 * np.sqrt(2)
rows = []
for a in [round(0.01 * i, 2) for i in range(1, 28)]:
    k = 0.999 * kap_psd(a, cc.CROSS)
    final, bad = cc.cover(a, k)
    ok = not bad and all(cc.iv_check(a, k, b) for b in final)
    lone = min(3 ** (1 / 3), (1 - np.sqrt(3) * a) ** (-2 / 3))
    rows.append(dict(a=a, kappa=k, certified=ok, boxes=len(final), delta=(3 ** (1 / 3) / k) ** 1.5,
                     delta_lone=(3 ** (1 / 3) / lone) ** 1.5))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(f'cert_grid_{variant}.json', 'w'), indent=1)
