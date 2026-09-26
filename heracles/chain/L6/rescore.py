# Re-score gap-scan / adversary sets with the stronger floored-programme optimiser (floor_opt.py):
# single-offset accounting (uniform discount delta(a) at a = A_s optimistic, a = c_lo pessimistic), the multi-offset lever
# (per-offset, per-frequency depth max(a_flip, c_plain sqrt(2/cov_k(D)))), and the critical discount delta_crit(T, eps).
# usage: rescore.py p file.jsonl    (writes rescore_p{p}.jsonl)
import sys, json, numpy as np
sys.path.insert(0, '..')
from floor_opt import floored_value, rate_matrix
from tools import gamma_star, cov_matrix

p = int(sys.argv[1]); src = sys.argv[2]; rng = np.random.default_rng(7 + p)
Gam, Sstar = gamma_star(p, rng)
A = cov_matrix(p)
delta = lambda a: np.maximum(1.0, np.sqrt(3) * (1 - np.sqrt(3) * np.asarray(a, float)))
done = set()
try:
    done = {(json.loads(l)['src'], tuple(json.loads(l)['T'])) for l in open(f'rescore_p{p}.jsonl')}
except FileNotFoundError:
    pass
out = open(f'rescore_p{p}.jsonl', 'a')
for line in open(src):
    r = json.loads(line); T = r['T']
    if (src, tuple(T)) in done:
        continue
    rec = dict(src=src, name=r.get('name', f"adv K={r.get('K')}"), T=T, K=len(T), A_star=r['A_star'], A_s=r['A_s'], c_lo=r['c_lo'])
    Ti = np.array(T) - 1
    Ap = min(r['A_star'], 0.47); c_plain = Ap * (2 + Ap) / 3
    lev = c_plain * np.sqrt(2 / A[:, Ti])
    for eps in (0.01, 0.03):
        if eps * len(T) > 0.9:
            continue
        val = lambda R: floored_value(p, T, R, eps, Gam, rng, seeds=[Sstar])
        for tag, a in (('opt', r['A_s']), ('pess', r['c_lo'])):
            d = float(delta(a))
            rec[f'single_{tag}_eps{eps}'] = val(rate_matrix(p, T, d)) if d > 1 else None
            dm = delta(np.maximum(a, lev))
            rec[f'lever_{tag}_eps{eps}'] = val(rate_matrix(p, T, dm)) if (dm > 1).any() else None
        # critical uniform discount
        if val(rate_matrix(p, T, np.sqrt(3))) <= 1:
            rec[f'delta_crit_eps{eps}'] = float(np.sqrt(3))
        else:
            lo, hi = 1.0, float(np.sqrt(3))
            for _ in range(8):
                mid = (lo + hi) / 2
                if val(rate_matrix(p, T, mid)) > 1 + 1e-9: hi = mid
                else: lo = mid
            rec[f'delta_crit_eps{eps}'] = lo
        dc = rec[f'delta_crit_eps{eps}']
        rec[f'depth_needed_eps{eps}'] = max(0.0, (1 - dc / np.sqrt(3)) / np.sqrt(3))
    out.write(json.dumps(rec) + '\n'); out.flush()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != 'src'}), flush=True)
