# Does the multi-offset lever (independent row densities at D and -D) close sets that single-offset aiming leaves open?
# Per offset D and frequency k in T, the proven relative depths are
#   flip (same density, shifts +-D/2):          a_flip  (flip-enlarged polyball depth; use c_lo pessimistic, A_s optimistic)
#   independent densities at +D and -D:          c_plain * sqrt(2 / cov_k(D)),  c_plain = plain-body polyball depth
#                                                (fable transfer from the one-sided depth A*: A'(2+A')/3, A' = min(A*, 0.47))
# a_k(D) = max of the two; discount delta(a_k(D)) on the lone-atom curve; offset-dependent share-floored programme.
# Reads gap_p{p}.jsonl, writes lever_p{p}.jsonl.  usage: lever_test.py p
import sys, json, numpy as np
sys.path.insert(0, '..')
from lever import floored_ratio_Ddep
from common import cover_matrix

p = int(sys.argv[1]); rng = np.random.default_rng(100 + p)
delta = lambda a: np.maximum(1.0, np.sqrt(3) * (1 - np.sqrt(3) * a))
A = cover_matrix(p)
done = set()
try:
    done = {json.loads(l)['name'] for l in open(f'lever_p{p}.jsonl')}
except FileNotFoundError:
    pass
out = open(f'lever_p{p}.jsonl', 'a')
for line in open(f'gap_p{p}.jsonl'):
    r = json.loads(line)
    if r['name'] in done:
        continue
    T = r['T']; Ti = np.array(T) - 1
    Ap = min(r['A_star'], 0.47); c_plain = Ap * (2 + Ap) / 3
    cov = A[:, Ti]                                            # [D, k in T]
    lev = c_plain * np.sqrt(2 / cov)
    rec = dict(name=r['name'], T=T, c_plain=c_plain)
    for tag, aflip in (('pess', r['c_lo']), ('opt', r['A_s'])):
        dmat = delta(np.maximum(aflip, lev))
        rec[f'frac_discounted_{tag}'] = float((dmat > 1 + 1e-12).mean())
        for eps in (0.01, 0.03):
            if eps * len(T) > 0.9:
                continue
            rec[f'lever_ratio_{tag}_eps{eps}'] = floored_ratio_Ddep(p, T, dmat, eps, r['Gamma_star'], rng, starts=15)
            rec[f'single_ratio_{tag}_eps{eps}'] = r.get(f'ratio_{tag}_eps{eps}')
    out.write(json.dumps(rec) + '\n'); out.flush()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
