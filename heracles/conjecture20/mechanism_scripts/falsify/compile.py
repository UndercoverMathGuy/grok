# Collect saved configurations, re-evaluate with the final (best) Gamma table, print the summary table.
import numpy as np, glob, json, math, re
from gget import gamma_table
from relmodel import Model
import sys; sys.path.insert(0,'..')
from prog3 import restricted
CS = 1 - (2 / 3) ** 0.5; CA = 1 / (2 * 3 ** 0.5)
GT = gamma_table(); rows = {}
for fn in glob.glob('res/run_p*.npy') + glob.glob('res/run3_p*.npy') + glob.glob('res/run2_p*.npy'):
    p = int(re.findall(r'_p(\d+)', fn)[0]); d = np.load(fn, allow_pickle=True).item()
    M = Model(p, 1.5, CS, GT[p]['Gamma'])
    R = rows.setdefault(p, dict(cstar=(9, None), pair=(-9, None), cap=(-9, None)))
    if 'run2' in fn:
        items = [('cstar', (d['c'], d['name'], [d['a'], d['v'], d['th']]))]
    else:
        items = [(k, v) for k, v in d.items()]
    for k, (val, name, cfg) in items:
        a, v, th = [np.asarray(x, float) for x in cfg]
        if k == 'cstar':
            S = np.nonzero(a > 1e-9 * a.max())[0]
            Q, yy = restricted(M.Knp, 1 / 1.5, list(S), np.random.default_rng(0), tries=4)
            if Q < np.inf and Q ** (-1.5) > M.G * (1 + 1e-9):
                print(f'WARNING p={p}: support of {name} has programme value {Q ** (-1.5):.6f} > table {M.G:.6f}')
            c, dd = M.cstar(a, v, th)
            cov, B, osc = M.parts(a, v, th); i = dd - 1
            lam = (osc[i] / 2) / np.sqrt(np.sum(v ** 2 * M.Knp[i]))
            if c < R['cstar'][0]: R['cstar'] = (c, dict(src=fn.split('/')[-1], name=name, lam=lam, supp=int(np.sum(a > 1e-6 * a.max())),
                                                     cheap=int(np.sum(v > 1e-6 * a.max())), covdef=1 - cov[i] / M.norm(a),
                                                     r_sh=M.ratio(a, v, th, CS), r_A=M.ratio(a, v, th, CA)))
        for key, kk in [('forced_pair_sh', 'pair'), ('forced_pair', 'pair'), ('forced_cap_sh', 'cap'), ('forced_cap', 'cap')]:
            if k == key:
                r = M.ratio(a, v, th, CS)
                if r > R[kk][0]: R[kk] = (r, dict(name=name, r_A=M.ratio(a, v, th, CA)))
print(' p     Gamma   K*  | c*     lam   supp/cheap  covdef  ratio@.1835 ratio@.289 | pair(sharp) [@.289]  cap(sharp) [@.289]')
for p in sorted(rows):
    R = rows[p]; c, I = R['cstar']; pr, PI = R['pair']; cr, CI = R['cap']
    print(f'{p:5d} {GT[p]["Gamma"]:.4f} {len(GT[p]["S"]):3d} | {c:.4f} {I["lam"]:.2f} {I["supp"]:4d}/{I["cheap"]:<4d} {I["covdef"]:.3f}  {I["r_sh"]:.3f}  {I["r_A"]:.3f} [{I["name"]}]'
          f' | {pr:.4f} [{PI["r_A"]:.4f}] {PI["name"]}  {cr:.4f} [{CI["r_A"]:.4f}] {CI["name"]}')
