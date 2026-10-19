# DICTIONARY LP search (a global search at the level of output weights).
# Dictionary: neurons with ONE frequency per slot, u_t(a) = sqrt(2/p) cos(2 pi k_t a/p + phi_t) (k_t = 0: 1/sqrt(p)),
#   w(c) = sqrt(2/p) cos(2 pi j c/p + psi), input phases on the grid pi k/g, psi on pi k/g (k < 2g); each column has weight lam >= 0
#   (table scale).  Balanced cost of a column: nu^{b/2} lam^r, r = b/nu (concave).  Gaps are LINEAR in lam.
# Reweighted l1 (concave cost):  min sum_i w_i lam_i  s.t.  g(x, delta; lam) >= 1  for all x, delta  (exact LP over all rows),
#   w_i = (lam_i + eps)^{r-1}, from several random initial weightings.  The best LP iterate is turned into a network and
#   polished by the general-network optimiser with the class constraint enforced exactly by per-slot frequency masks.
# Modes:
#   b1   |S| = 1 (feature 1): own pure-1 class-C neurons + FOREIGN neurons (any single-frequency-per-slot neuron that is not
#        pure in a frequency j != 1: beta_j = 0 exactly).  Ratio vs nu^{1-s} B_p kappa_min (n = 2: = D_{1}, pi = 3^{1-s}).
#   cS   class C with features in S (pure neurons only, frequencies in S); ratio vs 3^{1-s} B_p Gamma^{(S)}.
#   gS   general: pure neurons with frequency in S + all non-pure single-frequency-per-slot neurons (mixed modes, DC, channels)
#        whose pure frequency (if any) is in S; ratio vs 3^{1-s} B_p Gamma^{(S)} (n = 2).
# Usage: python dict_lp.py p n b mode grid nrest seed [S]     S: comma list (cS/gS), default F
import sys, os, itertools, json, time
import numpy as np
from scipy.optimize import linprog
import hlib as H, hlib2 as H2, glib as Gb
GL = H.GL

p = int(sys.argv[1]); n = int(sys.argv[2]); bexp = float(sys.argv[3]); mode = sys.argv[4]; g = int(sys.argv[5])
nrest = int(sys.argv[6]); seed = int(sys.argv[7])
F = (p - 1) // 2; nu = n + 1; s = nu / bexp; r = 1 / s
S = [int(v) for v in sys.argv[8].split(',')] if len(sys.argv) > 8 else list(range(1, F + 1))
rng = np.random.default_rng(seed)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out2')
tag = f"dict_p{p}_n{n}_b{bexp}_{mode}_g{g}_s{seed}"
kmin = 1 - np.cos(2 * np.pi / p)
if mode == 'b1':
    S = [1]; Dval = nu ** (1 - s) * H.Bp(p, n) * kmin
else:
    RS = H2.prog_bb(p, s, S=S, maxnodes=20000 if len(S) <= 12 else 300, restarts=30 if len(S) <= 12 else 300)
    Dval = nu ** (1 - s) * H.Bp(p, n) * RS['G']
a_ = np.arange(p)
X = np.indices((p,) * n).reshape(n, -1).T; N = X.shape[0]
Y = X.sum(1) % p
phis = np.pi * np.arange(g) / g


def slot_vecs(k):
    if k == 0: return [(0.0, np.full(p, 1 / np.sqrt(p)))]
    return [(ph, np.sqrt(2 / p) * np.cos(2 * np.pi * k * a_ / p + ph)) for ph in phis]


# ---------------------------------------------------------------- dictionary
cols = []      # (kvec, j, phases, psi)
if mode == 'b1':
    kopts = list(range(0, F + 1)); jopts = list(range(1, F + 1))
else:
    kopts = list(range(0, F + 1)) if mode == 'gS' else S; jopts = S if mode == 'cS' else list(range(1, F + 1))
specs = []
for j in jopts:
    for kv in itertools.product(kopts, repeat=n):
        pure = all(k == kv[0] for k in kv) and kv[0] == j
        if mode == 'b1':
            if pure and j != 1: continue                      # beta_j != 0 for j != 1
            if not pure and all(k == j for k in kv): continue
        elif mode == 'cS':
            if not pure: continue
        else:  # gS
            if pure and j not in S: continue
            if not pure and all(k == kv[0] for k in kv) and kv[0] not in S and kv[0] == j: continue
        specs.append((kv, j, pure))
nspec = len(specs)
# column budget: random phase subsets when too many
maxcols = int(os.environ.get('MAXCOLS', 60000))
psis = list(np.pi * np.arange(2 * g) / g)          # output phase grid (full circle, step pi/g): contains the IF alignment
allcols = []
for (kv, j, pure) in specs:
    phs = [list(phis) if k != 0 else [0.0] for k in kv]
    for ph in itertools.product(*phs):
        for psi in psis: allcols.append((kv, j, ph, psi, pure))
print(f"# {tag}: S={S} D={Dval:.6e} specs={nspec} dictionary columns={len(allcols)} rows={N * (p - 1)}", flush=True)


def colmat(cl):
    """gap rows (x, delta) for every column (unit table scale); returns (N*(p-1), len(cl))"""
    M = np.zeros((N * (p - 1), len(cl)))
    d = np.arange(1, p)
    for c, (kv, j, ph, psi, pure) in enumerate(cl):
        prod = np.ones(N)
        for t in range(n):
            k = kv[t]
            prod *= (np.full(p, 1 / np.sqrt(p)) if k == 0 else np.sqrt(2 / p) * np.cos(2 * np.pi * k * a_ / p + ph[t]))[X[:, t]]
        w = np.sqrt(2 / p) * np.cos(2 * np.pi * j * a_ / p + psi)
        G = prod[:, None] * (w[Y][:, None] - w[(Y[:, None] - d[None, :]) % p])
        M[:, c] = G.ravel()
    return M


def to_net(cl, lam):
    # columns sharing inputs and output frequency are MERGED into one neuron (output = sum of lam * w): cost can only drop
    groups = {}
    for (kv, j, ph, psi, pure), lm in zip(cl, lam):
        key = (tuple(kv), j, tuple(np.round(ph, 9)))
        w = np.sqrt(2 / p) * np.cos(2 * np.pi * j * a_ / p + psi) * lm
        groups[key] = groups.get(key, 0) + w
    Us = []; Ws = []; al = []
    for (kv, j, ph), Wv in groups.items():
        lm = np.linalg.norm(Wv)
        if lm <= 0: continue
        sc = lm ** (1.0 / nu)
        U = np.array([(np.full(p, 1 / np.sqrt(p)) if kv[t] == 0 else np.sqrt(2 / p) * np.cos(2 * np.pi * kv[t] * a_ / p + ph[t])) * sc for t in range(n)])
        Us.append(U); Ws.append(Wv / lm * sc)
        al.append([{kv[t]} for t in range(n)] + [{j}])
    return np.array(Us), np.array(Ws), al


def cc_from(U, W):
    # class-C model of an all-pure network (n any): b = conj(beta_i) per neuron, xi from the input phases
    freqs = []; bs = []; phs = []
    for i in range(U.shape[0]):
        l = int(np.argmax(np.abs(np.fft.fft(W[i]))[1:F + 1])) + 1
        nt = GL.Net(U[i:i + 1], W[i:i + 1], bexp)
        beta = nt.clean()[l - 1]
        # u_t(a) = A cos(2 pi l a/p + ph_t): fft at l = (A p/2) e^{i ph_t};  model phase phi_t = 2 ph_t;  model b = conj(beta_i)
        freqs.append(l); bs.append(np.conj(beta)); phs.append(np.angle(np.fft.fft(U[i], axis=1)[:, l]) * 2)
    cc = H.CC(p, n, freqs, s)
    x = cc.pack(np.array(bs), np.array(phs))
    return cc, x


t0 = time.time(); recs = []; best = (-np.inf, None)
for it in range(nrest):
    cl = allcols if len(allcols) <= maxcols else [allcols[i] for i in rng.choice(len(allcols), maxcols, replace=False)]
    if it == 0 or len(allcols) > maxcols: M = colmat(cl)
    w = np.ones(len(cl)) if it == 0 else rng.uniform(0.2, 5.0, len(cl)) ** 2
    lam_best = (-np.inf, None)
    for k in range(25):
        res = linprog(w, A_ub=-M, b_ub=-np.ones(M.shape[0]), bounds=(0, None), method='highs')
        if res.status != 0: break
        lam = res.x * (res.x > 1e-12 * res.x.max())
        # merge identical neurons that differ only by psi into one (cost decreases); evaluate true normalised margin
        cost = nu ** (bexp / 2) * np.sum(lam[lam > 0] ** r)
        h = (M @ lam).min()
        val = h / cost ** s / Dval
        if val > lam_best[0]: lam_best = (val, lam.copy())
        wn = (lam / lam.max() + 1e-6) ** (r - 1)
        if np.allclose(wn, w, rtol=1e-9): break
        w = wn
    val, lam = lam_best
    idx = np.nonzero(lam > 0)[0]
    U, W, al = to_net([cl[i] for i in idx], lam[idx])
    nt = GL.Net(U, W, bexp); v0 = nt.margin() / nt.cost() ** s / Dval
    if mode == 'cS':
        # class C: exact Riesz-product model, mass LP + full SLP polish; check the model against the table
        cc, xc = cc_from(U, W)
        Dc = nu ** (1 - s) * RS['G']
        if abs(cc.gnorm(xc) / Dc - v0) > 1e-6 * max(1, abs(v0)):
            print('# warning: class-C conversion mismatch', cc.gnorm(xc) / Dc, v0, flush=True)
        xc, vc = H2.polish(cc, xc)
        v1 = vc / Dc
        U2, W2 = cc.to_net(xc, bexp); nt = GL.Net(U2, W2, bexp)
    else:
        # polish with exact class constraints (each neuron keeps its slot frequencies)
        gnet = Gb.GNet(p, n, bexp, al)
        x = gnet.from_UW(U, W)
        try:
            x = gnet.lbfgs(x, taus=(3e-3, 1e-3, 3e-4, 1e-4), iters=800)
            x, v1 = gnet.slp(x, iters=80)
        except Exception as e:
            v1 = gnet.gamma(x)
        v1 /= Dval
        nt = gnet.net(x)
    cl_ = np.abs(nt.clean()); feats = [l for l in range(1, F + 1) if cl_[l - 1] > 1e-7 * cl_.max()]
    nforeign = sum(1 for i in idx if not cl[i][4])
    oute = [float(np.sum((nt.W @ GL.freq_proj(p, k)) ** 2)) for k in range(0, F + 1)]
    rec = dict(it=it, lp_ratio=val, net_ratio=v0, polished=v1, neurons=len(idx), nonpure=nforeign, feats=feats,
               out_energy_share=[round(e / sum(oute), 4) for e in oute], t=round(time.time() - t0))
    recs.append(rec); print(json.dumps(rec), flush=True)
    if v1 > best[0]: best = (v1, (nt.U, nt.W))
    if v1 > 1 + 1e-9 or val > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_{it}.npz"), U=nt.U, W=nt.W, p=p, n=n, b=bexp, S=S, ratio=v1, lp_ratio=val)
np.savez(os.path.join(OUT, f"best_{tag}.npz"), U=best[1][0], W=best[1][1], p=p, n=n, b=bexp, S=S, ratio=best[0])
print(f"# SUMMARY {tag}: runs={len(recs)} best LP ratio={max(z['lp_ratio'] for z in recs):.10f} best polished={best[0]:.10f} "
      f"max nonpure neurons in a solution={max(z['nonpure'] for z in recs)} time={time.time() - t0:.0f}s", flush=True)
