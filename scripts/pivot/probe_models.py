"""Hand-rolled numpy learners for the probe-ceiling study (no sklearn in env).

  logreg   L2 logistic regression, IRLS/Newton with a ridge path
  gbt      histogram gradient-boosted trees, level-wise, logistic loss
  mlp      2 hidden layers, ReLU, Adam, weight decay

All take pre-standardized dense float64 X.
"""
import numpy as np


# --------------------------------------------------------------- logistic
def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def logreg_fit(X, y, lam=1.0, iters=60, tol=1e-9):
    n, f = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(f + 1)
    R = lam * np.eye(f + 1)
    R[-1, -1] = 0.0                       # never penalize the intercept
    yf = y.astype(float)
    for _ in range(iters):
        p = _sig(Xb @ w)
        g = Xb.T @ (p - yf) + R @ w
        s = np.clip(p * (1 - p), 1e-8, None)
        H = (Xb * s[:, None]).T @ Xb + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


def logreg_pred(w, X):
    return X @ w[:-1] + w[-1]


# ------------------------------------------------------------------- gbt
def _bin_edges(X, nbins):
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return [np.unique(np.percentile(X[:, j], qs)) for j in range(X.shape[1])]


def _codes(X, edges, nbins):
    C = np.empty(X.shape, dtype=np.int32)
    for j, e in enumerate(edges):
        C[:, j] = np.searchsorted(e, X[:, j], side="left")
    return np.clip(C, 0, nbins - 1)


class _Tree:
    __slots__ = ("feat", "thr", "leaf")


def _fit_tree(C, g, h, depth, nbins, l2, min_child, rng, colsample):
    n, F = C.shape
    node = np.zeros(n, np.int64)
    feats = np.arange(F)
    if colsample < 1.0:
        feats = np.sort(rng.choice(F, max(1, int(F * colsample)), replace=False))
    Fs = len(feats)
    Csub = C[:, feats]
    off = (np.arange(Fs) * nbins)[None, :]
    tree = _Tree()
    tree.feat, tree.thr = [], []
    for lvl in range(depth):
        nn = 1 << lvl
        flat = (node[:, None] * (Fs * nbins) + off + Csub).ravel()
        size = nn * Fs * nbins
        G = np.bincount(flat, np.repeat(g, Fs), size).reshape(nn, Fs, nbins)
        H = np.bincount(flat, np.repeat(h, Fs), size).reshape(nn, Fs, nbins)
        GL, HL = G.cumsum(2), H.cumsum(2)
        GT, HT = GL[:, :, -1:], HL[:, :, -1:]
        GR, HR = GT - GL, HT - HL
        ok = (HL >= min_child) & (HR >= min_child)
        gain = (GL ** 2 / (HL + l2) + GR ** 2 / (HR + l2)
                - GT ** 2 / (HT + l2))
        gain = np.where(ok, gain, -np.inf)
        gain[:, :, -1] = -np.inf                      # empty right child
        fl = gain.reshape(nn, -1)
        best = fl.argmax(1)
        bf, bt = best // nbins, best % nbins
        dead = ~np.isfinite(fl[np.arange(nn), best])
        bt = np.where(dead, nbins, bt)                # send everything left
        tree.feat.append(feats[bf])
        tree.thr.append(bt.copy())
        go_r = (Csub[np.arange(n), bf[node]] > bt[node]) & ~dead[node]
        node = node * 2 + go_r.astype(np.int64)
    nl = 1 << depth
    Gl = np.bincount(node, g, nl)
    Hl = np.bincount(node, h, nl)
    tree.leaf = -Gl / (Hl + l2)
    return tree


def _tree_pred(tree, C):
    n = C.shape[0]
    node = np.zeros(n, np.int64)
    for bf, bt in zip(tree.feat, tree.thr):
        go_r = C[np.arange(n), bf[node]] > bt[node]
        node = node * 2 + go_r.astype(np.int64)
    return tree.leaf[node]


def gbt_fit(X, y, Xval=None, yval=None, rounds=400, lr=0.06, depth=3,
            nbins=32, l2=2.0, min_child=8.0, colsample=0.7, seed=0,
            patience=40, score=None):
    """Returns (model, best_rounds). `score` (higher better) drives early
    stopping on (Xval, yval); defaults to validation log-likelihood."""
    rng = np.random.default_rng(seed)
    edges = _bin_edges(X, nbins)
    C = _codes(X, edges, nbins)
    yf = y.astype(float)
    base = np.log(max(yf.mean(), 1e-6) / max(1 - yf.mean(), 1e-6))
    F = np.full(len(yf), base)
    trees = []
    Cv = Fv = None
    if Xval is not None:
        Cv = _codes(Xval, edges, nbins)
        Fv = np.full(len(yval), base)
    best, best_n, bad = -np.inf, 0, 0
    for t in range(rounds):
        p = _sig(F)
        g, h = p - yf, np.clip(p * (1 - p), 1e-6, None)
        tr = _fit_tree(C, g, h, depth, nbins, l2, min_child, rng, colsample)
        trees.append(tr)
        F += lr * _tree_pred(tr, C)
        if Cv is None:
            continue
        Fv += lr * _tree_pred(tr, Cv)
        s = (score(Fv) if score is not None
             else float(np.mean(yval * Fv - np.logaddexp(0, Fv))))
        if s > best + 1e-6:
            best, best_n, bad = s, t + 1, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model = dict(edges=edges, nbins=nbins, base=base, lr=lr, trees=trees)
    return model, (best_n if Cv is not None else rounds)


def gbt_pred(model, X, ntrees=None):
    C = _codes(X, model["edges"], model["nbins"])
    F = np.full(X.shape[0], model["base"])
    for tr in model["trees"][:ntrees]:
        F += model["lr"] * _tree_pred(tr, C)
    return F


# ------------------------------------------------------------------- mlp
def mlp_fit(X, y, hidden=(48, 24), steps=1500, lr=3e-3, wd=1e-3, seed=0,
            Xval=None, yval=None, score=None, eval_every=25):
    rng = np.random.default_rng(seed)
    dims = [X.shape[1]] + list(hidden) + [1]
    W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
         for i in range(len(dims) - 1)]
    b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(x) for x in b]
    vb = [np.zeros_like(x) for x in b]
    yf = y.astype(float)
    best, bestP, bad = -np.inf, None, 0

    def fwd(Z, Ws, bs):
        acts = [Z]
        for i in range(len(Ws) - 1):
            Z = np.maximum(Z @ Ws[i] + bs[i], 0.0)
            acts.append(Z)
        return (Z @ Ws[-1] + bs[-1]).ravel(), acts

    for t in range(1, steps + 1):
        out, acts = fwd(X, W, b)
        d = (_sig(out) - yf)[:, None] / len(yf)
        for i in range(len(W) - 1, -1, -1):
            gW = acts[i].T @ d + wd * W[i]
            gb = d.sum(0)
            if i > 0:
                d = (d @ W[i].T) * (acts[i] > 0)
            for arr, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                arr[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (
                    np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        if Xval is not None and t % eval_every == 0:
            ov, _ = fwd(Xval, W, b)
            s = (score(ov) if score is not None
                 else float(np.mean(yval * ov - np.logaddexp(0, ov))))
            if s > best + 1e-6:
                best, bad = s, 0
                bestP = ([w.copy() for w in W], [x.copy() for x in b])
            else:
                bad += 1
                if bad >= 12:
                    break
    if bestP is not None:
        W, b = bestP
    return (W, b)


def mlp_pred(model, X):
    W, b = model
    Z = X
    for i in range(len(W) - 1):
        Z = np.maximum(Z @ W[i] + b[i], 0.0)
    return (Z @ W[-1] + b[-1]).ravel()
