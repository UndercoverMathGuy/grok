"""Grokked-toy transfer test: do quadratic-net committees inherit the
transformer's mask-specific favorites?  Primary stat: toy-vs-transformer
committee overlap, same-mask vs cross-mask."""
import sys
from pathlib import Path
import mlx.core as mx
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from grok.config import Config
from grok.data import train_test_split
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

P, N, SIGMA, LR, WD = 113, 256, 0.3, 1e-3, 1.0
B1, B2, EPS = 0.9, 0.98, 1e-8
MAXS = 20000

a, b = np.divmod(np.arange(P * P), P)
all_lab = (a + b) % P

def spectrum(logits):
    x = (a[:, None] + b[:, None] - np.arange(P)[None, :]) % P
    Lx = np.zeros(P); np.add.at(Lx, x.ravel(), logits.ravel()); Lx /= P * P
    ks = np.arange(1, P // 2 + 1)
    return np.abs((2.0 / P) * (Lx[None, :] * np.cos(2 * np.pi * ks[:, None]
                  * np.arange(P)[None, :] / P)).sum(1))

def run_toy(seed, is_train, is_test):
    ta, tb = a[is_train], b[is_train]; n = ta.size
    Oa = np.zeros((n, P), np.float32); Oa[np.arange(n), ta] = 1
    Ob = np.zeros((n, P), np.float32); Ob[np.arange(n), tb] = 1
    y_np = np.full((n, P), -1.0 / P, np.float32)
    y_np[np.arange(n), (ta + tb) % P] += 1.0
    Oa_, Ob_, y = mx.array(Oa), mx.array(Ob), mx.array(y_np)
    rng = np.random.default_rng(seed)
    V = mx.array(rng.normal(0, SIGMA, (N, P)).astype(np.float32))
    W = mx.array(rng.normal(0, SIGMA, (N, P)).astype(np.float32))
    U = mx.array(rng.normal(0, SIGMA / np.sqrt(N), (P, N)).astype(np.float32))
    z = lambda: (mx.zeros_like(V), mx.zeros_like(W), mx.zeros_like(U))
    state = (V, W, U, *z(), *z(), mx.array(0.0))

    def step(V, W, U, mV, mW, mU, sV, sW, sU, t):
        H = V @ Oa_.T + W @ Ob_.T
        A = H * H
        R = A.T @ U.T - y
        gU = R.T @ A.T / n
        Bm = (U.T @ R.T) * (2 * H) / n
        gV = Bm @ Oa_; gW = Bm @ Ob_
        t = t + 1
        out = []
        for P_, g, m, s in ((V, gV, mV, sV), (W, gW, mW, sW), (U, gU, mU, sU)):
            m = B1 * m + (1 - B1) * g
            s = B2 * s + (1 - B2) * g * g
            out.append((P_ - LR * ((m / (1 - B1**t)) /
                       (mx.sqrt(s / (1 - B2**t)) + EPS) + WD * P_), m, s))
        (V, mV, sV), (W, mW, sW), (U, mU, sU) = out
        return V, W, U, mV, mW, mU, sV, sW, sU, t

    cstep = mx.compile(step)
    for s in range(MAXS):
        state = cstep(*state)
        if s % 1000 == 999:
            mx.eval(*state)
            Vn, Wn, Un = (np.array(t2, np.float64) for t2 in state[:3])
            logits = ((Vn[:, a] + Wn[:, b]) ** 2).T @ Un.T
            te = (logits[is_test].argmax(1) == all_lab[is_test]).mean()
            if te >= 0.999 and s >= 3999:   # grokked + a little consolidation
                return committee_from_coeffs(spectrum(logits)), te, s + 1
    return committee_from_coeffs(spectrum(logits)), te, MAXS

# transformer committees per mask
tf = {}
for d, c in discover():
    if c.p != P or c.data_seed not in (0, 1, 2):
        continue
    coeffs, acc, _ = final_coeffs_and_acc(d, c)
    if acc < 0.99:
        continue
    tf.setdefault(c.data_seed, []).append(set(committee_from_coeffs(coeffs)))

toy = {}
for ds in (0, 1, 2):
    cfg = Config(p=P, data_seed=ds)
    is_train, is_test = train_test_split(cfg)
    toy[ds] = []
    for seed in range(10):
        comm, te, steps = run_toy(1000 * ds + seed, is_train, is_test)
        toy[ds].append(set(comm))
        print(f"dseed{ds} toy{seed}: test {te:.3f} @ {steps}  K={len(comm)}  "
              f"{sorted(comm)}", flush=True)

def mean_overlap(A, Bl):
    v = [len(x & y) for x in A for y in Bl]
    return np.mean(v)

def chance(A, Bl):
    return np.mean([len(x) * len(y) / 56.0 for x in A for y in Bl])

print("\n=== toy-toy within-mask consistency ===")
for ds in (0, 1, 2):
    pairs = [len(toy[ds][i] & toy[ds][j]) for i in range(10) for j in range(i+1, 10)]
    ch = np.mean([len(toy[ds][i]) * len(toy[ds][j]) / 56.0
                  for i in range(10) for j in range(i+1, 10)])
    print(f"dseed{ds}: mean shared {np.mean(pairs):.2f} (chance {ch:.2f})")

print("\n=== toy vs TRANSFORMER, same-mask vs cross-mask ===")
same, cross = [], []
for ds_t in (0, 1, 2):
    for ds_f in (0, 1, 2):
        o = mean_overlap(toy[ds_t], tf[ds_f]); ch = chance(toy[ds_t], tf[ds_f])
        (same if ds_t == ds_f else cross).append(o - ch)
        tag = "SAME " if ds_t == ds_f else "cross"
        print(f"toy d{ds_t} vs tf d{ds_f} [{tag}]: {o:.2f} (chance {ch:.2f})")
print(f"\nmean excess overlap: same-mask {np.mean(same):+.3f}, "
      f"cross-mask {np.mean(cross):+.3f}")
