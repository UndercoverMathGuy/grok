# Check L6's correction: flip-enlarged depth <= two-sided K-mode depth (single-pattern directions); dense sets.
import numpy as np
def ratio_two_sided(p, z):
    S = np.arange(1, (p - 1) // 2 + 1); ze = np.exp(-2j * np.pi * 2 * np.outer(S, np.arange(p)) / p)
    return np.abs(np.real(z @ ze)).max() / np.abs(z).sum()
for p in (11, 13, 17):
    h = (p - 1) // 2; k = np.arange(1, h + 1)
    leg = np.array([1 if pow(int(x), (p - 1) // 2, p) == 1 else -1 for x in k], float)
    cands = {'legendre': leg, 'legendre*i': 1j * leg, 'chirp': np.exp(1j * np.pi * k ** 2 / h), 'alt': (-1.0) ** k}
    best = min((ratio_two_sided(p, z), n) for n, z in cands.items())
    print(f"dense p={p}: explicit single-pattern direction {best[1]} gives two-sided depth <= {best[0]:.4f}")
