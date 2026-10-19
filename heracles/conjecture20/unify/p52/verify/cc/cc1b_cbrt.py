import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np, iv
from fractions import Fraction as Fr
rng = np.random.default_rng(12345)
xs = np.array([5e-324, 1e-310, 2.2250738585072014e-308, 1e308, 1.7976931348623157e308, 1e-320])
x = xs; y = iv.up(iv.up(np.cbrt(x)))
for it in range(6):
    b_ = iv._cube_dn(y) < x
    y = np.where(b_, iv.up(y * (1 + 1e-15)), y)
print("unverified:", xs[iv._cube_dn(y) < x])
# relevant range for pow23 in the certificate: [0, 4]
x = np.concatenate([rng.uniform(0, 4, 2_000_000), np.linspace(0, 4, 100001)])
y = iv.up(iv.up(np.cbrt(x))); first = iv._cube_dn(y) < x
print("in [0,4]: values needing refinement:", int(first.sum()))
lo = iv.cbrt_lo(x[:200000]); 
print("cbrt_lo in [0,4] zeroed-out (fallback to 0):", int(np.sum((lo == 0) & (x[:200000] > 0))))
