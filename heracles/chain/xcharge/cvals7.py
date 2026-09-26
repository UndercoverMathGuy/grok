import sys; sys.path.insert(0, '..')
import io, contextlib, math
with contextlib.redirect_stdout(io.StringIO()):
    from transfer_flip import depth
for p in (5, 7, 11, 13):
    c, lam = depth(1 / math.sqrt(p - 1), 1 / (p - 1), n=120)
    print(f"p={p}: elementary A*>=1/(p-1)={1/(p-1):.3f}, A_s>=1/sqrt(p-1)={1/math.sqrt(p-1):.3f} -> c ~ {c:.4f}")
