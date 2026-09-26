import sys; sys.path.insert(0, '..')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from transfer_flip import depth
for As in (0.5, 0.52, 0.55, 0.559):
    c, lam = depth(As, 0.25, n=120)
    print(f"A*=1/4, A_s={As}: c ~ {c:.4f}, lam {lam.round(3)}")
