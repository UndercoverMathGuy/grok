import sys, numpy as np
sys.path.insert(0, '..')
from transfer_flip import depth as tdepth
from flip_ball import make, PLAIN, FLIP, depth2
print("flip-transfer bound c(A*=0.25, A_s=0.559):", round(tdepth(0.55902, 0.25)[0], 4), "  fable A(2+A)/3:", round(0.25 * 2.25 / 3, 4))
print("search (upper est.) plain:", round(depth2(5, [1, 2], PLAIN, extra=200), 4), " flip-enlarged:", round(depth2(5, [1, 2], FLIP, extra=200), 4))
