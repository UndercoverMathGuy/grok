import sys, time
sys.argv = ['cert_pt.py', 'pt', '1.5', '0.6']
from fractions import Fraction as Fr
import cert_pt as cp
t = time.time()
for i in range(5):
    cp.inner_low(Fr(0.731), Fr(-0.4123))
print("per call", (time.time()-t)/5)
t = time.time()
for i in range(5):
    cp.root_clusters([Fr(12), Fr(0), -6*Fr(0.731), -Fr(-0.4123)])
print("root_clusters per call", (time.time()-t)/5)
