import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch  # noqa: E402

torch.set_num_threads(1)
