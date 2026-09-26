"""Product and power networks on the full modular-addition table (NORMS A1; paper prelim.tex).

Parameters of a stack of B independent networks of the same shape (batched lockstep training):
    U : (B, n, m, p)   U[b, t, i, a] = u_{i,t}(a), the weight of neuron i on operand t for token a
    W : (B, m, p)      W[b, i, c]    = w_i(c), the readout of neuron i for candidate c
No biases. A single network is B = 1.

    product:  f(x)[c] = sum_i  prod_t U[t, i, a_t] * W[i, c]            (nu = n + 1)
    power:    f(x)[c] = sum_i (sum_t U[t, i, a_t])^n * W[i, c]           (nu = n + 1)

The full table has p^n inputs x = (a_1..a_n) in row-major order (a_1 slowest) and p candidates c;
the label is y(x) = (a_1 + ... + a_n) mod p. Logits are returned as (B, p^n, p).

Per-neuron norm (paper prelim.tex): ||omega_i||^2 = sum_t ||U[t, i, :]||^2 + ||W[i, :]||^2.
L_{2,q} norm: ||theta||_{2,q} = (sum_i ||omega_i||^q)^(1/q).
"""
from __future__ import annotations

import math

import torch

MODELS = ("product", "power")


def labels(p: int, n: int, device=None) -> torch.Tensor:
    """y(x) = (a_1 + ... + a_n) mod p for the p^n inputs in row-major order (int64, length p^n)."""
    a = torch.arange(p, device=device)
    y = torch.zeros(1, dtype=torch.int64, device=device)
    for _ in range(n):
        y = (y[:, None] + a[None, :]).reshape(-1)
    return y % p


def inputs(p: int, n: int, device=None) -> torch.Tensor:
    """(p^n, n) int64 array of the inputs (a_1..a_n) in the same row-major order as the logits."""
    grids = torch.meshgrid(*[torch.arange(p, device=device)] * n, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=1)


def _outer_chain(vecs):
    """vecs: list of (B, m, p) tensors -> (B, p^k, m) with entry prod_t vecs[t][b, i, a_t]."""
    B, m, p = vecs[0].shape
    G = vecs[0].transpose(1, 2)  # (B, p, m)
    for v in vecs[1:]:
        G = (G[:, :, None, :] * v.transpose(1, 2)[:, None, :, :]).reshape(B, -1, m)
    return G


def product_logits(U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Full logit table of product networks: (B, p^n, p).

    The CP contraction is split into a left block (the first nL operands) and a right block
    (the remaining operands and the readout), so the largest intermediate is p^max(nL, nR+1) x m
    and the main cost is one batched matmul of p^n x p x m multiply-adds.
    """
    B, n, m, p = U.shape
    nL = (n + 2) // 2  # left operands; right block = n - nL operands + readout
    left = _outer_chain([U[:, t] for t in range(nL)])  # (B, p^nL, m)
    right = _outer_chain([U[:, t] for t in range(nL, n)] + [W])  # (B, p^(n-nL+1), m)
    F = torch.bmm(left, right.transpose(1, 2))  # (B, p^nL, p^(n-nL+1))
    return F.reshape(B, p ** n, p)


def power_logits(U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Full logit table of power networks: (B, p^n, p), f = (sum_t u_t(a_t))^n . W."""
    B, n, m, p = U.shape
    S = U[:, 0].transpose(1, 2)  # (B, p, m)
    for t in range(1, n):
        S = (S[:, :, None, :] + U[:, t].transpose(1, 2)[:, None, :, :]).reshape(B, -1, m)
    H = S ** n
    return torch.bmm(H, W)


def logits(model: str, U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    if model == "product":
        return product_logits(U, W)
    if model == "power":
        return power_logits(U, W)
    raise ValueError(model)


def neuron_sqnorms(U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """||omega_i||^2 = sum_t ||U[t,i,:]||^2 + ||W[i,:]||^2, shape (B, m)."""
    return (U * U).sum(dim=(1, 3)) + (W * W).sum(dim=2)


def penalty(U: torch.Tensor, W: torch.Tensor, q) -> torch.Tensor:
    """sum_i ||omega_i||^q per network, shape (B,). q: float or (B,) tensor."""
    sq = neuron_sqnorms(U, W)
    if not torch.is_tensor(q):
        q = torch.full((U.shape[0],), float(q), dtype=U.dtype, device=U.device)
    return (sq.clamp_min(0) ** (q[:, None] / 2)).sum(1)


def penalty_grad(U: torch.Tensor, W: torch.Tensor, q) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic gradient of sum_i ||omega_i||^q: q ||omega_i||^(q-2) * (U, W)."""
    sq = neuron_sqnorms(U, W)
    if not torch.is_tensor(q):
        q = torch.full((U.shape[0],), float(q), dtype=U.dtype, device=U.device)
    coef = q[:, None] * sq.clamp_min(0) ** (q[:, None] / 2 - 1)  # (B, m); q = 2 -> 2 (0^0 = 1)
    return coef[:, None, :, None] * U, coef[:, :, None] * W


def l2q_norm(U: torch.Tensor, W: torch.Tensor, q) -> torch.Tensor:
    """||theta||_{2,q} = (sum_i ||omega_i||^q)^(1/q), shape (B,)."""
    if not torch.is_tensor(q):
        q = torch.full((U.shape[0],), float(q), dtype=U.dtype, device=U.device)
    return penalty(U, W, q) ** (1.0 / q)


def ce_margin_form(F: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the table, per network: shape (B,).

    CE(x) = log sum_c exp(f_c) - f_y = softplus( logsumexp_{c != y} (f_c - f_y) ).
    This is algebraically identical to the standard cross-entropy. It is used because, in
    float32, the standard form rounds the correct-class gradient softmax_y - 1 to 0 once the
    wrong-class mass S = sum_{c!=y} exp(f_c - f_y) falls below ~6e-8; here the correct-class
    gradient is -sum_{c!=y} (accurate small terms) and the label entry is masked out exactly.
    """
    B, N, p = F.shape
    mask = torch.zeros(N, p, dtype=torch.bool, device=F.device)
    mask[torch.arange(N, device=F.device), y] = True
    mask = mask.view(1, N, p)
    # f_y by a masked sum (p-1 exact zeros + f_y), so the backward pass is elementwise and
    # deterministic on every device (a gather would back-propagate through scatter_add).
    fy = F.masked_fill(~mask, 0.0).sum(dim=2, keepdim=True)
    D = (F - fy).masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(D, dim=2)
    return torch.nn.functional.softplus(lse).mean(dim=1)


def ce_standard(F: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard mean cross-entropy per network (reference implementation, for tests)."""
    B, N, p = F.shape
    return torch.stack([torch.nn.functional.cross_entropy(F[b], y) for b in range(B)])


def margins_from_logits(F: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """h = min_x min_{c != y(x)} f(x)[y(x)] - f(x)[c], per network: shape (B,)."""
    B, N, p = F.shape
    fy = torch.gather(F, 2, y.view(1, N, 1).expand(B, N, 1))
    mask = torch.zeros(N, p, dtype=torch.bool, device=F.device)
    mask[torch.arange(N, device=F.device), y] = True
    Fo = F.masked_fill(mask.view(1, N, p), float("-inf"))
    return (fy[..., 0] - Fo.max(dim=2).values).min(dim=1).values


def init_params(model: str, n: int, m: int, p: int, seed: int, init_scale: float,
                scheme: str = "normal") -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic init of ONE network (B = 1), generated in float64 on CPU from its own seed.

    scheme 'normal': every entry i.i.d. N(0, init_scale^2 / p), so each weight vector has
    expected squared norm init_scale^2. The returned tensors are float64 and already rounded to
    float32 precision, so float32 and float64 runs from the same seed start from the same point.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    if scheme != "normal":
        raise ValueError(scheme)
    std = init_scale / math.sqrt(p)
    U = torch.randn((1, n, m, p), generator=g, dtype=torch.float64) * std
    W = torch.randn((1, m, p), generator=g, dtype=torch.float64) * std
    return U.float().double(), W.float().double()
