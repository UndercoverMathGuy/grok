"""1-layer transformer for modular addition, in MLX.

Faithful port of the original torch model (Nanda et al.): no LayerNorm, no
biases in attention, learned positional embeddings, weights stored in the
same shapes/names as the paper so analysis formulas (e.g.
W_neur = W_in @ W_O @ W_V @ W_E) transfer directly.

Instead of torch-style hooks, every forward pass can record its intermediate
activations: use `logits, cache = model.run_with_cache(tokens)`. The cache is
a plain dict keyed like 'blocks.0.attn' / 'blocks.0.mlp.post', so grabbing an
activation is just a dict lookup — add new cache entries by editing the
`_stash` calls in the forward passes below.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import Config


def _stash(cache, name, x):
    if cache is not None:
        cache[name] = x
    return x


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, h, dh = cfg.d_model, cfg.num_heads, cfg.d_head
        init = lambda *shape: mx.random.normal(shape) / math.sqrt(d)
        self.W_K = init(h, dh, d)
        self.W_Q = init(h, dh, d)
        self.W_V = init(h, dh, d)
        self.W_O = init(d, h * dh)
        if cfg.attn_init == "isometric":
            # QR the same Gaussian draw (stream-preserving, like _embed_init):
            # rows of Q are orthonormal, split into h blocks of d_head rows ->
            # per-head value subspaces tile d_model; W_O = Q^T reassembles
            # them isometrically, so sum_h ||OV_h x||^2 = ||x||^2 for every x.
            # Row/column norms are 1, matching the Gaussian init's expectation.
            g = np.array(self.W_V, dtype=np.float64).reshape(h * dh, d)
            q, _ = np.linalg.qr(g.T)   # q: (d, h*dh), orthonormal columns
            self.W_V = mx.array(q.T.astype(np.float32)).reshape(h, dh, d)
            self.W_O = mx.array(q.astype(np.float32))
        else:
            assert cfg.attn_init == "normal"
        self.d_head = dh
        self._mask = mx.tril(mx.ones((cfg.n_ctx, cfg.n_ctx)))

    def _heads(self, y):
        # (batch, pos, head*d_head) -> (batch, head, pos, d_head)
        return y.reshape(*y.shape[:2], -1, self.d_head).transpose(0, 2, 1, 3)

    def _project(self, x, W):
        # One gemm per projection: W is (head, d_head, d_model); flattening it
        # to (head*d_head, d_model) turns the projection into a single
        # (batch*pos, d) @ (d, head*d_head) matmul. Broadcasting x against W
        # instead (x[:, None] @ W.transpose(0, 2, 1)) dispatches batch*head
        # tiny gemms and is ~15x slower on Metal.
        return self._heads(x @ W.reshape(-1, x.shape[-1]).T)

    def __call__(self, x, cache=None, prefix="", x_q=None):
        # x: (batch, pos, d_model). k/q/v: (batch, head, pos, d_head)
        # If x_q is given, queries come only from those positions (assumed to
        # be a suffix of x, e.g. x[:, -1:]) and no causal mask is needed.
        k = _stash(cache, prefix + "k", self._project(x, self.W_K))
        q = self._project(x if x_q is None else x_q, self.W_Q)
        v = _stash(cache, prefix + "v", self._project(x, self.W_V))
        k_t = k.transpose(0, 1, 3, 2)
        scores = q @ k_t / math.sqrt(self.d_head)
        if x_q is None:
            _stash(cache, prefix + "q", q)
            n = x.shape[-2]
            scores = mx.where(self._mask[:n, :n] == 1, scores, -1e10)
        # attn: (batch, head, query_pos, key_pos)
        attn = _stash(cache, prefix + "attn", mx.softmax(scores, axis=-1))
        z = _stash(cache, prefix + "z", attn @ v)
        z_flat = z.transpose(0, 2, 1, 3).reshape(z.shape[0], z.shape[2], -1)  # b q (i h)
        return z_flat @ self.W_O.T


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, m = cfg.d_model, cfg.d_mlp
        self.W_in = mx.random.normal((m, d)) / math.sqrt(d)
        self.b_in = mx.zeros(m)
        self.W_out = mx.random.normal((d, m)) / math.sqrt(d)
        self.b_out = mx.zeros(d)
        self.act = {"ReLU": nn.relu, "GeLU": nn.gelu}[cfg.act_type]

    def __call__(self, x, cache=None, prefix=""):
        pre = _stash(cache, prefix + "pre", x @ self.W_in.T + self.b_in)
        post = _stash(cache, prefix + "post", self.act(pre))
        return post @ self.W_out.T + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def __call__(self, x, cache=None, prefix=""):
        _stash(cache, prefix + "resid_pre", x)
        x = x + _stash(cache, prefix + "attn_out", self.attn(x, cache, prefix + "attn."))
        _stash(cache, prefix + "resid_mid", x)
        x = x + _stash(cache, prefix + "mlp_out", self.mlp(x, cache, prefix + "mlp."))
        return _stash(cache, prefix + "resid_post", x)


def _embed_init(cfg: Config, d: int, v: int):
    gauss = mx.random.normal((d, v)) / math.sqrt(d)
    if cfg.embed_init == "normal":
        return gauss
    assert cfg.embed_init == "orthogonal" and d >= v
    # Semi-orthogonal W_E: QR-orthonormalize the token columns, so
    # W_E^T W_E = I_v and every unit vocab-space direction (in particular
    # every Fourier vector) has energy exactly 1 — the same as the Gaussian
    # init's *expected* column norm, but with zero per-frequency fluctuation.
    # The Gaussian is drawn from the same mx stream as 'normal' so the rest
    # of the init (W_pos, blocks, W_U) sees an unchanged seed sequence.
    q, _ = np.linalg.qr(np.array(gauss, dtype=np.float64))
    return mx.array(q.astype(np.float32))


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, v = cfg.d_model, cfg.d_vocab
        self.embed = {"W_E": _embed_init(cfg, d, v)}
        self.pos_embed = {"W_pos": mx.random.normal((cfg.n_ctx, d)) / math.sqrt(d)}
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        self.unembed = {"W_U": mx.random.normal((d, v)) / math.sqrt(v)}

    @property
    def W_E(self):
        return self.embed["W_E"]

    @property
    def W_pos(self):
        return self.pos_embed["W_pos"]

    @property
    def W_U(self):
        return self.unembed["W_U"]

    def __call__(self, tokens, cache=None):
        # tokens: (batch, pos) ints -> logits: (batch, pos, d_vocab)
        x = self.W_E.T[tokens] + self.W_pos[: tokens.shape[-1]]
        for i, block in enumerate(self.blocks):
            x = block(x, cache, prefix=f"blocks.{i}.")
        return x @ self.W_U

    def run_with_cache(self, tokens):
        cache = {}
        logits = self(tokens, cache)
        return logits, cache

    def final_logits(self, tokens):
        """Logits at the final position only: (batch, d_vocab).

        Equivalent to `self(tokens)[:, -1]` but skips the MLP, attention
        queries, and unembedding at all other positions — in the last block
        those outputs are never read. Used for training; ~2x faster at
        n_ctx=3. Exact for any depth: earlier blocks still run in full.
        """
        x = self.W_E.T[tokens] + self.W_pos[: tokens.shape[-1]]
        for block in self.blocks[:-1]:
            x = block(x)
        block = self.blocks[-1]
        xf = x[:, -1:] + block.attn(x, x_q=x[:, -1:])
        xf = xf + block.mlp(xf)
        return (xf @ self.W_U)[:, 0]
