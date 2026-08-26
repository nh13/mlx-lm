# Copyright © 2026 Apple Inc. / mlx-community
#
# DeepSeek-V4 (Pro / Flash) for mlx-lm.
# Architecture: Multi-head Latent Attention (num_kv_heads=1) + grouped low-rank output,
# sliding-window + compressed KV + indexer topk (sparse attention), hash-routed MoE
# with sqrtsoftplus scoring, Manifold-constrained Hyper-Connections (mHC) replacing
# residuals. Weights are native FP8 (e4m3) with 128x128 block scaling (ue8m0).
#
# Reference: deepseek-ai/DeepSeek-V4 (Apr 2026). mHC: arXiv:2512.24880.

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .hyper_connection import HyperConnection, HyperHead
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1

    # Attention (MLA-style with single shared KV head)
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    attention_bias: bool = False
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)

    # Compressor / Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_rope_theta: float = 160000.0

    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Hyper-Connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # MTP (multi-token prediction). `num_nextn_predict_layers` in the base config
    # is a DeepSeek-V3-loader TRAP: it is set to 1 in V4-Flash checkpoints to stop
    # V3-shaped loaders from building 3 heads, and is NOT the DSpark backbone depth.
    num_nextn_predict_layers: int = 1

    # DSpark speculative-decoding head (arXiv:2607.05147). Present only in
    # MTP-preserving checkpoints (e.g. mlx-community DeepSeek-V4-Flash-*-mtp). The
    # head is a `dspark_num_layers`-stage sliding-window V4 backbone that drafts a
    # block of `dspark_block_size` future tokens per step from the residual streams
    # tapped at `dspark_target_layer_ids`, corrected by a VanillaMarkov head. The
    # stage count is an architectural constant of the DSpark-5 variant (not otherwise
    # derivable from the base config). When `dspark_target_layer_ids` is None, no
    # DSpark head is built.
    dspark_target_layer_ids: Optional[List[int]] = None
    dspark_block_size: int = 5
    dspark_markov_rank: int = 256
    dspark_noise_token_id: int = 128799
    dspark_num_layers: int = 3

    # RoPE / YaRN
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rms_norm_eps: float = 1e-6

    # Quantization (FP8 block)
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        # Auto-fill compress_ratios with V4 defaults if not specified, and
        # validate length / values. Adapted from @eauchs c6a7828 (#1192).
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        total_layers = self.num_hidden_layers + self.n_mtp_stages
        self.compress_ratios = list(self.compress_ratios[:total_layers])
        # MTP stages default to compress_ratio=0 (plain sliding-window MLA)
        while len(self.compress_ratios) < total_layers:
            self.compress_ratios.append(0)
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")

    @property
    def n_mtp_stages(self) -> int:
        """Number of DSpark MTP head stages to build (0 when the checkpoint has no
        DSpark head — i.e. `dspark_target_layer_ids` is unset)."""
        return self.dspark_num_layers if self.dspark_target_layer_ids else 0


# --------------------------------------------------------------------------- #
# Fused partial-RoPE Metal kernel                                             #
# --------------------------------------------------------------------------- #
#
# Decode dispatch reduction: the scalar-Python rotation (slice -> reshape ->
# index x0/x1 -> 4 muls + add/sub -> stack -> reshape) issues ~5 graph ops per
# rope call, and DeepseekV4 invokes rope ~3x per attention layer (q_pe, k_pe,
# inverse on attention output) -> 129 calls/token at L=1. Collapsing the chain
# into a single Metal kernel removes ~600 dispatches/token on the decode path.
#
# Adapted from @0xClandestine's optimization PR
# (https://github.com/Blaizzy/mlx-lm/pull/13) targeting Blaizzy's V4 branch.
# We use the rope-only signature (V4Attention splits nope/rope outside the
# rope call), so the nope passthrough loop is dropped from the source.
#
# One SIMD-group per (b, h, l) work item; lane t handles the interleaved
# pair (x[2t], x[2t+1]).

def _make_partial_rope_kernel():
    # Env-var escape hatch so benchmarks can A/B kernel ON vs OFF without
    # monkey-patching: MLX_LM_DISABLE_PARTIAL_ROPE_KERNEL=1 -> falls back
    # to the pure-MLX path used pre-2026-04-25.
    import os
    if os.environ.get("MLX_LM_DISABLE_PARTIAL_ROPE_KERNEL", "0") == "1":
        return None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        constexpr int DRH = D_ROPE / 2;
        int L_v = dims[0];
        int H_v = dims[1];
        uint l   = gid % (uint)L_v;
        uint tmp = gid / (uint)L_v;
        uint h   = tmp % (uint)H_v;
        uint b   = tmp / (uint)H_v;

        const auto xp = x  + ((uint64_t)b * H_v * L_v + h * L_v + l) * D_ROPE;
        auto       yp = y  + ((uint64_t)b * H_v * L_v + h * L_v + l) * D_ROPE;
        const auto cp = cos_s + l * DRH;
        const auto sp = sin_s + l * DRH;

        // Lane t handles one interleaved pair (x[2t], x[2t+1]).
        if ((int)tid < DRH) {
            float x0 = float(xp[2 * tid]);
            float x1 = float(xp[2 * tid + 1]);
            float c  = float(cp[tid]);
            float s  = float(sp[tid]);
            if (INVERSE) {
                store_elem(yp[2 * tid],     fma( x1, s, x0 * c));   //  x0*c + x1*s
                store_elem(yp[2 * tid + 1], fma(-x0, s, x1 * c));   // -x0*s + x1*c
            } else {
                store_elem(yp[2 * tid],     fma(-x1, s, x0 * c));   //  x0*c - x1*s
                store_elem(yp[2 * tid + 1], fma( x0, s, x1 * c));   //  x0*s + x1*c
            }
        }
    """
    return mx.fast.metal_kernel(
        name="ds4_partial_rope",
        input_names=["x", "cos_s", "sin_s", "dims"],
        output_names=["y"],
        header="template<typename T> inline void store_elem(device T& dst, float v) { dst = T(v); }",
        source=source,
    )


_partial_rope_kernel = _make_partial_rope_kernel()


class DeepseekV4RoPE(nn.Module):
    """DeepSeek-V4 rotary embedding.

    The reference implementation applies RoPE to the KV tensor before attention
    and applies the conjugate rotation to the attention output. The generic MLX
    RoPE layers do not expose an inverse path, so keep the small DeepSeek-specific
    implementation here.
    """

    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.dims = dims

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings
                        / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = math.floor(correction_dim(beta_fast))
            high = math.ceil(correction_dim(beta_slow))
            low = max(low, 0)
            high = min(high, dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default", "linear"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type {rope_type}")

        # This is derived from config, not a checkpoint parameter.
        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(self, x: mx.array, offset: int = 0, inverse: bool = False):
        dtype = x.dtype
        T = x.shape[-2]
        if isinstance(offset, mx.array):
            if offset.size == 1:
                offset = offset.item()
            else:
                B = offset.shape[0]
                pos = offset[:, None] + mx.arange(T, dtype=mx.float32)[None, :]
                theta = pos[..., None] * self.inv_freq[None, None, :]
                if inverse:
                    theta = -theta
                # theta: [B, T, dims//2]. Reshape for x dims: [B,H,T,D] or [B,1,T,D]
                target_shape = (B,) + (1,) * (x.ndim - 3) + (T, self.dims // 2)
                cos = mx.cos(theta).reshape(target_shape).astype(dtype)
                sin = mx.sin(theta).reshape(target_shape).astype(dtype)
                rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
                x0 = rot[..., 0]
                x1 = rot[..., 1]
                r0 = x0 * cos - x1 * sin
                r1 = x0 * sin + x1 * cos
                rotated = mx.stack([r0, r1], axis=-1).reshape(*x.shape[:-1], self.dims)
                if self.dims < x.shape[-1]:
                    return mx.concatenate([rotated, x[..., self.dims:]], axis=-1)
                return rotated
        # Fast path: fused Metal kernel for the rope-only 4D case used by
        # V4Attention. Falls through to the pure-MLX path on CPU, on Mode-B
        # (x has a nope tail), or on non-4D inputs (e.g. Indexer rope).
        # The kernel itself handles inverse via formula sign-flip; theta is
        # always forward-direction (do NOT negate it here).
        if (
            _partial_rope_kernel is not None
            and x.shape[-1] == self.dims
            and x.ndim == 4
        ):
            B, H, L, _ = x.shape
            pos = mx.arange(offset, offset + T, dtype=mx.float32)
            theta = pos[:, None] * self.inv_freq[None, :]
            cos = mx.cos(theta).astype(mx.float32)
            sin = mx.sin(theta).astype(mx.float32)
            dims_arr = mx.array([L, H], dtype=mx.int32)
            return _partial_rope_kernel(
                inputs=[x, cos, sin, dims_arr],
                template=[("D_ROPE", self.dims), ("INVERSE", 1 if inverse else 0)],
                grid=(B * H * L * 32, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]

        pos = mx.arange(offset, offset + T, dtype=mx.float32)
        theta = pos[:, None] * self.inv_freq[None, :]
        if inverse:
            theta = -theta

        broadcast_shape = (1,) * (x.ndim - 2) + theta.shape
        cos = mx.cos(theta).reshape(broadcast_shape).astype(dtype)
        sin = mx.sin(theta).reshape(broadcast_shape).astype(dtype)

        rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
        x0 = rot[..., 0]
        x1 = rot[..., 1]
        y = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
        y = y.reshape(*x.shape[:-1], self.dims)
        if x.shape[-1] == self.dims:
            return y
        return mx.concatenate([y, x[..., self.dims :]], axis=-1)


# --------------------------------------------------------------------------- #
# Gate (hash + score-based)                                                   #
# --------------------------------------------------------------------------- #

# Pre-allocated scalar zero for sqrtsoftplus: avoids mx.zeros_like() allocation per call.
_SCORE_ZERO = mx.array(0.0)


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    # sqrtsoftplus: sqrt(softplus(x))  — used by V4
    # Scalar broadcast avoids allocating a zeros tensor every call.
    return mx.sqrt(mx.logaddexp(scores, _SCORE_ZERO))


class MoEGate(nn.Module):
    """Routing gate. First `num_hash_layers` layers use a deterministic hash
    (token-id -> expert-id table) instead of learned score-based topk. Remaining
    layers run sqrtsoftplus scoring + e_score_correction_bias + topk, with
    post-softmax renormalization if score_func != 'softmax'."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_routed = args.n_routed_experts
        self.top_k = args.num_experts_per_tok
        self.hash = layer_idx < args.num_hash_layers
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob

        self.weight = mx.zeros((self.n_routed, args.hidden_size))
        # Cache transposed weight to avoid recomputing .T every forward call.
        self._weight_t = None
        if self.hash:
            # tid2eid: [vocab, top_k] int32 — predetermined expert routing per token id
            self.tid2eid = mx.zeros((args.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros((self.n_routed,), dtype=mx.float32)

    @property
    def weight_t(self):
        if self._weight_t is None:
            self._weight_t = self.weight.T
        return self._weight_t

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        # x: [B, S, D] or [N, D]
        if self.hash:
            # x shape -> [B*S, D]; input_ids -> [B, S] flattened to [B*S]
            flat = x.reshape(-1, x.shape[-1])
            scores = flat.astype(mx.float32) @ self.weight_t.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            ids = input_ids.reshape(-1)
            inds = self.tid2eid[ids].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            # Reshape inds/weights back to match x's leading dims so SwitchGLU
            # can broadcast against x: [B, S, top_k] (mirrors non-hash branch).
            inds = inds.reshape(*x.shape[:-1], self.top_k)
            weights = weights.reshape(*x.shape[:-1], self.top_k)
        else:
            scores = x.astype(mx.float32) @ self.weight_t.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            orig = scores
            biased = scores + self.e_score_correction_bias
            inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
            weights = mx.take_along_axis(orig, inds, axis=-1)

        if self.score_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.route_scale
        return inds, weights


# --------------------------------------------------------------------------- #
# MoE                                                                          #
# --------------------------------------------------------------------------- #

def _swiglu_limited(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return nn.silu(gate) * up


class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_swiglu_limited(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )
        self.gate = MoEGate(args, layer_idx)
        if args.n_shared_experts:
            self.shared_experts = DeepseekV4MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.n_shared_experts,
                swiglu_limit=0.0,
            )
        self.sharding_group = None
        self._combine = None   # lazily-compiled expert-combine (see __call__)

    def _experts(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:
        """Routed + shared expert combine — the graph-encode-heavy part of the MoE,
        independent of the gate. Compiled in __call__; bit-identical to the inline
        form. shared_experts runs before switch_mlp so MLX can overlap both (it does
        not depend on routing)."""
        shared_y = self.shared_experts(x) if hasattr(self, "shared_experts") else None
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        return y + shared_y if shared_y is not None else y

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, weights = self.gate(x, input_ids)
        # Compile the expert-combine on first use (batch=1 decode is host-encode-bound;
        # compiling roughly halves the host ops for this block, bit-identical — verified
        # maxdiff=0). Lazy so module init / weight load see the eager graph.
        if self._combine is None:
            self._combine = mx.compile(self._experts)
        y = self._combine(x, inds, weights)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


def _rope_at(rope, x_pe: mx.array, positions: mx.array) -> mx.array:
    """Interleaved-pair RoPE on x_pe[..., S, dims] at explicit `positions` [S]."""
    dims = rope.dims
    theta = positions[:, None] * rope.inv_freq[None, :]
    cos = mx.cos(theta).astype(x_pe.dtype)
    sin = mx.sin(theta).astype(x_pe.dtype)
    shape = (1,) * (x_pe.ndim - 2) + cos.shape
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    rot = x_pe[..., :dims].reshape(*x_pe.shape[:-1], dims // 2, 2)
    x0 = rot[..., 0]
    x1 = rot[..., 1]
    y = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
    return y.reshape(x_pe.shape)


def _rope_pool(rope, pool: mx.array, positions: mx.array, rope_head_dim: int) -> mx.array:
    """RoPE the last `rope_head_dim` dims of compressed pool rows [..., n, head_dim]
    at absolute `positions` [n]. Rows are roped ONCE at emit time (row j at its true
    absolute position j*ratio) so a later top-k gather carries each selected row's
    correct positional signal — matching the HF DeepSeek-V4 reference, which ropes
    compressed rows at emit and never repositions them. (The old code roped after the
    gather at arange(n)*ratio, mis-positioning the unordered selected rows.)"""
    nope = pool.shape[-1] - rope_head_dim
    p_nope, p_pe = mx.split(pool, [nope], axis=-1)
    p_pe = _rope_at(rope, p_pe, positions)
    return mx.concatenate([p_nope, p_pe], axis=-1)


@lru_cache(maxsize=32)
def _compressed_col_mask(
    S_q: int, offset: int, n_comp: int, r: int, dtype: mx.Dtype
) -> mx.array:
    """Contiguous compressed-pool mask columns, [S_q, n_comp]. Block j summarizes
    raw positions [j*r, (j+1)*r), so a query at absolute position offset+i attends it
    iff offset+i >= (j+1)*r-1. Memoized (depends only on these scalars) so a
    multi-token forward builds one mask per (offset, compress-ratio) rather than
    rebuilding it in every compressed layer. bool dtype -> bool; else additive 0/-inf."""
    comp_end = mx.arange(n_comp) * r + (r - 1)
    keep = (offset + mx.arange(S_q))[:, None] >= comp_end[None, :]   # [S_q, n_comp]
    if dtype == mx.bool_:
        return keep
    return mx.where(keep, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


def _compressed_mask(
    S_q: int, offset: int, n_comp: int, r: int, dtype: mx.Dtype,
    block_ids: Optional[mx.array],
) -> mx.array:
    """Compressed-pool mask columns [S_q, n_comp]. Query row i sits at absolute
    position offset+i and attends compressed column g (covering raw block b, ending at
    b*r+r-1) iff offset+i >= b*r+r-1, where b = block_ids[g] for a gathered (top-k)
    pool or g for a contiguous pool. The contiguous case is memoized; the gathered case
    is data-dependent (unordered top-k ids) so it is built per call. block_ids is 1-D
    (batch 1 — the single-sequence generation/DSpark path)."""
    if block_ids is None:
        return _compressed_col_mask(S_q, offset, n_comp, r, dtype)
    comp_end = block_ids.astype(mx.int32) * r + (r - 1)             # [n_comp]
    keep = (offset + mx.arange(S_q))[:, None] >= comp_end[None, :]  # [S_q, n_comp]
    if dtype == mx.bool_:
        return keep
    return mx.where(keep, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


# --------------------------------------------------------------------------- #
# Attention: MLA (num_kv_heads=1) + sliding window + optional compressed KV   #
# --------------------------------------------------------------------------- #

class CompressedKVCache(KVCache):
    """Cache for compressed-attention layers: sliding-window local cache + compressed KV pool.

    During prefill, the compressor produces all compressed rows at once.
    During decode, tokens accumulate in a buffer; every `ratio` tokens the
    buffer is compressed and the result is appended to the pool.

    Inherits from KVCache so external engines (vllm-mlx) recognize it via
    isinstance checks. All state is proxied through self.local (RotatingKVCache).
    """

    def __init__(self, max_size: int = 128):
        # Skip KVCache.__init__ — we proxy everything through self.local
        self.local = RotatingKVCache(max_size=max_size, keep=0)
        self._pool = None
        self._buf = None
        self._buf_count = 0
        self._abs_pos = 0
        self._index_pool = None
        self._index_buf = None
        self._index_abs_pos = 0
        # Speculative single-forward rollback state (see spec_begin/spec_trim).
        self._spec_armed = False
        self._spec_snap = None
        self._spec_raw = None
        self._spec_compressor = None
        self._spec_index_compressor = None
        self._spec_rope = None

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def values(self):
        return self.local.values

    @values.setter
    def values(self, value):
        self.local.values = value

    @property
    def pool(self):
        return self._pool

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    @property
    def state(self):
        ls = self.local.state
        ls = ls if isinstance(ls, tuple) else (ls,)
        empty = mx.array([], dtype=mx.bfloat16)
        extra = tuple(
            a if a is not None else empty
            for a in (self._pool, self._buf, self._index_pool, self._index_buf)
        )
        return (*ls, *extra)

    @state.setter
    def state(self, value):
        *ls, pool, buf, ipool, ibuf = value
        self.local.state = tuple(ls) if len(ls) != 1 else ls[0]
        self._pool = pool if pool.size > 0 else None
        self._buf = buf if buf.size > 0 else None
        self._index_pool = ipool if ipool.size > 0 else None
        self._index_buf = ibuf if ibuf.size > 0 else None
        self._buf_count = 0 if self._buf is None else self._buf.shape[1]

    @property
    def nbytes(self):
        n = self.local.nbytes
        if self._pool is not None:
            n += self._pool.nbytes
        if self._buf is not None:
            n += self._buf.nbytes
        if self._index_pool is not None:
            n += self._index_pool.nbytes
        if self._index_buf is not None:
            n += self._index_buf.nbytes
        return n

    @property
    def meta_state(self):
        return (
            self.local.meta_state,
            str(self._buf_count),
            str(self._abs_pos),
            str(self._index_abs_pos),
        )

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value[0]
        self._buf_count = int(value[1])
        self._abs_pos = int(value[2])
        self._index_abs_pos = int(value[3])

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        *ls, pool, buf, ipool, ibuf = state
        obj.local = RotatingKVCache.from_state(
            tuple(ls) if len(ls) != 1 else ls[0], meta_state[0]
        )
        obj._pool = pool if pool.size > 0 else None
        obj._buf = buf if buf.size > 0 else None
        obj._index_pool = ipool if ipool.size > 0 else None
        obj._index_buf = ibuf if ibuf.size > 0 else None
        obj._buf_count = int(meta_state[1])
        obj._abs_pos = int(meta_state[2])
        obj._index_abs_pos = int(meta_state[3])
        return obj

    def is_trimmable(self):
        return self.local.is_trimmable()

    def trim(self, n):
        return self.local.trim(n)

    def spec_snapshot(self):
        """Snapshot the compressed-pool state for speculative-verify rollback. The pool
        fields are immutable arrays (+ ints) rebuilt via concat, so this is a cheap
        reference save; the local sliding KV is trimmed back separately (it is
        trimmable). Restore with `spec_restore`."""
        return (self._pool, self._buf, self._buf_count, self._abs_pos,
                self._index_pool, self._index_buf, self._index_abs_pos)

    def spec_restore(self, snap):
        (self._pool, self._buf, self._buf_count, self._abs_pos,
         self._index_pool, self._index_buf, self._index_abs_pos) = snap

    def spec_begin(self):
        """Arm speculative capture for a verify forward. Snapshots the pre-verify
        pool state and starts recording the raw `accumulate` input so `spec_trim`
        can rebuild the pool for any accepted prefix in a single forward — no clean
        re-advance pass needed. Idempotent per verify; cleared by `spec_trim`."""
        self._spec_snap = self.spec_snapshot()
        self._spec_raw = None
        self._spec_compressor = None
        self._spec_index_compressor = None
        self._spec_rope = None
        self._spec_armed = True

    def spec_trim(self, keep: int):
        """Roll the compressed pool + buffers back to the first `keep` positions of
        the just-run verify forward, bit-exactly. Replays the tested
        `_accumulate_window` over the captured raw prefix from the pre-verify
        snapshot: the emitted rows for the accepted prefix are a prefix of (and
        identical to) the full verify rows, and the retained raw buffer is
        reconstructed exactly — including tokens the full forward already dropped at
        a compression boundary. The local sliding KV is trimmed by the caller."""
        if not self._spec_armed:
            return
        (pool0, buf0, _bc0, abs0, ipool0, ibuf0, iabs0) = self._spec_snap
        x = self._spec_raw
        if x is not None and self._spec_compressor is not None:
            xk = x[:, :keep]
            self._pool, self._buf, self._abs_pos = self._accumulate_window(
                pool0, buf0, abs0, xk, self._spec_compressor, self._spec_rope)
            self._buf_count = 0 if self._buf is None else self._buf.shape[1]
            if self._spec_index_compressor is not None:
                self._index_pool, self._index_buf, self._index_abs_pos = (
                    self._accumulate_window(
                        ipool0, ibuf0, iabs0, xk, self._spec_index_compressor))
        else:
            # No rows accumulated during verify — restore the pre-verify snapshot.
            self.spec_restore(self._spec_snap)
        self._spec_armed = False
        self._spec_raw = None

    @classmethod
    def merge(cls, caches):
        """Merge multiple CompressedKVCaches into a single batched cache."""
        merged = cls.__new__(cls)

        # Merge local rotating caches (delegates to BatchRotatingKVCache)
        merged.local = caches[0].local.merge([c.local for c in caches])

        # Merge compressed pools: pad to max length, stack along B
        pools = [c._pool for c in caches]
        if all(p is None for p in pools):
            merged._pool = None
        else:
            head_dim = next(p.shape[-1] for p in pools if p is not None)
            dtype = next(p.dtype for p in pools if p is not None)
            max_len = max(p.shape[1] if p is not None else 0 for p in pools)
            padded = []
            for p in pools:
                if p is None:
                    padded.append(mx.zeros((1, max_len, head_dim), dtype=dtype))
                elif p.shape[1] < max_len:
                    pad = mx.zeros((1, max_len - p.shape[1], head_dim), dtype=dtype)
                    padded.append(mx.concatenate([p, pad], axis=1))
                else:
                    padded.append(p)
            merged._pool = mx.concatenate(padded, axis=0)

        # Merge buffers: pad to max buf_count, stack along B
        bufs = [c._buf for c in caches]
        buf_counts = [c._buf_count for c in caches]
        if all(b is None for b in bufs):
            merged._buf = None
            merged._buf_count = 0
        else:
            D = next(b.shape[-1] for b in bufs if b is not None)
            dtype = next(b.dtype for b in bufs if b is not None)
            max_bc = max(buf_counts)
            padded = []
            for b, bc in zip(bufs, buf_counts):
                if b is None:
                    padded.append(mx.zeros((1, max_bc, D), dtype=dtype))
                elif b.shape[1] < max_bc:
                    pad = mx.zeros((1, max_bc - b.shape[1], D), dtype=dtype)
                    padded.append(mx.concatenate([b, pad], axis=1))
                else:
                    padded.append(b)
            merged._buf = mx.concatenate(padded, axis=0)
            merged._buf_count = max_bc

        merged._index_pool = cls._merge_field([c._index_pool for c in caches])
        merged._index_buf = cls._merge_field([c._index_buf for c in caches])
        merged._abs_pos = max(c._abs_pos for c in caches)
        merged._index_abs_pos = max(c._index_abs_pos for c in caches)

        return merged

    def filter(self, batch_indices):
        if hasattr(self.local, "filter"):
            self.local.filter(batch_indices)
        for attr in ("_pool", "_buf", "_index_pool", "_index_buf"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v[batch_indices])

    def extend(self, other):
        if hasattr(self.local, "extend"):
            self.local.extend(other.local)
        self._pool = self._extend_field(self._pool, other._pool)
        self._buf = self._extend_field(self._buf, other._buf)
        self._buf_count = max(self._buf_count, other._buf_count)
        self._index_pool = self._extend_field(self._index_pool, other._index_pool)
        self._index_buf = self._extend_field(self._index_buf, other._index_buf)
        self._abs_pos = max(self._abs_pos, other._abs_pos)
        self._index_abs_pos = max(self._index_abs_pos, other._index_abs_pos)

    def finalize(self):
        if hasattr(self.local, "finalize"):
            self.local.finalize()

    def extract(self, idx):
        extracted = CompressedKVCache.__new__(CompressedKVCache)
        extracted.local = self.local.extract(idx) if hasattr(self.local, "extract") else self.local
        def sl(a):
            return a[idx : idx + 1] if a is not None else None

        extracted._pool = sl(self._pool)
        extracted._buf = sl(self._buf)
        extracted._index_pool = sl(self._index_pool)
        extracted._index_buf = sl(self._index_buf)
        extracted._buf_count = self._buf_count
        extracted._abs_pos = self._abs_pos
        extracted._index_abs_pos = self._index_abs_pos
        return extracted

    @property
    def batch_size(self):
        if hasattr(self.local, 'batch_size'):
            return self.local.batch_size
        return 1

    @staticmethod
    def _merge_field(arrays):
        # Pad batch-1 [1, n, D] arrays to max length and stack along batch.
        if all(a is None for a in arrays):
            return None
        present = next(a for a in arrays if a is not None)
        D, dtype = present.shape[-1], present.dtype
        max_len = max(a.shape[1] if a is not None else 0 for a in arrays)
        out = []
        for a in arrays:
            if a is None:
                out.append(mx.zeros((1, max_len, D), dtype=dtype))
            elif a.shape[1] < max_len:
                pad = mx.zeros((a.shape[0], max_len - a.shape[1], D), dtype=dtype)
                out.append(mx.concatenate([a, pad], axis=1))
            else:
                out.append(a)
        return mx.concatenate(out, axis=0)

    @staticmethod
    def _extend_field(a, b):
        # Concatenate two [B, n, D] fields along batch, padding to max length.
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        max_len = max(a.shape[1], b.shape[1])
        def pad(x):
            if x.shape[1] < max_len:
                p = mx.zeros((x.shape[0], max_len - x.shape[1], x.shape[2]), dtype=x.dtype)
                return mx.concatenate([x, p], axis=1)
            return x
        return mx.concatenate([pad(a), pad(b)], axis=0)

    @staticmethod
    def _accumulate_window(pool, buf, abs_pos, x, compressor, rope=None):
        """Emit compressed rows for completed windows; return (pool, buf, abs_pos).

        Shared by the main compressed KV pool and the indexer pool. Each row is
        produced by calling `compressor` over exactly the raw-token window a
        single full-sequence forward would see (retaining the previous window
        for ratio-4 overlap), so incremental decode is bit-for-bit consistent
        with prefill, including prefill remainders and chunk boundaries.

        When `rope` is given (main pool only — the index pool stays unroped, since
        the indexer does not rope its queries), each newly emitted row is RoPE'd at
        its true absolute position (row j -> j*r) before it enters the pool, so a
        later top-k gather carries the correct positional signal. Deterministic in
        the row index, so `spec_trim`'s replay reproduces the pool bit-exactly.
        """
        r = compressor.ratio
        overlap = compressor.overlap
        buf = x if buf is None else mx.concatenate([buf, x], axis=1)
        abs_pos += x.shape[1]
        raw_start = abs_pos - buf.shape[1]
        n_rows = 0 if pool is None else pool.shape[1]
        n_before = n_rows
        while (n_rows + 1) * r <= abs_pos:
            w = n_rows
            need_start = ((w - 1) * r) if (overlap and w > 0) else (w * r)
            s = need_start - raw_start
            e = (w + 1) * r - raw_start
            rows = compressor(buf[:, s:e])
            row = rows[:, -1:]
            pool = row if pool is None else mx.concatenate([pool, row], axis=1)
            n_rows += 1
        if rope is not None and n_rows > n_before:
            # RoPE just-emitted rows [n_before, n_rows) at positions [n_before*r, ...).
            pos = mx.arange(n_before, n_rows, dtype=mx.float32) * r
            roped = _rope_pool(rope, pool[:, n_before:], pos, compressor.rope_head_dim)
            pool = roped if n_before == 0 else mx.concatenate([pool[:, :n_before], roped], axis=1)
        keep_abs = max(((n_rows - 1) * r) if overlap else (n_rows * r), 0)
        drop = keep_abs - raw_start
        if drop > 0:
            buf = buf[:, drop:]
        return pool, buf, abs_pos

    def accumulate(
        self, x: mx.array, compressor: "Compressor", rope=None
    ) -> Optional[mx.array]:
        """Main compressed-KV pool: buffer raw tokens, emit rows on boundaries. Pass
        `rope` (the compress RoPE) to pre-rope emitted rows at their true positions."""
        # getattr guard: caches rebuilt via __new__ (from_state/merge/extract) skip
        # __init__, so the spec fields may be absent; they are only armed by spec_begin.
        if getattr(self, "_spec_armed", False):
            self._spec_raw = (
                x if self._spec_raw is None
                else mx.concatenate([self._spec_raw, x], axis=1)
            )
            self._spec_compressor = compressor
            self._spec_rope = rope
        self._pool, self._buf, self._abs_pos = self._accumulate_window(
            self._pool, self._buf, self._abs_pos, x, compressor, rope
        )
        self._buf_count = 0 if self._buf is None else self._buf.shape[1]
        return self._pool

    def accumulate_index(self, x: mx.array, compressor: "Compressor") -> Optional[mx.array]:
        """Indexer pool: same windowing as accumulate, separate compressor/state.

        Lets the lightweight index compressor accumulate across decode steps so
        top-k block selection works during generation (it previously recomputed
        on the single decode token, produced no rows, and disabled top-k).
        """
        if getattr(self, "_spec_armed", False):
            # Same raw `x` as accumulate(); only the index compressor differs.
            self._spec_index_compressor = compressor
        self._index_pool, self._index_buf, self._index_abs_pos = self._accumulate_window(
            self._index_pool, self._index_buf, self._index_abs_pos, x, compressor
        )
        return self._index_pool


class Compressor(nn.Module):
    """Learned gated pooling over `ratio` consecutive tokens for KV compression.

    At prefill, produces ~ seq/ratio compressed KV rows. At decode, accumulates
    tokens in a state buffer and emits a compressed row every `ratio` steps.
    Pure-MLX; a fused Metal kernel may replace this in a follow-up.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.ratio = compress_ratio
        self.overlap = compress_ratio == 4
        out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(self.dim, out_dim, bias=False)
        self.wgate = nn.Linear(self.dim, out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, out_dim), dtype=mx.float32)
        self.norm  = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

    def _overlap_transform(self, tensor: mx.array, value: float) -> mx.array:
        B, S, R, _ = tensor.shape
        D = self.head_dim
        out = mx.full((B, S, 2 * R, D), value, dtype=tensor.dtype)
        out[:, :, R:] = tensor[:, :, :, D:]
        out[:, 1:, :R] = tensor[:, :-1, :, :D]
        return out

    def __call__(self, x: mx.array) -> mx.array:
        # Prefill-only MVP: chunk x into windows of `ratio` tokens. Ratio-4
        # layers use the overlapping layout from the reference implementation.
        # Returns compressed KV: [B, S//ratio, head_dim] (bf16).
        B, S, _ = x.shape
        r = self.ratio
        keep = (S // r) * r
        if keep == 0:
            return mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        xf = x[:, :keep].astype(mx.float32)
        kv = self.wkv(xf).reshape(B, keep // r, r, -1)
        score = self.wgate(xf).reshape(B, keep // r, r, -1) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))
        weights = mx.softmax(score, axis=2, precise=True)
        kv = (kv * weights).sum(axis=2)
        return self.norm(kv.astype(x.dtype))


class V4Attention(nn.Module):
    """V4 attention block.

    Checkpoint shapes (Flash):
        n_heads=64, head_dim=512, rope_head_dim=64 (nope=448)
        q_lora_rank=1024,  wq_a: [dim, 1024], wq_b: [1024, n_heads*head_dim]
        wkv: [dim, head_dim]            (single shared K=V head, MQA-style)
        attn_sink: [n_heads] fp32
        wo_a: [n_heads*head_dim/n_groups, n_groups*o_lora_rank]
        wo_b: [n_groups*o_lora_rank, dim]
        For compress_ratio != 0: compressor.wkv/wgate/ape/norm; and if ratio==4, indexer.*

    Forward path (MVP):
        - Project Q (64 heads), K=V (1 head); apply RoPE to last `rope_head_dim` dims.
        - For ratio=0 layers: sliding window mask of size `sliding_window`.
        - For ratio!=0 layers: append compressed KV rows to attend to (no topk filtering
          yet — full compressed cache). Use attn_sink via SDPA `sinks=` argument.
        - Grouped low-rank output projection: wo_a per group -> concat -> wo_b.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.window = args.sliding_window
        self.eps = args.rms_norm_eps

        ratios = args.compress_ratios or []
        self.compress_ratio = ratios[layer_idx] if layer_idx < len(ratios) else 0

        self.scale = self.head_dim ** -0.5

        # q path
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # kv path (single shared head)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        # attention sink (per-head learnable bias added in softmax denominator)
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # grouped low-rank output projection
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(group_feat, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=args.attention_bias)

        # RoPE: two separate instances, selected per layer type in __call__.
        # Sliding-window ("main") layers use PLAIN RoPE at rope_theta (10000) with
        # NO YaRN scaling; compressed (CSA/HCA) layers use YaRN RoPE at
        # compress_rope_theta (160000). This matches the DeepSeek-V4 reference
        # (HF DeepseekV4Config.__post_init__: main={"rope_type": "default"},
        # compress={**yarn}) — "pure sliding-window layers use plain RoPE ... YaRN
        # applies only to compressor layers." Passing rope_scaling to the main rope
        # wrongly applied factor-16 frequency interpolation on sliding layers.
        # Separate instances also avoid tying the main rotation to the wrong base
        # on compressed layers (cf. #1192 CJK token-drop report, @Blaizzy@b78ccb1).
        self.rope = DeepseekV4RoPE(self.rope_head_dim, args.rope_theta, scaling_config=None)
        self.compress_rope = DeepseekV4RoPE(
            self.rope_head_dim, args.compress_rope_theta, args.rope_scaling,
        )

        # Compressor / Indexer — present only when compress_ratio > 0
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        B, S = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        out = out.reshape(B, S, self.n_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            # Batched grouped quantized matmul: collapse the per-group Python
            # loop (8 dispatches) into a single mx.quantized_matmul call by
            # treating the group dim as a broadcast batch dim. Adapted from
            # @Blaizzy's pc/add-deepseekv4flash-model branch.
            #
            # Shapes:
            #   out (after transpose): [G, B, S, group_feat]
            #   weight (reshaped):     [G, 1, o_lora_rank, group_feat / pack_factor]
            #   scales:                [G, 1, o_lora_rank, group_feat / group_size]
            # Single dispatch returns [G, B, S, o_lora_rank], then transpose
            # back to [B, S, G, o_lora_rank] -> [B, S, G * o_lora_rank].
            out_g = out.transpose(2, 0, 1, 3)
            weight = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            scales = self.wo_a.scales.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            )
            out_g = mx.quantized_matmul(
                out_g,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=self.wo_a.group_size,
                bits=self.wo_a.bits,
                mode=self.wo_a.mode,
            )
            out = out_g.transpose(1, 2, 0, 3).reshape(B, S, self.n_groups * self.o_lora_rank)
            if "bias" in self.wo_a:
                out = out + self.wo_a.bias
            return out

        wa = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, group_feat)
        out = mx.einsum("bsgd,grd->bsgr", out, wa)
        out = out.reshape(B, S, self.n_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, S, _ = x.shape

        # --- Q (shared intermediate reused by indexer) ---
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rms_norm(q, weight=None, eps=self.eps)

        # --- K = V (shared single-head) ---
        kv = self.kv_norm(self.wkv(x))
        kv = kv.reshape(B, 1, S, self.head_dim)

        offset = cache.offset if cache is not None else 0

        # Apply RoPE only to the last rope_head_dim dims
        q_nope, q_pe = mx.split(q,  [self.nope_head_dim], axis=-1)
        k_nope, k_pe = mx.split(kv, [self.nope_head_dim], axis=-1)
        attn_rope = self.compress_rope if self.compress_ratio else self.rope
        q_pe = attn_rope(q_pe, offset=offset)
        k_pe = attn_rope(k_pe, offset=offset)
        q = mx.concatenate([q_nope, q_pe], axis=-1)
        k = v = mx.concatenate([k_nope, k_pe], axis=-1)

        # --- Compressed sparse attention ---
        # Pool rows are RoPE'd at their true absolute positions when emitted (in the
        # cache) or below (cacheless prefill), so the top-k gather carries each
        # selected row's correct positional signal — matching the HF reference.
        rope_head_dim = self.head_dim - self.nope_head_dim
        compressed_k = compressed_v = None
        gathered_ids = None
        if self.compress_ratio:
            comp_cache = cache if isinstance(cache, CompressedKVCache) else None
            if comp_cache is not None:
                pool = comp_cache.accumulate(x, self.compressor, self.compress_rope)
            elif S > 1:
                pool = self.compressor(x)
                if pool.shape[1] > 0:
                    pos = mx.arange(pool.shape[1], dtype=mx.float32) * self.compress_ratio
                    pool = _rope_pool(self.compress_rope, pool, pos, rope_head_dim)
                else:
                    pool = None
            else:
                pool = None

            # Accumulate the indexer pool in lockstep so top-k block
            # selection works during decode (S==1), not just prefill.
            index_pool = None
            if hasattr(self, "indexer"):
                if comp_cache is not None:
                    index_pool = comp_cache.accumulate_index(x, self.indexer.compressor)
                elif S > 1:
                    index_pool = self.indexer.compressor(x)
                    index_pool = index_pool if index_pool.shape[1] > 0 else None

            if pool is not None:
                ckv = pool
                if hasattr(self, "indexer") and ckv.shape[1] > self.args.index_topk:
                    topk_idx = self.indexer(x, qr, index_pool)
                    if topk_idx is not None:
                        gathered_ids = topk_idx                        # [B, topk]
                        idx = mx.broadcast_to(
                            topk_idx[:, :, None],
                            (B, topk_idx.shape[1], self.head_dim),
                        )
                        ckv = mx.take_along_axis(ckv, idx, axis=1)
                compressed_k = ckv[:, None, :, :]                      # rows already roped
                compressed_v = compressed_k

        # Update KV cache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Prepend compressed KV to cached KV for sparse attention
        if compressed_k is not None:
            k = mx.concatenate([compressed_k, k], axis=2)
            v = mx.concatenate([compressed_v, v], axis=2)
            n_comp = compressed_k.shape[2]
            if mask is not None:
                # Compressed pool columns are ATTENDED (bool True) and CAUSAL: query
                # row i (absolute position offset+i) attends compressed block b iff
                # offset+i >= b*r+r-1, where b is the true block id (gathered_ids after
                # top-k, else the contiguous index). Decode (S==1) has mask=None. The
                # offset term (previously omitted) is required for any forward at
                # offset>0 past the sliding window — chunked prefill and DSpark verify.
                S_q = mask.shape[-2]
                block_ids = gathered_ids[0] if gathered_ids is not None else None
                comp_mask = _compressed_mask(
                    S_q, offset, n_comp, self.compress_ratio, mask.dtype, block_ids
                )
                comp_mask = mx.broadcast_to(
                    comp_mask, list(mask.shape[:-2]) + [S_q, n_comp]
                )
                mask = mx.concatenate([comp_mask, mask], axis=-1)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )

        out_nope, out_pe = mx.split(out, [self.nope_head_dim], axis=-1)
        out_pe = attn_rope(out_pe, offset=offset, inverse=True)
        out = mx.concatenate([out_nope, out_pe], axis=-1)

        # Grouped low-rank projection: [B, n_heads, S, head_dim] -> [B, S, n_heads*head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)


class Indexer(nn.Module):
    """Top-k selector over compressed KV rows for ratio-4 sparse attention.

    Two-pass design: this module uses a lightweight compressor (index_head_dim,
    typically 128) to score all compressed rows cheaply, then returns topk
    indices used to gather from the main attention compressor's output
    (head_dim, typically 512). This reduces per-layer attention from O(S/4)
    to O(topk) compressed rows — 500x at 1M context with topk=512.

    Checkpoint params:
        wq_b: [q_lora_rank, n_heads * index_head_dim]
        weights_proj: [hidden_size, n_heads]
        compressor.{wkv, wgate, ape, norm}
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.scale = args.index_head_dim ** -0.5
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.compressor = Compressor(args, compress_ratio, self.head_dim)

    def __call__(
        self,
        x: mx.array,
        q_intermediate: mx.array,
        ck: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Score compressed rows and return topk indices.

        Args:
            x: [B, S, D] hidden state (fed to the lightweight compressor).
            q_intermediate: [B, S, q_lora_rank] post wq_a+q_norm (shared with main attn).

        Returns:
            topk_indices [B, topk] or None when there are too few compressed rows.
            Indices are shared across heads (head-weighted scores are aggregated).
        """
        B, S, _ = x.shape

        if ck is None:
            ck = self.compressor(x)
        n_compressed = ck.shape[1]
        if n_compressed == 0:
            return None

        q = self.wq_b(q_intermediate)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)

        scores = (q @ ck[:, None].transpose(0, 1, 3, 2)) * self.scale

        hw = mx.sigmoid(self.weights_proj(x))
        hw = hw.transpose(0, 2, 1)[..., None]
        scores = scores * hw

        agg = scores.sum(axis=2).mean(axis=1)

        topk = min(self.index_topk, n_compressed)
        return mx.argpartition(-agg, kth=topk - 1, axis=-1)[:, :topk]


# --------------------------------------------------------------------------- #
# Block                                                                       #
# --------------------------------------------------------------------------- #

class DeepseekV4Block(nn.Module):
    """V4 block: mHC-wrapped (attention-norm -> attention), mHC-wrapped (moe-norm -> moe).

    The block maintains `hc_mult` parallel hidden-state copies. Each sub-layer
    reduces them to 1 via hc_pre, applies its block, then expands back via hc_post.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = V4Attention(args, layer_idx)
        self.hc_attn = HyperConnection(
            args.hidden_size, args.hc_mult,
            args.rms_norm_eps, args.hc_sinkhorn_iters, args.hc_eps,
        )

        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = DeepseekV4MoE(args, layer_idx)
        self.hc_ffn = HyperConnection(
            args.hidden_size, args.hc_mult,
            args.rms_norm_eps, args.hc_sinkhorn_iters, args.hc_eps,
        )

    def __call__(self, h: mx.array, mask, cache, input_ids: mx.array) -> mx.array:
        # h: [B, S, hc, D]
        # Attention half
        residual = h
        y, post, comb = self.hc_attn.hc_pre(h)
        y = self.attn_norm(y)
        y = self.attn(y, mask=mask, cache=cache)
        h = self.hc_attn.hc_post(y, residual, post, comb)

        # FFN half
        residual = h
        y, post, comb = self.hc_ffn.hc_pre(h)
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        h = self.hc_ffn.hc_post(y, residual, post, comb)
        return h


# --------------------------------------------------------------------------- #
# MTP Block (next-N-token prediction head, from Blaizzy/mlx-lm PR #15)        #
# --------------------------------------------------------------------------- #

class VanillaMarkov(nn.Module):
    """DSpark Markov head (arXiv:2607.05147, VanillaMarkov variant).

    Produces a per-position logit *bias* from the previously sampled token,
    correcting the parallel backbone's suffix decay. The bias is
    `markov_w2(markov_w1(prev_token))`: an embedding lookup into a `markov_rank`
    space followed by a linear projection back to vocab space. Memoryless — it
    conditions only on the immediately preceding token. V4-Flash uses the plain
    (Vanilla) variant: only `markov_w1`/`markov_w2`, no gating/RNN state.
    """

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def prev_embedding(self, prev_token: mx.array) -> mx.array:
        """Rank-space embedding of the previous token (also consumed by the
        confidence head). `prev_token`: integer array [...]."""
        return self.markov_w1(prev_token)

    def logit_bias(self, prev_token: mx.array) -> mx.array:
        """Logit bias to add to the backbone logits at the current block position."""
        return self.markov_w2(self.markov_w1(prev_token))


class DSparkConfidenceHead(nn.Module):
    """DSpark confidence head: predicts per-position acceptance (survival) from the
    stage hidden state concatenated with the Markov rank-space embedding of the
    previous token (`hidden_size + markov_rank` -> 1). Drives *optional* adaptive
    block length; unused for the fixed-gamma baseline (Phase 5).
    """

    def __init__(self, hidden_size: int, markov_rank: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=False)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features)


class DSparkStage(DeepseekV4Block):
    """One stage of the DSpark MTP backbone (arXiv:2607.05147).

    Each stage IS a full V4 transformer block (MLA with num_kv_heads=1, mHC-wrapped
    attention + MoE) forced to plain sliding-window attention: the stages carry no
    compressor/indexer tensors, and their `layer_idx` (>= num_hash_layers) selects
    bias-corrected softmax routing rather than hash routing. Subclassing
    DeepseekV4Block places the block submodules at FLAT paths (`mtp.<s>.attn`,
    `mtp.<s>.hc_attn`, `mtp.<s>.ffn`, ...) — matching the checkpoint's tensor names
    AND its per-path quantization map (e.g. `mtp.0.attn.wq_a` @ 6-bit); nesting them
    under a `.block.` submodule would miss those per-path bits at load.

    Stage-specific head tensors are asymmetric across the stages:
      - first stage (mtp.0): `main_proj` (concat of the tapped residual streams ->
        hidden) + `main_norm` — the entry/fusion of the target model's context.
      - last stage (mtp.N-1): `hc_head` (mHC stream collapse) + `norm`, plus the
        draft heads `markov_head` (VanillaMarkov) and `confidence_head`.
    """

    def __init__(self, args: ModelArgs, stage_idx: int):
        # layer_idx = num_hidden_layers + stage_idx lands past `num_hash_layers`
        # (softmax routing) on a compress_ratio=0 slot (plain sliding-window MLA) —
        # matching the checkpoint's mtp stages.
        super().__init__(args, args.num_hidden_layers + stage_idx)
        self.stage_idx = stage_idx
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == args.n_mtp_stages - 1

        if self.is_first:
            n_tap = len(args.dspark_target_layer_ids)
            self.main_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.main_proj = nn.Linear(
                n_tap * args.hidden_size, args.hidden_size, bias=False
            )

        if self.is_last:
            self.hc_head = HyperHead(
                args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
            )
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.markov_head = VanillaMarkov(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(
                args.hidden_size, args.dspark_markov_rank
            )

    # NOTE: __call__ is inherited from DeepseekV4Block — the stage's block-body
    # forward (mHC-wrapped MLA + MoE) IS a DSpark backbone stage. Cross-attention
    # over [context, block] falls out of the inherited self-attention once the
    # stage's KV cache is pre-filled with the context K=V by `update_ctx` below.

    def update_ctx(self, fused: mx.array, ctx_offset: int, cache: Any) -> None:
        """Pre-fill this stage's KV cache with the DSpark context K=V.

        The context is the shared fused target signal main_norm(main_proj(tap_cat))
        (computed once on the first stage), projected here through THIS stage's own
        wkv/kv_norm/rope — mirroring the K=V path of V4Attention exactly. After this,
        the inherited block forward's self.attn(block, cache=cache) appends the block
        K=V and attends over [context, block], i.e. cross-attention, with rope offsets
        flowing from cache.offset. `fused`: [B, L_ctx, D]; roped at absolute ctx_offset.
        """
        a = self.attn
        B, L, _ = fused.shape
        kv = a.kv_norm(a.wkv(fused)).reshape(B, 1, L, a.head_dim)
        k_nope, k_pe = mx.split(kv, [a.nope_head_dim], axis=-1)
        # Stages are plain sliding-window (compress_ratio 0) -> plain main rope.
        k_pe = a.rope(k_pe, offset=ctx_offset)
        k = mx.concatenate([k_nope, k_pe], axis=-1)
        cache.update_and_fetch(k, k)  # K==V; advances cache.offset by L


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

class DeepseekV4Model(nn.Module, PipelineMixin):
    def __init__(self, args: ModelArgs):
        super().__init__()
        PipelineMixin.__init__(self)
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DeepseekV4Block(args, i) for i in range(args.num_hidden_layers)]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Final HC head (reduces hc copies -> 1 before lm_head)
        self.hc_head = HyperHead(
            args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
        )

    def _collapse(self, h: mx.array, mode: str) -> mx.array:
        """Collapse an mHC stream stack [B,S,hc,D] -> [B,S,D] for a DSpark tap.
        The exact reduction the DSpark head was trained against is not specified in
        any public reference, so it is selectable and resolved empirically."""
        if mode == "hc_head":
            return self.hc_head(h)
        if mode == "stream0":
            return h[:, :, 0, :]
        if mode == "mean":
            return h.mean(axis=2)
        raise ValueError(f"unknown tap collapse mode: {mode}")

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_raw_hidden: bool = False,
        capture_layer_ids: Optional[List[int]] = None,
        capture_collapse: str = "hc_head",
    ):
        h = self.embed_tokens(inputs)                        # [B, S, D]
        # Expand to hc_mult parallel copies
        h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]))
        # Make it contiguous — broadcast_to gives a view
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * self.num_layers

        first_cache = cache[0]
        if isinstance(first_cache, CompressedKVCache):
            first_cache = first_cache.local
        elif isinstance(first_cache, (list, tuple)):
            first_cache = first_cache[0]
        mask = create_attention_mask(
            h[:, :, 0, :],
            first_cache if first_cache is not None else None,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        tapset = set(capture_layer_ids) if capture_layer_ids is not None else None
        captured = [] if tapset is not None else None
        for i in range(self.num_layers):
            layer_idx = self.start_idx + i
            h = self.layers[layer_idx](h, mask, cache[i], inputs)
            if tapset is not None and layer_idx in tapset:
                captured.append(self._collapse(h, capture_collapse))

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            last_cache = cache[-1]
            if last_cache is not None:
                lc = last_cache.local if isinstance(last_cache, CompressedKVCache) else last_cache
                if hasattr(lc, 'keys') and lc.keys is not None:
                    lc.keys = mx.depends(lc.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # Reduce [B,S,hc,D] -> [B,S,D] then RMSNorm
        out = self.norm(self.hc_head(h))
        if captured is not None:
            # DSpark tap: concat the collapsed residual streams at the tapped layers.
            return out, mx.concatenate(captured, axis=-1)   # [B, S, n_tap * D]
        if return_raw_hidden:
            return out, h
        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # DSpark MTP head (built only for MTP-preserving checkpoints). Gated on the
        # DSpark tap config, NOT `num_nextn_predict_layers` (a V3-loader trap).
        if args.n_mtp_stages > 0:
            self.mtp = [DSparkStage(args, i) for i in range(args.n_mtp_stages)]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
    ):
        if return_hidden:
            h, h_raw = self.model(inputs, cache, return_raw_hidden=True)
            return self.lm_head(h), h_raw
        h = self.model(inputs, cache)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def pred(k: str):
            # Keep mHC params and gate biases in fp32
            if "hc_" in k or "e_score_correction_bias" in k or "attn_sink" in k:
                return False
            if k.endswith(".fn") or k.endswith(".base") or k.endswith(".scale"):
                return False
            return True
        return pred

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attn.compress_ratio:
                caches.append(CompressedKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def make_mtp_cache(self):
        if not hasattr(self, "mtp"):
            return None
        caches = []
        for mtp_block in self.mtp:
            attn = mtp_block.attn
            if attn.compress_ratio:
                caches.append(CompressedKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def mtp_forward(self, *args, **kwargs):
        # Legacy next-1-token API (generate.py --mtp path). The V3-shaped forward was
        # removed; DSpark decoding uses the dspark_* methods + a draft->verify loop
        # (Phase 4). The default (non-MTP) generate path is unaffected.
        raise NotImplementedError(
            "Legacy mtp_forward is not supported for the DSpark head; use the "
            "dspark_* methods with a draft->verify loop (Phase 4)."
        )

    # ------------------------------------------------------------------- #
    # DSpark speculative-decoding head (arXiv:2607.05147)                  #
    # ------------------------------------------------------------------- #

    def make_dspark_ctx_cache(self) -> Optional[List[Any]]:
        """One sliding-window KV cache per DSpark stage, holding that stage's
        projected context K=V (pre-filled by dspark_update_context)."""
        if not hasattr(self, "mtp"):
            return None
        return [RotatingKVCache(max_size=self.args.sliding_window) for _ in self.mtp]

    def dspark_tap(self, inputs: mx.array, cache=None, collapse: str = "hc_head"):
        """Target forward that also returns the DSpark context tap:
        (logits [B,S,V], tap_cat [B,S,n_tap*D]) where tap_cat concatenates the
        collapsed residual streams at `dspark_target_layer_ids`."""
        out, tap_cat = self.model(
            inputs, cache,
            capture_layer_ids=self.args.dspark_target_layer_ids,
            capture_collapse=collapse,
        )
        return self.lm_head(out), tap_cat

    def dspark_fuse(self, tap_cat: mx.array) -> mx.array:
        """Fuse the tapped target context once: main_norm(main_proj(tap_cat)) -> [B,L,D].
        main_proj/main_norm live only on the first stage but the fused signal feeds
        every stage's context projection (per-stage wkv)."""
        s0 = self.mtp[0]
        return s0.main_norm(s0.main_proj(tap_cat))

    def dspark_update_context(
        self, tap_cat: mx.array, ctx_offset: int, ctx_caches: List[Any]
    ) -> None:
        """Append the fused context of `tap_cat`'s positions to every stage's KV
        cache, roped at absolute `ctx_offset`."""
        fused = self.dspark_fuse(tap_cat)
        for stage, cache in zip(self.mtp, ctx_caches):
            stage.update_ctx(fused, ctx_offset, cache)

    def dspark_backbone(
        self, block_ids: mx.array, ctx_caches: List[Any], mask=None,
        return_hidden: bool = False,
    ):
        """Run the DSpark backbone over a draft block -> base logits [B, blk, V].

        `block_ids` [B, blk] = [anchor, noise, ...]. `ctx_caches` must already hold
        the context K=V (via dspark_update_context); each stage's block rope offset
        flows from its cache.offset, so the block sits at absolute [ctx_len, ...] and
        cross-attends [context, block]. mask=None gives the single-block layout
        (full attention over the whole context + bidirectional block).

        With `return_hidden`, also returns the last stage's post-norm hidden [B, blk, D]
        (pre-lm_head) — the confidence head's per-position input."""
        h = self.model.embed_tokens(block_ids)                        # [B, blk, D]
        hc = self.args.hc_mult
        h = mx.contiguous(
            mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], hc, h.shape[2]))
        )
        for stage, cache in zip(self.mtp, ctx_caches):
            h = stage(h, mask, cache, block_ids)                      # cross-attn block forward
        last = self.mtp[-1]
        out = last.norm(last.hc_head(h))                              # collapse streams + norm
        logits = self.lm_head(out)                                     # [B, blk, V]
        return (logits, out) if return_hidden else logits

    def _dspark_draft(
        self, base_logits: mx.array, hidden: mx.array, first_prev: int,
        cap: Optional[int], threshold: float,
    ) -> List[int]:
        """Draft a block: greedy sampling with sequential VanillaMarkov correction
        (arXiv:2607.05147) plus optional confidence-based trim (§3.2.1), built as one
        graph and read back with a SINGLE device sync.

        `base_logits` [blk, V] are the backbone's parallel per-position logits
        (logits_start=0: position i predicts draft token i); the Markov head conditions
        each position on the previously sampled token, so the block is sampled
        sequentially. With `threshold > 0`, the confidence head (input:
        concat(post-norm hidden_i, markov_w1(prev_token_i))) trims the draft to the
        longest prefix whose cumulative survival probability stays >= threshold (min 1).
        Fusing the ~gamma+1 per-token syncs into one matters because the loop is
        host-sync-bound, not FLOP-bound, at batch=1. Returns the drafted token ids."""
        last = self.mtp[-1]
        markov = last.markov_head
        k = base_logits.shape[0] if cap is None else min(cap, base_logits.shape[0])
        prev = mx.array([first_prev])
        toks = []
        for i in range(k):                                   # sequential Markov graph
            step = base_logits[i] + markov.logit_bias(prev)[0]
            nxt = mx.argmax(step, axis=-1, keepdims=True)
            toks.append(nxt)
            prev = nxt
        d = mx.concatenate(toks)                             # [k]
        if threshold > 0.0 and k > 1:
            prev_ids = mx.concatenate([mx.array([first_prev]), d[:-1]])       # [k]
            feats = mx.concatenate(
                [hidden[:k], markov.prev_embedding(prev_ids)], axis=-1
            )
            surv = mx.cumprod(mx.sigmoid(last.confidence_head(feats)[:, 0]))  # [k]
            keep = mx.maximum(mx.sum((surv >= threshold).astype(mx.int32)), 1)
            mx.eval(d, keep)                                 # one sync
            return [int(x) for x in d[: int(keep.item())].tolist()]
        mx.eval(d)                                           # one sync
        return [int(x) for x in d.tolist()]

    def _dspark_spec_begin(self, caches: List[Any]) -> List[int]:
        """Arm single-forward spec rollback before a verify forward: record each
        cache's pre-verify offset and arm compressed-pool capture. Returns the
        offsets so `_dspark_spec_commit` knows where the accepted prefix begins."""
        offsets = []
        for c in caches:
            offsets.append(c.offset)
            if isinstance(c, CompressedKVCache):
                c.spec_begin()
        return offsets

    def _dspark_spec_commit(
        self, caches: List[Any], offsets: List[int], keep: int
    ) -> None:
        """Roll each target cache back to `keep` positions past its pre-verify
        offset — the accepted prefix [pending] + accepted drafts — in a single
        forward (no clean re-advance). The compressed pool is rebuilt bit-exactly
        by `spec_trim`; the local sliding KV is trimmed to the same length."""
        for c, off in zip(caches, offsets):
            target = off + keep
            if isinstance(c, CompressedKVCache):
                c.spec_trim(keep)
                c.local.trim(c.local.offset - target)
            else:
                c.trim(c.offset - target)

    def dspark_generate(
        self,
        prompt_ids: List[int],
        max_tokens: int,
        *,
        cap: Optional[int] = None,
        collapse: str = "mean",
        confidence_threshold: float = 0.0,
    ):
        """Lossless DSpark speculative greedy decode (arXiv:2607.05147).

        CORRECTNESS: lossless in exact arithmetic. The V4 forward is causal (verified:
        appending tokens does not change earlier-position logits — zero information
        leak), so within one verify forward each position's argmax is independent of the
        drafts after it; the committed sequence (accepted greedy-prefix + bonus) is
        exactly the target's own greedy continuation. On a QUANTIZED checkpoint the
        output matches plain greedy only up to quantization near-tie flips — plain greedy
        is itself not reproducible between incremental-cache and full-recompute (a
        length-dependent rounding effect), so bit-exact identity is not a meaningful gate
        there.

        PERFORMANCE: one target forward per round. The verify forward's own KV/pool
        entries for the accepted prefix are committed in place via exact single-forward
        rollback (`spec_begin`/`spec_trim`): the compressed-KV pool is rebuilt from the
        pre-verify snapshot over the captured accepted raw tokens, so no clean re-advance
        pass is needed. This also makes the committed cache numerically self-consistent
        with the forward that made the accept decision.

        ADAPTIVE VERIFY LENGTH (arXiv:2607.05147 §3.2.1): with
        `confidence_threshold > 0`, the confidence head trims the draft to the longest
        prefix whose cumulative survival probability stays >= the threshold (min 1
        token), shrinking the verify width — the dominant per-round cost — on rounds
        where the drafter is unsure. This never changes which tokens are COMMITTED
        (still the verify forward's own argmax), so losslessness is preserved; a good
        value is ~0.2 (draft ~5 -> ~3, accept-len preserved, ~1.2x throughput on this
        2.4-bit checkpoint). `confidence_threshold = 0.0` (default) is identical to the
        loop above.

        Returns (token_ids, stats) with `mean_accept_len`, `mean_draft_len`, `rounds`.
        """
        if not hasattr(self, "mtp"):
            raise ValueError("checkpoint has no DSpark head (dspark_target_layer_ids unset)")
        bs = self.args.dspark_block_size
        cap = bs if cap is None else min(cap, bs)
        mask_id = self.args.dspark_noise_token_id

        tgt = self.make_cache()
        ctxc = self.make_dspark_ctx_cache()
        # Full-prompt prefill (so the first token equals plain greedy); seed the draft
        # context with the whole prompt tap.
        logits, tap = self.dspark_tap(mx.array([list(prompt_ids)]), tgt, collapse)
        self.dspark_update_context(tap, 0, ctxc)
        pending = int(mx.argmax(logits[0, -1]).item())
        out = [pending]
        committed_len = len(prompt_ids)   # positions held by the target cache
        accept_lens: List[int] = []
        draft_lens: List[int] = []

        while len(out) < max_tokens:
            # --- draft a block; the backbone appends the block to the ctx caches, so
            #     roll that back before it can poison the next round's context ---
            block = mx.array([[pending] + [mask_id] * (bs - 1)])
            ctx_off = [c.offset for c in ctxc]
            base, hidden = self.dspark_backbone(block, ctxc, return_hidden=True)
            drafted = self._dspark_draft(
                base[0], hidden[0], pending, cap, confidence_threshold
            )
            for c, off in zip(ctxc, ctx_off):
                c.trim(c.offset - off)
            draft_lens.append(len(drafted))

            # --- verify [pending] + drafted against the target (single forward;
            #     arm exact rollback so the accepted prefix is kept in place) ---
            offsets = self._dspark_spec_begin(tgt)
            vlogits, vtap = self.dspark_tap(mx.array([[pending] + drafted]), tgt, collapse)
            tt = mx.argmax(vlogits[0], axis=-1)
            # vectorized accept: length of the leading run of matches, one sync (forces tt)
            matches = (mx.array(drafted) == tt[: len(drafted)]).astype(mx.int32)
            n = int(mx.sum(mx.cumprod(matches)).item())
            bonus = int(tt[n].item())
            committed = drafted[:n] + [bonus]
            accept_lens.append(len(committed))

            # commit context with the accepted positions [pending]+accepted (verify tap is
            # causally valid to reuse), then roll the target cache back to that prefix.
            self.dspark_update_context(vtap[:, : n + 1, :], committed_len, ctxc)
            self._dspark_spec_commit(tgt, offsets, n + 1)
            committed_len += n + 1

            for t in committed:
                out.append(t)
                if len(out) >= max_tokens:
                    break
            pending = bonus

        stats = {
            "mean_accept_len": sum(accept_lens) / len(accept_lens) if accept_lens else 0.0,
            "mean_draft_len": sum(draft_lens) / len(draft_lens) if draft_lens else 0.0,
            "rounds": len(accept_lens),
        }
        return out[:max_tokens], stats

    # ------------------------------------------------------------------- #
    # Weight loading                                                      #
    # ------------------------------------------------------------------- #

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Handle DeepSeek-V4 checkpoint conversion.

        Supports both raw HF checkpoints (FP8/FP4 block scales) and
        pre-quantized MLX checkpoints (e.g. mlx-community 8-bit).

        Checkpoint naming (from HF):
            layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
            layers.N.attn.{q_norm,kv_norm,attn_sink}
            layers.N.attn.compressor.{wkv,wgate,ape,norm}
            layers.N.attn.indexer.{wq_b,weights_proj,compressor.*}
            layers.N.ffn.gate.{weight,bias,tid2eid}
            layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}
            layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}
            layers.N.{attn_norm,ffn_norm}.weight
            layers.N.hc_{attn,ffn}_{fn,base,scale}
            embed.weight, head.weight, hc_head_{fn,base,scale}
            mtp.<s>.{attn,ffn,attn_norm,ffn_norm,attn_hc,ffn_hc}.*  (DSpark stages,
              flat; kept iff self.mtp exists), plus stage heads
              mtp.0.{main_proj,main_norm}, mtp.N.{hc_head,norm,markov_head,
              confidence_head}. Dropped when the model has no DSpark head.

        MLX-quantized naming (community 8-bit):
            embed.{weight,biases,scales}, head.{weight,biases,scales}
            layers.N.attn.wo_a.G.{weight,biases,scales} (per-group)
            layers.N.ffn.experts.w{1,2,3}.{weight,biases,scales} (pre-stacked)
        """
        n_layers = self.args.num_hidden_layers

        # 1) Keep MTP weights only when self.mtp exists; drop layers beyond n_layers
        has_mtp = hasattr(self, "mtp")
        has_mtp_weights = any(k.startswith("mtp.") for k in weights)
        # Disable MTP module if weights are absent (e.g. quantized checkpoints)
        if has_mtp and not has_mtp_weights:
            del self.mtp
            has_mtp = False
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                if not has_mtp:
                    continue
                new_weights[k] = v
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    idx = int(parts[1])
                except ValueError:
                    new_weights[k] = v
                    continue
                if idx >= n_layers:
                    continue
            new_weights[k] = v
        weights = new_weights

        def _scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        # 2) FP8/FP4 block dequant:
        #    `X.weight` + `X.scale` -> dequantized bf16 `X.weight`
        #    Routed experts in Flash are FP4-packed int8; other scaled matrices
        #    are FP8 e4m3 with 128x128 block scales.
        def _dequant_fp8_block(weight: mx.array, scale: mx.array, bs: int = 128) -> mx.array:
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            scale = _scale_to_float(scale)
            m, n = weight.shape
            pad_b = (-m) % bs
            pad_s = (-n) % bs
            weight = mx.pad(weight, ((0, pad_b), (0, pad_s)))
            weight = weight.reshape(((m + pad_b) // bs, bs, (n + pad_s) // bs, bs))
            weight = (weight * scale[:, None, :, None]).reshape(m + pad_b, n + pad_s)
            return weight[:m, :n].astype(mx.bfloat16)

        def _dequant_fp4_block(weight: mx.array, scale: mx.array, bs: int = 32) -> mx.array:
            table = mx.array(
                [
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
                ],
                dtype=mx.float32,
            )
            packed = weight.astype(mx.uint8)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            unpacked = mx.stack([mx.take(table, low), mx.take(table, high)], axis=-1)
            unpacked = unpacked.reshape(weight.shape[0], weight.shape[1] * 2)
            scale = mx.repeat(_scale_to_float(scale), bs, axis=-1)
            return (unpacked * scale).astype(mx.bfloat16)

        new = {}
        for k, v in weights.items():
            if k.endswith(".scale"):
                wk = k[:-len(".scale")] + ".weight"
                weight = weights.get(wk)
                if (
                    weight is not None
                    and ".ffn.experts." in wk
                    and "shared_experts" not in wk
                    and weight.dtype in (mx.int8, mx.uint8)
                    and v.shape[-1] * 16 == weight.shape[-1]
                ):
                    new[wk] = _dequant_fp4_block(weight, v)
                elif weight is not None and weight.dtype in (mx.uint8,):
                    new[wk] = _dequant_fp8_block(weights[wk], v)
                else:
                    new[k] = v
            elif k not in new:
                new[k] = v
        weights = new

        # 3) Remap top-level names to our module structure
        #    Prefix-based remap handles both raw (.weight) and quantized
        #    (.weight, .biases, .scales) checkpoints.
        top_prefix_remap = {
            "embed.":      "model.embed_tokens.",
            "head.":       "lm_head.",
        }
        top_exact_remap = {
            "norm.weight":     "model.norm.weight",
            "hc_head_fn":      "model.hc_head.fn",
            "hc_head_base":    "model.hc_head.base",
            "hc_head_scale":   "model.hc_head.scale",
        }
        new = {}
        for k, v in weights.items():
            nk = k
            for old_pfx, new_pfx in top_prefix_remap.items():
                if nk.startswith(old_pfx):
                    nk = new_pfx + nk[len(old_pfx):]
                    break
            if nk in top_exact_remap:
                nk = top_exact_remap[nk]
            new[nk] = v
        weights = new

        # 4) Remap layer-level names: layers.N.X -> model.layers.N.X
        #    Also remap gate.bias -> gate.e_score_correction_bias,
        #    hc_{attn,ffn}_{fn,base,scale} -> hc_{attn,ffn}.{fn,base,scale},
        #    shared_experts.w{1,2,3} -> shared_experts.{gate,down,up}_proj
        new = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        # DSpark mtp stages are FLAT DeepseekV4Blocks (DSparkStage subclasses the
        # block), so `mtp.<s>.attn/ffn/attn_norm/...` map 1:1 to module paths — the
        # same generic renames below (attn_hc->hc_attn, gate.bias->e_score_correction_bias,
        # shared_experts.w{1,2,3}->{gate,down,up}_proj) apply to mtp keys unchanged.
        # Stage-head tensors (main_proj/main_norm, hc_head, markov_head,
        # confidence_head, norm) already carry their final names and pass through.
        for k, v in weights.items():
            nk = k
            # Add model. prefix for main-model layers (mtp.* keeps its own prefix)
            if nk.startswith("layers."):
                nk = "model." + nk

            # gate.bias -> gate.e_score_correction_bias
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")

            # hc_attn_fn -> hc_attn.fn (etc.) — raw HF checkpoint underscores
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".hc_{sub}.{param}")

            # attn_hc.X -> hc_attn.X (mlx-community/DeepSeek-V4-Flash-8bit
            # naming order: per-layer hyper-connections stored as <sub>_hc.<param>
            # rather than hc_<sub>.<param>). Apply after the underscore rename so
            # both naming orders converge to the model's hc_<sub>.<param> layout.
            for sub in ("attn", "ffn"):
                nk = nk.replace(f".{sub}_hc.", f".hc_{sub}.")

            # shared_experts.w1 -> shared_experts.gate_proj (etc.)
            for w_old, w_new in w_remap.items():
                nk = nk.replace(f".shared_experts.{w_old}.", f".shared_experts.{w_new}.")

            new[nk] = v
        weights = new

        # 5) Stack expert weights: experts.E.w{1,2,3}.weight -> switch_mlp.{gate,down,up}_proj.weight
        #    Also handle pre-stacked experts (community quants): experts.w{1,2,3}.X -> switch_mlp.{proj}.X
        expert_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for l in range(n_layers):
            prefix = f"model.layers.{l}.ffn.experts"
            for src, dst in expert_remap.items():
                # Case A: per-expert weights need stacking (raw HF checkpoint)
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stack = [weights.pop(f"{prefix}.{e}.{src}.weight")
                             for e in range(self.args.n_routed_experts)]
                    weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.weight"] = mx.stack(stack)
                # Case B: already-stacked (community quant) — rename experts.w1.X -> switch_mlp.gate_proj.X
                for suffix in ("weight", "biases", "scales"):
                    old = f"{prefix}.{src}.{suffix}"
                    if old in weights:
                        weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.{suffix}"] = weights.pop(old)

        # 6) Fuse split wo_a: community quants store wo_a.G.{weight,biases,scales}
        #    per-group; our model uses a single QuantizedLinear with grouped dequant.
        n_groups = self.args.o_groups
        for l in range(n_layers):
            prefix = f"model.layers.{l}.attn.wo_a"
            if f"{prefix}.0.weight" in weights:
                for suffix in ("weight", "biases", "scales"):
                    parts = []
                    for g in range(n_groups):
                        key = f"{prefix}.{g}.{suffix}"
                        if key in weights:
                            parts.append(weights.pop(key))
                    if parts:
                        weights[f"{prefix}.{suffix}"] = mx.concatenate(parts, axis=0)

        # 6b) Flatten pre-stacked 3D wo_a. Newer mixed-quant checkpoints store the
        #     grouped output projection as a single [n_groups, o_lora_rank, X] tensor
        #     (weight/scales/biases) rather than per-group split keys. Our
        #     QuantizedLinear expects it fused to [n_groups * o_lora_rank, X] and
        #     re-splits internally in the grouped matmul. Applies to both the main
        #     layers and the DSpark mtp stages (group-major flatten preserves order).
        for k in list(weights):
            if k.endswith(("attn.wo_a.weight", "attn.wo_a.scales", "attn.wo_a.biases")):
                v = weights[k]
                if v.ndim == 3:
                    weights[k] = v.reshape(v.shape[0] * v.shape[1], v.shape[2])

        # Stack routed expert weights for MTP stages (raw HF: per-expert w{1,2,3};
        # pre-stacked community quants already carry switch_mlp.* and no-op here).
        if has_mtp:
            for mtp_idx in range(self.args.n_mtp_stages):
                prefix = f"mtp.{mtp_idx}.ffn.experts"
                for src, dst in (
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    key0 = f"{prefix}.0.{src}.weight"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.weight")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"mtp.{mtp_idx}.ffn.switch_mlp.{dst}.weight"
                        ] = mx.stack(stacked)

        return weights

    # ------------------------------------------------------------------- #
    # Distributed sharding                                                 #
    # ------------------------------------------------------------------- #

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()
        for layer in self.model.layers:
            a = layer.attn
            a.wq_b = shard_linear(a.wq_b, "all-to-sharded", group=group)
            a.wo_b = shard_linear(a.wo_b, "sharded-to-all", group=group)
            a.n_heads //= N
            # Slice attn_sink to local heads (mirrors gpt_oss.py:308-312).
            # Order matters: must run AFTER `a.n_heads //= N` so the stride is
            # the post-division (local) head count.
            a.attn_sink = a.attn_sink[a.n_heads * R : a.n_heads * (R + 1)]

            # wo_a: shape (n_groups * o_lora_rank, group_feat).
            # group_feat = n_heads * v_head_dim / n_groups. After sharding,
            # n_heads //= N and n_groups //= N cancel in the ratio, so group_feat
            # stays constant. Only the OUTPUT dim (n_groups axis) gets sharded —
            # each rank owns n_groups//N consecutive groups.
            # wo_b is "sharded-to-all" so its input = n_groups_local * o_lora_rank.
            old_n_groups = a.n_groups
            new_n_groups = old_n_groups // N
            gs = new_n_groups * R
            ge = new_n_groups * (R + 1)
            if isinstance(a.wo_a, nn.QuantizedLinear):
                gf = a.wo_a.weight.shape[-1]
                w = a.wo_a.weight.reshape(old_n_groups, a.o_lora_rank, gf)
                a.wo_a.weight = w[gs:ge].reshape(new_n_groups * a.o_lora_rank, gf)
                sc_gf = a.wo_a.scales.shape[-1]
                s = a.wo_a.scales.reshape(old_n_groups, a.o_lora_rank, sc_gf)
                a.wo_a.scales = s[gs:ge].reshape(new_n_groups * a.o_lora_rank, sc_gf)
                if getattr(a.wo_a, "biases", None) is not None:
                    b_gf = a.wo_a.biases.shape[-1]
                    b = a.wo_a.biases.reshape(old_n_groups, a.o_lora_rank, b_gf)
                    a.wo_a.biases = b[gs:ge].reshape(new_n_groups * a.o_lora_rank, b_gf)
            else:
                gf = a.wo_a.weight.shape[-1]
                w = a.wo_a.weight.reshape(old_n_groups, a.o_lora_rank, gf)
                a.wo_a.weight = w[gs:ge].reshape(new_n_groups * a.o_lora_rank, gf)
            a.n_groups = new_n_groups

            if isinstance(layer.ffn, DeepseekV4MoE):
                layer.ffn.sharding_group = group
                if hasattr(layer.ffn, "shared_experts"):
                    shard_inplace(layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group)
                    shard_inplace(layer.ffn.shared_experts.up_proj,   "all-to-sharded", group=group)
                shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
                shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
                shard_inplace(layer.ffn.switch_mlp.up_proj,   "all-to-sharded", group=group)
