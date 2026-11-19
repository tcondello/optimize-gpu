#!/usr/bin/env python3
"""
Ultra-Fast Qwen Inference Engine for NVIDIA eGPUs on Apple Silicon macOS (TinyGPU + tinygrad).

Key architectural optimizations over stock `tinygrad/apps/llm.py`:
1. Zero-Mount NVRTC Daemon (`nvrtc_daemon.py`): compiles CUDA -> SASS cubin in ~15 ms (vs 550 ms)
   with zero VirtioFS mounts (fixing Docker Desktop `service fs` hypervisor crashes).
2. Raw GGUF Ampere `__dp4a` + Warp-Shuffle GEMV/GEMM (`cuda_kernels.py`): streams Q4_K/Q6_K/Q8_0
   directly at 480-715 GB/s (up to 94% of RTX 3080's 760 GB/s GDDR6X peak, vs 17-39 GB/s stock).
3. Weight Projection Fusion: concatenates `ffn_gate + ffn_up` and `attn_q + attn_k (+ attn_v)`
   directly in raw GGUF block space with zero VRAM bloat.
4. Kernel Fusion:
   - Fused `RMSNorm + Q8_1` activation quantization
   - Fused `SwiGLU + Q8_1` activation quantization
   - Fused `Residual Add` inside `attn_output` and `ffn_down` GEMV epilogues
   - Fused `QK-RMSNorm + Half-Split RoPE + FP16 KV-Cache Write`
   - Single-pass Online-Softmax GQA `Flash-Decoding` & `Flash-Prefill`
   - Single-CTA warp-shuffle `fast_argmax_1d` for greedy decoding (12 us vs 200 us)
5. Single-Graph HCQ Execution (`JIT_BATCH_SIZE=0`): captures all ~320 kernels of a 36-layer Qwen
   step into ONE hardware command queue submission (< 0.4 ms Thunderbolt dispatch overhead).
6. Full Qwen2.5 + Qwen3 support (including QKV attention biases that caused gibberish in stock llm.py).
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import struct
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Configure tinygrad environment BEFORE importing tinygrad
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("JIT_BATCH_SIZE", "0")

TINYGRAD_REPO = os.getenv("TINYGRAD_REPO", str(Path.home() / "tinygrad"))
if TINYGRAD_REPO not in sys.path and Path(TINYGRAD_REPO).exists():
    sys.path.insert(0, TINYGRAD_REPO)

from nvrtc_daemon import (  # noqa: E402
    ensure_compiler_daemon,
    install_fast_tinygrad_compiler,
    install_nv_thunderbolt_membar_opt,
)

ensure_compiler_daemon()
install_fast_tinygrad_compiler()
install_nv_thunderbolt_membar_opt()

from tinygrad import Context, Device, Tensor, TinyJit, UOp, dtypes, nn  # noqa: E402
from tinygrad.apps.llm import SimpleTokenizer, precompute_freqs_cis  # noqa: E402
from tinygrad.helpers import DEBUG, GlobalCounters, Timing, colored, fetch, prod, round_up, stderr_log  # noqa: E402
from tinygrad.nn.state import ggml_data_to_tensor  # noqa: E402
from tinygrad.viz.serve import HTTPRequestHandler, TCPServerWithReuse  # noqa: E402

from cuda_kernels import (  # noqa: E402
    fast_argmax_1d,
    flash_decode_gqa,
    flash_decode_gqa_q8_1,
    flash_prefill_gqa,
    fused_qk_rope_kv_populate,
    gguf_embedding_lookup,
    gguf_quant_gemv,
    quantize_q8_1,
    rmsnorm_quantize_q8_1,
    select_token_row,
    swiglu_quantize_q8_1,
)

QWEN_MODELS: dict[str, str] = {
    "qwen3:4b": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
    "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
    "qwen3:14b": "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
    "qwen2.5:3b": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "qwen2.5:7b": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}

SUPPORTED_RAW_TYPES: dict[int, tuple[int, int]] = {
    8: (32, 34),     # Q8_0
    12: (256, 144),  # Q4_K
    14: (256, 210),  # Q6_K
}

PREFILL_CHUNK = 32


def parse_gguf_header(path: Path) -> tuple[dict, dict[str, tuple[tuple[int, ...], int, int]], int]:
    """Parse GGUF metadata and tensor descriptors directly from local disk in ~5 ms."""
    kv_data: dict = {}
    t_infos: dict[str, tuple[tuple[int, ...], int, int]] = {}
    with open(path, "rb") as f:
        def ru32(): return struct.unpack("<I", f.read(4))[0]
        def ri32(): return struct.unpack("<i", f.read(4))[0]
        def ru64(): return struct.unpack("<Q", f.read(8))[0]
        def ri64(): return struct.unpack("<q", f.read(8))[0]
        def rstr(): return f.read(ru64()).decode("utf-8")

        fmts = {0: "<c", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        sizes = {k: struct.calcsize(v) for k, v in fmts.items()}

        def rval(t: int):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = ri32(), ru64()
                return [rval(et) for _ in range(n)]
            return struct.unpack(fmts[t], f.read(sizes[t]))[0]

        magic = f.read(4)
        version = ri32()
        if magic != b"GGUF" or version not in (2, 3):
            raise ValueError(f"Invalid GGUF file: {path}")
        n_tensors, n_kv = ri64(), ri64()
        for _ in range(n_kv):
            k = rstr()
            t = ri32()
            kv_data[k] = rval(t)
        for _ in range(n_tensors):
            name = rstr()
            ndim = ru32()
            dims = tuple(ru64() for _ in range(ndim))
            ggml_type = ri32()
            offset = ru64()
            t_infos[name] = (dims, ggml_type, offset)
        alignment = kv_data.get("general.alignment", 32)
        data_start = round_up(f.tell(), alignment)
    return kv_data, t_infos, data_start


def quantize_f32_to_q8_0_bytes(w_2d: Tensor) -> Tensor:
    """Convert any fallback weight matrix (N, K) into raw GGUF Q8_0 bytes on GPU once at load time."""
    N, K = w_2d.shape
    assert K % 32 == 0
    wf = w_2d.float().reshape(N, K // 32, 32)
    amax = wf.abs().max(axis=-1, keepdim=True)
    d = (amax / 127.0).cast(dtypes.float16)
    inv_d = (d.float() > 0).where(1.0 / d.float(), 0.0)
    qs = (wf * inv_d).round().clip(-127, 127).cast(dtypes.int8).bitcast(dtypes.uint8)
    d_bytes = d.bitcast(dtypes.uint8)  # (N, K//32, 2)
    return d_bytes.cat(qs, dim=-1).flatten().contiguous().realize()


@dataclass
class QuantWeight:
    raw: Tensor
    ggml_type: int
    out_features: int
    in_features: int
    bias: Tensor | None = None


@dataclass(frozen=True)
class QwenConfig:
    arch: str
    num_blocks: int
    dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    norm_eps: float
    vocab_size: int
    max_context: int
    has_qk_norm: bool


class FastQwenBlock:
    def __init__(
        self,
        config: QwenConfig,
        attn_norm_w: Tensor,
        ffn_norm_w: Tensor,
        qkv_fused: QuantWeight | None,
        qk_fused: QuantWeight | None,
        v_proj: QuantWeight | None,
        q_proj: QuantWeight | None,
        k_proj: QuantWeight | None,
        attn_output: QuantWeight,
        gate_up_fused: QuantWeight | None,
        ffn_gate: QuantWeight | None,
        ffn_up: QuantWeight | None,
        ffn_down: QuantWeight,
        q_norm_w: Tensor | None,
        k_norm_w: Tensor | None,
        freqs_cis: Tensor,
        kv_alloc_ctx: int,
    ):
        self.cfg = config
        self.attn_norm_w = attn_norm_w
        self.ffn_norm_w = ffn_norm_w
        self.qkv_fused = qkv_fused
        self.qk_fused = qk_fused
        self.v_proj = v_proj
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.attn_output = attn_output
        self.gate_up_fused = gate_up_fused
        self.ffn_gate = ffn_gate
        self.ffn_up = ffn_up
        self.ffn_down = ffn_down
        self.q_norm_w = q_norm_w
        self.k_norm_w = k_norm_w
        self.freqs_cis = freqs_cis
        self.cache_kv = Tensor.zeros(
            2, 1, config.n_kv_heads, kv_alloc_ctx, config.head_dim,
            dtype=dtypes.float16, device="NV",
        ).contiguous().realize()

    def _compute_qkv(self, xq: Tensor, xds: Tensor, T: int) -> tuple[Tensor, Tensor | None]:
        if self.qkv_fused is not None:
            w = self.qkv_fused
            return gguf_quant_gemv(xq, xds, w.raw, w.ggml_type, w.out_features, w.in_features, bias=w.bias, out_shape=(1, T, w.out_features)), None
        if self.qk_fused is not None and self.v_proj is not None:
            wqk, wv = self.qk_fused, self.v_proj
            qk = gguf_quant_gemv(xq, xds, wqk.raw, wqk.ggml_type, wqk.out_features, wqk.in_features, bias=wqk.bias, out_shape=(1, T, wqk.out_features))
            v = gguf_quant_gemv(xq, xds, wv.raw, wv.ggml_type, wv.out_features, wv.in_features, bias=wv.bias, out_shape=(1, T, wv.out_features))
            return qk, v
        assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
        q = gguf_quant_gemv(xq, xds, self.q_proj.raw, self.q_proj.ggml_type, self.q_proj.out_features, self.q_proj.in_features, bias=self.q_proj.bias, out_shape=(1, T, self.q_proj.out_features))
        k = gguf_quant_gemv(xq, xds, self.k_proj.raw, self.k_proj.ggml_type, self.k_proj.out_features, self.k_proj.in_features, bias=self.k_proj.bias, out_shape=(1, T, self.k_proj.out_features))
        v = gguf_quant_gemv(xq, xds, self.v_proj.raw, self.v_proj.ggml_type, self.v_proj.out_features, self.v_proj.in_features, bias=self.v_proj.bias, out_shape=(1, T, self.v_proj.out_features))
        return q.cat(k, dim=-1).contiguous(), v

    def forward(self, x: Tensor, start_pos: UOp, is_decode: bool) -> Tensor:
        T = x.shape[1]
        # 1. Fused Attention RMSNorm + Q8_1 Quantization
        xq, xds = rmsnorm_quantize_q8_1(x, self.attn_norm_w, self.cfg.norm_eps)

        # 2. Fused QKV __dp4a Projection (zero .cat() copy when QK and V have different quant types!)
        qk_or_qkv, v_sep = self._compute_qkv(xq, xds, T)

        # 3. Fused QK-RMSNorm + Half-Split RoPE + FP16 KV-Cache Store + Flash Attention
        q_roped, kv_after = fused_qk_rope_kv_populate(
            qk_or_qkv, self.cache_kv, self.freqs_cis, start_pos,
            self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim,
            self.cfg.rope_dim, self.cfg.norm_eps, self.q_norm_w, self.k_norm_w,
            v_separate=v_sep,
        )
        if is_decode:
            aq, ads = flash_decode_gqa_q8_1(
                q_roped, kv_after, start_pos,
                self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim,
            )
        else:
            attn_out = flash_prefill_gqa(
                q_roped, kv_after, start_pos, T,
                self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim,
            )
            aq, ads = quantize_q8_1(attn_out)

        # 4. Output Projection with Fused Residual Add (`h = x + attn_output(attn_out)`)
        wo = self.attn_output
        h = gguf_quant_gemv(aq, ads, wo.raw, wo.ggml_type, wo.out_features, wo.in_features, bias=wo.bias, residual=x, out_shape=(1, T, self.cfg.dim))

        # 5. Fused FFN RMSNorm + Q8_1 Quantization
        hq, hds = rmsnorm_quantize_q8_1(h, self.ffn_norm_w, self.cfg.norm_eps)

        # 6. Fused Gate+Up __dp4a Projection + Fused SwiGLU + Q8_1 Quantization
        if self.gate_up_fused is not None:
            wgu = self.gate_up_fused
            gate_up = gguf_quant_gemv(hq, hds, wgu.raw, wgu.ggml_type, wgu.out_features, wgu.in_features, out_shape=(1, T, wgu.out_features))
        else:
            assert self.ffn_gate is not None and self.ffn_up is not None
            wg, wu = self.ffn_gate, self.ffn_up
            g = gguf_quant_gemv(hq, hds, wg.raw, wg.ggml_type, wg.out_features, wg.in_features, out_shape=(1, T, wg.out_features))
            u = gguf_quant_gemv(hq, hds, wu.raw, wu.ggml_type, wu.out_features, wu.in_features, out_shape=(1, T, wu.out_features))
            gate_up = g.cat(u, dim=-1).contiguous()

        sq, sds = swiglu_quantize_q8_1(gate_up)

        # 7. Down Projection with Fused Residual Add (`out = h + ffn_down(swiglu)`)
        wd = self.ffn_down
        return gguf_quant_gemv(sq, sds, wd.raw, wd.ggml_type, wd.out_features, wd.in_features, bias=wd.bias, residual=h, out_shape=(1, T, self.cfg.dim))


class FastQwenEngine:
    def __init__(self, model_ref: str, max_context: int = 4096):
        t0 = time.perf_counter()
        resolved = QWEN_MODELS.get(model_ref, model_ref)
        if resolved.startswith(("http://", "https://")):
            local_path = Path(fetch(resolved))
        else:
            local_path = Path(resolved).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Model file not found: {local_path}")

        self.local_path = local_path
        kv, t_infos, data_start = parse_gguf_header(local_path)
        self.kv = kv
        arch = kv["general.architecture"]
        if arch not in ("qwen2", "qwen3", "llama"):
            raise ValueError(f"Unsupported architecture '{arch}' in FastQwenEngine (expected qwen2/qwen3/llama)")

        ctx_len = min(max_context, int(kv[f"{arch}.context_length"]))
        n_heads = int(kv[f"{arch}.attention.head_count"])
        n_kv_heads = int(kv[f"{arch}.attention.head_count_kv"])
        dim = int(kv[f"{arch}.embedding_length"])
        head_dim = int(kv.get(f"{arch}.attention.key_length", dim // n_heads))
        rope_dim = int(kv.get(f"{arch}.rope.dimension_count", head_dim))
        has_qk_norm = "blk.0.attn_q_norm.weight" in t_infos

        self.config = QwenConfig(
            arch=arch,
            num_blocks=int(kv[f"{arch}.block_count"]),
            dim=dim,
            hidden_dim=int(kv[f"{arch}.feed_forward_length"]),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            rope_theta=float(kv[f"{arch}.rope.freq_base"]),
            norm_eps=float(kv[f"{arch}.attention.layer_norm_rms_epsilon"]),
            vocab_size=len(kv["tokenizer.ggml.tokens"]),
            max_context=ctx_len,
            has_qk_norm=has_qk_norm,
        )
        self.model_name = kv.get("general.name") or kv.get("general.basename") or local_path.stem
        self.tok = SimpleTokenizer.from_gguf_kv(kv)
        self.bos_id: int | None = kv.get("tokenizer.ggml.bos_token_id") if kv.get("tokenizer.ggml.add_bos_token", True) else None
        self.eos_id: int = int(kv["tokenizer.ggml.eos_token_id"])
        self.im_end_id: int = int(self.tok._special_tokens.get("<|im_end|>", self.eos_id))
        self.stop_ids: set[int] = {self.eos_id, self.im_end_id} | {
            int(tid) for s, tid in self.tok._special_tokens.items() if s in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")
        }

        # Stream GGUF to NV VRAM via high-speed DMA (~2.5 GB/s over Thunderbolt 3)
        raw_gpu = Tensor(local_path).to("NV").realize()
        file_nbytes = local_path.stat().st_size

        def load_f16_or_f32_1d(name: str, as_f32: bool = False) -> Tensor | None:
            if name not in t_infos:
                return None
            dims, typ, off = t_infos[name]
            nel = prod(dims)
            t = ggml_data_to_tensor(raw_gpu[data_start + off:], nel, typ).reshape(*reversed(dims))
            return (t.float() if as_f32 else t.half()).contiguous().realize()

        def load_quant_linear(name: str, bias_name: str | None = None) -> QuantWeight:
            dims, typ, off = t_infos[name]
            in_f, out_f = int(dims[0]), int(dims[1])
            nel = in_f * out_f
            bias_t = load_f16_or_f32_1d(bias_name, as_f32=True) if bias_name else None
            if typ in SUPPORTED_RAW_TYPES:
                bsz = SUPPORTED_RAW_TYPES[typ]
                nbytes = (nel // bsz[0]) * bsz[1]
                raw_slice = raw_gpu[data_start + off : data_start + off + nbytes].contiguous().realize()
                return QuantWeight(raw=raw_slice, ggml_type=typ, out_features=out_f, in_features=in_f, bias=bias_t)
            # Fallback: convert non-(Q4_K/Q6_K/Q8_0) linear weight to Q8_0 once on GPU
            w_deq = ggml_data_to_tensor(raw_gpu[data_start + off:], nel, typ).reshape(out_f, in_f)
            q8_raw = quantize_f32_to_q8_0_bytes(w_deq)
            return QuantWeight(raw=q8_raw, ggml_type=8, out_features=out_f, in_features=in_f, bias=bias_t)

        def fuse_quant_weights(weights: list[QuantWeight]) -> QuantWeight | None:
            if len({w.ggml_type for w in weights}) != 1:
                return None
            has_any_bias = any(w.bias is not None for w in weights)
            fused_bias = None
            if has_any_bias:
                b_parts = [
                    w.bias if w.bias is not None else Tensor.zeros(w.out_features, dtype=dtypes.float32, device="NV")
                    for w in weights
                ]
                fused_bias = b_parts[0].cat(*b_parts[1:], dim=0).contiguous().realize()
            fused_raw = weights[0].raw.cat(*[w.raw for w in weights[1:]], dim=0).contiguous().realize()
            return QuantWeight(
                raw=fused_raw,
                ggml_type=weights[0].ggml_type,
                out_features=sum(w.out_features for w in weights),
                in_features=weights[0].in_features,
                bias=fused_bias,
            )

        # Embedding & Output head
        emb_dims, emb_typ, emb_off = t_infos["token_embd.weight"]
        if emb_typ in SUPPORTED_RAW_TYPES:
            bsz = SUPPORTED_RAW_TYPES[emb_typ]
            emb_nbytes = (prod(emb_dims) // bsz[0]) * bsz[1]
            self.emb_raw = raw_gpu[data_start + emb_off : data_start + emb_off + emb_nbytes].contiguous().realize()
            self.emb_type = emb_typ
        else:
            w_emb = ggml_data_to_tensor(raw_gpu[data_start + emb_off:], prod(emb_dims), emb_typ).reshape(int(emb_dims[1]), int(emb_dims[0]))
            self.emb_raw = quantize_f32_to_q8_0_bytes(w_emb)
            self.emb_type = 8

        if "output.weight" in t_infos:
            self.lm_head = load_quant_linear("output.weight")
        else:
            self.lm_head = QuantWeight(
                raw=self.emb_raw, ggml_type=self.emb_type,
                out_features=self.config.vocab_size, in_features=self.config.dim,
            )
        self.output_norm_w = load_f16_or_f32_1d("output_norm.weight", as_f32=False)
        assert self.output_norm_w is not None

        kv_alloc_ctx = ctx_len + PREFILL_CHUNK
        freqs_cis = precompute_freqs_cis(self.config.rope_dim, kv_alloc_ctx, self.config.rope_theta).float().contiguous().realize()

        self.blocks: list[FastQwenBlock] = []
        for i in range(self.config.num_blocks):
            p = f"blk.{i}"
            attn_norm_w = load_f16_or_f32_1d(f"{p}.attn_norm.weight")
            ffn_norm_w = load_f16_or_f32_1d(f"{p}.ffn_norm.weight")
            assert attn_norm_w is not None and ffn_norm_w is not None

            wq = load_quant_linear(f"{p}.attn_q.weight", f"{p}.attn_q.bias")
            wk = load_quant_linear(f"{p}.attn_k.weight", f"{p}.attn_k.bias")
            wv = load_quant_linear(f"{p}.attn_v.weight", f"{p}.attn_v.bias")
            wo = load_quant_linear(f"{p}.attn_output.weight", f"{p}.attn_output.bias")

            qkv_fused = fuse_quant_weights([wq, wk, wv])
            qk_fused = None
            if qkv_fused is None:
                qk_fused = fuse_quant_weights([wq, wk])
                if qk_fused is not None:
                    wq = wk = None  # free individual slices

            wg = load_quant_linear(f"{p}.ffn_gate.weight")
            wu = load_quant_linear(f"{p}.ffn_up.weight")
            wd = load_quant_linear(f"{p}.ffn_down.weight")
            gate_up_fused = fuse_quant_weights([wg, wu])
            if gate_up_fused is not None:
                wg = wu = None

            qnw = load_f16_or_f32_1d(f"{p}.attn_q_norm.weight") if has_qk_norm else None
            knw = load_f16_or_f32_1d(f"{p}.attn_k_norm.weight") if has_qk_norm else None

            self.blocks.append(FastQwenBlock(
                config=self.config,
                attn_norm_w=attn_norm_w,
                ffn_norm_w=ffn_norm_w,
                qkv_fused=qkv_fused,
                qk_fused=qk_fused,
                v_proj=wv if qkv_fused is None else None,
                q_proj=wq if (qkv_fused is None and qk_fused is None) else None,
                k_proj=wk if (qkv_fused is None and qk_fused is None) else None,
                attn_output=wo,
                gate_up_fused=gate_up_fused,
                ffn_gate=wg,
                ffn_up=wu,
                ffn_down=wd,
                q_norm_w=qnw,
                k_norm_w=knw,
                freqs_cis=freqs_cis,
                kv_alloc_ctx=kv_alloc_ctx,
            ))

        # Release initial raw file buffer now that fused layer slices are realized
        del raw_gpu
        gc.collect()
        Device["NV"].synchronize()

        self._cached_tokens: list[int] = []
        self._v_start_pos = UOp.variable("start_pos", 0, ctx_len - 1)
        self._v_valid_last = UOp.variable("valid_last", 0, PREFILL_CHUNK - 1)

        # Specialized single-HCQGraph JITs for T=32 chunked prefill and T=1 token rollout
        self._prefill_greedy_jit = TinyJit(self._forward_prefill_greedy)
        self._decode_greedy_jit = TinyJit(self._forward_decode_greedy)
        self._prefill_sample_jit = TinyJit(self._forward_prefill_sample)
        self._decode_sample_jit = TinyJit(self._forward_decode_sample)

        load_dt = time.perf_counter() - t0
        print(
            f"[FastQwenEngine] Loaded '{self.model_name}' ({file_nbytes / 1e9:.2f} GB, "
            f"{self.config.num_blocks} layers, ctx={ctx_len}) on {Device['NV'].arch} in {load_dt:.2f}s"
        )

    def _compute_last_logits(self, x_last: Tensor) -> Tensor:
        oq, ods = rmsnorm_quantize_q8_1(x_last, self.output_norm_w, self.config.norm_eps)
        lh = self.lm_head
        return gguf_quant_gemv(oq, ods, lh.raw, lh.ggml_type, lh.out_features, lh.in_features, out_shape=(1, lh.out_features))

    def _sample_gumbel(self, logits: Tensor, temp: Tensor) -> Tensor:
        noisy = logits / temp.maximum(1e-6) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()
        return fast_argmax_1d(noisy)

    def _forward_prefill_greedy(self, chunk_tokens: Tensor, start_pos: UOp, valid_last: UOp) -> Tensor:
        x = gguf_embedding_lookup(chunk_tokens, self.emb_raw, self.emb_type, self.config.dim)
        for blk in self.blocks:
            x = blk.forward(x, start_pos, is_decode=False)
        x_last = select_token_row(x, valid_last)
        logits = self._compute_last_logits(x_last)
        return fast_argmax_1d(logits).realize()

    def _forward_prefill_sample(self, chunk_tokens: Tensor, start_pos: UOp, valid_last: UOp, temp: Tensor) -> Tensor:
        x = gguf_embedding_lookup(chunk_tokens, self.emb_raw, self.emb_type, self.config.dim)
        for blk in self.blocks:
            x = blk.forward(x, start_pos, is_decode=False)
        x_last = select_token_row(x, valid_last)
        logits = self._compute_last_logits(x_last)
        return self._sample_gumbel(logits, temp).realize()

    def _forward_decode_greedy(self, token: Tensor, start_pos: UOp) -> Tensor:
        x = gguf_embedding_lookup(token, self.emb_raw, self.emb_type, self.config.dim)
        for blk in self.blocks:
            x = blk.forward(x, start_pos, is_decode=True)
        logits = self._compute_last_logits(x)
        return fast_argmax_1d(logits).realize()

    def _forward_decode_sample(self, token: Tensor, start_pos: UOp, temp: Tensor) -> Tensor:
        x = gguf_embedding_lookup(token, self.emb_raw, self.emb_type, self.config.dim)
        for blk in self.blocks:
            x = blk.forward(x, start_pos, is_decode=True)
        logits = self._compute_last_logits(x)
        return self._sample_gumbel(logits, temp).realize()

    def warmup(self) -> None:
        t0 = time.perf_counter()
        dummy_chunk = Tensor([0] * PREFILL_CHUNK, dtype=dtypes.int32, device="NV").reshape(1, PREFILL_CHUNK).realize()
        dummy_tok = Tensor([[0]], dtype=dtypes.int32, device="NV").realize()
        for i in range(3):
            self._prefill_greedy_jit(dummy_chunk, self._v_start_pos.bind(0), self._v_valid_last.bind(0))
            self._decode_greedy_jit(dummy_tok, self._v_start_pos.bind(i + 1))
        Device["NV"].synchronize()
        self._cached_tokens = []
        print(f"[FastQwenEngine] Warmup & single-HCQGraph capture finished in {time.perf_counter() - t0:.2f}s")

    def get_start_pos(self, tokens: list[int]) -> int:
        return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))

    def reset_cache(self) -> None:
        self._cached_tokens = []

    def encode_chat(self, messages: list[dict[str, str]], enable_thinking: bool = False) -> list[int]:
        ids: list[int] = [self.bos_id] if self.bos_id is not None else []
        for msg in messages:
            ids += self.tok.role(msg["role"])
            ids += self.tok.encode(msg["content"])
            ids += self.tok.end_turn(self.im_end_id)
        ids += self.tok.role("assistant")
        if not enable_thinking and "qwen3" in self.model_name.lower() and "instruct" not in self.model_name.lower():
            ids += self.tok.encode("<think>\n\n</think>\n\n")
        return ids

    def generate(self, tokens: list[int], max_tokens: int | None = None, temperature: float = 0.0):
        tokens = list(tokens)
        start_pos = self.get_start_pos(tokens)
        prompt_len = len(tokens)
        greedy = temperature <= 1e-6
        temp_t = None if greedy else Tensor([temperature], dtype=dtypes.float32, device="NV").realize()

        out_tok: Tensor | None = None
        generated = 0

        # Phase 1: Prompt Prefill
        while start_pos < prompt_len:
            rem = prompt_len - start_pos
            # For single-token prompt (or 1 remaining token), use fast T=1 decode graph directly!
            if rem == 1:
                tok_in = Tensor([[tokens[start_pos]]], dtype=dtypes.int32, device="NV").contiguous()
                sp = self._v_start_pos.bind(start_pos)
                out_tok = self._decode_greedy_jit(tok_in, sp) if greedy else self._decode_sample_jit(tok_in, sp, temp_t)
                start_pos += 1
            else:
                take = min(PREFILL_CHUNK, rem)
                chunk_ids = tokens[start_pos : start_pos + take] + [0] * (PREFILL_CHUNK - take)
                chunk_in = Tensor([chunk_ids], dtype=dtypes.int32, device="NV").contiguous()
                sp = self._v_start_pos.bind(start_pos)
                vl = self._v_valid_last.bind(take - 1)
                out_tok = (
                    self._prefill_greedy_jit(chunk_in, sp, vl)
                    if greedy
                    else self._prefill_sample_jit(chunk_in, sp, vl, temp_t)
                )
                start_pos += take

        assert out_tok is not None
        next_id = int(out_tok.item())
        tokens.append(next_id)
        self._cached_tokens = tokens[:-1]
        generated += 1
        yield next_id
        if max_tokens is not None and generated >= max_tokens:
            return

        # Phase 2: Autoregressive Rollout (GPU-to-GPU out_tok buffer piped directly into decode_jit!)
        while len(tokens) < self.config.max_context:
            sp = self._v_start_pos.bind(start_pos)
            out_tok = self._decode_greedy_jit(out_tok, sp) if greedy else self._decode_sample_jit(out_tok, sp, temp_t)
            start_pos += 1
            next_id = int(out_tok.item())
            tokens.append(next_id)
            self._cached_tokens = tokens[:-1]
            generated += 1
            yield next_id
            if max_tokens is not None and generated >= max_tokens:
                return


def make_http_handler(engine: FastQwenEngine):
    class FastHandler(HTTPRequestHandler):
        def log_request(self, code="-", size="-"):
            pass

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_data(json.dumps({"object": "list", "data": [{"id": engine.model_name, "object": "model"}]}).encode())
            else:
                from tinygrad.apps.llm import CHAT_HTML
                self.send_data(CHAT_HTML, content_type="text/html")

        def run_model(self, ids: list[int], req_model: str, include_usage=False, max_tokens: int | None = None, temperature: float = 0.0):
            cache_start_pos = engine.get_start_pos(ids)
            stderr_log(
                f"{self.path}  {colored('--', 'BLACK')}  "
                f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  "
            )
            tmpl = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": req_model}
            yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}], **tmpl}
            out: list[int] = []
            finish_reason = "stop"
            st = time.perf_counter()
            pt = st
            dec = engine.tok.stream_decoder()
            for next_id in engine.generate(ids, max_tokens=max_tokens, temperature=temperature):
                if len(out) == 0:
                    pt = time.perf_counter()
                    stderr_log(f"prefill:{(len(ids)-cache_start_pos)/max(pt-st, 1e-6):5.0f} tok/s  {colored('--', 'BLACK')}  ")
                if next_id in engine.stop_ids:
                    break
                out.append(next_id)
                yield {"choices": [{"index": 0, "delta": {"content": dec(next_id)}, "finish_reason": None}], **tmpl}
            else:
                if max_tokens is not None and len(out) >= max_tokens:
                    finish_reason = "length"
            if (tail := dec()):
                yield {"choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}], **tmpl}
            yield {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], **tmpl}
            if include_usage:
                yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
            et = time.perf_counter()
            stderr_log(
                f"gen:{(len(out)-1)/max(et-pt, 1e-6) if len(out) > 1 else 0:5.1f} tok/s  {colored('--', 'BLACK')}  "
                f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n"
            )

        def do_POST(self):
            raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.loads(raw_body.decode("utf-8"))
            if self.path == "/v1/chat/completions":
                ids = engine.encode_chat(body["messages"])
                max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
                chunks = self.run_model(
                    ids, body.get("model", engine.model_name),
                    not body.get("stream") or body.get("stream_options", {}).get("include_usage", False),
                    max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                )
                if body.get("stream"):
                    self.stream_json(chunks)
                else:
                    out, finish_reason, last_c = [], "stop", {}
                    for c in chunks:
                        last_c = c
                        if c["choices"] and c["choices"][0].get("delta", {}).get("content"):
                            out.append(c["choices"][0]["delta"]["content"])
                        if c["choices"] and c["choices"][0].get("finish_reason"):
                            finish_reason = c["choices"][0]["finish_reason"]
                    self.send_data(json.dumps({
                        **last_c, "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(out)}, "finish_reason": finish_reason}],
                    }).encode())
            else:
                raise RuntimeError(f"Unhandled path {self.path}")

    return FastHandler


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Qwen Engine on macOS NVIDIA eGPU (TinyGPU + Ampere __dp4a)")
    parser.add_argument("--model", "-m", default="qwen3:4b", help=f"Preset ({', '.join(QWEN_MODELS.keys())}) or URL / local GGUF path")
    parser.add_argument("--max-context", type=int, default=4096, help="Max KV context length (default: 4096)")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Run a single chat prompt and print response + speed metrics")
    parser.add_argument("--max-tokens", "-n", type=int, default=128, help="Max tokens to generate for --prompt (default: 128)")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Sampling temperature (default: 0.0 greedy fast-path)")
    parser.add_argument("--benchmark", nargs="?", type=int, const=30, metavar="COUNT", help="Benchmark raw token generation speed")
    parser.add_argument("--serve", nargs="?", type=int, const=8000, metavar="PORT", help="Serve OpenAI-compatible API + Web UI on PORT")
    args = parser.parse_args()

    engine = FastQwenEngine(args.model, max_context=args.max_context)

    if args.benchmark is not None:
        gen = engine.generate(toks := [engine.bos_id or 0], max_tokens=args.benchmark, temperature=0.0)
        times_ns: list[int] = []
        for idx in range(args.benchmark):
            GlobalCounters.reset()
            t_st = time.perf_counter_ns()
            tok_id = next(gen)
            dt_ns = time.perf_counter_ns() - t_st
            times_ns.append(dt_ns)
            toks.append(tok_id)
            print(
                f"step {idx+1:2d}: {dt_ns/1e6:7.2f} ms, {1e9/dt_ns:7.2f} tok/s, "
                f"{GlobalCounters.global_mem/dt_ns:7.2f} GB/s, "
                f"{GlobalCounters.global_mem//1_000_000}/{GlobalCounters.mem_used//1_000_000} MB"
            )
        if len(times_ns) > 3:
            steady = times_ns[3:]
            avg_ms = (sum(steady) / len(steady)) / 1e6
            best_ms = min(steady) / 1e6
            print(f"\nSteady-state avg: {avg_ms:.2f} ms/tok ({1000.0/avg_ms:.2f} tok/s) | Peak: {best_ms:.2f} ms/tok ({1000.0/best_ms:.2f} tok/s)")
        return

    if args.prompt is not None:
        engine.warmup()
        ids = engine.encode_chat([{"role": "user", "content": args.prompt}])
        dec = engine.tok.stream_decoder()
        t_start = time.perf_counter()
        t_first = None
        count = 0
        for tid in engine.generate(ids, max_tokens=args.max_tokens, temperature=args.temperature):
            if t_first is None:
                t_first = time.perf_counter()
            if tid in engine.stop_ids:
                break
            sys.stdout.write(dec(tid))
            sys.stdout.flush()
            count += 1
        sys.stdout.write(dec() + "\n")
        t_end = time.perf_counter()
        ttft_ms = ((t_first or t_end) - t_start) * 1000.0
        prefill_tps = len(ids) / max((t_first or t_end) - t_start, 1e-6)
        gen_tps = max(count - 1, 1) / max(t_end - (t_first or t_start), 1e-6)
        print(
            f"\n--- Metrics: prompt={len(ids)} toks (TTFT {ttft_ms:.1f} ms, {prefill_tps:.0f} tok/s prefill) | "
            f"generated={count} toks @ {gen_tps:.2f} tok/s ---"
        )
        return

    if args.serve is not None:
        engine.warmup()
        print(f"[FastQwenEngine] Serving OpenAI API + Chat UI at http://localhost:{args.serve}/")
        TCPServerWithReuse(("", args.serve), make_http_handler(engine)).serve_forever()
        return

    # Interactive CLI chat
    engine.warmup()
    ids = []
    while True:
        try:
            user_in = input(">>> ")
        except EOFError:
            break
        if not ids and engine.bos_id is not None:
            ids.append(engine.bos_id)
        ids += engine.tok.role("user") + engine.tok.encode(user_in) + engine.tok.end_turn(engine.im_end_id) + engine.tok.role("assistant")
        dec = engine.tok.stream_decoder()
        t0 = time.perf_counter()
        t1 = None
        n_out = 0
        for tid in engine.generate(ids, temperature=args.temperature):
            if t1 is None:
                t1 = time.perf_counter()
            if tid in engine.stop_ids:
                ids += engine.tok.end_turn(engine.im_end_id)
                sys.stdout.write(dec() + "\n")
                break
            ids.append(tid)
            sys.stdout.write(dec(tid))
            sys.stdout.flush()
            n_out += 1
        dt_gen = time.perf_counter() - (t1 or t0)
        if n_out > 1:
            print(f"[{n_out} tokens @ {(n_out-1)/dt_gen:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
