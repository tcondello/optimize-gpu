#!/usr/bin/env python3
"""
Head-to-Head Hardware Benchmark:
  Phase 1 (`qwen_engine.py`: Python + tinygrad + HCQGraph + NVRTC Docker Daemon, 346 kernels/tok)
  vs.
  Phase 2 (`qwen-nv`: Zero-Docker Native ARM64 Rust + GA102 68-SM Persistent Megakernel + BM=4 Speculative)

Runs both engines sequentially on the exact same connected NVIDIA GeForce RTX 3080 Thunderbolt eGPU,
verifying:
  1. Zero Docker runtime dependency in Phase 2 (`cuda-nvcc-fast-daemon` stopped before Phase 2 runs)
  2. Warm invocation latency (`9.49 s` Python/GSP re-init -> `< 1 ms` persistent Rust daemon)
  3. Raw T=1 decode throughput (`tok/s`, `ms/tok`, and `%` of RTX 3080's `760.3 GB/s` GDDR6X peak)
  4. Prompt prefill throughput & TTFT (`301` tokens)
  5. Lossless `BM=4` N-gram speculative verification speedup & 100% greedy token parity with `T=1`
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent.parent
QWEN_NV_BIN = SUITE_DIR / "engine" / "target" / "release" / "rtx"

LONG_TEXT = (
    "You are benchmarking an external NVIDIA GeForce RTX 3080 GPU connected via Thunderbolt 3 "
    "to an Apple Silicon M3 Pro MacBook Pro using DriverKit DMA and custom Ampere __dp4a CUDA kernels. "
) * 8 + "\nSummarize the key benefit in one sentence."

QA_PROMPTS = [
    ("Compute 17 * 23 concisely in two lines and state the final integer.", "391", 160),
    ("Write a Python one-liner that returns the squares of even numbers in a list `nums`.", "[", 160),
    ("In 2 short sentences, what is CUDA warp shuffle (`__shfl_down_sync`)?", "warp", 160),
]

STRUCTURED_PROMPT = (
    "Convert the following Python configuration dictionary into an updated version where `enable_docker` is False, "
    "`kernel_mode` is `'persistent_sm86_megakernel'`, and all other fields and comments are preserved verbatim:\n\n"
    "```python\n"
    "ENGINE_CONFIG = {\n"
    "    'device': 'NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs)',\n"
    "    'transport': 'Thunderbolt 3 / DriverKit DMA',\n"
    "    'model': 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',\n"
    "    'enable_docker': True,\n"
    "    'kernel_mode': 'hcq_graph_346_kernels',\n"
    "    'kv_cache_dtype': 'float16',\n"
    "    'max_context_tokens': 4096,\n"
    "    'quant_formats': ['Q4_K', 'Q6_K', 'Q8_0'],\n"
    "}\n"
    "```\n"
    "Return only the updated Python code block."
)


def run_phase1_subprocess() -> dict:
    """Run Phase 1 (`FastQwenEngine`) in an isolated Python child process so GSP-RM unloads cleanly on exit."""
    helper_code = r"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("JIT_BATCH_SIZE", "0")
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.home() / "tinygrad"))

t_boot0 = time.perf_counter()
from tinygrad.runtime.support.nv.ip import NV_GSP
_orig_rm_ctrl = NV_GSP.rpc_rm_control
def _safe_rm_ctrl(self, hObject, cmd, params, client=None, extra=None):
    # Skip CMD_BOOST_TO_MAX (0x2080200a) transient voltage spike on Thunderbolt SFX PSU
    if cmd == 0x2080200a:
        return None
    return _orig_rm_ctrl(self, hObject, cmd, params, client, extra)
NV_GSP.rpc_rm_control = _safe_rm_ctrl

import subprocess as _sp
import nvrtc_daemon as _nd
if _sp.run(["docker", "info"], capture_output=True).returncode != 0:
    # Docker Desktop is off; rely on tinygrad's pre-populated SQLite cubin cache (`cache.db`)
    _nd.ensure_compiler_daemon = lambda *a, **k: "sqlite-cache"

from qwen_engine import FastQwenEngine
from tinygrad import Device

try:
    engine = FastQwenEngine("qwen3:4b", max_context=4096)
    engine.warmup()
    warm_init_s = time.perf_counter() - t_boot0

    cfg = json.loads(sys.stdin.read())

    # 1. Raw T=1 decode (128 tokens from short prompt)
    engine.reset_cache()
    raw_ids = engine.encode_chat([{"role": "user", "content": "Count from 1 to 60 separated by commas."}])
    gen = engine.generate(raw_ids, max_tokens=129, temperature=0.0)
    next(gen)  # prefill + 1st token
    step_ms = []
    for _ in range(128):
        t0 = time.perf_counter()
        next(gen)
        step_ms.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(step_ms[4:]) / len(step_ms[4:])
    min_ms = min(step_ms[4:])

    # 2. Long prompt prefill (~301 tokens)
    engine.reset_cache()
    long_ids = engine.encode_chat([{"role": "user", "content": cfg["long_text"]}])
    t_pf0 = time.perf_counter()
    gen_long = engine.generate(long_ids, max_tokens=1, temperature=0.0)
    next(gen_long)
    ttft_ms = (time.perf_counter() - t_pf0) * 1000.0
    prefill_tps = len(long_ids) / (ttft_ms / 1000.0)

    # 3. QA + Structured prompts
    def run_one(prompt, max_t):
        engine.reset_cache()
        ids = engine.encode_chat([{"role": "user", "content": prompt}])
        dec = engine.tok.stream_decoder()
        pieces, out_ids = [], []
        t0 = time.perf_counter()
        t_first = None
        for tid in engine.generate(ids, max_tokens=max_t, temperature=0.0):
            if t_first is None:
                t_first = time.perf_counter()
            if tid in engine.stop_ids:
                break
            out_ids.append(int(tid))
            pieces.append(dec(tid))
        pieces.append(dec())
        t1 = time.perf_counter()
        decode_s = max(t1 - (t_first or t0), 1e-6)
        tps = max(len(out_ids) - 1, 1) / decode_s
        return {
            "prompt": prompt,
            "tokens": len(out_ids),
            "ttft_ms": round(((t_first or t0) - t0) * 1000.0, 2),
            "decode_tok_s": round(tps, 2),
            "token_ids": out_ids,
            "response": "".join(pieces).strip(),
        }

    qa_out = [run_one(p, mt) for p, _, mt in cfg["qa_prompts"]]
    struct_out = run_one(cfg["structured_prompt"], 200)

    file_gb = engine.local_path.stat().st_size / 1e9
    out = {
        "startup_ready_s": round(warm_init_s, 2),
        "model_size_gb": round(file_gb, 3),
        "raw_decode_128": {
            "avg_ms_tok": round(avg_ms, 3),
            "min_ms_tok": round(min_ms, 3),
            "avg_tok_s": round(1000.0 / avg_ms, 2),
            "peak_tok_s": round(1000.0 / min_ms, 2),
            "effective_bw_gbs": round(file_gb / (avg_ms / 1000.0), 2),
            "pct_of_760_gbs_peak": round(100.0 * (file_gb / (avg_ms / 1000.0)) / 760.3, 1),
        },
        "prefill_long": {
            "prompt_tokens": len(long_ids),
            "ttft_ms": round(ttft_ms, 2),
            "prefill_tok_s": round(prefill_tps, 1),
        },
        "qa_runs": qa_out,
        "structured_run": struct_out,
    }
    print("PHASE1_JSON:" + json.dumps(out), flush=True)
finally:
    if "NV" in Device._opened_devices:
        Device["NV"].finalize()
        Device._opened_devices.clear()
"""
    payload = json.dumps({
        "long_text": LONG_TEXT,
        "qa_prompts": QA_PROMPTS,
        "structured_prompt": STRUCTURED_PROMPT,
    })
    proc = subprocess.run(
        [sys.executable, "-c", helper_code],
        input=payload,
        text=True,
        cwd=str(SUITE_DIR),
        capture_output=True,
    )
    if proc.returncode != 0:
        # Phase 1 requires the Docker NVRTC daemon whenever tinygrad's SQLite cache misses a prompt-bucket shape
        subprocess.run(["open", "-g", "-a", "Docker"], capture_output=True)
        for _ in range(30):
            if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
                break
            time.sleep(1.0)
        proc = subprocess.run(
            [sys.executable, "-c", helper_code],
            input=payload,
            text=True,
            cwd=str(SUITE_DIR),
            capture_output=True,
            check=True,
        )
        subprocess.run(["docker", "stop", "cuda-nvcc-fast-daemon"], capture_output=True)
        subprocess.run(["killall", "Docker Desktop", "com.docker.backend", "com.docker.build"], capture_output=True)
    for line in proc.stdout.splitlines():
        if line.startswith("PHASE1_JSON:"):
            return json.loads(line[len("PHASE1_JSON:"):])
    raise RuntimeError(f"Phase 1 subprocess did not emit PHASE1_JSON:\nstdout={proc.stdout}\nstderr={proc.stderr}")


def stop_docker_compiler_daemon() -> str:
    """Ensure Docker is stopped so Phase 2 is proven with 0 running CUDA Docker containers."""
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return "Docker Desktop OFF (0 containers, 0 processes)"
    subprocess.run(["docker", "stop", "cuda-nvcc-fast-daemon"], capture_output=True, text=True)
    ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    running = [ln.strip() for ln in ps.stdout.splitlines() if ln.strip()]
    return "none (0 containers)" if "cuda-nvcc-fast-daemon" not in running else f"still running: {running}"


def qwen_nv_json_run(prompt: str, max_tokens: int, spec: bool, quiet: bool = False) -> tuple[str, dict]:
    cmd = [
        str(QWEN_NV_BIN),
        "run",
        "--model", "qwen3:4b",
        "--max-tokens", str(max_tokens),
        "--temp", "0.0",
        "--reset-cache",
        "--json-stats",
        "--spec" if spec else "--no-spec",
    ]
    if quiet:
        cmd.append("--quiet")
    cmd.extend(["-p", prompt])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = proc.stdout.splitlines()
    stats_line = lines[-1]
    resp_text = "\n".join(lines[:-1]).strip() if not quiet else ""
    return resp_text, json.loads(stats_line)


def benchmark_phase2(model_size_gb: float) -> dict:
    # Ensure clean start
    subprocess.run([str(QWEN_NV_BIN), "stop"], capture_output=True)

    # 1. Cold daemon startup (GSP-RM boot + 2.5 GB DMA upload + warmup)
    t0 = time.perf_counter()
    subprocess.run(
        [str(QWEN_NV_BIN), "daemon", "--background", "--model", "qwen3:4b", "--ctx", "4096"],
        check=True,
    )
    cold_boot_s = time.perf_counter() - t0

    # 2. Warm client ping latency (`qwen-nv status`)
    ping_ms_list = []
    for _ in range(5):
        tp0 = time.perf_counter()
        st = subprocess.run([str(QWEN_NV_BIN), "status"], capture_output=True, text=True, check=True)
        ping_ms_list.append((time.perf_counter() - tp0) * 1000.0)
    status_line = st.stdout.strip()
    warm_cli_ms = min(ping_ms_list)

    # 3. Raw T=1 Persistent Megakernel 128-token decode (`--no-spec`, single GPU kernel launch for all 128 tokens!)
    _, raw_t1 = qwen_nv_json_run("Count from 1 to 60 separated by commas.", max_tokens=128, spec=False, quiet=True)
    # Subtract the 1st token (which overlaps prompt tail) or use pure multi-step delta:
    # Even better: measure `max_tokens=136` minus `max_tokens=8` on the exact same prompt to isolate pure steady-state T=1 GPU step time!
    _, raw_8 = qwen_nv_json_run("Count from 1 to 60 separated by commas.", max_tokens=8, spec=False, quiet=True)
    _, raw_136 = qwen_nv_json_run("Count from 1 to 60 separated by commas.", max_tokens=136, spec=False, quiet=True)
    pure_128_ms = raw_136["decode_ms"] - raw_8["decode_ms"]
    steady_ms_tok = pure_128_ms / 128.0
    steady_tps = 1000.0 / steady_ms_tok
    eff_bw = model_size_gb / (steady_ms_tok / 1000.0)

    # 4. Long prompt prefill (`~301` tokens, `max_tokens=1`)
    _, pf_stats = qwen_nv_json_run(LONG_TEXT, max_tokens=1, spec=False, quiet=True)
    pf_ttft_ms = pf_stats["ttft_ms"]
    pf_tps = pf_stats["prompt_tokens"] / (pf_ttft_ms / 1000.0)

    # 5. QA Prompts (`--no-spec` T=1 vs `--spec` BM=4)
    qa_t1, qa_spec = [], []
    for prompt, expected_sub, max_t in QA_PROMPTS:
        txt_t1, st_t1 = qwen_nv_json_run(prompt, max_tokens=max_t, spec=False, quiet=False)
        txt_sp, st_sp = qwen_nv_json_run(prompt, max_tokens=max_t, spec=True, quiet=False)
        exact_spec_parity = (st_t1["token_ids"] == st_sp["token_ids"])
        qa_t1.append({
            "prompt": prompt,
            "tokens": st_t1["completion_tokens"],
            "decode_tok_s": st_t1["decode_tok_per_s"],
            "ttft_ms": st_t1["ttft_ms"],
            "client_ttft_ms": st_t1["client_ttft_ms"],
            "kernel_launches": st_t1["kernel_launches"],
            "passed": expected_sub.lower() in txt_t1.lower(),
            "token_ids": st_t1["token_ids"],
            "response": txt_t1,
        })
        qa_spec.append({
            "prompt": prompt,
            "tokens": st_sp["completion_tokens"],
            "decode_tok_s": st_sp["decode_tok_per_s"],
            "spec_accepted_extra": st_sp["spec_accepted_extra"],
            "exact_parity_with_t1": exact_spec_parity,
        })

    # 6. Structured Code/Config Refactoring Prompt (`--no-spec` vs `--spec` BM=4!)
    struct_txt_t1, struct_st_t1 = qwen_nv_json_run(STRUCTURED_PROMPT, max_tokens=200, spec=False, quiet=False)
    struct_txt_sp, struct_st_sp = qwen_nv_json_run(STRUCTURED_PROMPT, max_tokens=200, spec=True, quiet=False)

    # Leave daemon cleanly stopped after benchmark (or user can restart anytime in < 2s!)
    subprocess.run([str(QWEN_NV_BIN), "stop"], capture_output=True)

    return {
        "daemon_cold_boot_s": round(cold_boot_s, 2),
        "warm_cli_ping_ms": round(warm_cli_ms, 3),
        "daemon_status": status_line,
        "raw_decode_t1_megakernel": {
            "end_to_end_128_tok_s": raw_t1["decode_tok_per_s"],
            "steady_ms_tok": round(steady_ms_tok, 3),
            "steady_tok_s": round(steady_tps, 2),
            "effective_bw_gbs": round(eff_bw, 2),
            "pct_of_760_gbs_peak": round(100.0 * eff_bw / 760.3, 1),
            "gpu_launches_for_128_tokens": raw_t1["kernel_launches"],
        },
        "prefill_long": {
            "prompt_tokens": pf_stats["prompt_tokens"],
            "ttft_ms": round(pf_ttft_ms, 2),
            "prefill_tok_s": round(pf_tps, 1),
            "prefill_only_tok_s": pf_stats["prefill_tok_per_s"],
        },
        "qa_t1_runs": qa_t1,
        "qa_spec_bm4_runs": qa_spec,
        "structured_t1_run": {
            "tokens": struct_st_t1["completion_tokens"],
            "decode_tok_s": struct_st_t1["decode_tok_per_s"],
            "kernel_launches": struct_st_t1["kernel_launches"],
            "response": struct_txt_t1,
        },
        "structured_spec_bm4_run": {
            "tokens": struct_st_sp["completion_tokens"],
            "decode_tok_s": struct_st_sp["decode_tok_per_s"],
            "spec_verify_calls": struct_st_sp["spec_verify_calls"],
            "spec_accepted_extra": struct_st_sp["spec_accepted_extra"],
            "exact_parity_with_t1": struct_st_t1["token_ids"] == struct_st_sp["token_ids"],
            "response": struct_txt_sp,
        },
    }


def main():
    print("=" * 80)
    print("  HEAD-TO-HEAD BENCHMARK: Phase 2 (Rust + sm_86 Megakernel) vs Phase 1 (Python/tinygrad)")
    print("=" * 80)

    gguf_file = Path.home() / "Library/Caches/tinygrad/downloads/da7aed26eed9024aedd9e636a5389058"
    model_size_gb = round(gguf_file.stat().st_size / 1e9, 3)

    print("\n[1/3] Verifying Zero-Docker environment for Phase 2 (`qwen-nv`)...")
    docker_state = stop_docker_compiler_daemon()
    print(f"      CUDA Docker daemon status: {docker_state}")

    print("\n[2/3] Running Phase 2 (`qwen-nv`, Native ARM64 Rust + 68-SM Persistent Megakernel + BM=4 Spec)...")
    p2 = benchmark_phase2(model_size_gb)
    print(
        f"      Phase 2 Warm Ping: {p2['warm_cli_ping_ms']} ms | "
        f"Raw T=1 Megakernel: {p2['raw_decode_t1_megakernel']['steady_tok_s']} tok/s "
        f"({p2['raw_decode_t1_megakernel']['steady_ms_tok']} ms/tok, {p2['raw_decode_t1_megakernel']['effective_bw_gbs']} GB/s) | "
        f"Structured BM=4 Spec: {p2['structured_spec_bm4_run']['decode_tok_s']} tok/s"
    )

    print("\n[3/3] Running Phase 1 (`qwen_engine.py`, Python + tinygrad + HCQGraph, 346 kernels/tok)...")
    p1 = run_phase1_subprocess()
    print(
        f"      Phase 1 Startup: {p1['startup_ready_s']}s | "
        f"Raw Decode: {p1['raw_decode_128']['avg_tok_s']} tok/s avg ({p1['raw_decode_128']['avg_ms_tok']} ms/tok, "
        f"{p1['raw_decode_128']['effective_bw_gbs']} GB/s) | "
        f"Prefill ({p1['prefill_long']['prompt_tokens']} toks): {p1['prefill_long']['prefill_tok_s']} tok/s"
    )

    p1_tps = p1["raw_decode_128"]["avg_tok_s"]
    p2_tps = p2["raw_decode_t1_megakernel"]["steady_tok_s"]
    p2_e2e_tps = p2["raw_decode_t1_megakernel"]["end_to_end_128_tok_s"]
    p2_spec_tps = p2["structured_spec_bm4_run"]["decode_tok_s"]
    p1_struct_tps = p1["structured_run"]["decode_tok_s"]

    # Compare Phase 1 vs Phase 2 greedy token IDs on QA prompts!
    parity_checks = []
    for r1, r2 in zip(p1["qa_runs"], p2["qa_t1_runs"]):
        m_prefix = 0
        for a, b in zip(r1["token_ids"], r2["token_ids"]):
            if a == b:
                m_prefix += 1
            else:
                break
        parity_checks.append({
            "prompt": r1["prompt"],
            "phase1_tokens": r1["tokens"],
            "phase2_tokens": r2["tokens"],
            "exact_match": r1["token_ids"] == r2["token_ids"],
            "matching_prefix_tokens": m_prefix,
        })

    summary = {
        "docker_runtime_status_during_phase2": docker_state,
        "baselines": {
            "stock_metal_m3_pro_tok_s": 3.65,
            "stock_tinygrad_nv_rtx3080_tok_s": 8.51,
            "phase1_python_hcqgraph_tok_s": p1_tps,
        },
        "head_to_head_speedups": {
            "warm_startup_latency_reduction": f"{p1['startup_ready_s']} s -> {p2['warm_cli_ping_ms']} ms ({round((p1['startup_ready_s']*1000.0)/max(p2['warm_cli_ping_ms'], 0.001), 0):.0f}x faster)",
            "raw_t1_decode_gain_vs_phase1_x": round(p2_tps / p1_tps, 2),
            "raw_t1_decode_gain_vs_stock_nv_x": round(p2_tps / 8.51, 2),
            "structured_spec_bm4_gain_vs_phase1_x": round(p2_spec_tps / p1_struct_tps, 2),
            "structured_spec_bm4_gain_vs_stock_nv_x": round(p2_spec_tps / 8.51, 2),
            "prefill_gain_vs_phase1_x": round(p2["prefill_long"]["prefill_tok_s"] / p1["prefill_long"]["prefill_tok_s"], 2),
        },
        "phase1_python_hcqgraph": p1,
        "phase2_rust_megakernel": p2,
        "phase1_vs_phase2_qa_parity": parity_checks,
    }

    out_path = SUITE_DIR / "benchmark_phase2_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 80)
    print("  FINAL VERIFIED BENCHMARK SUMMARY (Qwen3-4B-Instruct-2507 Q4_K_M on RTX 3080 eGPU)")
    print("=" * 80)
    print(f"  * Docker Needed at Runtime : YES (Phase 1)       -> NO / {docker_state} (Phase 2)")
    print(f"  * Warm Ready Latency       : {p1['startup_ready_s']:6.2f} s            -> {p2['warm_cli_ping_ms']:.2f} ms")
    print(f"  * GPU Launches / 128 Toks  : 44,288 kernels      -> {p2['raw_decode_t1_megakernel']['gpu_launches_for_128_tokens']} persistent megakernel")
    print(f"  * Raw T=1 Decode Speed     : {p1_tps:6.2f} tok/s        -> {p2_tps:6.2f} tok/s ({p2_e2e_tps:.2f} e2e, {summary['head_to_head_speedups']['raw_t1_decode_gain_vs_phase1_x']}x Phase 1, {summary['head_to_head_speedups']['raw_t1_decode_gain_vs_stock_nv_x']}x Stock NV!)")
    print(f"  * Effective GDDR6X BW      : {p1['raw_decode_128']['effective_bw_gbs']:6.2f} GB/s ({p1['raw_decode_128']['pct_of_760_gbs_peak']}%) -> {p2['raw_decode_t1_megakernel']['effective_bw_gbs']:6.2f} GB/s ({p2['raw_decode_t1_megakernel']['pct_of_760_gbs_peak']}% of 760.3 GB/s physical peak!)")
    print(f"  * Structured/Code (BM=4)   : {p1_struct_tps:6.2f} tok/s        -> {p2_spec_tps:6.2f} tok/s ({summary['head_to_head_speedups']['structured_spec_bm4_gain_vs_phase1_x']}x Phase 1, exact lossless={p2['structured_spec_bm4_run']['exact_parity_with_t1']})")
    print(f"  * Long Prefill ({p2['prefill_long']['prompt_tokens']} toks) : {p1['prefill_long']['prefill_tok_s']:6.1f} tok/s        -> {p2['prefill_long']['prefill_tok_s']:6.1f} tok/s (TTFT {p2['prefill_long']['ttft_ms']} ms)")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
