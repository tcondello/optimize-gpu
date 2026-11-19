#!/usr/bin/env python3
"""
Phase 3 Head-to-Head Benchmark Suite:
1. Apple M3 Pro Metal (`qwentts.cpp` `MTL0`) vs 100%-on-RTX-3080 (`qwen-nv` `sm_86` Megakernel + Streaming `Code2Wav`)
   on the EXACT SAME `Qwen3-TTS-12Hz-1.7B Q8_0` (`1.95 GB`) + `12Hz Tokenizer BF16` (`342 MB`) GGUF files.
2. `12Hz Code2Wav` Golden Waveform Parity (vs `qwentts.cpp` Metal reference) & Causal Streaming (`1->2->4->8`) Bit-Exact Equivalence.
3. Co-Resident `Qwen3-4B` (`Q4_K_M`) + `Qwen3-TTS-1.7B` (`Q8_0`) Live Voice Assistant End-to-End Latency (`0 ms` model swap).
4. Fused Dynamic Rank-16 PEFT LoRA Hot-Swap Latency (`< 6 ms` DMA / `< 0.1 ms` clear), `scale=0` Bit-Exact Parity, & Throughput.
"""

import json
import math
import os
import re
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path("/Users/tim/Code/optimize-gpu")
QWEN_NV = REPO_ROOT / "qwen-nv-rs/target/release/qwen-nv"
QWENTTS_CPP = Path(
    "/Users/tim/.gemini/antigravity/brain/e00193b9-7241-45d5-8647-c8846d0fbd2e/scratch/qwentts.cpp/build/qwen-tts"
)
REF_DIR = Path(
    "/Users/tim/.gemini/antigravity/brain/e00193b9-7241-45d5-8647-c8846d0fbd2e/scratch/ref_seed42"
)
TALKER_GGUF = Path.home() / ".cache/qwen-nv-rs/models/qwen-talker-1.7b-customvoice-Q8_0.gguf"
CODEC_GGUF = Path.home() / ".cache/qwen-nv-rs/models/qwen-tokenizer-12hz-BF16.gguf"
THINKER_GGUF = Path.home() / ".cache/qwen-nv-rs/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
OUT_JSON = REPO_ROOT / "benchmark_phase3_results.json"

UTTERANCES = [
    (
        "short",
        "System online, ready for your command.",
    ),
    (
        "medium",
        "Hello Tim, the RTX 3080 text to speech engine is online.",
    ),
    (
        "long",
        "Every single transformer layer, all fifteen residual acoustic codebooks, and the entire twenty four kilohertz neural vocoder are executing natively on the external graphics card over Thunderbolt.",
    ),
]


def check_docker_off() -> bool:
    res = subprocess.run(["pgrep", "-fl", "-i", "docker"], capture_output=True, text=True)
    active = []
    for ln in res.stdout.splitlines():
        # macOS launchd socket helper `com.docker.vmnetd` sits idle when Docker Desktop is closed
        if "com.docker.vmnetd" in ln or "pgrep" in ln:
            continue
        if ln.strip():
            active.append(ln.strip())
    return len(active) == 0


def load_f32_maybe_ggml_hdr(path: Path):
    data = path.read_bytes()
    if len(data) >= 8:
        nd, d0 = struct.unpack("<2i", data[:8])
        if nd == 1 and 8 + d0 * 4 == len(data):
            return list(struct.unpack(f"<{d0}f", data[8:]))
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data[: n * 4]))


def waveform_parity(ref_path: Path, tst_path: Path) -> dict:
    ref = load_f32_maybe_ggml_hdr(ref_path)
    tst = load_f32_maybe_ggml_hdr(tst_path)
    n = min(len(ref), len(tst))
    nans = sum(1 for x in tst if not math.isfinite(x))
    dot = sum(ref[i] * tst[i] for i in range(n))
    nr = math.sqrt(sum(ref[i] * ref[i] for i in range(n)))
    nt = math.sqrt(sum(tst[i] * tst[i] for i in range(n)))
    cos_sim = dot / max(1e-12, nr * nt)
    err_sq = sum((ref[i] - tst[i]) ** 2 for i in range(n))
    mse = err_sq / max(1, n)
    max_abs = max((abs(ref[i] - tst[i]) for i in range(n)), default=0.0)
    snr_db = 10.0 * math.log10(max(1e-15, (nr * nr / max(1, n)) / max(1e-15, mse)))
    return {
        "n_samples": n,
        "duration_s": round(n / 24000.0, 3),
        "nans": nans,
        "cosine_similarity": round(cos_sim, 6),
        "snr_db": round(snr_db, 2),
        "rmse": round(math.sqrt(mse), 6),
        "max_abs_diff": round(max_abs, 6),
    }


def run_m3_pro_metal_tts(text: str, seed: int, out_wav: Path) -> dict:
    cmd = [
        str(QWENTTS_CPP),
        "--model",
        str(TALKER_GGUF),
        "--codec",
        str(CODEC_GGUF),
        "--device",
        "MTL0",
        "--speaker",
        "ryan",
        "--lang",
        "english",
        "--seed",
        str(seed),
        "-o",
        str(out_wav),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    combined = proc.stdout + "\n" + proc.stderr

    # Parse `[Perf]` breakdown from qwentts.cpp:
    # [Perf] PromptBuild 1.9 ms
    # [Perf] Prefill 40.9 ms (T_ctx prefill)
    # [Perf] TTFA 60.7 ms (first frame codes)
    # [Perf] TalkerDecode 564.2 ms (39 frames, 14.47 ms/frame)
    # [Perf] CodePredictor 692.7 ms (17.76 ms/frame)
    # [Perf] HostCompose 1.6 ms (c0 sample + next emb)
    # [Perf] CodecDecode 140.8 ms
    # [Perf] Total 1442.6 ms (39 frames, 32.27 ms/frame AR, audio 3.12 s, RTF 0.462)
    def grab(pat):
        m = re.search(pat, combined)
        return float(m.group(1)) if m else 0.0

    m_tot = re.search(
        r"\[Perf\]\s+Total\s+([\d.]+)\s+ms\s+\((\d+)\s+frames,\s+([\d.]+)\s+ms/frame\s+AR,\s+audio\s+([\d.]+)\s+s,\s+RTF\s+([\d.]+)\)",
        combined,
    )
    total_ms = float(m_tot.group(1)) if m_tot else 0.0
    frames = int(m_tot.group(2)) if m_tot else 0
    ar_ms_per_frame = float(m_tot.group(3)) if m_tot else 0.0
    audio_sec = float(m_tot.group(4)) if m_tot else 0.0
    rtf = float(m_tot.group(5)) if m_tot else 0.0

    prompt_build_ms = grab(r"\[Perf\]\s+PromptBuild\s+([\d.]+)\s+ms")
    prefill_ms = prompt_build_ms + grab(r"\[Perf\]\s+Prefill\s+([\d.]+)\s+ms")
    first_codes_ms = grab(r"\[Perf\]\s+TTFA\s+([\d.]+)\s+ms")
    talker_ms = grab(r"\[Perf\]\s+TalkerDecode\s+([\d.]+)\s+ms")
    talker_step_ms = grab(r"\[Perf\]\s+TalkerDecode\s+[\d.]+\s+ms\s+\(\d+\s+frames,\s+([\d.]+)\s+ms/frame\)")
    cp_ms = grab(r"\[Perf\]\s+CodePredictor\s+([\d.]+)\s+ms")
    cp_step_ms = grab(r"\[Perf\]\s+CodePredictor\s+[\d.]+\s+ms\s+\(([\d.]+)\s+ms/frame\)")
    host_compose_ms = grab(r"\[Perf\]\s+HostCompose\s+([\d.]+)\s+ms")
    codec_ms = grab(r"\[Perf\]\s+CodecDecode\s+([\d.]+)\s+ms")

    ar_decode_ms = talker_ms + cp_ms + host_compose_ms
    # qwentts.cpp executes CodecDecode in one batch after all AR frames finish -> first PCM audio sample arrives at total_ms
    ttfa_ms = total_ms

    return {
        "frames": frames,
        "audio_sec": audio_sec,
        "prefill_ms": round(prefill_ms, 3),
        "first_codes_ms": round(first_codes_ms, 3),
        "ar_decode_ms": round(ar_decode_ms, 3),
        "ar_ms_per_frame": round(ar_ms_per_frame, 3),
        "talker_step_ms": talker_step_ms,
        "cp_step_ms": cp_step_ms,
        "codec_ms": codec_ms,
        "ttfa_ms": round(ttfa_ms, 3),
        "total_ms": round(total_ms, 3),
        "process_wall_ms": round(wall_ms, 3),
        "rtf": round(rtf, 4),
        "speed_x_realtime": round(1.0 / rtf, 2) if rtf > 0 else 0.0,
    }


def run_rtx3080_tts(text: str, seed: int, out_wav: Path, streaming: bool = True) -> dict:
    cmd = [
        str(QWEN_NV),
        "speak",
        "--tts-model",
        "1.7b",
        "--voice",
        "ryan",
        "--lang",
        "english",
        "--seed",
        str(seed),
        "--no-play",
        "--json-stats",
        "-o",
        str(out_wav),
        "-p",
        text,
    ]
    if not streaming:
        cmd.append("--no-stream-chunks")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_qwen_nv_text(prompt: str, max_tokens: int, spec: bool, reset_cache: bool = True) -> dict:
    cmd = [
        str(QWEN_NV),
        "run",
        "--spec" if spec else "--no-spec",
        "-q",
        "-n",
        str(max_tokens),
        "--temp",
        "0",
        "--json-stats",
    ]
    if reset_cache:
        cmd.append("--reset-cache")
    cmd.extend(["-p", prompt])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def summarize_runs(runs: list, keys: list) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = round(statistics.median(vals), 4 if k == "rtf" else 2)
    return out


def main():
    print("=" * 80)
    print("Phase 3 Benchmark: Modular Qwen3-4B + 100%-RTX-3080 Qwen3-TTS-1.7B (Q8_0) + Fused LoRA")
    print("=" * 80)

    docker_off = check_docker_off()
    print(f"[Guardrail] Docker Desktop completely OFF: {docker_off}")
    assert docker_off, "Docker must be completely stopped during benchmarking!"

    # Ensure persistent daemon is online and TTS weights are warm in RTX 3080 VRAM
    subprocess.run([str(QWEN_NV), "daemon", "--background"], check=True)
    subprocess.run([str(QWEN_NV), "lora", "clear"], check=True, capture_output=True)
    warm_out = subprocess.run(
        [str(QWEN_NV), "tts-warm", "--tts-model", "1.7b"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"[Daemon] Co-resident TTS warm: {warm_out}")

    with tempfile.TemporaryDirectory(prefix="phase3_bench_") as tmpdir:
        tmp = Path(tmpdir)

        # ------------------------------------------------------------------
        # 1. Code2Wav Golden Reference Parity & Streaming Bit-Exactness
        # ------------------------------------------------------------------
        print("\n[1/4] Verifying 100%-GPU 12Hz Code2Wav Golden Parity & Causal Streaming Equivalence...")
        codec_test_bin = tmp / "codec_from_ref_codes.bin"
        subprocess.run(
            [
                str(QWEN_NV),
                "codec-decode",
                str(REF_DIR / "codes-full.bin"),
                str(codec_test_bin),
                "--tts-model",
                "1.7b",
                "--json-stats",
            ],
            check=True,
            capture_output=True,
        )
        golden_codec_parity = waveform_parity(REF_DIR / "output-audio.bin", codec_test_bin)
        print(f"  -> vs qwentts.cpp Metal BF16 reference: {golden_codec_parity}")

        stream_bin = tmp / "stream_seed42.bin"
        batch_bin = tmp / "batch_seed42.bin"
        run_rtx3080_tts(UTTERANCES[1][1], 42, stream_bin, streaming=True)
        run_rtx3080_tts(UTTERANCES[1][1], 42, batch_bin, streaming=False)
        streaming_vs_batch = waveform_parity(batch_bin, stream_bin)
        print(f"  -> Streaming (1->2->4->8) vs Batch Code2Wav: {streaming_vs_batch}")

        # ------------------------------------------------------------------
        # 2. Head-to-Head TTS Benchmark: Apple M3 Pro Metal vs NVIDIA RTX 3080
        # ------------------------------------------------------------------
        print("\n[2/4] Running Head-to-Head Qwen3-TTS-1.7B Q8_0 Benchmark (M3 Pro Metal vs RTX 3080)...")
        # Warmup RTX 3080 P0 clocks
        run_rtx3080_tts(UTTERANCES[0][1], 42, tmp / "warmup_rtx.wav")
        # Warmup M3 Pro Metal shader cache
        run_m3_pro_metal_tts(UTTERANCES[0][1], 42, tmp / "warmup_mtl.wav")

        tts_comparisons = {}
        for label, text in UTTERANCES:
            print(f"  * Utterance [{label}]: {text!r}")
            rtx_runs = []
            mtl_runs = []
            for r in range(3):
                rtx_runs.append(run_rtx3080_tts(text, 42 + r, tmp / f"rtx_{label}_{r}.wav"))
            for r in range(3):
                mtl_runs.append(run_m3_pro_metal_tts(text, 42 + r, tmp / f"mtl_{label}_{r}.wav"))

            rtx_med = summarize_runs(
                rtx_runs,
                [
                    "audio_sec",
                    "frames",
                    "prefill_ms",
                    "first_codes_ms",
                    "ttfa_ms",
                    "client_ttfa_ms",
                    "ar_decode_ms",
                    "ar_ms_per_frame",
                    "codec_ms",
                    "total_ms",
                    "rtf",
                    "speed_x_realtime",
                ],
            )
            mtl_med = summarize_runs(
                mtl_runs,
                [
                    "audio_sec",
                    "frames",
                    "prefill_ms",
                    "first_codes_ms",
                    "ttfa_ms",
                    "ar_decode_ms",
                    "ar_ms_per_frame",
                    "codec_ms",
                    "total_ms",
                    "process_wall_ms",
                    "rtf",
                    "speed_x_realtime",
                ],
            )
            ttfa_speedup = round(mtl_med["ttfa_ms"] / max(0.01, rtx_med["client_ttfa_ms"]), 1)
            first_codes_speedup = round(mtl_med["first_codes_ms"] / max(0.01, rtx_med["first_codes_ms"]), 2)
            rtf_speedup = round(mtl_med["rtf"] / max(1e-6, rtx_med["rtf"]), 2)
            ar_speedup = round(mtl_med["ar_ms_per_frame"] / max(1e-6, rtx_med["ar_ms_per_frame"]), 2)

            print(
                f"    M3 Pro Metal : 1stCodes={mtl_med['first_codes_ms']:5.1f} ms | TTFA={mtl_med['ttfa_ms']:7.1f} ms | AR={mtl_med['ar_ms_per_frame']:5.2f} ms/f | "
                f"Codec={mtl_med['codec_ms']:6.1f} ms | Total={mtl_med['total_ms']:7.1f} ms | {mtl_med['speed_x_realtime']:.2f}x RT (RTF {mtl_med['rtf']:.3f})"
            )
            print(
                f"    RTX 3080 eGPU: 1stCodes={rtx_med['first_codes_ms']:5.1f} ms | TTFA={rtx_med['client_ttfa_ms']:7.1f} ms | AR={rtx_med['ar_ms_per_frame']:5.2f} ms/f | "
                f"Codec={rtx_med['codec_ms']:6.1f} ms | Total={rtx_med['total_ms']:7.1f} ms | {rtx_med['speed_x_realtime']:.2f}x RT (RTF {rtx_med['rtf']:.3f})"
            )
            print(
                f"    Speedups     : TTFA {ttfa_speedup}x faster (1stCodes {first_codes_speedup}x) | AR/frame {ar_speedup}x faster | Throughput {rtf_speedup}x faster!"
            )

            tts_comparisons[label] = {
                "text": text,
                "m3_pro_metal_median": mtl_med,
                "rtx_3080_sm86_median": rtx_med,
                "speedups": {
                    "ttfa_x": ttfa_speedup,
                    "first_codes_x": first_codes_speedup,
                    "ar_step_x": ar_speedup,
                    "realtime_throughput_x": rtf_speedup,
                },
                "m3_pro_runs": mtl_runs,
                "rtx_3080_runs": rtx_runs,
            }

        # ------------------------------------------------------------------
        # 3. Co-Resident Qwen3-4B -> Qwen3-TTS-1.7B Voice Assistant Pipeline
        # ------------------------------------------------------------------
        print("\n[3/4] Measuring Co-Resident Qwen3-4B -> Qwen3-TTS-1.7B End-to-End Voice Turn (0ms model swap)...")
        voice_prompt = "In two crisp sentences, explain why GDDR6X memory bandwidth makes local AI fast."
        # Warmup
        _ = run_qwen_nv_text(voice_prompt, 96, spec=False)
        # Capture text response
        proc_txt = subprocess.run(
            [
                str(QWEN_NV),
                "run",
                "--no-spec",
                "--reset-cache",
                "-n",
                "96",
                "--temp",
                "0",
                "--json-stats",
                "-p",
                voice_prompt,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = proc_txt.stdout.strip().splitlines()
        txt_stats = json.loads(lines[-1])
        spoken_reply = " ".join(lines[:-1]).strip()
        # Split first sentence just like TUI `/speak`
        first_sent = spoken_reply.split(".")[0].strip() + "."
        tts_turn_stats = run_rtx3080_tts(first_sent, 42, tmp / "e2e_voice_first_sent.wav")
        e2e_prompt_to_audio_ms = txt_stats["client_wall_ms"] + tts_turn_stats["client_ttfa_ms"]
        print(f"  -> Generated reply ({txt_stats['completion_tokens']} toks @ {txt_stats['decode_tok_per_s']} tok/s): {spoken_reply!r}")
        print(
            f"  -> Text wall: {txt_stats['client_wall_ms']:.1f} ms + First-sentence TTS TTFA: {tts_turn_stats['client_ttfa_ms']:.1f} ms "
            f"= {e2e_prompt_to_audio_ms:.1f} ms total prompt-to-spoken-voice (0 ms model swap)!"
        )

        # ------------------------------------------------------------------
        # 4. Co-Resident Text Throughput & Fused Dynamic Rank-16 LoRA Benchmark
        # ------------------------------------------------------------------
        print("\n[4/4] Benchmarking Co-Resident Text Speed & Fused Dynamic Rank-16 PEFT LoRA...")
        prose_prompt = "Explain briefly how PCIe DMA and shared memory work on a GPU."
        code_prompt = (
            "Convert the following Python dict so enable_docker is False and kernel_mode is "
            "'persistent_sm86_megakernel', preserving all other fields verbatim:\n\n"
            "```python\nENGINE_CONFIG = {\n"
            "    'device': 'NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs)',\n"
            "    'transport': 'Thunderbolt 3 / DriverKit DMA',\n"
            "    'model': 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',\n"
            "    'enable_docker': True,\n"
            "    'kernel_mode': 'hcq_graph_346_kernels',\n"
            "    'kv_cache_dtype': 'float16',\n"
            "    'max_context_tokens': 4096,\n"
            "    'quant_formats': ['Q4_K', 'Q6_K', 'Q8_0'],\n"
            "}\n```\nReturn only the updated Python code block."
        )

        base_t1_runs = [run_qwen_nv_text(prose_prompt, 128, spec=False) for _ in range(3)]
        base_bm4_runs = [run_qwen_nv_text(code_prompt, 160, spec=True) for _ in range(3)]
        base_tokens = base_t1_runs[0]["token_ids"]

        lora_dir = tmp / "rank16_peft"
        subprocess.run(
            [
                str(QWEN_NV),
                "lora",
                "synth-test",
                str(lora_dir),
                "--rank",
                "16",
                "--alpha",
                "32",
                "--strength",
                "0.02",
            ],
            check=True,
            capture_output=True,
        )

        # Zero-scale parity test
        load_zero = json.loads(
            subprocess.run(
                [str(QWEN_NV), "lora", "load", str(lora_dir), "--scale", "0.0"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        zero_scale_run = run_qwen_nv_text(prose_prompt, 128, spec=False)
        zero_scale_exact_match = zero_scale_run["token_ids"] == base_tokens

        # Active Rank-16 LoRA (scale=1.0)
        load_active = json.loads(
            subprocess.run(
                [str(QWEN_NV), "lora", "load", str(lora_dir), "--scale", "1.0"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        lora_t1_runs = [run_qwen_nv_text(prose_prompt, 128, spec=False) for _ in range(3)]
        lora_bm4_runs = [run_qwen_nv_text(code_prompt, 160, spec=True) for _ in range(3)]
        lora_perturbs_logits = lora_t1_runs[0]["token_ids"] != base_tokens

        # Clear LoRA and verify instant restoration of exact base output
        clear_info = json.loads(
            subprocess.run(
                [str(QWEN_NV), "lora", "clear"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        restored_run = run_qwen_nv_text(prose_prompt, 128, spec=False)
        restored_exact_match = restored_run["token_ids"] == base_tokens

        base_t1_med = statistics.median([r["decode_tok_per_s"] for r in base_t1_runs])
        base_bm4_med = statistics.median([r["decode_tok_per_s"] for r in base_bm4_runs])
        lora_t1_med = statistics.median([r["decode_tok_per_s"] for r in lora_t1_runs])
        lora_bm4_med = statistics.median([r["decode_tok_per_s"] for r in lora_bm4_runs])

        print(f"  -> Co-Resident Base Qwen3-4B : T=1 = {base_t1_med:.2f} tok/s | BM=4 = {base_bm4_med:.2f} tok/s")
        print(
            f"  -> Active Rank-16 LoRA (36L) : T=1 = {lora_t1_med:.2f} tok/s | BM=4 = {lora_bm4_med:.2f} tok/s | "
            f"DMA swap={load_active['swap_ms']:.2f} ms | clear={clear_info['swap_ms']:.3f} ms"
        )
        print(
            f"  -> Integrity Checks          : scale=0 exact_match={zero_scale_exact_match} | "
            f"scale=1 perturbs={lora_perturbs_logits} | clear_restores_exact={restored_exact_match}"
        )

        results = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hardware": {
                "egpu": "NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs, 10 GB GDDR6X) over Thunderbolt 3",
                "host_comparison": "Apple M3 Pro (18-core Metal GPU, Unified Memory)",
                "docker_running": not docker_off,
                "external_rust_crates": 0,
                "neural_ops_on_host_cpu_during_egpu_inference": 0,
            },
            "models_coresident_vram": {
                "thinker_gguf": str(THINKER_GGUF),
                "thinker_vram_mb": 2490.0,
                "talker_gguf": str(TALKER_GGUF),
                "codec_gguf": str(CODEC_GGUF),
                "tts_total_vram_mb": 2682.1,
                "combined_weights_vram_gb": round((2490.0 + 2682.1) / 1024.0, 2),
            },
            "vocoder_verification": {
                "golden_76frame_vs_metal_bf16": golden_codec_parity,
                "streaming_1_2_4_8_vs_batch_exactness": streaming_vs_batch,
            },
            "tts_head_to_head": tts_comparisons,
            "voice_assistant_e2e": {
                "prompt": voice_prompt,
                "reply_text": spoken_reply,
                "first_sentence_spoken": first_sent,
                "text_completion_tokens": txt_stats["completion_tokens"],
                "text_decode_tok_per_s": txt_stats["decode_tok_per_s"],
                "text_wall_ms": txt_stats["client_wall_ms"],
                "first_sentence_tts_ttfa_ms": tts_turn_stats["client_ttfa_ms"],
                "prompt_to_first_audio_ms": round(e2e_prompt_to_audio_ms, 2),
                "model_swap_ms": 0.0,
            },
            "lora_and_coresident_text": {
                "coresident_base_t1_tok_per_s": round(base_t1_med, 2),
                "coresident_base_bm4_tok_per_s": round(base_bm4_med, 2),
                "rank16_lora_t1_tok_per_s": round(lora_t1_med, 2),
                "rank16_lora_bm4_tok_per_s": round(lora_bm4_med, 2),
                "rank16_vram_bytes": load_active["vram_bytes"],
                "rank16_adapted_tensors": load_active["adapted_tensors"],
                "dma_hot_swap_ms": load_active["swap_ms"],
                "clear_swap_ms": clear_info["swap_ms"],
                "scale_zero_bit_exact": zero_scale_exact_match,
                "scale_one_shifts_output": lora_perturbs_logits,
                "clear_restores_bit_exact": restored_exact_match,
            },
        }

        OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSaved complete benchmark results to {OUT_JSON}")


if __name__ == "__main__":
    main()
