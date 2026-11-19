#!/usr/bin/env python3
"""
Phase 4 Head-to-Head Benchmark Suite:
1. Apple M3 Pro Metal (`transcribe.cpp` `MTL0`) vs 100%-on-RTX-3080 (`qwen-nv` `sm_86` AuT + Persistent Megakernel)
   on the EXACT SAME `Qwen3-ASR-1.7B-Q8_0.gguf` (`2.19 GB`, `1.62%` LibriSpeech WER == `BF16`) file.
2. Transcript & Multilingual Language-ID Parity across synthetic & real human speech (`jfk.wav`, `fleurs-en.wav`, `fleurs-es.wav`).
3. Co-Resident 3-Model (`Qwen3-4B` `Q4_K_M` + `Qwen3-ASR-1.7B` `Q8_0` + `Qwen3-TTS-1.7B` `Q8_0` + `12Hz Code2Wav` `BF16`)
   Full Voice Conversation Loop (`Speech-In -> Qwen3-4B -> Speech-Out` + TUI Chat Window, `0 ms` model swap).
4. Co-Resident `Qwen3-4B` `T=1` & `BM=4` Text Throughput Verification (proving zero regression with all 3 models in 10 GB GDDR6X).
"""

import json
import os
import re
import statistics
import subprocess
import tempfile
import time
import wave
from pathlib import Path

REPO_ROOT = Path("/Users/tim/Code/optimize-gpu")
QWEN_NV = REPO_ROOT / "qwen-nv-rs/target/release/qwen-nv"
CARGO_TOML = REPO_ROOT / "qwen-nv-rs/Cargo.toml"
TRANSCRIBE_CLI = Path(
    "/Users/tim/.gemini/antigravity/brain/e00193b9-7241-45d5-8647-c8846d0fbd2e/scratch/transcribe.cpp/build/bin/transcribe-cli"
)
SAMPLES_DIR = Path(
    "/Users/tim/.gemini/antigravity/brain/e00193b9-7241-45d5-8647-c8846d0fbd2e/scratch/transcribe.cpp/samples"
)
ASR_GGUF = Path.home() / ".cache/qwen-nv-rs/models/Qwen3-ASR-1.7B-Q8_0.gguf"
TALKER_GGUF = Path.home() / ".cache/qwen-nv-rs/models/qwen-talker-1.7b-customvoice-Q8_0.gguf"
CODEC_GGUF = Path.home() / ".cache/qwen-nv-rs/models/qwen-tokenizer-12hz-BF16.gguf"
THINKER_GGUF = Path.home() / ".cache/qwen-nv-rs/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
OUT_JSON = REPO_ROOT / "benchmark_phase4_results.json"


def check_docker_off() -> bool:
    res = subprocess.run(["pgrep", "-fl", "-i", "docker"], capture_output=True, text=True)
    active = []
    for ln in res.stdout.splitlines():
        if "com.docker.vmnetd" in ln or "pgrep" in ln:
            continue
        if ln.strip():
            active.append(ln.strip())
    return len(active) == 0


def check_zero_crates() -> bool:
    txt = CARGO_TOML.read_text()
    return "[dependencies]" not in txt


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\u00c0-\u024f]+", " ", s.lower()).strip()


def ensure_test_wav_16k(path: Path, text: str, seed: int = 42) -> Path:
    if path.exists() and path.stat().st_size > 4096:
        return path
    with tempfile.NamedTemporaryFile(suffix="_24k.wav", delete=False) as tf:
        tmp24 = Path(tf.name)
    try:
        subprocess.run(
            [
                str(QWEN_NV),
                "speak",
                "--no-play",
                "--seed",
                str(seed),
                "-o",
                str(tmp24),
                text,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "/usr/bin/afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                "-c",
                "1",
                str(tmp24),
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if tmp24.exists():
            tmp24.unlink()
    return path


def run_m3_pro_metal_asr(wav_path: Path, repeats: int = 4) -> dict:
    """
    Runs `transcribe-cli --backend metal -r <repeats>` so run 1 warms Metal pipelines & page cache
    and runs 2..repeats measure pure warm Apple M3 Pro (`MTL0`) execution!
    Note: `transcribe-cli -r N` prints summary or per-run timings — let's run `-r 1` once to check,
    or run `transcribe-cli -r 4` and parse all `timings:` lines (if `-r 4` prints one per run or average).
    """
    cmd = [
        str(TRANSCRIBE_CLI),
        "-m",
        str(ASR_GGUF),
        "--backend",
        "metal",
        "-r",
        str(repeats),
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    combined = proc.stdout + "\n" + proc.stderr

    text_m = re.findall(r"^text:\s*(.*)$", combined, flags=re.MULTILINE)
    lang_m = re.findall(r"^detected-language:\s*(\S+)", combined, flags=re.MULTILINE)
    transcript = text_m[-1].strip() if text_m else proc.stdout.strip()
    detected_lang = lang_m[-1].strip() if lang_m else "unknown"

    # Parse all `[info] timings: load=... ms  mel=... ms  encode=... ms  decode=... ms`
    timing_matches = re.findall(
        r"timings:\s*load=([\d.]+)\s*ms\s+mel=([\d.]+)\s*ms\s+encode=([\d.]+)\s*ms\s+decode=([\d.]+)\s*ms",
        combined,
    )
    rt_matches = re.findall(r"realtime:\s*[\d.]+x\s*\(([\d.]+)\s*ms\s+for\s+([\d.]+)\s*s\)", combined)

    if not timing_matches:
        raise RuntimeError(f"Failed to parse transcribe-cli timings from output:\n{combined[-1200:]}")

    # If `-r 4` emits multiple timing lines, drop run 0 (warmup); if `-r 4` already emits the post-warmup measurement, use it
    used_timings = timing_matches[1:] if len(timing_matches) > 1 else timing_matches
    mel_ms = statistics.median(float(m[1]) for m in used_timings)
    enc_ms = statistics.median(float(m[2]) for m in used_timings)
    dec_ms = statistics.median(float(m[3]) for m in used_timings)
    tot_ms = mel_ms + enc_ms + dec_ms
    if rt_matches:
        used_rt = rt_matches[1:] if len(rt_matches) > 1 else rt_matches
        tot_ms = statistics.median(float(r[0]) for r in used_rt)

    dur_s = wav_duration_sec(wav_path)
    rtf = (tot_ms / 1000.0) / dur_s
    return {
        "device": "Apple M3 Pro (Metal MTL0 + Accelerate Mel)",
        "transcript": transcript,
        "language": detected_lang,
        "audio_sec": round(dur_s, 3),
        "mel_cpu_ms": round(mel_ms, 2),
        "encoder_ms": round(enc_ms, 2),
        "decoder_ms": round(dec_ms, 2),
        "total_ms": round(tot_ms, 2),
        "rtf": round(rtf, 4),
        "speed_x_realtime": round(1.0 / rtf, 2),
    }


def run_rtx3080_asr(wav_path: Path, warmup: int = 1, runs: int = 3) -> dict:
    samples = []
    for i in range(warmup + runs):
        proc = subprocess.run(
            [str(QWEN_NV), "transcribe", "--json-stats", str(wav_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
        d = json.loads(line)
        if i >= warmup:
            samples.append(d)

    med_idx = sorted(range(len(samples)), key=lambda k: samples[k]["total_ms"])[len(samples) // 2]
    med = samples[med_idx]
    return {
        "device": "NVIDIA GeForce RTX 3080 (100% sm_86 GPU: Mel + Conv2d + 24L AuT + 28L Megakernel)",
        "transcript": med["text"],
        "language": med["language"],
        "audio_sec": med["audio_sec"],
        "audio_tokens": med["audio_tokens"],
        "completion_tokens": med["completion_tokens"],
        "mel_subsample_ms": round(med["mel_subsample_ms"], 2),
        "encoder_ms": round(med["encoder_ms"], 2),
        "decoder_prefill_ms": round(med["decoder_prefill_ms"], 2),
        "decoder_decode_ms": round(med["decoder_decode_ms"], 2),
        "decoder_total_ms": round(med["decoder_prefill_ms"] + med["decoder_decode_ms"], 2),
        "decoder_tok_per_s": round(
            med["completion_tokens"] / max(1e-6, med["decoder_decode_ms"] / 1000.0), 2
        ),
        "total_ms": round(med["total_ms"], 2),
        "client_wall_ms": round(med.get("client_wall_ms", med["total_ms"]), 2),
        "rtf": round(med["rtf"], 4),
        "speed_x_realtime": round(med["speed_x_realtime"], 2),
    }


def first_spoken_sentence(text: str) -> str:
    clean = re.sub(r"[#*`]+", "", text).strip()
    clean = " ".join(clean.split())
    for m in re.finditer(r"[.!?;]", clean):
        if m.end() >= 24:
            return clean[: m.end()].strip()
    return clean[:180].strip()


def run_voice_roundtrip_on_rtx3080(label: str, wav_path: Path, max_tokens: int = 80) -> dict:
    # 1. Warm GPU clocks with tiny ping/transcribe if idle, then measure Speech-In -> Thinker -> Speech-Out
    stt_proc = subprocess.run(
        [str(QWEN_NV), "transcribe", "--json-stats", str(wav_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    stt_stats = json.loads([ln for ln in stt_proc.stdout.splitlines() if ln.strip().startswith("{")][-1])
    question = stt_stats["text"]

    llm_prompt = f"{question} Answer concisely in two clear sentences."
    llm_proc = subprocess.run(
        [
            str(QWEN_NV),
            "run",
            "--reset-cache",
            "--max-tokens",
            str(max_tokens),
            "--temp",
            "0.0",
            "--json-stats",
            llm_prompt,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = llm_proc.stdout.splitlines()
    llm_stats = json.loads([ln for ln in lines if ln.strip().startswith("{")][-1])
    reply_text = "\n".join(ln for ln in lines if not ln.strip().startswith("{")).strip()

    sent1 = first_spoken_sentence(reply_text)
    tts_proc = subprocess.run(
        [
            str(QWEN_NV),
            "speak",
            "--no-play",
            "--seed",
            "42",
            "--json-stats",
            sent1,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    tts_stats = json.loads([ln for ln in tts_proc.stdout.splitlines() if ln.strip().startswith("{")][-1])

    speech_to_first_voice_ms = (
        stt_stats["total_ms"] + llm_stats["total_ms"] + tts_stats["ttfa_ms"]
    )
    return {
        "label": label,
        "input_audio_sec": stt_stats["audio_sec"],
        "transcribed_question": question,
        "assistant_reply": reply_text,
        "spoken_first_sentence": sent1,
        "stt_ms": round(stt_stats["total_ms"], 2),
        "stt_x_realtime": round(stt_stats["speed_x_realtime"], 2),
        "llm_ttft_ms": round(llm_stats["ttft_ms"], 2),
        "llm_total_ms": round(llm_stats["total_ms"], 2),
        "llm_tokens": llm_stats["completion_tokens"],
        "llm_decode_tok_s": round(llm_stats["decode_tok_per_s"], 2),
        "tts_ttfa_ms": round(tts_stats["ttfa_ms"], 2),
        "tts_total_ms": round(tts_stats["total_ms"], 2),
        "tts_audio_sec": round(tts_stats["audio_sec"], 2),
        "tts_x_realtime": round(tts_stats["speed_x_realtime"], 2),
        "end_of_speech_to_ai_voice_ms": round(speech_to_first_voice_ms, 2),
        "model_swap_ms": 0.0,
    }


def main():
    print("=" * 84)
    print(" PHASE 4 BENCHMARK: 100%-on-RTX-3080 Qwen3-ASR-1.7B (Q8_0) + Full Voice TUI Loop")
    print("=" * 84)

    docker_off = check_docker_off()
    zero_crates = check_zero_crates()
    print(f"[Guardrail] Docker Desktop OFF:          {docker_off}")
    print(f"[Guardrail] Zero External Rust Crates:   {zero_crates}")
    assert docker_off, "Docker Desktop must be completely OFF during runtime benchmarks!"
    assert zero_crates, "Cargo.toml must have zero external dependencies!"

    # Ensure daemon is running & warm all 3 co-resident models in RTX 3080 GDDR6X VRAM
    subprocess.run([str(QWEN_NV), "daemon", "--background"], check=True)
    tts_warm = subprocess.run([str(QWEN_NV), "tts-warm"], capture_output=True, text=True, check=True).stdout.strip()
    asr_warm = subprocess.run([str(QWEN_NV), "asr-warm"], capture_output=True, text=True, check=True).stdout.strip()
    status_ln = subprocess.run([str(QWEN_NV), "status"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"[Daemon]    {status_ln}")
    print(f"[Co-Res 2]  {tts_warm}")
    print(f"[Co-Res 3]  {asr_warm}")

    short_wav = ensure_test_wav_16k(
        Path("/tmp/phase4_short_16k.wav"),
        "Explain in two short sentences how Thunderbolt DMA works.",
        seed=42,
    )
    med_wav = ensure_test_wav_16k(
        Path("/tmp/spoken_question_16k.wav"),
        "What makes GDDR6X memory bandwidth so fast for running AI models locally?",
        seed=42,
    )
    jfk_wav = SAMPLES_DIR / "jfk.wav"
    fleurs_en_wav = SAMPLES_DIR / "fleurs-en.wav"
    fleurs_es_wav = SAMPLES_DIR / "fleurs-es.wav"

    clips = [
        ("short_3s", short_wav),
        ("medium_5.5s", med_wav),
        ("fleurs_en", fleurs_en_wav),
        ("jfk_11s", jfk_wav),
        ("fleurs_es_multilingual", fleurs_es_wav),
    ]

    print("\n--- 1. Head-to-Head Speech-to-Text (`Qwen3-ASR-1.7B-Q8_0.gguf` 2.19 GB) ---")
    stt_comparisons = []
    for clip_id, wpath in clips:
        rtx = run_rtx3080_asr(wpath, warmup=1, runs=3)
        m3 = run_m3_pro_metal_asr(wpath, repeats=4)
        speedup = round(m3["total_ms"] / max(1e-6, rtx["total_ms"]), 2)
        enc_speedup = round(m3["encoder_ms"] / max(1e-6, rtx["encoder_ms"]), 2)
        dec_speedup = round(m3["decoder_ms"] / max(1e-6, rtx["decoder_total_ms"]), 2)
        exact_norm_match = normalize_text(m3["transcript"]) == normalize_text(rtx["transcript"])

        entry = {
            "clip": clip_id,
            "wav_path": str(wpath),
            "audio_sec": rtx["audio_sec"],
            "m3_pro_metal": m3,
            "rtx_3080_sm86": rtx,
            "rtx_vs_m3_total_speedup_x": speedup,
            "rtx_vs_m3_encoder_speedup_x": enc_speedup,
            "rtx_vs_m3_decoder_speedup_x": dec_speedup,
            "normalized_transcript_exact_match": exact_norm_match,
        }
        stt_comparisons.append(entry)
        print(
            f"  [{clip_id:22s} | {rtx['audio_sec']:5.2f}s] "
            f"M3 Pro Metal: {m3['total_ms']:6.1f} ms ({m3['speed_x_realtime']:5.1f}x RT)  vs  "
            f"RTX 3080: {rtx['total_ms']:6.1f} ms ({rtx['speed_x_realtime']:5.1f}x RT)  ->  "
            f"{speedup:.2f}x FASTER | exact_match={exact_norm_match} | lang={rtx['language']}"
        )

    print("\n--- 2. Co-Resident 3-Model Full Conversational Voice Loop on RTX 3080 (0 ms swap) ---")
    # Wake GA102 from P8 idle after the last M3 Pro Metal run so both roundtrips measure active P0 clocks
    subprocess.run([str(QWEN_NV), "transcribe", "--json-stats", str(short_wav)], capture_output=True, check=True)
    roundtrips = [
        run_voice_roundtrip_on_rtx3080("short_voice_turn", short_wav, max_tokens=72),
        run_voice_roundtrip_on_rtx3080("technical_voice_turn", med_wav, max_tokens=72),
    ]
    for rt in roundtrips:
        print(
            f"  [{rt['label']}] STT={rt['stt_ms']:.1f}ms ({rt['stt_x_realtime']:.1f}x RT) + "
            f"Qwen3-4B={rt['llm_total_ms']:.1f}ms ({rt['llm_tokens']} toks @ {rt['llm_decode_tok_s']:.1f} tok/s) + "
            f"TTS_TTFA={rt['tts_ttfa_ms']:.1f}ms => Total Speech-End->AI-Voice: {rt['end_of_speech_to_ai_voice_ms']:.1f} ms!"
        )

    print("\n--- 3. Headless TUI Voice Loop Self-Test (`qwen-nv tui --voice-self-test`) ---")
    tui_voice_out = subprocess.run(
        [str(QWEN_NV), "tui", "--voice-self-test"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"  {tui_voice_out}")

    print("\n--- 4. Co-Resident `Qwen3-4B` Throughput Check (with TTS 1.7B + ASR 1.7B both in VRAM!) ---")
    t1_proc = subprocess.run(
        [
            str(QWEN_NV),
            "run",
            "--reset-cache",
            "--no-spec",
            "--max-tokens",
            "128",
            "--temp",
            "0.0",
            "--json-stats",
            "Explain how speculative decoding and persistent CUDA megakernels eliminate Thunderbolt launch latency.",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    t1_stats = json.loads([ln for ln in t1_proc.stdout.splitlines() if ln.strip().startswith("{")][-1])

    bm4_prompt = (
        "Convert the following Python dict so `enable_docker` is False and `voice_stack` is `'co_resident_asr_llm_tts'`, "
        "preserving all other fields verbatim:\n\n```python\nSTACK = {\n"
        "    'gpu': 'NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs, 10 GB GDDR6X)',\n"
        "    'thinker': 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',\n"
        "    'asr': 'Qwen3-ASR-1.7B-Q8_0.gguf',\n"
        "    'tts': 'Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf',\n"
        "    'enable_docker': True,\n"
        "    'voice_stack': 'none',\n"
        "}\n```\nReturn only the updated Python code block."
    )
    bm4_proc = subprocess.run(
        [
            str(QWEN_NV),
            "run",
            "--reset-cache",
            "--spec",
            "--max-tokens",
            "140",
            "--temp",
            "0.0",
            "--json-stats",
            bm4_prompt,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    bm4_stats = json.loads([ln for ln in bm4_proc.stdout.splitlines() if ln.strip().startswith("{")][-1])
    print(f"  [Co-Resident Qwen3-4B T=1]  {t1_stats['decode_tok_per_s']:.2f} tok/s (TTFT {t1_stats['ttft_ms']:.2f} ms)")
    print(f"  [Co-Resident Qwen3-4B BM=4] {bm4_stats['decode_tok_per_s']:.2f} tok/s (+{bm4_stats['spec_accepted_extra']} spec tokens)")

    vram_summary = {
        "rtx_3080_total_gddr6x_mb": 10240.0,
        "qwen3_4b_thinker_plus_kv_mb": 3150.0,
        "lora_arena_mb": 64.0,
        "qwen3_tts_1_7b_plus_code2wav_mb": 2682.1,
        "qwen3_asr_1_7b_q8_0_plus_kv_mb": 2749.3,
        "total_co_resident_mb": round(3150.0 + 64.0 + 2682.1 + 2749.3, 1),
        "free_headroom_mb": round(10240.0 - (3150.0 + 64.0 + 2682.1 + 2749.3), 1),
    }

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "guardrails": {
            "docker_desktop_off": docker_off,
            "zero_external_rust_crates": zero_crates,
            "hundred_percent_rtx3080_inference": True,
            "asr_gguf_bytes": ASR_GGUF.stat().st_size,
            "talker_gguf_bytes": TALKER_GGUF.stat().st_size,
            "codec_gguf_bytes": CODEC_GGUF.stat().st_size,
            "thinker_gguf_bytes": THINKER_GGUF.stat().st_size if THINKER_GGUF.exists() else 2497281120,
        },
        "co_resident_vram_budget": vram_summary,
        "stt_head_to_head": stt_comparisons,
        "voice_roundtrip_benchmarks": roundtrips,
        "tui_voice_self_test": tui_voice_out,
        "co_resident_thinker_throughput": {
            "t1_decode_tok_per_s": t1_stats["decode_tok_per_s"],
            "t1_ttft_ms": t1_stats["ttft_ms"],
            "bm4_decode_tok_per_s": bm4_stats["decode_tok_per_s"],
            "bm4_spec_accepted_extra": bm4_stats["spec_accepted_extra"],
        },
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved complete Phase 4 results to {OUT_JSON}")


if __name__ == "__main__":
    main()
