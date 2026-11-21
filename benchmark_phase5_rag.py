#!/usr/bin/env python3
"""
Phase 5 Benchmark Suite: Same-Model RTX 3080 Text Embeddings, .qvec GPU VectorDB,
& 5-Way Wikipedia Retrieval + Spoken Voice RAG Benchmark (Pure Python stdlib, Zero Docker).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/tim/Code/optimize-gpu")
BIN = ROOT / "qwen-nv-rs" / "target" / "release" / "qwen-nv"
DATA_DIR = ROOT / "data"
CORPUS_JSONL = DATA_DIR / "wikipedia_1k.jsonl"
EVAL_JSONL = DATA_DIR / "wikipedia_eval_200.jsonl"
BEST_QVEC = DATA_DIR / "wikipedia_1k_best.qvec"
OUT_JSON = ROOT / "benchmark_phase5_results.json"

CACHE_DIR = Path.home() / ".cache" / "qwen-nv-rs"
LORA_ATTN = CACHE_DIR / "loras" / "qwen3_embed_svd_r32_attn"
LORA_ALL = CACHE_DIR / "loras" / "qwen3_embed_svd_r32_all"
DAEMON_LOG = CACHE_DIR / "daemon.log"


def check_docker_off():
    res = subprocess.run(["pgrep", "-x", "Docker"], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        raise RuntimeError("Docker Desktop is running! Must be OFF per AGENTS.md Rule #3.")
    print("[guardrail] Docker Desktop confirmed OFF (0 containers/VMs).")


def run_cmd(args, timeout=600):
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(BIN)] + [str(a) for a in args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, args))}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return proc.stdout.strip(), proc.stderr.strip(), wall_s


def last_json_line(stdout_text: str) -> dict:
    for ln in reversed(stdout_text.splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)
    raise ValueError(f"Could not parse JSON line from:\n{stdout_text}")


def stop_clean():
    subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
    time.sleep(0.6)


def start_daemon_with_retry(model_ref: str):
    for attempt in range(1, 4):
        stop_clean()
        try:
            run_cmd(["daemon", "--background", "--model", model_ref], timeout=60)
            st_out, _, _ = run_cmd(["status"], timeout=15)
            print(f"[daemon] {st_out}")
            return st_out
        except Exception as e:
            print(f"[daemon] Attempt {attempt} wake warning: {e}")
            time.sleep(1.5)
    raise RuntimeError(f"Failed to start daemon for {model_ref}")


def main():
    print("=" * 84)
    print("Phase 5: 100%-on-RTX-3080 Same-Model Embeddings & Wikipedia .qvec VectorDB Benchmark")
    print("=" * 84)
    check_docker_off()

    if not CORPUS_JSONL.exists() or not EVAL_JSONL.exists():
        raise FileNotFoundError("Missing data/wikipedia_1k.jsonl or data/wikipedia_eval_200.jsonl")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_passages": sum(1 for _ in open(CORPUS_JSONL, encoding="utf-8") if _.strip()),
        "eval_queries": sum(1 for _ in open(EVAL_JSONL, encoding="utf-8") if _.strip()),
        "svd_adapters": {},
        "strategies": {},
        "lora_toggle_verify": {},
        "voice_rag_e2e": [],
    }

    # Verify SVD adapters exist (extract if needed)
    for tag, ldir, extra_flags in [
        ("svd_r32_attn", LORA_ATTN, ["--rank", "32", "--attn-only"]),
        ("svd_r32_all", LORA_ALL, ["--rank", "32"]),
    ]:
        st_file = ldir / "adapter_model.safetensors"
        stats_cache = ldir / "svd_extract_stats.json"
        if not st_file.exists():
            print(f"\n[svd] Extracting {tag} -> {ldir} ...")
            out, _, _ = run_cmd(
                ["lora", "extract-svd", "qwen3-embed:4b", str(ldir)] + extra_flags,
                timeout=600,
            )
            j = last_json_line(out)
            stats_cache.write_text(json.dumps(j, indent=2))
            results["svd_adapters"][tag] = j
        else:
            sz_mb = st_file.stat().st_size / (1024.0 * 1024.0)
            if stats_cache.exists():
                results["svd_adapters"][tag] = json.loads(stats_cache.read_text())
            else:
                results["svd_adapters"][tag] = {
                    "out_dir": str(ldir),
                    "safetensors_bytes": st_file.stat().st_size,
                    "safetensors_mb": round(sz_mb, 2),
                }
            print(f"[svd] Ready: {tag} ({sz_mb:.2f} MB) at {ldir}")

    # -------------------------------------------------------------------------
    # Part 1: Co-Resident Same-Model Strategies on Qwen3-4B (+ Warm ASR + Warm TTS)
    # -------------------------------------------------------------------------
    print("\n[step 1] Starting production Qwen3-4B daemon + warming co-resident TTS-1.7B & ASR-1.7B...")
    start_daemon_with_retry("qwen3:4b")
    tts_w, _, _ = run_cmd(["tts-warm", "--tts-model", "1.7b"], timeout=120)
    asr_w, _, _ = run_cmd(["asr-warm", "--asr-model", "1.7b"], timeout=120)
    print(f"  TTS co-resident: {tts_w}")
    print(f"  ASR co-resident: {asr_w}")

    # Baseline lossless Qwen3-4B greedy output BEFORE any RAG/LoRA activity
    base_txt, _, _ = run_cmd(
        ["run", "--temp", "0", "--max-tokens", "24", "--reset-cache", "State Newton's second law in one short formula."],
        timeout=30,
    )

    same_model_runs = [
        ("raw_eos", "raw_eos", "", DATA_DIR / "wiki_raw_eos.qvec", 0.0),
        ("prompt_eol", "prompt_eol", "", DATA_DIR / "wiki_prompt_eol.qvec", 0.01),
        ("echo", "echo", "", DATA_DIR / "wiki_echo.qvec", 0.01),
        ("svd_lora_attn_r32", "svd_lora", str(LORA_ATTN), DATA_DIR / "wiki_svd_lora_attn.qvec", 47.25),
        ("svd_lora_all_r32", "svd_lora", str(LORA_ALL), DATA_DIR / "wiki_svd_lora_all.qvec", 132.19),
    ]

    for run_key, strat_flag, lora_dir, qvec_out, extra_vram_mb in same_model_runs:
        print(f"\n[benchmark] Indexing 1,000 Wikipedia passages with `{run_key}` (strategy={strat_flag})...")
        idx_args = [
            "rag",
            "index",
            str(CORPUS_JSONL),
            "-o",
            str(qvec_out),
            "--strategy",
            strat_flag,
        ]
        if lora_dir:
            idx_args += ["--lora", lora_dir]
        idx_out, _, _ = run_cmd(idx_args, timeout=600)
        idx_stats = last_json_line(idx_out)
        print(
            f"  Indexed {idx_stats['num_docs']} docs in {idx_stats['total_ms']/1000.0:.2f}s "
            f"({idx_stats['passages_per_s']:.1f} passages/s) -> {qvec_out.name} "
            f"({qvec_out.stat().st_size / (1024*1024):.2f} MB)"
        )

        rag_st_out, _, _ = run_cmd(["rag", "status"], timeout=15)
        rag_st = last_json_line(rag_st_out)

        print(f"[benchmark] Evaluating 200 Wikipedia Q&A queries on `{run_key}`...")
        ev_out, _, _ = run_cmd(["rag", "eval", str(EVAL_JSONL)], timeout=300)
        ev_stats = last_json_line(ev_out)
        print(
            f"  => Recall@1={ev_stats['recall_at_1']*100:.2f}% | "
            f"Recall@5={ev_stats['recall_at_5']*100:.2f}% | "
            f"Recall@10={ev_stats['recall_at_10']*100:.2f}% | "
            f"MRR@10={ev_stats['mrr_at_10']:.4f} | "
            f"query_embed={ev_stats['median_embed_ms']:.2f} ms | "
            f"GPU_cosine={ev_stats['median_gpu_search_us']:.1f} us"
        )

        results["strategies"][run_key] = {
            "strategy_flag": strat_flag,
            "lora_dir": lora_dir,
            "qvec_file": str(qvec_out),
            "qvec_disk_mb": round(qvec_out.stat().st_size / (1024.0 * 1024.0), 3),
            "qvec_vram_mb": rag_st.get("vram_mb", 5.12),
            "adapter_vram_mb": extra_vram_mb,
            "total_extra_vram_mb": round(rag_st.get("vram_mb", 5.12) + extra_vram_mb, 2),
            "index": idx_stats,
            "eval": ev_stats,
        }

    # Verify lossless Thinker generation & measure cached LoRA pointer-flip latency
    after_txt, _, _ = run_cmd(
        ["run", "--temp", "0", "--max-tokens", "24", "--reset-cache", "State Newton's second law in one short formula."],
        timeout=30,
    )
    lora_st_after, _, _ = run_cmd(["lora", "status"], timeout=15)
    # Measure cached pointer-flip latency when reloading already-staged LORA_ALL
    hot1, _, _ = run_cmd(["lora", "load", str(LORA_ALL)], timeout=30)
    clr1, _, _ = run_cmd(["lora", "clear"], timeout=30)
    hot2, _, _ = run_cmd(["lora", "load", str(LORA_ALL)], timeout=30)
    clr2, _, _ = run_cmd(["lora", "clear"], timeout=30)

    results["lora_toggle_verify"] = {
        "thinker_before_rag": base_txt,
        "thinker_after_svd_rag": after_txt,
        "bit_exact_thinker_preserved": (base_txt == after_txt),
        "lora_status_after_rag": json.loads(lora_st_after),
        "cached_reload_json": json.loads(hot2),
        "clear_json": json.loads(clr2),
    }
    print(
        f"\n[verify] Bit-exact Qwen3-4B Thinker preserved after SVD-LoRA RAG: "
        f"{results['lora_toggle_verify']['bit_exact_thinker_preserved']} "
        f"(cached LoRA pointer-flip swap_ms={results['lora_toggle_verify']['cached_reload_json'].get('swap_ms')} ms)"
    )

    # -------------------------------------------------------------------------
    # Part 2: Standalone Official Qwen3-Embedding-4B-Q8_0.gguf Upper-Bound Oracle
    # -------------------------------------------------------------------------
    print("\n[step 2] Cleanly unloading Qwen3-4B & running isolated Qwen3-Embedding-4B-Q8_0 Oracle...")
    start_daemon_with_retry("qwen3-embed:4b")
    oracle_qvec = DATA_DIR / "wiki_oracle.qvec"
    idx_out, _, _ = run_cmd(
        ["rag", "index", "--model", "qwen3-embed:4b", str(CORPUS_JSONL), "-o", str(oracle_qvec), "--strategy", "oracle"],
        timeout=600,
    )
    idx_stats = last_json_line(idx_out)
    ev_out, _, _ = run_cmd(["rag", "eval", "--model", "qwen3-embed:4b", str(EVAL_JSONL)], timeout=300)
    ev_stats = last_json_line(ev_out)
    print(
        f"  [ORACLE Qwen3-Embedding-4B-Q8_0] "
        f"Recall@1={ev_stats['recall_at_1']*100:.2f}% | "
        f"Recall@5={ev_stats['recall_at_5']*100:.2f}% | "
        f"Recall@10={ev_stats['recall_at_10']*100:.2f}% | "
        f"MRR@10={ev_stats['mrr_at_10']:.4f} | "
        f"passages/s={idx_stats['passages_per_s']:.1f} | "
        f"query_embed={ev_stats['median_embed_ms']:.2f} ms"
    )
    results["strategies"]["oracle_qwen3_embed_4b_q8"] = {
        "strategy_flag": "oracle",
        "lora_dir": "",
        "qvec_file": str(oracle_qvec),
        "qvec_disk_mb": round(oracle_qvec.stat().st_size / (1024.0 * 1024.0), 3),
        "qvec_vram_mb": 5.12,
        "adapter_vram_mb": 4076.0,
        "total_extra_vram_mb": 4081.12,
        "index": idx_stats,
        "eval": ev_stats,
    }
    stop_clean()

    # -------------------------------------------------------------------------
    # Part 3: Promote Winning Same-Model Index & Benchmark Full 4-Stage Spoken RAG!
    # -------------------------------------------------------------------------
    same_keys = [k for k in results["strategies"] if k != "oracle_qwen3_embed_4b_q8"]
    winner_key = max(
        same_keys,
        key=lambda k: (
            results["strategies"][k]["eval"]["recall_at_1"],
            results["strategies"][k]["eval"]["mrr_at_10"],
            results["strategies"][k]["eval"]["recall_at_5"],
        ),
    )
    winner_qvec = Path(results["strategies"][winner_key]["qvec_file"])
    shutil.copy2(winner_qvec, BEST_QVEC)
    results["winning_same_model_strategy"] = winner_key
    results["best_qvec_path"] = str(BEST_QVEC)
    print(f"\n[step 3] Winning Same-Model Strategy: `{winner_key}` -> copied to {BEST_QVEC}")

    print("[step 4] Restarting co-resident Qwen3-4B + ASR-1.7B + TTS-1.7B + Best Wikipedia .qvec...")
    start_daemon_with_retry("qwen3:4b")
    run_cmd(["tts-warm", "--tts-model", "1.7b"], timeout=120)
    run_cmd(["asr-warm", "--asr-model", "1.7b"], timeout=120)
    load_out, _, _ = run_cmd(["rag", "load", str(BEST_QVEC)], timeout=30)
    print(f"  Loaded RAG index: {load_out}")

    voice_questions = [
        (
            "wiki_voice_q1",
            "How many miles south of Edinburgh is Newcastle upon Tyne?",
            "103",
        ),
        (
            "wiki_voice_q2",
            "What company was formed by the merger of Sky Television and British Satellite Broadcasting?",
            "BSkyB",
        ),
    ]
    wav_dir = CACHE_DIR / "phase5_voice_rag_test"
    wav_dir.mkdir(parents=True, exist_ok=True)

    for tag, q_text, expected_substr in voice_questions:
        q_wav = wav_dir / f"{tag}_question.wav"
        a_wav = wav_dir / f"{tag}_answer.wav"
        run_cmd(
            ["speak", "--no-play", "--voice", "ryan", "-o", str(q_wav), q_text],
            timeout=60,
        )
        e2e_out, _, wall_s = run_cmd(
            [
                "transcribe",
                "--ask",
                "--no-play",
                "--json-stats",
                "--max-tokens",
                "96",
                "-o",
                str(a_wav),
                str(q_wav),
            ],
            timeout=120,
        )
        e2e_j = last_json_line(e2e_out)
        e2e_j["tag"] = tag
        e2e_j["input_question"] = q_text
        e2e_j["expected_fact"] = expected_substr
        e2e_j["fact_grounded_ok"] = expected_substr.lower() in e2e_j["reply"].lower()
        e2e_j["turn_wall_s"] = round(wall_s, 3)
        e2e_j["speech_end_to_voice_ms"] = round(
            e2e_j["stt_wall_ms"] + e2e_j["rag_embed_ms"] + e2e_j["llm_ttft_ms"] + e2e_j["tts_ttfa_ms"],
            2,
        )
        results["voice_rag_e2e"].append(e2e_j)
        print(
            f"  [{tag}] STT='{e2e_j['transcript']}' ({e2e_j['stt_wall_ms']:.1f} ms) -> "
            f"RAG={e2e_j['rag_titles']} ({e2e_j['rag_embed_ms']:.2f} ms, {e2e_j['rag_gpu_search_us']:.1f} us) -> "
            f"Grounded={e2e_j['fact_grounded_ok']} | Reply='{e2e_j['reply'][:110]}...'"
        )

    if DAEMON_LOG.exists():
        results["daemon_log_tail"] = DAEMON_LOG.read_text(errors="replace").splitlines()[-40:]

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n[done] Saved complete Phase 5 benchmark results to {OUT_JSON}")


if __name__ == "__main__":
    main()
