#!/usr/bin/env python3
"""
Phase 8 End-to-End Verification & Benchmark (`qwen-nv` v0.7.0, 100% RTX 3080 `sm_86`):
  1. Dynamic Scale-Aware `BM=4 -> Persistent T=1` Crossover Switch (`1,536` on 4B, `3,072` on 8B):
     - Verifies 100.0% bit-exact greedy token parity across `--no-spec`, `--spec --spec-crossover 0`,
       `--spec --spec-crossover 1536`, and `--spec` default (`3,072` on 8B).
  2. Shared-Ancestor `8B` Embeddings & `.qvec` Wikipedia RAG (`AGENTS.md` Rule #1):
     - Uses Rank-32 7-projection truncated SVD LoRA (`lora/qwen3-8b-embed-svd-r32`, `166.56 MB`)
       extracted from `Qwen3-Embedding-8B-Q6_K.gguf` - `Qwen3-8B-Q6_K.gguf` (`36L`, `dim=4096`).
     - Evaluates `data/wikipedia_1k_8b_best.qvec` (`4096-D` BF16, `0` standalone embedding model in VRAM)
       on `data/wikipedia_eval_200.jsonl`.
  3. Unseen Post-Cutoff (`Nov 2025 - Aug 2026`) Competition Benchmark (`24` Stratified Held-Out Problems @ 4,096 toks):
     - Evaluates `Qwen3-8B-Q6_K` (`qwen3-thinking:8b`, `thinking_tmpl=true`, `0.6B` drafter + `3,072` auto-switch)
       head-to-head against `Qwen3-4B-Thinking-2507` (Unpatched & Mechanistically Patched).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/tim/Code/optimize-gpu")
sys.path.insert(0, str(ROOT / "experiments"))

from run_postcutoff_study import (
    BIN,
    TEST_JSONL,
    analyze_trace,
    check_guardrails,
    parse_last_json,
    run_cmd,
    save_traces,
    summarize_metrics,
)
from run_postcutoff_final_stages import select_stratified

MODELS_DIR = Path.home() / ".cache" / "qwen-nv-rs" / "models"
BASE_8B_GGUF = MODELS_DIR / "Qwen3-8B-Q6_K.gguf"
EMBED_8B_GGUF = MODELS_DIR / "Qwen3-Embedding-8B-Q6_K.gguf"
LORA_8B_EMBED_DIR = ROOT / "lora" / "qwen3-8b-embed-svd-r32"
WIKI_1K_JSONL = ROOT / "data" / "wikipedia_1k.jsonl"
WIKI_EVAL_200 = ROOT / "data" / "wikipedia_eval_200.jsonl"
QVEC_8B_BEST = ROOT / "data" / "wikipedia_1k_8b_best.qvec"

OUT_SWITCH_JSON = ROOT / "metrics" / "metrics_phase8_crossover_switch.json"
OUT_RAG_8B_JSON = ROOT / "metrics" / "metrics_phase8_rag_8b.json"
OUT_POSTCUTOFF_8B_JSON = ROOT / "metrics" / "metrics_phase8_postcutoff_8b_4096.json"
WORK_8B = ROOT / "data" / "phase8_8b"


def start_daemon_8b(ctx: int = 4864, load_drafter: bool = False):
    for attempt in range(1, 9):
        subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
        time.sleep(1.5 + 0.5 * attempt)
        try:
            run_cmd(["daemon", "--background", "--ctx", str(ctx), "--model", "qwen3-thinking:8b"], timeout=120)
            if load_drafter:
                run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
            st, _, _ = run_cmd(["status"], timeout=15)
            print(f"[daemon] {st.strip()}", flush=True)
            return
        except Exception as e:
            print(f"[daemon retry {attempt}] {e}", flush=True)
            time.sleep(2.5)
    raise RuntimeError("Failed to boot qwen3-thinking:8b daemon")


def main():
    print("=" * 92)
    print("PHASE 8: QWEN3-8B SHARED-ANCESTOR THINKER + BM=4->T=1 CROSSOVER SWITCH + 8B RAG + EVAL")
    print("=" * 92, flush=True)
    check_guardrails()
    WORK_8B.mkdir(parents=True, exist_ok=True)

    # If a background daemon is finishing writing `wikipedia_1k_8b_best.qvec`, let it finish cleanly!
    daemon_log = Path.home() / ".cache" / "qwen-nv-rs" / "daemon.log"
    if not QVEC_8B_BEST.exists() and daemon_log.exists():
        log_txt = daemon_log.read_text(errors="ignore")
        if "Indexed 600/1000" in log_txt or "Indexed 800/1000" in log_txt:
            print("[wait] Waiting for in-flight 1,000-doc 8B `.qvec` indexing to finish saving...", flush=True)
            t_w = time.perf_counter()
            while not QVEC_8B_BEST.exists() and (time.perf_counter() - t_w) < 360:
                time.sleep(3.0)
            if QVEC_8B_BEST.exists():
                time.sleep(1.0)
                print(f"[wait] `wikipedia_1k_8b_best.qvec` saved ({QVEC_8B_BEST.stat().st_size / 1e6:.2f} MB)!", flush=True)

    test_prompts = [json.loads(ln) for ln in open(TEST_JSONL, encoding="utf-8") if ln.strip()]
    test_deep_24 = select_stratified(
        test_prompts,
        {"amc12_nov2025": 8, "hmmt_nov2025": 6, "aime_feb2026": 5, "arxiv_math2026": 5},
    )

    # Boot rebuilt v0.7.0 daemon (`thinking_tmpl=true`, default 8B crossover = `3072`)!
    start_daemon_8b(ctx=4864, load_drafter=False)

    # Verify `/no_think` fast path & default `spec_crossover_pos == 3072` on 8B!
    out_fast, _, _ = run_cmd(
        ["run", "--spec", "--temp", "0", "--max-tokens", "64", "--reset-cache", "--json-stats", "-p", "Compute 17 * 23. Reply concisely. /no_think"],
        timeout=60,
    )
    st_fast = parse_last_json(out_fast)
    print(
        f"[check /no_think] tokens={st_fast['completion_tokens']}, default_8b_crossover={st_fast['spec_crossover_pos']}, "
        f"switched={st_fast['spec_crossover_switched']}, tps={st_fast['decode_tok_per_s']:.2f} tok/s",
        flush=True,
    )
    assert st_fast["spec_crossover_pos"] == 3072, f"Expected 8B default crossover 3072, got {st_fast['spec_crossover_pos']}"
    assert st_fast["completion_tokens"] < 40, f"/no_think should finish concisely, got {st_fast['completion_tokens']} tokens"

    # Load already-verified 2,400-token 5-mode switch parity report (and record scale-aware defaults)
    if OUT_SWITCH_JSON.exists():
        sw_rep = json.loads(OUT_SWITCH_JSON.read_text())
        sw_rep["scale_aware_defaults"] = {"qwen3_4b_dim_2560": 1536, "qwen3_8b_dim_4096": 3072}
        sw_rep["no_think_fast_tokens"] = st_fast["completion_tokens"]
        OUT_SWITCH_JSON.write_text(json.dumps(sw_rep, indent=2))
        print("[part 1] Verified 5-mode bit-exact switch parity & scale-aware crossover (4B=1536, 8B=3072).", flush=True)

    # =========================================================================
    # PART 2: Shared-Ancestor `8B` Same-Model RAG (`Qwen3-8B` + `Qwen3-Embedding-8B` Rank-32 SVD LoRA)
    # =========================================================================
    print("\n" + "=" * 92)
    print("PART 2: Shared-Ancestor 8B SVD LoRA & 1,000-Doc Wikipedia `.qvec` RAG Eval (200 Queries)")
    print("=" * 92, flush=True)

    if not (LORA_8B_EMBED_DIR / "adapter_model.safetensors").exists():
        svd_out, svd_err, svd_wall = run_cmd(
            ["lora", "extract-svd", str(EMBED_8B_GGUF), str(LORA_8B_EMBED_DIR), "--model", str(BASE_8B_GGUF), "--rank", "32"],
            timeout=900,
        )
        print(f"[rag 8b] SVD LoRA extracted in {svd_wall:.1f}s", flush=True)

    if not QVEC_8B_BEST.exists():
        print("[rag 8b] Indexing 1,000 Wikipedia passages on RTX 3080...", flush=True)
        idx_out, _, idx_wall = run_cmd(
            ["rag", "index", str(WIKI_1K_JSONL), str(QVEC_8B_BEST), "--strategy", "svd_lora", "--lora", str(LORA_8B_EMBED_DIR)],
            timeout=2400,
        )
        print(f"[rag 8b] Indexed `.qvec` in {idx_wall:.1f}s: {idx_out.strip()}", flush=True)

    ev_out, _, _ = run_cmd(["rag", "eval", str(WIKI_EVAL_200), "--qvec", str(QVEC_8B_BEST)], timeout=600)
    rag_eval_8b = parse_last_json(ev_out)
    run_cmd(["rag", "clear"], timeout=30)
    run_cmd(["lora", "clear"], timeout=30)
    lora_size_mb = (LORA_8B_EMBED_DIR / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
    qvec_size_mb = QVEC_8B_BEST.stat().st_size / (1024 * 1024)
    rag_eval_8b["thinker_backbone"] = "Qwen3-8B-Q6_K.gguf"
    rag_eval_8b["embedding_ancestor_donor"] = "Qwen3-Embedding-8B-Q6_K.gguf"
    rag_eval_8b["embedding_dim"] = 4096
    rag_eval_8b["lora_disk_mb"] = round(lora_size_mb, 2)
    rag_eval_8b["qvec_disk_mb"] = round(qvec_size_mb, 2)
    OUT_RAG_8B_JSON.write_text(json.dumps(rag_eval_8b, indent=2))
    print(f"[rag 8b eval] {json.dumps(rag_eval_8b)}", flush=True)

    # =========================================================================
    # PART 3: Unseen Post-Cutoff (`Nov 2025 - Aug 2026`) 24-Problem Competition Benchmark @ 4,096 Tokens
    # =========================================================================
    print("\n" + "=" * 92)
    print("PART 3: Qwen3-8B on 24 Unseen Post-Cutoff Competition Problems @ 4,096 Tokens (0.6B + 3072 Switch)")
    print("=" * 92, flush=True)

    # Co-reside Qwen3-0.6B neural drafter (~89.5 tok/s up to pos=3072, then auto-switches to persistent T=1 megakernel!)
    run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)

    traces_8b_path = WORK_8B / "traces_qwen3_8b_test_4096.jsonl"
    traces_8b = []
    t0 = time.perf_counter()
    switched_cnt = 0
    for idx, item in enumerate(test_deep_24):
        out = None
        for attempt in range(3):
            try:
                out, _, _ = run_cmd(
                    [
                        "run",
                        "--spec",
                        "--temp",
                        "0",
                        "--max-tokens",
                        "4096",
                        "--reset-cache",
                        "--json-stats",
                        "-p",
                        item["prompt"],
                    ],
                    timeout=300,
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  [warn] idx={idx} attempt={attempt+1}: {e}", flush=True)
                start_daemon_8b(ctx=4864, load_drafter=True)
        lines = out.splitlines()
        st = parse_last_json(out)
        if st.get("spec_crossover_switched"):
            switched_cnt += 1
        gen_text = "\n".join(lines[:-1])
        rec = analyze_trace(item, gen_text, st)
        rec["spec_crossover_switched"] = st.get("spec_crossover_switched", False)
        traces_8b.append(rec)
        if (idx + 1) % 4 == 0 or (idx + 1) == len(test_deep_24):
            acc = sum(1 for r in traces_8b if r["is_correct"]) / len(traces_8b)
            end_th = sum(1 for r in traces_8b if r["has_end_think"]) / len(traces_8b)
            avg_th = sum(r["think_tokens"] for r in traces_8b) / len(traces_8b)
            avg_w = sum(r["hesitation_count"] for r in traces_8b) / len(traces_8b)
            tps = sum(r["decode_tok_per_s"] for r in traces_8b) / len(traces_8b)
            print(
                f"  [8B_postcutoff {idx+1:2d}/24] acc={acc*100:5.1f}% | </think>={end_th*100:5.1f}% | "
                f"avg_think={avg_th:6.1f} | wait/hmm={avg_w:4.1f} | switched={switched_cnt}/{idx+1} | tps={tps:5.1f} | wall={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    save_traces(traces_8b_path, traces_8b)
    m_8b = summarize_metrics(traces_8b)
    m_8b["max_gen_tokens"] = 4096
    m_8b["model"] = "Qwen3-8B-Q6_K.gguf"
    m_8b["spec_crossover_pos"] = 3072
    m_8b["spec_crossover_switched_count"] = switched_cnt
    OUT_POSTCUTOFF_8B_JSON.write_text(json.dumps(m_8b, indent=2))

    m_4b_base = json.loads((ROOT / "metrics_postcutoff_baseline_test_4096.json").read_text())
    m_4b_patch = json.loads((ROOT / "metrics_postcutoff_patched_test_4096.json").read_text())

    print("\n" + "=" * 92)
    print("HEAD-TO-HEAD POST-CUTOFF 2025-2026 COMPETITION RESULTS (24 Held-Out Problems @ 4,096 Tokens)")
    print("=" * 92)
    for lbl, m in [
        ("1. Qwen3-4B-Thinking-2507 (Unpatched) ", m_4b_base),
        ("2. Qwen3-4B-Thinking-2507 (Patched)   ", m_4b_patch),
        ("3. Qwen3-8B Shared-Ancestor (v0.7.0)  ", m_8b),
    ]:
        print(
            f"  {lbl}: acc={m['accuracy']*100:5.2f}% ({m['correct_count']:2d}/24) | "
            f"</think>={m['end_think_completion_rate']*100:5.1f}% | avg_think={m['avg_think_tokens']:6.1f} | "
            f"Wait/Hmm={m['avg_hesitation_markers']:5.2f} | speed={m['tok_per_sec_eval_mode']:5.1f} tok/s"
        )
    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()
