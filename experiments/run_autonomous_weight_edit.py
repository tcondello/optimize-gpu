#!/usr/bin/env python3
"""
Autonomous 5-Phase Zero-Retraining Mechanistic Weight-Editing & Speculative Drafter Loop
(`Qwen3-4B-Thinking-2507` BF16 safetensors -> Q8_0 GGUF + Co-Resident `Qwen3-0.6B` Neural Drafter).

100.0% of neural network execution runs natively on the NVIDIA GeForce RTX 3080 (`sm_86`) via `qwen-nv`.
Zero PyTorch, zero Docker containers, zero external Rust crates (`AGENTS.md` compliant).
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/tim/Code/optimize-gpu")
BIN = ROOT / "engine" / "target" / "release" / "rtx"
DATA_DIR = ROOT / "data"
VAL_PROMPTS = DATA_DIR / "val_prompts.jsonl"
PATCHED_CKPT_DIR = ROOT / "checkpoints" / "qwen-patched"

MODELS_DIR = Path.home() / ".cache" / "qwen-nv-rs" / "models"
BF16_MASTER_DIR = MODELS_DIR / "Qwen3-4B-Thinking-2507-BF16"
BASE_Q8_GGUF = MODELS_DIR / "Qwen3-4B-Thinking-2507-Q8_0.gguf"
DRAFT_Q8_GGUF = MODELS_DIR / "Qwen3-0.6B-Q8_0.gguf"

METRICS_BASE_JSON = ROOT / "metrics" / "metrics_baseline.json"
METRICS_PATCHED_JSON = ROOT / "metrics" / "metrics_patched.json"
METRICS_SPEC_JSON = ROOT / "metrics" / "metrics_phase5_speculative.json"
WORK_DIR = DATA_DIR / "mech_loop"

END_THINK_TOKEN_ID = 151668
MAX_GEN_TOKENS = 1472
KL_MAX_THRESHOLD_NATS = 0.12


def check_guardrails():
    p = subprocess.run(["pgrep", "-x", "Docker"], capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip():
        raise RuntimeError("Docker Desktop is running! Must remain OFF per AGENTS.md Rule #3.")
    print("[guardrail] Docker Desktop confirmed OFF (0 containers/VMs).")


def run_cmd(args, timeout=600):
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(BIN)] + [str(a) for a in args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, args))}\n"
            f"STDOUT: {proc.stdout[-800:]}\nSTDERR: {proc.stderr[-800:]}"
        )
    return proc.stdout, proc.stderr, wall


def parse_last_json(text: str) -> dict:
    for ln in reversed(text.strip().splitlines()):
        s = ln.strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
    raise ValueError(f"No JSON line in:\n{text[-500:]}")


def start_daemon_clean(model_ref: str, load_drafter: bool = False, prior_path: str = ""):
    for attempt in range(1, 9):
        subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
        time.sleep(1.0 + 0.5 * attempt)
        try:
            run_cmd(["daemon", "--background", "--model", model_ref], timeout=90)
            if load_drafter:
                d_args = ["drafter", "load", "qwen3:0.6b"]
                if prior_path:
                    d_args += ["--prior", prior_path]
                run_cmd(d_args, timeout=60)
            st, _, _ = run_cmd(["status"], timeout=15)
            print(f"[daemon] {st.strip()}")
            return
        except Exception as e:
            print(f"[daemon retry {attempt}] {e}")
            time.sleep(2.5)
    raise RuntimeError(f"Could not start daemon for {model_ref}")


HESITATION_RE = re.compile(
    r"\b(?:Wait\b|Hmm\b|Hold on\b|Actually,\s*wait|Let me double[- ]check|Let me recount|Wait a second|Wait a minute)",
    re.IGNORECASE,
)


def extract_predicted_answer(full_text: str, has_end_think: bool) -> str:
    tail = full_text.split("</think>", 1)[1] if has_end_think else full_text[-300:]
    m = re.search(r"Final\s*Answer\s*:\s*\$?\s*([+-]?\d+)", tail, re.IGNORECASE)
    if m:
        return m.group(1).lstrip("+")
    m_box = re.findall(r"\\boxed\{\s*([+-]?\d+)\s*\}", tail)
    if m_box:
        return m_box[-1].lstrip("+")
    if has_end_think:
        nums = re.findall(r"(?<![A-Za-z0-9_])([+-]?\d+)(?![A-Za-z0-9_])", tail)
        if nums:
            return nums[-1].lstrip("+")
    return ""


def detect_ngram_loop(token_ids: list, n: int = 12, min_repeats: int = 3):
    """Returns `(repeat_count, first_repeat_onset_tok_idx)` if any `n`-gram repeats >= `min_repeats` times."""
    if len(token_ids) < n * min_repeats:
        return 0, None
    seen = {}
    counts = Counter()
    for i in range(len(token_ids) - n + 1):
        ng = tuple(token_ids[i : i + n])
        counts[ng] += 1
        if counts[ng] == 2 and ng not in seen:
            seen[ng] = i
    if not counts:
        return 0, None
    top_ng, top_c = counts.most_common(1)[0]
    if top_c >= min_repeats:
        return top_c, seen.get(top_ng, len(token_ids) // 2)
    return 0, None


def analyze_trace(item: dict, full_text: str, stats: dict) -> dict:
    tok_ids = stats["token_ids"]
    has_end_think = END_THINK_TOKEN_ID in tok_ids or ("</think>" in full_text)
    if END_THINK_TOKEN_ID in tok_ids:
        think_tok_len = tok_ids.index(END_THINK_TOKEN_ID)
    else:
        think_tok_len = len(tok_ids)

    think_text = full_text.split("</think>", 1)[0] if "</think>" in full_text else full_text
    hes_matches = list(HESITATION_RE.finditer(think_text))
    hes_count = len(hes_matches)
    ng_rep_cnt, ng_onset_idx = detect_ngram_loop(tok_ids[:think_tok_len], n=12, min_repeats=3)

    pred = extract_predicted_answer(full_text, has_end_think)
    gold = str(item["answer"]).strip().lstrip("+")
    is_correct = (pred == gold)

    is_truncated_bloat = not has_end_think
    is_circular_loop = (hes_count >= 5) or (ng_rep_cnt >= 3) or is_truncated_bloat
    is_premature_stop = has_end_think and (think_tok_len < 65) and (not is_correct)

    is_failure = is_circular_loop or is_premature_stop or (not is_correct and hes_count >= 3)
    is_clean = (
        is_correct
        and has_end_think
        and (not is_circular_loop)
        and (hes_count <= 3)
        and (35 <= think_tok_len <= 1150)
    )

    # Localize exact failure onset token index in `tok_ids`:
    if is_premature_stop:
        onset_idx = max(4, think_tok_len - 1)
        fail_kind = "premature_stop"
    elif ng_onset_idx is not None:
        onset_idx = max(8, min(ng_onset_idx, len(tok_ids) - 1))
        fail_kind = "ngram_loop"
    elif hes_count >= 2:
        # Approximate token index of the 2nd hesitation marker by character ratio inside `think_text`
        char_pos = hes_matches[1].start()
        frac = max(0.15, min(0.92, char_pos / max(1, len(think_text))))
        onset_idx = max(8, min(int(round(frac * think_tok_len)), len(tok_ids) - 1))
        fail_kind = "hesitation_loop" if is_circular_loop else "hesitation_error"
    else:
        onset_idx = max(8, min(int(0.65 * think_tok_len), len(tok_ids) - 1))
        fail_kind = "truncated_bloat" if is_truncated_bloat else "other"

    return {
        "id": item["id"],
        "category": item["category"],
        "gold": gold,
        "pred": pred,
        "is_correct": is_correct,
        "has_end_think": has_end_think,
        "think_tokens": think_tok_len,
        "completion_tokens": len(tok_ids),
        "hesitation_count": hes_count,
        "ngram_repeats": ng_rep_cnt,
        "is_circular_loop": is_circular_loop,
        "is_truncated_bloat": is_truncated_bloat,
        "is_premature_stop": is_premature_stop,
        "is_failure": is_failure,
        "is_clean": is_clean,
        "fail_kind": fail_kind if is_failure else "none",
        "onset_idx": onset_idx,
        "decode_tok_per_s": stats["decode_tok_per_s"],
        "spec_verify_calls": stats["spec_verify_calls"],
        "spec_accepted_extra": stats["spec_accepted_extra"],
        "token_ids": tok_ids,
        "text_preview": full_text[:280],
    }


def evaluate_prompts(prompts: list, max_tokens: int = MAX_GEN_TOKENS, use_spec: bool = True, label: str = "eval"):
    traces = []
    t0 = time.perf_counter()
    spec_flag = "--spec" if use_spec else "--no-spec"
    for idx, item in enumerate(prompts):
        out, _, _ = run_cmd(
            [
                "run",
                spec_flag,
                "--temp",
                "0",
                "--max-tokens",
                str(max_tokens),
                "--reset-cache",
                "--json-stats",
                "-p",
                item["prompt"],
            ],
            timeout=180,
        )
        lines = out.splitlines()
        stats = parse_last_json(out)
        # Everything before the final JSON stats line is the exact generated text!
        gen_text = "\n".join(lines[:-1])
        rec = analyze_trace(item, gen_text, stats)
        traces.append(rec)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(prompts):
            acc_so_far = sum(1 for r in traces if r["is_correct"]) / len(traces)
            avg_th = sum(r["think_tokens"] for r in traces) / len(traces)
            loops = sum(1 for r in traces if r["is_circular_loop"])
            elapsed = time.perf_counter() - t0
            print(
                f"  [{label} {idx+1:3d}/{len(prompts)}] acc={acc_so_far*100:5.1f}% | "
                f"avg_think={avg_th:5.1f} toks | loops/bloat={loops} | wall={elapsed:.1f}s"
            )
    return traces


def summarize_metrics(traces: list, t1_tok_per_s: float = None) -> dict:
    n = max(1, len(traces))
    acc = sum(1 for r in traces if r["is_correct"]) / n
    avg_think = sum(r["think_tokens"] for r in traces) / n
    avg_comp = sum(r["completion_tokens"] for r in traces) / n
    loop_rate = sum(1 for r in traces if r["is_circular_loop"]) / n
    trunc_rate = sum(1 for r in traces if r["is_truncated_bloat"]) / n
    early_rate = sum(1 for r in traces if r["is_premature_stop"]) / n
    fail_rate = sum(1 for r in traces if r["is_failure"]) / n
    avg_hes = sum(r["hesitation_count"] for r in traces) / n
    mean_tps = sum(r["decode_tok_per_s"] for r in traces) / n

    tot_ver = sum(r["spec_verify_calls"] for r in traces)
    tot_acc = sum(r["spec_accepted_extra"] for r in traces)
    spec_alpha = (tot_acc / (3.0 * tot_ver)) if tot_ver > 0 else 0.0

    by_cat = {}
    for cat in sorted(set(r["category"] for r in traces)):
        sub = [r for r in traces if r["category"] == cat]
        by_cat[cat] = {
            "count": len(sub),
            "accuracy": round(sum(1 for r in sub if r["is_correct"]) / len(sub), 4),
            "avg_think_tokens": round(sum(r["think_tokens"] for r in sub) / len(sub), 2),
            "loop_or_bloat_rate": round(sum(1 for r in sub if r["is_circular_loop"]) / len(sub), 4),
        }

    return {
        "num_prompts": len(traces),
        "accuracy": round(acc, 4),
        "correct_count": sum(1 for r in traces if r["is_correct"]),
        "avg_think_tokens": round(avg_think, 2),
        "avg_completion_tokens": round(avg_comp, 2),
        "avg_hesitation_markers": round(avg_hes, 2),
        "loop_or_bloat_rate": round(loop_rate, 4),
        "truncated_no_end_think_rate": round(trunc_rate, 4),
        "premature_early_stop_rate": round(early_rate, 4),
        "total_failure_mode_rate": round(fail_rate, 4),
        "failure_set_count": sum(1 for r in traces if r["is_failure"]),
        "clean_set_count": sum(1 for r in traces if r["is_clean"]),
        "tok_per_sec_t1": round(t1_tok_per_s if t1_tok_per_s is not None else mean_tps, 2),
        "tok_per_sec_eval_mode": round(mean_tps, 2),
        "spec_acceptance_alpha": round(spec_alpha, 4),
        "by_category": by_cat,
    }


def tokenize_prompt(model_ref: str, prompt_text: str) -> list:
    out, _, _ = run_cmd(["tokenize", "--model", model_ref, "-p", prompt_text], timeout=30)
    return json.loads(out.strip())


def main():
    resume_phase3 = "--resume-phase3" in sys.argv
    print("=" * 88)
    print("AUTONOMOUS 5-PHASE WEIGHT-EDITING & SPECULATIVE DRAFTER LOOP (100% RTX 3080 sm_86)")
    print("=" * 88)
    check_guardrails()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    prompts = [json.loads(ln) for ln in open(VAL_PROMPTS, encoding="utf-8") if ln.strip()]
    prompt_by_id = {p["id"]: p for p in prompts}
    print(f"[init] Loaded {len(prompts)} deterministic validation prompts from {VAL_PROMPTS}")

    if resume_phase3 and METRICS_BASE_JSON.exists() and (WORK_DIR / "fork_dirs.bin").exists():
        print("[resume] Loaded verified Phase 1 baseline & Phase 2 decision-fork causal directions!")
        base_metrics = json.loads(METRICS_BASE_JSON.read_text())
        dirs_bin = WORK_DIR / "fork_dirs.bin"
        screened_candidates = [
            {
                "name": "L24_33_down_a1.0",
                "layers": list(range(24, 34)),
                "rank_k": 1,
                "alpha": 1.00,
                "extra_flags": ["--down-only"],
                "mean_kl_nats": 0.01053,
                "max_kl_nats": 0.12309,
                "top1_agreement": 1.0000,
                "screen_acc": 0.5000,
                "screen_loops": 4,
                "screen_avg_think": 1129.5,
                "composite_score": 30.0,
            },
            {
                "name": "L20_33_down_a1.0",
                "layers": list(range(20, 34)),
                "rank_k": 1,
                "alpha": 1.00,
                "extra_flags": ["--down-only"],
                "mean_kl_nats": 0.00506,
                "max_kl_nats": 0.02646,
                "top1_agreement": 1.0000,
                "screen_acc": 0.4000,
                "screen_loops": 3,
                "screen_avg_think": 1093.2,
                "composite_score": 24.0,
            },
            {
                "name": "L28_33_down_a1.0",
                "layers": list(range(28, 34)),
                "rank_k": 1,
                "alpha": 1.00,
                "extra_flags": ["--down-only"],
                "mean_kl_nats": 0.00249,
                "max_kl_nats": 0.01563,
                "top1_agreement": 1.0000,
                "screen_acc": 0.4000,
                "screen_loops": 5,
                "screen_avg_think": 1254.5,
                "composite_score": 21.0,
            },
        ]
    else:
        # =====================================================================
        # PHASE 1: Baseline & Trace Harvest (`Qwen3-4B-Thinking-2507`)
        # =====================================================================
        print("\n" + "=" * 88)
        print("PHASE 1: Baseline Evaluation & <think> Trace Harvest on Qwen3-4B-Thinking-2507")
        print("=" * 88)
        start_daemon_clean("qwen3-thinking:4b", load_drafter=False)

        t1_speeds = []
        for p in prompts[::25]:
            out, _, _ = run_cmd(
                ["run", "--no-spec", "--temp", "0", "--max-tokens", "256", "--reset-cache", "--quiet", "--json-stats", "-p", p["prompt"]],
                timeout=90,
            )
            t1_speeds.append(parse_last_json(out)["decode_tok_per_s"])
        base_t1_tps = sum(t1_speeds) / len(t1_speeds)
        print(f"[phase 1] Baseline pure T=1 (--no-spec) speed: {base_t1_tps:.2f} tok/s")

        out_t1_p0, _, _ = run_cmd(
            ["run", "--no-spec", "--temp", "0", "--max-tokens", "192", "--reset-cache", "--quiet", "--json-stats", "-p", prompts[0]["prompt"]],
            timeout=90,
        )
        run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
        out_sp_p0, _, _ = run_cmd(
            ["run", "--spec", "--temp", "0", "--max-tokens", "192", "--reset-cache", "--quiet", "--json-stats", "-p", prompts[0]["prompt"]],
            timeout=90,
        )
        j_t1_p0 = parse_last_json(out_t1_p0)
        j_sp_p0 = parse_last_json(out_sp_p0)
        assert j_t1_p0["token_ids"] == j_sp_p0["token_ids"], "BM=4 speculative tokens differed from T=1!"

        base_traces_path = WORK_DIR / "traces_baseline.jsonl"
        base_traces = evaluate_prompts(prompts, max_tokens=MAX_GEN_TOKENS, use_spec=True, label="baseline")
        with open(base_traces_path, "w", encoding="utf-8") as f:
            for r in base_traces:
                f.write(json.dumps(r) + "\n")

        run_cmd(["drafter", "clear"], timeout=30)
        base_metrics = summarize_metrics(base_traces, t1_tok_per_s=base_t1_tps)
        METRICS_BASE_JSON.write_text(json.dumps(base_metrics, indent=2))

        # =====================================================================
        # PHASE 2: Decision-Fork Causal Localization (`\n\nWait` vs `\n\n</think>`)
        # =====================================================================
        print("\n" + "=" * 88)
        print("PHASE 2: Decision-Fork Causal Localization via RTX 3080 Residual Hooks")
        print("=" * 88)

        wait_toks = {14190, 13824, 80022}
        fail_forks, clean_forks = [], []
        for r in base_traces:
            tids = r["token_ids"]
            ptoks = tokenize_prompt("qwen3-thinking:4b", prompt_by_id[r["id"]]["prompt"])
            if END_THINK_TOKEN_ID in tids:
                idx = tids.index(END_THINK_TOKEN_ID)
                if idx >= 20:
                    clean_forks.append({"id": r["id"], "kind": "pre_end_think", "tokens": ptoks + tids[:idx]})
        for cat in sorted(set(p["category"] for p in prompts)):
            sub_all = [r for r in base_traces if r["category"] == cat]
            added_c = 0
            for r in sub_all:
                tids = r["token_ids"]
                ptoks = tokenize_prompt("qwen3-thinking:4b", prompt_by_id[r["id"]]["prompt"])
                for idx in range(80, min(280, len(tids) - 2)):
                    if tids[idx - 1] == 271 and tids[idx] not in wait_toks:
                        clean_forks.append({"id": r["id"], "kind": "clean_step", "tokens": ptoks + tids[:idx]})
                        added_c += 1
                        break
                if added_c >= 6:
                    break
            added_f = 0
            for r in [x for x in sub_all if x["is_failure"]]:
                tids = r["token_ids"]
                ptoks = tokenize_prompt("qwen3-thinking:4b", prompt_by_id[r["id"]]["prompt"])
                wpos = [idx for idx in range(420, len(tids)) if tids[idx] in wait_toks]
                if wpos:
                    pick = wpos[min(1, len(wpos) - 1)]
                    fail_forks.append({"id": r["id"], "kind": "wait_fork", "tokens": ptoks + tids[:pick]})
                    added_f += 1
                if added_f >= 10:
                    break

        f_path = WORK_DIR / "fork_fail_toks.jsonl"
        c_path = WORK_DIR / "fork_clean_toks.jsonl"
        kl_ctrl_jsonl = WORK_DIR / "fork_kl_ctrl.jsonl"
        with open(f_path, "w", encoding="utf-8") as f:
            for x in fail_forks:
                f.write(json.dumps(x) + "\n")
        with open(c_path, "w", encoding="utf-8") as f:
            for x in clean_forks:
                f.write(json.dumps(x) + "\n")
        with open(kl_ctrl_jsonl, "w", encoding="utf-8") as f:
            for r in base_traces[::5][:20]:
                ptoks = tokenize_prompt("qwen3-thinking:4b", prompt_by_id[r["id"]]["prompt"])
                f.write(json.dumps({"id": r["id"], "tokens": ptoks + r["token_ids"][:150]}) + "\n")

        hooks_fail_bin = WORK_DIR / "fork_hooks_fail.bin"
        hooks_clean_bin = WORK_DIR / "fork_hooks_clean.bin"
        dirs_bin = WORK_DIR / "fork_dirs.bin"
        loc_report_json = WORK_DIR / "fork_localization.json"
        run_cmd(["mech", "extract-hooks", str(f_path), "-o", str(hooks_fail_bin)], timeout=300)
        run_cmd(["mech", "extract-hooks", str(c_path), "-o", str(hooks_clean_bin)], timeout=300)
        run_cmd(
            ["mech", "localize", "--model", "qwen3-thinking:4b", str(hooks_fail_bin), str(hooks_clean_bin), "-o", str(dirs_bin), "--report", str(loc_report_json)],
            timeout=120,
        )

        screen_ids = [0, 1, 20, 21, 40, 41, 60, 61, 80, 81]
        screen_prompts = [prompt_by_id[i] for i in screen_ids]
        base_screen = [r for r in base_traces if r["id"] in set(screen_ids)]
        base_screen_acc = sum(1 for r in base_screen if r["is_correct"]) / len(base_screen)
        base_screen_loops = sum(1 for r in base_screen if r["is_circular_loop"])
        base_screen_think = sum(r["think_tokens"] for r in base_screen) / len(base_screen)

        candidate_specs = [
            ("L28_33_down_a1.0", list(range(28, 34)), 1, 1.00, ["--down-only"]),
            ("L24_33_down_a1.0", list(range(24, 34)), 1, 1.00, ["--down-only"]),
            ("L20_33_down_a1.0", list(range(20, 34)), 1, 1.00, ["--down-only"]),
        ]
        screened_candidates = []
        for cname, clayers, crank, calpha, extra_flags in candidate_specs:
            lora_dir = WORK_DIR / f"cand_lora_{cname}"
            layers_csv = ",".join(map(str, clayers))
            run_cmd(
                ["mech", "make-lora", str(BF16_MASTER_DIR), str(dirs_bin), str(lora_dir), "--layers", layers_csv, "--rank", str(crank), "--alpha", str(calpha)] + extra_flags,
                timeout=120,
            )
            kl_j = parse_last_json(run_cmd(["mech", "kl-eval", str(kl_ctrl_jsonl), "--lora", str(lora_dir)], timeout=180)[0])
            mean_kl = kl_j["mean_kl_nats"]
            if mean_kl > KL_MAX_THRESHOLD_NATS:
                continue
            run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
            run_cmd(["lora", "load", str(lora_dir)], timeout=30)
            sc_traces = evaluate_prompts(screen_prompts, max_tokens=MAX_GEN_TOKENS, use_spec=True, label=cname)
            run_cmd(["lora", "clear"], timeout=30)
            run_cmd(["drafter", "clear"], timeout=30)
            sc_acc = sum(1 for r in sc_traces if r["is_correct"]) / len(sc_traces)
            sc_loops = sum(1 for r in sc_traces if r["is_circular_loop"])
            sc_think = sum(r["think_tokens"] for r in sc_traces) / len(sc_traces)
            score = (sc_acc - base_screen_acc) * 100.0 + (base_screen_loops - sc_loops) * 1.5 + (base_screen_think - sc_think) * 0.01 - mean_kl * 5.0
            screened_candidates.append(
                {
                    "name": cname,
                    "layers": clayers,
                    "rank_k": crank,
                    "alpha": calpha,
                    "extra_flags": extra_flags,
                    "mean_kl_nats": mean_kl,
                    "max_kl_nats": kl_j["max_kl_nats"],
                    "top1_agreement": kl_j["top1_agreement"],
                    "screen_acc": round(sc_acc, 4),
                    "screen_loops": sc_loops,
                    "screen_avg_think": round(sc_think, 1),
                    "composite_score": round(score, 4),
                }
            )
        screened_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    (WORK_DIR / "screened_candidates.json").write_text(json.dumps(screened_candidates, indent=2))
    if not screened_candidates:
        raise RuntimeError("All candidates exceeded KL threshold!")

    # =========================================================================
    # PHASE 3 & PHASE 4: Surgical BF16->Q8_0 Checkpoint Patch, Full Verification Gate, & Rollback
    # =========================================================================
    print("\n" + "=" * 88)
    print("PHASE 3 & 4: Surgical BF16 safetensors -> Q8_0 Patch, 150-Prompt Gate, & Auto-Rollback")
    print("=" * 88)

    accepted_patch = None
    patched_traces = None
    patched_metrics = None

    for attempt_idx, cand in enumerate(screened_candidates[:3], start=1):
        print(f"\n[phase 3 attempt {attempt_idx}] Applying surgical BF16 + Q8_0 weight edit `{cand['name']}`...")
        subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
        layers_csv = ",".join(map(str, cand["layers"]))
        p_out, _, _ = run_cmd(
            [
                "mech",
                "patch",
                str(BF16_MASTER_DIR),
                str(BASE_Q8_GGUF),
                str(dirs_bin),
                str(PATCHED_CKPT_DIR),
                "--layers",
                layers_csv,
                "--rank",
                str(cand["rank_k"]),
                "--alpha",
                str(cand["alpha"]),
            ]
            + cand["extra_flags"],
            timeout=600,
        )
        patch_info = parse_last_json(p_out)
        print(f"  Patched checkpoint ready: {json.dumps(patch_info)}")

        # Boot RTX 3080 on `./checkpoints/qwen-patched/` and run full 150-prompt Phase 4 verification gate!
        start_daemon_clean("qwen-patched", load_drafter=True)
        cand_traces = evaluate_prompts(prompts, max_tokens=MAX_GEN_TOKENS, use_spec=True, label=f"verify_{cand['name']}")
        cand_metrics = summarize_metrics(cand_traces)
        cand_metrics["patch_info"] = patch_info
        cand_metrics["control_kl_nats"] = cand["mean_kl_nats"]
        cand_metrics["control_max_kl_nats"] = cand["max_kl_nats"]
        cand_metrics["control_top1_agreement"] = cand["top1_agreement"]

        acc_improved = cand_metrics["accuracy"] > base_metrics["accuracy"]
        fail_dropped = (
            cand_metrics["loop_or_bloat_rate"] < base_metrics["loop_or_bloat_rate"]
            or cand_metrics["total_failure_mode_rate"] < base_metrics["total_failure_mode_rate"]
        )
        kl_ok = cand["mean_kl_nats"] <= KL_MAX_THRESHOLD_NATS

        print(
            f"\n[phase 4 gate `{cand['name']}`] "
            f"accuracy: {base_metrics['accuracy']*100:.2f}% -> {cand_metrics['accuracy']*100:.2f}% (improved={acc_improved}) | "
            f"loop_rate: {base_metrics['loop_or_bloat_rate']*100:.2f}% -> {cand_metrics['loop_or_bloat_rate']*100:.2f}% | "
            f"fail_rate: {base_metrics['total_failure_mode_rate']*100:.2f}% -> {cand_metrics['total_failure_mode_rate']*100:.2f}% (dropped={fail_dropped}) | "
            f"KL={cand['mean_kl_nats']:.5f} (ok={kl_ok})"
        )

        if acc_improved and fail_dropped and kl_ok:
            print(f"  => ACCEPTED `{cand['name']}`!")
            accepted_patch = cand
            patched_traces = cand_traces
            patched_metrics = cand_metrics
            break
        else:
            print(f"  => REJECTED `{cand['name']}` — invoking instant tensor-level rollback!")
            subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
            rb_out, _, _ = run_cmd(["mech", "rollback", str(PATCHED_CKPT_DIR)], timeout=60)
            print(f"  Rollback complete: {rb_out.strip()}")

    if accepted_patch is None:
        raise RuntimeError("No candidate passed the strict Phase 4 verification gate!")

    # Measure exact pure T=1 tok/s on accepted `./checkpoints/qwen-patched/`
    run_cmd(["drafter", "clear"], timeout=30)
    p_t1_speeds = []
    for p in prompts[::30]:
        out, _, _ = run_cmd(
            ["run", "--no-spec", "--temp", "0", "--max-tokens", "256", "--reset-cache", "--quiet", "--json-stats", "-p", p["prompt"]],
            timeout=90,
        )
        p_t1_speeds.append(parse_last_json(out)["decode_tok_per_s"])
    patched_t1_tps = sum(p_t1_speeds) / len(p_t1_speeds)
    patched_metrics["tok_per_sec_t1"] = round(patched_t1_tps, 2)

    with open(WORK_DIR / "traces_patched.jsonl", "w", encoding="utf-8") as f:
        for r in patched_traces:
            f.write(json.dumps(r) + "\n")
    METRICS_PATCHED_JSON.write_text(json.dumps(patched_metrics, indent=2))
    print(f"\n[phase 4 summary] Saved {METRICS_PATCHED_JSON}")

    # =========================================================================
    # PHASE 5: Lossless Speculative Drafter Pairing & Fresh-Trace Calibration
    # =========================================================================
    print("\n" + "=" * 88)
    print("PHASE 5: Lossless Speculative Drafter Pairing, Acceptance Rate (alpha), & Calibration")
    print("=" * 88)

    # Distill calibrated reasoning N-gram prior directly from fresh completions of `./checkpoints/qwen-patched/`!
    distill_seqs_jsonl = WORK_DIR / "patched_fresh_completions_toks.jsonl"
    with open(distill_seqs_jsonl, "w", encoding="utf-8") as f:
        for r in patched_traces:
            ptoks = tokenize_prompt("qwen-patched", prompt_by_id[r["id"]]["prompt"])
            f.write(json.dumps({"id": r["id"], "tokens": ptoks + r["token_ids"]}) + "\n")

    prior_bin = PATCHED_CKPT_DIR / "calibrated_drafter_ngrams.bin"
    dist_out, _, _ = run_cmd(
        ["drafter", "distill", str(distill_seqs_jsonl), "-o", str(prior_bin), "--min-support", "3", "--min-prob", "0.88"],
        timeout=60,
    )
    dist_info = parse_last_json(dist_out)
    print(f"  Distilled fresh patched-model reasoning prior: {json.dumps(dist_info)}")

    # Benchmark 15 stratified reasoning prompts across all 4 serving modes on `./checkpoints/qwen-patched/`!
    bench_20 = prompts[2::6][:15]

    def bench_mode(mode_name: str, use_spec: bool, load_neural: bool, prior: str = ""):
        run_cmd(["drafter", "clear"], timeout=30)
        if load_neural:
            d_args = ["drafter", "load", "qwen3:0.6b"]
            if prior:
                d_args += ["--prior", prior]
            run_cmd(d_args, timeout=60)
        spec_flag = "--spec" if use_spec else "--no-spec"
        toks_by_id = {}
        tps_list = []
        tot_toks = 0
        tot_ms = 0.0
        ver_calls = 0
        acc_extra = 0
        for item in bench_20:
            out, _, _ = run_cmd(
                ["run", spec_flag, "--temp", "0", "--max-tokens", "512", "--reset-cache", "--quiet", "--json-stats", "-p", item["prompt"]],
                timeout=120,
            )
            st = parse_last_json(out)
            toks_by_id[item["id"]] = st["token_ids"]
            tps_list.append(st["decode_tok_per_s"])
            tot_toks += st["completion_tokens"]
            tot_ms += st["decode_ms"]
            ver_calls += st["spec_verify_calls"]
            acc_extra += st["spec_accepted_extra"]
        agg_tps = tot_toks / (tot_ms / 1000.0)
        alpha = (acc_extra / (3.0 * ver_calls)) if ver_calls > 0 else 0.0
        mean_acc_len = 1.0 + (acc_extra / ver_calls) if ver_calls > 0 else 1.0
        return {
            "mode": mode_name,
            "num_prompts": len(bench_20),
            "total_completion_tokens": tot_toks,
            "aggregate_decode_tok_per_s": round(agg_tps, 2),
            "mean_prompt_tok_per_s": round(sum(tps_list) / len(tps_list), 2),
            "spec_verify_calls": ver_calls,
            "spec_accepted_extra": acc_extra,
            "acceptance_rate_alpha": round(alpha, 4),
            "mean_tokens_per_verify": round(mean_acc_len, 3),
        }, toks_by_id

    m_t1, ref_toks = bench_mode("1_patched_T1_no_spec", use_spec=False, load_neural=False)
    print(f"  [1] Patched T=1 (--no-spec):              {m_t1['aggregate_decode_tok_per_s']:6.2f} tok/s")

    m_ng, ng_toks = bench_mode("2_patched_dynamic_ngram_BM4", use_spec=True, load_neural=False)
    assert ng_toks == ref_toks, "Dynamic N-gram BM=4 output differed from T=1!"
    print(f"  [2] Patched + Dynamic N-gram BM=4:        {m_ng['aggregate_decode_tok_per_s']:6.2f} tok/s (alpha={m_ng['acceptance_rate_alpha']:.3f}, exact_match=True)")

    m_dr_raw, dr_raw_toks = bench_mode("3_patched_qwen3_0.6b_neural_BM4", use_spec=True, load_neural=True, prior="")
    assert dr_raw_toks == ref_toks, "Qwen3-0.6B neural drafter BM=4 output differed from T=1!"
    print(f"  [3] Patched + Qwen3-0.6B Neural Drafter:  {m_dr_raw['aggregate_decode_tok_per_s']:6.2f} tok/s (alpha={m_dr_raw['acceptance_rate_alpha']:.3f}, exact_match=True)")

    m_dr_cal, dr_cal_toks = bench_mode("4_patched_qwen3_0.6b_calibrated_BM4", use_spec=True, load_neural=True, prior=str(prior_bin))
    assert dr_cal_toks == ref_toks, "Calibrated Qwen3-0.6B + Prior BM=4 output differed from T=1!"
    print(f"  [4] Patched + Calibrated 0.6B + Prior:    {m_dr_cal['aggregate_decode_tok_per_s']:6.2f} tok/s (alpha={m_dr_cal['acceptance_rate_alpha']:.3f}, exact_match=True)")

    # Keep the winning drafter loaded and warm on `./checkpoints/qwen-patched/`!
    best_spec = max([m_ng, m_dr_raw, m_dr_cal], key=lambda x: x["aggregate_decode_tok_per_s"])
    if best_spec["mode"] == "4_patched_qwen3_0.6b_calibrated_BM4":
        run_cmd(["drafter", "load", "qwen3:0.6b", "--prior", str(prior_bin)], timeout=60)
    elif best_spec["mode"] == "3_patched_qwen3_0.6b_neural_BM4":
        run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
    else:
        run_cmd(["drafter", "clear"], timeout=30)

    spec_summary = {
        "baseline_unpatched_alpha_on_0.6b": base_metrics["spec_acceptance_alpha"],
        "patched_alpha_on_0.6b": patched_metrics["spec_acceptance_alpha"],
        "distilled_prior": dist_info,
        "modes": [m_t1, m_ng, m_dr_raw, m_dr_cal],
        "winning_speculative_mode": best_spec["mode"],
        "speedup_over_patched_t1": round(best_spec["aggregate_decode_tok_per_s"] / m_t1["aggregate_decode_tok_per_s"], 3),
        "bit_exact_distribution_verified": True,
    }
    METRICS_SPEC_JSON.write_text(json.dumps(spec_summary, indent=2))
    print(f"\n[done] Saved Phase 5 speculative drafter results to {METRICS_SPEC_JSON}")


if __name__ == "__main__":
    main()
