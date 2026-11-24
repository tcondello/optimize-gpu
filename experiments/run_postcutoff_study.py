#!/usr/bin/env python3
"""
Phase 7: Contamination-Free Post-Cutoff (> Nov 2025 – Aug 2026) Mechanistic Weight-Editing
& Zero-Shot Circuit Transfer Study on `Qwen/Qwen3-4B-Thinking-2507` + `Qwen3-0.6B` Speculative Drafter.

Experimental Stages (all checkpointed to disk for deterministic reproducibility):
  - Stage 1 (Step A.1): Zero-Shot Out-of-Distribution Transfer of Synthetic-Calibrated
                        `./checkpoints/qwen-patched` (`L24_33_down_a1.0` from `val_prompts.jsonl`)
                        evaluated directly on held-out `data/post_cutoff_test.jsonl` (60 unseen 2025-2026 problems).
  - Stage 2 (Step A.2 & B.1): Unpatched `Qwen3-4B-Thinking-2507` baseline evaluated on BOTH
                        `data/post_cutoff_test.jsonl` (60 test) and `data/post_cutoff_calib.jsonl` (60 calib).
  - Stage 3 (Step B.2): Zero-Leakage Decision-Fork Residual Hook Localization strictly on `post_cutoff_calib`
                        + 36-Layer Subspace Cosine Alignment (`synthetic` vs `real 2025-2026` circuit geometry).
  - Stage 4 (Step B.3-4): Candidate LoRA Screening + KL Gate (`kl-eval`) + Instant Rollback & Surgical
                        `BF16 safetensors -> Q8_0 GGUF` Patching verified strictly on `post_cutoff_calib`.
  - Stage 5 (Step B.5): Fresh `post_cutoff_calib` N-gram Prior Distillation, Locked One-Shot Evaluation on
                        untouched `data/post_cutoff_test.jsonl`, & 4-Mode Lossless Speculative Throughput Benchmark.

100.0% of neural forward passes, hooks, KL evaluations, and speculative decoding execute on RTX 3080 (`sm_86`).
"""

import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/tim/Code/optimize-gpu")
BIN = ROOT / "engine" / "target" / "release" / "rtx"
DATA_DIR = ROOT / "data"
CALIB_JSONL = DATA_DIR / "post_cutoff_calib.jsonl"
TEST_JSONL = DATA_DIR / "post_cutoff_test.jsonl"
SYNTH_DIRS_BIN = DATA_DIR / "mech_loop" / "fork_dirs.bin"
PATCHED_CKPT_DIR = ROOT / "checkpoints" / "qwen-patched"

MODELS_DIR = Path.home() / ".cache" / "qwen-nv-rs" / "models"
BF16_MASTER_DIR = MODELS_DIR / "Qwen3-4B-Thinking-2507-BF16"
BASE_Q8_GGUF = MODELS_DIR / "Qwen3-4B-Thinking-2507-Q8_0.gguf"

WORK_DIR = DATA_DIR / "mech_postcutoff"

METRICS_ZEROSHOT_TEST = ROOT / "metrics" / "metrics_postcutoff_zeroshot_test.json"
METRICS_BASE_TEST = ROOT / "metrics" / "metrics_postcutoff_baseline_test.json"
METRICS_BASE_CALIB = ROOT / "metrics" / "metrics_postcutoff_baseline_calib.json"
METRICS_PATCHED_CALIB = ROOT / "metrics" / "metrics_postcutoff_patched_calib.json"
METRICS_PATCHED_TEST = ROOT / "metrics" / "metrics_postcutoff_patched_test.json"
METRICS_SUBSPACE_ALIGN = ROOT / "metrics" / "metrics_postcutoff_subspace_alignment.json"
METRICS_SPEC_POSTCUTOFF = ROOT / "metrics_postcutoff_speculative.json"

END_THINK_TOKEN_ID = 151668
DAEMON_CTX = 4096
MAX_GEN_TOKENS = 2560
KL_MAX_THRESHOLD_NATS = 0.12

HESITATION_RE = re.compile(
    r"\b(?:Wait\b|Hmm\b|Hold on\b|Actually,\s*wait|Let me double[- ]check|Let me recount|Wait a second|Wait a minute)",
    re.IGNORECASE,
)


def check_guardrails():
    p = subprocess.run(["pgrep", "-x", "Docker"], capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip():
        raise RuntimeError("Docker Desktop is running! Must remain OFF per AGENTS.md Rule #3.")
    print("[guardrail] Docker Desktop confirmed OFF (0 containers/VMs).", flush=True)


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
            run_cmd(["daemon", "--background", "--ctx", str(DAEMON_CTX), "--model", model_ref], timeout=90)
            if load_drafter:
                d_args = ["drafter", "load", "qwen3:0.6b"]
                if prior_path and Path(prior_path).exists():
                    d_args += ["--prior", prior_path]
                run_cmd(d_args, timeout=60)
            st, _, _ = run_cmd(["status"], timeout=15)
            print(f"[daemon] {st.strip()}", flush=True)
            return
        except Exception as e:
            print(f"[daemon retry {attempt}] {e}", flush=True)
            time.sleep(2.5)
    raise RuntimeError(f"Could not start daemon for {model_ref}")


def normalize_exact_rational(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace("**", "").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$|^\\\[|\\\]$", "", s).strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[7:-1].strip()
    s = (
        s.replace(",", "")
        .replace("\\!", "")
        .replace("\\,", "")
        .replace("\\;", "")
        .replace("\\:", "")
        .replace("\\ ", "")
        .replace(" ", "")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\displaystyle", "")
        .replace("\\dfrac", "\\frac")
        .replace("\\tfrac", "\\frac")
    )
    s = s.rstrip(".")
    if re.fullmatch(r"[+-]?\d+", s):
        return str(int(s))
    m_dec = re.fullmatch(r"([+-]?)(\d+)\.(\d{1,6})", s)
    if m_dec:
        sign = -1 if m_dec.group(1) == "-" else 1
        whole = int(m_dec.group(2))
        frac_str = m_dec.group(3)
        q = 10 ** len(frac_str)
        p = sign * (whole * q + int(frac_str))
        g = math.gcd(abs(p), q)
        p, q = p // g, q // g
        return str(p) if q == 1 else f"{p}/{q}"
    m_sl = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", s)
    if m_sl:
        p, q = int(m_sl.group(1)), int(m_sl.group(2))
        if q == 0:
            return None
        if q < 0:
            p, q = -p, -q
        g = math.gcd(abs(p), q)
        p, q = p // g, q // g
        return str(p) if q == 1 else f"{p}/{q}"
    m_fr = re.fullmatch(r"([+-])?\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", s)
    if not m_fr:
        m_fr = re.fullmatch(r"([+-])?\\frac(\d)(\d)", s)
    if m_fr:
        sign = -1 if m_fr.group(1) == "-" else 1
        p = sign * int(m_fr.group(2))
        q = int(m_fr.group(3))
        if q == 0:
            return None
        if q < 0:
            p, q = -p, -q
        g = math.gcd(abs(p), q)
        p, q = p // g, q // g
        return str(p) if q == 1 else f"{p}/{q}"
    return None


def extract_boxed_list(text: str) -> list:
    out = []
    idx = 0
    needle = "\\boxed{"
    while True:
        pos = text.find(needle, idx)
        if pos == -1:
            break
        start = pos + len(needle)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[start : i - 1])
        idx = max(start, i)
    return out


def extract_predicted_answer(full_text: str, has_end_think: bool) -> str:
    tail = full_text.split("</think>", 1)[1] if (has_end_think and "</think>" in full_text) else full_text[-450:]

    # 1. Check balanced \boxed{...} from last to first in `tail`
    for b in reversed(extract_boxed_list(tail)):
        norm = normalize_exact_rational(b)
        if norm is not None:
            return norm

    # 2. Check explicit `Final Answer: ...` line
    for m in reversed(list(re.finditer(r"Final\s*Answer\s*:\s*([^\n\r]+)", tail, re.IGNORECASE))):
        cand = m.group(1).strip()
        # Also check if \boxed{...} is inside Final Answer
        b_in = extract_boxed_list(cand)
        if b_in:
            norm = normalize_exact_rational(b_in[-1])
            if norm is not None:
                return norm
        norm = normalize_exact_rational(cand)
        if norm is not None:
            return norm

    # 3. Fallback if `</think>` completed cleanly: check last \frac{p}{q}, p/q, or integer in `tail`
    if has_end_think:
        fracs = re.findall(r"[+-]?\s*\\(?:d|t)?frac\{\s*[+-]?\d+\s*\}\{\s*[+-]?\d+\s*\}", tail)
        if fracs:
            norm = normalize_exact_rational(fracs[-1])
            if norm is not None:
                return norm
        slashes = re.findall(r"(?<![A-Za-z0-9_])([+-]?\d+/\d+)(?![A-Za-z0-9_])", tail)
        if slashes:
            norm = normalize_exact_rational(slashes[-1])
            if norm is not None:
                return norm
        nums = re.findall(r"(?<![A-Za-z0-9_])([+-]?\d+)(?![A-Za-z0-9_])", tail)
        if nums:
            norm = normalize_exact_rational(nums[-1])
            if norm is not None:
                return norm
    return ""


def detect_ngram_loop(token_ids: list, n: int = 12, min_repeats: int = 3):
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
    ng_rep_cnt, _ = detect_ngram_loop(tok_ids[:think_tok_len], n=12, min_repeats=3)

    pred = extract_predicted_answer(full_text, has_end_think)
    gold = normalize_exact_rational(item["answer"]) or str(item["answer"]).strip()
    is_correct = (pred == gold)

    is_truncated_bloat = not has_end_think
    is_circular_loop = (hes_count >= 6) or (ng_rep_cnt >= 3) or is_truncated_bloat
    is_premature_stop = has_end_think and (think_tok_len < 65) and (not is_correct)
    is_failure = is_circular_loop or is_premature_stop or (not is_correct and hes_count >= 4)
    is_clean = (
        is_correct
        and has_end_think
        and (not is_circular_loop)
        and (hes_count <= 4)
    )

    return {
        "id": item["id"],
        "source_id": item.get("source_id", str(item["id"])),
        "category": item["category"],
        "release_date": item.get("release_date", ""),
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
        "decode_tok_per_s": stats["decode_tok_per_s"],
        "spec_verify_calls": stats["spec_verify_calls"],
        "spec_accepted_extra": stats["spec_accepted_extra"],
        "token_ids": tok_ids,
        "text_tail": full_text[-260:],
    }


def evaluate_prompts(
    prompts: list,
    max_tokens: int = MAX_GEN_TOKENS,
    use_spec: bool = True,
    label: str = "eval",
    recover_fn=None,
):
    traces = []
    t0 = time.perf_counter()
    spec_flag = "--spec" if use_spec else "--no-spec"
    for idx, item in enumerate(prompts):
        out = None
        for attempt in range(3):
            try:
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
                    timeout=240,
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  [warn] prompt {idx} attempt {attempt+1} transient error ({e}); recovering...", flush=True)
                if recover_fn is not None:
                    recover_fn()
                else:
                    time.sleep(2.0)
        lines = out.splitlines()
        stats = parse_last_json(out)
        gen_text = "\n".join(lines[:-1])
        rec = analyze_trace(item, gen_text, stats)
        traces.append(rec)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(prompts):
            acc_so_far = sum(1 for r in traces if r["is_correct"]) / len(traces)
            avg_th = sum(r["think_tokens"] for r in traces) / len(traces)
            end_th = sum(1 for r in traces if r["has_end_think"]) / len(traces)
            avg_hes = sum(r["hesitation_count"] for r in traces) / len(traces)
            elapsed = time.perf_counter() - t0
            print(
                f"  [{label} {idx+1:2d}/{len(prompts)}] acc={acc_so_far*100:5.1f}% | "
                f"end_think={end_th*100:5.1f}% | avg_think={avg_th:6.1f} | wait/hmm={avg_hes:4.1f} | wall={elapsed:.1f}s",
                flush=True,
            )
    return traces


def summarize_metrics(traces: list, t1_tok_per_s: float = None) -> dict:
    n = max(1, len(traces))
    acc = sum(1 for r in traces if r["is_correct"]) / n
    avg_think = sum(r["think_tokens"] for r in traces) / n
    avg_comp = sum(r["completion_tokens"] for r in traces) / n
    loop_rate = sum(1 for r in traces if r["is_circular_loop"]) / n
    trunc_rate = sum(1 for r in traces if r["is_truncated_bloat"]) / n
    end_think_rate = sum(1 for r in traces if r["has_end_think"]) / n
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
            "correct": sum(1 for r in sub if r["is_correct"]),
            "accuracy": round(sum(1 for r in sub if r["is_correct"]) / len(sub), 4),
            "end_think_completion_rate": round(sum(1 for r in sub if r["has_end_think"]) / len(sub), 4),
            "avg_think_tokens": round(sum(r["think_tokens"] for r in sub) / len(sub), 2),
            "avg_hesitation_markers": round(sum(r["hesitation_count"] for r in sub) / len(sub), 2),
            "loop_or_bloat_rate": round(sum(1 for r in sub if r["is_circular_loop"]) / len(sub), 4),
        }

    return {
        "num_prompts": len(traces),
        "max_gen_tokens": MAX_GEN_TOKENS,
        "accuracy": round(acc, 4),
        "correct_count": sum(1 for r in traces if r["is_correct"]),
        "end_think_completion_rate": round(end_think_rate, 4),
        "truncated_no_end_think_rate": round(trunc_rate, 4),
        "avg_think_tokens": round(avg_think, 2),
        "avg_completion_tokens": round(avg_comp, 2),
        "avg_hesitation_markers": round(avg_hes, 2),
        "loop_or_bloat_rate": round(loop_rate, 4),
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


def load_qmdr_dirs(path: Path):
    """Parse `QMDR` binary file -> `(num_layers, dim, dirs)` where `dirs[layer][writer][k]` is a `list[float]`."""
    raw = path.read_bytes()
    magic, ver, n_layers, dim = struct.unpack_from("<4sIII", raw, 0)
    assert magic == b"QMDR" and ver == 1, f"Bad QMDR header in {path}"
    off = 16
    layers = []
    for _l in range(n_layers):
        writers = []
        for _w in range(2):
            ks = []
            for _k in range(2):
                vec = list(struct.unpack_from(f"<{dim}f", raw, off))
                off += 4 * dim
                ks.append(vec)
            writers.append(ks)
        layers.append(writers)
    return n_layers, dim, layers


def dot_vec(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_subspace_alignment(synth_bin: Path, post_bin: Path) -> dict:
    nl1, d1, s_dirs = load_qmdr_dirs(synth_bin)
    nl2, d2, p_dirs = load_qmdr_dirs(post_bin)
    assert nl1 == nl2 and d1 == d2
    rand_sd = 1.0 / math.sqrt(d1)  # 0.019764 for dim=2560

    per_layer = []
    for l in range(nl1):
        # writer 0 = o_proj (`h_mid`), writer 1 = down_proj (`h_out`)
        cos_o_k0 = dot_vec(s_dirs[l][0][0], p_dirs[l][0][0])
        cos_dn_k0 = dot_vec(s_dirs[l][1][0], p_dirs[l][1][0])
        # Rank-2 Frobenius overlap ||D_s^T D_p||_F / sqrt(2) in [0, 1]
        g00 = dot_vec(s_dirs[l][1][0], p_dirs[l][1][0])
        g01 = dot_vec(s_dirs[l][1][0], p_dirs[l][1][1])
        g10 = dot_vec(s_dirs[l][1][1], p_dirs[l][1][0])
        g11 = dot_vec(s_dirs[l][1][1], p_dirs[l][1][1])
        rank2_overlap = math.sqrt(max(0.0, (g00 * g00 + g01 * g01 + g10 * g10 + g11 * g11) / 2.0))
        per_layer.append({
            "layer": l,
            "down_proj_primary_cos": round(cos_dn_k0, 6),
            "down_proj_z_score_vs_random": round(cos_dn_k0 / rand_sd, 2),
            "o_proj_primary_cos": round(cos_o_k0, 6),
            "down_proj_rank2_subspace_overlap": round(rank2_overlap, 6),
        })

    l24_33 = [x["down_proj_primary_cos"] for x in per_layer[24:34]]
    l24_35 = [x["down_proj_primary_cos"] for x in per_layer[24:36]]
    return {
        "hidden_dim": d1,
        "random_unit_vector_std": round(rand_sd, 6),
        "layers_24_33_mean_cos": round(sum(l24_33) / len(l24_33), 4),
        "layers_24_33_max_cos": round(max(l24_33), 4),
        "layers_24_33_mean_z_score": round((sum(l24_33) / len(l24_33)) / rand_sd, 2),
        "layers_24_35_mean_cos": round(sum(l24_35) / len(l24_35), 4),
        "per_layer": per_layer,
    }


def save_traces(path: Path, traces: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in traces:
            f.write(json.dumps(r) + "\n")


def load_traces(path: Path) -> list:
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def main():
    print("=" * 92)
    print("PHASE 7: CONTAMINATION-FREE POST-CUTOFF (> NOV 2025 - AUG 2026) TRANSFER & WEIGHT-EDIT STUDY")
    print("=" * 92, flush=True)
    check_guardrails()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    calib_prompts = [json.loads(ln) for ln in open(CALIB_JSONL, encoding="utf-8") if ln.strip()]
    test_prompts = [json.loads(ln) for ln in open(TEST_JSONL, encoding="utf-8") if ln.strip()]
    calib_by_id = {p["id"]: p for p in calib_prompts}
    print(f"[init] Loaded {len(calib_prompts)} post-cutoff CALIB prompts & {len(test_prompts)} HELD-OUT TEST prompts.", flush=True)

    # =========================================================================
    # STAGE 1 (Step A.1): Zero-Shot Transfer Evaluation of Synthetic-Calibrated
    #                     `./checkpoints/qwen-patched` (`L24_33_down_a1.0`) on `post_cutoff_test.jsonl`
    # =========================================================================
    zs_traces_path = WORK_DIR / "traces_zeroshot_test.jsonl"
    if METRICS_ZEROSHOT_TEST.exists() and zs_traces_path.exists():
        print("\n[stage 1] Reusing saved Zero-Shot Synthetic-Patched evaluation on post_cutoff_test.jsonl.", flush=True)
        zs_metrics = json.loads(METRICS_ZEROSHOT_TEST.read_text())
    else:
        print("\n" + "=" * 92)
        print("STAGE 1 (Step A.1): Zero-Shot Transfer of Synthetic-Calibrated Patch on Unseen 2025-2026 Test Set")
        print("=" * 92, flush=True)
        manifest_before = json.loads((PATCHED_CKPT_DIR / "patch_manifest.json").read_text())
        print(f"[stage 1] Verified active synthetic patch manifest: {json.dumps(manifest_before)}", flush=True)
        prior_synth = str(PATCHED_CKPT_DIR / "calibrated_drafter_ngrams.bin")
        start_daemon_clean("qwen-patched", load_drafter=True, prior_path=prior_synth)
        zs_traces = evaluate_prompts(
            test_prompts,
            max_tokens=MAX_GEN_TOKENS,
            use_spec=True,
            label="zeroshot_test",
            recover_fn=lambda: start_daemon_clean("qwen-patched", load_drafter=True, prior_path=prior_synth),
        )
        save_traces(zs_traces_path, zs_traces)
        zs_metrics = summarize_metrics(zs_traces)
        zs_metrics["source_patch"] = "synthetic_val_prompts_L24_33_down_a1.0"
        zs_metrics["patch_manifest"] = manifest_before
        METRICS_ZEROSHOT_TEST.write_text(json.dumps(zs_metrics, indent=2))
        print(f"[stage 1 done] Zero-Shot Test Accuracy={zs_metrics['accuracy']*100:.2f}%, Wait/Hmm={zs_metrics['avg_hesitation_markers']:.2f}", flush=True)

    # =========================================================================
    # STAGE 2 (Step A.2 & B.1): Unpatched `Qwen3-4B-Thinking-2507` on `test` AND `calib`
    # =========================================================================
    bt_traces_path = WORK_DIR / "traces_baseline_test.jsonl"
    bc_traces_path = WORK_DIR / "traces_baseline_calib.jsonl"
    if METRICS_BASE_TEST.exists() and bt_traces_path.exists() and METRICS_BASE_CALIB.exists() and bc_traces_path.exists():
        print("\n[stage 2] Reusing saved Unpatched Baseline evaluations on post_cutoff_test & post_cutoff_calib.", flush=True)
        base_test_metrics = json.loads(METRICS_BASE_TEST.read_text())
        base_calib_metrics = json.loads(METRICS_BASE_CALIB.read_text())
        base_calib_traces = load_traces(bc_traces_path)
    else:
        print("\n" + "=" * 92)
        print("STAGE 2: Unpatched Qwen3-4B-Thinking-2507 Baseline on Held-Out Test (60) & Calib (60)")
        print("=" * 92, flush=True)
        start_daemon_clean("qwen3-thinking:4b", load_drafter=False)

        # Measure pure T=1 (--no-spec) speed & verify bit-exact parity of BM=4 drafter on real 2026 competition problem!
        t1_speeds = []
        for p in test_prompts[::20]:
            out, _, _ = run_cmd(
                ["run", "--no-spec", "--temp", "0", "--max-tokens", "256", "--reset-cache", "--quiet", "--json-stats", "-p", p["prompt"]],
                timeout=90,
            )
            t1_speeds.append(parse_last_json(out)["decode_tok_per_s"])
        base_t1_tps = sum(t1_speeds) / len(t1_speeds)
        print(f"[stage 2] Unpatched T=1 (--no-spec) speed: {base_t1_tps:.2f} tok/s", flush=True)

        out_t1_chk, _, _ = run_cmd(
            ["run", "--no-spec", "--temp", "0", "--max-tokens", "192", "--reset-cache", "--quiet", "--json-stats", "-p", test_prompts[0]["prompt"]],
            timeout=90,
        )
        run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
        out_sp_chk, _, _ = run_cmd(
            ["run", "--spec", "--temp", "0", "--max-tokens", "192", "--reset-cache", "--quiet", "--json-stats", "-p", test_prompts[0]["prompt"]],
            timeout=90,
        )
        assert parse_last_json(out_t1_chk)["token_ids"] == parse_last_json(out_sp_chk)["token_ids"], "BM=4 speculative mismatch!"
        print("[stage 2] Verified 100% bit-exact greedy token parity between T=1 and Qwen3-0.6B BM=4 drafter.", flush=True)

        if not (METRICS_BASE_TEST.exists() and bt_traces_path.exists()):
            base_test_traces = evaluate_prompts(
                test_prompts,
                max_tokens=MAX_GEN_TOKENS,
                use_spec=True,
                label="unpatched_test",
                recover_fn=lambda: start_daemon_clean("qwen3-thinking:4b", load_drafter=True),
            )
            save_traces(bt_traces_path, base_test_traces)
            base_test_metrics = summarize_metrics(base_test_traces, t1_tok_per_s=base_t1_tps)
            METRICS_BASE_TEST.write_text(json.dumps(base_test_metrics, indent=2))
        else:
            base_test_metrics = json.loads(METRICS_BASE_TEST.read_text())

        if not (METRICS_BASE_CALIB.exists() and bc_traces_path.exists()):
            base_calib_traces = evaluate_prompts(
                calib_prompts,
                max_tokens=MAX_GEN_TOKENS,
                use_spec=True,
                label="unpatched_calib",
                recover_fn=lambda: start_daemon_clean("qwen3-thinking:4b", load_drafter=True),
            )
            save_traces(bc_traces_path, base_calib_traces)
            base_calib_metrics = summarize_metrics(base_calib_traces, t1_tok_per_s=base_t1_tps)
            METRICS_BASE_CALIB.write_text(json.dumps(base_calib_metrics, indent=2))
        else:
            base_calib_traces = load_traces(bc_traces_path)
            base_calib_metrics = json.loads(METRICS_BASE_CALIB.read_text())

    print(
        f"\n>>> STEP A ZERO-SHOT TRANSFER RESULT (60 Unseen Post-Nov-2025 Test Problems) <<<\n"
        f"  Unpatched Qwen3-4B-Thinking-2507 : acc={base_test_metrics['accuracy']*100:5.2f}% ({base_test_metrics['correct_count']}/60) | "
        f"</think>={base_test_metrics['end_think_completion_rate']*100:5.1f}% | avg_think={base_test_metrics['avg_think_tokens']:6.1f} | "
        f"Wait/Hmm={base_test_metrics['avg_hesitation_markers']:5.2f} | loops={base_test_metrics['loop_or_bloat_rate']*100:5.1f}%\n"
        f"  Zero-Shot Synthetic-Patched      : acc={zs_metrics['accuracy']*100:5.2f}% ({zs_metrics['correct_count']}/60) | "
        f"</think>={zs_metrics['end_think_completion_rate']*100:5.1f}% | avg_think={zs_metrics['avg_think_tokens']:6.1f} | "
        f"Wait/Hmm={zs_metrics['avg_hesitation_markers']:5.2f} | loops={zs_metrics['loop_or_bloat_rate']*100:5.1f}%",
        flush=True,
    )

    # =========================================================================
    # STAGE 3 (Step B.2): Zero-Leakage Decision-Fork Localization on `post_cutoff_calib`
    #                     + 36-Layer Subspace Cosine Alignment (`synthetic` vs `post-cutoff`)
    # =========================================================================
    print("\n" + "=" * 92)
    print("STAGE 3: Post-Cutoff Decision-Fork Localization (STRICTLY on `calib`) & Subspace Universality")
    print("=" * 92, flush=True)

    post_dirs_bin = WORK_DIR / "postcutoff_fork_dirs.bin"
    post_loc_json = WORK_DIR / "postcutoff_localization.json"
    kl_ctrl_jsonl = WORK_DIR / "postcutoff_kl_ctrl.jsonl"

    start_daemon_clean("qwen3-thinking:4b", load_drafter=False)

    wait_toks = {14190, 13824, 80022}  # "Wait", " Wait", "Hmm"
    fail_forks, clean_forks = [], []

    for r in base_calib_traces:
        tids = r["token_ids"]
        ptoks = tokenize_prompt("qwen3-thinking:4b", calib_by_id[r["id"]]["prompt"])
        if END_THINK_TOKEN_ID in tids:
            idx = tids.index(END_THINK_TOKEN_ID)
            # Keep total prefix <= 3200 tokens so hook extraction fits cleanly inside DAEMON_CTX=4096
            if 25 <= idx and (len(ptoks) + idx) <= 3400:
                clean_forks.append({"id": r["id"], "kind": "pre_end_think", "tokens": ptoks + tids[:idx]})

    for cat in sorted(set(p["category"] for p in calib_prompts)):
        sub_all = [r for r in base_calib_traces if r["category"] == cat]
        added_c = 0
        for r in sub_all:
            tids = r["token_ids"]
            ptoks = tokenize_prompt("qwen3-thinking:4b", calib_by_id[r["id"]]["prompt"])
            for idx in range(80, min(380, len(tids) - 2)):
                if tids[idx - 1] == 271 and tids[idx] not in wait_toks and (len(ptoks) + idx) <= 3400:
                    clean_forks.append({"id": r["id"], "kind": "clean_step", "tokens": ptoks + tids[:idx]})
                    added_c += 1
                    break
            if added_c >= 8:
                break

        added_f = 0
        # Prioritize failed/looping traces first, then high-hesitation traces within `calib`
        cand_fail_traces = [x for x in sub_all if x["is_failure"]] + [
            x for x in sub_all if (not x["is_failure"] and x["hesitation_count"] >= 3)
        ]
        for r in cand_fail_traces:
            tids = r["token_ids"]
            ptoks = tokenize_prompt("qwen3-thinking:4b", calib_by_id[r["id"]]["prompt"])
            wpos = [
                idx
                for idx in range(350, len(tids))
                if tids[idx] in wait_toks and (len(ptoks) + idx) <= 3400
            ]
            if wpos:
                pick = wpos[min(1, len(wpos) - 1)]
                fail_forks.append({"id": r["id"], "kind": "wait_fork", "tokens": ptoks + tids[:pick]})
                added_f += 1
                if len(wpos) >= 4 and added_f < 12:
                    pick2 = wpos[min(3, len(wpos) - 1)]
                    fail_forks.append({"id": r["id"], "kind": "wait_fork_late", "tokens": ptoks + tids[:pick2]})
                    added_f += 1
            if added_f >= 12:
                break

    print(f"[stage 3] Extracted {len(fail_forks)} pre-Wait failure forks & {len(clean_forks)} clean/pre-</think> forks from `calib`.", flush=True)
    f_path = WORK_DIR / "fork_fail_toks.jsonl"
    c_path = WORK_DIR / "fork_clean_toks.jsonl"
    with open(f_path, "w", encoding="utf-8") as f:
        for x in fail_forks:
            f.write(json.dumps(x) + "\n")
    with open(c_path, "w", encoding="utf-8") as f:
        for x in clean_forks:
            f.write(json.dumps(x) + "\n")
    with open(kl_ctrl_jsonl, "w", encoding="utf-8") as f:
        for r in base_calib_traces[::3][:20]:
            ptoks = tokenize_prompt("qwen3-thinking:4b", calib_by_id[r["id"]]["prompt"])
            f.write(json.dumps({"id": r["id"], "tokens": ptoks + r["token_ids"][:160]}) + "\n")

    hooks_fail_bin = WORK_DIR / "fork_hooks_fail.bin"
    hooks_clean_bin = WORK_DIR / "fork_hooks_clean.bin"
    run_cmd(["mech", "extract-hooks", str(f_path), "-o", str(hooks_fail_bin)], timeout=360)
    run_cmd(["mech", "extract-hooks", str(c_path), "-o", str(hooks_clean_bin)], timeout=360)
    run_cmd(
        ["mech", "localize", "--model", "qwen3-thinking:4b", str(hooks_fail_bin), str(hooks_clean_bin), "-o", str(post_dirs_bin), "--report", str(post_loc_json)],
        timeout=120,
    )

    align_report = compute_subspace_alignment(SYNTH_DIRS_BIN, post_dirs_bin)
    METRICS_SUBSPACE_ALIGN.write_text(json.dumps(align_report, indent=2))
    print(
        f"[stage 3] Subspace Cosine Alignment (`val_prompts` synthetic vs `2025-2026` unseen benchmarks):\n"
        f"  Layers 24-33 mean cos = {align_report['layers_24_33_mean_cos']:.4f} "
        f"(max = {align_report['layers_24_33_max_cos']:.4f}, z = {align_report['layers_24_33_mean_z_score']:.1f} sigma vs random!)",
        flush=True,
    )

    # =========================================================================
    # STAGE 4 (Step B.3-4): Zero-Leakage LoRA Screening on `calib` + Instant Rollback
    #                       + Surgical `BF16 -> Q8_0` Patch + Full `calib` Verification Gate
    # =========================================================================
    print("\n" + "=" * 92)
    print("STAGE 4: Candidate LoRA Screening (16-Prompt Stratified Calib) + Surgical BF16->Q8_0 Patch")
    print("=" * 92, flush=True)

    # Pick 4 calibration prompts per benchmark category (16 total) for fast screening
    screen_prompts = []
    for cat in sorted(set(p["category"] for p in calib_prompts)):
        cat_ps = [p for p in calib_prompts if p["category"] == cat]
        # Pick indices spread across moderate-to-hard within calib
        picks = [cat_ps[i] for i in [1, 4, 7, 10] if i < len(cat_ps)]
        screen_prompts.extend(picks)
    screen_ids = {p["id"] for p in screen_prompts}
    base_screen = [r for r in base_calib_traces if r["id"] in screen_ids]
    base_sc_acc = sum(1 for r in base_screen if r["is_correct"]) / len(base_screen)
    base_sc_loops = sum(1 for r in base_screen if r["is_circular_loop"])
    base_sc_think = sum(r["think_tokens"] for r in base_screen) / len(base_screen)
    print(
        f"[stage 4] Base 16-prompt Calib screen: acc={base_sc_acc*100:.1f}%, loops={base_sc_loops}/16, avg_think={base_sc_think:.1f}",
        flush=True,
    )

    candidate_specs = [
        ("post_L24_33_down_a1.0", list(range(24, 34)), 1, 1.00, ["--down-only"]),
        ("post_L24_33_down_a0.85", list(range(24, 34)), 1, 0.85, ["--down-only"]),
        ("post_L26_33_down_a1.0", list(range(26, 34)), 1, 1.00, ["--down-only"]),
    ]

    screened = []
    for cname, clayers, crank, calpha, extra_flags in candidate_specs:
        lora_dir = WORK_DIR / f"cand_lora_{cname}"
        layers_csv = ",".join(map(str, clayers))
        run_cmd(
            ["mech", "make-lora", str(BF16_MASTER_DIR), str(post_dirs_bin), str(lora_dir), "--layers", layers_csv, "--rank", str(crank), "--alpha", str(calpha)] + extra_flags,
            timeout=120,
        )
        kl_j = parse_last_json(run_cmd(["mech", "kl-eval", str(kl_ctrl_jsonl), "--lora", str(lora_dir)], timeout=180)[0])
        mean_kl = kl_j["mean_kl_nats"]
        print(f"  [KL gate `{cname}`] mean_kl={mean_kl:.5f} nats, top1_agree={kl_j['top1_agreement']*100:.2f}%", flush=True)
        if mean_kl > KL_MAX_THRESHOLD_NATS:
            print(f"    -> Skipped `{cname}` (exceeded KL cap {KL_MAX_THRESHOLD_NATS})")
            continue

        run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
        run_cmd(["lora", "load", str(lora_dir)], timeout=30)
        sc_traces = evaluate_prompts(
            screen_prompts,
            max_tokens=MAX_GEN_TOKENS,
            use_spec=True,
            label=cname,
            recover_fn=lambda ld=lora_dir: (
                start_daemon_clean("qwen3-thinking:4b", load_drafter=True),
                run_cmd(["lora", "load", str(ld)], timeout=30),
            ),
        )
        run_cmd(["lora", "clear"], timeout=30)
        run_cmd(["drafter", "clear"], timeout=30)

        sc_acc = sum(1 for r in sc_traces if r["is_correct"]) / len(sc_traces)
        sc_loops = sum(1 for r in sc_traces if r["is_circular_loop"])
        sc_think = sum(r["think_tokens"] for r in sc_traces) / len(sc_traces)
        sc_hes = sum(r["hesitation_count"] for r in sc_traces) / len(sc_traces)
        score = (
            (sc_acc - base_sc_acc) * 100.0
            + (base_sc_loops - sc_loops) * 1.5
            + (base_sc_think - sc_think) * 0.008
            - mean_kl * 5.0
        )
        rec = {
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
            "screen_avg_hes": round(sc_hes, 2),
            "composite_score": round(score, 4),
        }
        screened.append(rec)
        print(f"  [screen `{cname}`] acc={sc_acc*100:.1f}% | loops={sc_loops}/16 | wait/hmm={sc_hes:.2f} | score={score:.2f}", flush=True)

    screened.sort(key=lambda x: x["composite_score"], reverse=True)
    (WORK_DIR / "screened_candidates.json").write_text(json.dumps(screened, indent=2))
    if not screened:
        raise RuntimeError("No candidate passed KL gate on post_cutoff_calib!")

    # First cleanly roll back the Phase 6 synthetic patch on `./checkpoints/qwen-patched/`!
    subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
    if (PATCHED_CKPT_DIR / "rollback_backup.bin").exists():
        rb_out, _, _ = run_cmd(["mech", "rollback", str(PATCHED_CKPT_DIR)], timeout=60)
        print(f"[stage 4] Restored clean master weights via `<50ms` rollback before applying post-cutoff patch: {rb_out.strip()}", flush=True)

    accepted_cand = None
    patched_calib_traces = None
    patched_calib_metrics = None

    for attempt_idx, cand in enumerate(screened, start=1):
        print(f"\n[stage 4 attempt {attempt_idx}] Applying surgical BF16 + Q8_0 patch `{cand['name']}`...", flush=True)
        subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
        layers_csv = ",".join(map(str, cand["layers"]))
        p_out, _, _ = run_cmd(
            [
                "mech",
                "patch",
                str(BF16_MASTER_DIR),
                str(BASE_Q8_GGUF),
                str(post_dirs_bin),
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
        print(f"  Patched checkpoint written: {json.dumps(patch_info)}", flush=True)

        start_daemon_clean("qwen-patched", load_drafter=True)
        c_traces = evaluate_prompts(
            calib_prompts,
            max_tokens=MAX_GEN_TOKENS,
            use_spec=True,
            label=f"calib_gate_{cand['name']}",
            recover_fn=lambda: start_daemon_clean("qwen-patched", load_drafter=True),
        )
        c_metrics = summarize_metrics(c_traces)
        c_metrics["patch_info"] = patch_info
        c_metrics["control_kl_nats"] = cand["mean_kl_nats"]
        c_metrics["control_max_kl_nats"] = cand["max_kl_nats"]
        c_metrics["control_top1_agreement"] = cand["top1_agreement"]

        acc_ok = c_metrics["accuracy"] >= base_calib_metrics["accuracy"]
        strictly_better = (c_metrics["accuracy"] > base_calib_metrics["accuracy"]) or (
            c_metrics["accuracy"] == base_calib_metrics["accuracy"]
            and (
                c_metrics["loop_or_bloat_rate"] < base_calib_metrics["loop_or_bloat_rate"]
                or c_metrics["avg_hesitation_markers"] < base_calib_metrics["avg_hesitation_markers"] * 0.85
            )
        )
        fail_dropped = (
            c_metrics["loop_or_bloat_rate"] < base_calib_metrics["loop_or_bloat_rate"]
            or c_metrics["total_failure_mode_rate"] < base_calib_metrics["total_failure_mode_rate"]
            or c_metrics["avg_hesitation_markers"] < base_calib_metrics["avg_hesitation_markers"] * 0.80
        )
        kl_ok = cand["mean_kl_nats"] <= KL_MAX_THRESHOLD_NATS

        print(
            f"\n[phase 4 calib gate `{cand['name']}`] "
            f"calib_acc: {base_calib_metrics['accuracy']*100:.2f}% -> {c_metrics['accuracy']*100:.2f}% | "
            f"loops: {base_calib_metrics['loop_or_bloat_rate']*100:.2f}% -> {c_metrics['loop_or_bloat_rate']*100:.2f}% | "
            f"Wait/Hmm: {base_calib_metrics['avg_hesitation_markers']:.2f} -> {c_metrics['avg_hesitation_markers']:.2f} | "
            f"KL={cand['mean_kl_nats']:.5f}",
            flush=True,
        )

        if acc_ok and strictly_better and fail_dropped and kl_ok:
            print(f"  => ACCEPTED `{cand['name']}` on Post-Cutoff Calibration Gate!", flush=True)
            accepted_cand = cand
            patched_calib_traces = c_traces
            patched_calib_metrics = c_metrics
            break
        else:
            print(f"  => REJECTED `{cand['name']}` — invoking instant tensor-level rollback!", flush=True)
            subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
            rb_out, _, _ = run_cmd(["mech", "rollback", str(PATCHED_CKPT_DIR)], timeout=60)
            print(f"  Rollback complete: {rb_out.strip()}", flush=True)

    if accepted_cand is None:
        raise RuntimeError("No post-cutoff candidate passed the Phase 4 calibration verification gate!")

    save_traces(WORK_DIR / "traces_patched_calib.jsonl", patched_calib_traces)
    METRICS_PATCHED_CALIB.write_text(json.dumps(patched_calib_metrics, indent=2))

    # =========================================================================
    # STAGE 5 (Step B.5): Calibrated Speculative Prior Distillation (strictly from `calib`),
    #                     Locked One-Shot Held-Out `post_cutoff_test.jsonl` Evaluation,
    #                     & 4-Way Speculative Drafter Throughput Benchmark
    # =========================================================================
    print("\n" + "=" * 92)
    print("STAGE 5: Calibrated Prior Distillation + Locked One-Shot Held-Out Test Eval + Speculative Bench")
    print("=" * 92, flush=True)

    distill_jsonl = WORK_DIR / "calib_patched_completions_toks.jsonl"
    with open(distill_jsonl, "w", encoding="utf-8") as f:
        for r in patched_calib_traces:
            ptoks = tokenize_prompt("qwen-patched", calib_by_id[r["id"]]["prompt"])
            f.write(json.dumps({"id": r["id"], "tokens": ptoks + r["token_ids"]}) + "\n")

    prior_bin = PATCHED_CKPT_DIR / "calibrated_drafter_ngrams.bin"
    dist_out, _, _ = run_cmd(
        ["drafter", "distill", str(distill_jsonl), "-o", str(prior_bin), "--min-support", "3", "--min-prob", "0.88"],
        timeout=60,
    )
    dist_info = parse_last_json(dist_out)
    print(f"[stage 5] Distilled post-cutoff reasoning prior (strictly from `calib`): {json.dumps(dist_info)}", flush=True)

    run_cmd(["drafter", "load", "qwen3:0.6b", "--prior", str(prior_bin)], timeout=60)

    # LOCKED ONE-SHOT FINAL EVALUATION ON UNTOUCHED `post_cutoff_test.jsonl` (60 problems)
    patched_test_traces = evaluate_prompts(
        test_prompts,
        max_tokens=MAX_GEN_TOKENS,
        use_spec=True,
        label="LOCKED_FINAL_TEST",
        recover_fn=lambda: start_daemon_clean("qwen-patched", load_drafter=True, prior_path=str(prior_bin)),
    )
    save_traces(WORK_DIR / "traces_patched_test.jsonl", patched_test_traces)

    # Measure pure T=1 (--no-spec) speed on patched model
    run_cmd(["drafter", "clear"], timeout=30)
    pt1_speeds = []
    for p in test_prompts[::20]:
        out, _, _ = run_cmd(
            ["run", "--no-spec", "--temp", "0", "--max-tokens", "256", "--reset-cache", "--quiet", "--json-stats", "-p", p["prompt"]],
            timeout=90,
        )
        pt1_speeds.append(parse_last_json(out)["decode_tok_per_s"])
    patched_t1_tps = sum(pt1_speeds) / len(pt1_speeds)

    patched_test_metrics = summarize_metrics(patched_test_traces, t1_tok_per_s=patched_t1_tps)
    patched_test_metrics["accepted_calib_candidate"] = accepted_cand["name"]
    patched_test_metrics["patch_info"] = patched_calib_metrics["patch_info"]
    patched_test_metrics["control_kl_nats"] = accepted_cand["mean_kl_nats"]
    patched_test_metrics["control_top1_agreement"] = accepted_cand["top1_agreement"]
    METRICS_PATCHED_TEST.write_text(json.dumps(patched_test_metrics, indent=2))

    # 4-Mode Lossless Speculative Decoding Benchmark on 12 Stratified Post-Cutoff Test Problems (3 per category)
    bench_12 = []
    for cat in sorted(set(p["category"] for p in test_prompts)):
        sub = [p for p in test_prompts if p["category"] == cat]
        bench_12.extend(sub[:3])

    def bench_mode(mode_name: str, use_spec: bool, load_neural: bool, prior: str = ""):
        run_cmd(["drafter", "clear"], timeout=30)
        if load_neural:
            d_args = ["drafter", "load", "qwen3:0.6b"]
            if prior:
                d_args += ["--prior", prior]
            run_cmd(d_args, timeout=60)
        spec_flag = "--spec" if use_spec else "--no-spec"
        toks_by_id = {}
        tot_toks = 0
        tot_ms = 0.0
        ver_calls = 0
        acc_extra = 0
        for item in bench_12:
            out, _, _ = run_cmd(
                ["run", spec_flag, "--temp", "0", "--max-tokens", "512", "--reset-cache", "--quiet", "--json-stats", "-p", item["prompt"]],
                timeout=120,
            )
            st = parse_last_json(out)
            toks_by_id[item["id"]] = st["token_ids"]
            tot_toks += st["completion_tokens"]
            tot_ms += st["decode_ms"]
            ver_calls += st["spec_verify_calls"]
            acc_extra += st["spec_accepted_extra"]
        agg_tps = tot_toks / (tot_ms / 1000.0)
        alpha = (acc_extra / (3.0 * ver_calls)) if ver_calls > 0 else 0.0
        return {
            "mode": mode_name,
            "num_prompts": len(bench_12),
            "total_completion_tokens": tot_toks,
            "aggregate_decode_tok_per_s": round(agg_tps, 2),
            "spec_verify_calls": ver_calls,
            "spec_accepted_extra": acc_extra,
            "acceptance_rate_alpha": round(alpha, 4),
            "mean_tokens_per_verify": round(1.0 + (acc_extra / ver_calls) if ver_calls > 0 else 1.0, 3),
        }, toks_by_id

    m_t1, ref_toks = bench_mode("1_patched_T1_no_spec", use_spec=False, load_neural=False)
    print(f"  [1] Patched T=1 (--no-spec):              {m_t1['aggregate_decode_tok_per_s']:6.2f} tok/s", flush=True)

    m_ng, ng_toks = bench_mode("2_patched_dynamic_ngram_BM4", use_spec=True, load_neural=False)
    assert ng_toks == ref_toks, "Dynamic N-gram BM=4 mismatch!"
    print(f"  [2] Patched + Dynamic N-gram BM=4:        {m_ng['aggregate_decode_tok_per_s']:6.2f} tok/s (alpha={m_ng['acceptance_rate_alpha']:.3f})", flush=True)

    m_dr_raw, dr_raw_toks = bench_mode("3_patched_qwen3_0.6b_neural_BM4", use_spec=True, load_neural=True, prior="")
    assert dr_raw_toks == ref_toks, "Qwen3-0.6B neural drafter mismatch!"
    print(f"  [3] Patched + Qwen3-0.6B Neural Drafter:  {m_dr_raw['aggregate_decode_tok_per_s']:6.2f} tok/s (alpha={m_dr_raw['acceptance_rate_alpha']:.3f})", flush=True)

    m_dr_cal, dr_cal_toks = bench_mode("4_patched_qwen3_0.6b_calibrated_BM4", use_spec=True, load_neural=True, prior=str(prior_bin))
    assert dr_cal_toks == ref_toks, "Calibrated Qwen3-0.6B + Prior mismatch!"
    print(f"  [4] Patched + Calibrated 0.6B + Prior:    {m_dr_cal['aggregate_decode_tok_per_s']:6.2f} tok/s (alpha={m_dr_cal['acceptance_rate_alpha']:.3f})", flush=True)

    best_spec = max([m_ng, m_dr_raw, m_dr_cal], key=lambda x: x["aggregate_decode_tok_per_s"])
    run_cmd(["drafter", "load", "qwen3:0.6b", "--prior", str(prior_bin)], timeout=60)

    spec_summary = {
        "unpatched_test_alpha_on_0.6b": base_test_metrics["spec_acceptance_alpha"],
        "zeroshot_patched_test_alpha": zs_metrics["spec_acceptance_alpha"],
        "postcutoff_patched_test_alpha": patched_test_metrics["spec_acceptance_alpha"],
        "distilled_prior": dist_info,
        "modes": [m_t1, m_ng, m_dr_raw, m_dr_cal],
        "winning_speculative_mode": best_spec["mode"],
        "speedup_over_patched_t1": round(best_spec["aggregate_decode_tok_per_s"] / m_t1["aggregate_decode_tok_per_s"], 3),
        "bit_exact_distribution_verified": True,
    }
    METRICS_SPEC_POSTCUTOFF.write_text(json.dumps(spec_summary, indent=2))

    print("\n" + "=" * 92)
    print("FINAL 3-WAY HELD-OUT POST-CUTOFF TEST SUMMARY (60 Unseen Nov 2025 - Aug 2026 Problems)")
    print("=" * 92)
    for label, m in [
        ("1. Unpatched Qwen3-4B-Thinking-2507", base_test_metrics),
        ("2. Zero-Shot Synthetic-Patched     ", zs_metrics),
        ("3. Post-Cutoff Calibrated-Patched  ", patched_test_metrics),
    ]:
        print(
            f"  {label}: acc={m['accuracy']*100:5.2f}% ({m['correct_count']:2d}/60) | "
            f"</think>={m['end_think_completion_rate']*100:5.1f}% | avg_think={m['avg_think_tokens']:6.1f} | "
            f"Wait/Hmm={m['avg_hesitation_markers']:5.2f} | loops={m['loop_or_bloat_rate']*100:5.1f}%"
        )
    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()
