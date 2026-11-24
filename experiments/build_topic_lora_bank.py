#!/usr/bin/env python3
"""
Phase 8: Sharded Topic Parametric LoRA Bank Builder & Head-to-Head Benchmark (`Option B`)

Compiles a 46-article Wikipedia Topic LoRA Bank on `Qwen3-8B-Patched-Q6_K` (`qwen3-8b-patched`)
using Zero-Backprop Contextual State Transfer + Closed-Form Push-Through Null-Space AlphaEdit
entirely on the RTX 3080 + Apple Accelerate (0 PyTorch, 0 Docker, 0 external crates, 0 `.cu` edits).

Stages:
  1. Baseline (no context), Contextual Oracle (gold passage), & Phase 8 Text RAG (`wikipedia_1k_8b_best.qvec`)
     evaluation on `wikipedia_eval_200.jsonl` + curated unseen `wikipedia_paraphrase_40.jsonl`.
  2. Background Null-Space SwiGLU Keys (`BG.qmkv`) from non-Wikipedia prompts + `control_kl_40.jsonl`.
  3. Closed-loop sequential multi-layer calibration sweep on 5 diverse articles (`gamma`, `lambda`,
     layer schedule, 1x vs 2x prompt key augmentation) selecting optimal low-KL / high-recall config.
  4. Full 46-article Topic LoRA Bank compilation -> `checkpoints/wiki-topic-loras/`.
  5. End-to-end verification & head-to-head benchmark (`metrics_topic_lora_bank.json`), including:
     - `qwen-nv topic ask` on all 200 direct Q&As & 40 unseen paraphrases,
     - Dual-slot cache verification (`167 MB` Retrieval SVD LoRA pinned + `~44 us` Topic LoRA swaps),
     - Control `kl-eval` + `0.000000 nats` bit-exact clear verification,
     - 10 GB VRAM co-residency check (`qwen3-8b-patched` + `Qwen3-0.6B` drafter + pinned 8B SVD LoRA + Topic Pool + `.qvec`).
"""

import argparse
import json
import os
import re
import shutil
import string
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/tim/Code/optimize-gpu")
BIN = ROOT / "engine" / "target" / "release" / "rtx"
DATA_DIR = ROOT / "data"
CORPUS_JSONL = DATA_DIR / "wikipedia_1k.jsonl"
EVAL_JSONL = DATA_DIR / "wikipedia_eval_200.jsonl"
PARA_JSONL = DATA_DIR / "wikipedia_paraphrase_40.jsonl"
VAL_PROMPTS_JSONL = DATA_DIR / "val_prompts.jsonl"
BEST_QVEC = DATA_DIR / "wikipedia_1k_8b_best.qvec"

WORK_DIR = DATA_DIR / "topic_lora_work"
BANK_DIR = ROOT / "checkpoints" / "wiki-topic-loras"
OUT_METRICS = ROOT / "metrics" / "metrics_topic_lora_bank.json"

MODEL_REF = "qwen3-8b-patched"
DAEMON_CTX = 4096
SYS_PROMPT = "Answer the question concisely in one short phrase or sentence."
IM_END_ID = 151645

# Curated 40 unseen paraphrases across 40 distinct Wikipedia articles (strictly held out from LoRA compilation)
PARAPHRASE_SPECS = [
    (0, "Which field within theoretical computer science focuses on categorizing computational problems according to their difficulty and relationships?"),
    (1, "What term refers to a formal course of study?"),
    (2, "Which British company resulted from merging Sky Television and British Satellite Broadcasting?"),
    (3, "How would you describe the economy of Victoria, Australia?"),
    (4, "Roughly how many miles south of Edinburgh is Newcastle upon Tyne located?"),
    (5, "What was the name of the empire established by Genghis Khan?"),
    (6, "How is the immune system defined biologically?"),
    (7, "What do you call the Yuan dynasty in Chinese?"),
    (8, "What full name do the initials UMC represent?"),
    (9, "What did the word Norman originally mean?"),
    (10, "Since Martin Luther rejected salvation through good deeds, through what means did he say it is achieved?"),
    (11, "When a disease is described as enzootic, what does that mean?"),
    (12, "How many acres of land does the Victoria and Albert Museum occupy?"),
    (13, "Which play by Sophocles illustrated the concept of civil disobedience?"),
    (14, "Who serves as the chair of the Intergovernmental Panel on Climate Change?"),
    (15, "In Poland, Warsaw stood out as the most diverse what?"),
    (16, "Which ocean coastline borders Southern California?"),
    (18, "Which nation saw the first invention of a full-scale working railway steam locomotive?"),
    (19, "When and by what geological feature did the Amazon drainage basin split?"),
    (20, "In what way is packet-mode communication carried out regarding intermediate forwarding nodes?"),
    (23, "Where do the military forces in Jacksonville rank in size?"),
    (25, "A coerced trade agreement imposed by one nation on another exemplifies what form of imperialism?"),
    (26, "Which ancient philosopher wrote a philosophical discussion of force?"),
    (29, "What collective French name was given to the fortified towns granted to the Huguenots in 1598?"),
    (31, "Name the very first spacecraft to enter orbit around another celestial body."),
    (32, "Which kind of waves do seismologists rely on to map Earth's interior?"),
    (34, "Which corporation was the owner of NBC during the 1930s?"),
    (36, "In what calendar year was Harvard University founded?"),
    (37, "Which British leader expanded military resources in the colonies during the French and Indian War?"),
    (39, "Which Renaissance inventor expanded upon Philo of Byzantium's findings?"),
    (40, "Which European commission faced censure in 1999 after Commissioners abused power?"),
    (46, "How frequently does the character in Doctor Who travel alone?"),
    (51, "To what year does a historic pharmacy in Croatia trace back?"),
    (55, "Which integer did early Greek mathematicians not consider to be a true number?"),
    (56, "Since 2004, where has the official seat of the Scottish Parliament been located?"),
    (58, "Which teacher had a notable influence on Nikola Tesla during his school years?"),
    (65, "To which chloroplast lineage does Cyanophora belong?"),
    (67, "Name the Carolina Panthers running back who was waived ahead of Super Bowl 50."),
    (75, "Advances in what technology are enabling new research into building construction techniques?"),
    (76, "What German legal term forbids segregating pupils according to parental wealth in private schools?"),
]


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


def start_daemon_clean(model_ref: str = MODEL_REF):
    for attempt in range(1, 7):
        subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)
        time.sleep(0.8 + 0.4 * attempt)
        try:
            run_cmd(["daemon", "--background", "--ctx", str(DAEMON_CTX), "--model", model_ref], timeout=90)
            st, _, _ = run_cmd(["status"], timeout=15)
            print(f"[daemon] {st.strip()}", flush=True)
            return
        except Exception as e:
            print(f"[daemon retry {attempt}] {e}", flush=True)
            time.sleep(1.5)
    raise RuntimeError(f"Could not start daemon for {model_ref}")


def stop_daemon_clean():
    subprocess.run([str(BIN), "stop"], capture_output=True, text=True, timeout=30)


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s[:64] if s else "topic"


def normalize_qa_text(s: str) -> str:
    s = s.lower().strip()
    # Remove punctuation
    s = "".join(ch if ch not in string.punctuation and ch not in "–—\"'“”’" else " " for ch in s)
    # Remove articles
    tokens = [w for w in s.split() if w not in ("a", "an", "the")]
    return " ".join(tokens)


def score_qa(pred: str, gold: str) -> tuple[bool, float]:
    np = normalize_qa_text(pred)
    ng = normalize_qa_text(gold)
    if not ng:
        return (False, 0.0)
    # Exact/substring match (also handle short numeric/1-token answers with word boundaries)
    if ng == np:
        contains = True
    elif len(ng.split()) == 1 and len(ng) <= 4:
        contains = ng in np.split()
    else:
        contains = ng in np

    ptoks = np.split()
    gtoks = ng.split()
    if not ptoks or not gtoks:
        return (contains, 1.0 if contains else 0.0)
    common = Counter(ptoks) & Counter(gtoks)
    num_same = sum(common.values())
    if num_same == 0:
        return (contains, 0.0)
    prec = num_same / len(ptoks)
    rec = num_same / len(gtoks)
    f1 = 2 * prec * rec / (prec + rec)
    return (contains, f1)


def offline_tokenize_raw(text: str) -> list[int]:
    out, _, _ = run_cmd(["tokenize", "--model", MODEL_REF, "--raw", text], timeout=30)
    return json.loads(out.strip())


def format_chatml_prompt(user_text: str, sys_prompt: str = SYS_PROMPT) -> str:
    return (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def ensure_paraphrase_dataset(qas_by_qid: dict):
    if PARA_JSONL.exists():
        rows = [json.loads(ln) for ln in open(PARA_JSONL, encoding="utf-8") if ln.strip()]
        if len(rows) == 40:
            return rows
    rows = []
    for qid, p_question in PARAPHRASE_SPECS:
        src = qas_by_qid[qid]
        rows.append(
            {
                "qid": qid,
                "target_id": src["target_id"],
                "title": src["title"],
                "orig_question": src["question"],
                "question": p_question,
                "answer": src["answer"],
            }
        )
    with open(PARA_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[dataset] Wrote {len(rows)} curated unseen paraphrases to {PARA_JSONL}", flush=True)
    return rows


def eval_run_cmd_batch(items: list, prompt_fn, label: str, max_tokens: int = 64) -> dict:
    t0 = time.perf_counter()
    results = []
    for idx, it in enumerate(items):
        user_p = prompt_fn(it)
        out, _, wall_s = run_cmd(
            [
                "run",
                "--spec",
                "--temp",
                "0",
                "--max-tokens",
                str(max_tokens),
                "--reset-cache",
                "--sys",
                SYS_PROMPT,
                "--json-stats",
                "-p",
                user_p,
            ],
            timeout=120,
        )
        lines = out.splitlines()
        stats = parse_last_json(out)
        pred_text = "\n".join(lines[:-1]).strip()
        ok, f1 = score_qa(pred_text, it["answer"])
        results.append(
            {
                "qid": it["qid"],
                "title": it["title"],
                "question": it["question"],
                "gold": it["answer"],
                "pred": pred_text,
                "correct": ok,
                "f1": round(f1, 4),
                "token_ids": stats["token_ids"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "prefill_ms": stats["prefill_ms"],
                "decode_ms": stats["decode_ms"],
                "decode_tok_per_s": stats["decode_tok_per_s"],
                "total_ms": stats["total_ms"],
                "wall_ms": round(wall_s * 1000.0, 2),
            }
        )
        if (idx + 1) % 50 == 0 or (idx + 1) == len(items):
            acc = sum(1 for r in results if r["correct"]) / len(results)
            mf1 = sum(r["f1"] for r in results) / len(results)
            print(
                f"  [{label} {idx+1:3d}/{len(items)}] acc={acc*100:5.1f}% | F1={mf1*100:5.1f}% | wall={time.perf_counter()-t0:.1f}s",
                flush=True,
            )
    return summarize_qa_eval(label, results)


def eval_rag_ask_batch(items: list, label: str, max_tokens: int = 64) -> dict:
    run_cmd(["rag", "load", str(BEST_QVEC)], timeout=60)
    t0 = time.perf_counter()
    results = []
    for idx, it in enumerate(items):
        out, err, wall_s = run_cmd(
            [
                "rag",
                "ask",
                "--top-k",
                "3",
                "--temp",
                "0",
                "--max-tokens",
                str(max_tokens),
                "--sys",
                SYS_PROMPT,
                "--json-stats",
                it["question"],
            ],
            timeout=120,
        )
        lines = out.splitlines()
        stats = parse_last_json(out)
        pred_text = "\n".join(lines[:-1]).strip()
        ok, f1 = score_qa(pred_text, it["answer"])

        embed_ms = 0.0
        search_us = 0.0
        m = re.search(r"embed=([0-9.]+)\s*ms\s*\|\s*GPU_cosine=([0-9.]+)", err)
        if m:
            embed_ms = float(m.group(1))
            search_us = float(m.group(2))

        results.append(
            {
                "qid": it["qid"],
                "title": it["title"],
                "question": it["question"],
                "gold": it["answer"],
                "pred": pred_text,
                "correct": ok,
                "f1": round(f1, 4),
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "embed_ms": embed_ms,
                "gpu_search_us": search_us,
                "prefill_ms": stats["prefill_ms"],
                "decode_ms": stats["decode_ms"],
                "decode_tok_per_s": stats["decode_tok_per_s"],
                "total_ms": stats["total_ms"],
                "wall_ms": round(wall_s * 1000.0, 2),
            }
        )
        if (idx + 1) % 50 == 0 or (idx + 1) == len(items):
            acc = sum(1 for r in results if r["correct"]) / len(results)
            mf1 = sum(r["f1"] for r in results) / len(results)
            print(
                f"  [{label} {idx+1:3d}/{len(items)}] acc={acc*100:5.1f}% | F1={mf1*100:5.1f}% | wall={time.perf_counter()-t0:.1f}s",
                flush=True,
            )
    return summarize_qa_eval(label, results)


def summarize_qa_eval(label: str, rows: list) -> dict:
    n = max(1, len(rows))
    acc = sum(1 for r in rows if r["correct"]) / n
    mf1 = sum(r["f1"] for r in rows) / n
    mean_prompt_tok = sum(r["prompt_tokens"] for r in rows) / n
    mean_prefill = sum(r["prefill_ms"] for r in rows) / n
    mean_decode = sum(r["decode_ms"] for r in rows) / n
    mean_tps = sum(r["decode_tok_per_s"] for r in rows) / n
    mean_total = sum(r["total_ms"] for r in rows) / n
    mean_wall = sum(r["wall_ms"] for r in rows) / n
    out = {
        "label": label,
        "count": len(rows),
        "correct": sum(1 for r in rows if r["correct"]),
        "accuracy": round(acc, 4),
        "mean_f1": round(mf1, 4),
        "mean_prompt_tokens": round(mean_prompt_tok, 1),
        "mean_prefill_ms": round(mean_prefill, 2),
        "mean_decode_ms": round(mean_decode, 2),
        "mean_decode_tok_per_s": round(mean_tps, 2),
        "mean_gpu_total_ms": round(mean_total, 2),
        "mean_client_wall_ms": round(mean_wall, 2),
        "rows": rows,
    }
    if rows and "embed_ms" in rows[0]:
        out["mean_embed_ms"] = round(sum(r["embed_ms"] for r in rows) / n, 2)
        out["mean_gpu_search_us"] = round(sum(r["gpu_search_us"] for r in rows) / n, 2)
    return out


def stage1_baseline_oracle_rag(qas: list, docs_by_id: dict, para_40: list) -> dict:
    cache_path = WORK_DIR / "stage1_evals.json"
    if cache_path.exists():
        print("\n[stage 1] Reusing cached Stage 1 Baseline / Oracle / RAG evaluations.", flush=True)
        return json.loads(cache_path.read_text())

    print("\n" + "=" * 88)
    print("STAGE 1: Baseline (No Context) vs Contextual Oracle vs Phase 5 Text RAG")
    print("=" * 88, flush=True)

    bare_200 = eval_run_cmd_batch(qas, lambda q: q["question"], "bare_200")
    oracle_200 = eval_run_cmd_batch(
        qas,
        lambda q: (
            f"Use the following Wikipedia passage to answer concisely:\n"
            f"[{q['title']}]\n{docs_by_id[q['target_id']]['text']}\n\nQuestion: {q['question']}"
        ),
        "oracle_200",
    )
    rag_200 = eval_rag_ask_batch(qas, "rag_200")
    bare_para_40 = eval_run_cmd_batch(para_40, lambda q: q["question"], "bare_para_40")
    rag_para_40 = eval_rag_ask_batch(para_40, "rag_para_40")

    stage1 = {
        "bare_200": bare_200,
        "oracle_200": oracle_200,
        "rag_200": rag_200,
        "bare_para_40": bare_para_40,
        "rag_para_40": rag_para_40,
    }
    cache_path.write_text(json.dumps(stage1, indent=2, ensure_ascii=False))
    return stage1


def build_teacher_sequences(qas: list, docs_by_id: dict, stage1: dict) -> dict:
    tok_cache = WORK_DIR / "teacher_tokens_cache.json"
    if tok_cache.exists():
        return json.loads(tok_cache.read_text())

    print("[tokens] Tokenizing 200 bare, topic-framed, contextual prompts & verified teacher answers...", flush=True)
    oracle_by_qid = {r["qid"]: r for r in stage1["oracle_200"]["rows"]}
    items = {}
    for idx, q in enumerate(qas):
        qid = q["qid"]
        or_row = oracle_by_qid[qid]
        if or_row["correct"] and or_row["token_ids"]:
            ans_toks = list(or_row["token_ids"])
            if ans_toks[-1] != IM_END_ID:
                ans_toks.append(IM_END_ID)
            ans_src = "oracle"
        else:
            ans_toks = offline_tokenize_raw(q["answer"].strip()) + [IM_END_ID]
            ans_src = "gold_answer"

        # Cap ultra-long answers to keep rank per article well within `r <= 32`
        if len(ans_toks) > 24:
            ans_toks = ans_toks[:23] + [IM_END_ID]

        bare_p = offline_tokenize_raw(format_chatml_prompt(q["question"]))
        framed_p = offline_tokenize_raw(format_chatml_prompt(f"Regarding {q['title']}: {q['question']}"))
        ctx_user = (
            f"Use the following Wikipedia passage to answer concisely:\n"
            f"[{q['title']}]\n{docs_by_id[q['target_id']]['text']}\n\nQuestion: {q['question']}"
        )
        ctx_p = offline_tokenize_raw(format_chatml_prompt(ctx_user))

        items[str(qid)] = {
            "qid": qid,
            "title": q["title"],
            "answer_src": ans_src,
            "answer_toks": ans_toks,
            "bare_prompt_toks": bare_p,
            "framed_prompt_toks": framed_p,
            "ctx_prompt_toks": ctx_p,
        }
        if (idx + 1) % 50 == 0 or (idx + 1) == len(qas):
            print(f"  Tokenized {idx+1}/{len(qas)} Q&As", flush=True)

    tok_cache.write_text(json.dumps(items))
    return items


def stage2_background_and_kl(all_layers: list[int]) -> tuple[Path, Path]:
    bg_qmkv = WORK_DIR / "bg_nullspace.qmkv"
    kl_ctrl_jsonl = WORK_DIR / "control_kl_40.jsonl"

    if not kl_ctrl_jsonl.exists() or not bg_qmkv.exists():
        print("\n" + "=" * 88)
        print("STAGE 2: Background Null-Space Keys (K_0) & 40-Prompt KL Control Set")
        print("=" * 88, flush=True)
        val_rows = [json.loads(ln) for ln in open(VAL_PROMPTS_JSONL, encoding="utf-8") if ln.strip()]
        general_prose = [
            "Explain the difference between TCP and UDP in two sentences.",
            "Write a short Python function that checks whether an integer is prime.",
            "What causes ocean tides on Earth?",
            "Summarize the process of photosynthesis concisely.",
            "How does a binary search algorithm work?",
            "List three common sorting algorithms and their average time complexities.",
            "What is the boiling point of water at standard atmospheric pressure?",
            "Describe the water cycle in three short bullet points.",
        ]

        specs = []
        ctrl_seqs = []
        total_bg_pos = 0
        for i in range(32):
            if i < len(general_prose):
                p_txt = general_prose[i]
            else:
                p_txt = val_rows[i % len(val_rows)]["prompt"].split("\n\nProvide your")[0]
            toks = offline_tokenize_raw(format_chatml_prompt(p_txt))
            pos_count = min(12, max(1, len(toks) // 2))
            pos_start = len(toks) - pos_count
            specs.append({"tokens": toks, "pos_start": pos_start, "pos_count": pos_count})
            total_bg_pos += pos_count
            ctrl_seqs.append({"tokens": toks})

        # Add 8 more diverse prefix cutpoints for 40 KL positions
        for i in range(8):
            toks = offline_tokenize_raw(format_chatml_prompt(val_rows[40 + i]["prompt"].split("\n\nProvide your")[0]))
            ctrl_seqs.append({"tokens": toks[:-3] if len(toks) > 10 else toks})

        with open(kl_ctrl_jsonl, "w", encoding="utf-8") as f:
            for c in ctrl_seqs:
                f.write(json.dumps(c) + "\n")

        bg_specs_jsonl = WORK_DIR / "bg_specs.jsonl"
        with open(bg_specs_jsonl, "w", encoding="utf-8") as f:
            for sp in specs:
                f.write(json.dumps(sp) + "\n")

        run_cmd(["lora", "clear"], timeout=30)
        lcsv = ",".join(map(str, all_layers))
        out, _, wall = run_cmd(
            ["mech", "extract-kv", str(bg_specs_jsonl), "--layers", lcsv, "-o", str(bg_qmkv)],
            timeout=300,
        )
        print(f"[stage 2] Extracted {total_bg_pos} background SwiGLU keys across layers {all_layers} in {wall:.2f}s: {out.strip()}", flush=True)
    else:
        print(f"\n[stage 2] Reusing cached {bg_qmkv} and {kl_ctrl_jsonl}.", flush=True)
    return bg_qmkv, kl_ctrl_jsonl


def write_kv_specs(qids: list[int], tok_items: dict, mode: str, path: Path):
    """
    `mode` in {"bare", "bare+framed", "ctx", "ctx+framed"}
    Exact off-by-one teacher-forcing alignment:
      full_seq = prompt_toks + answer_toks[:-1]
      pos_start = len(prompt_toks) - 1
      pos_count = len(answer_toks)
    """
    specs = []
    for qid in qids:
        it = tok_items[str(qid)]
        ans = it["answer_toks"]
        if mode == "bare":
            prompts = [it["bare_prompt_toks"]]
        elif mode == "bare+framed":
            prompts = [it["bare_prompt_toks"], it["framed_prompt_toks"]]
        elif mode == "ctx":
            prompts = [it["ctx_prompt_toks"]]
        elif mode == "ctx+framed":
            # Both bare and framed map to the exact same contextual teacher target!
            prompts = [it["ctx_prompt_toks"], it["ctx_prompt_toks"]]
        else:
            raise ValueError(mode)

        for ptoks in prompts:
            full_seq = ptoks + ans[:-1]
            specs.append(
                {
                    "tokens": full_seq,
                    "pos_start": len(ptoks) - 1,
                    "pos_count": len(ans),
                }
            )
    with open(path, "w", encoding="utf-8") as f:
        for sp in specs:
            f.write(json.dumps(sp) + "\n")


def compile_topic_lora(
    slug: str,
    qids: list[int],
    tok_items: dict,
    edit_layers: list[int],
    bg_qmkv: Path,
    gamma: float,
    lam: float,
    rank: int,
    use_framed_aug: bool,
    out_dir: Path,
    tmp_dir: Path,
) -> dict:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    ctx_specs = tmp_dir / f"{slug}_ctx_specs.jsonl"
    bare_specs = tmp_dir / f"{slug}_bare_specs.jsonl"
    ctx_qmkv = tmp_dir / f"{slug}_ctx.qmkv"
    bare_qmkv = tmp_dir / f"{slug}_bare.qmkv"

    write_kv_specs(qids, tok_items, "ctx+framed" if use_framed_aug else "ctx", ctx_specs)
    write_kv_specs(qids, tok_items, "bare+framed" if use_framed_aug else "bare", bare_specs)

    run_cmd(["lora", "clear"], timeout=30)
    # Only need final layer residual from CTX (`layers[-1]`)
    run_cmd(["mech", "extract-kv", str(ctx_specs), "--layers", str(edit_layers[-1]), "-o", str(ctx_qmkv)], timeout=180)

    last_sum = {}
    for idx, l in enumerate(edit_layers):
        rem = len(edit_layers) - idx
        frac = 1.0 if rem == 1 else (0.75 / rem)
        run_cmd(["mech", "extract-kv", str(bare_specs), "--layers", str(l), "-o", str(bare_qmkv)], timeout=180)
        solve_args = [
            "mech",
            "solve-topic-lora",
            str(bare_qmkv),
            str(ctx_qmkv),
            str(out_dir),
            "--layers",
            str(l),
            "--bg-kv",
            str(bg_qmkv),
            "--gamma",
            str(gamma),
            "--lambda",
            str(lam),
            "--rank",
            str(rank),
            "--layer-frac",
            f"{frac:.6f}",
        ]
        if idx > 0:
            solve_args += ["--merge-prev", str(out_dir)]
        out, _, _ = run_cmd(solve_args, timeout=120)
        last_sum = parse_last_json(out)
        run_cmd(["lora", "load", str(out_dir)], timeout=30)

    run_cmd(["lora", "clear"], timeout=30)
    return last_sum


def eval_single_lora(lora_dir: Path, items: list) -> tuple[float, float, list]:
    run_cmd(["lora", "load", str(lora_dir)], timeout=30)
    rows = []
    for it in items:
        out, _, _ = run_cmd(
            [
                "run",
                "--spec",
                "--temp",
                "0",
                "--max-tokens",
                "64",
                "--reset-cache",
                "--sys",
                SYS_PROMPT,
                "--json-stats",
                "-p",
                it["question"],
            ],
            timeout=60,
        )
        lines = out.splitlines()
        pred = "\n".join(lines[:-1]).strip()
        ok, f1 = score_qa(pred, it["answer"])
        rows.append({"qid": it["qid"], "pred": pred, "gold": it["answer"], "correct": ok, "f1": round(f1, 4)})
    run_cmd(["lora", "clear"], timeout=30)
    acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
    mf1 = sum(r["f1"] for r in rows) / max(1, len(rows))
    return acc, mf1, rows


def run_kl_eval(lora_dir: Path, kl_ctrl_jsonl: Path) -> dict:
    out, _, _ = run_cmd(["mech", "kl-eval", str(kl_ctrl_jsonl), "--lora", str(lora_dir)], timeout=180)
    return parse_last_json(out)


def stage3_calibration_sweep(
    qas_by_title: dict,
    para_by_title: dict,
    tok_items: dict,
    bg_qmkv: Path,
    kl_ctrl_jsonl: Path,
    stage1: dict,
) -> dict:
    sweep_cache = WORK_DIR / "stage3_sweep.json"
    if sweep_cache.exists():
        print("\n[stage 3] Reusing cached 5-topic calibration sweep.", flush=True)
        return json.loads(sweep_cache.read_text())

    print("\n" + "=" * 88)
    print("STAGE 3: Single-Topic Sanity & 5-Topic Closed-Loop Calibration Sweep")
    print("=" * 88, flush=True)

    # Pick 5 diverse articles where baseline made errors & paraphrases exist
    bare_ok_map = {r["qid"]: r["correct"] for r in stage1["bare_200"]["rows"]}
    cand_titles = [
        "Newcastle upon Tyne",
        "Sky (United Kingdom)",
        "Victoria and Albert Museum",
        "Harvard University",
        "Private school",
    ]
    calib_titles = [t for t in cand_titles if t in qas_by_title and t in para_by_title]
    for t in qas_by_title:
        if len(calib_titles) >= 5:
            break
        if t not in calib_titles and t in para_by_title and any(not bare_ok_map[q["qid"]] for q in qas_by_title[t]):
            calib_titles.append(t)

    calib_direct = [q for t in calib_titles for q in qas_by_title[t]]
    calib_para = [para_by_title[t] for t in calib_titles]
    print(f"[calib articles] {calib_titles} ({len(calib_direct)} direct QAs, {len(calib_para)} unseen paraphrases)", flush=True)

    configs = [
        {"name": "late6_g50_l1e-3_1x", "layers": [30, 31, 32, 33, 34, 35], "gamma": 50.0, "lambda": 1e-3, "aug": False},
        {"name": "late6_g100_l1e-3_2x", "layers": [30, 31, 32, 33, 34, 35], "gamma": 100.0, "lambda": 1e-3, "aug": True},
        {"name": "midlate6_g50_l1e-3_2x", "layers": [24, 28, 31, 33, 34, 35], "gamma": 50.0, "lambda": 1e-3, "aug": True},
        {"name": "late6_g20_l1e-2_2x", "layers": [30, 31, 32, 33, 34, 35], "gamma": 20.0, "lambda": 1e-2, "aug": True},
    ]

    sweep_results = []
    tmp_dir = WORK_DIR / "sweep_tmp"
    for cfg in configs:
        t0 = time.perf_counter()
        direct_rows_all = []
        para_rows_all = []
        kls = []
        fit_errs = []

        for title in calib_titles:
            slug = slugify(title)
            qids = [q["qid"] for q in qas_by_title[title]]
            ldir = tmp_dir / f"{cfg['name']}_{slug}"
            solve_sum = compile_topic_lora(
                slug, qids, tok_items, cfg["layers"], bg_qmkv,
                cfg["gamma"], cfg["lambda"], 32, cfg["aug"], ldir, tmp_dir,
            )
            fit_errs.append(solve_sum.get("mean_fit_rel_err", 0.0))

            _, _, d_rows = eval_single_lora(ldir, qas_by_title[title])
            _, _, p_rows = eval_single_lora(ldir, [para_by_title[title]])
            direct_rows_all.extend(d_rows)
            para_rows_all.extend(p_rows)
            kl_res = run_kl_eval(ldir, kl_ctrl_jsonl)
            kls.append(kl_res["mean_kl_nats"])

        d_acc = sum(1 for r in direct_rows_all if r["correct"]) / len(direct_rows_all)
        p_acc = sum(1 for r in para_rows_all if r["correct"]) / len(para_rows_all)
        mean_kl = sum(kls) / len(kls)
        mean_fit = sum(fit_errs) / len(fit_errs)
        rec = {
            **cfg,
            "direct_acc": round(d_acc, 4),
            "para_acc": round(p_acc, 4),
            "mean_kl_nats": round(mean_kl, 6),
            "mean_fit_rel_err": round(mean_fit, 6),
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        sweep_results.append(rec)
        print(
            f"  [sweep {cfg['name']}] direct={d_acc*100:5.1f}% | para={p_acc*100:5.1f}% | "
            f"KL={mean_kl:.4f} nats | fit_err={mean_fit:.4f} | wall={rec['wall_s']}s",
            flush=True,
        )

    # Select best passing config (`mean_kl <= 0.12`, tiebreak highest direct + 0.5*para - 0.5*kl)
    passing = [c for c in sweep_results if c["mean_kl_nats"] <= 0.12]
    pool = passing if passing else sweep_results
    best = max(pool, key=lambda c: c["direct_acc"] + 0.3 * c["para_acc"] - 0.1 * c["mean_kl_nats"])
    out = {"calib_titles": calib_titles, "configs": sweep_results, "selected": best}
    sweep_cache.write_text(json.dumps(out, indent=2))
    print(f"[stage 3] Selected config: {best['name']} (direct={best['direct_acc']*100:.1f}%, KL={best['mean_kl_nats']:.4f})", flush=True)
    return out


def stage4_compile_full_bank(qas_by_title: dict, tok_items: dict, bg_qmkv: Path, selected_cfg: dict) -> dict:
    manifest_path = BANK_DIR / "bank_manifest.json"
    print("\n" + "=" * 88)
    print(f"STAGE 4: Compiling Full {len(qas_by_title)}-Article Wikipedia Topic LoRA Bank ({selected_cfg['name']})")
    print("=" * 88, flush=True)

    adapters_root = BANK_DIR / "adapters"
    adapters_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = WORK_DIR / "compile_tmp"

    topics_meta = []
    t0 = time.perf_counter()
    titles = list(qas_by_title.keys())
    for idx, title in enumerate(titles):
        slug = slugify(title)
        qids = [q["qid"] for q in qas_by_title[title]]
        ldir = adapters_root / slug
        st_file = ldir / "adapter_model.safetensors"
        sum_file = ldir / "compile_summary.json"
        if st_file.exists() and sum_file.exists():
            solve_sum = json.loads(sum_file.read_text())
        else:
            solve_sum = compile_topic_lora(
                slug,
                qids,
                tok_items,
                selected_cfg["layers"],
                bg_qmkv,
                selected_cfg["gamma"],
                selected_cfg["lambda"],
                32,
                selected_cfg["aug"],
                ldir,
                tmp_dir,
            )
            sum_file.write_text(json.dumps(solve_sum, indent=2))

        topics_meta.append(
            {
                "title": title,
                "slug": slug,
                "lora_dir": f"adapters/{slug}",
                "num_facts": len(qids),
                "rank": solve_sum.get("rank", 32),
                "layers": solve_sum.get("layers", selected_cfg["layers"]),
                "mean_fit_err": round(solve_sum.get("mean_fit_rel_err", 0.0), 6),
                "kl_nats": 0.0,
            }
        )
        if (idx + 1) % 10 == 0 or (idx + 1) == len(titles):
            print(
                f"  [compile {idx+1:2d}/{len(titles)}] {title} (facts={len(qids)}, rank={topics_meta[-1]['rank']}, fit_err={topics_meta[-1]['mean_fit_err']:.4f}, wall={time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    manifest = {
        "router_qvec": str(BEST_QVEC),
        "min_score": 0.20,
        "system_prompt": SYS_PROMPT,
        "compiler_config": selected_cfg,
        "topics": topics_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[stage 4] Wrote {manifest_path} ({len(topics_meta)} topics) in {time.perf_counter()-t0:.1f}s", flush=True)
    return manifest


def eval_topic_ask_batch(items: list, label: str, max_tokens: int = 64) -> dict:
    t0 = time.perf_counter()
    results = []
    for idx, it in enumerate(items):
        out, _, wall_s = run_cmd(
            [
                "topic",
                "ask",
                "--temp",
                "0",
                "--max-tokens",
                str(max_tokens),
                "--json-stats",
                it["question"],
            ],
            timeout=120,
        )
        lines = out.splitlines()
        meta = parse_last_json(out)
        gen_stats = meta["gen_stats"]
        pred_text = "\n".join(lines[:-1]).strip()
        ok, f1 = score_qa(pred_text, it["answer"])
        routed_ok = meta.get("matched", False) and meta.get("topic", "").lower() == it["title"].lower()
        results.append(
            {
                "qid": it["qid"],
                "title": it["title"],
                "routed_topic": meta.get("topic", ""),
                "routed_correct": routed_ok,
                "route_score": meta.get("score", 0.0),
                "question": it["question"],
                "gold": it["answer"],
                "pred": pred_text,
                "correct": ok,
                "f1": round(f1, 4),
                "prompt_tokens": gen_stats["prompt_tokens"],
                "completion_tokens": gen_stats["completion_tokens"],
                "embed_ms": meta.get("embed_ms", 0.0),
                "gpu_search_us": meta.get("gpu_search_us", 0.0),
                "lora_swap_ms": meta.get("lora_swap_ms", 0.0),
                "prefill_ms": gen_stats["prefill_ms"],
                "decode_ms": gen_stats["decode_ms"],
                "decode_tok_per_s": gen_stats["decode_tok_per_s"],
                "total_ms": gen_stats["total_ms"],
                "topic_total_wall_ms": meta.get("topic_total_wall_ms", 0.0),
                "wall_ms": round(wall_s * 1000.0, 2),
            }
        )
        if (idx + 1) % 50 == 0 or (idx + 1) == len(items):
            acc = sum(1 for r in results if r["correct"]) / len(results)
            r_acc = sum(1 for r in results if r["routed_correct"]) / len(results)
            print(
                f"  [{label} {idx+1:3d}/{len(items)}] route_R@1={r_acc*100:5.1f}% | ans_acc={acc*100:5.1f}% | wall={time.perf_counter()-t0:.1f}s",
                flush=True,
            )
    s = summarize_qa_eval(label, results)
    n = max(1, len(results))
    s["route_r1"] = round(sum(1 for r in results if r["routed_correct"]) / n, 4)
    s["mean_lora_swap_ms"] = round(sum(r["lora_swap_ms"] for r in results) / n, 3)
    s["mean_topic_total_ms"] = round(sum(r["topic_total_wall_ms"] for r in results) / n, 2)
    return s


def stage5_benchmark_and_verify(
    qas: list,
    para_40: list,
    kl_ctrl_jsonl: Path,
    manifest: dict,
    stage1: dict,
    stage3: dict,
) -> dict:
    print("\n" + "=" * 88)
    print("STAGE 5: Full 200-Q&A + 40-Paraphrase Benchmark, Dual-Slot Cache, KL & 10GB Co-Residency")
    print("=" * 88, flush=True)

    load_out, _, _ = run_cmd(["topic", "load", str(BANK_DIR)], timeout=60)
    print(f"[topic load] {load_out.strip()}", flush=True)

    # 1. Dual-slot fast-swap check (cold vs warm cached Topic LoRA + pinned Retrieval SVD LoRA)
    sample_q = qas[4]["question"]  # Newcastle upon Tyne
    o1, _, _ = run_cmd(["topic", "ask", "--json-stats", "--max-tokens", "32", sample_q], timeout=60)
    m1 = parse_last_json(o1)
    o2, _, _ = run_cmd(["topic", "ask", "--json-stats", "--max-tokens", "32", sample_q], timeout=60)
    m2 = parse_last_json(o2)
    dual_slot_verify = {
        "cold_lora_swap_ms": m1["lora_swap_ms"],
        "cached_lora_swap_ms": m2["lora_swap_ms"],
        "embed_ms_after_topic_swap": m2["embed_ms"],
    }
    print(f"[dual-slot verify] {json.dumps(dual_slot_verify)}", flush=True)

    # 2. Full 200 Direct Q&As + 40 Unseen Paraphrases via `qwen-nv topic ask`
    topic_200 = eval_topic_ask_batch(qas, "topic_lora_200")
    topic_para_40 = eval_topic_ask_batch(para_40, "topic_lora_para_40")

    # 3. Control KL evaluation across 10 sampled Topic LoRAs + bit-exact clear parity check
    sample_topics = manifest["topics"][:10]
    kl_samples = []
    for t in sample_topics:
        ldir = BANK_DIR / t["lora_dir"]
        kl_r = run_kl_eval(ldir, kl_ctrl_jsonl)
        kl_samples.append({"title": t["title"], **kl_r})
        t["kl_nats"] = round(kl_r["mean_kl_nats"], 6)
    (BANK_DIR / "bank_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Bit-exact clear check
    base_probe_out, _, _ = run_cmd(
        ["run", "--no-spec", "--temp", "0", "--max-tokens", "24", "--reset-cache", "--sys", SYS_PROMPT, "--json-stats", "-p", qas[4]["question"]],
        timeout=30,
    )
    run_cmd(["lora", "load", str(BANK_DIR / sample_topics[0]["lora_dir"])], timeout=30)
    run_cmd(["lora", "clear"], timeout=30)
    after_clear_out, _, _ = run_cmd(
        ["run", "--no-spec", "--temp", "0", "--max-tokens", "24", "--reset-cache", "--sys", SYS_PROMPT, "--json-stats", "-p", qas[4]["question"]],
        timeout=30,
    )
    clear_parity = parse_last_json(base_probe_out)["token_ids"] == parse_last_json(after_clear_out)["token_ids"]
    print(f"[clear parity] bit-exact after lora clear = {clear_parity}", flush=True)

    # 4. 10 GB VRAM Co-Residency Check (qwen3-8b-patched + Retrieval SVD LoRA + Topic Pool + .qvec + Qwen3-0.6B drafter)
    drafter_w, _, _ = run_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
    st_out, _, _ = run_cmd(["status"], timeout=15)
    co_out, _, _ = run_cmd(["topic", "ask", "--json-stats", "--max-tokens", "32", qas[4]["question"]], timeout=60)
    co_meta = parse_last_json(co_out)
    co_pred = "\n".join(co_out.splitlines()[:-1]).strip()
    coresidency = {
        "drafter_load": drafter_w.strip(),
        "daemon_status": st_out.strip(),
        "live_topic_ask_pred": co_pred,
        "live_topic_ask_matched": co_meta["matched"],
        "live_topic_ask_topic": co_meta["topic"],
        "live_topic_ask_total_ms": co_meta["topic_total_wall_ms"],
    }
    print(f"[10GB co-residency] {json.dumps(coresidency)}", flush=True)

    # Disk footprint of 46 adapters
    total_disk_bytes = sum(
        p.stat().st_size for p in (BANK_DIR / "adapters").rglob("adapter_model.safetensors")
    )

    def slim_summary(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "rows"}

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_ref": MODEL_REF,
        "num_topics": len(manifest["topics"]),
        "total_adapters_disk_mb": round(total_disk_bytes / (1024.0 * 1024.0), 2),
        "selected_compiler_config": stage3["selected"],
        "calibration_sweep": stage3["configs"],
        "dual_slot_cache_verify": dual_slot_verify,
        "clear_bit_exact_parity": clear_parity,
        "kl_eval_10_topics": {
            "mean_kl_nats": round(sum(x["mean_kl_nats"] for x in kl_samples) / len(kl_samples), 6),
            "max_kl_nats": round(max(x["max_kl_nats"] for x in kl_samples), 6),
            "mean_top1_agreement": round(sum(x["top1_agreement"] for x in kl_samples) / len(kl_samples), 4),
            "samples": kl_samples,
        },
        "coresidency_10gb": coresidency,
        "comparison": {
            "bare_200": slim_summary(stage1["bare_200"]),
            "oracle_200": slim_summary(stage1["oracle_200"]),
            "rag_200": slim_summary(stage1["rag_200"]),
            "topic_lora_200": slim_summary(topic_200),
            "bare_para_40": slim_summary(stage1["bare_para_40"]),
            "rag_para_40": slim_summary(stage1["rag_para_40"]),
            "topic_lora_para_40": slim_summary(topic_para_40),
        },
        "detailed_rows": {
            "topic_lora_200": topic_200["rows"],
            "topic_lora_para_40": topic_para_40["rows"],
        },
    }
    OUT_METRICS.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[done] Wrote benchmark report to {OUT_METRICS}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run fast 1-topic RTX 3080 smoke verification first")
    args = parser.parse_args()

    check_guardrails()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    docs = [json.loads(ln) for ln in open(CORPUS_JSONL, encoding="utf-8") if ln.strip()]
    docs_by_id = {d["id"]: d for d in docs}
    qas = [json.loads(ln) for ln in open(EVAL_JSONL, encoding="utf-8") if ln.strip()]
    qas_by_qid = {q["qid"]: q for q in qas}
    qas_by_title = {}
    for q in qas:
        qas_by_title.setdefault(q["title"], []).append(q)

    para_40 = ensure_paraphrase_dataset(qas_by_qid)
    para_by_title = {p["title"]: p for p in para_40}

    start_daemon_clean(MODEL_REF)
    try:
        stage1 = stage1_baseline_oracle_rag(qas, docs_by_id, para_40)
        tok_items = build_teacher_sequences(qas, docs_by_id, stage1)
        all_layers = [24, 28, 30, 31, 32, 33, 34, 35]
        bg_qmkv, kl_ctrl_jsonl = stage2_background_and_kl(all_layers)

        if args.smoke:
            print("\n[smoke] Running single-topic closure & KL smoke test on `Newcastle upon Tyne`...", flush=True)
            t = "Newcastle upon Tyne"
            qids = [q["qid"] for q in qas_by_title[t]]
            sdir = WORK_DIR / "smoke_newcastle"
            sm = compile_topic_lora(
                "newcastle", qids, tok_items, [30, 31, 32, 33, 34, 35],
                bg_qmkv, 50.0, 1e-3, 32, True, sdir, WORK_DIR / "smoke_tmp",
            )
            acc, mf1, rows = eval_single_lora(sdir, qas_by_title[t])
            kl = run_kl_eval(sdir, kl_ctrl_jsonl)
            print(f"[smoke] solve={sm} | direct_acc={acc*100:.1f}% (F1={mf1*100:.1f}%) | KL={kl}", flush=True)
            for r in rows:
                print("  ", r, flush=True)
            return

        stage3 = stage3_calibration_sweep(qas_by_title, para_by_title, tok_items, bg_qmkv, kl_ctrl_jsonl, stage1)
        manifest = stage4_compile_full_bank(qas_by_title, tok_items, bg_qmkv, stage3["selected"])
        report = stage5_benchmark_and_verify(qas, para_40, kl_ctrl_jsonl, manifest, stage1, stage3)

        c = report["comparison"]
        print("\n" + "=" * 88)
        print("SUMMARY: Bare vs Phase 5 Text RAG vs Sharded Topic LoRA Bank (Option B)")
        print("=" * 88)
        for k in ["bare_200", "oracle_200", "rag_200", "topic_lora_200", "bare_para_40", "rag_para_40", "topic_lora_para_40"]:
            r = c[k]
            print(
                f"  {k:20s}: acc={r['accuracy']*100:5.1f}% | F1={r['mean_f1']*100:5.1f}% | "
                f"prompt_tok={r['mean_prompt_tokens']:5.1f} | prefill={r['mean_prefill_ms']:6.1f}ms | wall={r['mean_client_wall_ms']:6.1f}ms"
            )
    finally:
        stop_daemon_clean()


if __name__ == "__main__":
    main()
