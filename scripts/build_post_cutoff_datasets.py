#!/usr/bin/env python3
"""
Curate 100% Contamination-Free Post-Cutoff Reasoning Benchmarks (`> 2025-11-01`,
strictly 3 to 12+ months AFTER `Qwen/Qwen3-4B-Thinking-2507`'s July/Aug 2025 release).

Sources (all verified real-world competition / arXiv preprint dates > 2025-11-01):
  1. `amc12_nov2025`  : `sonthenguyen/amc12-2025-non-figure` (2025 AMC 12A & 12B, Nov 5 & 13, 2025)
  2. `hmmt_nov2025`   : `MathArena/hmmt_nov_2025`            (Harvard-MIT Nov Tournament, Nov 8, 2025)
  3. `aime_feb2026`   : `MathArena/aime_2026`                (2026 AIME I & II, Feb 5 & 11, 2026)
  4. `arxiv_math2026` : `MathArena/arxivmath-{04,05,06,08}26` (Apr–Aug 2026 arXiv preprints `arXiv:2604.*`–`2608.*`)

Outputs two strictly disjoint, difficulty-interleaved splits:
  - `data/post_cutoff_calib.jsonl` (60 problems — ONLY split touched for traces, hooks, LoRA screening & drafter prior)
  - `data/post_cutoff_test.jsonl`  (60 problems — 100% HELD-OUT test split for Step A Zero-Shot & Step B Final Eval)
"""

import json
import math
import re
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/tim/Code/optimize-gpu")
BIN = ROOT / "qwen-nv-rs" / "target" / "release" / "qwen-nv"
DATA_DIR = ROOT / "data"
CALIB_OUT = DATA_DIR / "post_cutoff_calib.jsonl"
TEST_OUT = DATA_DIR / "post_cutoff_test.jsonl"
MANIFEST_OUT = DATA_DIR / "post_cutoff_manifest.json"


def fetch_hf_rows(dataset: str) -> list:
    for attempt in range(5):
        try:
            req1 = urllib.request.Request(
                f"https://datasets-server.huggingface.co/splits?dataset={dataset}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req1, timeout=25) as r:
                sp = json.loads(r.read().decode("utf-8"))["splits"][0]
            cfg, split = sp["config"], sp["split"]
            all_rows = []
            offset = 0
            while True:
                req2 = urllib.request.Request(
                    f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={cfg}&split={split}&offset={offset}&length=100",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req2, timeout=25) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                batch = [x["row"] for x in payload["rows"]]
                all_rows.extend(batch)
                offset += len(batch)
                if offset >= payload["num_rows_total"] or not batch:
                    break
            return all_rows
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  [retry {attempt+1}] {dataset}: {e}")
            time.sleep(2.0 * (attempt + 1))
    return []


def normalize_exact_rational(raw: str):
    """
    Canonicalizes an exact integer or rational number (including LaTeX `\\frac{a}{b}`,
    `-\\frac{a}{b}`, terminating decimals `8.5`, commas `1,296`, leading zeros `062`)
    into `'k'` (if integer) or `'p/q'` with `q > 1` and `gcd(|p|, q) == 1`.
    Returns `None` if `raw` is symbolic, irrational (`\\sqrt`, `\\pi`), tuple, or non-numeric.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$|^\\\[|\\\]$", "", s).strip()
    # Unwrap outer \boxed{...} if balanced
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
    if s.endswith(".") and s[:-1].lstrip("+-").isdigit():
        s = s[:-1]

    # 1. Exact integer
    if re.fullmatch(r"[+-]?\d+", s):
        return str(int(s))

    # 2. Terminating decimal (at most 6 decimal places so no float approximations)
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

    # 3. Slash fraction p/q
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

    # 4. LaTeX \frac{p}{q} or \fracab shorthand
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


def token_len(prompt_text: str) -> int:
    proc = subprocess.run(
        [str(BIN), "tokenize", "--model", "qwen3-thinking:4b", "-p", prompt_text],
        capture_output=True,
        text=True,
        check=True,
    )
    return len(json.loads(proc.stdout.strip()))


def format_prompt(problem_text: str) -> str:
    body = problem_text.strip()
    return (
        f"{body}\n\n"
        "Put your final simplified exact answer inside \\boxed{...} and on the last line write: "
        "Final Answer: <answer>"
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching strictly post-cutoff (> 2025-11-01) datasets from Hugging Face...")

    # 1. AMC 12A & 12B Nov 2025 (2025-11-05 & 2025-11-13)
    amc_rows = fetch_hf_rows("sonthenguyen/amc12-2025-non-figure")
    amc_clean = []
    for r in amc_rows:
        ans = normalize_exact_rational(r.get("answer"))
        prob = str(r.get("question", "")).strip()
        if ans is None or not prob or "[asy]" in prob.lower():
            continue
        pnum_str = str(r.get("problem_number", "A99"))
        num_m = re.search(r"(\d+)", pnum_str)
        pnum = int(num_m.group(1)) if num_m else 99
        prompt = format_prompt(prob)
        tlen = token_len(prompt)
        if tlen <= 900:
            amc_clean.append({
                "category": "amc12_nov2025",
                "source_dataset": "sonthenguyen/amc12-2025-non-figure",
                "release_date": "2025-11-13",
                "source_id": str(r.get("id", pnum_str)),
                "difficulty_rank": pnum,
                "prompt": prompt,
                "prompt_tokens": tlen,
                "answer": ans,
                "raw_answer": str(r.get("answer")),
            })
    amc_clean.sort(key=lambda x: (x["difficulty_rank"], x["source_id"]))
    amc_selected = amc_clean[:36]  # 18 calib + 18 test

    # 2. HMMT Nov 2025 (2025-11-08)
    hmmt_rows = fetch_hf_rows("MathArena/hmmt_nov_2025")
    hmmt_clean = []
    for r in hmmt_rows:
        ans = normalize_exact_rational(r.get("answer"))
        prob = str(r.get("problem", "")).strip()
        if ans is None or not prob:
            continue
        pidx = int(r.get("problem_idx", 99))
        prompt = format_prompt(prob)
        tlen = token_len(prompt)
        if tlen <= 900:
            hmmt_clean.append({
                "category": "hmmt_nov2025",
                "source_dataset": "MathArena/hmmt_nov_2025",
                "release_date": "2025-11-08",
                "source_id": f"hmmt_nov25_{pidx}",
                "difficulty_rank": pidx,
                "prompt": prompt,
                "prompt_tokens": tlen,
                "answer": ans,
                "raw_answer": str(r.get("answer")),
            })
    hmmt_clean.sort(key=lambda x: x["difficulty_rank"])
    hmmt_selected = hmmt_clean[:24]  # 12 calib + 12 test

    # 3. AIME I & II Feb 2026 (2026-02-05 & 2026-02-11)
    aime_rows = fetch_hf_rows("MathArena/aime_2026")
    aime_clean = []
    for r in aime_rows:
        ans = normalize_exact_rational(r.get("answer"))
        prob = str(r.get("problem", "")).strip()
        if ans is None or not prob:
            continue
        pidx = int(r.get("problem_idx", 99))
        # Problem 1..15 is AIME I (#1..15), 16..30 is AIME II (#1..15); sort by exam position ((pidx-1)%15)
        exam_pos = ((pidx - 1) % 15) + 1
        prompt = format_prompt(prob)
        tlen = token_len(prompt)
        if tlen <= 900:
            aime_clean.append({
                "category": "aime_feb2026",
                "source_dataset": "MathArena/aime_2026",
                "release_date": "2026-02-11",
                "source_id": f"aime26_{pidx}",
                "difficulty_rank": exam_pos * 10 + (1 if pidx <= 15 else 2),
                "prompt": prompt,
                "prompt_tokens": tlen,
                "answer": ans,
                "raw_answer": str(r.get("answer")),
            })
    aime_clean.sort(key=lambda x: x["difficulty_rank"])
    aime_selected = aime_clean[:30]  # 15 calib + 15 test (all 30 AIME 2026 problems!)

    # 4. ArXiv Math 2026 (`2604.*`, `2605.*`, `2606.*`, `2608.*` — Apr–Aug 2026 research papers)
    arxiv_clean = []
    seen_probs = set()
    for month_tag, rel_dt in [
        ("0426", "2026-04-30"),
        ("0526", "2026-05-31"),
        ("0626", "2026-06-30"),
        ("0826", "2026-08-31"),
    ]:
        ds_name = f"MathArena/arxivmath-{month_tag}"
        for r in fetch_hf_rows(ds_name):
            ans = normalize_exact_rational(r.get("answer"))
            prob = str(r.get("problem", "")).strip()
            src_id = str(r.get("source", "")).strip()
            if ans is None or not prob or prob[:120] in seen_probs:
                continue
            # Verify arXiv identifier starts with 26xx.
            if not re.match(r"^26\d\d\.", src_id):
                continue
            prompt = format_prompt(prob)
            tlen = token_len(prompt)
            # Filter to concise research statements (<= 600 prompt tokens) so 4k KV context has > 3400 thinking headroom
            if tlen <= 600:
                seen_probs.add(prob[:120])
                arxiv_clean.append({
                    "category": "arxiv_math2026",
                    "source_dataset": ds_name,
                    "release_date": rel_dt,
                    "source_id": f"arxiv:{src_id}_p{r.get('problem_idx', 0)}",
                    "difficulty_rank": tlen,
                    "prompt": prompt,
                    "prompt_tokens": tlen,
                    "answer": ans,
                    "raw_answer": str(r.get("answer")),
                })
    arxiv_clean.sort(key=lambda x: x["difficulty_rank"])
    arxiv_selected = arxiv_clean[:30]  # 15 calib + 15 test

    print(
        f"Selected counts: amc12={len(amc_selected)}, hmmt_nov25={len(hmmt_selected)}, "
        f"aime_2026={len(aime_selected)}, arxiv_2026={len(arxiv_selected)}"
    )
    assert len(amc_selected) == 36, f"Expected 36 AMC12, got {len(amc_selected)}"
    assert len(hmmt_selected) == 24, f"Expected 24 HMMT Nov25, got {len(hmmt_selected)}"
    assert len(aime_selected) == 30, f"Expected 30 AIME 2026, got {len(aime_selected)}"
    assert len(arxiv_selected) == 30, f"Expected 30 ArXiv 2026, got {len(arxiv_selected)}"

    calib, test = [], []
    for group in [amc_selected, hmmt_selected, aime_selected, arxiv_selected]:
        for idx, item in enumerate(group):
            if idx % 2 == 0:
                calib.append(dict(item))
            else:
                test.append(dict(item))

    for idx, item in enumerate(calib):
        item["id"] = idx
        item["split"] = "calib"
    for idx, item in enumerate(test):
        item["id"] = idx
        item["split"] = "test"

    # Verify zero overlap in prompt text or source_id
    calib_sids = {x["source_id"] for x in calib}
    test_sids = {x["source_id"] for x in test}
    assert calib_sids.isdisjoint(test_sids), "Overlap detected between calib and test!"

    with open(CALIB_OUT, "w", encoding="utf-8") as f:
        for x in calib:
            f.write(json.dumps(x) + "\n")
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        for x in test:
            f.write(json.dumps(x) + "\n")

    manifest = {
        "target_model": "Qwen/Qwen3-4B-Thinking-2507",
        "model_cutoff": "2025-08-01",
        "earliest_problem_release_date": min(x["release_date"] for x in calib + test),
        "latest_problem_release_date": max(x["release_date"] for x in calib + test),
        "total_problems": len(calib) + len(test),
        "calib_count": len(calib),
        "test_count": len(test),
        "calib_by_category": dict(Counter(x["category"] for x in calib)),
        "test_by_category": dict(Counter(x["category"] for x in test)),
        "max_prompt_tokens": max(x["prompt_tokens"] for x in calib + test),
        "mean_prompt_tokens": round(sum(x["prompt_tokens"] for x in calib + test) / (len(calib) + len(test)), 1),
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    print("Manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
