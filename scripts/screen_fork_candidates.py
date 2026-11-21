#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/tim/Code/optimize-gpu")
from run_autonomous_weight_edit import analyze_trace, parse_last_json

ROOT = Path("/Users/tim/Code/optimize-gpu")
BIN = ROOT / "qwen-nv-rs/target/release/qwen-nv"
WORK = ROOT / "data/mech_loop"
BF16 = Path.home() / ".cache/qwen-nv-rs/models/Qwen3-4B-Thinking-2507-BF16"
DIRS = WORK / "fork_dirs.bin"
KL_CTRL = WORK / "fork_kl_ctrl.jsonl"

prompts = {p["id"]: p for p in (json.loads(ln) for ln in open(ROOT / "data/val_prompts.jsonl") if ln.strip())}
traces = {t["id"]: t for t in (json.loads(ln) for ln in open(WORK / "traces_baseline.jsonl") if ln.strip())}

screen_ids = [0, 1, 20, 21, 40, 41, 60, 61, 80, 81]
base_acc = sum(1 for i in screen_ids if traces[i]["is_correct"]) / len(screen_ids)
base_end = sum(1 for i in screen_ids if traces[i]["has_end_think"])
base_th = sum(traces[i]["think_tokens"] for i in screen_ids) / len(screen_ids)
base_hes = sum(traces[i]["hesitation_count"] for i in screen_ids) / len(screen_ids)
print(
    f"Baseline on 10-prompt 5-category screen: acc={base_acc*100:.1f}%, "
    f"finished_</think>={base_end}/10, avg_think={base_th:.1f}, avg_hes={base_hes:.1f}"
)

specs = [
    ("L20_33_down_a1.0", list(range(20, 34)), 1, 1.00, ["--down-only"]),
]

subprocess.run([str(BIN), "drafter", "clear"], check=True, capture_output=True)
kl_map = {}
for name, layers, rk, alpha, flags in specs:
    ldir = WORK / f"fork_lora_{name}"
    lcsv = ",".join(map(str, layers))
    subprocess.run(
        [
            str(BIN),
            "mech",
            "make-lora",
            str(BF16),
            str(DIRS),
            str(ldir),
            "--layers",
            lcsv,
            "--rank",
            str(rk),
            "--alpha",
            str(alpha),
        ]
        + flags,
        check=True,
        capture_output=True,
    )
    out = subprocess.run(
        [str(BIN), "mech", "kl-eval", str(KL_CTRL), "--lora", str(ldir)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    kj = parse_last_json(out)
    kl_map[name] = kj
    print(
        f"  [KL {name:20s}] mean_kl={kj['mean_kl_nats']:.5f} nats | "
        f"max_kl={kj['max_kl_nats']:.5f} | top1_agree={kj['top1_agreement']*100:.1f}%"
    )

subprocess.run([str(BIN), "drafter", "load", "qwen3:0.6b"], check=True, capture_output=True)
results = []
for name, layers, rk, alpha, flags in specs:
    kj = kl_map[name]
    if kj["mean_kl_nats"] > 0.30:
        print(f"  [SKIP {name}] KL too high ({kj['mean_kl_nats']:.4f})")
        continue
    ldir = WORK / f"fork_lora_{name}"
    subprocess.run([str(BIN), "lora", "load", str(ldir)], check=True, capture_output=True)
    t0 = time.perf_counter()
    recs = []
    for pid in screen_ids:
        item = prompts[pid]
        out = subprocess.run(
            [
                str(BIN),
                "run",
                "--spec",
                "--temp",
                "0",
                "--max-tokens",
                "1472",
                "--reset-cache",
                "--json-stats",
                "-p",
                item["prompt"],
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        lines = out.splitlines()
        st = parse_last_json(out)
        recs.append(analyze_trace(item, "\n".join(lines[:-1]), st))
    subprocess.run([str(BIN), "lora", "clear"], check=True, capture_output=True)
    acc = sum(1 for r in recs if r["is_correct"]) / len(recs)
    end_c = sum(1 for r in recs if r["has_end_think"])
    loops = sum(1 for r in recs if r["is_circular_loop"])
    avg_th = sum(r["think_tokens"] for r in recs) / len(recs)
    avg_hes = sum(r["hesitation_count"] for r in recs) / len(recs)
    print(
        f"  [SCREEN {name:18s}] acc={acc*100:5.1f}% (vs {base_acc*100:.1f}%) | "
        f"end_</think>={end_c:2d}/10 (vs {base_end}/10) | loops={loops:2d} | "
        f"avg_think={avg_th:6.1f} | avg_hes={avg_hes:4.1f} | KL={kj['mean_kl_nats']:.4f} | {time.perf_counter()-t0:.1f}s"
    )
    results.append(
        {
            "name": name,
            "layers": layers,
            "rank_k": rk,
            "alpha": alpha,
            "flags": flags,
            "kl": kj,
            "acc": acc,
            "end_think_of_10": end_c,
            "loops": loops,
            "avg_think": avg_th,
            "avg_hes": avg_hes,
        }
    )

(WORK / "fork_screen_results.json").write_text(json.dumps(results, indent=2))
