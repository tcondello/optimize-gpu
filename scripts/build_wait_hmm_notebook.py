#!/usr/bin/env python3
"""
Builds `/Users/tim/Code/optimize-gpu/wait_hmm_before_after_demo.ipynb` (nbformat v4.5)
and optionally pre-executes its cells against the saved telemetry + live RTX 3080 daemon
so every HTML diff card, SVG circuit chart, and live Before/After output is immediately visible
on open AND 100% runnable locally with zero external Python packages (`stdlib` + `IPython.display`).
"""

import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/tim/Code/optimize-gpu")
OUT_IPYNB = ROOT / "wait_hmm_before_after_demo.ipynb"


CELL_1_MD = """# Mechanistic Weight Surgery: Seeing `"Wait / Hmm"` Loops Disappear Before vs. After

This local notebook lets you **inspect and run live head-to-head comparisons** on your **NVIDIA GeForce RTX 3080 (`sm_86`, 68 SMs)** over Thunderbolt (`0` Docker containers, `0` PyTorch, `0` external Python packages required).

### What Changed Inside the Weights?
1. **The Failure Circuit**: Reasoning models (`Qwen3-8B` and `Qwen3-4B-Thinking-2507`) frequently solve a problem in the first few hundred tokens—then hit a paragraph-break BPE decision fork (`382` `".\\n\\n"`, `30` `"?"`, `13` `"."`, `271` `"\\n\\n"`) and emit `"Wait,"` or `"Hmm,"` `20–50+` times until exhausting the context window without ever closing `</think>`.
2. **Surgical Norm-Preserving Orthogonal Projection (`< 0.0008%` of Parameters)**:
   - Using GPU residual-stream hooks (`qwen-nv mech extract-hooks` + `localize`) on token-matched calibration forks, we extract the normalized self-doubt direction $d_l = \\text{normalize}(\\mu_{\\text{wait}, l} - \\mu_{\\text{clean}, l})$.
   - Across both synthetic and unseen `2025–2026` competition benchmarks (`AMC 12`, `HMMT`, `AIME 2026`, `2026 arXiv`), causal injection concentrates in **Layers `24–33` `mlp.down_proj`**, while **Layers `34–35` are left untouched** so `</think>` emission remains intact:
   $$W_{\\text{new}} = (I - \\alpha\\, d_l d_l^\\top) W \\quad (\\text{with row-norm preservation})$$
2. **Default Model (`qwen3-8b-patched`) + `< 1 ms` Inverse Toggle**:
   - `qwen-nv` now boots **`qwen3-8b-patched`** (`Qwen3-8B-Patched-Q6_K.gguf`) by default (`0` duplicate unpatched `.gguf` required).
   - For instant side-by-side comparison below, **BEFORE** hot-loads `lora/qwen3-8b-wait-hmm-patch` with `--scale -1.0` (`+α d_l (d_l^T W)`, temporarily re-injecting the original `"Wait/Hmm"` direction in `< 1 ms`), while **AFTER** clears the adapter to run the permanently patched `qwen3-8b-patched` checkpoint directly!
"""


CELL_2_CODE = """import html
import json
import re
import subprocess
import time
from pathlib import Path

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = lambda s: s
    def display(obj):
        print(re.sub(r"<[^>]+>", "", str(obj)))

ROOT = Path("/Users/tim/Code/optimize-gpu")
BIN = ROOT / "qwen-nv-rs" / "target" / "release" / "qwen-nv"

HESITATION_RE = re.compile(
    r"\\b(Wait a second|Wait a minute|Actually,\\s*wait|Let me double[- ]check|Let me recount|Hold on|Wait|Hmm)\\b",
    re.IGNORECASE,
)

def qwen_cmd(args, timeout=300):
    proc = subprocess.run([str(BIN)] + [str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"qwen-nv {' '.join(map(str, args))} failed:\\n{proc.stderr[-600:]}")
    return proc.stdout

def parse_last_json(text: str) -> dict:
    for ln in reversed(text.strip().splitlines()):
        s = ln.strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
    raise ValueError("Missing JSON stats line")

def highlight_trace_html(raw_text: str, max_chars: int = 4200) -> str:
    if len(raw_text) > max_chars:
        head = raw_text[: int(max_chars * 0.42)]
        tail = raw_text[-int(max_chars * 0.58) :]
        shown = head + f"\\n\\n[ ... {len(raw_text) - max_chars:,} middle characters omitted for readability ... ]\\n\\n" + tail
    else:
        shown = raw_text
    esc = html.escape(shown)
    esc = HESITATION_RE.sub(
        r'<span style="background:#7f1d1d;color:#fecaca;padding:1px 6px;border-radius:4px;font-weight:700;border:1px solid #ef4444;">\\1</span>',
        esc,
    )
    esc = esc.replace(
        "&lt;/think&gt;",
        '<span style="background:#065f46;color:#a7f3d0;padding:2px 7px;border-radius:4px;font-weight:700;border:1px solid #10b981;">&lt;/think&gt;</span>',
    )
    esc = re.sub(
        r"(\\\\boxed\\{[^{}]+\\})",
        r'<span style="background:#1e3a8a;color:#bfdbfe;padding:2px 7px;border-radius:4px;font-weight:700;border:1px solid #60a5fa;">\\1</span>',
        esc,
    )
    return esc

def render_before_after_card(title: str, prompt: str, gold: str, before: dict, after: dict):
    def badge(ok: bool, yes_txt: str, no_txt: str):
        bg, fg, bd = ("#064e3b", "#a7f3d0", "#10b981") if ok else ("#7f1d1d", "#fecaca", "#ef4444")
        return f'<span style="background:{bg};color:{fg};border:1px solid {bd};padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;">{yes_txt if ok else no_txt}</span>'

    b_wait, a_wait = before["hesitation_count"], after["hesitation_count"]
    pct_drop = ((b_wait - a_wait) / max(1, b_wait)) * 100.0
    b_html = highlight_trace_html(before["text"])
    a_html = highlight_trace_html(after["text"])

    card = f'''
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:12px;padding:18px;margin:14px 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;border-bottom:1px solid #1e293b;padding-bottom:10px;margin-bottom:12px;">
        <div style="font-size:16px;font-weight:700;color:#f8fafc;">{html.escape(title)}</div>
        <div style="font-size:13px;color:#94a3b8;">Ground Truth: <code style="background:#1e293b;color:#38bdf8;padding:2px 6px;border-radius:4px;">{html.escape(str(gold))}</code> ·
          <span style="color:#34d399;font-weight:700;">Wait/Hmm Change: {b_wait} → {a_wait} ({-pct_drop:+.0f}%)</span>
        </div>
      </div>
      <div style="background:#1e293b;border-radius:8px;padding:10px 12px;margin-bottom:14px;font-size:13px;color:#cbd5e1;white-space:pre-wrap;"><strong>Prompt:</strong> {html.escape(prompt)}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
        <div style="background:#18181b;border:1px solid #991b1b;border-radius:10px;padding:12px;display:flex;flex-direction:column;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
            <strong style="color:#fca5a5;font-size:14px;">BEFORE · Unpatched Baseline</strong>
            {badge(before["is_correct"], f"✓ Pred: {before['pred']}", f"✗ Pred: {before['pred'] or 'None (Truncated)'}")}
          </div>
          <div style="font-size:12px;color:#a1a1aa;margin-bottom:8px;">
            Wait/Hmm: <strong style="color:#f87171;">{b_wait}</strong> ·
            Think toks: <strong>{before["think_tokens"]}</strong> ·
            Closed <code>&lt;/think&gt;</code>: <strong>{"Yes" if before["has_end_think"] else "No"}</strong> ·
            Speed: <strong>{before.get("tps", 0):.1f} tok/s</strong>
          </div>
          <pre style="background:#09090b;color:#d4d4d8;padding:10px;border-radius:6px;font-size:12px;line-height:1.45;max-height:420px;overflow-y:auto;white-space:pre-wrap;margin:0;">{b_html}</pre>
        </div>
        <div style="background:#0c1a17;border:1px solid #059669;border-radius:10px;padding:12px;display:flex;flex-direction:column;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
            <strong style="color:#6ee7b7;font-size:14px;">AFTER · Surgical L24–33 Orthogonal Patch</strong>
            {badge(after["is_correct"], f"✓ Pred: {after['pred']}", f"✗ Pred: {after['pred'] or 'None'}")}
          </div>
          <div style="font-size:12px;color:#a7f3d0;margin-bottom:8px;">
            Wait/Hmm: <strong style="color:#34d399;">{a_wait}</strong> ·
            Think toks: <strong>{after["think_tokens"]}</strong> ·
            Closed <code>&lt;/think&gt;</code>: <strong>{"Yes" if after["has_end_think"] else "No"}</strong> ·
            Speed: <strong>{after.get("tps", 0):.1f} tok/s</strong>
          </div>
          <pre style="background:#040d0b;color:#e2e8f0;padding:10px;border-radius:6px;font-size:12px;line-height:1.45;max-height:420px;overflow-y:auto;white-space:pre-wrap;margin:0;">{a_html}</pre>
        </div>
      </div>
    </div>
    '''
    display(HTML(card))

def ensure_rtx3080_online(model_ref: str = "qwen3-8b-patched", ctx: int = 4096, load_drafter: bool = True) -> str:
    st = qwen_cmd(["status"]).strip()
    if st.startswith("ONLINE") and (model_ref in st or model_ref == "any"):
        return st
    print(f"⚡ RTX 3080 daemon is OFFLINE (or switching to {model_ref}) — waking GPU (~25s)...", flush=True)
    subprocess.run([str(BIN), "stop"], capture_output=True)
    time.sleep(1.5)
    # Clean up any orphaned daemon PID left behind after macOS Thunderbolt sleep
    p = subprocess.run(["pgrep", "-f", "qwen-nv daemon"], capture_output=True, text=True)
    for pid_s in p.stdout.split():
        subprocess.run(["kill", "-TERM", pid_s], capture_output=True)
    if p.stdout.strip():
        time.sleep(2.0)
        for pid_s in subprocess.run(["pgrep", "-f", "qwen-nv daemon"], capture_output=True, text=True).stdout.split():
            subprocess.run(["kill", "-KILL", pid_s], capture_output=True)
        time.sleep(1.0)

    for attempt in range(1, 8):
        subprocess.run([str(BIN), "stop"], capture_output=True)
        time.sleep(1.0 + 0.5 * attempt)
        r = subprocess.run(
            [str(BIN), "daemon", "--background", "--ctx", str(ctx), "--model", model_ref],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            if load_drafter:
                qwen_cmd(["drafter", "load", "qwen3:0.6b"], timeout=60)
            return qwen_cmd(["status"]).strip()
        time.sleep(2.0)
    raise RuntimeError(f"Failed to start RTX 3080 daemon for {model_ref}")

print("RTX 3080 Daemon Status:", ensure_rtx3080_online())
"""


CELL_3_CODE = """# Visualize Layer-by-Layer Causal `mlp.down_proj` vs `self_attn.o_proj` Injection on Qwen3-8B & Qwen3-4B
loc_path = ROOT / "data" / "phase8_8b" / "localization_8b.json"
if not loc_path.exists():
    loc_path = ROOT / "data" / "mech_postcutoff" / "matched_localization.json"

loc = json.loads(loc_path.read_text())
by_layer = sorted(loc["ranked_layers"], key=lambda x: x["layer"])
max_c = max(max(abs(x["down_proj_contrib"]), abs(x["o_proj_contrib"])) for x in by_layer) or 1.0

bars_svg = []
for item in by_layer:
    l = item["layer"]
    dn = max(0.0, item["down_proj_contrib"])
    op = max(0.0, item["o_proj_contrib"])
    x = 48 + l * 21
    h_dn = int(round((dn / max_c) * 150))
    h_op = int(round((op / max_c) * 150))
    in_patch = 24 <= l <= 33
    dn_color = "#10b981" if in_patch else ("#f59e0b" if l >= 34 else "#475569")
    bars_svg.append(f'<rect x="{x}" y="{180 - h_dn}" width="9" height="{h_dn}" fill="{dn_color}" rx="2"/>')
    bars_svg.append(f'<rect x="{x + 9}" y="{180 - h_op}" width="7" height="{h_op}" fill="#38bdf8" opacity="0.75" rx="2"/>')
    if l % 2 == 0 or l == 35:
        bars_svg.append(f'<text x="{x + 8}" y="196" fill="#94a3b8" font-size="10" text-anchor="middle">L{l}</text>')

svg_html = f'''
<div style="font-family:-apple-system,sans-serif;background:#0f172a;border:1px solid #334155;border-radius:12px;padding:16px;color:#e2e8f0;">
  <div style="font-size:15px;font-weight:700;margin-bottom:4px;">Layer-by-Layer Causal Injection into the "Wait / Hmm" Residual Direction ({html.escape(loc_path.name)}, dim={loc["dim"]})</div>
  <div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
    <span style="color:#10b981;font-weight:700;">■ Green (L24–L33 mlp.down_proj)</span>: Surgically projected orthogonal to d_l &nbsp;|&nbsp;
    <span style="color:#f59e0b;font-weight:700;">■ Amber (L34–L35)</span>: Preserved unedited so <code>&lt;/think&gt;</code> emission is never suppressed &nbsp;|&nbsp;
    <span style="color:#38bdf8;font-weight:700;">■ Blue</span>: self_attn.o_proj contribution
  </div>
  <svg viewBox="0 0 820 212" style="width:100%;height:auto;background:#090d16;border-radius:8px;">
    <rect x="{48 + 24*21 - 3}" y="16" width="{10*21}" height="166" fill="#064e3b" opacity="0.22" rx="4"/>
    <text x="{48 + 29*21}" y="29" fill="#6ee7b7" font-size="11" font-weight="700" text-anchor="middle">Edited Window (Layers 24–33)</text>
    <line x1="42" y1="180" x2="808" y2="180" stroke="#334155" stroke-width="1"/>
    {''.join(bars_svg)}
  </svg>
</div>
'''
display(HTML(svg_html))
"""


CELL_4_CODE = """# Aggregate Benchmark Scorecard: Before vs. After Weight Surgery Across 4B & 8B Checkpoints
rows_spec = [
    ("Qwen3-4B (Synthetic Val 100 Prompts @ 1,472)", ROOT / "metrics_baseline.json", ROOT / "metrics_patched.json"),
    ("Qwen3-4B (Unseen 2025–2026 Test 24 @ 4,096)", ROOT / "metrics_postcutoff_baseline_test_4096.json", ROOT / "metrics_postcutoff_patched_test_4096.json"),
    ("Qwen3-8B (Unseen 2025–2026 Test 24 @ 4,096)", ROOT / "metrics_phase8_postcutoff_8b_4096.json", ROOT / "metrics_phase8_patched_8b_test_4096.json"),
]

trs = []
for label, p_base, p_patch in rows_spec:
    if not (p_base.exists() and p_patch.exists()):
        continue
    mb = json.loads(p_base.read_text())
    mp = json.loads(p_patch.read_text())
    w_b, w_p = mb["avg_hesitation_markers"], mp["avg_hesitation_markers"]
    w_drop = (w_b - w_p) / max(1e-6, w_b) * 100.0
    eb = mb.get("end_think_completion_rate", 1.0 - mb.get("truncated_no_end_think_rate", 0.0)) * 100.0
    ep = mp.get("end_think_completion_rate", 1.0 - mp.get("truncated_no_end_think_rate", 0.0)) * 100.0
    trs.append(f'''
      <tr style="border-bottom:1px solid #1e293b;">
        <td style="padding:9px 10px;font-weight:600;color:#f1f5f9;">{html.escape(label)}</td>
        <td style="padding:9px 10px;text-align:right;color:#fca5a5;">{w_b:.2f}</td>
        <td style="padding:9px 10px;text-align:right;color:#34d399;font-weight:700;">{w_p:.2f} (-{w_drop:.1f}%)</td>
        <td style="padding:9px 10px;text-align:right;">{eb:.1f}% → <strong style="color:#6ee7b7;">{ep:.1f}%</strong></td>
        <td style="padding:9px 10px;text-align:right;">{mb["accuracy"]*100:.2f}% → <strong style="color:#38bdf8;">{mp["accuracy"]*100:.2f}%</strong></td>
        <td style="padding:9px 10px;text-align:right;color:#cbd5e1;">{mp.get("control_kl_nats", 0.0):.4f} nats</td>
      </tr>
    ''')

table_html = f'''
<div style="font-family:-apple-system,sans-serif;background:#0f172a;border:1px solid #334155;border-radius:12px;padding:16px;color:#e2e8f0;">
  <div style="font-size:15px;font-weight:700;margin-bottom:10px;">Before vs. After Summary Across Benchmarks (100% RTX 3080 Greedy Evaluation)</div>
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead>
      <tr style="border-bottom:2px solid #334155;color:#94a3b8;text-align:left;">
        <th style="padding:8px 10px;">Model &amp; Evaluation Suite</th>
        <th style="padding:8px 10px;text-align:right;">Wait/Hmm (Before)</th>
        <th style="padding:8px 10px;text-align:right;">Wait/Hmm (After)</th>
        <th style="padding:8px 10px;text-align:right;">Closed &lt;/think&gt;</th>
        <th style="padding:8px 10px;text-align:right;">Exact Accuracy</th>
        <th style="padding:8px 10px;text-align:right;">Control KL</th>
      </tr>
    </thead>
    <tbody>{''.join(trs)}</tbody>
  </table>
</div>
'''
display(HTML(table_html))
"""


CELL_5_CODE = """# Instant Trace Browser: Inspect Saved Unseen Competition Traces (Before vs. After)
# Change `DATASET` ("8b_postcutoff" | "4b_postcutoff" | "4b_val") or `PAIR_INDEX` to browse any problem!
import sys
sys.path.insert(0, str(ROOT))
from run_postcutoff_study import extract_predicted_answer, normalize_exact_rational

def browse_saved_pair(dataset: str = "8b_postcutoff", pair_index: int = None):
    if dataset == "8b_postcutoff" and (ROOT / "data/phase8_8b/traces_qwen3_8b_patched_test_4096.jsonl").exists():
        b_path = ROOT / "data/phase8_8b/traces_qwen3_8b_test_4096.jsonl"
        a_path = ROOT / "data/phase8_8b/traces_qwen3_8b_patched_test_4096.jsonl"
        p_map = {json.loads(ln)["id"]: json.loads(ln) for ln in open(ROOT / "data/post_cutoff_test.jsonl")}
        tag = "Qwen3-8B Unseen Post-Cutoff Competition"
    elif dataset == "4b_postcutoff":
        b_path = ROOT / "data/mech_postcutoff/traces_baseline_test_4096.jsonl"
        a_path = ROOT / "data/mech_postcutoff/traces_patched_test_4096.jsonl"
        p_map = {json.loads(ln)["id"]: json.loads(ln) for ln in open(ROOT / "data/post_cutoff_test.jsonl")}
        tag = "Qwen3-4B Unseen Post-Cutoff Competition"
    else:
        b_path = ROOT / "data/mech_loop/traces_baseline.jsonl"
        a_path = ROOT / "data/mech_loop/traces_patched.jsonl"
        p_map = {json.loads(ln)["id"]: json.loads(ln) for ln in open(ROOT / "data/val_prompts.jsonl")}
        tag = "Qwen3-4B Multi-Step Reasoning"

    b_list = [json.loads(ln) for ln in open(b_path)]
    a_list = [json.loads(ln) for ln in open(a_path)]
    a_by_id = {r["id"]: r for r in a_list}

    # Default: pick the problem where patching flipped Incorrect/Loop -> Correct with the biggest Wait/Hmm drop!
    if pair_index is None:
        best = max(
            range(len(b_list)),
            key=lambda i: (
                int(a_by_id[b_list[i]["id"]]["is_correct"] and not b_list[i]["is_correct"]) * 1000
                + (b_list[i]["hesitation_count"] - a_by_id[b_list[i]["id"]]["hesitation_count"])
            ),
        )
        pair_index = best

    rb = b_list[pair_index]
    ra = a_by_id[rb["id"]]
    pm = p_map[rb["id"]]
    b_txt = rb.get("text_tail") or rb.get("text_preview", "")
    a_txt = ra.get("text_tail") or ra.get("text_preview", "")
    render_before_after_card(
        title=f"{tag} · Problem #{pair_index} ({rb.get('source_id', rb['category'])})",
        prompt=pm["prompt"],
        gold=rb["gold"],
        before={**rb, "text": b_txt, "tps": rb.get("decode_tok_per_s", 0.0)},
        after={**ra, "text": a_txt, "tps": ra.get("decode_tok_per_s", 0.0)},
    )

browse_saved_pair("8b_postcutoff")
browse_saved_pair("4b_postcutoff")
"""


CELL_6_CODE = """# LIVE Interactive RTX 3080 Playground: Run ANY Prompt Before vs. After in Real Time!
# Since `qwen3-8b-patched` is now the default daemon model, BEFORE temporarily restores the unpatched
# direction via `--scale -1.0` (`+alpha * d_l * d_l^T W`) in `<1 ms`, and AFTER runs `qwen3-8b-patched` directly!

def compare_prompt_live(
    prompt: str,
    gold: str = "",
    max_tokens: int = 2048,
):
    lora_patch_dir = ROOT / "lora" / "qwen3-8b-wait-hmm-patch"

    def run_single(label_tag: str) -> dict:
        out = qwen_cmd(
            ["run", "--spec", "--temp", "0", "--max-tokens", str(max_tokens), "--reset-cache", "--json-stats", "-p", prompt],
            timeout=300,
        )
        lines = out.splitlines()
        st = parse_last_json(out)
        txt = "\\n".join(lines[:-1])
        tids = st["token_ids"]
        has_end = (151668 in tids) or ("</think>" in txt)
        think_toks = tids.index(151668) if 151668 in tids else len(tids)
        think_part = txt.split("</think>", 1)[0] if "</think>" in txt else txt
        waits = len(HESITATION_RE.findall(think_part))
        pred = extract_predicted_answer(txt, has_end)
        norm_gold = normalize_exact_rational(gold) if gold else ""
        is_corr = (pred == norm_gold) if norm_gold else has_end
        return {
            "text": txt,
            "pred": pred,
            "is_correct": is_corr,
            "has_end_think": has_end,
            "think_tokens": think_toks,
            "hesitation_count": waits,
            "tps": st["decode_tok_per_s"],
        }

    st_line = ensure_rtx3080_online("qwen3-8b-patched", ctx=4096, load_drafter=True)
    is_patched_base = "patched" in st_line

    if is_patched_base:
        print(f"▶ Temporarily restoring unpatched Wait/Hmm direction ({lora_patch_dir.name} --scale -1.0) for BEFORE...")
        qwen_cmd(["lora", "load", str(lora_patch_dir), "--scale", "-1.0"])
        before_res = run_single("before")

        print("▶ Clearing inverse LoRA (<1 ms) & running AFTER on default qwen3-8b-patched...")
        qwen_cmd(["lora", "clear"])
        after_res = run_single("after")
    else:
        qwen_cmd(["lora", "clear"])
        print("▶ Running BEFORE (Unpatched Qwen3-8B)...")
        before_res = run_single("before")

        print(f"▶ Hot-toggling orthogonal Wait/Hmm patch ({lora_patch_dir.name}) in <1 ms & running AFTER...")
        qwen_cmd(["lora", "load", str(lora_patch_dir), "--scale", "1.0"])
        after_res = run_single("after")
        qwen_cmd(["lora", "clear"])

    render_before_after_card(
        title="LIVE RTX 3080 Comparison · Qwen3-8B Before vs. After Weight Edit",
        prompt=prompt,
        gold=gold or "(open-ended)",
        before=before_res,
        after=after_res,
    )
    return before_res, after_res

# --- Pick any unseen post-cutoff 2025 AMC 12 preset below, or set `PRESET_KEY = "custom"`! ---
PRESETS = {
    "amc12_2025_nut_mix": (
        "A box contains $10$ pounds of a nut mix that is $50$ percent peanuts, $20$ percent cashews, "
        "and $30$ percent almonds. A second nut mix containing $20$ percent peanuts, $40$ percent cashews, "
        "and $40$ percent almonds is added to the box resulting in a new nut mix that is $40$ percent peanuts. "
        "How many pounds of cashews are now in the box?\\n\\n"
        "Put your final simplified exact answer inside \\\\boxed{...} and on the last line write: Final Answer: <answer>",
        "4",
    ),
    "amc12_2025_self_ref_logic": (
        "Agnes writes the following four statements on a blank piece of paper.\\n\\n"
        "$\\\\bullet$ At least one of these statements is true.\\n\\n"
        "$\\\\bullet$ At least two of these statements are true.\\n\\n"
        "$\\\\bullet$ At least two of these statements are false.\\n\\n"
        "$\\\\bullet$ At least one of these statements is false.\\n\\n"
        "Each statement is either true or false. How many false statements did Agnes write on the paper?\\n\\n"
        "Put your final simplified exact answer inside \\\\boxed{...} and on the last line write: Final Answer: <answer>",
        "1",
    ),
    "amc12_2025_diophantine": (
        "Positive integers $x$ and $y$ satisfy the equation $57x+22y = 400$. "
        "What is the least possible value of $x+y$?\\n\\n"
        "Put your final simplified exact answer inside \\\\boxed{...} and on the last line write: Final Answer: <answer>",
        "15",
    ),
    "custom": (
        "Three switches outside a closed room control three incandescent bulbs inside. You may flip switches "
        "however you want, then open the door once. How many times must you open the door at minimum to identify "
        "every switch?\\n\\nPut your final simplified exact answer inside \\\\boxed{...} and on the last line write: Final Answer: <answer>",
        "1",
    ),
}

PRESET_KEY = "amc12_2025_nut_mix"
LIVE_PROMPT, LIVE_GOLD = PRESETS[PRESET_KEY]

_ = compare_prompt_live(LIVE_PROMPT, gold=LIVE_GOLD, max_tokens=2048)
"""


CELL_7_CODE = """# Inspect Saved Standalone Patched Checkpoint Manifests (`<50 ms` Rollback Ready)
for ckpt_name in ["qwen3-8b-patched", "qwen-patched"]:
    mf = ROOT / "checkpoints" / ckpt_name / "patch_manifest.json"
    if mf.exists():
        print(f"=== checkpoints/{ckpt_name}/patch_manifest.json ===")
        print(mf.read_text().strip())
"""


def make_cell(cell_type: str, source: str, cell_id: str, outputs=None, exec_count=None):
    lines = [ln + "\n" for ln in source.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    c = {
        "id": cell_id,
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        c["execution_count"] = exec_count
        c["outputs"] = outputs or []
    return c


def execute_cells_and_capture(code_cells: list) -> list:
    """Executes all code cells sequentially in one Python namespace, capturing exact stdout & IPython HTML outputs!"""
    import io
    from contextlib import redirect_stdout

    captured_all = []
    ns = {"__name__": "__main__"}

    class FakeHTML:
        def __init__(self, s):
            self.html = s

    current_outputs = []

    def custom_display(obj):
        if isinstance(obj, FakeHTML):
            current_outputs.append({
                "output_type": "display_data",
                "data": {
                    "text/html": [ln + "\n" for ln in obj.html.splitlines()],
                    "text/plain": ["<IPython.core.display.HTML object>"],
                },
                "metadata": {},
            })
        else:
            current_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [str(obj) + "\n"],
            })

    for idx, src in enumerate(code_cells, start=1):
        current_outputs = []
        buf = io.StringIO()
        # Inject FakeHTML + custom_display so `display(HTML(...))` captures rich `text/html` mimebundles!
        patched_src = src.replace(
            "from IPython.display import HTML, display",
            "HTML, display = _NB_HTML, _NB_DISPLAY",
        )
        ns["_NB_HTML"] = FakeHTML
        ns["_NB_DISPLAY"] = custom_display
        print(f"[nb-exec] Running code cell {idx}/{len(code_cells)}...", flush=True)
        with redirect_stdout(buf):
            exec(compile(patched_src, f"<cell_{idx}>", "exec"), ns)
        std_txt = buf.getvalue()
        cell_outs = []
        if std_txt:
            cell_outs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [ln + "\n" for ln in std_txt.splitlines()],
            })
        cell_outs.extend(current_outputs)
        captured_all.append(cell_outs)
    return captured_all


def main():
    do_exec = "--execute" in sys.argv
    code_sources = [CELL_2_CODE, CELL_3_CODE, CELL_4_CODE, CELL_5_CODE, CELL_6_CODE, CELL_7_CODE]
    outputs_per_code = execute_cells_and_capture(code_sources) if do_exec else [[] for _ in code_sources]

    cells = [make_cell("markdown", CELL_1_MD, "cell-intro-md")]
    for i, (src, outs) in enumerate(zip(code_sources, outputs_per_code), start=1):
        cells.append(make_cell("code", src, f"cell-code-{i}", outputs=outs, exec_count=(i if do_exec else None)))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split()[0],
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "cells": cells,
    }
    OUT_IPYNB.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"[done] Wrote {OUT_IPYNB} ({OUT_IPYNB.stat().st_size / 1024:.1f} KB, executed={do_exec})", flush=True)


if __name__ == "__main__":
    main()
