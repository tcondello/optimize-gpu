# `optimize-gpu` — 277 tok/s Qwen Text + 99.5% Recall@10 Same-Model RAG + 42.7x STT + 6.0x Streaming TTS (`26 ms` TTFA) + Fused LoRA on an Apple Silicon MacBook Pro + NVIDIA RTX 3080 Thunderbolt eGPU

**Zero Docker. Zero Python. Zero external Rust crates. Zero model modifications. 100% of neural, DSP, embedding, & vector search inference on the RTX 3080.**

A native `aarch64-apple-darwin` Rust GPU runtime, persistent co-resident daemon, TrueColor Voice+Text+RAG Terminal UI ([`qwen-tui`](./qwen-tui)), CLI speech recognizer ([`qwen-listen`](./qwen-listen)), CLI speech synthesizer ([`qwen-speak`](./qwen-speak)), and custom **NVIDIA Ampere (`GA102`, `sm_86`, 68 SMs) persistent CUDA megakernel suite** running **co-resident inside 10 GB GDDR6X (`8.78 GB` used, `0 ms` swap)**:
1. **`Qwen3-4B-Instruct-2507` (`Q4_K_M`, `2.50 GB`)** from `8.51 tok/s` to **`167.3 – 277.5 tok/s`** (`20.5x – 32.6x` speedup, `< 1 ms` warm daemon ping),
2. **Fused Dynamic Rank-$r$ (`r \le 32`) PEFT LoRA** hot-swapping onto the base Thinker in **`5.1 ms` cold DMA / `44 µs` cached pointer-flip** (`0.0%` overhead when inactive, `96` registers preserved),
3. **Co-Resident 100%-on-RTX-3080 `Qwen3-TTS-12Hz-1.7B` (`Q8_0`, `2.68 GB` w/ `BF16` `Code2Wav`)** streaming `24 kHz` speech at **`6.05x` real-time (`RTF = 0.165`)** with **`26.0 – 39.7 ms` Time-To-First-Audio (`43.6x – 144.9x` faster TTFA than M3 Pro Metal)**,
4. **Co-Resident 100%-on-RTX-3080 `Qwen3-ASR-1.7B` (`Q8_0`, `2.75 GB`, `1.62%` LibriSpeech WER)** transcribing `16 kHz` speech (including GPU Hann STFT + Slaney Mel + `NHWC` `Conv2d` + `24L` `AuT` Encoder + `28L` `Qwen3` Megakernel) at **`36.1x – 42.7x` real-time (`2.63x – 3.14x` faster than M3 Pro Metal)**, and
5. **Same-Model RTX 3080 Text Embeddings & `.qvec` `BF16` GPU VectorDB (`130.9 MB` total VRAM, `0` standalone embedding model)** achieving **`82.0%` `Recall@1` / `97.5%` `Recall@5` / `99.5%` `Recall@10` (`0.8877` `MRR@10`)** on `1,000` Wikipedia passages (`99.5%` of the standalone `4.08 GB` `Qwen3-Embedding-4B-Q8_0` oracle in **`31.2x` less VRAM**) with **`308 µs` GPU cosine search**.

---

## Verified Hardware Benchmark (`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, 2.50 GB)

**System Under Test**: Apple MacBook Pro (`M3 Pro`, macOS Darwin `25.6.0`) $\leftrightarrow$ Thunderbolt 3 (`x4 PCIe`, `~2.3 GB/s` DMA) $\leftrightarrow$ Razer Core X Chroma $\leftrightarrow$ **NVIDIA GeForce RTX 3080 (`GA102`, `sm_86`, 68 SMs, 10 GB GDDR6X @ `760.3 GB/s`)**.

All Phase 2 numbers were measured with **Docker Desktop completely shut down (`0` containers, `0` processes)** via [`benchmark_phase2.py`](./benchmark_phase2.py) (raw output in [`benchmark_phase2_results.json`](./benchmark_phase2_results.json)):

| Stack / Backend | Docker at Runtime | Warm Ready Latency | GPU Launches / 128 Toks | Raw `T=1` Decode | Code / Config (`BM=4` Lossless Spec) | Effective GDDR6X BW | Speedup vs Stock NV |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock `tinygrad` (`DEV=METAL`, M3 Pro) | No | — | ~48,000 | 3.65 tok/s (`274.3 ms`) | 3.65 tok/s | 9.5 GB/s | 0.43x |
| Stock `tinygrad` (`DEV=NV`, RTX 3080) | Yes (`nvcc`) | ~12.4 s | ~65,000 | 8.51 tok/s (`117.5 ms`) | 8.51 tok/s | 22.1 GB/s | 1.00x |
| **Phase 1**: Python + `HCQGraph` ([`qwen_engine.py`](./qwen_engine.py)) | Yes (`nvrtc`) | 10.49 s | 44,288 (`346/tok`) | 117.71 tok/s (`8.50 ms`) | 108.93 tok/s | 294.0 GB/s (`38.7%`) | 13.8x |
| **Phase 2**: Native Rust + `sm_86` Megakernel ([`qwen-nv`](./qwen-nv-rs)) | **NO (`0` containers)** | **5.97 ms CLI (`~1.9 ms` IPC)** | **2 (`1` prefill + `1` megakernel)** | **172.83 tok/s (`5.79 ms`, `174.2` e2e)** | **260.14 tok/s (`3.84 ms`, `4/4` exact)** | **431.6 GB/s (`56.8%` `T=1`) / `649.6 GB/s` eff.** | **20.5x – 30.6x** |

> **100% Pure Serving Layer**: The model weights (`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, `2,497,281,120` bytes, SHA-256 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`) are opened strictly read-only (`PROT_READ` `mmap`). Zero layers are pruned, zero weights are re-quantized, all 36 layers + FP16 KV cache run in full precision, and `BM=4` speculative verification is 100% mathematically lossless (`argmax` bit-identical to `T=1`).

---

## Phase 3 Verified Benchmark: Co-Resident `Qwen3-TTS-12Hz-1.7B` (`Q8_0`) + Fused Rank-16 LoRA

Measured with **Docker Desktop completely OFF** via [`benchmark_phase3_tts.py`](./benchmark_phase3_tts.py) (raw telemetry in [`benchmark_phase3_results.json`](./benchmark_phase3_results.json)). Both engines ran on the **exact same GGUF files** (`qwen-talker-1.7b-customvoice-Q8_0.gguf` `1.95 GB` + `qwen-tokenizer-12hz-BF16.gguf` `342 MB`). Every neural layer (`28L` Talker + `5L x 15-step` `CodePredictor` + Split-RVQ + `8L` Sliding-Window Transformer + `2x` ConvNeXt + `4-stage` `SnakeBeta` `Code2Wav`) runs **100% on the RTX 3080** (`0.999990` cosine similarity / `46.80 dB` SNR vs `BF16` reference; `0.000000` streaming-vs-batch RMSE):

| Utterance (`24 kHz` Mono) | Backend | 1st Frame Codes | Playable TTFA | AR (`Talker` + `15x CP`) | `Code2Wav` | Total Time | `RTF` | Speed vs Real-Time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Short** (`~2.8s`) | Apple M3 Pro (`qwentts.cpp` `MTL0`) | 60.6 ms | 1,135.1 ms | 31.90 ms/f | 111.3 ms | 1,135.1 ms | 0.458 | 2.18x |
| **Short** (`~2.8s`) | **RTX 3080 (`qwen-nv` `sm_86`)** | **19.9 ms** | **26.0 ms (`43.6x`)** | **11.05 ms/f (`2.89x`)** | **73.1 ms** | **505.7 ms** | **0.167** | **5.99x (`2.74x`)** |
| **Medium** (`~6.0s`) | Apple M3 Pro (`qwentts.cpp` `MTL0`) | 61.6 ms | 2,863.1 ms | 31.77 ms/f | 265.3 ms | 2,863.1 ms | 0.443 | 2.26x |
| **Medium** (`~6.0s`) | **RTX 3080 (`qwen-nv` `sm_86`)** | **24.6 ms** | **30.7 ms (`93.2x`)** | **11.09 ms/f (`2.86x`)** | **131.2 ms** | **925.6 ms** | **0.165** | **6.05x (`2.68x`)** |
| **Long** (`~13.1s`) | Apple M3 Pro (`qwentts.cpp` `MTL0`) | 78.1 ms | 5,746.3 ms | 31.41 ms/f | 515.7 ms | 5,746.3 ms | 0.439 | 2.28x |
| **Long** (`~13.1s`) | **RTX 3080 (`qwen-nv` `sm_86`)** | **33.5 ms** | **39.7 ms (`144.9x`)** | **11.18 ms/f (`2.81x`)** | **303.3 ms** | **2,163.2 ms** | **0.165** | **6.05x (`2.66x`)** |

- **Co-Resident Live Voice Turn (`Qwen3-4B` $\rightarrow$ `Qwen3-TTS-1.7B`, `0 ms` Model Swap)**: `52`-token answer generated in `351.3 ms` + `36.2 ms` first-sentence TTS TTFA = **`387.6 ms` prompt-to-spoken-voice** while `Qwen3-4B` maintains **`268.7 tok/s`** (`BM=4`) / **`164.8–175.0 tok/s`** (`T=1`).
- **Fused Dynamic Rank-16 PEFT LoRA**: `5.07 ms` PCIe DMA hot-swap (`144` FP16 matrices across 36 layers), **`44 µs` clear**, `148.8 tok/s` (`T=1`) / `188.4 tok/s` (`BM=4`) when active, `128/128` bit-exact token parity at `scale=0` and after `clear`.

---

## Phase 4 Verified Benchmark: Co-Resident `Qwen3-ASR-1.7B` (`Q8_0`) Speech-to-Text + Full Voice TUI Loop

Measured with **Docker Desktop completely OFF** via [`benchmark_phase4_voice.py`](./benchmark_phase4_voice.py) (raw telemetry in [`benchmark_phase4_results.json`](./benchmark_phase4_results.json)). Both engines ran on the **exact same `Qwen3-ASR-1.7B-Q8_0.gguf` (`2.19 GB`)** file. Every single stage—including the `16 kHz` reflect-padded 400-pt Hann STFT, 128-bin `fp64` Slaney Mel filterbank, `3x` `NHWC` `float4` `Conv2d` subsampler, `24`-layer Bidirectional Flash-Attention `AuT` Encoder, and `28`-layer `Qwen3` persistent decode megakernel (`~265 tok/s`)—executes **100% on the RTX 3080**:

| Speech Clip (`16 kHz` Mono) | Backend | Mel + Subsample | 24L `AuT` Encoder | 28L `Qwen3` Decoder | Total Time | `RTF` | Speed vs Real-Time | Exact Transcript |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Short Command** (`4.64s` EN) | Apple M3 Pro (`transcribe.cpp` `MTL0`) | 0.95 ms (CPU) | 36.49 ms | 366.94 ms | 404.40 ms | 0.0872 | 11.47x | — |
| **Short Command** (`4.64s` EN) | **RTX 3080 (`qwen-nv` `sm_86`)** | **17.68 ms** | **9.13 ms (`4.00x`)** | **101.79 ms (`3.60x`)** | **128.61 ms** | **0.0277** | **36.08x (`3.14x`)** | **100% (`en`)** |
| **Tech Question** (`5.52s` EN) | Apple M3 Pro (`transcribe.cpp` `MTL0`) | 1.98 ms (CPU) | 45.06 ms | 391.71 ms | 438.70 ms | 0.0795 | 12.58x | — |
| **Tech Question** (`5.52s` EN) | **RTX 3080 (`qwen-nv` `sm_86`)** | **20.27 ms** | **10.35 ms (`4.35x`)** | **121.43 ms (`3.23x`)** | **152.05 ms** | **0.0275** | **36.30x (`2.89x`)** | **100% (`en`)** |
| **FLEURS English** (`10.56s` EN) | Apple M3 Pro (`transcribe.cpp` `MTL0`) | 2.00 ms (CPU) | 59.48 ms | 607.98 ms | 669.50 ms | 0.0634 | 15.77x | — |
| **FLEURS English** (`10.56s` EN) | **RTX 3080 (`qwen-nv` `sm_86`)** | **38.83 ms** | **18.36 ms (`3.24x`)** | **197.16 ms (`3.08x`)** | **254.37 ms** | **0.0241** | **41.51x (`2.63x`)** | **100% (`en`)** |
| **JFK Speech** (`11.00s` EN) | Apple M3 Pro (`transcribe.cpp` `MTL0`) | 4.53 ms (CPU) | 66.38 ms | 683.39 ms | 754.30 ms | 0.0686 | 14.58x | — |
| **JFK Speech** (`11.00s` EN) | **RTX 3080 (`qwen-nv` `sm_86`)** | **38.25 ms** | **19.01 ms (`3.49x`)** | **200.32 ms (`3.41x`)** | **257.59 ms** | **0.0234** | **42.70x (`2.93x`)** | **100% (`en`)** |
| **FLEURS Spanish** (`9.72s` ES) | Apple M3 Pro (`transcribe.cpp` `MTL0`) | 5.74 ms (CPU) | 62.89 ms | 673.92 ms | 742.60 ms | 0.0764 | 13.09x | — |
| **FLEURS Spanish** (`9.72s` ES) | **RTX 3080 (`qwen-nv` `sm_86`)** | **34.62 ms** | **16.81 ms (`3.74x`)** | **198.96 ms (`3.39x`)** | **250.40 ms** | **0.0258** | **38.82x (`2.97x`)** | **100% (`es`)** |

- **3-Model Co-Resident Full Voice Loop (`Speech-In -> Qwen3-4B -> Speech-Out`, `8.65 GB / 10.24 GB` GDDR6X, `0 ms` Model Swap)**:
  - `134.2 ms` `Qwen3-ASR-1.7B` STT + `275.3 ms` `Qwen3-4B` (`40` tokens @ `165.4 tok/s`) + `32.7 ms` `Qwen3-TTS-1.7B` TTFA = **`442.1 ms` total end-of-speech to first spoken AI audio** (`510.6 ms` on a 47-token technical answer), while `Qwen3-4B` sustains **`277.48 tok/s`** (`BM=4`) / **`167.34 tok/s`** (`T=1`).

---

## Phase 5 Verified Benchmark: Same-Model RTX 3080 Embeddings, `.qvec` GPU VectorDB, & Spoken Wikipedia RAG

Measured with **Docker Desktop completely OFF** and **`0` `.cu` kernel edits** via [`benchmark_phase5_rag.py`](./benchmark_phase5_rag.py) (raw telemetry in [`benchmark_phase5_results.json`](./benchmark_phase5_results.json)) over [`data/wikipedia_1k.jsonl`](./data/wikipedia_1k.jsonl) (`1,000` Wikipedia passages) and [`data/wikipedia_eval_200.jsonl`](./data/wikipedia_eval_200.jsonl) (`200` Q&A queries). Per [`AGENTS.md`](./AGENTS.md), **zero standalone embedding models are deployed alongside the Thinker** — embeddings are extracted directly on the loaded `Qwen3-4B` backbone while `Qwen3-ASR-1.7B` + `Qwen3-TTS-1.7B` remain co-resident in VRAM:

| Embedding Strategy | Backbone in VRAM | Extra VRAM (`Adapter + .qvec`) | `Recall@1` | `Recall@5` | `Recall@10` | `MRR@10` | Query Embed | GPU `BF16` Cosine |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_eos` (Control) | `Qwen3-4B` (`Q4_K_M`) | **`4.88 MB`** (`0 B` weights) | `0.00%` | `0.00%` | `0.50%` | `0.0006` | `24.67 ms` | `307.9 µs` |
| `prompt_eol` + GPU Center | `Qwen3-4B` (`Q4_K_M`) | **`4.89 MB`** (`0 B` weights) | `1.00%` | `8.00%` | `13.50%` | `0.0433` | `35.29 ms` | `307.0 µs` |
| `echo` (Copy-#2 + GPU Center) | `Qwen3-4B` (`Q4_K_M`) | **`4.89 MB`** (`0 B` weights) | `25.00%` | `48.50%` | `60.50%` | `0.3501` | `45.91 ms` | `307.6 µs` |
| `svd_lora` (`r=32`, Attn-Only) | `Qwen3-4B` (`Q4_K_M`) | `49.92 MB` (`45.04 MB` LoRA) | `4.50%` | `10.00%` | `13.00%` | `0.0707` | `67.72 ms` | `307.9 µs` |
| **`svd_lora` (`r=32`, All 7 Projs)** | **`Qwen3-4B` (`Q4_K_M`)** | **`130.94 MB` (`126.06 MB` LoRA)** | **`82.00%`** | **`97.50%`** | **`99.50%`** | **`0.8877`** | **`102.82 ms`** | **`308.2 µs`** |
| *Standalone Oracle* (`Qwen3-Embedding-4B-Q8_0`) | *Separate `4.08 GB` Model* | *`4,081 MB` (OOMs w/ Voice)* | *`85.00%`* | *`98.50%`* | *`100.00%`* | *`0.9125`* | *`55.45 ms`* | *`307.7 µs`* |

- **Why All-7-Projection SVD (`82.0%` `Recall@1`) Beats Attention-Only SVD (`4.5%` `Recall@1`) by `18.2x`**: Contrastive bi-encoder training in same-ancestor `Qwen/Qwen3-Embedding-4B` rotates the output representation space inside the SwiGLU `gate_proj`/`up_proj`/`down_proj` MLP matrices (`15.91%` Rank-32 Frobenius energy), collapsing causal rogue activations (`dim[7]: 0.3105 -> 0.0123`). Hot-toggling the cached `126 MB` 7-projection adapter takes **`44 µs` ON / `37 µs` OFF** and preserves **100% bit-exact `Qwen3-4B` generation**.
- **4-Stage Spoken Wikipedia Voice RAG (`Speech-In -> RTX 3080 STT -> RTX 3080 Wiki .qvec -> Qwen3-4B Cited Answer -> RTX 3080 TTS`, `8.78 GB` VRAM)**: `153.8 ms` STT + `109.8 ms` SVD-LoRA embed + `310.2 µs` GPU cosine search + `592.6 ms` 469-token grounded prefill + `39.9 ms` TTS TTFA = **`896.2 ms` spoken question to spoken cited Wikipedia answer**.

---

## Phase 6 Verified Benchmark: Autonomous Zero-Retraining Weight Editing (`Qwen3-4B-Thinking-2507`) + Co-Resident `Qwen3-0.6B` Lossless Speculative Drafter

Executed 100% natively on the RTX 3080 (`0` PyTorch, `0` Docker, `0` external crates) via [`run_autonomous_weight_edit.py`](./run_autonomous_weight_edit.py) over [`data/val_prompts.jsonl`](./data/val_prompts.jsonl) (`100` deterministic reasoning problems; raw metrics in [`metrics_baseline.json`](./metrics_baseline.json), [`metrics_patched.json`](./metrics_patched.json), and [`metrics_phase5_speculative.json`](./metrics_phase5_speculative.json)):

- **Surgical Orthogonal `BF16 safetensors` $\rightarrow$ `Q8_0 GGUF` Patch (`Layers 24–33` `mlp.down_proj`, `25,600` DOFs = `0.000636%` of `4.02B` params)**:
  - **Macro Accuracy**: **`20.00%` $\rightarrow$ `38.00%` (`+90.0%` relative gain, zero retraining)** — `gsm_math` `95% -> 100%` (`-30.5%` `<think>` tokens), `algo_trace` `0% -> 50%`, `crt_deceptive` `5% -> 40%`.
  - **Overthinking Suppression & Control Fidelity**: `"Wait / Hmm"` self-doubt markers slashed by **`72.9%` (`8.98 -> 2.43` / trace)**, circular loop rate cut **`92.0% -> 54.0%`**, and **`100.0%` top-1 agreement (`mean_kl = 0.01053 nats`)** preserved on mid-sentence control tokens.
- **Co-Resident `Qwen3-0.6B-Q8_0` Neural Drafter + Fresh-Trace Calibrated Prior (`checkpoints/qwen-patched/`)**:
  - Draft acceptance rate $\alpha$ **improved** after patching (`0.6588 -> 0.6776`), reaching **`144.17 tok/s`** (`588.5 GB/s` effective `Q8_0` bandwidth = **`77.4%` of RTX 3080 GDDR6X peak**, `7,680 / 7,680` **100% bit-exact** token parity vs `T=1`).

---

## Architecture

```mermaid
flowchart LR
    subgraph Host["macOS ARM64 (M3 Pro — Zero Python, Zero Docker, Zero Crates)"]
        TUI["qwen-tui (TrueColor Terminal UI / OpenAI HTTP :8000)"]
        DMN["Persistent Rust GPU Daemon (daemon.rs)"]
        SHM["Apple DART Shared Memory (MAP_SYSMEM_FD)"]
        TG["TinyGPU.app DriverKit Bridge (tinygpu.sock)"]
        TUI <-->|"Unix Socket (~1.9 ms)"| DMN
        DMN <-->|"Zero-Doorbell Token Stream"| SHM
        DMN <-->|"BAR0 / BAR1 MMIO"| TG
    end

    subgraph eGPU["Thunderbolt 3 -> Razer Core X -> NVIDIA RTX 3080 (GA102, 68 SMs, 10GB GDDR6X)"]
        GSP["GA102 GSP-RM (Booted Once, Resident)"]
        VRAM["VRAM: 2.50 GB Raw GGUF Slabs + FP16 KV Cache + L2 Scratch"]
        MK["1x 68-SM Persistent Megakernel (512 thr/SM = 1,088 Warps)"]
        GSP --- VRAM
        VRAM <-->|"431.6 - 649.6 GB/s"| MK
    end

    SHM <-->|"PCIe Posted MWr (4B / token)"| MK
    TG <-->|"PCIe x4 Tunnel"| GSP
```

### How It Works Under the Hood

1. **Zero-Dependency Native ARM64 Rust `GA102` Driver ([`nv_hw.rs`](./qwen-nv-rs/src/nv_hw.rs), [`tinygpu_client.rs`](./qwen-nv-rs/src/tinygpu_client.rs), [`nv_defs.rs`](./qwen-nv-rs/src/nv_defs.rs))**:
   - Speaks directly to `/Applications/TinyGPU.app`'s DriverKit Unix socket (`SCM_RIGHTS` `MAP_SYSMEM_FD` + `BAR0`/`BAR1` MMIO) without `tinygrad`, Python, or CUDA runtime libraries.
   - Parses VBIOS `FWSEC` (`FRTS`), boots Falcon + SEC2 + **GA102 GSP-RM** via 3-level Radix3 page tables, configures Turing/Ampere `MMU v2` 5-level page tables (`2 MB` VRAM hugepages + `4 KB` snooped host pages), and dispatches `QMDv03_00` descriptors directly to `AMPERE_COMPUTE_B` (`0xc7c0`) over `GPFIFO`.
   - Embeds the precompiled `sm_86` SASS binary ([`qwen_sm86_megakernel.cubin`](./qwen-nv-rs/kernels/qwen_sm86_megakernel.cubin)) via `include_bytes!` so **zero runtime compiler or Docker container is ever needed**.

2. **68-SM Persistent Multi-Token Megakernel ([`qwen_sm86_megakernel.cu`](./qwen-nv-rs/kernels/qwen_sm86_megakernel.cu))**:
   - Collapses `346` kernel launches per token (`44,288` launches for 128 tokens) into **1 persistent 68-SM launch** that executes all 36 transformer layers (`RMSNorm+Q8_1` $\rightarrow$ `QKV` $\rightarrow$ `QK-RMSNorm+RoPE+FP16 KV` $\rightarrow$ `Online-Softmax Flash-GQA+Q8_1` $\rightarrow$ `O+Residual` $\rightarrow$ `FFN RMSNorm+Q8_1` $\rightarrow$ `Gate+Up` $\rightarrow$ `SwiGLU+Q8_1` $\rightarrow$ `Down+Residual`), final `RMSNorm`, in-register `151,936`-row `LM_Head` argmax/Gumbel-Max sampler, and multi-token loop in-place on GPU.
   - Runs `512` threads (`16` warps = `32` `Q4_K` half-warps per SM, `2,176` concurrent rows across 68 SMs) with `LDG.E.128.CONSTANT` (`__ldg`) 128-byte L1 read-only cache coalescing and hardware `__dp4a` INT8 dot products, sustaining **`431.6 GB/s`** (`56.8%` of RTX 3080's `760.3 GB/s` GDDR6X ceiling) over a `2.3 GB/s` Thunderbolt cable.
   - Streams generated token IDs directly to host `MAP_SYSMEM_FD` via posted 4-byte PCIe `st.global.wt.s32` writes with **zero host round-trips between tokens**.

3. **512-Thread SMEM-Staged `BM=4` Lossless Speculative Verification ([`engine.rs`](./qwen-nv-rs/src/engine.rs))**:
   - Drafts 3 lookahead tokens from prompt + conversation `4`-grams/`3`-grams on CPU (`0` VRAM cost) and verifies `[cur_tok, d0, d1, d2]` in a single `BM=4` megakernel pass that stages `4 x K` `Q8_1` activations in `29.2 KB` shared memory (`LDS.128` @ `19 TB/s`), reading each quantized weight block once for all 4 tokens and reaching **`260.14 tok/s`** on code/JSON workloads.

4. **Hardware & macOS Kernel Root-Cause Fixes**:
   - **macOS `AppleT8110DART` `[SPTM] VIOLATION_T8110_DART_INVALID_ERR_MASK` Kernel Panic Prevention**: Guarantees `NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER` (with Falcon hardware halt fallback) executes in `Drop for NvGpu` and signal handlers *before* `tinygpu.sock` closes and unmaps DART IOVAs.
   - **PCIe Spec §2.4.1 `BAR1` Write-Combining Flush**: Issues a 4-byte Non-Posted `MRd` on `BAR1` (`gpput_off`) prior to ringing `BAR0 + 0xbb0090`, preventing `BAR0` doorbells from overtaking Apple Silicon `BAR1` WC buffers across the Thunderbolt bridge.
   - **Thunderbolt eGPU PSU Transient Protection**: Suppresses `CMD_BOOST_TO_MAX` voltage spikes and uses `nanosleep.u32 16` inside the L2 grid barrier to smooth $di/dt$ current draw on SFX enclosures.

---

## Quick Start

### Prerequisites
- macOS on Apple Silicon + Thunderbolt 3/4 NVIDIA Ampere (`sm_86`) eGPU with [`TinyGPU.app`](https://github.com/tinygrad/tinygpu_releases) installed at `/Applications/TinyGPU.app`
- Rust toolchain (`cargo`) — **no Docker and no external Rust crates required**

### 1. Launch the Interactive TrueColor Voice + Text Rust TUI (`qwen-tui`)
```bash
./qwen-tui
```
Auto-builds `qwen-nv` on first run (~8 seconds), starts the persistent background GPU daemon if not already running, and opens the full-screen Terminal UI:
- **`Ctrl+R` or `/listen` (`/mic`)** — **Speak a question into your Mac microphone**: captures `16 kHz` audio with live VU meter & ~1s pause auto-VAD (or press `Enter` / `Ctrl+R` to send immediately), transcribes 100% on the RTX 3080 via `Qwen3-ASR-1.7B Q8_0`, logs `▸ YOU` and streams `◆ Qwen3-4B-Instruct-2507` live in the TUI chat window, and speaks the answer aloud at `24 kHz` via `Qwen3-TTS-1.7B Q8_0`!
- **`/talk [on|off]`** — **Hands-free continuous voice conversation** (automatically re-opens the microphone after each spoken reply; `Esc` exits)
- `/speak [on|off]` — Toggle live **100%-on-RTX-3080 `Qwen3-TTS-1.7B Q8_0`** spoken replies (`24 kHz` CoreAudio streaming)
- `/say [text]` — Speak `<text>` (or the last assistant reply) aloud immediately │ `/voice <name>` — Switch speaker (`ryan`, `serena`, `vivian`, `aiden`, `uncle_fu`, `ono_anna`, `sohee`, `eric`, `dylan`)
- **`/rag [wiki|<path.qvec>|off|status]`** — Load the `1,000`-passage Wikipedia GPU `.qvec` VectorDB (`📚 RAG:1000` badge) so both spoken (`Ctrl+R` / `/talk`) and typed questions automatically retrieve & cite top-3 Wikipedia articles in `308 µs` on the RTX 3080!
- `/lora <path|off>` — `< 6 ms` hot-swap (`44 µs` cached flip) a HuggingFace PEFT `.safetensors` Rank-$r$ (`r \le 32`) LoRA adapter onto `Qwen3-4B`
- `/bench` (or `/bench t1`) — Run the live `~277 tok/s` `BM=4` or `~173 tok/s` `T=1` benchmark inside chat
- `/verify` — Display loaded GGUF file path, exact byte size, & unmodified model architecture
- `Ctrl+S` (`/spec`) — Toggle lossless `BM=4` speculative verification vs pure `T=1` megakernel
- `Ctrl+Y` (`/copy` or `/copy code`) — Copy last reply or fenced code block to macOS clipboard
- `Esc` — Cancel active mic recording, GPU text stream, & speaker output in `< 1 ms` │ `Ctrl+L` (`/clear`) — Reset KV cache │ `Ctrl+Q` — Quit TUI

### 2. CLI Speech-to-Text (`qwen-listen`), Streaming TTS (`qwen-speak`), LoRA, & OpenAI HTTP Server (`qwen-nv`)
```bash
# Transcribe a WAV file or live Mac microphone 100% on the RTX 3080 (36x - 42.7x real-time)
./qwen-listen /path/to/question.wav
./qwen-listen --mic

# Full CLI voice round-trip (Mic -> Qwen3-ASR-1.7B -> Qwen3-4B -> Qwen3-TTS-1.7B voice reply!)
./qwen-listen --mic --ask

# Speak live through Mac speakers (26 ms TTFA, 6.0x real-time on RTX 3080) and optionally save WAV
./qwen-speak "Hello Tim, the RTX 3080 text to speech engine is online."
./qwen-speak --voice serena -o /tmp/serena.wav "Every neural layer runs natively on the external graphics card."

# Dynamic Rank-r (r <= 32) PEFT LoRA hot-swap (< 6 ms DMA load, 44 us cached flip / clear) & same-ancestor SVD extraction
./qwen-nv-rs/target/release/qwen-nv lora extract-svd qwen3-embed:4b ~/.cache/qwen-nv-rs/loras/qwen3_embed_svd_r32_all --rank 32
./qwen-nv-rs/target/release/qwen-nv lora load /path/to/peft_adapter_dir
./qwen-nv-rs/target/release/qwen-nv lora status
./qwen-nv-rs/target/release/qwen-nv lora clear

# 100%-on-RTX-3080 Same-Model Embeddings & Wikipedia .qvec VectorDB (82.0% Recall@1, 99.5% Recall@10, 308 us cosine search)
./qwen-nv-rs/target/release/qwen-nv rag load ./data/wikipedia_1k_best.qvec
./qwen-nv-rs/target/release/qwen-nv rag query "Which theoretical computer science field classifies computational problems by difficulty?"
./qwen-nv-rs/target/release/qwen-nv rag ask "How many miles south of Edinburgh is Newcastle upon Tyne?"
./qwen-nv-rs/target/release/qwen-nv embed --strategy svd_lora --lora ~/.cache/qwen-nv-rs/loras/qwen3_embed_svd_r32_all "Search query"

# Stream a text prompt from CLI (--spec enabled by default; pass --no-spec for pure T=1)
./qwen-nv-rs/target/release/qwen-nv run --json-stats -p "Explain CUDA warp shuffle in 2 short sentences."

# Serve OpenAI-compatible `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/speech`, & `/v1/audio/transcriptions` on port 8000
./qwen-nv-rs/target/release/qwen-nv serve --port 8000

# IMPORTANT: Always unload GSP-RM cleanly before unplugging the Thunderbolt cable!
./qwen-nv-rs/target/release/qwen-nv stop
```

### 3. Reproduce the Head-to-Head Benchmarks
```bash
python3 ./benchmark_phase2.py
python3 ./benchmark_phase3_tts.py
python3 ./benchmark_phase4_voice.py
python3 ./benchmark_phase5_rag.py
```

---

## Engineering FAQ (Common Questions & Hardware Verification)

### 1. *"Why didn't you just run `ollama`, `llama.cpp` CUDA, or `vLLM` instead of writing a custom Rust driver and megakernel?"*
Because **none of them can see an NVIDIA GPU on Apple Silicon macOS**:
- Apple removed NVIDIA drivers after macOS 10.13 (2017), and NVIDIA has **never** shipped an `aarch64-apple-darwin` kernel driver, `libcuda`, `cudart`, or `cuBLAS`. There is no `/dev/nvidia0` on macOS, and Docker Desktop on macOS runs inside `Virtualization.framework`, which cannot pass a PCIe/Thunderbolt eGPU (`VFIO`) into a Linux container.
- With SIP enabled, the **only** user-space path to an NVIDIA eGPU on Apple Silicon is [`TinyGPU.app`](https://github.com/tinygrad/tinygpu_releases)'s DriverKit PCIe tunnel (`BAR0`/`BAR1` MMIO + Apple `DART` IOMMU `MAP_SYSMEM_FD`), booting NVIDIA's open-source RISC-V `GSP-RM` firmware from user space.
- `8.51 tok/s` wasn't a "misconfiguration"—it is what stock upstream `tinygrad` (`DEV=NV`) gets out of the box on `Qwen3-4B-Q4_K_M`, because stock `tinygrad` dequantizes `Q4_K` tensors in VRAM on every token (`22.1 GB/s` effective) and dispatches ~500 individual kernels per token with PCIe `SYSMEMBAR` flushes across the Thunderbolt bridge.

### 2. *"Are you 'cheating' by offloading layers or compute onto the M3 Pro chip?"*
**Zero neural network compute runs on the M3 Pro.** Both physics and the code prove the RTX 3080 (`GA102`) does 100% of inference:
- **Memory-Bandwidth Physics**: Single-token (`T=1`) decode must read all `2.497 GB` of weights per token. An Apple M3 Pro (`150 GB/s` unified memory bus) has a hard physical ceiling of `150 / 2.497 = 60.0 tok/s` at an impossible 100% bus efficiency (`~45–52 tok/s` in practice via Metal `llama.cpp` / `MLX`). Even summing M3 Pro (`150 GB/s`) + Thunderbolt 3 (`2.3 GB/s`) cannot physically reach **`172.8 tok/s` (`431.6 GB/s` sustained weight bandwidth)**. Only the RTX 3080's 320-bit GDDR6X bus (`760.3 GB/s` physical peak) can move `431.6 GB/s`.
- **2-Launch Execution**: For a 128-token `T=1` generation (`--no-spec`), `qwen-nv` issues **2 GPU launches total** (`1` chained QMD prefill + `1` persistent 68-SM decode megakernel). All 36 layers, QK-RMSNorm, RoPE, FP16 KV-cache updates, Flash-GQA, SwiGLU, `151,936`-row `LM_Head`, and `argmax` sampling execute inside the RTX 3080, writing back **only one 4-byte `int32` token ID per token** (`512 bytes` total over Thunderbolt).
- Even in `--spec` (`BM=4`) mode, the M3 Pro CPU only performs a microsecond `HashMap` lookup on prior token integers (`0` weights, `0` draft model); 100% of verification runs on the RTX 3080.

### 3. *"How do `172.8 tok/s` and `260.1 tok/s` compare to bare-metal Linux CUDA—and why are there two numbers?"*
We report both modes separately so the roofline math is completely transparent:
- **`172.83 tok/s` (`--no-spec`, pure `T=1` decode on *any* prompt)**: Sustains **`431.6 GB/s` (`56.8%` of RTX 3080's `760.3 GB/s` GDDR6X peak)**—matching what bare-metal Linux `llama.cpp` CUDA achieves on PCIe Gen4 x16 (`~165–185 tok/s` on 4B `Q4_K_M`), despite running over a `2.3 GB/s` macOS Thunderbolt 3 cable, because the persistent megakernel eliminates per-kernel Thunderbolt dispatch latency.
- **`260.14 tok/s` (`--spec`, lossless `BM=4` speculative verification on code/JSON/refactoring; `171–189 tok/s` on general Q&A)**: Loads each `Q4_K` superblock from GDDR6X **once** into registers and multiplies it against **4 candidate token activations staged in `29.2 KB` Shared Memory (`LDS.128` @ `19 TB/s`)**, amortizing the GDDR6X memory wall across up to 4 tokens per pass while accepting **only** tokens that match exact greedy `T=1` `argmax` (`4/4` bit-identical token-ID parity verified in [`benchmark_phase2_results.json`](./benchmark_phase2_results.json)).

---

## Repository Structure

```text
optimize-gpu/
├── AGENTS.md                                   # Workspace engineering guardrails (Shared-Ancestor Rule, 100% RTX 3080, Zero Crates/Docker)
├── qwen-tui                                    # One-command launcher for the Voice + Text + Wiki RAG + LoRA Rust TUI (Ctrl+R, /rag wiki, /talk)
├── qwen-listen                                 # One-command CLI launcher for 100%-RTX-3080 Qwen3-ASR-1.7B STT (--mic, --ask)
├── qwen-speak                                  # One-command CLI launcher for 100%-RTX-3080 Qwen3-TTS-1.7B streaming TTS
├── data/
│   ├── wikipedia_1k.jsonl                      # 1,000 real Wikipedia passages across 48 articles
│   ├── wikipedia_eval_200.jsonl                # 200 ground-truth Q&A evaluation queries
│   ├── wikipedia_1k_8b_best.qvec               # 4096-D Qwen3-8B SVD LoRA GPU vector index (90.5% R@1 / 100.0% R@10)
│   ├── wikipedia_1k_best.qvec                  # 2560-D Qwen3-4B SVD LoRA GPU vector index (82.0% R@1 / 99.5% R@10)
│   ├── post_cutoff_calib.jsonl                 # 60 strictly post-Nov-2025 unseen calibration problems (AMC12, HMMT, AIME 2026, ArXiv 2026)
│   ├── post_cutoff_test.jsonl                  # 60 strictly post-Nov-2025 held-out test problems
│   └── val_prompts.jsonl                       # 100 deterministic verified reasoning prompts across 5 categories
├── checkpoints/
│   ├── qwen3-8b-patched/                       # Default Thinker: Qwen3-8B-Patched-Q6_K.gguf (L24-33 down_proj, a=0.85) + <50ms rollback
│   └── qwen-patched/                           # Phase 6 Qwen3-4B patch manifest & calibrated N-gram prior
├── lora/
│   ├── qwen3-8b-embed-svd-r32/                 # Same-ancestor Qwen3-Embedding-8B Rank-32 SVD LoRA (166.6 MB, 90.5% R@1)
│   └── qwen3-8b-wait-hmm-patch/                # 43us Rank-1 L24-33 orthogonal Wait/Hmm adapter (324 KB; --scale -1.0 restores unpatched)
├── wait_hmm_before_after_demo.ipynb            # Interactive local Jupyter notebook for live RTX 3080 Wait/Hmm Before-vs-After inspection
├── qwen-nv-rs/                                 # Zero-crate, zero-Docker native Rust + sm_86 engine (v0.7.0)
│   ├── Cargo.toml                              # 0 external crate dependencies
│   ├── kernels/
│   │   ├── qwen_sm86_megakernel.cu             # 68-SM T=1 + BM=4 + Rank-r LoRA + Qwen3-ASR + Qwen3-TTS + Code2Wav kernels
│   │   └── qwen_sm86_megakernel.cubin          # Precompiled sm_86 SASS binary embedded via include_bytes!
│   └── src/
│       ├── main.rs                             # CLI (tui, run, mech, drafter, embed, rag, transcribe/listen, speak, lora, serve, ...)
│       ├── mech_patch.rs                       # Causal fork localization, BF16 safetensors -> Q8_0 surgical patcher, KL gate, & <50ms rollback
│       ├── drafter.rs                          # Co-resident 100%-GPU Qwen3-0.6B neural speculative drafter & calibrated N-gram prior
│       ├── tui.rs                              # TrueColor TUI with live Ctrl+R mic, /rag wiki, /talk hands-free, /speak, & /lora
│       ├── rag.rs                              # 100%-GPU embeddings (raw_eos, prompt_eol, echo, svd_lora) & .qvec BF16 cosine VectorDB
│       ├── asr.rs                              # 100%-GPU Qwen3-ASR-1.7B STFT/Mel + NHWC Conv2d + 24L AuT + 28L Megakernel
│       ├── tts.rs                              # 100%-GPU Qwen3-TTS-1.7B Talker + 15-step CodePredictor + Code2Wav
│       ├── lora.rs                             # Zero-crate PEFT .safetensors parser, Accelerate SVD extractor, & 44us dynamic LoRA arena
│       ├── audio.rs                            # Zero-crate macOS AudioToolbox 16 kHz mic VAD recorder & 24 kHz speaker player
│       ├── daemon.rs                           # Co-resident Unix-socket daemon & OpenAI HTTP (/v1/chat/completions, /v1/embeddings, /v1/audio/*)
│       ├── engine.rs                           # VRAM slab loader, KV prefix cache, residual hooks, logit export, & rollout loop
│       ├── gguf.rs                             # Read-only mmap GGUF v2/v3 parser, BF16/F32/Quant & BPE tokenizer
│       ├── nv_hw.rs                            # Native GA102 GSP-RM booter, MMU v2 page tables, & QMDv3 launcher
│       ├── nv_defs.rs                          # Ampere hardware constants, PTE/PDE encoders, & QMDv03_00 packer
│       └── tinygpu_client.rs                   # Direct Unix socket + SCM_RIGHTS MAP_SYSMEM_FD DriverKit client
├── run_autonomous_weight_edit.py               # Autonomous 5-phase mechanistic weight-editing + speculative drafter loop
├── metrics_baseline.json                       # Phase 1 baseline Qwen3-4B-Thinking-2507 metrics
├── metrics_patched.json                        # Phase 4 verified patched checkpoint metrics & KL control gate
├── metrics_phase5_speculative.json             # Phase 5 4-way lossless speculative drafter acceptance & throughput telemetry
├── benchmark_phase5_rag.py                     # Phase 5 benchmark: 6-way Wikipedia RAG retrieval + 4-stage Spoken Voice RAG E2E
├── benchmark_phase5_results.json               # Verified Phase 5 retrieval accuracy, VRAM, & latency telemetry
├── benchmark_phase4_voice.py                   # Phase 4 benchmark: RTX 3080 vs M3 Pro Metal STT + 3-Model Voice Loop E2E
├── benchmark_phase4_results.json               # Verified Phase 4 telemetry & exact transcript parity metrics
├── benchmark_phase3_tts.py                     # Phase 3 benchmark: RTX 3080 vs M3 Pro Metal TTS + LoRA + Voice E2E
├── benchmark_phase3_results.json               # Verified Phase 3 telemetry & golden waveform parity metrics
├── benchmark_phase2.py                         # Phase 2 benchmark: Native Rust Megakernel vs Phase 1 & Stock
└── benchmark_phase2_results.json               # Verified Phase 2 telemetry & token-parity output
```
