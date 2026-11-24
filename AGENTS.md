# Workspace Architecture & Engineering Rules (`optimize-gpu`)

## 1. Shared-Ancestor Thinker & Embedding Model Rule
- Whenever text embeddings or RAG retrieval are added to this stack, **the Thinker (generative/reasoning LLM) and the official Embedding model MUST share the exact same base model ancestor** (matching layer count, hidden dimension, GQA head layout, and tokenizer vocabulary — e.g., `meta-llama/Llama-3.1-8B` powering both `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` and `nvidia/Llama-Embed-Nemotron-8B`).
- **Never deploy a separate standalone embedding model** in VRAM alongside the Thinker. All embedding extraction must execute on the loaded Thinker backbone via either:
  1. Zero-weight causal pooling (`PromptEOL` / `Echo Embeddings` over `ws_hidden_last`), or
  2. Hot-toggling a low-rank adapter (`LoRA` / truncated SVD weight delta from the official same-ancestor embedding checkpoint) inside the existing dynamic LoRA arena.

## 2. 100% RTX 3080 Inference Guardrail
- **100.0% of neural network and audio DSP compute** (tokenization-to-waveform TTS, waveform-to-mel-to-text STT, LLM prefill/decode, LoRA, and RAG embeddings) **must execute on the NVIDIA GeForce RTX 3080 (`GA102`, `sm_86`, 68 SMs, 10 GB GDDR6X)** over Thunderbolt.
- The Apple Silicon host (`M3 Pro`) may only perform raw CoreAudio microphone/speaker I/O, terminal UI rendering, lightweight prompt/token plumbing, and isolated head-to-head Metal baseline benchmarks.

## 3. Zero Runtime Docker & Zero External Rust Crates
- `[dependencies]` in `qwen-nv-rs/Cargo.toml` must remain completely empty (`0` external crates).
- Precompiled `sm_86` CUDA binaries (`.cubin`) must be embedded via `include_bytes!`. Docker Desktop must remain completely OFF (`0` containers/VMs) during all runtime execution and benchmarks.
- Always cleanly unload `GSP-RM` (`qwen-nv stop` / `NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER`) before starting a new GPU process or disconnecting Thunderbolt, and never issue `CMD_BOOST_TO_MAX`.

## 4. Deep Modules & Zero-Slop Architectural Guardrail (`deslop-please`)
- All Rust code in `qwen-nv-rs` must pass `deslop-please` with **`0` errors and `0` `.deslop-please.allow` suppressions**:
  1. **`0` module cycles & `0` cross-module tangles**: Enforce a strict DAG (`hw` -> `{safetensors, tokenizer, gguf}` -> `{lora, mech_patch, topic_lora}` -> `{models, voice}` -> `runtime` -> `{tui, cli_ops, main}`).
  2. **`0` leaked struct fields (`field-leak`)**: Keep stateful struct fields private; expose behavior through methods that do real work rather than trivial field getters/setters or mutable external state pokes.
  3. **Deep modules with narrow interfaces (`shallow-module`)**: Any folder module with child modules must hide more functions than it exposes and keep each exposed type's external method surface `<= 7` items.
  4. **Bounded layout (`wide-directory` / `long-file`)**: Keep every directory `<= 15` entries and every `.rs` source file `<= 2,000` lines.
