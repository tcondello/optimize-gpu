//! Offline Weight Surgery (`mech`), LoRA Extraction (`lora`), Drafter Distillation (`drafter`),
//! and Standalone / Pareto / Suite `Inkling-2-Mini` & `Qwen` CLI Runners.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use crate::gguf::{resolve_model_path, GgufFile, QwenModelConfig, Utf8StreamDecoder};
use crate::hw::NvGpu;
use crate::lora::{extract_svd_peft_adapter, write_synthetic_peft_adapter};
use crate::mech_patch::{
    apply_surgical_weight_patch, build_ablation_lora_from_bf16, kl_divergence_from_logits,
    localize_causal_directions, rollback_surgical_weight_patch,
};
use crate::models::drafter::CalibratedNgramPrior;
use crate::models::inkling::InklingEngine;
use crate::models::qwen::QwenMegakernelEngine;
use crate::runtime::client::{DaemonControl, DaemonHooksClient, DaemonQuery};
use crate::runtime::session::format_stats_json;
use crate::topic_lora::{
    json_find_num, solve_alphaedit_lora_layer, write_or_merge_topic_peft_lora, QmkvFile,
};

fn parse_i32_slice_in_line(ln: &str) -> Option<Vec<i32>> {
    let sub = if let Some(idx) = ln.find("\"tokens\"") {
        &ln[idx + 8..]
    } else {
        ln
    };
    let open = sub.find('[')?;
    let close = sub[open + 1..].find(']')?;
    let inner = &sub[open + 1..open + 1 + close];
    let mut v = Vec::new();
    for p in inner.split(',') {
        let pt = p.trim();
        if !pt.is_empty() {
            if let Ok(val) = pt.parse::<i32>() {
                v.push(val);
            }
        }
    }
    if v.is_empty() { None } else { Some(v) }
}

fn parse_i32_tokens_jsonl(path: &Path) -> Result<Vec<Vec<i32>>, String> {
    let txt = fs::read_to_string(path)
        .map_err(|e| format!("Failed reading {}: {}", path.display(), e))?;
    let mut seqs = Vec::new();
    for raw_ln in txt.lines() {
        let ln = raw_ln.trim();
        if !ln.is_empty() {
            if let Some(v) = parse_i32_slice_in_line(ln) {
                seqs.push(v);
            }
        }
    }
    if seqs.is_empty() {
        return Err(format!("No token arrays found in {}", path.display()));
    }
    Ok(seqs)
}

fn parse_kv_specs_jsonl(path: &Path) -> Result<Vec<(Vec<i32>, usize, usize)>, String> {
    let txt = fs::read_to_string(path)
        .map_err(|e| format!("Failed reading {}: {}", path.display(), e))?;
    let mut specs = Vec::new();
    for raw_ln in txt.lines() {
        let ln = raw_ln.trim();
        if ln.is_empty() {
            continue;
        }
        let Some(toks) = parse_i32_slice_in_line(ln) else {
            continue;
        };
        let pos_start = json_find_num(ln, "pos_start")
            .map(|x| x.max(0.0) as usize)
            .unwrap_or_else(|| toks.len().saturating_sub(1));
        let pos_count = json_find_num(ln, "pos_count")
            .map(|x| x.max(1.0) as usize)
            .unwrap_or(1);
        specs.push((toks, pos_start, pos_count));
    }
    if specs.is_empty() {
        return Err(format!("No token specs found in {}", path.display()));
    }
    Ok(specs)
}

fn ensure_client_daemon(mref: &str, m_exp: bool, ctx: usize) -> Result<(), String> {
    if !m_exp && DaemonControl::ping().is_ok() {
        Ok(())
    } else {
        DaemonControl::ensure_running(mref, ctx)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_lora_subcmd(
    positional: &[String],
    model_ref: &str,
    model_explicit: bool,
    max_ctx: usize,
    lora_rank: usize,
    lora_rank_explicit: bool,
    lora_alpha: f32,
    lora_strength: f32,
    lora_scale: f32,
    attn_only: bool,
) -> Result<(), String> {
    let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
    match action {
        "extract-svd" => {
            let target_ref = positional.get(1).ok_or_else(|| {
                "Usage: qwen-nv lora extract-svd <TARGET_GGUF_OR_ALIAS> <OUT_DIR> [--rank 32] [--attn-only]".to_string()
            })?;
            let out_dir = positional.get(2).ok_or_else(|| {
                "Usage: qwen-nv lora extract-svd <TARGET_GGUF_OR_ALIAS> <OUT_DIR> [--rank 32] [--attn-only]".to_string()
            })?;
            let svd_rank = if lora_rank_explicit { lora_rank } else { 32 };
            let base_path = resolve_model_path(model_ref)?;
            let targ_path = resolve_model_path(target_ref)?;
            let gguf = GgufFile::open(&base_path)?;
            let tok = gguf.build_tokenizer()?;
            let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, model_ref)?;
            eprintln!(
                "[qwen-nv extract-svd] Dequantizing & running Randomized Rank-{} SVD across {} layers (include_mlp={})...",
                svd_rank,
                cfg.num_layers,
                !attn_only
            );
            let st = extract_svd_peft_adapter(
                &base_path,
                &targ_path,
                Path::new(out_dir),
                cfg.num_layers,
                svd_rank,
                !attn_only,
            )?;
            println!(
                "{{\"status\":\"ok\",\"out_dir\":\"{}\",\"num_layers\":{},\"rank\":{},\"include_mlp\":{},\"num_tensors\":{},\"safetensors_bytes\":{},\"safetensors_mb\":{:.2},\"attn_energy_pct\":{:.2},\"mlp_energy_pct\":{:.2},\"total_energy_pct\":{:.2},\"elapsed_s\":{:.3}}}",
                out_dir,
                st.num_layers,
                st.rank,
                st.include_mlp,
                st.num_tensors,
                st.safetensors_bytes,
                (st.safetensors_bytes as f64) / (1024.0 * 1024.0),
                st.attn_energy_pct,
                st.mlp_energy_pct,
                st.total_energy_pct,
                st.elapsed_s
            );
            Ok(())
        }
        "synth-test" => {
            let out_dir = positional
                .get(1)
                .ok_or_else(|| "Usage: qwen-nv lora synth-test <OUT_DIR> [--rank 16] [--strength 0.25]".to_string())?;
            let path = resolve_model_path(model_ref)?;
            let gguf = GgufFile::open(&path)?;
            let tok = gguf.build_tokenizer()?;
            let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, model_ref)?;
            let q_dim = cfg.n_heads * cfg.head_dim;
            let kv_dim = cfg.n_kv_heads * cfg.head_dim;
            write_synthetic_peft_adapter(
                Path::new(out_dir),
                cfg.num_layers,
                cfg.dim,
                q_dim,
                kv_dim,
                lora_rank,
                lora_alpha,
                lora_strength,
            )?;
            println!(
                "{{\"status\":\"ok\",\"out_dir\":\"{}\",\"layers\":{},\"rank\":{},\"alpha\":{:.1},\"strength\":{:.3}}}",
                out_dir, cfg.num_layers, lora_rank, lora_alpha, lora_strength
            );
            Ok(())
        }
        "load" => {
            let adapter_path = positional
                .get(1)
                .ok_or_else(|| "Usage: qwen-nv lora load <ADAPTER_DIR_OR_SAFETENSORS> [--scale 1.0]".to_string())?;
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            let t0 = Instant::now();
            let res = DaemonControl::query(DaemonQuery::LoraLoad {
                path: adapter_path.clone(),
                scale: lora_scale,
            })?;
            let client_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if let Some(stripped) = res.strip_suffix('}') {
                println!("{},\"client_ipc_ms\":{:.3}}}", stripped, client_ms);
            } else {
                println!("{}", res);
            }
            Ok(())
        }
        "clear" | "off" | "unload" => {
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            let t0 = Instant::now();
            let res = DaemonControl::query(DaemonQuery::LoraClear)?;
            let client_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if let Some(stripped) = res.strip_suffix('}') {
                println!("{},\"client_ipc_ms\":{:.3}}}", stripped, client_ms);
            } else {
                println!("{}", res);
            }
            Ok(())
        }
        "status" => {
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            println!("{}", DaemonControl::query(DaemonQuery::LoraStatus)?);
            Ok(())
        }
        other => Err(format!(
            "Unknown `lora` action '{}' (expected load, clear, status, extract-svd, or synth-test)",
            other
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_drafter_subcmd(
    positional: &[String],
    model_ref: &str,
    model_explicit: bool,
    max_ctx: usize,
    drafter_prior_path: &str,
    output_path: Option<&str>,
    min_support: u32,
    min_prob: f32,
) -> Result<(), String> {
    let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
    match action {
        "status" => {
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            println!("{}", DaemonControl::query(DaemonQuery::DrafterStatus)?);
            Ok(())
        }
        "clear" | "off" | "unload" => {
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            println!("{}", DaemonControl::query(DaemonQuery::DrafterClear)?);
            Ok(())
        }
        "load" | "warm" => {
            let d_ref = positional.get(1).map(|s| s.as_str()).unwrap_or("qwen3:0.6b");
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            println!(
                "{}",
                DaemonControl::query(DaemonQuery::DrafterLoad {
                    draft_ref: d_ref.to_string(),
                    prior_path: drafter_prior_path.to_string(),
                })?
            );
            Ok(())
        }
        "distill" => {
            let seqs_jsonl = positional.get(1).ok_or_else(|| {
                "Usage: qwen-nv drafter distill <SEQS.jsonl> -o <OUT_PRIOR.bin> [--min-support 2] [--min-prob 0.85]".to_string()
            })?;
            let out_p = output_path.ok_or("Pass -o <OUT_PRIOR.bin>")?;
            let seqs = parse_i32_tokens_jsonl(Path::new(seqs_jsonl))?;
            let prior = CalibratedNgramPrior::distill_from_sequences(&seqs, min_support, min_prob);
            prior.save_bin(Path::new(out_p))?;
            let (n5, n4) = prior.counts();
            println!(
                "{{\"status\":\"ok\",\"num_sequences\":{},\"map5_entries\":{},\"map4_entries\":{},\"out_path\":\"{}\"}}",
                seqs.len(),
                n5,
                n4,
                out_p
            );
            Ok(())
        }
        other => Err(format!(
            "Unknown `drafter` action '{}' (expected load, clear, status, or distill)",
            other
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_mech_subcmd(
    positional: &[String],
    model_ref: &str,
    model_explicit: bool,
    max_ctx: usize,
    output_path: Option<&str>,
    mech_report_path: Option<&str>,
    mech_layers: &[usize],
    mech_o_only: bool,
    mech_down_only: bool,
    mech_rescale_rows: bool,
    lora_rank: usize,
    lora_rank_explicit: bool,
    lora_alpha: f32,
    lora_alpha_explicit: bool,
    embed_lora_path: &str,
    lora_scale: f32,
    bg_kv_path: Option<&str>,
    topic_gamma: f32,
    topic_lambda: f32,
    topic_layer_frac: f32,
    topic_merge_prev: Option<&str>,
) -> Result<(), String> {
    let action = positional.first().map(|s| s.as_str()).unwrap_or("help");
    let eff_rank = if lora_rank_explicit { lora_rank } else { 1 };
    let eff_alpha = if lora_alpha_explicit { lora_alpha } else { 1.0 };
    let edit_o = !mech_down_only;
    let edit_dn = !mech_o_only;

    match action {
        "extract-hooks" => {
            let toks_jsonl = positional.get(1).ok_or_else(|| {
                "Usage: qwen-nv mech extract-hooks <TOKS.jsonl> -o <OUT_HOOKS.bin>".to_string()
            })?;
            let out_bin = output_path.ok_or("Pass -o <OUT_HOOKS.bin>")?;
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            let seqs = parse_i32_tokens_jsonl(Path::new(toks_jsonl))?;
            if let Some(p) = Path::new(out_bin).parent() {
                if !p.as_os_str().is_empty() {
                    fs::create_dir_all(p).map_err(|e| e.to_string())?;
                }
            }
            let mut f = fs::File::create(out_bin).map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            let mut floats_per_seq = 0usize;
            for seq in &seqs {
                let h = DaemonHooksClient::extract_hooks(seq)?;
                floats_per_seq = h.len();
                let mut raw = Vec::with_capacity(h.len() * 4);
                for &v in &h {
                    raw.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                f.write_all(&raw).map_err(|e| e.to_string())?;
            }
            println!(
                "{{\"status\":\"ok\",\"num_samples\":{},\"floats_per_sample\":{},\"out_path\":\"{}\",\"elapsed_s\":{:.3}}}",
                seqs.len(),
                floats_per_seq,
                out_bin,
                t0.elapsed().as_secs_f64()
            );
            Ok(())
        }
        "localize" => {
            let fail_bin = positional.get(1).ok_or_else(|| {
                "Usage: qwen-nv mech localize <FAIL_HOOKS.bin> <CLEAN_HOOKS.bin> -o <OUT_DIRS.bin> --report <REPORT.json>".to_string()
            })?;
            let clean_bin = positional.get(2).ok_or_else(|| {
                "Usage: qwen-nv mech localize <FAIL_HOOKS.bin> <CLEAN_HOOKS.bin> -o <OUT_DIRS.bin> --report <REPORT.json>".to_string()
            })?;
            let out_dirs = output_path.ok_or("Pass -o <OUT_DIRS.bin>")?;
            let out_rep = mech_report_path
                .unwrap_or("/Users/tim/Code/optimize-gpu/data/causal_localization.json");
            let path = resolve_model_path(model_ref)?;
            let gguf = GgufFile::open(&path)?;
            let tok = gguf.build_tokenizer()?;
            let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, model_ref)?;
            let ranked = localize_causal_directions(
                Path::new(fail_bin),
                Path::new(clean_bin),
                cfg.num_layers,
                cfg.dim,
                Path::new(out_dirs),
                Path::new(out_rep),
            )?;
            let top6: Vec<String> = ranked
                .iter()
                .take(6)
                .map(|s| {
                    format!(
                        "{{\"layer\":{},\"score\":{:.4},\"cos_sep\":{:.4},\"cohen_d\":{:.2}}}",
                        s.layer, s.combined_score, s.res_cos_sep, s.res_cohen_d
                    )
                })
                .collect();
            println!(
                "{{\"status\":\"ok\",\"dirs_bin\":\"{}\",\"report_json\":\"{}\",\"top_layers\":[{}]}}",
                out_dirs,
                out_rep,
                top6.join(",")
            );
            Ok(())
        }
        "make-lora" => {
            if positional.len() < 4 || mech_layers.is_empty() {
                return Err("Usage: qwen-nv mech make-lora <BF16_DIR> <DIRS.bin> <OUT_LORA_DIR> --layers <L0,L1,...> [--rank 1] [--alpha 1.0]".into());
            }
            build_ablation_lora_from_bf16(
                Path::new(&positional[1]),
                Path::new(&positional[2]),
                mech_layers,
                edit_o,
                edit_dn,
                eff_rank,
                eff_alpha,
                Path::new(&positional[3]),
            )?;
            println!(
                "{{\"status\":\"ok\",\"out_lora_dir\":\"{}\",\"layers\":{:?},\"rank\":{},\"alpha\":{:.3},\"edit_o\":{},\"edit_down\":{}}}",
                positional[3], mech_layers, eff_rank, eff_alpha, edit_o, edit_dn
            );
            Ok(())
        }
        "patch" => {
            if positional.len() < 5 || mech_layers.is_empty() {
                return Err("Usage: qwen-nv mech patch <BF16_DIR> <BASE_Q8_GGUF> <DIRS.bin> <OUT_CKPT_DIR> --layers <L0,L1,...> [--rank 1] [--alpha 1.0] [--no-rescale]".into());
            }
            let sum = apply_surgical_weight_patch(
                Path::new(&positional[1]),
                Path::new(&positional[2]),
                Path::new(&positional[3]),
                Path::new(&positional[4]),
                mech_layers,
                edit_o,
                edit_dn,
                eff_rank,
                eff_alpha,
                mech_rescale_rows,
            )?;
            println!(
                "{{\"status\":\"patched\",\"out_dir\":\"{}\",\"gguf_path\":\"{}\",\"num_tensors\":{},\"subspace_dof_params\":{},\"subspace_dof_pct\":{:.6},\"mean_rel_frobenius_delta\":{:.6},\"elapsed_s\":{:.3}}}",
                sum.out_dir,
                sum.gguf_path,
                sum.edited_tensors.len(),
                sum.subspace_dof_params,
                sum.subspace_dof_pct,
                sum.mean_rel_frobenius_delta,
                sum.elapsed_s
            );
            Ok(())
        }
        "rollback" => {
            let ckpt_dir = positional
                .get(1)
                .ok_or("Usage: qwen-nv mech rollback <OUT_CKPT_DIR>")?;
            let n_restored = rollback_surgical_weight_patch(Path::new(ckpt_dir))?;
            println!("{{\"status\":\"rolled_back\",\"restored_slices\":{}}}", n_restored);
            Ok(())
        }
        "kl-eval" => {
            let toks_jsonl = positional
                .get(1)
                .ok_or("Usage: qwen-nv mech kl-eval <CLEAN_TOKS.jsonl> --lora <LORA_DIR>")?;
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            let seqs = parse_i32_tokens_jsonl(Path::new(toks_jsonl))?;
            let _ = DaemonControl::query(DaemonQuery::LoraClear)?;
            let mut base_logits = Vec::with_capacity(seqs.len());
            for seq in &seqs {
                base_logits.push(DaemonHooksClient::eval_logits(seq)?);
            }
            if !embed_lora_path.is_empty() {
                let _ = DaemonControl::query(DaemonQuery::LoraLoad {
                    path: embed_lora_path.to_string(),
                    scale: lora_scale,
                })?;
            }
            let mut kls = Vec::with_capacity(seqs.len());
            let mut top1_same = 0usize;
            for (idx, seq) in seqs.iter().enumerate() {
                let q_log = DaemonHooksClient::eval_logits(seq)?;
                let p_log = &base_logits[idx];
                let kl = kl_divergence_from_logits(p_log, &q_log);
                let p_top = p_log
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|x| x.0)
                    .unwrap_or(0);
                let q_top = q_log
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|x| x.0)
                    .unwrap_or(0);
                if p_top == q_top {
                    top1_same += 1;
                }
                kls.push(kl);
            }
            if !embed_lora_path.is_empty() {
                let _ = DaemonControl::query(DaemonQuery::LoraClear)?;
            }
            let mean_kl = kls.iter().sum::<f64>() / (kls.len().max(1) as f64);
            let max_kl = kls.iter().copied().fold(0.0f64, f64::max);
            println!(
                "{{\"num_positions\":{},\"mean_kl_nats\":{:.6},\"max_kl_nats\":{:.6},\"top1_agreement\":{:.4}}}",
                seqs.len(),
                mean_kl,
                max_kl,
                (top1_same as f64) / (seqs.len().max(1) as f64)
            );
            Ok(())
        }
        "extract-kv" => {
            let specs_jsonl = positional.get(1).ok_or_else(|| {
                "Usage: qwen-nv mech extract-kv <SPECS.jsonl> --layers <L0,L1,...> -o <OUT.qmkv>".to_string()
            })?;
            let out_qmkv = output_path.ok_or("Pass -o <OUT.qmkv>")?;
            if mech_layers.is_empty() {
                return Err("Pass --layers <L0,L1,...> for `mech extract-kv`".into());
            }
            ensure_client_daemon(model_ref, model_explicit, max_ctx)?;
            let specs = parse_kv_specs_jsonl(Path::new(specs_jsonl))?;
            let t0 = Instant::now();
            let mut all_keys = Vec::<f32>::new();
            let mut all_res = Vec::<f32>::new();
            let mut total_positions = 0usize;
            let mut hidden_dim = 0usize;
            let mut dim = 0usize;

            for (toks, pos_start, pos_count) in &specs {
                let (k_f32, r_f32) =
                    DaemonHooksClient::extract_mlp_kv(toks, mech_layers, *pos_start, *pos_count)?;
                dim = r_f32.len() / (*pos_count).max(1);
                hidden_dim = k_f32.len() / ((*pos_count).max(1) * mech_layers.len());
                all_keys.extend_from_slice(&k_f32);
                all_res.extend_from_slice(&r_f32);
                total_positions += *pos_count;
            }
            let qmkv = QmkvFile {
                layers: mech_layers.to_vec(),
                hidden_dim,
                dim,
                num_positions: total_positions,
                keys: all_keys,
                final_res: all_res,
            };
            qmkv.write(Path::new(out_qmkv))?;
            println!(
                "{{\"status\":\"ok\",\"num_sequences\":{},\"num_positions\":{},\"layers\":{:?},\"hidden_dim\":{},\"dim\":{},\"out_path\":\"{}\",\"elapsed_s\":{:.3}}}",
                specs.len(),
                total_positions,
                mech_layers,
                hidden_dim,
                dim,
                out_qmkv,
                t0.elapsed().as_secs_f64()
            );
            Ok(())
        }
        "solve-topic-lora" => {
            if positional.len() < 4 {
                return Err("Usage: qwen-nv mech solve-topic-lora <BARE.qmkv> <CTX.qmkv> <OUT_LORA_DIR> --layers <L0,...> [--bg-kv <BG.qmkv>] [--gamma 50.0] [--lambda 0.001] [--rank 32] [--layer-frac 1.0] [--merge-prev <DIR>]".into());
            }
            let t0 = Instant::now();
            let bare = QmkvFile::read(Path::new(&positional[1]))?;
            let ctx = QmkvFile::read(Path::new(&positional[2]))?;
            if bare.num_positions != ctx.num_positions
                || bare.dim != ctx.dim
                || bare.hidden_dim != ctx.hidden_dim
            {
                return Err(format!(
                    "BARE.qmkv ([pos={}, dim={}, h={}) mismatches CTX.qmkv ([pos={}, dim={}, h={}])",
                    bare.num_positions,
                    bare.dim,
                    bare.hidden_dim,
                    ctx.num_positions,
                    ctx.dim,
                    ctx.hidden_dim
                ));
            }
            let target_layers = if mech_layers.is_empty() {
                bare.layers.clone()
            } else {
                mech_layers.to_vec()
            };
            let bg_opt = match bg_kv_path {
                Some(p) => Some(QmkvFile::read(Path::new(p))?),
                None => None,
            };
            let r_rows: Vec<f32> = ctx
                .final_res
                .iter()
                .zip(bare.final_res.iter())
                .map(|(&c, &b)| (c - b) * topic_layer_frac)
                .collect();
            let max_r = if lora_rank_explicit {
                lora_rank.clamp(1, 32)
            } else {
                32
            };
            let mut solved = Vec::with_capacity(target_layers.len());
            for &l in &target_layers {
                let k1_rows = bare.extract_layer_keys(l)?;
                let k0_rows = if let Some(ref bg) = bg_opt {
                    bg.extract_layer_keys(l)?
                } else {
                    Vec::new()
                };
                solved.push(solve_alphaedit_lora_layer(
                    l,
                    &k1_rows,
                    &r_rows,
                    &k0_rows,
                    bare.hidden_dim,
                    bare.dim,
                    topic_gamma,
                    topic_lambda,
                    max_r,
                )?);
            }
            let sum = write_or_merge_topic_peft_lora(
                Path::new(&positional[3]),
                &solved,
                topic_merge_prev.map(Path::new),
                bare.dim,
                bare.hidden_dim,
                None,
            )?;
            println!(
                "{{\"status\":\"ok\",\"out_dir\":\"{}\",\"layers\":{:?},\"rank\":{},\"num_positions\":{},\"mean_fit_rel_err\":{:.6},\"mean_bg_leak_rms\":{:.6},\"mean_delta_frob\":{:.4},\"safetensors_bytes\":{},\"elapsed_s\":{:.3}}}",
                sum.out_dir,
                sum.layers,
                sum.rank,
                bare.num_positions,
                sum.mean_fit_rel_err,
                sum.mean_bg_leak_rms,
                sum.mean_delta_frob,
                sum.safetensors_bytes,
                t0.elapsed().as_secs_f64()
            );
            Ok(())
        }
        other => Err(format!(
            "Unknown `mech` action '{}' (expected extract-hooks, extract-kv, localize, make-lora, solve-topic-lora, patch, rollback, or kl-eval)",
            other
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_inkling_subcmd(
    positional: &[String],
    eff_model: &str,
    max_ctx: usize,
    inkling_gamma: Option<usize>,
    seed: Option<u64>,
    temp: f32,
    temp_explicit: bool,
    max_tokens: usize,
    max_tokens_explicit: bool,
    spec: bool,
    spec_explicit: bool,
    inkling_pareto: bool,
    inkling_suite: bool,
    inkling_passes: Option<usize>,
    inkling_effort: Option<&str>,
    sys_prompt_opt: Option<&str>,
    raw_mode: bool,
    quiet: bool,
    json_stats: bool,
    output_path: Option<&str>,
) -> Result<(), String> {
    let _ = DaemonControl::stop();
    DaemonControl::install_signal_handlers();
    let mut gpu = NvGpu::init()?;
    let mut engine = InklingEngine::load(&mut gpu, eff_model, max_ctx)?;
    if let Some(g) = inkling_gamma {
        engine.set_dspark_gamma(g);
    }
    let eff_seed = seed.unwrap_or(42);
    let eff_temp = if temp_explicit { temp } else { 0.0 };
    let eff_max_toks = if max_tokens_explicit { max_tokens } else { 512 };
    let spec = if spec_explicit { spec } else { false };

    if inkling_pareto {
        let pareto_cases: &[(&str, &str, &str, usize)] = &[
            (
                "concise_factual_effort0",
                "What are the three main architectural innovations in a looped Mixture-of-Experts transformer? Reply in 3 concise bullet points.",
                "0",
                384,
            ),
            (
                "coding_rust_is_prime_effort0",
                "Write a concise zero-dependency Rust function `is_prime(n: u64) -> bool`.",
                "0",
                384,
            ),
            (
                "coding_rust_bsearch_effort0",
                "Write a concise zero-dependency generic Rust function `lower_bound<T: Ord>(slice: &[T], target: &T) -> usize` with 3 `assert_eq!` unit test examples.",
                "0",
                384,
            ),
            (
                "structured_json_effort0",
                "Output ONLY a valid minified JSON object with keys `gpu`, `sm_count`, `vram_gb`, `quant`, `loop_passes`, and `logical_layers` describing an NVIDIA RTX 3080 (68 SMs, 10 GB) running Inkling-2-Mini (3 passes, 48 logical layers, IQ2).",
                "0",
                256,
            ),
            (
                "math_amc12_nut_mix",
                "A box contains 10 pounds of a nut mix that is 50% peanuts, 20% cashews, and 30% almonds. A second nut mix containing 20% peanuts, 40% cashews, and 40% almonds is added to the box resulting in a new nut mix that is 40% peanuts. How many pounds of cashews are now in the box? Give the exact final number.",
                "0.2",
                640,
            ),
            (
                "math_bat_and_ball",
                "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "0.2",
                256,
            ),
        ];

        let strategies: &[(&str, bool, usize, bool)] = &[
            ("persistent_b1", false, 0, true),
            ("dspark_gamma2", true, 2, false),
            ("dspark_gamma3", true, 3, true),
            ("dspark_gamma4", true, 4, false),
            ("dspark_gamma7", true, 7, false),
        ];

        let mut baseline_tokens = HashMap::<String, Vec<i32>>::new();
        let mut rows_json = Vec::new();
        let model_file = resolve_model_path(eff_model)
            .ok()
            .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| eff_model.to_string());

        for &(strat_name, en_spec, gam, run_all) in strategies {
            if gam > 0 {
                engine.set_dspark_gamma(gam);
            }
            for (c_idx, &(case_name, prompt_s, eff_str, m_toks)) in pareto_cases.iter().enumerate() {
                if !run_all && c_idx == 4 {
                    continue;
                }
                let p_ids = engine.format_chat_prompt(prompt_s, None, Some(eff_str));
                let mut reply = String::new();
                let st = engine.generate_with_cancel(
                    &mut gpu,
                    &p_ids,
                    m_toks,
                    0.0,
                    eff_seed,
                    Some(3),
                    en_spec,
                    true,
                    None,
                    false,
                    |_, piece| {
                        reply.push_str(piece);
                        true
                    },
                )?;
                let lossless_match = if !en_spec {
                    baseline_tokens.insert(case_name.to_string(), st.token_ids.clone());
                    true
                } else if let Some(ref_toks) = baseline_tokens.get(case_name) {
                    ref_toks == &st.token_ids
                } else {
                    true
                };
                eprintln!(
                    "[pareto] {:16} | {:14} | {:28} | prompt={:3} ({:6.1} t/s) | gen={:3} @ {:6.2} tok/s (DSpark +{:3}/{:3}) | match={}",
                    model_file,
                    strat_name,
                    case_name,
                    st.prompt_tokens,
                    st.prefill_tok_per_s,
                    st.completion_tokens,
                    st.decode_tok_per_s,
                    st.spec_accepted_extra,
                    st.spec_verify_calls,
                    lossless_match
                );
                let esc_rep = reply
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "");
                rows_json.push(format!(
                    "{{\"model_file\":\"{}\",\"strategy\":\"{}\",\"case\":\"{}\",\"effort\":\"{}\",\"spec\":{},\"gamma\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.2},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.2},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.2},\"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"lossless_match\":{},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2},\"response\":\"{}\"}}",
                    model_file,
                    strat_name,
                    case_name,
                    eff_str,
                    en_spec,
                    gam,
                    st.prompt_tokens,
                    st.completion_tokens,
                    st.prefill_ms,
                    st.prefill_tok_per_s,
                    st.decode_ms,
                    st.decode_tok_per_s,
                    st.ttft_ms,
                    st.spec_verify_calls,
                    st.spec_accepted_extra,
                    lossless_match,
                    st.vram_weights_mb,
                    st.vram_kv_conv_mb,
                    esc_rep
                ));
            }
        }

        let summary_json = format!(
            "{{\"model_file\":\"{}\",\"model_name\":\"{}\",\"gpu\":\"NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs, 10GB GDDR6X)\",\"results\":[{}]}}",
            model_file,
            engine.model_name(),
            rows_json.join(",")
        );
        if let Some(out_p) = output_path {
            fs::write(out_p, &summary_json).map_err(|e| e.to_string())?;
        }
        if json_stats {
            println!("{}", summary_json);
        }
        gpu.shutdown_clean();
        return Ok(());
    }

    if inkling_suite {
        let bench_cases: &[(&str, &str, &str, usize, usize)] = &[
            (
                "concise_factual_effort0",
                "What are the three main architectural innovations in a looped Mixture-of-Experts transformer? Reply in 3 concise bullet points.",
                "0",
                3,
                384,
            ),
            (
                "math_amc12_nut_mix_thinking",
                "A box contains 10 pounds of a nut mix that is 50% peanuts, 20% cashews, and 30% almonds. A second nut mix containing 20% peanuts, 40% cashews, and 40% almonds is added to the box resulting in a new nut mix that is 40% peanuts. How many pounds of cashews are now in the box? Give the exact final number.",
                "0.2",
                3,
                640,
            ),
            (
                "coding_rust_fn_effort0",
                "Write a concise zero-dependency Rust function `is_prime(n: u64) -> bool`.",
                "0",
                3,
                384,
            ),
            (
                "loop_ablation_1pass",
                "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "0.2",
                1,
                256,
            ),
            (
                "loop_ablation_2pass",
                "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "0.2",
                2,
                256,
            ),
            (
                "loop_ablation_3pass",
                "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "0.2",
                3,
                256,
            ),
        ];

        let mut rows_json = Vec::new();
        for &(name, prompt_s, eff_str, n_pass, m_toks) in bench_cases {
            eprintln!(
                "\n=== [Inkling Suite: {} (passes={}, effort={}, spec={})] ===",
                name, n_pass, eff_str, spec
            );
            let p_ids = engine.format_chat_prompt(prompt_s, sys_prompt_opt, Some(eff_str));
            let mut reply = String::new();
            let mut out_stdout = io::stdout();
            let st = engine.generate_with_cancel(
                &mut gpu,
                &p_ids,
                m_toks,
                0.0,
                eff_seed,
                Some(n_pass),
                spec,
                true,
                None,
                false,
                |_, piece| {
                    reply.push_str(piece);
                    if !quiet {
                        let _ = out_stdout.write_all(piece.as_bytes());
                        let _ = out_stdout.flush();
                    }
                    true
                },
            )?;
            if !quiet {
                println!();
            }
            let esc_rep = reply
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "");
            rows_json.push(format!(
                "{{\"case\":\"{}\",\"passes\":{},\"logical_layers\":{},\"effort\":\"{}\",\"spec\":{},\"gamma\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.2},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.2},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.2},\"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2},\"response\":\"{}\"}}",
                name,
                st.loop_passes,
                st.logical_layers_per_tok,
                eff_str,
                spec && engine.has_dspark() && n_pass == 3,
                engine.dspark_gamma(),
                st.prompt_tokens,
                st.completion_tokens,
                st.prefill_ms,
                st.prefill_tok_per_s,
                st.decode_ms,
                st.decode_tok_per_s,
                st.ttft_ms,
                st.spec_verify_calls,
                st.spec_accepted_extra,
                st.vram_weights_mb,
                st.vram_kv_conv_mb,
                esc_rep
            ));
        }
        let summary_json = format!(
            "{{\"model\":\"{}\",\"gpu\":\"NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs, 10GB GDDR6X)\",\"results\":[{}]}}",
            engine.model_name(),
            rows_json.join(",")
        );
        if let Some(out_p) = output_path {
            fs::write(out_p, &summary_json).map_err(|e| e.to_string())?;
        }
        if json_stats {
            println!("{}", summary_json);
        }
        gpu.shutdown_clean();
        return Ok(());
    }

    let prompt = positional.join(" ");
    if prompt.is_empty() {
        gpu.shutdown_clean();
        return Err("Usage: qwen-nv inkling [--passes 1|2|3] [--effort 0|0.2|0.9] [--spec|--no-spec] [--dspark-gamma 1..7] [--max-tokens N] <PROMPT>".into());
    }
    let prompt_ids = if raw_mode {
        engine.encode_raw(&prompt)
    } else {
        engine.format_chat_prompt(&prompt, sys_prompt_opt, inkling_effort)
    };
    let mut out_stdout = io::stdout();
    let res = engine.generate_with_cancel(
        &mut gpu,
        &prompt_ids,
        eff_max_toks,
        eff_temp,
        eff_seed,
        inkling_passes,
        spec,
        true,
        None,
        raw_mode,
        |_, piece| {
            if !quiet && !piece.is_empty() {
                let _ = out_stdout.write_all(piece.as_bytes());
                let _ = out_stdout.flush();
            }
            true
        },
    );
    let st = match res {
        Ok(s) => s,
        Err(e) => {
            gpu.shutdown_clean();
            return Err(e);
        }
    };
    if !quiet {
        println!();
    }
    if json_stats {
        println!(
            "{{\"model\":\"{}\",\"loop_passes\":{},\"logical_layers\":{},\"spec\":{},\"gamma\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.3},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.3},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.3},\"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2}}}",
            st.model_name,
            st.loop_passes,
            st.logical_layers_per_tok,
            spec && engine.has_dspark() && st.loop_passes == 3,
            engine.dspark_gamma(),
            st.prompt_tokens,
            st.completion_tokens,
            st.prefill_ms,
            st.prefill_tok_per_s,
            st.decode_ms,
            st.decode_tok_per_s,
            st.ttft_ms,
            st.spec_verify_calls,
            st.spec_accepted_extra,
            st.vram_weights_mb,
            st.vram_kv_conv_mb
        );
    } else if !quiet {
        eprintln!(
            "[inkling-nv] {} passes ({} logical layers) | prompt={} ({:.1} tok/s) | gen={} @ {:.2} tok/s (DSpark +{} accepted / {} steps) | TTFT={:.1} ms | VRAM={:.0}+{:.0} MiB",
            st.loop_passes,
            st.logical_layers_per_tok,
            st.prompt_tokens,
            st.prefill_tok_per_s,
            st.completion_tokens,
            st.decode_tok_per_s,
            st.spec_accepted_extra,
            st.spec_verify_calls,
            st.ttft_ms,
            st.vram_weights_mb,
            st.vram_kv_conv_mb
        );
    }
    gpu.shutdown_clean();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_direct_prompt(
    model_ref: &str,
    max_ctx: usize,
    prompt: &str,
    raw_mode: bool,
    sys_prompt_opt: Option<&str>,
    inkling_effort: Option<&str>,
    inkling_passes: Option<usize>,
    inkling_gamma: Option<usize>,
    max_tokens: usize,
    temp: f32,
    seed: u64,
    spec: bool,
    quiet: bool,
    json_stats: bool,
) -> Result<(), String> {
    let _ = DaemonControl::stop();
    DaemonControl::install_signal_handlers();
    let mut gpu = NvGpu::init()?;

    if InklingEngine::is_inkling_ref(model_ref) {
        let mut engine = InklingEngine::load(&mut gpu, model_ref, max_ctx)?;
        if let Some(g) = inkling_gamma {
            engine.set_dspark_gamma(g);
        }
        let prompt_ids = if raw_mode {
            engine.encode_raw(prompt)
        } else {
            engine.format_chat_prompt(prompt, sys_prompt_opt, inkling_effort)
        };
        let mut out_stdout = io::stdout();
        let res = engine.generate_with_cancel(
            &mut gpu,
            &prompt_ids,
            max_tokens,
            temp,
            seed,
            inkling_passes,
            spec,
            true,
            None,
            raw_mode,
            |_, piece| {
                if !quiet && !piece.is_empty() {
                    let _ = out_stdout.write_all(piece.as_bytes());
                    let _ = out_stdout.flush();
                }
                true
            },
        );
        let st = match res {
            Ok(s) => s,
            Err(e) => {
                gpu.shutdown_clean();
                return Err(e);
            }
        };
        if !quiet {
            println!();
        }
        if json_stats {
            println!(
                "{{\"model\":\"{}\",\"loop_passes\":{},\"logical_layers\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.3},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.3},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.3},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2}}}",
                st.model_name,
                st.loop_passes,
                st.logical_layers_per_tok,
                st.prompt_tokens,
                st.completion_tokens,
                st.prefill_ms,
                st.prefill_tok_per_s,
                st.decode_ms,
                st.decode_tok_per_s,
                st.ttft_ms,
                st.vram_weights_mb,
                st.vram_kv_conv_mb
            );
        }
        gpu.shutdown_clean();
        Ok(())
    } else {
        let mut engine = QwenMegakernelEngine::load(&mut gpu, model_ref, max_ctx)?;
        let prompt_ids = if raw_mode {
            engine.encode_raw_prompt(prompt)
        } else if let Some(sys_p) = sys_prompt_opt {
            engine.encode_chat(
                &[
                    ("system".into(), sys_p.to_string()),
                    ("user".into(), prompt.to_string()),
                ],
                true,
            )
        } else {
            engine.encode_chat(&[("user".into(), prompt.to_string())], true)
        };

        let mut dec = Utf8StreamDecoder::new();
        let mut out_stdout = io::stdout();
        let res = engine.generate(&mut gpu, &prompt_ids, max_tokens, temp, spec, None, None, |_, tok_bytes| {
            if !quiet {
                let s = dec.push(tok_bytes);
                if !s.is_empty() {
                    let _ = out_stdout.write_all(s.as_bytes());
                    let _ = out_stdout.flush();
                }
            }
            true
        });
        let (out_ids, stats) = match res {
            Ok(pair) => pair,
            Err(e) => {
                gpu.shutdown_clean();
                return Err(e);
            }
        };
        if !quiet {
            let tail = dec.flush();
            if !tail.is_empty() {
                print!("{}", tail);
            }
            println!();
        }
        if json_stats {
            println!("{}", format_stats_json(&stats, &out_ids));
        } else {
            eprintln!(
                "[qwen-nv] {} tokens in {:.1} ms ({:.2} tok/s decode, TTFT {:.2} ms, {} GPU launches, +{} spec tokens)",
                stats.completion_tokens,
                stats.decode_ms,
                stats.decode_tok_per_s,
                stats.ttft_ms,
                stats.kernel_launches,
                stats.spec_accepted_extra
            );
        }
        gpu.shutdown_clean();
        Ok(())
    }
}
