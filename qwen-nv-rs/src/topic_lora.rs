//! Zero-Crate Closed-Form Push-Through AlphaEdit Topic LoRA Compiler & Bank Manifest.
//!
//! Synthesizes Rank-`r` (`r <= 32`) `mlp.down_proj` PEFT LoRA adapters directly from GPU
//! SwiGLU key (`k_l`) and contextual final-residual (`h_last^ctx - h_last^bare`) snapshots
//! using closed-form regularized Cholesky push-through + thin Eckart-Young SVD in Apple
//! `Accelerate` (`cblas_sgemm`) + pure Rust `f64`.
//!
//! Strictly enforces `AGENTS.md`: `0` external crates, `0` PyTorch, `0` Docker.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::gguf::{f16_to_f32, f32_to_f16_bits};
use crate::hw::{get_u32, put_u32};
use crate::lora::MAX_LORA_RANK;
use crate::safetensors::SafeTensorsFile;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

const QMKV_MAGIC: &[u8; 4] = b"QMKV";
const QMKV_VERSION: u32 = 1;

// -----------------------------------------------------------------------------
// Binary `QMKV` Snapshot Format (`[layers, hidden_dim]` keys + `[dim]` final_res)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct QmkvFile {
    pub layers: Vec<usize>,
    pub hidden_dim: usize,
    pub dim: usize,
    pub num_positions: usize,
    /// `[num_positions, layers.len(), hidden_dim]` row-major `f32`
    pub keys: Vec<f32>,
    /// `[num_positions, dim]` row-major `f32`
    pub final_res: Vec<f32>,
}

impl QmkvFile {
    pub fn write(&self, path: &Path) -> Result<(), String> {
        let n_kl = self.layers.len();
        if self.keys.len() != self.num_positions * n_kl * self.hidden_dim {
            return Err(format!(
                "QMKV keys len mismatch ({} != {})",
                self.keys.len(),
                self.num_positions * n_kl * self.hidden_dim
            ));
        }
        if self.final_res.len() != self.num_positions * self.dim {
            return Err(format!(
                "QMKV final_res len mismatch ({} != {})",
                self.final_res.len(),
                self.num_positions * self.dim
            ));
        }
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
        }

        let hdr_bytes = 24 + n_kl * 4;
        let body_floats = self.keys.len() + self.final_res.len();
        let mut buf = vec![0u8; hdr_bytes + body_floats * 4];
        buf[0..4].copy_from_slice(QMKV_MAGIC);
        put_u32(&mut buf, 4, QMKV_VERSION);
        put_u32(&mut buf, 8, n_kl as u32);
        put_u32(&mut buf, 12, self.hidden_dim as u32);
        put_u32(&mut buf, 16, self.dim as u32);
        put_u32(&mut buf, 20, self.num_positions as u32);
        for (i, &l) in self.layers.iter().enumerate() {
            put_u32(&mut buf, 24 + i * 4, l as u32);
        }

        let mut pos = hdr_bytes;
        let k_stride = n_kl * self.hidden_dim;
        for p in 0..self.num_positions {
            let k_slice = &self.keys[p * k_stride..(p + 1) * k_stride];
            for &v in k_slice {
                buf[pos..pos + 4].copy_from_slice(&v.to_bits().to_le_bytes());
                pos += 4;
            }
            let r_slice = &self.final_res[p * self.dim..(p + 1) * self.dim];
            for &v in r_slice {
                buf[pos..pos + 4].copy_from_slice(&v.to_bits().to_le_bytes());
                pos += 4;
            }
        }
        fs::write(path, buf).map_err(|e| format!("Failed writing {}: {}", path.display(), e))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("Failed reading {}: {}", path.display(), e))?;
        if raw.len() < 24 || &raw[0..4] != QMKV_MAGIC {
            return Err(format!("Invalid QMKV header in {}", path.display()));
        }
        let ver = get_u32(&raw, 4);
        if ver != QMKV_VERSION {
            return Err(format!("Unsupported QMKV version {}", ver));
        }
        let n_kl = get_u32(&raw, 8) as usize;
        let hidden_dim = get_u32(&raw, 12) as usize;
        let dim = get_u32(&raw, 16) as usize;
        let num_positions = get_u32(&raw, 20) as usize;
        let hdr_bytes = 24 + n_kl * 4;
        let k_stride = n_kl * hidden_dim;
        let need = hdr_bytes + num_positions * (k_stride + dim) * 4;
        if raw.len() != need {
            return Err(format!(
                "QMKV size mismatch in {} ({} != {})",
                path.display(),
                raw.len(),
                need
            ));
        }
        let mut layers = Vec::with_capacity(n_kl);
        for i in 0..n_kl {
            layers.push(get_u32(&raw, 24 + i * 4) as usize);
        }
        let mut keys = Vec::with_capacity(num_positions * k_stride);
        let mut final_res = Vec::with_capacity(num_positions * dim);
        let mut off = hdr_bytes;
        for _ in 0..num_positions {
            for _ in 0..k_stride {
                keys.push(f32::from_bits(get_u32(&raw, off)));
                off += 4;
            }
            for _ in 0..dim {
                final_res.push(f32::from_bits(get_u32(&raw, off)));
                off += 4;
            }
        }
        Ok(Self {
            layers,
            hidden_dim,
            dim,
            num_positions,
            keys,
            final_res,
        })
    }

    pub fn extract_layer_keys(&self, layer: usize) -> Result<Vec<f32>, String> {
        let kl_idx = self
            .layers
            .iter()
            .position(|&l| l == layer)
            .ok_or_else(|| format!("Layer {} not present in QMKV file (has {:?})", layer, self.layers))?;
        let k_stride = self.layers.len() * self.hidden_dim;
        let mut out = Vec::with_capacity(self.num_positions * self.hidden_dim);
        for p in 0..self.num_positions {
            let base = p * k_stride + kl_idx * self.hidden_dim;
            out.extend_from_slice(&self.keys[base..base + self.hidden_dim]);
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Pure-Rust Linear Algebra (`f64` Cholesky + Modified Gram-Schmidt + Jacobi)
// -----------------------------------------------------------------------------

fn cholesky_factor_f64(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 1e-15 || !sum.is_finite() {
                    return Err(format!("Non-SPD pivot at diag {} (val={:.3e})", i, sum));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(l)
}

fn solve_cholesky_multi_rhs(l: &[f64], m_tot: usize, n_rhs: usize) -> Vec<f64> {
    // Solves `(L L^T) Z = [I_{n_rhs}; 0]`, returning `Z [m_tot, n_rhs]`
    let mut z = vec![0.0f64; m_tot * n_rhs];
    let mut y = vec![0.0f64; m_tot];
    for col in 0..n_rhs {
        for v in &mut y {
            *v = 0.0;
        }
        // Forward substitution: `L y = e_col` (note y[0..col] == 0!)
        for i in col..m_tot {
            let mut sum = if i == col { 1.0 } else { 0.0 };
            for k in col..i {
                sum -= l[i * m_tot + k] * y[k];
            }
            y[i] = sum / l[i * m_tot + i];
        }
        // Back substitution: `L^T x = y`
        for i in (0..m_tot).rev() {
            let mut sum = y[i];
            for k in (i + 1)..m_tot {
                sum -= l[k * m_tot + i] * z[k * n_rhs + col];
            }
            z[i * n_rhs + col] = sum / l[i * m_tot + i];
        }
    }
    z
}

/// Modified Gram-Schmidt QR on columns of `mat [rows, cols]` (`rows >= cols`),
/// returning `(q [rows, cols], r [cols, cols])` such that `mat == q @ r`.
fn mgs_qr_cols(mat: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f64>) {
    let mut q = vec![0.0f64; rows * cols];
    let mut r = vec![0.0f64; cols * cols];
    for j in 0..cols {
        for row in 0..rows {
            q[row * cols + j] = mat[row * cols + j] as f64;
        }
        for _pass in 0..2 {
            for i in 0..j {
                let mut dot = 0.0f64;
                for row in 0..rows {
                    dot += q[row * cols + i] * q[row * cols + j];
                }
                r[i * cols + j] += dot;
                for row in 0..rows {
                    q[row * cols + j] -= dot * q[row * cols + i];
                }
            }
        }
        let mut norm_sq = 0.0f64;
        for row in 0..rows {
            let v = q[row * cols + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        r[j * cols + j] = norm;
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for row in 0..rows {
                q[row * cols + j] *= inv;
            }
        } else {
            for row in 0..rows {
                q[row * cols + j] = 0.0;
            }
        }
    }
    let q_f32: Vec<f32> = q.into_iter().map(|v| v as f32).collect();
    (q_f32, r)
}

fn jacobi_sym_eigen_f64(mut a: Vec<f64>, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    let max_sweeps = if p <= 64 { 40 } else { 60 };
    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0f64;
        for i in 0..p {
            for j in (i + 1)..p {
                max_off = max_off.max(a[i * p + j].abs());
            }
        }
        if max_off < 1e-13 {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let aij = a[i * p + j];
                if aij.abs() < 1e-15 {
                    continue;
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                let tau = (ajj - aii) / (2.0 * aij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[i * p + i] = aii - t * aij;
                a[j * p + j] = ajj + t * aij;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;

                for k in 0..p {
                    if k != i && k != j {
                        let aki = a[k * p + i];
                        let akj = a[k * p + j];
                        a[k * p + i] = c * aki - s * akj;
                        a[i * p + k] = a[k * p + i];
                        a[k * p + j] = s * aki + c * akj;
                        a[j * p + k] = a[k * p + j];
                    }
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..p).collect();
    order.sort_unstable_by(|&i, &j| {
        a[j * p + j]
            .partial_cmp(&a[i * p + i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut eigs = vec![0.0f64; p];
    let mut v_sorted = vec![0.0f64; p * p];
    for (new_c, &old_c) in order.iter().enumerate() {
        eigs[new_c] = a[old_c * p + old_c].max(0.0);
        for r in 0..p {
            v_sorted[r * p + new_c] = v[r * p + old_c];
        }
    }
    (eigs, v_sorted)
}

// -----------------------------------------------------------------------------
// Closed-Form Push-Through AlphaEdit Solver (`mlp.down_proj` LoRA)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SolvedLayerLora {
    pub layer: usize,
    pub rank: usize,
    /// Row-major `[rank, d_ff]`
    pub lora_a: Vec<f32>,
    /// Row-major `[dim, rank]`
    pub lora_b: Vec<f32>,
    pub edit_fit_rel_err: f64,
    pub bg_leak_rms: f64,
    pub delta_frob: f64,
}

/// Solve `min_Delta ||Delta K1 - R||_F^2 + gamma ||Delta K0||_F^2 + lambda ||Delta||_F^2`
/// via exact push-through Cholesky on the `[n + n_bg, n + n_bg]` Gram system,
/// then compress to `<= max_rank` via exact Eckart-Young thin QR+Jacobi SVD if `n > max_rank`.
///
/// - `k1_rows`: `[n, d_ff]` bare post-SwiGLU MLP keys
/// - `r_rows`: `[n, dim]` target residual delta rows (`layer_frac * (h_ctx - h_bare)`)
/// - `k0_rows`: `[n_bg, d_ff]` preserved background MLP keys (empty slice allowed)
pub fn solve_alphaedit_lora_layer(
    layer: usize,
    k1_rows: &[f32],
    r_rows: &[f32],
    k0_rows: &[f32],
    d_ff: usize,
    dim: usize,
    gamma: f32,
    lambda_rel: f32,
    max_rank: usize,
) -> Result<SolvedLayerLora, String> {
    if max_rank == 0 || max_rank > MAX_LORA_RANK {
        return Err(format!("max_rank {} must be in 1..={}", max_rank, MAX_LORA_RANK));
    }
    if k1_rows.is_empty() || k1_rows.len() % d_ff != 0 {
        return Err("Invalid k1_rows dimensions".into());
    }
    let n = k1_rows.len() / d_ff;
    if r_rows.len() != n * dim {
        return Err(format!("r_rows len {} != n({}) * dim({})", r_rows.len(), n, dim));
    }
    if !k0_rows.is_empty() && k0_rows.len() % d_ff != 0 {
        return Err("Invalid k0_rows dimensions".into());
    }
    let n_bg = if d_ff > 0 { k0_rows.len() / d_ff } else { 0 };
    let m_tot = n + n_bg;

    let mut sum_k1_sq = 0.0f64;
    for &v in k1_rows {
        let vf = v as f64;
        sum_k1_sq += vf * vf;
    }
    let mean_k1_sq = (sum_k1_sq / (n as f64)).max(1e-6);
    let base_lambda = (lambda_rel as f64).max(1e-8) * mean_k1_sq;

    let sqrt_gamma = gamma.max(0.0).sqrt();
    let mut m_rows = vec![0.0f32; m_tot * d_ff];
    m_rows[..n * d_ff].copy_from_slice(k1_rows);
    for i in 0..(n_bg * d_ff) {
        m_rows[n * d_ff + i] = k0_rows[i] * sqrt_gamma;
    }

    // Gram matrix `G_raw = M_rows @ M_rows^T` -> `[m_tot, m_tot]`
    let mut g_f32 = vec![0.0f32; m_tot * m_tot];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m_tot as i32,
            m_tot as i32,
            d_ff as i32,
            1.0,
            m_rows.as_ptr(),
            d_ff as i32,
            m_rows.as_ptr(),
            d_ff as i32,
            0.0,
            g_f32.as_mut_ptr(),
            m_tot as i32,
        );
    }

    let mut l_chol = None;
    let mut ridge = base_lambda;
    for _retry in 0..5 {
        let mut g_f64 = vec![0.0f64; m_tot * m_tot];
        for i in 0..m_tot {
            for j in 0..=i {
                let sym = 0.5 * ((g_f32[i * m_tot + j] as f64) + (g_f32[j * m_tot + i] as f64));
                g_f64[i * m_tot + j] = sym;
                g_f64[j * m_tot + i] = sym;
            }
            g_f64[i * m_tot + i] += ridge;
        }
        if let Ok(fac) = cholesky_factor_f64(&g_f64, m_tot) {
            l_chol = Some(fac);
            break;
        }
        ridge *= 10.0;
    }
    let l_fac = l_chol.ok_or_else(|| format!("Cholesky failed on layer {} (m_tot={})", layer, m_tot))?;

    let z_cols = solve_cholesky_multi_rhs(&l_fac, m_tot, n);
    let mut z_t = vec![0.0f32; n * m_tot];
    for c in 0..n {
        for r in 0..m_tot {
            z_t[c * m_tot + r] = z_cols[r * n + c] as f32;
        }
    }

    // `a_full [n, d_ff] = z_t [n, m_tot] @ m_rows [m_tot, d_ff]`
    let mut a_full = vec![0.0f32; n * d_ff];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            n as i32,
            d_ff as i32,
            m_tot as i32,
            1.0,
            z_t.as_ptr(),
            m_tot as i32,
            m_rows.as_ptr(),
            d_ff as i32,
            0.0,
            a_full.as_mut_ptr(),
            d_ff as i32,
        );
    }

    // `b_full [dim, n] = r_rows^T`
    let mut b_full = vec![0.0f32; dim * n];
    for p in 0..n {
        for row in 0..dim {
            b_full[row * n + p] = r_rows[p * dim + row];
        }
    }

    let (eff_rank, mut lora_a, mut lora_b) = if n <= max_rank {
        // Norm-balance each rank-1 column/row pair so FP16 dynamic range stays well-conditioned
        for j in 0..n {
            let mut nb_sq = 0.0f64;
            for row in 0..dim {
                let v = b_full[row * n + j] as f64;
                nb_sq += v * v;
            }
            let mut na_sq = 0.0f64;
            for col in 0..d_ff {
                let v = a_full[j * d_ff + col] as f64;
                na_sq += v * v;
            }
            let nb = nb_sq.sqrt();
            let na = na_sq.sqrt();
            if nb > 1e-12 && na > 1e-12 {
                let s = (nb / na).sqrt() as f32;
                for row in 0..dim {
                    b_full[row * n + j] /= s;
                }
                for col in 0..d_ff {
                    a_full[j * d_ff + col] *= s;
                }
            }
        }
        (n, a_full, b_full)
    } else {
        // Exact thin Eckart-Young rank-`max_rank` SVD of `B_full @ A_full`
        // `A_full^T = Q_a @ R_a` (`[d_ff, n]` QR), `B_full = Q_b @ R_b` (`[dim, n]` QR)
        let mut a_t = vec![0.0f32; d_ff * n];
        for r in 0..n {
            for col in 0..d_ff {
                a_t[col * n + r] = a_full[r * d_ff + col];
            }
        }
        let (q_b, r_b) = mgs_qr_cols(&b_full, dim, n);
        let (q_a, r_a) = mgs_qr_cols(&a_t, d_ff, n);

        // `C [n, n] = R_b [n, n] @ R_a^T [n, n]`
        let mut c_core = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += r_b[i * n + k] * r_a[j * n + k];
                }
                c_core[i * n + j] = sum;
            }
        }

        // `C_cov = C @ C^T` -> symmetric `[n, n]`
        let mut c_cov = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += c_core[i * n + k] * c_core[j * n + k];
                }
                c_cov[i * n + j] = sum;
                c_cov[j * n + i] = sum;
            }
        }

        let (eigs, u_mat) = jacobi_sym_eigen_f64(c_cov, n);
        let r_eff = max_rank.min(n);

        // Build `u_b [n, r_eff]` and `v_a [r_eff, n]` with balanced `sqrt(sigma_j)` scaling
        let mut u_b = vec![0.0f32; n * r_eff];
        let mut v_a = vec![0.0f32; r_eff * n];
        for j in 0..r_eff {
            let sigma = eigs[j].max(0.0).sqrt();
            if sigma > 1e-12 {
                let sq = sigma.sqrt();
                for r in 0..n {
                    u_b[r * r_eff + j] = (u_mat[r * n + j] * sq) as f32;
                }
                for c in 0..n {
                    let mut ut_c = 0.0f64;
                    for r in 0..n {
                        ut_c += u_mat[r * n + j] * c_core[r * n + c];
                    }
                    v_a[j * n + c] = (ut_c / sq) as f32;
                }
            }
        }

        let mut out_b = vec![0.0f32; dim * r_eff];
        let mut out_a = vec![0.0f32; r_eff * d_ff];
        unsafe {
            // `out_b [dim, r_eff] = Q_b [dim, n] @ u_b [n, r_eff]`
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                dim as i32,
                r_eff as i32,
                n as i32,
                1.0,
                q_b.as_ptr(),
                n as i32,
                u_b.as_ptr(),
                r_eff as i32,
                0.0,
                out_b.as_mut_ptr(),
                r_eff as i32,
            );
            // `out_a [r_eff, d_ff] = v_a [r_eff, n] @ Q_a^T [n, d_ff]`
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                r_eff as i32,
                d_ff as i32,
                n as i32,
                1.0,
                v_a.as_ptr(),
                n as i32,
                q_a.as_ptr(),
                n as i32,
                0.0,
                out_a.as_mut_ptr(),
                d_ff as i32,
            );
        }
        (r_eff, out_a, out_b)
    };

    // Simulate exact `F16` round-trip when evaluating diagnostics so metrics match GPU execution!
    for v in &mut lora_a {
        *v = f16_to_f32(f32_to_f16_bits(*v));
    }
    for v in &mut lora_b {
        *v = f16_to_f32(f32_to_f16_bits(*v));
    }

    // Diagnostics: edit fit error `||B (A K1) - R||_F / ||R||_F`
    let mut ak1 = vec![0.0f32; eff_rank * n];
    let mut pred_r = vec![0.0f32; dim * n];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            eff_rank as i32,
            n as i32,
            d_ff as i32,
            1.0,
            lora_a.as_ptr(),
            d_ff as i32,
            k1_rows.as_ptr(),
            d_ff as i32,
            0.0,
            ak1.as_mut_ptr(),
            n as i32,
        );
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            dim as i32,
            n as i32,
            eff_rank as i32,
            1.0,
            lora_b.as_ptr(),
            eff_rank as i32,
            ak1.as_ptr(),
            n as i32,
            0.0,
            pred_r.as_mut_ptr(),
            n as i32,
        );
    }

    let mut err_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for row in 0..dim {
        for p in 0..n {
            let target = r_rows[p * dim + row] as f64;
            let diff = (pred_r[row * n + p] as f64) - target;
            err_sq += diff * diff;
            ref_sq += target * target;
        }
    }
    let edit_fit_rel_err = (err_sq / ref_sq.max(1e-20)).sqrt();

    let bg_leak_rms = if n_bg > 0 {
        let mut ak0 = vec![0.0f32; eff_rank * n_bg];
        let mut pred_0 = vec![0.0f32; dim * n_bg];
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                eff_rank as i32,
                n_bg as i32,
                d_ff as i32,
                1.0,
                lora_a.as_ptr(),
                d_ff as i32,
                k0_rows.as_ptr(),
                d_ff as i32,
                0.0,
                ak0.as_mut_ptr(),
                n_bg as i32,
            );
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                dim as i32,
                n_bg as i32,
                eff_rank as i32,
                1.0,
                lora_b.as_ptr(),
                eff_rank as i32,
                ak0.as_ptr(),
                n_bg as i32,
                0.0,
                pred_0.as_mut_ptr(),
                n_bg as i32,
            );
        }
        let sq: f64 = pred_0.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (sq / ((dim * n_bg) as f64)).sqrt()
    } else {
        0.0
    };

    // Exact `||B A||_F = sqrt(tr((B^T B) (A A^T)))`
    let mut btb = vec![0.0f32; eff_rank * eff_rank];
    let mut aat = vec![0.0f32; eff_rank * eff_rank];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            eff_rank as i32,
            eff_rank as i32,
            dim as i32,
            1.0,
            lora_b.as_ptr(),
            eff_rank as i32,
            lora_b.as_ptr(),
            eff_rank as i32,
            0.0,
            btb.as_mut_ptr(),
            eff_rank as i32,
        );
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            eff_rank as i32,
            eff_rank as i32,
            d_ff as i32,
            1.0,
            lora_a.as_ptr(),
            d_ff as i32,
            lora_a.as_ptr(),
            d_ff as i32,
            0.0,
            aat.as_mut_ptr(),
            eff_rank as i32,
        );
    }
    let frob_sq: f64 = btb
        .iter()
        .zip(aat.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();

    Ok(SolvedLayerLora {
        layer,
        rank: eff_rank,
        lora_a,
        lora_b,
        edit_fit_rel_err,
        bg_leak_rms,
        delta_frob: frob_sq.max(0.0).sqrt(),
    })
}

// -----------------------------------------------------------------------------
// Multi-Layer PEFT `adapter_model.safetensors` Writer & `--merge-prev`
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TopicLoraWriteSummary {
    pub out_dir: String,
    pub layers: Vec<usize>,
    pub rank: usize,
    pub safetensors_bytes: usize,
    pub mean_fit_rel_err: f64,
    pub mean_bg_leak_rms: f64,
    pub mean_delta_frob: f64,
}

fn parse_layer_from_down_proj_key(k: &str) -> Option<(usize, bool)> {
    let idx = k.find(".layers.")?;
    let rest = &k[idx + 8..];
    let dot = rest.find('.')?;
    let layer = rest[..dot].parse::<usize>().ok()?;
    let tail = &rest[dot + 1..];
    if !tail.contains("down_proj") {
        return None;
    }
    if tail.contains("lora_A") {
        Some((layer, false))
    } else if tail.contains("lora_B") {
        Some((layer, true))
    } else {
        None
    }
}

pub fn write_or_merge_topic_peft_lora(
    out_dir: &Path,
    new_layers: &[SolvedLayerLora],
    merge_prev_dir: Option<&Path>,
    dim: usize,
    hidden_dim: usize,
    fixed_rank: Option<usize>,
) -> Result<TopicLoraWriteSummary, String> {
    // Map: layer -> (rank, a_f32 [rank, hidden_dim], b_f32 [dim, rank])
    let mut layer_map: HashMap<usize, (usize, Vec<f32>, Vec<f32>)> = HashMap::new();

    if let Some(prev_dir) = merge_prev_dir {
        let st_path = if prev_dir.is_dir() {
            prev_dir.join("adapter_model.safetensors")
        } else {
            prev_dir.to_path_buf()
        };
        if st_path.exists() {
            let st = SafeTensorsFile::open(&st_path)?;
            let mut prev_pairs: HashMap<usize, (Option<String>, Option<String>)> = HashMap::new();
            for k in st.tensor_names() {
                if let Some((l, is_b)) = parse_layer_from_down_proj_key(&k) {
                    let e = prev_pairs.entry(l).or_insert((None, None));
                    if is_b {
                        e.1 = Some(k);
                    } else {
                        e.0 = Some(k);
                    }
                }
            }
            for (l, (a_key_opt, b_key_opt)) in prev_pairs {
                if let (Some(ak), Some(bk)) = (a_key_opt, b_key_opt) {
                    let (a_vals, a_shape) = st.decode_f32(&ak)?;
                    let (b_vals, b_shape) = st.decode_f32(&bk)?;
                    if a_shape.len() == 2 && b_shape.len() == 2 && a_shape[1] == hidden_dim && b_shape[0] == dim {
                        let r = a_shape[0].min(b_shape[1]);
                        if r > 0 {
                            layer_map.insert(l, (r, a_vals, b_vals));
                        }
                    }
                }
            }
        }
    }

    for sl in new_layers {
        layer_map.insert(sl.layer, (sl.rank, sl.lora_a.clone(), sl.lora_b.clone()));
    }
    if layer_map.is_empty() {
        return Err("No layers to write into Topic PEFT LoRA".into());
    }

    let max_seen_rank = layer_map.values().map(|(r, _, _)| *r).max().unwrap_or(1);
    let out_rank = fixed_rank
        .map(|fr| fr.max(max_seen_rank))
        .unwrap_or(max_seen_rank)
        .clamp(1, MAX_LORA_RANK);

    let mut sorted_layers: Vec<usize> = layer_map.keys().copied().collect();
    sorted_layers.sort_unstable();

    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let cfg_json = format!(
        "{{\n  \"peft_type\": \"LORA\",\n  \"r\": {},\n  \"lora_alpha\": {:.1},\n  \"target_modules\": [\"down_proj\"]\n}}\n",
        out_rank, out_rank as f32
    );
    fs::write(out_dir.join("adapter_config.json"), cfg_json).map_err(|e| e.to_string())?;

    let mut body = Vec::<u8>::with_capacity(sorted_layers.len() * out_rank * (hidden_dim + dim) * 2);
    let mut header_entries = Vec::<String>::with_capacity(sorted_layers.len() * 2);

    for &l in &sorted_layers {
        let (r_src, a_src, b_src) = &layer_map[&l];
        let copy_r = (*r_src).min(out_rank);

        let a_start = body.len();
        for row_r in 0..out_rank {
            if row_r < copy_r {
                let row = &a_src[row_r * hidden_dim..(row_r + 1) * hidden_dim];
                for &v in row {
                    body.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                }
            } else {
                body.resize(body.len() + hidden_dim * 2, 0);
            }
        }
        let a_end = body.len();
        header_entries.push(format!(
            "\"base_model.model.model.layers.{}.mlp.down_proj.lora_A.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
            l, out_rank, hidden_dim, a_start, a_end
        ));

        let b_start = body.len();
        for row in 0..dim {
            for col_r in 0..out_rank {
                let val = if col_r < copy_r {
                    b_src[row * r_src + col_r]
                } else {
                    0.0
                };
                body.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
            }
        }
        let b_end = body.len();
        header_entries.push(format!(
            "\"base_model.model.model.layers.{}.mlp.down_proj.lora_B.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
            l, dim, out_rank, b_start, b_end
        ));
    }

    let mut hdr_json = format!("{{{}}}", header_entries.join(","));
    while hdr_json.len() % 8 != 0 {
        hdr_json.push(' ');
    }
    let mut file_bytes = Vec::with_capacity(8 + hdr_json.len() + body.len());
    file_bytes.extend_from_slice(&(hdr_json.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(hdr_json.as_bytes());
    file_bytes.extend_from_slice(&body);
    let st_bytes = file_bytes.len();
    fs::write(out_dir.join("adapter_model.safetensors"), file_bytes).map_err(|e| e.to_string())?;

    let denom = new_layers.len().max(1) as f64;
    let mean_fit = new_layers.iter().map(|s| s.edit_fit_rel_err).sum::<f64>() / denom;
    let mean_leak = new_layers.iter().map(|s| s.bg_leak_rms).sum::<f64>() / denom;
    let mean_frob = new_layers.iter().map(|s| s.delta_frob).sum::<f64>() / denom;

    Ok(TopicLoraWriteSummary {
        out_dir: out_dir.display().to_string(),
        layers: sorted_layers,
        rank: out_rank,
        safetensors_bytes: st_bytes,
        mean_fit_rel_err: mean_fit,
        mean_bg_leak_rms: mean_leak,
        mean_delta_frob: mean_frob,
    })
}

// -----------------------------------------------------------------------------
// Zero-Crate Topic Bank Manifest (`bank_manifest.json`) & GPU Hit Resolver
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TopicEntry {
    pub title: String,
    pub slug: String,
    pub lora_dir: String,
    pub num_facts: usize,
    pub rank: usize,
    pub layers: Vec<usize>,
    pub mean_fit_err: f64,
    pub kl_nats: f64,
}

#[derive(Clone, Debug)]
pub struct TopicBankManifest {
    pub bank_dir: PathBuf,
    pub router_qvec: String,
    pub min_score: f32,
    pub system_prompt: String,
    pub topics: Vec<TopicEntry>,
}

pub(crate) fn json_find_str(src: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let idx = src.find(&pat)?;
    let rest = src[idx + pat.len()..]
        .trim_start()
        .strip_prefix(':')?
        .trim_start()
        .strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Some(out);
        }
        if c == '\\' {
            match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                other => out.push(other),
            }
        } else {
            out.push(c);
        }
    }
    None
}

pub(crate) fn json_find_num(src: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\"", key);
    let idx = src.find(&pat)?;
    let rest = src[idx + pat.len()..].trim_start().strip_prefix(':')?.trim_start();
    let num_s: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
        .collect();
    num_s.parse::<f64>().ok()
}

fn json_find_usize_arr(src: &str, key: &str) -> Vec<usize> {
    let pat = format!("\"{}\"", key);
    let Some(idx) = src.find(&pat) else {
        return Vec::new();
    };
    let Some(rest) = src[idx + pat.len()..].trim_start().strip_prefix(':') else {
        return Vec::new();
    };
    let rest = rest.trim_start();
    let Some(inner_start) = rest.strip_prefix('[') else {
        return Vec::new();
    };
    let Some(end) = inner_start.find(']') else {
        return Vec::new();
    };
    inner_start[..end]
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect()
}

fn split_json_object_array(arr_body: &str) -> Vec<&str> {
    let b = arr_body.as_bytes();
    let mut objs = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'"' {
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                } else if b[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if b[i] == b'{' {
            let start = i;
            let mut depth = 1usize;
            i += 1;
            while i < b.len() && depth > 0 {
                if b[i] == b'"' {
                    i += 1;
                    while i < b.len() {
                        if b[i] == b'\\' {
                            i += 2;
                        } else if b[i] == b'"' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                    continue;
                }
                if b[i] == b'{' {
                    depth += 1;
                } else if b[i] == b'}' {
                    depth -= 1;
                }
                i += 1;
            }
            objs.push(&arr_body[start..i]);
            continue;
        }
        i += 1;
    }
    objs
}

impl TopicBankManifest {
    pub fn load(bank_dir: &Path) -> Result<Self, String> {
        let manifest_path = if bank_dir.is_file() {
            bank_dir.to_path_buf()
        } else {
            bank_dir.join("bank_manifest.json")
        };
        let resolved_bank_dir = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let txt = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed reading {}: {}", manifest_path.display(), e))?;

        let topics_pos = txt
            .find("\"topics\"")
            .ok_or_else(|| format!("Missing \"topics\" array in {}", manifest_path.display()))?;
        let header_part = &txt[..topics_pos];

        let router_qvec = json_find_str(header_part, "router_qvec")
            .or_else(|| json_find_str(&txt, "router_qvec"))
            .ok_or_else(|| format!("Missing \"router_qvec\" in {}", manifest_path.display()))?;
        let min_score = json_find_num(header_part, "min_score")
            .or_else(|| json_find_num(&txt, "min_score"))
            .unwrap_or(0.30) as f32;
        let system_prompt = json_find_str(header_part, "system_prompt")
            .or_else(|| json_find_str(&txt, "system_prompt"))
            .unwrap_or_else(|| "Answer the question concisely in one short phrase or sentence.".to_string());

        let arr_tail = &txt[topics_pos + 8..];
        let arr_open = arr_tail
            .find('[')
            .ok_or_else(|| "Malformed \"topics\" array".to_string())?;
        let arr_slice = &arr_tail[arr_open + 1..];
        // Find matching `]` outside string literals
        let ab = arr_slice.as_bytes();
        let mut depth = 1usize;
        let mut arr_end = ab.len();
        let mut idx = 0usize;
        while idx < ab.len() {
            if ab[idx] == b'"' {
                idx += 1;
                while idx < ab.len() {
                    if ab[idx] == b'\\' {
                        idx += 2;
                    } else if ab[idx] == b'"' {
                        idx += 1;
                        break;
                    } else {
                        idx += 1;
                    }
                }
                continue;
            }
            if ab[idx] == b'[' {
                depth += 1;
            } else if ab[idx] == b']' {
                depth -= 1;
                if depth == 0 {
                    arr_end = idx;
                    break;
                }
            }
            idx += 1;
        }

        let mut topics = Vec::new();
        for obj in split_json_object_array(&arr_slice[..arr_end]) {
            let Some(title) = json_find_str(obj, "title") else {
                continue;
            };
            let lora_dir = json_find_str(obj, "lora_dir").unwrap_or_default();
            let slug = json_find_str(obj, "slug").unwrap_or_else(|| title.to_lowercase().replace(' ', "_"));
            let num_facts = json_find_num(obj, "num_facts").unwrap_or(0.0) as usize;
            let rank = json_find_num(obj, "rank").unwrap_or(0.0) as usize;
            let layers = json_find_usize_arr(obj, "layers");
            let mean_fit_err = json_find_num(obj, "mean_fit_err").unwrap_or(0.0);
            let kl_nats = json_find_num(obj, "kl_nats").unwrap_or(0.0);
            topics.push(TopicEntry {
                title,
                slug,
                lora_dir,
                num_facts,
                rank,
                layers,
                mean_fit_err,
                kl_nats,
            });
        }
        if topics.is_empty() {
            return Err(format!("No topic entries found in {}", manifest_path.display()));
        }

        Ok(Self {
            bank_dir: resolved_bank_dir,
            router_qvec,
            min_score,
            system_prompt,
            topics,
        })
    }

    pub fn resolve_router_qvec_path(&self) -> PathBuf {
        let p = Path::new(&self.router_qvec);
        if p.is_absolute() {
            return p.to_path_buf();
        }
        let rel = self.bank_dir.join(p);
        if rel.exists() {
            rel
        } else {
            p.to_path_buf()
        }
    }

    pub fn resolve_topic_lora_path(&self, entry: &TopicEntry) -> PathBuf {
        let p = Path::new(&entry.lora_dir);
        if p.is_absolute() {
            return p.to_path_buf();
        }
        let rel = self.bank_dir.join(p);
        if rel.exists() {
            rel
        } else {
            p.to_path_buf()
        }
    }

    pub fn resolve_topic<'a>(
        &self,
        hits: impl IntoIterator<Item = (&'a str, f32)>,
        min_score_override: Option<f32>,
    ) -> Option<(&TopicEntry, f32)> {
        let thresh = min_score_override.unwrap_or(self.min_score);
        for (title, score) in hits {
            if score < thresh {
                continue;
            }
            if let Some(entry) = self
                .topics
                .iter()
                .find(|t| t.title.eq_ignore_ascii_case(title.trim()))
            {
                return Some((entry, score));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::prepare_lora_from_path;

    fn lcg_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0;
            out.push(u * 2.0 - 1.0);
        }
        out
    }

    #[test]
    fn test_cholesky_and_alphaedit_exact_fit() {
        let d_ff = 128usize;
        let dim = 32usize;
        let n = 8usize;
        let n_bg = 24usize;
        let k1 = lcg_vec(n * d_ff, 11);
        let r = lcg_vec(n * dim, 22);
        let k0 = lcg_vec(n_bg * d_ff, 33);

        let no_bg = solve_alphaedit_lora_layer(35, &k1, &r, &k0, d_ff, dim, 0.0, 1e-4, 32).unwrap();
        assert!(
            no_bg.edit_fit_rel_err < 1e-2,
            "Expected tight edit fit, got {}",
            no_bg.edit_fit_rel_err
        );

        let with_bg = solve_alphaedit_lora_layer(35, &k1, &r, &k0, d_ff, dim, 200.0, 1e-4, 32).unwrap();
        assert!(
            with_bg.edit_fit_rel_err < 5e-2,
            "Expected fit with null-space reg, got {}",
            with_bg.edit_fit_rel_err
        );
        assert!(
            with_bg.bg_leak_rms * 20.0 < no_bg.bg_leak_rms,
            "Expected >20x bg suppression (no_bg={}, with_bg={})",
            no_bg.bg_leak_rms,
            with_bg.bg_leak_rms
        );
    }

    #[test]
    fn test_rank_compression_eckart_young() {
        let d_ff = 128usize;
        let dim = 64usize;
        let n = 48usize;
        let true_rank = 12usize;

        let k1 = lcg_vec(n * d_ff, 44);
        // Generate ground-truth rank-12 target `R = W_true @ K1^T`
        let b_true = lcg_vec(dim * true_rank, 55);
        let a_true = lcg_vec(true_rank * d_ff, 66);
        let mut w_true = vec![0.0f32; dim * d_ff];
        let mut r_rows = vec![0.0f32; n * dim];
        for i in 0..dim {
            for c in 0..d_ff {
                let mut s = 0.0f32;
                for rr in 0..true_rank {
                    s += b_true[i * true_rank + rr] * a_true[rr * d_ff + c];
                }
                w_true[i * d_ff + c] = s * 0.1;
            }
        }
        for p in 0..n {
            for i in 0..dim {
                let mut s = 0.0f32;
                for c in 0..d_ff {
                    s += w_true[i * d_ff + c] * k1[p * d_ff + c];
                }
                r_rows[p * dim + i] = s;
            }
        }

        let solved = solve_alphaedit_lora_layer(35, &k1, &r_rows, &[], d_ff, dim, 0.0, 1e-6, 16).unwrap();
        assert_eq!(solved.rank, 16);
        assert!(
            solved.edit_fit_rel_err < 1e-2,
            "Expected tight compressed fit on rank-12 matrix, got {}",
            solved.edit_fit_rel_err
        );
        for &v in solved.lora_a.iter().chain(solved.lora_b.iter()) {
            assert!(v.is_finite() && v.abs() < 1000.0);
        }
    }

    #[test]
    fn test_qmkv_roundtrip_and_peft_merge() {
        let tmp = std::env::temp_dir().join("qwen_nv_test_topic_lora");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let layers = vec![34usize, 35usize];
        let hidden_dim = 64usize;
        let dim = 32usize;
        let num_positions = 4usize;
        let keys = lcg_vec(num_positions * layers.len() * hidden_dim, 77);
        let final_res = lcg_vec(num_positions * dim, 88);

        let qmkv = QmkvFile {
            layers: layers.clone(),
            hidden_dim,
            dim,
            num_positions,
            keys: keys.clone(),
            final_res: final_res.clone(),
        };
        let qmkv_path = tmp.join("test.qmkv");
        qmkv.write(&qmkv_path).unwrap();
        let read_back = QmkvFile::read(&qmkv_path).unwrap();
        assert_eq!(read_back.layers, layers);
        assert_eq!(read_back.keys, keys);
        assert_eq!(read_back.final_res, final_res);

        let k34 = read_back.extract_layer_keys(34).unwrap();
        let k35 = read_back.extract_layer_keys(35).unwrap();
        let r34 = solve_alphaedit_lora_layer(34, &k34, &final_res, &[], hidden_dim, dim, 0.0, 1e-3, 8).unwrap();
        let lora_dir = tmp.join("adapter");
        write_or_merge_topic_peft_lora(&lora_dir, &[r34], None, dim, hidden_dim, None).unwrap();

        let r35 = solve_alphaedit_lora_layer(35, &k35, &final_res, &[], hidden_dim, dim, 0.0, 1e-3, 8).unwrap();
        let sum = write_or_merge_topic_peft_lora(&lora_dir, &[r35], Some(&lora_dir), dim, hidden_dim, None).unwrap();
        assert_eq!(sum.layers, vec![34, 35]);

        let prep = prepare_lora_from_path(lora_dir.to_str().unwrap(), 1.0, 36, dim, hidden_dim, 32, 16).unwrap();
        assert_eq!(prep.num_adapted_tensors, 4);
        assert!((prep.scale - 1.0).abs() < 1e-5);
        assert_eq!(prep.layer_descs[34].active_mask, 1 << 6);
        assert_eq!(prep.layer_descs[35].active_mask, 1 << 6);

        let _ = fs::remove_dir_all(&tmp);
    }
}
