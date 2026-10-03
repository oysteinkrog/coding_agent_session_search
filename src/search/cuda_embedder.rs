//! Fork-only CUDA acceleration for the MiniLM quality embedder (feature `cuda`).
//!
//! The CPU producer stays in charge. In a `cuda` build, `FastEmbedder` loads
//! the upstream frankensearch F32 MiniLM producer (`AllMiniLmL6V2F32`) on the
//! CPU. That producer owns the embedding identity, embeds single texts
//! (queries), and is the fallback. Batches of at least `CASS_CUDA_MIN_BATCH`
//! texts (default 8) go to a candle BERT on the GPU instead
//! (`cuda_minibert.rs`).
//!
//! The GPU is started lazily, on the first large batch, so a short-lived
//! search process never touches CUDA. At start the GPU output for a fixed set
//! of texts is compared with the CPU F32 output; if any value differs by more
//! than `MAX_ABS_DIFF`, or CUDA cannot start, cass logs a warning and stays on
//! the CPU. A GPU error during a batch also falls back to the CPU for that
//! batch and turns the GPU off for the rest of the process.
//!
//! The GPU vectors equal the CPU F32 vectors to f32 rounding, not bit for
//! bit. Upstream certifies producers by exact bits; this fork accepts the
//! tolerance check instead, so GPU-made and CPU-made vectors share one
//! producer identity and the same index serves queries from either.
//!
//! Set `CASS_EMBED_DEVICE=cpu` to turn the GPU off.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use super::cuda_minibert::{BertConfig, MiniBert};
use tokenizers::Tokenizer;
use frankensearch::NativeEmbedder;
use frankensearch::core::traits::SyncEmbed;

/// Same limit as frankensearch `DEFAULT_MAX_LENGTH`.
const MAX_LENGTH: usize = 512;
const DIM: usize = 384;
/// Largest accepted difference between GPU and CPU F32 output values at start.
/// Measured differences are far below this (see the port notes); the margin
/// covers driver and cuBLAS version changes, not a different model.
const MAX_ABS_DIFF: f32 = 1.0e-4;
/// Upper bound on padded tokens (rows x longest row) in one forward pass.
/// 32 K tokens keeps the attention scores (12 heads x rows x len^2 x 4 bytes)
/// under about 1 GB at length 512.
const DEFAULT_TOKEN_BUDGET: usize = 32 * 1024;
const MAX_ROWS: usize = 1024;
const DEFAULT_MIN_BATCH: usize = 8;

/// Lazily started GPU path for one loaded MiniLM model directory.
pub(crate) struct CudaAccel {
    model_dir: PathBuf,
    state: OnceLock<Option<CudaMiniLm>>,
    disabled: AtomicBool,
    min_batch: usize,
}

impl CudaAccel {
    pub(crate) fn new(model_dir: &Path) -> Self {
        let off = dotenvy::var("CASS_EMBED_DEVICE")
            .map(|v| v.trim().eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        let min_batch = dotenvy::var("CASS_CUDA_MIN_BATCH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_MIN_BATCH);
        Self {
            model_dir: model_dir.to_path_buf(),
            state: OnceLock::new(),
            disabled: AtomicBool::new(off),
            min_batch,
        }
    }

    /// True once the GPU has started and passed the start check.
    pub(crate) fn is_active(&self) -> bool {
        !self.disabled.load(Ordering::Relaxed) && matches!(self.state.get(), Some(Some(_)))
    }

    /// Embed on the GPU, or return `None` so the caller uses the CPU producer.
    pub(crate) fn try_embed(
        &self,
        reference: &NativeEmbedder,
        texts: &[&str],
    ) -> Option<Vec<Vec<f32>>> {
        if texts.len() < self.min_batch || self.disabled.load(Ordering::Relaxed) {
            return None;
        }
        let gpu = self
            .state
            .get_or_init(|| match start_checked(&self.model_dir, reference) {
                Ok((gpu, diff)) => {
                    tracing::info!(
                        device = %gpu.device_label(),
                        max_abs_diff = diff,
                        "cuda MiniLM embedder started; matches the CPU F32 producer"
                    );
                    Some(gpu)
                }
                Err(reason) => {
                    tracing::warn!(%reason, "cuda MiniLM embedder unavailable; using CPU");
                    None
                }
            })
            .as_ref()?;
        match gpu.embed_batch(texts) {
            Ok(vectors) => Some(vectors),
            Err(reason) => {
                tracing::warn!(%reason, "cuda MiniLM batch failed; using CPU from now on");
                self.disabled.store(true, Ordering::Relaxed);
                None
            }
        }
    }
}

/// Texts for the start check: the frozen conformance corpus plus texts that
/// exercise padding (mixed lengths in one batch) and truncation (over 512
/// tokens).
fn check_texts() -> Vec<String> {
    let mut texts: Vec<String> = frankensearch::embed::model_manifest::MODEL_CONFORMANCE_TEXTS_V1
        .iter()
        .map(|t| (*t).to_owned())
        .collect();
    let sentence = "The indexer embeds each canonical message, then writes the vector \
                    shard and publishes it after the lexical generation. ";
    for repeat in [1usize, 3, 9, 40] {
        texts.push(sentence.repeat(repeat));
    }
    texts.push("fn main() { println!(\"{}\", 2 + 2); } // café 東京 résumé".to_owned());
    texts
}

fn start_checked(model_dir: &Path, reference: &NativeEmbedder) -> Result<(CudaMiniLm, f32), String> {
    let gpu = CudaMiniLm::load(model_dir, 0)?;
    let owned = check_texts();
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let expected = reference
        .embed_batch_sync(&texts)
        .map_err(|e| format!("cpu reference embed: {e}"))?;
    let actual = gpu.embed_batch(&texts)?;
    let diff = max_abs_diff(&actual, &expected)
        .ok_or_else(|| "gpu output shape or values are invalid".to_owned())?;
    if diff > MAX_ABS_DIFF {
        return Err(format!(
            "gpu output differs from the CPU F32 producer by {diff:e} (limit {MAX_ABS_DIFF:e})"
        ));
    }
    Ok((gpu, diff))
}

/// candle BERT MiniLM on one CUDA device.
///
/// Reproduces the frankensearch native MiniLM input and output contract: the
/// same `tokenizer.json`, truncation to 512 tokens, token type ids all 0, mean
/// pooling over every real token (specials included) and L2 normalization
/// when the norm is above zero. Batches are padded and masked, so padding
/// never enters the mean. The math is plain f32 (no TF32, no f16).
pub(crate) struct CudaMiniLm {
    model: Mutex<MiniBert>,
    device: Device,
    tokenizer: Tokenizer,
    token_budget: usize,
}

fn err(context: &str, error: impl std::fmt::Display) -> String {
    format!("{context}: {error}")
}

impl CudaMiniLm {
    /// Load on CUDA device `ordinal`. `model_dir` must already have passed the
    /// frankensearch manifest check; the CPU producer load does that.
    pub(crate) fn load(model_dir: &Path, ordinal: usize) -> Result<Self, String> {
        let device = Device::new_cuda(ordinal).map_err(|e| err("cuda device", e))?;
        let config_text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| err("read config.json", e))?;
        let config =
            BertConfig::from_json(&config_text).map_err(|e| err("parse config.json", e))?;
        if config.hidden != DIM {
            return Err(format!("config.json hidden_size {} is not {DIM}", config.hidden));
        }
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| err("load tokenizer.json", e))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| err("tokenizer truncation", e))?;
        tokenizer.with_padding(None);

        // Read (not mmap) the weights: cass denies unsafe code, and the file is
        // only 91 MB. The manifest check has already hashed it.
        let weights = std::fs::read(model_dir.join("model.safetensors"))
            .map_err(|e| err("read model.safetensors", e))?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)
            .map_err(|e| err("parse model.safetensors", e))?;
        let model = MiniBert::load(vb, config).map_err(|e| err("build MiniBert", e))?;
        let token_budget = dotenvy::var("CASS_CUDA_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v >= MAX_LENGTH)
            .unwrap_or(DEFAULT_TOKEN_BUDGET);
        Ok(Self {
            model: Mutex::new(model),
            device,
            tokenizer,
            token_budget,
        })
    }

    pub(crate) fn device_label(&self) -> String {
        format!("{:?}", self.device.location())
    }

    /// Embed a batch; output order matches `texts`.
    pub(crate) fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| err("tokenize", e))?;
        let ids: Vec<Vec<u32>> = encodings
            .iter()
            .map(|enc| enc.get_ids().iter().take(MAX_LENGTH).copied().collect())
            .collect();

        // Sort by length so each forward pass pads as little as possible.
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| ids[i].len());

        let mut out: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
        let model = self
            .model
            .lock()
            .map_err(|e| err("cuda embedder mutex", e))?;
        let mut start = 0;
        while start < order.len() {
            // Rows are sorted by length, so the newest row is the longest.
            let mut end = start + 1;
            while end < order.len()
                && end - start < MAX_ROWS
                && (end - start + 1) * ids[order[end]].len().max(1) <= self.token_budget
            {
                end += 1;
            }
            let chunk = &order[start..end];
            let rows: Vec<&[u32]> = chunk.iter().map(|&i| ids[i].as_slice()).collect();
            let vectors = self.forward_chunk(&model, &rows)?;
            for (&row, vector) in chunk.iter().zip(vectors) {
                out[row] = Some(vector);
            }
            start = end;
        }
        drop(model);
        out.into_iter()
            .map(|v| v.ok_or_else(|| "cuda embedder lost a row".to_owned()))
            .collect()
    }

    fn forward_chunk(&self, model: &MiniBert, rows: &[&[u32]]) -> Result<Vec<Vec<f32>>, String> {
        let n = rows.len();
        let len = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
        let mut id_buf = vec![0u32; n * len];
        let mut mask_buf = vec![0f32; n * len];
        for (r, row) in rows.iter().enumerate() {
            id_buf[r * len..r * len + row.len()].copy_from_slice(row);
            mask_buf[r * len..r * len + row.len()].fill(1.0);
        }
        let run = || -> candle_core::Result<Vec<Vec<f32>>> {
            let input_ids = Tensor::from_vec(id_buf, (n, len), &self.device)?;
            let mask = Tensor::from_vec(mask_buf, (n, len), &self.device)?;
            let padded = rows.iter().any(|r| r.len() != len);
            let hidden = model.forward(&input_ids, padded.then_some(&mask))?; // [n, len, 384]
            let summed = hidden.broadcast_mul(&mask.unsqueeze(2)?)?.sum(1)?; // [n, 384]
            let counts = mask.sum_keepdim(1)?; // [n, 1]
            summed.broadcast_div(&counts)?.to_vec2::<f32>()
        };
        let mut vectors = run().map_err(|e| err("cuda forward", e))?;
        for v in &mut vectors {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
        }
        Ok(vectors)
    }
}

/// Largest absolute difference between two vector sets of the same shape;
/// `None` when the shapes differ or a value is not finite.
pub(crate) fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        if x.len() != y.len() {
            return None;
        }
        for (p, q) in x.iter().zip(y) {
            let d = (p - q).abs();
            if !d.is_finite() {
                return None;
            }
            worst = worst.max(d);
        }
    }
    Some(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_abs_diff_rejects_shape_mismatch_and_nan() {
        let a = vec![vec![1.0, 0.0]];
        assert_eq!(max_abs_diff(&a, &a), Some(0.0));
        assert_eq!(max_abs_diff(&a, &[vec![1.0]]), None);
        assert_eq!(max_abs_diff(&a, &[]), None);
        assert_eq!(max_abs_diff(&a, &[vec![f32::NAN, 0.0]]), None);
        assert!((max_abs_diff(&a, &[vec![0.5, 0.25]]).unwrap() - 0.5).abs() < f32::EPSILON);
    }

    /// Needs a GPU and a model dir in `CASS_TEST_MINILM_DIR`; skipped otherwise.
    #[test]
    fn gpu_matches_cpu_f32_producer() {
        let Ok(dir) = std::env::var("CASS_TEST_MINILM_DIR") else {
            eprintln!("skipped: CASS_TEST_MINILM_DIR not set");
            return;
        };
        let dir = PathBuf::from(dir);
        let cpu = NativeEmbedder::load_model(
            &dir,
            frankensearch::NativeEmbeddingModel::AllMiniLmL6V2F32,
        )
        .expect("cpu f32 producer");
        let (_gpu, diff) = start_checked(&dir, &cpu).expect("gpu start check");
        assert!(diff <= MAX_ABS_DIFF);
    }
}
