//! Small BERT encoder for MiniLM on candle, written for GPU throughput
//! (fork-only, feature `cuda`).
//!
//! candle-transformers' `BertModel` runs every Linear as a batched GEMM with
//! one small matrix per sequence and uses the multi-pass softmax. Here every
//! Linear runs as one 2-D GEMM over all tokens of the batch, Q, K and V share
//! one fused GEMM, and softmax and LayerNorm use candle's fused kernels.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub(crate) struct BertConfig {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) layers: usize,
    pub(crate) intermediate: usize,
    pub(crate) max_positions: usize,
    pub(crate) eps: f32,
}

impl BertConfig {
    pub(crate) fn from_json(text: &str) -> std::result::Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let get = |k: &str| {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("config.json has no integer {k}"))
        };
        if v.get("hidden_act").and_then(|a| a.as_str()) != Some("gelu") {
            return Err("config.json hidden_act is not gelu".into());
        }
        if v.get("position_embedding_type").and_then(|a| a.as_str()) .is_some_and(|p| p != "absolute") {
            return Err("config.json position_embedding_type is not absolute".into());
        }
        Ok(Self {
            hidden: get("hidden_size")?,
            heads: get("num_attention_heads")?,
            layers: get("num_hidden_layers")?,
            intermediate: get("intermediate_size")?,
            max_positions: get("max_position_embeddings")?,
            eps: v
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-12) as f32,
        })
    }
}

/// `y = x W^T + b` with `W^T` stored contiguous, for 2-D `x`.
struct Dense {
    wt: Tensor,
    b: Tensor,
}

impl Dense {
    fn load(vb: &VarBuilder, name: &str, out: usize, inp: usize) -> Result<Self> {
        let w = vb.get((out, inp), &format!("{name}.weight"))?;
        let b = vb.get(out, &format!("{name}.bias"))?;
        Ok(Self { wt: w.t()?.contiguous()?, b })
    }

    fn fused(parts: &[Self]) -> Result<Self> {
        let wts: Vec<&Tensor> = parts.iter().map(|p| &p.wt).collect();
        let bs: Vec<&Tensor> = parts.iter().map(|p| &p.b).collect();
        Ok(Self { wt: Tensor::cat(&wts, 1)?.contiguous()?, b: Tensor::cat(&bs, 0)? })
    }
}

impl Module for Dense {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.wt)?.broadcast_add(&self.b)
    }
}

struct Norm {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl Norm {
    fn load(vb: &VarBuilder, name: &str, h: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            w: vb.get(h, &format!("{name}.weight"))?,
            b: vb.get(h, &format!("{name}.bias"))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm(&x.contiguous()?, &self.w, &self.b, self.eps)
    }
}

struct Layer {
    qkv: Dense,
    out: Dense,
    ln1: Norm,
    inter: Dense,
    ffn_out: Dense,
    ln2: Norm,
}

pub(crate) struct MiniBert {
    word: Tensor,
    pos: Tensor,
    type0: Tensor,
    ln: Norm,
    layers: Vec<Layer>,
    cfg: BertConfig,
}

impl MiniBert {
    pub(crate) fn load(vb: VarBuilder, cfg: BertConfig) -> Result<Self> {
        let h = cfg.hidden;
        // sentence-transformers exports have no "bert." prefix; HF BertModel ones may.
        let vb = if vb.contains_tensor("embeddings.word_embeddings.weight") {
            vb
        } else {
            vb.pp("bert")
        };
        let word = vb
            .pp("embeddings.word_embeddings")
            .get_unchecked("weight")?;
        let pos = vb.get((cfg.max_positions, h), "embeddings.position_embeddings.weight")?;
        let types = vb.pp("embeddings.token_type_embeddings").get_unchecked("weight")?;
        let type0 = types.get(0)?;
        let ln = Norm::load(&vb, "embeddings.LayerNorm", h, cfg.eps)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("encoder.layer.{i}");
            let q = Dense::load(&vb, &format!("{p}.attention.self.query"), h, h)?;
            let k = Dense::load(&vb, &format!("{p}.attention.self.key"), h, h)?;
            let v = Dense::load(&vb, &format!("{p}.attention.self.value"), h, h)?;
            layers.push(Layer {
                qkv: Dense::fused(&[q, k, v])?,
                out: Dense::load(&vb, &format!("{p}.attention.output.dense"), h, h)?,
                ln1: Norm::load(&vb, &format!("{p}.attention.output.LayerNorm"), h, cfg.eps)?,
                inter: Dense::load(&vb, &format!("{p}.intermediate.dense"), cfg.intermediate, h)?,
                ffn_out: Dense::load(&vb, &format!("{p}.output.dense"), h, cfg.intermediate)?,
                ln2: Norm::load(&vb, &format!("{p}.output.LayerNorm"), h, cfg.eps)?,
            });
        }
        Ok(Self { word, pos, type0, ln, layers, cfg })
    }

    /// `ids` is `[n, len]` u32; `mask` is `[n, len]` with 1 for real tokens, or
    /// `None` when no row is padded. Returns `[n, len, hidden]` in the model dtype.
    pub(crate) fn forward(&self, ids: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (n, len) = ids.dims2()?;
        let h = self.cfg.hidden;
        let heads = self.cfg.heads;
        let hd = h / heads;
        let dtype = self.word.dtype();
        let emb = self.word.index_select(&ids.flatten_all()?, 0)?.reshape((n, len, h))?;
        let emb = emb
            .broadcast_add(&self.pos.narrow(0, 0, len)?)?
            .broadcast_add(&self.type0)?
            .reshape((n * len, h))?;
        let mut x = self.ln.forward(&emb)?;
        // Additive mask over keys: 0 for real tokens, f32::MIN for padding.
        let bias = match mask {
            Some(m) => Some(
                ((m.ones_like()? - m)? * f64::from(f32::MIN))?
                    .to_dtype(dtype)?
                    .reshape((n, 1, 1, len))?,
            ),
            None => None,
        };
        let scale = 1.0 / (hd as f64).sqrt();
        for layer in &self.layers {
            let qkv = layer.qkv.forward(&x)?.reshape((n, len, 3, heads, hd))?;
            let q = (qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()? * scale)?;
            let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let mut scores = q.matmul(&k.t()?)?; // [n, heads, len, len]
            if let Some(b) = &bias {
                scores = scores.broadcast_add(b)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs
                .matmul(&v)? // [n, heads, len, hd]
                .transpose(1, 2)?
                .contiguous()?
                .reshape((n * len, h))?;
            let attn = layer.out.forward(&ctx)?;
            x = layer.ln1.forward(&(attn + &x)?)?;
            let inner = layer.inter.forward(&x)?.gelu_erf()?;
            let ffn = layer.ffn_out.forward(&inner)?;
            x = layer.ln2.forward(&(ffn + &x)?)?;
        }
        x.reshape((n, len, h))
    }
}
