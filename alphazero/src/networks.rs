use tch::nn;
use tch::{Tensor, Kind, Device};


// Bert base architecture (roughly)
pub const BERT_BASE_CONFIG: Config = Config {
    input_dim: 64,          // 64 squares, 1 piece per square
    embedding_dim: 832,     // 6 piece types per side plus one for no piece, 64 squares ((6 * 2) + 1) * 64 = 832
    hidden_dim: 768,
    n_blocks: 8,
    n_heads: 12,
    mlp_expansion_factor: 4,
    dropout_rate: 0.1,
    policy_mlp_dim: 256,
    value_mlp_dim: 256,
    move_dim: 1968,         // Number of possible chess moves as defined in move_map.rs
    replay_buffer_capacity: 100_000,
};


#[derive(Debug, Copy, Clone)]
pub struct Config {
    pub input_dim: i64,
    pub embedding_dim: i64,
    hidden_dim: i64,
    n_blocks: i64,
    n_heads: i64,
    mlp_expansion_factor: i64,
    dropout_rate: f64,
    policy_mlp_dim: i64,
    value_mlp_dim: i64,
    pub move_dim: i64,
    pub replay_buffer_capacity: i64,
}

#[derive(Debug)]
struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl nn::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.weight.tr()) + &self.bias
    }
}

fn linear(vs: nn::Path, input_dim: i64, output_dim: i64) -> Linear {
    Linear {
        weight: vs.kaiming_normal("weight", &[output_dim, input_dim]),
        bias: vs.zeros("bias", &[output_dim]),
    }
}

fn linear_no_bias(vs: nn::Path, input_dim: i64, output_dim: i64) -> Linear {
    Linear {
        weight: vs.kaiming_normal("weight", &[output_dim, input_dim]),
        bias: vs.zeros_no_train("bias", &[output_dim]),
    }
}


fn causal_self_attention(p: &nn::Path, cfg: Config) -> impl nn::ModuleT {
    let key   = linear(p / "key", cfg.hidden_dim, cfg.hidden_dim);
    let query = linear(p / "query", cfg.hidden_dim, cfg.hidden_dim);
    let value = linear(p / "value", cfg.hidden_dim, cfg.hidden_dim);

    let output_proj = linear(p / "output_proj", cfg.hidden_dim, cfg.hidden_dim);

    nn::func_t(move |xs, train| {
        let (batch_size, seq_len, _) = xs.size3().unwrap();
        let sizes = [batch_size, seq_len, cfg.n_heads, cfg.hidden_dim / cfg.n_heads];
        let k = xs.apply(&key).view(sizes).transpose(1, 2);
        let q = xs.apply(&query).view(sizes).transpose(1, 2);
        let v = xs.apply(&value).view(sizes).transpose(1, 2);

        let attn = q.matmul(&k.transpose(-2, -1)) / f64::sqrt(sizes[3] as f64);
        let attn = attn.softmax(-1, Kind::Float).dropout(cfg.dropout_rate, train);

        let ys = attn.matmul(&v).transpose(1, 2).contiguous().view([batch_size, seq_len, cfg.hidden_dim]);
        
        ys.apply(&output_proj).dropout(cfg.dropout_rate, train)
    })
}


fn encoder_block(p: &nn::Path, cfg: Config) -> impl nn::ModuleT {
    let layer_norm1 = nn::layer_norm(p / "layer_norm1", vec![cfg.hidden_dim], Default::default());
    let layer_norm2 = nn::layer_norm(p / "layer_norm2", vec![cfg.hidden_dim], Default::default());
    let attn = causal_self_attention(p, cfg);

    let linear1 = linear(p / "linear1", cfg.hidden_dim, cfg.hidden_dim * cfg.mlp_expansion_factor);
    let linear2 = linear_no_bias(p / "linear2", cfg.hidden_dim * cfg.mlp_expansion_factor, cfg.hidden_dim);

    nn::func_t(move |xs, train| {
        let xs = xs + xs.apply(&layer_norm1).apply_t(&attn, train);
        let ys = xs.apply(&layer_norm2)
                   .apply(&linear1).gelu("none")
                   .apply(&linear2).dropout(cfg.dropout_rate, train);
        xs + ys
    })
}

pub fn chess_transformer(path: &nn::Path, cfg: Config) -> nn::SequentialT {
    let piece_embedding = nn::embedding(
        path / "piece_embedding", 
        cfg.embedding_dim, 
        cfg.hidden_dim, 
        Default::default(),
        );
    let layer_norm_final = nn::layer_norm(path / "layer_norm_final", vec![cfg.hidden_dim], Default::default()); 
    let mut encoder = nn::seq_t();
    for block_idx in 0..cfg.n_blocks {
        encoder = encoder.add(encoder_block(&(path / block_idx), cfg));
    }
    nn::seq_t().add(
    nn::func_t(move |xs, train| {
        let embeddings = xs.apply(&piece_embedding);
        let ys = embeddings.apply_t(&encoder, train);
        ys.apply(&layer_norm_final)
    }))
    .add(
        nn::func_t(move |xs, _| {
            xs.mean_dim(Some([1i64].as_slice()), false, Kind::Float)
        }))
}


pub fn policy_mlp(p: &nn::Path, cfg: Config) -> nn::SequentialT {
    // Wait to apply softmax, might want to use cross entropy loss which does a log softmax.
    let linear1 = linear(p / "linear1", cfg.hidden_dim, cfg.policy_mlp_dim);
    let linear2 = linear(p / "linear2", cfg.policy_mlp_dim, cfg.move_dim);
    nn::seq_t().add(
    nn::func_t(move |xs, _| {
        xs.apply(&linear1).gelu("none").apply(&linear2)
    })
    )
}

pub fn value_mlp(p: &nn::Path, cfg: Config) -> nn::SequentialT {
    let linear1 = linear(p / "linear1", cfg.hidden_dim, cfg.value_mlp_dim);
    let linear2 = linear(p / "linear2", cfg.value_mlp_dim, 1);
    nn::seq_t().add(
    nn::func_t(move |xs, _| {
        xs.apply(&linear1).gelu("none").apply(&linear2).tanh()
    })
    )
}
