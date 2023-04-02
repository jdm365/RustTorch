use progress_bar::*;

use crate::replay_buffer::*;
use tch::nn;
use tch::{ Tensor, Kind, Device, nn::OptimizerConfig, TchError };//, autocast };



#[derive(Debug, Copy, Clone)]
pub struct Config {
    pub input_dim: i64,
    pub embedding_dim: i64,
    pub hidden_dim: i64,
    pub n_blocks: i64,
    pub n_heads: i64,
    pub mlp_expansion_factor: i64,
    pub dropout_rate: f64,
    pub policy_mlp_dim: i64,
    pub value_mlp_dim: i64,
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


pub struct Networks {
    state_processing_network: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head:  nn::SequentialT,
    optimizer: nn::Optimizer,
    var_store: nn::VarStore,
}



impl Networks {
    pub fn new(cfg: Config) -> Self {
        let var_store = nn::VarStore::new(Device::cuda_if_available());
        let state_processing_network = chess_transformer(
            &(var_store.root() / "state_processing_network"), 
            cfg 
            );
        let policy_head = policy_mlp(
            &(var_store.root() / "policy_head"), 
            cfg 
            );
        let value_head = value_mlp(
            &(var_store.root() / "value_head"), 
            cfg 
            );

        let optimizer = nn::Adam::default().build(&var_store, 1e-3).expect("Optimizer failed to build");
        Networks {
            state_processing_network,
            policy_head,
            value_head,
            optimizer,
            var_store,
        }
    }

    pub fn forward(&self, board: [i32; 64], train: bool) -> Option<(Vec<f32>, f32)> {
        // Get probabilities of all moves from policy net.
        // FuncT::forward_t() -> (&Tensor, train: bool) -> Tensor
        let tensor_board = self.get_tensor_board(board, true);
        
        let (probs_tensor, value_tensor) = self.tensor_forward(tensor_board, train).expect("Forward pass failed");

        let probs = Vec::from(probs_tensor.to_kind(Kind::Float).to_device(tch::Device::Cpu));
        let value = f32::from(value_tensor.to_kind(Kind::Float).to_device(tch::Device::Cpu));

        return Some((probs, value));
    }

    pub fn tensor_forward(&self, tensor_board: Tensor, train: bool) -> Option<(Tensor, Tensor)> {
        // Get probabilities of all moves from policy net.
        // FuncT::forward_t() -> (&Tensor, train: bool) -> Tensor
        /*
        autocast(train, || {
            let processed_state = tensor_board.apply_t(&self.state_processing_network, train);

            let probs = processed_state.apply_t(&self.policy_head, train).softmax(-1, Kind::Float);
            let value = processed_state.apply_t(&self.value_head, train).tanh();

            return Some((probs, value));
        })
        */
        let processed_state = tensor_board.apply_t(&self.state_processing_network, train);

        let probs = processed_state.apply_t(&self.policy_head, train).softmax(-1, Kind::Float);
        let value = processed_state.apply_t(&self.value_head, train).tanh();

        return Some((probs, value));
    }

    pub fn get_tensor_board(&self, board: [i32; 64], to_gpu: bool) -> Tensor {
        match to_gpu {
            true => Tensor::of_slice(&board).to_kind(Kind::Int).to_device(tch::Device::cuda_if_available()).unsqueeze(0),
            false => Tensor::of_slice(&board).to_kind(Kind::Int).to_device(tch::Device::Cpu).unsqueeze(0)
        }
    }

    pub fn train(&mut self, replay_buffer: &ReplayBuffer, n_iters: i64, batch_size: i64) {
        let two = Tensor::of_slice(&[2.0]).to_kind(Kind::Float).to_device(Device::cuda_if_available());

        init_progress_bar(n_iters as usize);
        set_progress_bar_action("Training", Color::Green, Style::Bold);
        for _ in 0..n_iters {
            let (target_states, target_probs, target_rewards) = replay_buffer.sample(batch_size);

            let (probs, value) = self.tensor_forward(target_states, true).expect("Forward pass failed");

            let actor_loss = -(target_probs * probs.log()).sum_dim_intlist(Some([-1i64].as_slice()), false, Kind::Float);
            let critic_loss = (target_rewards - value).pow(&two).sum(Kind::Float) / batch_size;
            let total_loss = actor_loss.mean(Kind::Float) + critic_loss;

            self.optimizer.backward_step(&total_loss);
            self.optimizer.zero_grad();
            inc_progress_bar();
        }
        // Save network and optimizer config to file.
        match self.save("saved_models/networks_test.pth") {
            Ok(_) => {},
            Err(e) => println!("Error saving network config: {}", e),
        }
        match self.load("saved_models/networks_test.pth") {
            Ok(_) => {},
            Err(e) => println!("Error loading network config: {}", e),
        }
    }

    pub fn train_mt(&mut self, replay_buffer: &ReplayBufferMT, n_iters: i64, batch_size: i64) {
        let two = Tensor::of_slice(&[2.0]).to_kind(Kind::Float).to_device(Device::cuda_if_available());

        init_progress_bar(n_iters as usize);
        set_progress_bar_action("Training", Color::Green, Style::Bold);
        for _ in 0..n_iters {
            let (target_states, target_probs, target_rewards) = replay_buffer.sample(batch_size);

            let (probs, value) = self.tensor_forward(target_states, true).expect("Forward pass failed");

            let actor_loss = -(target_probs * probs.log()).sum_dim_intlist(Some([-1i64].as_slice()), false, Kind::Float);
            let critic_loss = (target_rewards - value).pow(&two).sum(Kind::Float) / batch_size;
            let total_loss = actor_loss.mean(Kind::Float) + critic_loss;

            self.optimizer.backward_step(&total_loss);
            self.optimizer.zero_grad();
            inc_progress_bar();
        }
        // Save network and optimizer config to file.
        match self.save("saved_models/networks_test.pth") {
            Ok(_) => {},
            Err(e) => println!("Error saving network config: {}", e),
        }
        match self.load("saved_models/networks_test.pth") {
            Ok(_) => {},
            Err(e) => println!("Error loading network config: {}", e),
        }
    }

    fn save(&self, filename: &str) -> Result<(), TchError> {
        println!("...Saving Network Config to {}...", filename);
        self.var_store.save(filename)?;
        Ok(())
    }

    #[allow(dead_code)]
    fn load(&mut self, filename: &str) -> Result<(), TchError> {
        println!("...Loading Network Config from {}...", filename);
        self.var_store.load(filename)?;
        Ok(())
    }

}
