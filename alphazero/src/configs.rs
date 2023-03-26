use crate::networks::Config;



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



// Smaller Transformer
pub const SMALL_CONFIG: Config = Config {
    input_dim: 64,          // 64 squares, 1 piece per square
    embedding_dim: 832,     // 6 piece types per side plus one for no piece, 64 squares ((6 * 2) + 1) * 64 = 832
    hidden_dim: 256,
    n_blocks: 4,
    n_heads: 8,
    mlp_expansion_factor: 2,
    dropout_rate: 0.1,
    policy_mlp_dim: 256,
    value_mlp_dim: 256,
    move_dim: 1968,         // Number of possible chess moves as defined in move_map.rs
    replay_buffer_capacity: 100_000,
};
