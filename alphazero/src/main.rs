use std::sync::Arc;

pub mod configs;
use crate::configs::*;

pub mod chess_game;

pub mod mcts;

pub mod move_map;
use crate::move_map::*;

pub mod networks;
use crate::networks::*;

pub mod replay_buffer;
use crate::replay_buffer::*;

pub mod play_games;
use crate::play_games::play_game_chess;


#[allow(dead_code)]
fn main() {
    const N_GAMES: usize = 1024;
    const N_THREADS: usize = 512;
    const N_MCTS_SIMS: usize = 800;

    let move_hash = Arc::new(get_move_hash());
    // let mut networks = Networks::new(BERT_LARGE_CONFIG);
    // let mut networks = Networks::new(BERT_BASE_CONFIG);
    let mut networks = Networks::new(SMALL_CONFIG);
    // let mut networks = Networks::new(TINY_CONFIG);

    let mut replay_buffer = ReplayBuffer::new(
        SMALL_CONFIG.replay_buffer_capacity, 
        SMALL_CONFIG.input_dim, 
        SMALL_CONFIG.move_dim,
        );

    let start = std::time::Instant::now();
    // Single Threaded
    for _ in 0..N_GAMES {
        play_game_chess(move_hash.clone(), &mut networks, &mut replay_buffer, N_MCTS_SIMS);
    }
    let end = std::time::Instant::now();
    println!("Single Thread Time elapsed: {} ms", (end - start).as_millis());
}
