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
use crate::play_games::*;


fn main() {
    const N_GAMES: usize = 1024;
    const N_THREADS: usize = 64;
    const N_MCTS_SIMS: usize = 400;

    let move_hash = Arc::new(get_move_hash());

    // let config = BERT_LARGE_CONFIG;
    // let config = BERT_BASE_CONFIG;
    let config = SMALL_CONFIG;
    // let config = TINY_CONFIG;

    let mut networks = Networks::new(config);
    match networks.load("saved_models/networks_test.pth") {
        Ok(_) => {},
        Err(x) => println!("Error loading model: {:?}", x),
    }

    /*
    let mut replay_buffer = ReplayBuffer::new(
        config.replay_buffer_capacity, 
        config.input_dim, 
        config.move_dim,
        );
    */

    let mut replay_buffer = ReplayBufferMT::new(
        config.replay_buffer_capacity, 
        config.input_dim, 
        config.move_dim,
        N_THREADS as i64,
        );

    std::env::set_var("RAYON_NUM_THREADS", N_THREADS.to_string());

    /*
    for _ in 0..N_GAMES {
        play_game_chess(move_hash.clone(), &mut networks, &mut replay_buffer, N_MCTS_SIMS);
    }
    */

    for _ in 0..(N_GAMES / N_THREADS) {
        play_ngames_chess(
            move_hash.clone(), 
            &mut networks, 
            &mut replay_buffer, 
            N_MCTS_SIMS, 
            N_THREADS, 
            40,
            false
            );
    }
}
