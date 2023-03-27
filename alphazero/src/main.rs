#![allow(unused_imports)]

use rand::Rng; 
use std::thread;
use std::sync::{ Arc, RwLock };
use std::collections::HashMap;
use std::time::Instant;

pub mod configs;
use crate::configs::*;

pub mod chess_game;
use crate::chess_game::ChessGame;

use chess::ChessMove;

use rayon::prelude::*;

mod mcts;
use crate::mcts::*;

pub mod move_map;
use crate::move_map::*;

pub mod networks;
use crate::networks::*;


fn play_game_chess(
    move_hash: Arc<HashMap<ChessMove, usize>>, 
    networks: &mut Networks, 
    replay_buffer: &mut ReplayBuffer,
    n_mcts_sims: usize,
    ) {
    let mut game = ChessGame::new(move_hash.clone());

    let mut reward = 0;
    for idx in 0..200 {
        // Make reference immutable
        let best_move = run_mcts(&mut game, &*networks, replay_buffer, n_mcts_sims);

        println!("Move {}: {:?}", (idx / 2) + 1 as usize, move_hash.iter().find(|(_, &v)| v == best_move).unwrap().0.to_string());
        match game.make_move(best_move) {
            Some(x) => {
                reward = x;
                break;
            },
            None => {},
        }
    }
    match replay_buffer.store_episode(reward as i32) {
        true => {
            networks.train(replay_buffer, 128, 256);
        },
        false => {},
    }
}

#[allow(dead_code)]
fn main() {
    const N_GAMES: usize = 1024;
    const N_THREADS: usize = 512;
    const N_MCTS_SIMS: usize = 400;

    let move_hash = Arc::new(get_move_hash());
    // let mut networks = Networks::new(BERT_BASE_CONFIG);
    let mut networks = Networks::new(SMALL_CONFIG);


    // Multithreaded
    /*
    let start = std::time::Instant::now();
    let chunk_size = N_GAMES / N_THREADS;
    rayon::scope(|s| {
        for _ in 0..N_THREADS {
            s.spawn(|_| {
                for _ in 0..chunk_size {
                    play_game_chess(move_hash.clone());
                }
            });
        }
    });
    let end = std::time::Instant::now();
    println!("Rayon Time elapsed: {} ms", (end - start).as_millis());
    */

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
