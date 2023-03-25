#![allow(unused_imports)]

use rand::Rng; 
use std::thread;
use std::sync::{ Arc, RwLock };
use std::collections::HashMap;
use std::time::Instant;

mod connect4;
use crate::connect4::Connect4Funcs;
use crate::connect4::Connect4Game;

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


#[allow(dead_code)]
fn play_game_connect4() {
    let mut game = Connect4Game::new();
    loop {
        let action = rand::thread_rng().gen_range(0..7);
        match game.make_move(action) {
            Some(_) => {
                break;
            },
            None => {},
        }
    }
}

fn play_game_chess(move_hash: Arc<HashMap<ChessMove, usize>>) {
    let mut game = ChessGame::new(move_hash);

    let networks = Networks::new(BERT_BASE_CONFIG);
    for _ in 0..200 {
        let best_move = run_mcts(&mut game, &networks, 800);

        println!("Best Move: {:?}", best_move);
        match game.make_move(best_move) {
            Some(_) => {
                break;
            },
            None => {},
        }
    }
}

#[allow(dead_code)]

fn main() {
    const N_GAMES: usize = 131_072;
    const N_THREADS: usize = 512;
    let move_hash = Arc::new(get_move_hash());


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


    let start = std::time::Instant::now();
    // Single Threaded
    for _ in 0..N_GAMES {
        play_game_chess(move_hash.clone());
    }
    let end = std::time::Instant::now();
    println!("Single Thread Time elapsed: {} ms", (end - start).as_millis());
}
