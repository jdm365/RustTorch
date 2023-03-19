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
use crate::mcts::Node;

pub mod move_map;
use crate::move_map::*;

pub mod networks;


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

fn play_game_chess(move_hash: &Arc<RwLock<HashMap<ChessMove, usize>>>) {
    let mut game = ChessGame::new(&move_hash.read().unwrap());
    for _ in 0..200 {
        // End game after 100 moves if not ended
        match game.make_move_random() {
            Some(_) => {
                break;
            },
            None => {},
        }
    }
}


fn main() {
    const N_GAMES: usize = 131_072;
    const N_THREADS: usize = 512;
    let move_hash = Arc::new(RwLock::new(get_move_hash()));

    let chunk_size = N_GAMES / N_THREADS;

    let start = std::time::Instant::now();

    // Now play N_GAMES again on N_THREADS threads using rayon
    rayon::scope(|s| {
        for _ in 0..N_THREADS {
            //let move_hash = move_hash.clone();
            s.spawn(|_| {
                for _ in 0..chunk_size {
                    play_game_chess(&move_hash);
                }
            });
        }
    });


    let end = std::time::Instant::now();
    println!("Rayon Time elapsed: {} ms", (end - start).as_millis());
}
