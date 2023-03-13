#![allow(unused_imports)]

use rand::Rng; 
use std::thread;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;

mod connect4;
use crate::connect4::Connect4Funcs;
use crate::connect4::Connect4Game;

pub mod chess_game;
use crate::chess_game::ChessFuncs;
use crate::chess_game::ChessGame;

use chess::ChessMove;

/*
mod mcts;
use crate::mcts::NodeFuncs;
use crate::mcts::Node;
*/

pub mod move_map;
use crate::move_map::*;


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

fn play_game_chess(move_hash: &HashMap<ChessMove, usize>) {
    let mut game = ChessGame::new(&move_hash);
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
    const N_GAMES: usize = 524_288;
    const N_THREADS: usize = 512;
    let move_hash = Arc::new(get_move_hash());

    let chunk_size = N_GAMES / N_THREADS;


    let start = std::time::Instant::now();

    let handles: Vec<_> = (0..N_THREADS).map(|_| {
        let move_hash = move_hash.clone();
        thread::spawn(move || {
            for _ in 0..chunk_size {
                play_game_chess(&move_hash);
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let end = std::time::Instant::now();
    println!("Time elapsed: {} ms", (end - start).as_millis());
}
