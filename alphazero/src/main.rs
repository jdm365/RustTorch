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

fn play_game_chess(move_hash: Arc<HashMap<ChessMove, usize>>, networks: &mut Networks) {
    let mut game = ChessGame::new(move_hash.clone());

    let mut reward = 0;
    for idx in 0..200 {
        let best_move = run_mcts(&mut game, networks, 800);

        println!("Move {}: {:?}", (idx / 2) + 1 as usize, move_hash.iter().find(|(_, &v)| v == best_move).unwrap().0.to_string());
        match game.make_move(best_move) {
            Some(x) => {
                reward = x;
                break;
            },
            None => {},
        }
    }
    networks.store_episode(reward as i32);
}

#[allow(dead_code)]

fn main() {
    const N_GAMES: usize = 131_072;
    const N_THREADS: usize = 512;
    let move_hash = Arc::new(get_move_hash());
    let mut networks = Networks::new(BERT_BASE_CONFIG);


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
        play_game_chess(move_hash.clone(), &mut networks);
    }
    let end = std::time::Instant::now();
    println!("Single Thread Time elapsed: {} ms", (end - start).as_millis());
}
