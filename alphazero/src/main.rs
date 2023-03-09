#![allow(unused_imports)]

use rand::Rng; 
use std::thread;

mod connect4;
use crate::connect4::Connect4Funcs;
use crate::connect4::Connect4Game;

pub mod chess_game;
use crate::chess_game::ChessFuncs;
use crate::chess_game::ChessGame;

mod mcts;
use crate::mcts::NodeFuncs;
use crate::mcts::Node;

pub mod move_map;


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

fn play_game_chess() {
    let mut game = ChessGame::new();
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
    const N_GAMES: usize = 100_000;

    for i in 0..N_GAMES {
        // play_game_connect4(&mut game);
        thread::spawn(move || {
            play_game_chess();
        });

        if (i+1) % 1000 == 0 {
            println!("Game {} of {}", i+1, N_GAMES);
        }
    }
}
