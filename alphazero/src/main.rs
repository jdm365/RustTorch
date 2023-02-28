#![allow(unused_imports)]

use rand::Rng; 

mod game;
use crate::game::Game;
use crate::game::Connect4Game;

mod game_torch;
use crate::game_torch::GameTorch;
use crate::game_torch::Connect4GameTorch;



fn play_game_torch(game: &mut Connect4GameTorch) {
    loop {
        let action = rand::thread_rng().gen_range(0..7) as f32;
        match game.make_move(action) {
            Some(_) => {
                break;
            },
            None => {},
        }
    }
    game.reset_board();
}

fn play_game(game: &mut Connect4Game) {
    loop {
        let action = rand::thread_rng().gen_range(0..7);
        match game.make_move(action) {
            Some(_) => {
                break;
            },
            None => {},
        }
    }
    game.reset_board();
}


fn main() {
    const N_GAMES: usize = 10000;

    // Torch Version
    let mut game = Connect4GameTorch::new();
    for i in 0..N_GAMES {
        play_game_torch(&mut game);

        if (i+1) % 100 == 0 {
            println!("Game {} of {}", i+1, N_GAMES);
        }
    }

    let mut game = Connect4Game::new();
    for i in 0..N_GAMES {
        play_game(&mut game);

        if (i+1) % 100 == 0 {
            println!("Game {} of {}", i+1, N_GAMES);
        }
    }
}
