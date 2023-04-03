use std::sync::Arc;
use std::collections::HashMap;
use chess::ChessMove;


use crate::chess_game::ChessGame;
use crate::mcts::{ run_mcts, run_mcts_multithreaded };
use crate::networks::Networks;
use crate::replay_buffer::*;


pub fn play_game_chess(
    move_hash: Arc<HashMap<ChessMove, usize>>, 
    networks: &mut Networks, 
    replay_buffer: &mut ReplayBuffer,
    n_mcts_sims: usize,
    max_moves: usize,
    ) {
    let mut game = ChessGame::new(move_hash.clone());

    let mut reward = 0;
    for idx in 0..(2 * max_moves) {
        // Make reference immutable
        let best_move = run_mcts(&game, &*networks, replay_buffer, n_mcts_sims);

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



pub fn play_ngames_chess(
    move_hash: Arc<HashMap<ChessMove, usize>>, 
    networks: &mut Networks, 
    replay_buffer: &mut ReplayBufferMT,
    n_mcts_sims: usize,
    num_threads: usize,
    max_moves: usize,
    verbose: bool,
    ) {
    let games = vec![ChessGame::new(move_hash.clone()); num_threads];

    let mut rewards: Vec<Option<i8>> = vec![None; num_threads];

    for idx in 0..(2 * max_moves) {
        let start = std::time::Instant::now();

        let mut input_games: Vec<ChessGame> = games.iter().filter(|x| x.get_status().is_none()).map(|x| x.clone()).collect();
        let active_thread_idxs: Vec<usize> = games.iter().filter(|x| x.get_status().is_none()).enumerate().map(|(idx, _)| idx).collect();
        let best_moves = run_mcts_multithreaded(&input_games, &*networks, replay_buffer, n_mcts_sims);

        for (thread_idx, best_move) in best_moves.iter().enumerate() {
            if verbose {
                println!(
                    "Thread {} Move {}: {:?}", 
                    thread_idx,
                    (idx / 2) + 1 as usize, 
                    move_hash.iter().find(|(_, &v)| v == *best_move).unwrap().0.to_string()
                    );
            }

            match input_games[thread_idx].make_move(*best_move) {
                Some(x) => {
                    rewards[active_thread_idxs[thread_idx]] = Some(x);
                    match replay_buffer.store_episode(x as i32, active_thread_idxs[thread_idx]) {
                        true => {
                            networks.train_mt(replay_buffer, 512, 256);
                        },
                        false => {},
                    }
                },
                None => {},
            }
        }
        println!("Moves/s: {}", (num_threads as f64) / start.elapsed().as_secs_f64());
        println!("Approx Games/s: {}", (num_threads as f64) / (max_moves as f64 * 2.0 * start.elapsed().as_secs_f64()));
        println!("");
    }


    for (thread_idx, reward) in rewards.iter().enumerate() {
        match reward {
            Some(_) => {},
            None => {
                match replay_buffer.store_episode(0, thread_idx) {
                    true => {
                        networks.train_mt(replay_buffer, 512, 256);
                    },
                    false => {},
                }
            },
        }
    }
}
