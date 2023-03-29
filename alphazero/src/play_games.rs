use std::sync::Arc;
use std::collections::HashMap;
use chess::ChessMove;


use crate::chess_game::ChessGame;
use crate::mcts::run_mcts;
use crate::networks::Networks;
use crate::replay_buffer::ReplayBuffer;


pub fn play_game_chess(
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
