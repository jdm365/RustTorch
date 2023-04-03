use std::collections::HashMap;
use rand::Rng;

use tch::{ Tensor, Kind, Device };

use crate::replay_buffer::*;
use crate::networks::Networks;
use crate::chess_game::ChessGame;


// use std::sync::{Arc, Mutex};
use rayon::prelude::*;



#[derive(Debug, Clone)]
pub struct Node {
    // game: ChessGame,
    visit_count: usize,
    value_sum: f32,
    child_nodes: HashMap<usize, usize>,   // key, value = (move_idx, node_idx `in arena`)
    prior: f32,
}



impl Node {
    fn new(prior: f32) -> Self {
        Node {
            visit_count: 0,
            value_sum: 0.00,
            child_nodes: HashMap::new(),
            prior
        }
    }

    #[inline]
    fn calc_ucb(&self, child_node: &Node) -> f32 {
        let actor_weight = child_node.prior;
        let value_weight = -child_node.value_sum;
        let visit_weight = (self.visit_count as f32).sqrt() / (child_node.visit_count + 1) as f32;

        return value_weight + actor_weight * visit_weight;
    }
        

    fn select_move(&self, node_arena: &NodeArena) -> usize {
        // Use mapping iterable to calculate UCB, find max, and get best action. 
        // Return child Node of best action.
        let mut best_move = 0;
        let mut best_ucb = -100000.00;

        for (_, arena_idx) in self.child_nodes.iter() {
            let ucb = self.calc_ucb(node_arena.get(*arena_idx));

            if ucb > best_ucb {
                best_ucb = ucb;
                best_move = *arena_idx;
            }
        }
        best_move
    }

    fn select_move_final(&self, node_arena: &NodeArena) -> (Vec<f32>, usize) {
        // Use mapping iterable to calculate UCB, find max, and get best action. 
        // Return child Node of best action.
        let temperature = 1.00;

        let mut probs: [f32; 1968] = [0.00; 1968];
        for (move_idx, arena_idx) in self.child_nodes.iter() {
            probs[*move_idx] = (node_arena.get(*arena_idx).visit_count as f32).powf(1.00 / temperature);
        }
        let sum: f32 = probs.iter().sum();
        let probs = probs.iter().map(|&x| x / sum).collect::<Vec<_>>();

        // Choose action based on probs.
        // TODO: Check this.
        let mut rng = rand::thread_rng();
        let mut rand_num = rng.gen_range(0.00..1.00);
        let mut move_idx = 0;
        for (idx, &prob) in probs.iter().enumerate() {
            rand_num -= prob;
            if rand_num <= 0.00 {
                move_idx = idx;
                break;
            }
        }
        (probs, move_idx)
    }

}


#[derive(Debug, Clone)]
pub struct NodeArena {
    arena: Vec<Node>,
}

impl NodeArena {
    fn new() -> Self {
        NodeArena {
            arena: Vec::new(),
        }
    }

    fn get(&self, idx: usize) -> &Node {
        &self.arena[idx]
    }

    fn get_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.arena[idx]
    }

    fn push(&mut self, node: Node) -> usize {
        self.arena.push(node);
        self.arena.len() - 1
    }
}



fn run_mcts_sim(game: &ChessGame, networks: &Networks, node_arena: &mut NodeArena) {
    /*
    MCTS: 4 steps
    Step 1: Selection       - Traverse the tree following maximized UCB until you arrive at a 
                              state which has not been evaluated by policy and value nets. 
                              Select that node.
    Step 2: Expansion       - Enumerate all possible moves in selected state. *Mask invlid moves.
    Step 3: Simulation      - Misnomer for AlphaZero approach. Makes more sense in classical
                              MCTS. Here it just means query the networks and get scores
                              for all moves in the position (Masking illegal moves) and 
                              value of the position in the form of a predicted probability
                              of winning.
    Step 4: Backpropogation - Update the search path with visit_count and evaulation sum.
    */
    let player = game.get_current_player();

    let mut search_path = vec![0];
    let mut arena_idx = 0;

    let mut game = game.clone();
    let mut value: Option<i8> = None;
    let mut new_arena_idx;
    
    while node_arena.get(arena_idx).child_nodes.len() != 0 {
        new_arena_idx = node_arena.get(arena_idx).select_move(&node_arena);
        search_path.push(new_arena_idx);
        let move_idx = node_arena.get(arena_idx).child_nodes.iter()
                                                            .find(|(_, &v)| v == new_arena_idx)
                                                            .expect("Move not found").0;
        value = game.make_move(*move_idx);
        arena_idx = new_arena_idx;
    }
    

    match value {
        Some(value) => {
            // Backprop
            let mut factor = (player * game.get_current_player()) as f32;
            for node_idx in search_path.iter().rev() {
                let node = node_arena.get_mut(*node_idx);
                node.value_sum += value as f32 * factor;
                node.visit_count += 1;
                factor *= -1.00;
            }
        },
        None => {
            let (probs, values) = match networks.forward(game.get_board(), false) {
                Some((p, v)) => (p, v),
                None => panic!("Networks failed to return values")
            };
            /*
            // DEBUG PERF
            let probs = vec![1.00; 1968];
            let values = 0.50;
            */

            // Expand. Need to call here to satisfy borrow checker.
            let move_mask = game.get_move_mask();
            let mut move_probs = move_mask.iter().zip(probs.iter()).map(|(&x, &y)| x * y).collect::<Vec<_>>();
            let sum: f32 = move_probs.iter().sum();
            move_probs = move_probs.iter().map(|&x| x / sum).collect::<Vec<_>>();

            // Create new nodes for all moves.
            for move_idx in 0..1968 {
                if move_probs[move_idx] == 0.00 {
                    continue;
                }

                let child_node = Node::new(move_probs[move_idx]);
                let arena_idx_child = node_arena.push(child_node);
                node_arena.get_mut(arena_idx).child_nodes.insert(move_idx, arena_idx_child);
            }

            // Backprop
            let mut factor = (player * game.get_current_player()) as f32;
            for node_idx in search_path.iter().rev() {
                let node = node_arena.get_mut(*node_idx);
                node.value_sum += values * factor;
                node.visit_count += 1;
                factor *= -1.00;
            }
        }
    }
    
}


fn run_mcts_nsims_multithreaded(
    orig_games: &Vec<ChessGame>,
    networks: &Networks, 
    num_threads: usize,
    num_sims: usize,
    ) -> Vec<NodeArena> {
    /*
    MCTS: 4 steps
    Step 1: Selection       - Traverse the tree following maximized UCB until you arrive at a 
                              state which has not been evaluated by policy and value nets. 
                              Select that node.
    Step 2: Expansion       - Enumerate all possible moves in selected state. *Mask invlid moves.
    Step 3: Simulation      - Misnomer for AlphaZero approach. Makes more sense in classical
                              MCTS. Here it just means query the networks and get scores
                              for all moves in the position (Masking illegal moves) and 
                              value of the position in the form of a predicted probability
                              of winning.
    Step 4: Backpropogation - Update the search path with visit_count and evaulation sum.
    */
    let players = orig_games.iter().map(|g| g.get_current_player()).collect::<Vec<_>>();
    let root_nodes = vec![Node::new(0.00); orig_games.len()];

    let mut node_arenas = vec![NodeArena::new(); num_threads];
    for idx in 0..num_threads {
        node_arenas[idx].push(root_nodes[idx].clone());
    }

    for _ in 0..num_sims {
        let mut search_paths = vec![vec![0]; orig_games.len()];
        let mut games = orig_games.iter().map(|g| g.clone()).collect::<Vec<_>>();
        let mut values: Vec<Option<i8>> = vec![None; games.len()];
        let mut arena_idxs = vec![0; games.len()];

        // node_arenas.par_iter_mut().enumerate().for_each(|(idx, arena)| {
        node_arenas.par_iter_mut().zip(games.par_iter_mut())
                                  .zip(values.par_iter_mut())
                                  .zip(arena_idxs.par_iter_mut())
                                  .zip(search_paths.par_iter_mut())
                                  .for_each(|((((arena, game), value), arena_idx), search_path)| {
            let mut new_arena_idx;
            while arena.get(*arena_idx).child_nodes.len() != 0 {
                new_arena_idx = arena.get(*arena_idx).select_move(&arena);
                search_path.push(new_arena_idx);
                let move_idx = arena.get(*arena_idx).child_nodes.iter()
                                                                .find(|(_, &v)| v == new_arena_idx)
                                                                .expect("Move not found").0;
                *value = game.make_move(*move_idx);
                *arena_idx = new_arena_idx;
            }
        });

        let tensor_board = Tensor::zeros(&[games.len() as i64, 64], (Kind::Int, Device::cuda_if_available()));
        for (idx, (game, value)) in games.iter_mut().zip(values.iter()).enumerate() {
            match value {
                Some(_) => {},
                None => {
                    tensor_board.narrow(0, idx as i64, 1).copy_(&networks.get_tensor_board(game.get_board(), true));
                },
            }
        }
        let (probs_tensor, value_tensor) = match networks.tensor_forward(tensor_board, false) {
            Some((p, v)) => (p, v),
            None => panic!("Networks failed to return values")
        };

        let probs_net: Vec<Vec<f32>> = Vec::from(probs_tensor.to_kind(Kind::Float).to_device(tch::Device::Cpu));
        let values_net: Vec<f32>     = Vec::from(value_tensor.to_kind(Kind::Float).to_device(tch::Device::Cpu));

        // values.par_iter_mut().enumerate().for_each(|(idx, true_value)| {
        values.par_iter_mut().zip(games.par_iter_mut())
                             .zip(probs_net.par_iter())
                             .zip(values_net.par_iter())
                             .zip(search_paths.par_iter_mut())
                             .zip(node_arenas.par_iter_mut())
                             .zip(arena_idxs.par_iter())
                             .zip(players.par_iter())
                             .for_each(|(((((((true_value, game), probs_net), values_net), search_path), node_arena), arena_idx), player)| {
            // BACKPROP
            match true_value {
                Some(value) => {
                    let mut factor = (player * game.get_current_player()) as f32;
                    for node_idx in search_path.iter().rev() {
                        let node = node_arena.get_mut(*node_idx);
                        node.value_sum += *value as f32 * factor;
                        node.visit_count += 1;
                        factor *= -1.00;
                    }
                },
                None => {
                    // EXPAND
                    let move_mask = game.get_move_mask();
                    let mut move_probs = move_mask.iter().zip(probs_net.iter()).map(|(&x, &y)| x * y).collect::<Vec<_>>();
                    let sum: f32 = move_probs.iter().sum();
                    move_probs = move_probs.iter().map(|&x| x / sum).collect::<Vec<_>>();

                    for move_idx in 0..1968 {
                        if move_probs[move_idx] == 0.00 {
                            continue;
                        }

                        let child_node = Node::new(move_probs[move_idx]);
                        let arena_idx_child = node_arena.push(child_node);
                        node_arena.get_mut(*arena_idx).child_nodes.insert(move_idx, arena_idx_child);
                    }

                    // BACKPROP
                    let mut factor = (player * game.get_current_player()) as f32;
                    for node_idx in search_path.iter().rev() {
                        let node = node_arena.get_mut(*node_idx);
                        node.value_sum += values_net * factor;
                        node.visit_count += 1;
                        factor *= -1.00;
                    }
                }
            }
        });
    }
    node_arenas
}


pub fn run_mcts(
    game: &ChessGame,
    networks: &Networks,
    replay_buffer: &mut ReplayBuffer,
    n_sims: usize,
    ) -> usize {
    let root = Node::new(0.00);
    let mut node_arena = NodeArena::new();
    node_arena.push(root);

    for _ in 0..n_sims {
        run_mcts_sim(game, networks, &mut node_arena);
    }

    // Return best move based on mcts
    let (probs, best_move) = node_arena.get(0).select_move_final(&node_arena);
    replay_buffer.push(
        networks.get_tensor_board(game.get_board(), false), 
        Tensor::of_slice(&probs).to_kind(Kind::Float).to_device(Device::Cpu),
        );
    best_move
}



pub fn run_mcts_multithreaded(
    games: &Vec<ChessGame>, 
    networks: &Networks, 
    replay_buffer: &mut ReplayBufferMT, 
    num_sims: usize,
    ) -> Vec<usize> {
    let num_threads = games.len();


    let node_arenas = run_mcts_nsims_multithreaded(games, networks, num_threads, num_sims);

    // Return best move based on mcts
    let mut best_moves = Vec::new();

    for thread_idx in 0..num_threads {
        let (probs, best_move) = node_arenas[thread_idx].get(0).select_move_final(&node_arenas[thread_idx]);
        replay_buffer.push(
            networks.get_tensor_board(games[thread_idx].get_board(), false), 
            Tensor::of_slice(&probs).to_kind(Kind::Float).to_device(Device::Cpu),
            thread_idx,
            );
        best_moves.push(best_move);
    }
    best_moves
}
