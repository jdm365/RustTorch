use std::collections::HashMap;
use rand::Rng;

use tch::{ Tensor, Kind, Device };

use crate::replay_buffer::ReplayBuffer;
use crate::networks::Networks;
use crate::chess_game::ChessGame;




#[derive(Debug, Clone)]
pub struct Node {
    game: ChessGame,
    visit_count: usize,
    value_sum: f32,
    child_nodes: HashMap<usize, usize>,   // key, value = (move_idx, node_idx `in arena`)
    prior: f32,
}



impl Node {
    fn new(game: ChessGame, prior: f32) -> Self {
        Node {
            game,
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



fn run_mcts_sim(networks: &Networks, node_arena: &mut NodeArena) {
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
    let mut search_path = vec![0];
    let mut arena_idx = 0;
    

    while node_arena.get(arena_idx).child_nodes.len() != 0 {
        arena_idx = node_arena.get(arena_idx).select_move(&node_arena);
        search_path.push(arena_idx);
    }

    let value = node_arena.get(arena_idx).game.get_status();

    match value {
        Some(value) => {
            // Backprop
            for node_idx in search_path.iter().rev() {
                let node = node_arena.get_mut(*node_idx);
                node.value_sum += value as f32 * node.game.get_current_player() as f32;
                node.visit_count += 1;
            }
        },
        None => {
            let (probs, values) = match networks.forward(node_arena.get(arena_idx).game.get_board(), false) {
                Some((p, v)) => (p, v),
                None => panic!("Networks failed to return values")
            };
            /*
            // DEBUG PERF
            let probs = vec![1.00; 1968];
            let values = 0.50;
            */

            // Expand. Need to call here to satisfy borrow checker.
            let move_mask = node_arena.get_mut(arena_idx).game.get_move_mask();
            let mut move_probs = move_mask.iter().zip(probs.iter()).map(|(&x, &y)| x * y).collect::<Vec<_>>();
            let sum: f32 = move_probs.iter().sum();
            move_probs = move_probs.iter().map(|&x| x / sum).collect::<Vec<_>>();

            // Create new nodes for all moves.
            for move_idx in 0..1968 {
                if move_probs[move_idx] == 0.00 {
                    continue;
                }

                let mut new_game = node_arena.get_mut(arena_idx).game.clone();
                new_game.make_move(move_idx);
                let child_node = Node::new(new_game, move_probs[move_idx]);
                let arena_idx_child = node_arena.push(child_node);
                node_arena.get_mut(arena_idx).child_nodes.insert(move_idx, arena_idx_child);
            }

            // Backprop
            for node_idx in search_path.iter().rev() {
                let node = node_arena.get_mut(*node_idx);
                node.value_sum += values * node.game.get_current_player() as f32;
                node.visit_count += 1;
            }
        }
    }
    
}


pub fn run_mcts(
    game: &ChessGame, 
    networks: &Networks, 
    replay_buffer: &mut ReplayBuffer, 
    n_sims: usize,
    ) -> usize {
    let root = Node::new(game.clone(), 0.00);
    let mut node_arena = NodeArena::new();
    node_arena.push(root);

    for _ in 0..n_sims {
        run_mcts_sim(networks, &mut node_arena);
    }

    // Return best move based on mcts
    let (probs, best_move) = node_arena.get(0).select_move_final(&node_arena);
    replay_buffer.push(
        networks.get_tensor_board(game.get_board(), false), 
        Tensor::of_slice(&probs).to_kind(Kind::Float).to_device(Device::Cpu),
        );
    best_move
}
