use Iterator;

use std::collections::HashMap;
use tch::nn;
use tch::{ Tensor, Kind, Device};

use crate::chess_game::ChessGame;

use crate::networks::{ chess_transformer, policy_mlp, value_mlp, Config };

use rand::Rng;




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

    fn select_move_final(&self, node_arena: &NodeArena) -> usize {
        // Use mapping iterable to calculate UCB, find max, and get best action. 
        // Return child Node of best action.
        let mut best_move = 0;
        let mut best_ucb = -100000.00;

        for (move_idx, arena_idx) in self.child_nodes.iter() {
            let ucb = self.calc_ucb(node_arena.get(*arena_idx));

            if ucb > best_ucb {
                best_ucb = ucb;
                best_move = *move_idx;
            }
        }
        best_move
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

    let (probs, values) = match networks.forward(node_arena.get(arena_idx).game.get_board(), false) {
        Some((p, v)) => (p, v),
        None => panic!("Networks failed to return values")
    };

    // node_arena.get_mut(arena_idx).expand(&probs, node_arena);

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


pub fn run_mcts(game: &ChessGame, networks: &Networks, n_sims: usize) -> usize {
    let root = Node::new(game.clone(), 0.00);
    let mut node_arena = NodeArena::new();
    node_arena.push(root);

    for _ in 0..n_sims {
        run_mcts_sim(networks, &mut node_arena);
    }

    // Return best move based on mcts
    node_arena.get(0).select_move_final(&node_arena)
}


#[allow(dead_code)]
struct ReplayBuffer {
    states: Tensor,
    probs: Tensor,
    values: Tensor,
    rewards: Tensor,
    capacity: i64,
    cntr: i64
}




#[allow(dead_code)]
impl ReplayBuffer {
    fn new(capacity: i64, input_dim: i64, n_actions: i64) -> Self {
        ReplayBuffer {
            states: Tensor::zeros(&[capacity, input_dim], (Kind::Float, Device::cuda_if_available())),
            probs: Tensor::zeros(&[capacity, n_actions], (Kind::Float, Device::cuda_if_available())),
            values: Tensor::zeros(&[capacity, 1], (Kind::Float, Device::cuda_if_available())),
            rewards: Tensor::zeros(&[capacity, 1], (Kind::Float, Device::cuda_if_available())),
            capacity,
            cntr: 0
        }
    }

    fn push(&mut self, state: Tensor, probs: Tensor, value: Tensor, reward: Tensor) {
        self.cntr = self.cntr % self.capacity;
        self.states.get(self.cntr).copy_(&state);
        self.probs.get(self.cntr).copy_(&probs);
        self.values.get(self.cntr).copy_(&value);
        self.rewards.get(self.cntr).copy_(&reward);
        self.cntr += 1;
    }

    fn sample(&self, batch_size: i64) -> Option<(Tensor, Tensor, Tensor, Tensor)> {
        let idxs = Tensor::randint(self.capacity - 1 , &[batch_size], (Kind::Int64, Device::cuda_if_available()));

        let states  = self.states.index_select(0, &idxs);
        let probs   = self.probs.index_select(0, &idxs);
        let values  = self.values.index_select(0, &idxs);
        let rewards = self.rewards.index_select(0, &idxs);

        return Some((states, probs, values, rewards));

    }
}


#[allow(dead_code)]
pub struct Networks {
    state_processing_network: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head:  nn::SequentialT,
    replay_buffer: ReplayBuffer,
}



impl Networks {
    pub fn new(cfg: Config) -> Self {
        let var_store = nn::VarStore::new(Device::cuda_if_available());
        let state_processing_network = chess_transformer(
            &(var_store.root() / "state_processing_network"), 
            cfg 
            );
        let policy_head = policy_mlp(
            &(var_store.root() / "policy_head"), 
            cfg 
            );
        let value_head = value_mlp(
            &(var_store.root() / "value_head"), 
            cfg 
            );
        Networks {
            state_processing_network,
            policy_head,
            value_head,
            replay_buffer: ReplayBuffer::new(cfg.replay_buffer_capacity, cfg.input_dim, cfg.move_dim),
        }
    }

    fn forward(&self, board: [i32; 64], train: bool) -> Option<(Vec<f32>, f32)> {
        // Get probabilities of all moves from policy net.
        // FuncT::forward_t() -> (&Tensor, train: bool) -> Tensor
        let tensor_board = (Tensor::of_slice(&board).to_kind(Kind::Int).to_device(tch::Device::cuda_if_available())).unsqueeze(0);

        let processed_state = tensor_board.apply_t(&self.state_processing_network, train);

        let _probs = processed_state.apply_t(&self.policy_head, train).softmax(-1, Kind::Float);
        let _value = processed_state.apply_t(&self.value_head, train).tanh();

        let probs_vec = Vec::from(_probs.to_kind(Kind::Float).to_device(tch::Device::Cpu));
        let value = f32::from(_value.to_kind(Kind::Float).to_device(tch::Device::Cpu));

        return Some((probs_vec, value));
    }
}
