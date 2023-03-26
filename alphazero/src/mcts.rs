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


pub fn run_mcts(game: &ChessGame, networks: &mut Networks, n_sims: usize) -> usize {
    let root = Node::new(game.clone(), 0.00);
    let mut node_arena = NodeArena::new();
    node_arena.push(root);

    for _ in 0..n_sims {
        run_mcts_sim(networks, &mut node_arena);
    }

    // Return best move based on mcts
    let (probs, best_move) = node_arena.get(0).select_move_final(&node_arena);
    networks.replay_buffer.push(
        networks.get_tensor_board(game.get_board(), false), 
        Tensor::of_slice(&probs).to_kind(Kind::Float).to_device(Device::Cpu),
        );
    best_move
}


#[allow(dead_code)]
struct ReplayBuffer {
    states: Tensor,
    probs: Tensor,
    rewards: Tensor,
    episode_states: Tensor,
    episode_probs: Tensor,
    capacity: i64,
    episode_cntr: i64,
    buffer_ptr: i64,
}




#[allow(dead_code)]
impl ReplayBuffer {
    fn new(capacity: i64, input_dim: i64, n_actions: i64) -> Self {
        ReplayBuffer {
            states: Tensor::zeros(&[capacity, input_dim], (Kind::Int, Device::Cpu)),
            probs: Tensor::zeros(&[capacity, n_actions], (Kind::Float, Device::Cpu)),
            rewards: Tensor::zeros(&[capacity, 1], (Kind::Float, Device::Cpu)),
            episode_states: Tensor::zeros(&[1024, input_dim], (Kind::Float, Device::Cpu)),
            episode_probs: Tensor::zeros(&[1024, n_actions], (Kind::Float, Device::Cpu)),
            capacity,
            episode_cntr: 0,
            buffer_ptr: 0,
        }
    }

    pub fn push(&mut self, state: Tensor, probs: Tensor) {
        self.episode_states.slice(0, self.episode_cntr, self.episode_cntr + 1, 1).copy_(&state);
        self.episode_probs.slice(0, self.episode_cntr, self.episode_cntr + 1, 1).copy_(&probs);
        self.episode_cntr += 1;
    }

    pub fn store_episode(&mut self, reward: i32) {
        // Reward is -1, 0, 1
        // To create reward tensor for episode, backtrack through episode 
        // and add reward for each state and prob pair multiplying by -1
        // each time.
        let prev_buffer_ptr = match self.capacity < self.episode_cntr + self.buffer_ptr {
            true => 0,
            false => self.buffer_ptr,
        };

        self.buffer_ptr += self.episode_cntr;
        self.episode_cntr = 0;

        if reward == 0 {
            self.states.slice(0, prev_buffer_ptr, self.buffer_ptr, 1).copy_(&self.episode_states);
            self.probs.slice(0, prev_buffer_ptr, self.buffer_ptr, 1).copy_(&self.episode_probs);
            self.rewards.slice(0, prev_buffer_ptr, self.buffer_ptr, 1).copy_(&Tensor::zeros(&[self.episode_cntr, 1], (Kind::Float, Device::Cpu)));

        }
        else {
            let mut rewards = vec![reward as f32; self.episode_cntr as usize];
            for idx in (0..self.episode_cntr - 1).rev() {
                rewards[idx as usize] *= -1.0;
            }
            let reward_tensor = Tensor::of_slice(&rewards).to_kind(Kind::Float).to_device(Device::Cpu);

            self.states.slice(0, prev_buffer_ptr, self.buffer_ptr, 1).copy_(&self.episode_states);
            self.probs.slice(0, prev_buffer_ptr, self.buffer_ptr, 1).copy_(&self.episode_probs);
            self.rewards.slice(0, prev_buffer_ptr, self.buffer_ptr, 1).copy_(&reward_tensor);

        }
        self.episode_states = self.episode_states.zero_();
        self.episode_probs  = self.episode_probs.zero_();

    }

    pub fn sample(&self, batch_size: i64) -> Option<(Tensor, Tensor, Tensor)> {
        let idxs = Tensor::randint(self.capacity - 1 , &[batch_size], (Kind::Int64, Device::cuda_if_available()));

        let states  = self.states.index_select(0, &idxs);
        let probs   = self.probs.index_select(0, &idxs);
        let rewards = self.rewards.index_select(0, &idxs);

        return Some((states, probs, rewards));

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
        let tensor_board = self.get_tensor_board(board, true);

        let processed_state = tensor_board.apply_t(&self.state_processing_network, train);

        let _probs = processed_state.apply_t(&self.policy_head, train).softmax(-1, Kind::Float);
        let _value = processed_state.apply_t(&self.value_head, train).tanh();

        let probs_vec = Vec::from(_probs.to_kind(Kind::Float).to_device(tch::Device::Cpu));
        let value = f32::from(_value.to_kind(Kind::Float).to_device(tch::Device::Cpu));

        return Some((probs_vec, value));
    }

    pub fn get_tensor_board(&self, board: [i32; 64], to_gpu: bool) -> Tensor {
        match to_gpu {
            true => Tensor::of_slice(&board).to_kind(Kind::Int).to_device(tch::Device::cuda_if_available()).unsqueeze(0),
            false => Tensor::of_slice(&board).to_kind(Kind::Int).to_device(tch::Device::Cpu).unsqueeze(0)
        }
    }

    pub fn store_episode(&mut self, reward: i32) {
        self.replay_buffer.store_episode(reward);
    }
}
