use Iterator;

use std::collections::HashMap;
use tch::nn;
use tch::{ Tensor, Kind, Device};

use crate::chess_game::ChessGame;
use crate::chess_game::ChessFuncs;

use crate::networks::{ chess_transformer, policy_mlp, value_mlp };

use rand::Rng;

pub struct Node<'a> {
    game: ChessGame<'a>,
    visit_count: usize,
    value_sum: f32,
    child_nodes: HashMap<usize, Node<'a>>,   // key, value = (move_idx, Node)
    prior: f32,
}



impl<'a> Node<'a> {
    fn new(_game: ChessGame, _prior: f32) -> Self {
        Node {
            game: _game,
            visit_count: 0,
            value_sum: 0.00,
            child_nodes: HashMap::new(),
            prior: _prior
        }
    }

    fn expand(&mut self, probs: &Vec<f32>) -> () {
        // Mask illegal moves
        let move_mask = self.game.get_move_mask();
        let mut move_probs = move_mask.iter().zip(probs.iter()).map(|(&x, &y)| x * y).collect::<Vec<_>>();
        let sum: f32 = move_probs.iter().sum();
        move_probs = move_probs.iter().map(|&x| x / sum).collect::<Vec<_>>();
        
        // Create new nodes for all moves.
        for move_idx in 0..1968 {
            if move_probs[move_idx] == 0.00 {
                continue;
            }

            self.game.make_move(move_idx);
            self.child_nodes.insert(
                move_idx, 
                Node::new(self.game, move_probs[move_idx])
                );
        }
    }


    fn expanded(&self) -> bool {
        return self.visit_count != 0;
    }


    #[inline]
    fn calc_ucb(&self, child: &Node) -> f32 {
        let actor_weight = child.prior;
        let value_weight = -child.value_sum;
        let visit_weight = (self.visit_count as f32).sqrt() / (child.visit_count + 1) as f32;

        return value_weight + actor_weight * visit_weight;
    }
        

    fn select_move(&self) -> Node {
       // Use mapping iterable to calculate UCB, find max, and get best action. 
       // Return child Node of best action.
       let move_idx = self.child_nodes.iter()
                                      .map(|(k, v)| (k, self.calc_ucb(v)))
                                      .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                                      .unwrap().0;
       return self.child_nodes[move_idx];
    }
}



fn run_mcts_sim(game: &ChessGame, node: Node, networks: &Networks) {
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
    let search_path = vec![&node];
    while node.expanded() {
        node = node.select_move();
        search_path.push(&node);
    }
    let (probs, values) = match networks.forward_inference(node.game.get_board()) {
        Some((p, v)) => (p, v),
        None => panic!("Networks failed to return values")
    };

    node.expand(&probs);
    
    // Backprop
    for node in search_path.iter().rev() {
        node.value_sum += values * game.get_current_player() as f32;
        node.visit_count += 1;
    }
}


fn run_mcts(game: &ChessGame, n_sims: usize) {
    let root = Node::new(game, 0.00);

    for _ in 0..n_sims {
        run_mcts_sim(game, root)
    }
}


struct ReplayBuffer {
    states: Tensor,
    probs: Tensor,
    values: Tensor,
    rewards: Tensor,
    capacity: i64,
    size:i64 
}




impl ReplayBuffer {
    fn new() -> Self {
        ReplayBuffer {
            states:  Tensor::empty(&[0, 0, 0, 0], (Kind::Float, Device::cuda_if_available())),
            probs:   Tensor::empty(&[0, 0, 0, 0], (Kind::Float, Device::cuda_if_available())),
            values:  Tensor::empty(&[0, 0, 0, 0], (Kind::Float, Device::cuda_if_available())),
            rewards: Tensor::empty(&[0, 0, 0, 0], (Kind::Float, Device::cuda_if_available())),
            capacity: 0,
            size: 0
        }
    }

    fn push(&mut self, state: Tensor, probs: Tensor, value: Tensor, reward: Tensor) -> () {
        self.states = Tensor::cat(&[self.states, state], 0);
        self.probs = Tensor::cat(&[self.probs, probs], 0);
        self.values = Tensor::cat(&[self.values, value], 0);
        self.rewards = Tensor::cat(&[self.rewards, reward], 0);
        self.size += 1;

        if self.size == self.capacity {
            self.states  = self.states.narrow(0, 1, self.capacity - 1);
            self.probs   = self.probs.narrow(0, 1, self.capacity - 1);
            self.values  = self.values.narrow(0, 1, self.capacity - 1);
            self.rewards = self.rewards.narrow(0, 1, self.capacity - 1);
            self.size -= 1;
        }
    }

    fn sample(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor)> {
        if self.size < batch_size as i64 {
            return None;
        }

        let idxs = Tensor::randint(self.size, &[batch_size as i64], (Kind::Int64, Device::cuda_if_available()));

        let states  = self.states.index_select(0, &idxs);
        let probs   = self.probs.index_select(0, &idxs);
        let values  = self.values.index_select(0, &idxs);
        let rewards = self.rewards.index_select(0, &idxs);

        return Some((states, probs, values, rewards));

    }
}


struct Networks {
    state_processing_network: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head: nn::SequentialT,
    replay_buffer: ReplayBuffer,
}


impl Networks {
    fn new() -> Self {
        Networks {
            state_processing_network: nn::seq_t(),
            policy_head: nn::seq_t(),
            value_head: nn::seq_t(),
            replay_buffer: ReplayBuffer::new(),
        }
    }

    fn forward_inference(&self, board: &Vec<f32>) -> Option<(Vec<f32>, f32)> {
        // Get probabilities of all moves from policy net.
        // FuncT::forward_t() -> (&Tensor, train: bool) -> Tensor
        let tensor_board = Tensor::of_slice(board).to_kind(Kind::Float).to_device(tch::Device::cuda_if_available());
        let processed_state = tensor_board.apply_t(&self.state_processing_network, false);
        let _probs = processed_state.apply_t(&self.policy_head, false).softmax(-1, Kind::Float);
        let _value = processed_state.apply_t(&self.value_head, false).tanh();

        let probs_vec = Vec::from(_probs.to_kind(Kind::Float).to_device(tch::Device::Cpu));
        let value = f32::from(_value.to_kind(Kind::Float).to_device(tch::Device::Cpu));

        return Some((probs_vec, value));
    }
}
