use Iterator;

use std::collections::HashMap;
use tch::nn;

use crate::chess_game::ChessGame;
use crate::chess_game::ChessFuncs;

// Assume chess for now. Num possible moves = 8 x 8 x 73 = 1744
// 1 Node ~= 25 KB. If m simulations and n threads (concurrent games),
// then m x n x 25 KB are required. E.g. 16 threads 800 sims -> ~200 MB.

pub struct Node {
    game: ChessGame,
    visit_count: usize,
    value_sum: f32,
    child_nodes: HashMap<i32, Node>,   // key, value = move_idx, Node
    prior: f32,
}


pub trait NodeFuncs {
    fn new(game: ChessGame, prior: f32) -> Self;
    fn expand(&mut self, probs: [&f32; 1744]) -> ();
    fn expanded(&self) -> bool;
    fn calc_ucb(&self, child: &Node) -> f32;
    fn select_move(&self) -> Node;
}


impl NodeFuncs for Node {
    fn new(_game: ChessGame, _prior: f32) -> Self {
        Node {
            game: _game,
            visit_count: 0,
            value_sum: 0.00,
            child_nodes: HashMap::new(),
            prior: _prior
        }
    }

    fn expand(&mut self, probs: &[f32; 1744]) -> () {
        // Mask illegal moves
        let move_probs = probs.iter()
                              .zip(self.game.get_move_mask().iter())
                              .map(|(&a, &b)| a * b)
                              .collect::<Vec<f32>>()
                              .try_into()
                              .unwrap();
        move_probs /= move_probs.sum();

        // Create new nodes for all moves.
        for move_idx in 0..1744 {
            if move_probs[move_idx] == 0.00 {
                continue;
            }

            let new_game = self.game.make_move(move_idx);
            self.child_nodes[move_idx] = Node::new(
                &new_game, 
                move_probs[move_idx] 
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
        let visit_weight = (self.visit_count as f32).sqrt() / (child.visit_count + 1);

        return value_weight + actor_weight * visit_weight;
    }
        

    fn select_move(&self) -> Node {
       // Use mapping iterable to calculate UCB, find max, and get best action. 
       // Return child Node of best action.
       return self.child_nodes
            .iter()
            .map(|(key, value)| (key, self.calc_ucb(value)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1;
    }
}



fn run_mcts_sim(
    game: &ChessGame, 
    node: Node, 
    policy_net: &nn::ModuleT, 
    value_net:  &nn::ModuleT
    ) {
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
    let probs  = policy_net(node.game.get_board());
    let values = value_net(node.game.get_board());

    probs  = Vec::from(probs);
    values = Vec::from(values);

    node.expand(probs);
    
    // Backprop
    for node in search_path.iter().rev() {
        node.value_sum += values * game.get_current_player();
        node.visit_count += 1;
    }
}


fn run_mcts(
    game: &ChessGame, 
    policy_net: &nn::ModuleT, 
    value_net: &nn::ModuleT, 
    n_sims: usize
    ) {
    let root = Node::new(game, 0.00);

    for _ in 0..n_sims {
        run_mcts_sim(
            game, 
            root,
            policy_net, 
            value_net
        )
    }
}
