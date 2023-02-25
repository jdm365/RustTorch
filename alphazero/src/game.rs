use tch::{ Tensor, Kind, Device };


pub struct Connect4Game {
    board: Tensor,
    current_player: f32,
    result: Option<f32>,
    game_over: bool,
}


pub trait Game {
    fn new() -> Self;
    fn get_result(&self, move_col: f32, move_row: f32) -> Option<f32>;
    fn get_possible_moves(&self) -> Tensor;
    fn make_move(&mut self, move_col: f32) -> Option<f32>;
}


impl Game for Connect4Game {
    fn new() -> Self {
        Connect4Game {
            board: Tensor::zeros(&[6, 7], (Kind::Float, Device::Cpu)),
            current_player: 1.0,
            result: None,
            game_over: false,
        }
    }

    fn get_result(&self, move_col: f32, move_row: f32) -> Option<f32> {
        // Checks if the game is over and returns the result
        // 1 if player 1 wins, 0 if draw, None if game is not over
        //
        // General method: 
        // 1. Get sub_tensor - row, col or diag of last move.
        // 2. Multiply sub_tensor by self.current_player
        // 3. Apply ReLU
        // 4. Sum over all elements
        // 5. If sum >= 4, game is over

        // Vertical check
        let mut sub_tensor = self.board.narrow_copy(1, move_col as i64, 1);
        sub_tensor *= self.current_player;
        if bool::from(sub_tensor.relu().sum(Kind::Float).ge(4.0)) {
            return Some(self.current_player);
        }

        // Horizontal check
        sub_tensor = self.board.narrow_copy(0, move_row as i64, 1);
        sub_tensor *= self.current_player;
        if bool::from(sub_tensor.relu().sum(Kind::Float).ge(4.0)) {
            return Some(self.current_player);
        }
        /*

        // Diagonal check 1
        println!("Diagonal check 1");
        for col in 0..n_checks {
            // Get relevant 4x4 subtensor and select diag
            if self.board.narrow(0, move_row as i64 - col - 4, 4).narrow(1, move_col as i64 + col, 4).fliplr().diag(0).view([-1]).equal(&connect4) {
                return Some(self.current_player);
            }
            println!("Move row: {}", move_row);
            println!("Move col: {}", col);
            println!("");
        }

        // Diagonal check 2
        println!("Diagonal check 2");
        for col in 0..n_checks {
            // Get relevant 4x4 subtensor and select diag
            println!("Move row: {}", move_row as i64 - col - 3);
            println!("Move col: {}", move_col as i64 + col - 3);
            self.board.narrow(0, move_row as i64 - col - 3, 4).narrow(1, move_col as i64 + col - 3, 4).print();
            println!("");
            if self.board.narrow(0, move_row as i64 - col - 3, 4).narrow(1, move_col as i64 + col - 3, 4).diag(0).view([-1]).equal(&connect4) {
                return Some(self.current_player);
            }
        }
        */

        return None;
        }


    fn get_possible_moves(&self) -> Tensor {
        // Returns a tensor of possible moves
        // 1 if the move is possible, 0 if not
        if self.game_over {
            panic!("Game is over");
        }
        let zero = Tensor::zeros(&[7], (Kind::Float, Device::Cpu));

        return self.board.get(0).eq_tensor(&zero).to_kind(Kind::Float);
    }


    fn make_move(&mut self, move_col: f32) -> Option<f32> {
        // Makes a move

        if self.game_over {
            panic!("Game is over");
        }

        let zero = Tensor::zeros(&[1], (Kind::Float, Device::Cpu)).squeeze();
        let mut one = Tensor::ones(&[1], (Kind::Float, Device::Cpu)).squeeze();
        one *= self.current_player;
        let mut move_row: f32 = 0.0;

        for _row in 0..6 {
            let row = 5 - _row;
            if self.board.get(row).get(move_col as i64).equal(&zero) {
                self.board.get(row).get(move_col as i64).copy_(&one);
                move_row = row as f32;
                break;
            }
        }

        // Check if the game is over
        self.result = self.get_result(move_col, move_row);

        match self.result {
            Some(_) => {
                self.game_over = true;
                return Some(self.current_player);
            },
            None => {
                self.current_player *= -1.0; 
                return None;
            },
        }
    }
}





#[cfg(test)]
// Test Connect4Game functions
mod tests {
    use super::*;

    #[test]
    fn test_get_result() {
        let mut game = Connect4Game::new();
        assert_eq!(game.get_result(0.0, 0.0), None);
        game.make_move(0.0);
        game.make_move(1.0);
        game.make_move(0.0);
        game.make_move(1.0);
        game.make_move(0.0);
        game.make_move(1.0);
        assert_eq!(game.make_move(0.0), Some(1.0));
    }

    #[test]
    fn test_get_possible_moves() {
        let mut game = Connect4Game::new();
        assert_eq!(game.get_possible_moves(), Tensor::of_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        game.board.print();
        assert_eq!(game.get_possible_moves(), Tensor::of_slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    }
}

