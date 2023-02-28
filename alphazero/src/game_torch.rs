use tch::{ Tensor, Kind, Device };


pub struct Connect4GameTorch {
    board: Tensor,
    current_player: f32,
    result: Option<f32>,
    game_over: bool,
}


pub trait GameTorch {
    fn new() -> Self;
    fn get_result(&self, move_col: f32, move_row: f32) -> Option<f32>;
    fn get_possible_moves(&self) -> Tensor;
    fn make_move(&mut self, move_col: f32) -> Option<f32>;
    fn reset_board(&mut self);
}


impl GameTorch for Connect4GameTorch {
    fn new() -> Self {
        Connect4GameTorch {
            board: Tensor::zeros(&[6, 7], (Kind::Float, Device::Cpu)),
            current_player: 1.0,
            result: None,
            game_over: false,
        }
    }

    fn reset_board(&mut self) {
        self.board = Tensor::zeros(&[6, 7], (Kind::Float, Device::Cpu));
        self.current_player = 1.0;
        self.result = None;
        self.game_over = false;
    }


    fn get_result(&self, move_col: f32, move_row: f32) -> Option<f32> {
        // Checks if the game is over and returns the result
        // 1 if player 1 wins, 0 if draw, None if game is not over
        //
        // General method: 
        // 1. Get sub_tensor - row, col or diag of last move.
        // 2. Multiply sub_tensor by self.current_player
        // 3. Apply ReLU
        // 4. For each 1x4 sub_tensor, take the dot product with player_tensor
        // 5. If any of the dot products are 4, return the current player

        // Init 1x4 tensor
        let player_tensor = Tensor::ones(&[4], (Kind::Float, Device::Cpu)).squeeze();

        // Vertical check
        let mut sub_tensor = self.board.narrow_copy(1, move_col as i64, 1);
        sub_tensor *= self.current_player;
        sub_tensor = sub_tensor.relu();

        if bool::from(sub_tensor.squeeze().narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) 
            ||
           bool::from(sub_tensor.squeeze().narrow(0, 1, 4).dot(&player_tensor).eq(4.0)) 
            ||
           bool::from(sub_tensor.squeeze().narrow(0, 2, 4).dot(&player_tensor).eq(4.0)) 
        {
            return Some(self.current_player);
        }

        // Horizontal check
        sub_tensor = self.board.narrow_copy(0, move_row as i64, 1);
        sub_tensor *= self.current_player;
        sub_tensor = sub_tensor.relu();
        if bool::from(sub_tensor.squeeze().narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) 
            ||
           bool::from(sub_tensor.squeeze().narrow(0, 1, 4).dot(&player_tensor).eq(4.0)) 
            ||
           bool::from(sub_tensor.squeeze().narrow(0, 2, 4).dot(&player_tensor).eq(4.0)) 
            ||
           bool::from(sub_tensor.squeeze().narrow(0, 3, 4).dot(&player_tensor).eq(4.0)) 
        {
            return Some(self.current_player);
        }

        // Diag lengths: 1 2 3 4 5 6 6 5 4 3 2 1
        // Diag colsum mappings: 0 1 2 3 4 5 5 4 3 2 1 0

        // Diagonal check 1
        let mut diag_len = (move_row + move_col + 1.0) as i64;
        let mut upper = true;
        if diag_len > 6 {
            upper = false;
            diag_len = 12 - diag_len;
        }

        if upper {
            sub_tensor = self.board.narrow(0, 0, diag_len).narrow_copy(1, 0, diag_len).fliplr().diag(0).view([-1]);
        }
        else {
            sub_tensor = self.board.narrow(0, 6 - diag_len, diag_len).narrow_copy(1, 7 - diag_len, diag_len).fliplr().diag(0).view([-1]);
        }
        sub_tensor *= self.current_player;
        sub_tensor = sub_tensor.relu();

        if diag_len == 4 {
            if bool::from(sub_tensor.squeeze().narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) {
                return Some(self.current_player);
            }
        }
        else if diag_len == 5 {
            if bool::from(sub_tensor.squeeze().narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) 
                ||
               bool::from(sub_tensor.squeeze().narrow(0, 1, 4).dot(&player_tensor).eq(4.0)) 
            {
                return Some(self.current_player);
            }
        }
        else if diag_len == 6 {
            if bool::from(sub_tensor.squeeze().narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) 
                ||
               bool::from(sub_tensor.squeeze().narrow(0, 1, 4).dot(&player_tensor).eq(4.0)) 
                ||
               bool::from(sub_tensor.squeeze().narrow(0, 2, 4).dot(&player_tensor).eq(4.0)) 
            {
                return Some(self.current_player);
            }
        }

        // Diagonal check 2
        diag_len = (move_row + (6.0 - move_col) + 1.0) as i64;
        upper = true;
        if diag_len > 6 {
            upper = false;
            diag_len = 13 - diag_len;
        }

        if upper {
            sub_tensor = self.board.narrow(0, 0, diag_len).narrow_copy(1, 7 - diag_len, diag_len).diag(0).view([-1]);
        }
        else {
            sub_tensor = self.board.narrow(0, 6 - diag_len, diag_len).narrow_copy(1, 0, diag_len).diag(0).view([-1]);
        }
        sub_tensor *= self.current_player;
        sub_tensor = sub_tensor.relu();

        if diag_len == 4 {
            if bool::from(sub_tensor.narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) {
                return Some(self.current_player);
            }
        }
        else if diag_len == 5 {
            if bool::from(sub_tensor.narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) 
                ||
               bool::from(sub_tensor.narrow(0, 1, 4).dot(&player_tensor).eq(4.0)) 
            {
                return Some(self.current_player);
            }
        }
        else if diag_len == 6 {
            if bool::from(sub_tensor.narrow(0, 0, 4).dot(&player_tensor).eq(4.0)) 
                ||
               bool::from(sub_tensor.narrow(0, 1, 4).dot(&player_tensor).eq(4.0)) 
                ||
               bool::from(sub_tensor.narrow(0, 2, 4).dot(&player_tensor).eq(4.0)) 
            {
                return Some(self.current_player);
            }
        }
        
        if bool::from(self.board.abs().sum(Kind::Float).eq(42.0)) {
            return Some(0.0);
        }

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
        let mut game = Connect4GameTorch::new();

        // Test vertical win player 1
        assert_eq!(game.get_result(0.0, 0.0), None);
        game.make_move(0.0); game.make_move(1.0);
        game.make_move(0.0); game.make_move(1.0);
        game.make_move(0.0); game.make_move(1.0);
        assert_eq!(game.make_move(0.0), Some(1.0));

        game.reset_board();

        // Test vertical win player 2
        game.make_move(0.0); game.make_move(1.0);
        game.make_move(0.0); game.make_move(1.0);
        game.make_move(4.0); game.make_move(1.0);
        game.make_move(4.0); assert_eq!(game.make_move(1.0), Some(-1.0));

        game.reset_board();

        // Test horizontal win
        game.make_move(0.0); game.make_move(6.0);
        game.make_move(1.0); game.make_move(6.0);
        game.make_move(2.0); game.make_move(6.0);
        assert_eq!(game.make_move(3.0), Some(1.0));

        game.reset_board();

        // Test diagonal win reverse diag
        game.make_move(0.0); game.make_move(1.0);
        game.make_move(1.0); game.make_move(2.0);
        game.make_move(2.0); game.make_move(3.0);
        game.make_move(2.0); game.make_move(3.0);
        
        game.make_move(3.0); game.make_move(6.0);
        assert_eq!(game.make_move(3.0), Some(1.0));

        game.reset_board();

        // Test diagonal win standard diag
        game.make_move(3.0); game.make_move(2.0);
        game.make_move(2.0); game.make_move(1.0);
        game.make_move(1.0); game.make_move(0.0);
        game.make_move(1.0); game.make_move(0.0);
        game.make_move(0.0); game.make_move(4.0);
        assert_eq!(game.make_move(0.0), Some(1.0));
    }

    #[test]
    fn test_get_possible_moves() {
        let mut game = Connect4GameTorch::new();
        assert_eq!(game.get_possible_moves(), Tensor::of_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        game.make_move(0.0);
        assert_eq!(game.get_possible_moves(), Tensor::of_slice(&[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    }
}

