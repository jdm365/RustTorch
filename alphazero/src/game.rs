use tch::{ Tensor, Kind, Device };


struct Connect4Game {
    board: Tensor,
    current_player: f32,
    result: Option<f32>,
}


trait Game {
    fn new() -> Self;
    fn get_result(&self, move_col: f32, move_row: f32) -> Option<f32>;
    fn get_possible_moves(&self) -> Tensor;
    fn make_move(&mut self, move_col: f32);
}


impl Game for Connect4Game {
    fn new() -> Self {
        Connect4Game {
            board: Tensor::zeros(&[6, 7], (Kind::Float, Device::Cpu)),
            current_player: 1.0,
            result: None,
        }
    }

    fn get_result(&self, move_col: f32, move_row: f32) -> Option<f32> {
        // Checks if the game is over and returns the result
        // 1 if player 1 wins, 0 if draw, None if game is not over
        // If we have the last move then we can reduce the 
        // number of terminal checks from 120 to 13. -> Might be wrong on former
        let connect4 = Tensor::of_slice(&[
            self.current_player, 
            self.current_player, 
            self.current_player, 
            self.current_player
        ]).to_kind(Kind::Float);

        // Vertical check
        if move_row <= 2.0 {
            if self.board.narrow(1, move_col as i64, 4).equal(&connect4) {
                return Some(self.current_player);
            }
        }

        // Horizontal checks
        let n_checks = 4 - (move_col as i64 - 3).abs() % 4;
        let left_to_right = move_col <= 3.0;

        if left_to_right {
            for col in 0..n_checks {
                if self.board.get(move_row as i64).narrow(0, col, 4).equal(&connect4) {
                    return Some(self.current_player);
                }
            }
        }
        else {
            for col in 0..n_checks {
                if self.board.get(move_row as i64).narrow(0, 6 - col, 4).equal(&connect4) {
                    return Some(self.current_player);
                }
            }
        }

        // Diagonal check 1

        

        return None;
        }


    fn get_possible_moves(&self) -> Tensor {
        // Returns a tensor of possible moves
        // 1 if the move is possible, 0 if not
        let zero = Tensor::zeros(&[7], (Kind::Float, Device::Cpu));

        return self.board.get(0).eq_tensor(&zero).to_kind(Kind::Float);
    }


    fn make_move(&mut self, move_col: f32) {
        // Makes a move



        self.current_player = -self.current_player;

        // Check if the game is over
        self.result = self.get_result(move_col, move_row);

        match self.result {
            Some(_) => println!("Game over!"),
            None => {},
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
        assert_eq!(game.get_result(), None);
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        assert_eq!(game.get_result(), Some(1));
    }

    #[test]
    fn test_get_possible_moves() {
        let mut game = Connect4Game::new();
        assert_eq!(game.get_possible_moves(), Tensor::of_slice(&[1, 1, 1, 1, 1, 1, 1]));
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        assert_eq!(game.get_possible_moves(), Tensor::of_slice(&[0, 1, 1, 1, 1, 1, 1]));
    }
}

