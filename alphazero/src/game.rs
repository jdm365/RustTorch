pub struct Connect4Game {
    board: [i8; 42],
    current_player: i8,
    result: Option<i8>,
    game_over: bool,
}


pub trait Game {
    fn new() -> Self;
    fn get_result(&self, move_col: usize, move_row: usize) -> Option<i8>;
    fn get_possible_moves(&self) -> Vec<i8>;
    fn make_move(&mut self, move_col: usize) -> Option<i8>;
    fn reset_board(&mut self);
}

fn relu(x: i8) -> i8 {
    if x > 0 {
        return x;
    } else {
        return 0;
    }
}


impl Game for Connect4Game {
    fn new() -> Self {
        Connect4Game {
            board: [0; 42],
            current_player: 1,
            result: None,
            game_over: false,
        }
    }

    fn reset_board(&mut self) {
        self.board = [0; 42];
        self.current_player = 1;
        self.result = None;
        self.game_over = false;
    }


    fn get_result(&self, move_col: usize, move_row: usize) -> Option<i8> {
        // Checks if the game is over and returns the result
        // 1 if player 1 wins, 0 if draw, None if game is not over
        //
        // General method: 
        // 1. Get sub_tensor - row, col or diag of last move.
        // 2. Multiply sub_tensor by self.current_player
        // 3. Apply ReLU
        // 4. For each 1x4 sub_tensor, take the dot product with player_tensor
        // 5. If any of the dot products are 4, return the current player


        // Vertical check
        let mut sum = 0;
        for row in 0..6 {
            sum += 1;
            if relu(self.current_player * self.board[row * 7 + move_col]) == 0 {
                sum = 0;
            }
            if sum == 4 {
                return Some(self.current_player);
            }
        }


        // Horizontal check
        sum = 0;
        for col in 0..7 {
            sum += 1;
            if relu(self.current_player * self.board[move_row * 7 + col]) == 0 {
                sum = 0;
            }
            if sum == 4 {
                return Some(self.current_player);
            }
        }


        // Diagonal check 1
        let mut diag_len = (move_row + (6 - move_col) + 1) as usize;
        if diag_len > 6 {
            diag_len = 13 - diag_len;
        }
        sum = 0;
        let move_idx = move_row * 7 + move_col;
        let init_add = ((41 - move_idx) / 8) as usize;
        for i in 0..diag_len {
            sum += 1;
            if relu(self.current_player * self.board[move_idx + 8 * (init_add - i)]) == 0 {
                sum = 0;
            }
            if sum == 4 {
                return Some(self.current_player);
            }
        }


        // Diagonal check 2
        diag_len = (move_row + move_col + 1) as usize;
        if diag_len > 6 {
            diag_len = 12 - diag_len;
        }
        sum = 0;
        let init_add = ((41 - move_idx) / 6) - 1 as usize;
        for i in 0..diag_len {
            sum += 1;
            if relu(self.current_player * self.board[move_idx + 6 * (init_add - i)]) == 0 {
                sum = 0;
            }
            if sum == 4 {
                return Some(self.current_player);
            }
        }

        
        if self.board.iter().all(|&x| x != 0) {
            return Some(0);
        }

        return None;
        }


    fn get_possible_moves(&self) -> Vec<i8> {
        // Returns an array of possible moves
        // 1 if the move is possible, 0 if not
        if self.game_over {
            panic!("Game is over");
        }
        return self.board[0..7].iter().map(|&x| (x == 0) as i8).collect();
    }


    fn make_move(&mut self, move_col: usize) -> Option<i8> {
        // Makes a move

        if self.game_over {
            panic!("Game is over");
        }

        let mut move_row = 0;
        for row in 1..=6 {
            if self.board[(6 - row) * 7 + move_col] == 0 {
                move_row = 6 - row;
                self.board[move_row * 7 + move_col] = self.current_player;
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
                self.current_player *= -1; 
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

        // Test vertical win player 1
        assert_eq!(game.get_result(0, 0), None);
        game.make_move(0); game.make_move(1);
        game.make_move(0); game.make_move(1);
        game.make_move(0); game.make_move(1);
        assert_eq!(game.make_move(0), Some(1));

        game.reset_board();

        // Test vertical win player 2
        game.make_move(0); game.make_move(1);
        game.make_move(0); game.make_move(1);
        game.make_move(4); game.make_move(1);
        game.make_move(4); assert_eq!(game.make_move(1), Some(-1));

        game.reset_board();

        // Test horizontal win
        game.make_move(0); game.make_move(6);
        game.make_move(1); game.make_move(6);
        game.make_move(2); game.make_move(6);
        assert_eq!(game.make_move(3), Some(1));

        game.reset_board();

        // Test diagonal win reverse diag
        game.make_move(0); game.make_move(1);
        game.make_move(1); game.make_move(2);
        game.make_move(2); game.make_move(3);
        game.make_move(2); game.make_move(3);
        
        game.make_move(3); game.make_move(6);
        assert_eq!(game.make_move(3), Some(1));

        game.reset_board();

        // Test diagonal win standard diag
        game.make_move(3); game.make_move(2);
        game.make_move(2); game.make_move(1);
        game.make_move(1); game.make_move(0);
        game.make_move(1); game.make_move(0);
        game.make_move(0); game.make_move(4);
        assert_eq!(game.make_move(0), Some(1));
    }

    #[test]
    fn test_get_possible_moves() {
        let mut game = Connect4Game::new();
        assert_eq!(game.get_possible_moves(), [1, 1, 1, 1, 1, 1, 1]);
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        game.make_move(0);
        println!("{:?}", game.board);
        assert_eq!(game.get_possible_moves(), [0, 1, 1, 1, 1, 1, 1]);
    }
}

