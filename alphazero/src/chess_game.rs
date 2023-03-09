use rand::seq::IteratorRandom;
use chess::{ ChessMove, GameResult, Board, MoveGen };

#[allow(dead_code)]
pub struct ChessGame {
    board: Board,
    current_player: i8,
    result: Option<i8>,
    move_hash: [ChessMove; 1744],
}

// 8 x 8 x 73 = 1744

pub trait ChessFuncs {
    fn new() -> Self;
    fn make_move(&mut self, _move: usize) -> Option<i8>;
    fn _make_move(&mut self, _move: ChessMove) -> Option<i8>;
    fn get_move_mask(&self) -> [f32; 1744];
    fn make_move_random(&mut self) -> Option<i8>;
    fn get_current_player(&self) -> i8;
    fn get_board(&self) -> Board;
}


impl ChessFuncs for ChessGame {
    fn new() -> Self {
        ChessGame {
            board: Board::default(),
            current_player: 1,
            result: None,
            move_hash: [ChessMove::null(); 1744],
        }
    }

    fn get_move_mask(&self) -> [f32; 1744] {
        let mask = [0.00; 1744];
        return mask;
    }

    fn make_move(&self, move_idx: usize) -> Option<i8> {
        let _move = self.move_hash[move_idx];
        return self._make_move(_move);
    }

    fn _make_move(&mut self, _move: ChessMove) -> Option<i8> {
        let new_board = self.board.clone();
        new_board.make_move(_move, &mut self.board);

        match self.board.status() {
            chess::BoardStatus::Checkmate => {
                return Some(self.current_player);
            },
            chess::BoardStatus::Stalemate => {
                return Some(0);
            },
            chess::BoardStatus::Ongoing => {
                self.current_player *= -1; 
                return None;
            }
        }
    }

    fn make_move_random(&mut self) -> Option<i8> {
        let iterable = MoveGen::new_legal(&self.board);

        let mut rng = rand::thread_rng();
        let random_move = iterable.choose(&mut rng).unwrap();

        return self._make_move(random_move);
    }

    fn get_current_player(&self) -> i8 {
        return self.current_player;
    }

    fn get_board(&self) -> Board {
        return self.board.clone();
    }

}
