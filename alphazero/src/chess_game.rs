use std::collections::HashMap;
use rand::seq::IteratorRandom;
use chess::{ ChessMove, GameResult, Board, MoveGen };

use crate::move_map::*;

#[allow(dead_code)]
pub struct ChessGame<'a> {
    board: Board,
    current_player: i8,
    result: Option<i8>,
    move_hash: &'a HashMap<ChessMove, usize>
}


pub trait ChessFuncs<'a> {
    fn new<'b>(move_hash: &'b HashMap<ChessMove, usize>) -> Self where 'b: 'a;
    fn make_move(&mut self, _move: usize) -> Option<i8>;
    fn _make_move(&mut self, _move: ChessMove) -> Option<i8>;
    fn get_move_mask(&self) -> [usize; 1968];
    fn make_move_random(&mut self) -> Option<i8>;
    fn get_current_player(&self) -> i8;
    fn get_board(&self) -> Board;
}


impl<'a> ChessFuncs<'a> for ChessGame<'a> {
    fn new<'b>(_move_hash: &'b HashMap<ChessMove, usize>) -> Self where 'b: 'a {
        ChessGame {
            board: Board::default(),
            current_player: 1,
            result: None,
            move_hash: _move_hash,
        }
    }

    fn get_move_mask(&self) -> [usize; 1968] {
        let mut mask = [0; 1968];
        let iterable = MoveGen::new_legal(&self.board);
        for _move in iterable {
            let chess_move = match self.move_hash.get(&_move) {
                Some(chess_move) => chess_move,
                None => panic!("Move {:?} not found in move hash", _move),
            };
            mask[*chess_move] = 1;
        }
        return mask;
    }

    fn make_move(&mut self, move_idx: usize) -> Option<i8> {
        let _move = match self.move_hash.keys().nth(move_idx) {
            Some(_move) => _move,
            None => panic!("Index out of range"),
        };
        return self._make_move(*_move);
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

        let mask = self.get_move_mask();

        let random_move = match iterable
            .filter(|_move| mask[*self.move_hash.get(_move).unwrap()] == 1)
            .choose(&mut rand::thread_rng()) {
                Some(_move) => _move,
                None => panic!("No legal moves"),
        };

        return self._make_move(random_move);
    }

    fn get_current_player(&self) -> i8 {
        return self.current_player;
    }

    fn get_board(&self) -> Board {
        return self.board.clone();
    }

}
