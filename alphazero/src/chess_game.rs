use std::collections::HashMap;
use rand::seq::IteratorRandom;
use chess::{ ChessMove, GameResult, Board, MoveGen };

use crate::move_map::*;

#[allow(dead_code)]
pub struct ChessGame {
    board: Board,
    current_player: i8,
    result: Option<i8>,
    move_hash: HashMap<ChessMove, usize>
}


impl ChessGame {
    pub fn new(_move_hash: HashMap<ChessMove, usize>) -> Self {
        ChessGame {
            board: Board::default(),
            current_player: 1,
            result: None,
            move_hash: _move_hash,
        }
    }

    pub fn get_move_mask(&self) -> [f32; 1968] {
        let mut mask = [0.00; 1968];
        let iterable = MoveGen::new_legal(&self.board);
        for _move in iterable {
            let chess_move = match self.move_hash.get(&_move) {
                Some(chess_move) => chess_move,
                None => panic!("Move {:?} not found in move hash", _move),
            };
            mask[*chess_move] = 1.00;
        }
        return mask;
    }

    pub fn make_move(&mut self, move_idx: usize) -> Option<i8> {
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

    pub fn make_move_random(&mut self) -> Option<i8> {
        let iterable = MoveGen::new_legal(&self.board);

        let mask = self.get_move_mask();

        let random_move = match iterable
            .filter(|_move| mask[*self.move_hash.get(_move).unwrap()] == 1.00)
            .choose(&mut rand::thread_rng()) {
                Some(_move) => _move,
                None => panic!("No legal moves"),
        };

        return self._make_move(random_move);
    }

    pub fn get_current_player(&self) -> i8 {
        return self.current_player;
    }

    pub fn get_board(&self) -> [u8; 768] {
        /*
        ------------------------------------------------------------------------------------------------------------
        Board mapping representation:
        0: Empty
        1: Piece
        
        Boolean array where piece at 12 * square_idx + piece_idx is 1
        piece_idxs: White Pawn: 0, 
                    White Knight: 1, 
                    White Bishop: 2, 
                    White Rook: 3, 
                    White Queen: 4, 
                    White King: 5

                    Black Pawn: 6, 
                    Black Knight: 7, 
                    Black Bishop: 8, 
                    Black Rook: 9, 
                    Black Queen: 10, 
                    Black King: 11

        square_idx: rank * 8 + file where file is 0-7 for a-h
        ------------------------------------------------------------------------------------------------------------
        */
        let mut representation: [u8; 768] = [0; 768];

        let white_bitboard = self.board.color_combined(chess::Color::White);
        let black_bitboard = self.board.color_combined(chess::Color::Black);

        let pawn_bitboard   = self.board.pieces(chess::Piece::Pawn);
        let knight_bitboard = self.board.pieces(chess::Piece::Knight);
        let bishop_bitboard = self.board.pieces(chess::Piece::Bishop);
        let rook_bitboard   = self.board.pieces(chess::Piece::Rook);
        let queen_bitboard  = self.board.pieces(chess::Piece::Queen);
        let king_bitboard   = self.board.pieces(chess::Piece::King);

        representation[0..64].copy_from_slice(&(white_bitboard & pawn_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[64..128].copy_from_slice(&(white_bitboard & knight_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[128..192].copy_from_slice(&(white_bitboard & bishop_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[192..256].copy_from_slice(&(white_bitboard & rook_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[256..320].copy_from_slice(&(white_bitboard & queen_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[320..384].copy_from_slice(&(white_bitboard & king_bitboard).to_size(0).to_be_bytes().to_vec());

        representation[384..448].copy_from_slice(&(black_bitboard & pawn_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[448..512].copy_from_slice(&(black_bitboard & knight_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[512..576].copy_from_slice(&(black_bitboard & bishop_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[576..640].copy_from_slice(&(black_bitboard & rook_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[640..704].copy_from_slice(&(black_bitboard & queen_bitboard).to_size(0).to_be_bytes().to_vec());
        representation[704..768].copy_from_slice(&(black_bitboard & king_bitboard).to_size(0).to_be_bytes().to_vec());

        return representation;
    }
}

