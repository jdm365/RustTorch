use std::collections::HashMap;
use std::sync::Arc;

use rand::seq::IteratorRandom;
use chess::{ ChessMove, GameResult, Board, MoveGen, Rank, File, Square };

use crate::move_map::*;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ChessGame {
    board: Board,
    current_player: i8,
    result: Option<i8>,
    move_hash: Arc<HashMap<ChessMove, usize>>
}


impl ChessGame {
    pub fn new(move_hash: Arc<HashMap<ChessMove, usize>>) -> Self {
        ChessGame {
            board: Board::default(),
            current_player: 1,
            result: None,
            move_hash,
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
        let _move = match self.move_hash.iter().find(|(_move, &idx)| idx == move_idx) {
            Some(_move) => _move.0,
            None => panic!("Index out of range"),
        };
        return self._make_move(*_move);
    }

    fn _make_move(&mut self, _move: ChessMove) -> Option<i8> {
        let new_board = self.board.clone();
        new_board.make_move(_move, &mut self.board);

        match self.board.status() {
            chess::BoardStatus::Checkmate => {
                self.result = Some(self.current_player * -1);
                return Some(self.current_player);
            },
            chess::BoardStatus::Stalemate => {
                self.result = Some(0);
                return Some(0);
            },
            chess::BoardStatus::Ongoing => {
                self.current_player *= -1; 
                return None;
            }
        }
    }

    pub fn get_status(&self) -> Option<i8> {
        return self.result;
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

    pub fn get_board_old(&self) -> [u8; 768] {
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
        let mut mask: [u8; 768] = [0; 768];

        let white_bitboard = self.board.color_combined(chess::Color::White);
        let black_bitboard = self.board.color_combined(chess::Color::Black);

        let pawn_bitboard   = self.board.pieces(chess::Piece::Pawn);
        let knight_bitboard = self.board.pieces(chess::Piece::Knight);
        let bishop_bitboard = self.board.pieces(chess::Piece::Bishop);
        let rook_bitboard   = self.board.pieces(chess::Piece::Rook);
        let queen_bitboard  = self.board.pieces(chess::Piece::Queen);
        let king_bitboard   = self.board.pieces(chess::Piece::King);

        // Endianness not important as long as it is consistent
        mask[0..64].copy_from_slice(&u64_tobit_array((white_bitboard & pawn_bitboard).to_size(0) as u64));
        mask[64..128].copy_from_slice(&u64_tobit_array((white_bitboard & knight_bitboard).to_size(0) as u64));
        mask[128..192].copy_from_slice(&u64_tobit_array((white_bitboard & bishop_bitboard).to_size(0) as u64));
        mask[192..256].copy_from_slice(&u64_tobit_array((white_bitboard & rook_bitboard).to_size(0) as u64));
        mask[256..320].copy_from_slice(&u64_tobit_array((white_bitboard & queen_bitboard).to_size(0) as u64));
        mask[320..384].copy_from_slice(&u64_tobit_array((white_bitboard & king_bitboard).to_size(0) as u64));

        mask[384..448].copy_from_slice(&u64_tobit_array((black_bitboard & pawn_bitboard).to_size(0) as u64));
        mask[448..512].copy_from_slice(&u64_tobit_array((black_bitboard & knight_bitboard).to_size(0) as u64));
        mask[512..576].copy_from_slice(&u64_tobit_array((black_bitboard & bishop_bitboard).to_size(0) as u64));
        mask[576..640].copy_from_slice(&u64_tobit_array((black_bitboard & rook_bitboard).to_size(0) as u64));
        mask[640..704].copy_from_slice(&u64_tobit_array((black_bitboard & queen_bitboard).to_size(0) as u64));
        mask[704..768].copy_from_slice(&u64_tobit_array((black_bitboard & king_bitboard).to_size(0) as u64));

        return mask;
    }

    pub fn get_board(&self) -> [i32; 64] {
        // Smaller function which gets nn board representation.
        // Piece + Position embedding index -> [0, 64 * 13) = [0, 832) - 32 for pawns on respective first rank.
        // [0, 800) for all 64 squares.
        // NOTE: Going to start with [0, 832) to limit complexity.
        // get embedding idx by square_idx * 13 + piece_idx + 6 * (is_black)
        let mut representation: [i32; 64] = [0; 64];
        for idx in 0..64 {
            let rank = Rank::from_index((idx / 8) as usize);
            let file = File::from_index(idx % 8);
            let square = Square::make_square(rank, file);
            representation[idx] = match self.board.piece_on(square) {
                Some(piece) => {
                    let offset = match self.board.color_on(square) {
                        Some(color) => match color {
                            chess::Color::White => 0,
                            chess::Color::Black => 6,
                        },
                        None => panic!("No color on square"),
                    };
                    let piece_idx = match piece {
                        chess::Piece::Pawn => 0 + offset,
                        chess::Piece::Knight => 1 + offset,
                        chess::Piece::Bishop => 2 + offset,
                        chess::Piece::Rook => 3 + offset,
                        chess::Piece::Queen => 4 + offset,
                        chess::Piece::King => 5 + offset,
                    };
                    (13 * idx) as i32 + piece_idx
                },
                None => 0,
            };
        }

        return representation;
        }
}



#[inline]
fn u64_tobit_array(value: u64) -> [u8; 64] {
    let mut array = [0; 64];
    for i in 0..64 {
        array[i] = ((value >> i) & 1) as u8;
    }
    return array;
}

