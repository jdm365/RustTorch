def rook_move_gen():
    file_mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}

    moves = []
    for src_rank in range(1, 9):
        for src_file in range(1, 9):
            moves.append('')
            for dst_rank in range(1, 9):

                if (src_rank == dst_rank):
                    continue
                _src_file = file_mapping[src_file]
                moves.append(f'ChessMove::new(chess::Square::{_src_file}{src_rank}, chess::Square::{_src_file}{dst_rank}, None),')

            moves.append('')
            for dst_file in range(1, 9):

                if (src_file == dst_file):
                    continue
                _src_file = file_mapping[src_file]
                _dst_file = file_mapping[dst_file]
                moves.append(f'ChessMove::new(chess::Square::{_src_file}{src_rank}, chess::Square::{_dst_file}{src_rank}), None),')
    assert len(moves) == 16 * 64, f'Expected 896 moves, got {len(moves)}'
    return moves


def bishop_move_gen():
    file_mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
    moves = []

    for src_rank in range(1, 9):
        for src_file in range(1, 9):
            moves.append('')
            for i in range(1, 8):
                dst_rank1, dst_file1 = src_rank + i, src_file + i
                dst_rank2, dst_file2 = src_rank - i, src_file + i
                dst_rank3, dst_file3 = src_rank + i, src_file - i
                dst_rank4, dst_file4 = src_rank - i, src_file - i
                
                if dst_rank1 < 9 and dst_file1 < 9:
                    _src_file = file_mapping[src_file]
                    _dst_file = file_mapping[dst_file1]
                    moves.append(f'ChessMove::new(chess::Square::{_src_file}{src_rank}, chess::Square::{_dst_file}{dst_rank1}), None),')

                if dst_rank2 > 0 and dst_file2 < 9:
                    _src_file = file_mapping[src_file]
                    _dst_file = file_mapping[dst_file2]
                    moves.append(f'ChessMove::new(chess::Square::{_src_file}{src_rank}, chess::Square::{_dst_file}{dst_rank2}), None),')

                if dst_rank3 < 9 and dst_file3 > 0:
                    _src_file = file_mapping[src_file]
                    _dst_file = file_mapping[dst_file3]
                    moves.append(f'ChessMove::new(chess::Square::{_src_file}{src_rank}, chess::Square::{_dst_file}{dst_rank3}), None),')

                if dst_rank4 > 0 and dst_file4 > 0:
                    _src_file = file_mapping[src_file]
                    _dst_file = file_mapping[dst_file4]
                    moves.append(f'ChessMove::new(chess::Square::{_src_file}{src_rank}, chess::Square::{_dst_file}{dst_rank4}), None),')
                
    assert len(moves) == 624, f'Expected 624 moves, got {len(moves)}'
    return moves


def knight_move_gen():
    file_mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
    moves = []

    for src_rank in range(1, 9):
        for src_file in range(1, 9):
            moves.append('')
            dst_ranks = [src_rank+2, src_rank+1, src_rank-1, src_rank-2,
                         src_rank-2, src_rank-1, src_rank+1, src_rank+2
                         ]
            dst_files = [src_file+1, src_file+2, src_file+2, src_file+1,
                         src_file-1, src_file-2, src_file-2, src_file-1
                         ]

            for dst_rank, dst_file in zip(dst_ranks, dst_files):
                if dst_rank < 1 or dst_rank > 8:
                    continue
                if dst_file < 1 or dst_file > 8:
                    continue
                _src_file = file_mapping[src_file]
                _dst_file = file_mapping[dst_file]
                moves.append(f'ChessMove::new(chess::Square::{_src_file}{src_rank}, chess::Square::{_dst_file}{dst_rank}), None),')

    assert len(moves) == 400, f'Expected 400 moves, got {len(moves)}'
    return moves




if __name__ == '__main__':
    rook_moves = rook_move_gen()
    ## Write all rook moves to rook_moves.txt where each move is on a new line
    with open('rook_moves.txt', 'w') as f:
        for move in rook_moves:
            f.write(f'\t{move}\n')

    bishop_moves = bishop_move_gen()
    ## Write all bishop moves to bishop_moves.txt where each move is on a new line
    with open('bishop_moves.txt', 'w') as f:
        for move in bishop_moves:
            f.write(f'\t{move}\n')

    knight_moves = knight_move_gen()
    ## Write all bishop moves to knight_moves.txt where each move is on a new line
    with open('knight_moves.txt', 'w') as f:
        for move in knight_moves:
            f.write(f'\t{move}\n')
