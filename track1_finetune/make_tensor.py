from pathlib import Path

import numpy as np
import chess.pgn

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def make_tensor(board):
    tensor = np.zeros((12,8, 8,), dtype=np.int32)
    for square, piece in board.piece_map().items():
        channel = piece.piece_type - 1

        if piece.color == chess.BLACK:
            channel += 6

        row = 7 - square // 8
        col = square % 8

        tensor[channel, row, col] = 1

    return tensor
with open(DATA_DIR / "Chessgame.pgn", encoding="utf-8-sig") as file:
    game = chess.pgn.read_game(file) 
    board = game.board()
    tensor = make_tensor(board)
    print(tensor)
