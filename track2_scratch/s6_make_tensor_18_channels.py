from pathlib import Path

import numpy as np
import chess.pgn
from track2_scratch.s3_make_tensor_12_channels import make_tensor_12

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def make_tensor_18(board):

    encoded = np.zeros((18, 8, 8), dtype=np.float32)

    encoded[:12] = make_tensor_12(board)
    encoded[12, :, :] = float(board.turn == chess.WHITE)

    encoded[13, :, :] = float(
        board.has_kingside_castling_rights(chess.WHITE)
    )
    encoded[14, :, :] = float(
        board.has_queenside_castling_rights(chess.WHITE)
    )
    encoded[15, :, :] = float(
        board.has_kingside_castling_rights(chess.BLACK)
    )
    encoded[16, :, :] = float(
        board.has_queenside_castling_rights(chess.BLACK)
    )
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)

        encoded[17, row, col] = 1

    return encoded
with open(DATA_DIR / "Chessgame.pgn", encoding="utf-8-sig") as file:
    game = chess.pgn.read_game(file) 
    board = game.board()
    tensor = make_tensor_18(board)
    print(tensor)
