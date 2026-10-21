from pathlib import Path

import chess.pgn

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "Chessgame.pgn", encoding="utf-8-sig") as file:
    game = chess.pgn.read_game(file)        
board = game.board()

for move in game.mainline_moves():
    print(move.uci())
    board.push(move)
    print(board)