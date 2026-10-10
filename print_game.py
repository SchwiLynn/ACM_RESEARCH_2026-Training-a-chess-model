import chess.pgn
import pandas as pd
with open("Chessgame.pgn", encoding="utf-8-sig") as file:
    game = chess.pgn.read_game(file)        
board = game.board()

for move in game.mainline_moves():
    print(board.san(move))
    board.push(move)
    print(board)