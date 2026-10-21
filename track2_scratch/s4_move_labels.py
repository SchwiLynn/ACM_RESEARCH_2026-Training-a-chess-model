"""S4: turn a chess move into a single number (a "label") and back.

The model's output layer has 64 * 64 = 4096 slots, one for every possible
(from-square, to-square) pair. A move's label is the index of its slot.

python-chess numbers squares 0..63:
    a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63
so  label = from_square * 64 + to_square
"""

import chess

NUM_LABELS = 64 * 64  # 4096


def move_to_label(move: chess.Move) -> int:
    """Move -> number in [0, 4095]."""
    return move.from_square * 64 + move.to_square


def label_to_move(label: int, board: chess.Board | None = None) -> chess.Move:
    """Number -> Move.

    The label only stores two squares, so it cannot say WHICH piece a pawn
    promotes to. If you pass the board, a pawn reaching the last rank is
    returned as a queen promotion (the usual default). Without the board,
    you get a plain from/to move.
    """
    if not 0 <= label < NUM_LABELS:
        raise ValueError(f"label must be in [0, {NUM_LABELS - 1}], got {label}")

    from_square = label // 64   # integer division: how many full 64s fit
    to_square = label % 64      # the remainder

    promotion = None
    if board is not None:
        piece = board.piece_at(from_square)
        last_rank = chess.square_rank(to_square) in (0, 7)
        if piece is not None and piece.piece_type == chess.PAWN and last_rank:
            promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)


if __name__ == "__main__":
    # ---- tests: run with `python s4_move_labels.py` ----

    # 1. Known values: e2e4 -> e2 is square 12, e4 is square 28
    e2e4 = chess.Move.from_uci("e2e4")
    assert move_to_label(e2e4) == 12 * 64 + 28 == 796
    assert label_to_move(796) == e2e4

    # 2. Round trip on a few ordinary moves
    for uci in ["e2e4", "g1f3", "a2a3", "h7h5", "b8c6"]:
        move = chess.Move.from_uci(uci)
        assert label_to_move(move_to_label(move)) == move, uci

    # 3. Edge labels: lowest (a1a1) and highest (h8h8)
    assert label_to_move(0) == chess.Move(0, 0)
    assert label_to_move(4095) == chess.Move(63, 63)

    # 4. Round trip on EVERY legal move from the start position
    board = chess.Board()
    for move in board.legal_moves:
        assert label_to_move(move_to_label(move)) == move
    print(f"start position: all {board.legal_moves.count()} legal moves round-trip OK")

    # 5. Different moves never share a label (within one position)
    labels = [move_to_label(m) for m in board.legal_moves]
    assert len(labels) == len(set(labels))

    # 6. Promotion: queen and knight promotion get the SAME label (the limitation)
    promo_board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    queen = chess.Move.from_uci("a7a8q")
    knight = chess.Move.from_uci("a7a8n")
    assert move_to_label(queen) == move_to_label(knight)
    assert label_to_move(move_to_label(queen), promo_board) == queen  # queen by default
    print("promotion: queen and knight share one label (known limitation)")

    # 7. Bad labels are rejected
    for bad in (-1, 4096):
        try:
            label_to_move(bad)
            raise AssertionError("should have raised")
        except ValueError:
            pass

    print("all S4 tests passed")
