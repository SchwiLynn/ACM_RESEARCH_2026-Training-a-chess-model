"""PGN -> per-move table (one row per half-move).

Produces the columns Maia-2 needs for inference and fine-tuning, plus the
columns we need for filtering and for the date split.

Maia-2's inference requires exactly: board, move, active_elo, opponent_elo.
It ignores any extra columns, so this table works unchanged with F3.

Usage:
    python build_dataset.py data/thakt_rapid.pgn --target thakt -o data/thakt_moves.csv
    python3 shared/build_dataset.py data/thakt_rapid.pgn --target thakt -o data/thakt_all.csv

    # keep only the target player's moves, past the opening, not in time trouble
    python build_dataset.py thakt_rapid.pgn --target thakt --only-target \
        --min-ply 11 --min-clock 30 -o data/thakt_moves.csv
"""

import argparse
import csv
from pathlib import Path

import chess
import chess.pgn

from maia2.utils import create_elo_dict, map_to_category, mirror_move

# Result from White's point of view.
WHITE_SCORE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
ELO_DICT = create_elo_dict()

FIELDS = [
    # --- required by maia2 inference---
    "board",          # FEN *before* the move was played
    "move",           # the move actually played, UCI
    "active_elo",     # rating of the player making this move
    "opponent_elo",
    "active_elo_bucket",    # Maia-2 rating group of the mover (0-10)
    "opponent_elo_bucket",
    # --- needed for the fine-tuning loss (value head) ---
    "active_win",     # game result from the mover's point of view: 1 / -1 / 0
    "mirrored",       # 1 if Black moved (board and move are flipped)
    # --- needed for filtering and splitting ---
    "game_id",
    "date",           # UTCDate, for the newest-20%-is-test split
    "time_control",
    "ply",            # 1-indexed half-move number
    "mover",          # username of the player to move
    "is_target",      # 1 if mover == --target
    "clock",          # mover's remaining seconds AFTER this move ('' if absent)
]


def game_id_from_site(site):
    """https://lichess.org/abcd1234 -> abcd1234"""
    if not site:
        return ""
    return site.rstrip("/").rsplit("/", 1)[-1]


def rows_from_game(game, target):
    h = game.headers

    result = h.get("Result", "*")
    if result not in WHITE_SCORE:
        return  # unfinished / unknown result: no value-head label
    white_score = WHITE_SCORE[result]

    try:
        white_elo = int(h.get("WhiteElo", ""))
        black_elo = int(h.get("BlackElo", ""))
    except ValueError:
        return  

    white = h.get("White", "")
    black = h.get("Black", "")
    gid = game_id_from_site(h.get("Site", ""))
    date = h.get("UTCDate") or h.get("Date", "")
    tc = h.get("TimeControl", "")
    target_lower = target.lower() if target else None

    board = game.board()
    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        if move is None:
            break

        mover_is_white = board.turn == chess.WHITE
        mover = white if mover_is_white else black
        clock = node.clock()  # seconds left after the move, or None

        yield {
            "board": board.fen() if mover_is_white else board.mirror().fen(),            
            "move": move.uci() if mover_is_white else mirror_move(move.uci()),
            "active_elo": white_elo if mover_is_white else black_elo,
            "opponent_elo": black_elo if mover_is_white else white_elo,
            "active_elo_bucket": map_to_category(white_elo if mover_is_white else black_elo, ELO_DICT),
            "opponent_elo_bucket": map_to_category(black_elo if mover_is_white else white_elo, ELO_DICT),
            "active_win": white_score if mover_is_white else - white_score,
            "mirrored": 0 if mover_is_white else 1,
            "game_id": gid,
            "date": date,
            "time_control": tc,
            "ply": ply,
            "mover": mover,
            "is_target": int(target_lower is not None
                             and mover.lower() == target_lower),
            "clock": "" if clock is None else round(clock, 1),
        }

        board.push(move)


def build(pgn_path, out_path, target, only_target, min_ply, min_clock):
    pgn_path = Path(pgn_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games = kept = skipped_games = 0
    dropped_ply = dropped_clock = dropped_not_target = 0
    missing_clock = 0
    dates = []

    with open(pgn_path, encoding="utf-8-sig") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        writer.writeheader()

        while True:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            games += 1

            rows = list(rows_from_game(game, target))
            if not rows:
                skipped_games += 1
                continue

            for row in rows:
                if only_target and not row["is_target"]:
                    dropped_not_target += 1
                    continue
                if row["ply"] < min_ply:
                    dropped_ply += 1
                    continue
                if row["clock"] == "":
                    missing_clock += 1
                elif min_clock is not None and row["clock"] < min_clock:
                    dropped_clock += 1
                    continue

                writer.writerow(row)
                kept += 1
                if row["date"]:
                    dates.append(row["date"])

    print(f"read            {games} games ({skipped_games} skipped: no result or no Elo)")
    if only_target:
        print(f"dropped         {dropped_not_target} moves by the opponent")
    print(f"dropped         {dropped_ply} moves with ply < {min_ply}")
    if min_clock is not None:
        print(f"dropped         {dropped_clock} moves with clock < {min_clock}s")
    if missing_clock:
        print(f"note            {missing_clock} moves had no clock in the PGN (kept)")
    print(f"wrote           {kept} rows -> {out_path}")
    if dates:
        print(f"date range      {min(dates)} .. {max(dates)}")
    if kept == 0:
        print("\nNothing was written. Check --target spelling against the "
              "White/Black headers in the PGN, and whether your filters are "
              "too strict.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pgn", help="input PGN file")
    p.add_argument("--target", default=None,
                   help="Lichess username whose style we want to learn")
    p.add_argument("-o", "--out", default=None,
                   help="output CSV (default: <pgn stem>_moves.csv)")
    p.add_argument("--only-target", action="store_true",
                   help="keep only moves made by --target")
    p.add_argument("--min-ply", type=int, default=1,
                   help="drop moves before this ply (11 skips the first 5 moves "
                        "each side, which are mostly opening theory)")
    p.add_argument("--min-clock", type=float, default=None,
                   help="drop moves with less than this many seconds left")
    args = p.parse_args()

    if args.only_target and not args.target:
        p.error("--only-target needs --target")

    out = args.out or f"{Path(args.pgn).stem}_moves.csv"
    build(args.pgn, out, args.target, args.only_target,
          args.min_ply, args.min_clock)


if __name__ == "__main__":
    main()