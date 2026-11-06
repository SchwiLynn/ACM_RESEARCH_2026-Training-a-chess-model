"""Evaluate Maia-3 5M through UCI on the frozen player test sets.

Install Maia-3 from https://github.com/CSSLab/maia3 first.
Colab example (run from the project root after mounting the shared data):
    python track2_finetunemaia3/s8_maia3_baseline.py --limit 1000
    python track2_finetunemaia3/s8_maia3_baseline.py

Default outputs go to MyDrive/acm_chess/baselines_maia3 on Colab.
Use --history none for a separate FEN-only comparison.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import chess
import chess.engine
import chess.pgn
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared"))
from evaluation_metrics import evaluate_predictions, format_report


def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      promotion=move.promotion)


def load_games(path, wanted):
    """Keep only test games, without importing Maia-2's dataset builder."""
    games = {}
    with path.open(encoding="utf-8-sig") as stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            gid = game.headers.get("Site", "").rstrip("/").rsplit("/", 1)[-1]
            if gid in wanted:
                if gid in games:
                    raise ValueError(f"Duplicate PGN game_id {gid} in {path}")
                if game.errors:
                    raise ValueError(f"Invalid PGN game {gid}: {game.errors}")
                games[gid] = (game.board(), list(game.mainline_moves()))
    missing = wanted - games.keys()
    if missing:
        raise ValueError(f"PGN is missing test games: {sorted(missing)[:5]}")
    return games


def position_for_row(row, games):
    """Return an original-orientation board, preserving its move stack."""
    mirrored = int(row.mirrored)
    if mirrored not in (0, 1):
        raise ValueError("mirrored must be 0 or 1")
    normalized = chess.Board(row.board)
    original = normalized.mirror() if mirrored else normalized
    expected_move = chess.Move.from_uci(row.move)
    original_move = mirror_move(expected_move) if mirrored else expected_move
    if original_move not in original.legal_moves:
        raise ValueError(f"Illegal target move at {row.game_id}, ply {row.ply}")
    if games is None:
        return original
    initial, moves = games[str(row.game_id)]
    ply = int(row.ply)
    if not 1 <= ply <= len(moves):
        raise ValueError(f"Invalid ply {ply} in game {row.game_id}")
    board = initial.copy()
    # The target move itself must never enter the model's history.
    for move in moves[:ply - 1]:
        board.push(move)
    # Board.mirror changes the fullmove counter's interpretation for Black.
    # Compare position fields and halfmove clock, not the fullmove counter.
    if board.fen().split()[:5] != original.fen().split()[:5]:
        raise ValueError(f"PGN/CSV position mismatch: {row.game_id}, ply {ply}")
    if moves[ply - 1] != original_move:
        raise ValueError(f"PGN/CSV target mismatch: {row.game_id}, ply {ply}")
    return board


def predict(engine, data, games):
    predictions = []
    start = time.perf_counter()
    for index, row in enumerate(data.itertuples(index=False), start=1):
        board = position_for_row(row, games)
        infos = engine.analyse(
            board, chess.engine.Limit(depth=1), multipv=3,
            options={"SelfElo": int(row.active_elo),
                     "OppoElo": int(row.opponent_elo)},
        )
        moves = [info["pv"][0] for info in sorted(
            infos, key=lambda info: info.get("multipv", 1)) if info.get("pv")]
        if len(moves) != min(3, board.legal_moves.count()):
            raise ValueError(f"Incomplete MultiPV at {row.game_id}, ply {row.ply}")
        if len(set(moves)) != len(moves) or any(m not in board.legal_moves for m in moves):
            raise ValueError("Engine returned duplicate or illegal moves")
        if int(row.mirrored):
            moves = [mirror_move(move) for move in moves]
        values = [move.uci() for move in moves]
        predictions.append(values + [None] * (3 - len(values)))
        if index % 1000 == 0:
            print(f"  {index}/{len(data)} positions", flush=True)
    elapsed = time.perf_counter() - start
    output = data.copy()
    output[["prediction1", "prediction2", "prediction3"]] = predictions
    return output, elapsed


def compare_maia2(path, data):
    baseline = pd.read_csv(path, dtype={"game_id": str})
    keys = ["game_id", "ply"]
    if baseline.duplicated(keys).any():
        raise ValueError(f"Duplicate positions in {path}")
    aligned = data[keys + ["board", "move"]].merge(
        baseline, on=keys, how="left", suffixes=("_test", ""),
        validate="one_to_one", indicator=True)
    if (aligned["_merge"] != "both").any():
        raise ValueError(f"Maia-2 baseline does not cover the selected test rows: {path}")
    for column in ("board", "move"):
        if not aligned[column].eq(aligned[f"{column}_test"]).all():
            raise ValueError(f"Maia-2 {column} differs from the frozen test set: {path}")
    return evaluate_predictions(aligned)


def main():
    drive = Path("/content/drive/MyDrive/acm_chess")
    storage = drive if drive.exists() else ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=storage / "data")
    parser.add_argument("--out-dir", type=Path, default=storage / "baselines_maia3")
    parser.add_argument("--maia2-dir", type=Path, default=storage / "baselines")
    parser.add_argument("--players", nargs="+", help="default: every split folder with test.csv")
    parser.add_argument("--history", choices=["pgn", "none"], default="pgn")
    parser.add_argument("--limit", type=int, help="positions per player for timing only")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--timeout", type=float, default=600,
                        help="engine timeout including initial model download, in seconds")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    players = args.players or sorted(p.parent.name for p in
                                     (args.data_root / "splits").glob("*/test.csv"))
    if not players:
        parser.error("No frozen test sets found. Set --data-root to the shared data folder.")
    command = [sys.executable, "-m", "maia3.uci", "--model", "maia3-5m",
               "--device", args.device, "--temperature", "0", "--multipv", "3"]
    if args.history == "pgn":
        command.append("--use-uci-history")
    destination = args.out_dir / args.history
    if args.limit:
        destination = destination / f"sample_{args.limit}"
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    with chess.engine.SimpleEngine.popen_uci(command, timeout=args.timeout) as engine:
        # Maia-3 loads its checkpoint on isready. Exclude that from timing.
        engine.ping()
        for player in players:
            print(f"\n===== {player} ({args.history}) =====", flush=True)
            data = pd.read_csv(args.data_root / "splits" / player / "test.csv",
                               dtype={"game_id": str})
            required = {"board", "move", "active_elo", "opponent_elo",
                        "game_id", "ply", "mirrored"}
            if required - set(data.columns):
                raise ValueError(f"Missing test columns: {sorted(required - set(data.columns))}")
            if data.empty or data[list(required)].isna().any().any():
                raise ValueError(f"Empty or incomplete test set for {player}")
            if data.duplicated(["game_id", "ply"]).any():
                raise ValueError(f"Duplicate test positions for {player}")
            if args.limit:
                data = data.head(args.limit).copy()
            games = None
            if args.history == "pgn":
                games = load_games(args.data_root / "raw" / f"{player}_rapid.pgn",
                                   set(data.game_id))
            output, elapsed = predict(engine, data, games)
            target = destination / f"{player}.csv"
            temporary = target.with_suffix(".csv.tmp")
            output.to_csv(temporary, index=False)
            os.replace(temporary, target)
            report = evaluate_predictions(output)
            print(format_report(report))
            speed = len(output) / elapsed
            print(f"{speed:.2f} positions/sec; estimated 240k positions: {240000 / speed / 3600:.2f} hours")
            record = {"player": player, "history": args.history,
                      "sample_limit": args.limit, "positions": len(output),
                      "games": report["n_games"], "seconds": elapsed,
                      "positions_per_second": speed,
                      "maia3_top1": report["top1"]["acc"],
                      "maia3_top3": report["top3"]["acc"]}
            baseline_path = args.maia2_dir / f"{player}.csv"
            if baseline_path.exists():
                baseline = compare_maia2(baseline_path, data)
                record.update(maia2_top1=baseline["top1"]["acc"],
                              maia2_top3=baseline["top3"]["acc"])
            else:
                print(f"Maia-2 comparison pending: missing {baseline_path}")
            metadata = {"engine_command": command, "engine": engine.id,
                        "summary": record, "evaluation": report}
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            summary.append(record)
            pd.DataFrame(summary).to_csv(destination / "comparison.csv", index=False)
    print("\nAccuracies are fractions (0–1):")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
