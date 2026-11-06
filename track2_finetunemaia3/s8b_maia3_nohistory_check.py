"""S8b: independent Maia-3 baseline check (pre-trained 5M, no fine-tuning, no move history).

A second, independent implementation of S8, used to cross-check Tung's
s8_maia3_baseline.py (run it with --history none to compare). It runs the
Maia-3 model in batches in Python instead of through the UCI engine, and
--check-uci confirms the two give the same top-3 moves.

Does for Maia-3 what F3 does for Maia-2: top-3 predictions on every player's
data/splits/<player>/test.csv, scored with shared/evaluation_metrics.py.

Setup (Colab, after notebook sections 1-3):
    !pip install -q git+https://github.com/CSSLab/maia3.git

Usage (from the repo root):
    # 1. time it first (the first 1,000 positions of one player)
    python track2_finetunemaia3/s8b_maia3_nohistory_check.py data/splits/thakt/test.csv --limit 1000 -o /tmp/thakt_1000.csv

    # 2. one player
    python track2_finetunemaia3/s8b_maia3_nohistory_check.py data/splits/thakt/test.csv -o results_maia3.csv

    # 3. every player + the comparison table with Maia-2's baseline
    python track2_finetunemaia3/s8b_maia3_nohistory_check.py --all data/splits \
        --out-dir /content/drive/MyDrive/acm_chess/baselines_maia3 \
        --maia2-dir /content/drive/MyDrive/acm_chess/baselines

    Add --check-uci 100 to any run to compare the first 100 positions with the
    official Maia-3 UCI engine (MultiPV 3, SelfElo/OppoElo). It should say 100/100.

HOW THE FLIP WORKS
    Our tables store every position from the mover's side: when Black moved,
    `board` and `move` are mirrored (mirrored = 1), so the mover looks like White.
      1. Un-flip: chess.Board(board).mirror() gives the real position (Black to move).
      2. Maia-3 flips Black-to-move boards again inside tokenize_board() and
         predicts in the mover's frame (its move list is written from White's side).
      3. So Maia-3's raw predictions are already in the same frame as our `move`
         column. "Flipping back" = keeping them in that frame instead of converting
         them to the real board, which is what the UCI engine does.

NO HISTORY
    Maia-3 can look at the last 8 positions. Here it only gets the current one,
    repeated 8 times. That is exactly what the official engine does when it is
    not started with --use-uci-history.
"""

import argparse
import hashlib
import sys
import time
from collections import deque
from pathlib import Path

import chess
import chess.engine
import pandas as pd
import torch
from torch.amp import autocast

from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import resolve_checkpoint_path
from maia3.uci import load_model
from maia3.uci import parse_args as maia3_parse_args
from maia3.utils import get_all_possible_moves, mirror_move

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "shared"))
from evaluation_metrics import evaluate_predictions, format_report  # noqa: E402

ALL_MOVES = get_all_possible_moves()              # Maia-3's move list (4,352 moves)
MOVE_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}


# ---------------------------------------------------------------- model

def load_maia3(model="maia3-5m", device="auto", checkpoint_path=None):
    """Load Maia-3 exactly the way its UCI engine does (same config, same weights)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    argv = ["--model", model, "--device", device]
    if checkpoint_path:
        argv += ["--checkpoint-path", str(checkpoint_path)]
    cfg = maia3_parse_args(argv)
    if cfg.checkpoint_path is None:  # download from Hugging Face (cached after the first time)
        cfg.checkpoint_path = resolve_checkpoint_path(
            cfg.model_spec,
            checkpoint_filename=cfg.checkpoint_filename,
            cache_dir=cfg.cache_dir,
            revision=cfg.revision,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            token=cfg.hf_token,
        )
    return load_model(cfg), cfg


def real_board(fen, mirrored):
    """Our stored board -> the real position (un-flip Black's moves)."""
    board = chess.Board(fen)
    return board.mirror() if int(mirrored) == 1 else board


def encode(fen, mirrored, cfg):
    """One position -> (model input, legal-move mask), no history."""
    board = real_board(fen, mirrored)
    history = deque([tokenize_board(board)], maxlen=cfg.history)
    tokens = get_historical_tokens(history, cfg, base=0.0, inc=0.0,
                                   clk_left_before=0.0, clk_ponder=0.0)
    return tokens, get_legal_moves_mask(board, MOVE_INDEX)


@torch.no_grad()
def predict(model, cfg, df, batch_size=512):
    """Add prediction1-3 and probability1-3 (moves in the mover's frame, like `move`)."""
    preds = {f"{c}{k}": [] for c in ("prediction", "probability") for k in (1, 2, 3)}
    use_amp = cfg.use_amp and cfg.device.startswith("cuda")

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        encoded = [encode(f, m, cfg) for f, m in zip(chunk["board"], chunk["mirrored"])]
        tokens = torch.stack([t for t, _ in encoded]).to(cfg.device)
        legal = torch.stack([m for _, m in encoded]).to(cfg.device)
        self_elo = torch.tensor(chunk["active_elo"].values, dtype=torch.long, device=cfg.device)
        oppo_elo = torch.tensor(chunk["opponent_elo"].values, dtype=torch.long, device=cfg.device)

        with autocast("cuda", enabled=use_amp):
            logits, _value, _ = model(tokens, self_elo, oppo_elo)
        logits = logits.float().masked_fill(~legal, float("-inf"))  # only legal moves
        top_p, top_i = torch.topk(torch.softmax(logits, dim=-1), k=3)

        for probs, idxs in zip(top_p.tolist(), top_i.tolist()):
            for k, (p, i) in enumerate(zip(probs, idxs), start=1):
                legal_pick = p > 0  # fewer than 3 legal moves -> leftover slots are empty
                preds[f"prediction{k}"].append(ALL_MOVES[i] if legal_pick else None)
                preds[f"probability{k}"].append(p if legal_pick else 0.0)

    out = df.reset_index(drop=True).copy()
    for col, values in preds.items():
        out[col] = values
    return out


# ---------------------------------------------------------------- checks

def count_illegal(df):
    """Every prediction must be legal on our stored board. Non-zero = a flip bug."""
    bad = 0
    for row in df.itertuples():
        board = chess.Board(row.board)
        for m in (row.prediction1, row.prediction2, row.prediction3):
            if isinstance(m, str) and chess.Move.from_uci(m) not in board.legal_moves:
                bad += 1
    return bad


def check_against_uci(df, n, model_name, cfg, checkpoint_path=None):
    """Run the official Maia-3 UCI engine on the first n positions and compare top-3."""
    cmd = [sys.executable, "-m", "maia3.uci", "--model", model_name,
           "--device", cfg.device, "--temperature", "0", "--multipv", "3"]
    if checkpoint_path:
        cmd += ["--checkpoint-path", str(checkpoint_path)]
    engine = chess.engine.SimpleEngine.popen_uci(cmd)
    same = 0
    mismatches = []
    try:
        for row in df.head(n).itertuples():
            board = real_board(row.board, row.mirrored)
            engine.configure({"SelfElo": int(row.active_elo), "OppoElo": int(row.opponent_elo)})
            infos = engine.analyse(board, chess.engine.Limit(nodes=1), multipv=3)
            uci = [info["pv"][0].uci() for info in infos if info.get("pv")]
            if int(row.mirrored) == 1:  # engine answers on the real board -> flip back
                uci = [mirror_move(m) for m in uci]
            ours = [m for m in (row.prediction1, row.prediction2, row.prediction3) if isinstance(m, str)]
            if uci == ours:
                same += 1
            else:
                mismatches.append((row.Index, ours, uci))
    finally:
        engine.quit()
    print(f"UCI check: {same}/{n} positions have the same top-3 as the official engine")
    for idx, ours, uci in mismatches[:5]:
        print(f"  row {idx}: ours {ours}  engine {uci}")
    return same, mismatches


# ---------------------------------------------------------------- runs

def run_file(model, cfg, csv_path, out_path, args):
    df = pd.read_csv(csv_path)
    full_rows = len(df)
    if args.limit:
        df = df.head(args.limit)
    print(f"\n{csv_path}: {len(df)} of {full_rows} positions on {cfg.device.upper()}")

    start = time.time()
    out = predict(model, cfg, df, args.batch_size)
    secs = time.time() - start
    rate = len(df) / secs if secs > 0 else float("inf")
    print(f"{rate:.0f} positions/sec ({secs:.1f}s); this whole test set ({full_rows:,} rows) "
          f"would take about {full_rows / rate / 60:.1f} min")

    illegal = count_illegal(out)
    print(f"illegal predictions: {illegal}" + ("  <-- flip bug, stop and check" if illegal else ""))

    if args.check_uci:
        check_against_uci(out, min(args.check_uci, len(out)), args.model, cfg, args.checkpoint_path)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    return out


def summary_row(player, df, label):
    r = evaluate_predictions(df)
    pct = lambda x: round(100 * x, 1)
    row = {"player": player, "model": label, "test_games": r["n_games"],
           "positions": r["n_positions"],
           "top1": pct(r["top1"]["acc"]),
           "top1_ci_low": pct(r["top1"]["ci"][0]), "top1_ci_high": pct(r["top1"]["ci"][1]),
           "top3": pct(r["top3"]["acc"])}
    for phase, s in r["phases"].items():
        row[f"top1_{phase}"] = pct(s["acc"]) if s["n"] else None
    return row


def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def run_all(model, cfg, args):
    out_dir = Path(args.out_dir)
    rows = []
    seen = {}  # test.csv content -> first folder with it (skips duplicate copies)
    test_files = []
    for test_csv in sorted(Path(args.all).glob("*/test.csv")):
        digest = file_hash(test_csv)
        if digest in seen:
            print(f"{test_csv.parent.name}: same test.csv as {seen[digest]}, skipping the duplicate")
            continue
        seen[digest] = test_csv.parent.name
        test_files.append(test_csv)
    total = sum(sum(1 for _ in open(f)) - 1 for f in test_files)
    print(f"{len(test_files)} players, {total:,} positions in total")
    for test_csv in test_files:
        player = test_csv.parent.name
        out_csv = out_dir / f"{player}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"\n{player}: already done ({out_csv}), loading it (use --overwrite to redo)")
            out = pd.read_csv(out_csv)
        else:
            out = run_file(model, cfg, test_csv, out_csv, args)
        rows.append(summary_row(player, out, "Maia-3 5M"))

        maia2_csv = Path(args.maia2_dir) / f"{player}.csv" if args.maia2_dir else None
        if maia2_csv and maia2_csv.exists():
            rows.append(summary_row(player, pd.read_csv(maia2_csv), "Maia-2"))
        elif maia2_csv:
            print(f"  no Maia-2 baseline at {maia2_csv}")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    wide = table.pivot(index="player", columns="model", values=["top1", "top3"])
    print("\nTop-1 / top-3 accuracy (%)\n")
    print(wide.to_string())
    print(f"\nsaved {out_dir / 'summary.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="one player's test.csv")
    p.add_argument("-o", "--out", default="results_maia3.csv", help="output CSV for one file")
    p.add_argument("--all", metavar="SPLITS_DIR", help="run every <SPLITS_DIR>/<player>/test.csv")
    p.add_argument("--out-dir", help="with --all: where to save <player>.csv and summary.csv")
    p.add_argument("--maia2-dir", help="with --all: Maia-2 baseline folder, for the comparison table")
    p.add_argument("--model", default="maia3-5m")
    p.add_argument("--checkpoint-path", help="local .pt file instead of downloading")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--limit", type=int, help="only the first N positions (for timing)")
    p.add_argument("--check-uci", type=int, default=0, metavar="N",
                   help="compare the first N positions with the official UCI engine")
    p.add_argument("--overwrite", action="store_true", help="with --all: redo finished players")
    args = p.parse_args()

    if bool(args.input) == bool(args.all):
        p.error("give either one test.csv or --all SPLITS_DIR")
    if args.all and not args.out_dir:
        p.error("--all needs --out-dir")

    model, cfg = load_maia3(args.model, args.device, args.checkpoint_path)

    if args.all:
        run_all(model, cfg, args)
    else:
        out = run_file(model, cfg, args.input, args.out, args)
        print()
        print(format_report(evaluate_predictions(out)))


if __name__ == "__main__":
    main()
