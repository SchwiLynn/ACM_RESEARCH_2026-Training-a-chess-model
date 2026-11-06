"""S9: fine-tune pretrained Maia-3 5M on one player's frozen training split.

Run from the project root after installing CSSLab/maia3:
    python track2_finetunemaia3/s9_finetune.py --test data
    python track2_finetunemaia3/s9_finetune.py --test loss
    python track2_finetunemaia3/s9_finetune.py --test overfit
    python track2_finetunemaia3/s9_finetune.py --test run --device cuda

Uses S8's PGN reconstruction and Maia-3's own tokenization/move vocabulary.
Training choices: SGD, all parameters enabled, policy cross-entropy plus
0.1 * outcome cross-entropy. Clock prediction is omitted because the CSV
does not provide its training target. These are experiment defaults.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import resolve_checkpoint_path
from maia3.uci import load_model, parse_args as maia3_args
from maia3.utils import get_all_possible_moves

from s8_maia3_baseline import load_games, position_for_row

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared"))
from evaluation_metrics import evaluate_predictions, format_report

ALL_MOVES = get_all_possible_moves()
MOVE_INDEX = {move: index for index, move in enumerate(ALL_MOVES)}
CONFIG = {
    "lr": 1e-4,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "batch_size": 64,
    "max_epochs": 20,
    "patience": 3,
    "value_weight": 0.1,
    "seed": 0,
}


def load_base_model(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or select a GPU runtime.")
    cfg = maia3_args(["--model", "maia3-5m", "--device", device])
    cfg.checkpoint_path = resolve_checkpoint_path(cfg.model_spec)
    return load_model(cfg)


class PlayerDataset(Dataset):
    def __init__(self, df, cfg, games):
        self.rows = list(df.itertuples(index=False))
        self.cfg = cfg
        self.games = games

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        board = position_for_row(row, self.games)
        legal = get_legal_moves_mask(board, MOVE_INDEX)
        target = MOVE_INDEX[row.move]
        if not legal[target]:
            raise ValueError(f"Illegal normalized label: {row.game_id}, ply {row.ply}")
        # Include the current board and preceding boards, never the target move.
        history = [tokenize_board(board)]
        while board.move_stack and len(history) < self.cfg.history:
            board.pop()
            history.append(tokenize_board(board))
        tokens = get_historical_tokens(
            list(reversed(history)), self.cfg,
            base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
        # Maia-3 outcome classes are loss, draw, win.
        outcome = int(row.active_win) + 1
        return tokens, target, int(row.active_elo), int(row.opponent_elo), legal, outcome


def make_loader(df, cfg, batch_size, shuffle, games):
    return DataLoader(PlayerDataset(df, cfg, games), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def compute_loss(model, batch, value_weight=0.1):
    device = next(model.parameters()).device
    tokens, moves, self_elo, oppo_elo, legal, outcomes = [x.to(device) for x in batch]
    logits, value_logits, _ = model(tokens, self_elo, oppo_elo)
    policy_loss = F.cross_entropy(logits.masked_fill(~legal, float("-inf")), moves)
    value_loss = F.cross_entropy(value_logits, outcomes)
    total = policy_loss + value_weight * value_loss
    return total, {"move": policy_loss.detach(), "value": value_loss.detach()}


def set_trainable(model, mode="all"):
    if mode != "all":
        raise ValueError("S9 supports trainable='all'")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, value_weight=0.1):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_loss(model, batch, value_weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        size = len(batch[1])
        total += loss.item() * size
        count += size
    if not count:
        raise ValueError("Training split is empty")
    return total / count


@torch.no_grad()
def predict(model, df, games, batch_size=64):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    rows = []
    try:
        for batch in make_loader(df, model.cfg, batch_size, False, games):
            tokens, _, es, eo, legal, _ = [x.to(device) for x in batch]
            logits, _, _ = model(tokens, es, eo)
            probabilities = logits.masked_fill(~legal, float("-inf")).softmax(dim=1)
            if not torch.isfinite(probabilities).all():
                raise ValueError("Non-finite prediction probabilities")
            values, indices = probabilities.topk(3, dim=1)
            for probs, moves in zip(values.cpu().tolist(), indices.cpu().tolist()):
                result = {}
                for rank, (prob, move) in enumerate(zip(probs, moves), start=1):
                    result[f"prediction{rank}"] = ALL_MOVES[move] if prob > 0 else None
                    result[f"probability{rank}"] = prob
                rows.append(result)
    finally:
        model.train(was_training)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def evaluate(model, df, games, batch_size=64):
    return evaluate_predictions(predict(model, df, games, batch_size))


def finetune(model, train_df, val_df, config, games):
    set_trainable(model)
    loader = make_loader(train_df, model.cfg, config["batch_size"], True, games)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    best_score = -1.0
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, config["max_epochs"] + 1):
        loss = train_one_epoch(model, loader, optimizer, config["value_weight"])
        predictions = predict(model, val_df, games, config["batch_size"])
        score = float(predictions.prediction1.eq(predictions.move).mean())
        record = {"epoch": epoch, "train_loss": loss, "val_top1": score}
        history.append(record)
        print(record, flush=True)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = {"model_state_dict": best_state,
                          "training_config": config, "history": history.copy(),
                          "epoch": epoch, "val_top1": score}
            # Strip CLI-only ModelSpec so weights_only=True can reload the file.
            checkpoint["model_config"] = {
                k: v for k, v in vars(model.cfg).items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            path = Path(config["checkpoint"])
            temporary = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history


def read_split(path):
    df = pd.read_csv(path, dtype={"game_id": str})
    required = {"board", "move", "active_elo", "opponent_elo", "active_win",
                "mirrored", "game_id", "ply"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError(f"{path}: empty or incomplete split")
    if not df.active_win.isin([-1, 0, 1]).all():
        raise ValueError(f"{path}: active_win must be -1, 0, or 1")
    if df.duplicated(["game_id", "ply"]).any():
        raise ValueError(f"{path}: duplicate positions")
    return df


def main():
    drive = Path("/content/drive/MyDrive/acm_chess")
    storage = drive if drive.exists() else ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", required=True, choices=["data", "loss", "overfit", "run"])
    parser.add_argument("--player", default="thakt")
    parser.add_argument("--data-root", type=Path, default=storage / "data")
    parser.add_argument("--train-file", default="train_20_seed0.csv",
                        help="filename inside the player's split folder, e.g. train_all.csv")
    parser.add_argument("--out-dir", type=Path, default=storage / "finetuned_maia3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=CONFIG["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = args.data_root / "splits" / args.player
    train_df = read_split(splits / args.train_file)
    frames = [train_df]
    if args.test == "run":
        val_df = read_split(splits / "val.csv")
        test_df = read_split(splits / "test.csv")
        frames += [val_df, test_df]
        for left in range(len(frames)):
            for right in range(left + 1, len(frames)):
                if set(frames[left].game_id) & set(frames[right].game_id):
                    raise ValueError("Train, validation and test must have disjoint game IDs")
    else:
        train_df = train_df.head(64).copy()
        frames = [train_df]
    wanted = set(pd.concat(frames).game_id)
    games = load_games(args.data_root / "raw" / f"{args.player}_rapid.pgn", wanted)
    model = load_base_model(args.device)
    batch_size = min(args.batch_size, len(train_df))
    loader = make_loader(train_df, model.cfg, batch_size, False, games)
    batch = next(iter(loader))
    if args.test == "data":
        tokens, moves, es, eo, legal, outcomes = batch
        assert tokens.shape == (len(moves), 64, 12 * model.cfg.history + 1)
        assert legal.shape == (len(moves), len(ALL_MOVES))
        assert legal.gather(1, moves[:, None]).all()
        assert outcomes.min() >= 0 and outcomes.max() <= 2
        print("data: PASS", tokens.shape, "vocabulary:", len(ALL_MOVES))
        return
    set_trainable(model)
    if args.test == "loss":
        model.train()
        loss, parts = compute_loss(model, batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        print("loss: PASS", loss.item(), {k: v.item() for k, v in parts.items()})
        return
    if args.test == "overfit":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.eval()
        initial = float(compute_loss(model, batch)[0].detach())
        for epoch in range(1, 201):
            train_one_epoch(model, loader, optimizer)
            if epoch % 10 == 0:
                model.eval()
                loss = float(compute_loss(model, batch)[0].detach())
                predictions = predict(model, train_df, games, batch_size)
                accuracy = float(predictions.prediction1.eq(predictions.move).mean())
                print(f"epoch {epoch}: loss {loss:.4f}, top-1 {accuracy:.1%}", flush=True)
                if accuracy >= 0.9 and loss <= initial * 0.5:
                    print("overfit: PASS")
                    return
        raise AssertionError("Could not reach 90% accuracy and halve loss on the training sample")
    destination = args.out_dir / args.player / Path(args.train_file).stem
    destination.mkdir(parents=True, exist_ok=True)
    config = {**CONFIG, "max_epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "seed": args.seed,
              "checkpoint": str(destination / "best.pt")}
    print("Base Maia-3 on the frozen test set:", flush=True)
    before = predict(model, test_df, games, args.batch_size)
    baseline = evaluate_predictions(before)
    print(format_report(baseline))
    before.to_csv(destination / "before.csv", index=False)
    model, history = finetune(model, train_df, val_df, config, games)
    after = predict(model, test_df, games, args.batch_size)
    report = evaluate_predictions(after)
    after.to_csv(destination / "after.csv", index=False)
    print("\nFine-tuned Maia-3 on the same frozen test set:")
    print(format_report(report))
    print(f"Top-1 change: {(report['top1']['acc'] - baseline['top1']['acc']) * 100:+.2f} percentage points")
    results = {"player": args.player, "train_file": args.train_file,
               "config": config, "history": history, "before": baseline, "after": report}
    (destination / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
