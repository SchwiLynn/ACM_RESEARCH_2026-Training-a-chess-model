"""F5: fine-tune Maia-2 on one player's games.

Fill in every function marked TODO. How you do it is up to you.
Each part has a test at the bottom. Run them from the repo root:

    python3 track1_finetune/f5_finetune.py --test data      # make_loader
    python3 track1_finetune/f5_finetune.py --test loss      # compute_loss
    python3 track1_finetune/f5_finetune.py --test overfit   # train_one_epoch (needs the two above)
    python3 track1_finetune/f5_finetune.py --test run       # finetune (needs everything)

data, loss and overfit run on a laptop CPU. run needs a GPU (Colab).

Useful places in the maia2 source:
    maia2/main.py   MAIA2Dataset, MAIA2Model.forward, train_chunks (the training step)
    maia2/train.py  the three loss functions, the optimizer
Our CSVs are already in Maia-2's training format (see shared/build_dataset.py).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from maia2 import model as maia2_model_lib
from maia2.main import MAIA2Dataset
from maia2.utils import get_all_possible_moves

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "track1_finetune"))
from evaluation_metrics import evaluate_predictions, format_report  # noqa: E402
from F3 import maia2_predict  # noqa: E402

ALL_MOVES = get_all_possible_moves()
ALL_MOVES_DICT = {m: i for i, m in enumerate(ALL_MOVES)}

# Columns Maia-2's training dataset expects, in this order.
TRAIN_COLUMNS = ["board", "move", "active_elo_bucket", "opponent_elo_bucket", "active_win"]

# TODO: decide these (Maia-individual recipe, see the F2 table in the log and Agrima's config).
CONFIG = {
    "optimizer": "sgd",       # Maia-individual: SGD with momentum
    "lr": 1e-4,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 20,
    "patience": 3,            # stop after this many epochs without val improvement
    "trainable": "all",       # Maia-individual trains every layer
}


def load_base_model(device="auto"):
    """Base Maia-2 rapid model. Settings are on model.cfg."""
    return maia2_model_lib.from_pretrained(type="rapid", device=device)


def make_loader(df, cfg, batch_size, shuffle):
    """CSV rows -> torch DataLoader built on Maia-2's MAIA2Dataset.

    df:  a split CSV loaded with pandas (e.g. train_20_seed0.csv)
    cfg: model.cfg
    Each batch must be:
        boards (B,18,8,8) float, moves (B,) long, elo_self (B,), elo_oppo (B,),
        side_info (B,2021) float, active_win (B,)
    """
    raise NotImplementedError  # TODO


def compute_loss(model, batch):
    """One batch -> (total_loss, {"move": ..., "side_info": ..., "value": ...}).

    Same three losses and weights Maia-2 used in pretraining
    (see train.py and train_chunks in main.py).
    total_loss must be a tensor you can call .backward() on.
    """
    raise NotImplementedError  # TODO


def set_trainable(model, mode):
    """Choose which parameters train. Return the number of trainable parameters.

    "all"  - every layer (Maia-individual)
    Others are optional for now. Part names: chess_cnn, to_patch_embedding,
    pos_embedding, transformer, elo_embedding, last_ln, fc_1, fc_2, fc_3, fc_3_1.
    """
    raise NotImplementedError  # TODO


def train_one_epoch(model, loader, optimizer):
    """One pass over loader. Return the average total loss."""
    raise NotImplementedError  # TODO


def evaluate(model, df):
    """Score a model on a split with F3 + evaluation_metrics. Returns the results dict."""
    preds = maia2_predict(df.reset_index(drop=True), maia2_model=model, results="/tmp/f5_preds.csv")
    model.train()  # maia2_predict switches the model to eval mode
    return evaluate_predictions(preds)


def finetune(model, train_df, val_df, config):
    """Fine-tune model on train_df, keep the version with the best val top-1.

    Return (model, history). history is a list with one dict per epoch,
    at least {"epoch", "train_loss", "val_top1"}.
    """
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------- tests

SPLITS = ROOT / "data" / "splits" / "thakt"


def test_data():
    model = load_base_model("cpu")
    df = pd.read_csv(SPLITS / "train_20_seed0.csv")
    boards, moves, es, eo, side, win = next(iter(make_loader(df, model.cfg, 64, False)))
    assert boards.shape == (64, 18, 8, 8), boards.shape
    assert moves.shape == (64,) and moves.max() < len(ALL_MOVES)
    assert es.min() >= 0 and es.max() <= 10
    assert side.shape == (64, 2021), side.shape
    assert set(win.tolist()) <= {-1, 0, 1}

    # The base model's best legal move on these batches must match F3's prediction1.
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(boards, es, eo)
    legal = side[:, -len(ALL_MOVES):].bool()
    ours = [ALL_MOVES[i] for i in logits.masked_fill(~legal, float("-inf")).argmax(1).tolist()]
    f3 = maia2_predict(df.head(64), maia2_model=model, results="/tmp/f5_test.csv")["prediction1"].tolist()
    same = sum(a == b for a, b in zip(ours, f3))
    print(f"data: shapes OK, {same}/64 predictions match F3")
    assert same == 64


def test_loss():
    model = load_base_model("cpu")
    df = pd.read_csv(SPLITS / "train_20_seed0.csv")
    batch = next(iter(make_loader(df, model.cfg, 64, True)))
    model.train()
    total, parts = compute_loss(model, batch)
    total.backward()
    assert torch.isfinite(total)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print("loss:", round(total.item(), 4), {k: round(float(v), 4) for k, v in parts.items()})


def test_overfit():
    model = load_base_model("cpu")
    df = pd.read_csv(SPLITS / "train_20_seed0.csv").head(64)
    loader = make_loader(df, model.cfg, 64, True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # high lr on purpose: we want to memorize
    model.train()
    for epoch in range(1, 201):
        loss = train_one_epoch(model, loader, opt)
        if epoch % 10 == 0:
            top1 = evaluate(model, df)["top1"]["acc"]
            print(f"epoch {epoch:3d}  loss {loss:.4f}  top-1 on the same 64 {top1:.1%}")
            if top1 >= 0.9:
                print("overfit: PASS")
                return
    raise AssertionError("could not memorize 64 positions: the training step is broken")


def test_run():
    model = load_base_model("auto")
    train_df = pd.read_csv(SPLITS / "train_20_seed0.csv")
    val_df = pd.read_csv(SPLITS / "val.csv")
    test_df = pd.read_csv(SPLITS / "test.csv")
    set_trainable(model, CONFIG["trainable"])
    model, history = finetune(model, train_df, val_df, CONFIG)
    for h in history:
        print(h)
    print("\nthakt, 20 games, seed 0  (base Maia-2 was 49.1%)")
    print(format_report(evaluate(model, test_df)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["data", "loss", "overfit", "run"], required=True)
    {"data": test_data, "loss": test_loss, "overfit": test_overfit, "run": test_run}[p.parse_args().test]()