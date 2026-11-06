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
import copy
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

# Maia-individual recipe (Section 4.1): train all layers using SGD with momentum and lr=1e-4.
CONFIG = {
    "optimizer": "sgd",       # Maia-individual: SGD with momentum
    "lr": 1e-4,
    "momentum": 0.9,
    "weight_decay":1e-5,
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
    data = list(df[TRAIN_COLUMNS].itertuples(index=False, name=None))
    dataset = MAIA2Dataset(data, ALL_MOVES_DICT, cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_loss(model, batch):
    """One batch -> (total_loss, {"move": ..., "side_info": ..., "value": ...}).

    Same three losses and weights Maia-2 used in pretraining
    (see train.py and train_chunks in main.py).
    total_loss must be a tensor you can call .backward() on.
    """
    device = next(model.parameters()).device
    boards, moves, elo_self, elo_oppo, side_info, active_win = [
        t.to(device) for t in batch
    ]

    logits_maia, logits_side_info, logits_value = model(
        boards, elo_self.long(), elo_oppo.long()
    )

    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    criterion_value = nn.MSELoss()

    loss_move = criterion_maia(logits_maia, moves.long())

    side_coeff = getattr(model.cfg, "side_info_coefficient", 1.0)
    loss_side_info = criterion_side_info(logits_side_info, side_info.float()) * side_coeff

    val_coeff = getattr(model.cfg, "value_coefficient", 1.0)
    loss_value = criterion_value(logits_value.squeeze(-1), active_win.float()) * val_coeff

    total_loss = loss_move + loss_side_info + loss_value
    parts = {
        "move": loss_move.item(),
        "side_info": loss_side_info.item(),
        "value": loss_value.item(),
    }
    return total_loss, parts


def set_trainable(model, mode):
    """Choose which parameters train. Return the number of trainable parameters.

    "all"  - every layer (Maia-individual)
    Others are optional for now. Part names: chess_cnn, to_patch_embedding,
    pos_embedding, transformer, elo_embedding, last_ln, fc_1, fc_2, fc_3, fc_3_1.
    """
    if mode == "all":
        # Unfreeze all parameters (Maia-individual recipe)
        for p in model.parameters():
            p.requires_grad = True
    else:
        # Convert single string (e.g. "chess_cnn") to a list for clean matching
        parts = [mode] if isinstance(mode, str) else list(mode)

        for name, p in model.named_parameters():
            # Unfreeze if parameter name starts with any of the requested part names
            p.requires_grad = any(name.startswith(part) for part in parts)

    # Return total count of active trainable weights
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer):
    """One pass over loader. Return the average total loss."""
    model.train()  # Put the model into training mode
    total_loss_sum = 0.0
    n_batches = 0

    # Determine whether the model is running on CPU or GPU
    device = next(model.parameters()).device

    for batch in loader:
        # Move batch tensors to match the model's device (CPU/GPU)
        batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]

        # 1. Clear old gradients from the previous step
        optimizer.zero_grad()

        # 2. Compute the current batch loss
        total_loss, _ = compute_loss(model, batch)

        # 3. Calculate gradients via backpropagation
        total_loss.backward()

        # 4. Update the model parameters
        optimizer.step()

        # 5. Track total loss
        total_loss_sum += total_loss.item()
        n_batches += 1

    # Return the average loss across all batches in this epoch
    return total_loss_sum / n_batches if n_batches > 0 else 0.0

def evaluate(model, df):
    """Score a model on a split with F3 + evaluation_metrics. Returns the results dict."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results = Path(temp_dir) / "f5_preds.csv"
        preds = maia2_predict(
            df.reset_index(drop=True), maia2_model=model, results=results
        )
    model.train()  # maia2_predict switches the model to eval mode
    return evaluate_predictions(preds)


def finetune(model, train_df, val_df, config):
    """Fine-tune model on train_df, keep the version with the best val top-1.

    Return (model, history). history is a list with one dict per epoch,
    at least {"epoch", "train_loss", "val_top1"}.
    """
    set_trainable(model, config.get("trainable", "all"))
    loader = make_loader(train_df, model.cfg, config["batch_size"], shuffle=True)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer_name = config.get("optimizer", "sgd").lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=config["lr"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
        )
    elif optimizer_name in ("adam", "adamw"):
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    history = []
    best_val_top1 = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    patience = config.get("patience", 3)

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = train_one_epoch(model, loader, optimizer)
        val_top1 = evaluate(model, val_df)["top1"]["acc"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_top1": round(val_top1, 4),
        })

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


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
    with tempfile.TemporaryDirectory() as temp_dir:
        results = Path(temp_dir) / "f5_test.csv"
        f3 = maia2_predict(
            df.head(64), maia2_model=model, results=results
        )["prediction1"].tolist()
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