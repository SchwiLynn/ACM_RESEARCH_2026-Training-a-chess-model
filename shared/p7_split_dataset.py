"""P7: split one player's move CSV into test / val / train, plus training subsets.

  test  = newest 20% of games
  val   = newest 10% of the remaining games
  train = the rest
Training subsets: shuffle the train games with a seed and take the first N.
Because every size takes the first N of the same shuffle, the subsets are nested
(20 games are inside 50, 50 inside 200, ...).

Usage:
    python3 shared/p7_split_dataset.py data/thakt_moves.csv
Writes to data/splits/thakt_moves/:
    test.csv  val.csv  train_all.csv  train_20_seed0.csv  train_50_seed0.csv  ...
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def order_games_newest_first(df):
    games = (df.reset_index()
               .groupby("game_id", sort=False)
               .agg(date=("date", "first"), first_row=("index", "min")))
    games = games.sort_values(["date", "first_row"], ascending=[False, True])
    return list(games.index)


def split(df, test_frac, val_frac):
    games = order_games_newest_first(df)
    n_test = round(len(games) * test_frac)
    test = games[:n_test]
    rest = games[n_test:]
    n_val = round(len(rest) * val_frac)
    val = rest[:n_val]
    train = rest[n_val:]
    return train, val, test


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="one player's move CSV from build_dataset.py")
    p.add_argument("--out-dir", default=None, help="default: data/splits/<csv name>")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 200, 1000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.out_dir or Path("data/splits") / Path(args.csv).stem)
    out.mkdir(parents=True, exist_ok=True)

    train, val, test = split(df, args.test_frac, args.val_frac)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))

    def save(name, game_ids):
        rows = df[df["game_id"].isin(game_ids)]
        rows.to_csv(out / f"{name}.csv", index=False)
        print(f"  {name:<20} {len(game_ids):>5} games {len(rows):>7} rows")

    print(f"{args.csv}: {df['game_id'].nunique()} games, {len(df)} rows -> {out}")
    save("test", test)
    save("val", val)
    save("train_all", train)

    for seed in args.seeds:
        shuffled = list(np.random.default_rng(seed).permutation(train))
        for n in args.sizes:
            if n > len(train):
                continue
            save(f"train_{n}_seed{seed}", shuffled[:n])

    skipped = [n for n in args.sizes if n > len(train)]
    if skipped:
        print(f"  skipped sizes {skipped}: only {len(train)} training games")

    tdates = df[df["game_id"].isin(test)]["date"]
    rdates = df[df["game_id"].isin(train + val)]["date"]
    print(f"  test dates  {tdates.min()} .. {tdates.max()}")
    print(f"  train+val   {rdates.min()} .. {rdates.max()}")


if __name__ == "__main__":
    main()