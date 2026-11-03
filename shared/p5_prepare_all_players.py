"""P5: download, build and split every player in a list.

For each username:
  1. download all rated Rapid games  -> data/raw/<user>_rapid.pgn
  2. build the move table             -> data/moves/<user>_moves.csv
  3. split by game (P7)               -> data/splits/<user>/
Players already downloaded are skipped (use --redownload to fetch again).
Prints a summary table at the end and saves it to data/p5_summary.csv.

Usage:
    python3 shared/p5_prepare_all_players.py configs/eval_players.txt
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SHARED = Path(__file__).resolve().parent
RAW, MOVES, SPLITS = Path("data/raw"), Path("data/moves"), Path("data/splits")


def run(cmd, cwd=None):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([sys.executable, *cmd], cwd=cwd, check=True)


def download(user, redownload):
    pgn = RAW / f"{user}_rapid.pgn"
    if pgn.exists() and not redownload:
        print(f"  already downloaded: {pgn}")
        return pgn
    for attempt in (1, 2):
        run([SHARED / "download_lichess_games.py", user], cwd=RAW)
        if pgn.exists() and pgn.stat().st_size > 0:
            return pgn
        if attempt == 1:
            print("  download failed, waiting 60s and retrying once (Lichess rate limit)")
            time.sleep(60)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("players", help="text file, one Lichess username per line")
    p.add_argument("--redownload", action="store_true")
    args = p.parse_args()

    users = [l.strip() for l in open(args.players)
             if l.strip() and not l.strip().startswith("#")]
    for d in (RAW, MOVES, SPLITS):
        d.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, user in enumerate(users, 1):
        print(f"\n[{i}/{len(users)}] {user}")
        pgn = download(user, args.redownload)
        if pgn is None:
            print(f"  SKIPPED {user}: download failed")
            summary.append({"player": user, "status": "download failed"})
            continue

        moves = MOVES / f"{user}_moves.csv"
        run([SHARED / "build_dataset.py", pgn, "--target", user, "--only-target",
             "--min-ply", "11", "--min-clock", "30", "-o", moves])
        run([SHARED / "p7_split_dataset.py", moves, "--out-dir", SPLITS / user])

        df = pd.read_csv(moves)
        train = pd.read_csv(SPLITS / user / "train_all.csv")
        summary.append({
            "player": user, "status": "ok",
            "games": df["game_id"].nunique(), "rows": len(df),
            "train_games": train["game_id"].nunique(),
            "first_date": df["date"].min(), "last_date": df["date"].max(),
        })

    table = pd.DataFrame(summary)
    table.to_csv("data/p5_summary.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()