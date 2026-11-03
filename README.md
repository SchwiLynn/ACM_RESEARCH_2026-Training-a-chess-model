# ACM Research 2026 - Training a Chess Model

Fine-tune Maia-2 on one player's games to predict that player's moves.

## Setup

Python 3.10 - 3.12.

```bash
pip install -r requirements.txt
```

## Run

From the repo root:

```bash
# 1. Download games
cd data
python3 ../shared/download_lichess_games.py thakt --max 300
cd ..

# 2. Build move table
python3 shared/build_dataset.py data/thakt_rapid.pgn --target thakt --only-target --min-ply 11 --min-clock 30 -o data/thakt_moves.csv

# 3. Evaluate Maia-2
python3 track1_finetune/F3.py data/thakt_moves.csv -o data/thakt_baseline.csv
```

Fine-tuning runs on Colab: `notebooks/colab_setup.ipynb`.

## Folders

- `shared/` - download, build dataset, evaluation
- `track1_finetune/` - Track 1
- `track2_finetunemaia3/` - Track 2
- `notebooks/` - Colab notebooks
- `data/` - local data, not in git

## References

- Maia: https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf
- Maia-individual: https://www.cs.toronto.edu/~ashton/pubs/maia-individual-kdd2022.pdf
- Maia-2: https://www.cs.toronto.edu/~ashton/pubs/maia2-neurips2024.pdf
- Maia-2 code: https://github.com/CSSLab/maia2