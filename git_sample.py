import io

import chess.pgn
import requests
import zstandard as zstd

URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst"
)

MAX_GAMES = 1000
games_saved = 0
positions_saved = 0

with requests.get(URL, stream=True, timeout=60) as response:
    response.raise_for_status()
    decompressor = zstd.ZstdDecompressor()

    with decompressor.stream_reader(response.raw) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as pgn:
            with open("sample.pgn", "w+", encoding="utf-8") as output:
                while games_saved < MAX_GAMES:
                    game = chess.pgn.read_game(pgn)

                    if game is None:
                        break

                    if game.errors:
                        continue

                    event = game.headers.get("Event", "")
                    white_elo = game.headers.get("WhiteElo", "")
                    black_elo = game.headers.get("BlackElo", "")
                    if "Rated Rapid game" not in event:
                        continue
                    if not white_elo.isdigit() or not black_elo.isdigit():
                        continue
                    if not (
                        1200 <= int(white_elo) <= 1800
                        and 1200 <= int(black_elo) <= 1800
                    ):
                        continue
                    positions = sum(1 for i in game.mainline_moves())

                    exporter = chess.pgn.StringExporter(
                        variations=False,
                        comments=False,
                    )
                    output.write(game.accept(exporter) + "\n\n")

                    games_saved += 1
                    positions_saved += positions

                    if games_saved % 100 == 0:
                        print(f"Saved {games_saved} games", flush=True)

print(f"Games saved: {games_saved}")
print(f"Positions saved: {positions_saved}")
print("Output: sample.pgn")

