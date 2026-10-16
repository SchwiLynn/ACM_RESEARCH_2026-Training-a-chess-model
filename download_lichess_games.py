import argparse
import requests
import chess.pgn
from datetime import datetime

HEADERS = {"User-Agent": "acm-research-p2 (student project)"}


def download_games(username, max_games=None):
    url = f"https://lichess.org/api/games/user/{username}"
    params = {"perfType": "rapid", "rated": "true"}
    if max_games is not None:
        params["max"] = max_games

    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=(10, 300))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return

    filename = f"{username}_rapid.pgn"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    dates = []
    count = 0
    with open(filename, encoding="utf-8") as f:
        while True:
            h = chess.pgn.read_headers(f)
            if h is None:
                break
            count += 1
            try:
                dates.append(datetime.strptime(h.get("Date", ""), "%Y.%m.%d"))
            except ValueError:
                pass

    if count == 0:
        print("No rated Rapid games found for", username)
        return

    print("saved:", filename)
    print("games:", count)
    print("oldest:", min(dates).date())
    print("newest:", max(dates).date())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Lichess player's rated Rapid games as PGN.")
    parser.add_argument("username", help="Lichess username")
    parser.add_argument("--max", type=int, default=None, help="optional limit on number of games")
    args = parser.parse_args()
    download_games(args.username, args.max)
