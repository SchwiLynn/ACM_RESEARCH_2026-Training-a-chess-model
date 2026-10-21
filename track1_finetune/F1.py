from maia2 import model, inference

maia2_model = model.from_pretrained(type="rapid", device="auto")
prepared = inference.prepare()
start_fen = "r1b1k1nr/p4ppp/2p5/1p2n3/1b2N3/1B6/PPP2PPP/RNB1K2R w KQkq - 0 1"
elo_self = 1500
elo_oppo = 1500
move_probs, white_expected_score = inference.inference_each(
    maia2_model, prepared, start_fen, elo_self, elo_oppo
)
sorted_moves = sorted(move_probs.items(), key=lambda item: item[1], reverse=True)
print(f"Predicted move probabilities for White (rating {elo_self}):")

for i, (move, prob) in enumerate(sorted_moves[:3]):
    print(f"  {i+1}. Move: {move}, Probability: {prob:.2%}")

print(f"\nWhite expected score: {white_expected_score:.3f}")