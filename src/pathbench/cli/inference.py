import argparse
import json

# Inference API should be small and stable

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--input", required=True, help="slide features path or tiles")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    # load model (Lightning checkpoint), run predict, save JSON output
    res = {"status": "ok", "probs": [0.1, 0.9]}
    with open(args.output, "w") as f:
        json.dump(res, f)
    print(args.output)