import argparse
import os

import torch

from .data import build_vocab, load_text
from .model import MyCustomToyGPT, device

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "..", "data", "input.txt"
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()

    text = load_text(DATA_PATH)
    chars, stoi, itos, encode, decode = build_vocab(text)
    vocab_size = len(chars)

    model = MyCustomToyGPT(vocab_size).to(device)

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {CKPT_PATH}")
    else:
        print("No checkpoint found. Generating from randomly initialized weights.")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out_idx = model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()
    print(decode(out_idx))


if __name__ == "__main__":
    main()
