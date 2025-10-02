import argparse
import os

import torch
from torch import optim

from .data import build_vocab, get_batch, load_text, make_splits
from .metrics import estimate_loss
from .model import (
    MyCustomToyGPT,
    batch_size,
    block_size,
    device,
    dropout,
    eval_interval,
    eval_iters,
    learning_rate,
    max_iters,
    n_embd,
    n_head,
    n_layer,
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "..", "data", "input.txt"
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate a text sample after training completes.",
    )
    args = parser.parse_args()

    torch.manual_seed(1337)

    text = load_text(DATA_PATH)
    chars, stoi, itos, encode, decode = build_vocab(text)
    vocab_size = len(chars)

    data = [stoi[c] for c in text]
    train_data, val_data = make_splits(data, train_ratio=0.9)

    def batcher(split):
        return get_batch(split, train_data, val_data, batch_size)

    model = MyCustomToyGPT(vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, batcher, eval_iters, device)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = batcher("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "meta": {
                "vocab": chars,
                "block_size": block_size,
                "n_embd": n_embd,
                "n_layer": n_layer,
                "n_head": n_head,
                "dropout": dropout,
            },
        },
        CKPT_PATH,
    )
    print(f"Saved checkpoint to {CKPT_PATH}")

    if args.sample:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample = model.generate(context, max_new_tokens=500)[0].tolist()
        print("".join([chars[i] for i in sample]))

if __name__ == "__main__":
    main()
