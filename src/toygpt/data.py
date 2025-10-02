import os

import torch

from .model import block_size, device


def load_text(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find {path}. Run `python -m toygpt.ensure_data` or `python scripts/download_data.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ix):
        return "".join(itos[i] for i in ix)

    return chars, stoi, itos, encode, decode


def make_splits(encoded, train_ratio=0.9):
    data = torch.tensor(encoded, dtype=torch.long)
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def get_batch(split, train_data, val_data, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
