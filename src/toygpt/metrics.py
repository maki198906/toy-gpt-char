from typing import Callable, Dict, Tuple

import torch

BatchFn = Callable[[str], Tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def estimate_loss(
    model, get_batch_fn: BatchFn, eval_iters: int, device
) -> Dict[str, float]:
    """Average cross-entropy over `eval_iters` mini-batches for both splits."""
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    if was_training:
        model.train()
    return out
