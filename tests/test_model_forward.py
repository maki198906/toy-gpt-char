import torch

from toygpt.model import MyCustomToyGPT, block_size, device


def test_forward_pass_produces_logits_and_loss():
    vocab_size = 16
    batch_size = 2

    model = MyCustomToyGPT(vocab_size).to(device)
    model.eval()

    idx = torch.randint(0, vocab_size, (batch_size, block_size), device=device)

    logits, _ = model(idx)
    assert logits.shape == (batch_size, block_size, vocab_size)

    _, loss = model(idx, idx)
    assert loss is not None
