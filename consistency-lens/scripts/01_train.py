"""Minimal single-process training loop.

The *real* entry-point will wire up DeepSpeed & read ``lens.yaml``.  For the MVP
we just train Decoder/Encoder for a handful of steps to prove gradients flow.
"""

from __future__ import annotations

import argparse

import torch
from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from torch.utils.data import DataLoader


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train Consistency-Lens MVP")
    parser.add_argument("--activation_dir", type=str, default="consistency-lens/data/activations")
    parser.add_argument("--model_name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ActivationDataset(args.activation_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No .pt files found in {args.activation_dir}.  Run scripts/00_dump_activations.py first.")

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

    dec = Decoder(DecoderConfig(model_name=args.model_name)).to(device)
    enc = Encoder(args.model_name).to(device)
    orig = OrigWrapper(args.model_name, load_in_8bit=False)
    orig.model.to(device)

    params = list(dec.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    step_iter = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(loader)
            batch = next(step_iter)

        # Move batch tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = train_step(batch, {"dec": dec, "enc": enc, "orig": orig}, {})

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if step % 1 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
