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
from lens.training.optim import param_groups
from lens.utils.logging import init as log_init, log as log_metrics
from lens.training.schedules import get_schedule_value
import yaml
from transformers import AutoTokenizer
from torch.amp import autocast, GradScaler
from contextlib import nullcontext


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train Consistency-Lens MVP")
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens.yaml", help="Path to the lens.yaml config file.")
    # Allow overriding some key parameters from the command line
    parser.add_argument("--model_name", type=str, help="Override model_name from config.")
    parser.add_argument("--activation_dir", type=str, help="Override activation_dir from config.")
    parser.add_argument("--max_train_steps", type=int, help="Override max_train_steps from config.")
    parser.add_argument("--learning_rate", type=float, help="Override learning_rate from config.")
    parser.add_argument("--t_text", type=int, help="Override t_text from config.")
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with CLI arguments if provided
    if args.model_name: config['model_name'] = args.model_name
    if args.activation_dir: config['activation_dumper']['output_dir'] = args.activation_dir # Assuming structure
    if args.max_train_steps: config['max_train_steps'] = args.max_train_steps
    if args.learning_rate: config['learning_rate'] = args.learning_rate
    if args.t_text: config['t_text'] = args.t_text

    # Use overridden values or defaults from config
    model_name = config['model_name']
    activation_dir = config.get('activation_dumper', {}).get('output_dir', 'consistency-lens/data/activations')
    max_steps = config['max_train_steps']
    learning_rate = config['learning_rate']
    t_text = config['t_text']
    wandb_config = config.get("wandb", {})
    prompt_text = config.get("decoder_prompt", "Explain: ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_init(project=wandb_config.get("project", "consistency-lens"), config=config, mode=wandb_config.get("mode", "offline"))

    dataset = ActivationDataset(activation_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No .pt files found in {activation_dir}.  Run scripts/00_dump_activations.py first.")

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate)

    dec = Decoder(DecoderConfig(model_name=model_name, n_prompt_tokens=config['decoder_n_prompt_tokens'])).to(device)
    # Set textual prompt
    tok = AutoTokenizer.from_pretrained(model_name)
    dec.set_prompt(prompt_text, tok)
    enc = Encoder(model_name).to(device)
    orig = OrigWrapper(model_name, load_in_8bit=False)
    orig.model.to(device)

    groups = param_groups(dec, learning_rate) + param_groups(enc, learning_rate)
    opt = torch.optim.AdamW(groups)

    scaler = GradScaler(device_type="cuda", enabled=device.type == "cuda")

    step_iter = iter(loader)
    for step in range(max_steps):
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(loader)
            batch = next(step_iter)

        # Move batch tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)

        cast_ctx = autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with cast_ctx:
            current_tau = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps)
            current_alpha = get_schedule_value(config['alpha_schedule'], step, max_steps)

            losses = train_step(
                batch, {"dec": dec, "enc": enc, "orig": orig}, {"tau": current_tau, "T_text": t_text, "alpha": current_alpha}
            )
            loss = losses["total"]

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if step % 10 == 0:
            print(
                f"step {step:04d} | loss {loss.item():.4f} | mse {losses['mse'].item():.4f} | "
                f"ce {losses['ce'].item():.4f} | kl {losses['kl'].item():.4f} | tau {current_tau:.3f} | alpha {current_alpha:.3f}"
            )
            log_metrics(
                {
                    "loss/total": loss.item(),
                    "loss/mse": losses["mse"].item(),
                    "loss/ce": losses["ce"].item(),
                    "loss/kl": losses["kl"].item(),
                    "params/tau": current_tau,
                    "params/alpha": current_alpha,
                },
                step=step,
            )


if __name__ == "__main__":
    main()
