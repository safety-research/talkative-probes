"""Minimal single-process training loop.

The *real* entry-point will wire up DeepSpeed & read ``lens.yaml``.  For the MVP
we just train Decoder/Encoder for a handful of steps to prove gradients flow.
"""

from __future__ import annotations

import argparse
import logging

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
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Logging setup (console). W&B handled via lens.utils.logging.
    # ---------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

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
    activation_dir = config['activation_dumper']['output_dir']
    max_steps = config['max_train_steps']
    learning_rate = config['learning_rate']
    t_text = config['t_text']
    wandb_config = config['wandb']
    prompt_text = config['decoder_prompt']
    ce_weight = config.get('ce_weight', 0.01)
    kl_base_weight = config.get('kl_base_weight', 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_init(project=wandb_config['project'], config=config, mode=wandb_config['mode'])

    log.info("Starting training run – model=%s, activations=%s, steps=%d", model_name, activation_dir, max_steps)

    dataset = ActivationDataset(activation_dir)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No .pt files found in {activation_dir}.  Run scripts/00_dump_activations.py first."
        )

    log.info("Activation dataset size: %d samples", len(dataset))

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate)

    dec_raw = Decoder(DecoderConfig(model_name=model_name, n_prompt_tokens=config['decoder_n_prompt_tokens']))
    enc_raw = Encoder(model_name)

    if config.get('compile_models', True):
        dec = torch.compile(dec_raw).to(device)
        enc = torch.compile(enc_raw).to(device)
    else:
        dec = dec_raw.to(device)
        enc = enc_raw.to(device)

    orig = OrigWrapper(model_name, load_in_8bit=False)
    orig.model.to(device)

    groups = param_groups(dec, learning_rate) + param_groups(enc, learning_rate)
    opt = torch.optim.AdamW(groups)
    # gradscaler takes device not device_type
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")

    start_step = 0
    if args.resume:
        from lens.utils import checkpoint as ckpt
        rec = ckpt.load(args.resume, models={"dec": dec, "enc": enc}, optim=opt, map_location=device)
        start_step = int(rec.get("step", 0)) + 1

    step_iter = iter(loader)
    for step in range(start_step, max_steps):
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
                batch,
                {"dec": dec, "enc": enc, "orig": orig},
                {
                    "tau": current_tau,
                    "T_text": t_text,
                    "alpha": current_alpha,
                    "ce_weight": ce_weight,
                    "kl_base_weight": kl_base_weight,
                },
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
            log.info(
                "step %04d | loss %.4f | mse %.4f | ce %.4f | kl %.4f | tau %.3f | alpha %.3f",
                step,
                loss.item(),
                losses["mse"].item(),
                losses["ce"].item(),
                losses["kl"].item(),
                current_tau,
                current_alpha,
            )
            log_metrics(
                {
                    "loss/total": loss.item(),
                    "loss/mse": losses["mse"].item(),
                    "loss/ce": losses["ce"].item(),
                    "loss/kl": losses["kl"].item(),
                    "params/tau": current_tau,
                    "params/alpha": current_alpha,
                    "params/ce_w": ce_weight,
                    "params/kl_w": kl_base_weight,
                },
                step=step,
            )

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if step % args.save_every == 0 and step > 0:
            from lens.utils import checkpoint

            checkpoint.save(
                path=f"outputs/ckpt_step_{step}.pt",
                models={"dec": dec, "enc": enc},
                optim=opt,
                step=step,
                tau=current_tau,
                alpha=current_alpha,
            )
            log.info("Saved checkpoint at step %d", step)


if __name__ == "__main__":
    main()
