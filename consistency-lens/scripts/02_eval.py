"""Evaluate a saved checkpoint on a held-out activation set."""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
import re
import json

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from lens.utils import checkpoint as ckpt_util
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union

from transformers import PreTrainedTokenizerBase
from lens.utils.embedding_remap import remap_embeddings
from lens.evaluation.verbose_samples import (
    generate_autoregressive_continuation,
    get_top_n_tokens,
    print_formatted_table,
    print_verbose_sample_details,
    process_and_print_verbose_batch_samples,
)
import hydra
from omegaconf import DictConfig, OmegaConf


def get_project_root() -> Path:
    """Get the project root directory (consistency-lens folder)."""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    return project_root


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root if it's a relative path."""
    path = Path(path_str)
    if not path.is_absolute():
        project_root = get_project_root()
        return project_root / path
    return path


def extract_dataset_info(activation_dir: str) -> dict:
    """Extract model name, layer, and dataset info from activation directory path.
    
    Expected format: .../dataset_name/model_name/layer_X/split_name/
    """
    parts = Path(activation_dir).parts
    info = {
        'model_name': None,
        'layer': None,
        'dataset': None,
        'split': None
    }
    
    # Find layer_X pattern
    for i, part in enumerate(parts):
        if part.startswith('layer_'):
            layer_match = re.match(r'layer_(\d+)', part)
            if layer_match:
                info['layer'] = int(layer_match.group(1))
                # Model name should be one level up
                if i > 0:
                    info['model_name'] = parts[i-1]
                # Dataset should be two levels up
                if i > 1:
                    info['dataset'] = parts[i-2]
                # Split should be one level down
                if i < len(parts) - 1:
                    info['split'] = parts[i+1]
                break
    
    return info


def extract_checkpoint_info(checkpoint_path: str) -> dict:
    """Extract information from checkpoint path/name."""
    ckpt_name = Path(checkpoint_path).parent.name
    info = {
        'run_name': ckpt_name,
        'step': None
    }
    
    # Try to extract step number from filename
    filename = Path(checkpoint_path).name
    step_match = re.search(r'step[_]?(\d+)', filename)
    if step_match:
        info['step'] = int(step_match.group(1))
    
    return info


def _setup_logging() -> logging.Logger:
    """Sets up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def _build_models(
    cfg: Dict[str, Any], device: torch.device
) -> Tuple[Dict[str, torch.nn.Module], PreTrainedTokenizerBase, OrigWrapper]:
    """Builds and initializes the models (Decoder, Encoder, Original Model Wrapper, Tokenizer)."""
    model_name = cfg["model_name"]
    
    # Decoder
    dec_cfg = DecoderConfig(model_name=model_name)
    dec = Decoder(dec_cfg)
    tokenizer_name = cfg.get("tokenizer_name", model_name)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    logging.getLogger(__name__).info(f"Tokenizer name: {tokenizer_name}")

    # Encoder (created before remap)
    encoder_train_cfg = cfg.get('trainable_components', {}).get('encoder', {})
    enc_cfg = EncoderConfig(model_name=model_name, **encoder_train_cfg)
    enc = Encoder(enc_cfg)

    # Original Model Wrapper
    orig = OrigWrapper(model_name, load_in_8bit=False)
    orig.model.to(device)

    new_vocab_size = tok.vocab_size

    from lens.utils.embedding_remap import remap_embeddings
    base_tok = AutoTokenizer.from_pretrained(model_name)
    if cfg.get("tokenizer_name", model_name) != model_name:
        remap_embeddings(dec.base, base_tok, tok)
        if enc.base is not None:
            remap_embeddings(enc.base, base_tok, tok)
        remap_embeddings(orig.model, base_tok, tok)
        logging.getLogger(__name__).info("Remapped all model embeddings to new tokenizer")

    # Initialize encoder soft prompt from text if specified
    encoder_soft_prompt_text = encoder_train_cfg.get('soft_prompt_init_text')
    if encoder_soft_prompt_text:
        enc.set_soft_prompt_from_text(encoder_soft_prompt_text, tok)
        logging.getLogger(__name__).info(f"Initialized encoder soft prompt from text: {encoder_soft_prompt_text}")

    # Ensure Decoder's independent LM head matches new vocab
    if dec.out.weight.size(0) != new_vocab_size:
        import torch.nn as nn
        d_model = dec.base.config.hidden_size
        dec.out = nn.Linear(d_model, new_vocab_size, bias=False)
        with torch.no_grad():
            dec.out.weight.copy_(dec.base.get_output_embeddings().weight)
        logging.getLogger(__name__).info("Resized dec.out to new vocab size")

    models_dict = {"dec": dec.to(device), "enc": enc.to(device)}
    return models_dict, tok, orig


def _load_checkpoint(checkpoint_path: str, models: Dict[str, torch.nn.Module], device: torch.device) -> Dict[str, Any]:
    """Loads model weights from a checkpoint and returns checkpoint metadata."""
    # Load checkpoint data first
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Handle model state dicts with special care for prompt_ids
    if "models" in ckpt_data:
        for model_name, model in models.items():
            if model_name in ckpt_data["models"]:
                state_dict = ckpt_data["models"][model_name]
                
                # Special handling for decoder's prompt_ids buffer
                if model_name == "dec" and "prompt_ids" in state_dict:
                    # Remove prompt_ids from state dict to avoid size mismatch
                    # We'll set it properly later using set_prompt()
                    prompt_ids_backup = state_dict.pop("prompt_ids")
                    try:
                        model.load_state_dict(state_dict, strict=False)
                    finally:
                        # Restore it to the checkpoint data for reference
                        state_dict["prompt_ids"] = prompt_ids_backup
                else:
                    model.load_state_dict(state_dict)
    
    # Load optimizer state if needed (for evaluation we don't really need this)
    # Dummy optimizer just for compatibility with checkpoint format
    dec = models["dec"]
    enc = models["enc"]
    all_params = list(dec.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params=all_params, lr=1e-4)
    
    if "optim" in ckpt_data:
        try:
            opt.load_state_dict(ckpt_data["optim"])
        except ValueError:
            # Param group mismatch, skip optimizer state
            pass
    
    # Log checkpoint metadata
    logging.info(f"Checkpoint metadata:")
    logging.info(f"  - Step: {ckpt_data.get('step', 'N/A')}")
    logging.info(f"  - Epoch: {ckpt_data.get('epoch', 'N/A')}")
    
    if 'config' in ckpt_data:
        ckpt_config = ckpt_data['config']
        logging.info(f"  - Model: {ckpt_config.get('model_name', 'N/A')}")
        logging.info(f"  - Tokenizer: {ckpt_config.get('tokenizer_name', 'N/A')}")
        logging.info(f"  - Layer L: {ckpt_config.get('layer_l', 'N/A')}")
        logging.info(f"  - Decoder prompt: '{ckpt_config.get('decoder_prompt', 'N/A')}'")
        logging.info(f"  - t_text: {ckpt_config.get('t_text', 'N/A')}")
    
    return ckpt_data


def _prepare_data(
    cfg: Dict[str, Any], effective_act_dir: str, log: logging.Logger
) -> DataLoader:
    """Loads and prepares the dataset and DataLoader."""
    log.info(f"Loading activations from {effective_act_dir}")
    full_ds = ActivationDataset(effective_act_dir)

    # Get evaluation config
    eval_cfg = cfg.get('evaluation', {})
    vf = eval_cfg.get('val_fraction', None)
    split_seed = eval_cfg.get('split_seed', cfg.get('split_seed', 42))

    if vf is not None and 0 < vf < 1.0:
        vsz = int(len(full_ds) * vf)
        tsz = len(full_ds) - vsz
        if tsz == 0 or vsz == 0:
            log.warning(f"Validation split resulted in an empty train or validation set (tsz={tsz}, vsz={vsz}). Using full dataset for evaluation.")
            ds = full_ds
        else:
            _, ds = random_split(
                full_ds,
                [tsz, vsz],
                generator=torch.Generator().manual_seed(split_seed),
            )
    else:
        ds = full_ds
        if vf == 0.0:
            log.info("val_fraction is 0, using full dataset for evaluation (as validation set).")
        elif vf == 1.0:
             log.info("val_fraction is 1, using full dataset for evaluation.")
        elif vf is None:
            log.info("val_fraction is None, using full dataset for evaluation.")

    batch_size = eval_cfg.get('batch_size', 4)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    log.info(f"Loaded {len(ds)} samples for evaluation.")
    return loader


def _process_and_print_verbose_batch_samples(
    batch: Dict[str, Any],
    cfg: Dict[str, Any],
    models: Dict[str, torch.nn.Module],
    orig: OrigWrapper,
    tok: PreTrainedTokenizerBase,
    sch_args: Dict[str, Any],
    device: torch.device,
    printed_count_so_far: int,
    cached_prefix_ids: torch.Tensor | None = None
) -> int:
    """Wrapper for the shared verbose sample processing function."""
    eval_cfg = cfg.get('evaluation', {})
    return process_and_print_verbose_batch_samples(
        batch=batch,
        cfg=cfg,
        models=models,
        orig=orig,
        tok=tok,
        sch_args=sch_args,
        device=device,
        num_samples=eval_cfg.get('verbose_samples', 3),
        top_n_analysis=eval_cfg.get('top_n_analysis', 3),
        printed_count_so_far=printed_count_so_far,
        generate_continuation=True,
        continuation_tokens=30,
        cached_prefix_ids=cached_prefix_ids
    )

def evaluate_custom_strings(
    strings: List[str],
    models: Dict[str, torch.nn.Module],
    orig: OrigWrapper,
    cfg: Dict[str, Any],
    device: torch.device,
    tok: PreTrainedTokenizerBase,
    log: logging.Logger,
) -> Dict[str, Any]:
    """Evaluates the lens on custom input strings.
    
    Args:
        strings: List of strings to analyze
        models: Dictionary containing 'dec' and 'enc' models
        orig: Original model wrapper
        cfg: Configuration dictionary
        device: Device to run on
        tok: Tokenizer
        log: Logger
        
    Returns:
        Dictionary with analysis results for each string
    """
    models["dec"].eval()
    models["enc"].eval()
    
    results = []
    layer_l = cfg.get('layer_l', 5)
    
    for idx, text in enumerate(strings):
        log.info(f"Processing string {idx+1}/{len(strings)}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Tokenize the input
        inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        
        # Get activations at layer L
        with torch.no_grad():
            # Run through original model to get activations
            orig_outputs = orig.model(input_ids, output_hidden_states=True)
            hidden_states = orig_outputs.hidden_states
            
            # Extract activation at layer L
            # Note: hidden_states includes embedding layer, so layer L is at index L+1
            # Get the last token position by default
            seq_len = input_ids.shape[1]
            last_pos = seq_len - 1
            A = hidden_states[layer_l + 1][:, last_pos, :]  # Shape: (1, hidden_size)
            
            # Also get the full sequence for potential position analysis
            A_full_seq = hidden_states[layer_l + 1]  # Shape: (1, seq_len, hidden_size)
        
        # Generate explanation through decoder
        tau = cfg.get("gumbel_tau_schedule", {}).get("end_value", 1.0)
        t_text = cfg.get("t_text", 5)
        
        # Use the same generation method as in training
        if models["dec"].config.use_flash_attention:
            gen = models["dec"].generate_soft_kv_flash(A, max_length=t_text, gumbel_tau=tau)
        elif models["dec"].config.use_kv_cache:
            gen = models["dec"].generate_soft_kv_cached(A, max_length=t_text, gumbel_tau=tau)
        else:
            gen = models["dec"].generate_soft(A, max_length=t_text, gumbel_tau=tau)
        
        # Get reconstruction
        A_hat = models["enc"](gen.generated_text_embeddings)
        
        # Decode the generated tokens
        generated_tokens = gen.hard_token_ids[0].cpu().tolist()
        generated_text = tok.decode(generated_tokens, skip_special_tokens=True)
        
        # Compute reconstruction error
        mse = torch.nn.functional.mse_loss(A, A_hat).item()
        
        # Get predictions from original vs reconstructed
        with torch.no_grad():
            # Original model's next token prediction
            orig_logits = orig.model(input_ids).logits[0, -1, :]  # Last position
            orig_probs = torch.softmax(orig_logits, dim=-1)
            
            # To get reconstructed predictions, we need to inject A_hat back
            # This requires using the model's forward with custom hidden states
            # For now, we'll compute cosine similarity as a proxy
            cosine_sim = torch.nn.functional.cosine_similarity(A, A_hat, dim=-1).mean().item()
        
        # Analyze multiple positions if requested
        position_results = []
        num_positions = min(5, A_full_seq.shape[1])  # Analyze up to 5 positions
        
        for pos in range(num_positions):
            A_pos = A_full_seq[:, pos, :].unsqueeze(0)
            
            # Generate for this position
            if models["dec"].config.use_flash_attention:
                gen_pos = models["dec"].generate_soft_kv_flash(A_pos, max_length=t_text, gumbel_tau=tau)
            elif models["dec"].config.use_kv_cache:
                gen_pos = models["dec"].generate_soft_kv_cached(A_pos, max_length=t_text, gumbel_tau=tau)
            else:
                gen_pos = models["dec"].generate_soft(A_pos, max_length=t_text, gumbel_tau=tau)
            
            pos_tokens = gen_pos.hard_token_ids[0].cpu().tolist()
            pos_text = tok.decode(pos_tokens, skip_special_tokens=True)
            
            position_results.append({
                'position': pos,
                'token': tok.decode([input_ids[0, pos].item()]) if pos < len(input_ids[0]) else '<pad>',
                'explanation': pos_text
            })
        
        # Get the token at the analyzed position
        analyzed_token = tok.decode([input_ids[0, last_pos].item()])
        
        result = {
            'input_text': text,
            'input_length': len(input_ids[0]),
            'layer': layer_l,
            'analyzed_position': last_pos,
            'analyzed_token': analyzed_token,
            'main_explanation': generated_text,
            'reconstruction_mse': mse,
            'cosine_similarity': cosine_sim,
            'position_analyses': position_results,
            'generated_token_ids': generated_tokens,
        }
        
        results.append(result)
        
        # Print summary
        log.info(f"  Analyzed token: '{analyzed_token}' at position {last_pos}")
        log.info(f"  Explanation: {generated_text}")
        log.info(f"  MSE: {mse:.6f}, Cosine Sim: {cosine_sim:.4f}")
    
    return {'string_analyses': results}


def _evaluate_model(
    loader: DataLoader,
    models: Dict[str, torch.nn.Module],
    orig: OrigWrapper,
    cfg: Dict[str, Any],
    device: torch.device,
    tok: PreTrainedTokenizerBase,
    log: logging.Logger,
) -> dict:
    """Runs the evaluation loop."""
    total_loss = 0.0
    n_seen = 0
    printed_verbose_total = 0

    eval_cfg = cfg.get('evaluation', {})
    num_batches = eval_cfg.get('num_batches', 25)
    verbose_samples = eval_cfg.get('verbose_samples', 3)

    models["dec"].eval()
    models["enc"].eval()
    
    # Cache tokenized natural language prefix if specified
    cached_prefix_ids = None
    lm_loss_natural_prefix_text = cfg.get('lm_loss_natural_prefix')
    if lm_loss_natural_prefix_text and isinstance(lm_loss_natural_prefix_text, str):
        cached_prefix_ids = tok(lm_loss_natural_prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        log.info(f"Cached natural language prefix: '{lm_loss_natural_prefix_text}' ({cached_prefix_ids.shape[1]} tokens)")

    with torch.no_grad():
        for b_idx, batch in tqdm(enumerate(loader), total=num_batches, desc="Evaluating"):
            if b_idx >= num_batches:
                log.info(f"Reached max {num_batches} batches.")
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            sch_args = {
                "tau": cfg["gumbel_tau_schedule"]["end_value"],
                "t_text": cfg["t_text"],
                "alpha": cfg["alpha_schedule"].get("value", 0.1),
                "ce_weight": cfg.get("ce_weight", 0.01),
                "kl_base_weight": cfg.get("kl_base_weight", 1.0),
                "lm_base_weight": cfg.get("lm_base_weight", 0.0),
                "entropy_weight": cfg.get("entropy_weight", 0.0),
                "mse_weight": cfg.get("mse_weight", 0.0),
            }
            
            current_models_for_step = {"dec": models["dec"], "enc": models["enc"], "orig": orig}
            losses = train_step(batch, current_models_for_step, sch_args,
                              lm_loss_natural_prefix=cfg.get('lm_loss_natural_prefix'),
                              tokenizer=tok)
            
            if printed_verbose_total < verbose_samples:
                num_printed_this_batch = _process_and_print_verbose_batch_samples(
                    batch, cfg, models, orig, tok, sch_args, device, printed_verbose_total, cached_prefix_ids
                )
                printed_verbose_total += num_printed_this_batch

            total_loss += losses["total"].item() * batch["A"].size(0)
            n_seen += batch["A"].size(0)

    results = {
        'total_samples': n_seen,
        'avg_loss': total_loss / n_seen if n_seen > 0 else 0.0,
        'verbose_samples_printed': printed_verbose_total
    }
    
    if n_seen > 0:
        avg_loss = total_loss / n_seen
        log.info("Eval loss_total: %.4f over %d samples", avg_loss, n_seen)
    else:
        log.info("No samples processed during evaluation.")
    
    return results


# ----------------------------------------------------------------------------
# Hydra entry point
# ----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(hcfg: DictConfig) -> None:
    """Main entry point for evaluating a Consistency-Lens checkpoint."""
    log = _setup_logging()
    cfg = OmegaConf.to_container(hcfg, resolve=True)
    
    # Get evaluation config
    eval_cfg = cfg.get('evaluation', {})
    
    # Check if checkpoint is specified
    checkpoint_path = cfg.get('checkpoint')
    if not checkpoint_path:
        log.error("No checkpoint specified. Use: checkpoint=path/to/checkpoint.pt")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Extract checkpoint info
    ckpt_info = extract_checkpoint_info(checkpoint_path)
    
    # Setup output directory if requested
    output_dir = None
    if eval_cfg.get('output_dir'):
        output_dir = Path(eval_cfg['output_dir'])
    elif eval_cfg.get('save_results', False):
        # Auto-generate output dir based on checkpoint
        base_eval_dir = Path("outputs/evaluations")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_eval_dir / f"{ckpt_info['run_name']}_step{ckpt_info['step']}_{timestamp}"
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Evaluation results will be saved to: {output_dir}")

    # Fetch tokenizer_name early to construct paths correctly
    model_name_for_tokenizer = cfg["model_name"] 
    tokenizer_name = cfg.get("tokenizer_name", model_name_for_tokenizer)

    models_dict, tok, orig_model = _build_models(cfg, device)
    ckpt_data = _load_checkpoint(checkpoint_path, models_dict, device)
    
    # Set decoder prompt from checkpoint if available
    if 'config' in ckpt_data:
        ckpt_config = ckpt_data['config']
        ckpt_prompt = ckpt_config.get('decoder_prompt')
        if ckpt_prompt:
            log.info(f"Setting decoder prompt from checkpoint: '{ckpt_prompt}'")
            models_dict['dec'].set_prompt(ckpt_prompt, tok)
            
            # Warn if prompt differs from current config
            current_prompt = cfg.get('decoder_prompt', 'Explain: ')
            if ckpt_prompt != current_prompt:
                log.warning(f"Decoder prompt in checkpoint ('{ckpt_prompt}') differs from current config ('{current_prompt}')")
        else:
            # Fallback to config prompt if not in checkpoint
            log.warning("No decoder prompt found in checkpoint metadata, using config prompt")
            models_dict['dec'].set_prompt(cfg.get('decoder_prompt', 'Explain: '), tok)
            
        # Check tokenizer consistency
        ckpt_tokenizer = ckpt_config.get('tokenizer_name')
        current_tokenizer = cfg.get('tokenizer_name', cfg['model_name'])
        if ckpt_tokenizer and ckpt_tokenizer != current_tokenizer:
            log.warning(f"Tokenizer in checkpoint ('{ckpt_tokenizer}') differs from current config ('{current_tokenizer}'). "
                       f"This may cause issues if vocabularies differ significantly.")
            
        # Check layer_l consistency
        ckpt_layer = ckpt_config.get('layer_l')
        current_layer = cfg.get('layer_l')
        if ckpt_layer is not None and current_layer is not None and ckpt_layer != current_layer:
            log.warning(f"Layer L in checkpoint ({ckpt_layer}) differs from current config ({current_layer}). "
                       f"Make sure activation data matches the checkpoint's layer.")
    else:
        # Old checkpoint without config - use current config
        log.warning("Checkpoint has no config metadata (old format?), using current config values")
        models_dict['dec'].set_prompt(cfg.get('decoder_prompt', 'Explain: '), tok)

    # Check if custom strings are provided
    custom_strings = eval_cfg.get('custom_strings', [])
    custom_strings_file = eval_cfg.get('custom_strings_file', None)
    
    # Load strings from file if provided
    if custom_strings_file:
        strings_path = Path(custom_strings_file)
        if strings_path.exists():
            with open(strings_path, 'r') as f:
                file_strings = [line.strip() for line in f if line.strip()]
                custom_strings.extend(file_strings)
                log.info(f"Loaded {len(file_strings)} strings from {custom_strings_file}")
        else:
            log.warning(f"Custom strings file not found: {custom_strings_file}")
    
    # If custom strings are provided, evaluate them instead of dataset
    if custom_strings:
        layer_l = cfg.get('layer_l')
        log.info("=" * 60)
        log.info("CUSTOM STRING EVALUATION")
        log.info("=" * 60)
        log.info(f"Checkpoint: {checkpoint_path}")
        log.info(f"Run Name: {ckpt_info['run_name']}")
        log.info(f"Step: {ckpt_info.get('step', 'unknown')}")
        log.info(f"Number of strings: {len(custom_strings)}")
        log.info(f"Layer: {layer_l}")
        log.info("=" * 60)
        
        results = evaluate_custom_strings(
            custom_strings, models_dict, orig_model, cfg, device, tok, log
        )
        
        # Add metadata
        results['mode'] = 'custom_strings'
        results['num_strings'] = len(custom_strings)
    else:
        # Normal dataset evaluation
        # Determine activation directory
        base_eval_act_dir_str = eval_cfg.get('activation_dir') or cfg.get("val_activation_dir")
        layer_l = cfg.get('layer_l')
        
        effective_act_dir: str | None = None
        if base_eval_act_dir_str:
            base_path = resolve_path(base_eval_act_dir_str)
            model_name_clean = cfg['model_name'].replace("/", "_")
            effective_act_dir = str(base_path.parent / model_name_clean / f"layer_{layer_l}" / base_path.name)

        if not effective_act_dir:
            log.critical("No effective activation directory could be determined for evaluation. Exiting.")
            return
            
        # Extract dataset info
        dataset_info = extract_dataset_info(effective_act_dir)
        
        # Log evaluation info
        log.info("=" * 60)
        log.info("EVALUATION INFO")
        log.info("=" * 60)
        log.info(f"Checkpoint: {checkpoint_path}")
        log.info(f"Run Name: {ckpt_info['run_name']}")
        log.info(f"Step: {ckpt_info.get('step', 'unknown')}")
        log.info(f"Dataset: {dataset_info.get('dataset', 'unknown')}")
        log.info(f"Model: {dataset_info.get('model_name', 'unknown')}")
        log.info(f"Layer: {dataset_info.get('layer', 'unknown')}")
        log.info(f"Split: {dataset_info.get('split', 'unknown')}")
        log.info("=" * 60)
        
        loader = _prepare_data(cfg, effective_act_dir, log)

        results = _evaluate_model(loader, models_dict, orig_model, cfg, device, tok, log)
        results['dataset_info'] = dataset_info
        results['mode'] = 'dataset'
    
    # Add metadata to results
    results['checkpoint_path'] = str(checkpoint_path)
    results['checkpoint_info'] = ckpt_info
    results['eval_config'] = eval_cfg
    results['timestamp'] = datetime.now().isoformat()
    
    # Save results if requested
    if output_dir:
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved evaluation results to: {results_file}")
        
        # Also save a summary text file
        summary_file = output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Run Name: {ckpt_info['run_name']}\n")
            f.write(f"Step: {ckpt_info.get('step', 'unknown')}\n")
            
            if results.get('mode') == 'custom_strings':
                f.write(f"Mode: Custom String Analysis\n")
                f.write(f"Number of strings: {results.get('num_strings', 0)}\n")
                f.write(f"Layer: {cfg.get('layer_l', 'unknown')}\n")
                f.write(f"\nString Analyses:\n")
                f.write(f"----------------\n")
                
                if 'string_analyses' in results:
                    for i, analysis in enumerate(results['string_analyses']):
                        f.write(f"\n{i+1}. Input: {analysis['input_text'][:50]}{'...' if len(analysis['input_text']) > 50 else ''}\n")
                        f.write(f"   Analyzed token: '{analysis['analyzed_token']}' at position {analysis['analyzed_position']}\n")
                        f.write(f"   Explanation: {analysis['main_explanation']}\n")
                        f.write(f"   MSE: {analysis['reconstruction_mse']:.6f}, Cosine Sim: {analysis['cosine_similarity']:.4f}\n")
            else:
                dataset_info = results.get('dataset_info', {})
                f.write(f"Dataset: {dataset_info.get('dataset', 'unknown')}/{dataset_info.get('split', 'unknown')}\n")
                f.write(f"Model: {dataset_info.get('model_name', 'unknown')} (Layer {dataset_info.get('layer', 'unknown')})\n")
                f.write(f"\nResults:\n")
                f.write(f"--------\n")
                f.write(f"Average Loss: {results.get('avg_loss', 0.0):.6f}\n")
                f.write(f"Total Samples: {results.get('total_samples', 0)}\n")
            
            f.write(f"\nTimestamp: {results['timestamp']}\n")
        
        log.info(f"Saved evaluation summary to: {summary_file}")


if __name__ == "__main__":
    main()