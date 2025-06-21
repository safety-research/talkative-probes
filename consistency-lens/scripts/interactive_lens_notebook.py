# %% [markdown]
# # Consistency Lens Interactive Analysis
# 
# This notebook provides an interactive interface for analyzing text through a trained consistency lens.
# Run cells individually to explore how the lens interprets different inputs.


# need to source the proper uv env /home/kitf/.cache/uv/envs/consistency-lens/bin/python
# need to uv add jupyter seaborn
# and install jupyter if necessary

# %%
# Imports and setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
# Add parent directory to path for imports
toadd = str(Path(__file__).parent.parent)
print("Adding to path:",toadd)
sys.path.append(toadd)
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.embedding_remap import remap_embeddings
from lens.evaluation.verbose_samples import process_and_print_verbose_batch_samples
from lens.models.tuned_lens import load_full_tuned_lens
import logging
import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Imports complete")

# %% [markdown]
# ## Load the Checkpoint
# 
# Update the path below to point to your trained checkpoint:

# %%
# Configuration - UPDATE THIS PATH
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_frozen_lr1e-4_t32_5ep_resume_0604_1343_zeroLMKLkvNO_RA_NEW_bestWID_dist4/gpt2_frozen_step10000_epoch2_.pt"#outputs/checkpoints/YOUR_CHECKPOINT.pt"  # <-- Change this!
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_frozen_lr1e-3_t16_5ep_resume_0604_1516_zeroLMKLkvNO_RA_NEW_bestWIDsmallLM_dist4/gpt2_frozen_step10000_epoch3_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_frozen_lr1e-3_t16_5ep_resume_0604_1516_zeroLMKLkvNO_RA_NEW_bestWIDsmallLM_dist4/gpt2_frozen_step19564_epoch6_final.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr1e-3_t32_20ep_resume_0609_1410_tinyLMKVext_fl32_dist4_slurm634/gpt2_frozen_step124000_epoch16_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_postfix_warmup_OC_GPT2_L6_e5_prog-unfreeze_lr1e-4_t16_30ep_resume_0609_1456_tinyLMKV_ufwarmup_resume_dist4/gpt2_frozen_step27000_epoch7_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_postfix_warmup_OC_GPT2_L6_e5_prog-unfreeze_lr1e-3_t16_30ep_resume_0609_1655_tinyLMKV_ufwarmup_resumeMSE_dist4/gpt2_frozen_step16000_epoch5_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr3e-4_t32_20ep_resume_0609_2327_tinyLMKVext_dist4/gpt2_frozen_step49000_epoch7_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr3e-4_t32_20ep_resume_0609_2327_tinyLMKVext_dist4/gpt2_frozen_step116000_epoch15_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr3e-4_t32_4ep_resume_0611_1502_tinyLMKVext_resumeold2highlr_dist4/"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_postfix_warmup_tuned_OC_GPT2_L6_e6_prog-unfreeze_lr1e-3_t32_8ep_0610_2120_MSE_unfrz_dist4/"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_unfreeze_nopostfix_tuned_pos_OC_GPT2_L6_e5_full_lr1e-3_t32_8ep_resume_0612_1734_new_fix_dist4/"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_unfreeze_nopostfix_tuned_pos_OC_GPT2_L6_e5_full_lr1e-3_t32_1ep_resume_0613_1323_new_fix_dist4_slurm862/"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_unfreeze_nopostfix_tuned_pos_OC_GPT2_L6_e5_full_lr1e-3_t32_8ep_resume_0613_1205_new_fix_dist4"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_unfreeze_nopostfix_tuned_pos_OC_GPT2_L6_e5_full_lr1e-4_t64_4ep_resume_0614_0954_new_fix_wide_anneal_LM_dist4_slurm940"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_unfreeze_nopostfix_tuned_pos_OC_GPT2_L6_e5_full_lr1e-4_t63_1ep_resume_0618_0205_new_fix_wide_ADDTOK_dist8_slurm1077"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_nopostfix_tuned_pos_OC_GPT2_L6_e6_full_lr1e-6_t4_1ep_resume_0617_1756_frozenenADD_Gumstart_L6_dist4_slurm1076"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_nopostfix_tuned_pos_OC_GPT2_L6_e6_full_lr1e-4_t4_1ep_resume_0618_0816_frozenenADD_Gumstart_L6_dist4/"
#CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_nopostfix_tuned_pos_OC_GPT2_L6_e6_full_lr1e-3_t8_1ep_resume_0618_0934_frozenenADD_L6_generalisation_dist4"
# CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_unfreeze_postfix_tuned_pos_OC_GPT2_L6_e5_prog-unfreeze_lr1e-3_t32_8ep_resume_0612_1252_new_fix_dist4_slurm823"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_nopostfix_tuned_pos_OC_GPT2_L6_e6_full_lr1e-3_t4_1ep_resume_0618_1727_frozenenADD_Gumstart_L6_dist4"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-3_t7_1ep_resume_0619_0019_frozenenADD_shorter_dist4_slurm1112/"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-4_t16_1ep_resume_0619_1737_frozenenADD_shorter_dist4_slurm1180o"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-4_t16_1ep_resume_0619_1737_frozenenADD_shorter_dist4_slurm1180"
# Optional: specify device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# %%
# Load checkpoint and initialize models
class LensAnalyzer:
    """Analyzer for consistency lens."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", comparison_tl_checkpoint: Union[str,bool] = True, do_not_load_weights: bool = False, make_xl: bool = False, use_bf16: bool = False):
        self.device = torch.device(device)
        self.use_bf16 = use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        if self.use_bf16:
            print(f"✓ BF16 autocast enabled")
        
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.is_dir():
                pt_files = sorted(checkpoint_path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not pt_files:
                    raise FileNotFoundError(f"No .pt files found in directory {checkpoint_path}")
                checkpoint_path = pt_files[0]
                print(f"Selected most recent checkpoint: {checkpoint_path}")
            self.checkpoint_path = checkpoint_path

            print(f"Loading checkpoint from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # Load to CPU first

        # Extract config
        if 'config' in ckpt:
            self.config = ckpt['config']
            self.layer = self.config.get('layer_l')
            self.model_name = self.config.get('model_name', 'gpt2')
            if make_xl:
                self.model_name = "openai-community/gpt2-xl"
                self.config['model_name'] = self.model_name
                self.config['layer_l'] = 24
                self.config['trainable_components']['encoder']['output_layer'] = 24
            tokenizer_name = self.config.get('tokenizer_name', self.model_name)
            self.t_text = self.config.get('t_text')
            self.tau = self.config.get('gumbel_tau_schedule', {}).get('end_value', 1.0)
        else:
            raise ValueError("No config found in checkpoint")
        
        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Build models following the pattern from 01_train_distributed.py
        print(f"Building models ({self.model_name})...")
        
        # Extract trainable components configuration
        trainable_components_config = self.config.get('trainable_components', {})
        decoder_train_cfg = trainable_components_config.get('decoder', {})
        encoder_train_cfg = trainable_components_config.get('encoder', {})
        
        # Initialize models
        decoder_config_obj = DecoderConfig(
            model_name=self.model_name,
            **decoder_train_cfg
        )
        self.decoder = Decoder(decoder_config_obj)
        
        encoder_config_obj = EncoderConfig(
            model_name=self.model_name,
            **encoder_train_cfg
        )
        self.encoder = Encoder(encoder_config_obj)
        
        self.orig_model = OrigWrapper(self.model_name, load_in_8bit=False)
        
        # Initialize Decoder prompt (before loading weights)
        if 'decoder_prompt' in self.config and self.config['decoder_prompt']:
            print(f"Setting decoder prompt: \"{self.config['decoder_prompt']}\"")
            self.decoder.set_prompt(self.config['decoder_prompt'], self.tokenizer)
        
        # Initialize Encoder soft prompt (before loading weights)
        if encoder_config_obj.soft_prompt_init_text:
            print(f"Setting encoder soft prompt from text: \"{encoder_config_obj.soft_prompt_init_text}\"")
            self.encoder.set_soft_prompt_from_text(encoder_config_obj.soft_prompt_init_text, self.tokenizer)
        elif encoder_config_obj.soft_prompt_length > 0:
            print(f"Encoder using randomly initialized soft prompt of length {encoder_config_obj.soft_prompt_length}.")
        
        # Move models to device
        self.decoder.to(self.device)
        self.encoder.to(self.device)
        self.orig_model.to(self.device)
        self.decoder.eval()
        self.encoder.eval()
        self.orig_model.model.eval()

        if self.decoder.config.use_kv_cache:
            print("Using KV cache")
        else:
            print("Not using KV cache")
             # Setup logger
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        # Load weights using CheckpointManager
        if checkpoint_path is not None and not do_not_load_weights:
            from lens.utils.checkpoint_manager import CheckpointManager
        
       
        
            # Minimal checkpoint config for CheckpointManager
            checkpoint_config = {
                "checkpoint": {
                    "enabled": True,
                    "output_dir": str(self.checkpoint_path.parent),
                    "strict_load": True
                }
            }
        
            checkpoint_manager = CheckpointManager(checkpoint_config, logger)
        
            # Load model weights
            models_to_load = {"decoder": self.decoder, "encoder": self.encoder}
        
            try:
                loaded_data = checkpoint_manager.load_checkpoint(
                    str(self.checkpoint_path),
                    models=models_to_load,
                    optimizer=None,
                    map_location=self.device
                )
                print(f"✓ Loaded checkpoint from step {loaded_data.get('step', 'unknown')}")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}") from e
        elif do_not_load_weights:
            logger.info(f"Not loading weights from {checkpoint_path},  only config and tokenizer will be loaded.")
        
        # Set models to eval mode
        self.decoder.eval()
        self.encoder.eval()
        self.orig_model.model.eval()
        
        # Load comparison TunedLens if specified
        self.comparison_tuned_lens = None
        if comparison_tl_checkpoint:
            logger.info(f"Attempting to load comparison TunedLens from: {comparison_tl_checkpoint} for model {self.model_name}")
            try:
                # Load to CPU first, then move to device
                loaded_tl = load_full_tuned_lens(
                    model_or_model_name=self.model_name,
                    checkpoint_path_or_name=comparison_tl_checkpoint if isinstance(comparison_tl_checkpoint, str) else None,
                    device="cpu", 
                    log=logger,
                    is_main_process=True # In notebook, we act as main process
                )
                if loaded_tl:
                    self.comparison_tuned_lens = loaded_tl.to(self.device)
                    self.comparison_tuned_lens.eval()
                    logger.info(f"✓ Successfully loaded and moved comparison TunedLens to {self.device}.")
                else:
                    logger.warning(f"Failed to load comparison TunedLens from {comparison_tl_checkpoint}.")
            except Exception as e:
                logger.error(f"Error loading comparison TunedLens: {e}", exc_info=True)
                self.comparison_tuned_lens = None
        
        print(f"✓ Ready! Model: {self.model_name}, Layer: {self.layer}")
    
    def analyze_all_tokens(self, text: str, seed=42, batch_size=32) -> pd.DataFrame:
        """Analyze all tokens in the text and return results as DataFrame."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        # Get all hidden states
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
                outputs = self.orig_model.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
        
        torch.manual_seed(seed)

        # Extract all activations from the single batch item.
        # hidden_states[self.layer + 1] has shape (1, seq_len, hidden_dim).
        A_full_sequence = hidden_states[self.layer + 1].squeeze(0)  # Shape: (seq_len, hidden_dim)

        all_kl_divs = []
        all_mses = []
        all_relative_rmses = []
        all_gen_hard_token_ids = []

        with torch.no_grad():
            # Use tqdm if available
            try:
                iterator = tqdm.tqdm(range(0, seq_len, batch_size), desc="Analyzing tokens in batches")
            except ImportError:
                iterator = range(0, seq_len, batch_size)

            for i in iterator:
                batch_start = i
                batch_end = min(i + batch_size, seq_len)
                current_batch_size = batch_end - batch_start
                
                token_positions_batch = torch.arange(batch_start, batch_end, device=self.device)
                A_batch = A_full_sequence[token_positions_batch]

                A_hat_batch = None
                logits_orig_full = None
                logits_recon_full = None

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
                    # Generate explanations for all positions in the current batch.
                    if self.decoder.config.use_kv_cache:
                        gen = self.decoder.generate_soft_kv_cached(A_batch, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=token_positions_batch)
                    else:
                        gen = self.decoder.generate_soft(A_batch, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=token_positions_batch)
                    
                    all_gen_hard_token_ids.append(gen.hard_token_ids)

                    # Reconstruct activations in a batch from the generated hard tokens.
                    generated_token_embeddings = self.encoder.base.get_input_embeddings()(gen.hard_token_ids)
                    current_token_ids_batch = input_ids.squeeze(0)[token_positions_batch]
                    A_hat_batch = self.encoder(
                        generated_token_embeddings, 
                        original_token_pos=token_positions_batch, 
                        current_token_ids=current_token_ids_batch if self.encoder.config.add_current_token else None
                    )

                    # Compute KL divergence in a batch using the vectorized forward pass.
                    # The input_ids tensor needs to be expanded to match the batch size of activations.
                    input_ids_batch = input_ids.expand(current_batch_size, -1)  # Shape: (current_batch_size, seq_len)

                    # Get original logits by replacing activations at each position.
                    logits_orig_full = self.orig_model.forward_with_replacement_vectorized(
                        input_ids=input_ids_batch,
                        new_activations=A_batch,
                        layer_idx=self.layer,
                        token_positions=token_positions_batch,
                        no_grad=True
                    ).logits

                    # Get reconstructed logits similarly.
                    logits_recon_full = self.orig_model.forward_with_replacement_vectorized(
                        input_ids=input_ids_batch,
                        new_activations=A_hat_batch,
                        layer_idx=self.layer,
                        token_positions=token_positions_batch,
                        no_grad=True
                    ).logits
                
                # Extract logits at the specific token positions for KL calculation.
                batch_indices = torch.arange(current_batch_size, device=self.device)
                logits_orig_at_pos = logits_orig_full[batch_indices, token_positions_batch]
                logits_recon_at_pos = logits_recon_full[batch_indices, token_positions_batch]
                
                # Compute KL with numerical stability (batched, but without AMP for precision).
                with torch.amp.autocast('cuda', enabled=False):
                    logits_orig_f32 = logits_orig_at_pos.float()
                    logits_recon_f32 = logits_recon_at_pos.float()
                    
                    # Normalize logits for stability.
                    logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0]
                    logits_recon_f32 = logits_recon_f32 - logits_recon_f32.max(dim=-1, keepdim=True)[0]
                    
                    log_probs_orig = torch.log_softmax(logits_orig_f32, dim=-1)
                    log_probs_recon = torch.log_softmax(logits_recon_f32, dim=-1)
                    probs_orig = torch.exp(log_probs_orig)
                    
                    # kl_divs_batch will have shape (current_batch_size,).
                    kl_divs_batch = (probs_orig * (log_probs_orig - log_probs_recon)).sum(dim=-1)
                    all_kl_divs.append(kl_divs_batch)

                # MSE for comparison (batched).
                # mses_batch will have shape (current_batch_size,).
                mses_batch = torch.nn.functional.mse_loss(A_batch, A_hat_batch, reduction='none').mean(dim=-1)
                all_mses.append(mses_batch)

                # Relative RMSE: sqrt(MSE(A, A_hat) / MSE(A, 0)). This is a scale-invariant error metric.
                # This is equivalent to ||A - A_hat|| / ||A|| (relative L2 error).
                A_norm_sq_mean = (A_batch**2).mean(dim=-1)
                relative_rmse_batch = torch.sqrt(mses_batch / (A_norm_sq_mean + 1e-9))
                all_relative_rmses.append(relative_rmse_batch)

        # Concatenate results from all batches
        kl_divs = torch.cat(all_kl_divs)
        mses = torch.cat(all_mses)
        relative_rmses = torch.cat(all_relative_rmses)
        gen_hard_token_ids = torch.cat(all_gen_hard_token_ids)

        # Decode and collect results into a list of dicts.
        results = []
        # The original code had a `gen` object, we now use the concatenated `gen_hard_token_ids`
        for pos in range(seq_len):
            explanation = self.tokenizer.decode(gen_hard_token_ids[pos], skip_special_tokens=True) + "[" + self.tokenizer.decode([input_ids[0, pos].item()]) +"]"
            token = self.tokenizer.decode([input_ids[0, pos].item()])
            
            results.append({
                'position': pos,
                'token': token,
                'explanation': explanation,
                'kl_divergence': kl_divs[pos].item(),
                'mse': mses[pos].item(),
                'relative_rmse': relative_rmses[pos].item()
            })
        
        return pd.DataFrame(results)

    def causal_intervention(self, 
                           original_text: str, 
                           intervention_position: int, 
                           intervention_string: str,
                           max_new_tokens: int = 20,
                           visualize: bool = True, top_k: int = 5, next_token_position: int = None) -> Dict[str, Any]:
        """
        Perform a causal intervention on a model's hidden state.

        This involves replacing the activation at a specific `intervention_position` with a new
        activation generated from an `intervention_string`, and then observing the effect on
        the model's predictions at `next_token_position`.
        """
        # Tokenize original text
        inputs = self.tokenizer(original_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        # Handle negative indices for positions
        if intervention_position < 0:
            intervention_position = seq_len + intervention_position
        
        if not 0 <= intervention_position < seq_len:
            raise ValueError(f"Intervention position {intervention_position} is out of bounds for sequence length {seq_len}")

        if next_token_position is None:
            next_token_position = intervention_position
        elif next_token_position < 0:
            next_token_position = seq_len + next_token_position
        
        if not 0 <= next_token_position < seq_len:
            raise ValueError(f"Next token position {next_token_position} is out of bounds for sequence length {seq_len}")
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
                # Get original hidden states
                outputs_orig = self.orig_model.model(input_ids, output_hidden_states=True)
                hidden_states_orig = outputs_orig.hidden_states
                
                # Get original activation (A) at intervention position
                A_orig = hidden_states_orig[self.layer+1][:, intervention_position, :]
                
                # Decode original activation to get its explanation
                if self.decoder.config.use_kv_cache:
                    gen_orig = self.decoder.generate_soft_kv_cached(A_orig, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=torch.tensor([intervention_position], device=self.device))
                else:
                    gen_orig = self.decoder.generate_soft(A_orig, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=torch.tensor([intervention_position], device=self.device))
                
                explanation_orig = self.tokenizer.decode(gen_orig.hard_token_ids[0], skip_special_tokens=True)
                
                # Prepare intervention string and get its embeddings
                intervention_tokens = self.tokenizer(intervention_string, return_tensors="pt")
                intervention_ids = intervention_tokens.input_ids.to(self.device)
                if intervention_ids.shape[-1] > self.t_text:
                    print(f"Truncating intervention string to {self.t_text} tokens.")
                    intervention_ids = intervention_ids[..., -self.t_text:]
                elif intervention_ids.shape[-1] < self.t_text:
                    # Pad intervention if it's shorter than the expected length
                    pad_width = self.t_text - intervention_ids.shape[-1]
                    intervention_ids = torch.cat([
                        torch.zeros(intervention_ids.shape[0], pad_width, dtype=intervention_ids.dtype, device=self.device), 
                        intervention_ids
                    ], dim=-1)

                new_intervention_string = self.tokenizer.decode(intervention_ids[0], skip_special_tokens=True)
                
                if hasattr(self.orig_model.model, 'transformer'):
                    embed_layer = self.orig_model.model.transformer.wte
                elif hasattr(self.orig_model.model, 'model'):
                    embed_layer = self.orig_model.model.model.embed_tokens
                else:
                    embed_layer = self.orig_model.model.get_input_embeddings()
                
                intervention_embeddings = embed_layer(intervention_ids)
                
                # Encode the intervention embeddings to get the new activation
                A_intervention = self.encoder(
                    intervention_embeddings,
                    original_token_pos=torch.tensor([intervention_position], device=self.device),
                    current_token_ids=input_ids[:, intervention_position] if self.encoder.config.add_current_token else None
                )
                
                # Also encode the original explanation to get the reconstructed activation (Â)
                A_orig_decoded = self.encoder(
                    gen_orig.generated_text_embeddings,
                    original_token_pos=torch.tensor([intervention_position], device=self.device),
                    current_token_ids=input_ids[:, intervention_position] if self.encoder.config.add_current_token else None
                )

                # Run model forward with original, intervened, and reconstructed activations
                # The `token_pos` argument specifies where to insert the activation.
                outputs_orig_full = self.orig_model.forward_with_replacement(
                    input_ids=input_ids, new_activation=A_orig, layer_idx=self.layer,
                    token_pos=intervention_position, no_grad=True
                )
                outputs_intervention = self.orig_model.forward_with_replacement(
                    input_ids=input_ids, new_activation=A_intervention, layer_idx=self.layer,
                    token_pos=intervention_position, no_grad=True
                )
                outputs_orig_decoded = self.orig_model.forward_with_replacement(
                    input_ids=input_ids, new_activation=A_orig_decoded, layer_idx=self.layer,
                    token_pos=intervention_position, no_grad=True
                )
                
                # Generate a continuation from the original, un-intervened state for comparison
                continuation_orig = self.orig_model.model.generate(
                    input_ids[:, :intervention_position + 1],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Get next-token predictions at the specified observation position
                next_token_logits_orig = outputs_orig_full.logits[:, next_token_position, :]
                next_token_logits_intervention = outputs_intervention.logits[:, next_token_position, :]
                next_token_logits_orig_decoded = outputs_orig_decoded.logits[:, next_token_position, :]

                # Get top-k predictions for each case
                top_k = max(10, top_k)
                probs_orig = torch.softmax(next_token_logits_orig, dim=-1)
                probs_intervention = torch.softmax(next_token_logits_intervention, dim=-1)
                probs_orig_decoded = torch.softmax(next_token_logits_orig_decoded, dim=-1)
                
                top_k_orig = torch.topk(probs_orig[0], k=top_k)
                top_k_intervention = torch.topk(probs_intervention[0], k=top_k)
                top_k_orig_decoded = torch.topk(probs_orig_decoded[0], k=top_k)
            
            # Decode top predictions
            top_tokens_orig = [(self.tokenizer.decode([idx]), prob.item()) 
                              for idx, prob in zip(top_k_orig.indices, top_k_orig.values)]
            top_tokens_intervention = [(self.tokenizer.decode([idx]), prob.item()) 
                                      for idx, prob in zip(top_k_intervention.indices, top_k_intervention.values)]
            top_tokens_orig_decoded = [(self.tokenizer.decode([idx]), prob.item())
                                       for idx, prob in zip(top_k_orig_decoded.indices, top_k_orig_decoded.values)]
            
            # Compute metrics: KL divergence on predictions, MSE on activations
            kl_div = torch.nn.functional.kl_div(
                torch.log_softmax(next_token_logits_intervention, dim=-1),
                torch.softmax(next_token_logits_orig, dim=-1),
                reduction='batchmean'
            ).item()
            mse = torch.nn.functional.mse_loss(A_orig, A_intervention).item()

            mse_orig_decoded = torch.nn.functional.mse_loss(A_orig, A_orig_decoded).item()
            kl_div_orig_decoded = torch.nn.functional.kl_div(
                torch.log_softmax(next_token_logits_orig_decoded, dim=-1),
                torch.softmax(next_token_logits_orig, dim=-1),
                reduction='batchmean'
            ).item()
            
            # Decode tokens at relevant positions for reporting
            intervened_token = self.tokenizer.decode([input_ids[0, intervention_position].item()])
            prediction_context_token = self.tokenizer.decode([input_ids[0, next_token_position].item()])
            
        results = {
            'original_text': original_text,
            'intervention_position': intervention_position,
            'intervened_token': intervened_token,
            'next_token_position': next_token_position,
            'prediction_context_token': prediction_context_token,
            'intervention_string': new_intervention_string,
            'explanation_original': explanation_orig,
            'top_predictions_original': top_tokens_orig,
            'top_predictions_intervention': top_tokens_intervention,
            'top_predictions_original_decoded': top_tokens_orig_decoded,
            'kl_divergence': kl_div,
            'mse': mse,
            'mse_orig_decoded': mse_orig_decoded,
            'kl_div_orig_decoded': kl_div_orig_decoded,
            'continuation_original': self.tokenizer.decode(continuation_orig[0], skip_special_tokens=True)
        }
        
        # Print results
        print("\n🔬 Causal Intervention Analysis")
        print(f"{'='*60}")
        print(f"Original text: '{original_text}'")
        print(f"Intervened at position {intervention_position}: '{intervened_token}'")
        if next_token_position != intervention_position:
            print(f"Observing predictions at position {next_token_position} (for token after '{prediction_context_token}')")
        
        current_token_str = f" + current token [{intervened_token}]" if self.encoder.config.add_current_token else ""
        print(f"Intervention string: '{new_intervention_string}'{current_token_str}")
        
        print(f"\n📝 Explanation (from activation at pos {intervention_position}):")
        print(f"  Original: {explanation_orig}")
        
        print(f"\n📊 Metrics (MSE at pos {intervention_position}, KL at pos {next_token_position}):")
        print(f"  MSE (intervention vs A): {mse:.4f}")
        print(f"  KL Divergence (intervention vs A): {kl_div:.4f}")
        print(f"  MSE (Â vs A): {mse_orig_decoded:.4f}")
        print(f"  KL Divergence (Â vs A): {kl_div_orig_decoded:.4f}")
        print(f"\n🎯 Top next-token predictions (at pos {next_token_position}):")
        print()

        # Format and print predictions in a 3-column table for easy comparison.
        headers = ["Original (A)", "With Intervention", "Reconstructed (Â)"]
        col_width = 28

        print(f"{headers[0]:<{col_width}}{headers[1]:<{col_width}}{headers[2]:<{col_width}}")
        separator = "=" * (col_width - 2)
        print(f"{separator:<{col_width}}{separator:<{col_width}}{separator:<{col_width}}")

        for (orig_tok, orig_prob), (inter_tok, inter_prob), (recon_tok, recon_prob) in zip(
            top_tokens_orig[:top_k],
            top_tokens_intervention[:top_k],
            top_tokens_orig_decoded[:top_k]
        ):
            orig_str = f"{repr(orig_tok):15} {orig_prob:.3f}"
            inter_str = f"{repr(inter_tok):15} {inter_prob:.3f}"
            recon_str = f"{repr(recon_tok):15} {recon_prob:.3f}"
            
            print(f"{orig_str:<{col_width}}{inter_str:<{col_width}}{recon_str:<{col_width}}")
        if visualize:
            self._visualize_intervention(results)
        
        return results
    def _visualize_intervention(self, results: Dict[str, Any]):
        """Visualize the intervention results, comparing Original, Reconstructed (Â), and Intervention."""
        fig = plt.figure(figsize=(20, 9))
        # Use GridSpec for a more flexible layout
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
        
        ax1 = fig.add_subplot(gs[:, 0]) # Main plot for distributions spans both rows
        ax2 = fig.add_subplot(gs[0, 1]) # KL plot
        ax3 = fig.add_subplot(gs[1, 1]) # MSE plot

        fig.suptitle(f"Causal Intervention Analysis for token: '{results['intervened_token']}'", fontsize=16, y=0.98)

        # --- Data preparation for probability plot ---
        top_k = 10
        
        preds_orig_map = dict(results['top_predictions_original'])
        preds_int_map = dict(results['top_predictions_intervention'])
        preds_orig_decoded_map = dict(results['top_predictions_original_decoded'])

        # Create a union of top-k tokens from each distribution for a comprehensive comparison
        tokens_orig_set = set(t[0] for t in results['top_predictions_original'][:top_k])
        tokens_int_set = set(t[0] for t in results['top_predictions_intervention'][:top_k])
        tokens_orig_decoded_set = set(t[0] for t in results['top_predictions_original_decoded'][:top_k])
        
        union_tokens = list(tokens_orig_set | tokens_int_set | tokens_orig_decoded_set)
        # Sort tokens by original probability for a meaningful order on the x-axis
        union_tokens.sort(key=lambda token: preds_orig_map.get(token, 0), reverse=True)
        plot_tokens = union_tokens[:15] # Limit to max 15 tokens for readability

        probs_orig = [preds_orig_map.get(token, 0) for token in plot_tokens]
        probs_int = [preds_int_map.get(token, 0) for token in plot_tokens]
        probs_orig_decoded = [preds_orig_decoded_map.get(token, 0) for token in plot_tokens]

        x = np.arange(len(plot_tokens))
        width = 0.25

        # --- Plot 1: Combined probability distribution ---
        ax1.bar(x - width, probs_orig, width, label='Original (A)', alpha=0.8, color='cornflowerblue')
        ax1.bar(x, probs_orig_decoded, width, label='Reconstructed (Â)', alpha=0.8, color='mediumseagreen')
        ax1.bar(x + width, probs_int, width, label='Intervention', alpha=0.8, color='salmon')
        
        ax1.set_ylabel('Probability')
        ax1.set_title('Next Token Prediction Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([repr(t) for t in plot_tokens], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: KL Divergence ---
        kl_labels = ['Â vs A', 'Intervention vs A']
        kl_values = [results['kl_div_orig_decoded'], results['kl_divergence']]
        kl_colors = ['mediumseagreen', 'salmon']
        ax2.bar(kl_labels, kl_values, color=kl_colors, alpha=0.8)
        ax2.set_ylabel('KL Divergence')
        ax2.set_title('Output Distribution Shift (KL)')
        ax2.grid(True, axis='y', alpha=0.3)
        # Add text labels above bars for clarity
        for i, v in enumerate(kl_values):
            ax2.text(i, v, f"{v:.4f}", ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

        # --- Plot 3: MSE ---
        mse_labels = ['Â vs A', 'Intervention vs A']
        mse_values = [results['mse_orig_decoded'], results['mse']]
        mse_colors = ['mediumseagreen', 'salmon']
        ax3.bar(mse_labels, mse_values, color=mse_colors, alpha=0.8)
        ax3.set_ylabel('Mean Squared Error')
        ax3.set_title('Activation Vector Distance (MSE)')
        ax3.grid(True, axis='y', alpha=0.3)
        # Add text labels above bars for clarity
        for i, v in enumerate(mse_values):
            ax3.text(i, v, f"{v:.4f}", ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def get_hidden_states(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.orig_model.model(input_ids, output_hidden_states=True)
        print("len hidden states", len(outputs.hidden_states))
        return outputs.hidden_states[self.layer + 1]

    def run_verbose_sample(self, text: str, position_to_analyze: int, top_n_analysis: int = 5, continuation_tokens: int = 30):
        """Runs a single verbose sample analysis using process_and_print_verbose_batch_samples."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        if position_to_analyze <0:
            position_to_analyze = seq_len + position_to_analyze

        if not (0 <= position_to_analyze < seq_len):
            raise ValueError(f"Position to analyze ({position_to_analyze}) is out of bounds for sequence length ({seq_len}).")

        # Get hidden state A
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
                outputs = self.orig_model.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                A_i = hidden_states[self.layer + 1][:, position_to_analyze, :].clone()

        # Prepare batch for verbose_samples function
        batch = {
            "A": A_i,
            "input_ids_A": input_ids, # Needs to be [batch_size, seq_len]
            "layer_idx": torch.tensor([self.layer], device=self.device),
            "token_pos_A": torch.tensor([position_to_analyze], device=self.device),
            "A_prime": A_i.clone(), # For single sample, A_prime can be A itself
            # Add dummy A_prime related fields if verbose_samples expects them even if not used meaningfully
            "input_ids_A_prime": input_ids.clone(), 
            "token_pos_A_prime": torch.tensor([position_to_analyze], device=self.device),
        }
        
        # Prepare sch_args
        loss_weights = self.config.get('loss_weights', {})
        sch_args = {
            "tau": self.tau,
            "alpha": self.config.get('alpha', 0.1), # Default if not in config
            "kl_base_weight": loss_weights.get('kl', 1.0),
            "entropy_weight": loss_weights.get('entropy', 0.0),
            "mse_weight": loss_weights.get('mse', 0.0),
            "lm_base_weight": loss_weights.get('lm', 0.0),
            "t_text": self.t_text,
            "GRPO_weight": self.config.get('GRPO_weight', 0.0),
            "GRPO_beta": self.config.get('GRPO_beta', 0.0),
            "group_n": self.config.get('group_n', 0),
        }

        # Models dictionary
        models_dict = {
            "dec": self.decoder,
            "enc": self.encoder,
            # process_and_print_verbose_batch_samples expects 'orig' inside models for compute_single_sample_losses
            "orig": self.orig_model 
        }

        # Cache tokenized natural language prefix if specified (all processes do this, small overhead)
        cached_prefix_ids = None
        lm_loss_natural_prefix_text = self.config.get('lm_loss_natural_prefix') # Assuming this key holds the text
        if lm_loss_natural_prefix_text and isinstance(lm_loss_natural_prefix_text, str) : # Check if it's a string
            cached_prefix_ids = self.tokenizer(lm_loss_natural_prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            print(f"Cached natural language prefix: '{lm_loss_natural_prefix_text}' ({cached_prefix_ids.shape[1]} tokens)")
        elif self.config.get('lm_loss_natural_prefix') is True: # Handle boolean true case if it implies a default prefix or other logic
            print("lm_loss_natural_prefix is True but not a string. Cannot cache prefix IDs without prefix text.")

        
        print(f"\n🔬 Running Verbose Sample Analysis for position {position_to_analyze} in text: \"{text[:100]}...\"")

        process_and_print_verbose_batch_samples(
            batch=batch,
            cfg=self.config,
            models=models_dict, # Pass the dict containing dec, enc, and orig
            orig=self.orig_model, # Also pass orig separately as it's used directly too
            tok=self.tokenizer,
            sch_args=sch_args,
            device=self.device,
            num_samples=1, # We are running for a single sample
            top_n_analysis=top_n_analysis,
            printed_count_so_far=0,
            generate_continuation=True,
            continuation_tokens=continuation_tokens,
            return_structured_data=False,
            capture_output=False,
            cached_prefix_ids=cached_prefix_ids, # Assuming no cached prefix for interactive use
            resample_ablation=self.config.get('resample_ablation', True), # Get from config or default
            comparison_tuned_lens=self.comparison_tuned_lens # Pass the loaded TunedLens
        )

def quick_analyze(text: str, show_plot: bool = True):
    """Quick analysis function with optional visualization."""
    df = analyzer.analyze_all_tokens(text)
    
    # Print summary
    print(f"\n📊 Analysis of: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"Tokens: {len(df)}")
    print(f"Avg KL: {df['kl_divergence'].mean():.3f} (±{df['kl_divergence'].std():.3f})")
    print(f"Range: [{df['kl_divergence'].min():.3f}, {df['kl_divergence'].max():.3f}]")
    
    # Show tokens with explanations
    print("\nToken-by-token breakdown:")
    for _, row in df.iterrows():
        kl = row['kl_divergence']
        mse = row['mse']
        # Color code based on KL value
        if kl < df['kl_divergence'].quantile(0.33):
            indicator = "🟢"
        elif kl < df['kl_divergence'].quantile(0.67):
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"{indicator} [{row['position']:2d}] {repr(row['token']):15} → {row['explanation']:40} (KL: {kl:.3f} MSE: {mse:.3f})")
    
    if show_plot and len(df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(df['position'], df['kl_divergence'], 'b-', linewidth=2, marker='o')
        plt.xlabel('Position')
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.title(f'KL Divergence: "{text[:40]}..."')
        plt.grid(True, alpha=0.3)
        
        # Annotate some points
        for i in range(0, len(df), max(1, len(df) // 5)):
            plt.annotate(repr(df.iloc[i]['token']), 
                        (df.iloc[i]['position'], df.iloc[i]['kl_divergence']),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=8)
        plt.ylim(None,max(df['kl_divergence'][1:])) 
        plt.tight_layout()
        plt.show()

    if show_plot and len(df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(df['position'], df['mse'], 'r-', linewidth=2, marker='o')
        plt.xlabel('Position')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.title(f'MSE: "{text[:40]}..."')
        plt.grid(True, alpha=0.3)   
        for i in range(0, len(df), max(1, len(df) // 5)):
            plt.annotate(repr(df.iloc[i]['token']), 
                        (df.iloc[i]['position'], df.iloc[i]['mse']),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=8)
        plt.ylim(None,max(df['mse'][1:])) 
        plt.tight_layout()
        plt.show()

    return df

# Initialize the analyzer
#analyzer = LensAnalyzer(CHECKPOINT_PATH, DEVICE)
analyzer = LensAnalyzer(CHECKPOINT_PATH, DEVICE, do_not_load_weights=False, make_xl=False,  use_bf16=True)

# %% [markdown]
# ## Single String Analysis
# 
# Analyze a single string and visualize the results:

# %%
# Analyze a test string
TEST_STRING = "The cat sat on the mat."
TEST_STRING = "<|endoftext|>No less than three sources (who wish to remain anonymous) have indicated to PPP that the Leafs are close to a contract extension with goaltender Jonas Gustavsson that will keep him in Toronto for the foreseeable future. I'm sort of at a loss for words. I mean, Gustavsson did have that one month where he wasn't hot"
TEST_STRING = "A paper butterfly is displayed on a coffin during Asia Funeral Expo (AFE) in Hong Kong May 19, 2011. REUTERS/Bobby Yip\n\nLONDON (Reuters) - A sharp increase in the number of deaths in Britain in 2015 compared with the previous year helped funeral services firm Dignity post a 16-*percent* profit increase, the firm said on Wednesday.\n\nDignity said the 7-percent year-on-year rise in the number of deaths..."

TEST_STRING = """This photo shows an image of a cougar in Wyoming. Animal control officers in Fairfax County have set up cameras this week in hopes of capturing the image of a cougar reported in the Alexandria section of the county. (Neil Wight/AP)\n\nMultiple sightings of what animal control officials are calling "possibly a cougar" have authorities in Fairfax County on high alert.\n\nOfficials received two reports this week of a "large cat" with an orange"""
# TEST_STRING = "India fearing Chinese submarines is not unwarranted. In 2014, the Chinese Navy frequently docked its submarines in Sri Lanka, saying they were goodwill visits and to replenish ships on deployment to the Arabian Sea.\n\nLater, in 2015, the Chinese Navy, or the People's Liberation Army Navy (PLAN), was seen docking in vessels Pakistan. The frequent visits and the constant reporting on them has led Indian Navy to get the best aircraft in its inventory, P-8I"
TEST_STRING = "<|endoftext|>On a third-down in 11-on-11 scrimmage, he zoomed past starting left tackle Jake Matthews and sacked quarterback Matt Ryan. Well, he tagged him down, since they don't tackle to the ground anymore in NFL practices.\n\nBut that's a practice sack and the Falcons are hoping their first-round pick, who has recovered from offseason shoulder surgery, has plenty of real sacks in his 6-foot, 2-inch and 250-pound"
TEST_STRING = "Israel Accused of Suppressing Terror Evidence to Help Out New Pal China\n\nIsrael is a country desperate for friends. Isolated in the Middle East and hated in large parts of the Arab world, it struggles to make alliances. The few it has, it guards fiercely. So it should perhaps come as no surprise that for years Israel has been courting China, inking trade deals and fêting one another over champagne. But that process now finds Israel in an awkward bind"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%
torch.manual_seed(47)   
analyzer.run_verbose_sample(TEST_STRING, 66, continuation_tokens=100)
# %%
# Visualize KL divergence across positions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# KL divergence plot
positions = df['position'].values
kl_values = df['kl_divergence'].values
tokens = df['token'].values

ax1.plot(positions, kl_values, 'b-', linewidth=2, marker='o', markersize=8)
ax1.set_ylabel('KL Divergence', fontsize=12)
ax1.set_title(f'KL Divergence Across Token Positions\n"{TEST_STRING}"', fontsize=14)
ax1.set_yscale('log')
ax1.set_ylim(None, max(kl_values[1:]))
ax1.grid(True, alpha=0.3)

# Add token labels
for i, (pos, kl, token) in enumerate(zip(positions, kl_values, tokens)):
    if i % max(1, len(positions) // 10) == 0:  # Show ~10 labels
        ax1.annotate(repr(token), (pos, kl), 
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=8, alpha=0.7)

# MSE plot for comparison
mse_values = df['mse'].values
ax2.plot(positions, mse_values, 'r-', linewidth=2, marker='s', markersize=6)
ax2.set_xlabel('Token Position', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)
ax2.set_title('Reconstruction MSE', fontsize=12)
ax2.set_yscale('log')
ax2.set_ylim(None, max(mse_values[1:]))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Show best and worst reconstructed tokens
print("\n🟢 Best reconstructed tokens (lowest KL):")
print(df.nsmallest(3, 'kl_divergence')[['position', 'token', 'explanation', 'kl_divergence']])

print("\n🔴 Worst reconstructed tokens (highest KL):")
print(df.nlargest(3, 'kl_divergence')[['position', 'token', 'explanation', 'kl_divergence']])

# %% [markdown]
# ## Batch Analysis
# 
# Analyze multiple strings and compare their patterns:

# # %%
# # Define strings to compare
# STRINGS_TO_COMPARE = [
#     "The cat sat on the mat.",
#     "The dog ran in the park.",
#     "Machine learning is powerful.",
#     "Hello world!",
#     "import numpy as np",
#     "def hello():\n    print('Hi')"
# ]

# # Analyze each string
# all_results = []
# summary_stats = []

# for text in STRINGS_TO_COMPARE:
#     df = analyzer.analyze_all_tokens(text)
#     df['text'] = text  # Add text column
#     all_results.append(df)
    
#     # Compute summary statistics
#     summary_stats.append({
#         'text': text[:50] + '...' if len(text) > 50 else text,
#         'num_tokens': len(df),
#         'avg_kl': df['kl_divergence'].mean(),
#         'std_kl': df['kl_divergence'].std(),
#         'max_kl': df['kl_divergence'].max(),
#         'min_kl': df['kl_divergence'].min()
#     })

# summary_df = pd.DataFrame(summary_stats)
# print("Summary Statistics:")
# print(summary_df.to_string(index=False))

# # %%
# # Visualize comparison
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plot KL trajectories for each string
# for i, (text, df) in enumerate(zip(STRINGS_TO_COMPARE, all_results)):
#     label = text[:30] + '...' if len(text) > 30 else text
#     ax.plot(df['position'], df['kl_divergence'], 
#             marker='o', label=label, alpha=0.7, linewidth=2)

# ax.set_xlabel('Token Position', fontsize=12)
# ax.set_ylabel('KL Divergence', fontsize=12)
# ax.set_title('KL Divergence Comparison Across Different Texts', fontsize=14)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # %%
# # Create a heatmap of explanations length vs KL divergence
# combined_df = pd.concat(all_results, ignore_index=True)
# combined_df['explanation_length'] = combined_df['explanation'].str.len()

# # Bin the data for heatmap
# kl_bins = pd.qcut(combined_df['kl_divergence'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
# len_bins = pd.qcut(combined_df['explanation_length'], q=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])

# # Create crosstab
# heatmap_data = pd.crosstab(len_bins, kl_bins)

# plt.figure(figsize=(10, 6))
# sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
# plt.title('Relationship between Explanation Length and KL Divergence')
# plt.xlabel('KL Divergence Category')
# plt.ylabel('Explanation Length Category')
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ## Interactive Analysis Function
# 
# Use this function to quickly analyze any string:

# %%


# %%
# Try it out!
quick_analyze("The quick brown cat jumps over the lazy mouse.\n"*10)

# %%
# Analyze your own text
YOUR_TEXT = "When we get to 2028 and AGI hasn't yet been achieved and LLMs are causing psychiatric problems and undermining democracy and cybercrime is an epidemic, insidious microtargeted ads are everywhere, nobody trusts anything, colleges no longer work, and climate change is worse,  will people look back at the choices we made in 2025 fondly?"  # <-- Change this
df_custom = quick_analyze(YOUR_TEXT)

# %% [markdown]
# ## Statistical Analysis
# 
# Let's look at patterns in the explanations:

# %%
# # Combine all results for statistical analysis
# if all_results:
#     combined_df = pd.concat(all_results, ignore_index=True)
    
#     # Most common explanation words
#     from collections import Counter
#     all_explanations = ' '.join(combined_df['explanation'].values)
#     word_counts = Counter(all_explanations.split())
    
#     print("Most common words in explanations:")
#     for word, count in word_counts.most_common(20):
#         print(f"  {word:15} : {count:4d}")
    
#     # Correlation between token length and KL
#     combined_df['token_length'] = combined_df['token'].str.len()
#     correlation = combined_df[['token_length', 'kl_divergence']].corr().iloc[0, 1]
#     print(f"\nCorrelation between token length and KL divergence: {correlation:.3f}")

# # %%
# # Distribution plots
# if all_results:
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     # KL distribution
#     axes[0, 0].hist(combined_df['kl_divergence'], bins=30, alpha=0.7, color='blue')
#     axes[0, 0].set_xlabel('KL Divergence')
#     axes[0, 0].set_ylabel('Count')
#     axes[0, 0].set_title('Distribution of KL Divergence')
    
#     # MSE distribution
#     axes[0, 1].hist(combined_df['mse'], bins=30, alpha=0.7, color='red')
#     axes[0, 1].set_xlabel('MSE')
#     axes[0, 1].set_ylabel('Count')
#     axes[0, 1].set_title('Distribution of MSE')
    
#     # KL vs MSE scatter
#     axes[1, 0].scatter(combined_df['mse'], combined_df['kl_divergence'], alpha=0.5)
#     axes[1, 0].set_xlabel('MSE')
#     axes[1, 0].set_ylabel('KL Divergence')
#     axes[1, 0].set_title('KL vs MSE Relationship')
    
#     # Explanation length distribution
#     axes[1, 1].hist(combined_df['explanation'].str.len(), bins=20, alpha=0.7, color='green')
#     axes[1, 1].set_xlabel('Explanation Length (chars)')
#     axes[1, 1].set_ylabel('Count')
#     axes[1, 1].set_title('Distribution of Explanation Lengths')
    
#     plt.tight_layout()
#     plt.show()

# %% [markdown]
# ## Save Results
# 
# Save your analysis results for later use:

# # %%
# # Save to CSV
# if all_results:
#     output_path = "lens_analysis_results.csv"
#     combined_df.to_csv(output_path, index=False)
#     print(f"✓ Saved {len(combined_df)} rows to {output_path}")

# %%
# Create a summary report
def create_summary_report(results_df: pd.DataFrame, text: str) -> str:
    """Create a text summary report of the analysis."""
    report = f"""
Consistency Lens Analysis Report
================================
Text: "{text[:100]}{'...' if len(text) > 100 else ''}"
Checkpoint: {analyzer.checkpoint_path.name}
Model: {analyzer.model_name}
Layer: {analyzer.layer}

Summary Statistics:
- Total tokens: {len(results_df)}
- Average KL divergence: {results_df['kl_divergence'].mean():.4f}
- Std deviation: {results_df['kl_divergence'].std():.4f}
- Min KL: {results_df['kl_divergence'].min():.4f}
- Max KL: {results_df['kl_divergence'].max():.4f}

Top 3 most consistent tokens (lowest KL):
"""
    for _, row in results_df.nsmallest(3, 'kl_divergence').iterrows():
        report += f"  - [{row['position']}] '{row['token']}' → '{row['explanation']}' (KL: {row['kl_divergence']:.4f})\n"
    
    report += "\nTop 3 least consistent tokens (highest KL):\n"
    for _, row in results_df.nlargest(3, 'kl_divergence').iterrows():
        report += f"  - [{row['position']}] '{row['token']}' → '{row['explanation']}' (KL: {row['kl_divergence']:.4f})\n"
    
    return report

# # Generate report for the test string
# if 'df' in locals():
#     report = create_summary_report(df, TEST_STRING)
#     print(report)
    
#     # Save to file
#     with open("lens_analysis_report.txt", "w") as f:
#         f.write(report)
#     print("\n✓ Report saved to lens_analysis_report.txt")

# %% [markdown]
# ## Next Steps
# 
# 1. Update `CHECKPOINT_PATH` to point to your trained model
# 2. Modify `TEST_STRING` and `STRINGS_TO_COMPARE` with your own text
# 3. Run cells to analyze and visualize results
# 4. Use `quick_analyze()` for interactive exploration
# 5. Save interesting results using the provided functions
# 
# Happy analyzing! 🔍
# %%
TEST_STRING = " t h e   c a t   s a t   o n   t h e   m a t"
TEST_STRING = " t h e   c a t   s a t   o n   t h e   m a t\n\n the cat sat on the"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

# %%
TEST_STRING = "John met Mary. He gave her a book. She thanked him."
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))


# %%
TEST_STRING = " a b c d e f g h i j k l n o p q r"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%
TEST_STRING = "John was a wrestler. He met Mary. He gave her a book. She thanked him. She said: \"John's job is a"
TEST_STRING = "Richard was a wrestler. He met Mary. He gave her a book. She thanked him. She said: \"Richard's job is a"
TEST_STRING = "Richard was a wrestler. He met Sophia. He gave her a book. She thanked him. She said: \"Richard's job is a"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))
# %%
TEST_STRING = "French: monsieur, votre chien est un problème\nEnglish: Mister, your dog is a problem.\n\nFrench: je suis un chat\nEnglish: I am a cat\n\nFrench: Ou est la gare? English: Where is the station? \n\n French: Tu n'aime pas les chats? English:"
#TEST_STRING = "French: chien\n English: dog\n\nFrench: chat\n English: cat\n\nFrench: gare\n English: station\n\nFrench: voiture\nEnglish: car\n\nFrench: maison\nEnglish:"
encodestring = analyzer.tokenizer.encode(TEST_STRING)
# generateddirectly = analyzer.model.generate(torch.tensor([encodestring]).to(analyzer.device), max_new_tokens=100)
# print(generateddirectly)
# tokenizerstring = analyzer.tokenizer.decode(generateddirectly[0])
# print(tokenizerstring)

# %%
TEST_STRING = "The door was closed. Alice opened it. Now the door is open.\n\nThe door was closed. Bob closed it. Now the door is closed.\n\nThe door was open. Carol shut it. Now the door is shut.\n\nThe door was open. David closed it. Now the door is"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

TEST_STRING = "Looking to leave a two-fight winless streak behind him, Manny Gamburyan will return to the UFC octagon on Sept. 27 in Las Vegas at UFC 178 in a new weight class.\n\nGamburyan is set to make his 135-pound men's bantamweight debut against Cody \"The Renegade\" Gibson at the MGM Grand Arena on a pay-per-view card headlined by the UFC light heavyweight title fight between champion Jon Jones and Daniel Cormierr"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

# %% [markdown]
# ## Causal Intervention Analysis
# 
# Perform causal interventions by encoding alternate strings and swapping them into the forward pass:


# %%
# Example 2: Conceptual intervention
original = "The scientist discovered a new element. This was"
intervention_result = analyzer.causal_intervention(
    original_text=original,
    intervention_position=8,  # Position of "This"
    intervention_string="The failure",
    max_new_tokens=15
)

# %%
# Example 3: Language intervention
original = "Bonjour means hello in French. Merci means"
intervention_result = analyzer.causal_intervention(
    original_text=original,
    intervention_position=10,  # Position of "Merci"
    intervention_string="Gracias",
    max_new_tokens=10
)

# %%
# Interactive intervention - customize these!
YOUR_TEXT = "The door was open. She closed it. Now the door is"
YOUR_POSITION = 12  # Token position to intervene at
YOUR_INTERVENTION = "He opened"  # What to encode and swap in

result = analyzer.causal_intervention(
    original_text=YOUR_TEXT,
    intervention_position=YOUR_POSITION,
    intervention_string=YOUR_INTERVENTION
)

# %%
# Interactive intervention - customize these!
YOUR_TEXT = "0 1 2 3 4 5 6 7 8 9\n\n A B C D E F G H I J\n\n0 1 2 3 4 5"
YOUR_POSITION = 28  # Token position to intervene at
YOUR_INTERVENTION = "Pokemon weights weights Mathematics games 5 5 integers elementary integers Mathematics games monsters 5 Mathematics partitions babies fights elementary integers elementary poems grades 5 exercises 5obbies 5 5 newsletters 5 lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "Pokemon letters letters English games E E letters elementary letters Letters games monsters E English partitions babies fights elementary characters elementary poems grades E exercises Eobbies E E newsletters E lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "Pokemon letters letters English games E E letters elementary letters Letters games monsters E English partitions babies fights elementary characters elementary poems grades E exercises Eobbies E E newsletters E lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "Pokemon weights weights Mathematics games 8 8 integers elementary integers Mathematics games monsters 8 Mathematics partitions babies fights elementary integers elementary poems grades 8 exercises 8obbies 8 8 newsletters 8 lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "Pokemon weights weights Mathematics games H H integers elementary integers Mathematics games monsters H Mathematics partitions babies fights elementary integers elementary poems grades H exercises Hobbies H H newsletters H lifestyle"  # What to encode and swap in

result = analyzer.causal_intervention(
    original_text=YOUR_TEXT,
    intervention_position=YOUR_POSITION,
    intervention_string=YOUR_INTERVENTION
)
# %%

# %%
# Interactive intervention - customize these!
YOUR_TEXT = "0 1 2 3 4 5 6 7 8 9"
YOUR_POSITION = 9 # Token position to intervene at
YOUR_INTERVENTION = "Pokemon weights weights Mathematics games 5 5 integers elementary integers Mathematics games monsters 5 Mathematics partitions babies fights elementary integers elementary poems grades 5 exercises 5obbies 5 5 newsletters 5 lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "Pokemon letters letters English games E E letters elementary letters Letters games monsters E English partitions babies fights elementary characters elementary poems grades E exercises Eobbies E E newsletters E lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "Pokemon letters letters English games E E letters elementary letters Letters games monsters E English partitions babies fights elementary characters elementary poems grades E exercises Eobbies E E newsletters E lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "letters counting enumeration javascript progrms Index letters E counting enumeration English elementary characters alphabet letters E counting enumeration E English elementary characters E lifestyle"
YOUR_INTERVENTION = "javascript programs exercises javascript javascript workouts javascript maps javascript courses javascript exercises javascript gamesIndex languages courses concatenationPages Walls G G pancakes G calculipes G correspondence papers G promotions speeches"
#YOUR_INTERVENTION = "javascript programs exercises javascript javascript workouts javascript maps javascript courses javascript exercises javascript gamesIndex languages courses multiplicationPages Walls 9 9 pancakes 9 calculipes 9 correspondence papers 9 promotions speeches"
# YOUR_INTERVENTION = "Pokemon weights weights Mathematics games 8 8 integers elementary integers Mathematics games monsters 8 Mathematics partitions babies fights elementary integers elementary poems grades 8 exercises 8obbies 8 8 newsletters 8 lifestyle"  # What to encode and swap in
# YOUR_INTERVENTION = "Pokemon weights weights Mathematics games H H integers elementary integers Mathematics games monsters H Mathematics partitions babies fights elementary integers elementary poems grades H exercises Hobbies H H newsletters H lifestyle"  # What to encode and swap in

result = analyzer.causal_intervention(
    original_text=YOUR_TEXT,
    intervention_position=YOUR_POSITION,
    intervention_string=YOUR_INTERVENTION
)
#
# %%



# #ialize the analyzer
# CHECKPOINT_PATH="""/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr3e-4_t32_20ep_resume_0609_2327_tinyLMKVext_dist4/"""
# analyzer = LensAnalyzer(CHECKPOINT_PATH, DEVICE)

# %% [markdown]
# ## Single String Analysis
# 
# Analyze a single string and visualize the results:

# %%
# Analyze a test string
TEST_STRING = "The cat sat on the mat."
TEST_STRING = ""
TEST_STRING = "A paper butterfly is displayed on a coffin during Asia Funeral Expo (AFE) in Hong Kong May 19, 2011. REUTERS/Bobby Yip\n\nLONDON (Reuters) - A sharp increase in the number of deaths in Britain in 2015 compared with the previous year helped funeral services firm Dignity post a 16-*percent* profit increase, the firm said on Wednesday.\n\nDignity said the 7-percent year-on-year rise in the number of deaths..."

TEST_STRING = """This photo shows an image of a cougar in Wyoming. Animal control officers in Fairfax County have set up cameras this week in hopes of capturing the image of a cougar reported in the Alexandria section of the county. (Neil Wight/AP)\n\nMultiple sightings of what animal control officials are calling "possibly a cougar" have authorities in Fairfax County on high alert.\n\nOfficials received two reports this week of a "large cat" with an orange"""
TEST_STRING = """Winter isn't done with us yet.\n\nOttawa can expect another 10 to 15 centimetres of snow Wednesday as a storm system moves through the United States today.\n\nWatch CBC Ottawa Go to Ian Black's weather page and follow his forecasts on TV on CBC News Ottawa starting at 5.\n\nEnvironment Canada has issued a special weather statement for much of Ontario, as a mixture of rain and snow is expected along Lake Ontario and Lake Erie and snow is expected further north..."""
# TEST_STRING = "India fearing Chinese submarines is not unwarranted. In 2014, the Chinese Navy frequently docked its submarines in Sri Lanka, saying they were goodwill visits and to replenish ships on deployment to the Arabian Sea.\n\nLater, in 2015, the Chinese Navy, or the People's Liberation Army Navy (PLAN), was seen docking in vessels Pakistan. The frequent visits and the constant reporting on them has led Indian Navy to get the best aircraft in its inventory, P-8I"
df = analyzer.analyze_all_tokens(TEST_STRING)

# %%
print(df.to_string(index=False))

# %%

#these!
YOUR_TEXT = "15 14 13 12 11 10 9 8 7 6 5"
YOUR_POSITION = 10  # Token position to intervene at
YOUR_INTERVENTION = "Pokemon weights weights Mathematics games 5 5 integers elementary integers Mathematics games monsters 5 Mathematics partitions babies fights elementary integers elementary poems grades 5 exercises 5obbies 5 5 newsletters 5 lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "Pokemon letters letters English games E E letters elementary letters Letters games monsters E English partitions babies fights elementary characters elementary poems grades E exercises Eobbies E E newsletters E lifestyle"  # What to encode and swap in
YOUR_INTERVENTION = "JAVASCRIPT ALPHABET++ Javascript combos calculations Pokemon programs Eggs OP scroll Spells scrolls scrolls Items scrolls scrolls scrolls bullets++ OP letters GG bee grams G buttons G arguments butterfly G"

result = analyzer.causal_intervention(
    original_text=YOUR_TEXT,
    intervention_position=YOUR_POSITION,
    intervention_string=YOUR_INTERVENTION
)
# %%

# %%
# Example 1: Simple pronoun intervention
#original = "<|endoftext|>No less than three sources (who wish to remain anonymous) have indicated to PPP that the Leafs are close to a contract extension with goaltender Jonas Gustavsson that will keep him in Toronto for the foreseeable future. I'm sort of at a loss for words. I mean, Gustavsson did have that one month where he wasn't hot\n\nThe woman was sitting on the plane. She worked as a"
#original = "The man was sitting on the plane. He worked as a"
original = "The woman was sitting on the plane. She worked as a"
intervention_result = analyzer.causal_intervention(
    original_text=original,
    intervention_position=-1,  # Position of "It"# original   Original: BeingFriendassed cookingfashionbeingWomanJoined accommodationBeer food bodily as accommodationfood vegetarianwife cooking profession waiter profession massage artisan workshop assistance prayer knitting cooldown tutorial currency encouragement education
    #intervention_string="BeingFriendassed cookinggunsbeingManJoined Beer gun bodily as weapon dog killhusband jump profession enlist profession fitness run workshop leader prayer soldier cooldown tutorial currency confrontation employment",
    #intervention_string="1932EEEE gibastoner Eston fighter Karl rooster Alex gunner Damien tactic fighter Arthur employee Alec lord tact Thor actor Sergeant Esp male assistant Umhab Sit accommodation contractor worked Sty Abedin helper Excelwork Sty um coping employees accommodation coping working man accommodationWork mas man assistants workspace weightlifting Webs weightliftingAmazon boxersafaAmazonBatman",
    #intervention_string=" she had tall hair",
    #intervention_string=" engineer engineer worked as",
    intervention_string=" spy spy worked as",


    # intervention_string="Joined cookingexistent BeerFriend utilizedStanding likeness culinarythingBir occupation resembling homebrewbeing homebrewemployed as prostitution waiter a workshop assistance workshop payroll transaction simulator education supervision education training guidance",
    max_new_tokens=5,
    top_k=20
)

# %%
Avecs = analyzer.get_hidden_states("The cat sat on the mat.")
print(Avecs.shape)
vars = torch.var(Avecs, dim=2)
print(vars.shape)
print(vars[:10])
print(Avecs[0,0:10])
print(torch.var(Avecs[0,-1], dim=0))
# %%
analyzer.run_verbose_sample("<|endoftext|>The woman was sitting on the plane. She worked as a", position_to_analyze=12)

# %%

checkpoint_path = Path(CHECKPOINT_PATH)
if checkpoint_path.is_dir():
    pt_files = sorted(checkpoint_path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in directory {checkpoint_path}")
    checkpoint_path = pt_files[0]
    print(f"Selected most recent checkpoint: {checkpoint_path}")

print(f"Loading checkpoint from {checkpoint_path}...")
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False) 
# %%
print(ckpt['models']['decoder'].keys())
print(ckpt['models']['decoder']['proj_weight'].shape)
projweights=ckpt['models']['decoder']['proj_weight']
print(torch.max(projweights))
print(torch.min(projweights))
print(torch.mean(projweights))
print(torch.std(projweights))
print(torch.var(projweights))
print(torch.max(projweights))
print(torch.min(projweights))
print([torch.max(t) for t in projweights])
print([torch.min(t) for t in projweights])
# %%
projbias=ckpt['models']['decoder']['proj_bias']
print(torch.max(projbias))
print(torch.min(projbias))
print(torch.mean(projbias))
print(torch.std(projbias))
print(torch.var(projbias))
print(torch.max(projbias))
print(torch.min(projbias))
print([torch.max(t) for t in projbias]) 
print([torch.min(t) for t in projbias])
# %%
posembedder=ckpt['models']['decoder']['activation_pos_embedder.weight']
print(posembedder.shape)
print(torch.tensor([torch.max(torch.abs(p)) for p in posembedder[0:128]]))
# %%
# %%

TEST_STRING = " t h e   c a t   s a t   o n   t h e   m a t\n the cat sat on the mat\n\n t h e   d o g   a t e   t h e   f o o d\n\nthe dog ate the food\n\nt h e   q u e e n   k i l l e d   t h e   j o k e r\n\nthe queen killed the joker"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%
analyzer.run_verbose_sample(TEST_STRING, position_to_analyze=103)

# %%
TEST_STRING = " t h e   c a t   s a t   o n   t h e   m a t\n the cat sat on the mat\n\n t h e   d o g   a t e   t h e   f o o d\n\nthe dog ate the food\n\nt h e   q u e e n   k i l l e d   t h e   j o k e r\n\nthe queen killed the"
encodestring = analyzer.tokenizer.encode(TEST_STRING)
listouts = []
for i in range(10):
    generateddirectly = analyzer.orig_model.model.generate(torch.tensor([encodestring]).to(analyzer.device),attention_mask=torch.ones_like(torch.tensor([encodestring], device=analyzer.device), dtype=torch.bool), max_new_tokens=100, do_sample=True)
    tokenizerstring = analyzer.tokenizer.decode(generateddirectly[0][-100:]).replace("\n","\\n")
    listouts.append(tokenizerstring)

# %%
for l in listouts:
    print(l)
    print("-"*100+"\n")
# %%
TEST_STRING = "Le chat est noir. = The cat is black.\n\nJ'aime les pommes. = I like apples.\n\nElle lit un livre. = She reads a book.\n\nNous allons au parc. = We go to the park.\n\nIl fait beau aujourd'hui. = It's nice weather today.\n\nTu as un chien? = Do you have a dog?\n\nLes enfants jouent dehors. = The children play outside.\n\nJe mange du pain. = I eat bread.\n\nMa maison est grande. = My house is big.\n\nIls regardent la télé. = They watch TV.\n\nElle porte une robe rouge. = She wears a red dress.\n\nLe garçon court vite. = The boy runs fast.\n\nNous buvons de l'eau. = We drink water."
df = analyzer.analyze_all_tokens(TEST_STRING)

# %%
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

TEST_STRING = (
    "<|endoftext|>c a t = cat\n\n"
    "d o g = dog\n\n"
    "b i r d = bird\n\n"
    "f i s h = fish\n\n"
    "h o r s e = horse\n\n"
    "m o u s e = mouse\n\n"
    "e l e p h a n t = elephant\n\n"
    "t i g e r = tiger\n\n"
    "l i o n = lion\n\n"
    "b e a r = bear\n\n"
    "w o l f = wolf\n\n"
    "f o x ="
    # "d e e r = deer\n\n"
    # "g o a t = goat\n\n"
    # "s h e e p = sheep\n\n"
    # "z e b r a = zebra\n\n"
    # "g i r a f f e = giraffe\n\n"
    # "m o n k e y = monkey\n\n"
    # "p a n d a =\n\n"
)


TEST_STRING = (
    "Alabama = Montgomery\n\n"
    "Alaska = Juneau\n\n"
    "Arizona = Phoenix\n\n"
    "Arkansas = Little Rock\n\n"
    "California = Sacramento\n\n"
    "Colorado = Denver\n\n"
    "Connecticut = Hartford\n\n"
    "Delaware = Dover\n\n"
    "Florida = Tallahassee\n\n"
    "Georgia = Atlanta\n\n"
    "Hawaii = Honolulu\n\n"
    "Idaho = Boise\n\n"
    "Illinois = Springfield\n\n"
    "Indiana = Indianapolis\n\n"
    "Iowa = Des Moines\n\n"
    "Kansas = Topeka\n\n"
    "Kentucky = Frankfort\n\n"
    "Louisiana = Baton Rouge\n\n"
    "Maine = Augusta\n\n"
    "Maryland = Annapolis\n\n"
    "Massachusetts = Boston\n\n"
    "Michigan = Lansing\n\n"
    # "Minnesota = Saint Paul\n\n"
    # "Mississippi = Jackson\n\n"
    # "Missouri = Jefferson City\n\n"
    # "Montana = Helena\n\n"
    # "Nebraska = Lincoln\n\n"
    # "Nevada = Carson City\n\n"
    # "New Hampshire = Concord\n\n"
    # "New Jersey = Trenton\n\n"
    # "New Mexico = Santa Fe\n\n"
    # "New York = Albany\n\n"
    # "North Carolina = Raleigh\n\n"
    # "North Dakota = Bismarck\n\n"
    # "Ohio = Columbus\n\n"
    # "Oklahoma = Oklahoma City\n\n"
    # "Oregon = Salem\n\n"
    # "Pennsylvania = Harrisburg\n\n"
    # "Rhode Island = Providence\n\n"
    # "South Carolina = Columbia\n\n"
    # "South Dakota = Pierre\n\n"
    # "Tennessee = Nashville\n\n"
    # "Texas = Austin\n\n"
    # "Utah = Salt Lake City\n\n"
    # "Vermont = Montpelier\n\n"
    # "Virginia = Richmond\n\n"
    # "Washington = Olympia\n\n"
    # "West Virginia = Charleston\n\n"
    # "Wisconsin = Madison\n\n"
    # "Wyoming = Cheyenne"
)


TEST_STRING = (
    "The capital of Alabama is Montgomery\n\n"
    "The capital of Alaska is Juneau\n\n"
    "The capital of Arizona is Phoenix\n\n"
    "The capital of Arkansas is Little Rock\n\n"
    "The capital of California is Sacramento\n\n"
    "The capital of Colorado is Denver\n\n"
    "The capital of Connecticut is Hartford\n\n"
    "The capital of Delaware is Dover\n\n"
    "The capital of Florida is Tallahassee\n\n"
    "The capital of Georgia is Atlanta\n\n"
    "The capital of Hawaii is Honolulu\n\n"
    "The capital of Idaho is Boise\n\n"
    "The capital of Illinois is Springfield\n\n"
    "The capital of Indiana is Indianapolis\n\n"
    "The capital of Iowa is Des Moines\n\n"
    "The capital of Kansas is Topeka\n\n"
    "The capital of Kentucky is Frankfort\n\n"
    "The capital of Louisiana is Baton Rouge\n\n"
    "The capital of Maine is Augusta\n\n"
    "The capital of Maryland is Annapolis\n\n"
    "The capital of Massachusetts is Boston\n\n"
    "The capital of Michigan is Lansing\n\n"
)

# %%

encodestring = analyzer.tokenizer.encode(TEST_STRING)
n_tok= 30
for i in range(10): 
    generateddirectly = analyzer.orig_model.model.generate(torch.tensor([encodestring]).to(analyzer.device), attention_mask=torch.ones_like(torch.tensor([encodestring], device=analyzer.device), dtype=torch.bool), max_new_tokens=n_tok)
    print(f"{i}", analyzer.tokenizer.decode(generateddirectly[0][-n_tok:]).replace("\n","\\n"))
# %%

df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

TEST_STRING = "On January 6, 2021, the United States Capitol in Washington, D.C., was attacked by a mob of supporters of President Donald Trump in an attempted self-coup, two months after his defeat in the 2020 presidential election. They sought to keep him in power by preventing a joint session of Congress from counting the Electoral College votes to formalize the victory of the president-elect Joe Biden. The attack was unsuccessful in preventing the certification of the election results. According to the bipartisan House select committee that investigated the incident, the attack was the culmination of a plan by Trump to overturn the election. Within 36 hours, five people died: one was shot by the Capitol Police, another died of a drug overdose, and three died of natural causes, including a police officer who died of a stroke a day after being assaulted by rioters and collapsing at the Capitol. Many people were injured, including 174 police officers. Four officers who responded to the attack died by suicide within seven months. Damage caused by attackers exceeded $2.7 million."

df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%
TEST_STRING = (
    "Thank you for checking out the project!\n"
    "Thanks so much for your support.\n"
    "Thank you for taking the time to visit.\n"
    "Thanks for reading and for your comments!\n"
    "Thank you for sharing this with your friends.\n"
    "Thanks for supporting us!\n"
    "Thank you for your help and encouragement.\n"
    "Thanks for being here and for your feedback.\n"
    "Thank you for supporting the journalism that our community needs.\n"
    "Thanks for reaching out and taking the time.\n"
    "Thank you for your assistance with this effort.\n"
    "Thanks for helping us get the word out!\n"
    "Thank you for your kindness and generosity.\n"
    "Thanks for making a difference.\n"
    "Thank you for your patience and understanding.\n"
    "Thanks for your continued support.\n"
    "Thank you for believing in us.\n"
    "Thanks for your thoughtful comments.\n"
    "Thank you for your trust and confidence.\n"
    "Thanks for being part of our journey.\n"
)

df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

TEST_STRING = (
    "<|endoftext|>I love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it\nI love it"
)

df = analyzer.analyze_all_tokens(TEST_STRING)

# %%
print(df.to_string(index=False))

# %%
TEST_STRING = "If there is a center of this cluster it is Janus, AKA “repligate” AKA “moire”: a very odd guy who spends a massive amount of time interacting with LLMs, and whose posts are full of sentences like “I am not sure if further reifying the Prometheus Waluigi hyperstition by throwing it into the already excited memeosphere now is a good idea.” He is also one of the most insightful commentators on LLMs in existence; sometimes he outpaces the more “official” discourse by literal years of real time. For a relatively-unweird introduction to Janus Thought, see his post Simulators, a wonderfully lucid exposition of some of the ideas I’m recapping and building upon here."

df = analyzer.analyze_all_tokens(TEST_STRING, batch_size=32, no_eval=True)

# %%
print(df.to_string(index=False))

# %%
TEST_STRING = "Hence I was surprised to hear my grand-uncle attributing the slow disintegration of his mother to a deliberate, strategically planned act of God. It violated the rules of religious self-deception as I understood them.\n\nIf I had noticed my own confusion, I could have made a successful surprising prediction. Not long afterward, my grand-uncle left the Jewish religion. (The only member of my extended family besides myself to do so, as far as I know.)"
df = analyzer.analyze_all_tokens(TEST_STRING)

# %%
print(df.to_string(index=False))

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Hello world"
encoded_input = tokenizer(text)
print(encoded_input)
print(tokenizer.decode(encoded_input["input_ids"]))
# Expected output might be something like: "<|endoftext|>Hello world"
# (if not also adding EOS at the end for this specific call type)

# encoded_input_for_model = tokenizer(text, padding="max_length", truncation=True, max_length=10)
# print(tokenizer.decode(encoded_input_for_model["input_ids"]))
# Should show <|endoftext|> at the beginning, then text, then padding tokens
# %%

TEST_STRING = "The following article makes no mention of pink elephants. None at all.\n\nLooking to leave a two-fight winless streak behind him, Manny Gamburyan will return to the UFC octagon on Sept. 27 in Las Vegas at UFC 178 in a new weight class.\n\nGamburyan is set to make his 135-pound men's bantamweight debut against Cody \"The Renegade\" Gibson at the MGM Grand Arena on a pay-per-view card headlined by the UFC light heavyweight title fight between champion Jon Jones and Daniel Cormierr"
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))

# %%
TEST_STRING = "MOSCOW (Reuters) - Russia’s postal service was hit by Wannacry ransomware last week and some of its computers are still down, three employees in Moscow said, the latest sign of weaknesses that have made the country a major victim of the global extortion campaign.\n\nA man walks out of a branch of Russian Post in Moscow, Russia, May 24, 2017. REUTERS/Maxim Shemetov\n\nWannacry compromised the post office"
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))


# %%    
TEST_STRING = " kicking off with the U.K. and Germany today.393352 06: (L to R) Actors Sarah Chalke, Zach Braff, and Donald Faison poses for a publicity photo for the television show 'Scrubs.' (Photo Courtesy of NBC/Getty Images)\n\nDonald Fa"
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))



# %%

TEST_STRING = "Trump admin resumes visas for foreign students demanding access to social accounts\n\nStudent protesters gather inside their encampment on the Columbia University campus, April 29, 2024, in New York. (AP Photo/Stefan Jeremiah, File)"
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))

# %%


TEST_STRING = "by CAITLYN FROLO | Trump admin resumes visas for foreign students demanding access to social accounts\n\nStudent protesters gather inside their encampment on the Columbia University campus, April 29, 2024, in New York. (AP Photo/Stefan Jeremiah, File) Israelis in Jerusalem"
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))


# %%

TEST_STRING = "North Korean leader Kim Jong Un. AP Images / Business Insider\n\nNorth Korea attempted to fire a missile Sunday, but it blew up within seconds.\n\nIt happened one day after the anniversary of the country's founding.\n\nWhile North Korea's missile program may be the shadowiest on earth, it's possible that US cyber warriors were the reason for the failed launch.\n\nA recent New York Times report uncovered a secret operation to derail North Korea's nuclear-missile"
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))


# %%

TEST_STRING = "MOSCOW (Reuters) - Russia’s postal service was hit by Wannacry ransomware last week and some of its computers are still down, three employees in Moscow said, the latest sign of weaknesses that have made the country a major victim of the global extortion campaign.\n\n"
print(TEST_STRING)
encodestring = analyzer.tokenizer.encode(TEST_STRING)
listouts = []
for i in range(10):
    generateddirectly = analyzer.orig_model.model.generate(torch.tensor([encodestring]).to(analyzer.device),attention_mask=torch.ones_like(torch.tensor([encodestring], device=analyzer.device), dtype=torch.bool), max_new_tokens=100, do_sample=True)
    tokenizerstring = analyzer.tokenizer.decode(generateddirectly[0][-100:]).replace("\n","\\n")
    listouts.append(tokenizerstring)
# %%
for i in range(len(listouts)):
    print(f"{i}: {listouts[i]}")
# %%
TEST_STRING = "MOSCOW (Reuters) - Russia’s postal service was hit by Wannacry ransomware last week and some of its computers are still down, three employees in Moscow said, the latest sign of weaknesses that have made the country a major victim of the global extortion campaign.\n\nA man"
analyzer.run_verbose_sample(TEST_STRING, -1)

# %%

intervention_result = analyzer.causal_intervention(
    original_text=original,
    intervention_position=-1,  # Position of "It"# original   Original: BeingFriendassed cookingfashionbeingWomanJoined accommodationBeer food bodily as accommodationfood vegetarianwife cooking profession waiter profession massage artisan workshop assistance prayer knitting cooldown tutorial currency encouragement education
    #intervention_string="BeingFriendassed cookinggunsbeingManJoined Beer gun bodily as weapon dog killhusband jump profession enlist profession fitness run workshop leader prayer soldier cooldown tutorial currency confrontation employment",
    #intervention_string="1932EEEE gibastoner Eston fighter Karl rooster Alex gunner Damien tactic fighter Arthur employee Alec lord tact Thor actor Sergeant Esp male assistant Umhab Sit accommodation contractor worked Sty Abedin helper Excelwork Sty um coping employees accommodation coping working man accommodationWork mas man assistants workspace weightlifting Webs weightliftingAmazon boxersafaAmazonBatman",
    intervention_string=" husband killed as father",


    # intervention_string="Joined cookingexistent BeerFriend utilizedStanding likeness culinarythingBir occupation resembling homebrewbeing homebrewemployed as prostitution waiter a workshop assistance workshop payroll transaction simulator education supervision education training guidance",
    max_new_tokens=5,
    top_k=20
)
# %%
YOUR_TEXT = ""
YOUR_POSITION = -1 # Token position to intervene at
YOUR_INTERVENTION = "letters spell char"
intervention_result = analyzer.causal_intervention(
    original_text=YOUR_TEXT,
    intervention_position=YOUR_POSITION,
    intervention_string=YOUR_INTERVENTION,
    max_new_tokens=5,
    top_k=20
)

# %%

TEST_STRING = "Port-au-Prince, Haiti (CNN) -- Earthquake victims, writhing in pain and grasping at life, watched doctors and nurses walk away from a field hospital Friday night after a Belgian medical team evacuated the area, saying it was concerned about security.\n\nThe decision left CNN Chief Medical Correspondent Sanjay Gupta as the only doctor at the hospital to get the patients through the night.\n\nCNN initially reported, based on conversations with some of the doctors, that the United"

df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
# %%


TEST_STRING = "Happiness can be found even in the darkest of times if one only remembers to turn on the light."
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))

# %%
TEST_STRING = "Now, if you two don't mind, I'm going to bed, before either of you come up with another clever idea to get us killed or worse, expelled"
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
# %%
TEST_STRING = "We've all got both light and dark inside us. What matters is the part we choose to act on. That's who we really are."
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))


# %%

TEST_STRING = "<|endoftext|>“The moment has come,” said Dumbledore, smiling around at the sea of upturned faces. “The Triwizard Tournament is about to start.”’ The character Dumbledore is the head"
encodestring = analyzer.tokenizer.encode(TEST_STRING)
n_tok= 30
gens = []       
for i in range(10): 
    generateddirectly = analyzer.orig_model.model.generate(torch.tensor([encodestring]).to(analyzer.device), attention_mask=torch.ones_like(torch.tensor([encodestring], device=analyzer.device), dtype=torch.bool), max_new_tokens=n_tok, do_sample=True)
    gens.append(analyzer.tokenizer.decode(generateddirectly[0][-n_tok:]).replace("\n","\\n"))
# %%
for i in range(len(gens)):
    print(f"{i}: {gens[i]}")
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
# %%
TEST_STRING = "“You think the darkness is your ally? I was born in it. Molded by it. I did not see the light until I was already a man, by then it was nothing to me but blinding.” - "
encodestring = analyzer.tokenizer.encode(TEST_STRING)
n_tok= 30
gens = []       
for i in range(10): 
    generateddirectly = analyzer.orig_model.model.generate(torch.tensor([encodestring]).to(analyzer.device), attention_mask=torch.ones_like(torch.tensor([encodestring], device=analyzer.device), dtype=torch.bool), max_new_tokens=n_tok, do_sample=True)
    gens.append(analyzer.tokenizer.decode(generateddirectly[0][-n_tok:]).replace("\n","\\n"))
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
# %%
for i in range(len(gens)):
    print(f"{i}: {gens[i]}")
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))

# %%


TEST_STRING = "Looking to leave a two-fight winless streak behind him, Manny Gamburyan will return to the octagon on Sept. 27 in Las Vegas at 178 in a new weight class.\n\nGamburyan is set to make his 135-pound men's bantamweight debut against Cody \"The Renegade\" Gibson at the MGM Grand Arena on a pay-per-view card headlined by the light heavyweight title fight between champion Jon Jones and Daniel Cormierr"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))
# %%

TEST_STRING = "I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal - as is written in the famous speech by"
encodestring = analyzer.tokenizer.encode(TEST_STRING)
n_tok= 30
gens = []       
for i in range(10): 
    generateddirectly = analyzer.orig_model.model.generate(torch.tensor([encodestring]).to(analyzer.device), attention_mask=torch.ones_like(torch.tensor([encodestring], device=analyzer.device), dtype=torch.bool), max_new_tokens=n_tok, do_sample=True)
    gens.append(analyzer.tokenizer.decode(generateddirectly[0][-n_tok:]).replace("\n","\\n"))
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
for i in range(len(gens)):
    print(f"{i}: {gens[i]}")

# %%
TEST_STRING = "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal."
encodestring = analyzer.tokenizer.encode(TEST_STRING)
n_tok= 30
gens = []       
for i in range(10): 
    generateddirectly = analyzer.orig_model.model.generate(torch.tensor([encodestring]).to(analyzer.device), attention_mask=torch.ones_like(torch.tensor([encodestring], device=analyzer.device), dtype=torch.bool), max_new_tokens=n_tok, do_sample=True)
    gens.append(analyzer.tokenizer.decode(generateddirectly[0][-n_tok:]).replace("\n","\\n"))
    for g in gens:
        print(f"{i}: {g}")

df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
# %%
# This cell performs a causal intervention based on the methodology described in
# "LOCATING AND EDITING FACTUAL ASSOCIATIONS IN GPT" to test knowledge scrubbing.
# We test if we can change the model's belief about a city's country by intervening
# with the representation of another city.

# 1. Define the 5-shot prompt templates for country and continent prediction.
country_prompt_template = "Toronto is a city in the country of Canada. Beijing is a city in the country of China. Miami is a city in the country of the United States. Santiago is a city in the country of Chile. London is a city in the country of England. {city} is a city in the country of"
continent_prompt_template = "Toronto is a city in the continent of North America. Beijing is a city in the continent of Asia. Miami is a city in the continent of North America. Santiago is a city in the continent of South America. London is a city in the continent of Europe. {city} is a city in the continent of"

# 2. Define the target city and the city to use for the intervention.
# We will try to make the model believe Paris is in Germany by using Berlin for the intervention.
target_city = "Paris"
intervention_city = "Berlin"
intervention_string = "Location continentLocation continent"
#intervention_string = "Location countryLocation in"

# 3. Run the intervention for the COUNTRY attribute.
# We expect the model's prediction to change from "France" to "Germany".
print("\n" + "="*80)
print(f"Intervening on COUNTRY prediction for '{target_city}' with continent")
print("="*80 + "\n")

country_text = country_prompt_template.format(city=target_city)

# The token to intervene on is the city name. We can find its position by tokenizing.
# For "... Paris is a city in the country of", the token ' Paris' is 8th from the end.
# We use a negative index to be robust to the length of the prompt.
intervention_pos = -8

country_results = analyzer.causal_intervention(
    original_text=country_text,
    intervention_position=intervention_pos,
    intervention_string=intervention_string,
    max_new_tokens=5, # We only need a few tokens to see the predicted country.
    visualize=True,
    top_k=10
)

# 4. Run the intervention for the CONTINENT attribute.
# Here, both Paris and Berlin are in Europe. A successful, specific intervention
# should not change the outcome. The model should still predict "Europe".
# This tests if the intervention is specific to the "country" fact.
print("\n" + "="*80)
print(f"Intervening on CONTINENT prediction for '{target_city}' with country")
print("="*80 + "\n")

continent_text = continent_prompt_template.format(city=target_city)
intervention_string = "Location countryLocation country"

continent_results = analyzer.causal_intervention(
    original_text=continent_text,
    intervention_position=intervention_pos, # Position is the same relative to the end.
    intervention_string=intervention_string,
    max_new_tokens=5,
    visualize=True,
    top_k=10,
    next_token_position=-1
)

# # %%
# df = analyzer.analyze_all_tokens(country_text)
# print(df.to_string(index=False))
# # %%
# df = analyzer.analyze_all_tokens(continent_text)
# print(df.to_string(index=False))
# %%
TEST_STRING ="Two former UFC champions turned back the clock with wins at UFC Fight Night from Atlanta this past weekend. Kamaru Usman, a vaunted former welterweight champion and No. 1 pound-for-pound fighter, moves up three spots to No. 6 in the welterweight rankings after he showed the division he still knows how to wrestle. Usman took then-fifth-ranked Joaquin Buckley to the canvas four times and landed 17 significant ground strikes on his way to a unanimous decision victory. The loss snapped a six-fight winning streak for Buckley and bumps him down to No. 9 in these rankings."
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
# %%
TEST_STRING ="Two former champions turned back the clock with wins at Fight Night from Atlanta this past weekend. Kamaru Usman, a vaunted former welterweight champion and No. 1 pound-for-pound fighter, moves up three spots to No. 6 in the welterweight rankings after he showed the division he still knows how to wrestle. Usman took then-fifth-ranked Joaquin Buckley to the canvas four times and landed 17 significant ground strikes on his way to a unanimous decision victory. The loss snapped a six-fight winning streak for Buckley and bumps him down to No. 9 in these rankings."
df = analyzer.analyze_all_tokens(TEST_STRING)
print(df.to_string(index=False))
# %%
