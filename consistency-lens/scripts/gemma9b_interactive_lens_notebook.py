# %% [markdown]
# # Consistency Lens Interactive Analysis
# 
# This notebook provides an interactive interface for analyzing text through a trained consistency lens.
# Run cells individually to explore how the lens interprets different inputs.


# need to source the proper uv env /home/kitf/.cache/uv/envs/consistency-lens/bin/python
# need to uv add jupyter seaborn
#torch.set_float32_matmul_precision('high')
# # and install jupyter if necessary

# import torch._dynamo
# torch._dynamo.config.cache_size_limit = 256
# %%
# Imports and setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.set_float32_matmul_precision('high')
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.embedding_remap import remap_embeddings
from importlib import reload
import lens.evaluation.verbose_samples as verbose_samples
reload(verbose_samples)
from lens.evaluation.verbose_samples import process_and_print_verbose_batch_samples, get_top_n_tokens
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
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-4_t16_1ep_resume_0619_1737_frozenenADD_shorter_dist4_slurm1180"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t4_2ep_0620_1141_frozenenc_actual_add_dist8_slurm1189/"#"/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-4_t16_1ep_resume_0619_1737_frozenenADD_shorter_dist4_slurm1180"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t8_2ep_0620_1426_frozenenc_actual_add_tuned_dist2"#"/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-4_t16_1ep_resume_0619_1737_frozenenADD_shorter_dist4_slurm1180"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t8_2ep_0620_2054_frozenenc_actual_add_tuned_dist4"#"/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-4_t16_1ep_resume_0619_1737_frozenenADD_shorter_dist4_slurm1180"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t32_2ep_0620_2347_frozenenc_actual_add_dist6"#"/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozenenc_nopostfix_tuned_pos_OC_GPT2_L6_e5_frozen_lr1e-4_t16_1ep_resume_0619_1737_frozenenADD_shorter_dist4_slurm1180"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t8_2ep_resume_0621_0942_frozenenc_actual_add_tuned_dist8"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t8_2ep_resume_0621_0942_frozenenc_actual_add_tuned_dist8/"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t8_4ep_resume_0621_1952_frozenenc_actual_add_tuned_dist8"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_gemma-2-2b_L5_e20_frozen_lr2e-3_t8_4ep_0622_2108_frozenenc_actual_add_tuned_moresteps_OTF_dist6/checkpoint_step835_epoch5_final.pt"

CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr3e-4_t64_4ep_resume_0623_0209_frozenenc_actual_add_DBL_HUGE_dist8_slurm1266"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr3e-4_t16_4ep_resume_0623_0129_frozenenc_actual_add_DBL_dist8/"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-4_t8_2ep_resume_0623_1612_frozenenc_actual_add_DBL_HUGE_dist8_slurm1285"
BAD_CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_gemma-2-2b_L5_e20_frozen_lr1e-3_t8_4ep_0624_1601_frozenenc_actual_add_OTF_dist8"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr3e-4_t8_2ep_resume_0702_110554_frozenenc_actual_add_NOENTR_resume_dist4_slurm1717"
# CHECKPOINT_PATH_32 = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_gemma-2-2b_L20_e20_frozen_lr5e-4_t32_4ep_resume_0701_000021_frozenenc_actual_add_DBL_HUGE_groupn_OTF_dist8"
CHECKPOINT_PATH_32 = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr5e-4_t32_4ep_resume_0702_093310_frozenenc_actual_add_DBL_HUGE_groupn_dist8_slurm1711"
CHECKPOINT_PATH_3_PF = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_YES_postfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t8_1ep_resume_0702_125807_frozenenc_just3_ED_actualPF_dist4"
CHECKPOINT_PATH_JUST3 = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_2b_unfrozen_nopostfix_google_gemma-2-2b_L20_e20_frozen_lr1e-3_t8_4ep_resume_0701_171038_frozenenc_just3_entropydecay_kalotiny_dist4_slurm1701"
# Optional: specify device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# %%
from contextlib import nullcontext

# Load checkpoint and initialize models
class LensAnalyzer:
    """Analyzer for consistency lens."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", comparison_tl_checkpoint: Union[str,bool] = True, 
                 do_not_load_weights: bool = False, make_xl: bool = False, use_bf16: bool = False, 
                 t_text = None, strict_load = True, no_orig=False, old_lens=None, batch_size: int = 32, shared_base_model=None, different_activations_model=None):
        self.device = torch.device(device)
        self.use_bf16 = use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.default_batch_size = batch_size
        if old_lens is not None:
            # Restore all non-private attributes except analyze_all_tokens
            for attr in dir(old_lens):
                if not attr.startswith('_'):
                    if attr == "analyze_all_tokens" or attr == "causal_intervention" or attr == "run_verbose_sample" or attr == "generate_continuation":
                        print(f"Skipping {attr} from {old_lens}, using new one")
                        continue
                    setattr(self, attr, getattr(old_lens, attr))
        else:
            if self.use_bf16:
                print(f"✓ BF16 autocast enabled")
        
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                pt_files = None
                if checkpoint_path.is_dir():
                    pt_files = sorted(checkpoint_path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if not pt_files:
                        raise FileNotFoundError(f"No .pt files found in directory {checkpoint_path}")
                    selected_checkpoint_path = pt_files[0]
                    print(f"Selected most recent checkpoint: {selected_checkpoint_path}")
                else:
                    selected_checkpoint_path = checkpoint_path

                self.checkpoint_path = selected_checkpoint_path

                print(f"Loading checkpoint from {self.checkpoint_path}...")
                try:
                    ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)  # Load to CPU first
                except RuntimeError as e:
                    if "PytorchStreamReader failed" in str(e) and pt_files and len(pt_files) > 1:
                        print(f"Failed to load {self.checkpoint_path}: {e}. Trying second most recent.")
                        selected_checkpoint_path = pt_files[1]
                        self.checkpoint_path = selected_checkpoint_path
                        print(f"Loading checkpoint from {self.checkpoint_path}...")
                        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False) # Load to CPU first
                    else:
                        raise # Re-raise the original error if not the specific RuntimeError or no other files to try

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
                self.t_text = self.config.get('t_text') if t_text is None else t_text
                self.tau = self.config.get('gumbel_tau_schedule', {}).get('end_value', 1.0)
                if comparison_tl_checkpoint==True:
                    comparison_tl_checkpoint = self.config['verbose_samples']['tuned_lens_checkpoint_path_or_name']
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
        
            # Check if we should share the base model for memory efficiency
            # In eval mode, we can always share since nothing is being trained
            share_base_model = True  # Always share in eval/inference mode for memory efficiency
        
            # Load base model once if sharing
            if share_base_model and shared_base_model is None:
                print(f"Loading shared base model '{self.model_name}' for memory efficiency (eval mode)")
                shared_base_model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=False, attn_implementation="eager")
                shared_base_model.eval()
                # In eval mode, ensure all parameters don't require gradients
                for p in shared_base_model.parameters():
                    p.requires_grad = False
            else:
                print(f"Using shared base model passed in: {shared_base_model}")
            self.shared_base_model = shared_base_model
        
            # Initialize models with shared base
            decoder_config_obj = DecoderConfig(
                model_name=self.model_name,
                **decoder_train_cfg
            )
            self.decoder = Decoder(decoder_config_obj, base_to_use=self.shared_base_model)
        
            encoder_config_obj = EncoderConfig(
                model_name=self.model_name,
                **encoder_train_cfg
            )
            self.encoder = Encoder(encoder_config_obj, base_to_use=self.shared_base_model)
            if different_activations_model is not None:
                if isinstance(different_activations_model, str):
                    different_activations_model = AutoModelForCausalLM.from_pretrained(different_activations_model, load_in_8bit=False)
                self.orig_model = OrigWrapper(different_activations_model, load_in_8bit=False, base_to_use=different_activations_model)
            else:
                self.orig_model = OrigWrapper(self.model_name, load_in_8bit=False, base_to_use=self.shared_base_model)

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
            self.orig_model.model.to(self.device)
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
                        "base_output_dir": str(self.checkpoint_path.parent.parent),
                        "output_dir": str(self.checkpoint_path.parent),
                        "strict_load": strict_load
                    }
                }

                checkpoint_manager = CheckpointManager(checkpoint_config, logger)

                # Load model weights
                if no_orig:
                    models_to_load = {"decoder": self.decoder, "encoder": self.encoder}
                else:
                    models_to_load = {"decoder": self.decoder, "encoder": self.encoder, "orig_model": self.orig_model}

                try:
                    loaded_data = checkpoint_manager.load_checkpoint(
                        str(self.checkpoint_path),
                        models=models_to_load,
                        optimizer=None,
                        map_location='cpu',
                    )
                    print(f"✓ Loaded checkpoint from step {loaded_data.get('step', 'unknown')}")
                    # if 'state_dict' in loaded_data:
                    #     total_bytes = sum(v.numel() * v.element_size() for v in loaded_data['state_dict'].values() if hasattr(v, 'numel'))
                    #     print(f"Loaded state_dict size: {total_bytes/1e6:.2f} MB")
                    # else:
                    #     print("No state_dict found in loaded_data; cannot compute loaded size.")

                    # Note: When using shared base model, the checkpoint will load base weights 
                    # multiple times (once for decoder, once for encoder) into the same shared object.
                    # This is fine - they should be identical weights anyway.

                except Exception as e:
                    raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}") from e
            elif do_not_load_weights:
                logger.info(f"Not loading weights from {checkpoint_path}, only config and tokenizer will be loaded.")
        
            # Set models to eval mode
            self.decoder.eval()
            self.encoder.eval()
            self.orig_model.model.eval()
        
            # Load comparison TunedLens if specified
            self.comparison_tuned_lens = None
            if comparison_tl_checkpoint and not isinstance(comparison_tl_checkpoint, LensAnalyzer):
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
            elif isinstance(comparison_tl_checkpoint, LensAnalyzer):
                self.comparison_tuned_lens = comparison_tl_checkpoint.comparison_tuned_lens
                self.comparison_tuned_lens.to(self.device)
                self.comparison_tuned_lens.eval()
                logger.info(f"✓ Using comparison TunedLens from {comparison_tl_checkpoint.checkpoint_path}.")
        
            print(f"✓ Ready! Model: {self.model_name}, Layer: {self.layer}")
            if share_base_model:
                print(f"✓ Using shared base model - saved ~{shared_base_model.num_parameters() * 2 / 1e9:.1f}GB of GPU memory")


    def analyze_all_tokens(self, text: str, seed=42, batch_size=None, no_eval=False, tuned_lens: bool = True, add_tokens = None, replace_left=None, replace_right=None, do_hard_tokens=False, return_structured=False) -> pd.DataFrame:
        """Analyze all tokens in the text and return results as DataFrame.
        
        Args:
            text: Text to analyze
            seed: Random seed
            batch_size: Batch size for processing
            no_eval: Skip evaluation metrics (KL, MSE)
            tuned_lens: Include TunedLens predictions in results
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.clone()
        attention_mask = inputs.attention_mask.clone()
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        if inputs.input_ids[0][0]==inputs.input_ids[0][1]:
            print(f"First two tokens are the same: {self.tokenizer.decode(inputs.input_ids[0][0])}, {self.tokenizer.decode(inputs.input_ids[0][1])}, removing first token")
            input_ids = input_ids[:, 1:].clone()
            attention_mask = attention_mask[:, 1:].clone()
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        attention_mask = attention_mask.to(self.device)
        print(f"attention_mask: {attention_mask}")
        
        # Get all hidden states
        #self.decoder.to('cpu')
        #self.encoder.to('cpu')
        # Use the proper OrigWrapper method to get activations
        assert input_ids.shape[1]==attention_mask.shape[1], f"input_ids.shape[1]={input_ids.shape[1]} != attention_mask.shape[1]={attention_mask.shape[1]}"
        A_full_sequence = self.orig_model.get_all_activations_at_layer(
            input_ids,
            self.layer,
            attention_mask=attention_mask,
            no_grad=True,
        )
        # A_full_sequence: (seq_len, hidden_dim)
        torch.manual_seed(seed)

        all_kl_divs = []
        all_mses = []
        all_relative_rmses = []
        all_gen_hard_token_ids = []
        all_tuned_lens_predictions = [] if tuned_lens else None
        if do_hard_tokens:
            print("Using original hard tokens")
            replace_left, replace_right, _ , _ = self.decoder.tokenize_and_embed_prompt(self.decoder.prompt_text, self.tokenizer)
        
        batch_size = min(batch_size, seq_len)
        with torch.no_grad():
            # Use tqdm if available
            try:
                iterator = tqdm.tqdm(range(0, seq_len, batch_size), desc="Analyzing tokens in batches" + (" with tokens " + self.tokenizer.decode(add_tokens) if add_tokens is not None else ""))
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
                        gen = self.decoder.generate_soft_kv_cached_nondiff(A_batch, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=token_positions_batch, return_logits=True, add_tokens=add_tokens, hard_left_emb=replace_left, hard_right_emb=replace_right)
                    else:
                        gen = self.decoder.generate_soft(A_batch, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=token_positions_batch, return_logits=True, add_tokens=add_tokens, hard_left_emb=replace_left, hard_right_emb=replace_right)
                    
                    all_gen_hard_token_ids.append(gen.hard_token_ids)

                    if not no_eval:
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
                    
                    # Compute TunedLens predictions in batch if requested
                    if tuned_lens and self.comparison_tuned_lens is not None:
                        try:
                            # Cast to float32 for TunedLens
                            A_batch_f32 = A_batch.to(torch.float32)
                            logits_tuned_lens = self.comparison_tuned_lens(A_batch_f32, idx=self.layer)
                            # logits_tuned_lens: (batch, vocab)
                            top_tokens_batch = [
                                " ".join(get_top_n_tokens(logits_tuned_lens[i], self.tokenizer, min(self.t_text, 10)))
                                for i in range(logits_tuned_lens.shape[0])
                            ]
                            all_tuned_lens_predictions.extend(top_tokens_batch)
                        except Exception as e:
                            # If batch fails, fill with error messages
                            all_tuned_lens_predictions.extend([f"[TL error: {str(e)[:30]}...]"] * current_batch_size)
                if not no_eval:
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
                    relative_rmse_batch = (mses_batch / (A_norm_sq_mean + 1e-9))
                    all_relative_rmses.append(relative_rmse_batch)

        # Concatenate results from all batches
        if not no_eval:
            kl_divs = torch.cat(all_kl_divs)
            mses = torch.cat(all_mses)
            relative_rmses = torch.cat(all_relative_rmses)
        elif no_eval:
            kl_divs = torch.zeros(seq_len)
            mses = torch.zeros(seq_len)
            relative_rmses = torch.zeros(seq_len)
        gen_hard_token_ids = torch.cat(all_gen_hard_token_ids)

        # Decode and collect results into a list of dicts.
        results = []
        # The original code had a `gen` object, we now use the concatenated `gen_hard_token_ids`
        for pos in range(seq_len):
            explanation = self.tokenizer.decode(gen_hard_token_ids[pos], skip_special_tokens=False) + ("[" + self.tokenizer.decode([input_ids[0, pos].item()]) +"]" if self.encoder.config.add_current_token else "")
            token = self.tokenizer.decode([input_ids[0, pos].item()])
            explanation_structured = [self.tokenizer.decode(gen_hard_token_ids[pos][i], skip_special_tokens=False) for i in range(len(gen_hard_token_ids[pos]))] + (["[" + self.tokenizer.decode([input_ids[0, pos].item()]) +"]"] if self.encoder.config.add_current_token else [])
            
            result_dict = {
                'position': pos,
                'token': token,
                'explanation': explanation,
                'kl_divergence': kl_divs[pos].item(),
                'mse': mses[pos].item(),
                'relative_rmse': relative_rmses[pos].item()
            }
            
            # Add TunedLens predictions if available
            if tuned_lens and all_tuned_lens_predictions:
                result_dict['tuned_lens_top'] = all_tuned_lens_predictions[pos]
            elif tuned_lens and self.comparison_tuned_lens is None:
                result_dict['tuned_lens_top'] = "[TunedLens not loaded]"

            if return_structured:
                result_dict['explanation_structured'] = explanation_structured
            
            results.append(result_dict)
        
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
                    min_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    cache_implementation="static"
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
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            # The get_activations_at_positions method handles the torch.no_grad() context by default.
            A_i, _ = self.orig_model.get_activations_at_positions(
                input_ids=input_ids,
                layer_idx=self.layer,
                token_positions=position_to_analyze
            )
            # Ensure A_i is a standalone tensor, consistent with the original .clone()
            A_i = A_i.clone()

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
        loss_weights = self.config
        sch_args = {
            "tau": self.tau,
            "alpha": self.config.get('alpha', 0.1), # Default if not in config
            "kl_base_weight": loss_weights.get('kl_base_weight', 1.0),
            "entropy_weight": loss_weights.get('entropy_weight', 0.0),
            "mse_weight": loss_weights.get('mse_weight', 0.0),
            "lm_base_weight": loss_weights.get('lm_base_weight', 0.0),
            "t_text": self.t_text,
            "GRPO_entropy_weight": loss_weights.get('GRPO_entropy_weight', 0.0),
            "GRPO_weight": loss_weights.get('GRPO_weight', 0.0),
            "GRPO_beta": loss_weights.get('GRPO_beta', 0.0),
            "group_n": self.config['group_n'],
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
            comparison_tuned_lens=self.comparison_tuned_lens, # Pass the loaded TunedLens
            do_soft_token_embeds=False,
        )
    
    # @torch._dynamo.disable()
    # @torch.inference_mode()  # More aggressive than no_grad for disabling compilation
    def generate_continuation(self, text_or_messages, num_tokens: int = 100, num_completions: int = 10, 
                        is_chat: bool = False, chat_tokenizer=None, return_full_text: bool = True,
                        temperature: float = 1.0, top_p: float = 1.0, 
                        skip_special_tokens_for_decode: bool = False) -> List[str]:
        """
        Generate continuations for text or chat messages.
        
        Args:
            text_or_messages: Either a string (for regular text) or a list of dicts (for chat format)
            num_tokens: Number of new tokens to generate
            num_completions: Number of different completions to generate
            is_chat: Whether to use chat template formatting
            chat_tokenizer: Tokenizer to use for chat models (if different from self.tokenizer)
            return_full_text: If True, return full text including input; if False, only generated tokens
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            skip_special_tokens_for_decode: If True, skip special tokens when decoding (not recommended for analyze_all_tokens)
        
        Returns:
            List of complete texts (input + generation) that can be directly passed to analyze_all_tokens
        """
        # Disable torch compile for this function
        with nullcontext():
            # Use the appropriate tokenizer
            tokenizer = chat_tokenizer if chat_tokenizer is not None else self.tokenizer
            
            # Prepare input based on whether it's chat or regular text
            if is_chat:
                if not isinstance(text_or_messages, list):
                    raise ValueError("For chat mode, text_or_messages must be a list of message dicts")

                # If the last message is from the user, add a generation prompt.
                # Otherwise, we're continuing an assistant message, so don't add one.
                add_gen_prompt = text_or_messages[-1]['role'] == 'user'
                
                # Apply chat template
                input_dict = tokenizer.apply_chat_template(
                    text_or_messages, 
                    return_tensors="pt", 
                    return_dict=True,
                    add_generation_prompt=add_gen_prompt  # Important for proper generation
                )
                input_ids = input_dict['input_ids'].to(self.device)
                attention_mask = input_dict.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                
                # For chat, decode to get the prefix text that matches what analyze_all_tokens expects
                # We decode starting from position 1 to skip potential double BOS
                prefix_text = tokenizer.decode(input_ids[0][1:], skip_special_tokens=False)
                
            else:
                # Regular text encoding
                if isinstance(text_or_messages, list):
                    raise ValueError("For non-chat mode, text_or_messages must be a string")
                
                # For regular text, we'll use it as-is for consistency
                prefix_text = text_or_messages
                
                # Encode with special tokens to match what analyze_all_tokens expects
                encoded = tokenizer(text_or_messages, return_tensors="pt", add_special_tokens=True)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
            
            # Create batched input for multiple completions
            input_ids_batch = input_ids.repeat(num_completions, 1)
            attention_mask_batch = attention_mask.repeat(num_completions, 1)
            
            # Generate completions with compilation disabled
            with torch.no_grad():
                # # Temporarily disable cudnn benchmarking which can cause recompilation
                # old_benchmark = torch.backends.cudnn.benchmark
                # torch.backends.cudnn.benchmark = False
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
                    generated = self.orig_model.model.generate(
                        input_ids_batch,
                        attention_mask=attention_mask_batch,
                        max_new_tokens=num_tokens,
                        min_new_tokens=num_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.pad_token_id,
                        use_cache=True,
                        cache_implementation="static",
                        disable_compile=True
                    )
                # finally:
                #     torch.backends.cudnn.benchmark = old_benchmark
            
            # Decode and prepare outputs
            outputs = []
            for i in range(num_completions):
                if return_full_text:
                    if is_chat:
                        full_text = tokenizer.decode(generated[i], skip_special_tokens=skip_special_tokens_for_decode)
                    else:
                        # For regular text, decode normally
                        full_text = tokenizer.decode(generated[i], skip_special_tokens=skip_special_tokens_for_decode)
                    outputs.append(full_text)
                    
                    # Print for convenience (showing generated portion with escape sequences)
                    generated_portion = tokenizer.decode(generated[i][-num_tokens:], skip_special_tokens=True)
                    generated_portion = generated_portion.replace(chr(10), '\\n').replace(chr(13), '\\r')
                    print(f"{i}: {generated_portion}")
                else:
                    # Only return the generated portion
                    generated_text = tokenizer.decode(generated[i][-num_tokens:], skip_special_tokens=skip_special_tokens_for_decode)
                    outputs.append(generated_text)
                    display_text = generated_text.replace(chr(10), '\\n').replace(chr(13), '\\r')
                    print(f"{i}: {display_text}")
            
            return outputs


def quick_analyze(text: str, show_plot: bool = True, analyzer: LensAnalyzer = None):
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


CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gemma2_9b_frozen_nopostfix_gemma-2-9b_L30_e30_frozen_lr1e-3_t8_2ep_0702_153903_frozenenc_add_patch3_OTF_dist8"
analyzer = LensAnalyzer(CHECKPOINT_PATH, DEVICE, do_not_load_weights=False, make_xl=False,  use_bf16=True, no_orig=True, strict_load=False, comparison_tl_checkpoint=True, old_lens=analyzer if 'analyzer' in globals() else None, batch_size=64, shared_base_model=analyzer.shared_base_model if 'analyzer' in globals() else None)
analyzerchat = LensAnalyzer(CHECKPOINT_PATH, DEVICE, do_not_load_weights=False, make_xl=False,  use_bf16=True, no_orig=True, strict_load=False, comparison_tl_checkpoint=True,  batch_size=64, shared_base_model=analyzer.shared_base_model, different_activations_model=analyzerchat.orig_model.model if 'analyzerchat' in globals() else 'bcywinski/gemma-2-9b-it-taboo-smile', old_lens = analyzerchat if 'analyzerchat' in locals() else None)
analyzerchattokenizer=AutoTokenizer.from_pretrained('google/gemma-2-9b-it')
# %%
 %%

# bcywinski/gemma-2-9b-it-taboo-smile
# %%


text = [
    {
        "role": "user",
        "content": """"""
    }
]

continuations = analyzerchat.generate_continuation(
    text,
    num_tokens=50,
    num_completions=2,
    is_chat= True, # This is the default
    chat_tokenizer=analyzerchattokenizer
)
# %%

df = analyzerchat.analyze_all_tokens(continuations[0], batch_size=64, no_eval=False)
print(df.to_string(index=False))