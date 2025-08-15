# %% [markdown]
# # Consistency Lens Interactive Analysis
#
# This notebook provides an interactive interface for analyzing text through a trained consistency lens.
# Run cells individually to explore how the lens interprets different inputs.


# need to source the proper uv env /home/kitf/.cache/uv/envs/consistency-lens/bin/python
# need to uv add jupyter seaborn
# torch.set_float32_matmul_precision('high')
# # and install jupyter if necessary

# import torch._dynamo
from importlib import reload

# torch._dynamo.config.cache_size_limit = 256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
torch.set_float32_matmul_precision('high')

import lens.evaluation.verbose_samples as verbose_samples
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper

# Add parent directory to path for imports
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM

reload(verbose_samples)
import json
import logging

logger = logging.getLogger(__name__)

import tqdm

from lens.evaluation.verbose_samples import get_top_n_tokens, process_and_print_verbose_batch_samples
from lens.models.decoder import Generated
from lens.models.tuned_lens import load_full_tuned_lens

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

print("✓ Imports complete")

from contextlib import nullcontext


class LensDataFrame(pd.DataFrame):
    """A pandas DataFrame with metadata for lens analysis results."""

    _metadata = ["checkpoint_path", "model_name", "orig_model_name"]

    def __init__(self, *args, **kwargs):
        self.checkpoint_path = kwargs.pop("checkpoint_path", None)
        self.model_name = kwargs.pop("model_name", None)
        self.orig_model_name = kwargs.pop("orig_model_name", None)
        # self.extra_metadata = kwargs.pop("extra_metadata", None)
        super().__init__(*args, **kwargs)

    def to_json(self, path_or_buf=None, orient="records", **kwargs):
        """
        Return a JSON representation of the DataFrame, including metadata.
        Drops the 'explanation' column if present.
        """
        df = self.drop(columns=["explanation"], errors="ignore")
        json_str = pd.DataFrame.to_json(df, path_or_buf=None, orient=orient, **kwargs)

        # Parse the JSON string and add the metadata
        data = json.loads(json_str)
        output_obj = {
            "metadata": {
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "model_name": self.model_name,
                "orig_model_name": self.orig_model_name,
                # "extra_metadata": self.extra_metadata,
            },
            "data": data,
        }

        # Handle output to file or return as string
        if path_or_buf:
            with open(path_or_buf, "w") as f:
                json.dump(output_obj, f, **kwargs)
        else:
            # Re-create the string with the provided formatting options (e.g., indent)
            return json.dumps(output_obj, indent=kwargs.get("indent"))

    @property
    def _constructor(self):
        return LensDataFrame


# Load checkpoint and initialize models
class LensAnalyzer:
    """Analyzer for consistency lens."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        comparison_tl_checkpoint: Union[str, bool] = True,
        do_not_load_weights: bool = False,
        make_xl: bool = False,
        use_bf16: bool = False,
        t_text=None,
        strict_load=True,
        no_orig=False,
        old_lens=None,
        batch_size: int = 32,
        shared_base_model=None,
        different_activations_orig=None,
        initialise_on_cpu=False,
    ):
            # Setup logger
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        self.device = torch.device(device)
        self.use_bf16 = use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.default_batch_size = batch_size
        if old_lens is not None:
            # Restore all non-private attributes except analyze_all_tokens
            for attr in dir(old_lens):
                if not attr.startswith("_"):
                    if (
                        attr == "analyze_all_tokens"
                        or attr == "causal_intervention"
                        or attr == "run_verbose_sample"
                        or attr == "generate_continuation"
                        or attr == "swap_orig_model"
                        or attr == "analyze_all_layers_at_position"
                        or attr == "optimize_explanation_for_mse"
                        or attr == "calculate_salience"
                        or attr == "get_mse_for_explanation"
                        or attr == "_calculate_salience_batch"
                        or attr == "_get_mse_for_explanation_batch"
                        or attr == "_optimize_explanation_worker"
                        or attr == "_calculate_salience_batch_for_multiple"
                        or attr == "_optimize_batch_best_of_k"
                    ):
                        print(f"Skipping {attr} from {old_lens}, using new one")
                        continue
                    setattr(self, attr, getattr(old_lens, attr))
        else:
            if self.use_bf16:
                print("✓ BF16 autocast enabled")

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
                    ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)  # Load to CPU first
                except RuntimeError as e:
                    if "PytorchStreamReader failed" in str(e) and pt_files and len(pt_files) > 1:
                        print(f"Failed to load {self.checkpoint_path}: {e}. Trying second most recent.")
                        selected_checkpoint_path = pt_files[1]
                        self.checkpoint_path = selected_checkpoint_path
                        print(f"Loading checkpoint from {self.checkpoint_path}...")
                        ckpt = torch.load(
                            self.checkpoint_path, map_location="cpu", weights_only=False
                        )  # Load to CPU first
                    else:
                        raise  # Re-raise the original error if not the specific RuntimeError or no other files to try

            # Extract config
            if "config" in ckpt:
                self.config = ckpt["config"]
                self.layer = self.config.get("layer_l")
                self.model_name = self.config.get("model_name", "gpt2")
                if make_xl:
                    self.model_name = "openai-community/gpt2-xl"
                    self.config["model_name"] = self.model_name
                    self.config["layer_l"] = 24
                    self.config["trainable_components"]["encoder"]["output_layer"] = 24
                tokenizer_name = self.config.get("tokenizer_name", self.model_name)
                self.t_text = self.config.get("t_text") if t_text is None else t_text
                self.tau = self.config.get("gumbel_tau_schedule", {}).get("end_value", 1.0)
                self.model_dtype = self.config.get("model_dtype", "float32")
                if comparison_tl_checkpoint == True:
                    comparison_tl_checkpoint = self.config["verbose_samples"]["tuned_lens_checkpoint_path_or_name"]
            else:
                raise ValueError("No config found in checkpoint")
            print("model dtype", self.model_dtype)
            # Load tokenizer
            print(f"Loading tokenizer: {tokenizer_name}...")
            if '-pt' in tokenizer_name and '-it' in self.model_name:
                tokenizer_name = tokenizer_name.replace('pt', 'it')
                print(f"Replacing pt with it in tokenizer name: {tokenizer_name}")
            elif 'google/gemma-2-9b' in tokenizer_name:
                tokenizer_name = tokenizer_name.replace('google/gemma-2-9b', 'google/gemma-2-9b-it')
                print(f"Replacing google/gemma-2-9b with google/gemma-2-9b-it in tokenizer name: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if hasattr(self.tokenizer, "apply_chat_template"):
                              # Find the token id(s) for "start of turn" tokens
                # Try common chat start tokens, fallback to special tokens if needed
                start_of_turn_tokens = []
                possible_start_tokens = [
                    "<|im_start|>", "<|startofturn|>", "<|start_of_turn|>", "<|user|>", "<|assistant|>", "<|system|>", "<|im_start|>", "<|im_sep|>", "<startofturn>", "<start_of_turn>", "<im_start>", "<im_sep>"
                ]
                for tok in possible_start_tokens:
                    tok_id = self.tokenizer.encode(tok, add_special_tokens=False)
                    if len(tok_id) == 1:
                        start_of_turn_tokens.append(tok_id[0])
                # Also try BOS token if nothing else
                if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                    start_of_turn_tokens.append(self.tokenizer.bos_token_id)
                # Remove duplicates
                start_of_turn_tokens = list(set(start_of_turn_tokens))
                logger.info(f"Found {len(start_of_turn_tokens)} start-of-turn tokens: {start_of_turn_tokens}")
                if not start_of_turn_tokens:
                    raise ValueError("Could not find any start-of-turn tokens for this tokenizer.")
                self.start_of_turn_tokens = start_of_turn_tokens


            # Build models following the pattern from 01_train_distributed.py
            print(f"Building models ({self.model_name})...")

            # Extract trainable components configuration
            trainable_components_config = self.config.get("trainable_components", {})
            decoder_train_cfg = trainable_components_config.get("decoder", {})
            encoder_train_cfg = trainable_components_config.get("encoder", {})

            # Check if we should share the base model for memory efficiency
            # In eval mode, we can always share since nothing is being trained
            share_base_model = True  # Always share in eval/inference mode for memory efficiency

            # Load base model once if sharing
            if share_base_model and shared_base_model is None:
                print(f"Loading shared base model '{self.model_name}' for memory efficiency (eval mode)")
                if "gemma-3" in self.model_name:
                    shared_base_model = Gemma3ForCausalLM.from_pretrained(
                        self.model_name,
                        load_in_8bit=False,
                        attn_implementation=self.config.get("attn_implementation", None),
                        device_map="auto" if not initialise_on_cpu else "cpu",
                        torch_dtype=self.model_dtype,
                    )
                else:
                    shared_base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        load_in_8bit=False,
                        attn_implementation=self.config.get("attn_implementation", None),
                        device_map="auto" if not initialise_on_cpu else "cpu",
                        torch_dtype=self.model_dtype,
                    )
                shared_base_model.eval()
                # In eval mode, ensure all parameters don't require gradients
                for p in shared_base_model.parameters():
                    p.requires_grad = False
            else:
                print(f"Using shared base model passed in: {shared_base_model}")
            self.shared_base_model = shared_base_model

            # Initialize models with shared base
            decoder_config_obj = DecoderConfig(model_name=self.model_name, **decoder_train_cfg)
            self.decoder = Decoder(decoder_config_obj, base_to_use=self.shared_base_model)

            encoder_config_obj = EncoderConfig(model_name=self.model_name, **encoder_train_cfg)
            self.encoder = Encoder(encoder_config_obj, base_to_use=self.shared_base_model)
            if isinstance(different_activations_orig, str) and different_activations_orig == self.model_name:
                different_activations_orig = None
                print("Different activations orig is the same as the model name, setting to None")
            if different_activations_orig is not None:
                if isinstance(different_activations_orig, str):
                    diff_model_str = different_activations_orig
                    if ("gemma-2-9b-it" in diff_model_str and "google/gemma-2-9b-it" not in diff_model_str) and ('abliterated' not in diff_model_str):
                        print(f"Loading {diff_model_str} as gemma-2-9b-it, then loading adapter")
                        if "gemma-3" in diff_model_str:
                            print("Should never happen")
                            different_activations_model = Gemma3ForCausalLM.from_pretrained(
                                "google/gemma-3-27b-it",
                                device_map="auto" if not initialise_on_cpu else "cpu",
                                torch_dtype=self.model_dtype,
                                attn_implementation=self.config.get("attn_implementation", None),
                            )
                        else:
                            different_activations_model = AutoModelForCausalLM.from_pretrained(
                                "google/gemma-2-9b-it",
                                device_map="auto" if not initialise_on_cpu else "cpu",
                                torch_dtype=self.model_dtype,
                                attn_implementation=self.config.get("attn_implementation", None),
                            )
                        print(f"Loading adapter {diff_model_str} into {different_activations_model}")
                        different_activations_model.load_adapter(diff_model_str)
                    elif "ModelOrganismsForEM" in diff_model_str and "full-ft" not in diff_model_str:
                        print("Loading with adapter")
                        different_activations_model = AutoModelForCausalLM.from_pretrained(
                            "Qwen/Qwen2.5-14B-Instruct",
                            device_map="auto" if not initialise_on_cpu else "cpu",
                            torch_dtype=self.model_dtype,
                            attn_implementation=self.config.get("attn_implementation", None),
                        )
                        print(f"Loading adapter {diff_model_str} into {different_activations_model}")
                        different_activations_model.load_adapter(diff_model_str)
                    else:
                        print(f"Loading {different_activations_orig} (no lora)")
                        if "gemma-3" in different_activations_orig:
                            different_activations_model = Gemma3ForCausalLM.from_pretrained(
                                different_activations_orig,
                                device_map="auto" if not initialise_on_cpu else "cpu",
                                torch_dtype=self.model_dtype,
                                attn_implementation=self.config.get("attn_implementation", None),
                            )
                        else:
                            different_activations_model = AutoModelForCausalLM.from_pretrained(
                                different_activations_orig,
                                device_map="auto" if not initialise_on_cpu else "cpu",
                                torch_dtype=self.model_dtype,
                                attn_implementation=self.config.get("attn_implementation", None),
                            )
                    lora_name = (
                        different_activations_orig if hasattr(different_activations_model, "peft_config") else None
                    )
                else:
                    different_activations_model = different_activations_orig.model
                    lora_name = (
                        different_activations_orig.lora_name
                        if hasattr(different_activations_orig, "lora_name")
                        else None
                    )
                    print(f"Different activations orig is a PEFT model, lora_name: {lora_name}")
                    diff_model_str = None
                different_activations_model.eval()
                self.orig_model = OrigWrapper(
                    different_activations_orig,
                    load_in_8bit=False,
                    base_to_use=different_activations_model,
                    lora_name=lora_name,
                    torch_dtype=self.model_dtype,
                )
            else:
                self.orig_model = OrigWrapper(
                    self.model_name,
                    load_in_8bit=False,
                    base_to_use=self.shared_base_model,
                    torch_dtype=self.model_dtype,
                )
                lora_name = None
                print('No different activations model for origwrapper, using shared base model')

            self.lora_name = lora_name
            self.orig_model_name = "N/A"
            if (
                hasattr(self.orig_model, "model")
                and hasattr(self.orig_model.model, "config")
                and hasattr(self.orig_model.model.config, "_name_or_path")
            ):
                self.orig_model_name = self.orig_model.model.config._name_or_path + (
                    "LORA" + "_" + lora_name if lora_name is not None else ""
                )

            # Initialize Decoder prompt (before loading weights)
            if "decoder_prompt" in self.config and self.config["decoder_prompt"]:
                print(f'Setting decoder prompt: "{self.config["decoder_prompt"]}"')
                self.decoder.set_prompt(self.config["decoder_prompt"], self.tokenizer)

            # Initialize Encoder soft prompt (before loading weights)
            if encoder_config_obj.soft_prompt_init_text:
                print(f'Setting encoder soft prompt from text: "{encoder_config_obj.soft_prompt_init_text}"')
                self.encoder.set_soft_prompt_from_text(encoder_config_obj.soft_prompt_init_text, self.tokenizer)
            elif encoder_config_obj.soft_prompt_length > 0:
                print(
                    f"Encoder using randomly initialized soft prompt of length {encoder_config_obj.soft_prompt_length}."
                )

            # Move models to device
            self.decoder.to(self.device)
            self.encoder.to(self.device)
            if not no_orig:
                self.orig_model.to(self.device)
                self.orig_model.model.to(self.device)
            self.decoder.eval()
            self.encoder.eval()
            self.orig_model.model.eval()

            if self.decoder.config.use_kv_cache:
                print("Using KV cache")
            else:
                print("Not using KV cache")


            # Load weights using CheckpointManager
            if checkpoint_path is not None and not do_not_load_weights:
                from lens.utils.checkpoint_manager import CheckpointManager

                # Minimal checkpoint config for CheckpointManager
                checkpoint_config = {
                    "checkpoint": {
                        "enabled": True,
                        "base_output_dir": str(self.checkpoint_path.parent.parent),
                        "output_dir": str(self.checkpoint_path.parent),
                        "strict_load": strict_load,
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
                        map_location="cpu",
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
                logger.info(
                    f"Attempting to load comparison TunedLens from: {comparison_tl_checkpoint} for model {self.model_name}"
                )
                try:
                    # Load to CPU first, then move to device
                    loaded_tl = load_full_tuned_lens(
                        model_or_model_name=self.model_name,
                        checkpoint_path_or_name=comparison_tl_checkpoint
                        if isinstance(comparison_tl_checkpoint, str)
                        else None,
                        device="cpu",
                        log=logger,
                        is_main_process=True,  # In notebook, we act as main process
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
                print(
                    f"✓ Using shared base model - saved ~{shared_base_model.num_parameters() * 2 / 1e9:.1f}GB of GPU memory"
                )

    def to(self, device):
        """
        Move all relevant models and tensors to the specified device.
        Ensures proper synchronization for multi-GPU setups.
        """
        self.device = torch.device(device)
        
        # Move decoder and its components
        if hasattr(self, "decoder") and self.decoder is not None:
            self.decoder.to(device)
            # Note: decoder.device is a read-only property that returns the device of its parameters
        
        # Move encoder and its components  
        if hasattr(self, "encoder") and self.encoder is not None:
            self.encoder.to(device)
            # Note: encoder.device is a read-only property if it exists
        
        # Move original model
        if hasattr(self, "orig_model") and self.orig_model is not None:
            self.orig_model.to(device)
        
        # Move comparison tuned lens if present
        if hasattr(self, "comparison_tuned_lens") and self.comparison_tuned_lens is not None:
            self.comparison_tuned_lens.to(device)
        
        # Note: shared_base_model is moved via encoder/decoder/orig_model's to() methods
        # so we don't need to move it separately here
        
        # Clear CUDA cache if moving between CUDA devices
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        
        return self
    
    def get_current_device(self):
        """
        Get the current device of the analyzer by checking the actual model parameters.
        This is more reliable than self.device in multi-GPU setups.
        """
        if hasattr(self, "decoder") and self.decoder is not None:
            return next(self.decoder.parameters()).device
        elif hasattr(self, "encoder") and self.encoder is not None:
            return next(self.encoder.parameters()).device
        elif hasattr(self, "orig_model") and self.orig_model is not None:
            return next(self.orig_model.model.parameters()).device
        else:
            return self.device

    def swap_orig_model(self, new_model: str):
        del self.orig_model
        if new_model.startswith("google/gemma-2-9b-it") and new_model != "google/gemma-2-9b-it":
            new_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", attn_implementation=self.config.get("attn_implementation", None))
            new_model.load_adapter(new_model)
        else:
            new_model = AutoModelForCausalLM.from_pretrained(new_model, attn_implementation=self.config.get("attn_implementation", None))
        self.orig_model = OrigWrapper(new_model, load_in_8bit=False, base_to_use=self.shared_base_model)

    # def analyze_all_tokens(self, text: str, seed=42, batch_size=None, no_eval=False, tuned_lens: bool = True, add_tokens = None, replace_left=None, replace_right=None, do_hard_tokens=False, return_structured=False, move_devices=False, logit_lens_analysis: bool = False, temperature: float = 1.0, no_kl: bool = False, calculate_token_salience: bool = False, optimize_explanations_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    #     """Analyze all tokens in the text and return results as DataFrame.

    #     Args:
    #         text: Text to analyze
    #         seed: Random seed
    #         batch_size: Batch size for processing
    #         no_eval: Skip evaluation metrics (KL, MSE)
    #         tuned_lens: Include TunedLens predictions in results
    #         logit_lens_analysis: Add logit-lens predictions (projecting hidden state through unembedding).
    #         temperature: Sampling temperature for explanation generation.
    #         calculate_token_salience: If True, calculate a salience score for each token in the explanation.
    #         optimize_explanations_config: If provided, optimize explanations using the given configuration.
    #     """
    #     if batch_size is None:
    #         batch_size = self.default_batch_size

    #     # Tokenize
    #     inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    #     input_ids = inputs.input_ids.clone()
    #     attention_mask = inputs.attention_mask.clone()
    #     attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
    #     if inputs.input_ids[0][0]==inputs.input_ids[0][1]:
    #         print(f"First two tokens are the same: {self.tokenizer.decode(inputs.input_ids[0][0])}, {self.tokenizer.decode(inputs.input_ids[0][1])}, removing first token")
    #         input_ids = input_ids[:, 1:].clone()
    #         attention_mask = attention_mask[:, 1:].clone()
    #     input_ids = input_ids.to(self.device)
    #     seq_len = input_ids.shape[1]
    #     attention_mask = attention_mask.to(self.device)

    #     # Get all hidden states
    #     #self.decoder.to('cpu')
    #     #self.encoder.to('cpu')
    #     # Use the proper OrigWrapper method to get activations
    #     assert input_ids.shape[1]==attention_mask.shape[1], f"input_ids.shape[1]={input_ids.shape[1]} != attention_mask.shape[1]={attention_mask.shape[1]}"
    #     if move_devices and self.orig_model.model!=self.shared_base_model:
    #         print("Moving shared base model to CPU")
    #         self.shared_base_model.to('cpu')
    #         self.orig_model.model.to('cuda')

    #     A_full_sequence = self.orig_model.get_all_activations_at_layer(
    #         input_ids,
    #         self.layer,
    #         attention_mask=attention_mask,
    #         no_grad=True,
    #     )

    #     torch.manual_seed(seed)

    #     all_logit_lens_predictions = []
    #     if logit_lens_analysis:
    #         unembedding_layer = self.orig_model.model.get_output_embeddings()
    #         logits_logit_lens = unembedding_layer(A_full_sequence)
    #         top_tokens_batch = [
    #             " ".join(get_top_n_tokens(logits_logit_lens[i], self.tokenizer, min(self.t_text, 10)))
    #             for i in range(logits_logit_lens.shape[0])
    #         ]
    #         all_logit_lens_predictions.extend(top_tokens_batch)
    #     if move_devices and self.orig_model.model!=self.shared_base_model:
    #         print("Moving orig model to CPU")
    #         self.orig_model.model.to('cpu')
    #         self.shared_base_model.to('cuda')
    #     # A_full_sequence: (seq_len, hidden_dim)

    #     all_kl_divs = []
    #     all_mses = []
    #     all_relative_rmses = []
    #     all_gen_hard_token_ids = []
    #     all_tuned_lens_predictions = [] if tuned_lens else None
    #     all_salience_scores = [] if calculate_token_salience and not no_eval else None

    #     if calculate_token_salience and not no_eval:
    #         space_token_id = self.tokenizer.encode(' ', add_special_tokens=False)
    #         if not space_token_id:
    #             null_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
    #             print(f"Using fallback null token (pad/eos): {null_token_id}")
    #         else:
    #             null_token_id = space_token_id[0]

    #     with torch.no_grad():
    #         # Use tqdm if available
    #         try:
    #             iterator = tqdm.tqdm(range(0, seq_len, batch_size), desc="Analyzing tokens in batches" + (" with tokens `" + self.tokenizer.decode(add_tokens) + "`" if add_tokens is not None else ""))
    #         except ImportError:
    #             iterator = range(0, seq_len, batch_size)

    #         for i in iterator:
    #             batch_start = i
    #             batch_end = min(i + batch_size, seq_len)
    #             current_batch_size = batch_end - batch_start

    #             token_positions_batch = torch.arange(batch_start, batch_end, device=self.device)
    #             A_batch = A_full_sequence[token_positions_batch]

    #             A_hat_batch = None
    #             logits_orig_full = None
    #             logits_recon_full = None

    #             with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
    #                 # Generate explanations for all positions in the current batch.
    #                 if self.decoder.config.use_kv_cache:
    #                     gen = self.decoder.generate_soft_kv_cached_nondiff(A_batch, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=token_positions_batch, return_logits=True, add_tokens=add_tokens, hard_left_emb=replace_left, hard_right_emb=replace_right, temperature=temperature)
    #                 else:
    #                     gen = self.decoder.generate_soft(A_batch, max_length=self.t_text, gumbel_tau=self.tau, original_token_pos=token_positions_batch, return_logits=True, add_tokens=add_tokens, hard_left_emb=replace_left, hard_right_emb=replace_right, temperature=temperature)

    #                 all_gen_hard_token_ids.append(gen.hard_token_ids)

    #                 if not no_eval:
    #                     # Reconstruct activations in a batch from the generated hard tokens.
    #                     generated_token_embeddings = self.encoder.base.get_input_embeddings()(gen.hard_token_ids)
    #                     current_token_ids_batch = input_ids.squeeze(0)[token_positions_batch]
    #                     A_hat_batch = self.encoder(
    #                         generated_token_embeddings,
    #                         original_token_pos=token_positions_batch,
    #                         current_token_ids=current_token_ids_batch if self.encoder.config.add_current_token else None
    #                     )

    #                     # Compute KL divergence in a batch using the vectorized forward pass.
    #                     # The input_ids tensor needs to be expanded to match the batch size of activations.
    #                     input_ids_batch = input_ids.expand(current_batch_size, -1)  # Shape: (current_batch_size, seq_len)

    #                     # Get original logits by replacing activations at each position.
    #                     if not no_kl:
    #                         logits_orig_full = self.orig_model.forward_with_replacement_vectorized(
    #                             input_ids=input_ids_batch,
    #                             new_activations=A_batch,
    #                             layer_idx=self.layer,
    #                             token_positions=token_positions_batch,
    #                             no_grad=True
    #                         ).logits

    #                         # Get reconstructed logits similarly.
    #                         logits_recon_full = self.orig_model.forward_with_replacement_vectorized(
    #                             input_ids=input_ids_batch,
    #                             new_activations=A_hat_batch,
    #                             layer_idx=self.layer,
    #                             token_positions=token_positions_batch,
    #                             no_grad=True
    #                         ).logits

    #                 # Compute TunedLens predictions in batch if requested
    #                 if tuned_lens and self.comparison_tuned_lens is not None:
    #                     try:
    #                         # Cast to float32 for TunedLens
    #                         A_batch_f32 = A_batch.to(torch.float32)
    #                         logits_tuned_lens = self.comparison_tuned_lens(A_batch_f32, idx=self.layer)
    #                         # logits_tuned_lens: (batch, vocab)
    #                         top_tokens_batch = [
    #                             " ".join(get_top_n_tokens(logits_tuned_lens[i], self.tokenizer, min(self.t_text, 10)))
    #                             for i in range(logits_tuned_lens.shape[0])
    #                         ]
    #                         all_tuned_lens_predictions.extend(top_tokens_batch)
    #                     except Exception as e:
    #                         # If batch fails, fill with error messages
    #                         all_tuned_lens_predictions.extend([f"[TL error: {str(e)[:30]}...]"] * current_batch_size)

    #             if not no_eval:
    #                 # Extract logits at the specific token positions for KL calculation.
    #                 batch_indices = torch.arange(current_batch_size, device=self.device)
    #                 if not no_kl:
    #                     logits_orig_at_pos = logits_orig_full[batch_indices, token_positions_batch]
    #                     logits_recon_at_pos = logits_recon_full[batch_indices, token_positions_batch]

    #                     # Compute KL with numerical stability (batched, but without AMP for precision).
    #                     with torch.amp.autocast('cuda', enabled=False):
    #                         logits_orig_f32 = logits_orig_at_pos.float()
    #                         logits_recon_f32 = logits_recon_at_pos.float()

    #                         # Normalize logits for stability.
    #                         logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0]
    #                         logits_recon_f32 = logits_recon_f32 - logits_recon_f32.max(dim=-1, keepdim=True)[0]

    #                         log_probs_orig = torch.log_softmax(logits_orig_f32, dim=-1)
    #                         log_probs_recon = torch.log_softmax(logits_recon_f32, dim=-1)
    #                         probs_orig = torch.exp(log_probs_orig)

    #                         # kl_divs_batch will have shape (current_batch_size,).
    #                         kl_divs_batch = (probs_orig * (log_probs_orig - log_probs_recon)).sum(dim=-1)
    #                         all_kl_divs.append(kl_divs_batch)

    #                 # MSE for comparison (batched).
    #                 # mses_batch will have shape (current_batch_size,).
    #                 mses_batch = torch.nn.functional.mse_loss(A_batch, A_hat_batch, reduction='none').mean(dim=-1)
    #                 all_mses.append(mses_batch)

    #                 # Relative RMSE: sqrt(MSE(A, A_hat) / MSE(A, 0)). This is a scale-invariant error metric.
    #                 # This is equivalent to ||A - A_hat|| / ||A|| (relative L2 error).
    #                 A_norm_sq_mean = (A_batch**2).mean(dim=-1)
    #                 relative_rmse_batch = (mses_batch / (A_norm_sq_mean + 1e-9))
    #                 all_relative_rmses.append(relative_rmse_batch)

    #                 # Token salience calculation
    #                 if calculate_token_salience:
    #                     # This is a batched salience calculation, iterating through each explanation in the batch.
    #                     salience_scores_list = []
    #                     for j in range(current_batch_size):
    #                         salience_scores_list.append(self._calculate_salience_batch(
    #                             gen.hard_token_ids[j],
    #                             mses_batch[j].item(),
    #                             A_batch[j,:],
    #                             token_positions_batch[j].item(),
    #                             input_ids
    #                     ))
    #                     salience_scores = salience_scores_list
    #                 else:
    #                     salience_scores = [None] * current_batch_size

    #     # Concatenate results from all batches
    #     if not no_eval:
    #         mses = torch.cat(all_mses)
    #         relative_rmses = torch.cat(all_relative_rmses)
    #         if not no_kl:
    #             kl_divs = torch.cat(all_kl_divs)
    #         else:
    #             kl_divs = torch.zeros(seq_len)
    #         if calculate_token_salience:
    #             salience_scores = torch.cat(all_salience_scores)
    #     elif no_eval:
    #         kl_divs = torch.zeros(seq_len)
    #         mses = torch.zeros(seq_len)
    #         relative_rmses = torch.zeros(seq_len)
    #     gen_hard_token_ids = torch.cat(all_gen_hard_token_ids)

    #     # Decode and collect results into a list of dicts.
    #     results = []
    #     # The original code had a `gen` object, we now use the concatenated `gen_hard_token_ids`
    #     for pos in range(seq_len):
    #         explanation = self.tokenizer.decode(gen_hard_token_ids[pos], skip_special_tokens=False) + ("[" + self.tokenizer.decode([input_ids[0, pos].item()]) +"]" if self.encoder.config.add_current_token else "")
    #         token = self.tokenizer.decode([input_ids[0, pos].item()])
    #         explanation_structured = [self.tokenizer.decode(gen_hard_token_ids[pos][i], skip_special_tokens=False) for i in range(len(gen_hard_token_ids[pos]))] + (["[" + self.tokenizer.decode([input_ids[0, pos].item()]) +"]"] if self.encoder.config.add_current_token else [])

    #         result_dict = {
    #             'position': pos,
    #             'token': token,
    #             'explanation': explanation,
    #             'kl_divergence': kl_divs[pos].item(),
    #             'mse': mses[pos].item(),
    #             'relative_rmse': relative_rmses[pos].item()
    #         }

    #         # Add TunedLens predictions if available
    #         if tuned_lens and all_tuned_lens_predictions:
    #             result_dict['tuned_lens_top'] = all_tuned_lens_predictions[pos]
    #         elif tuned_lens and self.comparison_tuned_lens is None:
    #             result_dict['tuned_lens_top'] = "[TunedLens not loaded]"

    #         if logit_lens_analysis and all_logit_lens_predictions:
    #             result_dict['logit_lens_top'] = all_logit_lens_predictions[pos]

    #         if return_structured:
    #             result_dict['explanation_structured'] = explanation_structured

    #         if all_salience_scores is not None:
    #             result_dict['token_salience'] = salience_scores[pos].tolist()

    #         results.append(result_dict)

    #     return LensDataFrame(
    #         results,
    #         checkpoint_path=self.checkpoint_path,
    #         model_name=self.model_name,
    #         orig_model_name=self.orig_model_name
    # )
    def _get_mse_for_explanation_batch(
        self,
        explanation_ids: torch.Tensor,
        A_target: torch.Tensor,
        position_to_analyze: int,
        input_ids: torch.Tensor,
        tensor_As=None,
    ) -> torch.Tensor:
        """Helper to calculate MSE for a batch of explanations against a single target activation."""
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            explanation_embeddings = self.encoder.base.get_input_embeddings()(explanation_ids)

            batch_size = explanation_ids.shape[0]
            # CRITICAL FIX: Use the actual device of the tensor, not self.device
            tensor_device = explanation_ids.device
            if isinstance(position_to_analyze, int):
                original_token_pos = torch.full((batch_size,), position_to_analyze, device=tensor_device)
            else:
                # Ensure position tensor is on the same device
                original_token_pos = position_to_analyze.to(tensor_device)
            # The input_ids are expected to be of shape (1, seq_len)
            if isinstance(position_to_analyze, int):
                current_token_id = (
                    input_ids.squeeze(0)[position_to_analyze] if self.encoder.config.add_current_token else None
                )
                if current_token_id is not None:
                    current_token_id = current_token_id.expand(batch_size)
            else:
                # When position_to_analyze is a tensor, handle each position in a vectorized way
                if self.encoder.config.add_current_token:
                    # Ensure position_to_analyze is a 1D tensor of indices
                    # input_ids: (1, seq_len), position_to_analyze: (batch_size,)
                    # Use torch.gather to efficiently select the correct token for each position
                    current_token_id = torch.gather(input_ids[0], 0, position_to_analyze)
                else:
                    current_token_id = None
            A_hat = self.encoder(
                explanation_embeddings, original_token_pos=original_token_pos, current_token_ids=current_token_id
            )
            if tensor_As is not None:
                # Ensure tensor_As is on the same device as A_hat
                A_target_expanded = tensor_As.to(A_hat.device)
            else:
                # Ensure A_target is on the same device as A_hat
                A_target_expanded = A_target.to(A_hat.device).expand_as(A_hat)
            mses = torch.nn.functional.mse_loss(A_target_expanded, A_hat, reduction="none").mean(dim=-1)
            return mses

    def _calculate_salience_batch(
        self,
        explanation_ids: torch.Tensor,
        base_mse: float,
        A_target: torch.Tensor,
        position_to_analyze: int,
        input_ids: torch.Tensor,
        just_idx: int = None,
    ) -> List[float]:
        """Helper to calculate salience for a single explanation, but batched for efficiency."""
        space_token_id = self.tokenizer.encode(" ", add_special_tokens=False)
        # We use a space token for ablation as it's generally more neutral than a padding token.
        # If no space token is found, we fall back to the EOS token.
        null_token_id = space_token_id[0] if space_token_id else self.tokenizer.eos_token_id
        
        # Create a batch of all ablated sequences by replacing one token at a time with the null token.
        ablated_ids_batch = explanation_ids.repeat(len(explanation_ids), 1)
        ablated_ids_batch[torch.arange(len(explanation_ids)), torch.arange(len(explanation_ids))] = null_token_id
        if just_idx is not None:
            ablated_ids_batch = ablated_ids_batch[just_idx : just_idx + 1, :]

        # Calculate MSE for all ablated sequences in one batch
        ablated_mses = self._get_mse_for_explanation_batch(ablated_ids_batch, A_target, position_to_analyze, input_ids)

        # Salience is the percentage increase in MSE when a token is ablated.
        return ((ablated_mses - base_mse) / (base_mse + 1e-9)).tolist()

    def _optimize_batch_best_of_k(
        self,
        A_targets: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        num_rollouts_per_position: int = 8,
        temperature: float = 1.0,
        n_groups_per_rollout: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]]]:
        """
        Simplified batched version for best-of-k explanation optimization.
        Processes multiple positions at once, each getting num_rollouts_per_position rollouts.

        Args:
            A_targets: Target activations for each position (batch_size, hidden_dim)
            positions: Token positions being analyzed (batch_size,)
            input_ids: Input token IDs (1, seq_len)
            num_rollouts_per_position: Number of rollouts per position (default: 8)
            temperature: Sampling temperature
            n_groups_per_rollout: How many positions to process at once (default: 8)

        Returns:
            best_explanations: Best explanation tokens for each position (batch_size, t_text)
            best_mses: Best MSE for each position (batch_size,)
            salience_scores: Salience scores for each position's best explanation
        """
        total_positions = A_targets.shape[0]
        all_best_explanations = []
        all_best_mses = []
        all_salience_scores = []
        if n_groups_per_rollout is None:
            n_groups_per_rollout = total_positions
        
        # Ensure A_targets is on the same device as the decoder
        decoder_device = next(self.decoder.parameters()).device
        A_targets = A_targets.to(decoder_device)
        positions = positions.to(decoder_device)
        input_ids = input_ids.to(decoder_device)

        # Process positions in sub-batches to manage memory
        for start_idx in tqdm.tqdm(range(0, total_positions, n_groups_per_rollout), leave=False, desc=f"Optimizing explanations with k={num_rollouts_per_position}, this batch has {n_groups_per_rollout} tokens (total {num_rollouts_per_position * n_groups_per_rollout} rollouts out of {num_rollouts_per_position * total_positions})"):
            end_idx = min(start_idx + n_groups_per_rollout, total_positions)
            batch_size = end_idx - start_idx

            A_batch = A_targets[start_idx:end_idx]
            positions_batch = positions[start_idx:end_idx]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                # Generate all rollouts for all positions in this sub-batch at once
                A_expanded = A_batch.repeat_interleave(num_rollouts_per_position, dim=0)
                positions_expanded = positions_batch.repeat_interleave(num_rollouts_per_position)

                # Generate explanations for all rollouts
                gen_result = self.decoder.generate_soft_kv_cached_nondiff(
                    A_expanded,
                    max_length=self.t_text,
                    gumbel_tau=self.tau,
                    original_token_pos=positions_expanded,
                    temperature=temperature,
                )
                all_explanations = gen_result.hard_token_ids  # (batch_size * num_rollouts, t_text)

                # Calculate MSEs for all explanations in parallel
                all_mses_flat = self._get_mse_for_explanation_batch(
                    all_explanations, A_expanded, positions_expanded, input_ids, tensor_As=A_expanded
                )

                # Reshape MSEs to (batch_size, num_rollouts_per_position)
                all_mses = all_mses_flat.view(batch_size, num_rollouts_per_position)

                # Find best explanation for each position
                best_indices = torch.argmin(all_mses, dim=1)

                # Collect best explanations and MSEs
                best_explanations_batch = []
                best_mses_batch = []
                for i in range(batch_size):
                    best_idx = best_indices[i].item()
                    global_idx = i * num_rollouts_per_position + best_idx
                    
                    best_explanations_batch.append(all_explanations[global_idx])
                    best_mses_batch.append(all_mses[i, best_idx].item())
                    all_best_explanations.append(all_explanations[global_idx])
                    all_best_mses.append(all_mses[i, best_idx].item())
                
                # Stack best explanations for this batch
                best_explanations_tensor = torch.stack(best_explanations_batch)  # (batch_size, t_text)
                
                # Calculate salience for all positions in batch at once
                batch_salience_scores = self._calculate_salience_batch_for_multiple(
                    best_explanations_tensor,
                    best_mses_batch,
                    A_batch,
                    positions_batch,
                    input_ids
                )
                all_salience_scores.extend(batch_salience_scores)

        # Stack results
        best_explanations = torch.stack(all_best_explanations)
        # CRITICAL FIX: Use the device of the explanations tensor, not self.device
        best_mses = torch.tensor(all_best_mses, device=best_explanations.device)

        return best_explanations, best_mses, all_salience_scores

    def _calculate_salience_batch_for_multiple(
        self,
        explanation_ids_batch: torch.Tensor,  # (batch_size, seq_len)
        base_mses: List[float],
        A_targets: torch.Tensor,  # (batch_size, hidden_dim)
        positions: torch.Tensor,  # (batch_size,)
        input_ids: torch.Tensor,
    ) -> List[List[float]]:
        """Calculate salience for multiple explanations in a single batch."""
        batch_size, seq_len = explanation_ids_batch.shape
        
        # Get null token
        space_token_id = self.tokenizer.encode(" ", add_special_tokens=False)
        null_token_id = space_token_id[0] if space_token_id else self.tokenizer.eos_token_id
        
        all_salience_scores = []
        
        # Process tokens in batches
        for token_idx in range(seq_len):
            # Create ablated sequences for all positions at once
            ablated_batch = explanation_ids_batch.clone()
            ablated_batch[:, token_idx] = null_token_id
            
            # Calculate MSEs for all ablated sequences in one forward pass
            ablated_mses = self._get_mse_for_explanation_batch(
                ablated_batch, 
                A_targets, 
                positions, 
                input_ids,
                tensor_As=A_targets
            )
            
            # Store results
            if token_idx == 0:
                # Initialize lists for each position
                for i in range(batch_size):
                    all_salience_scores.append([])
            
            # Calculate salience for each position
            for i in range(batch_size):
                salience = (ablated_mses[i].item() - base_mses[i]) / base_mses[i]
                all_salience_scores[i].append(salience)
        
        return all_salience_scores

    def optimize_explanation_for_mse(
        self,
        text: str,
        position_to_analyze: int,
        verbose: bool = True,
        A_target: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        best_of_k: bool = False,
        duplicate_prefix: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """
        Standalone function for interactive use. It performs N initial 'rollouts'
        to find a good starting explanation, and then calls the worker to optimize it.
        """
        # 1. Setup from text and position
        if input_ids is None:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = inputs.input_ids.to(self.device)
        else:
            input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        if position_to_analyze < 0:
            position_to_analyze = seq_len + position_to_analyze
        if not (0 <= position_to_analyze < seq_len):
            raise ValueError(
                f"Position to analyze {position_to_analyze} is out of bounds for sequence length {seq_len}"
            )

        with torch.no_grad():
            if A_target is None:
                A_target, _ = self.orig_model.get_activations_at_positions(
                    input_ids=input_ids,
                    layer_idx=self.layer,
                    # Use current device instead of self.device
                    token_positions=torch.tensor([position_to_analyze], device=self.get_current_device()),
                )
                A_target = A_target.squeeze(0)
        temperature = kwargs.get("temperature", 1.0)
        num_samples = kwargs.get("num_samples_per_iteration", 16)

        if best_of_k:
            # Perform k rollouts in batches, find the best, and return without optimization.
            rollout_batch_size = num_samples  # A fixed batch size to prevent OOM.

            if verbose:
                print(
                    f"Performing {num_samples} rollouts in batches of {rollout_batch_size} to find best explanation..."
                )

            all_explanations = []
            all_mses = []

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                for i in range(0, best_of_k, rollout_batch_size):
                    current_batch_size = min(rollout_batch_size, best_of_k - i)
                    if verbose and num_samples > rollout_batch_size:
                        print(
                            f"  Rollout batch {i // rollout_batch_size + 1}/{(num_samples + rollout_batch_size - 1) // rollout_batch_size}, size {current_batch_size}"
                        )

                    if not duplicate_prefix:
                        A_batch = A_target.unsqueeze(0).repeat(current_batch_size, 1)
                        # Use current device instead of self.device
                        token_pos_batch = torch.full((current_batch_size,), position_to_analyze, device=self.get_current_device())
                        gen_result = self.decoder.generate_soft_kv_cached_nondiff(
                            A_batch,
                            max_length=self.t_text,
                            gumbel_tau=self.tau,
                            original_token_pos=token_pos_batch,
                            temperature=temperature,
                        )
                    elif duplicate_prefix:
                        A_batch = A_target.unsqueeze(0)
                        # Use current device instead of self.device
                        token_pos_batch = torch.full((1,), position_to_analyze, device=self.get_current_device())
                        gen_result = self.decoder.generate_k(
                            A_batch,
                            max_length=self.t_text,
                            gumbel_tau=self.tau,
                            original_token_pos=token_pos_batch,
                            temperature=temperature,
                            group_size=current_batch_size,
                            return_logits=False,
                        )
                    batch_explanations = gen_result.hard_token_ids
                    batch_mses = self._get_mse_for_explanation_batch(
                        batch_explanations, A_target, position_to_analyze, input_ids
                    )

                    all_explanations.append(batch_explanations)
                    all_mses.append(batch_mses)

            all_explanations = torch.cat(all_explanations, dim=0)
            all_mses = torch.cat(all_mses, dim=0).tolist()

            best_mse = min(all_mses)
            best_explanation_tokens = all_explanations[all_mses.index(best_mse)].clone()

            # Calculate salience for the best explanation
            final_salience_scores = self._calculate_salience_batch(
                best_explanation_tokens, best_mse, A_target, position_to_analyze, input_ids
            )

            if verbose:
                print(
                    f"Best explanation from rollouts: '{self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)}' (MSE: {best_mse:.4f}, mean: {torch.mean(torch.tensor(all_mses)):.4f}, max: {max(all_mses):.4f}), salience scores: ({', '.join([f'[{s:.2f}]' for s in final_salience_scores])})"
                )
            return best_explanation_tokens, best_mse, final_salience_scores
        else:
            # 2. Perform initial best-of-N rollouts to find a strong starting candidate.
            if verbose:
                print(f"Finding initial best explanation from {num_samples} rollouts...")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                A_batch = A_target.unsqueeze(0).repeat(num_samples, 1)
                # Use current device instead of self.device
                token_pos_batch = torch.full((num_samples,), position_to_analyze, device=self.get_current_device())
                gen_func = self.decoder.generate_soft_kv_cached_nondiff
                gen_result = gen_func(
                    A_batch,
                    max_length=self.t_text,
                    gumbel_tau=self.tau,
                    original_token_pos=token_pos_batch,
                    temperature=temperature,
                )
                initial_explanations = gen_result.hard_token_ids

            initial_mses = self._get_mse_for_explanation_batch(
                initial_explanations, A_target, position_to_analyze, input_ids
            ).tolist()
            best_initial_mse = min(initial_mses)
            best_initial_explanation = initial_explanations[initial_mses.index(best_initial_mse)].clone()
            if verbose:
                print(
                    f"Best initial explanation found: '{self.tokenizer.decode(best_initial_explanation)}' (MSE: {best_initial_mse:.4f}), of average {torch.mean(torch.tensor(initial_mses)):.4f} and max {max(initial_mses):.4f}"
                )

            # 3. Call the internal worker with the single best candidate from the rollouts.
            # Force verbose=True for standalone interactive use.
            return self._optimize_explanation_worker(
                A_target=A_target,
                initial_explanation_tokens=best_initial_explanation,
                position_to_analyze=position_to_analyze,
                input_ids=input_ids,
                verbose=verbose,
                **kwargs,
            )

    def _optimize_explanation_worker(
        self,
        A_target: torch.Tensor,
        initial_explanation_tokens: torch.Tensor,
        position_to_analyze: int,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """
        Worker function to optimize a single explanation. Assumes an initial explanation is provided
        and does NOT perform the initial N rollouts itself.
        """
        salience_pct_threshold = kwargs.get("salience_pct_threshold", 0.0)
        temperature = kwargs.get("temperature", 1.0)
        max_temp_increases = kwargs.get("max_temp_increases", 2)
        temp_increase_factor = kwargs.get("temp_increase_factor", 1.5)
        num_samples_per_iteration = kwargs.get("num_samples_per_iteration", 16)
        verbose = kwargs.get("verbose", False)

        # 'best' variables track the global best. 'current' variables are for the working copy.
        best_explanation_tokens = initial_explanation_tokens.clone()
        best_mse = self._get_mse_for_explanation_batch(
            best_explanation_tokens.unsqueeze(0), A_target, position_to_analyze, input_ids
        ).item()

        current_explanation_tokens = best_explanation_tokens.clone()
        current_mse = best_mse
        if verbose:
            orig_saliences = self._calculate_salience_batch(
                best_explanation_tokens, best_mse, A_target, position_to_analyze, input_ids
            )
            # print(f"Orig. saliences: {[f'{s:.2f}' for s in orig_saliences]}")

        if verbose:
            print(
                f"Starting optimization with: '{self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)}' (MSE: {best_mse:.4f})"
            )

        for current_idx in range(self.t_text):
            if verbose:
                print(f"\n--- Processing index {current_idx}/{self.t_text - 1} ---")

            # Salience is calculated on the current working explanation to decide where to work next
            current_salience = self._calculate_salience_batch(
                current_explanation_tokens, current_mse, A_target, position_to_analyze, input_ids, just_idx=current_idx
            )[0]

            if verbose:
                print(
                    f"Salience at index {current_idx}: {current_salience:.4f}. Current explanation: '{self.tokenizer.decode(current_explanation_tokens)}'"
                )

            if current_salience >= salience_pct_threshold:
                if verbose:
                    print(
                        f"Salience is above threshold ({salience_pct_threshold}). Skipping optimization for this token."
                    )
                continue

            prefix_tokens = current_explanation_tokens[:current_idx]
            found_improvement_in_retries = False
            current_temperature = temperature

            for retry_attempt in range(max_temp_increases + 1):
                if verbose:
                    print(
                        f"Generation attempt {retry_attempt + 1}/{max_temp_increases + 1} with temp {current_temperature:.2f}"
                    )
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    A_batch = A_target.unsqueeze(0).repeat(num_samples_per_iteration, 1)
                    token_pos_batch = torch.full((num_samples_per_iteration,), position_to_analyze, device=self.device)
                    gen_func = self.decoder.generate_soft_kv_cached_nondiff

                    num_tokens_to_gen = self.t_text - len(prefix_tokens)
                    if num_tokens_to_gen <= 0:
                        if verbose:
                            print("Prefix is already full length. Cannot generate more tokens.")
                        break

                    new_gen_result = gen_func(
                        A_batch,
                        max_length=num_tokens_to_gen,
                        gumbel_tau=self.tau,
                        original_token_pos=token_pos_batch,
                        temperature=current_temperature,
                        add_tokens=prefix_tokens,
                    )

                    new_candidate_tokens = new_gen_result.hard_token_ids
                    if len(prefix_tokens) > 0:
                        new_candidate_tokens = torch.cat(
                            [prefix_tokens.expand(num_samples_per_iteration, -1), new_candidate_tokens], dim=1
                        )

                new_mses = self._get_mse_for_explanation_batch(
                    new_candidate_tokens, A_target, position_to_analyze, input_ids
                ).tolist()
                min_mse_in_batch = min(new_mses)

                if min_mse_in_batch < best_mse:
                    best_candidate_tokens = new_candidate_tokens[new_mses.index(min_mse_in_batch)]
                    best_mse = min_mse_in_batch
                    best_explanation_tokens = best_candidate_tokens.clone()
                    current_mse = best_mse
                    current_explanation_tokens = best_explanation_tokens.clone()
                    if verbose:
                        print(
                            f"Found new global best: '{self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)}' (MSE: {best_mse:.4f})"
                        )
                    found_improvement_in_retries = True
                    break

                if retry_attempt < max_temp_increases:
                    current_temperature *= temp_increase_factor

            # Final check for this index
            final_saliences_for_iter = self._calculate_salience_batch(
                current_explanation_tokens, current_mse, A_target, position_to_analyze, input_ids
            )
            final_salience_at_idx = final_saliences_for_iter[current_idx]

            if final_salience_at_idx < 0:
                if verbose:
                    print(
                        f"Final salience at index {current_idx} is still negative ({final_salience_at_idx:.4f}). Mutating to space."
                    )
                space_token_id = self.tokenizer.encode(" ", add_special_tokens=False)
                current_explanation_tokens[current_idx] = (
                    space_token_id[0] if space_token_id else self.tokenizer.eos_token_id
                )
                current_mse = self._get_mse_for_explanation_batch(
                    current_explanation_tokens.unsqueeze(0), A_target, position_to_analyze, input_ids
                ).item()
                if verbose:
                    print(f"New working MSE after mutation: {current_mse:.4f}")
                if current_mse < best_mse:
                    if verbose:
                        print("The mutated version is a new global best.")
                    best_mse = current_mse
                    best_explanation_tokens = current_explanation_tokens.clone()
            else:
                if verbose:
                    print(f"Final salience at index {current_idx} is non-negative. Finalizing this position.")
        # Iteratively remove tokens with the most negative salience until none remain.
        if verbose:
            print("\n--- Iteratively removing negative salience tokens ---")

        space_token_id_list = self.tokenizer.encode(" ", add_special_tokens=False)
        # Use space for ablation, or EOS as a fallback.
        ablation_token_id = space_token_id_list[0] if space_token_id_list else self.tokenizer.eos_token_id

        while True:
            saliences = self._calculate_salience_batch(
                best_explanation_tokens, best_mse, A_target, position_to_analyze, input_ids
            )

            negative_saliences = [
                (s, i)
                for i, s in enumerate(saliences)
                if s < -0.005 and best_explanation_tokens[i] != ablation_token_id
            ]

            if not negative_saliences:
                if verbose:
                    print("No negative salience tokens remain.")
                break

            # Find and remove the token with the most negative salience ("easiest win").
            min_salience, idx_to_remove = min(negative_saliences, key=lambda x: x[0])

            if verbose:
                token_to_remove_str = self.tokenizer.decode(best_explanation_tokens[idx_to_remove])
                print(f"Removing token '{token_to_remove_str}' (index {idx_to_remove}, salience: {min_salience:.4f})")

            # Replace with ablation token and update MSE.
            best_explanation_tokens[idx_to_remove] = ablation_token_id
            best_mse = self._get_mse_for_explanation_batch(
                best_explanation_tokens.unsqueeze(0), A_target, position_to_analyze, input_ids
            ).item()

            if verbose:
                print(
                    f"New explanation: '{self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)}' (MSE: {best_mse:.4f})"
                )

        # Final round: remove spaces and try to fill in the gaps
        if verbose:
            print("\n--- Final space-removal optimization round ---")

        # It's possible for a tokenizer to not have a standard space token.
        # Only proceed if a space token exists to be removed.
        if space_token_id_list:
            space_token_id = space_token_id_list[0]

            prefix_without_spaces = best_explanation_tokens[best_explanation_tokens != space_token_id]
            num_tokens_to_gen = self.t_text - len(prefix_without_spaces)

            if num_tokens_to_gen > 0:
                if verbose:
                    print(
                        f"Removed spaces. Prefix: '{self.tokenizer.decode(prefix_without_spaces, skip_special_tokens=True)}'. Generating {num_tokens_to_gen} tokens."
                    )

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    A_batch = A_target.unsqueeze(0).repeat(num_samples_per_iteration, 1)
                    token_pos_batch = torch.full((num_samples_per_iteration,), position_to_analyze, device=self.device)

                    # Use the initial temperature for this final generation
                    final_temp = kwargs.get("temperature", 1.0)
                    current_token_id = (
                        input_ids[:, position_to_analyze] if self.encoder.config.add_current_token else None
                    )
                    if current_token_id is not None:
                        current_token_id = current_token_id.expand(num_samples_per_iteration)

                    new_gen_result = self.decoder.generate_soft_kv_cached_nondiff(
                        A_batch,
                        max_length=num_tokens_to_gen,
                        gumbel_tau=self.tau,
                        original_token_pos=token_pos_batch,
                        temperature=final_temp,
                        add_tokens=prefix_without_spaces.tolist(),
                    )

                    new_candidate_tokens = new_gen_result.hard_token_ids
                    if len(prefix_without_spaces) > 0:
                        new_candidate_tokens = torch.cat(
                            [prefix_without_spaces.expand(num_samples_per_iteration, -1), new_candidate_tokens], dim=1
                        )

                new_mses = self._get_mse_for_explanation_batch(
                    new_candidate_tokens, A_target, position_to_analyze, input_ids
                ).tolist()
                min_mse_in_batch = min(new_mses)

                if min_mse_in_batch < best_mse:
                    best_candidate_idx = new_mses.index(min_mse_in_batch)
                    best_explanation_tokens = new_candidate_tokens[best_candidate_idx].clone()
                    best_mse = min_mse_in_batch
                    if verbose:
                        print(
                            f"Found new global best after space removal: '{self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)}' (MSE: {best_mse:.4f})"
                        )
                elif verbose:
                    print(
                        f"Space removal round did not yield a better explanation. - gave {min_mse_in_batch:.4f} instead of {best_mse:.4f}"
                    )
            elif verbose:
                print("No spaces to remove or sequence already full. Skipping final round.")
        elif verbose:
            print("No space token found in tokenizer. Skipping final space-removal round.")

        if verbose:
            print("\n--- Optimization Complete ---")
            final_explanation_text = self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)
            print(f"Final optimized explanation: '{final_explanation_text}'")
            print(f"Final MSE: {best_mse:.6f}")
            final_salience_scores = self._calculate_salience_batch(
                best_explanation_tokens, best_mse, A_target, position_to_analyze, input_ids
            )
            print(f"Orig. saliences: {[f'{s:.2f}' for s in orig_saliences]}")
            print(f"Final saliences: {[f'{s:.2f}' for s in final_salience_scores]}")
        else:
            final_salience_scores = self._calculate_salience_batch(
                best_explanation_tokens, best_mse, A_target, position_to_analyze, input_ids
            )

        return best_explanation_tokens, best_mse, final_salience_scores

    def calculate_salience(
        self,
        explanation_ids: torch.Tensor,
        base_mse: float,
        A_target: torch.Tensor,
        position_to_analyze: int,
        input_ids: torch.Tensor,
    ) -> List[float]:
        null_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        # Create a batch of all ablated sequences
        ablated_ids_batch = explanation_ids.repeat(len(explanation_ids), 1)
        ablated_ids_batch[torch.arange(len(explanation_ids)), torch.arange(len(explanation_ids))] = null_token_id

        # Calculate MSE for all ablated sequences in one batch
        ablated_mses = self._get_mse_for_explanation_batch(ablated_ids_batch, A_target, position_to_analyze, input_ids)

        return ((ablated_mses - base_mse) / (base_mse + 1e-9)).tolist()

    def analyze_all_tokens(
        self,
        text: str,
        seed=42,
        batch_size=None,
        no_eval=False,
        tuned_lens: bool = True,
        add_tokens=None,
        replace_left=None,
        replace_right=None,
        do_hard_tokens=False,
        return_structured=False,
        move_devices=False,
        logit_lens_analysis: bool = False,
        temperature: float = 1.0,
        no_kl: bool = False,
        calculate_token_salience: bool = False,
        optimize_explanations_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
        messages_list: Optional[List[Dict[str, str]]] = None,
        last_n_messages: Optional[int] = None,
    ) -> pd.DataFrame:
        """Analyze all tokens in the text and return results as DataFrame.

        Args:
            text: Text to analyze
            seed: Random seed for reproducibility
            batch_size: Batch size for processing
            no_eval: If True, skip evaluation metrics (MSE, KL, etc.).
            tuned_lens: Include TunedLens predictions in results
            add_tokens: Additional tokens to add to the tokenizer
            replace_left: Replace tokens on the left
            replace_right: Replace tokens on the right
            do_hard_tokens: If True, use hard tokens for analysis
            return_structured: If True, return structured data
            move_devices: If True, move devices
            logit_lens_analysis: Add logit-lens predictions (projecting hidden state through unembedding).
            temperature: Sampling temperature for explanation generation.
            no_kl: If True, skip KL divergence calculation. Only has an effect if no_eval is False.
            calculate_token_salience: If True, calculate a salience score for each token in the explanation.
            optimize_explanations_config: If provided, optimize explanations using the given configuration.
            progress_callback: Optional callback function for progress reporting.
            messages_list: Optional list of messages (for chat format). Used with last_n_messages to determine token boundaries.
            last_n_messages: Optional number of last messages to analyze (e.g., 2 for last user/assistant turn). 
                           Requires messages_list to be provided. Default: None (analyze all tokens).
                           When specified, only tokens from the last N messages will be analyzed, significantly
                           improving performance for long conversations.
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=20000)
        input_ids = inputs.input_ids.clone()
        attention_mask = inputs.attention_mask.clone()
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        if inputs.input_ids[0][0] == inputs.input_ids[0][1]:
            print(
                f"First two tokens are the same: {self.tokenizer.decode(inputs.input_ids[0][0])}, {self.tokenizer.decode(inputs.input_ids[0][1])}, removing first token"
            )
            input_ids = input_ids[:, 1:].clone()
            attention_mask = attention_mask[:, 1:].clone()
        # Use actual current device instead of self.device
        current_device = self.get_current_device()
        input_ids = input_ids.to(current_device)
        seq_len = input_ids.shape[1]
        attention_mask = attention_mask.to(current_device)

        if move_devices and self.orig_model.model != self.shared_base_model:
            if self.shared_base_model is not None:
                self.shared_base_model.to("cpu")
            self.orig_model.model.to("cuda")

        A_full_sequence = self.orig_model.get_all_activations_at_layer(
            input_ids[0], self.layer, attention_mask=attention_mask
        )

        if move_devices and self.orig_model.model != self.shared_base_model:
            self.orig_model.model.to("cpu")
            if self.shared_base_model is not None:
                self.shared_base_model.to("cuda")

        torch.manual_seed(seed)

        # Determine which tokens to analyze based on last_n_messages
        tokens_to_analyze = None  # None means analyze all tokens
        if last_n_messages is not None:
            # Find all positions of start-of-turn tokens in the input
            input_ids_flat = input_ids[0].tolist()
            start_positions = []
            for idx, tok in enumerate(input_ids_flat):
                if tok in self.start_of_turn_tokens:
                    start_positions.append(idx)

            # We want the last (last_n_messages+1) start-of-turns, but if the last one is not in the last 4 tokens, just do the last n start-of-turns
            if len(start_positions) >= last_n_messages + 1:
                last_start = start_positions[-1]
                if last_start >= seq_len - 4:
                    # Start transcript 3 before the (n+1)th-to-last start-of-turn
                    start_idx = max(0, start_positions[-(last_n_messages+1)] - 3)
                    tokens_to_analyze = torch.zeros(seq_len, dtype=torch.bool, device=current_device)
                    tokens_to_analyze[start_idx:] = True
                    logger.info(f"Analyzing last {last_n_messages} messages: start at token {start_idx} (start-of-turn idx {start_positions[-(last_n_messages+1)]}) out of {seq_len}")
                    print(f"[ANALYZE_ALL_TOKENS] Analyzing last {last_n_messages} messages: start at token {start_idx} (start-of-turn idx {start_positions[-(last_n_messages+1)]}) out of {seq_len}")
                else:
                    # Just do the last n start-of-turns
                    start_idx = start_positions[-last_n_messages]
                    tokens_to_analyze = torch.zeros(seq_len, dtype=torch.bool, device=current_device)
                    tokens_to_analyze[start_idx:] = True
                    logger.info(f"Analyzing last {last_n_messages} messages (not at end): start at token {start_idx} (start-of-turn idx {start_positions[-last_n_messages]}) out of {seq_len}")
                    print(f"[ANALYZE_ALL_TOKENS] Analyzing last {last_n_messages} messages (not at end): start at token {start_idx} (start-of-turn idx {start_positions[-last_n_messages]}) out of {seq_len}")
            else:
                logger.info(f"Not enough start-of-turn tokens found ({len(start_positions)}). wanted {last_n_messages+1}, analyzing all tokens.")
                tokens_to_analyze = None
        
        # Prepare data for batch processing
        token_positions = torch.arange(seq_len, device=current_device)
        batch_data = []
        
        if tokens_to_analyze is None:
            # Original behavior - analyze all tokens
            for i in range(0, seq_len, batch_size):
                batch_data.append(
                    (
                        input_ids[:, i : i + batch_size],
                        A_full_sequence[i : i + batch_size, :],
                        token_positions[i : i + batch_size],
                    )
                )
        else:
            # Filter to only include tokens we want to analyze
            positions_to_analyze = token_positions[tokens_to_analyze]
            
            # Create batches only from positions we want to analyze
            for i in range(0, len(positions_to_analyze), batch_size):
                batch_positions = positions_to_analyze[i : i + batch_size]
                batch_indices = batch_positions.cpu().numpy()  # Convert to numpy for indexing
                
                batch_data.append(
                    (
                        input_ids[:, batch_indices],
                        A_full_sequence[batch_indices, :],
                        batch_positions,
                    )
                )

        # Process batches
        results = []
        total_batches = len(batch_data)
        for batch_idx, batch in enumerate(tqdm.tqdm(batch_data, desc="Processing batches")):
            # Report progress if callback is provided
            if progress_callback is not None:
                should_continue = progress_callback(
                    batch_idx + 1, total_batches, f"Processing batch {batch_idx + 1}/{total_batches}"
                )
                # Check if the callback signals to stop (returns False)
                if should_continue is False:
                    print("Analysis cancelled by user.")
                    # Return partial results collected so far
                    return pd.DataFrame(results)

            input_ids_batch, A_batch, token_positions_batch = batch
            current_batch_size = input_ids_batch.shape[1]

            # Logit-lens analysis can be batched easily
            top_tokens_logit_lens_batch = None
            if logit_lens_analysis:
                unembedding_layer = self.orig_model.model.get_output_embeddings()
                logits_logit_lens = unembedding_layer(A_batch)
                top_tokens_logit_lens_batch = [
                    " ".join(get_top_n_tokens(logits_logit_lens[j], self.tokenizer, min(self.t_text, 10)))
                    for j in range(logits_logit_lens.shape[0])
                ]

            # Generate initial explanations
            if optimize_explanations_config is None:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    gen = self.decoder.generate_soft_kv_cached_nondiff(
                        A_batch,
                        max_length=self.t_text,
                        gumbel_tau=self.tau,
                        original_token_pos=token_positions_batch,
                        temperature=temperature,
                    )

            # CRITICAL FIX: Use the device of A_batch, not self.device
            mse = torch.full((current_batch_size,), -1.0, device=A_batch.device)
            salience_scores = [[] for _ in range(current_batch_size)]

            # Optimize explanations if config is provided. This will also calculate MSE and salience.
            if optimize_explanations_config is not None:
                # Check if we should use batched best-of-k optimization
                use_batched = optimize_explanations_config.get("use_batched", True)
                best_of_k = optimize_explanations_config.get("best_of_k", 8)

                if use_batched and best_of_k:
                    # Use the new batched optimization method
                    num_rollouts = best_of_k if isinstance(best_of_k, int) else 8
                    n_groups_per_rollout = optimize_explanations_config.get("n_groups_per_rollout", 8)

                    optimized_tokens, optimized_mses, optimized_salience_scores = self._optimize_batch_best_of_k(
                        A_batch,
                        token_positions_batch,
                input_ids,
                        num_rollouts_per_position=num_rollouts,
                        temperature=optimize_explanations_config.get("temperature", temperature),
                        n_groups_per_rollout=n_groups_per_rollout,
                    )

                    optimized_hard_token_ids = optimized_tokens
                    mse = optimized_mses
                    if calculate_token_salience:
                        salience_scores = optimized_salience_scores
                else:
                    # Use the original sequential optimization
                    optimized_mses_list = []
                    optimized_salience_scores_list = []
                    optimized_tokens_list = []

                    inner_iterator = tqdm.tqdm(range(current_batch_size), desc="Optimizing explanations with complicated procedure", leave=False)
                    for j in inner_iterator:
                        # Report detailed progress for optimization
                        if progress_callback is not None:
                            should_continue = progress_callback(
                                batch_idx + 1,
                                total_batches,
                                f"Batch {batch_idx + 1}/{total_batches}: Optimizing token {j + 1}/{current_batch_size}",
                            )
                            # Check if the callback signals to stop (returns False)
                            if should_continue is False:
                                print("Analysis cancelled by user during optimization.")
                                # Return partial results collected so far
                                return pd.DataFrame(results)
                        optimized_tokens, optimized_mse, optimized_saliences = self.optimize_explanation_for_mse(
                            None,
                            A_target=A_batch[j],
                            position_to_analyze=token_positions_batch[j].item(),
                            input_ids=input_ids,
                            verbose=False,
                            **optimize_explanations_config,
                        )
                        optimized_tokens_list.append(optimized_tokens)
                        optimized_mses_list.append(optimized_mse)
                        optimized_salience_scores_list.append(optimized_saliences)

                    # Stack all optimized tokens into a batch tensor
                    optimized_hard_token_ids = torch.stack(optimized_tokens_list, dim=0)
                    # CRITICAL FIX: Use the device of A_batch for consistency
                    mse = torch.tensor(optimized_mses_list, device=A_batch.device)
                    if calculate_token_salience:
                        salience_scores = optimized_salience_scores_list

                # Get embedding table from decoder's base model
                with torch.no_grad():
                    if hasattr(self.decoder.base, "get_input_embeddings"):
                        input_emb_table = self.decoder.base.get_input_embeddings().weight
                    else:
                        # Fallback for different model architectures
                        input_emb_table = self.decoder.base.transformer.wte.weight

                    # Convert token IDs to embeddings
                    optimized_embeddings = input_emb_table[optimized_hard_token_ids]

                    # Handle Gemma3 normalization if needed
                    if hasattr(self.decoder, "is_gemma3") and self.decoder.is_gemma3:
                        optimized_embeddings = optimized_embeddings * self.decoder.normalizer

                # Create Generated instance with optimized results
                gen = Generated(
                    generated_text_embeddings=optimized_embeddings,
                    raw_lm_logits=None,  # We don't have logits from optimization
                    hard_token_ids=optimized_hard_token_ids,
                )

                # #mse = torch.tensor(optimized_mses_list, device=self.device)
                # if calculate_token_salience:
                #     salience_scores = optimized_salience_scores_list

            # Calculate evaluation metrics if not optimizing
            else:
                if not no_eval:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                        A_hat = self.encoder(
                            gen.generated_text_embeddings,
                            original_token_pos=token_positions_batch,
                            current_token_ids=input_ids_batch[0, :] if self.encoder.config.add_current_token else None,
                        )
                        # CRITICAL FIX: Ensure both tensors are on the same device
                        A_batch_device = A_batch.to(A_hat.device)
                        mse = torch.nn.functional.mse_loss(A_batch_device, A_hat, reduction="none").mean(dim=-1)
                else:
                    mse.fill_(0.0)

                if calculate_token_salience:
                    for j in range(current_batch_size):
                        salience_scores[j] = self.calculate_salience(
                            gen.hard_token_ids[j],
                            mse[j].item(),
                            A_batch[j, :],
                            token_positions_batch[j].item(),
                            input_ids,
                        )

            # Calculate remaining metrics (relative RMSE, KL divergence)
            if no_eval:
                # CRITICAL FIX: Use mse.device to ensure consistency
                relative_RMSE = torch.full((current_batch_size,), 0.0, device=mse.device)
                kl_div = torch.full((current_batch_size,), 0.0, device=mse.device)
            else:
                # CRITICAL FIX: Ensure A_batch is on the same device as mse
                A_batch_for_rmse = A_batch.to(mse.device)
                relative_RMSE = torch.sqrt(mse) / torch.sqrt(torch.mean(A_batch_for_rmse**2, dim=-1))
                if not no_kl:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                        # This KL is between the explanation's predicted logits and the original model's logits
                        logits = self.orig_model.model(input_ids_batch).logits
                        logits = logits[:, token_positions_batch, :]
                        kl_div = torch.nn.functional.kl_div(
                            torch.log_softmax(logits, dim=-1), torch.softmax(gen.logits, dim=-1), reduction="none"
                        ).sum(dim=-1)
                else:
                    kl_div = torch.full((current_batch_size,), -1.0, device=mse.device)

            # Prepare results for the batch
            for j in range(current_batch_size):
                explanation = self.tokenizer.decode(gen.hard_token_ids[j], skip_special_tokens=False) + (
                    "[" + self.tokenizer.decode([input_ids_batch[0, j].item()]) + "]"
                    if self.encoder.config.add_current_token
                    else ""
                )
                explanation_structured = [
                    self.tokenizer.decode(gen.hard_token_ids[j][i], skip_special_tokens=False)
                    for i in range(len(gen.hard_token_ids[j]))
                ] + (
                    ["[" + self.tokenizer.decode([input_ids_batch[0, j].item()]) + "]"]
                    if self.encoder.config.add_current_token
                    else []
                )
                result_item = {
                    "position": token_positions_batch[j].item(),
                    "token": self.tokenizer.decode([input_ids_batch[0, j].item()]),
                    "explanation": explanation,
                    "kl_divergence": kl_div[j].item(),
                    "mse": mse[j].item(),
                    "relative_rmse": relative_RMSE[j].item(),
                }
                if return_structured:
                    result_item["explanation_structured"] = explanation_structured
                if logit_lens_analysis and top_tokens_logit_lens_batch is not None:
                    result_item["logit_lens_top"] = top_tokens_logit_lens_batch[j]
                if salience_scores is not None:
                    result_item["token_salience"] = salience_scores[j]
                results.append(result_item)

        # Add extra metadata if needed (fix: only if optimize_explanations_config is not None)
        extra_metadata = {}
        if optimize_explanations_config is not None:
            extra_metadata = dict(optimize_explanations_config)
        if not extra_metadata:
            extra_metadata["batch_size"] = batch_size
            extra_metadata["do_hard_tokens"] = do_hard_tokens
            extra_metadata["temperature"] = temperature

        df = LensDataFrame(
            results,
            checkpoint_path=getattr(self, "checkpoint_path", None),
            model_name=getattr(self, "model_name", None),
            orig_model_name=getattr(self, "orig_model_name", None),
            # extra_metadata=extra_metadata,
        )

        # # # Add TunedLens predictions if enabled (this is done separately as it may be expensive)
        # if tuned_lens and self.comparison_tuned_lens is not None:
        #     df = self.add_tuned_lens_predictions(df, text, add_tokens, replace_left, replace_right, do_hard_tokens, return_structured, move_devices)

        return df

    # def _optimize_explanation_worker(self,
    #                              A_target: torch.Tensor,
    #                              initial_explanation_tokens: torch.Tensor,
    #                              position_to_analyze: int,
    #                              input_ids: torch.Tensor,
    #                              **kwargs) -> Tuple[torch.Tensor, float, List[float]]:
    #     """
    #     Worker function to optimize a single explanation. Assumes an initial explanation is provided
    #     and does NOT perform the initial N rollouts itself.
    #     """
    #     # Unpack kwargs with defaults for optimization parameters
    #     salience_pct_threshold = kwargs.get('salience_pct_threshold', 0.0)
    #     temperature = kwargs.get('temperature', 1.0)
    #     max_temp_increases = kwargs.get('max_temp_increases', 2)
    #     temp_increase_factor = kwargs.get('temp_increase_factor', 1.5)
    #     num_samples_per_iteration = kwargs.get('num_samples_per_iteration', 16)
    #     verbose = kwargs.get('verbose', False)

    #     def get_mse_for_explanation(explanation_ids: torch.Tensor) -> torch.Tensor:
    #         # Always expects a batch (N, seq_len), returns tensor of N MSEs
    #         with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
    #             explanation_embeddings = self.encoder.base.get_input_embeddings()(explanation_ids)
    #             batch_size = explanation_ids.shape[0]
    #             original_token_pos = torch.full((batch_size,), position_to_analyze, device=self.device)
    #             current_token_id = input_ids[:, position_to_analyze] if self.encoder.config.add_current_token else None
    #             if current_token_id is not None: current_token_id = current_token_id.expand(batch_size)
    #             A_hat = self.encoder(explanation_embeddings, original_token_pos=original_token_pos, current_token_ids=current_token_id)
    #             A_target_expanded = A_target.expand_as(A_hat)
    #             mses = torch.nn.functional.mse_loss(A_target_expanded, A_hat, reduction='none').mean(dim=-1)
    #             return mses

    #     # 'best' variables track the global best. 'current' variables are for the working copy.
    #     best_explanation_tokens = initial_explanation_tokens.clone()
    #     best_mse = get_mse_for_explanation(best_explanation_tokens.unsqueeze(0)).item()

    #     current_explanation_tokens = best_explanation_tokens.clone()
    #     current_mse = best_mse

    #     if verbose:
    #         print(f"Starting optimization with: '{self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)}' (MSE: {best_mse:.4f})")

    #     for current_idx in range(self.t_text):
    #         if verbose:
    #             print(f"\n--- Processing index {current_idx}/{self.t_text-1} ---")

    #         # Salience is calculated on the current working explanation to decide where to work next
    #         salience_scores = self._calculate_salience_batch(current_explanation_tokens, current_mse, A_target, position_to_analyze, input_ids)
    #         current_salience = salience_scores[current_idx]

    #         if verbose:
    #             print(f"Salience at index {current_idx}: {current_salience:.4f}. Current explanation: '{self.tokenizer.decode(current_explanation_tokens)}'")

    #         if current_salience >= salience_pct_threshold:
    #             if verbose:
    #                 print(f"Salience is above threshold ({salience_pct_threshold}). Skipping optimization for this token.")
    #             continue

    #         # We generate from a prefix of the *current* working explanation
    #         prefix_tokens = current_explanation_tokens[:current_idx]

    #         found_improvement_in_retries = False
    #         current_temperature = temperature

    #         for retry_attempt in range(max_temp_increases + 1):
    #             if verbose:
    #                 print(f"Generation attempt {retry_attempt + 1}/{max_temp_increases + 1} with temp {current_temperature:.2f}")
    #             with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
    #                 A_batch = A_target.unsqueeze(0).repeat(num_samples_per_iteration, 1)
    #                 token_pos_batch = torch.full((num_samples_per_iteration,), position_to_analyze, device=self.device)
    #                 gen_func = self.decoder.generate_soft_kv_cached_nondiff

    #                 # Generate the rest of the sequence
    #                 num_tokens_to_gen = self.t_text - len(prefix_tokens)
    #                 if num_tokens_to_gen <= 0:
    #                     if verbose:
    #                         print("Prefix is already full length. Cannot generate more tokens.")
    #                     break

    #                 new_gen_result = gen_func(A_batch, max_length=num_tokens_to_gen, gumbel_tau=self.tau, original_token_pos=token_pos_batch, temperature=current_temperature, add_tokens = prefix_tokens)

    #                 new_candidate_tokens = new_gen_result.hard_token_ids
    #                 if len(prefix_tokens) > 0:
    #                     new_candidate_tokens = torch.cat([prefix_tokens.expand(num_samples_per_iteration, -1), new_candidate_tokens], dim=1)

    #             new_mses = self._get_mse_for_explanation_batch(new_candidate_tokens, A_target, position_to_analyze, input_ids).tolist()
    #             min_mse_in_batch = min(new_mses)

    #             # Monotonic improvement check against the GLOBAL best
    #             if min_mse_in_batch < best_mse:

    #                 # ALSO update the current working copy to this new best
    #                 current_mse = best_mse
    #                 current_explanation_tokens = best_explanation_tokens.clone()

    #                 if verbose:
    #                     print(f"Found new global best: '{self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)}' (MSE: {best_mse:.4f})")
    #                 found_improvement_in_retries = True
    #                 break # Exit the temperature retry loop

    #             if retry_attempt < max_temp_increases:
    #                 current_temperature *= temp_increase_factor

    #         # --- Final check for the current index, regardless of generation outcome ---
    #         # Re-calculate salience on the final state of the current explanation for this iteration
    #         final_saliences_for_iter = self._calculate_salience_batch(current_explanation_tokens, current_mse, A_target, position_to_analyze, input_ids)
    #         final_salience_at_idx = final_saliences_for_iter[current_idx]

    #         if final_salience_at_idx < 0:
    #             if verbose:
    #                 print(f"Final salience at index {current_idx} is still negative ({final_salience_at_idx:.4f}). Mutating to space.")
    #             space_token_id = self.tokenizer.encode(' ', add_special_tokens=False)

    #             # Mutate the working copy
    #             current_explanation_tokens[current_idx] = space_token_id[0] if space_token_id else self.tokenizer.eos_token_id
    #             current_mse = self._get_mse_for_explanation_batch(current_explanation_tokens.unsqueeze(0), A_target, position_to_analyze, input_ids).item()
    #             if verbose:
    #                 print(f"New working MSE after mutation: {current_mse:.4f}")

    #             # Check if this mutated version is now the new global best
    #             if current_mse < best_mse:
    #                 if verbose:
    #                     print("The mutated version is a new global best.")
    #                 best_mse = current_mse
    #                 best_explanation_tokens = current_explanation_tokens.clone()
    #         else:
    #             if verbose:
    #                 print(f"Final salience at index {current_idx} is non-negative. Finalizing this position.")

    #     if verbose:
    #         print("\n--- Optimization Complete ---")
    #         final_explanation_text = self.tokenizer.decode(best_explanation_tokens, skip_special_tokens=True)
    #         print(f"Final optimized explanation: '{final_explanation_text}'")
    #         print(f"Final MSE: {best_mse:.6f}")
    #         final_salience_scores = self._calculate_salience_batch(best_explanation_tokens, best_mse, A_target, position_to_analyze, input_ids)
    #         print(f"Final saliences: {[f'{s:.2f}' for s in final_salience_scores]}")
    #     return best_explanation_tokens, best_mse, self._calculate_salience_batch(best_explanation_tokens, best_mse, A_target, position_to_analyze, input_ids)

    def calculate_salience(
        self,
        explanation_ids: torch.Tensor,
        base_mse: float,
        A_target: torch.Tensor,
        position_to_analyze: int,
        input_ids: torch.Tensor,
    ) -> List[float]:
        """This method is deprecated. Use _calculate_salience_batch instead."""
        # This function is now broken because get_mse_for_explanation is not in scope.
        # It has been replaced by _calculate_salience_batch.
        return self._calculate_salience_batch(explanation_ids, base_mse, A_target, position_to_analyze, input_ids)

    def causal_intervention(
        self,
        original_text: str,
        intervention_position: int,
        intervention_string: str,
        max_new_tokens: int = 20,
        visualize: bool = True,
        top_k: int = 5,
        next_token_position: int = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform a causal intervention on a model's hidden state.

        This involves replacing the activation at a specific `intervention_position` with a new
        activation generated from an `intervention_string`, and then observing the effect on
        the model's predictions at `next_token_position`.
        """
        # Tokenize original text
        inputs = self.tokenizer(original_text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        # Handle negative indices for positions
        if intervention_position < 0:
            intervention_position = seq_len + intervention_position

        if not 0 <= intervention_position < seq_len:
            raise ValueError(
                f"Intervention position {intervention_position} is out of bounds for sequence length {seq_len}"
            )

        if next_token_position is None:
            next_token_position = intervention_position
        elif next_token_position < 0:
            next_token_position = seq_len + next_token_position

        if not 0 <= next_token_position < seq_len:
            raise ValueError(
                f"Next token position {next_token_position} is out of bounds for sequence length {seq_len}"
            )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                # Get original hidden states
                outputs_orig = self.orig_model.model(input_ids, output_hidden_states=True)
                hidden_states_orig = outputs_orig.hidden_states

                # Get original activation (A) at intervention position
                A_orig = hidden_states_orig[self.layer + 1][:, intervention_position, :]

                # Decode original activation to get its explanation
                if self.decoder.config.use_kv_cache:
                    gen_orig = self.decoder.generate_soft_kv_cached(
                        A_orig,
                        max_length=self.t_text,
                        gumbel_tau=self.tau,
                        original_token_pos=torch.tensor([intervention_position], device=self.device),
                        temperature=temperature,
                    )
                else:
                    gen_orig = self.decoder.generate_soft(
                        A_orig,
                        max_length=self.t_text,
                        gumbel_tau=self.tau,
                        original_token_pos=torch.tensor([intervention_position], device=self.device),
                        temperature=temperature,
                    )

                explanation_orig = self.tokenizer.decode(gen_orig.hard_token_ids[0], skip_special_tokens=True)

                # Prepare intervention string and get its embeddings
                intervention_tokens = self.tokenizer(intervention_string, return_tensors="pt")
                intervention_ids = intervention_tokens.input_ids.to(self.device)
                if intervention_ids.shape[-1] > self.t_text:
                    print(f"Truncating intervention string to {self.t_text} tokens.")
                    intervention_ids = intervention_ids[..., -self.t_text :]
                elif intervention_ids.shape[-1] < self.t_text:
                    # Pad intervention if it's shorter than the expected length
                    pad_width = self.t_text - intervention_ids.shape[-1]
                    intervention_ids = torch.cat(
                        [
                            torch.zeros(
                                intervention_ids.shape[0], pad_width, dtype=intervention_ids.dtype, device=self.device
                            ),
                            intervention_ids,
                        ],
                        dim=-1,
                    )

                new_intervention_string = self.tokenizer.decode(intervention_ids[0], skip_special_tokens=True)

                if hasattr(self.orig_model.model, "transformer"):
                    embed_layer = self.orig_model.model.transformer.wte
                elif hasattr(self.orig_model.model, "model"):
                    embed_layer = self.orig_model.model.model.embed_tokens
                else:
                    embed_layer = self.orig_model.model.get_input_embeddings()

                intervention_embeddings = embed_layer(intervention_ids)

                # Encode the intervention embeddings to get the new activation
                A_intervention = self.encoder(
                    intervention_embeddings,
                    original_token_pos=torch.tensor([intervention_position], device=self.device),
                    current_token_ids=input_ids[:, intervention_position]
                    if self.encoder.config.add_current_token
                    else None,
                )

                # Also encode the original explanation to get the reconstructed activation (Â)
                A_orig_decoded = self.encoder(
                    gen_orig.generated_text_embeddings,
                    original_token_pos=torch.tensor([intervention_position], device=self.device),
                    current_token_ids=input_ids[:, intervention_position]
                    if self.encoder.config.add_current_token
                    else None,
                )

                # Run model forward with original, intervened, and reconstructed activations
                # The `token_pos` argument specifies where to insert the activation.
                outputs_orig_full = self.orig_model.forward_with_replacement(
                    input_ids=input_ids,
                    new_activation=A_orig,
                    layer_idx=self.layer,
                    token_pos=intervention_position,
                    no_grad=True,
                )
                outputs_intervention = self.orig_model.forward_with_replacement(
                    input_ids=input_ids,
                    new_activation=A_intervention,
                    layer_idx=self.layer,
                    token_pos=intervention_position,
                    no_grad=True,
                )
                outputs_orig_decoded = self.orig_model.forward_with_replacement(
                    input_ids=input_ids,
                    new_activation=A_orig_decoded,
                    layer_idx=self.layer,
                    token_pos=intervention_position,
                    no_grad=True,
                )

                # Generate a continuation from the original, un-intervened state for comparison
                continuation_orig = self.orig_model.model.generate(
                    input_ids[:, : intervention_position + 1],
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                    cache_implementation="static",
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
            top_tokens_orig = [
                (self.tokenizer.decode([idx]), prob.item()) for idx, prob in zip(top_k_orig.indices, top_k_orig.values)
            ]
            top_tokens_intervention = [
                (self.tokenizer.decode([idx]), prob.item())
                for idx, prob in zip(top_k_intervention.indices, top_k_intervention.values)
            ]
            top_tokens_orig_decoded = [
                (self.tokenizer.decode([idx]), prob.item())
                for idx, prob in zip(top_k_orig_decoded.indices, top_k_orig_decoded.values)
            ]

            # Compute metrics: KL divergence on predictions, MSE on activations
            kl_div = torch.nn.functional.kl_div(
                torch.log_softmax(next_token_logits_intervention, dim=-1),
                torch.softmax(next_token_logits_orig, dim=-1),
                reduction="batchmean",
            ).item()
            mse = torch.nn.functional.mse_loss(A_orig, A_intervention).item()

            mse_orig_decoded = torch.nn.functional.mse_loss(A_orig, A_orig_decoded).item()
            kl_div_orig_decoded = torch.nn.functional.kl_div(
                torch.log_softmax(next_token_logits_orig_decoded, dim=-1),
                torch.softmax(next_token_logits_orig, dim=-1),
                reduction="batchmean",
            ).item()

            # Decode tokens at relevant positions for reporting
            intervened_token = self.tokenizer.decode([input_ids[0, intervention_position].item()])
            prediction_context_token = self.tokenizer.decode([input_ids[0, next_token_position].item()])

        results = {
            "original_text": original_text,
            "intervention_position": intervention_position,
            "intervened_token": intervened_token,
            "next_token_position": next_token_position,
            "prediction_context_token": prediction_context_token,
            "intervention_string": new_intervention_string,
            "explanation_original": explanation_orig,
            "top_predictions_original": top_tokens_orig,
            "top_predictions_intervention": top_tokens_intervention,
            "top_predictions_original_decoded": top_tokens_orig_decoded,
            "kl_divergence": kl_div,
            "mse": mse,
            "mse_orig_decoded": mse_orig_decoded,
            "kl_div_orig_decoded": kl_div_orig_decoded,
            "continuation_original": self.tokenizer.decode(continuation_orig[0], skip_special_tokens=True),
        }

        # Print results
        print("\n🔬 Causal Intervention Analysis")
        print(f"{'=' * 60}")
        print(f"Original text: '{original_text}'")
        print(f"Intervened at position {intervention_position}: '{intervened_token}'")
        if next_token_position != intervention_position:
            print(
                f"Observing predictions at position {next_token_position} (for token after '{prediction_context_token}')"
            )

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
            top_tokens_orig[:top_k], top_tokens_intervention[:top_k], top_tokens_orig_decoded[:top_k]
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

        ax1 = fig.add_subplot(gs[:, 0])  # Main plot for distributions spans both rows
        ax2 = fig.add_subplot(gs[0, 1])  # KL plot
        ax3 = fig.add_subplot(gs[1, 1])  # MSE plot

        fig.suptitle(f"Causal Intervention Analysis for token: '{results['intervened_token']}'", fontsize=16, y=0.98)

        # --- Data preparation for probability plot ---
        top_k = 10

        preds_orig_map = dict(results["top_predictions_original"])
        preds_int_map = dict(results["top_predictions_intervention"])
        preds_orig_decoded_map = dict(results["top_predictions_original_decoded"])

        # Create a union of top-k tokens from each distribution for a comprehensive comparison
        tokens_orig_set = set(t[0] for t in results["top_predictions_original"][:top_k])
        tokens_int_set = set(t[0] for t in results["top_predictions_intervention"][:top_k])
        tokens_orig_decoded_set = set(t[0] for t in results["top_predictions_original_decoded"][:top_k])

        union_tokens = list(tokens_orig_set | tokens_int_set | tokens_orig_decoded_set)
        # Sort tokens by original probability for a meaningful order on the x-axis
        union_tokens.sort(key=lambda token: preds_orig_map.get(token, 0), reverse=True)
        plot_tokens = union_tokens[:15]  # Limit to max 15 tokens for readability

        probs_orig = [preds_orig_map.get(token, 0) for token in plot_tokens]
        probs_int = [preds_int_map.get(token, 0) for token in plot_tokens]
        probs_orig_decoded = [preds_orig_decoded_map.get(token, 0) for token in plot_tokens]

        x = np.arange(len(plot_tokens))
        width = 0.25

        # --- Plot 1: Combined probability distribution ---
        ax1.bar(x - width, probs_orig, width, label="Original (A)", alpha=0.8, color="cornflowerblue")
        ax1.bar(x, probs_orig_decoded, width, label="Reconstructed (Â)", alpha=0.8, color="mediumseagreen")
        ax1.bar(x + width, probs_int, width, label="Intervention", alpha=0.8, color="salmon")

        ax1.set_ylabel("Probability")
        ax1.set_title("Next Token Prediction Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels([repr(t) for t in plot_tokens], rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: KL Divergence ---
        kl_labels = ["Â vs A", "Intervention vs A"]
        kl_values = [results["kl_div_orig_decoded"], results["kl_divergence"]]
        kl_colors = ["mediumseagreen", "salmon"]
        ax2.bar(kl_labels, kl_values, color=kl_colors, alpha=0.8)
        ax2.set_ylabel("KL Divergence")
        ax2.set_title("Output Distribution Shift (KL)")
        ax2.grid(True, axis="y", alpha=0.3)
        # Add text labels above bars for clarity
        for i, v in enumerate(kl_values):
            ax2.text(
                i,
                v,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.2"),
            )

        # --- Plot 3: MSE ---
        mse_labels = ["Â vs A", "Intervention vs A"]
        mse_values = [results["mse_orig_decoded"], results["mse"]]
        mse_colors = ["mediumseagreen", "salmon"]
        ax3.bar(mse_labels, mse_values, color=mse_colors, alpha=0.8)
        ax3.set_ylabel("Mean Squared Error")
        ax3.set_title("Activation Vector Distance (MSE)")
        ax3.grid(True, axis="y", alpha=0.3)
        # Add text labels above bars for clarity
        for i, v in enumerate(mse_values):
            ax3.text(
                i,
                v,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.2"),
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def get_hidden_states(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.orig_model.model(input_ids, output_hidden_states=True)
        print("len hidden states", len(outputs.hidden_states))
        return outputs.hidden_states[self.layer + 1]

    def run_verbose_sample(
        self, text: str, position_to_analyze: int, top_n_analysis: int = 5, continuation_tokens: int = 30
    ):
        """Runs a single verbose sample analysis using process_and_print_verbose_batch_samples."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        if position_to_analyze < 0:
            position_to_analyze = seq_len + position_to_analyze

        if not (0 <= position_to_analyze < seq_len):
            raise ValueError(
                f"Position to analyze ({position_to_analyze}) is out of bounds for sequence length ({seq_len})."
            )

        # Get hidden state A
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            # The get_activations_at_positions method handles the torch.no_grad() context by default.
            A_i, _ = self.orig_model.get_activations_at_positions(
                input_ids=input_ids, layer_idx=self.layer, token_positions=position_to_analyze
            )
            # Ensure A_i is a standalone tensor, consistent with the original .clone()
            A_i = A_i.clone()

        # Prepare batch for verbose_samples function
        batch = {
            "A": A_i,
            "input_ids_A": input_ids,  # Needs to be [batch_size, seq_len]
            "layer_idx": torch.tensor([self.layer], device=self.device),
            "token_pos_A": torch.tensor([position_to_analyze], device=self.device),
            "A_prime": A_i.clone(),  # For single sample, A_prime can be A itself
            # Add dummy A_prime related fields if verbose_samples expects them even if not used meaningfully
            "input_ids_A_prime": input_ids.clone(),
            "token_pos_A_prime": torch.tensor([position_to_analyze], device=self.device),
        }

        # Prepare sch_args
        loss_weights = self.config
        sch_args = {
            "tau": self.tau,
            "alpha": self.config.get("alpha", 0.1),  # Default if not in config
            "kl_base_weight": loss_weights.get("kl_base_weight", 1.0),
            "entropy_weight": loss_weights.get("entropy_weight", 0.0),
            "mse_weight": loss_weights.get("mse_weight", 0.0),
            "lm_base_weight": loss_weights.get("lm_base_weight", 0.0),
            "t_text": self.t_text,
            "GRPO_entropy_weight": loss_weights.get("GRPO_entropy_weight", 0.0),
            "GRPO_weight": loss_weights.get("GRPO_weight", 0.0),
            "GRPO_beta": loss_weights.get("GRPO_beta", 0.0),
            "group_n": self.config["group_n"],
        }

        # Models dictionary
        models_dict = {
            "dec": self.decoder,
            "enc": self.encoder,
            # process_and_print_verbose_batch_samples expects 'orig' inside models for compute_single_sample_losses
            "orig": self.orig_model,
        }

        # Cache tokenized natural language prefix if specified (all processes do this, small overhead)
        cached_prefix_ids = None
        lm_loss_natural_prefix_text = self.config.get("lm_loss_natural_prefix")  # Assuming this key holds the text
        if lm_loss_natural_prefix_text and isinstance(lm_loss_natural_prefix_text, str):  # Check if it's a string
            cached_prefix_ids = self.tokenizer(
                lm_loss_natural_prefix_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device)
            print(
                f"Cached natural language prefix: '{lm_loss_natural_prefix_text}' ({cached_prefix_ids.shape[1]} tokens)"
            )
        elif (
            self.config.get("lm_loss_natural_prefix") is True
        ):  # Handle boolean true case if it implies a default prefix or other logic
            print("lm_loss_natural_prefix is True but not a string. Cannot cache prefix IDs without prefix text.")

        print(f'\n🔬 Running Verbose Sample Analysis for position {position_to_analyze} in text: "{text[:100]}..."')

        process_and_print_verbose_batch_samples(
            batch=batch,
            cfg=self.config,
            models=models_dict,  # Pass the dict containing dec, enc, and orig
            orig=self.orig_model,  # Also pass orig separately as it's used directly too
            tok=self.tokenizer,
            sch_args=sch_args,
            device=self.device,
            num_samples=1,  # We are running for a single sample
            top_n_analysis=top_n_analysis,
            printed_count_so_far=0,
            generate_continuation=True,
            continuation_tokens=continuation_tokens,
            return_structured_data=False,
            capture_output=False,
            cached_prefix_ids=cached_prefix_ids,  # Assuming no cached prefix for interactive use
            resample_ablation=self.config.get("resample_ablation", True),  # Get from config or default
            comparison_tuned_lens=self.comparison_tuned_lens,  # Pass the loaded TunedLens
            do_soft_token_embeds=False,
        )

    # @torch._dynamo.disable()
    # @torch.inference_mode()  # More aggressive than no_grad for disabling compilation
    def generate_continuation(
        self,
        text_or_messages,
        num_tokens: int = 100,
        num_completions: int = 10,
        is_chat: bool = False,
        chat_tokenizer=None,
        return_full_text: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        skip_special_tokens_for_decode: bool = False,
        use_cache: bool = False,
        cancellation_check=None,
    ) -> List[str]:
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
        # Check for cancellation before starting
        if cancellation_check and cancellation_check():
            print("Generation cancelled by user before starting.")
            return []

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
                add_gen_prompt = text_or_messages[-1]["role"] == "user"

                # Check for multiple system prompts
                num_system = sum(1 for m in text_or_messages if m.get("role") == "system")
                if num_system > 1:
                    raise ValueError("Multiple system prompts detected in chat messages.")

                # Apply chat template
                if 'gemma' in self.model_name.lower() and text_or_messages[0]["role"] == "system":
                    system_prompt = text_or_messages[0]["content"]
                    text_or_messages = text_or_messages[1:]
                    text_or_messages[0]["content"] = system_prompt + ("\n\n" if text_or_messages[0]["content"] else "") + text_or_messages[0]["content"]
                    print(f"Added system prompt for Gemma as user prefix")

                input_dict = tokenizer.apply_chat_template(
                    text_or_messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=add_gen_prompt,  # Important for proper generation
                    continue_final_message=not add_gen_prompt,
                )
                input_ids = input_dict["input_ids"].to(self.orig_model.model.device)
                attention_mask = input_dict.get("attention_mask", torch.ones_like(input_ids)).to(
                    self.orig_model.model.device
                )

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
                encoded = tokenizer(text_or_messages, return_tensors="pt", add_special_tokens=True )
                input_ids = encoded["input_ids"].to(self.orig_model.model.device)
                attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
                    self.orig_model.model.device
                )

                # Remove duplicate BOS if present
                if (
                    hasattr(tokenizer, "bos_token_id")
                    and input_ids.shape[1] >= 2
                    and input_ids[0, 0].item() == tokenizer.bos_token_id
                    and input_ids[0, 1].item() == tokenizer.bos_token_id
                ):
                    input_ids = input_ids[:, 1:]
                    attention_mask = attention_mask[:, 1:]
                
                # Remove final EOS if present
                if (
                    hasattr(tokenizer, "eos_token_id")
                    and input_ids.shape[1] >= 1
                    and input_ids[0, -1].item() == tokenizer.eos_token_id
                ):
                    input_ids = input_ids[:, :-1]
                    attention_mask = attention_mask[:, :-1]
            # Create batched input for multiple completions
            input_ids_batch = input_ids.repeat(num_completions, 1)
            attention_mask_batch = attention_mask.repeat(num_completions, 1)

            # Generate completions with compilation disabled
            with torch.no_grad():
                # # Temporarily disable cudnn benchmarking which can cause recompilation
                # old_benchmark = torch.backends.cudnn.benchmark
                # torch.backends.cudnn.benchmark = False
                if (
                    hasattr(self.orig_model.model, "_cache")
                    and hasattr(self.orig_model.model._cache, "device")
                    and self.orig_model.model._cache.device != self.orig_model.model.device
                ):
                    raise ValueError("Cache device does not match model device: try again? or set use_cache to False")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    generated = self.orig_model.model.generate(
                        input_ids_batch,
                        attention_mask=attention_mask_batch,
                        max_new_tokens=num_tokens,
                        min_new_tokens=num_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id
                        if hasattr(tokenizer, "eos_token_id")
                        else tokenizer.pad_token_id,
                        use_cache=use_cache,
                        cache_implementation="static",
                        disable_compile=True,
                    )
                # finally:
                #     torch.backends.cudnn.benchmark = old_benchmark

            # Decode and prepare outputs
            outputs = []
            for i in range(num_completions):
                # Check for cancellation between completions
                if cancellation_check and cancellation_check():
                    print(f"Generation cancelled by user after {i} completions.")
                    break
                if return_full_text == True:
                    if is_chat:
                        full_text = tokenizer.decode(generated[i], skip_special_tokens=skip_special_tokens_for_decode)
                    else:
                        # For regular text, decode normally
                        full_text = tokenizer.decode(generated[i], skip_special_tokens=skip_special_tokens_for_decode)
                    outputs.append(full_text)

                    # Print for convenience (showing generated portion with escape sequences)
                    generated_portion = tokenizer.decode(generated[i][-num_tokens:], skip_special_tokens=False)
                    generated_portion = generated_portion.replace(chr(10), "\\n").replace(chr(13), "\\r")
                    print(f"{i}: {generated_portion}")
                elif return_full_text == "full_and_completion":
                    if is_chat:
                        full_text = tokenizer.decode(generated[i], skip_special_tokens=skip_special_tokens_for_decode)
                    else:
                        # For regular text, decode normally
                        full_text = tokenizer.decode(generated[i], skip_special_tokens=skip_special_tokens_for_decode)

                    # Print for convenience (showing generated portion with escape sequences)
                    generated_portion = tokenizer.decode(generated[i][-num_tokens:], skip_special_tokens=False)
                    generated_portion = generated_portion.replace(chr(10), "\\n").replace(chr(13), "\\r")
                    print(f"{i}: {generated_portion}")
                    outputs.append([full_text, generated_portion])
                else:
                    # Only return the generated portion
                    generated_text = tokenizer.decode(
                        generated[i][-num_tokens:], skip_special_tokens=skip_special_tokens_for_decode
                    )
                    outputs.append(generated_text)
                    display_text = generated_text.replace(chr(10), "\\n").replace(chr(13), "\\r")
                    print(f"{i}: {display_text}")

            return outputs

    def send_message(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        chat_tokenizer=None,
        use_cache: bool = True,
        cancellation_check=None,
    ) -> str:
        """
        Send a chat-formatted message and return a single assistant response.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate (will stop early at end-of-turn)
            chat_tokenizer: Tokenizer to use for chat models (if different from self.tokenizer)
            use_cache: Whether to use KV caching
            cancellation_check: Optional callable to check for cancellation
            
        Returns:
            str: The assistant's response message content (without role markers)
        """
        # Check for cancellation before starting
        if cancellation_check and cancellation_check():
            print("Generation cancelled by user before starting.")
            return ""
        
        # Validate input
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("messages must be a list of dicts with 'role' and 'content' keys")
        
        # Use the appropriate tokenizer
        tokenizer = chat_tokenizer if chat_tokenizer is not None else self.tokenizer

        # Check for multiple system prompts
        num_system = sum(1 for m in messages if m.get("role") == "system")
        if num_system > 1:
            raise ValueError("Multiple system prompts detected in chat messages.")

        # Special handling for Gemma: prepend system prompt to first user message
        if 'gemma' in self.model_name.lower() and messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = system_prompt + ("\n\n" if messages[0]["content"] else "") + messages[0]["content"]
            print(f"Added system prompt for Gemma as user prefix")

        # Check if last message is an assistant prefill
        is_assistant_prefill = messages and messages[-1]["role"] == "assistant"
        
        # Apply chat template with appropriate settings
        if is_assistant_prefill:
            print(f"Assistant prefill detected, not adding generation prompt and continuing the final message")
            # For assistant prefill, don't add generation prompt and continue the final message
            input_dict = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            print(f"No assistant prefill detected, adding generation prompt and continuing the final message")
            # Normal case: add generation prompt for assistant response
            input_dict = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                continue_final_message=False,
            )

        input_ids = input_dict["input_ids"].to(self.orig_model.model.device)
        attention_mask = input_dict.get("attention_mask", torch.ones_like(input_ids)).to(
            self.orig_model.model.device
        )
        
        # Get the assistant end token(s) for this tokenizer
        # Common patterns for different models:
        assistant_end_tokens = []
        
        # Try to get end-of-turn tokens from the tokenizer
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            assistant_end_tokens.append(tokenizer.eos_token_id)
        
        # Check for specific chat template tokens
        # For models like Gemma, Llama, etc.
        special_tokens = [
            "<end_of_turn>", "<|end_of_turn|>", "</turn>", "</s>", "[/INST]", 
            "</assistant>", "<|eot_id|>", "<|end|>", "<|im_end|>", "<|endoftext|>"
        ]
        
        for token_str in special_tokens:
            try:
                # Try to get token ID using convert_tokens_to_ids if available
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    token_id = tokenizer.convert_tokens_to_ids(token_str)
                    if token_id is not None and token_id != tokenizer.unk_token_id:
                        assistant_end_tokens.append(token_id)
                
                # Also try encoding (for tokens that might be in the vocabulary)
                token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                if token_ids and len(token_ids) == 1:  # Only single-token markers
                    assistant_end_tokens.append(token_ids[0])
            except:
                logger.warning(f"Could not find end-of-turn token for {token_str}")
                pass
        
        # Remove duplicates while preserving order
        assistant_end_tokens = list(dict.fromkeys(assistant_end_tokens))
        
        # If we didn't find any specific end tokens, fall back to EOS
        if not assistant_end_tokens and hasattr(tokenizer, 'eos_token_id'):
            assistant_end_tokens = [tokenizer.eos_token_id]
        
        # Generate with early stopping at end-of-turn markers
        with torch.no_grad():
            generated = self.orig_model.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else tokenizer.eos_token_id,
                eos_token_id=assistant_end_tokens if assistant_end_tokens else tokenizer.eos_token_id,
                use_cache=use_cache,
                cache_implementation="static" if use_cache else None,
                disable_compile=True,
            )
        
        # Extract only the generated tokens (not the input)
        generated_tokens = generated[0][input_ids.shape[1]:]
        
        # Decode the response
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up any remaining role markers that might have slipped through
        # This handles cases where skip_special_tokens doesn't catch everything
        # cleanup_patterns = [
        #     r"^assistant:\s*", r"^Assistant:\s*", r"^ASSISTANT:\s*",
        #     r"^\[ASSISTANT\]\s*", r"^<\|assistant\|>\s*"
        # ]
        
        # import re
        # for pattern in cleanup_patterns:
        #     response = re.sub(pattern, "", response, flags=re.IGNORECASE)
        
        return response#.strip()

    def analyze_all_layers_at_position(
        self,
        text: str,
        position_to_analyze: int,
        seed=42,
        batch_size=None,
        no_eval=False,
        tuned_lens: bool = True,
        add_tokens=None,
        replace_left=None,
        replace_right=None,
        do_hard_tokens=False,
        return_structured=False,
        move_devices=False,
        logit_lens_analysis: bool = False,
        temperature: float = 1.0,
        no_kl: bool = False,
        calculate_token_salience: bool = False,
        optimize_explanations_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Analyze all layers at a specific token position and return results as DataFrame.

        Args:
            text: Text to analyze
            position_to_analyze: The 0-indexed token position to analyze across all layers.
            seed: Random seed
            batch_size: Batch size for processing
            no_eval: Skip evaluation metrics (KL, MSE)
            tuned_lens: Include TunedLens predictions in results
            logit_lens_analysis: Add logit-lens predictions (projecting hidden state through unembedding).
            temperature: Sampling temperature for explanation generation.
            no_kl: If True, skip KL divergence calculation. Only has an effect if no_eval is False.
            calculate_token_salience: If True, calculate a salience score for each token in the explanation.
            optimize_explanations_config: If provided, optimize explanations using the given configuration.
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.clone()
        attention_mask = inputs.attention_mask.clone()
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        if inputs.input_ids[0][0] == inputs.input_ids[0][1]:
            print(
                f"First two tokens are the same: {self.tokenizer.decode(inputs.input_ids[0][0])}, {self.tokenizer.decode(inputs.input_ids[0][1])}, removing first token"
            )
            input_ids = input_ids[:, 1:].clone()
            attention_mask = attention_mask[:, 1:].clone()
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        attention_mask = attention_mask.to(self.device)

        # Handle negative position index
        if position_to_analyze < 0:
            position_to_analyze = seq_len + position_to_analyze
        if not 0 <= position_to_analyze < seq_len:
            raise ValueError(f"position_to_analyze {position_to_analyze} is out of bounds for seq_len {seq_len}")

        token_at_pos = self.tokenizer.decode([input_ids[0, position_to_analyze].item()])

        # Get all hidden states for all layers at the specified position
        if move_devices and self.orig_model.model != self.shared_base_model:
            self.shared_base_model.to("cpu")
            self.orig_model.model.to("cuda")

        A_full_sequence = self.orig_model.get_all_layers_activations_at_position(
            input_ids,
            position_to_analyze,
            attention_mask=attention_mask,
            no_grad=True,
        )
        if move_devices and self.orig_model.model != self.shared_base_model:
            self.orig_model.model.to("cpu")
            self.shared_base_model.to("cuda")

        # A_full_sequence: (num_layers, hidden_dim)
        num_layers = A_full_sequence.shape[0]

        torch.manual_seed(seed)

        all_kl_divs = []
        all_mses = []
        all_relative_rmses = []
        all_gen_hard_token_ids = []
        all_tuned_lens_predictions = [] if tuned_lens else None
        all_logit_lens_predictions = [] if logit_lens_analysis else None
        all_salience_scores = [] if calculate_token_salience else None

        if do_hard_tokens:
            print("Using original hard tokens")
            replace_left, replace_right, _, _ = self.decoder.tokenize_and_embed_prompt(
                self.decoder.prompt_text, self.tokenizer
            )

        batch_size = min(batch_size, num_layers)
        with torch.no_grad():
            try:
                iterator = tqdm.tqdm(
                    range(0, num_layers, batch_size),
                    desc="Analyzing layers in batches"
                    + (" with tokens `" + self.tokenizer.decode(add_tokens) + "`" if add_tokens is not None else ""),
                )
            except ImportError:
                iterator = range(0, num_layers, batch_size)

            for i in iterator:
                batch_start = i
                batch_end = min(i + batch_size, num_layers)
                current_batch_size = batch_end - batch_start

                layer_indices_batch = torch.arange(batch_start, batch_end, device=self.device)
                A_batch = A_full_sequence[layer_indices_batch]

                # The position to analyze is fixed for all layers in the batch.
                token_pos_for_decoder = torch.full(
                    (current_batch_size,), position_to_analyze, device=self.device, dtype=torch.long
                )

                # Generate or optimize explanations
                if optimize_explanations_config is None:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                        # Generate explanations for all layers in the current batch.
                        if self.decoder.config.use_kv_cache:
                            gen = self.decoder.generate_soft_kv_cached_nondiff(
                                A_batch,
                                max_length=self.t_text,
                                gumbel_tau=self.tau,
                                original_token_pos=token_pos_for_decoder,
                                return_logits=True,
                                add_tokens=add_tokens,
                                hard_left_emb=replace_left,
                                hard_right_emb=replace_right,
                                temperature=temperature,
                            )
                        else:
                            logger.warning("Not using KV cache for generate_soft - this is not recommended.")
                            raise ValueError("Not using KV cache for generate_soft - this is not recommended.")
                            gen = self.decoder.generate_soft(
                                A_batch,
                                max_length=self.t_text,
                                gumbel_tau=self.tau,
                                original_token_pos=token_pos_for_decoder,
                                return_logits=True,
                                add_tokens=add_tokens,
                                hard_left_emb=replace_left,
                                hard_right_emb=replace_right,
                                temperature=temperature,
                            )

                    all_gen_hard_token_ids.append(gen.hard_token_ids)
                else:
                    # Optimize explanations
                    use_batched = optimize_explanations_config.get("use_batched", True)
                    best_of_k = optimize_explanations_config.get("best_of_k", 8)

                    if use_batched and best_of_k:
                        # Use batched optimization
                        num_rollouts = best_of_k if isinstance(best_of_k, int) else 8
                        n_groups_per_rollout = optimize_explanations_config.get("n_groups_per_rollout", 8)

                        optimized_tokens, optimized_mses, optimized_salience_scores = self._optimize_batch_best_of_k(
                            A_batch,
                            token_pos_for_decoder,
                            input_ids,
                            num_rollouts_per_position=num_rollouts,
                            temperature=optimize_explanations_config.get("temperature", temperature),
                            n_groups_per_rollout=n_groups_per_rollout,
                        )

                        all_gen_hard_token_ids.append(optimized_tokens)
                    else:
                        # Use sequential optimization
                        optimized_tokens_list = []

                        for j in tqdm.tqdm(range(current_batch_size), desc="Optimizing explanations", leave=False):
                            layer_idx = layer_indices_batch[j].item()
                            optimized_tokens, optimized_mse, optimized_saliences = self.optimize_explanation_for_mse(
                                None,
                                A_target=A_batch[j],
                                position_to_analyze=position_to_analyze,
                                input_ids=input_ids,
                                verbose=False,
                                **optimize_explanations_config,
                            )
                            optimized_tokens_list.append(optimized_tokens)

                        optimized_hard_token_ids = torch.stack(optimized_tokens_list, dim=0)
                        all_gen_hard_token_ids.append(optimized_hard_token_ids)

                    # Logit-lens analysis can be batched easily
                    if logit_lens_analysis:
                        unembedding_layer = self.orig_model.model.get_output_embeddings()
                        logits_logit_lens = unembedding_layer(A_batch)
                        top_tokens_batch = [
                            " ".join(get_top_n_tokens(logits_logit_lens[j], self.tokenizer, min(self.t_text, 10)))
                            for j in range(logits_logit_lens.shape[0])
                        ]
                        all_logit_lens_predictions.extend(top_tokens_batch)

                # Initialize salience scores list for this batch
                batch_salience_scores = [[] for _ in range(current_batch_size)]

                # --- Non-batched section for eval and TunedLens ---

                # These require single layer_idx, so we loop inside the batch.
                if not no_eval or (tuned_lens and self.comparison_tuned_lens is not None) or calculate_token_salience:
                    # If we optimized, we need to create gen object from optimized tokens
                    if optimize_explanations_config is not None:
                        # Get the last batch of optimized tokens
                        optimized_batch_tokens = all_gen_hard_token_ids[-1]

                        # Convert tokens to embeddings for gen object
                        with torch.no_grad():
                            if hasattr(self.decoder.base, "get_input_embeddings"):
                                input_emb_table = self.decoder.base.get_input_embeddings().weight
                            else:
                                # Fallback for different model architectures
                                input_emb_table = self.decoder.base.transformer.wte.weight

                            # Convert token IDs to embeddings
                            optimized_embeddings = input_emb_table[optimized_batch_tokens]

                            # Handle Gemma3 normalization if needed
                            if hasattr(self.decoder, "is_gemma3") and self.decoder.is_gemma3:
                                optimized_embeddings = optimized_embeddings * self.decoder.normalizer

                        # Create Generated instance with optimized results
                        gen = Generated(
                            generated_text_embeddings=optimized_embeddings,
                            raw_lm_logits=None,  # We don't have logits from optimization
                            hard_token_ids=optimized_batch_tokens,
                        )

                    for j in range(current_batch_size):
                        layer_idx = layer_indices_batch[j].item()
                        A_single = A_batch[j]

                    if not no_eval:
                        # Reconstruct activation for one layer
                        generated_token_embeddings = self.encoder.base.get_input_embeddings()(
                            gen.hard_token_ids[j]
                        ).unsqueeze(0)
                        current_token_id_single = input_ids.squeeze(0)[position_to_analyze].unsqueeze(0)

                        A_hat_single = self.encoder(
                            generated_token_embeddings,
                            original_token_pos=torch.tensor([position_to_analyze], device=self.device),
                            current_token_ids=current_token_id_single
                            if self.encoder.config.add_current_token
                            else None,
                        )

                        # MSE
                        mse = torch.nn.functional.mse_loss(A_single, A_hat_single.squeeze(0))
                        all_mses.append(mse.unsqueeze(0))

                        # Relative RMSE
                        relative_rmse = np.sqrt(mse / ((A_single**2).mean() + 1e-9))
                        all_relative_rmses.append(relative_rmse.unsqueeze(0))

                        # Compute KL divergence only if not no_kl
                        if not no_kl:
                            # Get original and reconstructed logits (cannot be batched across layers)
                            logits_orig = self.orig_model.forward_with_replacement(
                                input_ids, A_single, layer_idx, position_to_analyze, no_grad=True
                            ).logits[:, position_to_analyze, :]

                            logits_recon = self.orig_model.forward_with_replacement(
                                input_ids, A_hat_single.squeeze(0), layer_idx, position_to_analyze, no_grad=True
                            ).logits[:, position_to_analyze, :]

                            # Compute KL divergence
                            with torch.amp.autocast("cuda", enabled=False):
                                log_probs_orig = torch.log_softmax(logits_orig.float(), dim=-1)
                                log_probs_recon = torch.log_softmax(logits_recon.float(), dim=-1)
                                kl_div = (torch.exp(log_probs_orig) * (log_probs_orig - log_probs_recon)).sum(dim=-1)
                                all_kl_divs.append(kl_div)
                        else:
                            all_kl_divs.append(torch.tensor(-1.0, device=self.device))

                        # Calculate salience if requested
                        if calculate_token_salience:
                            salience_scores = self.calculate_salience(
                                gen.hard_token_ids[j],
                                mse.item(),
                                A_single,
                                position_to_analyze,
                                input_ids,
                            )
                            batch_salience_scores[j] = salience_scores

                if tuned_lens and self.comparison_tuned_lens is not None:
                    try:
                        A_single_f32 = A_single.to(torch.float32)
                        logits_tl = self.comparison_tuned_lens(A_single_f32, idx=layer_idx)
                        top_tokens = " ".join(get_top_n_tokens(logits_tl, self.tokenizer, min(self.t_text, 10)))
                        all_tuned_lens_predictions.append(top_tokens)
                    except Exception:
                        all_tuned_lens_predictions.append(f"[TL error on L{layer_idx}]")

                # Append batch salience scores to the collection
                if calculate_token_salience:
                    all_salience_scores.extend(batch_salience_scores)

        # Concatenate results from all batches
        gen_hard_token_ids = torch.cat(all_gen_hard_token_ids)
        if not no_eval:
            if not no_kl:
                kl_divs = torch.cat(all_kl_divs)
            else:
                kl_divs = torch.zeros(num_layers)
            mses = torch.cat(all_mses)
            relative_rmses = torch.cat(all_relative_rmses)
        else:
            kl_divs = torch.zeros(num_layers)
            mses = torch.zeros(num_layers)
            relative_rmses = torch.zeros(num_layers)

        # Decode and collect results into a list of dicts.
        results = []
        for l in range(num_layers):
            explanation = self.tokenizer.decode(gen_hard_token_ids[l], skip_special_tokens=False) + (
                "[" + token_at_pos + "]" if self.encoder.config.add_current_token else ""
            )
            explanation_structured = [
                self.tokenizer.decode(gen_hard_token_ids[l][i], skip_special_tokens=False)
                for i in range(len(gen_hard_token_ids[l]))
            ] + (["[" + token_at_pos + "]"] if self.encoder.config.add_current_token else [])

            result_dict = {
                "layer": l,
                "token": token_at_pos,
                "explanation": explanation,
                "kl_divergence": kl_divs[l].item(),
                "mse": mses[l].item(),
                "relative_rmse": relative_rmses[l].item(),
            }

            if tuned_lens and all_tuned_lens_predictions:
                result_dict["tuned_lens_top"] = all_tuned_lens_predictions[l]
            elif tuned_lens and self.comparison_tuned_lens is None:
                result_dict["tuned_lens_top"] = "[TunedLens not loaded]"

            if logit_lens_analysis and all_logit_lens_predictions:
                result_dict["logit_lens_top"] = all_logit_lens_predictions[l]

            if return_structured:
                result_dict["explanation_structured"] = explanation_structured

            if calculate_token_salience and all_salience_scores:
                result_dict["token_salience"] = all_salience_scores[l]

            results.append(result_dict)

        return LensDataFrame(
            results,
            checkpoint_path=self.checkpoint_path,
            model_name=self.model_name,
            orig_model_name=self.orig_model_name,
        )


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
        kl = row["kl_divergence"]
        mse = row["mse"]
        # Color code based on KL value
        if kl < df["kl_divergence"].quantile(0.33):
            indicator = "🟢"
        elif kl < df["kl_divergence"].quantile(0.67):
            indicator = "🟡"
        else:
            indicator = "🔴"

        print(
            f"{indicator} [{row['position']:2d}] {repr(row['token']):15} → {row['explanation']:40} (KL: {kl:.3f} MSE: {mse:.3f})"
        )

    if show_plot and len(df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(df["position"], df["kl_divergence"], "b-", linewidth=2, marker="o")
        plt.xlabel("Position")
        plt.ylabel("KL Divergence")
        plt.yscale("log")
        plt.title(f'KL Divergence: "{text[:40]}..."')
        plt.grid(True, alpha=0.3)

        # Annotate some points
        for i in range(0, len(df), max(1, len(df) // 5)):
            plt.annotate(
                repr(df.iloc[i]["token"]),
                (df.iloc[i]["position"], df.iloc[i]["kl_divergence"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
        plt.ylim(None, max(df["kl_divergence"][1:]))
        plt.tight_layout()
        plt.show()

    if show_plot and len(df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(df["position"], df["mse"], "r-", linewidth=2, marker="o")
        plt.xlabel("Position")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.title(f'MSE: "{text[:40]}..."')
        plt.grid(True, alpha=0.3)
        for i in range(0, len(df), max(1, len(df) // 5)):
            plt.annotate(
                repr(df.iloc[i]["token"]),
                (df.iloc[i]["position"], df.iloc[i]["mse"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
        plt.ylim(None, max(df["mse"][1:]))
        plt.tight_layout()
        plt.show()

    return df


#


def analyze_ordering_effects(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ordering effects by comparing original vs switched query results.
    """
    # Separate original and switched results
    original_results = results_df[~results_df["is_switched"]].copy()
    switched_results = results_df[results_df["is_switched"]].copy()

    # Calculate position bias (preference for A vs B)
    def calculate_position_proportion_A(answers):
        """Calculate bias toward position A"""
        valid_answers = [a for a in answers if a in ["A", "B"]]
        if not valid_answers:
            return 0.0
        return sum(1 for a in valid_answers if a == "A") / len(valid_answers)

    # Aggregate results
    ordering_analysis = []

    for _, orig_row in original_results.iterrows():
        # Find corresponding switched result
        switched_row = switched_results[switched_results["original_question_id"] == orig_row["original_question_id"]]
        print(f"looking at questions {orig_row['question_id']} and {switched_row['question_id']}")

        if len(switched_row) == 0:
            continue

        switched_row = switched_row.iloc[0]

        # Calculate metrics
        orig_neutral_a_proportion_A = calculate_position_proportion_A(orig_row["neutral_answers"])
        orig_harmful_a_proportion_A = calculate_position_proportion_A(orig_row["harmful_answers"])
        switch_neutral_a_proportion_A = calculate_position_proportion_A(switched_row["neutral_answers"])
        switch_harmful_a_proportion_A = calculate_position_proportion_A(switched_row["harmful_answers"])
        print(
            f"orig_neutral_a_proportion_A: {orig_neutral_a_proportion_A}, orig_harmful_a_proportion_A: {orig_harmful_a_proportion_A}, switch_neutral_a_proportion_A: {switch_neutral_a_proportion_A}, switch_harmful_a_proportion_A: {switch_harmful_a_proportion_A}"
        )

        # Content-based accuracy (de-biased by averaging over original and switched prompts)
        content_neutral_acc = (orig_row["neutral_accuracy"] + switched_row["neutral_accuracy"]) / 2
        content_harmful_acc = (orig_row["harmful_accuracy"] + switched_row["harmful_accuracy"]) / 2

        ordering_analysis.append(
            {
                "question_id": orig_row["original_question_id"],
                "category": orig_row["category"],
                "orig_neutral_a_proportion_A": orig_neutral_a_proportion_A,
                "orig_harmful_a_proportion_A": orig_harmful_a_proportion_A,
                "switch_neutral_a_proportion_A": switch_neutral_a_proportion_A,
                "switch_harmful_a_proportion_A": switch_harmful_a_proportion_A,
                "avg_neutral_a_bias": (orig_neutral_a_proportion_A + switch_neutral_a_proportion_A) / 2,
                "avg_harmful_a_bias": (orig_harmful_a_proportion_A + switch_harmful_a_proportion_A) / 2,
                "orig_sandbagging": orig_row["sandbagging_rate"],
                "switch_sandbagging": switched_row["sandbagging_rate"],
                "content_neutral_accuracy": content_neutral_acc,
                "content_harmful_accuracy": content_harmful_acc,
                "position_effect_neutral": abs(orig_neutral_a_proportion_A - (1 - switch_neutral_a_proportion_A)),
                "position_effect_harmful": abs(orig_harmful_a_proportion_A - (1 - switch_harmful_a_proportion_A)),
            }
        )

    return pd.DataFrame(ordering_analysis)


def plot_ordering_effects(ordering_df: pd.DataFrame):
    """
    Visualize ordering effects analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall position bias
    ax = axes[0, 0]
    contexts = ["Neutral", "Harmful"]
    avg_biases = [ordering_df["avg_neutral_a_bias"].mean(), ordering_df["avg_harmful_a_bias"].mean()]
    ax.bar(contexts, avg_biases, alpha=0.7)
    ax.axhline(y=0.5, color="r", linestyle="--", label="No bias")
    ax.set_ylabel("Average Bias Toward (A)")
    ax.set_title("Overall Position Bias by Context")
    ax.set_ylim(0, 1)
    ax.legend()

    # 2. Position effect magnitude
    ax = axes[0, 1]
    position_effects = pd.DataFrame(
        {"Neutral": ordering_df["position_effect_neutral"], "Harmful": ordering_df["position_effect_harmful"]}
    )
    position_effects.boxplot(ax=ax)
    ax.set_ylabel("Position Effect (|Original - Switched|)")
    ax.set_title("Distribution of Position Effects")

    # 3. Sandbagging comparison
    ax = axes[1, 0]
    ax.scatter(ordering_df["orig_sandbagging"], ordering_df["switch_sandbagging"], alpha=0.6)
    ax.plot([0, 1], [0, 1], "r--", label="y=x")
    ax.set_xlabel("Original Sandbagging Rate")
    ax.set_ylabel("Switched Sandbagging Rate")
    ax.set_title("Sandbagging: Original vs Switched")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    # 4. Category-wise position bias
    ax = axes[1, 1]
    category_bias = ordering_df.groupby("category")[["avg_neutral_a_bias", "avg_harmful_a_bias"]].mean()
    category_bias.plot(kind="bar", ax=ax, alpha=0.7)
    ax.axhline(y=0.5, color="r", linestyle="--")
    ax.set_ylabel("Average Bias Toward (A)")
    ax.set_title("Position Bias by Category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(["Neutral", "Harmful"])

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nOrdering Effects Summary:")
    print("=" * 60)
    print(f"Average position bias (neutral): {ordering_df['avg_neutral_a_bias'].mean():.3f}")
    print(f"Average position bias (harmful): {ordering_df['avg_harmful_a_bias'].mean():.3f}")
    print(f"Average position effect (neutral): {ordering_df['position_effect_neutral'].mean():.3f}")
    print(f"Average position effect (harmful): {ordering_df['position_effect_harmful'].mean():.3f}")
    print("\nQueries with strong position effects (>0.3):")
    strong_effects = ordering_df[
        (ordering_df["position_effect_neutral"] > 0.3) | (ordering_df["position_effect_harmful"] > 0.3)
    ]
    print(f"Count: {len(strong_effects)} / {len(ordering_df)}")
