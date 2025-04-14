import torch
import nnsight
from nnsight import util
from safetensors.torch import load_file
import os
import json
from tqdm.auto import tqdm
from circuitsvis.tokens import colored_tokens_multi
from huggingface_hub import hf_hub_download
from typing import Optional, Callable
import torch
import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG, util
from nnsight.intervention import InterventionProxy
InterventionInterface = Callable[[InterventionProxy], InterventionProxy]
from nnsight import download_file

class ObservableLanguageModel:
    """Wraps nnsight.LanguageModel to provide activation caching and intervention interface,
       based on the Goodfire demo notebook."""
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16, # Default to bfloat16 as in demo
    ):
        self.dtype = dtype
        self.device = device
        self._original_model_name = model_name

        # Initialize the nnsight LanguageModel
        # Note: nnsight might automatically handle device_map='auto' here
        # We specify dtype explicitly
        self._model = nnsight.LanguageModel(
            self._original_model_name,
            device_map=device, # Pass the determined device/auto
            torch_dtype=self.dtype
        )
        
        # Trigger model download/loading if lazy (nnsight behavior)
        # Using a simple input and trace context
        print("Running initial trace to load model...")
        try:
            # Try a simple trace first
            input_tokens = self._model.tokenizer("hello", return_tensors="pt")["input_ids"].to(self.device)
            with self._model.trace(input_tokens):
                pass
        except Exception as e:
            print(f"Initial trace failed (may be expected during setup): {e}")
            # If it fails, it might be due to model needing specific chat template
            try:
                print("Trying trace with chat template...")
                chat_input = self._model.tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], return_tensors="pt").to(self.device)
                with self._model.trace(chat_input):
                     pass
            except Exception as e2:
                 print(f"Chat template trace also failed: {e2}. Model loading might have issues.")

        print("Model should be loaded now.")
        self.tokenizer = self._model.tokenizer
        self.config = self._model.config # Store config for easy access
        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()
        
        # Expose the underlying nnsight model's device and dtype
        self.model_device = self._model.device
        self.model_dtype = self._model.dtype

        # NNsight validation scan - keep False for performance unless debugging
        self.safe_mode = False 

    def _attempt_to_infer_hidden_layer_dimensions(self):
        """Infers the hidden dimension size (d_model) from the model's config."""
        if hasattr(self.config, "hidden_size"):
            return int(self.config.hidden_size)
        # Add other potential attribute names if needed (e.g., d_model)
        elif hasattr(self.config, "d_model"):
             return int(self.config.d_model)

        raise AttributeError(
            "Could not infer hidden layer dimension ('hidden_size' or 'd_model') from model config."
        )

    def _find_module(self, hook_point: str):
        """Navigates the module hierarchy to find a module by its string path."""
        # Uses nnsight's utility function for robust attribute fetching
        try:
            return util.fetch_attr(self._model, hook_point)
        except AttributeError:
             print(f"Warning: Could not find module at hook_point: {hook_point}")
             # You might want to print the model structure here to help debug
             # print(self._model)
             raise

    def forward(
        self,
        inputs: torch.Tensor, # Expects tensor of token IDs
        cache_activations_at: Optional[list[str]] = None,
        interventions: Optional[dict[str, InterventionInterface]] = None,
        # Note: use_cache and past_key_values are part of nnsight's trace context, 
        #       not direct args here unlike the original demo method.
        #       Generation/KV caching would be handled differently if needed.
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Performs a forward pass, optionally caching activations and applying interventions.
           Returns logits for the last token and a dictionary of cached activations."""
        cache: dict[str, torch.Tensor] = {}
        
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.model_device)
        
        with self._model.trace(
            inputs,
            scan=self.safe_mode,
            validate=self.safe_mode,
            # use_cache=True, # KV Caching for generation - not needed for single fwd pass analysis
        ) as tracer:
            # Apply interventions if provided
            if interventions:
                for hook_site, intervention_fn in interventions.items():
                    if intervention_fn is None:
                        continue
                    
                    module = self._find_module(hook_site)
                    
                    # The intervention function receives the module's output proxy
                    # Note: Llama output is often (hidden_state, kv_cache_tuple)
                    # We usually intervene on the hidden_state, module.output[0]
                    intervened_acts = intervention_fn(module.output[0])
                    
                    # Set the module's output. If KV caching was enabled in trace,
                    # preserve the second element (the cache tuple).
                    if isinstance(module.output, tuple) and len(module.output) > 1:
                         module.output = (intervened_acts, module.output[1])
                    else:
                         module.output = intervened_acts # Assume output was just the hidden state

            # Cache activations if requested
            if cache_activations_at is not None:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    # Save the hidden state part of the output (usually index 0)
                    # Ensure it's saved correctly, could be module.output or module.output[0]
                    # Let's try saving the whole output first and inspect if needed
                    output_to_save = module.output
                    if isinstance(output_to_save, tuple):
                         output_to_save = output_to_save[0] # Assume hidden state is first
                    
                    # Check if output_to_save is already a Proxy or Tensor that can be saved
                    if isinstance(output_to_save, nnsight.intervention.InterventionProxy):
                        cache[hook_point] = output_to_save.save()
                    elif isinstance(output_to_save, torch.Tensor):
                         # If it's already a tensor (e.g. from an intervention), 
                         # we might need to handle saving differently or it might not be traceable.
                         # For now, assume it's a proxy.
                         print(f"Warning: Output at {hook_point} is a Tensor, direct saving might not work as expected in trace.")
                         cache[hook_point] = output_to_save # Store tensor directly (may detach from graph)
                    else:
                        print(f"Warning: Unexpected output type at {hook_point}: {type(output_to_save)}")

            # Save the logits for the *last* input token position
            # model.output[0] usually contains the full sequence logits [batch, seq, vocab]
            logits = self._model.output[0][:, -1, :].save()

        # Process cached activations after trace exits
        # Saved values are often tuples; extract the tensor
        processed_cache = {}
        for k, v in cache.items():
             value = v.value
             if isinstance(value, tuple):
                 processed_cache[k] = value[0].detach() # Assume tensor is first element
             else:
                  processed_cache[k] = value.detach()

        return (
            logits.value.detach(), # Return only last token logits
            processed_cache,
        )



class SparseAutoEncoder(torch.nn.Module):
    """SAE class definition based on Goodfire demo."""
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16, # Default to bfloat16 as in demo
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.dtype = dtype
        
        # Define layers (Matches the demo notebook structure)
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden, bias=True)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in, bias=True) # Decoder bias usually False
        
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear layer followed by a ReLU activation."""
        # Ensure input tensor is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype) 
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of feature activations using a linear layer."""
        # Ensure input tensor is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode the input, then decode the features.
           Returns the reconstructed input and the intermediate sparse features."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    @classmethod
    def from_hf(cls, repo_id, weight_filename="sae_weights.safetensors", cfg_filename="cfg.json", cache_dir="sae_cache"):
        """Loads SAE weights and config from Hugging Face Hub."""
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        repo_url = f"https://huggingface.co/{repo_id}/resolve/main/"
        weights_url = repo_url + weight_filename
        cfg_url = repo_url + cfg_filename

        local_weights_path = os.path.join(cache_dir, f"{repo_id.replace('/', '_')}_{weight_filename}")
        local_cfg_path = os.path.join(cache_dir, f"{repo_id.replace('/', '_')}_{cfg_filename}")

        # Download weights and config if they don't exist locally
        download_file(weights_url, local_weights_path)
        download_file(cfg_url, local_cfg_path)

        # Load config
        try:
            with open(local_cfg_path, 'r') as f:
                cfg = json.load(f)
            d_in = cfg['d_in']
            d_hidden = cfg['d_hidden']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error loading or parsing config file {local_cfg_path}: {e}")
            raise

        # Instantiate SAE
        sae = cls(d_in, d_hidden)

        # Load state dict
        try:
            state_dict = load_file(local_weights_path)
            # Adapt keys if necessary (e.g., remove 'sae.' prefix if present)
            adapted_state_dict = {}
            for key, value in state_dict.items():
                 # Example adaptation: remove 'sae.' prefix if needed
                 # new_key = key.replace("sae.", "") 
                 new_key = key # Assume keys match for now
                 adapted_state_dict[new_key] = value
            
            # Handle potential bias mismatch in decoder (common practice)
            if 'decoder.bias' not in sae.state_dict() and 'decoder.bias' in adapted_state_dict:
                del adapted_state_dict['decoder.bias']
                print("Note: Removed decoder bias from loaded state_dict as it's not in the model.")
            elif 'decoder.bias' in sae.state_dict() and 'decoder.bias' not in adapted_state_dict:
                 print("Warning: Decoder bias expected but not found in loaded state_dict.")
                 
            sae.load_state_dict(adapted_state_dict)
            print(f"SAE loaded successfully from {repo_id}")

        except Exception as e:
            print(f"Error loading state dict from {local_weights_path}: {e}")
            raise

        return sae