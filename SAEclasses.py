import torch
import nnsight
import requests
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
from nnsight.intervention.graph.proxy import InterventionProxy
InterventionInterface = Callable[[InterventionProxy], InterventionProxy]

class ObservableLanguageModel:
    """Wraps nnsight.LanguageModel to provide activation caching and intervention interface,
       based on the Goodfire demo notebook."""
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16, # Default to bfloat16 as in demo
    ):
        self.dtype = dtype
        self.device = device
        self._original_model_name = model_name

        # Initialize the nnsight LanguageModel
        # Note: nnsight might automatically handle device_map='auto' here
        # We specify dtype explicitly
        print("device map used:",device_map)
        print("device used:",device)
        self._model = nnsight.LanguageModel(
            self._original_model_name,
            device_map=device_map, # Pass the determined device/auto
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
                    if isinstance(output_to_save, nnsight.intervention.graph.proxy.InterventionProxy):
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
        print("logits",logits.shape)
        print("logits",logits)
        # Process cached activations after trace exits
        # Saved values are often tuples; extract the tensor
        # Nice! That worked seamlessly, but hold on, how come we didn't need to call .value[0] 
        # on the result? In previous sections, we were just being explicit to get an understanding 
        # of Proxies and their value. In practice, however, nnsight knows that when outside of the
        # tracing context we only care about the actual value, and so printing, indexing, and applying functions all 
        # immediately return and reflect the data in .value. So for the rest of the tutorial we won't use it.
        # https://nnsight.net/notebooks/tutorials/walkthrough/
        processed_cache = {}
        for k, v in cache.items():
            value = v
            print(k, getattr(value, "shape", None), value)
            if torch.is_tensor(value):
                #print("value is a tensor", value)
                processed_cache[k] = value.detach()
            elif isinstance(value, tuple) and len(value) > 0 and torch.is_tensor(value[0]):
                #print("value is a tuple with tensor", value)
                processed_cache[k] = value[0].detach()
            elif torch.is_tensor(value[0]):
                #print("value is a tuple with tensor", value)
                processed_cache[k] = value[0].detach()
            else:
                #print("value is not a tensor or tuple of tensor", value)
                processed_cache[k] = value

            print(f"added {k}, shape {processed_cache[k].shape}")

        return (
            logits.detach(), # Return only last token logits
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


def download_file(url, fname):
    """Downloads a file from a URL, showing a progress bar."""
    if os.path.exists(fname):
        print(f"File already exists: {fname}")
        return
    print(f"Downloading {url} to {fname}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(fname, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(fname):
            os.remove(fname) # Clean up incomplete download
        raise

def load_sae_from_repo(
    repo_id: str,
    model_device: torch.device,
    model_dtype: torch.dtype,
    d_model: Optional[int] = None,
    expansion_factor: Optional[int] = None,
    layer: Optional[int] = None,
    hook_point: str = "hook_resid_pre", 
    weight_filename_sf: str = "sae_weights.safetensors",
    cfg_filename_sf: str = "cfg.json",
    weight_filename_gf: str = None,
    cache_dir: str = "sae_cache",
):
    """Loads an SAE from a Hugging Face repo, handling Goodfire, standard, and jbloom formats."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if repo_id.startswith("jbloom/"):
        print(f"Detected jbloom SAE format for {repo_id}")
        if layer is None:
            raise ValueError("layer parameter is required for jbloom SAE format.")
        
        # jbloom filename convention
        filename = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.{hook_point}_24576.pt"
        print(f"Loading jbloom SAE: {filename}")
        
        try:
            # Download .pt file using hf_hub_download
            local_weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                repo_type="model"
            )
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            raise
        
        # Create a more complete dummy sae_training module
        print("Setting up dummy sae_training module...")
        import sys
        import types
        from dataclasses import dataclass
        
        # Clean up any existing modules first
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('sae_training')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Create sae_training module
        sae_training = types.ModuleType('sae_training')
        sys.modules['sae_training'] = sae_training
        
        # Create config submodule
        config_module = types.ModuleType('sae_training.config')
        sys.modules['sae_training.config'] = config_module
        sae_training.config = config_module
        
        # Create a more complete dummy config class
        @dataclass
        class LanguageModelSAERunnerConfig:
            # Common attributes that might be expected
            d_in: int = d_model or 768
            d_sae: int = 24576
            device: str = 'cuda'
            dtype: torch.dtype = torch.float32
            l1_coefficient: float = 1e-3
            learning_rate: float = 1e-4
            train_batch_size_tokens: int = 4096
            context_size: int = 128
            
            def __init__(self, **kwargs):
                # Set defaults first
                self.d_in = d_model or 768
                self.d_sae = 24576
                self.device = 'cuda'
                self.dtype = torch.float32
                self.l1_coefficient = 1e-3
                self.learning_rate = 1e-4
                self.train_batch_size_tokens = 4096
                self.context_size = 128
                
                # Override with any provided kwargs
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def __getattr__(self, name):
                # Return None for any missing attributes
                return None
            
            def __setattr__(self, name, value):
                # Allow setting any attribute
                object.__setattr__(self, name, value)
            
            def __getstate__(self):
                return self.__dict__
            
            def __setstate__(self, state):
                self.__dict__.update(state)
        
        # Add the class to the config module
        config_module.LanguageModelSAERunnerConfig = LanguageModelSAERunnerConfig
        
        # Also add it to the main sae_training module for good measure
        sae_training.LanguageModelSAERunnerConfig = LanguageModelSAERunnerConfig
        
        print("Created comprehensive dummy sae_training module")
        
        # Now try loading with a more robust approach
        try:
            print("Loading checkpoint...")
            
            # First try the standard approach
            try:
                checkpoint = torch.load(local_weights_path, map_location=model_device, weights_only=False)
                print("Successfully loaded checkpoint with standard method!")
            except Exception as e:
                print(f"Standard loading failed: {e}")
                
                # If that fails, try with a custom pickle loader
                print("Trying custom pickle loading...")
                import pickle
                
                class CustomUnpickler(pickle.Unpickler):
                    def persistent_load(self, pid):
                        # Handle persistent load instructions
                        print(f"Persistent load called with: {pid}")
                        return None
                    
                    def find_class(self, module, name):
                        print(f"Looking for class: {module}.{name}")
                        if module.startswith('sae_training'):
                            if name == 'LanguageModelSAERunnerConfig':
                                return config_module.LanguageModelSAERunnerConfig
                            else:
                                # Return a dummy class for other sae_training classes
                                class DummyClass:
                                    def __init__(self, *args, **kwargs):
                                        pass
                                return DummyClass
                        return super().find_class(module, name)
                
                with open(local_weights_path, 'rb') as f:
                    checkpoint = CustomUnpickler(f).load()
                print("Successfully loaded checkpoint with custom unpickler!")
            
        except Exception as e:
            print(f"All loading methods failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Extract model configuration and weights
        print("Checkpoint type:", type(checkpoint))
        if isinstance(checkpoint, dict):
            print("Checkpoint keys:", list(checkpoint.keys()))
        
        # Extract config and dimensions
        d_in = d_model or 768  # Default for GPT-2 small
        d_sae = 24576  # Default for jbloom SAEs
        
        if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
            cfg = checkpoint['cfg']
            if hasattr(cfg, 'd_in'):
                d_in = cfg.d_in
            if hasattr(cfg, 'd_sae'):
                d_sae = cfg.d_sae
            print(f"Extracted config: d_in={d_in}, d_sae={d_sae}")
        
        # Find the state dict
        state_dict = None
        if isinstance(checkpoint, dict):
            # Try common state dict keys
            for key in ['state_dict', 'model_state_dict', 'model']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"Found state dict under key: {key}")
                    break
            
            # If no state dict found, check if checkpoint itself has weight keys
            if state_dict is None:
                weight_keys = [k for k in checkpoint.keys() if any(w in k.lower() for w in ['weight', 'bias', 'w_enc', 'w_dec'])]
                if weight_keys:
                    state_dict = checkpoint
                    print("Using checkpoint itself as state dict")
                    print(f"Weight keys found: {weight_keys}")
        else:
            state_dict = checkpoint
        
        if state_dict is None:
            raise ValueError("Could not find state dict in checkpoint")
        
        print(f"State dict keys: {list(state_dict.keys())}")
        
        # Instantiate SAE
        sae = SparseAutoEncoder(
            d_in=d_in,
            d_hidden=d_sae,
            device=model_device,
            dtype=model_dtype
        )
        
        # Map jbloom state dict keys to your SAE class
        adapted_state_dict = {}
        for key, value in state_dict.items():
            new_key = None
            
            # jbloom typically uses W_enc, b_enc, W_dec, b_dec
            if key == "W_enc":
                new_key = "encoder_linear.weight"
                # jbloom W_enc is shape (d_in, d_hidden), but nn.Linear expects (d_hidden, d_in)
                value = value.T  # Transpose
            elif key == "b_enc":
                new_key = "encoder_linear.bias"
            elif key == "W_dec":
                new_key = "decoder_linear.weight"
                # jbloom W_dec is shape (d_hidden, d_in), but nn.Linear expects (d_in, d_hidden)
                value = value.T  # Transpose
            elif key == "b_dec":
                new_key = "decoder_linear.bias"
            else:
                print(f"Skipping unknown key: {key}")
                continue
            
            if torch.is_tensor(value):
                adapted_state_dict[new_key] = value.to(device=model_device, dtype=model_dtype)
                print(f"Mapped {key} -> {new_key}, shape: {value.shape}")
            else:
                print(f"Warning: {key} is not a tensor, skipping")
        
        print("Final mapped state dict keys:", list(adapted_state_dict.keys()))
        
        # Validate we have the minimum required weights
        required_keys = ["encoder_linear.weight", "decoder_linear.weight"]
        missing_required = [k for k in required_keys if k not in adapted_state_dict]
        if missing_required:
            raise ValueError(f"Missing required keys: {missing_required}")
        
        # Handle decoder bias (SAEs usually don't have decoder bias)
        if 'decoder_linear.bias' not in sae.state_dict() and 'decoder_linear.bias' in adapted_state_dict:
            del adapted_state_dict['decoder_linear.bias']
            print("Note: Removed decoder bias from loaded state_dict")
        
        # Load the state dict
        missing_keys, unexpected_keys = sae.load_state_dict(adapted_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"jbloom SAE loaded successfully!")
        
    elif repo_id.startswith("Goodfire/"):
        print(f"Detected Goodfire SAE format for {repo_id}")
        if d_model is None or expansion_factor is None:
            raise ValueError("d_model and expansion_factor are required for Goodfire SAE format.")
        
        # Determine filename if not provided (Goodfire convention)
        if weight_filename_gf is None:
            model_name = repo_id.split('/')[1]
            weight_filename_gf = f"{model_name}.pth"
            print(f"Assuming Goodfire weight filename: {weight_filename_gf}")
        
        try:
            local_weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=weight_filename_gf,
                cache_dir=cache_dir,
                repo_type="model"
            )
        except Exception as e:
            if weight_filename_gf.endswith('.pth'):
                print(f"Failed to download {weight_filename_gf}, trying .pt format instead")
                weight_filename_gf = weight_filename_gf.replace('.pth', '.pt')
                local_weights_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=weight_filename_gf,
                    cache_dir=cache_dir,
                    repo_type="model"
                )
            else:
                raise
        
        d_hidden = d_model * expansion_factor
        
        sae = SparseAutoEncoder(
            d_in=d_model,
            d_hidden=d_hidden,
            device=model_device,
            dtype=model_dtype
        )
        
        try:
            state_dict = torch.load(local_weights_path, map_location=model_device, weights_only=True)
            print("SAE keys", state_dict.keys())
            sae.load_state_dict(state_dict)
            print(f"Goodfire SAE loaded successfully from {local_weights_path}")
        except Exception as e:
            print(f"Error loading state dict from {local_weights_path}: {e}")
            raise
    else:
        print(f"Assuming standard (.safetensors) SAE format for {repo_id}")
        repo_url = f"https://huggingface.co/{repo_id}/resolve/main/"
        weights_url = repo_url + weight_filename_sf
        cfg_url = repo_url + cfg_filename_sf

        local_weights_path = os.path.join(cache_dir, f"{repo_id.replace('/', '_')}_{weight_filename_sf}")
        local_cfg_path = os.path.join(cache_dir, f"{repo_id.replace('/', '_')}_{cfg_filename_sf}")

        download_file(weights_url, local_weights_path)
        download_file(cfg_url, local_cfg_path)

        try:
            with open(local_cfg_path, 'r') as f:
                cfg = json.load(f)
            d_in = cfg['d_in']
            d_hidden = cfg['d_sae']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error loading or parsing standard config file {local_cfg_path}: {e}")
            raise

        sae = SparseAutoEncoder(
            d_in=d_in,
            d_hidden=d_hidden, 
            device=model_device, 
            dtype=model_dtype
        )

        try:
            state_dict = load_file(local_weights_path, device=str(model_device))
            adapted_state_dict = {}
            for key, value in state_dict.items():
                 new_key = key
                 if key == "encoder.weight": new_key = "encoder_linear.weight"
                 if key == "encoder.bias": new_key = "encoder_linear.bias"
                 if key == "decoder.weight": new_key = "decoder_linear.weight"
                 if key == "decoder.bias": new_key = "decoder_linear.bias"
                 adapted_state_dict[new_key] = value
            
            if 'decoder_linear.bias' not in sae.state_dict() and 'decoder_linear.bias' in adapted_state_dict:
                del adapted_state_dict['decoder_linear.bias']
                print("Note: Removed decoder bias from loaded state_dict")
                 
            sae.load_state_dict(adapted_state_dict)
            print(f"Standard SAE loaded successfully")

        except Exception as e:
            print(f"Error loading state dict from {local_weights_path}: {e}")
            raise
            
    return sae