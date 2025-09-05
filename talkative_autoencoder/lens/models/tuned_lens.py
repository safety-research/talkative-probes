import logging
from pathlib import Path

#    from lens.models.tuned_lens import load_tuned_lens_translator_params
# Place this in 01_train_distributed.py or a suitable utility file
# from tuned_lens.nn import TunedLens # Will be imported dynamically
from typing import Dict, Optional, Tuple

import torch

# def load_tuned_lens_translator_params(
#     base_model_name: str,
#     lens_layer_idx: int,
#     checkpoint_path: str,
#     log: logging.Logger,
#     is_main_process: bool
# ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
#     """
#     Loads a TunedLens and extracts its translator's weight and bias.
#     The translator is expected to be an nn.Linear layer.
#     """
#     try:
#         from tuned_lens.nn import TunedLens # Dynamic import
#     except ImportError:
#         if is_main_process:
#             log.error("The 'tuned-lens' library is not installed. "
#                       "Please install it to use the 'tuned_lens_composition' initialization method.")
#         # Propagate the error to stop if this feature is critical
#         raise

#     if is_main_process:
#         log.info(f"Loading TunedLens translator: model='{base_model_name}', layer={lens_layer_idx}, ckpt='{checkpoint_path}'")

#     try:
#         # Ensure checkpoint_path is a string for torch.load
#         checkpoint_path_str = str(resolve_path(checkpoint_path)) # Use your existing resolve_path
#         if not Path(checkpoint_path_str).exists():
#             if is_main_process:
#                 log.error(f"TunedLens checkpoint not found: {checkpoint_path_str}")
#             return None, None

#         # Instantiate TunedLens. Note: Its constructor might need specific arguments
#         # or a config. We assume `lens_layer_name_or_idx` refers to the layer whose
#         # hidden state is the input to this specific lens's translator.
#         lens = TunedLens(model_name_or_path=base_model_name, lens_layer_name_or_idx=lens_layer_idx)

#         state_dict = torch.load(checkpoint_path_str, map_location='cpu')
#         lens.load_state_dict(state_dict)
#         lens.eval() # Set to evaluation mode

#         if hasattr(lens, 'translator') and isinstance(lens.translator, torch.nn.Linear):
#             # The translator's weights (W_k) and bias (b_k)
#             # W_k maps from hidden_dim of layer k to hidden_dim of final layer approx.
#             # Typically, these dimensions are the same (original model's hidden_size).
#             return lens.translator.weight.data.cpu().clone(), lens.translator.bias.data.cpu().clone()
#         else:
#             if is_main_process:
#                 log.error(f"TunedLens (layer {lens_layer_idx}, ckpt {checkpoint_path_str}) "
#                           "does not have a 'translator' nn.Linear attribute or it's not of the expected type.")
#             return None, None
#     except Exception as e:
#         if is_main_process:
#             log.error(f"Error loading or processing TunedLens (layer {lens_layer_idx}, ckpt {checkpoint_path}): {e}", exc_info=True)
#         return None, None


# def initialize_consistency_lens_projection(
#     model_component: torch.nn.Module,    # The Decoder or Encoder instance
#     component_config: DictConfig,        # Config for this specific component (e.g., cfg.trainable_components.decoder)
#     component_name: str,                 # "Decoder" or "Encoder" (for logging)
#     main_run_config: DictConfig,         # The main, fully resolved OmegaConf config for the run
#     log: logging.Logger,
#     is_main_process: bool
# ):
#     """Initializes the projection layer of a consistency-lens component."""
#     init_method = component_config.get('projection_init_method', 'default')

#     if init_method == 'tuned_lens_composition':
#         if is_main_process:
#             log.info(f"Attempting 'tuned_lens_composition' initialization for {component_name}.proj")

#         # --- Get configuration values ---
#         src_layer_idx = component_config.get('projection_init_tl_comp_src_layer')
#         tgt_layer_idx = component_config.get('projection_init_tl_comp_tgt_layer')

#         # Default to main model_name if specific TL model_name isn't provided
#         tl_model_name = component_config.get('projection_init_tl_model_name')
#         if not tl_model_name:
#             tl_model_name = main_run_config.get('model_name')
#             if is_main_process and component_config.get('projection_init_tl_model_name') is None: # only log if it was truly absent
#                 log.info(f"Using main model_name '{tl_model_name}' for TunedLens initialization of {component_name}.")

#         src_ckpt_path = component_config.get('projection_init_tl_comp_src_checkpoint')
#         tgt_ckpt_path = component_config.get('projection_init_tl_comp_tgt_checkpoint')
#         use_composed_bias = component_config.get('projection_init_tl_comp_use_composed_bias', False)

#         # --- Validate required config ---
#         if not all(isinstance(val, expected_type) for val, expected_type in
#                    [(src_layer_idx, int), (tgt_layer_idx, int), (tl_model_name, str),
#                     (src_ckpt_path, str), (tgt_ckpt_path, str)]):
#             if is_main_process:
#                 log.error(f"Missing or invalid type for one or more required parameters for 'tuned_lens_composition' "
#                           f"for {component_name}. Required: src_layer (int), tgt_layer (int), model_name (str), "
#                           f"src_checkpoint (str), tgt_checkpoint (str). Skipping initialization.")
#             return

#         # --- Load TunedLens translator parameters ---
#         W_src, b_src = load_tuned_lens_translator_params(tl_model_name, src_layer_idx, src_ckpt_path, log, is_main_process)
#         W_tgt, b_tgt = load_tuned_lens_translator_params(tl_model_name, tgt_layer_idx, tgt_ckpt_path, log, is_main_process)

#         if W_src is None or b_src is None or W_tgt is None or b_tgt is None:
#             if is_main_process:
#                 log.error(f"Failed to load all TunedLens parameters for {component_name}. Skipping composition initialization.")
#             return

#         # --- Perform the composition: h_tgt = pinv(W_tgt) * (W_src * h_src + b_src - b_tgt) ---
#         try:
#             # Ensure tensors are float for pinv and matmul if they aren't already
#             W_src_f, b_src_f = W_src.float(), b_src.float()
#             W_tgt_f, b_tgt_f = W_tgt.float(), b_tgt.float()

#             W_tgt_pinv = torch.linalg.pinv(W_tgt_f)

#             # Composed weight: W_P = pinv(W_tgt) * W_src
#             W_P = W_tgt_pinv @ W_src_f
#             # Composed bias: b_P = pinv(W_tgt) * (b_src - b_tgt)
#             b_P = W_tgt_pinv @ (b_src_f - b_tgt_f)

#         except Exception as e:
#             if is_main_process:
#                 log.error(f"Error during pseudo-inverse calculation or matrix multiplication for {component_name}: {e}", exc_info=True)
#             return

#         # --- Apply to the model_component's projection layer ---
#         # This assumes model_component has a single projection layer named 'proj'.
#         # If using per_layer_projections (e.g. in Decoder), this logic needs to be extended.
#         if not hasattr(model_component, 'proj') or not isinstance(model_component.proj, torch.nn.Linear):
#             if is_main_process:
#                 log.warning(f"{component_name} does not have a 'proj' attribute of type nn.Linear. Skipping initialization.")
#             return

#         proj_layer: torch.nn.Linear = model_component.proj

#         # Dimensionality Check:
#         # W_P maps from original model's hidden_dim (input to TL_src) to original model's hidden_dim (output of inv(TL_tgt)).
#         # So, W_P is expected to be [d_orig_hidden, d_orig_hidden].
#         # proj_layer.weight is [cl_output_dim, cl_input_dim].
#         # For direct copy: cl_output_dim == d_orig_hidden AND cl_input_dim == d_orig_hidden.

#         d_orig_hidden = W_P.shape[0] # Assuming square W_P from [d_orig_hidden, d_orig_hidden]

#         cl_output_dim = proj_layer.weight.shape[0]
#         cl_input_dim = proj_layer.weight.shape[1]

#         if cl_output_dim != d_orig_hidden or cl_input_dim != d_orig_hidden:
#             if is_main_process:
#                 log.error(
#                     f"Dimension mismatch for {component_name}.proj initialization. "
#                     f"Composed W_P has shape {W_P.shape} (implying original model hidden_dim={d_orig_hidden}). "
#                     f"{component_name}.proj.weight has shape {proj_layer.weight.shape} (output_dim={cl_output_dim}, input_dim={cl_input_dim}). "
#                     f"Both input and output dimensions of {component_name}.proj must match original model's hidden dimension ({d_orig_hidden}) "
#                     f"for this initialization. Skipping."
#                 )
#             return

#         # --- Copy weights and bias ---
#         try:
#             with torch.no_grad():
#                 proj_layer.weight.data.copy_(W_P.to(dtype=proj_layer.weight.dtype))
#                 if use_composed_bias:
#                     proj_layer.bias.data.copy_(b_P.to(dtype=proj_layer.bias.dtype))
#                 else:
#                     torch.nn.init.zeros_(proj_layer.bias.data)
#             if is_main_process:
#                 log.info(f"Successfully initialized {component_name}.proj using 'tuned_lens_composition' "
#                          f"(src_layer={src_layer_idx}, tgt_layer={tgt_layer_idx}). "
#                          f"Used composed bias: {use_composed_bias}.")
#         except Exception as e:
#             if is_main_process:
#                 log.error(f"Error copying composed W_P and b_P to {component_name}.proj: {e}", exc_info=True)

#     # elif init_method == 'some_other_method':
#         # pass
#     # else: (default initialization, do nothing here as PyTorch handles it)
#         # if is_main_process and init_method != 'default':
#             # log.warning(f"Unknown projection_init_method: '{init_method}' for {component_name}. Using default PyTorch initialization.")


# # --- In your main() function, after models are created on CPU, before DDP ---
# #
# # Example call structure:
# #
# #   # Convert Hydra config to a plain Python dict for most operations
# #   config_dict = OmegaConf.to_container(cfg, resolve=True)
# #
# #   # Initialize models (decoder, encoder) on CPU first
# #   decoder = Decoder(...)
# #   encoder = Encoder(...)
# #   # ... other model setup ...
# #
# #   if is_main(): # Perform initialization only on the main process
# #       # Initialize Decoder's projection layer
# #       if 'decoder' in config_dict.get('trainable_components', {}):
# #           initialize_consistency_lens_projection(
# #               model_component=decoder,
# #               component_config=config_dict['trainable_components']['decoder'], # Pass the specific dict
# #               component_name="Decoder",
# #               main_run_config=config_dict, # Pass the whole resolved dict
# #               log=log,
# #               is_main_process=True
# #           )
# #       # Initialize Encoder's projection layer (if applicable)
# #       if 'encoder' in config_dict.get('trainable_components', {}):
# #            initialize_consistency_lens_projection(
# #               model_component=encoder,
# #               component_config=config_dict['trainable_components']['encoder'],
# #               component_name="Encoder",
# #               main_run_config=config_dict,
# #               log=log,
# #               is_main_process=True
# #           )
# #
# #   # THEN, move models to device and wrap with DDP
# #   decoder, encoder, orig_model = setup_distributed_models(decoder, encoder, orig_model, device, rank, world_size)
# # ...


def load_full_tuned_lens(
    model_or_model_name: str,
    checkpoint_path_or_name: str,
    device: str = "cpu",
    log: logging.Logger = None,
    is_main_process: bool = True,
) -> Optional["TunedLens"]:
    """
    Loads a complete TunedLens containing translators for all layers.

    Args:
        model_or_model_name: Model name or instance for the base model
        checkpoint_path_or_name: Either:
            - Local path to checkpoint directory
            - HuggingFace model/space identifier (e.g., 'AlignmentResearch/tuned-lens-pythia-160m')
            - If None/empty, will try to load from default HF location based on model name
        device: Device to load the model on
        log: Logger instance
        is_main_process: Whether this is the main process

    Returns:
        TunedLens instance or None if loading fails
    """
    try:
        from tuned_lens import TunedLens

        from transformers import AutoModelForCausalLM
    except ImportError:
        if is_main_process and log:
            log.error(
                "The 'tuned-lens' library is not installed. "
                "Please install it to use the 'tuned_lens_composition' initialization method."
            )
        raise

    try:
        if model_or_model_name == "openai-community/gpt2":
            log.info(f"got {model_or_model_name}, cutting to gpt2")
            model_or_model_name = "gpt2"
        elif model_or_model_name == "openai-community/gpt2-xl":
            log.info(f"got {model_or_model_name}, cutting to gpt2-xl")
            model_or_model_name = "gpt2-xl"
        # Load the base model if we only have a name
        if isinstance(model_or_model_name, str):
            if is_main_process and log:
                log.info(f"Loading base model: {model_or_model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_or_model_name, device_map=device)
        else:
            model = model_or_model_name

        # Determine if this is a local path or HuggingFace identifier
        is_local_path = False
        if checkpoint_path_or_name:
            # Check if it's a local path
            try:
                checkpoint_path = Path(checkpoint_path_or_name)
                is_local_path = checkpoint_path.exists() and checkpoint_path.is_dir()
            except (OSError, ValueError):
                # Invalid path, assume it's a HF identifier
                is_local_path = False

        if not checkpoint_path_or_name:
            # Try default loading (will look for lens in default HF location)
            if is_main_process and log:
                log.info("Loading TunedLens from default HuggingFace location for model")
            lens = TunedLens.from_model_and_pretrained(model)
        elif is_local_path:
            # Local path
            if is_main_process and log:
                log.info(f"Loading TunedLens from local path: {checkpoint_path}")
            lens = TunedLens.from_model_and_pretrained(model, str(checkpoint_path))
        else:
            # HuggingFace identifier
            if is_main_process and log:
                log.info(f"Loading TunedLens from HuggingFace: {checkpoint_path_or_name}")
            # The tuned_lens library expects the pretrained identifier
            lens = TunedLens.from_model_and_pretrained(model, checkpoint_path_or_name)

        lens.eval()
        return lens

    except Exception as e:
        if is_main_process and log:
            log.error(f"Error loading TunedLens: {e}", exc_info=True)
        return None


def extract_translator_params(
    tuned_lens: "TunedLens", layer_idx: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract translator parameters for a specific layer from a TunedLens.
    For the final layer, returns identity transformation.

    Args:
        tuned_lens: Loaded TunedLens instance
        layer_idx: Layer index to extract translator for

    Returns:
        Tuple of (weight, bias) or (None, None) if extraction fails
    """
    try:
        num_layers = len(tuned_lens.layer_translators)

        # Final layer uses identity transformation
        if layer_idx >= num_layers:
            # Return identity matrix and zero bias
            d_model = tuned_lens.config.d_model
            identity = torch.eye(d_model, dtype=torch.float32)
            zero_bias = torch.zeros(d_model, dtype=torch.float32)
            return identity, zero_bias

        # Access the translator for the specific layer
        translator = tuned_lens.layer_translators[layer_idx]

        if hasattr(translator, "weight") and hasattr(translator, "bias"):
            return translator.weight.data.cpu().clone(), translator.bias.data.cpu().clone()
        else:
            return None, None

    except Exception:
        return None, None


def compute_composed_projection(
    W_src: torch.Tensor, b_src: torch.Tensor, W_tgt: torch.Tensor, b_tgt: torch.Tensor, log: logging.Logger = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the composed projection: h_tgt = W_tgt^(-1) * (W_src * h_src + b_src) + W_tgt^(-1) * (-b_tgt)

    This simplifies to:
    - Weight: W_P = W_tgt^(-1) * W_src
    - Bias: b_P = W_tgt^(-1) * (b_src - b_tgt)

    Returns:
        Tuple of (composed_weight, composed_bias)
    """
    # Ensure float precision for computation
    W_src_f, b_src_f = W_src.float(), b_src.float()
    W_tgt_f, b_tgt_f = W_tgt.float(), b_tgt.float()

    # Check if target is identity (final layer case)
    is_tgt_identity = torch.allclose(W_tgt_f, torch.eye(W_tgt_f.shape[0], device=W_tgt_f.device), atol=1e-6)

    if is_tgt_identity:
        W_P = W_src_f
        b_P = b_src_f - b_tgt_f
    else:
        # Log condition number and rank of W_tgt
        cond_num = torch.linalg.cond(W_tgt_f).item()
        rank = torch.linalg.matrix_rank(W_tgt_f).item()
        log.info(f"[TunedLens] W_tgt condition number: {cond_num:.2e}, rank: {rank}")
        # Compute pseudo-inverse
        W_tgt_pinv = torch.linalg.pinv(W_tgt_f)
        # log.warning(f"Using transpose instead of inverse")
        log.warning("Using pseudoinverse  - possibly unstable?")
        W_tgt_pinv = W_tgt_f.transpose(0, 1)
        W_P = W_tgt_pinv @ W_src_f
        b_P = W_tgt_pinv @ (b_src_f - b_tgt_f)

    return W_P, b_P


def initialize_consistency_lens_projection(
    model_component: torch.nn.Module,
    component_config: Dict,
    component_name: str,
    main_run_config: Dict,
    log: logging.Logger,
    is_main_process: bool,
    resolve_path_fn=None,
    tokenizer=None,
    device=None,
    orig_model=None,
    batch_tokens=None,
):
    """Initializes the projection layer(s) of a consistency-lens component.

    Configuration example for loading from HuggingFace:
    ```yaml
    trainable_components:
      decoder:
        projection_init_method: tuned_lens_composition
        # SIMPLEST: Omit checkpoint to auto-load from HuggingFace based on model
        # projection_init_tl_checkpoint: null

        # OR specify explicit HF identifier:
        # projection_init_tl_checkpoint: "AlignmentResearch/tuned-lens-pythia-160m"

        # OR for local path:
        # projection_init_tl_checkpoint: "/path/to/local/checkpoint"

        projection_init_tl_model_name: null  # Optional, defaults to main model_name from config
        projection_init_tl_comp_tgt_layer: 0  # For single projection mode
        projection_init_tl_comp_use_composed_bias: false
    ```

    The TunedLens will be loaded from:
    1. Default HF location for the model if checkpoint path is not specified (RECOMMENDED)
    2. Specified HuggingFace identifier if provided
    3. Local directory if the path exists
    """
    init_method = component_config.get("projection_init_method", "default")

    if init_method == "default":
        return
    if hasattr(model_component.config, "use_projection_layer") and not model_component.config.use_projection_layer:
        print("Skipping projection initialization as not using projection layer")
        return

    if is_main_process:
        log.info(f"Attempting '{init_method}' initialization for {component_name}")

    if init_method == "tuned_lens_composition":
        tl_checkpoint_path = component_config.get("projection_init_tl_checkpoint")

        # Get model name - use from config or default to main model
        tl_model_name = component_config.get("projection_init_tl_model_name")
        if not tl_model_name:
            tl_model_name = main_run_config.get("model_name")
            if is_main_process:
                log.info(f"Using main model_name '{tl_model_name}' for TunedLens")

        # Apply path resolution if needed
        if tl_checkpoint_path and resolve_path_fn:
            tl_checkpoint_path = str(resolve_path_fn(tl_checkpoint_path))

        # Load TunedLens - checkpoint_path can be None for auto-discovery
        if not tl_checkpoint_path:
            if is_main_process:
                log.info(
                    f"No checkpoint specified - will auto-load pre-trained TunedLens for '{tl_model_name}' from HuggingFace"
                )

        tuned_lens = load_full_tuned_lens(
            tl_model_name,
            tl_checkpoint_path,  # Can be None
            device="cpu",
            log=log,
            is_main_process=is_main_process,
        )

        if tuned_lens is None:
            raise RuntimeError(f"Failed to load TunedLens for {component_name}. Halting initialization.")

        # Get the number of layers in the base model
        num_model_layers = tuned_lens.config.num_hidden_layers
        if is_main_process:
            log.info(
                f"TunedLens has {len(tuned_lens.layer_translators)} translators for {num_model_layers} layer model"
            )

        # Get the source layer from config - this is where activations are extracted
        src_layer_idx = main_run_config.get("layer_l")
        if src_layer_idx is None:
            raise RuntimeError(
                "'layer_l' not found in config. Cannot determine source layer for tuned lens initialization."
            )

        # Validate source layer
        if src_layer_idx >= num_model_layers:
            raise RuntimeError(
                f"Source layer {src_layer_idx} is out of bounds for model with {num_model_layers} layers"
            )

        is_per_layer = (
            getattr(model_component.config, "per_layer_projections", False)
            if hasattr(model_component, "config")
            else False
        )

        if not is_per_layer:
            # Single projection layer mode
            if component_name == "Encoder":
                tgt_layer_idx = component_config.get("output_layer")
                if tgt_layer_idx is None:
                    raise RuntimeError(f"Encoder config for {component_name} must have 'output_layer' set.")
            else:
                tgt_layer_idx = component_config.get("projection_init_tl_comp_tgt_layer", 0)

            if is_main_process:
                log.info(
                    f"Single projection mode: mapping from extraction layer {src_layer_idx} to target layer {tgt_layer_idx}"
                )

            # Validate target layer
            if tgt_layer_idx >= num_model_layers:
                raise RuntimeError(
                    f"Target layer {tgt_layer_idx} is out of bounds for model with {num_model_layers} layers"
                )

            W_src, b_src = extract_translator_params(tuned_lens, src_layer_idx)
            W_tgt, b_tgt = extract_translator_params(tuned_lens, tgt_layer_idx)

            if any(x is None for x in [W_src, b_src, W_tgt, b_tgt]):
                raise RuntimeError(
                    f"Failed to extract translator parameters for layers {src_layer_idx} and {tgt_layer_idx}"
                )

            try:
                W_P, b_P = compute_composed_projection(W_src, b_src, W_tgt, b_tgt, log=log)
            except Exception as e:
                raise RuntimeError(f"Error computing composed projection: {e}")

            if not hasattr(model_component, "proj") or model_component.proj is None:
                raise RuntimeError(f"{component_name} does not have 'proj' attribute")

            proj_layer = model_component.proj
            with torch.no_grad():
                proj_layer.weight.data.copy_(W_P.to(dtype=proj_layer.weight.dtype))
                if component_config.get("projection_init_tl_comp_use_composed_bias", False):
                    proj_layer.bias.data.copy_(b_P.to(dtype=proj_layer.bias.dtype))
                else:
                    torch.nn.init.zeros_(proj_layer.bias.data)

            if is_main_process:
                log.info(
                    f"Initialized {component_name}.proj using tuned_lens_composition "
                    f"(src_layer={src_layer_idx}, tgt_layer={tgt_layer_idx})"
                )

        else:
            # Per-layer projections mode (Decoder only)
            if not hasattr(model_component, "proj_weight") or not hasattr(model_component, "proj_bias"):
                raise RuntimeError(
                    f"{component_name} with per_layer_projections=True does not have 'proj_weight' and 'proj_bias' attributes"
                )

            n_layers = model_component.proj_weight.shape[0]

            use_composed_bias = component_config.get("projection_init_tl_comp_use_composed_bias", True)
            initialized_count = 0

            if is_main_process:
                log.info(
                    f"Per-layer projection mode: mapping from extraction layer {src_layer_idx} to each decoder layer"
                )

            # Initialize each decoder layer's projection
            for decoder_layer_idx in range(n_layers):
                # Source is always the extraction layer (layer_l)
                # Target is the current decoder layer where we're patching
                if component_config["new_decoder"]:
                    if decoder_layer_idx == 0:
                        tgt_layer_idx = 0
                    else:
                        tgt_layer_idx = decoder_layer_idx - 1
                else:
                    tgt_layer_idx = decoder_layer_idx

                # Allow override for specific layers if needed
                tgt_override = component_config.get(f"projection_init_tl_comp_tgt_layer_{decoder_layer_idx}")
                if tgt_override is not None:
                    tgt_layer_idx = tgt_override

                # Validate target layer
                if tgt_layer_idx >= num_model_layers:
                    if is_main_process:
                        log.warning(
                            f"Target layer {tgt_layer_idx} is out of bounds for model with {num_model_layers} layers. Skipping decoder layer {decoder_layer_idx}"
                        )
                    continue

                # Extract translator parameters
                W_src, b_src = extract_translator_params(tuned_lens, src_layer_idx)
                W_tgt, b_tgt = extract_translator_params(tuned_lens, tgt_layer_idx)

                if any(x is None for x in [W_src, b_src, W_tgt, b_tgt]):
                    if is_main_process:
                        log.warning(
                            f"Failed to extract translator parameters for decoder layer {decoder_layer_idx} "
                            f"(src={src_layer_idx}, tgt={tgt_layer_idx})"
                        )
                    continue

                # Compute composed projection
                try:
                    W_P, b_P = compute_composed_projection(W_src, b_src, W_tgt, b_tgt, log=log)
                except Exception as e:
                    if is_main_process:
                        log.warning(f"Error computing composed projection for decoder layer {decoder_layer_idx}: {e}")
                    continue

                # Apply to this layer's projection weights
                with torch.no_grad():
                    model_component.proj_weight[decoder_layer_idx].data.copy_(
                        W_P.to(dtype=model_component.proj_weight.dtype)
                    )
                    if use_composed_bias:
                        model_component.proj_bias[decoder_layer_idx].data.copy_(
                            b_P.to(dtype=model_component.proj_bias.dtype)
                        )
                    else:
                        torch.nn.init.zeros_(model_component.proj_bias[decoder_layer_idx].data)

                initialized_count += 1

            if is_main_process:
                log.info(
                    f"Initialized {initialized_count}/{n_layers} per-layer projections for {component_name} "
                    f"using tuned_lens_composition (all from source layer {src_layer_idx})"
                )

    elif init_method == "scale":
        # if not all([tokenizer, device, orig_model, batch_tokens]):
        #     raise RuntimeError(
        #         f"To use 'scale' initialization for {component_name}, tokenizer, device, orig_model, and batch_tokens must be provided."
        #     )

        with torch.no_grad():
            # 1. Get hidden states from the provided batch
            inputs = batch_tokens.to(device)
            if main_run_config["dataset"]["on_the_fly"]["enabled"]:
                bscheck = main_run_config["dataset"]["on_the_fly"]["generation_batch_size"]
            else:
                bscheck = main_run_config["activation_dumper"]["batch_size"]
            bscheck = min(bscheck, 48)
            attention_mask = inputs[:bscheck] != tokenizer.pad_token_id
            outputs = orig_model.model(
                input_ids=inputs[:bscheck], attention_mask=attention_mask, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

            # 2. Calculate average token norms for each layer, ignoring pad tokens
            avg_norms = []
            for i, h in enumerate(hidden_states):
                # h: (batch, seq, d_model)
                if h.shape[1] > 5:
                    # Only consider tokens after the first 5
                    mask = attention_mask[:, 5:]
                    h_tokens = h[:, 5:, :]
                else:
                    mask = attention_mask
                    h_tokens = h

                # mask: (batch, seq), h_tokens: (batch, seq, d_model)
                token_norms = torch.linalg.norm(h_tokens.float(), dim=-1)  # (batch, seq)
                valid_mask = mask.bool()
                valid_token_norms = token_norms[valid_mask]
                if valid_token_norms.numel() > 0:
                    avg_norm = valid_token_norms.mean().item()
                else:
                    avg_norm = 0.0
                avg_norms.append(avg_norm)
                if is_main_process:
                    log.info(f"Layer {i} avg token norm (skipped 5, pad-masked): {avg_norm:.4f}")

            # 3. Apply scaling
            is_per_layer = (
                getattr(model_component.config, "per_layer_projections", False)
                if hasattr(model_component, "config")
                else False
            )

            if not is_per_layer:
                # Single projection layer (Encoder or Decoder)
                if component_name == "Encoder":
                    src_layer_idx = component_config.get("output_layer")
                    tgt_layer_idx = main_run_config.get("layer_l")
                else:  # Decoder
                    src_layer_idx = main_run_config.get("layer_l")
                    tgt_layer_idx = component_config.get("projection_init_tl_comp_tgt_layer", 0)

                if src_layer_idx is None or tgt_layer_idx is None:
                    raise RuntimeError("Source or target layer for scaling is not defined.")

                scale_factor = avg_norms[tgt_layer_idx] / (avg_norms[src_layer_idx] + 1e-9)
                log.info(f"Scale factor: {scale_factor:.4f} for component {component_name}")

                proj_layer = model_component.proj
                d_model = proj_layer.weight.shape[1]
                identity = torch.eye(d_model, device=device, dtype=proj_layer.weight.dtype)

                proj_layer.weight.data.copy_(identity * scale_factor)
                torch.nn.init.zeros_(proj_layer.bias.data)

                if is_main_process:
                    log.info(
                        f"Initialized single projection for {component_name} with scale factor {scale_factor:.4f} (src: {src_layer_idx}, tgt: {tgt_layer_idx})"
                    )

            else:
                # Per-layer projections (Decoder only)
                src_layer_idx = main_run_config.get("layer_l")
                if src_layer_idx is None:
                    raise RuntimeError("'layer_l' not found in config for per-layer scaling.")

                n_layers = model_component.proj_weight.shape[0]
                d_model = model_component.proj_weight.shape[2]
                identity = torch.eye(d_model, device=device)

                for decoder_layer_idx in range(n_layers):
                    tgt_layer_idx = decoder_layer_idx
                    if component_config["new_decoder"]:
                        if decoder_layer_idx == 0:
                            log.info("Should use norms of embedding vectors here, probably - fix later")
                            tgt_layer_idx = 0
                        else:
                            tgt_layer_idx = decoder_layer_idx - 1
                    else:
                        tgt_layer_idx = decoder_layer_idx

                    scale_factor = avg_norms[tgt_layer_idx] / (avg_norms[src_layer_idx] + 1e-9)

                    scaled_identity = identity * scale_factor
                    model_component.proj_weight[decoder_layer_idx].data.copy_(
                        scaled_identity.to(dtype=model_component.proj_weight.dtype)
                    )
                    torch.nn.init.zeros_(model_component.proj_bias[decoder_layer_idx].data)
                    log.info(
                        f"{component_name}, layer {decoder_layer_idx}, scale factor: {scale_factor:.4f}, src: {src_layer_idx}, tgt: {tgt_layer_idx}"
                    )

                if is_main_process:
                    log.info(
                        f"Initialized {n_layers} per-layer projections for {component_name} with norm scaling from source layer {src_layer_idx}."
                    )
