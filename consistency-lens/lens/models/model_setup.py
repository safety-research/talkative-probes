import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.embedding_remap import remap_embeddings

def initialize_raw_models_and_tokenizer(
    config: dict, 
    log: logging.Logger
) -> tuple[Decoder, Encoder, OrigWrapper, AutoTokenizer, AutoTokenizer]:
    """
    Initializes raw Decoder, Encoder, OrigWrapper models, and tokenizers.
    Handles embedding remapping, prompt setup, and decoder output head resizing.
    Models are NOT moved to device or compiled here.
    """
    model_name = config['model_name']
    tokenizer_name = config.get("tokenizer_name", model_name)
    
    decoder_train_cfg = config.get('trainable_components', {}).get('decoder', {})
    encoder_train_cfg = config.get('trainable_components', {}).get('encoder', {})

    # Initialize models
    decoder_config = DecoderConfig(model_name=model_name, **decoder_train_cfg)
    dec_raw = Decoder(decoder_config)
    log.info(f"Initialized Decoder model with config: {decoder_config}")

    encoder_config = EncoderConfig(model_name=model_name, **encoder_train_cfg)
    enc_raw = Encoder(encoder_config)
    log.info(f"Initialized Encoder model with config: {encoder_config}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    base_tok = AutoTokenizer.from_pretrained(model_name)
    log.info(f"Loaded tokenizer: {tokenizer_name}, Base tokenizer for {model_name}")

    new_vocab_size = tokenizer.vocab_size

    if tokenizer_name != model_name:
        log.info(f"Remapping embeddings from {model_name} to {tokenizer_name}")
        # Ensure base models exist before remapping
        if hasattr(dec_raw, 'base') and dec_raw.base is not None:
            remap_embeddings(dec_raw.base, base_tok, tokenizer)
        else:
            log.warning("Decoder raw does not have a 'base' model for embedding remapping.")

        if hasattr(enc_raw, 'base') and enc_raw.base is not None and enc_raw.config.use_base_model:
            remap_embeddings(enc_raw.base, base_tok, tokenizer)
        elif enc_raw.config.use_base_model:
            log.warning("Encoder raw does not have a 'base' model for embedding remapping, though use_base_model is true.")
        log.info("Attempted remapping for Decoder & Encoder embedding matrices to new tokenizer.")

    # Decoder prompt
    dec_raw.set_prompt(config["decoder_prompt"], tokenizer)
    log.info("Prompt for decoder: %s", str([tokenizer.decode(d) for d in dec_raw.prompt_ids]))

    # Encoder soft prompt
    encoder_soft_prompt_text = encoder_train_cfg.get('soft_prompt_init_text')
    if encoder_soft_prompt_text:
        enc_raw.set_soft_prompt_from_text(encoder_soft_prompt_text, tokenizer)
        log.info("Initialized encoder soft prompt from text: %s", encoder_soft_prompt_text)

    # Decoder output head resizing
    if hasattr(dec_raw, 'out') and dec_raw.out is not None and hasattr(dec_raw, 'base') and dec_raw.base is not None:
        if dec_raw.out.weight.size(0) != new_vocab_size:
            d_model = dec_raw.base.config.hidden_size
            dec_raw.out = nn.Linear(d_model, new_vocab_size, bias=False)
            with torch.no_grad():
                # Ensure base model has output embeddings
                if hasattr(dec_raw.base, 'get_output_embeddings') and dec_raw.base.get_output_embeddings() is not None:
                     dec_raw.out.weight.copy_(dec_raw.base.get_output_embeddings().weight)
                     log.info(f"Resized Decoder.out to new vocab size: {new_vocab_size}")
                else:
                    log.warning("Decoder base model does not have get_output_embeddings method or it returned None. Cannot copy weights for resized Decoder.out.")
    elif not (hasattr(dec_raw, 'base') and dec_raw.base is not None):
        log.warning("Decoder.out resizing skipped: Decoder.base is not initialized.")
    else:
        log.info("Decoder.out already matches new vocab size or not applicable.")


    # Original model wrapper (remap after creation)
    orig = OrigWrapper(model_name, load_in_8bit=config.get('load_orig_in_8bit', False)) # Added 8bit option from config
    if tokenizer_name != model_name:
        if hasattr(orig, 'model') and orig.model is not None:
            remap_embeddings(orig.model, base_tok, tokenizer)
            log.info("Remapped Orig model embeddings to new tokenizer.")
        else:
            log.warning("Orig model is not initialized. Skipping embedding remapping for Orig.")
            
    return dec_raw, enc_raw, orig, tokenizer, base_tok
