from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = ["remap_embeddings"]

def remap_embeddings(model: PreTrainedModel, old_tok: PreTrainedTokenizerBase, new_tok: PreTrainedTokenizerBase) -> None:
    """Align *model*'s input & output embedding rows to *new_tok*'s vocabulary.

    The algorithm copies rows whose token strings exist in the *old_tok* vocab.
    Rows for brand-new tokens are randomly initialised (std=0.02).
    The function operates in-place **on CPU tensors only**; call this *before*
    ``model.to(device)`` to minimise peak GPU memory.
    """
    with torch.no_grad():
        in_emb = model.get_input_embeddings()
        old_weight = in_emb.weight.clone()  # CPU tensor
        d_model = old_weight.size(1)

        new_vocab = new_tok.vocab_size
        # Resize to the new size first; this allocates the target matrix.
        model.resize_token_embeddings(new_vocab)
        in_emb = model.get_input_embeddings()

        # Build copy of old vocab lookup once for speed.
        old_vocab = old_tok.get_vocab()

        for new_id in range(new_vocab):
            token_str = new_tok.convert_ids_to_tokens(new_id)
            old_id = old_vocab.get(token_str, None)
            if old_id is not None and old_id < old_weight.size(0):
                in_emb.weight[new_id].copy_(old_weight[old_id])
            else:
                torch.nn.init.normal_(in_emb.weight[new_id], std=0.02)

        # If output embeddings exist and are separate, remap/tie them as well.
        try:
            out_emb = model.get_output_embeddings()
        except AttributeError:
            out_emb = None
        if out_emb is not None:
            if out_emb.weight.size(0) != new_vocab:
                # create new linear head and tie to input
                new_out = torch.nn.Linear(d_model, new_vocab, bias=False)
                new_out.weight.data.copy_(in_emb.weight.data)
                model.set_output_embeddings(new_out)
            else:
                out_emb.weight.data.copy_(in_emb.weight.data) 