import torch


def test_swap_hook_runs():
    from lens.models.orig import OrigWrapper

    m = OrigWrapper("sshleifer/tiny-gpt2", load_in_8bit=False)
    _ = m.forward_with_replacement(
        input_ids=m.model.reshape_inputs(torch.tensor([[m.model.config.bos_token_id]]))["input_ids"],  # type: ignore[attr-defined]
        new_activation=m.model.get_output_embeddings().weight[0].unsqueeze(0),  # type: ignore[attr-defined]
        layer_idx=0,
        token_pos=0,
    )
