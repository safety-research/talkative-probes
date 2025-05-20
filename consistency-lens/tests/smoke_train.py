import torch


def test_smoke_train():
    from lens.models.decoder import Decoder, DecoderConfig
    from lens.models.encoder import Encoder
    from lens.training.loop import train_step

    dec = Decoder(DecoderConfig(model_name="sshleifer/tiny-gpt2"))
    enc = Encoder("sshleifer/tiny-gpt2")

    batch = {
        "A": dec.proj.weight.new_zeros(1, dec.base.config.hidden_size),
        "A_prime": dec.proj.weight.new_zeros(1, dec.base.config.hidden_size),
        "input_ids_A": dec.proj.weight.new_zeros(1, 1, dtype=torch.long),
    }

    loss = train_step(batch, {"dec": dec, "enc": enc}, {})
    assert loss.requires_grad is True
