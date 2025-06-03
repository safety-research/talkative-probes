#!/usr/bin/env python3
"""Debug position embeddings issue."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def check_position_embedding_shapes():
    """Check shapes of position embeddings."""
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    transformer = model.transformer
    
    print("Checking Position Embedding Shapes")
    print("=" * 60)
    
    # Single position
    pos_ids_single = torch.tensor([5])
    pos_emb_single = transformer.wpe(pos_ids_single)
    print(f"Single position ID shape: {pos_ids_single.shape}")
    print(f"Single position embedding shape: {pos_emb_single.shape}")
    
    # Batch of positions
    pos_ids_batch = torch.tensor([[5, 6, 7]])
    pos_emb_batch = transformer.wpe(pos_ids_batch)
    print(f"\nBatch position IDs shape: {pos_ids_batch.shape}")
    print(f"Batch position embeddings shape: {pos_emb_batch.shape}")
    
    # What we have in the code
    current_position = 5
    batch_size = 2
    device = torch.device("cpu")
    
    position_ids = torch.arange(current_position, current_position + 1, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    print(f"\nOur position IDs shape: {position_ids.shape}")
    print(f"Position IDs: {position_ids}")
    
    # Try different approaches
    print("\nTrying different approaches:")
    
    # Approach 1: squeeze before wpe
    pos_emb1 = transformer.wpe(position_ids.squeeze(0))
    print(f"1. squeeze(0) -> wpe: {pos_emb1.shape}")
    
    # Approach 2: direct indexing
    pos_emb2 = transformer.wpe(position_ids[0])
    print(f"2. [0] -> wpe: {pos_emb2.shape}")
    
    # Approach 3: flatten
    pos_emb3 = transformer.wpe(position_ids.flatten())
    print(f"3. flatten -> wpe: {pos_emb3.shape}")
    
    # Test with hidden states
    hidden_states = torch.randn(batch_size, 1, 768)
    print(f"\nHidden states shape: {hidden_states.shape}")
    
    # Try adding
    try:
        result1 = hidden_states + pos_emb1.unsqueeze(0)
        print(f"hidden + pos_emb1.unsqueeze(0): {result1.shape}")
    except:
        print("hidden + pos_emb1.unsqueeze(0): FAILED")
    
    try:
        result2 = hidden_states + pos_emb2.unsqueeze(0).unsqueeze(0)
        print(f"hidden + pos_emb2.unsqueeze(0).unsqueeze(0): {result2.shape}")
    except:
        print("hidden + pos_emb2.unsqueeze(0).unsqueeze(0): FAILED")


if __name__ == "__main__":
    check_position_embedding_shapes()