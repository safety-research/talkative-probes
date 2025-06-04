# Talkative Probes - 3 Minute Talk Notes

## Opening (30 seconds)
- **Hook**: "What if we could have a conversation with the hidden layers of neural networks?"
- **Problem**: Neural network activations are high-dimensional vectors - hard to interpret
- **Solution**: Talkative Probes - a bidirectional interface between activations and natural language

## Core Concept (1 minute)
1. **Extract**: Take activation `A` from layer L of a frozen LLM
2. **Decode**: Train decoder `D` to convert `A` → natural language explanation
3. **Encode**: Train encoder `E` to convert explanation → reconstructed activation `A'`
4. **Key insight**: The reconstruction must preserve the original activation's function

## Technical Details (1 minute)
- **Training data**: 10B tokens, random positions & layers
- **Architecture**: Both D and E are LLM clones with trainable soft prompts
- **Loss function**: KL divergence ensures functional preservation
- **Innovation**: Learns general activation↔language mappings, not position-specific

## Applications & Impact (30 seconds)
- **Interpretability**: Understand what activations represent
- **Intervention**: Edit model behavior through natural language
- **Analysis**: Compare activations across models/layers
- **Future**: Foundation for more interpretable AI systems

## Key Takeaway
"Talkative Probes create a natural language API for neural network internals - making the black box conversational." 