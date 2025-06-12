# %% [markdown]
# # Consistency Lens Interactive Analysis
# 
# This notebook provides an interactive interface for analyzing text through a trained consistency lens.
# Run cells individually to explore how the lens interprets different inputs.


# need to source the proper uv env /home/kitf/.cache/uv/envs/consistency-lens/bin/python
# need to uv add jupyter seaborn
# and install jupyter if necessary

# %%
# Imports and setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
# Add parent directory to path for imports
toadd = str(Path(__file__).parent.parent)
print("Adding to path:",toadd)
sys.path.append(toadd)
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.embedding_remap import remap_embeddings

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Imports complete")

# %% [markdown]
# ## Load the Checkpoint
# 
# Update the path below to point to your trained checkpoint:

# %%
# Configuration - UPDATE THIS PATH
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_frozen_lr1e-4_t32_5ep_resume_0604_1343_zeroLMKLkvNO_RA_NEW_bestWID_dist4/gpt2_frozen_step10000_epoch2_.pt"#outputs/checkpoints/YOUR_CHECKPOINT.pt"  # <-- Change this!
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_frozen_lr1e-3_t16_5ep_resume_0604_1516_zeroLMKLkvNO_RA_NEW_bestWIDsmallLM_dist4/gpt2_frozen_step10000_epoch3_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_frozen_lr1e-3_t16_5ep_resume_0604_1516_zeroLMKLkvNO_RA_NEW_bestWIDsmallLM_dist4/gpt2_frozen_step19564_epoch6_final.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr1e-3_t32_20ep_resume_0609_1410_tinyLMKVext_fl32_dist4_slurm634/gpt2_frozen_step124000_epoch16_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_postfix_warmup_OC_GPT2_L6_e5_prog-unfreeze_lr1e-4_t16_30ep_resume_0609_1456_tinyLMKV_ufwarmup_resume_dist4/gpt2_frozen_step27000_epoch7_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_postfix_warmup_OC_GPT2_L6_e5_prog-unfreeze_lr1e-3_t16_30ep_resume_0609_1655_tinyLMKV_ufwarmup_resumeMSE_dist4/gpt2_frozen_step16000_epoch5_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr3e-4_t32_20ep_resume_0609_2327_tinyLMKVext_dist4/gpt2_frozen_step49000_epoch7_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr3e-4_t32_20ep_resume_0609_2327_tinyLMKVext_dist4/gpt2_frozen_step116000_epoch15_.pt"
CHECKPOINT_PATH = "/workspace/kitf/talkative-probes/consistency-lens/outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr3e-4_t32_4ep_resume_0611_1502_tinyLMKVext_resumeold2highlr_dist4/"
# Optional: specify device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# %%
# Load checkpoint and initialize models
class LensAnalyzer:
    """Analyzer for consistency lens."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            pt_files = sorted(checkpoint_path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not pt_files:
                raise FileNotFoundError(f"No .pt files found in directory {checkpoint_path}")
            checkpoint_path = pt_files[0]
            print(f"Selected most recent checkpoint: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path

        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # Load to CPU first

        # Extract config
        if 'config' in ckpt:
            self.config = ckpt['config']
            self.layer = self.config.get('layer_l')
            self.model_name = self.config.get('model_name', 'gpt2')
            tokenizer_name = self.config.get('tokenizer_name', self.model_name)
            self.t_text = self.config.get('t_text')
            self.tau = self.config.get('gumbel_tau_schedule', {}).get('end_value', 1.0)
        else:
            raise ValueError("No config found in checkpoint")
        
        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Build models following the pattern from 01_train_distributed.py
        print(f"Building models ({self.model_name})...")
        
        # Extract trainable components configuration
        trainable_components_config = self.config.get('trainable_components', {})
        decoder_train_cfg = trainable_components_config.get('decoder', {})
        encoder_train_cfg = trainable_components_config.get('encoder', {})
        
        # Initialize models
        decoder_config_obj = DecoderConfig(
            model_name=self.model_name,
            **decoder_train_cfg
        )
        self.decoder = Decoder(decoder_config_obj)
        
        encoder_config_obj = EncoderConfig(
            model_name=self.model_name,
            **encoder_train_cfg
        )
        self.encoder = Encoder(encoder_config_obj)
        
        self.orig_model = OrigWrapper(self.model_name, load_in_8bit=False)
        
        # Initialize Decoder prompt (before loading weights)
        if 'decoder_prompt' in self.config and self.config['decoder_prompt']:
            print(f"Setting decoder prompt: \"{self.config['decoder_prompt']}\"")
            self.decoder.set_prompt(self.config['decoder_prompt'], self.tokenizer)
        
        # Initialize Encoder soft prompt (before loading weights)
        if encoder_config_obj.soft_prompt_init_text:
            print(f"Setting encoder soft prompt from text: \"{encoder_config_obj.soft_prompt_init_text}\"")
            self.encoder.set_soft_prompt_from_text(encoder_config_obj.soft_prompt_init_text, self.tokenizer)
        elif encoder_config_obj.soft_prompt_length > 0:
            print(f"Encoder using randomly initialized soft prompt of length {encoder_config_obj.soft_prompt_length}.")
        
        # Move models to device
        self.decoder.to(self.device)
        self.encoder.to(self.device)
        self.orig_model.to(self.device)
        self.decoder.eval()
        self.encoder.eval()
        self.orig_model.model.eval()
        
        # Load weights using CheckpointManager
        from lens.utils.checkpoint_manager import CheckpointManager
        import logging
        
        # Setup logger
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        # Minimal checkpoint config for CheckpointManager
        checkpoint_config = {
            "checkpoint": {
                "enabled": True,
                "output_dir": str(self.checkpoint_path.parent)
            }
        }
        
        checkpoint_manager = CheckpointManager(checkpoint_config, logger)
        
        # Load model weights
        models_to_load = {"decoder": self.decoder, "encoder": self.encoder}
        
        try:
            loaded_data = checkpoint_manager.load_checkpoint(
                str(self.checkpoint_path),
                models=models_to_load,
                optimizer=None,
                map_location=self.device
            )
            print(f"✓ Loaded checkpoint from step {loaded_data.get('step', 'unknown')}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}") from e
        
        # Set models to eval mode
        self.decoder.eval()
        self.encoder.eval()
        self.orig_model.model.eval()
        
        print(f"✓ Ready! Model: {self.model_name}, Layer: {self.layer}")
    
    def analyze_all_tokens(self, text: str) -> pd.DataFrame:
        """Analyze all tokens in the text and return results as DataFrame."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        # Get all hidden states
        with torch.no_grad():
            outputs = self.orig_model.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        results = []
        torch.manual_seed(42)
        for pos in range(seq_len):
            with torch.no_grad():
                # Extract activation
                A = hidden_states[self.layer + 1][:, pos, :]
                
                # Generate explanation
                if self.decoder.config.use_kv_cache:
                    gen = self.decoder.generate_soft_kv_cached(A, max_length=self.t_text, gumbel_tau=self.tau)
                else:
                    gen = self.decoder.generate_soft(A, max_length=self.t_text, gumbel_tau=self.tau)
                
                # Reconstruct
                A_hat = self.encoder(gen.generated_text_embeddings)
                
                # Compute KL divergence
                logits_orig = self.orig_model.forward_with_replacement(
                    input_ids=input_ids,
                    new_activation=A,
                    layer_idx=self.layer,
                    token_pos=pos,
                    no_grad=True
                ).logits
                
                logits_recon = self.orig_model.forward_with_replacement(
                    input_ids=input_ids,
                    new_activation=A_hat,
                    layer_idx=self.layer,
                    token_pos=pos,
                    no_grad=True
                ).logits
                
                # Extract logits at position
                logits_orig_at_pos = logits_orig[:, pos, :]
                logits_recon_at_pos = logits_recon[:, pos, :]
                
                # Compute KL with numerical stability
                with torch.amp.autocast('cuda', enabled=False):
                    logits_orig_f32 = logits_orig_at_pos.float()
                    logits_recon_f32 = logits_recon_at_pos.float()
                    logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0]
                    logits_recon_f32 = logits_recon_f32 - logits_recon_f32.max(dim=-1, keepdim=True)[0]
                    log_probs_orig = torch.log_softmax(logits_orig_f32, dim=-1)
                    log_probs_recon = torch.log_softmax(logits_recon_f32, dim=-1)
                    probs_orig = torch.exp(log_probs_orig)
                    kl_div = (probs_orig * (log_probs_orig - log_probs_recon)).sum(dim=-1).mean().item()
                
                # MSE for comparison
                mse = torch.nn.functional.mse_loss(A, A_hat).item()
                
                # Decode
                explanation = self.tokenizer.decode(gen.hard_token_ids[0], skip_special_tokens=True)
                token = self.tokenizer.decode([input_ids[0, pos].item()])
                
                results.append({
                    'position': pos,
                    'token': token,
                    'explanation': explanation,
                    'kl_divergence': kl_div,
                    'mse': mse
                })
        
        return pd.DataFrame(results)
    
    def causal_intervention(self, 
                           original_text: str, 
                           intervention_position: int, 
                           intervention_string: str,
                           max_new_tokens: int = 20,
                           visualize: bool = True) -> Dict[str, Any]:
        """Perform causal intervention by swapping in an encoded representation.
        
        Args:
            original_text: The original input text
            intervention_position: Token position to intervene at (0-indexed)
            intervention_string: String to encode and swap in
            max_new_tokens: Number of tokens to generate after intervention
            visualize: Whether to show visualization of the intervention
            
        Returns:
            Dictionary containing intervention results and analysis
        """
        # Tokenize original text
        inputs = self.tokenizer(original_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        if intervention_position >= seq_len:
            raise ValueError(f"Intervention position {intervention_position} exceeds sequence length {seq_len}")
        
        with torch.no_grad():
            # Get original hidden states
            outputs_orig = self.orig_model.model(input_ids, output_hidden_states=True)
            hidden_states_orig = outputs_orig.hidden_states
            
            # Get original activation at intervention position
            A_orig = hidden_states_orig[self.layer + 1][:, intervention_position, :]
            
            # Decode original activation to get its explanation
            if self.decoder.config.use_kv_cache:
                gen_orig = self.decoder.generate_soft_kv_cached(A_orig, max_length=self.t_text, gumbel_tau=self.tau)
            else:
                gen_orig = self.decoder.generate_soft(A_orig, max_length=self.t_text, gumbel_tau=self.tau)
            
            explanation_orig = self.tokenizer.decode(gen_orig.hard_token_ids[0], skip_special_tokens=True)
            
            # Create soft embeddings for intervention string
            # First tokenize it
            intervention_tokens = self.tokenizer(intervention_string, return_tensors="pt")
            intervention_ids = intervention_tokens.input_ids.to(self.device)
            
            # Get embeddings for intervention text
            if hasattr(self.orig_model.model, 'transformer'):
                embed_layer = self.orig_model.model.transformer.wte
            elif hasattr(self.orig_model.model, 'model'):
                embed_layer = self.orig_model.model.model.embed_tokens
            else:
                embed_layer = self.orig_model.model.get_input_embeddings()
            
            intervention_embeddings = embed_layer(intervention_ids)
            
            # Encode the intervention embeddings
            A_intervention = self.encoder(intervention_embeddings)
            
            # Take mean if multiple tokens
            #if A_intervention.shape[1] > 1:
            #    A_intervention = A_intervention.mean(dim=1, keepdim=True)
            #A_intervention = A_intervention.squeeze(1)
            
            # Get predictions with original activation
            outputs_orig_full = self.orig_model.forward_with_replacement(
                input_ids=input_ids,
                new_activation=A_orig,
                layer_idx=self.layer,
                token_pos=intervention_position,
                no_grad=True
            )
            
            # Get predictions with intervention
            outputs_intervention = self.orig_model.forward_with_replacement(
                input_ids=input_ids,
                new_activation=A_intervention,
                layer_idx=self.layer,
                token_pos=intervention_position,
                no_grad=True
            )
            
            # Generate continuations
            continuation_orig = self.orig_model.model.generate(
                input_ids[:, :intervention_position+1],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # For intervention, we need to patch in the activation during generation
            # This is more complex - for now, let's just compare next-token predictions
            
            # Get next token predictions
            next_token_logits_orig = outputs_orig_full.logits[:, intervention_position, :]
            next_token_logits_intervention = outputs_intervention.logits[:, intervention_position, :]
            
            # Get top predictions
            top_k = 10
            probs_orig = torch.softmax(next_token_logits_orig, dim=-1)
            probs_intervention = torch.softmax(next_token_logits_intervention, dim=-1)
            
            top_k_orig = torch.topk(probs_orig[0], k=top_k)
            top_k_intervention = torch.topk(probs_intervention[0], k=top_k)
            
            # Decode top predictions
            top_tokens_orig = [(self.tokenizer.decode([idx]), prob.item()) 
                              for idx, prob in zip(top_k_orig.indices, top_k_orig.values)]
            top_tokens_intervention = [(self.tokenizer.decode([idx]), prob.item()) 
                                      for idx, prob in zip(top_k_intervention.indices, top_k_intervention.values)]
            
            # Compute KL divergence between distributions
            kl_div = torch.nn.functional.kl_div(
                torch.log_softmax(next_token_logits_intervention, dim=-1),
                torch.softmax(next_token_logits_orig, dim=-1),
                reduction='batchmean'
            ).item()
            
            # Decode what actually happened
            intervened_token = self.tokenizer.decode([input_ids[0, intervention_position].item()])
            
            # Decode intervention explanation
            if self.decoder.config.use_kv_cache:
                gen_intervention = self.decoder.generate_soft_kv_cached(A_intervention, max_length=self.t_text, gumbel_tau=self.tau)
            else:
                gen_intervention = self.decoder.generate_soft(A_intervention, max_length=self.t_text, gumbel_tau=self.tau)
            
            explanation_intervention = self.tokenizer.decode(gen_intervention.hard_token_ids[0], skip_special_tokens=True)
            
        results = {
            'original_text': original_text,
            'intervention_position': intervention_position,
            'intervened_token': intervened_token,
            'intervention_string': intervention_string,
            'explanation_original': explanation_orig,
            'explanation_intervention': explanation_intervention,
            'top_predictions_original': top_tokens_orig,
            'top_predictions_intervention': top_tokens_intervention,
            'kl_divergence': kl_div,
            'continuation_original': self.tokenizer.decode(continuation_orig[0], skip_special_tokens=True)
        }
        
        # Print results
        print(f"\n🔬 Causal Intervention Analysis")
        print(f"{'='*60}")
        print(f"Original text: '{original_text}'")
        print(f"Intervened at position {intervention_position}: '{intervened_token}'")
        print(f"Intervention string: '{intervention_string}'")
        print(f"\n📝 Explanations:")
        print(f"  Original: {explanation_orig}")
        print(f"  Intervention: {explanation_intervention}")
        print(f"\n📊 KL Divergence: {kl_div:.4f}")
        print(f"\n🎯 Top next-token predictions:")
        print(f"\nOriginal:")
        for token, prob in top_tokens_orig[:5]:
            print(f"  {repr(token):15} {prob:.3f}")
        print(f"\nWith intervention:")
        for token, prob in top_tokens_intervention[:5]:
            print(f"  {repr(token):15} {prob:.3f}")
        
        if visualize:
            self._visualize_intervention(results)
        
        return results
    
    def _visualize_intervention(self, results: Dict[str, Any]):
        """Visualize the intervention results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot probability distributions
        tokens_orig = [t[0] for t in results['top_predictions_original'][:10]]
        probs_orig = [t[1] for t in results['top_predictions_original'][:10]]
        tokens_int = [t[0] for t in results['top_predictions_intervention'][:10]]
        probs_int = [t[1] for t in results['top_predictions_intervention'][:10]]
        
        x = np.arange(len(tokens_orig))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, probs_orig, width, label='Original', alpha=0.8, color='blue')
        bars2 = ax1.bar(x + width/2, probs_int, width, label='Intervention', alpha=0.8, color='red')
        
        ax1.set_xlabel('Tokens')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Next Token Predictions\nKL Divergence: {results["kl_divergence"]:.4f}')
        ax1.set_xticks(x)
        ax1.set_xticklabels([repr(t) for t in tokens_orig], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot KL divergence as a bar chart
        ax2.bar(['Original\n→\nOriginal', 'Original\n→\nIntervention'], 
                [0, results['kl_divergence']], 
                color=['gray', 'red'], alpha=0.7)
        ax2.set_ylabel('KL Divergence')
        ax2.set_title('Distribution Shift')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Initialize the analyzer
analyzer = LensAnalyzer(CHECKPOINT_PATH, DEVICE)

# %% [markdown]
# ## Single String Analysis
# 
# Analyze a single string and visualize the results:

# %%
# Analyze a test string
TEST_STRING = "The cat sat on the mat."
TEST_STRING = "<|endoftext|>No less than three sources (who wish to remain anonymous) have indicated to PPP that the Leafs are close to a contract extension with goaltender Jonas Gustavsson that will keep him in Toronto for the foreseeable future. I'm sort of at a loss for words. I mean, Gustavsson did have that one month where he wasn't hot"
TEST_STRING = "A paper butterfly is displayed on a coffin during Asia Funeral Expo (AFE) in Hong Kong May 19, 2011. REUTERS/Bobby Yip\n\nLONDON (Reuters) - A sharp increase in the number of deaths in Britain in 2015 compared with the previous year helped funeral services firm Dignity post a 16-*percent* profit increase, the firm said on Wednesday.\n\nDignity said the 7-percent year-on-year rise in the number of deaths..."

TEST_STRING = """This photo shows an image of a cougar in Wyoming. Animal control officers in Fairfax County have set up cameras this week in hopes of capturing the image of a cougar reported in the Alexandria section of the county. (Neil Wight/AP)\n\nMultiple sightings of what animal control officials are calling "possibly a cougar" have authorities in Fairfax County on high alert.\n\nOfficials received two reports this week of a "large cat" with an orange"""
# TEST_STRING = "India fearing Chinese submarines is not unwarranted. In 2014, the Chinese Navy frequently docked its submarines in Sri Lanka, saying they were goodwill visits and to replenish ships on deployment to the Arabian Sea.\n\nLater, in 2015, the Chinese Navy, or the People's Liberation Army Navy (PLAN), was seen docking in vessels Pakistan. The frequent visits and the constant reporting on them has led Indian Navy to get the best aircraft in its inventory, P-8I"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%
# Visualize KL divergence across positions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# KL divergence plot
positions = df['position'].values
kl_values = df['kl_divergence'].values
tokens = df['token'].values

ax1.plot(positions, kl_values, 'b-', linewidth=2, marker='o', markersize=8)
ax1.set_ylabel('KL Divergence', fontsize=12)
ax1.set_title(f'KL Divergence Across Token Positions\n"{TEST_STRING}"', fontsize=14)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Add token labels
for i, (pos, kl, token) in enumerate(zip(positions, kl_values, tokens)):
    if i % max(1, len(positions) // 10) == 0:  # Show ~10 labels
        ax1.annotate(repr(token), (pos, kl), 
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=8, alpha=0.7)

# MSE plot for comparison
mse_values = df['mse'].values
ax2.plot(positions, mse_values, 'r-', linewidth=2, marker='s', markersize=6)
ax2.set_xlabel('Token Position', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)
ax2.set_title('Reconstruction MSE', fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Show best and worst reconstructed tokens
print("\n🟢 Best reconstructed tokens (lowest KL):")
print(df.nsmallest(3, 'kl_divergence')[['position', 'token', 'explanation', 'kl_divergence']])

print("\n🔴 Worst reconstructed tokens (highest KL):")
print(df.nlargest(3, 'kl_divergence')[['position', 'token', 'explanation', 'kl_divergence']])

# %% [markdown]
# ## Batch Analysis
# 
# Analyze multiple strings and compare their patterns:

# %%
# Define strings to compare
STRINGS_TO_COMPARE = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "Machine learning is powerful.",
    "Hello world!",
    "import numpy as np",
    "def hello():\n    print('Hi')"
]

# Analyze each string
all_results = []
summary_stats = []

for text in STRINGS_TO_COMPARE:
    df = analyzer.analyze_all_tokens(text)
    df['text'] = text  # Add text column
    all_results.append(df)
    
    # Compute summary statistics
    summary_stats.append({
        'text': text[:50] + '...' if len(text) > 50 else text,
        'num_tokens': len(df),
        'avg_kl': df['kl_divergence'].mean(),
        'std_kl': df['kl_divergence'].std(),
        'max_kl': df['kl_divergence'].max(),
        'min_kl': df['kl_divergence'].min()
    })

summary_df = pd.DataFrame(summary_stats)
print("Summary Statistics:")
print(summary_df.to_string(index=False))

# %%
# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Plot KL trajectories for each string
for i, (text, df) in enumerate(zip(STRINGS_TO_COMPARE, all_results)):
    label = text[:30] + '...' if len(text) > 30 else text
    ax.plot(df['position'], df['kl_divergence'], 
            marker='o', label=label, alpha=0.7, linewidth=2)

ax.set_xlabel('Token Position', fontsize=12)
ax.set_ylabel('KL Divergence', fontsize=12)
ax.set_title('KL Divergence Comparison Across Different Texts', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Create a heatmap of explanations length vs KL divergence
combined_df = pd.concat(all_results, ignore_index=True)
combined_df['explanation_length'] = combined_df['explanation'].str.len()

# Bin the data for heatmap
kl_bins = pd.qcut(combined_df['kl_divergence'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
len_bins = pd.qcut(combined_df['explanation_length'], q=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])

# Create crosstab
heatmap_data = pd.crosstab(len_bins, kl_bins)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Relationship between Explanation Length and KL Divergence')
plt.xlabel('KL Divergence Category')
plt.ylabel('Explanation Length Category')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interactive Analysis Function
# 
# Use this function to quickly analyze any string:

# %%
def quick_analyze(text: str, show_plot: bool = True):
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
        kl = row['kl_divergence']
        # Color code based on KL value
        if kl < df['kl_divergence'].quantile(0.33):
            indicator = "🟢"
        elif kl < df['kl_divergence'].quantile(0.67):
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"{indicator} [{row['position']:2d}] {repr(row['token']):15} → {row['explanation']:40} (KL: {kl:.3f})")
    
    if show_plot and len(df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(df['position'], df['kl_divergence'], 'b-', linewidth=2, marker='o')
        plt.xlabel('Position')
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.title(f'KL Divergence: "{text[:40]}..."')
        plt.grid(True, alpha=0.3)
        
        # Annotate some points
        for i in range(0, len(df), max(1, len(df) // 5)):
            plt.annotate(repr(df.iloc[i]['token']), 
                        (df.iloc[i]['position'], df.iloc[i]['kl_divergence']),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    return df

# %%
# Try it out!
quick_analyze("The quick brown cat jumps over the lazy mouse.\n"*10)

# %%
# Analyze your own text
YOUR_TEXT = "Enter your text here!"  # <-- Change this
df_custom = quick_analyze(YOUR_TEXT)

# %% [markdown]
# ## Statistical Analysis
# 
# Let's look at patterns in the explanations:

# %%
# Combine all results for statistical analysis
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Most common explanation words
    from collections import Counter
    all_explanations = ' '.join(combined_df['explanation'].values)
    word_counts = Counter(all_explanations.split())
    
    print("Most common words in explanations:")
    for word, count in word_counts.most_common(20):
        print(f"  {word:15} : {count:4d}")
    
    # Correlation between token length and KL
    combined_df['token_length'] = combined_df['token'].str.len()
    correlation = combined_df[['token_length', 'kl_divergence']].corr().iloc[0, 1]
    print(f"\nCorrelation between token length and KL divergence: {correlation:.3f}")

# %%
# Distribution plots
if all_results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # KL distribution
    axes[0, 0].hist(combined_df['kl_divergence'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('KL Divergence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of KL Divergence')
    
    # MSE distribution
    axes[0, 1].hist(combined_df['mse'], bins=30, alpha=0.7, color='red')
    axes[0, 1].set_xlabel('MSE')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of MSE')
    
    # KL vs MSE scatter
    axes[1, 0].scatter(combined_df['mse'], combined_df['kl_divergence'], alpha=0.5)
    axes[1, 0].set_xlabel('MSE')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL vs MSE Relationship')
    
    # Explanation length distribution
    axes[1, 1].hist(combined_df['explanation'].str.len(), bins=20, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Explanation Length (chars)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Explanation Lengths')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Save Results
# 
# Save your analysis results for later use:

# %%
# Save to CSV
if all_results:
    output_path = "lens_analysis_results.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(combined_df)} rows to {output_path}")

# %%
# Create a summary report
def create_summary_report(results_df: pd.DataFrame, text: str) -> str:
    """Create a text summary report of the analysis."""
    report = f"""
Consistency Lens Analysis Report
================================
Text: "{text[:100]}{'...' if len(text) > 100 else ''}"
Checkpoint: {analyzer.checkpoint_path.name}
Model: {analyzer.model_name}
Layer: {analyzer.layer}

Summary Statistics:
- Total tokens: {len(results_df)}
- Average KL divergence: {results_df['kl_divergence'].mean():.4f}
- Std deviation: {results_df['kl_divergence'].std():.4f}
- Min KL: {results_df['kl_divergence'].min():.4f}
- Max KL: {results_df['kl_divergence'].max():.4f}

Top 3 most consistent tokens (lowest KL):
"""
    for _, row in results_df.nsmallest(3, 'kl_divergence').iterrows():
        report += f"  - [{row['position']}] '{row['token']}' → '{row['explanation']}' (KL: {row['kl_divergence']:.4f})\n"
    
    report += "\nTop 3 least consistent tokens (highest KL):\n"
    for _, row in results_df.nlargest(3, 'kl_divergence').iterrows():
        report += f"  - [{row['position']}] '{row['token']}' → '{row['explanation']}' (KL: {row['kl_divergence']:.4f})\n"
    
    return report

# Generate report for the test string
if 'df' in locals():
    report = create_summary_report(df, TEST_STRING)
    print(report)
    
    # Save to file
    with open("lens_analysis_report.txt", "w") as f:
        f.write(report)
    print("\n✓ Report saved to lens_analysis_report.txt")

# %% [markdown]
# ## Next Steps
# 
# 1. Update `CHECKPOINT_PATH` to point to your trained model
# 2. Modify `TEST_STRING` and `STRINGS_TO_COMPARE` with your own text
# 3. Run cells to analyze and visualize results
# 4. Use `quick_analyze()` for interactive exploration
# 5. Save interesting results using the provided functions
# 
# Happy analyzing! 🔍
# %%
TEST_STRING = " t h e   c a t   s a t   o n   t h e   m a t"
TEST_STRING = " t h e   c a t   s a t   o n   t h e   m a t\n\n the cat sat on the"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

# %%
TEST_STRING = "John met Mary. He gave her a book. She thanked him."
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%
TEST_STRING = "French: monsieur, votre chien est un problème\nEnglish: Mister, your dog is a problem.\n\nFrench: je suis un chat\nEnglish: I am a cat\n\nFrench: Ou est la gare? English: Where is the station? \n\n French: Tu n'aime pas les chats? English:"
#TEST_STRING = "French: chien\n English: dog\n\nFrench: chat\n English: cat\n\nFrench: gare\n English: station\n\nFrench: voiture\nEnglish: car\n\nFrench: maison\nEnglish:"
TEST_STRING = " H"
encodestring = analyzer.tokenizer.encode(TEST_STRING)
# generateddirectly = analyzer.model.generate(torch.tensor([encodestring]).to(analyzer.device), max_new_tokens=100)
# print(generateddirectly)
# tokenizerstring = analyzer.tokenizer.decode(generateddirectly[0])
# print(tokenizerstring)

# %%
TEST_STRING = "The door was closed. Alice opened it. Now the door is open.\n\nThe door was closed. Bob closed it. Now the door is closed.\n\nThe door was open. Carol shut it. Now the door is shut.\n\nThe door was open. David closed it. Now the door is"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

TEST_STRING = "Looking to leave a two-fight winless streak behind him, Manny Gamburyan will return to the UFC octagon on Sept. 27 in Las Vegas at UFC 178 in a new weight class.\n\nGamburyan is set to make his 135-pound men's bantamweight debut against Cody \"The Renegade\" Gibson at the MGM Grand Arena on a pay-per-view card headlined by the UFC light heavyweight title fight between champion Jon Jones and Daniel Cormierr"
df = analyzer.analyze_all_tokens(TEST_STRING)

# Display the results
print(f"Analysis of: '{TEST_STRING}'")
print(f"Total tokens: {len(df)}")
print("\nToken-by-token breakdown:")
print(df.to_string(index=False))

# %%

# %% [markdown]
# ## Causal Intervention Analysis
# 
# Perform causal interventions by encoding alternate strings and swapping them into the forward pass:

# %%
# Example 1: Simple pronoun intervention
original = "The cat sat on the mat. It was very comfortable."
intervention_result = analyzer.causal_intervention(
    original_text=original,
    intervention_position=10,  # Position of "It"
    intervention_string="The dog",
    max_new_tokens=10
)

# %%
# Example 2: Conceptual intervention
original = "The scientist discovered a new element. This was"
intervention_result = analyzer.causal_intervention(
    original_text=original,
    intervention_position=8,  # Position of "This"
    intervention_string="The failure",
    max_new_tokens=15
)

# %%
# Example 3: Language intervention
original = "Bonjour means hello in French. Merci means"
intervention_result = analyzer.causal_intervention(
    original_text=original,
    intervention_position=9,  # Position of "Merci"
    intervention_string="Gracias",
    max_new_tokens=10
)

# %%
# Interactive intervention - customize these!
YOUR_TEXT = "The door was open. She closed it. Now the door is"
YOUR_POSITION = 11  # Token position to intervene at
YOUR_INTERVENTION = "He opened"  # What to encode and swap in

result = analyzer.causal_intervention(
    original_text=YOUR_TEXT,
    intervention_position=YOUR_POSITION,
    intervention_string=YOUR_INTERVENTION
)

# %%