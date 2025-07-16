import logging
from collections import Counter
from transformers import AutoTokenizer
from lens.utils.logging import log_metrics # W&B logging

def log_epoch_token_statistics(epoch_decoded_tokens: list[int], 
                               tokenizer: AutoTokenizer, 
                               current_epoch_num: int, # 1-based
                               step: int, # Current global step (end of epoch)
                               log_interval_console: int, # For console logging conditional
                               log: logging.Logger):
    """Logs statistics about token occurrences at the end of an epoch."""
    if not epoch_decoded_tokens:
        return

    token_counts = Counter(epoch_decoded_tokens)
    if not token_counts:
        return

    most_common_token_id, most_common_count = token_counts.most_common(1)[0]
    total_tokens_in_epoch = len(epoch_decoded_tokens)
    frequency = most_common_count / total_tokens_in_epoch

    # Console logging - check if current step aligns for console log
    # The original script logs this with `log.info` if `step % log_interval == 0`.
    # We should be careful here, as `step` is end-of-epoch step.
    # If `log_interval_console` is for per-step console logging, this might always log
    # or never log depending on its value relative to epoch length.
    # Assuming `log_interval_console` is the general console logging interval.
    if step % log_interval_console == 0 : 
        log.info(
            f"Epoch {current_epoch_num} most common token: ID {most_common_token_id} = `{tokenizer.decode([most_common_token_id])}` "
            f"(Count: {most_common_count}/{total_tokens_in_epoch}, Freq: {frequency:.4f})"
        )
    
    # W&B logging
    log_metrics({
        "epoch_stats/most_common_token_id": most_common_token_id,
        "epoch_stats/most_common_token_count": most_common_count,
        "epoch_stats/most_common_token_freq": frequency,
        "epoch_stats/total_tokens_in_epoch": total_tokens_in_epoch,
    }, step=step)