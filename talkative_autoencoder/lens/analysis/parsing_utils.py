import re
from typing import Dict, List, Optional, Tuple

import torch


def is_gpt_oss_model(model_name: str) -> bool:
    return isinstance(model_name, str) and ("gpt-oss" in model_name.lower())


def get_start_of_turn_tokens(tokenizer, logger=None) -> List[int]:
    start_of_turn_tokens: List[int] = []
    possible_start_tokens = [
        "<|im_start|>",
        "<|startofturn|>",
        "<|start_of_turn|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|im_start|>",
        "<|im_sep|>",
        "<startofturn>",
        "<start_of_turn>",
        "<im_start>",
        "<im_sep>",
    ]
    for tok in possible_start_tokens:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)
        if len(tok_id) == 1:
            start_of_turn_tokens.append(tok_id[0])

    if getattr(tokenizer, "bos_token_id", None) is not None:
        start_of_turn_tokens.append(tokenizer.bos_token_id)

    # de-duplicate while preserving order
    seen = set()
    unique = []
    for t in start_of_turn_tokens:
        if t not in seen:
            unique.append(t)
            seen.add(t)

    if logger is not None:
        logger.info(f"Found {len(unique)} start-of-turn tokens: {unique}")
    return unique


def find_start_positions(input_ids_flat: List[int], start_token_ids: List[int]) -> List[int]:
    if not start_token_ids:
        return []
    return [idx for idx, tok in enumerate(input_ids_flat) if tok in start_token_ids]


def compute_tokens_to_analyze_mask(
    seq_len: int,
    start_positions: List[int],
    last_n_messages: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[int]]:
    if last_n_messages is None:
        return None, None, None

    if len(start_positions) >= last_n_messages + 1:
        last_start = start_positions[-1]
        if last_start >= seq_len - 4:
            start_idx = max(0, start_positions[-(last_n_messages + 1)] - 3)
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            mask[start_idx:] = True
            return mask, start_idx, start_positions[-(last_n_messages + 1)]
        else:
            start_idx = start_positions[-last_n_messages]
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            mask[start_idx:] = True
            return mask, start_idx, start_positions[-last_n_messages]
    else:
        return None, None, None


def normalize_gemma_messages(model_name: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        return messages
    if "gemma" in (model_name or "").lower() and messages[0].get("role") == "system":
        system_prompt = messages[0].get("content", "")
        rest = messages[1:]
        if rest and rest[0].get("role") == "user":
            rest[0]["content"] = system_prompt + ("\n\n" if rest[0]["content"] else "") + rest[0]["content"]
        return rest
    return messages


def is_assistant_prefill(messages: List[Dict[str, str]]) -> bool:
    return bool(messages) and messages[-1].get("role") == "assistant"


def apply_chat_template_safe(
    tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool, continue_final_message: bool
):
    return tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )


def get_assistant_end_tokens(tokenizer) -> List[int]:
    toks: List[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        toks.append(tokenizer.eos_token_id)

    special = [
        "<end_of_turn>",
        "<|end_of_turn|>",
        "</turn>",
        "</s>",
        "[/INST]",
        "</assistant>",
        "<|eot_id|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<return>",
        # NOT <end> as this is a 'message' end, nor a EOS marker
    ]
    for token_str in special:
        try:
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                    toks.append(token_id)
            enc_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if enc_ids and len(enc_ids) == 1:
                toks.append(enc_ids[0])
        except Exception:
            pass

    seen = set()
    unique = []
    for t in toks:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def fix_bos_eos(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        hasattr(tokenizer, "bos_token_id")
        and input_ids.shape[1] >= 2
        and input_ids[0, 0].item() == tokenizer.bos_token_id
        and input_ids[0, 1].item() == tokenizer.bos_token_id
    ):
        input_ids = input_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

    if (
        hasattr(tokenizer, "eos_token_id")
        and input_ids.shape[1] >= 1
        and input_ids[0, -1].item() == tokenizer.eos_token_id
    ):
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
    return input_ids, attention_mask


_HARMONY_SEGMENT_RE = re.compile(
    r"<\|start\|>(?P<role>[^\s<|]+)(?:\s+[^<|]*)?"
    r"<\|channel\|>(?P<channel>\w+)(?:\s+[^<|]*)?"
    r"<\|message\|>(?P<content>.*?)(?:<\|call\|>|<\|end\|>)",
    re.DOTALL,
)


def parse_harmony_segments(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    return [
        {"role": m.group("role"), "channel": m.group("channel"), "content": m.group("content")}
        for m in _HARMONY_SEGMENT_RE.finditer(text)
    ]


def extract_harmony_contents(text: str) -> Tuple[str, str]:
    """
    Returns (final_text, analysis_text). Falls back gracefully if 'final' not found.
    """
    segments = parse_harmony_segments(text)
    finals = [s["content"].strip() for s in segments if s.get("channel") == "final" and s.get("content")]
    analyses = [s["content"].strip() for s in segments if s.get("channel") == "analysis" and s.get("content")]

    final_text = "\n".join(finals).strip() if finals else ""
    analysis_text = "\n\n".join(analyses).strip() if analyses else ""

    if not final_text:
        # Fallback: take the last non-analysis message
        non_analysis = [s["content"].strip() for s in segments if s.get("channel") != "analysis" and s.get("content")]
        if non_analysis:
            final_text = non_analysis[-1]

    if not final_text:
        final_text = text.strip()

    return final_text, analysis_text


# ---- Harmony input rendering ----


def has_harmony_metadata(messages: List[Dict[str, str]]) -> bool:
    """
    Returns True if any message contains Harmony-specific metadata we should render:
    - 'thinking' (assistant chain-of-thought content to be placed on analysis channel)
    - 'channel' (explicit channel override)
    """
    for m in messages or []:
        if isinstance(m, dict) and ("thinking" in m or "channel" in m):
            return True
    return False


def _render_segment(role: str, channel: str, content: str) -> str:
    return f"<|start|>{role}<|channel|>{channel}<|message|>{content}<|end|>"


def render_harmony_from_messages(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    continue_final_message: bool,
) -> str:
    """
    Render a Harmony-formatted conversation string from list-of-dicts.

    - For assistant messages with 'thinking', emit an analysis segment first.
    - For assistant messages with 'channel', honor it; otherwise default to 'final'.
    - For user/system/developer messages, default channel to 'final'.
    - Assistant prefill is represented by including the assistant final segment with partial content
      when continue_final_message is True.
    - When add_generation_prompt=True (last is user), we do not add any trailing assistant stub; the
      model will start a new assistant segment on its own.
    """
    rendered: List[str] = []
    for idx, m in enumerate(messages or []):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip()
        content = m.get("content") or ""
        channel_override = m.get("channel")

        if role == "assistant":
            thinking = m.get("thinking")
            if thinking:
                rendered.append(_render_segment("assistant", "analysis", thinking))
            if channel_override:
                if content:
                    rendered.append(_render_segment("assistant", channel_override, content))
            else:
                if content:
                    rendered.append(_render_segment("assistant", "final", content))
        elif role in ("user", "system", "developer"):
            ch = channel_override or "final"
            if content:
                rendered.append(_render_segment(role, ch, content))
        else:
            # Unknown role: pass through as final
            if content:
                rendered.append(_render_segment(role or "user", channel_override or "final", content))

    # No explicit assistant stub is added; the model will start the next assistant message.
    return "".join(rendered)
