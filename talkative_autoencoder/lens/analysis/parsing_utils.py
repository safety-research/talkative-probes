import re
from typing import Dict, List, Optional, Tuple

import torch

# Harmony package integration (required)
try:
    from openai_harmony import (
        Conversation as _HarmonyConversation,
    )
    from openai_harmony import (
        HarmonyEncodingName,
        StreamableParser,
        load_harmony_encoding,
    )
    from openai_harmony import (
        Message as _HarmonyMessage,
    )
    from openai_harmony import (
        Role as _HarmonyRole,
    )

    _HARMONY_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    _HARMONY_PKG_AVAILABLE = True
except Exception:
    _HARMONY_PKG_AVAILABLE = False
    # Central functionality; do not silently fall back
    print("[Harmony] openai_harmony unavailable; required for Harmony parsing/rendering")


def is_gpt_oss_model(model_name: str) -> bool:
    return isinstance(model_name, str) and ("gpt-oss" in model_name.lower())


def get_start_of_turn_tokens(tokenizer, logger=None) -> List[int]:
    start_of_turn_tokens: List[int] = []
    possible_start_tokens = [
        "<|im_start|>",
        "<|startofturn|>",
        "<|start_of_turn|>",
        "<|start|>",
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
    r"<\|message\|>(?P<content>.*?)(?:<\|call\|>|<\|end\|>|<\|return\|>)",
    re.DOTALL,
)

# Regex for partial segments (when prefilling)
_HARMONY_PARTIAL_RE = re.compile(
    r"(?:<\|start\|>(?P<role>[^\s<|]+)(?:\s+[^<|]*)?)?"
    r"(?:<\|channel\|>(?P<channel>\w+)(?:\s+[^<|]*)?)?"
    r"(?:<\|message\|>)?(?P<content>.*?)(?:<\|call\|>|<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)


def parse_harmony_segments(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    if not _HARMONY_PKG_AVAILABLE:
        raise RuntimeError("openai_harmony is required for Harmony parsing; please install openai-harmony")

    # Prefer direct text parser if available
    parse_text_fn = getattr(_HARMONY_ENCODING, "parse_messages_from_completion_text", None)
    if callable(parse_text_fn):
        msgs = parse_text_fn(text, role=_HarmonyRole.ASSISTANT)
        segments: List[Dict[str, str]] = []
        for msg in msgs or []:
            # Prefer to_dict if available
            to_dict = getattr(msg, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
                segments.append(
                    {
                        "role": d.get("role") or "assistant",
                        "channel": d.get("channel") or "final",
                        "content": d.get("content") or "",
                    }
                )
            else:
                role = getattr(msg, "role", None) or "assistant"
                channel = getattr(msg, "channel", None) or "final"
                content = getattr(msg, "content", None) or ""
                segments.append({"role": str(role), "channel": str(channel), "content": content})
        if segments:
            return segments

    # Tokenize text, then parse tokens to messages (handles most cases)
    to_tokens = None
    for fn_name in ("string_to_tokens", "text_to_tokens", "encode_text_to_tokens"):
        fn = getattr(_HARMONY_ENCODING, fn_name, None)
        if callable(fn):
            to_tokens = fn
            break
    if not callable(to_tokens):
        raise RuntimeError("Harmony encoding missing string->tokens function; cannot parse")

    tokens = to_tokens(text)
    parse_tokens_fn = getattr(_HARMONY_ENCODING, "parse_messages_from_completion_tokens", None)
    if callable(parse_tokens_fn):
        msgs = parse_tokens_fn(tokens, _HarmonyRole.ASSISTANT)
        segments: List[Dict[str, str]] = []
        for msg in msgs or []:
            to_dict = getattr(msg, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
                segments.append(
                    {
                        "role": d.get("role") or "assistant",
                        "channel": d.get("channel") or "final",
                        "content": d.get("content") or "",
                    }
                )
            else:
                role = getattr(msg, "role", None) or "assistant"
                channel = getattr(msg, "channel", None) or "final"
                content = getattr(msg, "content", None) or ""
                segments.append({"role": str(role), "channel": str(channel), "content": content})
        if segments:
            return segments

    # Handle incomplete outputs using StreamableParser (package-based)
    stream = StreamableParser(_HARMONY_ENCODING, role=_HarmonyRole.ASSISTANT)
    for t in tokens:
        stream.process(t)
    role = getattr(stream, "current_role", None) or "assistant"
    channel = getattr(stream, "current_channel", None) or "final"
    content = getattr(stream, "current_content", None) or ""
    if content or role or channel:
        return [{"role": str(role), "channel": str(channel), "content": content}]

    return []


def extract_harmony_contents(text: str) -> Tuple[str, str]:
    """
    Returns (final_text, analysis_text). Handles both complete segments and prefill cases.
    """
    segments = parse_harmony_segments(text)
    finals = [_to_plain_text(s["content"]) for s in segments if s.get("channel") == "final" and s.get("content")]
    analyses = [_to_plain_text(s["content"]) for s in segments if s.get("channel") == "analysis" and s.get("content")]

    # Return the last segment per channel to match reference usage
    final_text = finals[-1].strip() if finals else ""
    analysis_text = analyses[-1].strip() if analyses else ""

    # If no segments found at all, treat entire text as final content
    if not segments:
        final_text = text

    return final_text, analysis_text


def extract_harmony_contents_from_tokens(tokens: List[int]) -> Tuple[str, str]:
    """
    Returns (final_text, analysis_text) parsed from completion token IDs using Harmony package.
    """
    if not _HARMONY_PKG_AVAILABLE:
        raise RuntimeError("openai_harmony is required for Harmony parsing; please install openai-harmony")
    parse_tokens_fn = getattr(_HARMONY_ENCODING, "parse_messages_from_completion_tokens", None)
    if not callable(parse_tokens_fn):
        raise RuntimeError("Harmony encoding missing token parser; cannot parse")
    msgs = parse_tokens_fn(tokens, _HarmonyRole.ASSISTANT)
    finals: List[str] = []
    analyses: List[str] = []
    for msg in msgs or []:
        d = msg.to_dict() if hasattr(msg, "to_dict") else None
        if d is None:
            channel = getattr(msg, "channel", None)
            content = getattr(msg, "content", None)
        else:
            channel = d.get("channel")
            content = d.get("content")
        content_text = _to_plain_text(content)
        if channel == "final" and content_text:
            finals.append(content_text)
        elif channel == "analysis" and content_text:
            analyses.append(content_text)
    # Return the last segment per channel to match reference usage
    final_text = finals[-1].strip() if finals else ""
    analysis_text = analyses[-1].strip() if analyses else ""
    return final_text, analysis_text


def _to_plain_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_to_plain_text(part) for part in content)
    if isinstance(content, dict):
        # Prefer explicit text field
        if isinstance(content.get("text"), str):
            return content["text"]
        # Some objects nest content
        if "content" in content:
            return _to_plain_text(content["content"])
        return ""
    return str(content)


def collapse_harmony_turn_start_positions(start_positions: List[int], text: str) -> List[int]:
    # Use local Harmony regex to infer roles from text for boundary collapsing only
    matches = list(_HARMONY_SEGMENT_RE.finditer(text))
    if not matches or len(matches) != len(start_positions):
        return start_positions
    roles: List[str] = [(m.group("role") or "").strip() for m in matches]
    collapsed: List[int] = []
    prev_role: Optional[str] = None
    for idx, role in enumerate(roles):
        if role == "assistant" and prev_role == "assistant":
            prev_role = role
            continue
        collapsed.append(start_positions[idx])
        prev_role = role
    return collapsed


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
) -> List[int]:
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
    if not _HARMONY_PKG_AVAILABLE:
        raise RuntimeError("openai_harmony is required for Harmony rendering; please install openai-harmony")

    # Build Harmony Conversation from messages (cookbook style)
    harmony_messages: List[_HarmonyMessage] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role_str = (m.get("role") or "").strip() or "user"
        content_str = m.get("content") or ""
        role_enum = {
            "system": _HarmonyRole.SYSTEM,
            "developer": _HarmonyRole.DEVELOPER,
            "user": _HarmonyRole.USER,
            "assistant": _HarmonyRole.ASSISTANT,
        }.get(role_str, _HarmonyRole.USER)
        if content_str:
            harmony_messages.append(_HarmonyMessage.from_role_and_content(role_enum, content_str))

    convo = _HarmonyConversation.from_messages(harmony_messages)
    render_fn = getattr(_HARMONY_ENCODING, "render_conversation_for_completion", None)
    if not callable(render_fn):
        raise RuntimeError("Harmony encoding missing render_conversation_for_completion; cannot render")

    prefill_ids = render_fn(convo, _HarmonyRole.ASSISTANT)
    if not isinstance(prefill_ids, list):
        raise RuntimeError("Harmony encoding returned unexpected type for prefill tokens")
    return prefill_ids


def get_harmony_stop_tokens() -> List[int]:
    if not _HARMONY_PKG_AVAILABLE:
        raise RuntimeError("openai_harmony is required for Harmony stop tokens; please install openai-harmony")
    stop_fn = getattr(_HARMONY_ENCODING, "stop_tokens_for_assistant_actions", None)
    if not callable(stop_fn):
        raise RuntimeError("Harmony encoding missing stop_tokens_for_assistant_actions")
    return stop_fn()


def harmony_prompt_string_from_messages(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    continue_final_message: bool,
) -> str:
    if not _HARMONY_PKG_AVAILABLE:
        raise RuntimeError("openai_harmony is required for Harmony rendering; please install openai-harmony")

    harmony_messages: List[_HarmonyMessage] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role_str = (m.get("role") or "").strip() or "user"
        content_str = m.get("content") or ""
        role_enum = {
            "system": _HarmonyRole.SYSTEM,
            "developer": _HarmonyRole.DEVELOPER,
            "user": _HarmonyRole.USER,
            "assistant": _HarmonyRole.ASSISTANT,
        }.get(role_str, _HarmonyRole.USER)
        if content_str:
            harmony_messages.append(_HarmonyMessage.from_role_and_content(role_enum, content_str))

    convo = _HarmonyConversation.from_messages(harmony_messages)
    render_fn = getattr(_HARMONY_ENCODING, "render_conversation_for_completion", None)
    if not callable(render_fn):
        raise RuntimeError("Harmony encoding missing render_conversation_for_completion; cannot render")
    prefill_ids = render_fn(convo, _HarmonyRole.ASSISTANT)
    to_str = getattr(_HARMONY_ENCODING, "tokens_to_string", None)
    if not callable(to_str):
        raise RuntimeError("Harmony encoding missing tokens_to_string; cannot convert to string")
    return to_str(prefill_ids)
